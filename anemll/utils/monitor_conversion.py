#!/usr/bin/env python3
"""
Monitor conversion progress in real-time.

Run this in a separate terminal while convert_model.sh is running:
    python anemll/utils/monitor_conversion.py --output /path/to/output

Shows:
- Current step and what's being processed
- Elapsed time per step
- Estimated time remaining
"""

import argparse
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


# Step definitions for different model types
STEPS = {
    "standard": [
        (1, "Embeddings", "_embeddings"),
        (2, "LM Head", "_lm_head"),
        (3, "FFN Chunks", "_FFN_chunk_"),
        (4, "Prefill Chunks", "_PF_chunk_"),
        (5, "Combined Models", "_FFN_PF_"),
        (6, "Compilation", ".mlmodelc"),
        (7, "Tokenizer/Meta", "meta.yaml"),
        (8, "Testing", None),
    ],
    "gemma3_rotate": [
        (1, "Embeddings", "_embeddings"),
        (2, "LM Head", "_lm_head"),
        (3, "FFN Chunks", "_FFN_chunk_"),
        (4, "Prefill Chunks", "_PF_chunk_"),
        ("4a", "FFN Rotate", "_FFN_rotate_chunk_"),
        ("4b", "Prefill Rotate", "_PF_rotate_chunk_"),
        (5, "Combined Models", "_FFN_PF_"),
        (6, "Compilation", ".mlmodelc"),
        (7, "Tokenizer/Meta", "meta.yaml"),
        (8, "Testing", None),
    ],
}

# Average step durations (in seconds) - will be calibrated from actual runs
# These are conservative estimates for ~4B models with LUT4
# Smaller models (1B) may be 2-3x faster
DEFAULT_STEP_DURATIONS = {
    "1": 120,       # Embeddings - ~2 min
    "2": 120,       # LM Head - ~2 min
    "3": 240,       # FFN Chunks - ~4 min per chunk
    "4": 240,       # Prefill Chunks - ~4 min per chunk
    "4a": 240,      # FFN Rotate - ~4 min per chunk
    "4b": 600,      # Prefill Rotate - can take 10+ min per chunk for 4B
    "5": 15,        # Combine - ~15s per function
    "6": 5,         # Compile - few seconds per model
    "7": 2,         # Tokenizer/Meta - instant
    "8": 30,        # Test - fast
}

# build_ctx_model defaults (scripts/build_ctx_model)
BUILD_CTX_DEFAULT_CONTEXTS = [512, 1024, 2048, 3072, 4096]
BUILD_CTX_DEFAULT_MAX_CONTEXT = 4096
BUILD_CTX_DEFAULT_STATE_ROOT = "/Volumes/Models/ANE/vibethinker_1.5b_state_transition"
BUILD_CTX_DEFAULT_CONTEXT_NAME_TEMPLATE = "vibethinker_1.5b_ctx{context}_fp16_hybrid"
BUILD_CTX_STEP_NAMES = {
    1: "Export max context",
    2: "Export remaining contexts",
    3: "Combine contexts",
}


def _split_command(command: str) -> list:
    """Split a command string safely."""
    try:
        return shlex.split(command)
    except Exception:
        return command.split()


def _get_cli_arg(tokens: list, flag: str, default=None):
    """Get --flag value from tokenized CLI command."""
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default


def _has_cli_flag(tokens: list, flag: str) -> bool:
    """Check whether a flag exists in tokenized CLI command."""
    return any(tok == flag or tok.startswith(flag + "=") for tok in tokens)


def _parse_contexts(raw_value) -> list:
    """Parse context list from CLI value like '512 1024' or '512,1024'."""
    if not raw_value:
        return BUILD_CTX_DEFAULT_CONTEXTS.copy()
    vals = []
    for part in str(raw_value).replace(",", " ").split():
        try:
            vals.append(int(part))
        except Exception:
            pass
    return vals or BUILD_CTX_DEFAULT_CONTEXTS.copy()


def _context_dir_for(context_root: str, name_template: str, ctx: int) -> Path:
    """Resolve per-context output directory path."""
    return Path(context_root) / name_template.replace("{context}", str(ctx))


def _has_context_infer_chunks(context_dir: Path, num_chunks: int = 3) -> bool:
    """Check expected infer chunk artifacts for a context export."""
    for idx in range(1, num_chunks + 1):
        idx_s = f"{idx:02d}"
        found = False
        for ext in ("mlpackage", "mlmodelc"):
            if (context_dir / f"qwen25_FFN_chunk_{idx_s}of{num_chunks:02d}.{ext}").exists():
                found = True
                break
            if (context_dir / f"qwen25_FFN_PF_chunk_{idx_s}of{num_chunks:02d}.{ext}").exists():
                found = True
                break
        if not found:
            return False
    return True


def _has_state_transition_meta(output_dir: Path) -> bool:
    """Check if combined state-transition meta.yaml exists with key fields."""
    meta_path = output_dir / "meta.yaml"
    if not meta_path.exists():
        return False
    try:
        text = meta_path.read_text()
    except Exception:
        return False
    return (
        "state_transition_infer_contexts:" in text
        and "state_transition_prefill_context:" in text
    )


def get_process_table() -> list:
    """Return process table entries as dict(pid, ppid, command)."""
    procs = []
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            line = line.rstrip()
            if not line:
                continue
            m = re.match(r"^\s*(\d+)\s+(\d+)\s+(.*)$", line)
            if not m:
                continue
            procs.append(
                {
                    "pid": int(m.group(1)),
                    "ppid": int(m.group(2)),
                    "command": m.group(3),
                }
            )
    except Exception:
        return []
    return procs


def _descendant_pids(root_pid: int, children_map: dict) -> set:
    """Get all descendant PIDs for a root PID."""
    out = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        for child in children_map.get(pid, []):
            if child in out:
                continue
            out.add(child)
            stack.append(child)
    return out


def detect_build_ctx_state(num_chunks: int = 3) -> dict:
    """
    Detect running scripts/build_ctx_model and infer top-level progress.

    Returns a dict with state when running, else None.
    """
    procs = get_process_table()
    if not procs:
        return None

    build_procs = [
        p for p in procs
        if "scripts/build_ctx_model" in p["command"]
        and "monitor_conversion.py" not in p["command"]
    ]
    if not build_procs:
        return None

    children_map = {}
    by_pid = {p["pid"]: p for p in procs}
    for p in procs:
        children_map.setdefault(p["ppid"], []).append(p["pid"])

    candidates = []
    for bp in build_procs:
        tokens = _split_command(bp["command"])
        step_raw = _get_cli_arg(tokens, "--step")
        if _has_cli_flag(tokens, "--all"):
            step_raw = "all"
        if not step_raw:
            # Script requires --step/--all; if absent, assume all.
            step_raw = "all"

        if str(step_raw).lower() == "all":
            planned_steps = [1, 2, 3]
        else:
            planned_steps = []
            for part in str(step_raw).replace(",", " ").split():
                try:
                    v = int(part)
                except Exception:
                    continue
                if v in (1, 2, 3):
                    planned_steps.append(v)
            if not planned_steps:
                planned_steps = [1, 2, 3]

        contexts = _parse_contexts(_get_cli_arg(tokens, "--contexts"))
        max_context = int(_get_cli_arg(tokens, "--max-context", BUILD_CTX_DEFAULT_MAX_CONTEXT))
        state_root = _get_cli_arg(tokens, "--state-root", BUILD_CTX_DEFAULT_STATE_ROOT)
        context_root = _get_cli_arg(tokens, "--context-root")
        if not context_root:
            context_root = str(Path(state_root) / "_contexts")
        output_dir = _get_cli_arg(tokens, "--output", state_root)
        context_name_template = _get_cli_arg(
            tokens,
            "--context-name-template",
            BUILD_CTX_DEFAULT_CONTEXT_NAME_TEMPLATE,
        )

        max_dir = _context_dir_for(context_root, context_name_template, max_context)
        nonmax_contexts = [c for c in contexts if c != max_context]

        done = {}
        if 1 in planned_steps:
            done[1] = _has_context_infer_chunks(max_dir, num_chunks=num_chunks)
        if 2 in planned_steps:
            done[2] = all(
                _has_context_infer_chunks(
                    _context_dir_for(context_root, context_name_template, ctx),
                    num_chunks=num_chunks,
                )
                for ctx in nonmax_contexts
            )
        if 3 in planned_steps:
            done[3] = _has_state_transition_meta(Path(output_dir))

        desc_pids = _descendant_pids(bp["pid"], children_map)
        desc_procs = [by_pid[pid] for pid in desc_pids if pid in by_pid]
        rebuild = next((p for p in desc_procs if "scripts/rebuild_vibethinker_hybrid.sh" in p["command"]), None)
        combine = next((p for p in desc_procs if "scripts/combine_vibethinker_infer_contexts.sh" in p["command"]), None)

        active_top_step = None
        active_output_dir = None
        active_context = None

        if combine:
            active_top_step = 3
            combine_tokens = _split_command(combine["command"])
            active_output_dir = _get_cli_arg(combine_tokens, "--output", output_dir)
        elif rebuild:
            rebuild_tokens = _split_command(rebuild["command"])
            active_context = _get_cli_arg(rebuild_tokens, "--context")
            active_output_dir = _get_cli_arg(rebuild_tokens, "--output")
            try:
                active_top_step = 1 if int(active_context) == int(max_context) else 2
            except Exception:
                active_top_step = 2

        if active_top_step is None:
            for s in planned_steps:
                if not done.get(s, False):
                    active_top_step = s
                    break
            if active_top_step is None and planned_steps:
                active_top_step = planned_steps[-1]

        if active_output_dir is None:
            if active_top_step == 1:
                active_output_dir = str(max_dir)
            elif active_top_step == 3:
                active_output_dir = output_dir

        candidates.append(
            {
                "pid": bp["pid"],
                "planned_steps": planned_steps,
                "done": done,
                "active_top_step": active_top_step,
                "active_output_dir": active_output_dir,
                "active_context": active_context,
                "max_context": max_context,
                "contexts": contexts,
                "context_root": context_root,
                "output_dir": output_dir,
            }
        )

    if not candidates:
        return None

    # Prefer candidates with an active nested command; else most recent PID.
    candidates.sort(
        key=lambda c: (
            0 if c.get("active_output_dir") else 1,
            -int(c.get("pid", 0)),
        )
    )
    return candidates[0]


def _build_ctx_progress_line(state: dict, width: int = 30) -> str:
    """Build top-level progress bar line for build_ctx_model steps."""
    planned = state.get("planned_steps", [])
    if not planned:
        return "  Build ctx: [pending]"
    done_map = state.get("done", {})
    active_step = state.get("active_top_step")
    done_count = sum(1 for s in planned if done_map.get(s, False))
    progress_units = float(done_count)
    if active_step in planned and not done_map.get(active_step, False):
        progress_units = min(float(len(planned)), progress_units + 0.5)
    filled = int(width * progress_units / len(planned))
    bar = "█" * filled + "░" * (width - filled)
    pct = (progress_units / len(planned)) * 100.0
    return f"  Build ctx: [{bar}] {pct:.0f}% ({done_count}/{len(planned)} steps done)"


def _is_build_ctx_complete(state: dict) -> bool:
    """Return True only when all planned build_ctx_model steps are complete."""
    if not state:
        return False
    planned = state.get("planned_steps", [])
    if not planned:
        return False
    done_map = state.get("done", {})
    return all(done_map.get(step, False) for step in planned)


def get_chunk_timings(output_dir: Path, cutoff_time: float) -> dict:
    """
    Analyze file creation times to estimate per-chunk duration.
    Returns dict with average times per step type.
    """
    timings = {}

    # Get all recent mlpackage files
    files = []
    for f in output_dir.glob("*.mlpackage"):
        try:
            mtime = f.stat().st_mtime
            if mtime > cutoff_time:
                files.append((f.name, mtime))
        except:
            pass

    if not files:
        return timings

    # Sort by time
    files.sort(key=lambda x: x[1])

    # Calculate time between consecutive files of same type
    ffn_times = []
    prefill_times = []
    ffn_rotate_times = []
    prefill_rotate_times = []

    prev_ffn = None
    prev_prefill = None
    prev_ffn_rotate = None
    prev_prefill_rotate = None

    for name, mtime in files:
        if '_FFN_' in name and 'rotate' not in name and '_PF_' not in name and 'chunk' in name:
            if prev_ffn:
                ffn_times.append(mtime - prev_ffn)
            prev_ffn = mtime
        elif 'prefill' in name and 'rotate' not in name and 'chunk' in name:
            if prev_prefill:
                prefill_times.append(mtime - prev_prefill)
            prev_prefill = mtime
        elif '_FFN_rotate' in name and 'chunk' in name:
            if prev_ffn_rotate:
                ffn_rotate_times.append(mtime - prev_ffn_rotate)
            prev_ffn_rotate = mtime
        elif 'prefill_rotate' in name and 'chunk' in name:
            if prev_prefill_rotate:
                prefill_rotate_times.append(mtime - prev_prefill_rotate)
            prev_prefill_rotate = mtime

    # Calculate averages
    if ffn_times:
        timings['3'] = sum(ffn_times) / len(ffn_times)
    if prefill_times:
        timings['4'] = sum(prefill_times) / len(prefill_times)
    if ffn_rotate_times:
        timings['4a'] = sum(ffn_rotate_times) / len(ffn_rotate_times)
    if prefill_rotate_times:
        timings['4b'] = sum(prefill_rotate_times) / len(prefill_rotate_times)

    # Use FFN time to estimate rotate if not available
    if '3' in timings:
        if '4a' not in timings:
            timings['4a'] = timings['3']  # Rotate takes similar time
        if '4' not in timings:
            timings['4'] = timings['3']  # Prefill similar to FFN

    return timings


def get_files_by_pattern(output_dir: Path, pattern: str) -> list:
    """Get files matching pattern in output directory."""
    files = list(output_dir.glob(f"*{pattern}*"))
    return sorted(files, key=lambda f: f.stat().st_mtime if f.exists() else 0)


def detect_step(output_dir: Path, num_chunks: int = 2, cutoff_time: float = 0) -> tuple:
    """
    Detect current conversion step based on files present.
    Returns (step_number, step_name, details, files_found)

    Args:
        cutoff_time: Only consider files modified after this timestamp
    """
    def is_recent(path: Path) -> bool:
        """Check if file was modified after cutoff."""
        if cutoff_time == 0:
            return True
        try:
            return path.stat().st_mtime > cutoff_time
        except:
            return False

    def filter_recent(files: list) -> list:
        """Filter to only recent files."""
        return [f for f in files if is_recent(f)]

    # Check for meta.yaml (step 7 complete) - must be recent
    meta_path = output_dir / "meta.yaml"
    if meta_path.exists() and is_recent(meta_path):
        mlmodelc_files = filter_recent(list(output_dir.glob("*.mlmodelc")))
        if mlmodelc_files:
            return (8, "Testing or Complete", f"{len(mlmodelc_files)} compiled models", mlmodelc_files)

    # Check for .mlmodelc files (step 6 - compilation)
    mlmodelc_files = filter_recent(list(output_dir.glob("*.mlmodelc")))
    if mlmodelc_files:
        return (6, "Compiling Models", f"{len(mlmodelc_files)} compiled", mlmodelc_files)

    # Check for combined FFN_PF files (step 5)
    combined_files = filter_recent(get_files_by_pattern(output_dir, "_FFN_PF_"))
    mlpackage_combined = [f for f in combined_files if f.suffix == ".mlpackage"]
    if mlpackage_combined:
        return (5, "Combining Models", f"{len(mlpackage_combined)} combined", mlpackage_combined)

    # Check for rotate prefill chunks (step 4b - Gemma3 only)
    # Pattern: gemma3_prefill_rotate_lut4_chunk_01of02.mlpackage
    pf_rotate_files = filter_recent(get_files_by_pattern(output_dir, "prefill_rotate"))
    pf_rotate_mlpackage = [f for f in pf_rotate_files if f.suffix == ".mlpackage" and "chunk" in f.name]
    if pf_rotate_mlpackage:
        return ("4b", "Converting Prefill Rotate", f"{len(pf_rotate_mlpackage)}/{num_chunks} chunks", pf_rotate_mlpackage)

    # Check for rotate FFN chunks (step 4a - Gemma3 only)
    # Pattern: gemma3_FFN_rotate_lut4_chunk_01of02.mlpackage
    ffn_rotate_files = filter_recent(get_files_by_pattern(output_dir, "_FFN_rotate"))
    ffn_rotate_mlpackage = [f for f in ffn_rotate_files if f.suffix == ".mlpackage" and "chunk" in f.name]
    if ffn_rotate_mlpackage:
        return ("4a", "Converting FFN Rotate", f"{len(ffn_rotate_mlpackage)}/{num_chunks} chunks", ffn_rotate_mlpackage)

    # Check for prefill chunks (step 4)
    # Pattern: gemma3_prefill_lut4_chunk_01of02.mlpackage (but NOT prefill_rotate)
    pf_files = filter_recent(get_files_by_pattern(output_dir, "prefill"))
    pf_mlpackage = [f for f in pf_files if f.suffix == ".mlpackage" and "chunk" in f.name and "rotate" not in f.name]
    if pf_mlpackage:
        return (4, "Converting Prefill", f"{len(pf_mlpackage)}/{num_chunks} chunks", pf_mlpackage)

    # Check for FFN chunks (step 3)
    # Pattern: gemma3_FFN_lut4_chunk_01of02.mlpackage (but NOT FFN_rotate or FFN_PF)
    ffn_files = filter_recent(get_files_by_pattern(output_dir, "_FFN_"))
    ffn_mlpackage = [f for f in ffn_files if f.suffix == ".mlpackage" and "chunk" in f.name and "rotate" not in f.name and "_PF_" not in f.name]
    if ffn_mlpackage:
        return (3, "Converting FFN", f"{len(ffn_mlpackage)}/{num_chunks} chunks", ffn_mlpackage)

    # Check for LM head (step 2)
    lm_head_files = filter_recent(get_files_by_pattern(output_dir, "_lm_head"))
    lm_mlpackage = [f for f in lm_head_files if f.suffix == ".mlpackage"]
    if lm_mlpackage:
        return (2, "Converting LM Head", "complete", lm_mlpackage)

    # Check for embeddings (step 1)
    embed_files = filter_recent(get_files_by_pattern(output_dir, "_embeddings"))
    embed_mlpackage = [f for f in embed_files if f.suffix == ".mlpackage"]
    if embed_mlpackage:
        return (1, "Converting Embeddings", "complete", embed_mlpackage)

    return (0, "Starting", "waiting for first file", [])


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}PB"


def get_disk_space(path: Path) -> tuple:
    """Get free and total disk space for the volume containing path."""
    import shutil
    try:
        usage = shutil.disk_usage(path)
        return usage.free, usage.total
    except Exception:
        return None, None


def get_system_disk_space() -> tuple:
    """Get free and total disk space for system volume (/)."""
    return get_disk_space(Path("/"))


def get_current_output_size(output_dir: Path) -> int:
    """Get total size of all files in output directory."""
    total = 0
    try:
        for item in output_dir.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        pass
    return total


def estimate_final_size(output_dir: Path, num_chunks: int) -> tuple:
    """
    Estimate final output size based on current files and remaining steps.
    Returns (current_size, estimated_total) in bytes.
    """
    current_size = get_current_output_size(output_dir)

    # Get sizes of existing mlpackage files to estimate remaining
    mlpackage_sizes = []
    for f in output_dir.glob("*.mlpackage"):
        try:
            size = sum(p.stat().st_size for p in f.rglob("*") if p.is_file())
            mlpackage_sizes.append((f.name, size))
        except Exception:
            pass

    if not mlpackage_sizes:
        # No files yet, use rough estimate based on typical models
        # 4B model with LUT4: ~8GB total output
        return current_size, current_size + 8 * 1024 * 1024 * 1024

    # Calculate average chunk size from FFN chunks
    chunk_sizes = [size for name, size in mlpackage_sizes if 'chunk' in name and '_FFN_' in name]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 500 * 1024 * 1024

    # Count what's done
    ffn_count = len([1 for name, _ in mlpackage_sizes if '_FFN_' in name and 'rotate' not in name and '_PF_' not in name])
    pf_count = len([1 for name, _ in mlpackage_sizes if 'prefill' in name and 'rotate' not in name])
    ffn_rot_count = len([1 for name, _ in mlpackage_sizes if '_FFN_rotate' in name])
    pf_rot_count = len([1 for name, _ in mlpackage_sizes if 'prefill_rotate' in name])
    combined_count = len([1 for name, _ in mlpackage_sizes if '_FFN_PF_' in name])

    # Estimate remaining based on what's missing
    estimated_remaining = 0

    # Estimate missing mlpackages
    if ffn_count < num_chunks:
        estimated_remaining += (num_chunks - ffn_count) * avg_chunk_size
    if pf_count < num_chunks:
        estimated_remaining += (num_chunks - pf_count) * avg_chunk_size
    if ffn_rot_count < num_chunks:
        estimated_remaining += (num_chunks - ffn_rot_count) * avg_chunk_size
    if pf_rot_count < num_chunks:
        estimated_remaining += (num_chunks - pf_rot_count) * avg_chunk_size

    # Combined models (roughly 2x chunk size per combined)
    if combined_count < num_chunks:
        estimated_remaining += (num_chunks - combined_count) * avg_chunk_size * 2

    # Compiled models (.mlmodelc) - roughly same size as mlpackage
    mlmodelc_count = len(list(output_dir.glob("*.mlmodelc")))
    expected_mlmodelc = num_chunks * 2 + 2  # FFN_PF chunks + embed + lmhead
    if mlmodelc_count < expected_mlmodelc:
        estimated_remaining += (expected_mlmodelc - mlmodelc_count) * avg_chunk_size

    return current_size, current_size + int(estimated_remaining)


def get_file_age(filepath: Path) -> float:
    """Get age of file in seconds."""
    if filepath.exists():
        return time.time() - filepath.stat().st_mtime
    return 0


def estimate_remaining(current_step, num_chunks: int, step_durations: dict, is_gemma3_rotate: bool, current_chunk: int = 1) -> float:
    """Estimate remaining time based on current step and average durations."""
    remaining = 0

    if is_gemma3_rotate:
        step_order = ["1", "2", "3", "4", "4a", "4b", "5", "6", "7", "8"]
    else:
        step_order = ["1", "2", "3", "4", "5", "6", "7", "8"]

    current_str = str(current_step)
    try:
        current_idx = step_order.index(current_str)
    except ValueError:
        current_idx = 0

    # Add remaining chunks for current step
    if current_str in ["3", "4", "4a", "4b"]:
        remaining_chunks = num_chunks - current_chunk
        if remaining_chunks > 0:
            remaining += remaining_chunks * step_durations.get(current_str, 300)

    # Add future steps
    for step in step_order[current_idx + 1:]:
        base_duration = step_durations.get(step, 60)
        # Multiply by chunks for chunk-based steps
        if step in ["3", "4", "4a", "4b"]:
            base_duration *= num_chunks
        elif step == "6":
            base_duration *= (num_chunks * 2 + 2)  # FFN + PF chunks + embed + lmhead
        remaining += base_duration

    return remaining


def get_running_process(output_dir: Path) -> tuple:
    """Check what conversion process is currently running."""
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.split('\n'):
            if 'converter' in line and str(output_dir) in line and 'grep' not in line:
                # Extract --part value
                if '--part ' in line:
                    part = line.split('--part ')[1].split()[0]
                    part_map = {
                        '1': (1, 'Embeddings'),
                        '3': (2, 'LM Head'),
                        '2': (3, 'FFN'),
                        '2_prefill': (4, 'Prefill'),
                        '2_rotate': ('4a', 'FFN Rotate'),
                        '2_prefill_rotate': ('4b', 'Prefill Rotate'),
                    }
                    if part in part_map:
                        return part_map[part]
        return None
    except:
        return None


def monitor(output_dir: Path, num_chunks: int = 2, refresh_interval: float = 2.0, since_minutes: int = 60):
    """Main monitoring loop."""
    print(f"\n{'='*70}")
    print(f"  CONVERSION PROGRESS MONITOR")
    print(f"  Output: {output_dir}")
    print(f"  Chunks: {num_chunks}")
    print(f"{'='*70}\n")
    print("Press Ctrl+C to exit\n")

    # Keep requested output as fallback, but allow dynamic active output tracking
    fallback_output_dir = output_dir
    active_output_dir = output_dir

    # Only consider files modified in the last N minutes as part of current run
    cutoff_time = time.time() - (since_minutes * 60)

    # Find earliest file creation time for this run
    earliest_file_time = None
    for f in output_dir.glob("*.mlpackage"):
        try:
            mtime = f.stat().st_mtime
            if mtime > cutoff_time:
                if earliest_file_time is None or mtime < earliest_file_time:
                    earliest_file_time = mtime
        except Exception:
            pass

    # Use earliest file time as start, or current time if no files yet
    start_time = earliest_file_time if earliest_file_time else time.time()
    step_start_times = {}
    step_durations = DEFAULT_STEP_DURATIONS.copy()
    last_step = None
    is_gemma3_rotate = False

    try:
        while True:
            # If scripts/build_ctx_model is running, track its active nested output path.
            build_ctx_state = detect_build_ctx_state(num_chunks=max(1, num_chunks))
            if build_ctx_state:
                active_candidate = (
                    build_ctx_state.get("active_output_dir")
                    or build_ctx_state.get("output_dir")
                )
                if active_candidate:
                    active_output_dir = Path(active_candidate)
                else:
                    active_output_dir = fallback_output_dir
            else:
                active_output_dir = fallback_output_dir

            # Detect current step from files
            step, step_name, details, files = detect_step(active_output_dir, num_chunks, cutoff_time)
            current_chunk_num = 1  # Default to chunk 1

            # Also check what process is currently running
            running = get_running_process(active_output_dir)
            if running:
                running_step, running_name = running
                # If running step is ahead of detected step, use running info
                step_order = ["0", "1", "2", "3", "4", "4a", "4b", "5", "6", "7", "8"]
                try:
                    if step_order.index(str(running_step)) > step_order.index(str(step)):
                        step = running_step
                        step_name = f"{running_name} (in progress)"
                        # Determine which chunk we're on based on existing files
                        chunk_num = 1
                        if running_step == "4a":  # FFN Rotate
                            existing = len([1 for f in active_output_dir.glob("*_FFN_rotate*chunk*.mlpackage")])
                            chunk_num = existing + 1
                        elif running_step == "4b":  # Prefill Rotate
                            existing = len([1 for pf in active_output_dir.glob("*prefill_rotate*chunk*.mlpackage")
                                          if pf.stat().st_mtime > cutoff_time])
                            chunk_num = existing + 1
                        elif running_step == 3:  # FFN
                            existing = len([1 for f in active_output_dir.glob("*_FFN_*chunk*.mlpackage")
                                          if 'rotate' not in f.name and '_PF_' not in f.name])
                            chunk_num = existing + 1
                        elif running_step == 4:  # Prefill
                            existing = len([1 for f in active_output_dir.glob("*prefill*chunk*.mlpackage")
                                          if 'rotate' not in f.name])
                            chunk_num = existing + 1
                        details = f"chunk {chunk_num}/{num_chunks}"
                        current_chunk_num = chunk_num
                except ValueError:
                    pass

            # Detect if this is a Gemma3 rotate conversion
            if str(step) in ["4a", "4b"]:
                is_gemma3_rotate = True

            # Track step transitions
            if step != last_step:
                if last_step is not None and str(last_step) in step_start_times:
                    # Record actual duration of completed step
                    actual_duration = time.time() - step_start_times[str(last_step)]
                    step_durations[str(last_step)] = actual_duration

                step_start_times[str(step)] = time.time()
                last_step = step

            # Calculate elapsed time for current step
            step_elapsed = 0
            if str(step) in step_start_times:
                step_elapsed = time.time() - step_start_times[str(step)]

            total_elapsed = time.time() - start_time

            # Update step durations with actual timings from this run
            actual_timings = get_chunk_timings(active_output_dir, cutoff_time)
            for k, v in actual_timings.items():
                step_durations[k] = v

            # If current step has been running longer than estimated, update estimate
            # This handles the case where we're on chunk 1 and it's taking longer than expected
            if step_elapsed > step_durations.get(str(step), 0):
                step_durations[str(step)] = step_elapsed * 1.1  # Add 10% buffer

            # Estimate remaining time
            remaining = estimate_remaining(step, num_chunks, step_durations, is_gemma3_rotate, current_chunk_num)

            # Get most recent file (only from recent files)
            recent_file = None
            recent_files = [f for f in files if f.exists() and f.stat().st_mtime > cutoff_time]
            if recent_files:
                recent_file = max(recent_files, key=lambda f: f.stat().st_mtime)

            # Clear screen and print status
            os.system('clear' if os.name != 'nt' else 'cls')

            print(f"{'='*70}")
            print(f"  CONVERSION PROGRESS MONITOR")
            print(f"{'='*70}")
            print()

            if build_ctx_state:
                print("  build_ctx_model:")
                print(_build_ctx_progress_line(build_ctx_state))
                planned = build_ctx_state.get("planned_steps", [])
                done_map = build_ctx_state.get("done", {})
                active_top = build_ctx_state.get("active_top_step")
                max_ctx = build_ctx_state.get("max_context")
                contexts = build_ctx_state.get("contexts", []) or []
                nonmax_contexts = [c for c in contexts if c != max_ctx]
                for top_step in planned:
                    if top_step == active_top and not _is_build_ctx_complete(build_ctx_state):
                        mark = "[>]"
                    elif done_map.get(top_step, False):
                        mark = "[x]"
                    else:
                        mark = "[ ]"
                    name = BUILD_CTX_STEP_NAMES.get(top_step, f"Step {top_step}")
                    extra = ""
                    if top_step == 1 and max_ctx is not None:
                        extra = f" (ctx={max_ctx})"
                    elif top_step == 2 and nonmax_contexts:
                        active_ctx = build_ctx_state.get("active_context")
                        if top_step == active_top and active_ctx:
                            extra = f" (active ctx={active_ctx})"
                        else:
                            extra = f" ({len(nonmax_contexts)} contexts)"
                    elif top_step == 3:
                        extra = f" (output={build_ctx_state.get('output_dir')})"
                    print(f"    {mark} Step {top_step}: {name}{extra}")
                print()

            # Progress bar
            if is_gemma3_rotate:
                total_steps = 10
                step_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "4a": 5, "4b": 6, "5": 7, "6": 8, "7": 9, "8": 10}
            else:
                total_steps = 8
                step_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8}

            current_progress = step_map.get(str(step), 0)
            bar_width = 50
            filled = int(bar_width * current_progress / total_steps)
            bar = "█" * filled + "░" * (bar_width - filled)
            percent = (current_progress / total_steps) * 100

            print(f"  Progress: [{bar}] {percent:.0f}%")
            print()
            print(f"  Step {step}: {step_name}")
            print(f"  Details: {details}")
            print()
            print(f"  Step elapsed:    {format_duration(step_elapsed)}")
            print(f"  Total elapsed:   {format_duration(total_elapsed)}")
            print(f"  Est. remaining:  {format_duration(remaining)}")
            print()

            if recent_file:
                age = get_file_age(recent_file)
                print(f"  Latest file: {recent_file.name}")
                print(f"  Since last file: {format_duration(age)}")

            # Show observed timings if available
            if actual_timings:
                print()
                print("  Observed chunk times:")
                for k, v in sorted(actual_timings.items()):
                    step_names = {'3': 'FFN', '4': 'Prefill', '4a': 'FFN Rotate', '4b': 'PF Rotate'}
                    name = step_names.get(k, k)
                    print(f"    {name}: {format_duration(v)}/chunk")

            # Show disk space and size estimates
            print()
            output_free, output_total = get_disk_space(active_output_dir)
            sys_free, sys_total = get_system_disk_space()
            current_size, estimated_total = estimate_final_size(active_output_dir, num_chunks)

            print(f"  Current size:    {format_bytes(current_size)}")
            print(f"  Est. final size: {format_bytes(estimated_total)}")

            if output_free is not None:
                output_pct = (output_free / output_total) * 100 if output_total else 0
                space_ok = "✓" if output_free > (estimated_total - current_size) else "⚠️ LOW"
                print(f"  Output volume:   {format_bytes(output_free)} free ({output_pct:.0f}%) {space_ok}")
            if sys_free is not None and str(active_output_dir).startswith('/Volumes'):
                # Only show system disk if output is on external volume
                sys_pct = (sys_free / sys_total) * 100 if sys_total else 0
                print(f"  System volume:   {format_bytes(sys_free)} free ({sys_pct:.0f}%)")

            print()
            if build_ctx_state and active_output_dir != fallback_output_dir:
                print(f"  Output (active): {active_output_dir}")
                print(f"  Output (fallback): {fallback_output_dir}")
            else:
                print(f"  Output: {active_output_dir}")
            print()
            print("  Press Ctrl+C to exit")
            print(f"{'='*70}")

            # Check if complete.
            # For build_ctx_model, only exit when all planned top-level steps are done.
            if build_ctx_state:
                if _is_build_ctx_complete(build_ctx_state):
                    print("\n  ✓ BUILD_CTX CONVERSION COMPLETE!")
                    print(f"    Total time: {format_duration(total_elapsed)}")
                    break
            else:
                # Legacy single-output conversion completion (meta.yaml must be recent)
                meta_path = active_output_dir / "meta.yaml"
                if step == 8 and meta_path.exists() and meta_path.stat().st_mtime > cutoff_time:
                    print("\n  ✓ CONVERSION COMPLETE!")
                    print(f"    Total time: {format_duration(total_elapsed)}")
                    break

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def get_config_from_meta(output_dir: Path) -> dict:
    """Try to read config from meta_progress.yaml or meta.yaml."""
    if yaml is None:
        return {}

    config = {}

    # First try meta_progress.yaml (created at start of conversion)
    progress_path = output_dir / "meta_progress.yaml"
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                meta = yaml.safe_load(f)
            if 'conversion' in meta:
                conv = meta['conversion']
                config['chunks'] = conv.get('num_chunks', 2)
                config['context_length'] = conv.get('context_length')
                config['batch_size'] = conv.get('batch_size')
                config['prefix'] = conv.get('prefix')
                config['architecture'] = conv.get('architecture')
                config['start_time'] = conv.get('start_time')
                config['lut_part2'] = conv.get('lut_part2')
                return config
        except Exception:
            pass

    # Fall back to meta.yaml (from previous run)
    meta_path = output_dir / "meta.yaml"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            # Extract chunk count from model filenames
            if 'models' in meta:
                for model in meta['models']:
                    name = model.get('name', '')
                    if 'chunk' in name and 'of' in name:
                        # Parse chunk_01of02 -> 2 chunks
                        match = re.search(r'of(\d+)', name)
                        if match:
                            config['chunks'] = int(match.group(1))
                            break
            return config
        except Exception:
            pass
    return {}


def find_running_conversion() -> str:
    """Find output directory of running convert_model.sh process."""
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.split('\n'):
            if 'convert_model.sh' in line and '--output' in line:
                # Extract --output value
                parts = line.split('--output')
                if len(parts) > 1:
                    # Get the path after --output
                    output_part = parts[1].strip().split()[0]
                    return output_part
            elif 'converter' in line and '--output' in line:
                # Also check for direct converter runs
                parts = line.split('--output')
                if len(parts) > 1:
                    output_part = parts[1].strip().split()[0]
                    return output_part
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Monitor conversion progress")
    parser.add_argument("--output", "-o", default=None, help="Output directory (auto-detected if not specified)")
    parser.add_argument("--chunks", "-c", type=int, default=None, help="Number of chunks (auto-detected from meta.yaml if not specified)")
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="Refresh interval in seconds (default: 2)")
    parser.add_argument("--since", "-s", type=int, default=60, help="Only consider files modified in last N minutes (default: 60)")
    args = parser.parse_args()

    # Auto-detect output directory from running process if not specified
    if args.output is None:
        build_ctx_state = detect_build_ctx_state(num_chunks=max(1, args.chunks or 3))
        if build_ctx_state:
            detected = build_ctx_state.get("active_output_dir") or build_ctx_state.get("output_dir")
            if detected:
                active_step = build_ctx_state.get("active_top_step")
                print(f"Auto-detected running build_ctx_model (step {active_step}): {detected}")
                args.output = detected
        if args.output is None:
            detected = find_running_conversion()
            if detected:
                print(f"Auto-detected running conversion: {detected}")
                args.output = detected
            else:
                print("Error: No running conversion found. Specify --output manually.")
                print("Usage: python monitor_conversion.py --output /path/to/output")
                return

    output_dir = Path(args.output)
    if not output_dir.exists():
        print(f"Warning: Output directory does not exist yet: {output_dir}")
        print("Will wait for it to be created...")
        while not output_dir.exists():
            time.sleep(1)

    # Auto-detect config from meta_progress.yaml or meta.yaml
    config = get_config_from_meta(output_dir)

    num_chunks = args.chunks
    if num_chunks is None:
        num_chunks = config.get('chunks', 2)

    # Show detected config
    if config:
        print("Config detected from meta file:")
        if 'chunks' in config:
            print(f"  Chunks: {config['chunks']}")
        if 'context_length' in config:
            print(f"  Context: {config['context_length']}")
        if 'architecture' in config:
            print(f"  Architecture: {config['architecture']}")
        if 'lut_part2' in config:
            print(f"  LUT: {config['lut_part2']}")
        if 'start_time' in config:
            print(f"  Started: {config['start_time']}")
        print()

    monitor(output_dir, num_chunks, args.interval, args.since)


if __name__ == "__main__":
    main()
