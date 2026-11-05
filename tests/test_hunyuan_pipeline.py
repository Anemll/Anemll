"""Exercise shell architecture routing and generated runtime metadata offline."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    shutil.which("jq") is None, reason="jq is required by conversion scripts"
)


@pytest.fixture
def shell_model(tmp_path):
    model = tmp_path / "checkpoint"
    model.mkdir()
    (model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "calls.jsonl"
    python = bin_dir / "python3"
    python.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['-m', 'anemll.ane_converter.hunyuan_converter']:\n"
        "    with open(os.environ['PYTHON_CAPTURE'], 'a') as f: f.write(json.dumps(args) + '\\n')\n"
        "else:\n"
        "    os.execv("
        + repr(sys.executable)
        + ", ["
        + repr(sys.executable)
        + "] + args)\n"
    )
    python.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        PYTHON_CAPTURE=str(capture),
        ANEMLL_AUTO_VENV="0",
    )
    return model, capture, env


@pytest.mark.parametrize(
    "script,step,part",
    [
        ("convert_model.sh", "1", "1"),
        ("convert_model.sh", "2", "3"),
        ("convert_model.sh", "3", "2"),
        ("convert_model.sh", "4", "2_prefill"),
        ("convert_monolith.sh", "1", "monolithic"),
        ("convert_monolith.sh", "2", "monolithic_prefill"),
    ],
)
@pytest.mark.parametrize("architecture_only", [False, True])
@pytest.mark.parametrize("pad", ["0", "1"])
def test_shell_routes_hunyuan(
    shell_model, tmp_path, script, step, part, architecture_only, pad
):
    model, capture, env = shell_model
    config = {"vocab_size": 120818, "architectures": ["HunYuanDenseV1ForCausalLM"]}
    if not architecture_only:
        config["model_type"] = "hunyuan_v1_dense"
    (model / "config.json").write_text(json.dumps(config))
    command = [
        "bash",
        str(ROOT / "anemll/utils" / script),
        "--model",
        str(model),
        "--output",
        str(tmp_path / "output"),
        "--skip-check",
        "--only",
        step,
        "--context",
        "128",
        "--batch",
        "16",
        "--state-pad",
        pad,
    ]
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 1
    args = calls[0]
    assert args[:2] == ["-m", "anemll.ane_converter.hunyuan_converter"]
    assert args[args.index("--part") + 1] == part
    assert args[args.index("--prefix") + 1] == "hunyuan"
    assert args[args.index("--context-length") + 1] == "128"
    assert args[args.index("--state-pad") + 1] == pad


@pytest.mark.parametrize(
    "script,step", [("convert_model.sh", "7"), ("convert_monolith.sh", "5")]
)
def test_shell_preserves_tokenizer_and_head_layout(shell_model, tmp_path, script, step):
    model, _, env = shell_model
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "hunyuan_v1_dense",
                "vocab_size": 120818,
            }
        )
    )
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "anemll/utils" / script),
            "--model",
            str(model),
            "--output",
            str(output),
            "--skip-check",
            "--only",
            step,
            "--argmax",
            "--state-pad",
            "1",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((output / "config.json").read_text()) == {
        "model_type": "hunyuan_v1_dense",
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    metadata = (output / "meta.yaml").read_text()
    assert "architecture: hunyuan_v1_dense" in metadata
    assert "split_lm_head: 16" in metadata
    assert "argmax_in_model: true" in metadata
    assert "lm_head_chunk_sizes:" in metadata

    assert "position_mask_prefill: true" in metadata
    assert "    update_mask_prefill: true" not in metadata
    assert "    prefill_dynamic_slice: true" not in metadata
    assert "state_pad: 1" in metadata
    assert "state_length: 513" in metadata


@pytest.mark.parametrize("script", ["convert_model.sh", "convert_monolith.sh"])
@pytest.mark.parametrize(
    "model_type,pad,message",
    [
        ("hunyuan_v1_dense", "2", "--state-pad must be 0 or 1"),
        ("qwen3", "1", "--state-pad is only supported for Hunyuan"),
    ],
)
def test_shell_rejects_invalid_state_padding(
    shell_model, tmp_path, script, model_type, pad, message
):
    model, capture, env = shell_model
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "anemll/utils" / script),
            "--model",
            str(model),
            "--output",
            str(tmp_path / "output"),
            "--skip-check",
            "--only",
            "1",
            "--state-pad",
            pad,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert message in result.stdout
    assert not capture.exists()
