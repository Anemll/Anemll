## Skills

A skill is a set of local instructions stored in a `SKILL.md` file.

### Available skills

- `ane-divergence-analysis`
  - Compare PyTorch vs CoreML/ANE and HuggingFace Transformers vs CoreML for decoder-only LLMs in this repo.
  - Use for token/logit divergence, instability, ANE lowering issues, and parity metrics (KL, correlation, entropy, repetition).
  - File: `/Users/anemll/.codex/skills/custom/ane-divergence-analysis/SKILL.md`

- `ane-lowering-plan`
  - Create an ANE-legal CoreML/MIL lowering plan for decoder-only Transformers from HF or JAX/Flax.
  - Use for conversion/debugging on ANE, KV-cache/state contract issues, sliding-window/global attention, and chunked FFN.
  - File: `/Users/anemll/.codex/skills/custom/ane-lowering-plan/SKILL.md`

- `anemll-macos-agent`
  - Control macOS UI via AnemllAgentHost HTTP API for screenshots, clicks, typing, and window/screen automation.
  - Use only when GUI interaction is required.
  - File: `/Users/anemll/.codex/skills/custom/anemll-macos-agent/SKILL.md`

### Trigger rules

- If a user names one of these skills directly or the task clearly matches it, use that skill for the turn.
- If multiple apply, use the minimal set and state the order briefly.
- If a skill file is missing/unreadable, state it and continue with the best fallback.

### Usage notes

- Open the selected `SKILL.md` and read only what is needed for the current task.
- Resolve relative paths from the skill directory first.
- Prefer scripts/templates referenced by the skill over re-implementing workflows manually.
