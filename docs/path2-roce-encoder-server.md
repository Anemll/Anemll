# Path 2: Mac RoCE Encoder Server Procedure

Runbook for the Mac-side virtual worker that serves encoder-only prefill over RoCE-macos. Unknown CLI flags, buffer sizes, and KV@240k footprint are marked **TBD**. Do not treat this document as evidence that the procedure was executed.

## Architecture

M5 Max runs DeepSeek-V4.1 Q2 encoder-only prefill (L0–19), packs KV, and fire-and-forget RDMA_WRITEs it to Sparks over RoCE-macos; Sparks are decode-only clients and never see prefill.

## Components

| Component | Role |
| --- | --- |
| Mac GGUF encoder runtime + KV packer | DeepSeek-V4.1 Q2 encoder-only prefill on L0–19; pack versioned KV buffer |
| Sparks decode-only runtime | Decode clients only; never run prefill |
| RoCE-macos transport | RC / RDMA_WRITE path (RMOE-style) between Mac and Sparks |
| Sequencer / control | Request/imm or control message; request ids; optional async ack only |

## Memory budget (established)

- 128 GiB unified on M5 Max
- 74.64 GiB encoder weights
- ~53 GiB headroom for OS + 240k KV
- If weights + KV + ~16 GiB OS > 128 GiB → layer-swap contingency
- Exact KV@240k size: **TBD** (do not invent)

## Link class (already documented)

Thunderbolt ConnectX RoCE ~50+ Gbit/s, ~11 µs RTT. Do not invent new measurements.

## Open constraint

Spark has no GPU doorbell / GPUDirect UAR. Host-staged CPU proxy via libmlx5.

## Implementation discipline

- Reuse RoCE-macos RC / RDMA_WRITE (RMOE-style).
- Versioned buffer: magic + version + header CRC32C pattern.
- Optional Phase order: file → stub → live.
- **Fire-and-forget handoff:** return to listening as soon as the local WRITE CQ completes. Do **not** block on decode-complete ack. Any ack is async-only.

---

## Phase 1: Mac-side RoCE encoder server

Build and run the Mac virtual worker that listens for Spark requests, runs encoder prefill, packs KV, and hands it off.

### Server loop

`listen → prefill → pack → WRITE-done → listen again`

### Procedure

1. **Stand up the Mac encoder server** on the M5 Max (exact CLI flags **TBD**). Bind the RoCE-macos listener and confirm it is accepting requests.
2. **Receive** an incoming RoCE signal/call from a Spark client over RoCE-macos (request/imm or control message).
3. **Prefill** DeepSeek-V4.1 Q2 encoder-only on L0–19. Produce the KV cache. Sparks never see this step.
4. **Pack** the KV into a versioned buffer (magic + version + header CRC32C). Exact on-wire layout **TBD**.
5. **RDMA_WRITE** the packed buffer to the Spark(s). Wait only for the **local WRITE CQ** to complete.
6. **Return to listen immediately.** Do not wait for decode-complete. Any ack is async-only and must not gate the listen loop.
7. Repeat from step 2.

### Phase 1 success criteria (server)

- Sparks GPU idle during Mac prefill
- KV checksum match after handoff
- Correct decode from handed-off KV only
- Transfer seconds-scale at ≥50 Gbit/s (documented link class)
- No CB decode cliff on Sparks
- Listen loop is free as soon as local WRITE CQ completes

---

## Phase 2: Standalone Spark client test (Spark → Mac RoCE)

*Spin-off agent task: pick up and execute in a separate agent; do not run from the session that owns the Path 2 server build/doc work.*

Documentation-only client test. Phase 2 is a **manual/scripted client test procedure in this doc**, not “launch another agent” from the Path 2 build session. This section does not claim the test was run.

### Preconditions

- Mac encoder server from Phase 1 is listening.
- RoCE-macos link is up (documented class: Thunderbolt ConnectX RoCE ~50+ Gbit/s, ~11 µs RTT).
- Spark is in the decode-only client role (never runs prefill).

### Test narrative

From a Spark: send a RoCE request signal to the Mac encoder server → server runs a **short** encoder prefill on L0–19 (DeepSeek-V4.1 Q2) → packs KV → **fire-and-forget** sends it back over RoCE → Spark verifies receipt with **checksum match**.

### Exact steps

1. **Bring up the Mac encoder server (Phase 1).** Confirm it is in `listen`. Exact server CLI **TBD**.
2. **Confirm the RoCE-macos link.** Use the documented class only (~50+ Gbit/s, ~11 µs RTT). Do not invent new measurements.
3. **On Spark: post the request / signal to Mac.** Exact client CLI flags **TBD**. Send a RoCE request/imm or control message with a request id.
4. **On Mac: observe recv.** Request is seen (request/imm or control on the wire).
5. **On Mac: short encoder prefill** DeepSeek-V4.1 Q2 L0–19 (short sequence; exact token count **TBD**).
6. **On Mac: pack** the KV into the versioned buffer (magic + version + header CRC32C).
7. **On Mac: RDMA_WRITE** the buffer; wait for **local WRITE CQ complete** only; **return to listen**. Must not wait for decode ack.
8. **On Spark: receive** the versioned KV buffer.
9. **On Spark: verify** magic/version/length, then **checksum match** (expected vs actual).
10. **Optional:** short decode from handed-off KV only (Spark GPU; no Mac prefill replay).

### Expected signals at each stage

| Stage | Expected signal |
| --- | --- |
| Spark posts request | Request/imm or control message on the wire |
| Mac recv | Request seen on Mac (request id logged) |
| Mac prefill | Short L0–19 Q2 prefill completes; Sparks GPU idle |
| Mac pack | Versioned KV buffer ready (magic + version + header CRC32C) |
| Mac WRITE | Local WRITE CQ complete; then return to listen |
| Spark recv | Receive of versioned KV buffer completes |
| Spark verify | Checksum OK; version/magic OK; payload length matches header |

### Pass / fail

**Pass**

- Version/magic OK
- Checksum match
- Payload length matches header
- Mac listen loop not blocked on decode-complete ack (fire-and-forget held)
- Optional: short decode from handed-off KV only succeeds

**Fail**

- No receive on Mac
- Checksum mismatch
- Truncated buffer (length ≠ header)
- Mac blocked waiting for decode-complete ack before accepting the next request (violates fire-and-forget)

### What to log

- Timestamps at: Spark post, Mac recv, prefill start/end, pack done, WRITE CQ complete, Mac back-to-listen, Spark recv complete, verify done
- Request id
- Bytes sent / received
- Checksum expected vs actual
- CQ wait µs (local WRITE completion only)
- Whether fire-and-forget held (Mac listening again before any decode-complete ack)

### CLI (TBD)

Exact Mac server flags, Spark client flags, and checksum command names are **TBD**. Do not invent them.

---

## TBD

- Exact KV@240k size
- Exact on-wire versioned-buffer layout beyond magic + version + header CRC32C
- Exact Mac server and Spark client CLI flags
- Short-prefill token count for Phase 2
