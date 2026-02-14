# TODO: Gemma3 Split KV Cache State Fixes

## Issues Found

### Issue 1: Hardcoded 512 in convert_model.sh
- **Lines 491, 526**: Check `$CONTEXT_LENGTH -gt 512` (hardcoded!)
- **Problem**: 4B model has sliding_window=1024, not 512
- **Impact**: Generates unnecessary rotate functions when context <= sliding_window

### Issue 2: State Declaration (Conversion Error)
- Declaring BOTH states for ALL chunks causes "unused state" error
- **Fix**: Each chunk declares only what it uses

### Issue 3: State Passing (Inference in chat.py)
- Chunks without global layers → pass only `kv_cache_local`
- Chunks with global layers → pass both caches
- Need to sync `kv_cache_local` between different state objects

---

## Fixes Required

### 1. convert_model.sh - Extract sliding_window from config

**Add after line ~256 (after MODEL_PATH is set):**
```bash
# Get sliding_window from model config.json (default 512 for 1B, 1024 for 4B)
SLIDING_WINDOW=$(python3 -c "import json; print(json.load(open('$MODEL_PATH/config.json')).get('sliding_window', 512))" 2>/dev/null || echo "512")
echo "Detected sliding_window: $SLIDING_WINDOW"
```

**Replace line 491:**
```bash
# OLD: if [ $CONTEXT_LENGTH -gt 512 ]; then
# NEW:
if [ $CONTEXT_LENGTH -gt $SLIDING_WINDOW ]; then
```

**Replace line 526:**
```bash
# OLD: if [ $CONTEXT_LENGTH -gt 512 ]; then
# NEW:
if [ $CONTEXT_LENGTH -gt $SLIDING_WINDOW ]; then
```

### 2. gemma3_converter.py - Revert State Declarations

**GetTransformerStates (~line 311-357):**
- Revert to conditional: only declare states that chunk uses
- Chunk without global layers → only `kv_cache_local`
- Chunk with global layers → both caches

**FFNWrapper.forward (~line 1164-1171):**
- Remove dummy touch code

**PrefillWrapper.forward (~line 1290-1296):**
- Remove dummy touch code

### 3. chat.py - Per-Chunk State Handling

**detect_cache_type():**
- Already detects per-chunk what states are declared ✓

**create_unified_state():**
- Create two state types: local-only and full (both caches)
- Track which chunks use which state type

**run_prefill() and generate_next_token():**
- Pass correct state to each chunk based on its requirements
- Sync `kv_cache_local` between states after each chunk call

---

## Files to Modify

| File | Changes |
|------|---------|
| `anemll/utils/convert_model.sh` | Extract sliding_window, replace hardcoded 512 |
| `anemll/ane_converter/gemma3_converter.py` | Revert state declarations, remove dummy touch |
| `tests/chat.py` | Per-chunk states with sync |

---

## Testing

1. **Conversion test**: Convert with context=1024, verify no rotate functions generated
2. **Inference test**: Run chat.py with multi-chunk model, verify correct state passing
3. **Sync test**: Verify kv_cache_local updates propagate between chunks
