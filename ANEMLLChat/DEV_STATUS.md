# ANEMLL Chat Development Status

**Date:** 2026-01-30
**Status:** MODEL LOADING WORKING - Chat Functional

## Current Testing Session (2026-01-30 11:23 PM)

### App State - WORKING
- ANEMLLChat running successfully
- Model loading functional
- Chat interface working
- Gemma 3 1B successfully loaded and active

### Verified Working Features
1. **Models list display** - Shows downloaded and available models
2. **Model downloading** - Downloads from HuggingFace work
3. **Model loading** - CoreML models load on ANE
4. **Chat inference** - Generates responses
5. **Model unload** - Can switch models

### Downloaded Models
Located in: `~/Documents/Models/`
- `anemll_anemll-Llama-3.2-1B-FAST-iOS_0.3.0` (complete, has meta.yaml)
- `anemll_anemll-google-gemma-3-1b-it-ctx4096_0.3.4` (complete, has meta.yaml)
- `anemll_anemll-Qwen3-4B-ctx1024_0.3.0` (complete, has meta.yaml)
- `anemll_anemll-llama-3.2-1B-iOSv2.0` (complete, has meta.yaml)
- `anemll_anemll-google-gemma-3-4b-qat4-ctx1024` (complete, has meta.yaml)
- `anemll_anemll-google-gemma-3-4b-it-qat-int4-unquantized-ctx4096_0.3.5` (complete, has meta.yaml)

### Recent Changes
1. **Added console logging** - Clear `[MODEL LOADING]`, `[MODEL LOADED]`, `[INFERENCE]` markers
2. **HuggingFace repos verified** - All 4 default repos accessible and returning HTTP 200

---

## Build & Run

```bash
# Build
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-0.3.5/ANEMLLChat
xcodebuild -project ANEMLLChat.xcodeproj -scheme ANEMLLChat -configuration Debug build

# Run
open /Users/anemll/Library/Developer/Xcode/DerivedData/ANEMLLChat-cfloixiatmalxidetdfouelsvvlm/Build/Products/Debug/ANEMLLChat.app

# Kill
pkill -f "ANEMLLChat"
```

## Key Files

### App Structure
```
ANEMLLChat/
├── ANEMLLChat/
│   ├── App/
│   │   ├── ANEMLLChatApp.swift      # Main app entry
│   │   └── ContentView.swift         # Root view with navigation
│   ├── ViewModels/
│   │   ├── ModelManagerViewModel.swift  # Model management
│   │   └── ChatViewModel.swift          # Chat state
│   ├── Views/
│   │   ├── Models/
│   │   │   ├── ModelListView.swift   # Model browser
│   │   │   └── ModelCard.swift       # Model display card
│   │   └── Chat/
│   │       └── ChatView.swift
│   ├── Services/
│   │   ├── DownloadService.swift     # HuggingFace downloads
│   │   ├── StorageService.swift      # Model storage
│   │   ├── InferenceService.swift    # CoreML inference
│   │   └── Logger.swift              # Logging system
│   └── Models/
│       └── ModelInfo.swift           # Model data structure
└── ANEMLLChat.xcodeproj
```

## Default Models (HuggingFace)

| Model | Repo ID | Size | Context |
|-------|---------|------|---------|
| LLaMA 3.2 1B | anemll/anemll-llama-3.2-1B-iOSv2.0 | 1.6 GB | 512 |
| Gemma 3 1B | anemll/anemll-google-gemma-3-1b-it-ctx4096_0.3.4 | 1.5 GB | 4096 |
| Qwen 3 4B | anemll/anemll-Qwen3-4B-ctx1024_0.3.0 | 4.0 GB | 1024 |
| LLaMA FAST | anemll/anemll-Llama-3.2-1B-FAST-iOS_0.3.0 | 1.2 GB | 512 |

## UI Automation (AnemllAgentHost)

A local macOS agent for UI automation via HTTP API.

### Setup
```bash
export ANEMLL_HOST="http://127.0.0.1:8765"
export ANEMLL_TOKEN="EDF0B1FC-6A62-4CCB-8E65-771F0DF2309A"  # Get from menu bar app
```

### Commands

**Health Check:**
```bash
curl -s -H "Authorization: Bearer $ANEMLL_TOKEN" "$ANEMLL_HOST/health"
```

**Take Screenshot:**
```bash
curl -s -H "Authorization: Bearer $ANEMLL_TOKEN" -X POST "$ANEMLL_HOST/screenshot"
# Saves to /tmp/anemll_last.png
```

**Click at Coordinates:**
```bash
curl -s -H "Authorization: Bearer $ANEMLL_TOKEN" -H "Content-Type: application/json" \
  -X POST "$ANEMLL_HOST/click" -d '{"x":960,"y":540}'
```

**Type Text:**
```bash
curl -s -H "Authorization: Bearer $ANEMLL_TOKEN" -H "Content-Type: application/json" \
  -X POST "$ANEMLL_HOST/type" -d '{"text":"Hello"}'
```

**Move Mouse:**
```bash
curl -s -H "Authorization: Bearer $ANEMLL_TOKEN" -H "Content-Type: application/json" \
  -X POST "$ANEMLL_HOST/move" -d '{"x":960,"y":540}'
```

### Workflow
1. Take screenshot
2. Analyze `/tmp/anemll_last.png`
3. Determine action (click, type)
4. Execute action
5. Screenshot again to verify

### Notes
- SwiftUI buttons may not respond to CGEvent clicks
- Permission dialogs can be clicked via osascript
- Bring app to front: `osascript -e 'tell application "ANEMLLChat" to activate'`

## Console Logging

Model loading now prints clear markers to console:
- `[MODEL LOADING] Starting to load model from: <path>`
- `[MODEL LOADED] Successfully loaded: <model_name>`
- `[MODEL ERROR] Failed to load model: <error>`
- `[INFERENCE] Starting generation with N input tokens`
- `[INFERENCE] Complete: N tokens at X.X tok/s`

## Known Issues (RESOLVED)

1. ~~**splitLMHead hardcoded** - Was set to 8, but Gemma needs 16~~ **FIXED** - Now reads from `config.splitLMHead`
2. **Incomplete downloads** - Some downloads interrupted by debugger kill
3. **Custom model naming** - `-gemma-3-4b-qat4-ctx1024` has malformed name (starts with dash)

## Recent Fixes (2026-01-30 11:35 PM)

1. **Fixed splitLMHead configuration** - InferenceService.swift was hardcoding `splitLMHead: 8` but Gemma 3 models need `splitLMHead: 16`. Changed to read from `config.splitLMHead`.
   - Error was: "MultiArray shape (8) does not match the shape (16) specified in the model description"

2. **UI Improvements**:
   - **Model status button** - Made larger with pill shape, chevron indicator, shows "No Model" when none loaded
   - **Input text box** - Added border overlay for better visibility
   - **Download speed** - Added fallback calculation using average speed from start time when recent history unavailable

## UI Issues - Text Layout (iPhone) - FIXED ✓

**Status:** RESOLVED (2026-01-31)

### Implementation:
Created `Views/Chat/MarkdownView.swift` with full markdown support:
- ✓ **Bold/Italic** - via `AttributedString(markdown:)`
- ✓ **Numbered lists** - `1. item` renders with proper indentation
- ✓ **Bullet lists** - `- item`, `* item`, `• item` all supported
- ✓ **Code blocks** - ``` fenced blocks with language label and monospace font
- ✓ **Tables** - Full table rendering with headers, dividers, and inline markdown in cells
- ✓ **Headings** - `#` through `######` with appropriate font sizes
- ✓ **Paragraphs** - Proper spacing between blocks

### Updated Files:
- `Views/Chat/MarkdownView.swift` - NEW: Full markdown parser and renderer
- `Views/Chat/MessageBubble.swift` - Uses `MarkdownView` for assistant messages
- Bot messages have solid background (`secondarySystemBackground`)

## Swift CLI Fix (2026-01-31 07:45 AM)

**Fixed hardcoded EOT token in InferenceManager.swift**
- Removed hardcoded `eotToken = 128009` (LLaMA 3 token)
- Now correctly uses `eosTokens` passed from tokenizer
- Gemma 3 tokens: [1, 106, 212] = `<eos>`, `<end_of_turn>`, `</s>`
- LLaMA 3 tokens: [128001, 128008, 128009] (detected dynamically)

**Verified**: Multi-turn test shows correct EOS token setup and `[Stop reason: eos_token]` on all responses. Performance: 66-69 tok/s on macOS, TTFT 125-668ms.

**Also checked**: `chat.py` and `chat_full.py` - no hardcoded token IDs found (clean).

## Testing Session (2026-01-31 07:25 AM)

### Verified on iPhone (via iPhone Mirroring)
1. **Model unload/load** - Successfully unloaded Gemma 3 1B and reloaded it
2. **Multi-turn conversations** - Works correctly, maintains context
3. **Markdown rendering** - Bullet lists, numbered lists, bold text all render properly
4. **Token generation speed** - 42-57 tok/s on iPhone
5. **Controller API** - Health, screenshot, click, type all working

### Notes
- App runs on actual iPhone hardware (iPhone Mirroring used for testing)
- Downloaded models are stored in iPhone's sandboxed Documents/Models directory
- Models on Mac ~/Documents/Models are separate from iPhone storage

## Recent Fixes (2026-01-31 02:15 AM)

1. **Auto-load last model on startup** - Fixed timing issue where auto-load was called before models finished loading
   - Moved auto-load call to end of `loadModels()` in ModelManagerViewModel
   - Added Settings toggle "Auto-load last model" to enable/disable
   - Added "Clear remembered model" button in Settings

2. **Scroll-to-bottom indicator** - Added floating button that appears when scrolled up
   - Shows chevron-down button when not at bottom
   - Click to instantly scroll to latest message
   - Smooth animation on appear/disappear

3. **Table inline markdown** - Fixed bold/italic rendering inside table cells

4. **Error toast notifications** - Created reusable ToastView component
   - Non-intrusive toast at top of screen (replaces alerts)
   - Auto-dismiss after 5 seconds
   - Supports error, warning, success, info types
   - Added to ChatView and ModelListView

5. **Improved scroll-to-bottom detection** - Fixed scroll indicator logic
   - Now correctly detects when bottom anchor is visible
   - Button appears only when scrolled up from bottom

6. **Fixed ModelCard layout on iPhone** - Prevented vertical text wrapping
   - Removed icons from metadata row to save space
   - Added `.fixedSize()` to prevent text breaking across lines
   - Format: `1.5 GB • 4,096 ctx • gemma` (horizontal, single line)

7. **Fixed MessageBubble layout on iPhone**
   - Assistant messages now expand full width (removed right spacer)
   - Added extra bottom padding (20pt) to prevent overlap with input bar
   - User messages still right-aligned with left spacer

## TODO

1. ~~Clean up incomplete model downloads~~ ✓ DONE - All models now complete
2. ~~Test longer conversations~~ ✓ DONE - Multi-turn works, markdown rendering works
3. ~~Test model switching~~ ✓ DONE - Unload/load works correctly
4. ~~Implement proper error display in UI~~ ✓ DONE (ToastView component)
5. ~~Add download resume capability~~ ✓ PARTIAL - Completed files are skipped on retry; full file-level resume not implemented
6. ~~**Fix markdown rendering in chat messages**~~ ✓ DONE
7. ~~**Add scroll indicator for long responses**~~ ✓ DONE
8. ~~**Improve message bubble styling with solid background**~~ ✓ DONE
9. ~~**Delete model with confirmation dialog**~~ ✓ ALREADY EXISTS (swipe, context menu, alert)
10. ~~**Remember last loaded model on startup**~~ ✓ DONE - Added Settings toggle to disable
