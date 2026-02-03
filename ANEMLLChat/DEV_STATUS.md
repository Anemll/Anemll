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

## Build & Export (Claude Notes)

### macOS (local Mac app)
```bash
# Build Debug .app
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-0.3.5/ANEMLLChat
xcodebuild -project ANEMLLChat.xcodeproj -scheme ANEMLLChat -configuration Debug build

# Launch built app
open /Users/anemll/Library/Developer/Xcode/DerivedData/ANEMLLChat-*/Build/Products/Debug/ANEMLLChat.app
```

### iPhone (device build/export)
```bash
# Build for device (generic iOS device)
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-0.3.5/ANEMLLChat
xcodebuild -project ANEMLLChat.xcodeproj -scheme ANEMLLChat -configuration Debug \
  -destination 'generic/platform=iOS' build

# Archive (for export / TestFlight / IPA)
xcodebuild -project ANEMLLChat.xcodeproj -scheme ANEMLLChat -configuration Release \
  -destination 'generic/platform=iOS' archive \
  -archivePath /tmp/ANEMLLChat.xcarchive

# Export IPA (needs export options plist with signing)
xcodebuild -exportArchive \
  -archivePath /tmp/ANEMLLChat.xcarchive \
  -exportPath /tmp/ANEMLLChatExport \
  -exportOptionsPlist /tmp/ANEMLLChatExportOptions.plist
```

**Claude gotcha**: iOS export requires a valid signing team + provisioning profile. If `xcodebuild -exportArchive` fails, open the `.xcarchive` in Xcode Organizer and export there once to generate a working `ExportOptions.plist`, then reuse it for CLI exports.

## Xcode Run Notes (Claude)

### No Simulator Needed
Do **not** launch an emulator/simulator. iPhone Mirroring is already connected and should be used for on-device testing. The device shows as a real iPhone target in Xcode.

### Run on iPhone (via Mirroring)
1) In Xcode, select the scheme **ANEMLLChat**.
2) In the device picker, choose the **connected iPhone** (iOS 17.x in this setup). Do **not** pick a simulator.
3) Press **Run** (▶). Xcode will build and deploy to the phone.
4) Use the mirrored iPhone window to verify the UI.

### Run on macOS
1) In Xcode, select the scheme **ANEMLLChat**.
2) Choose **My Mac** as the run destination.
3) Press **Run** (▶). The macOS app launches locally.

### Token Handling (Agent Host)
If a bearer token fails (e.g., `/health` returns unauthorized), ask the user for the current token from the menu bar app UI and re-export it before retrying.

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

## Recent Fixes (2026-01-31 10:30 AM)

1. **Fixed Delete button visibility on iOS**
   - Made delete button more prominent with bordered style and `.body` font (was `.caption`)
   - Changed from plain style to bordered button with red tint
   - Made "Show in Finder" swipe action and context menu item macOS-only (won't work on iOS)

2. **Fixed multi-turn conversation bug in Tokenizer.swift**
   - The fallback chat template formatting was only using the FIRST user message, causing context loss
   - Now properly formats ALL messages in the conversation history for multi-turn support
   - Fixed for all templates: Gemma/Gemma3, LLaMA 3, DeepSeek, Qwen/ChatML
   - This fixes the bug where model would respond to previous questions instead of current one

3. **Added copy button on hover for messages (macOS)**
   - Copy icon appears in top-right corner when hovering over messages
   - Uses `.ultraThinMaterial` background for subtle appearance
   - iOS uses context menu (long press) for copy
   - Copies message content to clipboard

4. **Verified macOS model info display**
   - Model status shows in top-right toolbar (green dot when loaded, orange when not)
   - Model list shows full details: name, description, size, context length, architecture tag
   - Delete button (red trash icon) visible next to Load button for downloaded models

5. **Imported app icon from anemll-chatbot**
   - Copied Assets.xcassets with full AppIcon.appiconset (all sizes for iOS, iPad, macOS, Watch)
   - Bundle identifier updated to `anemll.anemll-chat.demo` (matches anemll-chatbot for App Store)
   - Marketing version updated to 0.3.7

6. **Fixed macOS download progress not updating**
   - KVO on URLSessionDownloadTask.progress was not firing frequently on macOS
   - Added timer-based polling (0.5s interval) as fallback for macOS
   - iOS continues to use KVO only (works fine there)

7. **Improved model loading indicator visibility**
   - Replaced plain ProgressView with animated ModelLoadingIndicator
   - Features green pulsing background circle with rotating CPU icon
   - Shows "Loading..." text in green capsule-shaped background
   - Much more visible than the previous tiny spinner

## Recent Fixes (2026-02-02 9:30 PM)

1. **Settings - Prompt Options Renamed**
   - "No Prompt" → "Default Prompt" (standard inference, no additional prompting) - now the DEFAULT
   - "Model's Default" → "No Template" (raw inference without chat template)
   - "Model's Default (Thinking)" → "Thinking Mode"
   - "Model's Default (Non-Thinking)" → "Non-Thinking Mode"
   - Added "Reset to Defaults" button in Settings

2. **Copy Button Alignment Fixed**
   - Changed from ZStack to overlay alignment
   - Copy button now properly positioned at top-right
   - Always clickable even for short text

3. **Scroll/Chevron Behavior Improved**
   - Fixed chevron not appearing during first long inference
   - Added `onChange(of: chatVM.isGenerating)` to track generation state
   - Added `onChange(of: chatVM.streamingContent)` to update during streaming
   - Chevron now shows when content grows beyond visible area (>200 chars during streaming)
   - Scroll behavior: shows dots first, then scrolls user question to top when tokens arrive

4. **Liquid Glass UI (macOS 26+)**
   - Added `InputBarGlassModifier` with `glassEffect(.regular.interactive())`
   - Added `ScrollButtonGlassModifier` for scroll-to-bottom chevron
   - Added `ToolbarGlassModifier` for toolbar
   - Falls back to `.ultraThinMaterial` on older macOS versions

5. **Reduced Bottom Padding**
   - `contentBottomPadding` reduced from +120 to +48
   - Bottom spacer reduced from 48px to 8px
   - Fixed excessive whitespace at bottom of chat

## Recent Fixes (2026-01-31 7:20 PM)

1. **Model deletion bug investigation**
   - Added debug logging to StorageService.deleteModel to track:
     - Model ID being deleted
     - Computed path for deletion
     - Whether fileExists check passed
     - Result of removeItem operation
   - Added verification after deletion to confirm directory was actually removed
   - Found: Some models had mismatched registry state (isDownloaded: false but folder exists)

2. **Input validation for custom models**
   - Added whitespace trimming to model IDs and names in ModelManagerViewModel.addCustomModel
   - Added validation for repo ID format (must contain "/")
   - Fixed modelPath function to trim whitespace from model IDs
   - Found malformed model ID in registry: leading space in " anemll/anemll-Qwen..."

3. **iOS Files app visibility**
   - Added `UIFileSharingEnabled = true` to Info.plist
   - Added `LSSupportsOpeningDocumentsInPlace = true` to Info.plist
   - Added CoreML document type handler
   - Models folder now visible in iOS Files app under "On My Phone/ANEMLLChat"

4. **System prompt resolution** (from earlier session)
   - Added SystemPromptOption enum with: Model's Default, Thinking, Non-Thinking, No Prompt, Custom
   - Changed default temperature to 0.0 (greedy decoding)
   - Added resolveSystemPrompt() in InferenceService for template-aware prompt resolution
   - Supports Qwen /think and /no_think suffixes

5. **Model storage location aligned with anemll-chatbot**
   - macOS: Changed from `~/Documents/Models/` to `~/Documents/` (matches anemll-chatbot)
   - iOS: Remains `Documents/Models/` (sandboxed, visible in Files app)
   - Both apps now share the same model storage location on macOS
   - **MIGRATION**: Existing models in `~/Documents/Models/` need to be moved to `~/Documents/`

## Known Issues / TODO

1. ~~**CTX display** - Need to verify if "ctx" in message stats shows overall context buffer or just per-message tokens~~ ✓ VERIFIED - CTX shows **overall context** (grew from 22→410 tokens in multi-turn conversation)
2. **App icon in dock** - May need dock cache refresh to show new icon
3. **Model registry state mismatch** - Some models show isDownloaded: false in registry but folder exists on disk

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
