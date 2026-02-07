# Local Model Transfer (P0) - ANEMLLChat

## Goal
Ship a reliable local-first model transfer flow from macOS to iOS:
- macOS: drop/import/link/share model
- iOS: receive/bootstrap/pull/validate/import model
- Both: deterministic naming, integrity checks, and clear recovery UX

## P0 Tickets (Must Ship)

### MODEL-001 Smart model-name suggestion from drop path
- Accept folder drop and validate model-root structure.
- Auto-suggest model name from path patterns (converter-friendly):
  - If path ends with `/hf/ios`, use grandparent folder name.
  - If leaf folder is generic (`ios`, `hf`, `model`, `output`, `converted`), walk upward to first non-generic parent.
- Normalize internal ID (slug/lowercase) and keep display name separately.
- Resolve collisions deterministically (`name`, `name-2`, `name-3`) while allowing user override.

Acceptance:
- Correct suggestion for common converter layouts.
- User can edit suggestion before finalize.
- Same path always yields same suggested name.

### MODEL-002 Import vs Link decision flow
- After drop validation, prompt user:
  - `Import` (copy into app storage), or
  - `Link` (reference existing external folder).

Acceptance:
- Choice is mandatory and explicit before ingest starts.
- UI clearly explains tradeoffs (portability vs speed/local reference).

### MODEL-003 Import pipeline (copy + progress + cancel + atomic finalize)
- Copy required files with progress reporting.
- Support cancel at any point.
- Use temp staging and atomic finalize to prevent partial model state.

Acceptance:
- No half-imported models visible after cancel/failure.
- Completed import appears once, with consistent metadata.

### MODEL-004 Link pipeline (bookmark + launch verification + recovery)
- Persist canonical path/security bookmark for linked model source.
- Verify linked source availability on app launch/model open.
- Show recovery UI when source folder is missing/unavailable.

Acceptance:
- Linked models survive restart when source remains available.
- Missing source state is explicit and recoverable (re-link/remove).

### MODEL-005 Share package contract
- Define package manifest contract (`manifest.json`):
  - package/version metadata
  - required file list
  - per-file SHA256 and overall package hash
- Generate package for both imported and linked models.

Acceptance:
- Package generation is deterministic for the same source.
- Validation tooling can detect missing/corrupt files.

### MODEL-006 macOS Model View -> Share (AirDrop)
- Add share action in macOS model list/detail.
- Build transfer payload and invoke system share sheet/AirDrop.

Acceptance:
- User can share selected model from UI with one clear action.
- Share flow reports success/failure with actionable messaging.

### MODEL-007 iOS receive + pull transfer from macOS file server
- AirDrop payload is bootstrap metadata/auth (not full blob transfer).
- macOS launches temporary local transfer server per session.
- iOS receives bootstrap, authenticates, downloads manifest/files, verifies hashes, then imports.
- Required transport behaviors:
  - one-time session token + expiry
  - chunk/range download + resume
  - hash verification per file + package
  - auto-stop server on success/timeout

Acceptance:
- Large models import without embedding file blobs in AirDrop JSON.
- Interrupted transfers can resume.
- Invalid token/hash fails safely with clear error.

### MODEL-008 Compatibility gate on iOS
- Check package format/version/app compatibility before import finalize.
- Show explicit unsupported-version message with next steps.

Acceptance:
- Incompatible packages never reach active model list.
- Error includes reason and required app/package version.

### MODEL-009 Duplicate/conflict policy
- Deterministic handling when model already exists:
  - prompt for replace/keep-both/rename policy
  - stable rename behavior

Acceptance:
- No silent overwrite.
- Conflict handling is predictable across repeated imports.

### MODEL-010 End-to-end smoke tests
- Cover happy path:
  - macOS import
  - macOS link
  - macOS share via AirDrop bootstrap
  - iOS receive/pull/import
  - first inference success

Acceptance:
- Automated smoke run passes on at least one supported macOS+iOS pair.

## P0 Dependency Order
1. MODEL-005 package contract (unblocks share/receive validation)
2. MODEL-001 drop validation + naming
3. MODEL-002 import/link decision flow
4. MODEL-003 import pipeline
5. MODEL-004 link pipeline
6. MODEL-006 macOS share UI integration
7. MODEL-007 iOS receive + macOS transfer server
8. MODEL-008 compatibility gate
9. MODEL-009 duplicate/conflict handling
10. MODEL-010 end-to-end smoke tests

## Out of Scope for This P0 File
- MODEL-011 to MODEL-017 (recovery hardening, storage guardrails, integrity re-check command, expanded QA matrix, optimization, diagnostics, advanced sharing modes) are follow-on work.
