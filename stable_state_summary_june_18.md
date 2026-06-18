# Stable State Summary — June 18, 2026 (EOD Update)

**Commit:** `bb15b58`
**Tags:** `START_OVER_POINT_JUNE_18` / `snapshot-current`
**Date:** June 18, 2026 (23:10 UTC)
**Status:** All platforms synchronized — localhost, GitHub, morpheme.games

---

## Changes Since Last Stable Point (June 13 Night)

### 1. Board Transposition Coordinate Alignment
- **Files:** `static/js/play.js`
- Fixed cell matching, highlighting, and path rendering coordinates when the board is transposed (90-degree portrait rotation for mobile).
- Removed redundant untransposition and double-transposition of coordinate paths during word submission, ensuring that path coords submitted to the server remain perfectly aligned with raw board cell coordinates.
- Corrected transposition references in helper functions (`rebuildTileToWordsMap`, `updateBoardCell`, `findWordPathOnBoard`, `submitWord`, `showValidationFeedback`, `handleTournamentWord`, `handlePrivateMatchWord`).

### 2. Multi-Touch Swipe & Drag Selection Robustness
- **Files:** `static/js/play.js`
- Track the active touch pointer using `activeTouchIdentifier` to keep the current swiping/drawing gesture isolated and uninterrupted by other touch events (such as accidental secondary touches, taps, or releases).
- Prevented premature drag termination by only ending the swipe when the specific active finger is lifted.
- Ensured visual highlights (`selected`, `current`, `typing-highlight`) are cleanly and instantly cleared from the DOM upon drag selection termination/cancellation.

### 3. Either/Or (EO) Path Resolution & Tournament Word Validation
- **Files:** `static/js/play.js`, `templates/index.html`
- Implemented client-side Either/Or option expansion and dictionary checking locally in `handleTournamentWord` to align with the behaviour of public rooms.
- Ensured local validation properly checks candidate words against `window.lastGameState.all_words` (supporting arrays or objects/keys formats).
- Correctly forwarded paths from `submitWord` to `handleTournamentWord` instead of calling findWordPathOnBoard again.

### 4. Tournament Word Submission, Path highlights, & Redirection Fixes
- **Files:** `static/js/play.js`, `templates/index.html`
- **TypeError Fix:** Added type checking in `handleTournamentWord` to ensure `bonus_word` is a string before evaluating `.toUpperCase()`, preventing crashes during valid word submissions.
- **Race Condition Prevention:** Discard standard room state poll updates in `updateGameState` when a tournament or private match session is active, eliminating background redirection back to a 4x4 public room.
- **Cache Busting:** Bumped play.js cache version parameter to `v=174` in `templates/index.html`.

---

## All Changes This Session (Full June 18)

| # | Change | Files |
|---|--------|-------|
| 1 | Board transposition coordinate alignment | `static/js/play.js` |
| 2 | Multi-touch drag selection swipe isolation | `static/js/play.js` |
| 3 | Local Either/Or path validation for Tournaments | `static/js/play.js` |
| 4 | Fix tournament validation TypeErrors & race condition redirection | `static/js/play.js` |
| 5 | play.js cache version bump to v=174 | `templates/index.html` |

---

## Platform Sync Status
| Platform | Commit |
|----------|--------|
| localhost | `bb15b58` |
| GitHub (origin/main) | `bb15b58` |
| morpheme.games (132.148.72.249) | `bb15b58` |

---

## Infrastructure
- Server: Ubuntu 24.04 @ 132.148.72.249
- Process manager: PM2 (`morpheme` process, fork mode)
- Deploy script: `boggle-gen/scratch/deploy.py`
