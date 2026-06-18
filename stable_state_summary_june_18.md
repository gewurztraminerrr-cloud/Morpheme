# Stable State Summary — June 18, 2026 (Evening)

**Commit:** `25cf175`
**Tags:** `START_OVER_POINT_JUNE_18` / `snapshot-current`
**Date:** June 18, 2026 (17:54 CT)
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
- Bumped play.js cache version parameter to `v=173` in `templates/index.html`.

---

## All Changes This Session (Full June 18)

| # | Change | Files |
|---|--------|-------|
| 1 | Board transposition coordinate alignment | `static/js/play.js` |
| 2 | Multi-touch drag selection swipe isolation | `static/js/play.js` |
| 3 | Local Either/Or path validation for Tournaments | `static/js/play.js` |
| 4 | play.js cache version bump to v=173 | `templates/index.html` |

---

## Platform Sync Status
| Platform | Commit |
|----------|--------|
| localhost | `25cf175` |
| GitHub (origin/main) | `25cf175` |
| morpheme.games (132.148.72.249) | `25cf175` |

---

## Infrastructure
- Server: Ubuntu 24.04 @ 132.148.72.249
- Process manager: PM2 (`morpheme` process, fork mode)
- Deploy script: `boggle-gen/scratch/deploy.py`
