# Stable State Summary — June 3, 2026

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `ad70317` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `ad70317` / `snapshot-current` / `START_OVER_POINT_JUNE_3` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `ad70317` / `snapshot-current` | ✅ Fully Deployed & PM2 Reloaded |

**All environments are 100% synchronized at the latest commit `ad70317`.**
The local modifications to `game_room.py`, `static/css/play.css`, and `static/js/play.js` have been committed, pushed to GitHub, and successfully deployed to the remote production environment via the `deploy.py` script. The active production tags `snapshot-current` and `START_OVER_POINT_JUNE_3` have been updated and pushed to remote.

---

## Serving Versions

| File | Version / State | Description |
|------|-----------------|-------------|
| `game_room.py` | Commit `ad70317` | Updated short valid word validation error message to return uppercase format: `f"{word.upper()} IS TOO SHORT (MIN: {min_len_req}L)"`. |
| `static/js/play.js` | Commit `ad70317` | Bypassed local optimistic invalid validation for short words in standard play. Updated tournament and private match word validations to format messages correctly. Implemented robust wait dots bounce animation on mobile resume. |
| `static/css/play.css` | Commit `ad70317` | Added CSS styles and keyframes animation `wait-bounce` for wait-dots. |
| `board_generator.py` | Commit `e7a3740` | Resolved 3D cube neighbor transitions inside `_has_ing_sequence`, `_sanitize_forbidden_sequences`, and `_guarantee_no_ing`. Enforced ING sequence verification on both target and achieved difficulties. Supported 3D Either/Or layouts and added early-break optimization for protected tiles. |

---

## Work Completed Up To June 3, 2026

### 1. Fix Short Valid Word Validation Feedback
* **Goal achieved:** Prevented the client from displaying "Invalid Word" before replacing it with the server's "too short" validation message when a valid word is entered below the minimum length requirement.
* **Fix details:** The client now skips optimistic validation for words below the minimum length. In tournament and private match modes, the validation output is correctly formatted as `[WORD] IS TOO SHORT (MIN: [MIN]L)`.

### 2. Bouncing WAIT... Dots & Mobile Wake-up Fix
* **Goal achieved:** The waiting status message in rooms ("WAIT...") now features a wave-like sequential bouncing animation for the dots.
* **Fix details:** The animation resets via a robust force-restart technique (clearing the element, forcing reflow, and toggling styles in `requestAnimationFrame`) upon tab focus/visibility events, ensuring animations never freeze on mobile wake-up.

### 3. Fix 3D Cube Neighbor Resolution inside Sequence Checkers
* **Goal achieved:** The "ING" sequence checkers (`_has_ing_sequence`, `_sanitize_forbidden_sequences`, and `_guarantee_no_ing`) now correctly use `_get_cube_neighbors(f, r, c)` when checking 3D cube layouts.

### 4. Enforce ING Check on Promoted Boards
* **Goal achieved:** If a board generated for an "Easy" target difficulty achieves a uniqueness ratio that places it in the "Medium" or "Hard" difficulty label range, the generator now detects the achieved difficulty and runs "ING" checks/sanitization.

### 5. Support 3D Either/Or Layouts
* **Goal achieved:** Rewrote the Either/Or tile application block inside both `generate_board` and `_generate_emergency_compliant_board` to support 3D coordinates `(f, r, c)`.

---

## Key Files Tracked

| File | Location | Purpose |
|------|----------|---------|
| `game_room.py` | Production + GitHub | Server-side validation and formatting of short valid words. |
| `static/js/play.js` | Production + GitHub | Client-side validation logic, state polling wake-up handler, and wait dots formatting. |
| `static/css/play.css` | Production + GitHub | Styles for wait-dot animation. |
| `board_generator.py` | Production + GitHub | Resolved 3D neighbor transition logic, enforced target/achieved difficulty checks, and supported 3D Either/Or layouts. |
