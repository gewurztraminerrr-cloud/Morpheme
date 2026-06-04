# Stable State Summary — June 3, 2026

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `182436a` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `182436a` / `snapshot-current` / `START_OVER_POINT_JUNE_3` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `182436a` / `snapshot-current` | ✅ Fully Deployed & PM2 Reloaded |

**All environments are 100% synchronized at the latest commit `182436a`.**
The local modifications to `spinner_set.py` and `game_room.py` have been committed, pushed to GitHub, and successfully deployed to the remote production environment via the `deploy.py` script. The active production tags `snapshot-current` and `START_OVER_POINT_JUNE_3` have been updated and pushed to remote.

---

## Serving Versions

| File | Version / State | Description |
|------|-----------------|-------------|
| `spinner_set.py` | Commit `182436a` | Reverted Either/Or and Density format weights back to 2% weight, restoring the original spinner odds configuration. |
| `game_room.py` | Commit `182436a` | Prioritized `next_round_spinner_params` over `spinner_params` in `start_next_round` to fix parameter promotion race condition. Passed `board_format` captured as a ghost variable to `save_round_history` to prevent board format race conditions. |
| `templates/index.html` | Commit `aad7cd2` | Updated FAQ Spinner Word Count Range Distribution list and bumped play.js cache-buster query tag to `?v=126`. |
| `static/js/play.js` | Commit `618e0f0` | Enable swiped/moused/typed path highlighting and validation flashing in Density Format. |
| `static/css/play.css` | Commit `ad70317` | Added CSS styles and keyframes animation `wait-bounce` for wait-dots. |
| `board_generator.py` | Commit `e7a3740` | Resolved 3D cube neighbor transitions inside `_has_ing_sequence`, `_sanitize_forbidden_sequences`, and `_guarantee_no_ing`. Enforced ING sequence verification on both target and achieved difficulties. Supported 3D Either/Or layouts and added early-break optimization for protected tiles. |

---

## Work Completed on June 3, 2026

### 1. Reverted Format Weights in Spinner Set
* **Goal achieved:** Restored Either/Or and Density format odds weights back to 2%.
* **Fix details:**
  - Configured `spinner_set.py` weights to `[72, 12, 2, 2, 2, 2, 2, 2, 2, 1, 1]` to align with original percentages in the Spinner Set Odds window.

### 2. Resolved Parameter Promotion & Scoring Races
* **Goal achieved:** Fixed Either/Or scoring bug where SPATES gave 3 points instead of 6 on a transposed board.
* **Fix details:**
  - Prioritized `next_round_spinner_params` over the previous round's `spinner_params` in `start_next_round` so that newly pre-generated round formats are correctly promoted at the start of the round.
  - Added a `board_format` parameter to `save_round_history` and passed the ended round's format as a captured ghost variable from `start_next_round` and daily reset. This prevents the asynchronous database logger from reading the *newly promoted* format instead of the *ended* round's format.

### 3. Enabled Highlighting & Validation Flashing in Density Format
* **Goal achieved:** Ensured path drawing and validity flashes are visible on Density boards.
* **Fix details:**
  - Standardized custom/selected/typing highlight selectors in `play.css` and `play.js` to override synesthesia and density styles.
  - Recalculated cell density styling dynamically when selection and validation states change.

---

## Key Files Tracked

| File | Location | Purpose |
|------|----------|---------|
| `spinner_set.py` | Production + GitHub | Gameplay spinner word count range generation and weights. |
| `templates/index.html` | Production + GitHub | Main client HTML containing versioned references to static assets and static FAQ text. |
| `static/js/play.js` | Production + GitHub | Client-side intermission letter filtering logic and cache render key checks. |
| `game_room.py` | Production + GitHub | Server-side validation and formatting of short valid words. |
| `static/css/play.css` | Production + GitHub | Styles for wait-dot animation. |
| `board_generator.py` | Production + GitHub | Resolved 3D neighbor transition logic, enforced target/achieved difficulty checks, and supported 3D Either/Or layouts. |
