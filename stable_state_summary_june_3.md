# Stable State Summary — June 3, 2026

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `aad7cd2` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `aad7cd2` / `snapshot-current` / `START_OVER_POINT_JUNE_3` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `aad7cd2` / `snapshot-current` | ✅ Fully Deployed & PM2 Reloaded |

**All environments are 100% synchronized at the latest commit `aad7cd2`.**
The local modifications to `templates/index.html` and `spinner_set.py` have been committed, pushed to GitHub, and successfully deployed to the remote production environment via the `deploy.py` script. The active production tags `snapshot-current` and `START_OVER_POINT_JUNE_3` have been updated and pushed to remote.

---

## Serving Versions

| File | Version / State | Description |
|------|-----------------|-------------|
| `spinner_set.py` | Commit `aad7cd2` | Updated word count range weights for standard / non-greatest configurations to `[30, 30, 30, 1]` to align with target distribution. |
| `templates/index.html` | Commit `aad7cd2` | Updated FAQ Spinner Word Count Range Distribution list and bumped play.js cache-buster query tag to `?v=126`. |
| `static/js/play.js` | Commit `63e9fe0` | Optimized intermission letter filtering performance: pre-built coordinate lookup maps (`O(1)` word lookup) and implemented render key cache checks to prevent redundant heartbeat rendering on mobile. |
| `game_room.py` | Commit `ad70317` | Updated short valid word validation error message to return uppercase format: `f"{word.upper()} IS TOO SHORT (MIN: {min_len_req}L)"`. |
| `static/css/play.css` | Commit `ad70317` | Added CSS styles and keyframes animation `wait-bounce` for wait-dots. |
| `board_generator.py` | Commit `e7a3740` | Resolved 3D cube neighbor transitions inside `_has_ing_sequence`, `_sanitize_forbidden_sequences`, and `_guarantee_no_ing`. Enforced ING sequence verification on both target and achieved difficulties. Supported 3D Either/Or layouts and added early-break optimization for protected tiles. |

---

## Work Completed on June 3, 2026

### 1. Instant Intermission Letter Filtering on Mobile
* **Goal achieved:** Optimized intermission letter-filtering loading performance to be instant on mobile devices.
* **Fix details:** 
  - Pre-built a coordinate-lookup map (`rebuildTileToWordsMap`) mapping tile coords to matching words.
  - Cached the map per round, reducing coordinate traversal checks to direct `O(1)` lookups.
  - Added an intermission render key cache check (`window.lastRenderedIntermissionKey`) to prevent full list re-rendering on heartbeat ticks when the active tab remains unchanged.
  - Cleared redundant found tab click handlers.

### 2. Spinner Word Count Range Distribution Updates
* **Goal achieved:** Updated the "Spinner Word Count Range Distribution" to reflect new percentages in both the FAQ and gameplay.
* **Fix details:**
  - FAQ list in `index.html` updated to show: 9% 50-100 words, 30% 100-200 words, 30% 200-300 words, 30% 300-400 words, 1% 500+ words.
  - Configured `spinner_set.py` weights to `[30, 30, 30, 1]` for standard min-length configurations to match these exact percentages when the `50-100` range is excluded.
  - Bumped play.js query string to `v=126` for cache invalidation.

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
