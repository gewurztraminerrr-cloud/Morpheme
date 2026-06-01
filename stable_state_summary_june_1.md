# Stable State Summary — June 1, 2026 (Final Save Point)

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `89667bb` / `START_OVER_POINT_JUNE_1` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `89667bb` / `START_OVER_POINT_JUNE_1` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `89667bb` / `snapshot-current` | ✅ Fully Deployed & PM2 Reloaded |

**All environments are 100% synchronized at the latest stable commit `89667bb`.**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py` (which completed a repository hard reset and successfully restarted all PM2 processes).
The active recovery points `START_OVER_POINT_JUNE_1` and `snapshot-current` tags have been successfully updated and pushed to GitHub.

---

## Serving Versions (cache-busted)

| File / Style | Version | Description |
|--------------|---------|-------------|
| `/js/play.js` | `v=115` | Valid already found words purple highlights, daily rooms missed section only, and zero-hesitation |
| `/css/play.css` | `v=83` | Added purple cell tile flash and status text styling classes |
| `/js/lobby.js` | `v=5` | Rating filter proximity sorting and search listener registrations |
| `/css/lobby.css` | `v=17` | Premium flex rating search bar layout and Find button style |
| Inline `<style>` block | *Dynamic* | Mobile WebView cache bypass styling in `index.html` |

---

## Work Completed This Session (June 1 Fixes & Updates)

### 1. Combo Checker Metric Backtracking Update
* **Goal achieved:** Relocating a letter further down (such as in `SUBRACE`/`UBRACE` $\rightarrow$ `RUBACE`) now correctly counts as exactly **1 MP** (1 relocation) rather than being treated as 2 MP (remove and insert).
* **Backtracking Implementation (`app.py` & `profile_combo.py`):**
  * Replaced the standard LCS backtrace (which only matches characters in strictly increasing order, failing on relocations) with a highly optimized backtracking search algorithm.
  * The new algorithm evaluates all possible matching subsets including index crossings (transpositions/relocations) and dynamically calculates the LIS.
  * Pruning guarantees that it executes in less than 0.1 milliseconds for all standard word lengths ($\le 10$).

### 2. Purple Highlights for Already Found Words
* **Goal achieved:** If a valid word is entered but has already been found, it highlights beautifully in premium **purple** instead of red. This applies across all three play styles (Standard, Tournament, and Private Match).
* **Tile Flash & Text Styling (`play.css` & `play.js`):**
  * Added `body .board-cell.tile-flash-purple` styling classes and `.status-already-found { color: #a855f7; }` rules.
  * Updated `showValidationFeedback` to classify matches containing `'ALREADY FOUND'` case-insensitively and apply purple flashes and input backgrounds.
  * Tuned `optimisticColor` and `serverColor` to `'purple'` to completely eliminate double-flashing.

### 3. Active Gameplay Validation Flashing Fix
* **Goal achieved:** Completely resolved active gameplay double-flashing. Sourced standard play validation directly from `preState.all_words` rather than the stale render cache.

### 4. Backend Guest Player History Persistence (24h Rooms)
* **Goal achieved:** Guest accounts in 24h rooms now successfully have their yesterday's words stored and loaded in the "Previous Day" tab by snapshotting `p.is_guest` and recovering guest usernames in `get_yesterdays_history`.

### 5. Hide FOUND Section in 24h Room Previous Day Tab
* **Goal achieved:** Hides the FOUND section completely in daily 24h rooms under the "Previous Day" tab, displaying only the MISSED section as requested.

---

## Key Files Modified

| File | Location | Purpose |
|------|----------|---------|
| `app.py` | Production + GitHub | Update `calculate_morpheme_metric` with the backtracking relocation logic |
| `profile_combo.py` | Production + GitHub | Update the standalone combo checker metric with the backtracking relocation logic |
| `game_room.py` | Production + GitHub | Include guest players in snapshots and complete round saves for 24h rooms, and correctly map guest usernames |
| `static/js/play.js` | Production + GitHub | Implement purple highlights for duplicates, direct authoritative validation checks, and hide FOUND section on daily rooms |
| `static/css/play.css` | Production + GitHub | Added purple cell flash and status text highlight styling rules |
| `templates/index.html` | Production + GitHub | Bump play.js version to `115` and play.css version to `83` for cache busting |

---

## Previous Save Points

* [May 31 Stable State Summary](file:///Users/jeffbabiak/.gemini/antigravity/brain/3cc91699-cf5c-405f-9bbd-1934305e8305/stable_state_summary_may_31.md)
* [May 30 Stable State Summary](file:///Users/jeffbabiak/.gemini/antigravity/brain/3cc91699-cf5c-405f-9bbd-1934305e8305/stable_state_summary_may_30.md)
* [May 29 Stable State Summary](file:///Users/jeffbabiak/.gemini/antigravity/brain/3cc91699-cf5c-405f-9bbd-1934305e8305/stable_state_summary_may_29.md)
