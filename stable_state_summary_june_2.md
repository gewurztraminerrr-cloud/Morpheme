# Stable State Summary — June 2, 2026 (Start Over Point)

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `91d82dd` / `START_OVER_POINT_JUNE_2` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `91d82dd` / `START_OVER_POINT_JUNE_2` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `91d82dd` / `snapshot-current` | ✅ Fully Deployed & PM2 Reloaded |

**All environments are 100% synchronized at the latest stable commit `91d82dd`.**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py`.
The active recovery points `START_OVER_POINT_JUNE_2` and `snapshot-current` tags have been successfully updated and pushed to GitHub.

---

## Serving Versions (cache-busted)

| File / Style | Version | Description |
|--------------|---------|-------------|
| `/js/play.js` | `v=117` | Changed Medium difficulty color to FAQ blue, Either/Or path resolution in public rooms, valid already found words purple highlights, daily rooms missed section only, and zero-hesitation |
| `/css/play.css` | `v=84` | Changed Medium difficulty color to FAQ blue, purple cell tile flash and status text styling classes |
| `/js/lobby.js` | `v=5` | Rating filter proximity sorting and search listener registrations |
| `/css/lobby.css` | `v=17` | Premium flex rating search bar layout and Find button style |
| Inline `<style>` block | *Dynamic* | Mobile WebView cache bypass styling in `index.html` |

---

## Work Completed Up To June 2, 2026

### 1. Fix Lobby Player Count Refresh
* **Goal achieved:** Entering a room and then returning to the lobby by pressing the "Lobby" button in the top menu immediately decrements the player count of that room on the lobby buttons (e.g. from `[1]` to `[0]`), eliminating the stale player count race condition.
* **Bug Fix (`app.py`):** Removed the obsolete auto-re-add player block from the GET `/api/room/<room_id>/state` endpoint. This prevents in-flight `/state` polling requests from automatically re-adding players who explicitly left.

### 2. Combo Checker Metric Backtracking Update
* **Goal achieved:** Relocating a letter further down (such as in `SUBRACE`/`UBRACE` $\rightarrow$ `RUBACE`) now correctly counts as exactly **1 MP** (1 relocation) rather than being treated as 2 MP (remove and insert).
* **Backtracking Implementation (`app.py` & `profile_combo.py`):**
  * Replaced the LCS backtrace with a backtracking search algorithm evaluating index crossings and LIS.
  * Pruning guarantees execution in less than 0.1ms for standard word lengths ($\le 10$).

### 3. Purple Highlights for Already Found Words
* **Goal achieved:** If a valid word is entered but has already been found, it highlights in premium **purple** instead of red. This applies across all play styles.

### 4. Active Gameplay Validation Flashing Fix
* **Goal achieved:** Completely resolved active gameplay double-flashing by sourcing standard play validation directly from `preState.all_words` rather than the stale render cache.

### 5. Backend Guest Player History Persistence (24h Rooms)
* **Goal achieved:** Guest accounts in 24h rooms now successfully have their yesterday's words stored and loaded in the "Previous Day" tab by snapshotting `p.is_guest` and recovering guest usernames in `get_yesterdays_history`.

### 6. Hide FOUND Section in 24h Room Previous Day Tab
* **Goal achieved:** Hides the FOUND section completely in daily 24h rooms under the "Previous Day" tab, displaying only the MISSED section.

### 7. Either/Or Tile Flashing Fix in Public/Standard Rooms
* **Goal achieved:** Entering a word that uses an Either/Or tile (e.g. `L/T`) in public/standard rooms via drag/mouse immediately flashes blue (valid) or green (bonus) without showing a brief red (invalid) flash first.
* **Fix (`static/js/play.js`):** Replicated the private matches path resolution logic within `submitWord` in `play.js`. The client resolves the swipe path against `window.lastGameState.all_words` beforehand, allowing correct validation matching between local and remote checks.

### 8. Medium Difficulty Color Sync (Spinner Set & FAQ)
* **Goal achieved:** Changed the Medium difficulty color inside the Spinner Set parameters and Spinner Odds modal to match the Vibrant Blue color (`#60a5fa`, emoji `🔵`) used in the FAQ section, instead of golden/yellow.

---

## Key Files Tracked

| File | Location | Purpose |
|------|----------|---------|
| `app.py` | Production + GitHub | Remove obsolete auto-readd player logic from `get_room_state`; update `calculate_morpheme_metric` with the backtracking relocation logic |
| `profile_combo.py` | Production + GitHub | Update the standalone combo checker metric with the backtracking relocation logic |
| `game_room.py` | Production + GitHub | Include guest players in snapshots and complete round saves for 24h rooms, and correctly map guest usernames |
| `static/js/play.js` | Production + GitHub | Client-side Either/Or path resolution in `submitWord()`, purple highlights for duplicates, direct authoritative validation checks, and hide FOUND section on daily rooms |
| `static/css/play.css` | Production + GitHub | Changed Medium difficulty color to Vibrant Blue, added purple cell flash and status text highlight styling rules |
| `templates/index.html` | Production + GitHub | Changed Medium difficulty emoji and odds text color in the Spinner Set Odds modal; bump play.js to `v=117` and play.css to `v=84` |

---

## Previous Save Points

* [June 1 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_june_1.md)
* [May 31 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_may_31.md)
* [May 30 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_may_30.md)
* [May 29 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_may_29.md)
