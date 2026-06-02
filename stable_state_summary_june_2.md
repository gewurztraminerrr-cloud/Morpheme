# Stable State Summary — June 2, 2026 (Start Over Point)

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `7c0ac97` / `START_OVER_POINT_JUNE_2` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `7c0ac97` / `START_OVER_POINT_JUNE_2` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `7c0ac97` / `snapshot-current` | ✅ Fully Deployed & PM2 Reloaded |

**All environments are 100% synchronized at the latest commit `7c0ac97`.**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py`.
The active recovery points `START_OVER_POINT_JUNE_2` and `snapshot-current` tags have been successfully updated and pushed to GitHub.

---

## Serving Versions (cache-busted)

| File / Style | Version | Description |
|--------------|---------|-------------|
| `/js/play.js` | `v=119` | Standardized validation feedback to "[word] VALID", changed Medium difficulty color to FAQ blue, Either/Or path resolution in public rooms, valid already found words purple highlights, daily rooms missed section only, zero-hesitation, and immediate timer wake-up restoration with fetch locking |
| `/js/app.js` | `v=38` | Clear active room state immediately on gateway transition back to the lobby or other non-play pages to prevent stale player counts and idle timeouts |
| `/css/play.css` | `v=85` | Changed Medium difficulty color to FAQ blue, purple cell tile flash and status text styling classes, and changed bonus word validation flash color to lime green (#32cd32) |
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

### 9. Word Submission Validation Message Standardization
* **Goal achieved:** Unified all valid word submission messages to display `[word] VALID` instead of `Valid Word` or `[word] ACCEPTED` across all gameplay modes (standard, tournament, and private matches).
* **Fix (`game_room.py` & `static/js/play.js`):** Modified the server success message to return `[word] VALID` and updated client local checks and fallbacks to construct and display the standardized text.

### 10. Fix Session Eviction and Stale Player Count on Wake-up
* **Goal achieved:** If a player suspends/minimizes the tab (causing a page reload and gateway transition on wake-up) and returns via the "ENTER LOBBY" button, the active room state on the server is immediately cleared (changing the lobby count to `[0]` immediately instead of waiting for a 10-minute timeout). This also prevents the 10-minute idle kick alert from triggering unexpectedly while the user is active in the lobby.
* **Fix (`static/js/app.js` & `templates/index.html`):** Updated the gateway transition handler `handleGatewayTransition` to be `async` and call `window.leaveCurrentRoom()` when transitioning to any non-play page. Bumped `app.js` import to `v=38` in `templates/index.html`.

### 11. Lime Green Bonus Word Flashing Color
* **Goal achieved:** Changed the validation flash color of bonus words from standard emerald green (`#2ecc71`) to a premium, high-visibility lime green (`#32cd32`) to match the style of the bonus-highlighted cells.
* **Fix (`static/css/play.css` & `templates/index.html`):** Updated the `tile-flash-green` background, border, and box-shadow color definitions to `#32cd32`. Bumped `play.css` to `v=85` in `templates/index.html`.

### 12. Instant Wake-up Timer Restoration & Fetch Locking
* **Goal achieved:** Returning to an active game room after minimizing or backgrounding the tab now restores the countdown timer instantly without showing the sluggish "WAIT..." text. Overlapping concurrent status fetches on mobile connections are locked to prevent browser queue starvation and network delays.
* **Fix (`static/js/play.js` & `templates/index.html`):** Updated focus/visibility listeners to restart the local timer instantly if the round is still active. Introduced `isFetchingState` concurrency lock inside `updateGameState`. Bumped `play.js` to `v=119` in `templates/index.html`.

---

## Key Files Tracked

| File | Location | Purpose |
|------|----------|---------|
| `app.py` | Production + GitHub | Remove obsolete auto-readd player logic from `get_room_state`; update `calculate_morpheme_metric` with the backtracking relocation logic |
| `profile_combo.py` | Production + GitHub | Update the standalone combo checker metric with the backtracking relocation logic |
| `game_room.py` | Production + GitHub | Return f"{final_word} VALID" on successful word submissions; include guest players in snapshots/saves and recover guest usernames |
| `static/js/app.js` | Production + GitHub | Clear active room state immediately on gateway transitions to non-play pages |
| `static/js/play.js` | Production + GitHub | Client-side Either/Or path resolution, standardize success feedback text to `[word] VALID`, purple highlights for duplicates, direct validation checks, hide FOUND section on daily rooms, prevent concurrent state fetch queue congestion, and instantly restore timer countdown on focus/visibility wake-up |
| `static/css/play.css` | Production + GitHub | Changed Medium difficulty color to Vibrant Blue, added purple cell flash and status text highlight styling rules, and changed bonus word validation flash color to lime green (#32cd32) |
| `templates/index.html` | Production + GitHub | Changed Medium difficulty emoji and odds text color in the Spinner Set Odds modal; bump play.js to `v=119`, play.css to `v=85`, and app.js to `v=38` |

---

## Previous Save Points

* [June 1 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_june_1.md)
* [May 31 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_may_31.md)
* [May 30 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_may_30.md)
* [May 29 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_may_29.md)


