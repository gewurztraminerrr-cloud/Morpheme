# Morpheme Stable State Summary - June 19, 2026

This summary documents the stable state of the Morpheme application as of June 19, 2026. Localhost, GitHub origin, and `morpheme.games` are fully synchronized and verified under Commit ID **`f1f1bfa`** (and tagged as **`START_OVER_POINT_JUNE_19`**).

---

## 🚀 Key Improvements & Bug Fixes

### 1. Rating Persistence & App Retrieval Fixes
*   **Syntax Correction**: Resolved several syntax errors in [app.py](file:///Users/jeffbabiak/app.py) (invalid `elif` clauses left behind after removing overrides) in `create_room`, `join_room`, and the `get_room_state` reconstruction.
*   **Solo Match Persistence**: Reverted the hardcoded `1200` rating override in `start_solo_match` to dynamically load configuration-specific ratings for FCFS/split/3D solo play.
*   **Database Retention**: Allows active room configuration ratings to persist correctly inside `user_ratings` instead of reverting back to the `1200` starting default when exiting/re-entering a room.

### 2. My Rating Button Resolution & Visibility
*   **Configuration Specificity**: Reconfigured the `#my-rating-btn` click handler and updater in [lobby.js](file:///Users/jeffbabiak/static/js/lobby.js) to resolve configuration-specific ratings from `window.currentUserConfigRatings` for all game configurations (including FCFS/split/3D) instead of bypassing them.
*   **Correct Defaults**: If no rating exists in the database for the active room configuration, the button defaults to `1200` for active room game types (`fcfs`, `split`, `3d`) and the global rating (`window.lastPlayerRating || 1200`) for accumulative setups.
*   **Mobile Visibility**: Restored visibility of the "My Rating" button on mobile layout screen widths within `resetLobbyButtons()`.
*   **Immediate Lobby Updates**: Programmed [app.js](file:///Users/jeffbabiak/static/js/app.js)'s `showPage('page-lobby')` logic to fetch the latest configuration ratings whenever returning to the lobby page. This updates the local rating client cache immediately post-round.
*   **Average Rating Textbox Behavior**: The average rating filter textbox is now left blank by default when users select or go to any active room configuration, allowing them to click the "My Rating" button to populate it manually.

### 3. Static Asset Caching & Instant Audio Playback
*   **The Issue**: The global `add_cache_headers` middleware forced a `no-store, no-cache` policy on all requests, disabling caching for static assets. This required browser range requests to redownload the large 7.4 MB `lobby.mp3` on every refresh, leading to noticeable buffering delays when seeking to the 205s starting loop.
*   **Caching Optimization**: Updated `add_cache_headers` in [app.py](file:///Users/jeffbabiak/app.py) to identify static file endpoints (endpoint names or path extensions matching `.mp3`, `.wav`, `.js`, `.css`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.ico`) and apply a public cache policy (`Cache-Control: public, max-age=604800`) while keeping dynamic API endpoints uncached.
*   **Instant Playback**: The browser now caches static assets locally, allowing the music player to seek to `205` seconds and start playing instantly on reload/navigation.

### 4. Vertical Mobile Board Transposition
*   **Double-Transposition Bug**: Resolved the layout issue on mobile devices where board coordinates and selection paths got swapped twice.
*   **Unified Transposition**: Transposed grid arrays directly in a single `safelyTransposeState` utility applied across normal matches, tournament plays, and private matches, and mapped intermission path filters correspondingly.

### 5. Registration Initial Settings
*   **Default DB Init**: Added settings initialization during account creation to insert default board sizes and default selectable corner cutoff (39) into the database.

### 6. Motivating Rating brackets
*   **Inspiring Bracket Descriptions**: Populated custom bracket descriptions for the 35 color segments in `RATING_RANGES`. Clicking any segment of the `#game-color-bar` in the header now brings up the detailed motivational modal.

### 7. 24h Room Issue Fixes
*   **Daily Reset Eviction**: Active users and spectators in 24h rooms are now immediately evicted back to the Lobby at Chicago midnight when a daily reset occurs (round changes), preventing silent auto-rejoins and stale roster counts.
*   **Stale Board Elimination**: Reconstructed public 24h rooms with empty boards automatically trigger a background thread to generate the daily board immediately, resolving flashing or incorrect boards on first login.
*   **Stale Stats Sync ("21/4" Fix)**: Properly capture and save `previous_total_words` and `previous_total_points` on reset. Updated database loading logic in `load_previous_day_data()` to query `total_words_avail` and reconstruct yesterday's word scores from the database rather than relying on dynamic fallbacks.

### 8. Solo Settings Tweaks
*   **10 Minutes Option**: Added "10 Minutes" option (`600` seconds) to the Solo time limit configuration, allowing players to play longer practice sessions.
*   **Board Format Random Label**: Removed the misleading `(14% special)` text from the "Random" option inside Solo's Board Format dropdown.

### 9. Store Mobile View Alignment Fix
*   **Category Tab Sizing**: Added mobile styling rules under a `(max-width: 900px)` media query in `static/css/lobby.css` (expanded from 600px to cover tablet and simulated mobile layouts). The store tabs now flex to fill the screen evenly, with optimized gap (`4px`), padding (`6px 2px`), and font size (`0.72rem`) along with reduced `#page-store` padding (`10px`). This ensures all buttons ("Hardware", "Themes", "Avatars", "Perks") fit perfectly on any mobile device viewport together without distorting the top menu header container.

---

## 🛠 Active Features & Configuration
*   **Board Formats**: Normal, Checkerboard, Double, Triple, Valued Letters, Rotation, Penalty, Mania, Either/Or, Bonus Word, and Density.
*   **Grid Dimensions**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
*   **Dictionaries**: NWL (American) and CSW (International) Tries.
*   **Verification**: All python compilation checks and rating logic test scenarios verify as green.

---

**Latest Stable Commit ID**: `f1f1bfa` (tagged as `START_OVER_POINT_JUNE_19`)  
**GitHub Tag**: `START_OVER_POINT_JUNE_19`  
**Localhost & GitHub Sameness Status**: Synchronized  
**Production Server Status**: Green / Auto-Restart protection active / Live at commit `f1f1bfa`
