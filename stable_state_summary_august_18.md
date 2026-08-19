# Stable State Summary — August 18, 2026

This document records the 'Start Over' stable point for **Morpheme** as of August 18, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is fully aligned and synchronized.

---

## 1. Repository & Commit Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 18, 2026
* **Commit ID**: `495722b6b0c2656fe1be34d5fcb0a9446d61fb3c` (`495722b`)

---

## 2. Key Features & Fixes Included in This Stable State

### A. Critical Deadlock Resolution (`game_room.py`)
- Removed recursive/nested `delete_room()` calls inside `remove_player()` when triggered from `cleanup_rooms()`.
- Replaced with an `is_closing = True` flag checked during room cleanup cycles, completely preventing thread hangs, SQLite database lockouts, and login timeouts.

### B. Lobby Room Visibility & Accurate Counts (`app.py`, `lobby.js`, `index.html`)
- **Empty Room Filtering**: Strict filter ensuring 0-human rooms are never rendered in the lobby active rooms list.
- **Player Pills in Room Cards**: Active rooms properly display all human player usernames and rating indicators.
- **Button Count Updates**: `Show Rooms [N]` and `Start [N]` reflect live player counts accurately upon lobby arrival.
- **FCFS Button Text Normalization**: Single-line button text format ensuring regex patterns accurately target and update `[N]` badges.

### C. Dedicated Lobby Refresh Button (`templates/index.html`, `static/js/lobby.js`)
- Added a compact `🔄` Refresh button adjacent to "My Rating" in the Active Rooms panel.
- Reuses the animated spin effect (`profileRefreshSpin`) matching the Profile page.
- On user click: queries `/api/lobby-stats`, refreshes FCFS & Split Points button badges, and re-renders active room listings (player pills, average rating, empty room cleanup) if a game configuration is open. Per specification, Refresh does not touch Accumulative buttons.

### D. Game-Type Specific Polling & Interaction Rules (`static/js/lobby.js`)
- **Lobby Entry (`fetchLobbyStats('all')`)**: Immediately fetches and displays live, authoritative numbers across all buttons (`Show Rooms [N]` and `Start [N]`) at the moment of entry.
- **Accumulative Matrix (`acc-btn`)**: Auto-polls every 4 seconds in the background so player counts (`Start [N]`) stay updated in real time as players enter/exit.
- **FCFS & Split Points Matrix (`fcfs-btn`, `split-btn`)**: Does not auto-poll or shift room statuses unprompted while waiting in a configuration; newly created rooms never display automatically while viewing a list. Listings and button counts update strictly on explicit user action (clicking "Show Rooms" or pressing the 🔄 Refresh button).
- **Show Rooms Click**: Immediately fetches and renders active rooms in the right panel and updates the player count badge on that specific button. Completely removed legacy auto-create logic so viewing rooms never creates rooms.

### E. Client Asset Cache Busting (`templates/index.html`)
- Incremented global asset version query string (`?v=33032`) across all CSS and JavaScript references in `index.html` to guarantee mobile and desktop clients load the latest scripts without stale cache interference.

### F. CPU Throttling & Server Usage Optimization (`board_generator.py`, `game_room.py`)
- **Micro-Yield in Generation Loop**: Added a `5ms` micro-pause during board retry loops in `_generate_board_internal`, capping peak generation CPU below host alert thresholds.
- **Single-Worker Refill Cap**: Restricted background cache refills to 1 concurrent worker with a 20s inter-board generation pacing.
- **Heartbeat Loop Pacing**: Adjusted `_bg_cleanup_loop` interval in `game_room.py` from `0.1s` (100ms) to `0.25s` (250ms), reducing idle thread wakeups by 60% while maintaining crisp round transitions.
- **Startup Seeder Throttle**: Paced `seed_pregenerated_cache_bg` config seeding with 5-second delays to prevent initial boot CPU spikes.

### G. Spectator Mode Room Loading & Stale Room Handling (`game_room.py`, `play.js`, `lobby.js`, `index.html`)
- **Missing `get_spectator` Method**: Added `get_spectator(self, user_id)` to `GameRoom` class, resolving a 500 server error crash during `/api/room/<id>/state` polling for spectators.
- **Spectator State Transition**: Properly initialized `window.isSpectatorMode` in `handleJoinRoomInline`, restoring full board loading.
- **"Join Room" Button & Rating Restrictions**: Rooms with open slots display the prominent **"Join Room"** button for players within the rating range, and display clean **`SPECTATING`** mode when spectating a rating-restricted room outside their range.
- **404 Stale Room Cleanup**: If a user clicks a room card that has ended or expired, the client alerts *"This room has ended or is no longer active"* and immediately re-fetches the live room list to remove the stale card.

### H. Open Rooms & Closed Rooms Lobby Tabs, Spectate Rules & FAQ (`templates/index.html`, `lobby.js`, `lobby.css`)
- **Open Rooms & Closed Rooms Buttons**: Replaced the legacy `Find` button with dedicated `Open Rooms` and `Closed Rooms` tab buttons along the same row as `My Rating` and `🔄 Refresh`.
- **Mobile Single-Row Layout**: On mobile/tablet screens, all four buttons (`Open Rooms`, `Closed Rooms`, `My Rating`, and `🔄`) fit neatly into a single row below the proximity input, with button text stacked vertically ("Open / Rooms", "Closed / Rooms", "My / Rating").
- **Open Rooms Qualification**: Rooms qualify as Open if the user's rating is within the set limits (or unrestricted) **and** player count is `< 8`. Displays both `Join` and `Spectate` buttons.
- **Closed Rooms Qualification & Spectate-Only**: Rooms qualify as Closed if player count `== 8` (full) or rating is outside the room's limits. In Closed Rooms, **ONLY** the `Spectate` button is visible.
- **No Background Auto-Shifts**: If a room drops from 8 to 7 players, it remains in the current list until the user explicitly clicks `Refresh`.
- **Full Room Join Alert Popup**: If a room becomes full before clicking Join, alerts *"This room is now full. Please press the Refresh button to update the list of Open Rooms and Closed Rooms."* and refreshes room cards.
- **Scroll Position Preservation**: Captures and restores scroll position on `rooms-list` and dynamic containers across refreshes so list position and relative average rating views remain stable.
- **Comprehensive FAQ Section**: Added a dedicated FAQ entry and Quick Navigate shortcut covering all Active Rooms mechanics.

### I. User Follow Spectator Mode Logic (`app.py`, `play.js`, `tools.js`)
- **Authoritative Rating on Follow & Polling**: The server passes the user's exact configuration rating (`your_rating`) in `/api/room/<id>/state` snapshot.
- **Rating Limit Enforcement on Follow**: When following a player into a room whose rating limit does not align with the follower's rating (e.g. 1191 rating following into 1193–1300), the user is automatically transitioned into **`SPECTATING`** mode (`as_spectator: true`, `window.isSpectatorMode = true`).
- **8-Player Capacity Enforcement on Follow**: When following a user into a room that already has 8 people playing, the user is placed directly into **`SPECTATING`** mode.
- **Server Auto-Sync Safeguard**: `get_room_state` polling automatically categorizes unassigned users as spectators rather than active players if the room is full or if the user's rating is outside room limits, preventing mid-round rating/capacity bypasses.
- **Spectator Panel Display**: The definitions panel renders clean `SPECTATING` header without the "Join Room" button when rating limits are not met.

### J. Word Submission Robustness & Spectator Isolation (`game_room.py`, `play.js`)
- **Server Authoritative Validation**: Removed client-side early rejection return in `submitWord` (`play.js`), ensuring all submitted words reach the backend `/room/<id>/submit_word` endpoint for authoritative validation and dynamic board self-healing.
- **Active Player vs. Spectator Clean State**: Added strict cross-list pruning in `add_player` and `add_spectator` (`game_room.py`). When an active player submits words, `submit_word` verifies their active player status first and clears any stale spectator records.

### K. Tournament Finalized Champion Announcement (`static/js/tournaments.js`)
- Updated completed tournament finalized state banner to display: *"The champion has been crowned! Congratulations to [username of winner]! The next tournament signup period will begin shortly."*

### L. Tournament Play Words Panel Simplification (`static/js/play.js`)
- During Tournament rounds, the Words and History tab navigation bar (`#words-tabs-container`) is hidden, cleanly displaying the direct stream of words found by the player with point values and definition click inspection.

### M. Tournament Hall of Fame Trophy Icon (`templates/index.html`)
- Added a trophy icon (🏆) adjacent to the Tournament Hall of Fame section heading (`🏆 Tournament Hall of Fame`).

### N. Tournament Championship Bracket Gold Winner Styling (`static/css/style.css`, `static/js/tournaments.js`)
- Replaced the purple styling for tournament winners with radiant gold styling (`#ffd700`, `rgba(255, 215, 0, 0.15)` background with matching glowing indicator dot and `🏆 Champion` badge).

### O. Mobile Game Room Navigation Smooth Sliding Animation (`static/js/play.js`, `static/js/app.js`)
- Enabled smooth animated sliding transitions (`scrollTo({ left: targetLeft, behavior: 'smooth' })`) when tapping mobile bottom navigation buttons (`Players`, `Board`, `Words`) across game rooms, matching the sleek sliding experience of Tools and Settings.

### P. Desktop/Laptop Active Rooms Req Rating Position (`static/js/lobby.js`, `static/css/lobby.css`)
- On desktop and laptop viewports, the `Req: [range]` rating badge is positioned horizontally directly to the left of `Avg Rating: [rating]` in the Active Rooms list cards.

### Q. Forum Multi-Image Upload (4 Images Cap & macOS/Cross-Platform Reliability) (`app.py`, `static/js/forum.js`, `templates/index.html`)
- **4 Images Capacity**: Increased attachable image capacity from 3 to 4 for both threads and replies across backend ingestion (`app.py`), UI labels, upload limits, and preview counters.
- **macOS / Mac Laptop Multiple Upload Reliability**: Converted inner upload label containers to standard `<div>` elements, eliminating native double-triggering on Mac laptops (Safari and Chrome on macOS) and standardizing upload bridge behavior across mobile devices, desktops, Acer laptops, and Mac laptops.

### R. Mini-Profile Round Reviews Auto-Scroll Navigation (`static/js/tools.js`)
- When a user selects **Round Reviews** from any mini-profile overlay, the profile page opens with the **Round Reviews** tab active and automatically smooth-scrolls down to the reviews section for immediate inspection.

### S. Unscramble Tool HTML Structure & Initialization Fix (`templates/index.html`, `static/js/tools.js`)
- Resolved a stray extra `</div>` tag immediately following the Word Lists tool that was prematurely closing `.tools-content` and causing the `#tool-unscramble` container to be placed outside the tools content area (preventing it from receiving the `.active` class).
- Hooked `showTool('unscramble')` to automatically load and generate a new game round if the board is empty, ensuring full interactive content renders instantly when selecting Unscramble from the Tools sidebar.

### T. Lobby "My Rating" Initial State & Unselected Room Protection (`static/js/lobby.js`, `templates/index.html`)
- Cleared the initial hardcoded config in `lobby.js` by setting `currentLobbyConfig = null` and `window.currentLobbyConfig = null` upon lobby entry.
- Safeguarded `handleMyRatingButtonClick` and the `#my-rating-btn` click listener so that pressing "My Rating" when no room matrix configuration is selected leaves the "Sort rooms by proximity to average rating" textbox completely blank and performs no action.

### U. Settings "Cube Scale (3D)" Mobile Device Removal & FAQ Update (`templates/index.html`, `static/css/play.css`, `static/js/settings.js`)
- Completely hid the "Cube Scale (3D)" panel in Settings on mobile devices using CSS media queries (`max-width: 900px`), `body.is-mobile`, and runtime JavaScript detection (`applyMobileSettingsVisibility()`), restricting the setting exclusively to desktops and laptops.
- Updated the FAQ section title and explanation to clarify why 3D Cube mode and Cube Scale are exclusive to desktops and laptops, highlighting the lack of an efficient word entry method and the lack of screen real estate for a dedicated word entry textbox on mobile screens.

### V. Combo Checker Mobile MP Tables Vertical Length Expansion (`static/css/play.css`)
- Removed the legacy 250px fixed section height on `.result-section` and expanded mobile table container heights (`min-height: 400px; height: 420px;`) with inner table scroll areas (`min-height: 330px; height: 350px;`).
- Gave MP tables (0MP, 1MP, 2MP, 3MP) the same generous vertical length as 5LIC/LIC tables on mobile devices, ensuring full word lists remain clearly visible and scrollable.

### W. Lobby "My Rating" Configuration Rating Accuracy Fix (`static/js/lobby.js`, `templates/index.html`)
- Fixed `getUserConfigRating` in `lobby.js` and `index.html` which previously fell back to the user's global overall rating (e.g. `1205`) for unplayed configurations instead of the default `1200`.
- Ensured all "My Rating" button displays (`My Rating ([rating])`) and button clicks precisely reflect the specific configuration rating shown on the user's Profile page ratings grid (e.g., 1191 for FCFS 4x4 45s, and 1200 for unplayed configurations).

### X. Unscramble Sequence Stabilization & Persistent Session History (`static/js/tools.js`)
- **Eliminated Jumbled Word Flash / Changing**: Introduced an `isLoading` mutex and eliminated redundant duplicate generators on input focus, timeout, and tab switching, ensuring only one clean generation request runs and preventing the first jumbled word from flickering or changing sequences upon arrival.
- **Persistent Session History**: Added automatic `localStorage` synchronization (`morpheme_unscramble_history`), ensuring completed and revealed rounds are permanently retained across page reloads and tab navigations.
- **Rich Session History UI**: Always renders the Session History section under Status & Results, displaying the jumbled puzzle, time, found count vs total solutions, color-coded solution pills with definition lookups, and a "Clear History" action button.

### Y. Game Board Loading Card Centering & Alignment Fix (`static/css/play.css`, `static/js/play.js`)
- Resolved the layout issue where the "Generating [Format]…" loading card would occasionally appear scrunched along the left side before pressing Rotate.
- **Enhanced CSS Specificity**: Styled `.game-board-loading` and `#game-board.game-board-loading` across desktop and responsive mobile rules with `display: flex !important`, `align-items: center !important`, `justify-content: center !important`, `margin: 0 auto !important`, and `grid-template-columns: none !important`.
- **Clean Style Reset**: Updated `ensureLoadingCardStyles()` to programmatically clear any stale grid template column constraints and set clean flex-centering properties, ensuring the loading spinner, title, status ticker, and explanation text always render perfectly centered and readable.

---

## 3. Production Deployment Instructions

To synchronize the live server on `morpheme.games` with this exact commit:

```bash
cd /home/morpheme/morpheme && git pull origin main && pm2 restart all
```

---

## 4. Verification Checklist

1. **Lobby Arrival**: Opening the lobby immediately loads fresh room numbers across all matrix buttons without requiring manual action.
2. **Accumulative Matrix**: Entering/exiting an Accumulative room automatically updates the `Start [N]` badge for all players viewing the lobby.
3. **FCFS / Split Points**: Room status, player pills, and average rating updates strictly on "Show Rooms" click and "Refresh" button click. Newly created rooms do not appear unprompted while viewing a list.
4. **Open & Closed Rooms Tabs**: Switching between Open Rooms and Closed Rooms displays the proper qualifying rooms. Closed Rooms strictly show the Spectate button only.
5. **Full Room Popup Notification**: Joining a full room notifies the user to press Refresh to update the list of Open Rooms and Closed Rooms.
6. **Scroll Position Retention**: Refreshing room listings retains the user's scroll position in the rooms panel.
7. **User Follow Rating Restriction**: Following a user with a rating outside the room limit (e.g., 1191 rating into 1193–1300) places the user directly into `SPECTATING` mode.
8. **User Follow 8-Player Cap**: Following a user into an 8-player full room places the user directly into `SPECTATING` mode.
9. **Spectator Mode**: Clicking "Spectate" on any active room immediately loads the room, board, timer, and the definitions panel status.
10. **Refresh Animation**: Clicking the 🔄 button triggers the smooth rotation indicator and updates all lobby numbers.
11. **No Thread Lockups**: Database lock errors during login and room creation remain completely eliminated.
12. **Smooth Server Load**: Board transitions and background refills remain well below hosting provider high-resource thresholds.
