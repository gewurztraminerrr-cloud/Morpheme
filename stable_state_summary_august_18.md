# Stable State Summary — August 18, 2026

This document records the 'Start Over' stable point for **Morpheme** as of August 18, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is fully aligned and synchronized.

---

## 1. Repository & Commit Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 18, 2026
* **Commit ID**: `e368be213fa66dbd8b1328906322a3cf2c99b821` (`e368be2`)

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
- Incremented global asset version query string (`?v=33014`) across all CSS and JavaScript references in `index.html` to guarantee mobile and desktop clients load the latest scripts without stale cache interference.

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
