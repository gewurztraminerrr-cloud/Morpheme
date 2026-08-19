# Stable State Summary — August 19, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of August 19, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is synchronized.

---

## 1. Repository & Deployment Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 19, 2026
* **Commit ID**: `46d870e674f919f0b16addf43bff2f10534904b8` (`46d870e`)
* **Asset Version**: `v=33060`

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Session Expired Notice Suppression on > 1 Hour Return (`static/js/app.js`, `static/js/play.js`, `templates/index.html`)
- **Activity Timestamp Tracking**: Continuously tracks user interactions (`mousedown`, `keydown`, `touchstart`, `scroll`, heartbeat interval, and `beforeunload`) via `morpheme_last_active_timestamp` in `localStorage`.
- **Silent Room Cleanup on Fresh Return**: If a user closes Morpheme while in a room and returns after more than an hour ($> 3600\text{s}$), the app silently clears stale room state (`localStorage.removeItem('last_joined_room')`, `window.currentRoomId = null`) and suppresses the "Session Expired" / inactivity ejection modal popup.
- **Clean Inactivity Ejection Handling**: `ejectToLobby("inactivity")` in `play.js` verifies if the user's absence exceeded an hour; if so, it transitions silently to the lobby without presenting the popup notice.

### B. Accumulative Lobby Real-Time Auto-Polling & Live Count Synchronization (`static/js/lobby.js`, `static/js/app.js`, `templates/index.html`)
- **Global Polling Lifecycle**: Added top-level `lobbyStatsInterval` reference and exported `window.startStatsPolling` / `window.stopStatsPolling`.
- **Infallible Lobby Page Detection**: Replaced static DOM element lookup in `isOnLobby()` with `window.currentPageId === 'page-lobby'` and dynamic visibility verification.
- **Active Navigation Trigger**: Wired `showPage('page-lobby')` in `app.js` to immediately initiate full button stats updates (`fetchLobbyStats('all')`) and activate background polling upon lobby entry.
- **Fast Auto-Poll Interval**: Tuned background auto-polling from 4 seconds to 2 seconds (`2000ms`), ensuring player count updates (`Start [0]` $\rightarrow$ `Start [1]`) reflect almost instantly across all connected computers as players enter and exit rooms.

### C. Unscramble Tool Desktop & Laptop Full-Width Panel Expansion (`templates/index.html`, `static/css/play.css`)
- **Full Horizontal Width**: Removed the restrictive `max-width: 600px` bottleneck from `.unscramble-game-area` and `#unscramble-found-container`, expanding both to `max-width: 1200px` (full width of tools workspace).
- **Responsive Single-Row Sequence Fitting**: Styled `#unscramble-jumbled` with `white-space: nowrap !important`, `text-align: center !important`, and responsive font clamp scaling (`clamp(1.6rem, 3.8vw, 3.2rem)`) with fluid letter-spacing (`clamp(5px, 1.2vw, 15px)`), ensuring all jumbled sequences (up to 21 letters long) fit cleanly on one single line across Acer laptops, Mac laptops, and desktop screens without clipping or overflowing.

### D. Store Tool Collins Official Scrabble Words 2–15 Word List Clarification (`templates/index.html`)
- **Word List (No Definitions) Note**: Explicitly noted in the description and features bullet points that the Collins 2–15 book is strictly a comprehensive word list and contains no definitions.

### E. Acer Laptop & Desktop Layout Optimization (`static/css/play.css`)
- **Words List & Definitions Box Balance**: Set `.definitions-panel` to **`130px`** with clean `0.95em` header and `0.9em` body text, giving generous reading space for full lexicographical definitions while preserving vertical space for found words.
- **Players Panel & Chatbox Balance**: Set `.chat-panel` to **`150px`** with `28px` input height, giving spacious room to view 4–5 chat lines comfortably.
- **Sequence, Subanagrams & Lists Word Tables Max-Expansion**: Expanded `#page-tools` container height to `calc(100vh - 40px)` with reduced padding (`8px 12px` from `30px`), streamlined tool headers and control inputs, and set all results tables, columns, and scroll areas to full `100%` flex height, showing dozens of rows at once on Acer laptops and desktop viewports.

### F. Permanent 24h Daily Room Score Sums & 12AM Rollover Preservation (`app.py`, `game_room.py`, `static/js/play.js`)
- **Fixed 12AM Midnight Score Rollover**: Fixed the race condition where `self.players` was cleared before capturing `intermission_player_snapshots`, preventing scores from being credited to `daily_score_sums`. Players are now preserved through intermission, accurately snapshotted, and added to the cumulative permanent score sums in SQLite `daily_score_sums`.
- **Canonical 24h Room Keys**: Standardized all 24h room keys across database storage and API endpoints to `24h_4x4`, `24h_4x6`, `24h_5x7`, and `24h_6x8`, ensuring independent word counts and permanent score sums across all four 24h room sizes.
- **Stable 24h Room Singletons**: Configured `/api/room/create` to always assign stable singleton room IDs (`pub_v2_accumulative_{dims}_86400`) rather than random UUIDs, keeping players in the same persistent room.
- **Live Active Scores & Instant Non-Empty Rankings**: Enhanced `/api/daily-score-sums` to dynamically incorporate active daily players with scores > 0 so that once a user establishes a score in any 24h room, the Score Sum tab never reads "No players found" again.

### G. Critical Deadlock Resolution (`game_room.py`)
- Removed recursive/nested `delete_room()` calls inside `remove_player()` when triggered from `cleanup_rooms()`.
- Replaced with an `is_closing = True` flag checked during room cleanup cycles, completely preventing thread hangs, SQLite database lockouts, and login timeouts.

### H. Dedicated Lobby Refresh Button & Clean Interaction Rules (`templates/index.html`, `static/js/lobby.js`)
- Added a compact `🔄` Refresh button adjacent to "My Rating" in the Active Rooms panel.
- On user click: queries `/api/lobby-stats`, refreshes FCFS & Split Points button badges, and re-renders active room listings (player pills, average rating, empty room cleanup).
- Accumulative buttons auto-poll every 2 seconds; FCFS & Split Points update on entry, refresh click, or "Show Rooms" click.

### I. Open Rooms & Closed Rooms Lobby Tabs (`templates/index.html`, `static/js/lobby.js`, `static/css/lobby.css`)
- **Open Rooms & Closed Rooms Buttons**: Dedicated tab buttons along the same row as `My Rating` and `🔄 Refresh`.
- **Mobile Single-Row Layout**: Fits neatly into a single row on mobile devices with stacked text.
- **Clean Qualification Rules**: Open Rooms qualify if user rating is within room limits and player count `< 8` (shows `Join` and `Spectate`); Closed Rooms qualify if full or rating outside room limits (shows `Spectate` only).

### J. User Follow Spectator Mode Logic (`app.py`, `play.js`, `tools.js`)
- **Authoritative Rating & Capacity Enforcement**: Users following friends into full rooms or rooms outside their rating limits are automatically transitioned into `SPECTATING` mode.

### K. Forum Multi-Image Upload (4 Images Cap & macOS/Cross-Platform Reliability) (`app.py`, `static/js/forum.js`, `templates/index.html`)
- Increased image capacity from 3 to 4 with unified non-double-triggering upload bridge across macOS, desktop, and mobile browsers.

### L. Realistic 3D "ENTER LOBBY" Button (`templates/index.html`, `static/css/style.css`, `static/js/app.js`)
- 3D physical socket housing with multi-tiered 12-layer extrusion box shadows and persistent flattened press-down state.

### M. Lists Tool "View Full List" Anti-Copy & Exact Spot Return (`templates/index.html`, `static/css/play.css`, `static/js/tools.js`)
- Selection/copying suppression with interactive definition lookup on word click and seamless return to exact scroll position.

---

## 3. Production Deployment Instructions

To synchronize the live server on `morpheme.games` with this exact commit:

```bash
cd /home/morpheme/morpheme && git pull origin main && pm2 restart all
```
