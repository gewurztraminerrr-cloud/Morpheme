# Stable State Summary — August 18, 2026

This document records the 'Start Over' stable point for **Morpheme** as of August 18, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is fully aligned and synchronized.

---

## 1. Repository & Commit Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 18, 2026
* **Commit ID**: `9dbc14299b9cf995733d0ba5ae64bdf602a8bfaf` (`9dbc142`)

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
- Incremented global asset version query string (`?v=33003`) across all CSS and JavaScript references in `index.html` to guarantee mobile and desktop clients load the latest scripts without stale cache interference.

### F. CPU Throttling & Server Usage Optimization (`board_generator.py`, `game_room.py`)
- **Micro-Yield in Generation Loop**: Added a `5ms` micro-pause during board retry loops in `_generate_board_internal`, capping peak generation CPU below host alert thresholds.
- **Single-Worker Refill Cap**: Restricted background cache refills to 1 concurrent worker with a 20s inter-board generation pacing.
- **Heartbeat Loop Pacing**: Adjusted `_bg_cleanup_loop` interval in `game_room.py` from `0.1s` (100ms) to `0.25s` (250ms), reducing idle thread wakeups by 60% while maintaining crisp round transitions.
- **Startup Seeder Throttle**: Paced `seed_pregenerated_cache_bg` config seeding with 5-second delays to prevent initial boot CPU spikes.

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
3. **FCFS / Split Points**: Room status, player pills, and average rating updates on "Show Rooms" click and "Refresh" button click.
4. **Refresh Animation**: Clicking the 🔄 button triggers the smooth rotation indicator and updates all lobby numbers.
5. **No Thread Lockups**: Database lock errors during login and room creation remain completely eliminated.
