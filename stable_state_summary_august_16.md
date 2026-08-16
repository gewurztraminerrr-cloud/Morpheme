# Stable State Summary — August 16, 2026

## Commit ID (HEAD)
```
04f79ac  Critical fix: close setupLobbyEvents — all polling functions were trapped inside it
```

## GitHub
- **Repo:** https://github.com/gewurztraminerrr-cloud/Morpheme.git
- **Branch:** `main`
- **Status:** Local == GitHub (up to date, clean)

## Deployment
To sync morpheme.games with this state:
```bash
cd /home/morpheme/morpheme && git pull origin main && pm2 restart all
```

---

## What Was Fixed in This Session (Aug 15–16)

### 1. Critical Deadlock Fix (`game_room.py`) — `51f1467`
- `remove_player()` was calling `delete_room()` (acquires `self.lock`) from inside `cleanup_rooms()` which iterates `self.rooms` → thread deadlock → Flask thread pool exhausted → SQLite "database is locked" → login/room creation failing with 1-minute timeouts.
- **Fix:** Removed `delete_room()` from `remove_player()`. Now sets `is_closing=True` only. `cleanup_rooms()` detects the flag and deletes on next tick.

### 2. `remove_player` Guard Fix (`game_room.py`) — `858cfdb`
- Added `leaving_player` guard to prevent `cleanup_user_rooms()` from destroying a brand-new room during the create→cleanup→add_player window.

### 3. Lobby: Room Cards Show Player Names (`app.py`, `lobby.js`) — `cc67b73`, `054899c`
- `list_rooms()`: removed 60s grace period; only shows rooms with ≥1 human; added `game_type`, `board_dimensions`, `time_limit` fields to response.
- `get_lobby_stats()`: fixed missing `is_daily` variable (NameError silently swallowed stats loop → all buttons showed `[0]`); fixed `max(1, len(humans))` → `len(humans)`.
- Client-side filter: `rooms = rooms.filter(r => r.players && r.players.length > 0)` — never renders empty room cards.
- Fixed fallback inline renderer in `index.html` to render full player pills (it was rendering only ACTIVE badge + Avg Rating with no player names).

### 4. Lobby: Show Rooms `[N]` Counter Updates (`lobby.js`, `index.html`) — `d94f66c`, `9ac4f92`, `63c3937`
- Button count now updated directly inside `fetchAndRenderRooms` from rooms data (no extra network request, no separate stats interval race).
- Fixed FCFS button HTML: text was split across two lines (`Show\n    Rooms [0]`), breaking whitespace-sensitive regex.
- `updateLobbyButtons` now normalizes whitespace (`replace(/\s+/g, ' ').trim()`) before regex replacement.

### 5. Lobby Refresh Button (`index.html`, `lobby.js`) — `63c3937`
- Added `🔄` Refresh button to right of "My Rating" in the Active Rooms panel.
- Uses same `.profile-refresh-btn` style and `profileRefreshSpin` CSS animation as the Profile page refresh button.
- On click: spins icon, fetches all stats (all button types), re-fetches room list if a config is open, stops spinning after ≥600ms.

### 6. Accumulative Auto-Update + Per-Game-Type Polling (`lobby.js`) — `63c3937`
- **Auto-poll every 4s:** Only updates Accumulative `Start [N]` buttons (Accumulative is auto-join, no Show Rooms panel).
- **FCFS/SP buttons:** Update on (1) lobby entry, (2) 🔄 Refresh click, (3) "Show Rooms" click.
- **On lobby entry:** Immediately fetches ALL button counts so the correct numbers are shown from the moment a user lands.

### 7. CRITICAL: `setupLobbyEvents` Missing Closing Braces (`lobby.js`) — `04f79ac`
- **Root cause:** `setupLobbyEvents` was missing two closing `}}` — one for `if (isOnLobby())` and one for the function itself.
- **Effect:** Every function defined after line 518 (`startStatsPolling`, `fetchLobbyStats`, `updateLobbyButtons`, `fetchAndRenderRooms`, `handleLobbyRefresh`, etc.) was trapped inside `setupLobbyEvents`, itself gated on `if (isOnLobby())`.
- **Result:** When any user navigated TO the lobby from another page, none of those functions existed → no auto-update, no room polling, no Refresh button handler.
- **Fix:** Added the two missing `}}` at the correct location after the mobile scroll block.

### 8. FAQ: Browser Zoom Tip (`templates/index.html`) — `397abeb`
- Added "The features seem to be oversized or undersized" FAQ entry with browser zoom keyboard shortcut instructions.

---

## Current Lobby Behavior

| Scenario | Behavior |
|---|---|
| User opens lobby | All button counts immediately correct (full stats fetch on entry) |
| Accumulative player enters/leaves | `Start [N]` auto-updates within 4 seconds |
| FCFS/SP player enters | Count updates when watcher clicks "Show Rooms" or 🔄 Refresh |
| 🔄 Refresh clicked | All button counts updated + room list refreshed if config open |
| "Show Rooms" clicked | Room list with player names/ratings; that button's count updates |
| Player leaves room | Count decreases on next update cycle |

---

## Files Modified in This Session

| File | Purpose |
|---|---|
| `app.py` | `list_rooms()` filter/fields fix; `get_lobby_stats()` `is_daily` fix |
| `game_room.py` | `remove_player()` deadlock fix + `leaving_player` guard; `cleanup_rooms()` `is_closing` flag |
| `static/js/lobby.js` | Full polling rewrite; `updateLobbyButtons` normalization + `allButtons` flag; `handleLobbyRefresh`; CRITICAL `}}` fix |
| `templates/index.html` | Fallback renderer with player pills; FCFS button text single-line fix; Refresh button HTML; FAQ zoom entry |

---

## Dev Notes

- **Deploy to prod:** `cd /home/morpheme/morpheme && git pull origin main && pm2 restart all`
- **SSH:** Key auth only — must be run manually (not available in Antigravity sandbox)
- **Local dev:** `flask run --port 5001` or `python app.py`
- **DB:** SQLite at `DB_PATH` (configured in `app.py`)
- **Scratch files NOT committed to git:** `cookie1.txt`, `cookie2.txt`, `scratch/*.py`
