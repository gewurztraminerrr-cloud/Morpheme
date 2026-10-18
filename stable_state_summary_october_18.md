# Stable State Summary — October 18, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_18
```
*(Also tagged as `stable-2026-10-18`)*

## Latest Commit ID
```
4201804ecdb7ed3ea39bbdc613b85d3e29f98b28
```

## Synchronization Status

| Environment | Status | Details |
|:---|:---|:---|
| **Localhost** | ✅ Synchronized | Clean working tree on `main` at `4201804e` |
| **GitHub (`origin/main`)** | ✅ Synchronized | Pushed to `origin/main` & tags `START_OVER_POINT_OCTOBER_18` / `stable-2026-10-18` updated |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | Updated to `4201804e`, tags fetched, PM2 process online, live HTTP 200 OK |
| **App / Client** | ✅ Synchronized | Cache busters bumped (`v=1792345100`) for instant live propagation across all devices |

---

## Session Highlights & Architectural Updates (October 18, 2026)

### 1. Mobile Room Wake-Up & State Recovery Optimization (`play.js`, `app.py`)
- **Resolved Stalled Timer on Mobile Resume**: Fixed handling of the server `'waiting'` state during wake-up. Previously, when a mobile phone was locked/minimized for ~4 minutes, empty public rooms paused into `'waiting'`. Polling ignored `'waiting'`, leaving the timer stuck on whatever was set on wake-up. Added full `'waiting'` state support with normalized grayed board and dynamic placeholder `"Waiting for next round..."`.
- **Smooth Animated Waiting Indicator**: Replaced static `'...'` with animated bouncing dots (`WAIT...`) and dynamic labels (`Next Round:` during intermission and waiting).
- **Stale Fetch Promise Deadlock Resolution**: Fixed file-scope lexical variable `_activeUpdateGameStatePromise` reset on wake-up, eliminating 90–120s socket timeout lockups when reconnecting over cellular networks.
- **Intermission Overrun Handling**: When board generation exceeds the countdown, the client now smoothly displays animated `WAIT...` dots with the `Next Round:` label until the server's next round board generation completes.

### 2. Sub-15ms State Latency Optimizations (`app.py`, `play.js`)
- **Background Daemon Milestones**: Dispatched synchronous intermission milestones (`start_board_search` and `generate_spinner_params`) to background daemon threads in [app.py](file:///Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/app.py). State polling response times dropped from **300–750ms** down to **5–15ms**.
- **Mobile Cellular Socket Recovery**: Reduced initial fetch timeout to 2.5s and socket thaw retry delay to 50ms for instant recovery from dead mobile OS sockets.

### 3. Intermission Review Preservation & Fast-Forward Protections (`game_room.py`)
- **Preserved Full 60s Review Duration**: Eliminated premature fast-forwarding when a room has 0 active players so returning players always experience the full intermission review period.
- **Fixed `UnboundLocalError`**: Initialized snapshot ratings and player states cleanly at the top of state handlers.
- **Prevented Erroneous Player Evictions**: Prevented mobile `pagehide` and `beforeunload` events from mistakenly evicting active players during routine browser app switching.

### 4. False 12:00 AM Midnight Reset Eviction Fix (`play.js`)
- **Root Cause**: Visiting a 24-hour room set `timerFormatIs24h = true`, which was never cleared on room exit. When the 45s countdown in standard rooms subsequently reached 0:00, the client misclassified the room as a 24h room and triggered `ejectToLobby("daily_reset")`.
- **Solution**: Removed `|| timerFormatIs24h` from local timer reset evaluation. Standardized authoritative 24h room detection (`time_limit >= 7200`, `room_id` containing `24h`/`86400`, or `game_type === '24h'`). Ensured `timerFormatIs24h` and `_wasEverInRoster` are cleared on all room teardowns (`stopPolling()`, `ejectToLobby()`, `exitPlay()`).

### 5. Room Entry Self-Cancellation Fix (`lobby.js`, `play.js`, `templates/index.html`)
- **Root Cause**: When pressing "Start" in the lobby, `enterLobbyRoom` called `leaveCurrentRoom()` to clean up prior rooms. `leaveCurrentRoom()` unconditionally set `window._userLeftRoom = true;` and `window._isEnteringRoom = false;`. When `/api/room/create` finished ~50ms later, the client saw `_userLeftRoom === true`, aborted navigation to `page-play`, and hid the loading overlay, leaving the user stranded in the lobby.
- **Solution**:
  - `leaveCurrentRoom()` now guards flag mutation: `if (!window._isEnteringRoom && !window._isCreatingRoom)`.
  - Awaited `window.leaveCurrentRoom()` during pre-entry cleanup and explicitly re-asserted `window._isEnteringRoom = true;` and `window._userLeftRoom = false;` before create/join requests.
  - Bumped asset cache busters to `v=1792345100` in `templates/index.html`.

---

## Commit History (October 18, 2026)

| Commit | Summary |
|:---|:---|
| `4201804e` | **Fix race condition in room entry self-cancellation by preserving entering state in leaveCurrentRoom** |
| `e68273c3` | **Fix false 12AM midnight eviction in non-24h rooms by strictly synchronizing timerFormatIs24h and clearing on exit** |
| `54a745c4` | **Restore full 60s intermission review: eliminate premature intermission fast-forwarding** |
| `f4d60e78` | **Eliminate stuck WAIT notice on mobile return: fix empty intermission fast-forward, is_closing unblock, and cache floor length** |
| `bdf69852` | **Lock active board immutability and fix intermission staging and fast-forwarding** |
| `b81f0b44` | **Eliminate WAIT notice hangs: fix UnboundLocalError in game_room.py, prevent pagehide/beforeunload player eviction, and trust authoritative server time_remaining** |
| `29009ccb` | **Eliminate WAIT notice stall: auto-rejoin polling users to roster and instant wake empty intermissions** |
| `f412c673` | **Ultra-fast mobile wake-up and async server milestone processing** |
| `fa429b3e` | **Fix mobile wake-up state and timer display during intermission and waiting states** |
| `8985e007` | **fix(ui): clearly display Next Round label and placeholder during intermission on wake-up** |
| `0f94874c` | **fix(mobile): reset _lastLocalTimerTickTime on resume to eliminate initial tick thaw stutter** |
| `f0933b1a` | **fix(mobile): resolve 100ms interval abort loop and deduplicate game state fetch on wake-up** |
| `6e24890c` | **fix(mobile): unify wake-up listeners, prevent abort collision storm, and clear stale timer on resume** |
| `ca52ab7a` | **fix(mobile): eliminate premature WAIT... on mobile wake-up and sync timer early** |
| `f28439f8` | **Initialize user_rating_for_snapshot at top of get_room_state to prevent UnboundLocalError** |

---

## Server & Infrastructure
- **Production Host**: Ubuntu 24.04 LTS at `132.148.72.249`.
- **Application Stack**: Python 3.9 + Waitress WSGI managed by PM2 (`morpheme`).
- **Database**: SQLite (`morpheme.db`) with WAL mode, automated migrations, IP ban cache, and timeout tables.
- **Verification**: Verified live on production host `132.148.72.249` with HTTP 200 OK.
