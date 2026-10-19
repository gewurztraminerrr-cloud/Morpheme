# Stable State Summary — October 18, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_18
```
*(Also tagged as `stable-2026-10-18`)*

## Latest Commit ID
```
4e48de6b2007e28b6b7d57e89d0bb90365fbdba8
```

## Synchronization Status

| Environment | Status | Details |
|:---|:---|:---|
| **Localhost** | ✅ Synchronized | Clean working tree on `main` |
| **GitHub (`origin/main`)** | ✅ Synchronized | Pushed to `origin/main` & tags `START_OVER_POINT_OCTOBER_18` / `stable-2026-10-18` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | Updated to latest commit, tags fetched, PM2 process online, live HTTP 200 OK |
| **App / Client** | ✅ Synchronized | Cache busters bumped (`play.js?v=1792375000`) for instant live propagation across all devices |

---

## Session Highlights & Architectural Updates (October 18, 2026)

### 1. In-Game Room Chat Speed & Instant Delivery Optimization (`app.py`, `play.js`)
- **Resolved 10–20s Chat Lag & Dropped Messages**:
  - Identified that `submit_chat_message` was performing a synchronous external HTTP call to a nonexistent Gemini API model (`gemini-3.5-flash-lite`) with a 10-second timeout on every single chat message.
  - Replaced the synchronous external call with Morpheme's zero-latency in-memory regex engine (`check_local_moderation`), running in `<0.1ms`.
  - Dispatched AI moderation checks for clean text messages to an asynchronous background daemon thread (`_bg_moderate_room_chat`), completely decoupling content verification from the client's HTTP response.
  - Corrected the Gemini API model endpoint to `gemini-1.5-flash` with a strict 3-second timeout.
- **In-Memory Rating Cache**: Eliminated blocking SQLite queries (`sqlite3.connect(DB_PATH, timeout=30)`) on every chat message by resolving ratings directly from in-memory player/spectator objects and `_USER_RATING_CACHE`.
- **Immediate Authoritative Response**: Route now returns `jsonify({'success': True, 'chat_messages': room.chat_messages})` directly, eliminating reliance on secondary polling.
- **0ms Optimistic UI Rendering**: Newly typed messages appear in `#chat-history` in 1–3ms with the user's username, rating badge, and timestamp. Removed the client-side `isSendingChat` lock that dropped messages during high-latency requests, and synchronously cleared the input and character counter.

### 2. Desktop Game Room Layout & Visual Balancing (`play.css`, `style.css`)
- **Equalized Left and Right Spacing in Players Panel**: Adjusted padding and inner container margins (`padding: 0 10px; width: 100%; box-sizing: border-box;`) so player information rectangles have perfectly symmetrical horizontal margins on desktops and laptops.
- **Restored Light Grey Background for Players Panel**: Re-applied the light grey background (`rgba(0, 0, 0, 0.05)`) around the Players panel beneath the color chart on White themes for desktop/laptop screens (`body.theme-white.is-desktop .players-panel`).
- **Clean White Header Separator**: Maintained a pure white background in the area beneath the color chart and above the "Players" title on White layouts.
- **Words to Work On Table Sizing**: Adjusted the vertical length and padding of the Words to Work On table for balanced desktop display.

### 3. Intermission Winner Announcement & Definitions Panel Enhancements (`play.js`, `play.css`)
- **Structured Winner Announcement**: Restored winner announcements at the beginning of intermission, separated into two clear sections: congratulated user on top, followed by a horizontal divider and `"Select a word to see its definition"` below.
- **Golden Flashing Effect Restored**: Restored high-visibility golden flashing animation (`@keyframes gold-status-flash`) for winner announcements.
- **Definitions Panel Padding & Layout**: Enhanced typography, padding, and layout within the intermission definitions panel for comfortable reading.
- **Cross-Round Definition Persistence**: Retained the selected word's definition across round transitions if clicked during intermission, displayed placeholder at the start of intermission, and aligned mobile word titles to the left with red accents.

### 4. Countdown Expiry & WAIT Transition Optimizations (`play.js`, `app.py`)
- **Display WAIT... on Expiry**: When the round countdown expires, the timer display now immediately shifts to `WAIT...` rather than freezing at `0:00` while awaiting server transitions.
- **Asynchronous Milestone Staging**: Staged intermission board generation milestones with a 0.8s–1.2s lead-time, preventing synchronous 60-second emergency stalls.
- **High-Frequency Transition Polling**: Tightened polling intervals to 80ms during active state transitions with chained fetches to prevent queue flooding.

### 5. Mobile Wake-Up & State Recovery Optimization (`play.js`, `app.py`)
- **Resolved Stalled Timer on Mobile Resume**: Fixed handling of the server `'waiting'` state during wake-up. Empty public rooms pausing into `'waiting'` now display a normalized grayed board and dynamic placeholder `"Waiting for next round..."`.
- **Smooth Animated Waiting Indicator**: Replaced static `'...'` with animated bouncing dots (`WAIT...`) and dynamic labels (`Next Round:` during intermission and waiting).
- **Stale Fetch Promise Deadlock Resolution**: Reset `_activeUpdateGameStatePromise` on wake-up, eliminating 90–120s socket timeout lockups when reconnecting over cellular networks.
- **Prevented False Midnight Resets & Room Entry Self-Cancellations**: Ensured `timerFormatIs24h` is cleared on room exit, preventing premature 24h reset evictions in standard rooms. Guarded `leaveCurrentRoom()` during lobby room entry to avoid self-canceling room navigation.

---

## Complete Commit History (October 18, 2026)

| Commit | Summary |
|:---|:---|
| `4a2d52d3` | **Optimize game room chat for instant sending, zero lag, and optimistic delivery** |
| `f4e926a1` | **style: equalize horizontal spacing for player item rectangles in Players panel on desktop** |
| `f41b15f4` | **style: restore light grey panel background for Players panel below color chart on desktops and laptops** |
| `029ff088` | **style: make area beneath color chart and above Players title white on white layouts on desktop** |
| `eaaacbda` | **feat: enhance definitions panel padding and styling during intermission and restore golden flashing effect** |
| `c7d58021` | **feat: show winner announcement during intermission with user on top and definition placeholder below divider** |
| `86d857b7` | **fix: keep definition across rounds if pressed during intermission, show placeholder at start of intermission, and align mobile word title left with red color** |
| `d61db27b` | **Adjust Words to Work on table vertical length for equal padding on desktop and laptops** |
| `62b39136` | **fix: display WAIT... message instead of 0:00 when countdown hits zero or expires** |
| `55a7ee81` | **fix: eliminate 60s emergency board gen stall and fix wake-up intermission timer** |
| `071dc9eb` | **fix: eliminate background socket contention and wake-up WAIT... flash** |
| `634b3cd2` | **Eliminate background wake-up WAIT delay: 400ms grace window and fast socket-thaw detection** |
| `eb55a167` | **Eliminate WAIT transition delay: 1.2s lead-time milestone, grace delay, and rapid polling** |
| `06c15d64` | **Fix transition queue flooding: proactive pre-fetch, chained polling, and 50s watchdog timing** |
| `5c7c4cba` | **Optimize WAIT duration: synchronous round start, 80ms transition polling, and 0.8s milestone lead-time** |
| `0d61b13f` | **docs: add stable state summary for October 18, 2026 (Start Over Point)** |
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
- **Verification**: Verified live on production host `132.148.72.249` with HTTP 200 OK and headless Chrome UI testing.
