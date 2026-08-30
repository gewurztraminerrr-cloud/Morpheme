# Stable State Summary — August 30, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of August 30, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is fully synchronized.

---

## 1. Repository & Deployment Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 30, 2026
* **Latest Commit ID**: `898f0b5d5bfa780d6b9d6a364177eb0594a9ea4c` (`898f0b5`)
* **Production Host**: `132.148.72.249` (`morpheme.games`)
* **Synchronization Status**: **100% Synchronized** across Localhost, GitHub, and Production (`morpheme.games`).

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. 12:00 AM Daily Room Reset Notice & Single-Notice Guarantee (`static/js/play.js`)
- **Clear Explanatory Notice at Midnight Rollover**:
  - When a 24-hour daily room reaches 12:00 AM (midnight) and concludes, active users are returned to the Lobby and presented with a clear, priority modal notice:
    - **Title**: `Daily Room Reset (12:00 AM)`
    - **Message**:
      > *The 24-hour Daily Room has concluded and reset at 12:00 AM for the new day!*  
      > *You have been returned to the Lobby while the previous day's results are finalized.*  
      > *Entering the room again from the Lobby will show you the brand-new daily round and board!*
- **Strict Single-Notice Guarantee & Eviction Mutex**:
  - Added an eviction lock (`window._isEjectingToLobby`) and immediate polling cancellation in `ejectToLobby()` to eliminate duplicate modal popups caused by in-flight poll responses or concurrent timer triggers.
  - Added a 2-minute notice deduplication window (`window._lastDailyResetNoticeTime`) ensuring the user receives **exactly ONE notice** during the midnight rollover.
  - Handled the midnight player roster reset so clients identify it specifically as a daily round rollover rather than misclassifying it as idle inactivity.

---

### B. Tournament Concurrency & Database Locking Resolution (`tournament_logic.py`, `app.py`)
- **Elimination of Concurrent Board Generation Race Condition**:
  - Identified and resolved the root cause of database lockups and `504 Gateway Time-out` errors: when tournaments reached start time or round advancement, concurrent client polls to `/api/tournament/status` simultaneously ran `start_tournament()` / `advance_tournament()`, triggering duplicate board generation and colliding on `tournament_rounds` unique constraints (`sqlite3.IntegrityError`).
  - Added `threading.Lock()` with non-blocking acquisition (`if not self._lock.acquire(blocking=False): return`) to `TournamentManager.update_tournament_status()` so that only one worker thread manages tournament transitions while all other requests return immediately without stalling.
  - Added atomic state validation (`status == 'signup'` -> `status = 'active'`, `current_round == round_num`) and changed round insertions to `INSERT OR REPLACE INTO tournament_rounds`.
- **Database Context Manager Adoption**:
  - Converted `/api/tournament/status`, `/api/logout`, and `/api/user/account-info` to use the centralized `get_db()` context manager with WAL journal mode, 60s busy timeout, NORMAL synchronous mode, and guaranteed connection cleanup.

---

### C. Comprehensive "View All Pairings" Tournament Modal (`static/js/tournaments.js`, `tournament_logic.py`, `app.py`)
- **All-Round Grouped Matchup Display**:
  - Added `get_all_tournament_matchups(tid)` in `tournament_logic.py` and exposed `all_tournament_matchups` in `/api/tournament/status`, providing the full bracket history across all rounds (Round 1, Round 2, Finals).
  - Redesigned `showAllPairingsModal` to display pairings neatly grouped by round with matchup counters, current round badges, winner trophies (🏆), final scores, and BYE indicators.
  - Highlighted the viewing user's own matchup at the top of each round with distinct purple styling.
  - Properly wired the modal OK button, Close (X) button, and outside backdrop click handlers for smooth opening and dismissal.

---

### D. Tournament "PLAY YOUR TURN" Lobby Redirect Guard Fix (`static/js/lobby.js`)
- **Resolved Play Page Blocking for Active Tournament Turns**:
  - Fixed an issue where the lobby redirect guard blocked navigation to `page-play` during active tournament matches when `tournament_play_active` was set in local storage.

---

### E. Achievements Modal DOM Structure & Click Handler Fix (`templates/index.html`, `static/js/achievements.js`)
- **Hoisted `room-achievements-modal` to Top-Level Body**:
  - Resolved an issue where `room-achievements-modal` was nested inside `mini-profile-modal` (whose `display: none` parent container prevented achievements from rendering).
  - Moved the achievements modal to top-level `<body>` with high z-index, robust click handler with username fallback, modal-first display sequencing, and null-safe guards.

---

### F. Word Lists Expansion to 10,000 Words (`app.py`, `static/js/tools.js`)
- **Raised Default & Server-Side Display Caps**:
  - Increased the Lists tool default display cap and server-side limit from 1,000 to 10,000 words.
  - Cleaned up list column header formatting by removing redundant `"Type:"` labeling.

---

## 3. Commit History (August 27 – August 30, 2026)

| Commit ID | Message |
| :--- | :--- |
| `cb2c3d1` | **Ensure single notice on 12AM daily room reset with clear explanation and lobby return** |
| `b7e58e6` | **Fix tournament pairings modal display and prevent concurrent database lockups** |
| `8573be1` | **fix(tournaments): allow navigation to page-play when tournament_play_active is set — lobby redirect guard was blocking PLAY YOUR TURN** |
| `1f01128` | **fix(achievements): move room-achievements-modal to top-level body — was nested inside mini-profile-modal** |
| `e83d88d` | **fix(achievements): robust click handler with username fallback, modal shows first, null guards, window export** |
| `1f74aff` | **Lists: raise server-side cap from 1000 to 10000 words** |
| `540f41e` | **Lists: default display cap 5000->10000 words; remove Type: label** |
| `92be0af` | **Fix thumb twitching: guard scheduleUpdate against running while steady loader controls thumb** |
| `9eb56e0` | **Fix View Full List: physically append 4000 words/sec to DOM, counter reflects actual rendered count, thumb shrinks and rises as words load** |
| `f9b3a54` | **feat(tools): expand scroll batch size and remove artificial cap so scrolling immediately reaches current loaded pool** |
| `9ccf712` | **docs: record loaded pool expansion in August 27 summary** |

---

## 4. Verification & Health Check

- **Localhost**: All unit tests pass, Python files compile without warnings, and git status is clean.
- **GitHub**: Branch `main` is up to date with origin.
- **Production Server (`132.148.72.249` / `morpheme.games`)**:
  - PM2 process `morpheme` is online with 0 errors.
  - Dictionaries (NWL 199,429 words, CSW 281,598 words, Custom Added Words 469,764 words) loaded cleanly.
  - `/api/tournament/status` responds with `HTTP 200` and all round matchups.
  - Daily room rollover and notice mechanisms are active.
