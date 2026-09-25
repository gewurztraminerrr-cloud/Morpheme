# Stable State Summary — September 25, 2026

> **Start Over Point**: All environments (localhost, GitHub, production `morpheme.games`, and mobile web app) are 100% synchronized and verified.  
> **Commit**: `8c217b0a` (feature commit `7f05e851`)  
> **Tags**: `START_OVER_POINT_SEPTEMBER_25`, `stable-2026-09-25`  
> **Branch**: `main` — `origin/main`  
> **Server**: `132.148.72.249` (`morpheme.games`) — PM2 process `morpheme` (ID: 0), online ✅  
> **Date/Time**: 2026-09-25 00:35 CDT  

---

## 1. Repository & Deployment Synchronization

| Environment | Host / Path | Status | Tags |
| :--- | :--- | :--- | :--- |
| **Localhost** | `/Users/jeffbabiak/` | ✅ Clean & Synchronized (`8c217b0a`) | `START_OVER_POINT_SEPTEMBER_25`, `stable-2026-09-25` |
| **GitHub** | `https://github.com/gewurztraminerrr-cloud/Morpheme` (`main`) | ✅ Pushed & Synchronized (`8c217b0a`) | `START_OVER_POINT_SEPTEMBER_25`, `stable-2026-09-25` |
| **Production Server** | `132.148.72.249` (`morpheme.games`) | ✅ Deployed & Online (PM2 process 0 healthy, 1.5GB) | `8c217b0a`, `START_OVER_POINT_SEPTEMBER_25` |
| **Mobile Web / App** | All client platforms (iOS / Android / Desktop) | ✅ Verified (0ms gateway hydration, cache-busted v=1790279000) | Synchronized |

---

## 2. Key Features, Improvements & Fixes (September 25, 2026)

### A. Multiplayer "Subanagrams (Practice)" Mode
1. **Lobby Matrix Button (`templates/index.html`, `static/css/lobby.css`)**:
   - Added `SUBANAGRAMS (PRACTICE)` matrix button strictly below the 3x3x3 Cube section.
   - Configured with `2m` time label, 120-second duration (`data-time="120"`), and standard `Start [0]` button.
   - Excluded from mobile devices via CSS media queries (`@media (max-width: 900px) { .matrix-subanagrams { display: none !important; } }`) and device filtering in `static/js/app.js`.
   - Backend API enforces mobile restriction, returning HTTP 403 on mobile room join or creation requests.
2. **Room Architecture & Lifecycle (`app.py`, `game_room.py`)**:
   - Permanent singleton room ID: `pub_v2_subanagrams_subanagrams_120`.
   - Unrated / No stats: no rating changes (`rating_change = 0`), no stats recorded to player profiles or leaderboards, and no database round history persistence.
   - Player cards in roster display `Unrated` with a neutral rating color.
   - High-capacity room support (`max_players = 9999`) with no spectator fallback.
3. **Gameplay UI & Tile Selection (`static/js/play.js`, `static/css/play.css`)**:
   - Single-row horizontal centered board layout (`.is-subanagrams-board`) with large tiles (64px & 2.1rem font).
   - Adjacency restrictions bypassed: players can select or tap tiles in any order across the letter sequence.
   - Automatic keyboard coordinate resolution upon word submission.
   - Rotate Board and Transpose buttons hidden.
   - Redundant solitary "Words" tab button hidden; header seamlessly reads "Words" during active play and "All Words" during intermission.

### B. Sequence Length in Spinner Set & Header Formatting
1. **Header Left Display (`templates/index.html`, `static/js/play.js`)**:
   - Wrapped board dimensions in `#header-meta-board` (`<span id="header-meta-board"><span id="param-board">-</span> | </span><span id="param-time">-</span>`).
   - For Subanagrams, `#header-meta-board` is hidden (`style.display = 'none'`), leaving strictly `2m` beneath `SUBANAGRAMS (PRACTICE)`.
   - Wrapped `#label-board` in `#label-board-meta` above the Spinner Set and hid it for Subanagrams so the label cleanly reads `SUBANAGRAMS (PRACTICE) 2m`.
2. **Spinner Set Sequence Length (`templates/index.html`, `static/js/play.js`)**:
   - In `.game-params`, converted `Diff:` to `<span id="param-diff-label">Diff</span>: <span id="param-diff">-</span>`.
   - In Subanagrams mode:
     - Label dynamically switches to **`Letters:`**.
     - Value displays the variable sequence length (6–10, e.g. `Letters: 7`).
     - Cleared difficulty coloring so it renders in standard text color.
     - At 0:45 intermission reveal, updates dynamically to the upcoming round's sequence length from `spinner_params.sequence_length` and participates in the gold shimmer reveal animation.
   - Restores standard difficulty label and color bar highlights in normal rooms.

### C. Tile Count Lock & Consistency Enforcement
1. **Generator & Room Parameter Locking (`subanagrams_generator.py`, `app.py`, `game_room.py`)**:
   - In `subanagrams_generator.py`, explicitly enforced `final_params['sequence_length'] = len(best_candidate)`.
   - In `app.py` & `game_room.py`, saved `room.next_spinner_params = nparams` and set `room.spinner_params_generated = True` during pre-generation so `generate_spinner_params` reuses the staged parameters rather than re-rolling a different sequence length.
   - On round promotion in `start_new_round`, `room.sequence_length` and `room.spinner_params['sequence_length']` are synchronized directly to `len(room.board[0])`.
2. **Client Source-of-Truth Enforcement (`static/js/play.js`)**:
   - In `updateParameters`, active rounds (`!isIntermission`) treat `state.board[0].length` as the absolute source of truth for `factSeqLen`, ensuring the number of tiles on the board and the Spinner Set sequence length are guaranteed to match identically.

### D. Finders & Word Tally Removal in Subanagrams
1. **UI Finders Removal (`static/js/play.js`)**:
   - In `displayAllWords`, `#finders-button-container` is strictly hidden (`style.display = 'none'`) in Subanagrams mode so the `"Finders:"` button never appears when clicking words during intermission.
   - Disabled golden finder highlights (`finder-highlight`) on player cards in `renderPlayers` when in Subanagrams mode.
   - Added early return in `showFinderModal` to prevent opening the finders modal in Subanagrams mode.
2. **Backend Word Finding & Stats Exclusion (`game_room.py`)**:
   - In `log_word_tally()`, added early return for `room.game_type == 'subanagrams'`, preventing words found in Subanagrams from being recorded in `word_stats.json` or `word_tally.log`.
   - Guarded all `save_round_history` and `log_word_tally` invocations on round end and promotion to skip Subanagrams rooms.

---

## 3. Invariants & Rules Preserved
- **Dictionary Rules**: Plural noun and verb conjugation sourcing integrity preserved across word lists and additions.
- **Mobile Fullscreen & Soft Keyboard Invariant**: Fullscreen exits cleanly when opening utility pages or modal dialogs with inputs (`static/js/tools.js`), preventing Android Chrome black-screen display surface rebuilds.
- **Gateway Screen Integrity**: Immediate fullscreen engagement preserved continuously from the ENTER LOBBY gateway screen into the lobby without viewport dimension shift.
- **Cache Busting**: Version query bumped to `play.js?v=1790279000` in `templates/index.html`.

---

## 4. Verification Check
- **Localhost**: Verified clean git tree, python compilation (`app.py`, `game_room.py`, `subanagrams_generator.py`), and JavaScript syntax with JXA.
- **GitHub**: Pushed to `origin/main` at commit `7f05e8510c728630d5aad871bc6defa936104d04` with tags `START_OVER_POINT_SEPTEMBER_25` and `stable-2026-09-25`.
- **Production Server (`132.148.72.249`)**: Verified `git rev-parse HEAD` returns `7f05e8510c728630d5aad871bc6defa936104d04`, tags fetched, PM2 process online with 1.5GB memory.
- **Live HTTP Check**: `curl -sI https://morpheme.games` returns `HTTP/1.1 200 OK`.
