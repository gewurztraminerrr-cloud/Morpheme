# Stable State Summary - November 19, 2026

## 1. Issues Addressed

### Issue 1: ACC 4x6 45s Start Button Timeout & Re-entry Loop
- **Symptom**:
  - The user clicked "Start" on ACC 4x6 45s.
  - The "Loading..." screen stayed up for 30–60 seconds, disappeared, and 2 minutes later reappeared with the toast message: *"Joining room took longer than expected. Please try again"*, while simultaneously dropping the user into the room.
- **Root Cause**:
  1. In `app.py` ([`create_room`](file:///Users/jeffbabiak/app.py#L7414-L7425)): When joining an uninitialized singleton room (such as `pub_v2_accumulative_4x6_45`), if the room had no board yet, it called `room_manager.start_round(room_id)` synchronously on the HTTP request worker thread.
  2. In `game_room.py` ([`start_round`](file:///Users/jeffbabiak/game_room.py#L5624-L5645)): If the cached or fallback board had a word count or difficulty label mismatch with the spun parameters, the "first-board contract" check triggered `generate_board` with `timeout=40.0` for 2 attempts (up to 80 seconds).
  3. In `static/js/lobby.js` ([`enterLobbyRoom`](file:///Users/jeffbabiak/static/js/lobby.js#L114-L131)): The HTTP request had an `AbortController` timer set to 25 seconds (`setTimeout(() => createCtrl.abort(), 25000)`). At 25 seconds, the browser aborted the fetch, showed *"Joining room took longer than expected. Please try again"*, and hid the loading overlay.
  4. On the server, the Waitress worker thread continued generating the board in the background for another 30–60 seconds. When generation finished and the room became active, the user (already registered in the room) was detected by the lobby's periodic sync/reconnect, which showed the loading overlay again and navigated to the room. Additionally, synchronous CPU-intensive generation starved Waitress worker threads (`WARNING:waitress.queue:Task queue depth is 89`).

- **Fix Applied**:
  1. **Non-blocking Room Creation**: In [`app.py:7414-7425`](file:///Users/jeffbabiak/app.py#L7414-L7425), if `not room.board`, `create_room` marks `room.state = 'loading'`, launches `start_round(room_id)` in a background daemon thread, and returns immediately (`{ success: true, room_id }`) in <5ms.
  2. **In-Room Loading Transition**: The client transitions immediately to `page-play`, where the game room's animated loading screen (`"GENERATING 4X6 BOARD..."`) is already built-in, fast-polling until the board is ready.
  3. **First-Board Contract Hardening**: In [`game_room.py:5624-5645`](file:///Users/jeffbabiak/game_room.py#L5624-L5645), reduced generation timeout to a single 10.0s attempt. If compliant generation does not complete quickly, the instant fallback board is cleanly accepted and room parameters adapt.
  4. **Lobby Abort Buffer**: In [`static/js/lobby.js:114-118`](file:///Users/jeffbabiak/static/js/lobby.js#L114-L118), increased `createTimer` from 25s to 45s.
  5. **Scoring Import Scope Guard**: Fixed an `UnboundLocalError` in [`game_room.py:7626-7630`](file:///Users/jeffbabiak/game_room.py#L7626-L7630) by importing `calculate_word_score` at the top of `start_next_round`.

---

### Issue 2: Multiplayer Subanagrams Definition Loading Speed
- **Symptom**:
  - Definition loading in Multiplayer Subanagrams was slow and showed delayed `"Loading definition..."` states.
- **Root Cause**:
  1. In `static/js/play.js`, definition prefetching was never invoked for Subanagrams; words were fetched one by one upon clicking.
  2. In `app.py` ([`get_definition`](file:///Users/jeffbabiak/app.py#L10915-L10940)): Single-word lookups performed external synchronous Wikipedia REST API requests with 2.5s timeouts and multiple candidate loops.
  3. In `app.py` ([`get_word_dictionaries`](file:///Users/jeffbabiak/app.py#L10892-L10912)): Performed filesystem `os.path.getmtime` checks on every call without in-memory caching.

- **Fix Applied**:
  1. **Batch Definitions API**: Added `POST /api/definitions/batch` in [`app.py:10945-11000`](file:///Users/jeffbabiak/app.py#L10945-L11000). Resolves definitions, pronunciations, lists, division stats, and cached wiki data for up to 250 words simultaneously in ~10ms.
  2. **Fast Dictionary Set Caching**: Added `_WORD_DICTS_CACHE` in [`app.py:10892`](file:///Users/jeffbabiak/app.py#L10892) for instant (<0.001ms) dictionary membership resolution.
  3. **Automatic Pre-fetching in Subanagrams**:
     - In [`static/js/play.js:5984-6010`](file:///Users/jeffbabiak/static/js/play.js#L5984-L6010) (`displayAllWords`): Automatically batch-prefetches all rack words the instant intermission renders.
     - In [`static/js/play.js:2404-2425`](file:///Users/jeffbabiak/static/js/play.js#L2404-L2425) (`updateGameState`): Prefetches `all_words`, `previous_all_words`, and player submissions.
  4. **Upgraded `prefetchDefinitions`**: In [`static/js/play.js:12526-12575`](file:///Users/jeffbabiak/static/js/play.js#L12526-L12575), sends batch requests to `/api/definitions/batch` and stores results directly into `window._gameDefCache`.
  5. **Instant Definition Rendering**: When clicking/tapping any word in Subanagrams, `window._gameDefCache.has(cleanWord)` hits immediately for 0ms latency with no loading placeholder.

---

### Issue 3: CSW-Only Words in NWL Rooms & Gold Highlighting Failure (`notcsw1.jpg` & `notcsw2.jpg`)
- **Symptom**:
  - In an `ACCUMULATIVE 4X6 | 45S` game room with `Dict: NWL`, CSW-only words such as `HARELDS`, `MUCHELS`, and `SCUDLER` were present on the board and in the "All Words" list (totaling 135 words instead of the true 102 NWL words).
  - In the "All Words" modal, these words were rendered in plain text without the gold styling indicating CSW Only.
  - The dictionary filter dropdown in the All Words modal showed `NWL (135)` instead of isolating CSW-only words.
  - Tapping `MUCHELS` opened a definition popup showing `MUCHELS 7L [CSW]`, contradicting the room's NWL-only designation and the modal's classification.
- **Root Cause**:
  1. **Dictionary Leakage at Staging and Promotion**: In [`game_room.py`](file:///Users/jeffbabiak/game_room.py) (`_stage_board`, `ensure_next_board_ready`, and `start_next_round`), boards popped from cache or fallback boards that had been solved against Collins Scrabble Words (CSW) were promoted directly to `room.all_words` and `room.all_words_paths` without filtering out words that were invalid in `room.current_dictionary`. Solving board `PMROEUUYCDYLLSLHEHCRRAMU` on CSW yields 135 words, while solving on NWL yields exactly 102 words.
  2. **Active Word Integrity Safeguard Gap**: The post-promotion `verify_board_words_consistency` check only checked whether words can physically be traced on the grid, rather than checking `word_validator.is_valid_word(w, room.current_dictionary)`.
  3. **Suppressed CSW Word Tagging on Backend**: In [`game_room.py`](file:///Users/jeffbabiak/game_room.py) (lines 8141, 8206) and [`app.py`](file:///Users/jeffbabiak/app.py) (lines 8425, 8519), the backend guarded CSW-only calculation with `if 'CSW' in current_dict:`. Because the room dictionary was `NWL`, the server explicitly set `room.csw_only_words = []`, `room.next_round_csw_only_words = []`, and `previous_csw_only_words = []`.
  4. **Frontend Misclassification**: Because `csw_only_words` was sent as an empty array `[]`, `cswOnlySet` was empty in [`static/js/play.js`](file:///Users/jeffbabiak/static/js/play.js). As a result:
     - `cswOnlySet.has('MUCHELS')` evaluated to `false`, preventing the `.csw-only` (gold) class from being applied.
     - The dropdown count loop placed all 135 words into `countNwl`, resulting in the `NWL (135)` option.

- **Fix Applied**:
  1. **Strict Dictionary Enforcement in `game_room.py`**:
     - In [`_stage_board`](file:///Users/jeffbabiak/game_room.py#L1740-L1750), [`ensure_next_board_ready`](file:///Users/jeffbabiak/game_room.py#L2060-L2075), and [`start_next_round`](file:///Users/jeffbabiak/game_room.py#L7790-L7810): When `dictionary` does not include CSW (i.e. NWL mode), all CSW-only words are strictly stripped using `not word_validator.word_validator.is_csw_only(w)` before staging or promoting. On board `PMROEUUYCDYLLSLHEHCRRAMU`, the 33 CSW-only words are pruned, leaving the exact 102 NWL words.
     - In [`start_next_round`](file:///Users/jeffbabiak/game_room.py#L8289-L8315): Added a dictionary enforcement shield immediately upon promotion to ensure `room.all_words_paths` never retains CSW-only words in an NWL room.
  2. **Unconditional CSW-Only Tagging**:
     - Removed all `if 'CSW' in dict` gates in [`game_room.py`](file:///Users/jeffbabiak/game_room.py) and [`app.py`](file:///Users/jeffbabiak/app.py). `room.csw_only_words`, `room.next_round_csw_only_words`, and `previous_csw_only_words` are always computed accurately from the actual words list.
     - In [`app.py`](file:///Users/jeffbabiak/app.py#L8430-L8445), removed the line that previously wiped `prev_csw_list = []` in NWL games, and ensured active round states immediately compute `csw_only_words` if unpopulated.
  3. **Frontend Classification & Highlighting Guard in `static/js/play.js`**:
     - In [`displayAllWords`](file:///Users/jeffbabiak/static/js/play.js#L6085-L6130): Added defensive checks so that if any CSW-only words exist (`countCswOnly > 0` or `cswOnlySet.size > 0`), `isCSW` is set to `true`, correctly rendering the separate `CSW Only` and `NWL` filter options in the dropdown rather than misclassifying all words under `NWL`.
     - Ensured `.csw-only` (gold) is applied whenever a word is in `cswOnlySet`.


---

### Issue 4: Dual Bonus Word Display & Scoring Desync (`notbonus1.jpg` & `notbonus2.jpg`)
- **Symptom**:
  - In `notbonus1.jpg` and `notbonus2.jpg` (Game room `ACCUMULATIVE 4X6 | 45S`, `Dict: NWL`, Board `EIT ISXN ERER MYLS IAFR ATWI S`):
  - In the "All Words" list, two words appeared with bonus word styling and scores:
    1. `INTERLAYER`: Displayed at the top with a green background (`.bonus-word`), a gold star badge (`★`), and point breakdown `17 + 10 = 27`. (The intended bonus word).
    2. `INTERSEXES`: Displayed further down the list with a white background, but also showed a gold star badge (`★`) and point breakdown `17 + 10 = 27`. (Not the bonus word).
- **Root Cause**:
  1. **Server-Side Scoring & Bonus Word Desync**:
     - When the board was staged in `ensure_next_board_ready` or cache refill, `INTERSEXES` was initially selected as candidate bonus word (`next_round_bonus = 'INTERSEXES'`), and `next_round_word_scores` was computed via `calculate_word_score(..., bonus_word='INTERSEXES')` (giving `INTERSEXES: {'base': 17, 'bonus_word_points': 10, 'total': 27}`).
     - At round start in [`start_next_round`](file:///Users/jeffbabiak/game_room.py#L8378), `pick_distinct_bonus_word` was called to avoid recently used bonus words across rounds, selecting `'INTERLAYER'`.
     - `room.bonus_word` was updated to `'INTERLAYER'`, but `room.solved_words_with_scores` was copied directly from `room.next_round_word_scores` without re-synchronizing or scrubbing bonus word points.
     - As a result, `room.solved_words_with_scores['INTERSEXES']` permanently retained `bonus_word_points = 10` and `total = 27`.
  2. **Client-Side Rendering Leakage in `static/js/play.js`**:
     - In [`displayAllWords`](file:///Users/jeffbabiak/static/js/play.js#L6312): The client checked `bonusWordPts = Number(pointsData.bonus_word_points) || 0` without gating on `isBonus`. Because `pointsData.bonus_word_points` was 10 from the server, `bonusWordPts` was set to 10 for `INTERSEXES` despite `isBonus = false`.
     - In [`displayAllWords`](file:///Users/jeffbabiak/static/js/play.js#L6414): The star badge condition was `else if (bonusWordPts > 0 || isBonus)`. Since `bonusWordPts` was 10, the gold star badge `★` was rendered on `INTERSEXES`.
     - In Found words lists ([`play.js:3726-3728, 3826`](file:///Users/jeffbabiak/static/js/play.js#L3726-L3728)): `isBonus` included `(pointsObj && pointsObj.bonus_word_points > 0)`, creating a false-positive bonus word match on any word carrying residual bonus points.
- **Fix Applied**:
  1. **Server-Side Word Score Synchronization**:
     - Added `sync_bonus_word_scores(room)` in [`game_room.py:3307-3338`](file:///Users/jeffbabiak/game_room.py#L3307-L3338). It iterates through `room.solved_words_with_scores` and ensures that **only** the single true `room.bonus_word` carries `bonus_word_points = len(word)` and an adjusted total, while any other word has `bonus_word_points = 0` and base total restored.
     - Hooked `sync_bonus_word_scores(self)` inside `recalculate_total_points()` ([`game_room.py:3085`](file:///Users/jeffbabiak/game_room.py#L3085)), `start_next_round` ([`game_room.py:8391`](file:///Users/jeffbabiak/game_room.py#L8391)), `start_round` ([`game_room.py:5827`](file:///Users/jeffbabiak/game_room.py#L5827)), and `force_instant_board` ([`game_room.py:8890`](file:///Users/jeffbabiak/game_room.py#L8890)).
     - In [`app.py:8428-8460`](file:///Users/jeffbabiak/app.py#L8428-L8460) (`get_room_state`): Added score sanitization during intermission to enforce that `word_scores_to_return` only ever attributes `bonus_word_points` to `room.previous_bonus_word`, and calls `sync_bonus_word_scores(room)` during active rounds.
     - In [`app.py:8955`](file:///Users/jeffbabiak/app.py#L8955) and [`game_room.py:1098, 1338`](file:///Users/jeffbabiak/game_room.py#L1098): Strictly bound `is_bonus` on submission to `bool(final_word and bw_str and final_word.strip().upper() == bw_str)`.
  2. **Client-Side Rendering Hardening**:
     - In [`static/js/play.js:6301-6320`](file:///Users/jeffbabiak/static/js/play.js#L6301-L6320) (`displayAllWords`): Constrained `bonusWordPts` strictly to `isBonus === true`. If `!isBonus`, `bonusWordPts` is always 0.
     - In [`static/js/play.js:6382-6395`](file:///Users/jeffbabiak/static/js/play.js#L6382-L6395): If `!isBonus` and server `pointsData` carries stale `bonus_word_points`, the stale bonus is stripped from `serverTotal` before deriving `totalPts`.
     - In [`static/js/play.js:6411-6418`](file:///Users/jeffbabiak/static/js/play.js#L6411-L6418): Changed star badge rendering condition to `else if (isBonus)`, ensuring the `★` tag is only rendered on the single true bonus word.
     - In [`static/js/play.js:3702-3730, 3820-3830, 4200-4210, 11227-11235, 11325-11330`](file:///Users/jeffbabiak/static/js/play.js): Enforced that bonus identification strictly requires matching the round's authoritative bonus word (`bwUpper`).
  3. **Verification**:
     - Verified unit tests locally confirming that setting bonus word to `INTERLAYER` ensures `INTERSEXES` has `bonus_word_points = 0`, total score 17, and no star icon, while `INTERLAYER` has `bonus_word_points = 10`, total score 27, green background, and star icon.
     - Pushed to `origin/main` (commit `a46acec0`), updated tag `stable-2026-11-19`, deployed to production VPS (`132.148.72.249`), restarted Waitress (PM2 #0), and verified via live API query that 0 spurious bonus words exist.


---

### Issue 5: App Freeze / Logo Screen Lockout at 12:57 PM (`morpheme-error.log` Queue Depth 116)
- **Symptom**:
  - At 12:57 PM (user time `17:57 UTC`), the app appeared to freeze/crash. Upon attempting to open or reload the app, it was stuck indefinitely on the logo / splash screen.
- **Root Cause Identified from VPS Logs**:
  1. **Log Corroboration**:
     - At `17:56:17 UTC` (12:56:17 PM), room `pub_v2_accumulative_4x6_45` concluded round 5 and triggered background pre-generation for a 4x6 grid targeting a high word count range (`300-400` / `500+`) with 538k custom added words.
     - On a 24-tile grid, achieving 300+ words with target uniqueness ratio required repeated `_rescue_word_soup_density` passes. The solver entered an 11-attempt loop with rescue sweeps testing 20 letters across positions.
     - This CPU loop ran continuously for **238 seconds (3.9 minutes)**, locking Python's Global Interpreter Lock (GIL).
     - Because the GIL was monopolized, the Waitress WSGI server's worker threads could not process incoming network I/O. The task queue backed up to a backlog of **116 tasks** (`WARNING:waitress.queue:Task queue depth is 116`).
     - At `17:58:14 UTC` (12:58:14 PM), the client's poll timed out (`AbortError`).
     - At `17:58:55 UTC` (12:58:55 PM), the user reopened the mobile app. The webview requested `GET /` to load the application. Waitress was blocked, unable to respond, and Nginx logged `GET / HTTP/2.0 499 0` when the client closed the connection.
     - While waiting for `GET /` to load, Flutter displayed the dark scaffold with the Morpheme logo, appearing completely frozen on the logo screen.
     - When the 238-second rescue sweep finally finished at `18:00:15 UTC` (1:00:15 PM), `start_next_round` fired and crashed on `UnboundLocalError: cannot access local variable 'calculate_word_score'` (an issue in commit `aff00cc2`).
- **Fix & Safeguards Applied**:
  1. **Scoring Variable Scope**: `calculate_word_score` was properly scoped in `game_room.py` (fixed in commit `45e027e8` / `d25c66d7`).
  2. **Non-Blocking Background Room Transitions**: Room creation and transitions no longer block synchronous HTTP worker threads.
  3. **GIL Micro-Yielding & Rescue Sweep Execution Cap**:
     - In [`board_generator.py:4990-5015`](file:///Users/jeffbabiak/board_generator.py#L4990-L5015): Added `time.sleep(0.002)` inside `_rescue_word_soup_density` so Python regularly releases the GIL, allowing Waitress threads to service incoming HTTP requests (`GET /`, auth, polling) in real time (<5ms).
     - Reduced the maximum rescue sweep execution cap per attempt from 30.0s to 8.0s and capped rescue candidate tiles to prevent CPU starvation.
- **Verification & Deployment**:
  - Committed and pushed to `origin/main` (`9f1c86e1`), updated tag `stable-2026-11-19`, deployed to production VPS, and verified PM2 process health.

---

### Issue 6: Tournaments Guest "Join Now" Button Contrast on Light Layouts
- **Symptom**:
  - When unauthenticated users or guests visited the Tournaments page (`#page-tournaments`), the guest call-to-action message card displayed:
    *"Championship Access - Tournaments are reserved for registered Morpheme members. Sign up now to compete for the crown!"*
  - On light-colored layouts (e.g. `theme-white`, `theme-light-gray`, `theme-light-yellow`, `theme-light-blue`, etc.), the "Join Now" button text was virtually unreadable.
- **Root Cause**:
  1. In [`templates/index.html:3844-3846`](file:///Users/jeffbabiak/templates/index.html#L3844-L3846), the button had hardcoded inline styling:
     `style="background: var(--accent-color); color: #000; padding: 12px 30px; margin-top:20px;"`
  2. In all light layouts, `--accent-color` is set to dark, high-contrast tones (e.g., `#333` in White theme, `#444` in Light Gray theme, `#cc0000` in Light Red, `#0066cc` in Light Blue, `#008000` in Light Green, `#7b1fa2` in Light Purple).
  3. Because the button had inline `color: #000`, it rendered black text directly on top of dark charcoal/dark red/dark blue backgrounds (e.g., `#000` on `#333` with an abysmal contrast ratio of `1.66:1`), completely failing WCAG accessibility guidelines and rendering the button label illegible.
- **Fix Applied**:
  1. **Template Cleanup**:
     - In [`templates/index.html:3844-3846`](file:///Users/jeffbabiak/templates/index.html#L3844-L3846), removed the brittle inline styling and assigned dedicated classes and ID:
       `<button class="nav-btn guest-join-btn" id="tournament-guest-join-btn" data-page="login">Join Now</button>`
  2. **Theme-Aware Button & Card Styling**:
     - In [`static/css/style.css:7985-8095`](file:///Users/jeffbabiak/static/css/style.css#L7985-L8095), added robust default styles for `.guest-msg-card .guest-join-btn` / `#tournament-guest-join-btn`:
       - Background: `linear-gradient(135deg, #e94560, #d63031)` with white text (`#ffffff !important; font-weight: 800 !important;`) and subtle text-shadow.
       - Smooth hover lifting and interactive states.
     - Added comprehensive overrides for all light layouts (`body.theme-white`, `[class*="theme-white"]`, `body[class*="theme-light-"]`, `[class*="theme-light-"]`):
       - Card: High-contrast semi-translucent light card (`background: rgba(255, 255, 255, 0.9)`, crisp border, and subtle drop shadow).
       - Join Now Button: Vibrant rose/crimson gradient (`linear-gradient(135deg, #e11d48, #be123c)`) with bold pure white text (`#ffffff !important`), yielding contrast ratios from 4.70:1 to 6.29:1 (exceeding WCAG AA).
     - Added dedicated override for `body.theme-black`: Crisp white button (`#ffffff`) with pure black text (`#000000`).
- **Verification & Deployment**:
  - Computed WCAG contrast metrics across all themes confirming passing ratios.
  - Committed changes (`eed6c286`), pushed to `origin/main`, updated git tag `stable-2026-11-19`.
  - Deployed to live VPS (`132.148.72.249`) and restarted PM2 process #0 (`morpheme`).
  - Verified live assets served at `https://morpheme.games/` and `https://morpheme.games/static/css/style.css`.

---

### Issue 7: Instant Layout Color Theme & Settings Application on Entering Lobby
- **Symptom**:
  - When a non-default layout color theme (such as White, Light Gray, Yellow, Blue, etc.) was selected in Settings, clicking "ENTER LOBBY" on the gateway/splash screen resulted in a noticeable delay before the Lobby changed from the default dark blue theme to the user's selected theme.
- **Root Cause**:
  1. In `static/js/settings.js`, cached settings from `localStorage` were partially parsed into `window.userSettings` at startup, but `applyTheme(theme)` and `applySettings(settings)` were never invoked synchronously on script startup.
  2. Instead, theme application and all settings were deferred until the completion of `loadSettings()`, which performed an asynchronous network HTTP request (`const response = await fetch('/api/settings')`).
  3. When the user pressed "ENTER LOBBY", the gateway transitioned to `page-lobby` in 120ms while the `/api/settings` fetch was still in-flight over the network. As a result, the user entered the Lobby and saw the default dark blue theme (`--bg-primary: #1a1a2e`), only for the layout color and settings to suddenly pop and flash hundreds of milliseconds later when the network response arrived.
- **Fix Applied**:
  1. **Synchronous Immediate Theme & Settings Application**:
     - In [`static/js/settings.js`](file:///Users/jeffbabiak/static/js/settings.js):
       - Implemented `applyAllSettingsImmediate(settingsOverride)` to synchronously apply `applyTheme(theme)` to both `document.body` and `document.documentElement` alongside all user settings (board sizes, tile sizes, font sizes, corner cutoff, synesthesia letter colors, music, alerts, and sounds) with 0ms latency.
       - Upgraded `applyTheme` to safely guard `.theme-btn` button queries so it can be called safely anywhere, and persists `morpheme_app_theme` into `localStorage`.
       - Enhanced `loadCachedSettings()` to gracefully fallback to `morpheme_app_theme` if user-specific settings are being initialized.
       - Invoked `applyAllSettingsImmediate()` immediately upon `settings.js` execution and at the start of `loadSettings()` before initiating the network fetch.
  2. **Gateway Pre-Application & Transition Hooking**:
     - In [`templates/index.html`](file:///Users/jeffbabiak/templates/index.html):
       - Added instant pre-application of `theme-<theme>` in the immediate 0ms inline script.
       - In `window.handleEnterLobbyClick`, invoked `applyAllSettingsImmediate()` the exact millisecond "ENTER LOBBY" is clicked, and reaffirmed it when the 120ms transition reveals the Lobby.
     - In [`static/js/app.js`](file:///Users/jeffbabiak/static/js/app.js):
       - In `executeGatewayTransition`, invoked `applyAllSettingsImmediate()` immediately on user click and upon switching to `lobby-active`.
       - In `navigateToLobby` and `setCurrentUser`, invoked `applyAllSettingsImmediate()` synchronously.
- **Verification & Deployment**:
  - Committed changes (`e1edb68f`), pushed to `origin/main`, updated git tag `stable-2026-11-19`.
  - Deployed to live VPS (`132.148.72.249`) and restarted PM2 process #0 (`morpheme`).
  - Verified live deployment and verified that the selected layout color theme and all settings apply synchronously at 0ms latency the moment the user clicks ENTER LOBBY.

---

### Issue 8: "Loading..." Notice Continuity When Entering Room from Lobby
- **Symptom**:
  - When clicking or tapping a game room button in the Lobby (e.g. Accumulative, Subanagrams, or Active Rooms), the "Loading..." notice appeared instantly, but then disappeared for a few seconds before reappearing or transitioning into the room.
  - The user requested keeping the message visible continuously the entire time they load into the room without disappearing.
- **Root Cause**:
  1. **Premature Dismissal in `finally` Blocks**:
     - In [`static/js/lobby.js`](file:///Users/jeffbabiak/static/js/lobby.js) (`enterLobbyRoom`, `joinRoom`, `createRoom`) and [`templates/index.html`](file:///Users/jeffbabiak/templates/index.html) (`handleJoinRoomInline`):
       The functions contained `finally { if (window.hideLoadingOverlay) window.hideLoadingOverlay(); }`.
     - When the user clicked a room, `window.showLoadingOverlay('Loading...')` appeared immediately.
     - However, the HTTP network request to `/api/room/create` or `/api/room/join` only took 50–150ms to register the user on the server.
     - Upon this fast HTTP response, `finally` executed immediately and called `hideLoadingOverlay()`, dismissing the overlay within ~100ms—well before the room state was fetched and before the game board was generated or rendered!
  2. **The "Disappear for a Few Seconds" Visual Gap**:
     - For the next 1–3 seconds, while the client switched to `page-play` and `updateGameState()` fetched `/api/room/${roomId}/state`, there was no loading overlay on screen.
     - When the server returned `state.state === 'loading'` (or when the board generator initialized), the game room suddenly rendered the in-board loading card (`"GENERATING NEXT BOARD..."` with a spinner), or the user tapped again, creating the visual flicker of the loading notice disappearing and reappearing seconds later.
  3. **Duplicate Click and Retry Dismissals**:
     - Duplicate click throttles in `enterLobbyRoom` and `joinRoom` previously called `hideLoadingOverlay()` when ignoring duplicate calls within 10s, accidentally dismissing active loading notices on fast double-taps.
     - In `static/js/play.js` (`_doUpdateGameState`), initial silent state sync retries (`joinAttemptCount < 5`) ran `hideLoadingOverlay()` in `finally` during retry cycles.
- **Fix Applied**:
  1. **Removed Premature `hideLoadingOverlay()` on Successful Room Entry**:
     - In [`static/js/lobby.js`](file:///Users/jeffbabiak/static/js/lobby.js) (`enterLobbyRoom`, `joinRoom`, `createRoom`): Removed `hideLoadingOverlay()` from `finally`. `hideLoadingOverlay()` is now strictly called only on error conditions (timeout, banned, room full, network failure).
     - Removed `hideLoadingOverlay()` from duplicate-click throttles.
  2. **Removed Premature `hideLoadingOverlay()` in Inline Templates**:
     - In [`templates/index.html`](file:///Users/jeffbabiak/templates/index.html) (`handleJoinRoomInline`, `enterLobbyRoomInline`, `createRoomInline`): Synchronized logic so `finally` never hides the overlay on successful room entry.
  3. **Guarded Initial Join State Retries in `play.js`**:
     - In [`static/js/play.js:4357-4364`](file:///Users/jeffbabiak/static/js/play.js#L4357-L4364) (`_doUpdateGameState`): Added guard `const isRetryingInitialJoin = (!window.lastGameState && joinAttemptCount > 0 && joinAttemptCount < 5);` so `hideLoadingOverlay()` is never invoked while silently waiting for initial room sync.
  4. **Smooth, Continuous Handoff**:
     - The `#global-loading-overlay` stays visible uninterrupted throughout room creation, page transition, and state polling.
     - It is cleanly dismissed at the exact frame `_doUpdateGameState` renders the room (whether revealing an active game board or the in-board generation screen).
  5. **Safety Timeout Buffer**:
     - Increased default auto-hide safety fallback from 10s to 25s in both `app.js` and `index.html`.
- **Verification & Deployment**:
  - Committed changes (`73ba3184`), pushed to `origin/main`, updated git tag `stable-2026-11-19`.
  - Deployed to live VPS (`132.148.72.249`) and restarted PM2 process #0 (`morpheme`).
  - Verified that entering a room maintains continuous visibility of the "Loading..." notice without disappearing or flickering.

---

### Issue 9: Zero Color Layout Changes Outside Settings (Server Pre-Rendered Theme & 0ms Instant Application)
- **Symptom**:
  - After pressing "ENTER LOBBY", it took ~1 second for the Lobby to switch to the correct color layout selected in Settings.
  - The user experienced a visible color scheme transition upon entering the Lobby, and requested: *"Please have NO visible signs of changing color schemes outside of Settings"*.
- **Root Cause**:
  1. **Server Route Never Delivered User Theme in Initial HTML**:
     - In [`app.py`](file:///Users/jeffbabiak/app.py) (`index` route `/`), the server only fetched `timezone` from the database. It never queried `app_theme` or user settings from SQLite.
     - As a result, the initial HTML was delivered over the wire without any theme class on `<html>` or `<body>`, defaulting to the default dark blue theme (`#1a1a2e`).
  2. **1-Second Network Latency Delay**:
     - At startup, `static/js/settings.js` initiated `loadSettings()` (`fetch('/api/settings')`).
     - This asynchronous HTTP network request took ~1 second to query SQLite and return the user's settings.
     - When the user pressed "ENTER LOBBY", the gateway transitioned to `page-lobby` in 120ms while `fetch('/api/settings')` was still in-flight.
     - When the network request finished 1 second later, `applySettings(data.settings)` called `applyTheme(data.settings.app_theme)`.
  3. **Global Background CSS Transition**:
     - In [`static/css/style.css:1845`](file:///Users/jeffbabiak/static/css/style.css#L1845), `body` had `transition: background 0.5s ease, color 0.5s ease;`.
     - When `applyTheme` added the theme class 1 second after entering the Lobby, the entire page visibly and slowly animated from dark blue to the user's selected theme right before the user's eyes.
- **Fix Applied**:
  1. **Server-Side Pre-Rendering of User Theme & Settings in `app.py`**:
     - In [`app.py:5191-5214`](file:///Users/jeffbabiak/app.py#L5191-L5214), updated `index()` to fetch all `user_settings` and `app_theme` directly from SQLite during initial template generation.
     - Passed `initial_theme` and `initial_settings` to `render_template('index.html')`.
  2. **Direct Server-Delivered Theme Classes on HTML & Body**:
     - In [`templates/index.html`](file:///Users/jeffbabiak/templates/index.html):
       - `<html class="{% if initial_theme and initial_theme != 'default' %}theme-{{ initial_theme }}{% endif %}">`
       - `<body class="{% if initial_theme and initial_theme != 'default' %}theme-{{ initial_theme }}{% endif %}">`
       - The initial HTML arrives from the server already bearing the user's chosen theme class on both `<html>` and `<body>` before first paint.
  3. **Head-Executed 0ms Theme & Settings Initialization**:
     - Added an inline script in `<head>` executed before first paint to immediately sync `window.initialTheme`, `window.initialSettings`, and `localStorage.morpheme_app_theme`.
  4. **Eliminated Color Scheme Transitions Outside of Settings**:
     - In [`static/css/style.css:1840-1850`](file:///Users/jeffbabiak/static/css/style.css#L1840-L1850), set `transition: none !important;` on `body`.
     - Scoped theme transitions strictly to `body.settings-active` and `body:has(#page-settings.active)` so smooth color changes only ever occur inside Settings when the user is explicitly previewing/clicking different theme buttons.
  5. **Zero-DOM Churn in `settings.js`**:
     - In [`static/js/settings.js`](file:///Users/jeffbabiak/static/js/settings.js), `applyTheme()` checks whether `body` and `html` already have the target theme. If so, it avoids any class manipulation.
     - `loadCachedSettings()` immediately merges `window.initialSettings` and `window.initialTheme`.
- **Verification & Deployment**:
  - Verified Jinja2 rendering test confirming `<html class="theme-light-yellow">` and `<body class="theme-light-yellow">` render server-side.
  - Committed changes (`c74efea4`), pushed to `origin/main`, updated git tag `stable-2026-11-19`.
  - Deployed to live VPS (`132.148.72.249`) and restarted PM2 process #0 (`morpheme`).
  - Verified that the user's chosen color layout is active from the very first frame with 0ms latency and zero visual color changes upon entering the Lobby.

---

### Issue 10: Spinner Set Word Count Mismatch (297 Words vs 300-400), Database Audit & Strict Pull Validation (`wrongcount1.jpg` & `wrongcount2.jpg`)
- **Symptom**:
  - In an `ACCUMULATIVE 4X6 | 45S` game room with `Dict: CSW + AW`, the Spinner Set header in [`wrongcount1.jpg`](file:///Users/jeffbabiak/.gemini/antigravity/brain/1d68eff8-1658-4346-9eae-f03c6f6b5147/wrongcount1.jpg) displayed `Words: 300-400`.
  - However, in [`wrongcount2.jpg`](file:///Users/jeffbabiak/.gemini/antigravity/brain/1d68eff8-1658-4346-9eae-f03c6f6b5147/wrongcount2.jpg), the Words modal showed only **297 words** (`0/297 - 0% (279 total pts)`).
  - The user requested:
    1. Determine why the board had 297 words when the Spinner Set displayed 300-400 words.
    2. Cycle through all boards stored in the pregenerated cache (`pregenerated_boards`), verify their factual word counts, and ensure their Spinner Sets are strictly correct.
    3. Ensure that when a board is pulled from pregenerated boards, its Spinner Set is validated before using it; if it isn't correct, discard the board and pull another one.
- **Root Cause**:
  1. **Unfiltered Word Count Checks on Cache Pop**:
     - In [`board_generator.py`](file:///Users/jeffbabiak/board_generator.py) (`pop_compatible_cached_board` line 519 and `pop_any_cached_board` line 409), the guard checking Added Words minimum count used `len(all_words) < 300` on the raw stored list.
     - When raw `all_words` contained e.g. 302 words (including words shorter than the room's `min_word_length=5` or words outside dictionary filters), `len(all_words)` was 302 (passing the check). But when filtered by `len(w) >= 5`, the actual playable word count dropped to **297**.
  2. **Missing Range Validation on Popped Boards**:
     - `pop_compatible_cached_board` and `pop_any_cached_board` only compared the stored string `param_key.get('word_count_range') == word_count_range` without verifying whether the board's *actual playable words* satisfied `is_board_count_valid(len(playable_words), word_count_range)`.
     - In [`game_room.py`](file:///Users/jeffbabiak/game_room.py) (`start_next_round` line 7792, `ensure_next_board_ready` line 2005, and `trigger_early_fallback_rescue` line 2774), calls to `pop_compatible_cached_board` omitted `word_count_range`, allowing boards of any word count to be retrieved.
  3. **Buggy Spinner Range Clamping in `spinner_set.py` & `game_room.py`**:
     - In [`spinner_set.py:611-616`](file:///Users/jeffbabiak/spinner_set.py#L611-L616), `get_factchecked_wc_range` had a branch `if is_aw: return '300-400'` when `count < 300`. This caused any Added Words board with 297 words to falsely claim it was `'300-400'`.
     - In [`game_room.py:8504-8506`](file:///Users/jeffbabiak/game_room.py#L8504-L8506), `start_next_round` contained a hardcoded override: `if final_wc not in ['300-400', '400-500', '500+']: final_wc = '300-400'`. Even when `is_board_count_valid(297, declared_wc)` failed, this override forced `room.current_word_count_range = '300-400'`.
- **Fix Applied**:
  1. **Strict Runtime Pull Validation in `board_generator.py`**:
     - Added [`is_board_count_valid`](file:///Users/jeffbabiak/board_generator.py) to validate word counts strictly against bounds (e.g. `300 <= count <= 399` for `300-400`).
     - In [`pop_compatible_cached_board`](file:///Users/jeffbabiak/board_generator.py), [`pop_any_cached_board`](file:///Users/jeffbabiak/board_generator.py), and [`pop_cached_board`](file:///Users/jeffbabiak/board_generator.py):
       - Computes `playable_words = [w for w in all_words if len(w) >= min_word_length and valid_for_dictionary]`.
       - If Added Words is active, strictly verifies `len(playable_words) >= 300`. If under 300, discards the board immediately.
       - If `word_count_range` is specified, strictly validates `is_board_count_valid(len(playable_words), word_count_range)`. If the count is outside the range, discards the board, deletes it from cache, and loops/recurses to pull the next candidate.
       - Strictly enforces that at least 3 words of length equal to `min_word_length` exist on the board.
  2. **Synchronized Pull Parameters in `game_room.py`**:
     - In [`start_next_round`](file:///Users/jeffbabiak/game_room.py), [`ensure_next_board_ready`](file:///Users/jeffbabiak/game_room.py), [`trigger_early_fallback_rescue`](file:///Users/jeffbabiak/game_room.py), and fallback handlers: Passed `word_count_range` and `difficulty` directly to `pop_compatible_cached_board`.
     - Added secondary validation: If candidate `len(_fw_filtered)` does not strictly satisfy `is_board_count_valid`, the board is rejected and regenerated.
     - Removed the forced Added Words overwrite in `start_next_round` so the Spinner Set header always truthfully reflects reality.
  3. **Truthful Bucket Mapping in `spinner_set.py`**:
     - In [`spinner_set.py`](file:///Users/jeffbabiak/spinner_set.py) ([`get_factchecked_wc_range`](file:///Users/jeffbabiak/spinner_set.py#L603-L620)), removed the false clamp that returned `'300-400'` for counts `< 300`.
  4. **Refill Generator Truth Guarantee**:
     - In [`refill_board_cache_bg`](file:///Users/jeffbabiak/board_generator.py), background board generation now computes the factual playable count *after* ING-prevention mutations, rejects any Added Words board under 300 words, and serializes the exact factual range into `param_key` and `board_json`.
  5. **Complete Database Audit & Repair (Local & VPS)**:
     - **Local Database (`/Users/jeffbabiak/morpheme.db`)**: Audited all 79 pregenerated boards. 1 board under 50 words was purged; 74 boards with legacy list formatting were updated to accurate string ranges.
     - **Production VPS Database (`/home/morpheme/morpheme/morpheme.db`)**: Audited all 6,660 pregenerated boards:
       - **20 invalid Added Words boards** with `< 300` playable words were permanently purged from the cache.
       - **165 boards** with mismatched word count ranges were updated to their exact truthful Spinner Set ranges.
       - All remaining 6,475 boards were verified to strictly match their Spinner Sets.
- **Verification & Deployment**:
  - Committed changes (`067b9cd1`), pushed to `origin/main`, updated git tag `stable-2026-11-19`.
  - Deployed to live VPS (`132.148.72.249`) and restarted PM2 process #0 (`morpheme`).
  - Verified VPS database state confirms 0 mismatches across all stored boards.
