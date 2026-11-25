# Morpheme Stable State Summary – November 25, 2026

**Synchronization Checkpoint & Stable Anchor**  
- **Git Branch**: `main` (Synchronized: Localhost == GitHub `origin/main` == VPS `132.148.72.249`)
- **Production Server**: `132.148.72.249` ([morpheme.games](https://morpheme.games)) – Live & Healthy (HTTP 200 OK)
- **Localhost Development**: Port `5001` – Running & Healthy (HTTP 200 OK)
- **Date/Time**: November 25, 2026 (00:45 EST / 05:45 UTC)
- **Anchor Commit ID**: `96fe6e48` ("fix: prevent lock starvation and timeout when entering 6x8 accumulative rooms")

---

## 1. Key Achievements & Consolidated Features

### A. Mobile Navigation Black Screen Flash & Viewport Twitching Elimination
- **User Theory & Root Cause Analysis**:
  - **The AW (Added Words) Connection**: When user focused `added-word-input` in Mods -> Added Words, the mobile OS opened the virtual keyboard, which caused Android/mobile browsers to drop HTML5 fullscreen. In addition, `mods.js` repeatedly called `wordInput.focus()`.
  - **The Recurring Fullscreen Thrash**: In `static/js/app.js`, a document-wide listener on `pointerdown` and `touchstart` checked `if (!document.fullscreenElement && !document.webkitFullscreenElement) attemptFullscreen(true)`. Once fullscreen had dropped, **every single touch or tap** (e.g. tapping to navigate to Lobby, Tools, Settings, etc.) fired `attemptFullscreen(true)` twice per tap!
  - **The Black Screen Twitch**: On Android Chrome, calling `requestFullscreen()` causes the browser compositor to animate and flash a black screen while attempting to switch into immersive mode. If rejected or lost, every subsequent touch flashed black again.
  - **The Hidden Tools DOM Overload**: `mods.js` called `window.loadAddedWords('added')` (aliased to `tools.js:fetchListsData`), which fetched up to 500,000 words in the background and executed `startProgressiveRendering()` on the hidden Tools tab DOM on mobile devices.
  - **The CSS Opacity Dip**: `.page` in `static/css/style.css` had `animation: fadeIn 0.5s ease;` which forced incoming pages to start at `opacity: 0; transform: translateY(10px);`, exposing the dark body background beneath it for 500ms on every page switch.
- **Remediation**:
  1. **Fullscreen Manager Tamed (`app.js`)**: Completely removed the document-wide `['pointerdown', 'touchstart']` and `window.focus` listeners that repeatedly forced `attemptFullscreen(true)`. Mobile fullscreen is now only initiated upon explicit user gesture (e.g. tapping "ENTER LOBBY" at the gateway).
  2. **Seamless Page Activation (`app.js`)**: Updated `showPage(pageId)` to a two-pass mechanism: the incoming page is activated and made visible (`display: flex/block`, `opacity: 1`) *before* hiding old pages, guaranteeing zero empty-frame gap where body is exposed. Ensured `#page-loading` is explicitly hidden on navigation.
  3. **Zero-Lag Page Styles (`style.css`)**: Removed `animation: fadeIn 0.5s ease;` on `.page` and set `animation: none !important;` on `.page` and `.page.active`, eliminating the 500ms opacity-0 dark blink.
  4. **Thresholded `--vh` Recalculations (`app.js`)**: Debounced `--vh` style updates to only re-set when height shifts exceed 0.5px, preventing DOM style thrashing on minor address-bar shifts.
  5. **Mods Tab Cleared of Hidden List Downloads (`mods.js`)**: In `addAddedWord`, `removeAddedWord`, and prompt methods, removed unconditional calls to `window.loadAddedWords('added')`. Added Words cache invalidates cleanly without background HTTP queries or DOM node generation unless the Lists tab is actively open. Stopped forceful autofocus on mobile.
- **Commit History**: `4db2982b`.

---

### B. Mobile App Crash & Opening Logo Hang Resolution (Nov 23 Incident)
- **Incident Analysis**:
  - At 6:12 PM – 6:25 PM EST yesterday (Nov 23, 23:12 – 23:25 UTC), requests to `POST /api/definitions/batch` from mobile clients flooded the backend server with up to 250 words per batch.
  - Server-side `lookup_raw_definition_online()` executed synchronous external HTTPS calls to Wiktionary (`en.wiktionary.org`) inside the request threads, exhausting all 64 Waitress worker threads.
  - Nginx timed out with `504 Gateway Timeout`. Simple initial requests (`GET /`, `/api/session`, `/api/lobby-stats`) queued behind the deadlocked threads, leaving users stuck indefinitely on the opening splash/logo screen.
  - At 6:24:47 PM EST, process memory hit 3.65 GB RSS and the Linux kernel Out-Of-Memory (OOM) killer terminated `python3` (PID 137728).
- **Backend Remediation (`app.py`)**:
  - Enforced `allow_online=False` on batch definitions: batch calls strictly query local database (`wiktionary_definitions`) and memory cache. Never make outbound HTTPS calls to Wiktionary inside batch loops.
  - Added a 3.0-second time-budget ceiling per batch and clamped batch size to max 100 words.
  - Optimized `get_word_dictionaries()` to query existing `word_validator` in-memory sets in <0.01ms rather than synchronously re-loading text files from disk.
  - Added LRU/size caps to `_WIKI_DB_CACHE`, `_IS_NOUN_WIKI_CACHE`, `_WIKI_SUMMARY_CACHE`, and `NEGATIVE_DEFINITIONS_CACHE`.
  - Pre-warmed WordNet during background startup so common noun calculation runs in <1ms without blocking requests.
- **Frontend Remediation (`play.js` & `app.js` & `index.html`)**:
  - Implemented `window._inFlightPrefetchWords` set and 250ms debouncing in `play.js`: never emits duplicate requests for words already in flight.
  - Completely removed the cascading fallback loop that was firing 25 individual `/api/definition` calls upon failure.
  - Added a 5-second failsafe watchdog in `templates/index.html` to guarantee the opening/logo screen can never freeze or trap mobile users indefinitely.
  - Reduced `checkSession()` timeout to 3.5s with clean exception handling and UI fallback.
- **Commit History**: `4d4bcc54`.

---

### B. Intermission Words Filter Dropdowns (Order & Dynamic Counts)
- **Swapped Dropdown Hierarchy**:
  - Reordered the filter row controls above the intermission words table:
    1. **Word List Dropdown** (`#dictionary-filter-dropdown`): `CSW + AW`, `CSW`, `CSW Only`, `NWL`, `AW`
    2. **Length Dropdown** (`#length-filter-dropdown`): `All (found/total)`, `4LW (found/total)`, `5LW`, etc.
    3. **Order Dropdown** (`#order-filter-dropdown`): `Alphabetical`, `Random`
- **Dynamic Found / Total Word Counts**:
  - Length filter dropdown dynamically computes word counts against the active dictionary selection (`dictFilteredWords`):
    - Example: `All (34/58)`, `4LW (12/18)`, `5LW (15/22)`.
  - Switching between dictionary categories immediately recalculates counts and updates option labels in real time.
- **Commit History**: `115dec4e` (Dynamic counts), `f1f32a6f` (Swapped positions).

---

### C. 24-Hour Room 10-Minute Inactivity Kick Exemption
- **Root Cause Resolution**:
  - `static/js/app.js` contained duplicate global listeners (`DOMContentLoaded` and `visibilitychange`) that evicted players after 10 minutes of tab inactivity without verifying if the active room was a 24h daily room.
- **Implementation**:
  - Added `is24h` checks in `app.js` (line 650 and line 4504) and strengthened checks in `play.js` (lines 1429 and 1828) against room IDs containing `24h` / `86400` or `time_limit >= 7200`.
  - Confirmed 24h room finds remain properly isolated from leaderboards, find counts, and player ratings.
- **Commit History**: `be3ff605`.

---

### D. Selected Word Cell Visual Elevation
- **High-Contrast Dark Blue to White Breathing Pulse**:
  - Selected word card pulses from dark blue (`#0f2b5c`) to pure white (`#ffffff`) using CSS `@property` color interpolation.
  - Dynamically flips text color between white and `#0f2b5c` for AAA contrast compliance throughout the cycle.
- **Glowing Indicator Pin (`▶`)**:
  - Active pointer beacon at `left: -20px` bouncing smoothly with a 1.2s spring animation (`@keyframes pin-bounce`) toward the selected card.
  - Multi-tier neon drop-shadow glow matching the word category (Sky blue, Lavender for AW, Green for bonus, Gold for CSW, Rose for opponent).
- **Diagonal Light Sheen Sweep**:
  - 115-degree linear gradient sweep every 2.6 seconds giving the card a polished reflective sheen.
- **Commit History**: `95ffe0ac`, `b41691a4`.

---

### E. Universal Word Definition Popup Edge Detection
- **Auto-Repositioning**:
  - Viewport edge detection prevents popover cut-off when clicking words near the bottom or right of the screen.
  - Flips popover upward or adjusts coordinate offsets automatically across desktop and mobile devices.

---

### F. Dynamic Word Lookup Length Badge
- **Layout Alignment**:
  - Formatted word length as `<span class="tool-def-len-badge">${wordLen}L</span>` next to dictionary tags in Tools -> Word Charts -> Dynamic Word Lookup, matching the universal popup styling.

---

### G. Clean Baseline for Word Tallies & Dynamic Divisions
- **Zero Authentic Baseline**:
  - Purged artificial seed benchmark tallies from `word_stats.json` and `user_word_tallies`.
  - All words initialize at 0 finds with max rarity division (7+L: Div 5 of 5, 6L: Div 3 of 3, 5L: Div 2 of 2).
  - Word Charts Dynamic Word Lookup, Find Count, and live game scoring are locked to the exact same authenticated database source.
  - `word_finds` capped at 10 recent finders per word to protect SQLite database compactness.

---

### H. Concept 1: Word Length Radar (Desktop & Laptop Game Rooms)
- **Problem**:
  - In `hmm.png`, `.board-panel` stretched vertically to match the height of the left and right panels, leaving a 250px–350px empty field below Rotate and Transpose buttons.
- **Implementation & Refinements**:
  - Created a responsive, live-updating **Word Length Radar** (`#word-length-radar`) directly below the word input and rotate/transpose row.
  - Dynamically calculates active board letter lengths (`totalByLen` vs `foundByLen`) and renders animated progress bars with vibrant bracket colors (4L: Emerald, 5L: Amber, 6L: Purple, 7L: Pink, 8L+: Indigo).
  - Displays instant optimistic completion updates as the player submits words, plus completion checkmarks (`✓`) when a letter bracket is fully cleared.
  - In intermission, switches seamlessly to a round completion debrief (`Word Length Breakdown`).
  - **Refinements (Nov 24)**:
    - **Equal Padding via Auto Margins**: Replaced the disproportionate bottom margin with `margin: auto auto !important;` on `#word-length-radar` inside the flex column `.board-panel`. This dynamically divides all remaining available vertical space 50/50, ensuring the padding above the Word Chart (under Rotate/Transpose buttons) is mathematically equal to the padding below the Word Chart.
    - **Spaced-Out Radar Rows**: Spaced out the radar rows from a cramped 13px/15px to 18px height with 7px container gap, 8px bar tracks, and roomy 12px 16px 14px padding, giving each word-length bracket clear breathing room and distinct visual separation.
    - **Zero Vertical Scrolling**: Enforced `overflow-y: hidden !important; overflow-x: hidden !important;` on `body.is-desktop .board-panel`. Compact laptop media query (`max-height: 820px`) scales elements smoothly so the panel never overflows or triggers a scrollbar on laptops or desktops.
    - **Clean Header**: Removed the lightning bolt icon (`⚡`) to the left of the title.
    - **Intermission Badge Cleanup**: Removed the purple "Round Review" pill badge during intermission.
- **Strict Desktop / Laptop Only Constraint**:
  - Hidden unconditionally on mobile devices and tablets via `body.is-mobile #word-length-radar { display: none !important; }` and `@media (max-width: 992px) { display: none !important; }`.
  - In JS, skips execution if `window.innerWidth <= 992` or mobile user agent is detected.
- **Cache Busters**: `play.css?v=1795370008`, `play.js?v=1795390004`.
- **Commit History**: `62b8c47b`, `939d38fb`, `61be58ea`.

---

### I. Strict 300–400 Word Count Range Guaranteed on All 24h Rooms
- **Mandate**:
  - All 24h rooms (`time_limit >= 7200` or room ID containing `24h` / `86400`, e.g. `pub_v2_accumulative_4x4_86400`, `pub_v2_accumulative_4x6_86400`, `pub_v2_accumulative_5x7_86400`, `pub_v2_accumulative_6x8_86400`) must strictly have 300–400 words on each and every round.
  - Never 200–300 words, and never 400–500 words: simply 300–400 words (300 to 399 inclusive) each and every time.
- **Root Causes Identified & Resolved**:
  1. **Range Definition Boundary**: `get_factchecked_wc_range` and `is_board_count_valid` define `'300-400'` as `300 <= count <= 399` (inclusive; 400 is classified as `400-500`). `_parse_word_count_range` in `board_generator.py` previously parsed `"300-400"` as `(300, 400)`, allowing boards with exactly 400 words to be generated and accepted, which were then fact-checked into the `400-500` bucket. Fixed `_parse_word_count_range` so that standard ranges cap at `high - 1` (e.g., `(300, 399)`).
  2. **`SpinnerSet.sanitize_params` Priority**: The `elif is_24h:` branch was evaluated after `if p.get('_exact_wc_calculated'):`. If a pre-calculated or cached board was sanitized, its prior word count range remained. Moved `if is_24h:` to the top of `sanitize_params`, locking `word_count_range = '300-400'`, `board_format = 'Valued Letters'`, and `use_added_words = False`.
  3. **First-Board Contract Exemption**: In `game_room.py:start_round`, line 6087 previously checked `if _bad and not is_24h:`, which exempted 24h rooms from the first-board contract check and allowed non-compliant cached/fallback boards to be accepted. Removed `and not is_24h` and added strict 24h checks (`300 <= count <= 399`).
  4. **Unconstrained Cache Pops**: In `start_round` and `start_board_search`, `pop_compatible_cached_board` was called without a `word_count_range` constraint, allowing non-300–400 boards to be popped. Added `word_count_range='300-400'` filter and explicit `300 <= len(words) <= 399` acceptance guards.
  5. **Active Word Count Shield**: Enabled the active word count shield in `start_next_round` for 24h rooms (`is_24h_room`), healing any non-compliant board immediately before the round starts.
  6. **Database Persistence & Restore Validation**: On room creation in `create_room`, 24h boards loaded from `active_boards` in the SQLite DB are validated for `300 <= len(words) <= 399`. If a saved board from earlier has < 300 or > 399 words, it is rejected and a fresh, compliant 300–400 word board is generated.
- **Production Status**:
  - Live VPS verified via direct query to `/home/morpheme/morpheme/morpheme.db`:
    - `pub_v2_accumulative_4x4_86400`: **352 words**, range = **'300-400'**, format = **'Valued Letters'**
    - `pub_v2_accumulative_4x6_86400`: **307 words**, range = **'300-400'**, format = **'Valued Letters'**
    - `pub_v2_accumulative_5x7_86400`: **349 words**, range = **'300-400'**, format = **'Valued Letters'**
    - `pub_v2_accumulative_6x8_86400`: **300 words**, range = **'300-400'**, format = **'Valued Letters'**
- **Commit History**: `8ed27cd5`.

---

### J. Full Country Name Display on Flag Hover (Desktop & Laptop)
- **Mandate**:
  - When a user hovers their mouse over a user’s flag in game rooms on desktops/laptops, do not show "CA" / "US"... show "Canada" / "United States", and likewise for all flags.
  - Extend this effect across all other flags displayed across the application, such as Find Count table, Mini Profiles, Leaderboards, Lobby, Tournaments, Friends lists, Forum, and Store.
- **Root Cause & Architectural Solution**:
  - `window.getFlagHtml` in `static/js/forum.js` previously generated `alt="${code}" title="${code}"` with the 2-letter ISO country code.
  - Built an in-memory lookup map (`window.COUNTRY_CODE_TO_NAME` and `window.COUNTRY_FLAG_TO_NAME`) from the 247 countries defined in `window.ALL_FLAGS`.
  - Implemented `window.getCountryName(codeOrFlag)` which maps 2-letter codes, emojis, and ISO standards to their full country names, with fallback to `Intl.DisplayNames(['en'], { type: 'region' })`.
  - Updated `window.getFlagHtml` to set `alt="${escapedName}" title="${escapedName}"` with the full country name.
  - Updated `span.player-flag` in `static/js/play.js:5069` to carry the country name title attribute on game room player rosters.
  - Updated `static/js/tools.js:14301` ("Words Good At") to render flags with `window.getFlagHtml` and country name tooltips.
  - In Find Count (`tools.js`), Mini Profile (`tools.js`), Leaderboards (`leaderboard.js`), Lobby (`lobby.js`), Tournaments (`tournaments.js`), Friends (`tools.js`, `forum.js`), and Forum (`forum.js`), all flag icons now instantly show the full country name on hover.
  - Handled the neutral/unselected flag `'🏳️'` with a clean `"Unknown"` tooltip.
- **Cache Busters**: `forum.js?v=1795390001`, `play.js?v=1795390005`, `tools.js?v=1795390001`.
- **Commit History**: `ab9855fe`.

---

### K. 24h Room Word Validation Flash & Sound Reversion
- **Issue Reported**:
  - In 24h rooms, the color flash indicating validity of a word was sometimes very slow and sometimes flashed red when a valid word was entered and then blue quickly after, accompanied by both the failure buzzer and success chime sounds. Standard rooms (such as ACC 4x4 45s) worked normally.
- **Root Cause**:
  - In `static/js/play.js:11855`, the optimistic invalidation flash was unconditionally enabled for any word missing from the local client word list (`wordList`).
  - In 24h rooms, the server supports `DynamicAccept` for valid dictionary words geometrically present on the board that were not part of the initial 300–400 word sample.
  - When an unlisted valid word was submitted, the client prematurely executed `showValidationFeedback(..., false)` (red flash + failure buzz). When the server accepted it moments later, the client executed `showValidationFeedback(..., true)` (blue flash + success chime), producing the jarring double-flash and double-sound desync.
  - Furthermore, on 5x7 and 6x8 boards, `findWordPathOnBoard` was unconditionally populating `specialCoords` with `bonus_cell`, forcing exhaustive DFS and extensive object allocations across all 48 cells twice for every typed submission even though 24h rooms run `Valued Letters` format (which does not use bonus cells), causing noticeable latency before feedback rendered.
- **Remediation**:
  1. **Reverted 24h Optimistic Red Invalidation**:
     Restored the established rule from commit `85d51861` / `6bfe773e`:
     ```javascript
     } else if (!isPenaltyMode && !is24H) {
         // PREFACE: in standard timed rooms (e.g. ACC 4x4 45s), the board word list is complete (never truncated), so an unlisted word is invalid.
         // NOTE: In 24h rooms, the server dynamically validates & accepts valid dictionary words on the board, so do not optimistically flash red!
     ```
     In 24h rooms, words in the local list flash blue immediately (0ms). Unlisted words do not flash red prematurely; they cleanly wait for the authoritative server verdict, flashing blue or red exactly once with zero sound desync.
  2. **DFS Path Resolution Acceleration (`findWordPathOnBoard`)**:
     Guarded `specialCoords` bonus cell lookups so they only activate when the room's format is actually `Bonus Letter`. In `Valued Letters` and `Normal` rooms, DFS executes in a single pass without searching 48 cells twice, eliminating input latency.
  3. **O(1) Set Word Lookup & Clean String Trimming**:
     Cached `preState._allWordsSet` from `preState.all_words` for `O(1)` local lookup, and ensured submitted words are strictly trimmed so valid words never fail set membership.
- **Cache Busters**: `play.js?v=1795390006`.
- **Commit History**: `96a7e76d`.

---

### L. Default Tile Selectable Space (Octagon vs Diamond) Set to 31%
- **User Request**:
  - When a user creates an account, set the default mousing and swiping "Tile Selectable Space" in Gameplay & Preferences, Settings to 31% rather than 39%.
- **Implementation**:
  1. **New Account Registration & Guest Creation (`app.py`)**:
     Updated default settings inserted into `user_settings` table upon new user registration (`register()`) and guest user creation (`guest_login()`) from `('corner_cutoff', '39')` to `('corner_cutoff', '31')`.
  2. **Canonical Client-Side Settings Definition (`settings.js`)**:
     Updated `getDefaultSettings()` to set `corner_cutoff: 31`.
     Updated `applySettings()` fallback parsing to default missing/NaN values to `31`.
  3. **Interactive Hitbox Overlay CSS (`play.css`)**:
     Updated CSS clip-path fallback from `var(--corner-cutoff, 39%)` to `var(--corner-cutoff, 31%)`.
  4. **Settings UI & Hitbox Preview (`templates/index.html`)**:
     Updated the slider initial value to `31`, value label to `31%`, and preview tile SVG clip-path polygon to `polygon(31% 0%, 69% 0%, 100% 31%, 100% 69%, 69% 100%, 31% 100%, 0% 69%, 0% 31%)`.
- **Cache Busters**: `settings.js?v=1795390007`, `play.css?v=1795390007`.
- **Commit History**: `b11cf611`.

### M. Intermission Player Selection & Words Highlight in All Words
- **User Mandate**:
  - In game rooms during intermission (under the Players list):
    1. Clicking directly on a player's **username text** (`.player-username`):
       - Toggles that player's selection: selects them if unselected; deselects (`selectedPlayerUsername = null`) if already selected.
       - Slides smoothly over to the Words panel via `window.switchPlayPanel('words', true)`.
       - Ensures the All Words (`found`) tab is active.
       - Shows what words that selected player found highlighted in **blue** (`.player-word`) in the All Words table, replacing the blue color of the logged-in user's own words.
       - When clicking the username text again, reverts back to the user's own found words in blue.
       - Closes any open notepad popup and does **not** open the notepad popup modal.
    2. Clicking **anywhere else** on the player card item (`.player-item`):
       - Opens the tactile yellow lined notepad popup modal for that user (`window.showNotepadPopup(username)`).
- **Implementation & Bugs Resolved**:
  1. **Isolated `targetPlayer` Resolution (`play.js`)**:
     - Fixed an existing bug where `find` used an OR condition matching `currentUserIdStr`, causing the logged-in user to match even when a different player was selected if the logged-in user preceded them in `state.players`.
     - Separated lookup: when `selectedPlayerUsername` is set, matches strictly against that username across `state.players` and `state.previous_players`.
  2. **Intermission Word Extraction**:
     - **Root Cause of Incorrect Words**: During intermission, `player.submitted_words` contains the words found during the round that just concluded, while `player.previous_submitted_words` contains the words from the round *prior*. A previous check (`if (state.state === 'intermission')`) was incorrectly prioritizing `previous_submitted_words`, causing the words highlighted in blue to be from an earlier round rather than the user's authentic words from the completed round.
     - **Fix**: Updated extraction across `updateGameState`, `allPlayerFoundStrs`, the `remaining` tab, and `updateWordLengthRadar` to prioritize `player.submitted_words && player.submitted_words.length > 0` (matching `showNotepadPopup` 100%), with `previous_submitted_words` serving strictly as an empty-state fallback.
     - Restricted merging of `_localSubmittedWords` strictly to the logged-in user (`!selectedPlayerUsername || isMeSelected`), preventing local words from polluting another player's word list.
     - Filtered out any obfuscated strings (`?`) from being treated as words.
  3. **Instant DOM Re-render on Click & TDZ ReferenceError Fix**:
     - Clicking `.player-username` immediately executes `_doUpdateGameState(window.lastGameState)` synchronously (0ms latency), bypassing network debounce and rendering the blue word highlights instantly.
     - **Resolved Temporal Dead Zone ReferenceError**: Identified and fixed a critical runtime error in `_doUpdateGameState` where `currentUser` was evaluated in `isOtherSelected` before its `let` declaration 12 lines below, causing an uncaught `ReferenceError: Cannot access 'currentUser' before initialization` that aborted state updates before reaching `targetPlayer` and `displayAllWords`. In addition, safely resolved `currentUsername` inside `displayAllWords`.
     - Added `targetCount` to `currentRenderKey` in both `_doUpdateGameState` and `updateIntermissionRenderKey` so that switching target players forces an immediate cache invalidation and re-render of the All Words list.
  4. **Visual Elevation & Feedback**:
     - Added `.player-item.selected-player` border (`#409cff`), background glow, and bold blue username style.
     - Added cursor pointer and hover underline with color `#409cff` to `.player-username`.
     - Words panel header displays `All Words (${selectedPlayerUsername})` when another player is active, and the stats header prefixes the count with the player's username.
- **Cache Busters**: `play.js?v=1795390011`, `play.css?v=1795390010`.
- **Commit History**: `ab7af99b`, `0f47944d`, `738d9093`.

---

### N. End-of-Round Rating Qualification (Played from Start & Score >= 1)
- **User Mandate**:
  - A user entered the room mid-round and started playing. Their rating was affected at the end of the round.
  - Mandate: Only affect the ratings of people who played from the beginning of the round and had a score >= 1.
- **Root Cause & Remediation**:
  1. **Authoritative Starter Tracking (`round_starting_player_ids`)**:
     - At round start (`start_round()`), recorded the exact set of user IDs present in `room.players`: `room.round_starting_player_ids = {str(p.user_id) for p in room.players}`.
     - Previously, `add_player` used `prior_last_active >= round_started_at - 30`, allowing any player active anywhere on the site in the 30 seconds before round start to bypass mid-round detection when entering at second 40 of a 45-second round.
     - Now, anyone entering or re-entering while `self.state == 'active'` whose ID is not in `round_starting_player_ids` is strictly marked `joined_mid_round = True`.
     - In `submit_word()`, an added guard verifies that if a player submits a word while active and is not in `round_starting_player_ids`, their `joined_mid_round` flag is immediately set to `True`.
  2. **Score >= 1 and Mid-Round Rating Exclusion (`rating_logic.py`)**:
     - Updated `calculate_proportional_rating_change` so that `competitive_humans` strictly requires:
       `not is_ai and not is_guest and not joined_mid_round and score >= 1`.
     - Players with score 0 (or score < 1) or who joined mid-round receive `0` rating change and are omitted from all expectation calculations.
  3. **Placement, Database, and Bounty Safeguards (`game_room.py`)**:
     - In `game_room.py`, wrapped player rating processing in a strict qualification guard:
       `if not (played_from_start and has_score and is_eligible_human): p.rating_change = 0; continue`.
     - Unrated placement rounds are not recorded for players who did not play from the beginning or had score < 1.
     - Abandonment bounty distribution (`eligible_receivers`) requires `played_from_start and has_score and not is_unrated`.
  4. **Consecutive Round Starter Population (`start_next_round`)**:
     - Identified that while `start_round()` populated `room.round_starting_player_ids`, consecutive rounds transitioning from intermission via `start_next_round()` were not refreshing this set. As a result, players who continued playing into subsequent rounds were falsely failing the starter ID lookup.
     - Populated `room.round_starting_player_ids = {str(p.user_id) for p in room.players}` at both player reset and round activation in `start_next_round()`.
     - Guarded `if round_starters and str(user_id) not in round_starters:` so empty starter sets never inadvertently flag players as mid-round.
- **Commit History**: `3049ab7e`, `75167958`.

---

### O. Word Length Radar Redundant Percentage Badge Removal
- **User Mandate**:
  - "Get rid of “1% Found“ badge next to WORD LENGTH RADAR if the user enters a word. You are already saying the percentage found further to the right."
- **Remediation**:
  - In `static/js/play.js` (`updateWordLengthRadar`), removed the status pill badge (`<span class="radar-pill active">${overallPct}% Found</span>`) rendered in `.radar-title-group`.
  - In `static/css/play.css`, enforced `.radar-pill { display: none !important; }` to prevent any stale rendering.
  - The Word Length Radar title displays cleanly as `Word Length Radar` (in-game) or `Word Length Breakdown` (intermission), while overall totals and percentage remain on the right (`${grandFound} / ${grandTotal} (${overallPct}%)`).
- **Cache Busters**: `play.css?v=1795390011`, `play.js?v=1795390012`.
- **Commit History**: `44687031`.

---

### P. Immediate Round-End Rating Change Calculation & Display
- **User Mandate**:
  - "Can you calculate the change in rating faster so that as soon as the round ends, the change in rating becomes visible immediately (For example: “(+7)”)"
- **Root Cause Analysis**:
  - Previously in `game_room.py`, when a round transitioned from `active` to `intermission`, all rating updates (`calculate_proportional_rating_change`, unrated placement, and SQLite writes) were deferred to a detached background thread (`process_results_async()`).
  - When the client polled at the exact moment the round ended, `self.state` was already `'intermission'`, but `p.rating_change` was still uncomputed (`0`).
  - Seeing `state === 'intermission'`, the client stopped its rapid transition polling loop and dropped back to its 1000ms polling interval.
  - As a result, users experienced a 1–2 second delay where rating changes showed `(0)` before suddenly jumping to `(+7)`.
- **Backend Remediation (`game_room.py`)**:
  - Moved split mode score finalization, winners history, full proportional rating calculations, unrated placement, abandonment bounty, in-memory player updates (`p.rating`, `p.rating_change`), and database commits to run **synchronously** during the `ACTIVE -> INTERMISSION` transition within `check_and_update_state()`.
  - Background thread (`process_results_async`) now strictly handles heavy non-blocking I/O operations: `rm.save_round_history()` and `rm.log_word_tally()`.
  - Every state response returning `intermission` now contains fully computed `p.rating_change` and updated `p.rating` from the very first frame.
- **Frontend Remediation (`static/js/play.js`)**:
  - Added `computePredictedRatingChanges(players, boardFormat)` mirroring authoritative rating logic (excluding guests, AI, mid-round joiners, and players with score < 1).
  - In `updateLocalTimer()`, when countdown hits `0:00`, precomputes predicted changes immediately prior to calling `resetPlayersToViewAll()`.
  - In `renderPlayers()`, if `p.rating_change` is not yet received from the server during round-end/intermission, seamlessly displays the predicted change and adjusts `displayRating` with zero milliseconds of delay.
  - Clears predicted state upon receipt of authoritative server state or new round initialization.
- **Cache Busters**: `play.js?v=1795390013`.
- **Commit History**: `eded000d`.

---

### Q. Selected Player Cell Pulse Animation & Bonus Word Selected Text Shadow Elimination
- **User Mandates**:
  1. "If a username is selected under Players during intermission, have the cell flash dark blue and white in the same way you did for the selected word under All Words during intermission."
  2. "Also, when the Bonus Word is selected under All Words during intermission, when the background color of the cell is white, the black text of the word exhibits a shadow or glowing effect, making it difficult to read . The other cells work properly."
- **Root Cause & Remediation**:
  1. **Selected Player Dark Blue & White Breathing Pulse**:
     - Updated `.player-item.selected-player` in `static/css/play.css` to adopt `@keyframes selected-pulse-default` (3s breathing pulse between dark blue `#0f2b5c` and pure white `#ffffff`) and `@keyframes selected-sheen-sweep` (2.6s reflective linear gradient sheen), matching `.word-item.finder-active`.
     - Ensured all child elements inside `.player-item.selected-player` (`.player-username`, `.player-rating-val`, `.player-words-count`, `.player-score-val`, `.player-rank`) dynamically inherit `color: var(--selected-text-color, #ffffff) !important;` and `text-shadow: none !important;` across both dark and light themes (`theme-white`, `theme-light-gray`).
     - Preserved the golden trophy glow for exceptional performance (`.player-item.selected-player .trophy-icon`).
  2. **Bonus Word Selected Text Shadow Elimination**:
     - Found that `.word-item.bonus-word` had `text-shadow: 0 0 8px rgba(0, 0, 0, 0.8), 0 0 4px #000 !important;`. When selected (`.word-item.finder-active.bonus-word`), at the 50% white background phase of `selected-pulse-bonus`, the text turned dark blue (`#0f2b5c`), but retained the 8px black drop shadow, creating a blurry halo/glowing distortion. Other cells did not exhibit this because only bonus words had the heavy text shadow.
     - Added `text-shadow: none !important; filter: none !important;` across `.word-item.finder-active`, `.word-item.finder-active.bonus-word`, and all their child text/spans, cleanly eliminating the shadow when selected while preserving the normal green bonus pulse when unselected.
  3. **Selected Player Highlighting Smoothness & Jitter Elimination**:
     - Identified that `.player-item` had `transition: all 0.2s;` in line 520. When running CSS `@keyframes` animations, CSS transitions were intercepting property updates on every frame, causing stutter and jagged interpolation. Added `transition: none !important;` and `transform: translateZ(0) !important;` to `.player-item.selected-player` and descendants.
     - In `static/js/play.js`, clicking a username previously set `lastPlayersHtml = ''`, executed `_doUpdateGameState()` which wiped `innerHTML`, and immediately fired `updateGameState()` which wiped `innerHTML` again 100ms later upon network response. Replaced this with in-place `.selected-player` class toggling and prevented redundant server polling on click, matching the 60fps smoothness of `toggleWordSelection` under All Words.
  4. **Removal of Sheen on Selected Player Cell**:
     - Removed `background-image: linear-gradient(...)`, `background-size`, `background-repeat`, and `selected-sheen-sweep` from `.player-item.selected-player`, leaving pure smooth dark-blue to white breathing pulse (`selected-pulse-default`) without any diagonal sheen streak.
- **Cache Busters**: `play.css?v=1795390016`, `play.js?v=1795390015`.
- **Commit History**: `21b7b724`, `033e1b7b`, `6744c43d`.

---

### R. Bonus Word Flashing Dark Blue & White When Selected Under All Words
- **User Mandate**:
  - "On desktops and laptops, under All Words, the green Bonus Word does not flash dark blue and white like the rest of the cells below it."
- **Root Cause Analysis**:
  - In `static/css/play.css`, `.word-item.bonus-word`, `#submitted-words-list .word-item.bonus-word`, `#submitted-words-list .word-item.player-word.bonus-word`, and light theme rules (`theme-white`, `theme-light-gray`) had rules setting `background: rgba(34, 197, 94, 0.4) !important;` and `animation: bonusPulse 1.5s infinite;`.
  - Because `background:` is a CSS shorthand, it overwrote `background-color: var(--selected-cell-bg)` whenever the bonus word was selected (`.finder-active`).
  - Furthermore, in light themes, rules at the end of the stylesheet had higher specificity than `#submitted-words-list .word-item.finder-active.bonus-word`, forcing `bonusPulse` and green background even while active.
  - In addition, `@keyframes selected-pulse-bonus` pulsed with a green border (`#4ade80`) and green halo instead of the blue border/halo (`#93c5fd` / `#3b82f6`) used by all other cells below it in All Words.
- **Remediation**:
  1. **Scoping Unselected Bonus Word Rules**: Added `:not(.finder-active)` to all static green background and `bonusPulse` selectors (`.word-item.bonus-word:not(.finder-active)`, `#submitted-words-list .word-item.bonus-word:not(.finder-active)`, `#submitted-words-list .word-item.player-word.bonus-word:not(.finder-active)`, and all theme variants). Unselected bonus words remain vibrant green with `bonusPulse`.
  2. **Authoritative Selected Pulse Alignment**:
     - Added authoritative override for `.word-item.bonus-word.finder-active` across `#submitted-words-list`, `#previous-words-list`, and all light/dark themes to use `background: var(--selected-cell-bg, #0f2b5c) !important;`, `background-color: var(--selected-cell-bg, #0f2b5c) !important;`, `border: 2px solid var(--selected-border-color, #3b82f6) !important;`, blue halo box shadow, and `animation: selected-pulse-default 3s infinite ease-in-out, selected-sheen-sweep 2.6s infinite ease-in-out !important;`.
     - Updated `@keyframes selected-pulse-bonus` to pulse between dark blue (`#0f2b5c`) and pure white (`#ffffff`) with light blue border (`#93c5fd`), perfectly matching `selected-pulse-default`.
  3. **Preserved Gold Badge Pill**: Ensured `.bonus-word-tag` retains its gold pill styling (`background: #ffd700; color: #000; border: 1px solid #daa520`) with black star icon while active.
- **Cache Busters**: `play.css?v=1795390017`.
- **Commit History**: `b97bd3c8`.

---

### S. Player Roster Distinct Click Regions & Intermission Scoping
- **User Mandate**:
  - "Can you ensure that when the colored square and username are not being pressed, that for anywhere else on the cell, the notepad appears (only during the intermission)? When the username is pressed, it highlights blue what the user found under All Words (only during the intermission), and when the colored square indicating rating is pressed, their miniprofile appears (at any time it is pressed)? You don't have to highlight a user during a round; this was intended for during the intermission."
- **Root Cause & Remediation**:
  1. **Colored Square Indicating Rating (`.rating-square`)**:
     - Configured with `cursor: pointer !important;`, hover scale (`transform: scale(1.15); filter: brightness(1.2)`), and click listener that calls `window.showMiniProfile(username)` at **any time** (during live rounds and intermission).
  2. **Username (`.player-username`)**:
     - During intermission: clicking toggles `selectedPlayerUsername`, highlighting what that user found under All Words in blue, sliding to the Words panel, and triggering the smooth dark-blue/white pulse on the player cell. Has `cursor: pointer;` and turns blue with an underline on hover.
     - During round: clicking does nothing, does not select/highlight the user, has `cursor: default;`, and does not underline on hover.
  3. **Anywhere Else on Cell (`.player-item`)**:
     - During intermission: clicking anywhere on the cell outside the colored square and username (rank `#1`, rating text `1200 (+5)`, flag, words count, score `pts`, or empty card area) opens the user's yellow-lined notepad modal (`showNotepadPopup(username)`). Has `cursor: pointer;` and `title="Click to view notepad"`.
     - During round: clicking does nothing, does not open notepad, and has `cursor: default;`.
  4. **Unified Event Delegation (`#players-list`)**:
     - Centralized click handling via single listener on `#players-list` with `hasPlayerRosterClickListener`, guaranteeing zero duplicate event bindings or dropped listeners during in-place DOM updates or roster re-renders.
- **Cache Busters**: `play.css?v=1795390018`, `play.js?v=1795390016`.
- **Commit History**: `2ad4e125`.

---

### T. Players List Stuck on "Loading..." in Game Rooms
- **User Mandate**:
  - "and now the users in room under 'Players' wont load in game rooms. It simply says 'Loading...'"
- **Root Cause Analysis**:
  - In `static/js/play.js`, inside `renderPlayers(players, currentUser, state)`:
    - At function scope, `const isIntermission` was correctly computed.
    - However, inside the `itemsToRender.map((p) => { ... })` callback, a duplicate declaration `const isIntermission = ...` was defined lower down in the loop body.
    - In ES6 JavaScript, `const` declarations are block-scoped and subject to the **Temporal Dead Zone (TDZ)**: referencing `isIntermission` higher up inside the loop body before that lower line executed caused an uncaught `ReferenceError: Cannot access 'isIntermission' before initialization`.
    - This exception immediately terminated execution of `renderPlayers` on the very first player item, preventing `#players-list.innerHTML` from being assigned. Consequently, the DOM element remained permanently displaying its initial template placeholder: `<p class="placeholder">Loading...</p>`.
    - Furthermore, `state.your_username` was referenced without null guarding, risking an uncaught `TypeError` if `state` was passed as `null`.
- **Remediation**:
  1. **Removed Shadowed TDZ Declaration**:
     - Removed the duplicate `const isIntermission = ...` from inside `itemsToRender.map`, seamlessly using the outer function-scoped `isIntermission`.
  2. **Safe State Access**:
     - Standardized `curGameState = state || window.lastGameState || {};`.
     - Replaced direct `state.your_username` with `(curGameState && curGameState.your_username) || window.currentUsername` and passed `curGameState` to `isPlayerDNP(p, curGameState)`.
  3. **Defensive Error Handling**:
     - Wrapped the entire body of `renderPlayers` in a `try...catch (err) { console.error('[RenderPlayers Error]', err); }` block to guarantee unexpected errors are logged and do not crash the UI.
  4. **Preserved All Three Distinct Interaction Zones**:
     - Rating square (`.rating-square`): opens Mini-Profile at any time.
     - Username (`.player-username`): selects player to highlight their words under All Words strictly during intermission.
     - Anywhere else (`.player-item`): opens player notepad modal strictly during intermission.
- **Cache Busters**: `play.js?v=1795390017`.
- **Commit History**: `407ba4a2`.

---

### U. Scoreboard Frozen at Initial Score During Active Gameplay
- **User Mandate**:
  - "The scoring of the users under Players stays at 4 pts. I get many other words but they didn't count"
- **Root Cause Analysis**:
  - Verified from the server's live `word_tally.log` that the backend was receiving, validating, scoring, and tallying all words perfectly (e.g. TARES, RESET, ERASE, SAREE, TASTE, PARSE, TRESS, TRASS in `pub_v2_accumulative_4x6_45`).
  - However, in `static/js/play.js`, a structure-check block in `renderPlayers`:
    ```javascript
    if (existingItems.length === players.length && lastPlayersHtml !== null) {
        if (sameStructure) { ... return; }
    }
    ```
    was previously added to prevent CSS animations from restarting when selecting a player *during intermission*.
    - Because this check was not restricted to intermission, whenever a player scored points during an active round, the roster count and usernames remained the same (`sameStructure === true`). As a result, `renderPlayers` updated neither `.player-score-val` nor `.player-words-count`, did not re-rank, and returned early without assigning `listEl.innerHTML = html;`.
    - This caused the scoreboard to appear permanently frozen at whatever score was rendered first (e.g., 4 pts).
- **Remediation**:
  1. **Restricted In-Place Selection Toggle to Intermission**:
     - Added `if (isIntermission && existingItems.length === players.length && lastPlayersHtml !== null)`.
     - During active rounds (`isIntermission === false`), this early exit is completely bypassed: `listEl.innerHTML = html;` is executed on every score change, updating player scores, word counts, and ranks immediately.
     - During intermission, the in-place class toggle remains active so clicking usernames maintains the smooth 60fps selection pulse without tearing down the DOM.
  2. **Optimistic Local Score & Word Count Updates**:
     - In `submitWord`, updated `myPlayer.score = (myPlayer.score || 0) + localPts` and `myPlayer.words_count = myPlayer.submitted_words.length` locally so the scoreboard increments with 0ms visual latency.
     - Synchronized `me.words_count = me.submitted_words.length` upon authoritative server response.
- **Cache Busters**: `play.js?v=1795390018`.
- **Commit History**: `8ce0edc3`.

---

### V. Confining Username Hitbox Strictly to Text to Prevent Inadvertent Selection
- **User Mandate**:
  - "When I press on the cell (not username or colored square), it treats it as a username press, in the sense that it takes me to All Words with blue cells indicating the words they got. But if I press it again, then the notepad appears."
- **Root Cause Analysis**:
  - In `static/css/play.css`, `.player-username` had `flex: 1` and `min-width: 80px`.
  - Because of `flex: 1`, the `<span class="player-username">` element expanded horizontally across the entire middle of the top row, filling all empty space between the rating square and the rating value.
  - When users attempted to click "on the cell" by clicking in the empty space to the right of the username text, their click physically landed inside the stretched `.player-username` bounding box. This triggered `e.target.closest('.player-username')`, unintentionally selecting the player and sliding to All Words.
  - Subsequent clicks near the lower half of the cell (or outside the stretched span) missed `.player-username` and correctly opened the notepad popup.
- **Remediation**:
  1. **Strict Hitbox Confinement (`play.css`)**:
     - Changed `.player-username` from `flex: 1` to `flex: 0 0 auto !important; width: auto !important; max-width: 120px !important;`.
     - Confined the element's bounding box and hover underline strictly to the visible text characters of the username.
  2. **Dedicated Top-Row Flex Spacer (`play.js`)**:
     - Added `<div style="flex:1;"></div>` between `.player-username` and `.player-rating-val` in `.player-row-top`, matching `.player-row-bottom`.
     - Clicks in the empty space between the username and rating value now hit the card spacer/body and immediately trigger `showNotepadPopup(username)` on the first click.
- **Cache Busters**: `play.css?v=1795390019`, `play.js?v=1795390019`.
- **Commit History**: `aa5ab7ed`.

---

### W. Density Format Tile Stability and Stepwise Lightening
- **User Mandate**:
  - "The Density Format moves the shades of the tiles all over the place when entering words. Keep the shades on the board in place and ensure that for every valid word submitted, the tiles that use the letters go one shade lighter closer to white"
- **Root Cause Analysis**:
  1. **Shades Moving During Word Entry**:
     - In `applyDensityToCell`, `if (isHighlighted)` checked for `.selected`, `.current`, and `.typing-highlight`. When true, it stripped `background`, `background-color`, `color`, and `box-shadow` from the tile!
     - When typing into `word-input`, `handleWordInput` added `.typing-highlight` / `.selected` to matching tiles on every keystroke, stripping their grayscale density background and painting them bright yellow or accent blue.
     - On backspace or changing letters, unmatched tiles had their highlight classes removed and `applyDensityToCell` re-applied the grayscale background with a 0.4s fade animation.
     - As a result, simply typing or editing words caused board tiles to constantly flicker, strip, and shuffle their background shades.
  2. **DOM Teardown on Density Changes**:
     - In `renderBoard`, `if (boardJSON === lastRenderedBoardJSON && densityJSON === lastRenderedDensityJSON ...)` checked `densityJSON`.
     - Whenever density updated after a word submission, `densityJSON` changed, which bypassed the skip check and caused `renderBoard` to tear down all `.board-cell` DOM elements (`existingCells.forEach(el => el.remove())`), resetting active animations, DOM elements, and focus.
  3. **Continuous Sub-perceptible Shading vs Stepwise Lightening**:
     - Previously, density was calculated via raw continuous word counts normalized by `maxD` (which could be 60–100+ words). Submitting 1 word only moved lightness by ~1.2% (virtually imperceptible), while dropping to 0 words abruptly snapped to pure white.
- **Remediation**:
  1. **Keeping Shades Firmly in Place During Word Entry (`play.js` & `play.css`)**:
     - In `applyDensityToCell`, restricted the style-stripping guard solely to celebratory flashes and bonus cell highlights. `.selected`, `.current`, and `.typing-highlight` no longer strip the tile's background shade or text color.
     - Highlight selection during word entry is now applied via non-destructive ring overlays (`box-shadow: inset 0 0 0 3px ...` and `outline: 2px solid #ffffff !important`).
     - Added `.is-density` styling in `play.css` and `.format-density` on `#game-board` so typing/mouse selection displays crisp highlight rings around tiles while their background grayscale shade stays 100% solid and undisturbed.
  2. **In-Place Density Shading Updates Without DOM Teardown (`play.js`)**:
     - Added an in-place density update branch in `renderBoard`: when letters and board dimensions are unchanged and only density has updated, `renderBoard` applies new density shades in-place across existing `.board-cell` elements rather than destroying and recreating the DOM.
  3. **Stepwise Lightening Closer to White (`game_room.py` & `play.js`)**:
     - In `game_room.py`, `initialize_density` maps raw word counts to 5 discrete shade levels (`MAX_SHADES = 5`), with `max_cell_density = 5`.
     - In `applyDensityToCell`, each level maps to a distinct lightness step (`Math.max(20, Math.round(100 - (ratio * 80)))`), yielding discrete shades: 20% (darkest), 36%, 52%, 68%, 84%, and 100% (`#ffffff` pure white).
     - For every valid word submitted, `update_density_for_word` decrements the level of each tile in the word's path by 1. Each tile used visibly advances **one shade lighter closer to white** with a smooth CSS transition. Once white (level 0), it remains pure white.
     - Fixed `self.update_density_for_word(ai, wd['word'])` in bot word generation.
- **Cache Busters**: `play.css?v=1795390020`, `play.js?v=1795390020`.
- **Commit History**: `579f4d2e`.

---

### X. Equalizing Left and Right Padding Around Cells Under Words and All Words
- **User Mandate**:
  - "Can you make the padding on the left side of a cell under Words and All Words equal to the padding on the right side of it?"
- **Root Cause Analysis**:
  - In `static/css/play.css`, `#submitted-words-list` (the list container housing cells under Words during active play and All Words during intermission) and `#previous-words-list` were configured with:
    ```css
    padding-left: 22px;
    padding-right: 8px;
    ```
  - This asymmetry was introduced in an earlier commit to provide clearance for an exterior glowing indicator arrow (`.word-item.finder-active::before { left: -20px; }`).
  - As a result, every `.word-item` cell was pushed 22px away from the left edge of the panel container, but only 8px away from the right edge, causing cells to appear noticeably off-center and right-shifted.
- **Remediation**:
  1. **Equalizing Container Padding (`play.css`)**:
     - Updated `#submitted-words-list` and `#previous-words-list` to:
       ```css
       padding-left: 8px;
       padding-right: 8px;
       ```
     - The padding on the left side of every cell under Words and All Words is now exactly equal to the padding on the right side (8px on each side), centering the cells symmetrically within the container.
  2. **Refining Indicator Pin Offset (`play.css`)**:
     - Adjusted `.word-item.finder-active::before` to `left: -7px !important; font-size: 10px !important;` so the active pointer indicator fits cleanly within the 8px margin without clipping against `overflow-x: hidden`.
- **Cache Busters**: `play.css?v=1795390021`.
- **Commit History**: `c38a752f`.

---

### Y. Word Length Radar Complete Reset on New Round Start
- **User Mandate**:
  - "The WORD LENGTH RADAR doesn’t reset the information upon the start of a new round. Ensure when a new round starts, the meters, “7L [etc.]” and numerical values to the right of it all reset to the new information of new round."
- **Root Cause Analysis**:
  1. **Stale Words Fallback During Active Play**:
     - In `updateWordLengthRadar` (`static/js/play.js`), player found words were extracted via:
       ```javascript
       const pWords = (me.submitted_words && Array.isArray(me.submitted_words) && me.submitted_words.length > 0)
           ? me.submitted_words
           : (me.previous_submitted_words || []);
       ```
     - When a new round started, `me.submitted_words` was an empty array (`[]`, length 0).
     - Because `me.submitted_words.length > 0` was `false`, it evaluated the fallback `(me.previous_submitted_words || [])`, loading the completed round's submitted words into the new round!
     - As a result, the meters remained partially or fully filled, and counts showed the previous round's words instead of resetting to 0.
  2. **Board Length Totals Poisoned by Stale Server Metadata**:
     - Board length totals were derived from `state.total_counts_by_len` before falling back to `all_words`.
     - When a new round started, if `state.total_counts_by_len` was still referencing the completed round, `validKeys.length === 0` was `false`, preventing `totalByLen` from being derived from the new board's actual `all_words`.
     - This caused length badges (e.g. `3L`, `4L`, `5L`, `6L`, `7L`, `7L+`) and row totals to retain the previous round's structure and word counts.
- **Remediation**:
  1. **Strictly Guarded Active vs Intermission Words**:
     - In `updateWordLengthRadar`:
       ```javascript
       let pWords = [];
       if (isIntermission) {
           pWords = (me.submitted_words && Array.isArray(me.submitted_words) && me.submitted_words.length > 0)
               ? me.submitted_words
               : (me.previous_submitted_words || []);
       } else {
           // Strictly count words submitted in this active round! Never fall back to previous round!
           pWords = (me.submitted_words && Array.isArray(me.submitted_words)) ? me.submitted_words : [];
       }
       ```
     - During active gameplay, `pWords` strictly uses the current round's `submitted_words`, guaranteeing 0 found words at round start.
  2. **Authoritative Lengths and Totals from Active Word List**:
     - When `wordList` is present (`state.all_words` during active rounds), `totalByLen` is computed directly from `wordList`, ensuring length badges ("3L", "4L", "5L", "6L", "7L", "7L+", etc.) and row totals immediately reflect the new board's distribution.
     - Fallback to server `total_counts_by_len` only occurs if `wordList` is absent or empty and matches `state.current_round`.
  3. **Immediate Radar Re-render on Round Transition**:
     - Explicitly called `updateWordLengthRadar(state)` inside the `isNewRound` block in `updateGameState` and in `resetPlayUI()`.
     - Upon round start, title displays "Word Length Radar", overall stats reset to `0 / [new round total] (0%)`, every meter resets to `width: 0%`, and numerical values reset to `0 / [length total]`.
- **Cache Busters**: `play.js?v=1795390021`.
- **Commit History**: `86fe150d`.

---

### Z. Instant Validation for Words Using QU Tile with Optional U (e.g. QINDAR)
- **User Mandate**:
  - "I made the U optional in the QU tile, and when I submitted QINDAR, it validation process hesitated before accepting it as valid (blue validation flash).. Please make it validate instantly."
- **Root Cause Analysis**:
  1. **Two-Stage Fallback Roundtrip Lag**:
     - When tracing across a `QU` tile to spell `QINDAR`, the client path tracer recorded `word = "QUINDAR"`.
     - In `static/js/play.js`, path candidate permutation resolution previously only executed if `isEO` (Either/Or format) was true. In standard rooms, `word` remained `"QUINDAR"`.
     - `_allWordsSet` on client only contained `"QINDAR"` (not `"QUINDAR"`).
     - Because optimistic validation failed (`isInWordList === false`) but `couldQFallback` was detected, the client did not flash red immediately; instead, it sent `"QUINDAR"` across HTTP to the server.
     - The server rejected `"QUINDAR"`, returning `{ success: false }`.
     - Only upon receiving this HTTP rejection did client-side fallback trigger (`QU-tile fallback: retrying "QUINDAR" as "QINDAR"`), firing a *second* HTTP request for `"QINDAR"`.
     - The blue flash was delayed until the second server response returned (~200–400ms visual hesitation).
  2. **Server Solver Generation Gap (`board_generator.py`)**:
     - In `board_generator._solve_board`, `pre_split_board` did not map `'QU'` to `'Q'`.
     - Boards generated with `'QU'` tiles therefore missed Q-without-U words (like `QINDAR`) in `all_words`, preventing instant inclusion.
  3. **Word Validator DFS (`word_validator.py`) & Game Room (`game_room.py`)**:
     - `word_validator.find_word_on_board` specifically searched for `char == 'Q'` for 1-letter Q matches; `'QU'` tiles only permitted 2-letter `'QU'` branches.
     - `game_room.submit_word` only ran path permutations for Either/Or rooms and only branched `['QU']` for `'QU'` tiles.
- **Remediation**:
  1. **Synchronous Optimistic Q/QU Resolution (`static/js/play.js`)**:
     - Enabled candidate path resolution whenever the traced path touches an Either/Or tile (`/`) OR a `Q`/`QU` tile (`hasSpecialTileInPath`).
     - Any cell containing `'QU'` or `'Q'` branches both `'QU'` and `'Q'`, resolving `word` synchronously to the valid option (e.g. `"QINDAR"`) before optimistic validation runs.
     - In optimistic validation: if `word` contains `QU` and is not directly in `_allWordsSet`, but `word.replace(/QU/g, 'Q')` is in `_allWordsSet`, it immediately switches `word` to the `Q` variant (and vice-versa for typed input), flashing the blue validation halo in 0ms!
     - Handled in `submitWord`, `handleTournamentWord`, `handlePrivateMatchWord`, and `getLetterFromPathIndex`.
  2. **Server Board Generator Solver (`board_generator.py`)**:
     - In `_solve_board`, mapped `'QU'` tiles to `'Q'` in `pre_split_board` (both 2D and 3D). Because the solver logic for `'Q'` checks both `+ 'Q'` and `+ 'QU'`, boards with `'QU'` tiles now populate `all_words` with both Q-with-U (`QUICK`) and Q-without-U (`QINDAR`) words.
  3. **Authoritative Server Validation (`word_validator.py` & `game_room.py`)**:
     - Updated `WordValidator.find_word_on_board` starting coordinates and DFS search: `char in ('Q', 'QU')` tests both 2-letter (`QU`) and 1-letter (`Q`) path steps.
     - Updated `GameRoom.submit_word`: reconstructs path candidates whenever `path` contains `/`, `'Q'`, or `'QU'`, and recognizes `QU` -> `Q` word matches directly.
- **Cache Busters**: `play.js?v=1795390022`.
- **Commit History**: `6f5f404b`.

---

### AA. Automatic Disambiguation Resolution to Corresponding Meaning Article (e.g. COGENER -> Congener (chemistry))
- **User Mandate**:
  - "View the Wikipedia popup of the word COGENER… It leads to CONGENER, which is good, but please direct the user to the correct article of all of them that are listed as meanings of it. The logic applied should see what the definition of CONGENER is on the app, and then select the bullet point that corresponds to it. Apply this logic across all words of a similar situation."
- **Root Cause Analysis**:
  1. **Disambiguation Page Left Unresolved**:
     - When tapping Wikipedia for `COGENER`, Wikipedia automatically redirected `Cogener` to `Congener`.
     - `Congener` is indexed on Wikipedia as a disambiguation page (`type: disambiguation`, listing meanings for biology, chemistry, beverages, and biological species).
     - Previously, the app displayed the raw disambiguation page ("Congener may refer to...").
  2. **Page Target Stale on Parse**:
     - MediaWiki parse API was querying `page=safe_title` (`page=Cogener`), which without `&redirects=1` only parsed the `#REDIRECT [[Congener]]` wikicode instead of the target page's content bullets.
  3. **Target Word Definition Isolation**:
     - When resolving meanings, the system only checked `target_word`'s definition (`something of the same nature`). It did not inspect the destination headword definition (`CONGENER`: `"something of the same kind or nature, also COGENER"`).
- **Remediation (`app.py`)**:
  1. **Multi-Source Definition Aggregation**:
     - In `get_wikipedia_summary()` (`app.py`), the matching corpus now aggregates definitions from the queried word, any cross-referenced words (`ref_cand`), and the headword of the page Wikipedia redirected to (`dest_head`).
  2. **Redirect-Aware Page Parsing**:
     - MediaWiki `action=parse` now queries the destination title with `&redirects=1`, ensuring the actual disambiguation items (`<li>...</li>`) are parsed.
  3. **Semantic Bullet Point Scoring Algorithm**:
     - Parses each candidate bullet link, extracts the text and article summary, and scores candidates against app definitions using WordNet synonyms, taxonomy terms, and exact keyword matches.
     - Gives heavy preference to topical articles retaining the word root in their title (e.g. `Congener (chemistry)`), while filtering proper nouns and penalizing unrelated redirects (such as `Congener (biology)` which redirects to `Biological specificity`).
     - Selected `Congener (chemistry)` with top score: Title `Congener (chemistry)`, Description `Related substances`, Extract `In chemistry, congeners are chemical substances "related to each other by origin, structure, or function".`
  4. **Direct Overrides Pre-Warmed**:
     - Added `'COGENER'`, `'COGENERS'`, `'CONGENER'`, and `'CONGENERS'` pointing directly to `('Congener (chemistry)', 'https://en.wikipedia.org/wiki/Congener_(chemistry)')` in `WIKI_COMMON_NOUN_OVERRIDES` and `resolve_wiki_common_noun()`.
- **Commit History**: `9f0bd60f`.

---

### AB. Player Cell Click Targets Reorganization
- **User Request**:
  - "Move the function of the pressing of the colored square over to the username (so that when the user presses on the username in a cell, it takes them to the miniprofile). Move the current function of pressing on the username over to the point value (for example : 45 pts). Pressing on the point value the user got will show which words the user got under All Words, highlighting the words the user got blue. Maintain the function of pressing anywhere else display the notepad."
- **Implementation Details**:
  1. **Username Click -> Mini-Profile (`.player-username`)**:
     - Clicking the username triggers `window.showMiniProfile(uName)` (available at any time, during round or intermission).
     - Hover styling and cursor pointer added in `static/css/play.css`. Tooltip set to `'View Mini-Profile'`.
  2. **Point Value Click -> All Words Word Highlighting (`.player-score-val`)**:
     - Clicking the score (e.g. `45 pts`) during intermission toggles `selectedPlayerUsername`, switches to the All Words panel (`found` tab), slides to the Words column, and updates the board/word highlighting so the player's found words are highlighted in blue. Clicking again reverts to the user's own words.
     - Hover styling (`filter: brightness(1.2); color: #409cff; text-decoration: underline;`) and pointer cursor added in `static/css/play.css` when the cell is in `.is-intermission`.
     - Tooltip dynamically set to `'Click to highlight found words in All Words'` / `'Click to revert to your words'`.
  3. **Anywhere Else in Cell -> Notepad (`.player-item`)**:
     - Removed inline `onclick="...showMiniProfile..."` and `stopPropagation()` from the colored rating square (`.rating-square`).
     - Tapping anywhere else on the `.player-item` (colored square, rank, flag, input icon, trophy, mute button, padding) opens the notepad popup (`window.showNotepadPopup(uName)`) during intermission.
- **Commit History**: `c11aac38`.

---

### AC. Ironclad Prohibition of ING Sequences on Medium and Hard Difficulties
- **User Mandate & Problem**:
  - A Hard difficulty board was discovered containing an `ING` sequence (`I-N-G` adjacent along an orthogonal or diagonal path). In Morpheme, `ING` sequences are strictly forbidden on Medium and Hard difficulties.
- **Root Cause Analysis (5 Critical Loopholes Identified)**:
  1. **Protected Positions Bypass in Detector**: In `_has_ing_sequence` and `_guarantee_no_ing`, if `protected_positions` (the embedded bonus word) were passed, the algorithm checked `if protected and all(p in protected for p in visited): return False / return None`. When a bonus word contained `ING` (e.g. `SINGLE`, `GINGER`, `DOMINGO`, `KINGDOM`), the 3 tiles fell entirely within `protected`, causing the detector to falsely report that no `ING` sequence existed and refuse to break it.
  2. **Bonus Word `endswith` Limitation**: Word filters for bonus words only checked `not w.upper().endswith(('ING', 'INGS'))`. Bonus words with internal `ING` like `SINGLE` (ends in `LE`), `DOMINGO` (ends in `GO`), `FLAMINGO`, etc. were chosen as bonus words, embedding adjacent `I-N-G` sequences directly on the board.
  3. **Mania Abundance Injected After Sanitization Audits**: In `_generate_board_internal`, `_guarantee_no_ing` ran before Mania verification sweeps and final Mania guarantees. When placing abundance tiles (e.g. `G` in `G Mania`), `_apply_mania_to_board` did not check `_is_creating_forbidden_sequence` or verify difficulty, placing `G` directly adjacent to `I` and `N`.
  4. **Pregenerated Cached Boards Popped Without Difficulty Audit**: `pop_cached_board`, `pop_compatible_cached_board`, and `pop_any_cached_board` returned pregenerated boards without checking `_has_ing_sequence` against the target difficulty.
  5. **Missing Room-Level Gatekeeper**: When boards were set or promoted in `game_room.py` (`start_round()`, `start_next_round()`, `generate_spinner_params()`, and `start_board_search()`), there was no post-assignment sanitizer ensuring Medium/Hard boards were clean.
- **Remediation**:
  1. **Unconditional Detection & Breaking (`board_generator.py`)**:
     - Removed `protected_positions` bypass from `_has_ing_sequence` so any 3-tile adjacent `I-N-G` sequence anywhere on the board is detected unconditionally.
     - Updated `_guarantee_no_ing` to never skip breaking a sequence; if all tiles in the sequence are protected, it falls back to breaking the `G` tile, re-assigning it to an alternative consonant/vowel.
  2. **Bonus Word Candidate Filtering (`board_generator.py` & `game_room.py`)**:
     - Updated all candidate filters across `board_generator.py` and `game_room.py` (`pick_distinct_bonus_word`, `_select_bonus_word`, `_ensure_exact_bonus_word`, `_ensure_dictionary_distinct_words`, `get_emergency_fallback_board`, `_get_bonus_word`) to strictly exclude any word containing `'ING'` anywhere (`'ING' not in w.upper()`).
  3. **Mania Placement Guards (`board_generator.py`)**:
     - Added `difficulty` and `depth` parameters to `_apply_mania_to_board`. If `difficulty in ["Medium", "Hard"]`, the function checks `_is_creating_forbidden_sequence` before placing any letter and executes `_guarantee_no_ing` immediately following abundance placement.
  4. **Cache & Fallback Sanitization (`board_generator.py` & `game_room.py`)**:
     - `pop_cached_board`, `pop_compatible_cached_board`, and `pop_any_cached_board` verify that popped boards do not contain `_has_ing_sequence` if Medium/Hard is requested; non-compliant boards are discarded.
     - Added an **Absolute Final Gatekeeper** in `generate_board()` right before returning the board dictionary: verifies achieved/target difficulty, executes `_guarantee_no_ing`, and re-solves if necessary.
     - Added `sanitize_board_for_room_difficulty` method in `RoomManager` (`game_room.py`), deployed as an active gatekeeper across all room state transitions (`start_round()`, `start_next_round()`, `generate_spinner_params()`, and `start_board_search()`).
- **Commit History**: `e1bce8ff`.

---

### AD. Density Format Fine-Grained Lexical Shading Calibration
- **User Mandate & Problem**:
  - "The letters in Density Format turn white way too quickly… Many words used the letter, meaning the shade of the tile should have changed very slightly closer to white, not almost the whole way."
- **Root Cause Analysis**:
  - An earlier refactoring had compressed the initial density grid into 5 discrete integer buckets (`MAX_SHADES = 5`, `ceil(cnt / max_count * 5)`), and set `max_cell_density` to 5.
  - As a result, tiles with low-to-medium connectivity were compressed down to an initial value of 1 or 2, and the most heavily used tiles were capped at 5.
  - When a valid word was entered, `player.cell_density[r][c] -= 1` subtracted 1.
  - For a tile with value 1, a single submission reduced it to 0, causing `play.js` to immediately turn it 100% pure white (`#ffffff`). Even for a tile with value 5, finding only 5 words completely exhausted it to pure white, despite dozens of other words still using that letter on the board.
- **Remediation**:
  1. **Raw Word Counts Restored (`game_room.py`)**:
     - In `initialize_density`, eliminated the 5-tier integer bucketing. `density_grid` now directly stores the authentic raw count of words passing through each cell (e.g. 35, 48, 12, etc.), and `max_cell_density` is set to `max(1, max_count)` across the board.
  2. **Subtle Lightening per Word (`static/js/play.js`)**:
     - `applyDensityToCell` dynamically calculates `ratio = cur / maxD` using the true remaining word count.
     - With `grayLightness = Math.min(97, Math.max(20, Math.round(100 - (ratio * 80))))`, submitting a word that uses a letter with 50 words decrements `cur` by 1 and shifts lightness by only ~1.6%, subtly and gradually moving closer to white with every valid word.
     - A tile only turns 100% pure white (`#ffffff`) when all words passing through it are fully exhausted (`cur === 0`).
  3. **3D Cube Density Harmonized (`static/js/play.js`)**:
     - Updated 3D board cube cell density style to utilize the same dynamic max-density determination and continuous ratio calculation.
  4. **Cache Buster Incremented (`templates/index.html`)**:
     - Bumped `play.js` cache buster to `v=1795390024`.
- **Commit History**: `4b4c50b6`.

---

### AE. Continuous 'Loading...' Overlay Across Room Entry
- **User Mandate & Problem**:
  - "Please keep the “Loading…” message visible at all times while the room they pressed to enter loads, and have it disappear once the user enters the room. I have seen it before very recently where the “Loading…” message disappears, and then 20 seconds later they enter the room. The user will not know this and will get confused, and may be tempted to press it again"
- **Root Cause Analysis**:
  1. **Premature Auto-Hide Timeout**:
     - `showLoadingOverlay` previously had a hardcoded default 25-second auto-hide timeout (`timeoutDuration = autoHideMs || 25000`). When rooms required board generation/search or constraint satisfaction taking >25s, the overlay self-dismissed while the user remained on the lobby/configuration screen, leaving the UI appearing frozen for 20s before navigation completed.
  2. **Uncoordinated Failsafe Dismissals**:
     - Page navigation transitions and opening screen watchdog routines invoked `hideLoadingOverlay()` unconditionally without checking whether a room entry request (`window._isEnteringRoom`, `_isCreatingRoom`, or `_isStartingSoloMatch`) was active.
  3. **Solo Match Flag Disconnect**:
     - `startSoloMatch` in `private_matches.js` set `_isStartingSoloMatch` but omitted `_isEnteringRoom`, allowing lobby/page watchdogs to hide the overlay prematurely.
  4. **Short Double-Click Guard Window**:
     - Room entry click guards expired after only 10 seconds, allowing users to accidentally double-click room entry buttons during longer server-side generation phases.
- **Remediation**:
  1. **Forced Visibility Guard in Overlay Controller (`templates/index.html` & `static/js/app.js`)**:
     - `window.hideLoadingOverlay(force = false)` strictly refuses to hide if `!force && (window._isEnteringRoom || window._isCreatingRoom || window._isStartingSoloMatch)`.
     - In `showLoadingOverlay`, if an entry operation is active, the auto-hide timeout defaults to 120s failsafe and explicitly checks `_isEnteringRoom || _isCreatingRoom || _isStartingSoloMatch` before closing.
  2. **Unified Entry State Management (`static/js/private_matches.js`, `static/js/lobby.js`, `templates/index.html`)**:
     - `startSoloMatch` sets `window._isEnteringRoom = true`, `window._isStartingSoloMatch = true`, `window._isEnteringRoomTime = Date.now()`, and triggers `showLoadingOverlay('Loading...', 120000)`.
     - Extended all room button double-click cooldown checks across `enterLobbyRoom`, `enterLobbyRoomInline`, `createRoom`, `handleJoinRoomInline`, and `startSoloMatch` from 10s to 90s.
     - On genuine error or rejection (account timed out, room full, invalid config), `hideLoadingOverlay(true)` is invoked with `force=true` and error alerts are displayed.
  3. **Authoritative Room Render Dismissal (`static/js/play.js`)**:
     - The overlay is dismissed (`window.hideLoadingOverlay(true)`) and room entering flags are reset (`_isEnteringRoom = false; _isCreatingRoom = false; _isStartingSoloMatch = false;`) only when the room view actually renders in `updateGameState`, `renderBoard`, `renderSplitNotepads`, or `renderFCFSNotepads`.
  4. **Cache Busters Incremented (`templates/index.html`)**:
     - Incremented `app.js`, `lobby.js`, `play.js`, and `private_matches.js` cache busters to `v=1795390025`.
- **Commit History**: `96fe6e48`.

---

### AF. Lock Starvation & Timeout Prevention for 6x8 Accumulative & Empty Singleton Rooms
- **User Mandate & Problem**:
  - "View load.jpg in Downloads. The "Loading..." message stayed on the scren the whole time, but this is the message I saw when trying to enter ACC 6x8 45s room. Nobody was in the room."
  - Image inspection of `load.jpg` showed an error toast: `"Joining room took longer than expected. Please try again."`
- **Root Cause Analysis**:
  1. **Premature Client-Side Abort Timers (`static/js/lobby.js`)**:
     - `enterLobbyRoom` used a 45-second abort controller (`createTimer = setTimeout(() => createCtrl.abort(), 45000)`), and `joinRoom` used a 25-second abort controller. When server-side processing exceeded 45s, `fetch` threw an `AbortError`, triggering the toast.
  2. **Blocking Solver Loop Under Global Lock (`game_room.py`)**:
     - When waking up an idle singleton room (`create_room`), `game_room.py` held `with self.lock:`.
     - Non-subanagram singleton rooms did not check if `next_round_board` was already staged from background pregeneration.
     - `pop_compatible_cached_board` and `pop_any_cached_board` strictly required an exact match on `min_word_length` (e.g. `min_word_length: 6`), whereas pregenerated 6x8 boards in SQLite had `min_word_length: 7`.
     - When cache pop failed, the server fell into `get_emergency_fallback_board`, which ran 6 synchronous dense solves with `timeout=10.0s` each (up to 60 seconds of CPU-bound work) *while holding `self.lock`*.
     - This starved the Waitress queue (queue depth jumped to 29), blocked all incoming requests, and caused the client-side 45s timer to abort.
- **Remediation**:
  1. **Instant Staged Board Promotion (`game_room.py`)**:
     - In `create_room`, non-subanagram singleton rooms now check if `next_round_board` and `next_round_words` are already staged before attempting any cache query or solver. If present, it promotes them immediately in <1ms.
  2. **Multi-Stage Cache Relaxation (`board_generator.py`)**:
     - Upgraded `pop_any_cached_board` to a 5-tier relaxation system:
       - Tier 0: exact match (dimensions, min_len, dict, aw, word_count_range, difficulty).
       - Tier 1: relax word count range and difficulty.
       - Tier 2: relax target format.
       - Tier 3: relax minimum word length (any min length for that dimension).
       - Tier 4: relax dictionary & added words (any valid board for that dimension).
     - As long as any pregenerated board exists in SQLite for the dimensions, it is popped and delivered in <1ms, and the room's spinners and rules adapt to that board's parameters.
  3. **Non-Blocking Emergency Fallback Safeguard (`game_room.py`)**:
     - In `get_emergency_fallback_board`, reduced the dense retry loop from `range(6)` to `range(1)` and solve timeout from `10.0s` to `2.0s`. If the dense solve yields playable words, it accepts them immediately instead of spinning for 60 seconds.
     - Added tiered `pop_any_cached_board` fallbacks in `start_next_round` and room kickstart paths.
  4. **Harmonized Client Abort Timers (`static/js/lobby.js`)**:
     - Extended `createTimer` and `joinTimer` timeouts from 45s/25s to 120s (`120000ms`), aligning with the 120s overlay failsafe.
  5. **Cache Busters Incremented (`templates/index.html`)**:
     - Bumped `lobby.js` and `play.js` cache busters to `v=1795390026`.
- **Commit History**: `96fe6e48`.

---

## 2. Infrastructure, Server Health & Resource Usage

| Node / Environment | Host / Port | Git Revision | Status | Response Latency |
| :--- | :--- | :--- | :--- | :--- |
| **GoDaddy Production VPS** | `132.148.72.249:443` (`morpheme.games`) | `96fe6e48` | **Healthy** (PM2 Process 0) | ~60ms |
| **GitHub Remote** | `gewurztraminerrr-cloud/Morpheme:main` | `96fe6e48` | **Synchronized** | Verified |
| **Localhost Environment** | `127.0.0.1:5001` | `96fe6e48` | **Healthy** (Waitress Daemon) | <5ms |

### Key System Metrics Verified
- **Production HTTP Code**: `200 OK`
- **Batch Definition API Throughput**: 4 words in 441ms (0.11s/word, previously timing out after 60s)
- **Watchdog Timeout**: 5000ms max for opening screen
- **Waitress Worker Threads**: 64 healthy threads with zero starvation
