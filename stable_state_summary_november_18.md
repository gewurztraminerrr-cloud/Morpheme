# Stable State Summary — November 18, 2026

## 1. Overview of Changes

### Bonus Word Length Accuracy (10L vs 6L), Global Deduplication, and Transition Scoping Fix
- **User Problem Report**:
  - "The Spinner Set the Bonus Word was 10L, but the bonus word was GELATE (6L). GELATE was on the board. This is the second time in a row i have seen GELATE as bonus word."
  - Log audit also revealed `UnboundLocalError: cannot access local variable 'calculate_word_score' where it is not associated with a value` in `game_room.py:2332`.
- **Root Cause**:
  1. *Bonus Word Length Discrepancy*:
     - During 0:45 intermission, the Spinner Set revealed `bonus_word_length = 10`.
     - In `board_generator.py:_ensure_exact_bonus_word`, if embedding a 10L word onto the board failed on its first attempt, it fell back to:
       `suitable.sort(key=lambda w: (abs(len(w) - req_len), -len(w)))` which returned the closest word found on the board (`GELATE`, 6 letters).
     - In `game_room.py:pick_distinct_bonus_word`, lines 3168–3176 checked `valid_range = [w for w in words_list if 6 <= len(w) <= 10...]` *before* attempting to generate or embed a 10L bonus word, immediately latching onto 6L `GELATE`.
     - Lines 5673–5678 and 8210–8216 then mutated `room.spinner_params['bonus_word_length'] = len(room.bonus_word)` (6), causing a discrepancy between the 10L promised on the Spinner Set and the 6L bonus word on the board.
     - Fallback board generators (`get_emergency_fallback_board`, `pop_compatible_cached_board`) did not strictly enforce `bonus_word_len`, allowing cached boards with `bonus_word_len == 0` or missing bonus words to match.
  2. *Bonus Word Repeating Twice in a Row*:
     - Deduplication was tracked only locally inside individual room instances (`room.recent_bonus_words` vs `room.bonus_word_history`), which were separate lists that were not synchronized.
     - When a new room was created or kickstarted (e.g., when the user transitioned rooms or after a server restart), both lists on the new room were completely empty, allowing recent words like `GELATE` to be chosen again immediately.
     - Server lacked a persistent global bonus word history across rooms.
  3. *UnboundLocalError on `calculate_word_score`*:
     - In `game_room.py:2320`, `from scoring import calculate_word_score` was indented inside `if getattr(self, 'game_type', None) == 'subanagrams':`. Because Python treats imported names as local for the entire function scope, the `else:` branch threw `UnboundLocalError` when calling `calculate_word_score`.
- **Fixes Applied**:
  - **`game_room.py`**:
    - **Global Rolling History & Synchronization**:
      - Added thread-safe global registry `_GLOBAL_RECENT_BONUS_WORDS` (last 50 words) initialized automatically from SQLite `round_history` on startup.
      - In `_record_room_bonus_word`, synchronized updates to `room.recent_bonus_words`, `room.bonus_word_history`, and `_record_global_bonus_word`.
      - In `RoomManager._get_bonus_word`, included `_GLOBAL_RECENT_BONUS_WORDS[-30:]` in `final_exclude` so recent bonus words are never repeated across rooms.
    - **Strict Bonus Word Length Enforcement**:
      - In `pick_distinct_bonus_word`, strictly prioritized exact target length (`valid_exact`).
      - If no exact target-length word is currently on the board, calls `rm._get_bonus_word(length=target_len)` and embeds the word directly onto `room.board`, re-solving all word paths so the promised length bonus word is guaranteed playable on the grid.
      - Prioritized `frozen_revealed_params.get('bonus_word_length')` when determining `bw_l` at round start and promotion, ensuring the bonus word strictly matches the revealed Spinner Set length.
    - **Fallback Board Length Compliance**:
      - Added `bonus_word_len` parameter to `get_emergency_fallback_board` and passed it through all fallback pop/generation calls.
    - **Transition Scoping Fix**:
      - Moved `from scoring import calculate_word_score` in `check_and_update_state` before the `if/else` block, eliminating `UnboundLocalError`.
  - **`board_generator.py`**:
    - In `_ensure_exact_bonus_word`:
      - Added `exclude` parameter to prevent selecting recently used words.
      - When embedding an exact length word, iterates across up to 15 candidates with randomized start positions to guarantee embedding succeeds.
      - In the fallback, prioritizes exact target-length words and filters all candidates against `exclude_set`.
    - In `pop_compatible_cached_board`:
      - Enforced strict `cached_bw_l == int(bonus_word_len)` check when `bonus_word_len` is provided.
- **Verification**:
  - Automated test suite verified:
    - Target length 8 returns 8L words.
    - Target length 10 returns 10L words.
    - Consecutive rounds in the same room produce 0 repeats.
    - Consecutive rounds across newly created rooms produce 0 repeats via global registry.
  - Production deployed on VPS `132.148.72.249`, PM2 restarted, Waitress port 5001 returning HTTP 200 OK.
  - Git commit `c8de5f5f` and tags `stable-2026-11-18` and `live-tested-2026-11-18` pushed to `origin/main`.

---

### Added Words Word Count Range (>= 300 Only) & Bonus Word Repetition Prevention
- **User Problem Report**:
  Reviewing the Spinner Set for an hour in public rooms revealed two distinct issues in `justthisone1.jpg` and `justthisone2.jpg`:
  1. The round showed `Dict: CSW + AW`, but `Words: 100-200`. Rule: `"NWL + AW"` and `"CSW + AW"` word lists should **NEVER** have a word count below 300. The only valid options for them are **`300-400`**, **`400-500`**, and **`500+`** words. Nothing else.
  2. The bonus word `SALTCELLAR` appeared twice in a row across consecutive rounds in the room.
- **Root Cause**:
  1. *Word Count Range Below 300*:
     - In `spinner_set.py:586`, `get_factchecked_wc_range(count, use_added_words, dictionary)` took `use_added_words` and `dictionary` as parameters but completely ignored them when bucketing `count`, returning `'100-200'` whenever `count < 200` (e.g., 130 words).
     - In `spinner_set.py:245`, `sanitize_params` had an early return for `_exact_wc_calculated` that bypassed the `is_aw_effective` clamp, allowing ranges like `'100-200'` to persist.
     - In `board_generator.py:2721`, when Equality Freq hit attempt or timeout limits, it returned `best_equality_board` even if `dist > 0` and word count was only 130 words.
     - In `game_room.py:3518`, the emergency fallback board solved a random grid (yielding 130 words on a 4x6 board) and did not perform a rescue sweep to guarantee $\ge 300$ words for `+ AW`.
     - In `static/js/play.js:6683`, the contradiction guard re-bucketed contradictory boards into `'100-200'` without checking whether Added Words (`useAW`) was active.
  2. *Bonus Word Repeating Twice in a Row (`SALTCELLAR`)*:
     - When `start_next_round` promoted a board (from cache, staging, or emergency fallback), lines 8108–8135 checked `current_bw` against `target_words`, but **never** excluded `ghost_bonus` (the immediately preceding round's bonus word) or `room.previous_bonus_word`.
     - In `pre_generate_next_round` (line 7617), `_fbw` was accepted without checking if it matched `room.bonus_word`.
     - In `force_instant_board` and `existing_room` creation, `previous_bonus_word` was prematurely assigned to the *new* bonus word (`room.previous_bonus_word = room.bonus_word`), wiping out the record of what word actually preceded it.
- **Fixes Applied**:
  - **`spinner_set.py`**:
    - Updated `get_factchecked_wc_range` to enforce that if `use_added_words` is `True` or dictionary contains `+ AW` / `+AW` / `AW` / `ADDED`, the return value is **strictly** one of `'300-400'`, `'400-500'`, or `'500+'` (never `'50-100'`, `'100-200'`, or `'200-300'`).
    - In `sanitize_params`, guaranteed that any parameter set with `is_aw_effective` is strictly restricted to `['300-400', '400-500', '500+']`, even if `_exact_wc_calculated` or `_lock_word_count_range` is present.
  - **`board_generator.py`**:
    - In `generate_board` Equality Freq exit block, if `is_any_aw` is active and `count < 300`, automatically triggers `_perform_rescue_sweep` to guarantee the board achieves $\ge 300$ words before distribution.
  - **`game_room.py`**:
    - Created robust helper `pick_distinct_bonus_word` and `_record_room_bonus_word`:
      - Maintains a rolling history of the last 10 bonus words per room (`room.recent_bonus_words`).
      - Strictly excludes `room.previous_bonus_word`, `ghost_bonus`, `room.bonus_word`, and `room.recent_bonus_words`.
      - Excludes words ending in `ING` or `INGS`.
      - Automatically falls back to closest length candidates or dictionary lookup if the preferred candidate is in the exclusion set.
    - Integrated `pick_distinct_bonus_word` across all bonus word selection and promotion sites:
      - `start_next_round`
      - `pre_generate_next_round`
      - `start_round`
      - `force_instant_board`
      - `get_emergency_fallback_board`
    - In `get_emergency_fallback_board`, added rescue sweep safeguard when `is_aw` is True and words $< 300$, ensuring even random grid fallbacks reach $\ge 300$ words.
    - In `start_next_round`, added clamp to ensure `final_wc` for `_use_aw` rooms is never below 300.
  - **`static/js/play.js`**:
    - In `updateParameters`, guarded `rawWordRange` so that when `useAW` is true, the range is clamped to `['300-400', '400-500', '500+']`.
    - In the contradiction auto-alignment block, if `useAW` is true, mapped word counts strictly to `'500+'` (if $\ge 500$), `'400-500'` (if $\ge 400$), or `'300-400'`.
  - **`templates/index.html`**:
    - Bumped script cache versions to `?v=1794380008`.
  - **Verification**:
    - Unit tests executed and verified:
      - All count brackets (50 to 600) with `NWL + AW` and `CSW + AW` return only `'300-400'`, `'400-500'`, or `'500+'`.
      - 20-round consecutive simulation of bonus words ran with 0 back-to-back repeats.
    - Production deployed on VPS `132.148.72.249`, HTTP/2 200 OK verified.
    - Git commit `5d1938e4`, tags `stable-2026-11-18` and `live-tested-2026-11-18` pushed to origin.

---

### Word Charts Mobile Scrolling & Navigation (Vertical Scrolling, Horizontal Scroll & Swipe-Back Guard)
- **Problem**:
  1. On mobile devices, in Word Charts in Tools under "Point Values & Divisions" and "Dictionary Word Counts", dragging a finger vertically over the tables failed to scroll the page up or down (vertical scrolling was frozen).
  2. When scrolling horizontally towards the leftmost content, dragging past the leftmost boundary (`scrollLeft = 0`) accidentally triggered the mobile swipe-back gesture and kicked the user back to the list of tabs in Tools.
  3. The sidebar tab title was "Word Chart" (singular) instead of "Word Charts", and the description under the tab wrapped across multiple lines.
- **Root Cause**:
  1. *Frozen Vertical Scrolling*: The table containers in `templates/index.html` were assigned class `horizontal-scroll-container` (`overflow-y: hidden`), and `play.css:24195` had `overflow-y: hidden !important;` and `overscroll-behavior: contain !important;`. In CSS, `overscroll-behavior: contain` sets both horizontal and vertical overscroll containment; because the table container has no vertical scroll range, dragging vertically immediately terminated at the container boundary and prevented scrolling from bubbling up to `.tools-content` or the page. In addition, `attachHorizontalScrollChainingPrevention` attached an active non-passive touch listener that intercepted gestures.
  2. *Swipe-Back Kicking Out*: Mobile touch gesture handling on `.tools-content` watched for rightward swipes (`diffX > 80`) without checking if the touch originated inside `#tool-word-chart` or its table containers.
- **Fixes Applied**:
  - **CSS Styling (`static/css/play.css` & `templates/index.html`)**:
    - Removed `horizontal-scroll-container` from `.word-chart-table-container`.
    - Configured `#tool-word-chart .word-chart-table-container` with:
      `overflow-x: auto !important; overflow-y: visible !important; -webkit-overflow-scrolling: touch !important; touch-action: pan-x pan-y !important; overscroll-behavior-x: contain !important; overscroll-behavior-y: auto !important;`.
    - This allows native vertical scrolling gestures to pass unhindered directly to `.tools-content` while strictly containing horizontal scrolling within the table container.
  - **JavaScript Gesture Cleanliness (`static/js/tools.js`)**:
    - Removed `attachHorizontalScrollChainingPrevention` from `#tool-word-chart .word-chart-table-container` to allow 100% native 120Hz browser scroll momentum.
    - Updated mobile touch swipe handling in `.tools-content` to check `.closest('.horizontal-scroll-container, .word-chart-table-container, #subtab-pane-wc-points, #subtab-pane-wc-dicts, #tool-word-chart, [data-prevent-swipe-back="true"]')` on both `touchstart` and `touchend`, completely preventing unintended swipe-backs to the tools list from Word Charts.
  - **Title & Description Rewording (`templates/index.html`)**:
    - Updated sidebar tab title to **Word Charts**.
    - Shortened description to a single line: `Point values & dictionary counts`.
    - Updated pane header to `Word Charts` with single-line description: `Point values, dynamic divisions, and live dictionary counts.`.
    - Aligned FAQ references to refer to **Word Charts**.

---

### Triple Format Music Continuity Fix (Plays Uninterrupted from 0:10 Intermission through Active Round)
- **Problem**: When Triple Format was upcoming, the background music correctly started playing at `0:10` time during intermission. However, as soon as the timer reached `0:00` and the new round began, the music stopped and replayed from the very beginning (`0:00`) again.
- **Root Cause**:
  1. *State Update Ordering Race Condition*: When the intermission countdown reached `0:00`, `_doUpdateGameState` received the new active state from the server. It called `syncTimerWithServer` (which set `localEndTime` to `now + 45s`) and immediately invoked `updateLocalTimer()`. However, `window.lastGameState = state` was positioned ~75 lines *below* `updateLocalTimer()`. 
  2. Because `window.lastGameState` was still the old state (`state === 'intermission'`) while `remaining` was `45s`, `updateTripleMusicState` evaluated `remainingSec <= 50.0 && remainingSec > 10.0` inside the `isIntermission` block, triggering an immediate `audio.pause()` and `audio.currentTime = 0`.
  3. A few milliseconds later, when `window.lastGameState` was finally updated to `state === 'active'`, `safePlay()` was called, causing the audio track to restart from `0:00`.
  4. In addition, if a room momentarily touched transition states (`'waiting'`, `'loading'`), `updateTripleMusicState` defaulted to pausing and resetting `currentTime = 0`.
- **Fixes Applied (`static/js/play.js`)**:
  - **Immediate State Synchronization**:
    - Moved `window.lastGameState = state` and `const previousState = window.lastGameState` up to execute *immediately* after transposing the board state, before `syncTimerWithServer` or `updateLocalTimer` run.
  - **Triple Music Session Tracking (`window._tripleMusicSessionActive`)**:
    - Added session tracking flag `window._tripleMusicSessionActive`.
    - When `remainingSec <= 10.0` during intermission for an upcoming Triple round, `window._tripleMusicSessionActive` is set to `true`, `audio.currentTime = 0` is set only once, and playback starts.
    - Across the `0:00` boundary, transition states (`waiting`, `loading`), and into the active round, the music continues playing continuously without interruption. `audio.currentTime` is strictly preserved and never reset to 0.
    - During the active round, if audio was ever suspended by browser policy, `safePlay()` resumes seamlessly from the current playhead without resetting.
    - When the round ends, the music fades out during the first 10 seconds of intermission (`0:60` to `0:50`), and is only paused/reset once volume reaches 0 at `0:50`.
  - **Audio Unlock Guard**:
    - In `unlockAudio()`, guarded `tripleMusic.load()` so it only loads if `tripleMusic.paused && tripleMusic.currentTime === 0`, preventing any user click from reloading an already-playing track.
  - **Cleanup on Leave**:
    - In `clearGameUIAndCache()`, reset `window._tripleMusicSessionActive = false`.

---

### Either/Or Intended Word Resolution for Words Below Minimum Length (`SKIRT` vs `SEIRT`)
- **Problem**: When a user traced or typed a word using an Either/Or tile that fell below the minimum word length requirement (e.g., in `eitheror3.jpg`, tracing `S` $\rightarrow$ `E/K` $\rightarrow$ `I` $\rightarrow$ `R` $\rightarrow$ `T` on a 6L minimum board), the error notification incorrectly showed the arbitrary non-word combination: `"SEIRT IS TOO SHORT (MIN: 6L)"` instead of the user's intended valid English word `"SKIRT IS TOO SHORT (MIN: 6L)"`.
- **Root Cause**:
  1. The board's `all_words` dictionary list sent to the client only includes words satisfying the round's minimum length requirement (lengths $\ge$ 6L).
  2. For a 5-letter word like `SKIRT`, neither candidate (`SEIRT` nor `SKIRT`) existed in `preState.all_words`, leaving `validOptions` empty on the client.
  3. The client path resolution defaulted to the first letter on the tile (`E` in `E/K`), yielding `SEIRT`.
  4. The client then intercepted `effectiveLen < minLen` before sending to the server, immediately displaying `"SEIRT IS TOO SHORT (MIN: 6L)"`.
- **Fixes Applied**:
  - **New Dictionary Resolution Endpoint (`POST /api/resolve_either_or` in `app.py`)**:
    - Evaluates all candidate words generated by Either/Or tiles directly against the authoritative dictionary (`word_validator.is_valid_word`, NWL, CSW, AW) regardless of length constraints.
    - Returns all valid dictionary words in order, identifying the real word (`SKIRT`) while rejecting gibberish (`SEIRT`).
  - **Client-Side Candidate Resolution (`static/js/play.js`)**:
    - Added `resolveEitherOrCandidate(possibleWords, preferredWord, dictionary)` with client-side caching (`window._eitherOrWordCache`).
    - When `validOptions` is empty (e.g., words shorter than `min_word_length`), calls `resolveEitherOrCandidate` to detect the intended dictionary word.
    - If a valid dictionary word exists among the candidate options (e.g. `SKIRT`), `word` resolves to that word before checking length.
    - Applied identically across Multiplayer Game Rooms (`submitWord`), Tournament mode (`handleTournamentWord`), and Private Matches (`handlePrivateMatchWord`).
    - Result: Displays `"SKIRT IS TOO SHORT (MIN: 6L)"` accurately reflecting the player's intended word.

---

### QU Either/Or Tile Validation Flash Fix (Only Blue Flash on Valid Entry)
- **Problem**: When using the `QU` part of an Either/Or tile (e.g. `QU / U` as seen in `eitheror1.jpg` when spelling `QUITS` or `QUIT`), the board tiles and word input would initially flash **red** (accompanied by failure audio), and then immediately switch to **blue** ~50ms later when the server response arrived. The entry was valid and went through, but the initial red flash was jarring and incorrect.
- **Root Cause**:
  1. Either/Or tiles with `Q` are stored on the board as `Q/<vowel>` (e.g. `Q/U`). The tile display expands `Q` to show `QU`, but in `submitWord`, `handleTournamentWord`, and `handlePrivateMatchWord`, `finalPath` resolution split `cellVal` into `['Q', 'U']` and generated candidate words without expanding `'Q'` to `'QU'`.
  2. For the word `QUITS`, `possibleWords` was generated as `["QITS", "UITS"]`. Because `"QUITS"` was not in `possibleWords`, `validOptions` was empty, and the client code fell back to `word = possibleWords[0]` (`"QITS"`).
  3. The client-side instant optimistic check tested `"QITS"` against the board's `all_words` dictionary list. Because `"QITS"` was not valid, it instantly triggered `showValidationFeedback("QITS INVALID", false)` — producing the red flash and failure sound.
  4. The server backend (`game_room.py`) *did* expand `'Q'` to `'QU'`, so it accepted `"QUITS"`, awarded 5 points, and returned `success: true`. Upon receiving this, the client triggered a second `showValidationFeedback` call that flashed blue.
- **Fixes Applied**:
  - **Client-Side Either/Or Path Expansion (`static/js/play.js`)**:
    - In `submitWord`, `handleTournamentWord`, and `handlePrivateMatchWord`: when evaluating options from an Either/Or tile (`cellVal.includes('/')`), if an option is `'Q'` or `'QU'`, expand it to both `'QU'` and `'Q'`.
    - Also ensure standard tiles with `'Q'` along the path expand to `'QU'` and `'Q'`.
    - Cleaned brackets and slashes (`word.replace(/\[|\]|\//g, '').toUpperCase()`) so typed or bracket-formatted strings resolve cleanly.
    - Updated `getLetterFromPathIndex` so display formatting outputs `[QU/U]` and mouse-swiped word reconstruction retains `QU`.
    - `possibleWords` now includes `"QUITS"`, which matches `preState.all_words` immediately. The local optimistic check passes instantly with `optimisticColor = 'blue'`, flashing blue once at 0ms. Redundant second flashes are suppressed on server agreement, eliminating the initial red flash completely.
  - **Server-Side Either/Or Resolution Alignment (`game_room.py`)**:
    - Aligned Either/Or candidate generation in `submit_word` to support both `'QU'` and `'Q'`, and sanitized bracket/slash characters from incoming word strings.

---

### Word Chart Tool in Tools (Point Values per Length, Dynamic Divisions & Live Dictionary Counts)
- **Features & Placement**:
  - Placed directly next to the **Player Activity & Stats** tab in the Tools sidebar navigation (`data-tool="word-chart"` right after `data-tool="new-users"`).
  - Built with three dedicated inner subtabs to cleanly organize all information:
    1. **Point Values & Divisions (`#subtab-pane-wc-points`)**: Displays letter lengths (2L through 15L and 16+), base score points, number of rarity divisions, division point value ranges, hard word threshold designations, bonus word premiums, and total bonus word scores. Includes live dynamic division ranges and examples (5L, 6L, 7L, 8L) showing how find counts dictate points and division assignments.
    2. **Dictionary Word Counts (`#subtab-pane-wc-dicts`)**: Displays KPI aggregate cards and a complete per-letter-length breakdown table for NWL, CSW, AW (Added Words), 16+ Supplementary List, and Total Unique Words across all dictionaries.
    3. **Dynamic Word Lookup (`#subtab-pane-wc-lookup`)**: Allows players to enter any word to instantly inspect its dynamic point value, current division status (`Division X of K`), rarity find count from the tally file, difficulty designation (`Standard` vs `Hard Word`), bonus word score, dictionary list tags (NWL, CSW, AW), and definition popup link.
- **Dynamic Tally Division Updates**:
  - Word point values and division assignments dynamically update as the find tally (`word_stats.json` / `word_tally.log`) grows over time, powered by `tally_divisions.py` and automatic `mtime` cache invalidation.
- **FAQ Cleanup**:
  - Removed large tabular data and static point/count tables from the 2 FAQ questions in `templates/index.html` (`#faq-dictionaries` and `#faq-bonus-words`).
  - Replaced them with concise explanations directing players to the new dedicated Word Chart in Tools.
- **Responsive & Theme Support**:
  - Added full styling in `static/css/play.css` with responsive mobile layout and high-contrast support across Dark, Light, and White themes.

---

### Reverse Dictionary Performance Overhaul & 100% Simultaneous Word Delivery
- **Problem**: When searching broad or common concept words like `"plant"` in Reverse Dictionary under Tools, mobile devices crashed with Chrome's `"Aw, Snap!"` out-of-memory white screen. On desktop and laptop computers, the query took 6+ seconds to load, followed by massive DOM stutter and 30–60 second thread freezes that locked user navigation (such as navigating to the Lobby).
- **Root Causes**:
  1. *Catastrophic Hyponym Expansion*: `get_concept_synonyms('plant')` recursively traversed WordNet hyponyms down 6 levels across sister synsets, expanding a single concept to over 5,094 words (every moss, lichen, shrub, weed, plus generic terms like `being`, `actor`, `complex`, `set`).
  2. *Massive Unfiltered Payload*: The linear search loop matched 87,798 definitions, serializing a 12.38 MB JSON payload over HTTP.
  3. *Client DOM Meltdown*: `static/js/tools.js` ran `window.highlightKeywordsInDefinition()` 87,798 times in a single synchronous loop and assigned `tbody.innerHTML = rowsHtml` with ~88,000 table rows (~500,000 DOM elements), blowing past mobile browser heap memory limits and freezing the main thread.
- **Fixes Applied**:
  - **In-Memory Inverted Index (`app.py`)**:
    - Built `_REVERSE_DICT_INDEX` on startup mapping definition word tokens to definition indices in `_REVERSE_DICT_SEARCH_LIST`. Pre-warmed during background server startup in `warm_up_server_resources()`.
    - Set intersection across query token candidate lists drops query search time across 827,000+ definitions from 6.3s to **1.5ms – 3ms**.
  - **Concept Expansion Guardrails (`app.py`)**:
    - Focused `get_concept_synonyms(word)` on direct lemmas, similar adjectives, and immediate depth-1 hyponyms.
    - Added a noise-word exclusion filter (`set`, `actor`, `being`, `complex`, `state`, etc.) and capped concept terms to ~40 high-confidence semantic markers.
  - **100% Full-Match Delivery Without Capping (`app.py`)**:
    - Returns **all** matching words (e.g. all 10,004 matches for `"plant"`) in the initial response.
    - Server-side JSON serialization takes ~17 milliseconds; compressed gzip wire size is only **~290 KB**.
    - Full roundtrip load time over public HTTPS is **< 1.0 second** (0.92s).
  - **Progressive DOM Chunk Rendering (`static/js/tools.js`)**:
    - All 10,000+ matches reside in JavaScript memory for instant filtering.
    - Instead of injecting 10,000 rows into the layout tree simultaneously, the UI progressively renders 80 rows per chunk (`window._reverseDictChunkSize = 80`).
    - Initial table render injects in `< 2ms`, maintaining silky smooth 60fps scrolling.
    - Passive infinite scroll automatically appends subsequent 80-row chunks as the user scrolls down.
    - Letter-length dropdown switches instantly in client memory with 0ms delay.
  - **Result**: Query for `"plant"` delivers all 10,004 words in **0.92s** over public HTTPS. Memory usage on mobile is lightweight, `"Aw, Snap!"` crashes are eliminated, and navigation to the Lobby or other tabs is instantaneous.

---

### Stale Intermission Congratulatory Message & Celebration Overlay Fix
- **Problem**: In game rooms that transitioned into intermission, a congratulatory announcement from a round played hours earlier (`"Congratulations! 1st Place with X pts"`) or board-clearing celebration (`"Congratulations! You found all the words!"`) would still appear in the definitions panel during intermission, even when no players had entered words in the round that just ended.
- **Root Causes**:
  1. *Freshness & Round Matching*: `play.js` checked `latest.round === state.current_round`, but if a room was empty or idled in intermission or was woken up at Round 1, `state.current_round` matched the round number of the winner record from hours ago. There was no check on the age of the record timestamp.
  2. *No-Word/Zero-Score Rounds*: When a round finished where no one submitted words or scored (`max_score == 0`), `game_room.py` skipped updating `self.winners_history`, leaving the old winner record from hours ago as the top entry (`winners_history[0]`).
  3. *Fireworks & Overlay Persistence*: `startDefinitionFireworks()` attached `<h2 class="def-congrats-title">Congratulations!</h2>` to the definitions panel and hid `def-content`, but `stopDefinitionFireworks()` was not called on intermission transition, leaving the canvas/overlay alive across states.
- **Fixes Applied**:
  - **Frontend (`play.js`)**:
    - Added multi-factor validation for `hasActualWinner` during intermission:
      * Strict timestamp freshness: `(Date.now() - latest.timestamp) < 180000` (must have occurred within the last 3 minutes, not hours ago).
      * Active gameplay validation: Verified that at least one player in `state.players` scored points, submitted words, or had valid word counts in the concluded round (`anyoneEnteredWords`).
      * Server agreement: Checked `state.last_round_had_winner`.
    - Enforced that if `hasActualWinner` is false during intermission, `defContent` placeholder (`Select a word to see its definition`) is restored and any stale winner card or congratulatory overlay is wiped clean.
    - Added automatic `stopDefinitionFireworks()` invocation whenever the room transitions to intermission or updates in intermission, and restricted fireworks checks to `state.state === 'active'`.
  - **Backend (`game_room.py` & `app.py`)**:
    - When a round ends with `max_score == 0` (no words entered), `self.winners_history` is purged of any entry for `self.current_round`, and `self.last_round_had_winner = False` is recorded and sent via `get_room_state`.
    - When an empty room wakes up or restarts at Round 1, `last_round_had_winner = False` and any stale Round 1 entries from prior sessions are cleared.

---

### Plural Stripping Fix for Singular Nouns Ending in 'S' (`ECHITES` $\rightarrow$ `Echites` not `Echite`)
- Fixed a widespread issue in `check_is_noun_and_get_wiki` where non-WordNet singular words ending in `s` (such as `ECHITES`, `DIABETES`, `SERIES`, `SPECIES`, `RABIES`, `GENUS`, `LENS`, `WALRUS`, etc.) had the trailing `s` blindly stripped off into invalid non-words (`Echite`, `Diabete`, `Sery`, `Specy`, etc.).
- Restricted heuristic plural suffix stripping in `check_is_noun_and_get_wiki` strictly to cases where the word's definition indicates a plural (`is_plural_def`), and only when the candidate base word exists in `DEFINITIONS_CACHE`.
- Added explicit overrides for `ECHITES` and `ECHITESES` in `WIKI_COMMON_NOUN_OVERRIDES` to point cleanly to `Echites` (`https://en.wikipedia.org/wiki/Echites`).

---

### Definition & Wikipedia Article Overhaul for `RAMPION`
- Overhauled the generic/truncated definition (`"a European plant"`) across `Definitions.txt`, `wikdefs.txt`, `wikdefs_duplicate.txt`, and the SQLite `wiktionary_definitions` table in `morpheme.db`:
  - `RAMPION - a European biennial bellflower (Campanula rapunculus) cultivated for its edible tuberous roots and salad leaves [n -S]`
  - `RAMPIONS - RAMPION, a European biennial bellflower (Campanula rapunculus) cultivated for its edible tuberous roots and salad leaves [n]`
- Added explicit Wikipedia mapping for `RAMPION` and `RAMPIONS` to `https://en.wikipedia.org/wiki/Campanula_rapunculus`.
- Updated `get_wikipedia_summary()` to support custom target slugs from override URLs so Wikipedia loads the detailed botanical article and high-resolution thumbnail of *Campanula rapunculus* while retaining the button title `Rampion`.
- Added botanical set-index page detection (`index of plants`, `set index`, `list of plants with the same common name`) to automatically bypass messy Wikipedia disambiguation lists.

---

### Wikipedia Popup Variant Resolution & Disambiguation Resolver
- **Variant Spellings to Canonical Articles (`ANDESYTE` $\rightarrow$ `Andesite`)**:
  - Fixed the issue where `ANDESYTE` on Wikipedia returned 404 because Wikipedia only has `Andesite`.
  - Added explicit mapping in `WIKI_COMMON_NOUN_OVERRIDES` and generic automatic fallback in `/api/wikipedia_summary` to resolve any variant spelling pointing to a target word (e.g. `also ANDESITE`) to the canonical Wikipedia article.
- **Disambiguation Page Resolution (`CORSELET` $\rightarrow$ `Corslet`)**:
  - Resolved the issue where `CORSELET` returned Wikipedia's disambiguation stub ("Corselet may refer to: Corselet, a piece of armor"), whereas `CORSLET` showed the armor description and image.
  - Added an automatic disambiguation bullet resolver in `/api/wikipedia_summary` that parses Wikipedia bullet links and matches definition keywords (e.g. `armor`/`armour`), automatically loading the target article (`Corslet`) with its complete description and image.
  - Explicitly mapped `CORSELET`, `CORSELETS`, `CORSLET`, `CORSLETS` to `Corslet` in `WIKI_COMMON_NOUN_OVERRIDES`.
- **Symmetrical Noun Detection & Taxonomy Expansion (`FRIGATEBIRD` / `MESITE`)**:
  - Resolved the discrepancy in **Is Valid in Tools** where `FRIGATEBIRDS` and `MESITES` had blue Wikipedia links while their singular forms `FRIGATEBIRD` and `MESITE` did not.
  - Root cause: WordNet only had `frigate_bird` (with underscore), and `_NOUN_DEF_REGEX` missed definitions starting with `Any of five species of bird...` / `Any of three species of bird...`. Plurals matched `\bplural of\b` and thus were recognized as nouns while singulars were not.
  - Expanded `_NOUN_DEF_REGEX` to match `Any of [N] species of bird...` and taxonomy genus/family markers.
  - Added symmetrical noun propagation: whenever a plural is `plural of X`, $X$ is guaranteed to be a noun; and whenever a singular is a noun, its plural is guaranteed to be a noun.
  - Tagged `[n -S]` on `FRIGATEBIRD` and `MESITE` in `Definitions.txt`.
- **Mutual Synonym Cross-Referencing & Article Synchronization (`PAICHE`, `ARAPAIMA`, `PIRARUCU`)**:
  - Synchronized definitions and Wikipedia popups for `PAICHE`, `ARAPAIMA`, and `PIRARUCU`, which refer to the identical freshwater Amazon fish (*Arapaima gigas*).
  - Updated definitions in `Definitions.txt`, `wikdefs.txt`, `wikdefs_duplicate.txt`, and database table `wiktionary_definitions` so each word mutually cross-references all synonyms:
    - `PAICHE - (South America) a large freshwater food fish of the Amazon, aka arapaima, pirarucu (Arapaima gigas) [n -S]`
    - `ARAPAIMA - (Tupi) a large freshwater food fish of South America, aka pirarucu, paiche (Arapaima gigas) [n -S]`
    - `PIRARUCU - (Tupi) a large South American fish, aka arapaima, paiche (Arapaima gigas) [n -S]`
  - Mapped `PAICHE`, `PAICHES`, `ARAPAIMA`, `ARAPAIMAS`, `PIRARUCU`, `PIRARUCUS` to `Arapaima` on Wikipedia so opening Wikipedia on any of the three displays the exact same article with image.
  - Updated `_CROSS_REF_POINTER_REGEX` to handle `aka`, `a.k.a.`, and optional leading articles (`the`, `a`, `an`), allowing definition head patterns (e.g. `An arapaima, also called...`) to resolve across all similar words.

---

### Forum Thread Maintenance & Configurable Keep Limit in Mods Tab
- Added a dedicated **Forum Maintenance** section under the **Mods** tab in the top menu (`#page-mods`, pane `#mod-tab-forum-maintenance`).
- Allows moderators to set a custom number of threads to keep alive per category (defaults to `300`, freely customizable via number input).
- Displays a live category breakdown showing total threads, preserved count, and excess threads to prune with status indicators.
- Prunes excess threads ordered by pinned status, last reply timestamp, and ID, with cascading deletion across comments, bookmarks, reactions, and responders.
- Endpoints: `GET /api/mods/forum/threads_status?limit=[N]` and `POST /api/mods/forum/prune_threads`.

---

### AFFILE & AFFILED Dictionary Validation & Definition Cleanup
- Added root verb `AFFILE` and third-person form `AFFILES` to the Added Words dictionary (`added_words.txt`, `added_words_duplicate.txt`, `added_words_dates.txt`, `word_stats.json`).
- Definition overhaul for `AFFILE`, `AFFILED`, `AFFILES`, and `AFFILING` across `Definitions.txt`, `wikdefs.txt`, `wikdefs_duplicate.txt`, and the SQLite `wiktionary_definitions` table in `morpheme.db`.

---

### CSS FOUC / Primitive Unstyled Page Degradation Prevention
- **Problem**: In rare instances on cold start or network jitter, page elements degraded to an unstyled, primitive layout of default HTML controls (white boxes, lost positioning and styling).
- **Root Cause**:
  1. A CSS brace imbalance/orphan syntax error at line 21238 of `static/css/play.css` caused browser CSS parsers to discard downstream style blocks when parsing in error-recovery mode.
  2. The primary stylesheet was loaded without preload hints or failure recovery.
- **Fixes Applied**:
  - Removed the syntax glitch in `static/css/play.css`. Verified perfect brace balance (`diff=0`).
  - Added `<link rel="preload" href="/css/play.css?v=1794380000" as="style">` in `templates/index.html`.
  - Added `_handleStyleRetry(el)` auto-retry handlers attached with `onerror` on all external stylesheets in `templates/index.html` to automatically reload if a transient network error occurs.

---

### Lobby Music Static & Double-Start / Playback Hiccup
- **Problem**: When pressing "ENTER LOBBY" on mobile, laptop, and desktop devices, significant static accompanied the music, and the track hiccuped and began twice.
- **Root Cause**:
  1. Both an inline script in `templates/index.html` and `app.js` initiated audio independently. The inline script started early Web Audio buffer decoding, while `app.js` loaded 1s later and restarted playback from `0:00`.
  2. Seven redundant user-gesture listeners fired simultaneously on `document`, `window`, and button elements, spawning overlapping audio sources out-of-phase that produced severe comb-filtering static.
- **Fixes Applied**:
  - Unified playback under a single `LobbyMusicEngine` stream in `static/js/app.js` using `#lobby-music`.
  - Replaced duplicate inline Web Audio buffer decoding and the 7-event capture barrage in `templates/index.html` with a single call to `window.lobbyMusicEngine.play()`.
  - In `LobbyMusicEngine.play()`, if the audio is already playing (`!audio.paused && audio.currentTime > 0`), it preserves continuous playback without restarting from `0:00`.

---

### Ghost Intermission Bell on Entering Lobby or Browsing
- **Problem**: Out of nowhere while browsing or immediately upon pressing "ENTER LOBBY", the intermission bell (sound 10 seconds before next round) played.
- **Root Cause**:
  1. `static/js/play.js:466` called `window.intermissionBellAudio.play()` inside `unlockAudio()` to "warm up" the audio element on the very first user interaction. Unmuting inside the promise handler leaked audible frames of `bell1.wav` into the Lobby.
  2. `playIntermissionBell` and `armIntermissionVibrationSchedule` lacked page boundary guards, allowing pre-scheduled intermission timers from previous sessions to ring in the Lobby or Tools.
- **Fixes Applied**:
  - Removed `window.intermissionBellAudio.play()` entirely from `unlockAudio()` in `static/js/play.js`.
  - Added guards `if (typeof isOnPlayPage === 'function' && !isOnPlayPage()) return;` and `if (window.currentPageId && window.currentPageId !== 'page-play') return;` to `playIntermissionBell` and `armIntermissionVibrationSchedule`.
  - Explicitly cleared `window._intermissionVibrationTimer` in `leaveCurrentRoom()`, `stopPolling()`, and `ejectToLobby()`.

---

### Room Entry "Loading..." Overlay Cut-Off
- **Problem**: After pressing "ENTER LOBBY", clicking a room button sometimes caused the "Loading…" message to get cut off and fail to enter the room, forcing the user to tap the button a second time.
- **Root Cause**:
  1. Entering the lobby launched an unawaited floating `leaveCurrentRoom()` task in the background. If the user clicked a game room before that fetch finished, `leaveCurrentRoom()` completed *after* `enterLobbyRoom()` started, setting `window._userLeftRoom = true`.
  2. When `/api/room/create` succeeded, `enterLobbyRoom` saw `window._userLeftRoom === true`, aborted navigation to the room, and hid the loading overlay.
- **Fixes Applied**:
  - Replaced floating `leaveCurrentRoom()` calls upon entering the lobby with synchronous state resets (`localStorage.removeItem('last_joined_room')`, `window._userLeftRoom = false;`, `window.currentRoomId = null;`).
  - In `enterLobbyRoom`, `createRoom`, and `setupLobbyEvents` in `static/js/lobby.js`, updated abort checks so room creation only aborts if the user genuinely navigated away from the lobby (`window.currentPageId && window.currentPageId !== 'page-lobby' && window.currentPageId !== 'page-play'`), rather than being blocked by an asynchronous `_userLeftRoom` flag.
  - Ensured `leaveCurrentRoom` does not set `_userLeftRoom = true` if `window._isEnteringRoom || window._isCreatingRoom`.

---

### Mobile Lobby Carousel Twitching on Quick Swipe
- **Problem**: On mobile devices, swiping towards the Solo/With Friends window immediately after pressing "ENTER LOBBY" caused severe screen twitching and button flashing between the Solo and Main panels.
- **Root Cause**:
  1. `showPage('page-lobby')` in `static/js/app.js` and `lobbyObserver` in `static/js/lobby.js` queued a series of rigid `setTimeout` calls (0ms, 50ms, 150ms, 300ms) that blindly executed `window.scrollLobbyToMainPanel()` (`lobbyGrid.scrollLeft = mainPanel.offsetLeft`).
  2. When the user began swiping immediately, the native touch compositor and these JavaScript timers fought back and forth at 60Hz.
- **Fixes Applied**:
  - Introduced `window._userIsTouchingLobbyGrid` and touch-interruptible snap timers (`window._lobbySnapTimers`, `window._clearLobbySnapTimers`).
  - In `static/js/lobby.js`: attached `touchstart`, `touchend`, and `touchcancel` listeners on `.lobby-grid`. On touch, all pending snap timers are cancelled immediately and `_userIsTouchingLobbyGrid` is set to `true`.
  - In `switchLobbyPanel`, `scrollLobbyToMainPanel`, and `_restoreLobbyPanel`: if `_userIsTouchingLobbyGrid` is `true`, programmatic scrolling immediately returns without overriding the user's touch gesture.
  - In `static/js/app.js`: `showPage('page-lobby')` only executes snaps when `!window._userIsTouchingLobbyGrid && window._currentLobbyPanel === 'main'`.

---

### Solo Mode Mid-Round Board Swap Fix
- **Problem**: Shortly after beginning the first round of Solo mode, the board changed ~10 seconds into the round to a completely different board.
- **Root Cause**:
  1. *Kickstart Bypass for Solo Rooms*: In `app.py:16662`, `create_solo_match` passes `is_private=True, is_solo=True` to `room_manager.create_room`. In `game_room.py:4209`, the board kickstart check was `if not is_24h and not is_private:`. Because `is_private=True`, lines 4209–4440 were skipped entirely, bypassing the dedicated solo cache pop/generation at lines 4252–4293.
  2. *Fallback to Background Generation*: As a result, `create_room` returned an empty room without a board (`room.board = []`). In `app.py:16747`, `has_kickstarted_board` evaluated to `False`, so `app.py:16831` set `room.state = 'loading'` and dispatched a background thread running `start_round(room_id)`.
  3. *Premature 10s Rescue*: In `game_room.py:6333`, `check_6x8_rescue(room)` checked `if room.state == 'loading':` with a 10.0-second threshold. At 10 seconds, `check_6x8_rescue` triggered, pulled an emergency fallback board, assigned `room.board = e_board`, and transitioned the room to `state = 'active'`. The user saw this emergency fallback board and began playing Round 1.
  4. *Uncoordinated Background Overwrite*: Meanwhile, the original background thread running `start_round` finished its board generation ~10 seconds later and blindly executed `room.board = board; room.current_round += 1`, overwriting the live gameplay board mid-game without checking if the room had already become active.
  5. *Client Shield Bypassed*: On the client side in `static/js/play.js:2237`, `window.lastGameState = state;` had been placed before line 2957, neutralizing the `IMMUTABLE ACTIVE BOARD SHIELD` because line 2957 compared `window.lastGameState.board` (which was already mutated to the incoming new `state.board`) instead of comparing `previousState.board`.
- **Fixes Applied**:
  - **`game_room.py`**:
    - Line 4209: Changed kickstart check to `if not is_24h and (not is_private or room.is_solo):`. Solo matches now pop a compatible cached board in 0ms (or generate one) immediately during `create_room`.
    - Line 4401: Set `room._initial_board_delivered = True` upon kickstarting.
    - `_apply_kickstart_results`: Added concurrency guard under `with room._state_lock:` to abort and prevent overwriting board if `room.state == 'active'` and board is already populated.
    - `start_round`: Added concurrency guard under `with room._state_lock:` to abort and prevent overwriting board if `room.state == 'active'` and board is already populated.
    - `check_6x8_rescue`: Fixed `start_t` to check `_round_start_init_time` and `creation_time`, and increased loading rescue threshold from `10.0s` to `25.0s`.
  - **`app.py`**:
    - With `room.board` populated by `create_room`, `has_kickstarted_board` is `True`, so `app.py` avoids setting `state = 'loading'` and does not launch a redundant background `start_round` thread.
  - **`static/js/play.js`**:
    - In `IMMUTABLE ACTIVE BOARD SHIELD` (line 2956), updated check to compare incoming `state.board` against `previousState.board` (and preserved `previousState.all_words` and `previousState.all_words_paths`), strictly rejecting mid-round board swaps on the client.

---

### Start Solo Match Instant Loading Notice
- **Problem**: When the user pressed "Start Solo Match", there was no immediate visual feedback, causing an apparent stall or frozen UI while the room creation request resolved over the network.
- **Fixes Applied**:
  - **`templates/index.html`**:
    - Added immediate inline click trigger `onclick="if(window.showLoadingOverlay)window.showLoadingOverlay('Loading...');"` to `#start-solo-btn`.
  - **`static/js/private_matches.js`**:
    - Invoked `window.showLoadingOverlay('Loading...');` synchronously the instant the button is pressed.
    - Added re-entrancy prevention guard (`window._isStartingSoloMatch`) and disabled `#start-solo-btn` with reduced opacity to prevent accidental double-clicks.
    - Added proper error handling to dismiss the loading overlay with `hideLoadingOverlay()` if feasibility checks fail or the server returns an error.
    - Added automatic dismissal when the initial game state arrives and renders in [`play.js`](file:///Users/jeffbabiak/static/js/play.js#L4254).

---

### Spinner Set Accurate Word Count Range Alignment (150 Words vs "300-400" Header Fix)
- **Problem**: In game rooms (such as Accumulative 5x7 with Min 6L and Dict `CSW + AW` shown in `wrongwordcount1.jpg`, `wrongwordcount2.jpg`, and `wrongwordcount3.jpg`), the Spinner Set header showed `Words: 300-400`, but only 150 words were present on the board (`100-200` range).
- **Root Cause**:
  1. *Artificial Added Words Floor in `get_factchecked_wc_range`*: In `spinner_set.py:544-549`, `SpinnerSet.get_factchecked_wc_range` had an artificial clamp: `if is_aw: if count >= 500: return '500+'; if count >= 400: return '400-500'; return '300-400'`. Whenever `CSW + AW`, `NWL + AW`, or `AW` was enabled, any board word count below 400 (even 150 or 75 words on sparse or high-min-length boards) was forcibly mapped to `'300-400'`.
  2. *Sanitization Override in `sanitize_params`*: In `spinner_set.py:253`, `elif is_aw_effective:` was evaluated before `_exact_wc_calculated`, overwriting fact-checked board ranges to `['300-400', '400-500', '500+']`.
  3. *Client-Side Display Fallback*: `static/js/play.js:6544` displayed `state.current_word_count_range` without verifying whether the active board's authoritative word count contradicted the header range string.
- **Fixes Applied**:
  - **`spinner_set.py`**:
    - Removed the artificial Added Words floor from `SpinnerSet.get_factchecked_wc_range`. Word count ranges strictly and truthfully map to their mathematical buckets (`50-100`, `100-200`, `200-300`, `300-400`, `400-500`, `500+`) regardless of dictionary.
    - In `sanitize_params`, moved `p.get('_exact_wc_calculated') or p.get('_lock_word_count_range')` ahead of `is_aw_effective` to ensure solved boards retain their factual word count ranges.
  - **`static/js/play.js`**:
    - Added an active-round Truthful Range Guard in `updateParameters`: if `!isIntermission` and the authoritative word count contradicts the server's range string (e.g. board has 150 words but range says `300-400`), the client immediately re-aligns `factWordRange` to the truthful bucket (`100-200`).

---

### Spinner Set & Remaining Table Truthful Min Word Length Alignment (Opening Round 1 "Min: 5L" with 0 5LWs Fix)
- **Problem**: In opening Round 1 of game rooms (both Solo and Public rooms, e.g. Accumulative 4x6 45s shown in `fives1.jpg` and `fives2.jpg`), the Spinner Set displayed `Min: 5L` and the Remaining tab showed `5LW: 0 + 0 = 0`, even though the shortest word on the board was 6 letters (165 6-letter words). Every subsequent round after Round 1 was correct.
- **Root Cause**:
  1. *Opening Round Cached Board Parameter Drift*: In `create_room` (`game_room.py:4317-4385`), `board_min_l` was read from cached parameters (`r_params.get('min_word_length', 3)`). When a cached board was popped with target `min_word_length = 5`, `create_room` filtered `room.all_words` with `len(w) >= 5`. On this board, no 5-letter words existed (shortest word was 6 letters). `create_room` set `room.current_min_length = board_min_l` (5) and `room.spinner_params['min_word_length'] = board_min_l` (5) without aligning to `min(len(w) for w in room.all_words)`. In contrast, `start_next_round` (lines 8100–8128) and `start_round` (lines 5431–5459) DID align `min_word_length` to `actual_shortest`, which explains why the discrepancy only occurred on Round 1 upon opening a room.
  2. *Secondary Board Assignment Paths*: In `_apply_kickstart_results` (`game_room.py:4488`), `_apply_pregenerated_board` (`game_room.py:2129`), `check_6x8_rescue` (`game_room.py:6496`), and `force_instant_board` (`game_room.py:8496`), minimum length was also assigned from requested parameters without checking `min(len(w) for w in board_words)`.
  3. *Client-Side Remaining Table & Header Fallback*: In `static/js/play.js:3875`, `remaining-words-table` looped starting at `state.current_min_length || 3`. If `current_min_length` was 5 but the shortest word on the board was 6, it printed a redundant `5LW: 0 + 0 = 0 (0%)` row. In addition, `updateParameters` displayed `state.current_min_length` directly without validating against `state.all_words`.
- **Fixes Applied**:
  - **`game_room.py`**:
    - In `create_room`, `_apply_kickstart_results`, `_apply_pregenerated_board`, `check_6x8_rescue`, and `force_instant_board`, added strict alignment: if `actual_shortest = min(len(w) for w in words) > min_word_length`, automatically update `board_min_l = actual_shortest`, `room.current_min_length = actual_shortest`, and `room.spinner_params['min_word_length'] = actual_shortest`.
  - **`static/js/play.js`**:
    - In `remaining-words-table` generation, guarded `minLen` against `actualShortest` from `state.all_words` (or `state.previous_all_words`) so the table never renders empty `0 + 0 = 0` rows for word lengths shorter than the board's shortest word.
    - In `updateParameters`, added an active-round guard so `factMinLen` is clamped to `actualShortest` of `state.all_words`, ensuring 100% truthfulness in the Spinner Set display under all network conditions.

---

### Solo Mode Low Time Timer Text Red Color (`<= 10s`)
- **Problem**: When playing Solo mode, the timer text failed to turn red when the time remaining was $\le$ 10 seconds, in contrast to public rooms where the timer text visibly turns red at $\le$ 10 seconds.
- **Root Cause**:
  1. *Private Match Timer Loop*: In `startPrivateMatchTimer` (`static/js/play.js:14104`), the countdown loop manually updated `timerEl.textContent = ...` without ever setting `timerEl.style.color = '#ef4444'` or invoking `updateSpecialMatchTimer(remaining)`.
  2. *Strict Sub-second Boundary Condition in `updateLocalTimer`*: In `updateLocalTimer` (`static/js/play.js:7487`), the condition was strictly `remaining <= 10 && remaining > 0`. Because the display is driven by `seconds = Math.ceil(remaining)`, when the timer showed `0:10` (e.g. `remaining` was `10.05`), `remaining <= 10` evaluated to `false`, leaving the timer text uncolored until dropping below `10.00`.
  3. *Light & White Theme CSS Specificity Override*: In `static/css/style.css:1660`, `[class*="theme-white"] .timer-value` and `[class*="theme-light-"] .timer-value` declared `color: #000 !important;`. Setting an inline style `style.color = '#ef4444'` without `!important` was completely overridden in light/white layouts.
  4. *Missing Server-Sync Coloring*: `syncTimerWithServer` updated `#timer-value` text without reapplying `#ef4444` when `secs <= 10`.
- **Fixes Applied**:
  - **`static/js/play.js`**:
    - Connected `startPrivateMatchTimer` directly to `updateSpecialMatchTimer(remaining)`, ensuring private match turns and solo sessions use the unified low-time text styling and board panel warning pulse.
    - Updated `updateSpecialMatchTimer`, `updateLocalTimer`, and `syncTimerWithServer` to check `(remaining <= 10 || seconds <= 10) && seconds > 0`.
    - Applied `.low-time-warning` class and `style.setProperty('color', '#ef4444', 'important')` on `#timer-value` during low-time, cleanly clearing them when waiting or when timer $> 10$s.
    - Added timer cleanup in `clearGameUIAndCache` and `setTimerWaitingState`.
  - **`static/css/play.css` & `static/css/style.css`**:
    - Added `.timer-value.low-time-warning` rules with `color: #ef4444 !important;` across all themes and light/white theme classes.
    - Scoped `[class*="theme-white"] .timer-value:not(.low-time-warning)` and `[class*="theme-light-"] .timer-value:not(.low-time-warning)` so the black text override never blocks the red low-time warning.

### Multiplayer Subanagrams Spinner Set Odds & Parameter Accuracy Alignment
- **Problem**: In Multiplayer Subanagrams rooms, the Spinner Set header was inaccurate and did not match the percentages, ranges, and parameter specifications in the Subanagrams Spinner Set Odds modal (`#subanagrams-spinner-set-modal` in `templates/index.html:8875–9070`).
- **Root Cause**:
  1. *Odds & Min Length Drift in `spinner_set.py`*: For 8-letter sequences, `min_word_length` was grouped with 7-letter words (`elif sequence_length in [7, 8]: min_word_length = _rng.choice([3, 4])`), completely omitting 5L even though the Odds window specifies `8, 9, 10 Letters: 3L (33.3%) | 4L (33.3%) | 5L (33.3%)`.
  2. *Artificial Word Count Range Restrictions*: `word_count_range` was artificially restricted by length (e.g. 8 letters only got `100-200`, `200-300`, `300-400`), omitting `50-100` and `500+` even though the Odds window specifies equal 20% probability for each of the 5 ranges (`50-100`, `100-200`, `200-300`, `300-400`, `500+`).
  3. *Premature Fallback in `subanagrams_generator.py`*: `max_attempts` was capped at only 25 attempts; when a candidate sequence within the target range was not found in 25 iterations, the generator silently decremented `min_len` down to 4 or 3 (corrupting the spun min length) and overwrote `word_count_range` with the fallback candidate's range.
  4. *Delayed Staging in `game_room.py`*: During the 0:45 intermission reveal, `reveal_spinner_params` generated `new_params` but did not pre-generate and stage the Subanagrams board, causing parameters to drift or recalculate at 0:00 round start. In addition, `self._get_factchecked_wc_range` mapped counts between 400 and 499 to `'400-500'`, which does not exist in the Subanagrams odds.
  5. *Intermission Client-Side Header Inaccuracy in `play.js`*: During intermission, `play.js:updateParameters` read the previous round's board length before checking `sp.sequence_length`, causing the `Letters:` header to remain stuck showing the previous round's sequence length during the 0:45 reveal animation.
- **Fixes Applied**:
  - **`spinner_set.py`**:
    - Re-architected `SpinnerSet.generate_subanagrams_params` with a doubly stochastic joint distribution matrix ensuring the marginal probabilities across all rounds are mathematically exact:
      - `Sequence Length`: 6 Letters (20%), 7 Letters (20%), 8 Letters (20%), 9 Letters (20%), 10 Letters (20%).
      - `Target Word Count Range`: `50-100` (20%), `100-200` (20%), `200-300` (20%), `300-400` (20%), `500+` (20%).
      - `Min Word Length`: 6 Letters (100% 3L); 7 Letters (50% 3L, 50% 4L); 8, 9, 10 Letters (33.3% 3L, 33.3% 4L, 33.3% 5L).
      - `Dictionary`: CSW (25%), NWL (25%), CSW + AW (25%), NWL + AW (25%).
      - `Format`: Word Guaranteed (50%), Totally Random (50%).
    - Added `SpinnerSet.get_subanagrams_factchecked_wc_range(count)` mapping counts $\ge 400$ to `'500+'`, strictly adhering to the 5 Subanagrams odds brackets.
  - **`subanagrams_generator.py`**:
    - Increased `max_attempts` from 25 to 300, and added targeted passes so valid letter sets matching the target range and format are always discovered without delay ($< 30$ms).
    - Removed arbitrary `min_len` decrementing: the declared `min_word_length` is strictly enforced and guaranteed to have $\ge 50$ valid subanagrams.
    - Guaranteed that in `Word Guaranteed` mode, at least one authentic full sequence length dictionary word exists in the valid subanagrams.
  - **`game_room.py`**:
    - In `reveal_spinner_params`, pre-generate and stage `next_round_board`, `next_round_words`, `next_round_bonus`, and `next_round_spinner_params` immediately at the 0:45 intermission mark, locking parameters so the transition to round start at 0:00 is instantaneous ($< 1$ms) with 0% parameter drift.
    - Connected `start_next_round` to `SpinnerSet.get_subanagrams_factchecked_wc_range` for Subanagrams rooms.
    - Updated room kickstart and creation paths (`create_room`, `_apply_kickstart_results`) to freeze revealed parameters and set `current_word_count_range` on Round 1.
  - **`app.py`**:
    - In `get_room_state`, during intermission when parameters are revealed, updated `calc_seq_len` to read `sp_obj.get('sequence_length')`, ensuring `state.sequence_length` reflects the newly revealed round parameters.
  - **`static/js/play.js`**:
    - In `updateParameters`, updated `factSeqLen` for Subanagrams to use `sp.sequence_length` during the 0:45 intermission reveal, preventing stale sequence length display from the previous board.
    - Guarded `factMinLen` and `rawWordRange` so Subanagrams displays strictly adhere to the 5 odds ranges.

---

## 2. Commit & Deployment Details
- **Latest Commit ID**: `3d6aac38f97e2f5b5c97f8045e0aa06c138d6df8` (short: `3d6aac38`)
- **Previous Commits**: `65c4cd6021fb68afe8b48672488d777f0419461b`, `6c0754b46fc9678b667b0eb6aec05f414b385602`, `a41872a020389edc6b7d5f67c0125c9935c3644e`, `a406cd5e1b61ace30014dfba46920220140fa184`, `658ba7632823cd934e51219a2026bc71a029e65f`
- **Tags Updated & Synchronized**: `start-over`, `latest-stable-state`, `stable-2026-11-18`
- **Pushed to GitHub**: `main` branch and tags pushed to `origin` (`https://github.com/gewurztraminerrr-cloud/Morpheme.git`)
- **Production Server (morpheme.games)**: Deployed and restarted via PM2 on `132.148.72.249` (Port 5001, HTTP 200 and live endpoints verified on `https://morpheme.games`)
- **Synchronization Status**: Localhost, GitHub (`origin/main`), and `morpheme.games` are 100% in sync.

---

## 3. Files Modified
- [`app.py`](file:///Users/jeffbabiak/app.py): 
  - Added `GET /api/tools/word-chart` returning live dictionary word counts by letter length (lengths 2–15, 16+), point values, dynamic rarity division thresholds, dynamic max finds from `word_stats.json`, and dynamic division examples for 5L, 6L, 7L, 8L words.
  - Added `GET /api/tools/word-chart/lookup?word=<WORD>` returning dynamic point values, division designation (`Division X of K`), difficulty designation (`Standard` vs `Hard Word`), rarity find count from tally, bonus word points, dictionary list tags, and definitions.
  - Lifted the 2,500 cap to deliver 100% of all matching words in the initial response.
  - Added inverted index `_REVERSE_DICT_INDEX` and fast candidate token intersection for Reverse Dictionary.
  - Refactored `get_concept_synonyms` to prevent runaway hyponym expansion.
  - Added pre-warming in `warm_up_server_resources` for reverse dictionary indexing and query priming.
  - Restructured Wikipedia noun resolution and proper noun filtering pipeline.
  - Expanded `_NOUN_DEF_REGEX` to recognize bird, animal, and taxonomic definitions starting with `Any of [N] species of...`.
  - Updated `_CROSS_REF_POINTER_REGEX` to support `aka`/`a.k.a.` and ignore articles (`the`, `a`, `an`).
  - Added definition headword recognition (`An arapaima, ...` $\rightarrow$ `arapaima`).
  - Added symmetrical singular/plural noun propagation.
  - Added automatic disambiguation bullet link parsing and keyword resolution in `get_wikipedia_summary()`.
  - Added cross-reference fallback for variant spelling 404s.
  - Added overrides for `ANDESYTE(S)`, `CORSELET(S)`, `FRIGATEBIRD(S)`, `MESITE(S)`, `PAICHE(S)`, `ARAPAIMA(S)`, `PIRARUCU(S)`.
- [`templates/index.html`](file:///Users/jeffbabiak/templates/index.html):
  - Added `Word Chart` button in Tools navigation placed directly next to `Player Activity & Stats` (`new-users`).
  - Added `#tool-word-chart` tool pane with 3 subtabs: `Point Values & Divisions`, `Dictionary Word Counts`, and `Dynamic Word Lookup`.
  - Cleaned up FAQ Questions 1 and 2 (`#faq-dictionaries` and `#faq-bonus-words`) by removing tables and directing players to the dedicated Word Chart tool in Tools.
- [`static/js/tools.js`](file:///Users/jeffbabiak/static/js/tools.js):
  - Added `window.switchWordChartSubtab()`, `window.loadWordChartTool()`, `window.renderWordChartUI()`, `window.renderWordChartPoints()`, `window.renderWordChartDicts()`, `window.renderWordChartExamples()`, and `window.lookupWordChartWord()`.
  - Added Enter key listener on the lookup search input.
  - Progressive DOM chunk rendering (80 rows per chunk) with passive infinite scroll on `reverse-dict-table-scroll`.
  - In-memory letter-length switching (0ms delay) across all returned words.
- [`static/css/play.css`](file:///Users/jeffbabiak/static/css/play.css):
  - Added styling for Word Chart tables, subtabs, search inputs, KPI metric cards, and full light/white theme high-contrast adaptations.
- [`dictionaries/Definitions.txt`](file:///Users/jeffbabiak/dictionaries/Definitions.txt): Updated definitions and noun tags for `PAICHE`, `ARAPAIMA`, `PIRARUCU`, `FRIGATEBIRD`, `MESITE`.
- [`dictionaries/wikdefs.txt`](file:///Users/jeffbabiak/dictionaries/wikdefs.txt): Updated definitions for `PAICHE`, `ARAPAIMA`, `PIRARUCU`, `FRIGATEBIRD`, `MESITE`.
- [`dictionaries/wikdefs_duplicate.txt`](file:///Users/jeffbabiak/dictionaries/wikdefs_duplicate.txt): Synchronized with `wikdefs.txt`.
- [`dictionaries/aw_inflections.txt`](file:///Users/jeffbabiak/dictionaries/aw_inflections.txt): Cached inflection mappings for `FRIGATEBIRD`, `MESITE`, `PAICHE`.
- [`morpheme.db`](file:///Users/jeffbabiak/morpheme.db): Updated `wiktionary_definitions` table on localhost and VPS production database.
- [`static/js/play.js`](file:///Users/jeffbabiak/static/js/play.js):
  - Removed problematic `keepalive: true` from interactive `submitWord` requests, eliminating background connection stalls and aborted fetch promises on mobile browsers.
  - Implemented 3-attempt resilient retry loop with active exception handling and 4.5s per-attempt timeout.
  - Passed `finalPath` to `showValidationFeedback` on submission failure so the specific word's tiles are targeted.
- [`game_room.py`](file:///Users/jeffbabiak/game_room.py):
  - Decoupled Subanagrams board generation and scoring in `generate_spinner_params` from `room._state_lock`, eliminating lock starvation on word submissions.
  - Pre-computed `room.next_round_word_scores` and pre-populated `room.solved_words_with_scores` in `generate_spinner_params`, `ensure_next_board_ready`, and `start_next_round`.
  - Added fast subanagram scoring fallback in `transition_to_intermission` bypassing grid pathfinding.
  - Moved word score dictionary comprehension in `subanagrams_search_worker` outside `room._state_lock`.
  - Extended buzzer-beater grace period in `submit_word` from 1.5s to 3.0s with explicit diagnostics logging.
- [`subanagrams_generator.py`](file:///Users/jeffbabiak/subanagrams_generator.py):
  - Optimized candidate search loop (`max_attempts = 100`, targeted pass `range(50)`) reducing typical generation time from ~880ms to <100ms.

---

### Multiplayer Subanagrams Lock Contention & Word Submission Timeout Fix
- **Problem**: When entering a valid word on a Subanagrams board (e.g. entering `SOCAGE` on a 7L board in `pub_v2_subanagrams_subanagrams_120`), the client-side optimistic check initially succeeded and flashed blue. However, 10–15 seconds later, the tiles flashed red with invalid audio and displayed `"SUBMISSION ERROR"`.
- **Root Cause**:
  1. *State Lock Starvation*: In `generate_spinner_params`, `with room._state_lock:` wrapped `generate_subanagrams_board_and_words`, which spent up to 880–2000ms searching permutations and solving dictionaries while holding the exclusive room lock.
  2. *Missing Pre-Computed Scores*: Subanagram staging omitted populating `room.next_round_word_scores`. When the round ended, `room.solved_words_with_scores` was empty, triggering a synchronous scoring fallback across hundreds of words under `_state_lock` during `transition_to_intermission`.
  3. *Submit Timeout & Retries*: Incoming submissions (`submit_word` in `app.py`) attempted `room._state_lock.acquire(timeout=6.0)`. Because the lock was held across the transition and background generation, submissions timed out (`[SW-TIMING] LOCK FAIL after 6009ms word='SOCAGE'`), returning HTTP 503. The client retried once after 300ms, waited another 6–8s, hit the 8000ms `AbortController` timeout, threw an error, and triggered line 11234: `showValidationFeedback('Submission Error', false)`.
  4. *Buzzer-Beater Grace Period Expiry*: Submissions sent near the buzzer (`tr=0.0`) had their 1.5s grace period expire while blocked waiting for the server lock.
- **Fixes Applied**:
  1. **Decoupled Board Generation Outside Lock (`game_room.py`)**:
     - In `generate_spinner_params`, checked staging under a brief lock (<0.01ms), released the lock completely while running `generate_subanagrams_board_and_words` and pre-computing `calculate_word_score`, and re-acquired the lock for microseconds solely to assign results.
     - In `subanagrams_search_worker`, computed `calculate_word_score` outside `room._state_lock`.
  2. **Authoritative Pre-Computed Word Scores (`game_room.py`)**:
     - Pre-computed `room.next_round_word_scores` across all staging paths (`generate_spinner_params`, `ensure_next_board_ready`, `start_next_round`).
     - Ensured `room.solved_words_with_scores` is immediately populated at round start.
     - In `transition_to_intermission`, added fast dictionary scoring without grid pathfinding if scores are ever missing.
  3. **Extended Buzzer-Beater Grace Period (`game_room.py`)**:
     - Extended grace period from 1.5s to 3.0s (`(now - self.intermission_start_time) <= 3.0`) and added debug logging to guarantee legitimate in-flight submissions near round transitions are accepted and credited.
  4. **Generator Performance Optimization (`subanagrams_generator.py`)**:
     - Tuned candidate search attempts to 100 with 50 targeted passes, reducing generator runtime from ~880ms to <100ms.
- **Verification**:
  - Concurrent simulation test confirmed `_state_lock` acquisition times dropped to **0.00ms – 0.02ms** (down from >6,000ms).
  - 100% of concurrent submissions during background generation succeeded instantly without errors.

---

### Stuck on "Loading..." Screen Resolution (Lobby Navigation & Leave Room Reliability)
- **Problem**: When a user was in Multiplayer Subanagrams, returned to the lobby, and clicked an Accumulative room (`ACC 4x6 45s`), the browser got permanently stuck on the "Loading..." screen for several minutes.
- **Root Cause**:
  1. *Unbounded Leave Request Awaiting*: When navigating from one room to another, `enterLobbyRoom` called `leaveCurrentRoom()`, which awaited `fetch('/api/room/${roomId}/leave')` with `keepalive: true` and no timeout. During moments of server activity (or mobile socket suspension), `leave` hung in the queue for up to 60s until Nginx returned 504.
  2. *Missing Timeout on Room Creation & Joining*: `enterLobbyRoom` then called `fetch('/api/room/create')` with no timeout. If the server was processing or the mobile browser aborted the stalled socket (`HTTP 499`), the request hung.
  3. *Unbounded Loading Overlay*: `showLoadingOverlay('Loading...')` in `app.js` and `index.html` only set an auto-hide timeout if `autoHideMs` was explicitly provided. When called without `autoHideMs`, any hung network promise or duplicate-click rejection trapped the user behind the modal overlay indefinitely.
  4. *Stale Lockout in `_isEnteringRoom`*: If a prior room entrance attempt failed or hung, `window._isEnteringRoom` remained `true`. Subsequent button clicks hit `if (window._isEnteringRoom) return;` without dismissing the loading overlay.
- **Fixes Applied**:
  1. *Failsafe Overlay Auto-Hide (`templates/index.html` & `static/js/app.js`)*: Added a default failsafe auto-hide timeout of 10 seconds (`autoHideMs || 10000`) so the user can never be permanently stuck on a loading screen under any circumstance.
  2. *Non-Blocking Leave Room (`static/js/play.js`)*: Removed `keepalive: true` and added a strict 1.5-second `AbortController` timeout to `leaveCurrentRoom()` so room transitions never stall waiting on `/leave`.
  3. *Timed Room Creation & Joining (`static/js/lobby.js`)*: Added an 8-second `AbortController` timeout to both `/api/room/create` and `/api/room/${roomId}/join` with clear error handling.
  4. *Stale Entrance Lock Guard (`static/js/lobby.js`)*: Added timestamped expiration (10s) to `window._isEnteringRoom` and ensured `hideLoadingOverlay()` is called whenever duplicate clicks are rejected.

---

### Multiplayer Subanagrams Uniform Letter Input Highlight
- **Problem**: In Multiplayer Subanagrams, when inputting letters to form a word, tiles had conflicting highlight colors and brightness levels. Tapping a new tile applied `.current` with `filter: brightness(1.2)` and a brighter background, causing the most recent tile to look mismatched compared to earlier `.selected` tiles in the word. Typing on keyboard applied `.typing-highlight` with a different yellow color (`#ffcc00`) instead of the selection color.
- **Root Cause**:
  1. *Boggle Snake Head Artifact*: In Boggle grid modes, `.board-cell.current` used `filter: brightness(1.2)` to identify the active dragging head. In Subanagrams, there is no spatial grid snake; words are subanagram letter sequences, so brightening the newest tile made the letters of the word look mismatched.
  2. *Typing vs Mouse Discrepancy*: Typing on keyboard applied `.typing-highlight` without syncing `mouseState.selectedPath`, leading to differing colors between typed and tapped tiles.
- **Fixes Applied**:
  1. *CSS Standardization (`static/css/play.css`)*:
     - Enforced `.game-board.is-subanagrams-board .board-cell.selected`, `.current`, and `.typing-highlight` to share the exact same background (`var(--highlight-mouse-color, #409cff)`), border (3px solid #fff), text color, box-shadow, and `filter: none !important`.
     - Completely eliminated brightness filter disparity between tiles in the word.
  2. *Interaction Logic Standardization (`static/js/play.js`)*:
     - In `handleSubanagramsTileClick`, marked all active tiles uniformly as `.selected` without `.current`.
     - In `wordInputEl.addEventListener('input')`, synchronized `mouseState.selectedPath` and applied `.selected` alongside `.typing-highlight` for Subanagrams so typed and clicked tiles are 100% unified in color and path tracking.

---

### Multiplayer Subanagrams Mobile Layout: Shuffle Button Relocated to Timer & Full-Width Word Input
- **Problem**:
  1. On mobile devices in Multiplayer Subanagrams, the Shuffle button sat beside `#word-input` in `.word-input-row`, causing the "ENTER WORD" textbox to take only a fraction of the row's horizontal width while the "Enter" button directly above it stretched full width.
  2. The user requested:
     - Move the Shuffle button in multiplayer subanagrams on mobile devices to the right side of the timer display, occupying the exact same spot where the Rotate and Transpose buttons sit in standard game rooms.
     - Make the ENTER WORD textbox run along the exact same horizontal length as the Enter button above it.
- **Root Cause**:
  - In standard game rooms on mobile, `adjustPlayHeaderForDevice()` relocated `#timer-controls-group` (Rotate & Transpose) to `.timer-display` with `margin-left: auto`. In Subanagrams, Rotate & Transpose are hidden, leaving that spot vacant, while Shuffle was appended to `.word-input-row` next to `#word-input`.
  - Because `#shuffle-board-btn` was inside `.word-input-row`, `#word-input` was restricted to ~60–70% width, falling short of `#submit-word-btn` which spanned 100% of `.word-input-section`.
- **Fixes Applied**:
  1. **Shuffle Button Relocation on Mobile (`static/js/play.js`)**:
     - Updated `adjustPlayHeaderForDevice()` to detect `isSubanagrams` and `!isDesktop`.
     - On mobile in Subanagrams, moved `#shuffle-board-btn` into `.timer-display` (after `.timer-left-group`) with `margin-left: auto`, placing it in the exact same spot and alignment as Rotate & Transpose in standard game rooms.
     - On desktop, restored `#shuffle-board-btn` back to `wordInputRow` (before `clearWordBtn`) with standard desktop styling.
     - Automatically called `adjustPlayHeaderForDevice()` on room state updates, leave room cleanup, and viewport resize.
  2. **Timer Display Shuffle Styling (`static/css/play.css`)**:
     - Styled `.timer-display #shuffle-board-btn` to match the exact dimensions and position of Rotate & Transpose buttons: `height: 26px !important; line-height: 26px !important; padding: 3px 12px !important; font-size: 0.75rem !important; font-weight: 700 !important; border-radius: 6px !important; margin-left: auto !important; width: auto !important;` with sleek vibrant blue gradient.
  3. **Identical Horizontal Length for Word Input (`static/css/play.css`)**:
     - Configured `.word-input-row` on mobile Subanagrams to `width: 100% !important; max-width: 100% !important; margin: 0 !important; padding: 0 !important; box-sizing: border-box !important;`.
     - Configured `#word-input` on mobile Subanagrams to `width: 100% !important; max-width: 100% !important; flex: 1 1 100% !important; box-sizing: border-box !important; margin: 0 !important;`.
     - Ensured `#submit-word-btn` directly above it has matching `width: 100% !important; max-width: 100% !important; box-sizing: border-box !important; margin: 0 !important;`.
     - The ENTER WORD textbox now spans the exact same horizontal width as the Enter button above it.
  4. **Eliminated Duplicate Inline Shadow Function & Cache-Busted Assets (`templates/index.html`, `static/js/app.js`, `static/js/lobby.js`)**:
     - Removed a legacy duplicate definition of `adjustPlayHeaderForDevice()` at the bottom of `templates/index.html` (lines 10526–10632) that was shadowing `play.js`'s updated implementation and executing without the Shuffle relocation logic.
     - Replaced the duplicate script block with clean delegation listeners (`load`, `resize`, `DOMContentLoaded`) invoking `window.adjustPlayHeaderForDevice()`.
     - Bumped cache-busting query strings to `?v=1794380004` across `play.css`, `play.js`, `app.js`, and `lobby.js` so mobile devices (iOS Safari, Android Chrome) immediately fetch the live updated assets rather than serving stale cached scripts.
     - Ensured `adjustPlayHeaderForDevice()` runs reliably during `showPage('page-play')`, `enterLobbyRoom()`, and Subanagrams board rendering.

---

### Gateway "ENTER LOBBY" Button Navigation Resolution
- **Problem**: Clicking the "ENTER LOBBY" button (`#btn-enter-lobby-gateway`) on the gateway page (`#page-loading`) no longer transitioned the user into the lobby.
- **Root Cause**:
  1. *Uncaught SyntaxError in `static/js/lobby.js`*:
     - An inner `try...finally` block introduced to manage the room join `AbortController` prematurely closed the outer `try` statement, leaving a stray `} catch (err) {` at line 684 without an opening `try`.
     - In JavaScript, a `SyntaxError` prevents the entire script file from evaluating when loaded via `<script src="/js/lobby.js">`.
     - Consequently, none of the functions in `lobby.js` (including `fetchLobbyStats`, `startStatsPolling`, and room handlers) were defined on `window`.
  2. *Gateway State Lockout*:
     - When the user clicked "ENTER LOBBY", `executeGatewayTransition()` attempted to invoke `fetchLobbyStats()` and setup lobby state.
     - Because `lobby.js` failed to evaluate, exceptions were thrown or promises stalled, leaving `gatewayTransitioning = true` stuck and preventing subsequent clicks from doing anything.
- **Fixes Applied**:
  1. *Syntax Fix (`static/js/lobby.js`)*:
     - Wrapped lines 553–684 with a clean outer `try { ... } catch (err) { ... } finally { ... }` block, properly pairing all control flow structures.
     - Verified with JavaScriptCore (`jsc`) across all files in `static/js/` (zero syntax errors).
  2. *Cache-Busting & Deployment (`templates/index.html`)*:
     - Bumped cache-busting version strings to `?v=1794380004` for `lobby.js`, `play.js`, `play.css`, and `app.js`.
     - Deployed cleanly to production VPS (`132.148.72.249`), verified HTTP/2 200 OK.
---

### Gateway "ENTER LOBBY" Button Navigation Resolution (Temporal Dead Zone ReferenceError Fix)
- **Problem**: When clicking the "ENTER LOBBY" button on the gateway page (`#page-loading`), the button remained permanently flattened in Chrome after mouse release (and popped back up in Safari), but the user was not transitioned to the lobby.
- **Root Cause**:
  1. *Temporal Dead Zone ReferenceError in `play.js`*:
     - In `static/js/play.js`, `adjustPlayHeaderForDevice()` at line 124 executed `(typeof isTournamentPlay !== "undefined" && isTournamentPlay)`.
     - `let isTournamentPlay = false;` was declared later at line 465.
     - When `adjustPlayHeaderForDevice()` was called on script evaluation at line 157, `isTournamentPlay` was in the JavaScript Temporal Dead Zone (TDZ).
     - In ES6+, accessing a block-scoped variable in TDZ—even with `typeof`—immediately throws `ReferenceError: Cannot access "isTournamentPlay" before initialization`.
     - This unhandled exception caused `play.js` script execution to abort midway, preventing `window.BoardAudio` and subsequent modules from initializing.
  2. *Uncaught Exception in `updateAuthUI()` during Gateway Transition*:
     - In `static/js/app.js`, line 3883 (`renderGameColorBar()`) also contained `typeof isTournamentPlay !== "undefined" && isTournamentPlay`.
     - When the user clicked "ENTER LOBBY", `executeGatewayTransition()` set `gatewayTransitioning = true` and invoked `updateAuthUI()`.
     - `updateAuthUI()` called `renderGameColorBar()`, which threw `ReferenceError: isTournamentPlay is not defined`.
     - Because `updateAuthUI()` was called without a `try...catch` wrapper in `executeGatewayTransition()`, the uncaught exception immediately aborted the async transition handler before the `setTimeout` to hide `#page-loading` and show `#page-lobby` could be scheduled.
     - Consequently, `#page-loading` remained visible, the button stayed flattened with `gatewayTransitioning = true` permanently stuck, and all subsequent clicks were blocked.
- **Fixes Applied**:
  1. *Hoisted Global Scope Variable Declaration (`static/js/play.js`)*:
     - Hoisted `var isTournamentPlay = false;` and `var isPrivateMatchPlay = false;` to the very top of `play.js` (above `adjustPlayHeaderForDevice`).
     - Exposed `window.isTournamentPlay` and `window.isPrivateMatchPlay` globally.
     - Replaced `(typeof isTournamentPlay !== "undefined" && isTournamentPlay)` with safe `Boolean(window.isTournamentPlay || localStorage.getItem("tournament_play_active") || ...)` across `play.js` (lines 124, 4486, 5104) and `app.js` (lines 2408, 3883).
  2. *Safe Global Object Lookups (`static/js/app.js`)*:
     - Replaced bare `typeof BoardAudio !== "undefined"` lookups with safe `window.BoardAudio` object property checks (`window.BoardAudio && typeof window.BoardAudio.suspend === "function"`), preventing any ReferenceError if audio helpers are called during initial page setup.
  3. *Failsafe Gateway Execution Protection (`static/js/app.js` & `templates/index.html`)*:
     - Wrapped `updateAuthUI()` inside `try...catch` in both `executeGatewayTransition` (`app.js`) and inline `handleEnterLobbyClick` (`index.html`), guaranteeing that UI helper warnings never interrupt or block the user from entering the lobby.
  4. *Asset Cache Version Bump & Verification*:
     - Bumped asset cache versions to `?v=1794380004` across `play.css`, `app.js`, `lobby.js`, and `play.js`.
     - Verified with headless Chrome running live page simulation: 0 unhandled errors, button transitions cleanly to `#page-lobby` with `display: flex`.

---

### Lobby Content Brief Displacement Resolution on Entering Lobby
- **Problem**: Occasionally, immediately after pressing "ENTER LOBBY", the contents within the lobby appeared briefly displaced, shifting or jumping before settling into place.
- **Root Causes**:
  1. *Premature Partial Display & Desynchronized Body Classes*:
     - In `executeGatewayTransition` (`app.js`) and `handleEnterLobbyClick` (`index.html`), the transition timer set `pLobby.style.display = 'flex'` and `pLobby.classList.add('active')` *before* `showPage('page-lobby')` ran.
     - At that moment, `document.body` did not yet have the class `lobby-active`.
     - In `style.css`, `.top-menu-spacer` has `height: var(--top-menu-height, 58px) !important;` when `lobby-active` is absent, and `display: none !important;` when `lobby-active` is present.
     - Because `#page-lobby` was revealed before `body.lobby-active` was set, the lobby was initially rendered pushed down by 58px. Moments later, when `showPage()` toggled `body.lobby-active`, `.top-menu-spacer` collapsed, shifting the entire lobby layout upward by 58px.
  2. *Unintended CSS Transitions Firing on Page Reveal*:
     - In `static/css/lobby.css`:
       - `.lobby-grid` had `transition: grid-template-columns 0.3s ease, gap 0.3s ease;` (line 86).
       - `.rating-filter-container` had `transition: height 0.32s cubic-bezier(0.16, 1, 0.3, 1), min-height 0.32s cubic-bezier(0.16, 1, 0.3, 1), padding 0.32s cubic-bezier(0.16, 1, 0.3, 1);` (line 408).
       - `#rooms-list` had `transition: flex 0.32s cubic-bezier(0.16, 1, 0.3, 1), height 0.32s cubic-bezier(0.16, 1, 0.3, 1), max-height 0.32s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.22s ease, padding 0.28s ease, margin 0.28s ease;` (line 739).
       - When the page transitioned from `display: none` to `display: flex`, these leftover transitions from an older layout interpolated from 0 / auto over 300ms–320ms, visibly resizing and shifting the columns, filter buttons, and active rooms list.
  3. *Desktop Static DOM Reordering Flash*:
     - In `templates/index.html`, `#mobile-panel-solo` was statically located outside `#mobile-panel-main` as a direct child of `.lobby-grid`. On desktop, `.lobby-grid` has 2 columns; with 3 direct children, the rooms panel was initially displaced to row 2 until JavaScript's `adjustLobbyLayoutForDevice()` reparented `#mobile-panel-solo` into `#mobile-panel-main`.
  4. *Mobile Carousel Zero-Offset Race Condition*:
     - In `static/js/lobby.js`, `switchLobbyPanel('main')` relied strictly on `el.offsetLeft`. On initial reveal before the browser completes layout reflow, `mainPanel.offsetLeft` could briefly report `0`. This caused the carousel to scroll to `0` (the Solo & Friends panel), and 50ms later snap to the Main panel when `offsetLeft` became positive, resulting in a visible horizontal jump.
     - Additionally, `_setupLobbyGridScrollTracker` tracked scroll events when the lobby was not active, occasionally setting `window._currentLobbyPanel = 'solo'` from initial unrendered scroll state.
- **Fixes Applied**:
  1. *Atomic Layout & Body Class Synchronization (`static/js/app.js` & `templates/index.html`)*:
     - Set `document.body.classList.remove('loading-active')` and `document.body.classList.add('lobby-active')` *before* revealing `#page-lobby`, ensuring `.top-menu-spacer` is already collapsed (`display: none !important`) and flex layout is active on first paint.
     - Removed premature manual assignments of `pLobby.style.display = 'flex'` in transition handlers, delegating cleanly to `showPage('page-lobby')`.
  2. *Elimination of Layout-Displacing CSS Transitions (`static/css/lobby.css`)*:
     - Removed `transition: grid-template-columns 0.3s ease, gap 0.3s ease;` from `.lobby-grid`.
     - Removed `transition: height ..., min-height ..., padding ...;` from `.rating-filter-container`.
     - Removed `transition: flex ..., height ..., max-height ..., opacity ..., padding ..., margin ...;` from `#rooms-list`.
     - Content now renders immediately in its final, exact geometry without animation lag.
  3. *Immediate Parse-Time Desktop DOM Nesting (`templates/index.html`)*:
     - Added an immediate synchronous script right after `.lobby-grid` in `index.html` so that on desktop viewports (`window.innerWidth > 900`), `#mobile-panel-solo` is reparented inside `#mobile-panel-main` at initial HTML parse time before first paint.
  4. *Mobile Carousel Fallback & Active Page Guard (`static/js/lobby.js`)*:
     - In `switchLobbyPanel`, if `window._currentLobbyPanel === 'main'` and `targetLeft <= 0`, dynamically fall back to `soloPanel.offsetWidth` or `lobbyGrid.clientWidth` so the carousel never scrolls to `0`.
     - In `_setupLobbyGridScrollTracker`, added a guard checking `lobbyPage.classList.contains('active')` so background scroll events never contaminate panel tracking.
  5. *Asset Cache Version Bump*:
     - Bumped cache-busting strings to `?v=1794380005` across `lobby.css`, `play.css`, `app.js`, `lobby.js`, and `play.js`.

---

### Room Entry AbortController Signal Timeout & Network Error Resolution
- **Problem**: When attempting to enter the multiplayer subanagrams room (or other rooms), the client encountered a `Network error` with the word `"signal"` in the error notification (`Network error: signal is aborted without reason` / `The operation was aborted due to AbortSignal`).
- **Root Cause**:
  1. *Premature 8-Second Client AbortController*:
     - In `static/js/lobby.js`, `enterLobbyRoom()` and `joinRoom()` wrapped `/api/room/create` and `/api/room/<id>/join` in an `AbortController` configured with `setTimeout(() => ctrl.abort(), 8000)`.
     - When entering Subanagrams or public rooms after a server restart or when the board generator is generating candidate sequences, loading word validator tries, or under background CPU load, the server took ~9–11 seconds to generate and score the initial board.
     - At precisely 8,000ms, the client AbortController fired, causing `fetch` to reject with a native browser `AbortError` (`"signal is aborted without reason"` in Chrome/Safari).
     - The catch blocks in `lobby.js` and `index.html` simply passed `error.message` through to `showLobbyToast('Network error: ' + error.message, 'error')`, directly displaying the browser's internal signal abort text.
- **Fixes Applied**:
  1. *Increased Room Join/Creation Timeout (`static/js/lobby.js`)*:
     - Extended the `AbortController` timeout from 8,000ms to **25,000ms (25 seconds)** for both `enterLobbyRoom` and `joinRoom`, giving complex board generations ample headroom.
  2. *Graceful User-Friendly Timeout Handling (`static/js/lobby.js` & `templates/index.html`)*:
     - Added explicit detection for `AbortError` and messages containing `"signal"`, replacing the cryptic browser signal abort text with a clear, helpful message: `"Joining room took longer than expected. Please try again."`.
  3. *Asset Cache Version Bump*:
     - Bumped version strings to `?v=1794380006` across `lobby.css`, `play.css`, `app.js`, `lobby.js`, and `play.js`.

---

### Multiplayer Subanagrams Instant Room Entry & Server CPU Starvation Fix
- **Problem**:
  - When clicking the multiplayer subanagrams "Start" button in the lobby, the user was stuck on the "Loading..." screen for 10–20 seconds before the overlay disappeared, leaving the user in the lobby without entering the room.
- **Root Causes**:
  1. *Synchronous Subanagrams Board Generation Inside Request Handler*:
     - In `game_room.py:3875` and `game_room.py:4238`, `create_room()` synchronously called `generate_subanagrams_board_and_words(sp)` while holding the global `RoomManager` lock (`with self.lock:`).
     - Because dictionary tries (specifically CSW with 280k+ words and Counter objects) were not preloaded on server startup, the first user to enter Subanagrams suffered a 10–20 second synchronous delay on the HTTP thread.
     - Even if the room already had an active board or a staged `next_round_board`, the code did not check them on wakeup, needlessly wiping and regenerating the board.
  2. *Unbounded Cache Refill Worker CPU Thrashing*:
     - In `board_generator.py:698`, `refill_board_cache_bg` ran a `while True:` loop without a break condition on retry attempts.
     - Configuration `("4x6", "CSW", (50, 100), "Normal", 6, "Easy", False)` in `seed_pregenerated_cache_bg` required <= 15% uniqueness on a 4x6 grid with 6L minimum and CSW dictionary, which is mathematically near-impossible and resulted in 25 attempts per cycle (~60s of 100% CPU DFS search) without ever matching "Easy".
     - Because it saved the resulting board under "Hard", the "Easy" cache count never reached the target, causing the worker to run continuously in an infinite CPU loop.
     - Under Python's Global Interpreter Lock (GIL), this pegged the CPU and blocked Waitress worker threads from processing incoming HTTP requests (`waitress.queue:Task queue depth is 60`). Nginx or client AbortControllers timed out (`499 Client Closed Request`), aborting `/api/room/create` and dismissing the loading screen.
  3. *Missing Mobile Layout State Synchronization in Inline Room Entry*:
     - `enterLobbyRoomInline` in `templates/index.html` did not toggle `is-subanagrams` on `document.body` upon receiving room confirmation.
- **Fixes Applied**:
  1. *Subanagrams Background Pre-Warming (`app.py`)*:
     - Added `prewarm_subanagrams_bg()` to asynchronously pre-load the Subanagrams CSW dictionary and pre-initialize `pub_v2_subanagrams_subanagrams_120` 2 seconds after server boot.
     - When any user clicks "Start", the room and board are already active and ready in memory; `/api/room/create` returns in **< 2 milliseconds**.
  2. *Instant Promotion of Staged Boards & Active Board Guard (`game_room.py`)*:
     - In `create_room()` wakeup logic, added fast-paths to check if the room already has an active board (returns immediately in <1ms) or if `next_round_board` is staged (promotes it instantly without generating on the fly).
  3. *Unbounded Cache Refill Worker Guard (`board_generator.py`)*:
     - Added `if attempts > 3: break` in `refill_board_cache_bg` so impossible parameter sets terminate after 3 attempts rather than spinning forever.
     - Adjusted `("4x6", "CSW", (50, 100), "Normal", 6, "Easy", False)` to `"Hard"` in `seed_pregenerated_cache_bg`.
  4. *Body Class Synchronization (`templates/index.html`)*:
     - Added `document.body.classList.toggle('is-subanagrams', isSub)` in `enterLobbyRoomInline`.
  5. *Asset Cache Version Bump*:
     - Bumped asset version strings to `?v=1794380007` across `lobby.css`, `play.css`, `app.js`, `lobby.js`, and `play.js`.




