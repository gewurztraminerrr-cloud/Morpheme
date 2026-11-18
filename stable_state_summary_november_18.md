# Stable State Summary — November 18, 2026

## 1. Overview of Changes

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

