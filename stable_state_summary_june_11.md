# Stable State Summary — June 11, 2026 (Start Over Point)

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `b5127c82ac0cf4d778d2ba1b4248f445b3a01890` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `b5127c82ac0cf4d778d2ba1b4248f445b3a01890` / `snapshot-current` / `START_OVER_POINT_JUNE_11` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `b5127c82ac0cf4d778d2ba1b4248f445b3a01890` / `snapshot-current` | ✅ Fully Deployed & PM2 Restarted |

**All environments are 100% synchronized at the latest commit b5127c82ac0cf4d778d2ba1b4248f445b3a01890.**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py`.
The active recovery points `START_OVER_POINT_JUNE_11` and `snapshot-current` tags have been successfully updated and pushed to GitHub.

---

## Serving Versions (cache-busted)

| File / Style | Version | Description |
|--------------|---------|-------------|
| `/css/lobby.css` | `v=26` | Configured `.room-meta` flex column container layout to stack rating requirement badges vertically under average ratings. |
| `/css/play.css` | `v=88` | Added premium glassmorphic draggable scrollbar container styling for lists in the Tools section. |
| `/css/forum.css` | `v=9` | Configured pointer-events and absolute layering to fix mobile attachments/image chooser. |
| `/js/lobby.js` | `v=7` | Modified active rooms listing HTML structure to support clean vertical stacking of average ratings and requirement badges. |
| `/js/play.js` | `v=152` | Implemented Triple Format background music state engine, 50% Added Words roll, and zero-latency submitted words cache. |
| `/js/forum.js` | `v=37` | Implemented event redirection to native file input wrapper click handlers with bubble loop protection. |
| `/js/tools.js` | `v=41` | Built draggable scrollbar thumb and replaced upfront computational list loading warnings with a 20-second timeout warning notice. |
| `templates/index.html` | *Dynamic* | Bumps cache-busters for lobby.css (v=26), play.css (v=88), forum.css (v=9), lobby.js (v=7), play.js (v=152), forum.js (v=37), and tools.js (v=41). |

---

## Work Completed on June 11, 2026

### 1. Game View and Leaderboard Score Discrepancy Fix
* **Goal achieved:** Eliminate scoring discrepancies where a player's round score in the gameplay UI could sometimes be 1 point higher than the score displayed on Leaderboards and Achievements.
* **Implementation:**
  * Identified a race condition on round end: if a player submitted a word at the very last second of a round, the submission API handler was processing and accepting it in-memory *after* the intermission transition task had already snapshotted and written the round details to the SQLite database (using `save_round_history`). This resulted in the player's live score being incremented (e.g. from 101 to 102), but the database record remained at 101 points.
  * Modified `submit_word` in `game_room.py` to immediately reject word submissions with `success=False` if the room state is not `active` (e.g., if it has transitioned to `intermission` or `finished`). This ensures the live score and the database record stay perfectly in sync.

### 2. Triple Format Background Music Automation
* **Goal achieved:** Automatically control the background music (`triple.mp3`) during rounds featuring the **Triple Format** with smooth fade-in, looping, fade-out, and auto-stop.
* **Implementation:**
  * Copied `/Users/jeffbabiak/Downloads/triple.mp3` to `/Users/jeffbabiak/static/music/triple.mp3`.
  * Added `<audio id="triple-music" src="/music/triple.mp3" preload="auto" loop></audio>` to `templates/index.html`.
  * Added user-gesture pre-loading of the `#triple-music` audio element inside `play.js` on first page interaction to bypass mobile browser autoplay blocks.
  * Added cleanup to pause `#triple-music` when exiting game rooms in `clearGameUIAndCache()`.
  * Implemented `updateTripleMusicState(remaining)` inside `updateLocalTimer()` (running every 500ms) to:
    * Stop play if lobby music is disabled (`lobby_music === false`).
    * Fade-in the music starting 10 seconds before the round starts (intermission state), volume increasing from 0.0 to 1.0, and rewinding the track position to `0`.
    * Play the music at full volume (`1.0`) and loop it continuously during the active round.
    * Fade-out the music linearly from volume `1.0` to `0.0` over the first 10 seconds of intermission when transitioning from a Triple Format round, pausing it completely after 10 seconds.
    * Keep the music paused during standard intermission when the next format is not Triple.
  * Incremented cache-buster version of `/js/play.js` to `v=151`.

### 3. Active Rooms Rating Range Layout Alignment
* **Goal achieved:** Allow room rating limit tags (e.g., "Req: 1000 - 1200") to fit on a single row without awkward text-wrapping, positioned directly under "Avg Rating".
* **Implementation:**
  * Modified `.room-header-row` in `lobby.css` (version `v=26`) to align elements to `flex-start` so that wait states and rating elements align neatly at the top boundaries.
  * Restructured `.room-meta` as a flex container with `flex-direction: column` and `align-items: flex-end` to stack ratings cleanly.
  * Added styling for `.room-avg-rating` (with a subtle background overlay and padding) and `.rating-req-badge` (with a solid `white-space: nowrap` and standard `padding: 4px 8px`).
  * Updated the active rooms template inside `lobby.js` (version `v=7`) to use these classes instead of inline styles.

### 4. Tools Lists Timeout Warning Notice
* **Goal achieved:** Revert upfront warnings/blocks on Tools lists while protecting user browser session and server performance with a 20-second timeout.
* **Implementation:**
  * Updated `tools.js` (version `v=41`) to remove hard blocking rules on loading massive lists or Scrabble likelihood without filters.
  * Implemented a client-side fetch timeout of 20 seconds using `AbortController` and `setTimeout`.
  * If the data takes longer than 20 seconds to load, the request is aborted and a warning is displayed: *"This list is taking longer than 20 seconds to load because it is computationally heavy... Please select a specific word length or starting letter to reduce the size of the request."*

### 5. Mobile Draggable Custom Scrollbar for Tools Lists
* **Goal achieved:** Provide a draggable scrollbar thumb on mobile devices for lists in the Tools section to navigate large word lists quickly.
* **Implementation:**
  * Wrapped `#main-list-results` in `.list-scroll-area-wrapper` in `templates/index.html` and added custom track and thumb markup.
  * Added styling for a glassmorphic track and glowing cyan-blue gradient draggable thumb in `play.css` (version `v=88`), hiding default thin scrollbars on mobile.
  * Implemented unified touch/mouse dragging logic in `tools.js` with a `MutationObserver` that automatically updates the thumb position/size when lists are updated.

### 6. Duplicate Word Submission Audio Feedback Fix
* **Goal achieved:** Avoid playing a success sound followed by a failure sound when a user submits a word twice; only play the negative sound.
* **Implementation:**
  * Implemented `window._localSubmittedWords` Set in `play.js` (version `v=151`), initialized and cleared per round.
  * Checked this cache Set during optimistic pre-validation to immediately mark duplicate entries as `alreadyFound`, playing the failure sound once and flashing the board purple.
  * Added successfully validated words to the cache Set immediately upon receiving a successful server response.

### 7. Word Tally Double Counting Fix
* **Goal achieved:** Prevent newly discovered words from incrementing the player's total tally by 2 instead of 1.
* **Implementation:**
  * Modified `play.js` to rely directly on the exact tally returned by the server API response instead of doing client-side increments.

### 8. Notepad Scroll Arrow Shape Update
* **Goal achieved:** Change the circular/oval styling on the notepad scroll buttons to clean rectangles.
* **Implementation:**
  * Modified `lobby.css` (version `v=25`) and `index.html` classes to render scroll buttons as neat, square/rectangular controls.

### 9. Web Audio Bluetooth Latency Optimization
* **Goal achieved:** Eliminate sound effect delays when playing with Bluetooth earpieces on mobile.
* **Implementation:**
  * Configured `AudioContext` with `latencyHint: 'interactive'`.
  * Warmed up the audio context on the first user tap/click.
  * Implemented an inaudible, silent keep-alive oscillator (`0.00001` gain) to keep the Bluetooth connection warm.

### 10. Boggle Round Transition and GIL Starvation Fixes
* **Goal achieved:** Resolve round start delays, GIL interpreter locks, database write blockages, and intermission bell timing.
* **Implementation:**
  * Migrated database queries out of request loops.
  * Inserted yielding `time.sleep(0.001)` statements in the CPU-bound solver.
  * Allowed a `0.2`s clock tolerance on transition checks.
  * Configured rapid polling on both transition directions and updated the intermission bell to sound exactly at 10 seconds.

### 11. Forum Mobile Attachment Click Triggers
* **Goal achieved:** Fix the "Attach an image (optional)" and "Click to choose a file or drag and drop" buttons in the Forum on mobile devices to open the native photo/file chooser.
* **Implementation:**
  * Updated CSS layout (`forum.css?v=9`) to manage layering and pointer-events.
  * Added a delegation click handler in `forum.js` (version `v=37`) to listen to clicks on button wrappers and programmatically trigger `.click()` on the nested `<input type="file">`.
  * Wrapped the redirection logic with bubble protection (`event._forumTriggered`) to prevent infinite recursion, ensuring Safari and other mobile browsers open the native file selection dialog.

### 12. Added Words 50% Probability Integration & Uniqueness Parity
* **Goal achieved:** Ensure that custom "Added Words" (AW) have a 50% chance of being used on the board per round in standard play. The UI indicates this status on the Spinner Set (e.g., displaying `NWL + AW` or `CSW + AW` if active, and simply `NWL` or `CSW` if inactive).
* **Implementation:**
  * Updated `spinner_set.py` to roll a 50% chance (`random.random() < 0.5`) to set `use_added_words` to `True` or `False` if the global moderator configuration is enabled.
  * Propagated `use_added_words` thread-safely via `use_added_words_ctx` (ContextVar) and updated `game_room.py` to propagate and set `room.use_added_words` on room creation (kickstart), start of next round, solo parameter generation, 6x8 rescue fallback, and fallback board generation.
  * Updated `static/js/play.js` (version `v=151`) to dynamically render `+ AW` based on `sp.use_added_words` or `state.use_added_words` strictly.
  * **Uniqueness Parity:** Updated `board_generator.py` to treat Added Words as unique words inside all uniqueness ratio validation and board optimization methods when `use_added_words` is active. This ensures they correctly contribute to uniqueness thresholds required for Hard (e.g., 45%) difficulty boards.

### 13. Spinner Set Word Lists Probability Details & Odds Display
* **Goal achieved:** Update "Word Lists" in the Spinner Set Odds modal and wheel randomization logic to reflect standard joint configurations:
  * When Added Words (AW) are enabled globally, the wheel rolls with a 4-way split of 25% each:
    - 25% NWL
    - 25% CSW
    - 25% NWL + AW
    - 25% CSW + AW
  * When Added Words are disabled, it rolls with a 2-way split of 50% each (50% NWL / 50% CSW).
* **Implementation:**
  * Updated `spinner_set.py` to pick from the 4 options when AW is globally active, assigning `dictionary` and `use_added_words` accordingly.
  * Updated the Spinner Set Odds modal in `templates/index.html` to display two separate, clear lists: one for "Added Words Enabled" (listing all 4 possibilities at 25% each) and one for "Added Words Disabled" (listing NWL at 50% and CSW at 50%).

### 14. Random Word Generator Randomness Fix
* **Goal achieved:** Ensure that the Random Word Generator tool under the Tools section produces completely random words, particularly when the Added Words dictionary is selected.
* **Implementation:**
  * Modified `app.py` to disable definitions-cache filtering for the `'added_words'` dictionary so that custom words added by moderators (which do not have Scrabble-definitions in the static definitions cache) bypass the filter and can be chosen.
  * Configured `load_tools_dictionary` to skip local cache storage for `'added_words'` so that any custom moderator additions/deletions take effect immediately.

### 15. Included Valued Letters Format in Leaderboards & Achievements
* **Goal achieved:** Ensure that rounds using the "Valued Letters" format are included in Achievements and Leaderboards (reversing a previous exclusion) so that all gameplay accomplishments are accurately documented and rewarded.
* **Implementation:**
  * Removed the exclusion SQL filter `AND (board_format IS NULL OR board_format != 'Valued Letters')` in both achievements and leaderboard data endpoints in `app.py`.
  * This registers round details such as "jeffles" finding the word "EMULATORS" (which occurred in a Valued Letters round) correctly.

### 16. Leaderboard and Profile UTC Timestamps
* **Goal achieved:** Fix the display of time-of-event on the leaderboard and user profiles to be accurate across all timezones.
* **Implementation:**
  * Added a backend helper `format_chicago_to_utc` to convert local Chicago datetime values stored in the SQLite database to standard UTC ISO 8601 strings.
  * Wrapped profile statistics, achievements, and leaderboard query responses in `app.py` to format all timestamps to UTC before serving them to the client.
  * Replaced timezone-appending hacks in JavaScript with direct parsing of UTC strings, ensuring local browser conversion displays the exact, correct time.

### 17. Round Replay Timing and Dynamic Word Listing Fix
* **Goal achieved:** Fix the round replay modal in Leaderboards to dynamically list found words at their exact discovery timestamps instead of listing them all at once.
* **Implementation:**
  * Updated `tools.js` to correctly parse `round.timestamp` using `window.parseUTCTimestamp` (converting the UTC string to milliseconds relative to epoch and then to seconds), resolving a scale factor bug that caused word find-times to be incorrectly offset by 56 years.
  * Added fallback validation checks to handle parse failures gracefully and keep the playback animation running smoothly.

### 18. Word Tally Active Dictionary Support
* **Goal achieved:** Ensure that words found in dictionaries other than CSW (like the newly added word `PALAISES` in the NWL dictionary) are correctly registered and counted in cumulative tally stats.
* **Implementation:**
  * Updated `log_word_tally` in `game_room.py` to fetch the room's active dictionary using `active_dict = getattr(room, 'current_dictionary', 'NWL')` and pass it to `is_valid_word`, instead of hardcoding `'CSW'` validation.
