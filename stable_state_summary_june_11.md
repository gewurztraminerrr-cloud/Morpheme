# Stable State Summary — June 11, 2026 (Start Over Point)

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `99f76f6d068c8fcc5dd418a5be87e61756fc86dd` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `99f76f6d068c8fcc5dd418a5be87e61756fc86dd` / `snapshot-current` / `START_OVER_POINT_JUNE_11` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `99f76f6d068c8fcc5dd418a5be87e61756fc86dd` / `snapshot-current` | ✅ Fully Deployed & PM2 Restarted |

**All environments are 100% synchronized at the latest commit 99f76f6d068c8fcc5dd418a5be87e61756fc86dd.**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py`.
The active recovery points `START_OVER_POINT_JUNE_11` and `snapshot-current` tags have been successfully updated and pushed to GitHub.

---

## Serving Versions (cache-busted)

| File / Style | Version | Description |
|--------------|---------|-------------|
| `/css/lobby.css` | `v=26` | Configured `.room-meta` flex column container layout to stack rating requirement badges vertically under average ratings. |
| `/css/play.css` | `v=88` | Added premium glassmorphic draggable scrollbar container styling for lists in the Tools section. |
| `/js/lobby.js` | `v=7` | Modified active rooms listing HTML structure to support clean vertical stacking of average ratings and requirement badges without inline styles. |
| `/js/play.js` | `v=150` | Implemented 50% Added Words roll and dynamic UI suffix rendering, plus zero-latency submitted words cache. |
| `/js/tools.js` | `v=39` | Built draggable scrollbar thumb, implemented overload check safeguards, and added filter requirement warnings. |
| `templates/index.html` | *Dynamic* | Bumps cache-busters for lobby.css (v=26), play.css (v=88), lobby.js (v=7), play.js (v=150), and tools.js (v=39). |

---

## Work Completed on June 11, 2026

### 1. Active Rooms Rating Range Layout Alignment
* **Goal achieved:** Allow room rating limit tags (e.g. "Req: 1000 - 1200") to fit on a single row without awkward text-wrapping, positioned directly under "Avg Rating".
* **Implementation:**
  * Modified `.room-header-row` in [lobby.css](file:///Users/jeffbabiak/static/css/lobby.css) (version `v=26`) to align elements to `flex-start` so that wait states and rating elements align neatly at the top boundaries.
  * Restructured `.room-meta` as a flex container with `flex-direction: column` and `align-items: flex-end` to stack ratings cleanly.
  * Added styling for `.room-avg-rating` (with a subtle background overlay and padding) and `.rating-req-badge` (with a solid `white-space: nowrap` and standard `padding: 4px 8px`).
  * Updated the active rooms template inside [lobby.js](file:///Users/jeffbabiak/static/js/lobby.js) (version `v=7`) to use these classes instead of inline styles.

### 2. Tools Lists Server Overload & Browser Freeze Safeguards
* **Goal achieved:** Avoid browser freeze and server overload when loading massive dictionaries (e.g. NWL/CSW) or computing Scrabble likelihood without filters.
* **Implementation:**
  * Updated [tools.js](file:///Users/jeffbabiak/static/js/tools.js) (version `v=39`) with checks to block loading massive word lists (NWL, CSW, CSW Only, Uniques) when both length and starting letter filters are unset ("All"). Instead, it renders a friendly UI notification: *"Please select configurations that don't overload the server with too many computations."*
  * Blocked combining "Likelihood" with "Length: All" to avoid running CPU-heavy scoring loops over 190,000 words.
  * Added a persistent note in the filter controls panel in [templates/index.html](file:///Users/jeffbabiak/templates/index.html) reminding users to select specific configurations.

### 3. Mobile Draggable Custom Scrollbar for Tools Lists
* **Goal achieved:** Provide a draggable scrollbar thumb on mobile devices for lists in the Tools section to navigate large word lists quickly.
* **Implementation:**
  * Wrapped `#main-list-results` in `.list-scroll-area-wrapper` in [templates/index.html](file:///Users/jeffbabiak/templates/index.html) and added custom track and thumb markup.
  * Added styling for a glassmorphic track and glowing cyan-blue gradient draggable thumb in [play.css](file:///Users/jeffbabiak/static/css/play.css) (version `v=88`), hiding default thin scrollbars on mobile.
  * Implemented unified touch/mouse dragging logic in [tools.js](file:///Users/jeffbabiak/static/js/tools.js) with a `MutationObserver` that automatically updates the thumb position/size when lists are updated.

### 4. Duplicate Word Submission Audio Feedback Fix
* **Goal achieved:** Avoid playing a success sound followed by a failure sound when a user submits a word twice; only play the negative sound.
* **Implementation:**
  * Implemented `window._localSubmittedWords` Set in [play.js](file:///Users/jeffbabiak/static/js/play.js) (version `v=149`), initialized and cleared per round.
  * Checked this cache Set during optimistic pre-validation to immediately mark duplicate entries as `alreadyFound`, playing the failure sound once and flashing the board purple.
  * Added successfully validated words to the cache Set immediately upon receiving a successful server response.

### 5. Word Tally Double Counting Fix
* **Goal achieved:** Prevent newly discovered words from incrementing the player's total tally by 2 instead of 1.
* **Implementation:**
  * Modified `play.js` to rely directly on the exact tally returned by the server API response instead of doing client-side increments.

### 6. Notepad Scroll Arrow Shape Update
* **Goal achieved:** Change the circular/oval styling on the notepad scroll buttons to clean rectangles.
* **Implementation:**
  * Modified [lobby.css](file:///Users/jeffbabiak/static/css/lobby.css) (version `v=25`) and `index.html` classes to render scroll buttons as neat, square/rectangular controls.

### 7. Web Audio Bluetooth Latency Optimization
* **Goal achieved:** Eliminate sound effect delays when playing with Bluetooth earpieces on mobile.
* **Implementation:**
  * Configured `AudioContext` with `latencyHint: 'interactive'`.
  * Warmed up the audio context on the first user tap/click.
  * Implemented an inaudible, silent keep-alive oscillator (`0.00001` gain) to keep the Bluetooth connection warm.

### 8. Boggle Round Transition and GIL Starvation Fixes
* **Goal achieved:** Resolve round start delays, GIL interpreter locks, database write blockages, and intermission bell timing.
* **Implementation:**
  * Migrated database queries out of request loops.
  * Inserted yielding `time.sleep(0.001)` statements in the CPU-bound solver.
  * Allowed a `0.2`s clock tolerance on transition checks.
  * Configured rapid polling on both transition directions and updated the intermission bell to sound exactly at 10 seconds.

### 9. Forum Mobile Attachment Click Triggers
* **Goal achieved:** Fix file/image uploads on mobile devices by making sure taps on the button area open the native picker.
* **Implementation:**
  * Restructured layout positions for `<input type="file">` to be layered absolutely over the styling container with `pointer-events` configured so that mobile taps register directly on the native component.

### 10. Added Words 50% Probability Integration
* **Goal achieved:** Ensure that custom "Added Words" (AW) have a 50% chance of being used on the board per round in standard play. The UI indicates this status on the Spinner Set (e.g., displaying `NWL + AW` or `CSW + AW` if active, and simply `NWL` or `CSW` if inactive).
* **Implementation:**
  * Updated [spinner_set.py](file:///Users/jeffbabiak/spinner_set.py) to roll a 50% chance (`random.random() < 0.5`) to set `use_added_words` to `True` or `False` if the global moderator configuration is enabled.
  * Propagated `use_added_words` thread-safely via `use_added_words_ctx` (ContextVar) and updated [game_room.py](file:///Users/jeffbabiak/game_room.py) to propagate and set `room.use_added_words` on room creation (kickstart), start of next round, solo parameter generation, 6x8 rescue fallback, and fallback board generation.
  * Updated [static/js/play.js](file:///Users/jeffbabiak/static/js/play.js) (version `v=150`) to dynamically render `+ AW` based on `sp.use_added_words` or `state.use_added_words` strictly.
  * Incremented the cache-buster version of `play.js` in [templates/index.html](file:///Users/jeffbabiak/templates/index.html) to `?v=150`.
