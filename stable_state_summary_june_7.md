# Stable State Summary — June 7, 2026 (Start Over Point)

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `5de2016` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `5de2016` / `snapshot-current` / `START_OVER_POINT_JUNE_7` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `5de2016` / `snapshot-current` | ✅ Fully Deployed & PM2 Restarted |

**All environments are 100% synchronized at the latest commit 5de2016.**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py`.
The active recovery points `START_OVER_POINT_JUNE_7` and `snapshot-current` tags have been successfully updated and pushed to GitHub.

---

## Serving Versions (cache-busted)

| File / Style | Version | Description |
|--------------|---------|-------------|
| `/css/play.css` | `v=86` | Added overrides for `.bonus-highlight` elements when actively selected or highlighted to hide the lime green background and glow. |
| `/css/howtoplay.css` | `v=10` | Implemented visible list grid and button styles for FAQ quick navigation. |
| `/css/lobby.css` | `v=20` | Added styling for the "My Rating" button using a premium blue-purple gradient, and optimized input width/button sizes for mobile responsive layouts. |
| `/js/app.js` | `v=41` | Implemented scroll navigation and pulse highlight logic for FAQ links. |
| `/js/lobby.js` | `v=6` | Added click handler to populate rating-filter with current user rating and sort the active rooms list. |
| `/js/play.js` | `v=144` | Disabled definitions panel gold flashing animation at round complete in 24h rooms, and fixed transposition corruption on definition click during intermission. |
| `templates/index.html` | *Dynamic* | Replaced the quick-nav dropdown selector with the visible question link button grid, bumped howtoplay.css cache-buster to `v=10`, app.js cache-buster to `v=41`, play.js cache-buster to `v=144`, play.css cache-buster to `v=86`, `lobby.css` cache-buster to `v=18`, `lobby.js` cache-buster to `v=6`, added the "My Rating" button, and updated the Spinner Set Odds description and trial percentages. |

---

## Work Completed on June 7, 2026

### 1. Either/Or Word Involvement Optimization (Target 1/3)
* **Goal achieved:** Optimize Either/Or tile letter pairing and placement so that approximately 1/3 (33%) of the valid words generated on the board involve the Either/Or tile.
* **Implementation (`board_generator.py`):**
  * Updated the E/O tile selection loops (both standard and emergency paths) to temporarily apply each candidate cell and letter partner, solve the board using `_solve_board(store_paths=True)`, and calculate the exact percentage of words passing through the tile.
  * Checks for ratio difference from `0.333`. If a candidate cell is within 5% of `1/3` (28.3% to 38.3%), it is accepted immediately. Otherwise, it searches and selects the candidate with the smallest absolute difference from 1/3.
  * Corrected path membership lookup (`cell in temp_words[w]`) in `_solve_board` result parsing.

### 2. Highlight Selection Override for Either/Or and Bonus Letter Tiles
* **Goal achieved:** Ensure that when a player highlights or selects an Either/Or tile or a Bonus Letter tile, the lime green styling (background, border, text color, box-shadow, and pulsing animation) is completely overridden by the standard selection or typing highlight colors.
* **Implementation (`static/css/play.css` & `templates/index.html`):**
  * Added CSS rules to override `.board-cell.selected.bonus-highlight`, `.board-cell.current.bonus-highlight`, `.board-cell.typing-highlight.bonus-highlight`, and `.board-cell.review-highlight.bonus-highlight`.
  * Set these classes to use the standard theme accent colors (`--highlight-mouse-color` and `--highlight-typing-color`), reset text color to `#000`, and set `animation: none !important` to stop the pulsing lime green box-shadow animation when selected.
  * Injected these overrides both in the external stylesheet and the templates inline stylesheet for maximum safety and cache independence, and bumped the stylesheet version parameter in `templates/index.html` from `?v=85` to `?v=86`.

### 3. FAQ Question Hidden Features Update
* **Goal achieved:** Add information to the FAQ regarding clicking on the Spinner Set to see more information.
* **Implementation (`templates/index.html`):**
  * Added item 6 to the "Are there any additional features in game rooms that aren’t obvious?" FAQ answer: `<li><strong>Clicking on the Spinner Set</strong> displays more information about the likelihoods and meaning of each Spinner.</li>`.

### 4. Board Gesture & Word Validation Sound Effects
* **Goal achieved:** Play responsive, latency-free sound effects during swiping and word submission, with an option to toggle this in User Settings.
* **Implementation (`templates/index.html` & `static/js/settings.js` & `static/js/play.js`):**
  * Added a "Board Sound Effects" checkbox toggle to the Layout / App Theme settings grid inside `index.html`.
  * Configured `settings.js` to initialize, save (via debounced API POST requests to `/api/settings/update`), and toggle the sound preference `board_sounds` (defaulting to `true`).
  * Created a global, Web Audio API-powered `BoardAudio` class in `play.js` that synthesizes low-latency audio tones.
  * Configured `playTileSound(pathLen)` to play sine wave blips that arpeggiate in pitch as letters are added or backtracked.
  * Configured `playSuccessSound()` and `playFailureSound()` to trigger warm chimes or thud sounds respectively inside `showValidationFeedback()` for all gameplay validation scenarios.
  * Incremented cache-buster script tags in `index.html` to `settings.js?v=16` and `play.js?v=143`.

### 5. Stuck Board Transition Fallback Recovery
* **Goal achieved:** Automatically recover stalled room transitions at 0:00 within 10 seconds by resetting state locks and applying simplified parameters.
* **Implementation (`game_room.py`):**
  * Configured the stuck watchdog in `get_next_round_milestone` to detect if the room has been stuck in the `intermission` state for more than 10 seconds past the 0:00 timer mark.
  * Overrides the room's parameters with a guaranteed-fast fallback parameter set: Normal format (or Valued Letters with 50-100 count for 24h rooms), 50-100 word count range, Medium difficulty, NWL dictionary, min word length 3, and bonus word length 6.
  * Clears the staging board fields and resets all transition locks (`starting_round = False`), triggering a clean re-attempt on the next heartbeat or client poll.
  * Stale background generation and emergency generation tasks immediately detect the parameter change and abort, preventing resource contention.

### 6. Intermission Definition & Board Transposition Bug Fixes
* **Goal achieved:** Ensure that viewing a word's definition during intermission does not clear board tile highlights or corrupt the board transposition state on mobile devices.
* **Implementation (`static/js/play.js` & `templates/index.html`):**
  * Added a custom cached property `state._isAlreadyTransposed` and `state._isBoardTransposedValue` on the room state object.
  * Configured `updateGameState` to check if `state._isAlreadyTransposed` is set. If true, it restores `window.isBoardTransposed` from the cached value and skips the transposition logic, completely preventing rows/columns double-swapping and transposition state corruption.
  * Configured `reapplyBoardHighlights` to correctly restore the intermission letter filter highlight class `.intermission-highlight` to the clicked cell.
  * Incremented cache-buster script tags in `index.html` to `play.js?v=144`.

### 7. Spinner Set Odds Modal Updates
* **Goal achieved:** Updated the text description and difficulty percentages inside the "Diff: Difficulty & Uniqueness" section of the Spinner Set Odds modal based on 200 trials.
* **Implementation (`templates/index.html`):**
  * Updated description text to explain the 200 trials conducted for each 2D board size and the rationale behind the difficulty distribution.
  * Updated percentages under Easy, Medium, and Hard for 4x4, 4x6, 5x7, and 6x8/Cube according to the trial results.

### 8. Lobby sorting by user rating ("My Rating" Button)
* **Goal achieved:** Allowed users to instantly sort active game rooms by how close the room's average rating is to their own player rating, with mobile responsiveness layout optimizations.
* **Implementation (`templates/index.html` & `static/css/lobby.css` & `static/js/lobby.js`):**
  * Added the `<button id="my-rating-btn">My Rating</button>` element in `templates/index.html` inside the rating filter container next to the "Find" button.
  * Styled the button in `lobby.css` with a premium blue-to-purple gradient, lift transitions, and active press scale animations.
  * Programmed a click event listener in `lobby.js` that retrieves the current user rating (defaulting to 1000 if not logged in or invalid), updates the `#rating-filter` text field, sets the search filter value `window.activeRatingFilterValue`, and triggers `fetchAndRenderRooms()` to execute the sort.
  * Widened the `.lobby-grid` desktop columns from `2fr 1fr` to `1.7fr 1.3fr` and removed the overriding `min-width: 0` on `#rating-filter` to enforce a `min-width: 185px`. This keeps the "Filter by average rating" placeholder text fully visible and readable on desktop/laptop screens with the buttons placed side-by-side on the right.
  * Configured media queries (`max-width: 900px` and `max-width: 480px`) to wrap the layout (`flex-wrap: wrap`, `#rating-filter` at `width: 100%`, and buttons at `flex: 1`) so it neatly stacks on smaller mobile and tablet viewports.

### 9. Timezone-Aware Leaderboard & Achievements ("Day" Tab Fix)
* **Goal achieved:** Fixed a bug where the "Day" tab under Leaderboards and User Achievements/Skill Rankings displayed no records due to a double-conversion timezone offset mismatch in SQLite.
* **Implementation (`app.py`):**
  * Replaced the SQLite `date(timestamp, 'localtime') = date('now', 'localtime')` date function calls with a timezone-aware calculation in Python using `zoneinfo.ZoneInfo("America/Chicago")` (matching the game's authoritative server clock).
  * The calculations generate exact threshold string representations (`chicago_today_str`, `chicago_week_ago_str`, etc.) which are evaluated directly against the stored datetimes in SQLite, correcting the timezone offset mismatch and restoring all records to the "Day" tabs.

### 10. Intermission Leaver Stats Fix (Immediate Save Enhancement)
* **Goal achieved:** Ensure that when a player leaves a room during intermission, the round results they just played are saved correctly to achievements, leaderboards, and stats instead of being dropped.
* **Implementation (`game_room.py`):**
  * Implemented an immediate database save by calling `save_round_history` and `log_word_tally` inside the `process_results_async()` thread spawned at the start of intermission (transition ACTIVE -> INTERMISSION).
  * Stores a reference to `RoomManager` in a global variable `_room_manager_instance` upon initialization to allow the `GameRoom` background thread to call its database methods.
  * Captures participating players' snapshots synchronously at the exact moment the round ends, preventing players who leave or disconnect during intermission from being omitted.
  * Added a reset loop in the standard room transition path to clear the stats of players in `room.past_players` (in addition to `room.players`), preventing old scores from leaking into future rounds.
