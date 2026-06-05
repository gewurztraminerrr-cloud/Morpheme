# Stable State Summary — June 4, 2026 (Start Over Point)

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | [LATEST_COMMIT_ID] | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | [LATEST_COMMIT_ID] / `snapshot-current` / `START_OVER_POINT_JUNE_4` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | [LATEST_COMMIT_ID] / `snapshot-current` | ✅ Fully Deployed & PM2 Reloaded |

**All environments are 100% synchronized at the latest commit [LATEST_COMMIT_ID].**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py`.
The active recovery points `START_OVER_POINT_JUNE_4` and `snapshot-current` tags have been successfully updated and pushed to GitHub.

---

## Serving Versions (cache-busted)

| File / Style | Version | Description |
|--------------|---------|-------------|
| `/js/play.js` | `v=135` | Implemented a consecutive invalid path warning popup (triggering on 4 consecutive guesses connecting on the board but dictionary-invalid) across all play modes. Fixed client-side validation message for too-short words to immediately show `[word] IS TOO SHORT (MIN: XL)` rather than flashing `INVALID` first. Fixed intermission finders button display/stale value bug. Prevented intermission tile press for the first 5 seconds of intermission. |
| `/js/app.js` | `v=39` | Standardized FAQ dictionary stats table styling: color cells to match headers (NWL #60a5fa, CSW #fbbf24, AW #c084fc, 16+ List #f87171). |
| `templates/index.html` | *Dynamic* | Styled dictionary stats table headers, added missing period to Vibrant Blue FAQ text, and bumped play.js cache-buster to `v=135` and app.js cache-buster to `v=39`. |

---

## Work Completed on June 4, 2026

### 1. Consecutive Invalid Guesses Warning Popup & Exceptions
* **Goal achieved:** Inform players when they repeatedly guess words that form valid connections on the board but are not in the dictionary, helping them realize if they are typing invalid words. Exclude already-found words and too-short sequences to prevent false warning counts.
* **Implementation (`static/js/play.js`):**
  * Added a tracker for consecutive invalid path guesses (`wrongGuessesOnBoardCount`).
  * If a player inputs a word that connects on the board but is dictionary-invalid, the counter increments. If the word is valid, the counter resets.
  * Added exceptions (using a third parameter `isSpecialSkip = true` to `recordGuessResult`): if a submitted sequence is too short (smaller than the minimum word length from room/spinner settings) or has already been found (by anyone in FCFS play, or by the current player in standard, tournament, and private match play), the counter is NOT incremented or reset.
  * When the counter hits 4, a warning popup is shown advising the user about their consecutive invalid guesses.
  * Integrated across standard play, tournament play, and private match play.

### 2. FAQ Dictionary Stats Styling & Color Coordination
* **Goal achieved:** Enhanced the design and legibility of the FAQ dictionary summary by matching column and cell colors with their respective dictionary definitions.
* **Implementation (`templates/index.html` & `static/js/app.js`):**
  * CSW column headers and totals styled in gold (`#fbbf24`) and 16+ List in red (`#f87171`) (swapping previous styling).
  * Color-coded individual cell counts within `loadFAQDictionaryStats` table cells to match the color themes of the headers.

### 3. Vibrant Blue FAQ Text Clarification
* **Goal achieved:** Updated the Vibrant Blue description under FAQ highlights to accurately mention that it highlights words personally discovered during the round, or words another player found if a username is selected. Added a trailing period for sentence consistency.

### 4. Test Scenario Unpacking Fix
* **Goal achieved:** Fixed `test_scenarios.py` `scenario_1` to match the updated tuple size returned by `BoardGenerator.generate_board()`.

### 5. Intermission Finders Button Display and Update Fix
* **Goal achieved:** Selecting a word under the "All Words" list now immediately displays the "Finders" bar (correctly showing who found it, or "Finders: None" if no one found it) without needing a secondary letter-filtering event.
* **Implementation (`static/js/play.js` & `templates/index.html`):**
  * Added the current selected word (`highlightedFoundWord`) to the intermission rendering cache keys (`currentRenderKey` and `lastRenderedIntermissionKey`).
  * When a user selects a word, the change in `highlightedFoundWord` invalidates the render cache key, forcing an immediate re-render of the words panel, which successfully updates and displays the finders button container.
  * Bumped `play.js` version tag to `v=130` in `templates/index.html` to bypass browser caches.

### 6. Intermission Board Press Delay (5 Seconds)
* **Goal achieved:** Prevent players from filtering/pressing letters on the board immediately when intermission starts, ensuring a 5-second grace period.
* **Implementation (`static/js/play.js`):**
  * Added a check inside `handleIntermissionTilePress` to verify if at least 5 seconds have elapsed in intermission.
  * Dynamically calculates elapsed time using `localEndTime` and intermission duration (5s for 24h rooms, 60s for standard rooms). If elapsed time is less than 5 seconds, the press event is ignored.
  * Bumped `play.js` version tag to `v=130` in `templates/index.html`.

### 7. Instant Zero-Hesitation Feedback for Too-Short Words
* **Goal achieved:** Eliminate the network latency delay before showing validation feedback when a user submits a word that is too short.
* **Implementation (`static/js/play.js`):**
  * Updated `submitWord` client-side validation logic: if the word's length is smaller than the minimum word length, it evaluates it locally immediately.
  * Shows `${word} IS TOO SHORT (MIN: ${minLen}L)` instantly and sets `optimisticColor = 'red'`.
  * The server confirms this status, preventing double-flashing and eliminating the validation delay completely.

### 8. Board Generator Compliance & Transition Lag Fixes
* **Goal achieved:** Eliminate visual hangs and server watchdog triggers during intermission-to-active transitions on low-target rooms.
* **Implementation (`board_generator.py`):**
  * Dynamically scaled the target word-embedding number based on `word_count_range` (`max_words`).
  * Introduced consonant-biased, vowel-reduced `fill_weights` for empty cell fills to prevent massive accidental connections, guaranteeing immediate compliance without decimation loops.

### 9. Dynamic Submission Word Self-Healing
* **Goal achieved:** Prevent valid words on the board (like "TANGLER") from being rejected due to server truncation to fit targets.
* **Implementation (`game_room.py`):**
  * Added self-healing fallback to `submit_word`: if a valid dictionary word physically found on the board is missing from `all_words`, it is dynamically accepted, scored, and added to the round statistics.

---

## Key Files Tracked

| File | Location | Purpose |
|------|----------|---------|
| `static/js/play.js` | Production + GitHub | Client-side guess warning popups, too-short feedback handling, finders button rendering, and delay checks. |
| `static/js/app.js` | Production + GitHub | Client-side FAQ table styling. |
| `templates/index.html` | Production + GitHub | Table headers, FAQ text updates, and script cache-busters. |
| `board_generator.py` | Production + GitHub | Dynamic board generation target scaling and sparse cell filling. |
| `game_room.py` | Production + GitHub | Dynamic word self-healing and submission acceptance logic. |

---

## Previous Save Points

* [June 3 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_june_3.md)
* [June 2 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_june_2.md)
* [June 1 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_june_1.md)
* [May 30 Stable State Summary](file:///Users/jeffbabiak/stable_state_summary_may_30.md)
