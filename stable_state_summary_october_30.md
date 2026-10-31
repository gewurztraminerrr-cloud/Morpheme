# Stable State Summary — October 30, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_30
```
*(Also tagged as `stable-2026-10-30`)*

## Latest Commit ID
```
2255fa55c37bce9879acd3ed4713f2283fe30c5f
```
*(Short SHA: `2255fa55`)*

## Synchronization Status

| Environment | Status | Commit / Tag | Verification Details |
|:---|:---|:---|:---|
| **Localhost** | ✅ Synchronized | `2255fa55` | Working tree on `main`, zero uncommitted code changes |
| **GitHub (`origin/main`)** | ✅ Synchronized | `2255fa55` | Pushed to `origin/main` & tags `START_OVER_POINT_OCTOBER_30` / `stable-2026-10-30` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | `2255fa55` | Reset to latest origin/main, PM2 online, Waitress bound, live HTTP/2 200 OK |
| **App (Mobile WebView)** | ✅ Synchronized | `https://morpheme.games/` | Flutter app loads live synchronized webview with native audio/haptic bridges |

---

## Session Highlights & Architectural Updates (October 25 – October 30, 2026)

### 1. Dictionary & Lexicographical Synchronization
- **`AEROMONIASES` Addition**:
  - Definition: `plural of aeromoniasis ((pathology) infection with bacteria of the genus Aeromonas.)`
  - Synchronized across `dictionaries/added_words.txt`, `dictionaries/added_words_dates.txt` (date: `25/10/2026`), `dictionaries/added_words_duplicate.txt`, `dictionaries/Definitions.txt`, `dictionaries/wikdefs.txt`, and `dictionaries/wikdefs_duplicate.txt`.
  - Updated `dictionaries/word_stats.json` with latest statistics.
  - Synchronized SQLite database `morpheme.db` (`wiktionary_definitions` table) on both localhost and remote production environments.
  - Preserved alphabetical ordering and index synchronization.

### 2. Client Submission & Optimistic Word Validation
- **Removal of False Optimistic Red Flash (`static/js/play.js`)**:
  - Eliminated erroneous red flash feedback for valid words that temporarily were not in the client-side pre-generated word list.
  - Server-authoritative validation cleanly handles word validation without optimistic client misclassification.

### 3. Bonus Letter Format & Tile Rendering
- **Bonus Cell Guarantee at Round Promotion (`game_room.py`)**:
  - Enforced deterministic bonus cell generation on round promotion in Bonus Letter format games.
- **Intermission Bonus Tile Visibility (`static/js/play.js`)**:
  - Fixed issue where the bonus letter tile was hidden during round intermission.
  - Ensured the bonus tile renders reliably whenever `bonus_cell` matches during both active play and intermission phases.

### 4. Round Timing & Intermission Transition
- **Exact Intermission 0:00 Transition (`game_room.py`, `static/js/play.js`)**:
  - Synchronized round start timer to launch the subsequent round exactly at `0:00` intermission expiration, eliminating phase drift.

### 5. 24h & Non-Grid Room Engine Robustness
- **Midnight Transition Fix (`game_room.py`)**:
  - Resolved midnight `start_next_round` retry loop caused by `UnboundLocalError` on copy operations.
  - Guarded post-promotion failures to ensure unbroken continuity across daily resets.
- **Local Import & Dimension Parsing**:
  - Fixed `calculate_word_score` local import scoping.
  - Implemented safe dimension parsing for non-grid rooms.

### 6. Scoring Pre-Calculation & Word Value Sorting
- **In-Game Room Word Sorting (`static/js/play.js`)**:
  - Accurately pre-calculated scores and sorted found words by descending point value in game rooms.
- **Intermission Bonus Formula Equations**:
  - Guaranteed that all bonus words and bonus letter words display full bonus value equation breakdowns under "All Words" during intermission.

### 7. Definition Lookups & Proper Noun Filtering
- **Curated Dictionary Guard (`app.py`, `word_validator.py`)**:
  - Prevented false-positive proper noun filtering on valid curated dictionary entries such as `FIGURANT`, `DANSEUR`, and `ARMBAND`.

### 8. Game Room Audio & Autoplay Recovery
- **Triple Word Audio Playback (`static/js/play.js`)**:
  - Enabled `triple.mp3` sound playback in game rooms upon finding triple-value or high-tier words.
  - Added user gesture autoplay recovery so audio permissions seamlessly initialize on first interaction across desktop and mobile browsers.

### 9. Intermission Default View & Player Notepads
- **Intermission "VIEW ALL" Default (`static/js/play.js`)**:
  - At the start of intermission in all game rooms, the middle player control button below "Players" automatically defaults to "VIEW ALL".
- **Dictionary Source Tags in Notepads (`templates/index.html`, `static/css/play.css`, `static/js/play.js`)**:
  - In player notepads during intermission, dictionary tags (`NWL`, `CSW`, or stacked vertically if both) are displayed in small, crisp typography directly next to the point value of each word.

### 10. Achievements UI Refinement
- **Exceptional Performances Cleanup (`templates/index.html`)**:
  - Removed the `AVERAGE PERF` metric from the header banner of "EXCEPTIONAL PERFORMANCES" in every user's Achievements window.

### 11. Header Parameters Display
- **Time Unit Formatting & Bold Parameters (`templates/index.html`, `static/css/play.css`)**:
  - Replaced seconds with standard units in room headers: `86400s` -> `24h`, `600s` -> `10m`, and `180s` -> `3m`.
  - Styled parameter values below room type (e.g., `Accumulative`, `4x4`, `45s`) with bold font weight while preserving thin, un-bolded pipe dividers.

### 12. Remaining Tab Formula & Responsive Legend
- **Row Arithmetic Display (`static/js/play.js`, `static/css/play.css`)**:
  - Formatted each row in Remaining (all public and 24h rooms) as:
    `7LW                    51 + 49 = 100  (51%)`
    with left-aligned `.len-cell`, right-aligned `.count-cell`, and two non-breaking spaces before `(51%)`.
- **Responsive Vector Legend (`templates/index.html`, `static/css/play.css`)**:
  - Above the Remaining table, displays the explanatory legend:
    `[found] + [missed] = [total words]  [percent found]`
  - Built as an inline responsive vector SVG (`viewBox="0 0 380 16"` with `width: 100%`) that automatically stretches to match the table margins across all devices.
  - Scaled with safety padding so the beginning `[` and ending `]` brackets never clip or spill outside the panel on any mobile device, tablet, laptop, or desktop display.

### 13. Lobby Sliding Drawer Equal Width Layout
- **Equal Horizontal Width for Players in Lobby & Lobby Chat (`static/css/lobby.css`)**:
  - Updated `.lobby-slide-body` on desktops and laptops to `grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)`.
  - Increased horizontal space for the `PLAYERS IN LOBBY` panel of online users so user chips spread out across 50% of the drawer.
  - Decreased the horizontal width of `Lobby Chat` messages to match, making the horizontal length of both elements exactly equal.

### 14. Closed Rooms Default Sorting & Proximity Activation
- **Closed Rooms Default Order (`static/js/lobby.js`)**:
  - When the **Closed Rooms** button is clicked, it defaults to showing rooms with the **highest average rating first**.
  - If the user explicitly presses **Enter** in the "Sort rooms by proximity..." textbox (or clicks **My Rating**), the closed rooms then sort by proximity to that average rating value.
  - Switching back to the Closed Rooms tab resets the sorting back to highest average rating first.

### 15. Tools: Personal Timer Pause/Resume, Mobile Stacked Buttons & Scroll Stability
- **Pause & Resume Controls (`templates/index.html`, `static/js/tools.js`, `static/css/play.css`)**:
  - Added a Pause/Resume button to Personal Timer styled with the identical blue gradient (`linear-gradient(135deg, #00d2ff, #3a7bd5)`), border radius, box shadow, and typography as Start Timer.
  - On desktop and laptops, Pause and Stop Timer appear side by side.
  - On mobile devices, buttons stack vertically in a single column (`flex-direction: column !important; width: 100% !important;`), placing each button on its own individual row on top of each other with large, readable font.
  - Explicitly styled `#timer-stop-btn` with red gradient (`#ff4d4d` to `#d43f3f`) and Pause/Resume/Start with blue gradient (`#00d2ff` to `#3a7bd5`), preventing generic theme overrides from washing out button colors.
  - Clicking Pause freezes countdown interval, preserves remaining seconds, and toggles button in-place to "Resume".
  - Clicking Resume restarts countdown from the remaining time and toggles button in-place back to "Pause".
  - Clicking Stop Timer resets the timer, hides Pause/Resume and Stop, and reveals Start Timer.
- **Mobile Scroll Stability (`templates/index.html`, `static/css/play.css`)**:
  - Moved `.mobile-bottom-nav.tools-bottom-nav` inside `.tools-content-column` so `.tools-split-layout` contains exactly 2 panels instead of a phantom 3rd pane.
  - Added `touch-action: pan-y !important;` and `overscroll-behavior-x: none !important;` to `.tools-content`, `.tools-content-column`, `.tool-pane`, and `#tool-timer`.
  - Added strict `box-sizing: border-box !important;` and `overflow-x: hidden !important;` across `#tool-timer` and `#timer-display-container`, preventing any horizontal jitter, shift, or wobble when scrolling vertically on mobile devices.

### 16. Tools: Find Count Random Words Table Mobile Layout Fix
- **Mobile Row-by-Row Display & Vertical Elongation Fix (`templates/index.html`, `static/css/play.css`)**:
  - Removed accidental `history-table` class from `.random-words-table` in `templates/index.html`. Previously, `.history-table` forced mobile columns to `width: 22% !important; word-break: break-word !important;`, which caused words to break character-by-character vertically into a single-pixel column, elongating the table by hundreds of pixels vertically.
  - On mobile devices (`@media (max-width: 900px)` and `body.is-mobile`), random words run cleanly row-by-row (`flex-direction: column !important; width: 100% !important;`), with each word centered on its own individual row with a subtle divider line, generous touch target padding (`10px 12px`), and clear font size (`0.95rem`), completely eliminating any vertical squishing or character breaking.
  - On desktops and laptops (`@media (min-width: 901px)` and `body:not(.is-mobile)`), random words continue running horizontally in a single bar side-by-side (`flex-direction: row !important; flex: 1 1 0 !important;`).

---

## Verification & Validation

| Test / Check | Result | Verification Notes |
|:---|:---|:---|
| **Local Working Tree** | ✅ Clean | Clean working tree on `main` at `c427827d` |
| **Origin / GitHub** | ✅ Synchronized | Synchronized with `origin/main` at `c427827d` |
| **Remote Production** | ✅ Synchronized | Deployed to `132.148.72.249` at `c427827d`, PM2 online, HTTP/2 200 OK |
| **Mobile App (WebView)** | ✅ Synchronized | Points directly to `https://morpheme.games/` |
| **Dictionary Line Count & Integrity** | ✅ Verified | Byte-exact synchronization of dictionary text files and SQLite databases |
| **Git Tags** | ✅ Tagged | `START_OVER_POINT_OCTOBER_30` and `stable-2026-10-30` pointing to `c427827d` |
