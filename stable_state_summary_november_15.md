# Stable State Summary: November 15, 2026

## Overview
- **Release / Branch**: `main`
- **Git Tags**:
  - `START_OVER_POINT_NOVEMBER_15`
  - `STABLE_STATE_NOVEMBER_15`
  - `START_OVER_POINT_TODAY`
  - `STABLE_STATE_TODAY`
  - `stable-2026-11-15`
  - `start-over`
  - `latest-stable-state`
  - `current-stable-state`
- **Synchronized Targets**:
  - `localhost:5001` (Development & App)
  - `GitHub: gewurztraminerrr-cloud/Morpheme` (`origin/main`)
  - `Production VPS (132.148.72.249)` (`https://morpheme.games/`)

---

## Key Milestones & Fixes Included in November 15 Stable Point

### 1. Mobile Top Black Gap / Viewport Shift Recovery After External Navigation
- **Files**: [`static/js/app.js`](file:///Users/jeffbabiak/static/js/app.js), [`static/css/play.css`](file:///Users/jeffbabiak/static/css/play.css), [`templates/index.html`](file:///Users/jeffbabiak/templates/index.html)
- **Summary**:
  - Fixed the black horizontal bar appearing along the top of mobile screens when returning from external links (such as the Wikipedia button in word definitions) via the mobile system Back button.
  - Added `pageshow`, `visibilitychange`, `focus`, and `visualViewport` resize/scroll listeners to reset `window.scrollTo(0, 0)` and resynchronize `--app-height` across staggered frames (`0ms`, `50ms`, `150ms`, `350ms`).
  - Configured Wikipedia links to open in an isolated new tab (`target="_blank" rel="noopener noreferrer"`) so the Morpheme document and viewport are never unloaded.

### 2. Spinner Set Odds Window Enforcement & Randomness Across All Modes
- **Files**: [`spinner_set.py`](file:///Users/jeffbabiak/spinner_set.py), [`game_room.py`](file:///Users/jeffbabiak/game_room.py), [`board_generator.py`](file:///Users/jeffbabiak/board_generator.py), [`tournament_logic.py`](file:///Users/jeffbabiak/tournament_logic.py)
- **Summary**:
  - Enforced the `300-400`, `400-500`, and `500+` word count floor across all `+ AW` dictionaries (`NWL + AW` and `CSW + AW`) in Game Rooms, Solo / With Friends, and Tournaments.
  - Prevented cached/fallback boards from overwriting randomly spun `SpinnerSet` parameters (format, min length, difficulty, dictionary) in public rooms while preserving fixed user-chosen parameters in Solo mode.
  - Removed the premature 10-second intermission fallback overwrite in `check_and_update_state()`.

### 3. Authentic Added Words (`+ AW`) & CSW-Only Density and Guarantee
- **Files**: [`board_generator.py`](file:///Users/jeffbabiak/board_generator.py), [`game_room.py`](file:///Users/jeffbabiak/game_room.py), [`app.py`](file:///Users/jeffbabiak/app.py), [`word_validator.py`](file:///Users/jeffbabiak/word_validator.py)
- **Summary**:
  - Made `+ AW` detection unconditional in `BoardGenerator._solve_board` so any call with `"+ AW"` in `dictionary` or `use_added_words=True` always traverses both the base dictionary trie and `_ONLY_ADDED_` trie.
  - Guaranteed `room.use_added_words`, `room.added_words`, and `room.csw_only_words` stay synchronized across round promotion, bonus-cell re-solves, and intermission auto-heal paths.
  - Improved Word Soup seeding and tile evaluation on `+ AW` and `CSW` boards so specialized vocabulary is authentically represented on the grid.
  - Removed `DIAPASE` and all 3,825 standard NWL/CSW duplicate words from `dictionaries/added_words.txt`, `dictionaries/added_words_duplicate.txt`, `self.added_words`, and `self.added_trie`.

### 4. Reverse Dictionary Unified Alphabetical Sorting
- **Files**: [`app.py`](file:///Users/jeffbabiak/app.py), [`static/js/tools.js`](file:///Users/jeffbabiak/static/js/tools.js)
- **Summary**:
  - Updated Reverse Dictionary sorting to `(len, word)` on both backend and frontend so every word length group is ordered in a single unified A–Z alphabetical sequence across all searches.

### 5. Audio Polish: Silent Gesture Unlock & Triple Music Intermission Gate
- **Files**: [`static/js/play.js`](file:///Users/jeffbabiak/static/js/play.js), [`templates/index.html`](file:///Users/jeffbabiak/templates/index.html)
- **Summary**:
  - Muted `window.intermissionBellAudio` (`muted = true`, `volume = 0`) during the mobile gesture unlock cycle on **ENTER LOBBY** so no brief 0:10 bell blip is heard when entering the lobby.
  - Fixed `updateTripleMusicState()` so `authoritativeUpcomingFormat` during intermission is read strictly from the upcoming round's revealed spinner parameters rather than `state.current_board_format` (which holds `previous_board_format` during intermission), preventing Triple music from re-triggering at `0:10`–`0:00` when the next round is `Normal`.
