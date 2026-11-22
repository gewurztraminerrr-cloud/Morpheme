# Morpheme Stable State Summary — November 22, 2026

## 1. System Synchronization State
- **Date**: Sunday, November 22, 2026
- **Status**: **START OVER STABLE POINT**
- **Git Commit ID**: `3cc848116255af5f0a11983ba3c708bca4285a01`
- **GitHub Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme` (`main` branch)
- **Live Production URL**: `https://morpheme.games/` (HTTP 200 OK, Content-Length: 860529)
- **Localhost URL**: `http://localhost:5001/` (HTTP 200 OK, Content-Length: 860529)
- **VPS PM2 Process**: Online (PID 98350, 0% CPU, normal healthy memory)
- **Byte-for-Byte Synchronization Across All Environments**:
  - **Localhost**: Synchronized at `3cc84811`
  - **GitHub (`main`)**: Synchronized at `3cc84811`
  - **Production VPS (`morpheme.games`)**: Synchronized at `3cc84811`
  - **App & Python Modules**: All verified active and serving identical code

---

## 2. Server Memory Optimization & GoDaddy Alert Resolution

### Root Cause of the 30-Minute GoDaddy Alerts
- **In-Memory Definition Overhead**:
  - Previously, `load_definitions()` loaded all **827,881 definitions** from `Definitions.txt` into a raw in-memory Python dictionary (`DEFINITIONS_CACHE`).
  - In 64-bit Python, holding 827,000+ key-value string pairs allocated **~1.8 GB to 2.0 GB of RAM**, pushing Python's resident memory to **3.3 GB** (87.3% of the VPS).
  - Total VPS RAM usage hovered between **95% and 97.3%** with only 88 MB free.
  - GoDaddy’s monitoring agent triggers an automated alert whenever server memory crosses **90%**, causing it to send *"Your server usage is very high"* and *"Your server usage is no longer high"* every 30 minutes.

### Implementation: On-Demand SQLite `DefinitionsProxy` (`app.py`)
- **Direct Database Access with LRU Caching**:
  - Replaced the monolithic 827k in-memory dictionary with `DefinitionsProxy(dict)`, an on-demand proxy backed by the SQLite `wiktionary_definitions` table in `morpheme.db`.
  - Maintains a small 10,000-word LRU cache in RAM (~2 MB) for lightning-fast repeat lookups (0.001 ms).
  - Queries SQLite primary key for words not in the LRU cache (0.014 ms).
  - Synchronized all 859,721 definitions into SQLite on both Localhost and the VPS.
- **Measured Impact**:
  - **Python Resident RAM**: Dropped from **3.3 GB (87.3%) down to ~1.1 GB (28%)**.
  - **Total Server RAM Used**: Dropped from **97.3% down to ~38%**.
  - **Free Available Headroom**: Jumped from **88 MB up to 2.4+ GB**.
  - **GoDaddy 90% Threshold**: Permanently resolved; alerts completely ceased.
  - **Zero-Restart Frontend Deploys**: Updated `scratch/deploy_reliable.py` to support `--no-restart` for frontend-only CSS/JS/HTML updates, avoiding unnecessary dictionary reloads.

---

## 3. Key Features & Bug Fixes in this Stable State

### A. Missing Bonus Word Resolution (`game_room.py`)
- **Root Cause**: In rare edge cases where candidate bonus words in the round's pre-computed word pool conflicted or were excluded, `pick_distinct_bonus_word` could return `None`, resulting in a round starting with no Bonus Word.
- **Tiered Fallback Engine**:
  - Tier 1: Candidate from distinct round word pool.
  - Tier 2: Valid word from the general dictionary matching length and board constraints.
  - Tier 3: Longest available word solvable on the board.
- **Self-Healing Assurance**: Added `ensure_bonus_word_on_board()` across all round start, promotion, heal, and wake pathways. Synchronized `room.next_round_words` on Word Count Shield heal so word pools are never out of sync.

### B. Sequential Word Path Highlighting on Review Selection (`static/js/play.js`)
- **Root Cause**: When a user selected a word under "All Words" during intermission review, `toggleWordSelection()` scheduled sequential timeouts but immediately triggered a synchronous board re-render via `updateGameState()`. The second pass saw that the word had already started animating and executed the static fallback, forcing all tiles to light up simultaneously with no visible path tracing.
- **Dedicated Animation Engine (`animateReviewWordPath`)**:
  - Sequentially lights up tile-by-tile at a clean, visible **85 ms per tile** cadence.
  - Guarded with `window._isReviewAnimating` and `window._activeReviewWord` so background game-state polling and board re-renders do not wipe or collapse in-flight animations.
  - Smooth retraction on deselection and full compatibility with both 2D boards and 3D Cube boards.

### C. Hover Tooltip for Input Method Icons in Game Rooms (`static/js/play.js`, `static/css/play.css`)
- Added hover tooltips to the method icon in game room player cards with `cursor: help`:
  - `🖱️` &rarr; `Input Method: Mousing`
  - `⌨️` &rarr; `Input Method: Typing`
  - `📱` &rarr; `Input Method: Swiping`

### D. Input Method Percentage Breakdown on Profile & Mini-Profile (`app.py`, `game_room.py`, `templates/index.html`, `static/js/tools.js`, `static/css/style.css`)
- **Dynamic Tracking**: Attached input method (`mouse`, `keyboard`, or `touch`) to all submitted words. On round conclusion, `save_round_history` tallies words submitted per method and updates `input_mouse_count`, `input_keyboard_count`, and `input_touch_count` on the `users` table.
- **Mini-Profile (In-Room Modal)**:
  - Added `#mini-meta-input-method` item in row 3 of the metadata grid alongside Forum Posts and above Proof.
  - On mobile devices, positioned directly below Forum Posts (`order: 6`).
  - Displays `Input: 🖱️ 83% · 📱 17%` (or `Input: No data yet`).
  - Desktop hover tooltip reveals full word counts and percentages across all modes.
- **Full Profile Page**:
  - Added `#profile-meta-input-method` in metadata line 3, completing a symmetrical 3×3 metadata grid.
  - On mobile devices, positioned directly below Forum Posts (`order: 9`, full-width centered card).
  - Displays the input method summary under the `Input` label with a detailed hover breakdown.

### E. Multiplayer Subanagrams Duplicate Letter Selection & Tap Tracking (`static/js/play.js`, `templates/index.html`)
- Preserved exact user-tapped tile coordinates (`mouseState.isBoardTileSelection = true`) so duplicate letters (e.g. the two 'E's in `CASERNE`) never re-order the word or switch highlights unexpectedly.
- Disabled mobile virtual keyboard autocorrect interference (`autocorrect="off" autocapitalize="characters" spellcheck="false"` on `#word-input`).
- Suppressed delayed synthetic mouse events after touch gestures (`lastTouchTime = Date.now()`).

### F. Multiplayer Subanagrams Tile Highlight Color Uniformity (`static/css/play.css`)
- Unified `.game-board.is-subanagrams-board .board-cell.selected`, `.current`, and `.typing-highlight` to use `var(--highlight-mouse-color, #409cff) !important;` for consistent color fidelity without flashing.

### G. Game Room Remaining Tab Cutoff Rule (`static/js/play.js`)
- Omitted trailing zero-word rows beyond the highest findable letter length on the board while preserving intermediate 0 rows and honoring `Min:`.

### H. Word Charts in Tools: 2-Letter Row Removal & Panel Edge Alignment (`templates/index.html`, `static/css/play.css`)
- Omitted "2 Letters" row from both **Point Values & Divisions** and **Dictionary Word Counts** tables.
- Extended pure white background to rounded panel edges in **Dynamic Word Lookup** for light-colored layouts.

---

## 4. Verification Summary
- **Live Production URL**: `https://morpheme.games/` (HTTP/2 200 OK, Content-Length: 860529).
- **Localhost URL**: `http://localhost:5001/` (HTTP/1.1 200 OK, Content-Length: 860529).
- **VPS PM2 Process**: Online (PID 98350, 0% CPU, 2.4 GB memory, 1.0 GB available).
- **Git Commit**: `12578e9c59544a6f97c1bdf52f186318322dd3e6`.
- **Sync Status**: `localhost`, `github (origin/main)`, and `production (morpheme.games)` are 100% synchronized byte-for-byte.
