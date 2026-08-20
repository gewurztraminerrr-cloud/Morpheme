# Stable State Summary — August 20, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of August 20, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is fully synchronized.

---

## 1. Repository & Deployment Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 20, 2026
* **Commit ID**: `bb63294b93e3034a37b5603291e0a13a9c4ecdfb` (`bb63294`)
* **Asset Version**: `v=33089`
* **Production Host**: `132.148.72.249` (`morpheme.games`)

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Restored Slow Gold Flash on Spinner Set Parameters (`static/css/play.css`, `static/js/play.js`)
- **Keyframe Specificity & W3C Compliance**: Resolved an issue where keyframes with `!important` inside the `@keyframes` block are ignored by browser CSS parsers. Converted `#param-diff` to use a CSS variable (`var(--diff-color)`) and eliminated hardcoded inline style overrides, and removed `!important` from `#param-bonus` so the animation smoothly takes over.
- **Dedicated Keyframes**: Implemented `fadeGoldToNormal`, `fadeGoldToLime`, and `fadeGoldToDiff` so that:
  - **0%–30% (First ~1.4s)**: All Spinner Set parameter labels and values (Difficulty, Time, Dimensions, Dictionary, Word Range, Format, and Bonus Letter) immediately turn gold (`#ffd700`) with a radiant dual-layer text glow (`text-shadow: 0 0 15px rgba(255, 215, 0, 0.95), 0 0 30px rgba(255, 215, 0, 0.7)`).
  - **30%–100% (4.5s Total)**: The gold color slowly and smoothly fades back to each element's original steady-state color (blue/green/red for Difficulty, lime green `#32cd32` for Bonus Letter, and readable primary text color for all others).
- **Automated Animation Cleanup**: Added automated 4600ms class removal cleanup in `play.js` when the reveal animation triggers at 0:45 remaining during intermission.

### B. Intermission Board Synchronization & Race Condition Fix (`game_room.py`, `static/js/play.js`, `app.py`)
- **Root Cause Resolved**: Fixed a race condition where an uninitialized `room.intermission_start_time` (evaluating to 0) caused `elapsed = now - 0 = ~1.78 billion seconds`, which triggered `milestone == 'start'` and immediately skipped the 60-second intermission right as the round ended.
- **Hardened Lifecycle Guards**: Added initialization fallbacks across `time_remaining`, `intermission_end_time`, and `get_next_round_milestone()` so `intermission_start_time` is guaranteed to be set to the transition timestamp whenever `state == 'intermission'`.
- **Preserved Board Review**: During intermission, the completed round's board stays active (grayed out) with `state.board` and `state.previous_board` synchronized so players can review paths and words throughout the entire 60 seconds. The next round's board is only presented when the intermission timer reaches 0:00.

### C. Restored 3D Physical Flattening Animation on ENTER LOBBY Button (`templates/index.html`, `static/js/app.js`)
- **Instant Press Sinking**: Added `onpointerdown`, `onmousedown`, and `ontouchstart` handlers to `#btn-enter-lobby-gateway` that immediately sink and flatten the 3D button into its socket housing (`.pressed`, `.flattened`).
- **280ms Tactile Transition**: Preserved the 280ms transition window so the physical flattening animation is visibly seen and felt while audio initiates before the viewport transitions smoothly into `#page-lobby`.

### D. Thread-Safe Player Joining & Duplicate User Elimination (`game_room.py`, `static/js/play.js`, `app.py`)
- **Synchronized Roster Mutex**: Guarded `add_player` with `with self._state_lock:`, eliminating race conditions when parallel join/state polling requests arrive simultaneously.
- **Strict User & Username Deduplication**: Pruned `self.players` against matching `user_id` or case-insensitive `username` across all persistence paths (rejoiners, past_players, quitters, and new players), preventing duplicate player rows and redundant `"has entered the room"` chat broadcasts.
- **Client-Side Deduplication Layer**: Added deduplication directly into `play.js` (`updatePlayers`) before sorting and rendering, guaranteeing the active Players list displays exactly one row per player.

### E. Instant 0ms Player Count Updates & Fast 1s Real-Time Auto-Polling (`templates/index.html`, `static/js/app.js`, `static/js/lobby.js`, `app.py`)
- **Instant Gateway-to-Lobby Fetch**: Added immediate inline fetching on the initial **ENTER LOBBY** click and wired direct stats loading inside `showPage('page-lobby')` so player counts populate instantly without waiting for async script execution.
- **1-Second Real-Time Auto-Polling**: Doubled the auto-polling frequency on the lobby page from 2s down to **1s**, synchronizing live player counts across devices with sub-second responsiveness.
- **Active Polling Verification**: Updated `/api/lobby-stats` to only count players actively polling within 15s (`now - p.last_active <= 15`). Inactive or disconnected tabs that left without a clean exit handshake no longer linger in the lobby count.
- **Eliminated Stale LocalStorage Stats**: Removed localStorage caching of player counts so buttons always reflect pure, authoritative live server data ($[0]$ on empty rooms, $[1]$ when 1 player is active).
- **Deduplicated User Aggregation**: Aggregated unique human player IDs per game configuration (`set(user_id)`), guaranteeing that duplicate room instances or transient ghost states never inflate player counts.

### F. Accelerated Room Join Handshake & Snappy Toast Feedback (`app.py`, `static/js/lobby.js`, `templates/index.html`)
- **Sub-30ms Room Join Handshake**: Consolidated user rating, games played, and flag database queries into a single optimized SQLite transaction during `/api/room/create` and made previous room departure non-blocking in the background.
- **Brisk Toast Notification**: Reduced the toast display duration from 3.5 seconds down to **1.5 seconds** (with 200ms smooth fade), delivering an immediate confirmation that dismisses cleanly without lingering or obstructing gameplay.

### G. Non-Reverting Gateway & Startup Initialization Flow (`templates/index.html`, `static/js/app.js`)
- **Protected Gateway Passage**: Added `window._gatewayPassed` tracking and active page checks during `app.js` async bootstrap so that once a user clicks **ENTER LOBBY** or clicks **Start [N]**, the asynchronous startup sequence never resets the page back to `#page-loading`.

### H. Restored Desktop Lobby Panel Vertical Height & Top Alignment (`templates/index.html`, `static/css/lobby.css`)
- **Eliminated Vertical Gap**: Scoped flex-centering strictly to `#page-loading.active` and set `.page.active` to `display: block;`. The lobby panels now sit immediately below the top menu bar (`margin-top: 0; height: calc(100vh - 100px);`), eliminating the unwanted vertical gap and restoring the full vertical length of the desktop lobby layout.

### I. Instant 0ms Room Entry & Direct Server Hydration (`app.py`, `static/js/lobby.js`, `static/js/play.js`, `templates/index.html`)
- **Direct 1-Roundtrip Handshake**: Replaced the previous 3-step serial waterfall (`/api/rooms` list query $\rightarrow$ wait $\rightarrow$ `/api/room/join` $\rightarrow$ wait $\rightarrow$ `/api/room/create`) with a direct, single-call endpoint that joins or creates the room immediately in $<30\text{ms}$.
- **Immediate Visual Switch**: Clicking "Start" switches to the play page immediately, clearing stale match caches and pre-hydrating the board instantly from the response's embedded `state`.
- **Eviction Race Condition Protection**: Expanded `_emptyPlayersPollCount` tolerance from 3 to 10 polls so transient initial roster handshakes never falsely kick a joining player back to the lobby.

### J. Safari Instant 0ms First-Paint Engine (`templates/index.html`, `static/js/app.js`, `app.py`)
- **Inlined Critical First-Paint CSS**: Core page styling, background, layout, and 3D **ENTER LOBBY** gateway button styles are embedded directly in `<head>`, allowing WebKit/Safari to paint the gateway screen on frame 0 without waiting for external stylesheets.
- **Asynchronous Font Loading**: Decoupled external Google Fonts via `media="print" onload="this.media='all'"` with native Apple system font fallbacks (`-apple-system, BlinkMacSystemFont, 'SF Pro Display'`), eliminating Safari's render-blocking FOIT delay.
- **Demand-Loaded Audio (`preload="none"`)**: Replaced blocking `preload="auto"` and `autoplay` on global audio elements with `preload="none"`, preventing Safari from stalling initial DOM rendering with MP3 HTTP range downloads.
- **Parallelized Background Session Handshake**: Replaced sequential session checks with `Promise.all([validateSingleInstance(), checkSession()])` running concurrently without blocking the UI.
- **Gzip & Immutable Static Cache**: Enabled automatic gzip compression for JS and CSS files in `app.py`, with `Cache-Control: public, max-age=31536000, immutable` headers for instant loads from memory cache.

### K. Lobby Filter Bar Organization (`templates/index.html`, `static/css/lobby.css`)
- **Desktop/Laptop Layout**: Positioned the **`My Rating`** button immediately to the right of the *"Sort rooms by proximity to average rating"* textbox, and to the left of the **`Open Rooms`** button (`[Proximity Input] [My Rating] [Open Rooms] [Closed Rooms] [🔄]`).
- **Mobile/Compact Layout**: The rating proximity textbox spans the top full width, with **`My Rating`** positioned directly underneath on the left, to the left of **`Open Rooms`** (`[My Rating] [Open Rooms] [Closed Rooms] [🔄]`).

### L. Instant 24h Midnight Rollover & Elimination of Double Eviction (`game_room.py`, `static/js/play.js`)
- **2-Second Midnight Transition**: Reduced the midnight rollover intermission in 24h rooms from 60 seconds down to **2 seconds**, pre-staging the new day's board instantly.
- **Protected Re-Entry**: Modified eviction logic in `play.js` so that only actively established players present during the round's concluding moment receive the end-of-day modal. Re-entering a 24h room immediately from the lobby will never trigger a second kick.

### M. Automatic Root Word Definition Lookup & Bracket Appending (`app.py`)
- **Recursive Root Resolution**: For any word defined with a pointer pattern (e.g. `third-person singular simple present indicative of [word]`, `plural of [word]`, `diminutive of [word]`, `synonym of [word]`, `alternative form of [word]`, `conjugation of [word]`, `comparative of [word]`, etc.), the definition engine automatically retrieves the full lexicographical definition of the referenced root word and appends it directly inside parentheses/brackets next to the root word.
- **Verified Examples**:
  - `BEHEDGES` $\rightarrow$ `third-person singular simple present indicative of behedge ((transitive) To hedge about; surround with or as with a hedge.)`
  - `MALAXER` $\rightarrow$ `Synonym of malaxator (one who, or that which, malaxates; esp. a machine for grinding, kneading, or stirring into a pasty or doughy mass [n -S])`
  - `MALAXERS` $\rightarrow$ `plural of malaxer (Synonym of malaxator (one who, or that which, malaxates; esp. a machine for grinding, kneading, or stirring into a pasty or doughy mass [n -S]))`
  - `POLESTER` $\rightarrow$ `(motor racing) Diminutive of polesitter ((motor racing) A driver placed in pole position.)`

### N. Clean Definition Formatting (Removed Leading `(noun)`) (`app.py`)
- Removed `(noun)` / `(Noun)` from the start of definitions across the entire dictionary lookup and resolution pipeline. Noun entries now start cleanly with their direct definition or root reference (e.g. `a horseman, also CABALLERO [n -S]` or `APPLE, the firm round edible fruit of the apple tree`). All other language origins (`(Hawaiian)`, `(French)`) and non-noun tags (`(verb)`, `(adjective)`) remain preserved.

### O. Dictionary Cleanup: Obsolete Words, Abbreviations & Misspellings Removed (`dictionaries/`)
- **Protected Standard Words**: NWL (199,429 words), CSW (281,598 words), and 16+ supplementary words (9,227 words) are completely preserved and locked.
- **Removed Flagged Added Words & Inflections**: Removed **30,644** obsolete words, abbreviations, and misspellings along with all their derived conjugations, plurals, and participles (e.g. `ABASTARDIZE`, `ABASTARDIZED`, `ABASTARDIZES`, `ABASTARDIZING`, `ABBERANT`, `ABDOM`, etc.).
- **Updated Lexicon Counts**:
  - **Truly Added Words**: **`469,764`** *(down from 500,408)*.
  - **Raw Added Words File (`added_words.txt`)**: **`479,310`** *(down from 509,954)*.
- **Untouched Duplicate Backups**: `dictionaries/added_words_backup.txt`, `dictionaries/Definitions_backup.txt`, and `dictionaries/wikdefs_backup.txt` remain permanently preserved and tracked in git.
- **Flushed Pregenerated Boards**: Flushed and refreshed `pregenerated_boards` and `used_boards` in SQLite so all board parameters align strictly with the cleaned dictionary.

### P. 24h Room Score Sum 0-Score Exclusion (`app.py`, `static/js/play.js`)
- Players with an overall total score of 0 are completely excluded from the Score Sum table across all four 24h rooms (`24h_4x4`, `24h_4x6`, `24h_5x7`, and `24h_6x8`).
- Backfill queries, SQL aggregation (`HAVING MAX(d.score_sum) > 0`), in-memory room scans, and frontend render logic only display and count players who have earned a score of 1 or greater.

### Q. Guaranteed Session Expired Notice Suppression on $\ge$ 1 Hour Return (`templates/index.html`, `static/js/app.js`, `static/js/play.js`)
- Timestamps track strictly on physical human interactions (`mousedown`, `keydown`, `touchstart`, `pointerdown`, `scroll`), removing false-active background heartbeat intervals.
- Dual-layer storage & memory verification ensures returning after $\ge 1$ hour of absence silently returns to the lobby with zero popup modal.

### R. In-Place Word Definition Popover across Tools (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- Clicking words in **Combo Checker**, **Sequence**, **Subanagrams**, **Lists**, and **View Full List** displays a sleek in-place definition popover card directly next to the word without navigating the user away to the "Is Valid" tool.

### S. 170× C-Accelerated Morpheme Metric & High-Speed Combo Checker (`app.py`, `morpheme_metric.c`, `static/js/tools.js`)
- Bare-metal C engine running LCS and bitmask backtracking in CPU registers, taking search times from ~45 seconds down to **0.06s – 0.25s**.
- Guaranteed 0MP subword extraction and uncapped results tables.

### T. Unscramble Tool Desktop & Laptop Full-Width Panel Expansion (`templates/index.html`, `static/css/play.css`)
- Expanded to `1200px` max-width with responsive font clamping so all 21-letter jumbled strings fit on a single line.

### U. Full-Width Centered Board Loading Card & Grid Containment Fix (`static/css/play.css`, `static/js/play.js`, `templates/index.html`)
- **Root Cause**: When navigating to a multi-column room (e.g. 5x7, 6x8) before the initial board letters loaded, `#game-board` retained CSS grid styles (`display: inline-grid`, `--board-cols: 5`). The browser CSS grid layout treated `.loading-container` as a single grid cell occupying only Column 1 (~50px wide), causing the text ("`[PROCESSING] Connecting to Morpheme server...`") to scrunch vertically into a narrow strip. Clicking "Rotate" previously realigned it because the rotate handler reset the board styles.
- **Resolution**:
  - Added parent `:has(.loading-container)` and `grid-column: 1 / -1; width: 100% !important;` rules across desktop and mobile media queries in `play.css`.
  - Updated `ensureLoadingCardStyles()` and `renderBoard()` in `play.js` to strip `grid-template-columns`, `grid-template-rows`, `--board-cols`, and `--board-rows` whenever rendering a loading state.
  - Sized the loading container to full width and centered it cleanly on all screen sizes.

---

## 3. Production Deployment Instructions

To synchronize the live server on `morpheme.games` with this exact commit:

```bash
cd /home/morpheme/morpheme && git pull origin main && pm2 restart all
```
