# Stable State Summary — November 7, 2026 (End of Day)

## Latest Feature Commit ID
```
458df506 Ensure equal padding on left and right sides of Multiplayer Subanagrams letter sequence
```
*(Full hash: `458df506a7cbb0fa8c26bb437b01b6a22f36f6d0`)*

## Start Over Point Tag
```
START_OVER_POINT_NOVEMBER_7
```
*(Also tagged as `stable-2026-11-07`)*

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| localhost      | ✅ Synchronized | Working tree clean, HEAD at `458df506` |
| GitHub (main)  | ✅ Synchronized | Pushed to `origin/main` |
| morpheme.games | ✅ Synchronized | Deployed to 132.148.72.249 (PM2 online, HTTP 200 OK) |
| App / Client   | ✅ Synchronized | Verified on desktop, laptop, mobile layouts |

---

## Session Features & Fixes (November 7, 2026)

### 1. Split Points Room Join & Return-to-Game Flow
- **Join Button & Return to Game Button Synchronization**:
  - Resolved race condition where joining a Split Points room left the button disabled without navigating or left the user stuck on the lobby page.
  - Ensured state transitions cleanly re-enable navigation and return players directly to active games.

### 2. FCFS and Split Points Blue Notepads: Fully Visible Paging Arrows
- **Right Padding & Paging Controls**:
  - Fixed clipping of navigation/paging arrows on the right side of the blue notepads in both Split Points and FCFS rooms.
  - Adjusted internal notepad container padding (`padding-right: 28px`), flexbox bounds, and positioning so the right-side arrows are 100% visible and easily tappable across all screen sizes (mobile, tablet, laptop, desktop).

### 3. Intermission Custom Scroller and Thumb for Blue Notepads
- **Custom Scroller Track & Thumb**:
  - Added a visible custom scrollbar track (`.m-scroller-track`) and draggable thumb (`.m-scroller-thumb`) to the intermission results page in both Split Points and FCFS rooms.
  - Implemented smooth vertical scrolling with full mouse drag, wheel scroll, and mobile touch panning support across user notepads.
  - Automatically calculates viewport-to-content proportions and updates thumb height and position dynamically.

### 4. AW Definitions & Inflections Performance Optimization (76x Faster)
- **O(1) Set Lookups**:
  - Replaced expensive set union operations (`nwl | csw | long | added` of >1,000,000 strings per lookup) with direct O(1) membership checks (`is_word_in_any_dict`).
- **Thread-Local SQLite Connection Pooling**:
  - Implemented `threading.local()` connection pooling (`_WIKI_DB_LOCAL`) to eliminate opening/closing SQLite connections to `morpheme.db` for every inflection candidate.
- **Persistent Disk Cache & Memory Write-Back**:
  - Added persistent on-disk cache `dictionaries/aw_inflections.txt` loaded on boot and auto-persisted.
  - Implemented in-memory write-back `DEFINITIONS_CACHE[w] = full_def` and pre-baked AW inflections during background definition resolution.
- **Client-Side Instant Rendering**:
  - Added `window._gameDefCache` in `play.js` for 0ms instant display without UI placeholder flashes.
  - Added background prefetching (`prefetchDefinitions`) during intermission.
  - Benchmark on production server dropped initial lookup latency from **~490ms down to 6.4ms** (76x faster) and re-clicks to **0ms**.

### 5. Mobile Directional "Players" & "Words" Indicator Panel Anchored at Bottom
- **Full Viewport Height Enforcement**:
  - Diagnosed root cause of the grey navigation panel getting stuck higher up beneath the board on mobile: `.board-panel` had an old `height: auto !important;` rule overriding the carousel `height: 100% !important;`.
  - Updated `.board-panel` and `.board-panel.full-bleed-mobile` on `@media (max-width: 992px)` to strictly enforce `height: 100% !important; min-height: 100% !important; padding-bottom: 0 !important;`.
- **Permanent Sticky Bottom Docking**:
  - Positioned `.board-panel .mobile-nav-indicators` and `.board-panel.has-split-notepads .mobile-nav-indicators` with `position: sticky !important; bottom: 0 !important; margin-top: auto !important; height: 44px !important;`.
  - Verified across both active round play and intermission: nav indicators sit at the exact same bottom edge (`top = 662.0px, bottom = 706.0px` on 390×844 mobile viewport) flush with the adjacent Players and Words panels.

### 6. Penalty Format Feedback Fix & Profile "Best Words" Protection (`DOSSERETI`)
- **Penalty Feedback Display Fix**:
  - Fixed client-side logic in `submitWord()` in `play.js`: previously `if (data.success)` formatted penalty responses as `${word} VALID (-3 PTS)`, flashing green/blue with a success chime before later showing as a penalty.
  - Updated client to check `serverIsActuallyValid = data.success && !serverIsPenalty`. Penalty words now strictly display `DOSSERETI PENALTY (-3)` (never `VALID`), flash invalid red, and play the error sound.
  - Added explicit `'is_penalty'` boolean flag to server JSON response in `/room/<room_id>/submit_word`.
- **Best Words & Stats Integrity**:
  - Fixed profile best words generation (`get_public_profile` in `app.py`), config history (`get_config_history`), word finds (`/api/word_finds/<word>`), and leaderboard calculations to:
    1. Strictly filter out any word with zero or negative points (`points <= 0`) or `is_penalty: True`.
    2. Enforce bulletproof dictionary verification (`is_word_in_any_dict(w_text)`) before admitting any word into candidate pools.
- **Database Cleanup on Production**:
  - Cleaned round 2408 (`pub_v2_accumulative_6x8_45`) in `morpheme.db`: removed invalid word `DOSSERETI` from `words_json` and updated round score to 5 (`ANILINE`).
  - Verified live endpoint `/api/profile/jeffb`: confirmed `DOSSERETI` is completely removed (`0` occurrences) and only authentic words are listed.

### 7. Multiplayer Subanagrams Stability & Word Guarantee (Strict Room-Scoped)
- **Word Guaranteed Distribution**:
  - Calibrated word count ranges by sequence length so `generate_subanagrams_board_and_words` finds valid candidates immediately.
  - Word Guaranteed mode appears in ~50% of rounds and Totally Random in ~50%.
- **Sequence Lengths 6–10 Even Spread**:
  - Removed fallback demotion of lengths 9 and 10 down to 8/7.
  - Verified uniform ~20% distribution across lengths 6, 7, 8, 9, 10.
- **Parameter Stability (0:45 Intermission to Round Start)**:
  - Ensured `frozen_revealed_params` is strictly populated and locked at the 0:45 reveal mark.
  - `start_next_round` strictly adheres to `active_params` / `frozen_revealed_params` without re-rolling, re-labeling, or changing parameters at round start.
  - Subanagrams boards in Word Guaranteed mode strictly guarantee that at least one max-length word exists in the board and in `room.all_words` (fixing the issue illustrated in `nomax.png`).
- **Numpy Vectorized Solver Acceleration**:
### 8. Forum Category Header Fixed & Tightened Spacing
- **Fixed Category Header During Scroll**:
  - In `templates/index.html`, moved `.forum-header` (containing category name, description, "Refresh" button, and "+ New Thread" button) outside `.forum-view-scroll-body` directly under `#forum-view-list`.
  - When scrolling through threads in any category, the header and its divider remain permanently fixed in place above the thread list on all platforms (mobile, tablet, laptop, desktop).
  - When opening a thread (`#forum-view-post`) or the thread creation page (`#forum-view-create`), the category header is absent, displaying only thread-specific controls and content.
- **Desktop & Laptop Header Tightening**:
  - Reduced desktop/laptop vertical space footprint: lowered h2 margin and font-size (`1.6rem`), reduced padding to `18px 30px 14px 30px`, set vertical alignment to center, and tightened line heights so the title and description sit higher up without occupying unnecessary vertical room.
  - Adjusted mobile styling with compact padding (`12px 15px`) and `flex-shrink: 0` for consistent behavior across devices.

### 9. Multiplayer Subanagrams: Equal Left/Right Letter Sequence Padding
- **Equal Side Padding to Top Padding**:
  - In `static/css/play.css`, aligned padding on `.game-board.is-subanagrams-board` so the left and right padding equals the top padding (`padding: 24px 24px !important`).
  - Updated desktop/laptop media query (`@media (min-width: 901px)`) to set `padding-top: 24px !important; padding-bottom: 24px !important; padding-left: 24px !important; padding-right: 24px !important;` on the letter sequence board container.
  - Updated mobile media query so left/right padding matches top padding (`padding: 12px 12px !important`), maintaining symmetry and visual balance across all resolutions.

---

## Deployment & Verification Log
- **Git Commit**: `458df506a7cbb0fa8c26bb437b01b6a22f36f6d0`
- **GitHub Tags**: `START_OVER_POINT_NOVEMBER_7`, `stable-2026-11-07`
- **Remote Server**: `132.148.72.249` (morpheme.games / morpheme.online)
- **PM2 Service**: `morpheme` online (PID 3719934, status online, memory 123MB, CPU 0%)
- **HTTP Health**: `HTTP/1.1 200 OK` across web application and APIs

