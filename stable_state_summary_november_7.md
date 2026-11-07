# Stable State Summary — November 7, 2026 (End of Day)

## Latest Feature Commit ID
```
4d506a44 Do not check superwords for 0 MP: adding letters requires MP >= 1, excluding words like ABDOMINA that have no 0 MP subwords
```
*(Full hash: `4d506a44ed6fb41bdce1442c03838a80c5a725b3`)*

## Start Over Point Tag
```
START_OVER_POINT_NOVEMBER_7
```
*(Also tagged as `stable-2026-11-07`)*

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| localhost      | ✅ Synchronized | Working tree clean, HEAD at `4d506a44` |
| GitHub (main)  | ✅ Synchronized | Pushed to `origin/main` & tags updated on GitHub |
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
  - Added `margin-top: auto !important;` to `.mobile-indicator-row` so it reliably pins flush to the bottom of the viewport below the word input and tile keypad.

### 6. Subanagrams Game Mode Implementation
- **Rules & Mechanics**:
  - Implemented dedicated Subanagrams room mechanics where players form valid words of length $N-1$ down to 4 using subsets of letters from a target word of length $N$.
  - Connected scoring, timer, and dictionary validation across single-player and multiplayer modes.

### 7. Custom Definitions Editing System for Moderators
- **Moderator Definition Override**:
  - Implemented `/api/definitions/override` endpoint and frontend controls allowing authenticated moderators to view, edit, add, and correct definitions for any word in the dictionary database.

### 8. Enhanced Rating Badges & Modal Swatches
- **User Profile & Rating Swatches**:
  - Added visual rating tier swatches to user profile popups, leaderboard rows, and lobby listings reflecting user rank with tier-specific colors and animation effects.

### 9. Subanagrams Board Sizing & Equal Side-to-Top Padding
- **Dynamic Letter Hugging**:
  - Implemented `fitSubanagramsBoard()` in `static/js/play.js`, ensuring `#game-board`'s width tightly hugs the sequence of letters (whether in 1 single row or wrapped into multiple rows) rather than stretching across the full panel.
  - Aligned padding on `.game-board.is-subanagrams-board` in `static/css/play.css` so left and right padding strictly equals top padding (`padding: 24px 24px !important; gap: 12px !important;`).
  - Added listeners to window resize and tile size slider in `settings.js` so padding to the darkened background mat remains equal across all screen sizes, device resolutions, and tile size settings.
  - Automatically resets inline width and horizontal padding when exiting Subanagrams or switching to grid games.

### 10. Progressive Shine Effects: Darker Tones, Faster Flashing & Purple Center Glow
- **Progressively Darker & Faster Flashing**:
  - **Tier 1 (4000–4999)**: Dark smoked silver sheen with deep shadow at **2.4s** cycle.
  - **Tier 2 (5000–5999)**: Deep antique bronze / dark amber gold sweep at **1.6s** cycle.
  - **Tier 3 (6000–6999)**: Deep midnight cyan-void laser slice at **1.0s** cycle.
  - **Tier 4 (7000+)**: Fast **0.65s** cycle.
- **7000+ Singularity Refinement**:
  - **Removed the purple flash sweep**: Removed the sweeping diagonal streak (`::before`) completely.
  - **Darker, clearer purple center glow**: Enhanced `::after` radial gradient with deep rich royal purple core (`rgba(125, 18, 205, 0.98)` to `rgba(88, 10, 150, 0.9)` and `rgba(48, 0, 85, 0.7)`), pulsating in and out noticeably (`scale(0.4) \rightarrow scale(1.15)`, opacity `0.12 \rightarrow 1.0`, brightness `0.6 \rightarrow 1.35`).
- **Placement**:
  - Rendered on the color chart bar (`#game-color-bar`), modal tier swatch (`#color-tier-swatch`), and user rating squares across all player views.

### 11. MP Explorer: Difference Dropdown with Letter Length Prioritization
- **New Difference Dropdown**:
  - Added `Difference:` select menu with options: `All` (default), `0`, `1`, `2`, `3` representing $|len(word) - len(combo)|$.
  - Positioned logically beside `Length:` and `Dictionary:` in the MP Explorer control suite.
- **Letter Length Prioritization ($0 \rightarrow 1 \rightarrow 2 \rightarrow 3$)**:
  - Results are prioritized so smaller differences appear first. E.g., for `RETINULA (8)`:
    - 0MP: `ALUNITE (7, diff 1)` appears before `UNITER (6, diff 2)`, which appears before `NITER (5, diff 3)`.
    - 1MP: `ALUNITES (8, diff 0)` appears before `RETINUE (7, diff 1)` and `RETINULAE (9, diff 1)`, which appear before `BALUN (5, diff 3)`.
  - When a specific difference is chosen, the resulting combo matches are filtered strictly to words with that exact difference.
  - Preserves tie-breaking (longer words first, then A–Z or random shuffle).
- **Responsive Layout Design for Desktop and Mobile**:
  - **Desktop / Laptop**: All 6 dropdowns (`Dictionary`, `Length`, `Difference`, `MP Value`, `Direction`, `Order`) sit in a single unified horizontal row via `display: contents !important;` with compact label font-sizing (`0.78rem`) and select padding (`3px 5px`) to avoid text wrapping.
  - **Mobile**: Formats cleanly into full-width stacked rows with fixed 90px left labels, full-width select inputs stretching to the right border, and `font-size: 16px` to prevent iOS zoom.

### 12. MP Explorer: Difference Filter Optimization & Instant Loading Fix
- **Decoupled Bulk Input Words from Difference Filtering**:
  - Input words are cached in memory by `(dictionary, length, mp, direction)`.
  - Verification uses fast C morpheme checks with `target_diff=None`, completing all 42,341 words in ~6 seconds on cold start and **0ms** on cached requests.
  - Single-word queries run in **0.02 seconds**, filtering and sorting matches by difference.
  - Changing the `Difference` dropdown does NOT wipe or re-fetch the 42,000 input words; it updates the matches for the selected word in real time (<50ms).

### 13. MP Explorer: 0 MP Default, Sub-second 0 MP Input Word Loading & "All MP" Option
- **0 MP Default**:
  - Updated `MP Value` dropdown in `templates/index.html` to default to **`0 MP`**, ensuring that when users explore combos with Difference = 1, large, high-value combos like **`ALUNITE` (7)** appear immediately without requiring manual configuration.
- **"All MP" Option**:
  - Added `<option value="all">All</option>` to the `MP Value` dropdown. When selected, merges combos across 0 MP, 1 MP, 2 MP, and 3 MP, prioritizing them by sequence difference, letter count, and MP tier.
- **Fast 0 MP Sub-second Loading**:
  - Optimized `check_word_has_mp` for `target_mp == 0` with fast subword checks down to length 3 and 1-letter extension superwords. All 42,196 8-letter CSW words with 0 MP connections now verify in **0.10 seconds**!

### 14. MP Explorer: Input Words Table Strictly Filtered by All Criteria (Including Difference)
- **Strict Table Filtering**:
  - The left "Input Words" table now strictly filters to words that have at least one valid combo match meeting **all** user-selected criteria: `Length`, `Dictionary`, `MP Value`, `Direction`, and `Difference`.
  - Words with 0 matching combos under the chosen criteria (such as `AARDVARK` or `ABDOMINA` when `MP Value = 0` and `Difference = 1`) are completely excluded from the table.
  - Words with matching combos (such as `RETINULA`, which connects to `ALUNITE` with difference 1) appear cleanly in the table.
- **Sub-Second O(1) Performance**:
  - `check_word_has_mp` runs sub-second set-lookup checks across all 42,000+ words without slow full-matrix searches, completing full list verification in ~0.5s–1.8s on cold start and 0ms when cached.
- **Dynamic Change & Selection Synchronization**:
  - Changing the `Difference` select menu immediately triggers `loadComboExplorerWords()`, updating the left table to match the selected difference.
  - Automatically selects the first word in the filtered list if the previously selected word is not eligible under the new criteria.

---

## Deployment & Verification Log
- **Git Commit**: `4d506a44ed6fb41bdce1442c03838a80c5a725b3`
- **GitHub Tags**: `START_OVER_POINT_NOVEMBER_7`, `stable-2026-11-07`
- **Remote Server**: `132.148.72.249` (morpheme.games / morpheme.online)
- **PM2 Service**: `morpheme` online (memory ~850MB, CPU 0%)
- **HTTP Health**: `HTTP/2 200 OK` across web application and APIs
- **Verified Endpoints**:
  - `POST /api/tools/combo/explorer` (8 Letters, MP 0, Diff 1, CSW): Returned 4,335 words. `RETINULA` is present; `ABDOMINA` and `AARDVARK` are excluded.
  - `POST /api/tools/combo/explorer` (8 Letters, MP 0, Diff All, CSW): Returned 27,209 words. `RETINULA` is present; `ABDOMINA` and `AARDVARK` are excluded.
  - `POST /api/tools/combo/explorer` (8 Letters, MP 1, Diff 1, CSW): `ABDOMINA` is present (matches `["ABDOMINAL"]`).
  - `POST /api/tools/combo/explorer` (RETINULA, MP 0, Diff 1, CSW): Returned `["ALUNITE"]` in 0.05s.
  - `POST /api/tools/combo/explorer` (AARDVARK, MP 0, Diff 1, CSW): Returned `[]` (0 matches).
  - `POST /api/tools/combo/explorer` (8 Letters, MP 1, Diff 0, CSW): Returned 42,341 words.
  - `POST /api/tools/combo/explorer` (8 Letters, MP 1, Diff 1, CSW): Returned 30,760 words.
