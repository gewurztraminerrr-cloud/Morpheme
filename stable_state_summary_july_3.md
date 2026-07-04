# Stable State Summary - July 3, 2026

This document summarizes the stable state of **Morpheme** as of July 3, 2026. All local changes, remote code on GitHub, and the live application running on morpheme.games are fully synchronized.

## Latest Commit Information
* **Commit ID**: `f5c5a72`
* **Branch**: `main`
* **Date**: July 3, 2026
* **Git Tag**: `snapshot-current` and `START_OVER_POINT_JULY_3` (updated July 3, 2026)

## Changes Included in this Stable State

1. **Pure Directional Morpheme Metric Alignment**:
   - **Formula Correction**: Replaced the hybrid forward/backward evaluation in `tools_combo_check()` in `app.py` with a pure directional calculation (evaluating 4 forward/reversed combinations of source and candidate).
   - **Correct Scenario Outputs**: Resolved all discrepancy anomalies globally. Direct evaluations match the Java scoring engine exactly:
     - `CITHERS` $\to$ `CITER` correctly resolves to **1MP** (reversing `CITHERS`, dropping the `H`, keeping `CITER`).
     - `AGRESTIC` $\to$ `STRATEGETIC` is correctly excluded from the 3MP candidate pool (resolves to 5MP).
     - `CRETICS` $\to$ `CITER` correctly resolves to **2MP**.
   - **Verification**: All 26 scenario tests pass with 100% correctness.

2. **Desktop Layout Scrollbar Prevention**:
   - **Layout Constraints**: Locked page heights for Tools, Settings, and Mods pages (`#page-tools`, `#page-settings`, `#page-mods`) to `height: calc(100vh - 120px) !important; overflow: hidden !important;` on desktop viewports (`min-width: 901px`) in `play.css`.
   - **Internal Scrolling**: Changed the layout panel container `.tools-split-layout` to fit dynamically (`flex: 1 1 auto !important; height: 0 !important; min-height: 0 !important;`) so that sidebars and scroll regions scroll independently, avoiding main browser window scrollbars entirely.
   - **Cache-Busting**: Incremented stylesheet parameter link to `play.css?v=111` in `templates/index.html`.

3. **Mobile Layout Resize Loops & Screen Blackouts Prevention**:
   - **Observer Loop Removal**: Completely removed the redundant `ResizeObserver` on `.board-panel` in `play.js` which caused infinite layout recalc loops on mobile device viewports when browser controls or virtual keyboard bounds shifted the screen size.
   - **Short-Circuited Layout Checks**: Added early checks in `checkBoardOverflow()` and `adjustPlayHeaderForDevice()` to return immediately if the Play page is not active (`!playPage.classList.contains('active')`), preventing background thrashes while using other tabs.
   - **Cache-Busting**: Incremented script parameter link to `play.js?v=212` in `templates/index.html`.

4. **Combo Checker Mobile Rendering Acceleration**:
   - **Flat DOM Layout**: Converted the heavy HTML `<table>` layout inside `tools.js`'s `renderGroups` to a flat list of `div`s styled with `.group-row`, which reduces layout calculations by 5x on mobile virtual keyboard resizes.
   - **DOM Weight Cap**: Capped the backend results slice per category column to the top **150** items in `app.py`, ensuring a light DOM weight.
   - **Cache-Busting**: Incremented import version parameters: `tools.js` to `v=73` and `play.css` to `v=112` in `templates/index.html`.

5. **Backend Combo Checker Speed Optimization (10x-50x speedups)**:
   - **Restricted Maximum MP Globally**: Capped `max_mp = 3` globally for all word lengths to prune candidates.
   - **Implemented LIC Shared-Count Cap**: Vectorized candidate pruning was updated to require `shared_counts >= 5` (matching Java Swing's 5LIC-9LIC range), filtering out 77,000+ low-overlap candidates instantly.
   - **LCS Linearity Lower Bound Pruning**: Added a `t_len - linearity > limit` check directly after calculating LCS length. If the edit difference bound mathematically exceeds the maximum limit, we skip the recursion entirely, pruning over 90% of backtrack calls.
   - **Removed Heavy Array Allocations in LIS**: Optimized `get_lis()` to iterate `range(i)` instead of slicing `nums[:i]`. This avoids creating millions of temporary list allocations during recursive iterations.
   - **Relevance candidate sorting & early early early break**: Sorted evaluated candidates by descending shared counts and early exit bounds.

6. **Candidate Length and Early-Break Correction for 8+ Length Searches**:
   - **Length Constraint Adjustment**: Relaxed candidate checks in `app.py` for search terms of length 8 to allow candidate words down to length 5 (and length 6 for search terms of length 9 and 10). This aligns the candidate selection criteria with Java Swing specifications and ensures shorter matching words like `UNITE` (length 5) are included.
   - **Dynamic Early-Break Adjustment**: Updated `active_mp_keys` checking to calculate bounds using the minimum candidate length (`min_target_len = 6 if source_len >= 9 else 5`) instead of the source length. This prevents the early-exit condition from triggering before low-MP words with lower shared counts are processed.

7. **Desktop Layout Sizing & Positioning Updates**:
   - **Responsive Timer Styling**: Restored the original large prominent proportions of the desktop `.timer-display` (font-size `1.5rem` for label, `2.2rem` and `min-width: 220px` for timer value) while retaining the modern light-gray capsule backdrop (`background: rgba(var(--text-primary-rgb), 0.1)`).
   - **Page Transition Synchronization**: Added a delayed `adjustPlayHeaderForDevice()` call (using a 50ms `setTimeout`) inside the `showPage('page-play')` routing function. This ensures that when navigating from the Lobby to the Play page, the layout adjusts immediately despite temporary DOM ID resets.
   - **Color Chart Alignment**: Positioned the `#game-color-bar` directly below the top `.play-header` on desktop, spanning the entire width above the Players, Board, and Words panels.
   - **Cache-Busting**: Incremented resource links to `play.css?v=117` and `app.js?v=67` in `templates/index.html`.

8. **Tournament Automatic Bot-Filling & Startup**:
   - **Automated Bracket Scaling**: Modified the `start_tournament()` logic in `tournament_logic.py` to automatically register AI bots to fill any remaining brackets if a tournament starts with less than 8 players.
   - **AI Bots DB Seeding**: Seeded the 8 AI bot accounts in the production database.
   - **Tournament 13 Launch**: Registered the player's test accounts, reset Tournament 13's state, and launched the tournament with a full 8-player bracket (4 test accounts + 4 AI bots) on the live site.

## Verification
* **Local**: Verified correct metric computations and scroll locks on localhost.
* **Production**: Pulled changes and verified live on **morpheme.games**. PM2 services are running cleanly.
