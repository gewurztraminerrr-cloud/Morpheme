# Stable State Summary - June 27, 2026

This document summarizes the stable state of **Morpheme** as of June 27, 2026. All local changes, remote code on GitHub, and the live application running on morpheme.games are fully synchronized.

## Latest Commit Information
* **Commit ID**: `0cdbef4`
* **Branch**: `main`
* **Commit Message**: "Create June 27 Stable Point"
* **Date**: June 27, 2026

## Changes Included in this Stable State
1. **Tools, Settings, and Mods Layout Fixes (Desktop/Laptop)**:
   * **Constant Panel Height**: Locked the height of the `.tools-split-layout` container to exactly `620px` and the width to `1400px` on all screens `901px` and wider.
   * **Constant Category Button Size**: Locked the sidebar category buttons to a height of `76px` and a width of `220px` to prevent any button shifting or layout jumping when switching tabs.
   * **Empty Content Area**: Configured a constant dark background (`rgba(0, 0, 0, 0.25)`) and borders for the content area on the right, providing a large, stable empty space when no category is selected.
   * **Aligned Page Titles**: Added a centered "Mods" title to the Mods tab and aligned all page title headers ("Tools", "Settings", "Mods") to match the `1400px` layout grid.
2. **Mobile Layout Constraints**:
   * Removed desktop width constraints on mobile screens (widths `900px` and below) to ensure the category button sidebar spans exactly `100%` of the screen width and the content area remains completely hidden off-screen to the right until selected.
3. **Board Sizing Settings Refactor**:
   * Removed the main board size slider and updated settings descriptions.
   * Configured default sizes: 4x4: `82px`, 4x6: `82px`, 5x7: `65px`, 6x8: `54px`.
   * Linked the dimension-specific sliders to dynamically resize the 2D example board (`#preview-board`) on slider input.
4. **Intermission Tile Filtering Fix**:
   * Clicking any letter tile (including duplicate letter tiles like multiple "C"s) now correctly filters the "All Words" list on mobile and desktop.
5. **CSW + AW Dictionary in Solo Mode**:
   * Modified `board_generator.py` to preserve the `use_added_words` context variable if already set to `True` by the game room, ensuring Added Words (AW) are correctly generated on Solo boards.
6. **Easy Difficulty Board Generation**:
   * Adjusted uniqueness ratio ranges for 4x4 and 4x6 boards to prevent generator timeouts.
   * Purged all super-rare letters (`Q`, `Z`, `J`, `X`, `K`) on Easy difficulty (unless they are part of the protected bonus word) to guarantee a lower uniqueness ratio.
7. **Profile Stats Exclude 24h Rooms**:
   * Excluded 24-hour rooms (duration `>= 7200` seconds) from the main profile statistics (**GAMES**, **WIN RATE**, **AVG WPM**, **BEST**, and **PT SUM**) so that they only reflect active play sessions.
   * Filtered out rounds with a score of `0` from these statistics and config-specific averages.
8. **Dynamic human-like Solo Bots**:
   * Implemented a database-driven dynamic AI model where bots query the recent round history of real human players on the server, calculate their WPM, word length, and CSW-only knowledge by rating bracket, and dynamically configure their parameters to match.
9. **Premium Board Loader with Typewriter Status Ticker**:
   * Redesigned the board loading state to feature a premium glowing spinner, a clear explanation card of the real-time board generation process, and a live status ticker that rotates through the actual steps of the generation (DFS solving, uniqueness checks, etc.).
   * Made the loading state mobile-responsive and fixed a layout collapse issue where the container was squeezed into a single 50px cell on mobile viewports.

## Verification
* **Local**: Verified on Safari and Chrome (Desktop and Mobile).
* **Production**: Deployed via SSH/PM2 and verified live on **morpheme.games**.
