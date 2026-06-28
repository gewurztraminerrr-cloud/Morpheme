# Stable State Summary - June 27, 2026

This document summarizes the stable state of **Morpheme** as of June 27, 2026. All local changes, remote code on GitHub, and the live application running on morpheme.games are fully synchronized.

## Latest Commit Information
* **Commit ID**: `6beb512`
* **Branch**: `main`
* **Commit Message**: "Deploy updates"
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

## Verification
* **Local**: Verified on Safari and Chrome.
* **Production**: Deployed via SSH/PM2 and verified live on **morpheme.games**.
