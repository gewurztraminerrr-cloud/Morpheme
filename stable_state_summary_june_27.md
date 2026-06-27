# Morpheme Stable State Summary - June 27, 2026

This summary documents the stable state of the Morpheme application as of June 27, 2026. Localhost, GitHub origin, and `morpheme.games` are fully synchronized and verified under Commit ID **`5007457`** (and subsequently tagged as **`START_OVER_POINT_JUNE_27`** and **`snapshot-current`**).

---

## 🚀 Key Improvements & Bug Fixes

### 1. Settings Page Redesign (Split-Layout)
*   **Desktop & Laptop Layout**: Refactored the Settings container in [templates/index.html](file:///Users/jeffbabiak/templates/index.html) to use `.tools-split-layout`. It now features a sidebar on the left with three category tabs (**Appearance**, **Audio & Sounds**, and **Gameplay & Highlights**) and the corresponding configuration panels on the right.
*   **Mobile Sliding Navigation**: Implemented sliding transitions, swipe gestures (swipe-to-back), and category navigation in [settings.js](file:///Users/jeffbabiak/static/js/settings.js) for mobile devices, aligning the Settings UX with the Tools UX.
*   **Tile Selectable Space Relocation**: Moved the "Tile Selectable Space (Octagon vs Diamond)" panel from the *Appearance* category to the *Gameplay & Highlights* category per user feedback.

### 2. Vertical Page Scroll Containment & Hash Reset
*   **Disable Hash Auto-Scroll**: Modified [app.js](file:///Users/jeffbabiak/static/js/app.js) to temporarily clear the page container's `id` during transition. This prevents the browser's native layout engine from auto-scrolling the viewport down to the page container when the URL hash (e.g. `#page-settings` or `#page-tools`) becomes visible.
*   **Manual Scroll Restoration**: Set `history.scrollRestoration = 'manual'` at the beginning of [app.js](file:///Users/jeffbabiak/static/js/app.js) to block the browser from forcing history-based scroll offsets on page navigation or reload.
*   **Forced Scroll Reset**: Added a delayed scroll reset to `(0, 0)` in `showPage` to ensure that both the window and the page container are anchored at the very top, keeping the **"MORPHEME MORE-FEEM"** header fully visible.

### 3. Mobile Scroll Containment
*   **Viewport Lock**: Configured `#page-tools` and `#page-settings` on mobile in [play.css](file:///Users/jeffbabiak/static/css/play.css) to have a fixed height (`calc(100vh - 120px)`) and hidden vertical overflow (`overflow-y: hidden`).
*   **Independent Scrollbars**: Forced the `.tools-split-layout` to fill the remaining viewport height, confining all scrolling to the sidebar list and the active tool content pane.
*   **Active State Visibility**: Fixed a bug where inactive pages were shown on mobile by ensuring `#page-tools` and `#page-settings` default to `display: none` and only receive `display: flex !important` when active.

### 4. Lobby "Start" Button State Reset
*   **Lobby Observer Fix**: Modified `isOnLobby()` in [lobby.js](file:///Users/jeffbabiak/static/js/lobby.js) to use the cached `lobbyPage` closure reference. This ensures that even when the page container's ID is temporarily cleared during transition, the Lobby correctly registers that it is active and runs `resetLobbyButtons()`, restoring the "Start" buttons to their clickable, active states.

### 5. Category Descriptions in Tools, Mods, and Settings
*   **Structured Content**: Added nested `.tool-btn-title` and `.tool-btn-desc` `div` tags to all sidebar buttons in [templates/index.html](file:///Users/jeffbabiak/templates/index.html).
*   **Theme-Adaptive Styling**: Styled descriptions in [play.css](file:///Users/jeffbabiak/static/css/play.css) to be smaller (`0.72rem`) and use a theme-adaptive muted color (`rgba(var(--text-primary-rgb), 0.45)`) that adapts to light and dark themes.
*   **Caching Workaround**: Used `div` elements instead of inline `span` elements in the HTML to guarantee that the title and description stack vertically even if the user's browser has cached the older stylesheet.

### 6. Bounce Format Ball Count Ratio
*   **10:16 Ratio**: Changed the ball count formula in [play.js](file:///Users/jeffbabiak/static/js/play.js) to a ratio of 10 balls for every 16 letters (tiles): `const count = Math.round((rows * cols * 10) / 16);`.
*   **4x4 Board Optimization**: For a 4x4 board (16 letters), this automatically reduces the ball count from 11 to 10, removing exactly 1 ball as requested.

### 7. Page Title Headers
*   **Visual Alignment**: Added `<h2>` headers with the class `.page-title-header` to both the Tools and Settings pages in [templates/index.html](file:///Users/jeffbabiak/templates/index.html).
*   **CSS Layout**: Created the `.page-title-header` class in [play.css](file:///Users/jeffbabiak/static/css/play.css) to ensure perfect alignment on desktop and automatic side padding (`15px`) on mobile.

---

## 🛠 Active Features & Configuration
*   **Board Formats**: Normal, Checkerboard, Double, Triple, Valued Letters, Rotation, Penalty, Mania, Either/Or, Bonus Word, and Density.
*   **Grid Dimensions**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
*   **Dictionaries**: NWL (American) and CSW (International) Tries.

---

**Latest Stable Commit ID**: `5007457`  
**GitHub Tags**: `START_OVER_POINT_JUNE_27`, `snapshot-current`  
**Localhost & GitHub Sameness Status**: Synchronized  
**Production Server Status**: Green / PM2 Online / Live at commit `5007457`
