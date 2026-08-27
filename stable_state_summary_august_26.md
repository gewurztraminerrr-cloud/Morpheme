# Stable State Summary — August 26, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of August 26, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is fully synchronized.

---

## 1. Repository & Deployment Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 26, 2026
* **Latest Commit ID**: `b053e1aeb0cfb8d4f40f065538e1db4e5b2bb569` (`b053e1a`)
* **Production Host**: `132.148.72.249` (`morpheme.games`)
* **Synchronization Status**: **100% Synchronized** across Localhost, GitHub, and Production (`morpheme.games`).

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Combo Checker Input Re-Focus Black Screen Resolution (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Eliminated Viewport Collision on Textbox Re-Focus**:
  - Removed an obsolete `focus` listener on `#combo-input` that previously forced simultaneous scroll resets on 4 parent containers (`window.scrollTo(0,0)`, `toolsContent.scrollTop = 0`, `toolPane.scrollTop = 0`, `layoutEl.scrollLeft = ...`) while the virtual keyboard and results tables were active.
  - Set `font-size: 16px !important;` on `#combo-input` to prevent iOS Safari auto-zoom recalculations and GPU compositor crashes when tapping into the textbox to search subsequent words.

### B. Combo Checker Textbox & Button Height Sizing on Mobile (`static/css/play.css`, `templates/index.html`)
- **Uniform 46px Height Matching the "Check" Button**:
  - Sized `#combo-input` (`"ENTER LETTERS TO CHECK..."`) to `height: 46px; min-height: 46px; max-height: 46px; padding: 10px 16px; font-size: 16px; border-radius: 8px;` matching the exact vertical dimensions and curvature of the **"Check"** button (`#combo-search-btn`) and dictionary dropdown (`#combo-dict`).

### C. Windowed Chunk Rendering & Seamless Mobile Word Jump in Tools Full List (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Eliminated Mobile Black Screen & Flash on Word Search**:
  - Replaced bulk DOM wipes with seamless near-range appending (`appendNextFullListBatch` / `prependPrevFullListBatch`) and windowed focused chunks (~500 words) for distant queries.
  - Used direct instant `scrollTop` assignment on mobile devices (`isMobile ? fullListResults.scrollTop = centerOffset : scrollTo(...)`) to eliminate iOS Safari WebKit compositor collisions caused by simultaneous smooth scroll interpolation and virtual keyboard dismissal.
  - Added CSS layout and paint containment (`contain: content; transform: translateZ(0);`) on `#full-list-modal-results` and full `100dvh` dimensions on `#full-list-modal` to retain GPU compositing surfaces without dropping frames.
  - Optimized the `.jump-target-pulse` animation to use GPU-friendly background/outline transitions instead of intensive multi-layer box shadows.

### D. Smooth Sliding Transitions on Bottom Navigation Buttons (`static/js/play.js`, `templates/index.html`)
- **Animated Panel Sliding**:
  - Configured `window.switchPlayPanel()` to default to smooth scrolling (`smooth = true`), triggering animated sliding when pressing **"Board ➜"** in the Players panel, **"← Board"** in the Words panel, **"← Players"**, and **"Words ➜"** in the Board panel.
  - Preserved instant zero-latency snaps (`smooth = false`) for room initialization, orientation change recoveries, and programmatic panel resets.

### E. Mobile Intermission Bottom Navigation Bar in FCFS, SP, and CUBE Rooms (`static/css/play.css`, `templates/index.html`)
- **Always-Visible Bottom Navigation Bar**:
  - Removed the `display: none !important` override on `.board-panel.has-split-notepads .mobile-nav-indicators` so the bottom grey navigation bar with `"<- Players"` and `"Words ->"` remains accessible during intermission in **FCFS**, **SP (Split Points)**, and **CUBE (3D)** rooms on mobile devices.
- **Dedicated Flex Layout & Dedicated Bottom Clearance**:
  - Structured `.board-panel.has-split-notepads` as a full-height column flex container, allocating `calc(100% - 44px)` to the scrollable notepads container and pinning the 44px bottom navigation bar (`flex-shrink: 0; z-index: 100;`).
  - Allows users to swipe and tap seamlessly between the **Players**, **Board/Notepads**, and **Words** panes without notepads overlapping the navigation bar.

### F. Profile and Tools Controls Centering & Equal Panel Padding (`static/css/play.css`, `templates/index.html`)
- **Centered Search, Buttons, and Inputs**:
  - Centered the username search input, **Search** button, and **My Profile** button horizontally in the Profile control panel card (`justify-content: center; align-items: center;`).
  - Standardized uniform 46px heights across profile search inputs and action buttons.
  - Symmetrized card padding (`padding: 20px 24px` on desktop, `padding: 14px 16px` on mobile, `box-sizing: border-box`), ensuring equal space surrounds contents on all four sides.
  - Applied symmetrical card padding and centered layouts across all Tools tabs (**Subanagrams**, **Sequence Search**, **Word Lists**, **Word Validator**, **Combo Checker**, **Unscramble**, and **Random Word Generator**).

### G. Drop-Down Menu Label Simplification: "Added Words" to "AW" (`templates/index.html`)
- **Clean Drop-Down Text Across Tools & Game Interfaces**:
  - Updated all select dropdown options containing the text "Added Words" across the application to display **"AW"** instead (Random Word Generator, Combo Checker, Word Validator, Subanagrams, Sequence Search, Word Lists filter, and Unscramble).
  - Maintained all existing `value` bindings (`value="added_words"` and `value="added"`), ensuring seamless compatibility and identical functional behavior across all backend endpoints, queries, and event handlers.

### H. Lobby Live Player Count Accuracy & Accumulative Room Sync (`app.py`, `game_room.py`, `static/js/lobby.js`, `templates/index.html`)
- **Guest Session Auto-Initialization Hardening**:
  - `ensure_guest_session()` and `guest_login()` now incorporate a collision retry loop (up to 10 attempts), preventing random `UNIQUE constraint failed: users.username` errors in SQLite from leaving guest users unauthenticated.
- **Backend Room Wake-Up & State Machine Fixes**:
  - Corrected `check_and_update_state()` in `game_room.py` to reference `wc_label` instead of undefined `wc_lbl` when waking up paused rooms upon human player join.
  - Corrected `_bg` board generator reference in `game_room.py` watchdog fallback.
- **Lobby Stats Aggregation**:
  - `get_lobby_stats()` in `app.py` aggregates human player counts accurately across all game configurations, including 24-hour daily persistent archives in `past_players` and active participants within 60 seconds in real-time rooms.
- **Frontend Button State Preservation & Hydration**:
  - Guarded `fetchAndRenderRooms` in `lobby.js` and `index.html` so non-accumulative room list updates never overwrite `Start [N]` player counts on Accumulative buttons.
  - Added early stats hydration listeners on `DOMContentLoaded` and fallback defaults for `updateLobbyButtons()`.

### I. New FAQ Entry for Game Room Window Navigation (`templates/index.html`)
- **Quick Navigate & Detailed FAQ**:
  - Added quick navigate button and accordion entry in the **How to Play & FAQ** modal:
    - **Question**: *It's difficult to slide over to neighboring windows in game rooms. Is there an easier way to do this?*
    - **Answer**: *The grey bar along the bottom not only is pressable, but swipeable as well. Get into the habit of swiping across or pressing on the text on it and the navigation eventually becomes instinctual!*

---

## 3. Verification & Server Health

* **Localhost Status**: Fully working, all syntax and integration tests passing.
* **Production Deployment (`morpheme.games`)**:
  - PM2 Process `morpheme` (ID 0) online with `0%` CPU and healthy memory.
  - Multi-user live integration tests passed.
  - Zero open unclosed database locks or runtime exceptions in logs.
