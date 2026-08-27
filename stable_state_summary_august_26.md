# Stable State Summary — August 26, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of August 26, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is fully synchronized.

---

## 1. Repository & Deployment Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 26, 2026
* **Latest Commit ID**: `7c55885f8e6c7104e1ce638bc2c7e0ea5ef73e26` (`7c55885`)
* **Production Host**: `132.148.72.249` (`morpheme.games`)
* **Synchronization Status**: **100% Synchronized** across Localhost, GitHub, and Production (`morpheme.games`).

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Mobile Intermission Bottom Navigation Bar in FCFS, SP, and CUBE Rooms (`static/css/play.css`, `templates/index.html`)
- **Always-Visible Bottom Navigation Bar**:
  - Removed the `display: none !important` override on `.board-panel.has-split-notepads .mobile-nav-indicators` so the bottom grey navigation bar with `"<- Players"` and `"Words ->"` remains accessible during intermission in **FCFS**, **SP (Split Points)**, and **CUBE (3D)** rooms on mobile devices.
- **Dedicated Flex Layout & Dedicated Bottom Clearance**:
  - Structured `.board-panel.has-split-notepads` as a full-height column flex container, allocating `calc(100% - 44px)` to the scrollable notepads container and pinning the 44px bottom navigation bar (`flex-shrink: 0; z-index: 100;`).
  - Allows users to swipe and tap seamlessly between the **Players**, **Board/Notepads**, and **Words** panes without notepads overlapping the navigation bar.

### B. Profile and Tools Controls Centering & Equal Panel Padding (`static/css/play.css`, `templates/index.html`)
- **Centered Search, Buttons, and Inputs**:
  - Centered the username search input, **Search** button, and **My Profile** button horizontally in the Profile control panel card (`justify-content: center; align-items: center;`).
  - Standardized uniform 46px heights across profile search inputs and action buttons.
  - Symmetrized card padding (`padding: 20px 24px` on desktop, `padding: 14px 16px` on mobile, `box-sizing: border-box`), ensuring equal space surrounds contents on all four sides.
  - Applied symmetrical card padding and centered layouts across all Tools tabs (**Subanagrams**, **Sequence Search**, **Word Lists**, **Word Validator**, **Combo Checker**, **Unscramble**, and **Random Word Generator**).

### C. Drop-Down Menu Label Simplification: "Added Words" to "AW" (`templates/index.html`)
- **Clean Drop-Down Text Across Tools & Game Interfaces**:
  - Updated all select dropdown options containing the text "Added Words" across the application to display **"AW"** instead (Random Word Generator, Combo Checker, Word Validator, Subanagrams, Sequence Search, Word Lists filter, and Unscramble).
  - Maintained all existing `value` bindings (`value="added_words"` and `value="added"`), ensuring seamless compatibility and identical functional behavior across all backend endpoints, queries, and event handlers.

### D. Lobby Live Player Count Accuracy & Accumulative Room Sync (`app.py`, `game_room.py`, `static/js/lobby.js`, `templates/index.html`)
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

### E. New FAQ Entry for Game Room Window Navigation (`templates/index.html`)
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
