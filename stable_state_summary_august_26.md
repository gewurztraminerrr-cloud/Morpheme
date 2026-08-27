# Stable State Summary — August 26, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of August 26, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is fully synchronized.

---

## 1. Repository & Deployment Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 26, 2026
* **Latest Commit ID**: `263f4c08e5038ec113b28b7636e09e1e12ea9bb3` (`263f4c0`)
* **Production Host**: `132.148.72.249` (`morpheme.games`)
* **Synchronization Status**: **100% Synchronized** across Localhost, GitHub, and Production (`morpheme.games`).

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Mobile Profile Search Textbox & Button Vertical Length Alignment (`static/css/play.css`, `static/css/style.css`, `templates/index.html`)
- **Uniform 46px Height Matching the "Search" Button**:
  - Sized `#profile-search-input` (`"Enter username..."`) on mobile devices to `height: 46px !important; min-height: 46px !important; max-height: 46px !important; padding: 10px 16px !important; font-size: 16px !important; box-sizing: border-box !important;`.
  - Matched the exact vertical height/length of the **"Search"** button (`#profile-search-btn`) and **"My Profile"** button (`#profile-my-profile-btn`) directly beneath it.

### B. Strict Intermission-Only Trophy Display & Dynamic PE Thresholds (`app.py`, `game_room.py`, `static/js/play.js`, `templates/index.html`)
- **Mathematical PE Calibration**:
  - In matches with similar ratings where the winner scores more than double the opponents, the winner's performance efficiency represents a decisive blowout.
  - Dynamically scaled room-size thresholds:
    - **2 players**: $\text{PE} \ge 1.30$ (corresponds to doubling opponent's score)
    - **3 players**: $\text{PE} \ge 1.45$ (corresponds to doubling opponents' scores)
    - **4–5 players**: $\text{PE} \ge 1.55$
    - **6+ players**: $\text{PE} \ge 1.60$
  - Added direct domination award: $\text{score} == \text{max\_score}$ and $\text{score} \ge 1.75 \times \text{second\_score}$ ($\text{score} \ge 15$).
- **Strict Intermission-Bound Display**:
  - Enforced on both frontend (`renderPlayers`) and backend (`get_room_state`) that the trophy icon 🏆 is **strictly rendered only during `state.state === 'intermission'`**.
  - As soon as the active round starts (`state.state === 'active'`), the trophy icon automatically disappears from all players.

### C. Level Alignment for Rotate & Transpose Buttons on Mobile (`static/css/play.css`, `templates/index.html`)
- **Fixed Button Offset / Vertical Shifting on Press**:
  - Scoped out generic `.rotate-btn:hover` translateY shifts from timer controls.
  - Enforced `transform: none !important; box-shadow: none !important; vertical-align: middle !important;` across all pseudo-states (`:hover`, `:active`, `:focus`, `:focus-visible`) for `#rotate-board-btn`, `#transpose-board-btn`, and `.timer-display .rotate-btn` on mobile devices.
  - Pressing or tapping either button maintains flush, level horizontal alignment with its partner button without jumping or tilting.

### D. Cross-Browser Lobby Music on "ENTER LOBBY" Gateway (`static/js/app.js`, `templates/index.html`)
- **Resolved Firefox & Safari Audio Gatekeeper Block**:
  - Added native `autoplay` attribute directly to the `#lobby-music` `<audio>` element so browsers with native media autoplay policies initiate playback immediately during HTML parsing.
  - Added `Permissions-Policy: autoplay=(self)` meta tag and hidden `allow="autoplay"` frame delegation for browser engines requiring explicit policy declarations.
  - Implemented muted buffer fallback (`audio.muted = true; audio.play()`) so Safari and Firefox immediately buffer and start playback without engine rejection, and instantaneously unmute upon any gesture (`pointermove`, `pointerdown`, `touchstart`, `focus`, `click`).
  - Attached eager capture triggers directly to `#btn-enter-lobby-gateway`, `#gateway-housing`, and the `#page-loading` container.
  - Added Web Audio Context unlock (`AudioContext.resume()`) and multi-lifecycle event triggers (`canplay`, `loadeddata`, `DOMContentLoaded`, `load`, `focus`, `visibilitychange`) so Safari and Firefox immediately engage music playback upon presentation of the "ENTER LOBBY" button.
  - Integrated `handleLobbyMusicState()` into `showPage()` for seamless audio continuity during SPA page transitions.

### E. Desktop Color Chart Repositioning & Downward Mini-Popups (`static/css/style.css`, `templates/index.html`)
- **Color Chart Relocation**:
  - Moved the horizontal rating tier color chart (`#game-color-bar`) on desktops and laptops to above the game header (`.play-header`) and directly below the top navigation menu / separator.
- **Downward Mini-Popups (Tooltips)**:
  - Repositioned the tier name and rating range hover popups to appear smoothly downward (`top: calc(100% + 7px)`) with inverted upward-pointing indicator arrows.
  - Added boundary constraints for the leftmost (`:first-child`) and rightmost (`:last-child`) segments to prevent viewport clipping.

### F. Combo Checker Virtualized Column Rendering & Infinite Scrolling (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Eliminated 3-Second Mobile Viewport Freeze on Re-Focus**:
  - Replaced dumping 30,000–50,000 simultaneous DOM nodes in MP and LIC result columns with a fast, chunked rendering architecture (100 words initially rendered per column) and smooth infinite scrolling upon reaching the bottom of a column.
  - Added `content-visibility: auto; contain-intrinsic-size: 0 28px;` to `.group-row` to bypass layout calculation for off-screen table rows.
  - Throttled custom scrollbars with `requestAnimationFrame()` and removed deep DOM mutation observers.
  - Kept `#combo-input` font-size at `16px` to prevent iOS Safari auto-zoom recalculations.

### G. Combo Checker Textbox & Button Height Sizing on Mobile (`static/css/play.css`, `templates/index.html`)
- **Uniform 46px Height Matching the "Check" Button**:
  - Sized `#combo-input` (`"ENTER LETTERS TO CHECK..."`) to `height: 46px; min-height: 46px; max-height: 46px; padding: 10px 16px; font-size: 16px; border-radius: 8px;` matching the exact vertical dimensions and curvature of the **"Check"** button (`#combo-search-btn`) and dictionary dropdown (`#combo-dict`).

### H. Windowed Chunk Rendering & Seamless Mobile Word Jump in Tools Full List (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Eliminated Mobile Black Screen & Flash on Word Search**:
  - Replaced bulk DOM wipes with seamless near-range appending (`appendNextFullListBatch` / `prependPrevFullListBatch`) and windowed focused chunks (~500 words) for distant queries.
  - Used direct instant `scrollTop` assignment on mobile devices (`isMobile ? fullListResults.scrollTop = centerOffset : scrollTo(...)`) to eliminate iOS Safari WebKit compositor collisions caused by simultaneous smooth scroll interpolation and virtual keyboard dismissal.
  - Added CSS layout and paint containment (`contain: content; transform: translateZ(0);`) on `#full-list-modal-results` and full `100dvh` dimensions on `#full-list-modal` to retain GPU compositing surfaces without dropping frames.
  - Optimized the `.jump-target-pulse` animation to use GPU-friendly background/outline transitions instead of intensive multi-layer box shadows.

### I. Smooth Sliding Transitions on Bottom Navigation Buttons (`static/js/play.js`, `templates/index.html`)
- **Animated Panel Sliding**:
  - Configured `window.switchPlayPanel()` to default to smooth scrolling (`smooth = true`), triggering animated sliding when pressing **"Board ➜"** in the Players panel, **"← Board"** in the Words panel, **"← Players"**, and **"Words ➜"** in the Board panel.
  - Preserved instant zero-latency snaps (`smooth = false`) for room initialization, orientation change recoveries, and programmatic panel resets.

### J. Mobile Intermission Bottom Navigation Bar in FCFS, SP, and CUBE Rooms (`static/css/play.css`, `templates/index.html`)
- **Always-Visible Bottom Navigation Bar**:
  - Removed the `display: none !important` override on `.board-panel.has-split-notepads .mobile-nav-indicators` so the bottom grey navigation bar with `"<- Players"` and `"Words ->"` remains accessible during intermission in **FCFS**, **SP (Split Points)**, and **CUBE (3D)** rooms on mobile devices.
- **Dedicated Flex Layout & Dedicated Bottom Clearance**:
  - Structured `.board-panel.has-split-notepads` as a full-height column flex container, allocating `calc(100% - 44px)` to the scrollable notepads container and pinning the 44px bottom navigation bar (`flex-shrink: 0; z-index: 100;`).
  - Allows users to swipe and tap seamlessly between the **Players**, **Board/Notepads**, and **Words** panes without notepads overlapping the navigation bar.

### K. Profile and Tools Controls Centering & Equal Panel Padding (`static/css/play.css`, `templates/index.html`)
- **Centered Search, Buttons, and Inputs**:
  - Centered the username search input, **Search** button, and **My Profile** button horizontally in the Profile control panel card (`justify-content: center; align-items: center;`).
  - Standardized uniform 46px heights across profile search inputs and action buttons.
  - Symmetrized card padding (`padding: 20px 24px` on desktop, `padding: 14px 16px` on mobile, `box-sizing: border-box`), ensuring equal space surrounds contents on all four sides.
  - Applied symmetrical card padding and centered layouts across all Tools tabs (**Subanagrams**, **Sequence Search**, **Word Lists**, **Word Validator**, **Combo Checker**, **Unscramble**, and **Random Word Generator**).

### L. Drop-Down Menu Label Simplification: "Added Words" to "AW" (`templates/index.html`)
- **Clean Drop-Down Text Across Tools & Game Interfaces**:
  - Updated all select dropdown options containing the text "Added Words" across the application to display **"AW"** instead (Random Word Generator, Combo Checker, Word Validator, Subanagrams, Sequence Search, Word Lists filter, and Unscramble).
  - Maintained all existing `value` bindings (`value="added_words"` and `value="added"`), ensuring seamless compatibility and identical functional behavior across all backend endpoints, queries, and event handlers.

### M. Lobby Live Player Count Accuracy & Accumulative Room Sync (`app.py`, `game_room.py`, `static/js/lobby.js`, `templates/index.html`)
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

### N. New FAQ Entry for Game Room Window Navigation (`templates/index.html`)
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
