# Stable State Summary — September 17, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of **September 17, 2026**. The codebase, databases, assets, and styling across **Localhost**, **GitHub (`main`)**, and **Production (`morpheme.games` / `132.148.72.249`)** are 100% synchronized, verified, and active.

---

## 1. Repository & Deployment Synchronization

| Environment | Host / Branch | Latest Commit ID | Status |
| :--- | :--- | :--- | :--- |
| **Localhost** | `/Users/jeffbabiak` (`main`) | `30abed8a83416973e659b8764eb803362a98cb9c` | ✅ Clean & Synchronized |
| **GitHub** | `origin/main` | `30abed8a83416973e659b8764eb803362a98cb9c` | ✅ Clean & Synchronized |
| **Production Server** | `132.148.72.249` (`/home/morpheme/morpheme`) | `30abed8a83416973e659b8764eb803362a98cb9c` | ✅ Deployed & Online (`HTTP/2 200 OK`) |
| **PM2 Process** | `morpheme` (PID 0) | `30abed8a83416973e659b8764eb803362a98cb9c` | ✅ Healthy (`online`, uptime active) |
| **Flutter Mobile App** | `morpheme_word_game` | `30abed8a83416973e659b8764eb803362a98cb9c` | ✅ Synchronized (`https://morpheme.games/` audio bridge) |

- **Stable Save Point Date**: September 17, 2026
- **Latest Commit ID**: `30abed8a83416973e659b8764eb803362a98cb9c`
- **Active Git Tags**:
  - `START_OVER_POINT_SEPTEMBER_17`
  - `stable-2026-09-17`
  - `START_OVER_POINT`
  - `save-point-latest`
  - `start-over`
  - *(Historic reference preserved: `START_OVER_POINT_SEPTEMBER_16`, `START_OVER_POINT_SEPTEMBER_13`, `START_OVER_POINT_SEPTEMBER_10`)*
- **Active Cache-Buster Versions**:
  - `style.css?v=1789335000`
  - `lobby.css?v=1789339000`
  - `play.css?v=1789348000`
  - `howtoplay.css?v=1789330000`
  - `forum.css?v=1789336000`
  - `donate.css?v=1789324000`

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Navigation & Top Menu Positioning
1. **Desktop & Laptop Top Menu Position Pinning (`templates/index.html`, `static/css/style.css`, `static/css/lobby.css`, `static/css/play.css`)**:
   - On desktops and laptops (`@media (min-width: 901px)`), pinned the top menu bar (containing MORPHEME, MORE-FEEM pronunciation, and all navigation tab buttons) as `position: sticky; top: 0; z-index: 1000; background: var(--bg-primary);` when the user scrolls down after selecting any tab in the top menu (How to Play, Lobby, Tournaments, Leaderboards, Forum, Tools, Settings, Profile, Donate, Login).
   - In game rooms (`#page-play` / `body.play-active`), the top menu bar is strictly `position: static`, allowing players to see the top menu by scrolling while keeping the full 100vh gameplay arena unobstructed during active rounds.
   - Wrapped `<header class="header">` and `<div class="separator"></div>` inside `.top-menu-bar` to preserve fluid document flow, dynamic themed backgrounds (`var(--bg-primary)`), glowing separator line alignment, and zero interference with mobile fullscreen gestures.

---

### B. Gameplay & Room Logic
1. **Safe Parsing for Randomized Minimum Word Length (`game_room.py`)**:
   - Safely parses `min_word_length` in `initial_solo_params` when randomized or dynamically configured, preventing type mismatch or unexpected null handling during solo game creation.
2. **User Round Results & Rating Protection (`app.py`, `game_room.py`)**:
   - Prevented recording round results for users who earned 0 points or who joined mid-round after play was already in progress, preserving individual player statistics and rating integrity.

---

### C. Mobile Chat Experience
1. **Fluid Drag & Touch Scrolling (`static/css/play.css`, `static/js/play.js`)**:
   - Enabled smooth direct touch drag scrolling in the expanded mobile chatbox when message history overflows.
   - Set `flex-shrink: 0` for all chat message elements to prevent vertical text squishing during momentum scrolling.
   - Removed paint containment and applied `touch-action: pan-y` to support native-feeling mobile gesture physics.
2. **Chat State Persistence & Dedicated Dismissal (`static/css/play.css`, `static/js/play.js`)**:
   - Preserved expanded/collapsed state during message scrolling; chat now strictly requires tapping the dedicated 'X' button to dismiss.
   - Increased the 'X' close button hit target and visual contrast for effortless dismissal.

---

### D. Layout, Spacing & Visual Aesthetics
1. **Mobile Board Spacing & Symmetry (`static/css/play.css`, `templates/index.html`)**:
   - Added balanced vertical spacing above and below the board matching the exact distance from the Spinner Set to the countdown timer panel.
2. **Deep Red Slow Pulse on White Layouts (`static/css/play.css`, `templates/index.html`)**:
   - Aligned mobile board timer pulsing with desktop styling, rendering a smooth deep red slow pulse animation on white layouts.
3. **Lobby Active Rooms Panel Proportions (`static/css/lobby.css`, `templates/index.html`)**:
   - Slightly increased padding and dimension allowances between active rooms header panels and room card elements, enhancing readability across desktop and mobile screens.

---

## 3. Permanent Invariant Rules Enforced (`AGENTS.md`)

1. **Full List Modal (`openFullListModal`)**: Immediately and explicitly exits fullscreen (`document.exitFullscreen()`) upon modal invocation in `static/js/tools.js`. Never remove or disable.
2. **Android Virtual Keyboard Black Screen Prevention**:
   - Fullscreen is strictly exited when navigating to utility pages (Tools, Settings, Profile, Forum, How to Play, Donate) or opening modal dialogs with text inputs.
   - Automatic fullscreen re-engagement never triggers while on non-game utility pages or when any modal/input is active.
3. **Gateway Screen (`#page-loading`)**: Fullscreen is requested immediately on initial tap/press across the initial ENTER LOBBY screen so the Android system prompt appears on the gateway screen, and fullscreen is preserved continuously into the Lobby without layout shifting.
4. **Dictionary Suffix and Plural Sourcing Rules**: Plurals repeat the singular root definition; conjugated verbs (`-S`, `-ED`, `-ING`) repeat the base root verb definition; prefix decomposition produces rich lexicographical definitions.

---

## 4. Verification Checkpoint

- **Local Compilation**: Clean Python compile on `app.py`, `game_room.py`, and core modules.
- **Git State**: Local repository `/Users/jeffbabiak` is clean, with all commits pushed to GitHub `origin/main`.
- **Production Server (`132.148.72.249`)**: Working directory `/home/morpheme/morpheme` is synchronized with `origin/main`.
- **PM2 Daemon**: Process `morpheme` running `online` with zero errors.
- **Live Endpoint Verification**: `curl -sI https://morpheme.games` returns `HTTP/1.1 200 OK` (HTTP/2 enabled).
- **Latest Commit ID**:
  ```
  30abed8a83416973e659b8764eb803362a98cb9c
  ```
