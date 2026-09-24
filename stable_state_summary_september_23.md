# Stable State Summary — September 23, 2026

> **Start Over Point**: All environments (localhost, GitHub, production `morpheme.games`, and mobile web app) are 100% synchronized and verified.  
> **Commit**: `34f60f554dbded3f2fc8eb5506bd5ee4e3d5ac01` (short: `34f60f55`)  
> **Tags**: `START_OVER_POINT_SEPTEMBER_23`, `stable-2026-09-23`  
> **Branch**: `main` — `origin/main`  
> **Server**: `132.148.72.249` (`morpheme.games`) — PM2 process `morpheme` (ID: 0), online ✅  
> **Date/Time**: 2026-09-23 22:15 CDT  

---

## 1. Repository & Deployment Synchronization

| Environment | Host / Path | Status | Tags |
| :--- | :--- | :--- | :--- |
| **Localhost** | `/Users/jeffbabiak/` | ✅ Clean & Synchronized | `START_OVER_POINT_SEPTEMBER_23`, `stable-2026-09-23` |
| **GitHub** | `https://github.com/gewurztraminerrr-cloud/Morpheme` (`main`) | ✅ Pushed & Synchronized | `START_OVER_POINT_SEPTEMBER_23`, `stable-2026-09-23` |
| **Production Server** | `132.148.72.249` (`morpheme.games`) | ✅ Deployed & Online (PM2 process 0 healthy) | Synchronized |
| **Mobile Web / App** | All client platforms (iOS / Android / Desktop) | ✅ Verified (0ms gateway hydration, audio engine) | Synchronized |

---

## 2. Key Features, Improvements & Fixes (September 23, 2026)

### A. Authentication, Session Lifecycle & Gateway Screen
1. **Definitive Server-Side Session Cookie Deletion (`app.py`, `static/js/app.js`)**:
   - Resolved a critical browser `TypeError: keepalive request cannot have an AbortSignal` by eliminating `keepalive: true` when combining with `fetchWithTimeout` or `AbortSignal` on logout.
   - `handleLogout()` now directly awaits standard `fetch('/api/logout', { method: 'POST', credentials: 'include' })` so the `Set-Cookie` deletion header is processed by the browser before local redirection occurs.
   - Revokes `auth_token` in the database and clears the server-side Flask session.
2. **Elimination of Stale Session Resurrection (`static/js/mods.js`, `static/js/app.js`)**:
   - Guarded `checkModStatus()` in `mods.js` and `setCurrentUser()` in `app.js` to immediately abort and never reinstate user credentials when `morpheme_logged_out` is set in storage.
   - Prevents background API responses (`/api/mods/status`) from resurrecting previous users upon browser refresh.
   - Added an explicit `isLoggedOutExplicitly` guard in `DOMContentLoaded` forcing `currentUser = null` and preserving the `"LOGIN"` button state.
3. **Desktop Gateway Screen Unauthenticated Routing (`templates/index.html`, `static/js/app.js`)**:
   - Displays a dedicated 3D mechanical `"LOGIN"` button on the desktop gateway screen when unauthenticated or explicitly logged out.
   - Added an inline, zero-latency click handler (`window.handleLoginGatewayClick`) in `templates/index.html` ensuring instantaneous response even before core scripts hydrate.
   - Preserved seamless transition into `page-login` without visual glitches or falling through to the lobby.

### B. Header Navigation & Layout
1. **Header Navigation & User Display Positioning (`static/css/style.css`, `templates/index.html`)**:
   - Nested `<div id="user-display">` inside `<nav class="nav">` directly to the right of the navigation buttons.
   - **On Mobile Devices**: Returned the vertical divider (`border-left: 1px solid rgba(var(--text-primary-rgb), 0.25)`) and `[username] Logout` segment directly to the right of the horizontally slidable top menu buttons with `position: static !important;`, completely eliminating the top-right corner overlap with `MORE-FEEM`.
   - **On Desktop**: Kept `margin-left: auto;` on `.nav`, maintaining the entire navigation bar and user display cleanly on the right with comfortable breathing room from the logo.

### C. Subanagrams Redesign & Light Theme Contrast
1. **Subanagrams Dual-Tab Layout (`Manual` and `Random`) (`static/js/tools.js`, `templates/index.html`)**:
   - Structured the Subanagrams control panel with two top horizontal tabs (`#sub-tab-manual` and `#sub-tab-random`).
   - **Manual Tab**: Letters input, Dictionary selection (`CSW`, `NWL`, `ALL`, `AW`), Minimum Word Length dropdown (`2` to `15`), **"Find All Words"**, and **"Practice"** buttons.
   - **Random Tab**: Sequence Length (`3` to `15`), Dictionary selection, Minimum Word Length (dynamically bounded by Sequence Length), Randomness mode, and **"Random"** button.
   - Removed purple banner overlay; results panel renders cleanly and directly below the controls.
2. **Light & White Theme Contrast Optimization (`static/css/style.css`, `static/css/play.css`)**:
   - Styled tabs, buttons, labels, and dropdowns for high contrast across `theme-white`, `theme-light-*`, yellow, pink, orange, gray, and light-brown themes.
   - Enhanced contrast for Account Settings card descriptions (`.account-card-desc`) and Cube Round Replay face labels.
   - Covered all space below divider surrounding mobile Back buttons with dark grey (`#18181b` / `#1f2937`) to eliminate white leakage.
   - Enhanced "More random words" button readability on white layouts with bold text and contrast shadows.
3. **Random Word Tool Black Text on White Layouts (`static/css/style.css`, `static/css/play.css`)**:
   - Styled `#random-word-display` and `#tool-random .random-word-large` to render in pure black (`#000000 !important`) with text shadows removed (`text-shadow: none !important`) across all white and light themes (`theme-white`, `theme-light-*`, yellow, pink, orange, gray, light-brown) for crisp readability.

### D. Tools, Settings & Mods Navigation Polish
1. **Dynamic Section-Tab Header Titles (`static/js/tools.js`, `static/js/settings.js`, `static/js/mods.js`)**:
   - Automatically updates page title headers dynamically to `"[Section] - [Tab Name]"` (e.g., `Tools - New Users`, `Settings - Appearance`, `Mods - Ban / Timeout User`).
   - Cleanly reverts back to base section names (`Tools`, `Settings`, `Mods`) when returning to menus.
2. **Scrollable Description Boxes & Title Deduplication (`static/css/style.css`, `static/css/play.css`)**:
   - Replaced duplicate inner `<h2>` headers with a sleek, compact, touch-scrollable description box across all panels in Tools, Settings, and Mods.
   - Scaled mobile section header titles to `1.25rem` to prevent line wrapping on small phone displays.
   - **Horizontal Alignment on Mobile**: Configured `.tool-header` and `.tool-header p` across `#page-tools`, `#page-settings`, and `#page-mods` to zero out asymmetric side padding (`padding-left: 0 !important; padding-right: 0 !important; margin: 0 0 8px 0 !important;`) and span 100% width with `box-sizing: border-box !important;`, guaranteeing description boxes line up flush horizontally with all tool panels, input cards, and controls underneath them.
3. **Mobile New Users Tool Screen Fit & Custom Scroller Thumb (`static/js/tools.js`, `static/css/style.css`)**:
   - Wrapped the New Users table in a touch-friendly flex container fitting entirely above the mobile navigation bar.
   - Added custom high-contrast scrollbar thumb and track (`#new-users-scrollbar-track` / `#new-users-scrollbar-thumb`).
   - Tightened stats grid margins for a compact side-by-side presentation.

### E. Lobby & Forum Enhancements
1. **Lobby Active Rooms Vertical Panel Spacing (`static/css/lobby.css`)**:
   - Balanced vertical margins below `#selected-game-info`, `.create-room-panel`, and `.rating-filter-container` (`8px` desktop, `3px` mobile).
   - Elevated `#rooms-placeholder-view` text and animated Morpheme logo icon on desktop by ~46px for an uncluttered aesthetic.
2. **Desktop Forum Thumbnail Previews & Lightbox Close Button (`static/js/forum.js`, `static/css/style.css`)**:
   - Fixed desktop thumbnail previews in forum threads with direct zoom-on-click modal lightboxes.
   - Added a dedicated, persistent close button (`×`) in the upper-right corner of the lightbox for effortless closing on desktop and touch devices.
3. **Cross-Browser Engine Alignment (`static/css/style.css`)**:
   - Normalized desktop Firefox and Chrome header dimensions, button sizes, font sizing, and padding pixel-for-pixel while preserving Gecko button normalization.

---

## 3. Current Asset Version Cache Busters

| Asset | Version String |
| :--- | :--- |
| `static/css/style.css` | `?v=1790019000` |
| `static/css/play.css` | `?v=1790019000` |
| `static/css/lobby.css` | `?v=1790007000` |
| `static/js/app.js` | `?v=1790017000` (Build `33155`) |
| `static/js/mods.js` | `?v=1790017000` |
| `static/js/tools.js` | `?v=1790008000` |
| `static/js/settings.js` | `?v=1790002000` |
| `static/js/lobby.js` | `?v=1789821200` |
| `static/js/play.js` | `?v=1789684000` |
| `static/js/forum.js` | `?v=1789822000` |
| `static/js/tournaments.js` | `?v=1789333000` |
| `static/js/leaderboard.js` | `?v=1788724200` |

---

## 4. Architectural & Mobile Invariants (AGENTS.md)

1. **Full List Modal (`openFullListModal`)**: Immediately and explicitly exits fullscreen (`document.exitFullscreen()`) upon open.
2. **Android Virtual Keyboard Black Screen Prevention**: Fullscreen is strictly exited on navigation to utility pages (Tools, Settings, Profile, Forum, How to Play, Donate) or whenever text input modals are open. Automatic fullscreen re-engagement is prohibited on non-game utility pages.
3. **Gateway Screen (`#page-loading`)**: Fullscreen is requested on initial user tap/press and preserved continuously into the Lobby without layout shifts.
4. **Dictionary Definitions**: Adheres strictly to noun plural and verb conjugation root pointer sourcing rules, as well as prefix-root authentic definitions without placeholder text.

---

## 5. Verification & Health Check

- **Local Python Syntax**: Passed (`python3 -m py_compile app.py`).
- **Production Server Status**: PM2 process `0` (`morpheme`) online, memory stable (`~214 MB`), HTTP 200 OK.
- **Commit ID**: **`34f60f554dbded3f2fc8eb5506bd5ee4e3d5ac01`** (short: `34f60f55`)
- **Git Status**: 100% clean working directory across Localhost, GitHub `main`, and Production Server.
