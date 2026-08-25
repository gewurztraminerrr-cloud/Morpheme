# Stable State Summary — August 25, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of August 25, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is fully synchronized.

---

## 1. Repository & Deployment Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 25, 2026
* **Latest Commit ID**: `524157b17e943d827f697e08ee352863176c9f28` (`524157b`)
* **Asset Version**: `v=33133`
* **Production Host**: `132.148.72.249` (`morpheme.games`)
* **Synchronization Status**: **100% Synchronized** across Localhost, GitHub, and Production (`morpheme.games`).

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Custom Draggable Scrollbars in Combo Checker (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Dedicated Scrollbar for Every Column**:
  - In `#page-tools` under `#tool-combo`, each rendered column (for MP groups `0MP`, `1MP`, `2MP`, `3MP`, `4MP`, `5MP`, `6MP` and all LIC groups `1LIC`, `2LIC`, `3LIC`, etc.) now contains its own custom scrollbar track (`.custom-scrollbar-track`) and glowing draggable thumb (`.custom-scrollbar-thumb`).
- **Smooth Dragging Without Interruption**:
  - Standardized wrapper DOM structure (`.list-scroll-area-wrapper` and `.list-scroll-area`) to mirror the reliable Lists in Tools scrollbar architecture.
  - Added an `isDragging` guard so active drag movements are never fought, reset, or jittered by concurrent scroll event triggers.
  - Attached `ResizeObserver`, `MutationObserver`, and post-layout frame triggers so thumb sizes and tracks calculate immediately upon column rendering and respond dynamically to resizing or filtering.
  - Generous hit areas, interactive `cursor: grab`/`cursor: grabbing`, and glowing cyan/blue gradients match the rest of the application.

### B. Moderator Access Hardened Exclusively to `jeffb` (`app.py`, `static/js/mods.js`)
- **Backend Authorization Guard**:
  - Endpoints `/api/mods/add` and `/api/mods/remove` strictly enforce `session['username'].lower() == 'jeffb'`, returning `403 Forbidden` for any other moderator or user.
  - Protected root moderator accounts (`jeffb`, `system`) from deletion or removal.
  - Refactored `save_moderator()` and `remove_moderator()` to use centralized database context manager `with get_db() as conn:`.
- **Frontend Dynamic Interface**:
  - `checkModStatus()` and `loadModList()` in `mods.js` verify `is_root` status (`jeffb`).
  - For non-`jeffb` moderators, the "Add Moderator" input field, add button, and remove (`×`) buttons are hidden from the DOM.
  - Tab description updated to reflect read-only mode for non-root moderators.

### C. Lists Tool Anti-Highlighting & Anti-Selection on All Devices (`static/css/play.css`, `static/js/tools.js`, `templates/index.html`)
- **Complete Text Selection & Drag Prevention**:
  - Added `-webkit-user-select: none !important; user-select: none !important; -webkit-user-drag: none !important; -webkit-touch-callout: none !important;` across `#tool-lists`, `#lists-container`, `#main-list-results`, `.list-scroll-area`, `.list-column`, and `.list-item`.
  - Added `::selection { background: transparent !important; color: inherit !important; }` across all Lists elements on mobile, tablet, desktop, and laptop devices.
  - Injected `onselectstart="return false;"`, `ondragstart="return false;"`, `oncopy="return false;"`, and `oncontextmenu="return false;"` event listeners and attributes to prevent word selection while preserving word lookup clickability.

### D. Tools Typography Standardization Across Desktops & Laptops (`static/css/play.css`, `templates/index.html`)
- **Standardized Tab Titles & Subtitles**:
  - Standardized all Tools tab titles (`#page-tools .tool-header h2`) to `1.8rem`, `font-weight: 700`, matching the "Word of the Day" title.
  - Standardized all Tools tab descriptions (`#page-tools .tool-header p`) to `0.95rem`, `margin: 0 0 20px 0`, matching the "Word of the Day" description.
  - Cleaned up inline font overrides and centered alignment across all Tools panes.

### E. 24h Room Score Sum Monotonic Persistence & Database Concurrency (`db.py`, `app.py`)
- **Permanent Monotonic Score Sums**:
  - Fixed an issue where total scores in the Score Sum tab in 24h rooms could be reset or overwritten. At midnight (12:00 AM), the score earned in the round just concluded is permanently added to the user's historical Score Sum.
  - A user's total in Score Sum will never decrease below their accumulated total.
- **SQLite Concurrency & WAL Hardening**:
  - Fully eliminated SQLite deadlocks and 504 Gateway Timeouts by configuring WAL mode (`PRAGMA journal_mode=WAL;`), 30-second busy timeouts (`PRAGMA busy_timeout=30000;`), and `PRAGMA synchronous=NORMAL;` across `db.py` and remote server.

### F. Asset Version Synchronization
- All script and stylesheet link tags in `templates/index.html` bumped to cache-busting version `v=33133`.

---

## 3. Verification & Server Health

* **Localhost Status**: Fully working, all tests passing.
* **GitHub Repository**: Pushed and up-to-date with commit `524157b17e943d827f697e08ee352863176c9f28`.
* **Production Deployment (`morpheme.games`)**:
  - PM2 Process `morpheme` (ID 0) online with `0%` CPU and healthy memory.
  - Remote Git Branch `main` aligned at `524157b`.
  - HTTP endpoints (`/`, `/api/rooms`, `/api/leaderboard`) responding in sub-millisecond times ($<1\text{ms}$).
  - Zero open unclosed database locks (`lsof morpheme.db`).
