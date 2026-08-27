# Stable State Summary — August 26, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of August 26, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is fully synchronized.

---

## 1. Repository & Deployment Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 26, 2026
* **Latest Commit ID**: `284bf668b57732a39396bb6b2e1bf743cfec9e76` (`284bf66`)
* **Production Host**: `132.148.72.249` (`morpheme.games`)
* **Synchronization Status**: **100% Synchronized** across Localhost, GitHub, and Production (`morpheme.games`).

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Drop-Down Menu Label Simplification: "Added Words" to "AW" (`templates/index.html`)
- **Clean Drop-Down Text Across Tools & Game Interfaces**:
  - Updated all select dropdown options containing the text "Added Words" across the application to display **"AW"** instead (Random Word Generator, Combo Checker, Word Validator, Subanagrams, Sequence Search, Word Lists filter, and Unscramble).
  - Maintained all existing `value` bindings (`value="added_words"` and `value="added"`), ensuring seamless compatibility and identical functional behavior across all backend endpoints, queries, and event handlers.

### B. Lobby Live Player Count Accuracy & Accumulative Room Sync (`app.py`, `game_room.py`, `static/js/lobby.js`, `templates/index.html`)
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
  - Bumped `lobby.js` cache-buster version to `v=33189`.

### C. New FAQ Entry for Game Room Window Navigation (`templates/index.html`)
- **Quick Navigate & Detailed FAQ**:
  - Added quick navigate button and accordion entry in the **How to Play & FAQ** modal:
    - **Question**: *It's difficult to slide over to neighboring windows in game rooms. Is there an easier way to do this?*
    - **Answer**: *The grey bar along the bottom not only is pressable, but swipeable as well. Get into the habit of swiping across or pressing on the text on it and the navigation eventually becomes instinctual!*

### D. Real-Time Timeout Countdown & Expiration Handling (`app.py`, `static/js/app.js`, `static/js/play.js`, `static/js/lobby.js`)
- **Dynamic Real-Time Countdown**:
  - Timeout modal updates remaining time second-by-second in real-time without requiring a page reload or app restart.
  - Automatically unblocks and returns the user to the Lobby or Active Room seamlessly the instant the timeout duration expires.
- **Moderation & Private Messaging Enforcement**:
  - Timed-out users are strictly prevented from sending private messages, creating rooms, or entering matchmaking queues until the timeout period has lapsed.

### E. User Profile & Interface Polishing (`static/js/profile.js`, `static/css/style.css`)
- **Profile Search Transition**:
  - Eliminated previous user profile layout flash on new profile searches.
  - Constrained "About Me" custom scrollbar thumb height strictly within its track boundary.

---

## 3. Verification & Server Health

* **Localhost Status**: Fully working, all syntax and integration tests passing.
* **Production Deployment (`morpheme.games`)**:
  - PM2 Process `morpheme` (ID 0) online with `0%` CPU and healthy memory.
  - Multi-user live integration tests passed verifying immediate reflection of `Start [1]` and clean reset to `Start [0]`.
  - Zero open unclosed database locks or runtime exceptions in logs.
