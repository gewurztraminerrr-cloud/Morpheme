# Stable State Summary — November 8, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_NOVEMBER_8
```
*(Also tagged as `stable-2026-11-08`)*

## Latest Feature Commit ID
```
fa8b31e58e164af999dbc02452185e3394f2e0f1
```

## Synchronization Status

| Environment | Status | Commit / Tag | Verification Details |
|:---|:---|:---|:---|
| **Localhost** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_8` | Working tree clean on `main`, zero uncommitted tracked changes |
| **GitHub (`origin/main`)** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_8` | Pushed to `origin/main` & tags `START_OVER_POINT_NOVEMBER_8` / `stable-2026-11-08` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_8` | Fast-forward updated to `origin/main`, PM2 online (process `morpheme`), HTTP 200 OK |
| **App (Mobile WebView)** | ✅ Synchronized | `https://morpheme.games/` | Flutter app loads live synchronized webview with native bridges |

---

## Session Features & Fixes (November 8, 2026)

### 1. Inactive Player Room Roster Eviction at 10-Minute Boundary (`app.py`, `game_room.py`, `static/js/app.js`, `static/js/play.js`)
- **Immediate Roster Eviction Across All Devices**:
  - Addressed multi-browser / multi-platform presence divergence where a player inactive for 10 minutes was kicked to Lobby, but their username lingered in the game room Players list on mobile devices (whose screens went black/asleep) for an additional 10 minutes until screen wake.
  - Enforced instantaneous eviction from room rosters the moment the 10-minute inactivity threshold is reached, regardless of whether the user is on desktop, laptop, or mobile.
  - Implemented proactive backend auto-sweep in `game_room.py` and `app.py` that removes expired sessions directly from the room's active roster and immediately broadcasts updated roster state to all remaining players without waiting for a client-side wake ping.

### 2. Elimination of Empty / Zero-Match Words in Combo Checker MP Explorer (`app.py`, `static/js/tools.js`)
- **Non-Plural, Non-Inflected Verification Prior to Table Insertion**:
  - Resolved issue where input words such as `ABAMPERE` (Length 8, Diff 0, 1 MP) appeared in the left-hand candidate table, but clicking them yielded "No non-plural, non-inflection diff 0 1MP words found for ABAMPERE".
  - Updated `check_word_has_mp` in `app.py` to ensure candidate words are validated against matching non-plural, non-inflected partner words before being rendered in the MP Explorer candidate list.
  - Applied this filtering uniformly across all user configurations (Length, Difference, and MP Value settings) so that words without valid partners are never added to the candidate table.

### 3. Active Round Word Definition Lookups in Game Rooms (`static/js/play.js`)
- **Reliable Definition Lookups During Gameplay**:
  - Fixed an issue where clicking on a submitted word during an active round in game rooms failed to display its definition popup (particularly disruptive in 24h rooms where all words are played).
  - Eliminated conflicting DOM re-renders and inadvertent highlight clearing in `static/js/play.js`.
  - Ensured word definitions consistently load and display in active rounds without race conditions.

### 4. Intermission Word Click Handler Collision Resolution (`static/js/play.js`)
- **Intermission Definition & Board Tracing Integrity**:
  - Resolved an event listener collision where a global `#submitted-words-list` click listener intercepted clicks on intermission words, blocking the definition modal and board path tracing in `displayAllWords()`.
  - Added an intermission guard so clicking any word during intermission reliably renders its full definition, traces letter paths on the grid, and highlights finder indicators.

### 5. Lobby Placeholder Logo Display & Safari/WebKit Geometry Fix (`templates/index.html`, `static/css/lobby.css`, `static/js/lobby.js`)
- **Placeholder Logo Container Wrapping**:
  - Resolved issue where the Morpheme logo in the "No 'Show Rooms' selected" panel in the Lobby was invisible or collapsed.
  - Diagnosed a WebKit/Safari flexbox column bug causing `height: auto` SVGs to collapse to 0 height; resolved by wrapping the SVG in `<div class="placeholder-icon-wrap">` with explicit minimum and proportional dimensions (`min-width: 140px; min-height: 124px;` desktop, `min-width: 130px; min-height: 115px;` mobile).
- **Mobile Viewport Spacing & Drop Shadow Aesthetics**:
  - Tuned `#rooms-placeholder-view` padding on mobile (`padding: 24px 16px 16px`) to eliminate vertical overflow and unwanted scrolling.
  - Enhanced logo styling with dual cyan/teal (`rgba(56, 189, 248, 0.45)`) and white drop shadows on dark themes, with high-contrast shadows on light/white themes.
- **Dynamic Toggle Restoral**:
  - Added toggle-off / deselect support in `handleShowRoomsInline` and exposed `window.renderRoomsPlaceholderView()` so deselecting "Show Rooms" immediately restores the centered logo placeholder view.
  - Bumped asset query strings to `1794100000` to prevent caching issues.

### 6. Toggle / Retract Selected Words in Game Rooms Across Rounds and Intermission (`static/js/play.js`, `templates/index.html`)
- **Word Selection Retraction on Repeated Click**:
  - Implemented `toggleWordSelection(word, clickedElement)` in `static/js/play.js`.
  - When a user selects a word under Words or All Words (as well as FCFS feed, notepads, or previous day list) during active rounds or intermission, clicking that same word again immediately deselects/retracts it.
  - Instantly removes `.finder-active` highlight from word items, clears `.review-highlight` from board cells, removes `.finder-highlight` from player cards in the roster, and calls `clearDefinitionPanel()` to return the definition card back to its clean default state.
  - Bumped asset version query string to `1794101000`.
