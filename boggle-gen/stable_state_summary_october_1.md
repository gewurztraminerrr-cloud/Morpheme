# Stable State Summary — October 1, 2026

## Latest Feature Commit ID
```
953ab9ef0d45ee9659b85558eef7a15a81cae932
```

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_1
```
*(Also tagged as `stable-2026-10-01`)*

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| localhost      | ✅ Synchronized | Working tree clean |
| GitHub (main)  | ✅ Synchronized | Pushed to origin/main |
| morpheme.games | ✅ Synchronized | Deployed, HTTP 200 OK, PM2 Online |
| App / Client   | ✅ Synchronized | Web client & backend endpoints verified |

---

## Session Features & Fixes (October 1, 2026)

### 1. Gateway Screen Music Autoplay
- **Lobby Music on Gateway Screen**:
  - Music begins playing immediately when the `#page-loading` gateway screen (`ENTER LOBBY` button) appears, instead of waiting for the user to click into the Lobby.
  - Initialized both via early Web Audio API pre-decoded buffer and HTML5 audio element fallback in `templates/index.html`.
  - Registered one-time capture gesture listeners to `window` for `['pointerdown', 'touchstart', 'mousedown', 'keydown']` so if browser autoplay policy blocks zero-gesture audio, the very first touch anywhere unlocks and plays audio instantly.
  - In `static/js/app.js`, updated `handleLobbyMusicState()` and `playMusicOnFirstInteraction()` to treat `activePage === 'page-loading'` as eligible for music playback (`onLobby = (activePage === 'page-lobby' || onGateway)`).
  - All gateway button styling, CSS classes, physical elevation, and fullscreen transition mechanics remain in their baseline stable state.

### 2. Hardened Added Words Definition Engine for Regular Plurals (`GUTTUSES`)
- **Loose Web Search Regex Elimination**:
  - Previously, `lookup_web_search_definition` matched unrelated text when querying words ending in `-es` or other suffixes (e.g. matching an eaves trough definition from "gutter" when querying "GUTTUSES").
  - Hardened Match 1 and Match 5 in `lookup_web_search_definition` to enforce `\bThe meaning of [word] is\b` and require the target word to directly precede or accompany the noun definition snippet.
- **Plural & Suffix Resolution Priority in Added Words**:
  - In `ensure_aw_definitions_for_words`, morphological suffix resolution (`-IES`, `-ES`, `-S`, `-ED`, `-ING`, `-ERS`) now runs **first**, guaranteeing that regular plural forms inherit their root definitions directly without unverified web lookups polluting cache or database tables.
- **Validation Pre-check Database Safeguards**:
  - In `get_word_definitions_for_aw_check`, unverified web search results are no longer written to `morpheme.db` during pre-validation checks.

### 3. Comprehensive Classical & Irregular Plural Resolution (`LOCHI`)
- **Classical Latin, Greek, and Irregular Plural Patterns**:
  - Expanded `ensure_aw_definitions_for_words` and `get_definition_cached_or_online_with_guess` to support classical declensions so any irregular or classical plural entered in Added Words inherits its root definition:
    - **`-I` → `-US`**: (e.g. `LOCHI` → `LOCHUS`, `CACTI` → `CACTUS`, `ALUMNI` → `ALUMNUS`, `FUNGI` → `FUNGUS`, `SYLLABI` → `SYLLABUS`)
    - **`-AE` → `-A`**: (e.g. `LARVAE` → `LARVA`, `ALGAE` → `ALGA`, `NEBULAE` → `NEBULA`)
    - **`-A` → `-UM` / `-ON`**: (e.g. `STRATA` → `STRATUM`, `BACTERIA` → `BACTERIUM`, `CRITERIA` → `CRITERION`, `PHENOMENA` → `PHENOMENON`)
    - **`-ICES` → `-EX` / `-IX`**: (e.g. `VERTICES` → `VERTEX`, `MATRICES` → `MATRIX`, `INDICES` → `INDEX`)
    - **`-ES` → `-IS`**: (e.g. `CRISES` → `CRISIS`, `THESES` → `THESIS`, `SYNOPSES` → `SYNOPSIS`, `AXES` → `AXIS`)
- **Pointer Snippet Capture**:
  - Added regex pattern capture in `lookup_web_search_definition` to recognize dictionary pointer snippets (`[word] definition: plural of [target]`).
- **Guaranteed Target Definition Expansion**:
  - In `ensure_aw_definitions_for_words`, if any definition is a pointer (`plural of [target]`, etc.) without an existing parenthetical definition, the engine extracts `[target]`, resolves its definition, and appends it within `(...)` per `AGENTS.md` rules.

### 4. Synchronized Dictionary & Database Entries
- **`GUTTUS` & `GUTTUSES`**:
  - `GUTTUS`: `(noun) A type of ancient Greek and Roman vessel designed for pouring liquids.`
  - `GUTTUSES`: `plural of guttus (A type of ancient Greek and Roman vessel designed for pouring liquids.)`
- **`LOCHUS` & `LOCHI`**:
  - `LOCHUS`: `(noun) In ancient Greece, a body of infantry; in Sparta, one of the larger divisions in which able-bodied men were grouped.`
  - `LOCHI`: `plural of lochus (In ancient Greece, a body of infantry; in Sparta, one of the larger divisions in which able-bodied men were grouped.)`
- Synchronized across `Definitions.txt`, `wikdefs.txt`, `wikdefs_duplicate.txt`, and production `morpheme.db`.

### 5. Mobile Fullscreen & Continuous Fast Navigation Hardening
- **Single-Entry Fullscreen Request (Once on First App Entry)**:
  - Fullscreen is requested synchronously only when the user first enters the app via the gateway button (`ENTER LOBBY` / `LOGIN`).
  - The Android OS *"morpheme.games — to exit full screen…"* notice is presented once upon entering the app, as designed.
- **Elimination of Post-Minimize Re-engagement**:
  - Completely removed `visibilitychange`, `pageshow`, `focus`, and global `pointerdown`/`touchstart` background listeners that previously called `attemptFullscreen(false)` whenever a user tapped after returning from minimizing.
  - Users can minimize Morpheme, switch apps, return, and tap anywhere on the screen without triggering the Android OS notice or the 3-button system navigation bar.
- **Permanent Fullscreen Invariants Established (`.agents/AGENTS.md`)**:
  - Fullscreen is preserved continuously across all pages, top menu tabs (**Lobby**, **Play**, **Tools**, **Mods**, **Leaderboard**, **Settings**, **Profile**, **Forum**, **How to Play**, **Donate**), the Lobby Chat Drawer, textboxes, and dropdowns.
  - The Full List Modal (`openFullListModal`) in `tools.js` remains the **only** modal where `document.exitFullscreen()` is explicitly called.

### 6. Mobile Virtual Keyboard Chatbox Elevation
- **Interactive Widget Viewport Meta**:
  - Added `interactive-widget=resizes-content` to the viewport `<meta>` tag in `templates/index.html`.
- **Dynamic Viewport Tracking (`window.visualViewport`)**:
  - In `static/js/lobby.js` and `static/js/play.js`, wired `window.visualViewport` resize and scroll listeners to dynamically measure the virtual keyboard height (`window.innerHeight - visualViewport.height`).
  - Dynamically elevates `.lobby-chat-drawer` and `.chat-panel` with `style.bottom = kbHeight + 'px'` while keyboard is open (`.keyboard-open`).
  - Anchors the chat textbox and "Send" button directly above the keyboard, preventing the software keyboard from covering user input.
  - Smoothly dismisses and restores layout upon blur or closing the chat drawer/panel.

### 7. Expanded Mobile Chat Message Viewing Window
- **Lobby Chat Drawer (`.lobby-chat-drawer.keyboard-open`)**:
  - Expanded `.lobby-chat-history` `max-height` to `250px` (with responsive ceiling `min(260px, 38dvh)` and `min-height: 140px`), more than doubling previous visible message capacity so 5–7 messages are visible at once while typing.
  - Expanded `.lobby-chat-slide-panel` `max-height` up to `min(420px, 60dvh)` and `.lobby-slide-body` to `min(390px, 56dvh)`.
  - Tightened inner padding to dedicate maximum screen space to message history while hiding the player list during active typing.
  - Updated scroll listeners in `lobby.js` to smoothly scroll to the latest messages at 50ms, 150ms, and 300ms as the keyboard finishes rising.
- **Game Room Chat Panel (`.chat-panel.keyboard-open`)**:
  - Expanded `#chat-history` `max-height` to `min(260px, calc(100dvh - 370px))` with `min-height: 140px`, allowing 6–8 chat messages to be read comfortably while typing.
  - Expanded `.chat-panel` `max-height` up to `min(360px, calc(100dvh - 280px))`.
  - Updated scroll listeners in `play.js` to ensure the newest chat messages are scrolled into view when the input is focused.

---

## Verification & Health Check

1. **Production Health & Endpoints**:
   - `GET https://morpheme.games/` → `HTTP/2 200 OK`.
   - PM2 Process `0` (`morpheme`) online, active (1.5 GB memory).
2. **Asset Cache Busters**:
   - `app.js`: `v=1790770000`
   - `style.css`: `v=1790645000`
   - `lobby.css`: `v=1790680000`
   - `play.css`: `v=1790680000`
   - `lobby.js`: `v=1790680000`
   - `play.js`: `v=1790680000`
   - Viewport meta: `interactive-widget=resizes-content`
3. **Definitions Engine Verification**:
   - `GET https://morpheme.games/api/definition?word=GUTTUSES` → returns `"plural of guttus (A type of ancient Greek and Roman vessel designed for pouring liquids.)"`
   - `GET https://morpheme.games/api/definition?word=LOCHI` → returns `"plural of lochus (In ancient Greece, a body of infantry; in Sparta, one of the larger divisions in which able-bodied men were grouped.)"`
4. **Git & Production Synchronization**:
   - All code synchronized across `localhost`, `github.com/gewurztraminerrr-cloud/Morpheme`, and `morpheme.games`.
