# Stable State Summary — August 27, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of August 27, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is fully synchronized.

---

## 1. Repository & Deployment Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 27, 2026
* **Latest Commit ID**: `cb360efdd887b5aaa7f976cbb03e886a6857cb69` (`cb360ef`)
* **Production Host**: `132.148.72.249` (`morpheme.games`)
* **Synchronization Status**: **100% Synchronized** across Localhost, GitHub, and Production (`morpheme.games`).

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Real Progressive Block Rendering & Live Displayed Counter in "View Full List" (`static/js/tools.js`, `templates/index.html`)
- **Real Progressive Block-by-Block Rendering into the Table**:
  - Replaced simulated timer animation with **actual block-by-block DOM rendering**: words physically appear and append into the table in alphabetical order (`A` ➔ `AA` ➔ `AAH` ➔ `AAHS` ...).
  - As each new block of words is rendered into the DOM, the top counter dynamically updates to reflect the exact number of words processed and loaded into the table: `${loadedCount.toLocaleString()} / ${total.toLocaleString()} words` (e.g. `200 / 469,764 words` ➔ `600 / 469,764 words` ➔ `1,000 / 469,764 words` ➔ `1,400 / 469,764 words` ...).
- **Synchronized Real-Time Scrollbar Shrink & Ascent**:
  - As each block of words renders into the table, the custom scrollbar thumb height shrinks progressively in proportion to the growing list and stays positioned at the top of the track.
  - When the user scrolls down through the list, subsequent blocks continue appending smoothly in alphabetical order, updating the counter and keeping the scrollbar thumb accurately scaled and responsive to touch/drag.
- **Persistent Virtual List Scrollbar Scaling**:
  - The custom scrollbar maintains virtual scaling across the full lexicon space, guaranteeing the thumb permanently stays small and sleek (28px minimum height) after loading finishes.

### B. Combo Checker LIC Tables Restoration & Dynamic Calibration (`app.py`, `static/css/play.css`, `templates/index.html`)
- **Dynamic LIC Letter Threshold**:
  - Replaced the static `shared_counts >= 5` constraint with dynamic scaling based on search term length:
    - 3–5 letter words: `min_lic_shared = 3` (e.g. 3LIC, 4LIC, 5LIC)
    - 6 letter words: `min_lic_shared = 4` (e.g. 4LIC, 5LIC, 6LIC)
    - 7+ letter words: `min_lic_shared = 5` (e.g. 5LIC, 6LIC, 7+LIC)
  - Enables full LIC generation across all word lengths (including 3-, 4-, and 5-letter queries).
- **Mobile Container Containment Cleanup**:
  - Removed `contain: layout paint;` and `contain: content;` from `#tool-combo.tool-pane.active` and `.combo-results-container` on mobile, ensuring iOS WebKit renders both MP and LIC sections below the fold with smooth vertical touch scrolling.

### C. Full Lexicon Uncapped Streaming & In-Memory Caching for "View Full List" (`static/js/tools.js`, `templates/index.html`)
- **Uncapped Full Lexicon Streaming**:
  - Removed premature binding to the capped 1,000-word page list.
  - Fetches the complete word list (`no_limit=true`, 469,764 words for AW / 199,429 words for NWL / 281,598 words for CSW) and caches it in `window._cachedFullWordLists`.

### D. Global Scope & Immediate Availability for "View Full List" Modal (`static/js/tools.js`, `templates/index.html`, `static/css/play.css`)
- **Global Function Availability**:
  - Moved `window.openFullListModal`, `window.closeFullListModal`, and jump helpers out of closure scopes and declared them at top-level global scope in `tools.js` to ensure immediate availability upon script parse.
  - Added direct runtime DOM hoisting (`if (modal.parentElement !== document.body) document.body.appendChild(modal);`) to guarantee the modal is a direct child of `<body>` with `z-index: 9999999 !important`.

### E. Real-Time Dynamic Scrollbar Scaling & Thumb Positioning Across All Devices (`static/js/tools.js`, `templates/index.html`)
- **Universal Dynamic Thumb Scaling for Mobile, Laptops & Desktops**:
  - Integrated `MutationObserver` and live batch notification hooks into `initCustomScrollbarForElement()`.
  - Fully enabled for desktops, laptops, tablets, and mobile devices across all custom scrollbars in Tools (Word Lists, View Full List modal, Sequence Search, Subanagrams, Combo Checker, and Unscramble Session History).

### F. Unscramble Session History Scrollbar Thumb Movement Fix (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Resolved Static Thumb Locking**:
  - Removed `top: 0 !important;` from the `#unscramble-history-scrollbar-thumb` CSS rule in `play.css`, which was overriding runtime JavaScript position updates.
  - Hardened `initCustomScrollbarForElement()` in `static/js/tools.js` to update thumb top coordinates via `setProperty('top', ..., 'important')` and real-time scroll synchronization.

### G. Mobile Touch Responsiveness & Scrolling for Unscramble Session History (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Enhanced Mobile Touch & Scroll Propagation**:
  - Configured `-webkit-overflow-scrolling: touch !important; touch-action: pan-y !important; overscroll-behavior-y: contain !important;` on `.unscramble-history-list` so swiping anywhere within the Session History pane scrolls the history list smoothly on iOS and Android.
  - Added extended invisible touch hit targets (`::before` pseudo-element spanning 40px) to `#unscramble-history-scrollbar-thumb` and widened the track on mobile viewports (`width: 20px`), allowing fingers to grab and drag the thumb immediately.

### H. Elimination of Word Pill Vertical Clipping in Unscramble Session History (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Complete Word Pill Visibility**:
  - Enforced `min-height: fit-content !important; height: auto !important; padding: 12px 14px 14px 14px !important; overflow: visible !important;` on `.unscramble-history-item`.
  - Structured `.unscramble-history-words` with `min-height: 36px !important; height: auto !important; overflow: visible !important;` and `.clickable-word-link` with `display: inline-flex !important; align-items: center !important; justify-content: center !important; min-height: 34px !important; height: auto !important; line-height: 1.2 !important; border-radius: 8px !important; text-decoration: none !important;`.
  - Completely resolved the issue where the bottom half of word pills and letters were cut off by card boundaries.

### I. Unscramble Session History Single-Row Mobile Header Layout (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Single-Row Alignment on Mobile Devices**:
  - Enforced `flex-wrap: nowrap !important;` on `.unscramble-history-header` so the jumbled word, the `0/1 Found` / `1/1 Found` status pill, and the timestamp remain neatly aligned on the exact same horizontal row on mobile screens.
  - Scaled typography and padding (`0.95rem` for jumbled word, `0.72rem` for status pill, `0.7rem` for timestamp) to fit effortlessly across all small viewport widths with zero wrapping or truncation.

### J. Unscramble Session History Custom Draggable Scroller & Thumb (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Custom Draggable Scroller with Thumb**:
  - Integrated the custom scrollbar system (`initCustomScrollbarForElement`) into Unscramble's Session History with a dedicated track (`#unscramble-history-scrollbar-track`) and touch-responsive glowing cyan thumb (`#unscramble-history-scrollbar-thumb`).
  - Hides native browser scrollbars in favor of the custom draggable thumb, dynamically appearing whenever history rounds exceed the container height.
  - Sized the scroll container with comfortable height (`max-height: 380px` on desktop, `max-height: 350px` on mobile) with dedicated right padding to give full space to word badges without overlapping the track.

### K. Unscramble Session History "1/1 Found" Formatting & Layout (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Standardized "X/Y Found" Status Pill**:
  - Formatted rounds where all solutions are found to read cleanly as `1/1 Found` (or `X/Y Found`) in green (`color: #4ade80; background: rgba(46, 204, 113, 0.18); border: 1px solid rgba(46, 204, 113, 0.4);`) matching the structure of `0/1 Found`.
  - Positioned the completion timestamp (`HH:MM:SS`) immediately to the right of the status pill.
  - Sized and placed played words into a dedicated full-width flex container (`.unscramble-history-words`) with zero vertical or horizontal clipping on mobile devices.

### L. Mobile Tap Highlight & Blue Flash Removal on "View Full List" (`static/css/play.css`, `templates/index.html`)
- **Eliminated Mobile Blue Flash & Browser Focus Ring**:
  - Enforced `-webkit-tap-highlight-color: transparent !important;`, `-webkit-touch-callout: none !important;`, `outline: none !important;`, and `-webkit-focus-ring-color: transparent !important;` on `#list-view-full-btn`.
  - Scoped `:focus`, `:focus-visible`, `:active`, and `:hover` states to maintain consistent lavender theme styling (`rgba(167, 139, 250, 0.25)` background with `rgba(167, 139, 250, 0.7)` border) with zero blue flash or blue border on touch.

### M. Lists Tool Header Cleanup & "View Full List" Button Placement (`templates/index.html`, `static/js/tools.js`)
- **Replaced "Top Options" with "View Full List"**:
  - Removed the `"▲ Top Options"` button from the Lists tool header in Tools.
  - Positioned the `"⛶ View Full List"` button directly on the right side of the list header (`#list-column-header-title`), creating a clean, balanced layout alongside the dictionary title and count badge.

### N. FAQ AW Dictionary Description Refinement (`templates/index.html`)
- **Explicit Mention of 30,000 Filtered Words**:
  - Updated the **"What are the different dictionaries used?"** entry in the FAQ to explicitly state that close to 30,000 unwanted entries were removed from Wiktionary's word list because they were either obsolete terms, abbreviations, or misspellings.

### O. Deduplication of 16+ Words Between AW & 16+ List (`dictionaries/added_words.txt`, `dictionaries/16plus.txt`)
- **6,128 Duplicate 16+ Words Removed from Added Words (AW)**:
  - Deduplicated `dictionaries/added_words.txt` by removing all 6,128 words of length 16+ that already exist in the official supplementary 16+ List (`dictionaries/16plus.txt`).
  - Result: `16+ AW` now contains **38,242 unique words** with **0 overlap** with the **16+ List** (9,227 unique words). Total words in `added_words.txt` updated from 479,310 to **473,182**.

### P. Instant Zero-Latency Room Loading & Synchronous Board Readiness (`app.py`, `static/css/style.css`, `static/js/app.js`, `static/js/lobby.js`, `static/js/play.js`, `templates/index.html`)
- **Synchronous Board Readiness on Creation**:
  - `create_room()` in `app.py` now guarantees the room's first round board is popped and initialized synchronously before returning, eliminating the asynchronous thread gap where the frontend previously had to wait through multiple poll cycles for the board to generate.
  - Eliminated redundant network pre-fetch roundtrips (`/api/user/my_timeout_status`).
  - Total latency from clicking **"Start"** in the Lobby to the board rendering with all tiles and timer is now reduced to **~50–100 milliseconds**.
- **Glassmorphism Backdrop Blur**:
  - Features an atmospheric semi-transparent dark backdrop (`rgba(10, 15, 30, 0.65)`) with native cross-browser background blurring (`backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);`), glowing cyan spinner, and centered `"Loading..."` typography.
- **Guaranteed Immediate Dismissal on Board Render**:
  - Fixed CSS display override rules (`#global-loading-overlay.hidden` and inline `setProperty('display', 'none', 'important')`).
  - `hideLoadingOverlay()` is invoked directly from `renderBoard()`, `renderSplitNotepads()`, `renderFCFSNotepads()`, and `updateGameState()`, ensuring the overlay immediately and completely vanishes the exact millisecond the game board renders behind it.

### Q. Performance Efficiency (PE) Calibration & Dominant Score Trophy Awards (`game_room.py`, `app.py`, `static/js/play.js`, `templates/index.html`)
- **Mathematical PE Calibration**:
  - Calibrated Performance Efficiency ($\text{PE}$) dynamic evaluation so players who double the score of opponents near their rating or dominate the room are reliably awarded the trophy icon 🏆 during intermission:
    - **2 players**: $\text{PE} \ge 1.30$ (corresponds to doubling opponent's score)
    - **3 players**: $\text{PE} \ge 1.45$ (e.g. 63 pts vs 26 & 25 pts yields $\text{PE} = 1.60\text{x}$)
    - **4–5 players**: $\text{PE} \ge 1.55$
    - **6+ players**: $\text{PE} \ge 1.60$
  - Added direct blowout domination award: $\text{score} == \text{max\_score}$ and $\text{score} \ge 1.75 \times \text{second\_score}$ ($\text{score} \ge 15$).
  - Enabled PE calculation for guest participants with 1200 rating baseline.
- **Strict Intermission Display & Automatic Start-of-Round Removal**:
  - Trophies awarded are displayed strictly in the bottom-left corner of the player card during that round's **intermission**.
  - At the **start of the next round**, the trophy icon automatically disappears from all players on both backend state transitions and frontend UI rendering.

### R. Mobile Profile Search Input & Button Height Alignment (`static/css/play.css`, `static/css/style.css`, `templates/index.html`)
- **Uniform 46px Height Matching the "Search" Button**:
  - Standardized `#profile-search-input` (`"Enter username..."`) on mobile devices to `height: 46px !important; min-height: 46px !important; max-height: 46px !important; padding: 10px 16px !important; font-size: 16px !important; box-sizing: border-box !important;`.
  - Matched the exact vertical height and curvature of the **"Search"** button (`#profile-search-btn`) and **"My Profile"** button (`#profile-my-profile-btn`) beneath it.

### S. Level Alignment for Rotate & Transpose Buttons on Mobile (`static/css/play.css`, `templates/index.html`)
- **Fixed Button Offset / Vertical Shifting on Press**:
  - Scoped out generic `.rotate-btn:hover` translateY shifts from timer controls.
  - Enforced `transform: none !important; box-shadow: none !important; vertical-align: middle !important;` across all pseudo-states (`:hover`, `:active`, `:focus`, `:focus-visible`) for `#rotate-board-btn`, `#transpose-board-btn`, and `.timer-display .rotate-btn` on mobile devices.
  - Pressing or tapping either button maintains flush, level horizontal alignment with its partner button without jumping or tilting.

### T. Cross-Browser Lobby Music on "ENTER LOBBY" Gateway (`static/js/app.js`, `templates/index.html`)
- **Resolved Firefox & Safari Audio Gatekeeper Block**:
  - Added native `autoplay` attribute directly to the `#lobby-music` `<audio>` element so browsers with native media autoplay policies initiate playback immediately during HTML parsing.
  - Added `Permissions-Policy: autoplay=(self)` meta tag and hidden `allow="autoplay"` frame delegation for browser engines requiring explicit policy declarations.
  - Implemented muted buffer fallback (`audio.muted = true; audio.play()`) so Safari and Firefox immediately buffer and start playback without engine rejection, and instantaneously unmute upon any gesture (`pointermove`, `pointerdown`, `touchstart`, `focus`, `click`).
  - Attached eager capture triggers directly to `#btn-enter-lobby-gateway`, `#gateway-housing`, and the `#page-loading` container.
  - Added Web Audio Context unlock (`AudioContext.resume()`) and multi-lifecycle event triggers (`canplay`, `loadeddata`, `DOMContentLoaded`, `load`, `focus`, `visibilitychange`) so Safari and Firefox immediately engage music playback upon presentation of the "ENTER LOBBY" button.
  - Integrated `handleLobbyMusicState()` into `showPage()` for seamless audio continuity during SPA page transitions.

### U. Desktop Color Chart Repositioning & Downward Mini-Popups (`static/css/style.css`, `templates/index.html`)
- **Color Chart Relocation**:
  - Moved the horizontal rating tier color chart (`#game-color-bar`) on desktops and laptops to above the game header (`.play-header`) and directly below the top navigation menu / separator.
- **Downward Mini-Popups (Tooltips)**:
  - Repositioned the tier name and rating range hover popups to appear smoothly downward (`top: calc(100% + 7px)`) with inverted upward-pointing indicator arrows.
  - Added boundary constraints for the leftmost (`:first-child`) and rightmost (`:last-child`) segments to prevent viewport clipping.

### V. Combo Checker Virtualized Column Rendering & Infinite Scrolling (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- **Eliminated 3-Second Mobile Viewport Freeze on Re-Focus**:
  - Replaced dumping 30,000–50,000 simultaneous DOM nodes in MP and LIC result columns with a fast, chunked rendering architecture (100 words initially rendered per column) and smooth infinite scrolling upon reaching the bottom of a column.
  - Added `content-visibility: auto; contain-intrinsic-size: 0 28px;` to `.group-row` to bypass layout calculation for off-screen table rows.
  - Throttled custom scrollbars with `requestAnimationFrame()` and removed deep DOM mutation observers.
  - Kept `#combo-input` font-size at `16px` to prevent iOS Safari auto-zoom recalculations.

### W. Combo Checker Textbox & Button Height Sizing on Mobile (`static/css/play.css`, `templates/index.html`)
- **Uniform 46px Height Matching the "Check" Button**:
  - Sized `#combo-input` (`"ENTER LETTERS TO CHECK..."`) to `height: 46px; min-height: 46px; max-height: 46px; padding: 10px 16px; font-size: 16px; border-radius: 8px;` matching the exact vertical dimensions and curvature of the **"Check"** button (`#combo-search-btn`) and dictionary dropdown (`#combo-dict`).

### X. Smooth Sliding Transitions on Bottom Navigation Buttons (`static/js/play.js`, `templates/index.html`)
- **Animated Panel Sliding**:
  - Configured `window.switchPlayPanel()` to default to smooth scrolling (`smooth = true`), triggering animated sliding when pressing **"Board ➜"** in the Players panel, **"← Board"** in the Words panel, **"← Players"**, and **"Words ➜"** in the Board panel.
  - Preserved instant zero-latency snaps (`smooth = false`) for room initialization, orientation change recoveries, and programmatic panel resets.

### Y. Mobile Intermission Bottom Navigation Bar in FCFS, SP, and CUBE Rooms (`static/css/play.css`, `templates/index.html`)
- **Always-Visible Bottom Navigation Bar**:
  - Removed the `display: none !important` override on `.board-panel.has-split-notepads .mobile-nav-indicators` so the bottom grey navigation bar with `"<- Players"` and `"Words ->"` remains accessible during intermission in **FCFS**, **SP (Split Points)**, and **CUBE (3D)** rooms on mobile devices.
- **Dedicated Flex Layout & Dedicated Bottom Clearance**:
  - Structured `.board-panel.has-split-notepads` as a full-height column flex container, allocating `calc(100% - 44px)` to the scrollable notepads container and pinning the 44px bottom navigation bar (`flex-shrink: 0; z-index: 100;`).
  - Allows users to swipe and tap seamlessly between the **Players**, **Board/Notepads**, and **Words** panes without notepads overlapping the navigation bar.

### Z. Profile and Tools Controls Centering & Equal Panel Padding (`static/css/play.css`, `templates/index.html`)
- **Centered Search, Buttons, and Inputs**:
  - Centered the username search input, **Search** button, and **My Profile** button horizontally in the Profile control panel card (`justify-content: center; align-items: center;`).
  - Standardized uniform 46px heights across profile search inputs and action buttons.
  - Symmetrized card padding (`padding: 20px 24px` on desktop, `padding: 14px 16px` on mobile, `box-sizing: border-box`), ensuring equal space surrounds contents on all four sides.
  - Applied symmetrical card padding and centered layouts across all Tools tabs (**Subanagrams**, **Sequence Search**, **Word Lists**, **Word Validator**, **Combo Checker**, **Unscramble**, and **Random Word Generator**).

### AA. Drop-Down Menu Label Simplification: "Added Words" to "AW" (`templates/index.html`)
- **Clean Drop-Down Text Across Tools & Game Interfaces**:
  - Updated all select dropdown options containing the text "Added Words" across the application to display **"AW"** instead (Random Word Generator, Combo Checker, Word Validator, Subanagrams, Sequence Search, Word Lists filter, and Unscramble).
  - Maintained all existing `value` bindings (`value="added_words"` and `value="added"`), ensuring seamless compatibility and identical functional behavior across all backend endpoints, queries, and event handlers.

### AB. Lobby Live Player Count Accuracy & Accumulative Room Sync (`app.py`, `game_room.py`, `static/js/lobby.js`, `templates/index.html`)
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

### AC. FAQ Entry for Game Room Window Navigation (`templates/index.html`)
- **Quick Navigate & Detailed FAQ**:
  - Added quick navigate button and accordion entry in the **How to Play & FAQ** modal explaining that the bottom grey bar is both pressable and swipeable for swift panel navigation.

---

## 3. Verification & Server Health

* **Localhost Status**: Fully working, all syntax and integration tests passing.
* **Production Deployment (`morpheme.games`)**:
  - PM2 Process `morpheme` (ID 0) online with healthy memory.
  - Multi-user live integration tests passed.
  - Zero open unclosed database locks or runtime exceptions in logs.
