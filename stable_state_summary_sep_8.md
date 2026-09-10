# Stable State Summary – September 8, 2026

## Latest Commit Information
- **Stable Save Point**: September 8, 2026
- **Active Git Tags**:
  - `START_OVER_POINT`
  - `START_OVER_POINT_SEPTEMBER_8`
  - `stable-2026-09-08`
  - `save-point-latest`
  - `start-over`

---

## Synchronization Status
- **Localhost (`/Users/jeffbabiak`)**: Synchronized
- **GitHub (`origin/main` & Tags)**: Synchronized
- **Production Server (`132.148.72.249` / `morpheme.games`)**: Synchronized, PM2 online, HTTP 200 OK
- **Flutter Mobile App (`morpheme_word_game`)**: Synchronized (loads `https://morpheme.games/` with native SoLoud audio bridge)

---

## Features & Stable Specifications (September 8 Checkpoint)

### 1. Mobile Lobby Grey Scrollers
- **Visual Overhaul**: Updated all thin scrollers across the Lobby on mobile devices from cyan/blue (`rgba(0, 240, 255, ...)`) to a clean, subtle semi-transparent grey (`rgba(var(--text-primary-rgb), 0.35) rgba(255, 255, 255, 0.04)`).
- **Panels Updated**: Applied consistently to the Game Types panel, Solo & Friends panel, Active Rooms list (`#rooms-list`), and the Lobby Players list.
- **Removed Glows**: Eliminated neon gradients and cyan glow box-shadow effects in favor of clean flat thumbs.

### 2. Settings Mobile Tab Scroller Tracking
- **Container Overflow Correction**: Resolved an issue where `#page-settings` had `overflow-y: auto`, producing an outer vertical scrollbar on mobile that remained stationary because touch scrolling was intercepted by the inner tab container. Changed `#page-settings` to `overflow-y: hidden !important` and sized `#page-settings .tools-split-layout` to fit cleanly within the viewport.
- **Active Scroller**: Explicitly re-enabled and styled custom thin grey scrollbars (`width: 6px`, `rgba(var(--text-primary-rgb), 0.35)`) on `#page-settings .tools-content` so the scrollbar actively tracks and moves as the user scrolls content in Appearance, Audio, and Gameplay tabs.

### 3. Room Rating Limits Creator Range Enforcement
- **Validation**: When a room creator applies rating limits (`min_rating` or `max_rating`), the system validates that the creator's own rating for the specified game configuration falls within the designated range (`min_rating <= rating <= max_rating`).
- **Standard Modal Feedback**: If the creator's rating falls outside their required range, room creation is blocked and a standard popup modal informs the user:
  *“Your rating ([rating]) does not fall within your specified range ([min] - [max]). Please ensure your rating is in your specified range for room creation to continue.”*
- **Range Inversion Protection**: Disallows setting a minimum rating greater than the maximum rating.

### 4. Skill Rankings 24-Hour Removal & Room Rating Fallback Fix
- **Profile Skill Rankings Cleanup**: Removed all 24-Hour game configurations from the Skill Rankings table on user Profiles and removed the "24 Hours" option from the Time Limit dropdown filter. Purged legacy 24h rows from the `user_ratings` database table.
- **Rating Fallback Fix**: Resolved an issue where creating a room with specific parameters (e.g. FCFS 4x6, 45s) fell back to a different configuration's rating (e.g., 4x4 24h at 1282) instead of the user's actual rating for the room (1200). Updated both backend rating resolution and frontend `getUserConfigRating` matching logic.

### 5. Word Validator ("Is Valid") Mobile Text Cutoff Prevention
- **Dynamic Ceilings**: Added `maxCeiling` (48px on mobile `innerWidth <= 900`, 68px on desktop) so short words (`HELLO`, `CAT`, `HI`) stay readable without vertically overflowing the container.
- **Unconstrained Results Container**: Configured `#tool-is-valid #valid-results-container` with natural expansion (`height: auto !important; overflow: visible !important;`), allowing word results and definitions to expand and scroll smoothly.
- **Descender Protection**: Configured `flex-shrink: 0 !important;` and bottom padding on `.valid-status-val` to guarantee bottom descenders are never clipped.

### 6. Store in Tools Scrolling Enabled & Visible Scrollbars Restored
- **Card Scrolling**: Enabled vertical scrolling on each individual store product card (`overflow-y: auto !important; overflow-x: hidden !important; -webkit-overflow-scrolling: touch !important;`).
- **Scrollbar Styling**: Configured styled scrollbars (`scrollbar-width: thin !important;` and custom WebKit scrollbars).
- **Action Buttons Reachable**: Sized `.store-item` and added `flex-shrink: 0 !important;` to `.buy-now-btn` so "VIEW ON AMAZON" action buttons are completely visible and clickable.

### 7. Lists in Tools Truncation Notice Guidance
- **Notice Update**: Updated the 10,000-word limit banner in `static/js/tools.js` to:
  *“Showing the first 10,000 words. Please select the parameters above for more words and/or use “View Full List””*
- **Interactive Trigger**: Embedded clickable modal opener (`openFullListModal()`) directly within the notice text.

### 8. Store in Tools Green Book & Dual Currency Pricing
- **Product Update**: Replaced the black book with the green *Collins Official Scrabble Words (2-15 All Words)* edition (cover image, Amazon link, description).
- **Dual Currency**: Converted all 13 items in Store to dual CAD and USD pricing (e.g. `$21.99 CAD / $16.99 USD`).

### 9. Mobile Game Room Navigation Tap/Press Highlight Removal
- **Touch State Cleanup**: Eliminated the default blue press/tap highlight (`-webkit-tap-highlight-color: transparent !important; outline: none !important; user-select: none !important;`) on "Board", "Words", and "Players" bottom navigation buttons on mobile devices.

### 10. Added Words Duplicate Detection & List Preservation
- **Inclusive Search**: Enhanced duplicate checks in `app.py` to search through the `Added Words` file in addition to CSW/NWL/16plus dictionaries.
- **Position Preservation**: Words already in AW remain in their original positions and do not get moved to the top.
- **Top Positioning for FUNGATE**: Positioned `FUNGATE`, `FUNGATES`, `FUNGATED`, and `FUNGATING` at the top of the Added Words list.

### 11. Lobby Journey Banner Horizontal Expansion on Desktops
- **Desktop Sizing**: Expanded the horizontal length of the Lobby journey banner (`ENTER A ROOM TO CONTINUE YOUR JOURNEY`) across all desktop resolutions (`1024px`, `1280px`, `1366px`, `1440px`, `1920px`) with `white-space: nowrap !important;`, eliminating text clipping.

### 12. Lobby Rating Filter Enter & Blur Behavior & Dedicated FAQ
- **Search Optimization**: Configured the Lobby rating filter input to trigger only on `Enter` and `blur` (with `enterkeyhint="go"` for mobile virtual keyboards), preventing premature re-renders while typing.
- **Dedicated FAQ**: Added a dedicated FAQ entry and Quick-Nav link explaining the feature for mobile and desktop users.

### 13. Tools "View Full List" Bidirectional Growth & Amber Highlight
- **Smooth Virtual Growth**: Implemented smooth, zero-jump bidirectional virtual list expansion and permanent amber highlight tracking for jumped words.

### 14. Settings & Synesthesia Per-User Data Isolation
- **Per-User Namespacing**: Replaced global, unpartitioned `localStorage['morpheme_settings']` key with user-namespaced storage (`morpheme_settings_<username>`), ensuring no settings or colors ever bleed across accounts on shared browsers or devices.
- **Clean In-Memory Initialization**: Provided a canonical `getDefaultSettings()` definition. When a user logs in, settings cleanly initialize from user-specific defaults before applying server settings, preventing unconfigured settings from inheriting a previous user's values.
- **Synesthesia Reset & DOM Cleanup**: `applySettings` explicitly strips all 26 CSS custom properties (`--letter-*-color`) from `document.documentElement` before applying the authenticated user's specific colors. If a user has no custom colors, all pickers reset to `#111111` and no custom variables remain active.
- **Logout Memory & DOM Purge**: Integrated `window.resetSettingsToDefault()` directly into `handleLogout()`, immediately purging letter color styles, UI pickers, and in-memory state so nothing leaks into the login screen or subsequent user sessions.

### 15. Tile Selectable Space (Corner Cutoff) 39% Default Consistency
- **HTML Initial State**: Updated `#setting-corner-cutoff-val` (39%), `#setting-corner-cutoff` (value 39), and `#preview-hitbox-shape` (clip-path polygon using 39%/61%) in `templates/index.html` so the UI renders 39% immediately without waiting on JS.
- **Client-Side Fallback (`applySettings`)**: In `static/js/settings.js`, if `settings.corner_cutoff` is missing or undefined from the server, `applySettings` unconditionally applies `39%` to the CSS variable `--corner-cutoff`, the slider, the value label, and the preview tile hitbox clip-path.
- **Guest Database Initialization**: Updated `guest_login()` in `app.py` to insert `('corner_cutoff', '39')` and default `board_sizes` into the `user_settings` table upon creation, guaranteeing guest API responses deliver `corner_cutoff: '39'`.

### 16. Fixed Bottom Back Navigation Buttons (Tools, Mods, Settings, Forum)
- **Architectural Edge Docking (Nothing Underneath / Above)**: Restructured Tools, Mods, Settings, and Forum views so that fixed bottom navigation bars (`.mobile-bottom-nav`, `.forum-bottom-bar`) are direct docked flex children (`flex: 0 0 auto; margin: 0; padding: 10px 14px; border-top: 1px solid var(--input-border); z-index: 100`) anchored at the absolute bottom edge of their view containers with zero margin and zero space beneath them.
- **Tools, Mods, and Settings Mobile Back Navigation**: Wrapped `.tools-content` and `.mobile-bottom-nav` inside `.tools-content-column` (`flex: 0 0 100%` on mobile, scroll-snap child 2 of `.tools-split-layout`). Content scrolls independently in `.tools-content` above the bar. Tapping `← Back` (`#tools-mobile-back-btn`, `#mods-mobile-back-btn`, `#settings-mobile-back-btn`) smoothly scrolls the carousel to `left: 0` back to the menu tabs sidebar. On desktop/laptop (`@media (min-width: 901px)`), `.mobile-bottom-nav` is hidden (`display: none !important`) because the side menus remain permanently visible.
- **Forum Edge Docking (Desktop & Laptop & Mobile)**:
  - Inside `.forum-main`, made `.forum-view` a full-height non-scrolling flex container (`height: 100%; display: flex; flex-direction: column; overflow: hidden; padding: 0 !important; margin: 0 !important;`).
  - Wrapped thread/post content and comments inside `<div class="forum-view-scroll-body">` (`flex: 1 1 0; overflow-y: auto;`), ensuring content scrolls cleanly above the docked bottom bar.
  - In `#forum-view-post`: docked `.forum-bottom-bar` containing `← Back to category` (`#forum-back-to-list`) at the absolute bottom of the container across all screen sizes so it never sits high or leaves empty space below.
  - In `#forum-view-create`: docked `.forum-bottom-bar` containing `← Cancel` (`#forum-cancel-create`) at the absolute bottom.
  - In `#forum-view-list`: docked `.forum-bottom-bar.mobile-only-bottom-bar` containing `← Back` (`#forum-category-back-btn`) at the bottom on mobile to smoothly slide back to categories sidebar; hidden on desktop.
- **Page Container Hierarchy Integrity**: Fixed DOM hierarchy closing tag in `#page-tools` so that all top-level views (`#page-mods`, `#page-settings`, `#page-contact`, `#page-donate`) remain top-level sibling `.page` containers at depth 3 and are never mistakenly hidden when `#page-tools` is set to `display: none`.
- **Mobile Fixed Bottom Back Button Invariant (Tools, Mods, Settings)**:
  - Locked `#page-tools`, `#page-mods`, and `#page-settings` to `overflow: hidden; height: calc(100vh - 120px)` (`100dvh` supported) on mobile (`@media (max-width: 900px)` and `body.is-mobile`), preventing the outer page container from scrolling vertically.
  - `.tools-split-layout` fills available height (`flex: 1 1 0px; min-height: 0; overflow-y: hidden`).
  - `.tools-content-column` spans 100% width on slide 2 (`height: 100%; min-height: 0; overflow: hidden; display: flex; flex-direction: column`).
  - `.tools-content` acts as the scroll body (`flex: 1 1 0px; min-height: 0; overflow-y: auto; -webkit-overflow-scrolling: touch`), ensuring all content (short or long) scrolls independently above the bottom bar.
  - `.mobile-bottom-nav` sits docked at the absolute bottom edge (`flex: 0 0 auto; width: 100%`) in a permanently fixed position as the user scrolls, identical to Forum.
- **Mobile Bottom Back Button Padding Reduction**: Reduced the bottom spacing/padding below the fixed 'Back to Category' button (`.forum-bottom-bar`) and the mobile bottom Back buttons (`.mobile-bottom-nav`) from `10px + env(safe-area)` to `7px 14px 4px 14px`, providing a tight, docked aesthetic with only 4px below the button on mobile viewports.
- **Word of the Day Mobile Single-Line Dynamic Sizing**: Applied the binary-search font-fitting algorithm (`applyDynamicValidationStyle`) to `#wotd-display` inside `.wotd-container`, matching the single-line behavior of Is Valid. Long words are dynamically calibrated and scaled to fit the mobile screen width on a single line (`white-space: nowrap !important; word-break: keep-all !important; overflow-wrap: normal !important;`) without wrapping across multiple lines, while retaining the showy gold gradient and animations.
- **Omitted Archaic 2nd/3rd Person Verb Inflections from AW (`added_words.txt`, `wikdefs.txt`, `Definitions.txt`, `morpheme.db`)**: Removed **4,961** archaic verb inflections (2,489 third-person simple present indicative ending in *-eth*, 552 second-person simple past indicative ending in *-st*, and 1,920 second-person simple present indicative ending in *-st*). Total words in `added_words.txt` updated from 474,064 to **469,103**. Synchronized definitions and database, and updated the FAQ entry.
- **Enrichment of Spaced Target Definitions in AW (`wikdefs.txt`, `Definitions.txt`, `morpheme.db`)**: Enriched **1,001** word definitions (both singular alternative forms/spellings pointing to spaced terms and their corresponding noun plurals) with authentic definitions of the spaced target terms consulted directly from Wiktionary, repeating the root definition for plurals (e.g., `SOWTHISTLE` and `SOWTHISTLES`).
