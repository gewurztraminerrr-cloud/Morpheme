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
