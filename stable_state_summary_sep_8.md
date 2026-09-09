# Stable State Summary – September 8, 2026

## Latest Commit Information
- **Code Commit ID**: `50eefb8e` (and subsequent checkpoint commit)
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

## Changes Implemented & Stable Specifications

### 1. Word Validator ("Is Valid") Mobile Text Cutoff Prevention
- **Root Cause Addressed**: `applyDynamicValidationStyle` previously binary-searched font sizes up to `hi = 96` purely based on width, allowing short words (`HELLO`, `CAT`, `HI`) to expand up to 85px–96px and consume excessive vertical room. Concurrently, `#tool-is-valid #valid-results-container` was constrained by an `overflow: hidden !important; height: 100% !important;` rule on mobile, truncating the `"IS VALID"` status text below the word.
- **Dynamic Responsive Ceilings**: Added `maxCeiling` (48px on mobile `innerWidth <= 900`, 68px on desktop) so short words remain prominent, bold, and readable without vertically crowding the container. Long words (`LARYNGOPHARYNGEAL`) calculate down to ~18px based on width constraints and remain completely unaffected.
- **Unconstrained Mobile Results Container**: Removed `#tool-is-valid #valid-results-container` from the `overflow: hidden` rule and granted it `height: auto !important; min-height: 0 !important; max-height: none !important; overflow: visible !important; padding: 16px 14px !important;`, allowing word results and dictionary definitions to expand naturally and scroll smoothly with the tools page.
- **Flex-Shrink & Descender Protection**: Added `flex-shrink: 0 !important;` to `#valid-result-display`, `.valid-word-val`, and `.valid-status-val`. Configured `.valid-status-val` with `line-height: 1.35 !important; padding-bottom: 2px !important; margin-top: 6px !important;` to guarantee bottom descenders are never clipped.
- **Cache Busters Bumped**: Updated `play.css` and `tools.js` to `v=1788921000`.

### 2. Store in Tools Scrolling Enabled & Visible Scrollbars Restored
- **Card Scrolling**: Enabled vertical scrolling on each individual store product panel (`overflow-y: auto !important; overflow-x: hidden !important; -webkit-overflow-scrolling: touch !important;`) in `static/css/play.css` and `static/css/lobby.css`.
- **Visible Scrollbars**: Restored styled scrollbars (`scrollbar-width: thin !important;` and custom WebKit scrollbars with rounded thumbs and hover highlights).
- **Button Visibility**: Expanded `.store-item` `max-height` to `380px !important;` and added `flex-shrink: 0 !important;` to `.buy-now-btn` so "VIEW ON AMAZON" action buttons are completely visible, reachable, and unclipped.

### 3. Lists in Tools Truncation Notice Guidance
- **Notice Update**: Updated the 10,000-word limit banner in `static/js/tools.js` to:
  *“Showing the first 10,000 words. Please select the parameters above for more words and/or use “View Full List””*
- **Interactive Trigger**: Made the embedded `"View Full List"` phrase a clickable modal opener (`openFullListModal()`).

### 4. Store in Tools Green Book & Dual Currency Pricing
- **Product Update**: Replaced the black book with the green *Collins Official Scrabble Words (2-15 All Words)* product (cover image, Amazon link, and updated description).
- **Dual Currency Pricing**: Converted all 13 items in Store to dual CAD and USD pricing (e.g. `$21.99 CAD / $16.99 USD`).

### 5. Mobile Game Room Navigation Tap/Press Highlight Removal
- **Touch State Cleanup**: Eliminated the default blue press/tap highlight (`-webkit-tap-highlight-color: transparent !important; outline: none !important; user-select: none !important;`) on "Board", "Words", and "Players" bottom navigation buttons on mobile devices.

### 6. Added Words Duplicate Detection & List Preservation
- **Inclusive Duplicate Search**: Enhanced Mods Added Words duplicate checks in `app.py` to search through the `Added Words` file in addition to CSW/NWL/16plus dictionaries.
- **Accurate Feedback**: If a word already exists in Added Words, the user is notified that the word is already valid in AW (`'{word}' is already a valid word in Added Words (AW)`).
- **Position Preservation**: Words already in AW remain in their original positions and are prevented from moving to the top of the list.
- **Top Positioning for FUNGATE**: Positioned `FUNGATE`, `FUNGATES`, `FUNGATED`, and `FUNGATING` at the top of the Added Words list.

### 7. Lobby Journey Banner Horizontal Expansion on Desktops
- **Desktop Sizing**: Expanded the horizontal length of the Lobby journey banner (`ENTER A ROOM TO CONTINUE YOUR JOURNEY`) across all desktop resolutions (`1024px`, `1280px`, `1366px`, `1440px`, `1920px`) with `width: 100% !important; max-width: 100% !important; white-space: nowrap !important;`, eliminating text clipping on desktop browsers.

### 8. Lobby Rating Filter Enter & Blur Behavior & Dedicated FAQ
- **Search Optimization**: Configured the Lobby rating filter input to trigger only on `Enter` and `blur` (with `enterkeyhint="go"` for mobile virtual keyboards), preventing premature re-renders and list flickering while typing.
- **Dedicated FAQ**: Added a dedicated FAQ entry and Quick-Nav link explaining the feature for mobile and desktop users.

### 9. Added Words Lexicographical Sourcing & Rejection Safeguards
- **Rejection Rules**: Blocked additions of abbreviations, misspellings, and obsolete terms (and their derived forms) with clear, informative status notices.

### 10. Tools "View Full List" Bidirectional Growth & Amber Highlight
- **Smooth Virtual Growth**: Implemented smooth, zero-jump bidirectional virtual list expansion and permanent amber highlight tracking for jumped words.
