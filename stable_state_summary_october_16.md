# Stable State Summary — October 16, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_16
```
*(Also tagged as `stable-2026-10-16`)*

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| **localhost**      | ✅ Synchronized | Working tree clean, synced with origin/main |
| **GitHub (main)**  | ✅ Synchronized | Pushed to `origin/main` & tags updated on GitHub |
| **morpheme.games** | ✅ Synchronized | Deployed to 132.148.72.249 (PM2 online, verified live HTTP 200 OK) |
| **App / Client**   | ✅ Synchronized | Verified on desktop, laptop, tablet, and mobile layouts across all 39 themes |

---

## Session Highlights & Architectural Updates (October 15–16, 2026)

### 1. Morpheme Store Modernization & Dynamic Flag-Based Multi-Currency Engine
- **Profile Flag Currency Detection**:
  - In `static/js/store.js`, implemented automatic currency detection mapped to the user's selected profile country flag (`country_flag` via `/api/login` and `/api/auth/auto-login`).
  - Supports over 50 currencies (USD, CAD, GBP, EUR, AUD, JPY, CHF, INR, BRL, MXN, KRW, etc.) with real-time conversion rates and localized symbol formatting.
  - Dynamically updates all product prices when the user updates their flag in Settings.
- **Centralized Catalog (`static/data/store_catalog.json`)**:
  - Replaced hardcoded static HTML with a structured, versioned catalog containing 17 curated products across 5 categories.
  - Added catalog metadata: `last_catalog_update` (*“October 16, 2026”*) and `next_scheduled_update` (*“November 16, 2026”*).
  - Every individual product card tracks its own `last_changed_date` (*“Last changed on October 16, 2026”*).
- **Competitive Tournament Hardware Additions**:
  - Added dedicated tournament-grade computing hardware optimized for low latency and high-speed word path input:
    - **PC Laptops**: ASUS ROG Zephyrus G16 (240Hz OLED, ultra-low input lag).
    - **PC Desktops**: Custom Tournament Rig (Ryzen 9, liquid cooled, ultra-responsive tactile switches).
    - **Mac Laptops**: 16" Apple MacBook Pro (M4 Max, 120Hz ProMotion XDR display).
    - **Mac Desktops**: Apple Mac Studio + Studio Display 5K (600 nits, anti-reflective, unthrottled performance).
    - Expanded alongside existing iPad Pro, iPhone 16 Pro, Samsung Galaxy S25 Ultra, official CSW24 and NWL23 dictionaries, and tournament accessories.
- **Monthly Maintenance Automation (`scripts/monthly_store_update.py`)**:
  - Created standalone monthly automation script to bump global dates, replace discontinued hardware, and verify image assets.
- **Backend API Endpoint (`app.py`)**:
  - Added `/api/store/catalog` endpoint serving the catalog with `Cache-Control: no-cache`.

### 2. Mobile Store UI & Category Filter Overhaul
- **Dual-Card Status Banner**:
  - Replaced single-line, awkward text wrapping on mobile with `.store-status-dates-card`.
  - Icon-aligned vertical stack with live pulsating green beacon for *"Prices and products last updated on"* and calendar icon for *"Next store update:"*.
  - Styled with backdrop blur, subtle borders, and theme-adaptive coloring.
- **Prices Reflected Currency Card**:
  - Added dedicated `.store-currency-card` pairing *"Prices reflected for:"* with a high-contrast pill badge displaying the flag, country name, and currency code.
- **Modernized Pill Category Filters**:
  - Converted category filter buttons into touch-friendly pill chips (`height: 34px`, `border-radius: 17px`).
  - Added FontAwesome icons to each category (*All Products*, *Tournament Hardware*, *Phones & Tablets*, *PC & Mac*, *Dictionaries*, *Games & Gear*).
  - Enabled smooth horizontal momentum scrolling (`-webkit-overflow-scrolling: touch`) and subtle tap feedback (`transform: scale(0.96)`).

### 3. Mobile Game Room Chatbox Polish & Transition Stability
- **Clean Expansion & Retraction**:
  - In `static/css/play.css` and `static/js/play.js`, standardized the mobile game room chat drawer behavior to match the proven reliability of Lobby Chat.
  - Resolved race conditions where pressing "✕" would inadvertently trigger expansion rather than retraction.
  - Dynamically aligns the chat drawer with mobile virtual keyboards using the `window.visualViewport` API, preventing cutoff, layout jumping, or stuck open states when keyboards appear/disappear.

### 4. Forum Self-Reaction Prevention & UI Toast
- **Backend Authorization**:
  - In `app.py`, added strict author verification to `/api/forum/threads/<id>/posts/<post_id>/react`. Users cannot like or dislike their own threads, posts, or comments, returning an HTTP 400 error.
- **Frontend Prevention**:
  - In `static/js/forum.js`, disabled reaction clicks on a user's own posts and comments with an immediate toast notification (`"You cannot like or dislike your own post"`), preventing unnecessary network requests.

### 5. Restored Reaction Pill Border & Background Styling
- **Preserved Pill Container Contrast**:
  - In `static/css/forum.css`, resolved an issue where setting `opacity: 0.55` on own-post reaction buttons caused the translucent background (`rgba(255, 255, 255, 0.08)`) and border (`rgba(255, 255, 255, 0.18)`) to wash out into near-zero visibility against the post background.
  - Set `opacity: 1 !important` and kept full border/background contrast while setting `cursor: not-allowed !important` and suppressing hover color transitions.
  - In `static/js/forum.js`, ensured own-post buttons render in standard neutral state without an active accent class if a prior reaction was recorded.

### 6. Forum Long Post Truncation & "View Whole Post" Expansion
- **500-Character Truncation & Interactive Expansion**:
  - In `static/js/forum.js`, updated post and comment rendering to inspect body length. Posts exceeding 500 characters are truncated with an ellipsis (`"..."`) and followed by a `"View Whole Post"` button (`.forum-view-whole-btn`).
  - Added `window.expandWholeMessage(btn, postId, isComment)` to smoothly swap truncated text with full sanitized post content on click and remove the button.
- **Pre-Wrap Blank Line & Indentation Elimination**:
  - In `static/css/forum.css` and `static/js/forum.js`, fixed an issue where whitespace in template literals rendered as blank lines above post text or beneath the expansion button due to container `white-space: pre-wrap`.
  - Removed template literal newlines and applied `white-space: normal !important; margin: 0 !important; padding: 0 !important;` to `.forum-expandable-content`.

### 7. MP Word Explorer Mobile Horizontal Extension
- **Full Width Controls on Mobile Devices**:
  - In `static/css/play.css`, updated `@media (max-width: 768px)` and `body.is-mobile` rules for `.combo-explorer-controls`:
    - Changed the controls grid layout to `grid-template-columns: 1fr` and `width: 100%`.
    - Styled `.combo-explorer-control-group` with `display: flex`, `width: 100%`, and `flex: 1 1 100%`.
    - Standardized labels to `width: 90px; min-width: 90px; text-align: left;` and inputs/selects to `flex: 1 1 auto; width: 100%; font-size: 16px;`.
    - Dictionary, Length, and MP Value dropdowns stretch horizontally all the way to the right boundary of the panel.
    - Order buttons (Random / A–Z) span evenly 50/50 across the bottom row.

### 8. Desktop & Laptop "Back" Button for CC & FC Navigation in Tools
- **Desktop/Laptop Bottom Navigation Bar**:
  - In `static/css/play.css`, added responsive desktop overrides for `.tools-bottom-nav.show-def-back`, enabling the bottom "← Back" button bar within `.tools-content-column` on laptop and desktop viewports when navigated via definition popups.
  - Styled with backdrop blur (`backdrop-filter: blur(12px)`), dark gradients, subtle top borders, and theme-adaptive styling for white and light themes.
- **Multi-Hop Stack-Based Navigation History**:
  - In `static/js/tools.js`, introduced `window._defPopupNavStack` to record navigation history across sequential definition lookups (e.g., from Play or Lobby → CC → FC → CC).
  - Clicking "← Back" pops the stack and smoothly restores the exact origin page, tool, inputs, and scroll position. When the stack is empty, it hides the bottom back button on desktop.

### 9. Tools Tab Screen Freeze & Lockup Resolution
- **MutationObserver Guard**:
  - Added an `isActive && !wasActive` guard in `static/js/tools.js` so the observer only resets when `#page-tools` genuinely transitions from inactive to active, never during tool switching or class mutations within Tools.
- **Scoped Def-Back Visibility**:
  - Scoped `setDefBackNavVisibility` to strictly toggle `.show-def-back` on `.tools-bottom-nav`, completely avoiding mutating classes on `#page-tools`.
- **Restored Flexbox Layout**:
  - Removed the destructive `height: auto !important` override in `static/css/play.css`, allowing `.tools-content` to maintain its natural `flex: 1 1 0%` height and `overflow-y: auto`, providing fluid vertical scrolling on all screen sizes.

### 10. MP Word Explorer Padding & Aesthetic Framing Across All Platforms
- **Container Framing & Padding**:
  - In `static/css/play.css`, updated `#combo-explorer-section` (the standalone MP Word Explorer card beneath the MP/LIC tabs in Combo Checker) with balanced padding and visual framing on desktop, laptop, and mobile devices.
- **Content Alignment**:
  - Reset `#combo-explorer-section .combo-explorer-title` padding to `0 !important` and aligned margins, ensuring flush edge-to-edge alignment without horizontal overflow.

### 11. Lobby Guide ("WHAT AM I LOOKING AT?") Polish
- **Subanagrams Rack Length Range**:
  - Updated Subanagrams description from "7, 8 or 9 letters" to "6, 7, 8, 9, or 10 letters" in both the Lobby Guide overview card and modal FAQ.
- **3D Cube Dice Mathematics**:
  - Clarified the 3D Cube description: "3x3x3 Cube (27 letter dice, 54 visible letter faces across 6 sides; 9 on each face)".
- **Diagram Deduplication**:
  - Eliminated redundant duplicate dimension labels (`4x4`, `5x7`, `6x8`) situated above the column headers row.

### 12. Cache Busting & Asset Versions
- In `templates/index.html`:
  - `play.css?v=1791164500`
  - `store.js?v=1791164500`
  - `play.js?v=1791157800`
  - `forum.css?v=1791160800`
  - `forum.js?v=1791160800`
  - `lobby.css?v=1791150300`
  - `tools.js?v=1791150200`

---

## Complete Core System Architecture & Operational Reference

### 1. Game Rooms & Board Generation
- **Board Geometries**:
  - Standard 4x4 (Boggle), 5x5 (Big Boggle / Deluxe), 6x8 (Super Boggle), 3D Cube (4x4x4).
  - Single-player and multiplayer Subanagrams (practice / competitive).
  - 24-hour daily challenge boards with automatic midnight rollover and board pre-generation.
- **Validation Engine**:
  - Dual-dictionary validation: NWL (North American / NASPA) and CSW (Collins Scrabble Words / International).
  - Custom Added Words (AW) engine with fast prefix tries and SQLite persistence.
  - Real-time word path searching via optimized native C engine (`morpheme_metric.c` / C extension).

### 2. Tools Architecture
- **Morpheme Store**:
  - Dynamic currency conversion based on user country flag, real-time catalog loading, category filtering, responsive cards, and automated monthly updates.
- **Combo Checker (CC)**:
  - Microprocedures (MP) v2.0: Distance calculations (0MP to 3MP) between dictionary words.
  - Letters In Common (LIC): Shared letter combinations and overlapping word lists.
  - MP Word Explorer: Multi-length word discovery with filterable non-plural and non-inflection MP associations.
- **Find Count (FC)**:
  - Global word frequency statistics and discovery leaderboards tracking user finds since Morpheme began.
- **Lists & Words to Work On**:
  - Curated high-probability word lists, stems, subanagrams, and custom study sets.
  - Support for CSW Only, New NWL Words, New CSW Words, and New AW Words.
- **Sequence Tool & Unscramble**:
  - Interactive letter-sequence exploration and anagram solving.

### 3. Themes & Visual System
- Comprehensive support for 39 themes spanning dark, vivid, white, and soft light palettes.
- Universal custom scrollbars with drag-and-drop thumb tracking.
- Touch-friendly swipe gestures for mobile navigation, definition previews, and panel transitions.

### 4. Server & Infrastructure
- **Server**: Ubuntu 24.04 LTS at `132.148.72.249`.
- **Application Server**: Waitress WSGI running Python 3 with PM2 process manager (`morpheme`).
- **Database**: SQLite (`morpheme.db`) with automatic migrations, round history, used board hash logging, and WAL mode.
- **Deployment Script**: `scratch/deploy_remote.py` with pre-reset safety backup of custom words.
