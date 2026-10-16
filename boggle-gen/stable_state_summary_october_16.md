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

### 1. Desktop & Laptop "Back" Button for CC & FC Navigation in Tools
- **Desktop/Laptop Bottom Navigation Bar**:
  - In `static/css/play.css`, added responsive desktop overrides for `.tools-bottom-nav.show-def-back`, enabling the bottom "← Back" button bar within `.tools-content-column` on laptop and desktop viewports when navigated via definition popups.
  - Styled with backdrop blur (`backdrop-filter: blur(12px)`), dark gradients, subtle top borders, and theme-adaptive styling for white and light themes.
- **Multi-Hop Stack-Based Navigation History**:
  - In `static/js/tools.js`, introduced `window._defPopupNavStack` to record navigation history across sequential definition lookups (e.g., from Play or Lobby → CC → FC → CC).
  - Each stack entry preserves:
    - `originPage`: Originating page outside Tools (Play, Lobby, Forum, Profile, etc.).
    - `originTool`: Originating tool pane within Tools (Lists, Words to Work on, Sequence, Subanagrams, etc.).
    - `wasFullListModalOpen`: Boolean state preserving modal dialog visibility.
    - Saved scroll offsets (`savedListsScrollTop`, `savedWorkOnScrollTop`, `savedSequenceScrollTop`, `savedSequenceToolsContentScrollTop`).
    - Input values and subtab states (`comboWord`, `comboSubtab`, `findCountWord`).
  - Clicking "← Back" pops the stack and smoothly restores the exact origin page, tool, inputs, and scroll position. When the stack is empty, it hides the bottom back button on desktop.

### 2. Tools Tab Screen Freeze & Lockup Resolution
- **Root Cause Analysis**:
  - Two interrelated bugs caused the screen to lock up / freeze upon pressing the Tools tab:
    1. A `MutationObserver` on `#page-tools` listening to `['class']` changes unconditionally triggered `resetToolsTab(true)` whenever `toolsPage.classList.contains('active')` was true. Calling `setDefBackNavVisibility` previously mutated classes on `#page-tools`, firing the observer repeatedly and immediately stripping `.active` from any selected tool pane or button.
    2. An override rule `#page-tools.has-def-back .tools-content { height: auto !important; min-height: 0 !important; }` broke the internal flexbox scrolling container within `.tools-content-column` (`height: 100%; overflow: hidden`), expanding the content beyond the column bounds without scrollbars and preventing vertical scrolling.
- **Permanent Solution**:
  - **`MutationObserver` Guard**: Added an `isActive && !wasActive` guard in `static/js/tools.js` so the observer only resets when `#page-tools` genuinely transitions from inactive to active, never during tool switching or class mutations within Tools.
  - **Scoped Def-Back Visibility**: Scoped `setDefBackNavVisibility` to strictly toggle `.show-def-back` on `.tools-bottom-nav`, completely avoiding mutating classes on `#page-tools`.
  - **Restored Flexbox Layout**: Removed the destructive `height: auto !important` override in `static/css/play.css`, allowing `.tools-content` to maintain its natural `flex: 1 1 0%` height and `overflow-y: auto`, providing fluid vertical scrolling on all screen sizes.

### 3. MP Word Explorer Padding & Aesthetic Framing Across All Platforms
- **Container Framing & Padding**:
  - In `static/css/play.css`, updated `#combo-explorer-section` (the standalone MP Word Explorer card beneath the MP/LIC tabs in Combo Checker) with balanced padding and visual framing:
    - **Desktop & Laptop**: `padding: 16px 18px !important;`, `border-radius: 14px !important;`, `background: rgba(0, 0, 0, 0.22) !important;`, `border: 1px solid rgba(var(--text-primary-rgb), 0.12) !important;`, `margin-bottom: 24px !important;`, and `box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2) !important;`.
    - **Mobile Devices**: `padding: 10px 10px 14px 10px !important;`, `border-radius: 14px !important;`, `background: rgba(0, 0, 0, 0.25) !important;`, `border: 1px solid rgba(var(--text-primary-rgb), 0.16) !important;`, and `margin-bottom: 16px !important;`.
  - Harmoniously mirrors the proportions and visual styling of `.combo-unified-panel` above it.
- **Content Alignment**:
  - Reset `#combo-explorer-section .combo-explorer-title` padding to `0 !important` and aligned margins, ensuring the title, description box, dropdown controls (Dictionary, Length, MP Value, Order), and dual result cards (Input Words, MP Connections) align flush with one another.
  - Ensured `box-sizing: border-box !important;` and `width: 100% !important;` across the grid and control elements for clean edge-to-edge alignment without horizontal overflow.

### 4. Lobby "WHAT AM I LOOKING AT?" Button Refinements
- **Proportional Sizing on Mobile**:
  - In `static/css/lobby.css`, reduced the vertical size of the "WHAT AM I LOOKING AT?" button on mobile viewports so its height and padding are proportional to its appearance on laptops and desktops.
- **Vertical Spacing Buffer**:
  - Added dedicated vertical padding buffers above and below the button within the Lobby matrix layout.
  - Eliminated dark growth and glowing edge artifacts beneath the button on mobile viewports.

### 5. Cache Busting & Asset Updates
- In `templates/index.html`:
  - Bumped `play.css` cache-buster to `v=1791150300`.
  - Bumped `tools.js` cache-buster to `v=1791150200`.

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
