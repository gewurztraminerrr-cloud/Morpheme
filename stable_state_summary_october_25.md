# Stable State Summary — October 25, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_25
```
*(Also tagged as `stable-2026-10-25`)*

## Latest Commit ID
```
ce85bca8b781b4e60853d38e549b195e26d89162
```

## Synchronization Status

| Environment | Status | Details |
|:---|:---|:---|
| **Localhost** | ✅ Synchronized | Clean working tree on `main` |
| **GitHub (`origin/main`)** | ✅ Synchronized | Pushed to `origin/main` & tags `START_OVER_POINT_OCTOBER_25` / `stable-2026-10-25` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | Updated to latest commit, tags fetched, PM2 process online, live HTTP 200 OK |
| **App / Client** | ✅ Synchronized | Cache busters bumped (`play.css?v=1792903000`, `play.js?v=1792903000`, `tools.js?v=1792902000`, `app.js?v=1792902000`, `howtoplay.css?v=1792844000`) for instant live propagation across all devices |

---

## Session Highlights & Architectural Updates (October 25, 2026)

### 1. Game Board Equal Padding & Symmetric Centering Fix
- **Issue**: In desktop gameplay (verified in `notcenter.png` on 6x8 grids), the board card (`.board-panel`) had ~16px left padding but virtually 0px right padding, causing the board to appear visually skewed to the right.
- **Root Cause**:
  1. `.board-panel` had hardcoded `padding: 8px 16px 6px 16px !important;` on desktop. When large boards (such as 6x8) approached or exceeded `.board-panel`'s content box, CSS flexbox negative auto margins clamped `margin-left` to 0 (the 16px left-padding line), pushing the board against the right card border.
  2. In `static/js/play.js`, `applyPanelLayout` used asymmetric panel caps (`maxLeft = 380px`, `maxRight = 420px`), shifting the center column off-center and underestimating `requiredBoardWidth` (40px padding instead of 56px).
  3. In `static/css/play.css`, widescreen media queries assigned excessive width to side panels (`--left-panel-w: 460px; --right-panel-w: 460px;`), squeezing the center board panel.
- **Resolution**:
  1. Updated `body.is-desktop .board-panel` to `padding: 8px 0 6px 0 !important;` with `align-items: center !important;`, removing the asymmetric left anchor and ensuring `#game-board` with `margin: 0 auto !important;` is mathematically centered.
  2. Added `-webkit-fit-content !important; max-width: 100% !important; align-self: center !important; justify-self: center !important;` to `#game-board` across all desktop theme variants.
  3. Updated `applyPanelLayout` in `static/js/play.js` to calculate symmetric side panel widths (`newLeft == newRight`), reserve 56px board breathing room, and apply `grid-template-columns: ${newLeft}px 1fr ${newRight}px !important`.
  4. Streamlined widescreen side panel variables in `play.css` (380px at 1400px+, 420px at 1600px+, 460px at 1920px+) to guarantee abundant space for any board format.
  5. Bumped cache busters to `play.css?v=1792903000` and `play.js?v=1792903000`.

### 2. In-Game Rooms: Split "Add to Words to Work on" with "Search in Combo Checker"
- **Horizontal Split Layout**: In game rooms (`#page-play`), when a word is highlighted/selected in the words panel, `#work-on-game-btn-container` displays a responsive horizontal flex row containing two equal-width action buttons:
  - Button 1: "Add to Words to Work on" (`#add-to-work-on-game-btn`)
  - Button 2: "Search in Combo Checker" (`#search-in-combo-game-btn`)
- **CC Popup Logic Parity**: Clicking "Search in Combo Checker" triggers `window.searchWordInComboChecker(targetWord, targetDict, 'page-play')`, executing the identical logic attached to the "CC" button on definition popups:
  - Sets `_navigatingViaDefPopup = true` to preserve back navigation.
  - Pushes current position to `_defPopupNavStack` with `originPage: 'page-play'`.
  - Enables Tools back button (`setDefBackNavVisibility(true)`).
  - Navigates to Tools -> Combo Checker -> Manual Tab -> Microprocedures (MP).
  - Resolves room dictionary: if `+ AW` is active, sets Combo Checker dictionary to `ALL`; otherwise sets `NWL` or `CSW`.
  - Populates input with the selected word and runs search.
  - Scrolls all desktop and mobile containers cleanly to the top.
  - Clicking the Tools Back button seamlessly returns the user back to the active game room.
- **Theme Contrast**: Styled with high contrast across dark themes, light themes, and pure white theme.

### 3. Combo Checker Navigation Fix
- **Issue**: Clicking "Search in Combo Checker" transitioned to the Tools tab, but the pane appeared blank.
- **Root Cause**: `switchTab('tools')` invoked `resetToolsTab()`, which explicitly reset the active tool to the default (`tools-search`) unless `_navigatingViaDefPopup` was true. Additionally, a `MutationObserver` on `#tools-content` was triggering re-renders that cleared active search state.
- **Resolution**: Guarded `resetToolsTab` and observer handlers to ensure the active Combo Checker pane, inputs, and search results remain mounted and visible.

### 4. Added Words Dictionary Synchronization
- Synchronized removal of invalid inflection `TELEDIABETESES` from `dictionaries/added_words.txt` and `dictionaries/added_words_duplicate.txt` across all environments.
- Retained canonical base entry `TELEDIABETES` while preserving alphabetical ordering and index synchronization.

### 5. Architectural Clarification: Word Count Distribution & Truncation
- Investigated and documented the engine's word list sampling:
  - Boards naturally generate between 500 and 1,500+ valid words.
  - When raw board counts exceed the declared range ceiling (e.g. `400` in `300–400`), `proportionally_sample_words(words, max_accept)` normalizes the word distribution down to the exact ceiling (`400`).
  - Intermediate counts (like `350`) occur only when a grid naturally yields within `[min_accept, max_accept]` without triggering ceiling truncation.

---

## Verification & Validation

| Test / Check | Result | Verification Notes |
|:---|:---|:---|
| **Board Symmetric Padding** | ✅ Verified | Left margin: 23px, Right margin: 23px on 6x8 board card container; 0px asymmetric shift |
| **Combo Checker Navigation** | ✅ Verified | Search in Combo Checker mounts MP pane, sets dictionary, and runs query cleanly |
| **Dictionary Line Count** | ✅ Verified | Exactly 542,243 lines across `added_words.txt` and `added_words_duplicate.txt` |
| **Git Working Tree** | ✅ Verified | Clean on localhost, origin/main, and production |
| **Remote Production Health** | ✅ Verified | PM2 process online, HTTP 200 OK |
