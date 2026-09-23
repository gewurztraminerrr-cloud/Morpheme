# Stable State Summary — September 23, 2026

> **Start Over Point**: All environments (localhost, GitHub, production `morpheme.games`, and mobile web app) are fully synchronized.  
> **Commit**: `1cc518018aa30fd78e15faf8014211e74c13c710` (short: `1cc51801`)  
> **Tags**: `START_OVER_POINT_SEPTEMBER_23`, `stable-2026-09-23`  
> **Branch**: `main` — `origin/main`  
> **Server**: `132.148.72.249` (`morpheme.games`) — PM2 process `morpheme` (ID: 0), online ✅  
> **Date/Time**: 2026-09-23 ~13:14 CDT  

---

## 1. Sync Status

| Environment | Status | Commit | Tags |
|---|---|---|---|
| **localhost** (`/Users/jeffbabiak/`) | ✅ Clean working tree | `1cc51801` | `START_OVER_POINT_SEPTEMBER_23`, `stable-2026-09-23` |
| **GitHub (`origin/main`)** | ✅ Fully pushed & up to date | `1cc51801` | `START_OVER_POINT_SEPTEMBER_23`, `stable-2026-09-23` |
| **morpheme.games (Production)** | ✅ Deployed & verified (HTTP 200 OK) | `1cc51801` | Synchronized |
| **Mobile Web / App** | ✅ Fully aligned & verified | `1cc51801` | Synchronized |

---

## 2. Work Completed in This Session (September 23, 2026)

### 1. Subanagrams Redesign (Manual & Random Tabs) ✅
- **Two Horizontal Tabs**: Added horizontal tab navigation (`Manual` and `Random`) at the top of the Subanagrams control panel (`#sub-tab-manual` and `#sub-tab-random`).
- **Vertical Row-by-Row Layout**:
  - **Manual Tab**: Letters text input, Dictionary dropdown (`CSW`, `NWL`, `ALL`, `AW`), Minimum Word Length dropdown (`2` to `15`), **"Find All Words"** button, and **"Practice"** button.
  - **Random Tab**: Sequence length dropdown (`3` to `15`), Dictionary dropdown, Minimum word length dropdown (`2` to selected sequence length), Word of Max Length Guaranteed / Totally Random dropdown, and **"Random"** button.
- **Dynamic Dropdown Synchronization**: Changing the Sequence Length in Random mode dynamically adjusts the Minimum Word Length options so minimum length cannot exceed sequence length.
- **Removed Purple Banner**: Removed the purple banner message from above the list of words found. The found words panel now cleanly displays only the found words.
- **Backend `min_length` Integration**: Updated `/api/tools/subanagrams` and `tools.js` to honor custom minimum word length filtering across both manual and random generation.

### 2. Light & White Theme Contrast Optimization ✅
- Restyled all Subanagrams tabs, buttons, labels, and dropdowns for light and white themes (`theme-white`, `theme-light-*`, `theme-yellow`, `theme-pink`, `theme-orange`, `theme-gray`, `theme-light-brown`).
- Inactive tabs display clear borders and dark charcoal text (`#334155`), active tabs display distinct blue backgrounds (`#0284c7`) with white text.
- Enhanced contrast for Account Settings card descriptions (`.account-card-desc`) and Cube Round Replay face labels in white themes.

### 3. Dynamic Section-Tab Header Titles ✅
- **Dynamic Hyphenated Format**: Selecting any tab within **Tools**, **Settings**, or **Mods** updates the top header title from its base name to `"[Section] - [Tab Name]"` (e.g., `Tools - New Users`, `Settings - Appearance`, `Mods - Ban / Timeout User`).
- **Clean Menu Reversion**: Navigating back to the menu (via the mobile back button, leaving the section, or clicking "Tools", "Settings", or "Mods" in the top navigation bar) cleanly reverts the header title back to `"Tools"`, `"Settings"`, or `"Mods"`.

### 4. Scrollable Description Box in Content Panels ✅
- Styled the descriptions in all content panels across Tools, Settings, and Mods with a compact, touch-scrollable box design.
- Features a subtle border (`1px solid rgba(var(--text-primary-rgb), 0.16)`), comfortable padding (`6px 12px`), a max height of `52px` (desktop) / `42px` (mobile), and smooth touch scrolling.
- Full support across dark, white, and pastel themes with dedicated borders, backgrounds, and scrollbar styling.

### 5. Removed Duplicate Titles from Content Panels ✅
- Hidden the redundant `<h2>` section titles inside the content panels across Tools, Settings, and Mods (`display: none !important;`).
- The active tab is exclusively identified by the hyphenated page header title at the top, allowing the description box to sit directly at the very top of each content panel without visual clutter.

### 6. Mobile Spacing & Title Sizing ✅
- Reduced padding/margin underneath the section titles on mobile screens (`margin: 8px auto 2px auto !important`).
- Scaled mobile section header titles down to `1.25rem` to comfortably fit hyphenated tab names on all phone viewports.
- Maintained exact original desktop title font sizing (`2.2rem` for `.page-title-header`).

### 7. Lobby Active Rooms Vertical Panel Spacing ✅
- Added vertical spacing below the panels between `ACTIVE ROOMS` and the room list (`No "Show Rooms" selected`):
  1. Below **"Select a game type"** (`#selected-game-info`)
  2. Below **"+ Create room"** panel (`.create-room-panel`)
  3. Below **"Open Rooms"** panel (`.rating-filter-container`)
- Configured with `8px` bottom margins on desktop/laptop and `3px` bottom margins on mobile devices.

### 8. Desktop Active Rooms Placeholder Elevation ✅
- Reduced desktop `#rooms-placeholder-view` padding from `60px 0 20px;` to `24px 0 20px;` (elevating the `No "Show Rooms" selected` text higher up by 36px).
- Reduced gap between placeholder text and animated Morpheme logo icon from `34px` to `24px` (elevating the logo by ~46px total).
- Mobile styling inside `@media (max-width: 900px)` preserved with its dedicated layout.

### 9. Cross-Browser Normalization (Desktop Firefox & Chrome) ✅
- Removed separate browser-divergent enlargement overrides so Firefox on desktop shares the exact same layout, header dimensions, font sizes, and button padding as Chrome.
- Preserved Firefox button normalization (`button::-moz-focus-inner { border: 0 !important; padding: 0 !important; }`), ensuring button geometry matches pixel-for-pixel across Gecko and Blink engines.

### 10. Mobile New Users Tool Layout & Scroller Thumb ✅
- **Screen Fit**: Configured `.tools-content` on mobile when New Users is active to use a non-scrolling flex column (`overflow-y: hidden !important; height: 100% !important;`), with the table wrapper set to flex (`min-height: 0; max-height: 100%; height: 100%; overflow-y: auto;`). The table fits completely within the screen above the mobile bottom nav bar.
- **Scroller Thumb**: Added visible custom scrollbar track (`#new-users-scrollbar-track`) and thumb (`#new-users-scrollbar-thumb`) with extended touch hit targets, dynamically updated via `initCustomScrollbarForElement` in `tools.js`. Added styled native scrollbar fallbacks.
- **Tightened Stats Padding**: Reduced padding above `REGISTRATIONS THIS WEEK` (compacted header description margin to `4px`) and below `TOTAL USERS` (reduced stats grid bottom margin from `25px` to `6px`), arranging stat boxes into a compact side-by-side 2-column grid.

---

## 3. Current Cache Busters

| Asset | Query String |
|---|---|
| `static/css/style.css` | `?v=1790010000` |
| `static/css/play.css` | `?v=1790006000` |
| `static/css/lobby.css` | `?v=1790005000` |
| `static/js/tools.js` | `?v=1790006000` |
| `static/js/settings.js` | `?v=1790002000` |
| `static/js/mods.js` | `?v=1790002000` |
| `static/js/app.js` | `?v=1790002000` |
| `static/js/lobby.js` | `?v=1789821200` |
| `static/js/forum.js` | `?v=1789821100` |

---

## 4. Key Architectural Invariants & Reference Notes

1. **Root Directory**:
   - The application root and Git repository root is `/Users/jeffbabiak/` (NOT `/Users/jeffbabiak/boggle-gen/`).
2. **Mobile Fullscreen & Keyboard Rules (STRICT / PERMANENT)**:
   - Full List modal (`openFullListModal`) MUST explicitly and immediately call `document.exitFullscreen()`.
   - Never allow automatic fullscreen re-engagement on utility pages (Tools, Settings, Mods, Profile, Forum, Donate) or active modals/inputs to prevent Android black screen rebuilds.
   - Gateway screen (`#page-loading`) requests fullscreen on initial touch so the notice appears on the gateway screen and transitions smoothly into the Lobby.
3. **Deployment**:
   - Production server: `132.148.72.249`, PM2 process `0` (`morpheme`).
   - Deployment script: `python3 /Users/jeffbabiak/scratch/deploy_remote.py`.

---

## 5. Previous Checkpoints
- **Sep 20 checkpoint**: Commit `15551a7d` — [stable_state_sep20_2026.md](file:///Users/jeffbabiak/.gemini/antigravity/brain/796c1591-5e88-454c-9b8e-455f08646822/stable_state_sep20_2026.md)
- **Sep 19 checkpoint**: Commit `1ced14e1` — [stable_state_sep19_2026.md](file:///Users/jeffbabiak/.gemini/antigravity/brain/796c1591-5e88-454c-9b8e-455f08646822/stable_state_sep19_2026.md)
