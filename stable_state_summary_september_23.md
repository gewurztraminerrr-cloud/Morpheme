# Stable State Summary — September 23, 2026

> **Start Over Point**: All environments (localhost, GitHub, production `morpheme.games`, and mobile web app) are fully synchronized.  
> **Commit**: `89eb17359560f78bdf1b2d076ff7a5bc8aa224a1` (short: `89eb1735`)  
> **Branch**: `main` — `origin/main`  
> **Server**: `132.148.72.249` (`morpheme.games`) — PM2 process `morpheme` (ID: 0), online ✅  
> **Date/Time**: 2026-09-23 ~12:05 CDT  

---

## 1. Sync Status

| Environment | Status | Commit |
|---|---|---|
| **localhost** (`/Users/jeffbabiak/`) | ✅ Clean working tree | `89eb1735` |
| **GitHub (`origin/main`)** | ✅ Fully up to date | `89eb1735` |
| **morpheme.games (Production)** | ✅ Deployed & verified | `89eb1735` |
| **Mobile Web / App** | ✅ Fully aligned & verified | `89eb1735` |

---

## 2. Work Completed This Session (Sep 20–23, 2026)

### 1. Subanagrams Redesign (Manual & Random Tabs) ✅
- **Two Horizontal Tabs**: Added horizontal tab navigation (`Manual` and `Random`) at the top of the Subanagrams control panel (`#sub-tab-manual` and `#sub-tab-random`).
- **Vertical Row-by-Row Layout**:
  - **Manual Tab**: Letters text input, Dictionary dropdown (`CSW`, `NWL`, `ALL`, `AW`), Minimum Word Length dropdown (`2` to `15`), **"Find All Words"** button, and **"Practice"** button.
  - **Random Tab**: Sequence length dropdown (`3` to `15`), Dictionary dropdown, Minimum word length dropdown (`2` to selected sequence length), Word of Max Length Guaranteed / Totally Random dropdown, and **"Random"** button.
- **Dynamic Dropdown Synchronization**: Changing the Sequence Length in Random mode automatically updates the Minimum Word Length options so min length cannot exceed sequence length.
- **Removed Purple Banner**: Removed the purple "Your Word Found" and "Click word for definition" message banner from above the list of words found. The found words panel now cleanly displays only the words found, while keeping the helpful prompt above the submission input.
- **Backend `min_length` Integration**: Updated `/api/tools/subanagrams` and `tools.js` to honor custom minimum word length filtering across both manual and random generation.

### 2. Light & White Theme Contrast Optimization ✅
- Restyled all Subanagrams tabs, buttons, labels, and dropdowns for light and white themes (`theme-white`, `theme-light-*`, `theme-yellow`, `theme-pink`, `theme-orange`, `theme-gray`, `theme-light-brown`).
- Inactive tabs show clear borders and dark charcoal text (`#334155`), active tabs display distinct blue backgrounds (`#0284c7`) with white text, and inputs/buttons are easily readable across all themes.

### 3. Dynamic Section-Tab Header Titles ✅
- **Dynamic Hyphenated Format**: Selecting any tab within **Tools**, **Settings**, or **Mods** updates the top header title from its base name to `"[Section] - [Tab Name]"` (e.g., `Tools - New Users`, `Settings - Appearance`, `Mods - Ban / Timeout User`).
- **Clean Menu Reversion**: Navigating back to the menu (via the mobile back button, leaving the section, or clicking "Tools", "Settings", or "Mods" in the top navigation bar) reverts the header title back to `"Tools"`, `"Settings"`, or `"Mods"`.

### 4. Scrollable Description Box in Content Panels ✅
- Styled the descriptions in all content panels across Tools, Settings, and Mods with the same compact, touch-scrollable box design introduced for MP in Combo Checker on mobile.
- Features a subtle border (`1px solid rgba(var(--text-primary-rgb), 0.16)`), comfortable padding (`6px 12px`), a max height of `52px` (desktop) / `46px` (mobile), and smooth touch scrolling (`-webkit-overflow-scrolling: touch; touch-action: pan-y;`).
- Full support across dark, white, and pastel themes with dedicated borders, backgrounds, and scrollbar styling.

### 5. Removed Duplicate Titles from Content Panels ✅
- Hidden the redundant `<h2>` section titles inside the content panels across Tools, Settings, and Mods (`display: none !important;`).
- The active tab is now exclusively identified by the hyphenated page header title at the top, allowing the description box to sit directly at the very top of each content panel without visual clutter.

### 6. Mobile Spacing & Title Sizing ✅
- Reduced padding/margin underneath the section titles on mobile screens (`margin: 8px auto 2px auto !important`).
- Scaled mobile section header titles down to `1.25rem` to comfortably fit hyphenated tab names on all phone viewports.
- Maintained exact original desktop title font sizing (`2.2rem` for `.page-title-header`).

### 7. Lobby Active Rooms Vertical Panel Spacing ✅
- Added extra vertical spacing below the three panels between `ACTIVE ROOMS` and the room list (`No "Show Rooms" selected`):
  1. Below **"Select a game type"** (`#selected-game-info`)
  2. Below **"+ Create room"** panel (`.create-room-panel`)
  3. Below **"Open Rooms"** panel (`.rating-filter-container`)
- **Responsive Sizing**:
  - **Desktops and Laptops**: Configured with `8px` bottom margins (`15px` total vertical spacing between panels) for generous visual breathing room.
  - **Mobile Devices**: Configured with `3px` bottom margins (`11px` total vertical spacing) to add a few pixels extra space while preserving vertical list height.

---

## 3. Current Cache Busters

| Asset | Query String |
|---|---|
| `static/css/style.css` | `?v=1790002000` |
| `static/css/play.css` | `?v=1790003000` |
| `static/css/lobby.css` | `?v=1790004000` |
| `static/js/tools.js` | `?v=1790002000` |
| `static/js/settings.js` | `?v=1790002000` |
| `static/js/mods.js` | `?v=1790002000` |
| `static/js/app.js` | `?v=1790002000` |
| `static/js/lobby.js` | `?v=1789821200` |
| `static/js/forum.js` | `?v=1789821100` |

---

## 4. Key Architectural Invariants & Reference Notes

1. **Root Directory**:
   - The application root and Git repository root is `/Users/jeffbabiak/` (NOT `/Users/jeffbabiak/boggle-gen/`).
2. **Mobile Fullscreen & Keyboard Rules (STRICT)**:
   - Full List modal (`openFullListModal`) MUST explicitly and immediately call `document.exitFullscreen()`.
   - Never allow automatic fullscreen re-engagement on utility pages (Tools, Settings, Mods, Profile, Forum) or active modals to prevent Android black screen rebuilds.
3. **Deployment**:
   - Production server: `132.148.72.249`, PM2 process `0` (`morpheme`).
   - Deployment script: `python3 /Users/jeffbabiak/scratch/deploy_remote.py`.

---

## 5. Previous Checkpoints
- **Sep 20 checkpoint**: Commit `15551a7d` — [stable_state_sep20_2026.md](file:///Users/jeffbabiak/.gemini/antigravity/brain/796c1591-5e88-454c-9b8e-455f08646822/stable_state_sep20_2026.md)
- **Sep 19 checkpoint**: Commit `1ced14e1` — [stable_state_sep19_2026.md](file:///Users/jeffbabiak/.gemini/antigravity/brain/796c1591-5e88-454c-9b8e-455f08646822/stable_state_sep19_2026.md)
