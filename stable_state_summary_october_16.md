# Stable State Summary — October 16, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_16
```
*(Also tagged as `stable-2026-10-16`)*

## Synchronization Status

| Environment | Status | Commit / Notes |
|:---|:---|:---|
| **localhost** | ✅ Synchronized | Working tree clean, synced with origin/main |
| **GitHub (main)** | ✅ Synchronized | Pushed to `origin/main` & tags updated on GitHub |
| **morpheme.games** | ✅ Synchronized | Deployed to `132.148.72.249` (PM2 online, verified live HTTP 200 OK) |
| **App / Client** | ✅ Synchronized | Verified on desktop, laptop, tablet, and mobile layouts across all 39 themes |

---

## Session Highlights & Architectural Updates (October 15–16, 2026)

### 1. Combo Checker Architecture Modernization (Manual & MP Explorer)
- **Top-Level Main Tabs (Manual vs MP Explorer)**:
  - Segregated Combo Checker into two prominent top-level main tabs: **Manual** and **MP Explorer**, with custom SVG vector icons, glassmorphic container styling (`backdrop-filter: blur(16px)`), and high-contrast theme-adaptive active gradients.
  - Defaults to **Manual** upon entry and when clicking "CC" in definition popups.
  - Mobile touch deck (`@media (max-width: 900px)`) with segmented pill deck and 42px touch targets across all 39 color layouts.
- **Desktop/Laptop Viewport Screen Fit**:
  - Content in both Manual and MP Explorer fits cleanly onto the screen with zero outer vertical scrolling (`#combo-scroll-container` and `.tools-content` have `overflow-y: hidden`).
  - Vertical scrolling is strictly confined to the inner data tables and lists (`#mp-container`, `#lic-container`, `#combo-explorer-input-list`, `#combo-explorer-response-list`).
- **Single-Row MP Explorer Dropdowns**:
  - All 5 dropdown menus in MP Word Explorer (`Dictionary`, `Length`, `MP Value`, `Direction`, `Order`) sit along a single horizontal row on desktop and laptop viewports via flexbox `display: contents` row flattening.
- **Table Scrolling on Desktop & Laptop Fixed**:
  - Repaired disconnected flex chains across intermediate containers under `@media (min-width: 901px)` (`.combo-main-tab-panes`, `.combo-results-container`, `.combo-unified-panel`, `.combo-tabs-section`, `.combo-tab-panes`, `.result-section`, `.list-scroll-area-wrapper`, `.group-table-container`).
  - Enabled active scrollbars for `1MP`, `2MP`, `3MP` while cleanly hiding empty scrollbars for `0MP`.
  - Added dedicated gold/blue custom scrollbar tracks with mousewheel forwarding.
- **Vertical Length Extension of Manual Main Panel**:
  - Removed legacy desktop `padding-bottom: 60px !important` from `.combo-results-container`.
  - Stretched `.combo-unified-panel` to 100% full vertical height (`height: 100% !important; max-height: 100% !important; margin-bottom: 0 !important; box-sizing: border-box !important;`).
  - Directly matches the bottom boundary (`bottom: 717px` / `height: 473px`) of the MP Explorer panel, providing an additional 60px of vertical viewing height for the group tables.

### 2. MP Word Explorer & Combo Checker Connectivity
- **Directional & Sorting Controls**: Added Forwards, Backwards, and Both direction filter dropdowns, alongside Random and A-Z sorting buttons.
- **Title Word Definition Popups**: Clickable definition popover support for the selected word in the title banner and 0-matches message (e.g. `0MP from SCARIDS`).
- **Synchronized Combo Checker Dictionary from Definition Popups**:
  - When opening a definition popup from any word in MP Word Explorer, clicking "CC" seamlessly sets Combo Checker's dictionary (`#combo-dict`) to match whatever dictionary is currently selected in MP Word Explorer (`#combo-explorer-dict`).
  - Supports `AW` $\rightarrow$ `added_words`, `All` $\rightarrow$ `ALL`, `CSW`, `NWL`, `csw_only`, `uniqueNWL`, `new_nwl`, `new_csw`, `new_added`, and `all_new`.
  - Immediate execution of `runComboSearch()` with matching dictionary.
  - Multi-hop stack navigation (`window._defPopupNavStack`) preserves and restores previous Combo Checker dictionary and search state when navigating back.

### 3. CSS Brace Integrity & Layout Protection
- Resolved brace imbalances in `static/css/play.css`, bringing curly braces to 100% balance: **1,961 open braces / 1,961 close braces** (Difference: 0).
- Fully restored and protected layout structures for Settings, Tools, Mods, and game room sidecars across all 39 Morpheme themes.

### 4. Lobby Guide ("WHAT AM I LOOKING AT?") Persistent Dismissal
- **Multi-Layer State Synchronization**:
  - When a user dismisses the guide (*"Remove the link leading to this explanation in Lobby and do not show it again"*), the state (`lobby_guide_dismissed: 'true'`) is saved across all storage tiers:
    1. Browser `localStorage` (`morpheme_lobby_guide_dismissed`).
    2. Per-user settings cache (`window.userSettings.lobby_guide_dismissed` and `saveSettingsToStorage()`).
    3. UI toggle (`setting-lobby-guide-toggle.checked = false`).
    4. Flask server session (`session['lobby_guide_dismissed']`).
    5. Persistent database table `user_settings` in SQLite via `/api/settings/update` and `/api/save_setting`.
- **Startup & App Reload Resilience**:
  - `settings.js:getDefaultSettings()` and `loadCachedSettings()` respect cached state.
  - Removed legacy first-run reset check in `templates/index.html`. Preserves the clean "ENTER A ROOM TO CONTINUE YOUR JOURNEY" banner without regression upon app restart.

### 5. Forum Popups & Readability Across All 39 Color Layouts
- **Universal High-Contrast Theming**: Full theme-adaptive styling overhaul for all modal dialogs, popups, response windows, and composition forms in the Forum across all 39 Morpheme color layouts.
- **Floating Response Popup Window (`.forum-reply-popup`)**: High-contrast card, clean headers, legible placeholders, active focus rings, and styled window controls.
- **Self-Reaction Prevention**: Prohibited users from liking or disliking their own threads, posts, or comments, with user-friendly toast feedback.
- **View Entire Post**: Reworded post truncation action from "View Full Post" / "View Whole Post" to "View Entire Post" with smooth expansion.

### 6. AI Content Moderation & Anti-Spam Zero-Loophole Engine
- **Multi-Touchpoint Coverage**: Automatically screens and moderates user inputs across registration, profiles, lobby/game chats, and forums.
- **Zero-Latency Local Heuristic Guarantee**: Built-in regular expression and heuristic engine in `app.py:check_local_moderation` guaranteeing instant penalty enforcement.
- **Competitive Scrabble/Boggle Dictionary Immunity**: Allows valid dictionary words during legitimate gameplay context while denying immunity if targeted insult words or pronouns are detected.
- **Exponential Penalty Ladder with 24-Hour Forgiveness Decay**:
  - Automatic doubling formula: $\text{duration} = 10 \times 2^{(\text{offenses} - 1)}$ minutes.
  - For every 24 hours of clean play, recorded offense level decays by 1.
  - Dual-layer account and IP tracking via `ip_timeouts` table.

### 7. Multimodal Google Gemini Vision Pipeline & Lockdown Interceptor
- AI computer vision on binary image bytes using Google Gemini Vision on photo upload routes.
- Automatic visual lockdown and eviction for penalized accounts with capture-phase button interceptors and live countdown timers.

### 8. Morpheme Store Modernization & Dynamic Multi-Currency Engine
- Flag-based automated multi-currency converter across 50+ world currencies.
- Centralized structured catalog with tournament-grade laptops, rigs, displays, and official dictionaries.
- **Mobile Store Typography & Layout Overhaul**:
  - Centered labels for `"Prices and products last updated on"`, `"Next Store update will occur on"`, and `"Prices reflected for:"`.
  - Prominent, enlarged dates (`0.96rem` in accent pink `#ff007f` and `0.94rem` in crisp `#fff`) and location badge (`0.94rem` with flag and currency code).
  - Symmetrical dual-divider grouping creating a unified three-part dashboard presentation on mobile.
  - Compact vertical spacing directly between `"Prices reflected for:"` and the country flag/currency badge.

### 9. Cache Busting & Asset Versions
- In `templates/index.html`:
  - `play.css?v=1792213500`
  - `store.js?v=1792213000`
  - `tools.js?v=1792210000`
  - `style.css?v=1792208000`
  - `lobby.css?v=1791173000`
  - `app.js?v=1791173000`
  - `lobby.js?v=1792204000`
  - `play.js?v=1791172000`
  - `settings.js?v=1792204000`
  - `forum.css?v=1792196000`
  - `forum.js?v=1792200000`

---

## Server & Infrastructure
- **Remote Production Host**: Ubuntu 24.04 LTS at `132.148.72.249`.
- **Application Stack**: Python 3.9 + Waitress WSGI managed by PM2 (`morpheme`).
- **Database**: SQLite (`morpheme.db`) with WAL mode, automated migrations, IP ban cache, and timeout tables.

---

## Complete Core System Architecture & Operational Reference

### 1. Game Rooms & Board Generation
- **Board Geometries**: Standard 4x4 (Boggle), 5x5 (Big Boggle), 6x8 (Super Boggle), 3D Cube (4x4x4), and Subanagrams.
- **Validation Engine**: Dual-dictionary validation (NWL & CSW) with Added Words (AW) tries and optimized C engine (`morpheme_metric.c`).
- **24-Hour Daily Challenge**: Automated daily boards pre-generated at midnight with persistent player rankings.

### 2. Tools Architecture
- **Combo Checker (CC)**: Microprocedures (MP) v2.0, Letters In Common (LIC), and MP Word Explorer.
- **Find Count (FC)**: Discovery leaderboards and lifetime word frequency statistics.
- **Lists & Study Sets**: High-probability word lists, subanagrams, CSW Only, New NWL, and custom study sets.

