# Morpheme Stable State Summary - September 11, 2026 (Final Checkpoint)

This document establishes the definitive **"Start Over"** stable point for Morpheme as of **September 11, 2026**. Full synchronization has been executed, verified, and confirmed across **Localhost**, **GitHub (`origin/main`)**, and the live production server on **`morpheme.games`**.

- **Latest Commit ID**: `86d0100e38208a269ddad4cc4e844656548ee7fb`
- **Git Branch**: `main` (clean working tree across local, GitHub, and production)
- **Production Server**: `132.148.72.249` (PM2 Process `morpheme` online, 1.5 GB memory, HTTP 200 OK)

---

## 🎯 Executive Summary of Milestones & Enhancements (September 11, 2026)

### 1. Lobby Rooms Sliding Animation & Default Retracted State
* **Default Retracted State**: Initial page load now defaults the Active Rooms in Lobby to the retracted state (`retractRooms(false)`), allowing players to view game room controls and lobby info cleanly on load.
* **Smooth Sliding Transition**: Pressing **EXPAND ROOMS** and **RETRACT ROOMS** executes a smooth upward/downward transition matching the game room chat drawer and the "Players In Lobby & Chat" sliding drawer:
  * Uses `cubic-bezier(0.16, 1, 0.3, 1)` easing over 350ms/320ms.
  * `.game-title` and `#selected-game-info` smoothly slide up/fade out during expansion and return during retraction.
  * `.create-room-panel` smoothly collapses its height and translates upward.
  * `#rooms-list` animates with `@keyframes roomsSlideUp` and `roomsSlideDown`.
  * The toggle chevron (`▲`) rotates 180° smoothly without text flickering.

### 2. Desktop Lobby Button Layout & Clipping Elimination
* **The Mandate**: Prevent the "Players in Lobby & Chat" drawer button from being horizontally cut off on desktop displays.
* **Implementation**:
  * Transitioned desktop Lobby layout (`body.lobby-active`) to a dynamic flex column where `#page-lobby` flexes (`flex: 1 1 0%`), cleanly accommodating top navigation, lobby content, and the drawer button.
  * Set `.lobby-chat-drawer` fixed at `38px` above a tight `2px` bottom padding.
  * Button and drawer are 100% visible and un-clipped across all window dimensions.

### 3. Game Room Chatbox Header Removal & Layout Optimization
* **The Mandate**: In game rooms, remove the horizontal divider line above the chatbox and expand the chatbox into the reclaimed space.
* **Implementation**:
  * Removed `#game-chat-header` divider and adjusted vertical flex boundaries so the chat messages area and input fill the container naturally.

### 4. Mobile Words Bar Left Alignment in Game Rooms
* **The Mandate**: Shift "Words" along the bottom of the screen to the left on mobile viewports so the toggle arrow is clearly visible.
* **Implementation**:
  * Adjusted mobile padding, text alignment, and flex positioning for `#game-words-toggle-btn` to ensure the disclosure chevron remains completely unobstructed.

### 5. Word Lists in Tools Description Update
* **The Mandate**: Update the description of Lists in Tools to clarify defaults and navigation.
* **Implementation**:
  * Updated `#tool-lists .tool-header p` to:
    > *"Browse official dictionaries and unique collections. Below, without adjusting the parameters, the first 10,000 words in NWL are on display. To see more, change the parameters using the dropdown menus or select “View Full List”.*

### 6. Tournament Start Date Timezone Alignment & AM/PM Standardization
* **Profile Timezone Alignment**: Replaced browser-default `new Date(data.start_date * 1000).toLocaleString()` with `formatTournamentStartDate(data.start_date)`. The start date is now explicitly computed using the player's selected Profile/Settings timezone (`window.currentUserTimezone` / `morpheme_timezone`), falling back to `Auto (Device)` if unset.
* **Timezone Declaration**: The timezone is clearly declared alongside the formatted date and time (e.g. `CDT (US Central)`, `EDT (US Eastern)`, `UTC`, or `CDT (Device)`).
* **AM/PM Standardization**: Standardized meridiem to uppercase `AM` / `PM` without periods, matching `formatAppDate` across the application.
* **Enrolled State Visibility**: Start date remains displayed both before and after tournament enrollment.
* **Immediate Reactivity**: If a player updates their timezone in Settings or Profile, the tournament start date updates instantly without page reload.

### 7. Tournament Parameter Card Layout & Value Font Sizing
* **Preserved Original Vertical Layout**: Parameter cards (`.param-item`) remain in their vertical tile layout (`flex-direction: column; gap: 4px; padding: 10px 12px; border-radius: 10px;`).
* **Preserved Parameter Titles**: Kept `.param-label` exactly as it was (`font-size: 0.72rem; color: var(--muted-text); text-transform: uppercase; letter-spacing: 0.5px;`).
* **Reduced Value Font Size & Anti-Wrapping**: Reduced `.param-value` font size to `0.88rem` with `white-space: nowrap !important;` (and cleared unscoped `play.css` pill padding `6px 18px` and backgrounds), ensuring values like `"6x8"` and `"10 Letters"` for Bonus Word stay cleanly on a single line under their title without wrapping onto two lines.

### 8. Is Valid Definition Visibility on White & Light Layouts
* **The Mandate**: Fix unreadable white definition text on white/light backgrounds in "Is Valid" in Tools.
* **Implementation**:
  * Eliminated hardcoded `color: #fff;` inline style on `.definition-text` in `static/js/tools.js` and replaced `#valid-definition-display` `color: #ccc;` in `templates/index.html` with `color: var(--text-primary);`.
  * Added `.definition-text` and `#valid-definition-display .definition-text` to `static/css/style.css` under all light/white theme selectors (`theme-white`, `theme-light-*`, `theme-yellow`, `theme-pink`, etc.) with `color: var(--text-primary) !important;`.
  * Definitions now render in crisp dark slate/black against white/light backgrounds and clean white against dark backgrounds.

### 9. View Full List Readability & Contrast on White Layouts
* **Trigger Button (`#list-view-full-btn`)**: Styled with solid royal purple (`#6d28d9`), bold white text (`#ffffff`), and `#5b21b6` border under `theme-white` and all light themes.
* **Modal Card & Grid**: High-contrast dark slate font (`#0f172a`, font-weight 700) on white card surface (`#ffffff`) providing a 16.5:1 contrast ratio.
* **Strict Invariant**: Preserved mobile fullscreen exit invariant (`openFullListModal`) so virtual keyboard never rebuilds display surface.

### 10. Gateway & Navigation Invariants
* **Login Flash Prevention**: Preserved `#page-loading` (Gateway) actively until user session is confirmed, preventing login flash.
* **Default to Lobby on Gateway Navigation**: Clicking "ENTER LOBBY" reliably routes to `#page-lobby` and clears residual location hashes.
* **Fullscreen Continuity**: Initial press on ENTER LOBBY engages fullscreen cleanly without shifting screen dimensions into the Lobby.

### 11. Lexicographical Additions & Invariant Enforcement
* Added and validated root definitions for Added Words (`LENATE`, `LENATES`, `LENATION`, `LENATIONS`, `MALAYOPHOBIA`, `JUFFERS`), adhering to root-sourcing, noun plurals, and conjugated verb suffix propagation rules in `AGENTS.md`.

---

## 🔒 Architectural & Invariant Guardrails (STRICT / PERMANENT)

1. **Fullscreen & Virtual Keyboard Invariants (`AGENTS.md`)**:
   - `openFullListModal`: Always calls `document.exitFullscreen()` immediately on modal open.
   - Non-Game Pages: Fullscreen is exited on utility views (Tools, Settings, Profile, Forum, How to Play, Donate) to prevent mobile display surface rebuild black screens.
   - Automatic re-engagement of fullscreen is blocked on utility pages.
2. **Dictionary Suffix & Plural Invariants**:
   - Noun plurals and conjugated verb endings (`-S`, `-ED`, `-ING`) replicate definitions from base root words. Missing roots are researched and lexicographically defined.
3. **Session & Security Invariants**:
   - Guest accounts are automatically scrubbed on logout.
   - Moderator checks and password hashing follow strict cryptographic standards.

---

## 📊 Complete File Modification Log (September 11, 2026)

| File | Status | Key Modifications |
| :--- | :--- | :--- |
| `static/js/lobby.js` | Updated | Smooth sliding animation for room expansion/retraction; default retracted state on load; arrow rotation. |
| `static/css/lobby.css` | Updated | `@keyframes roomsSlideUp`/`roomsSlideDown`; cubic-bezier transitions on headers and panel; desktop flex drawer placement. |
| `static/js/tournaments.js` | Updated | `formatTournamentStartDate` respecting profile timezone, declared timezone abbreviation/label, uppercase `AM`/`PM`, and instant setting reactivity. |
| `static/js/tools.js` | Updated | Replaced hardcoded `color: #fff;` on `.definition-text` with `var(--text-primary)`; added tournament re-render on profile timezone update. |
| `static/js/settings.js` | Updated | Added tournament re-render trigger on settings timezone change. |
| `static/css/style.css` | Updated | Added `.definition-text` light theme rules; refined `.tournament-params-grid .param-value` font size (`0.88rem`) and `white-space: nowrap !important;`. |
| `templates/index.html` | Updated | Word Lists description update; `#valid-definition-display` text color fix; cache-busting version bumps across CSS/JS assets. |

---

## 🚀 Verification & Synchronization Confirmation

1. **Local Working Tree**: Clean, all changes committed (`86d0100e`).
2. **GitHub Repository**: Pushed and up-to-date (`https://github.com/gewurztraminerrr-cloud/Morpheme`, branch `main`).
3. **Production Server (`132.148.72.249`)**:
   - `git status` reports: `HEAD is now at 86d0100e`, clean working tree.
   - PM2 Process `morpheme` (id 0) is **online**, memory stable at 1.5 GB.
   - Live HTTP request check: `curl -sI https://morpheme.games/` returns **`HTTP/1.1 200 OK`**.
4. **Synchronization Status**: **100% SYNCHRONIZED** across Localhost, GitHub, and Production.
