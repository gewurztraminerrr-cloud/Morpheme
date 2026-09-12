# Morpheme Stable State Summary - September 11, 2026 (Final Checkpoint)

This document establishes the definitive **"Start Over"** stable point for Morpheme as of **September 11, 2026**. Full synchronization has been executed, verified, and confirmed across **Localhost**, **GitHub (`origin/main`)**, and the live production server on **`morpheme.games`**.

- **Latest Commit ID**: `a6bbd7b45a7fb5c0f9a6d7fdcef98e2141c09c13` (and checkpoint commit)
- **Git Branch**: `main` (clean working tree across local, GitHub, and production)
- **Production Server**: `132.148.72.249` (PM2 Process `morpheme` online, HTTP 200 OK)

---

## 🎯 Executive Summary of Milestones & Enhancements (September 11, 2026)

### 1. Mobile Back Buttons Smooth Sliding Transition (Tools, Settings, Mods)
* **The Mandate**: Provide the mobile "Back" buttons with the exact smooth sliding effect that selecting a tab from the Tools menu has.
* **Implementation**:
  * Updated `resetToolsTab`, `resetSettingsTab`, and `resetModsTab` in `static/js/tools.js`, `static/js/settings.js`, and `static/js/mods.js`:
    * When clicking "Back" or performing a horizontal rightward touch swipe, the viewport executes `layoutEl.scrollTo({ left: 0, behavior: 'smooth' })`, smoothly gliding from the active content pane back to the sidebar menu.
    * The active tool/settings/mods pane remains fully rendered during the 320ms transition so the user clearly sees the content pane glide out of view to the right while the menu glides in from the left.
    * Active classes and styles are cleanly reset after the 320ms slide completes.
    * When arriving on the page via page navigation or route changes, `immediate = true` is passed (`scrollLeft = 0`) so the menu displays immediately without an unwanted sliding animation upon entry.
  * Added tactile slide feedback on `.bottom-back-btn:active` (`transform: translateX(-4px) scale(0.98)`).

### 2. Mobile Back Button Container Panel Background (Lighter Grey for White Layouts)
* **The Mandate**: For white layouts on mobile devices, make the background color of the panel containing the "Back" and "Back to Category" buttons along the bottom of the content in Tools, Settings, Mods, and Forum a lighter grey, not black or dark grey.
* **Implementation**:
  * Resolved mobile CSS specificity trap where `body.is-mobile #page-... .mobile-bottom-nav` at line 10404 of `play.css` had forced pitch black (`#1b2230 -> #121722`) on mobile.
  * Added high-specificity override rules across `play.css`, `forum.css`, and directly inline in `templates/index.html` (`<style id="mobile-back-nav-theme-override">`):
    * Background: `linear-gradient(180deg, #e4e7ed 0%, #d8dce4 100%) !important;`
    * Border-top: `1.5px solid rgba(0, 0, 0, 0.16) !important;`
    * Box-shadow: `0 -4px 16px rgba(0, 0, 0, 0.08), inset 0 1px 0 #ffffff !important;`
  * Applied consistently across `.mobile-bottom-nav` (Tools, Settings, Mods) and `.forum-bottom-bar` (Forum).

### 3. Forum "Attach images (up to 4)" File Picker Restoration
* **The Mandate**: Restore the photo attachment file chooser dialog popup in Forum thread creation and commenting.
* **Implementation**:
  * Restored native HTML `<label for="forum-comment-image" class="file-upload-box">` and `<label for="forum-post-image" class="file-upload-box">` in `templates/index.html`.
  * Added fallback click listeners in `static/js/forum.js` to ensure the file chooser triggers reliably across all mobile browsers and webviews.

### 4. Lobby Rooms Sliding Animation & Default Retracted State
* **Default Retracted State**: Initial page load now defaults the Active Rooms in Lobby to the retracted state (`retractRooms(false)`), allowing players to view game room controls and lobby info cleanly on load.
* **Smooth Sliding Transition**: Pressing **EXPAND ROOMS** and **RETRACT ROOMS** executes a smooth upward/downward transition matching the game room chat drawer:
  * Uses `cubic-bezier(0.16, 1, 0.3, 1)` easing over 350ms/320ms.
  * `.game-title` and `#selected-game-info` smoothly slide up/fade out during expansion and return during retraction.
  * `.create-room-panel` smoothly collapses its height and translates upward.
  * `#rooms-list` animates with `@keyframes roomsSlideUp` and `roomsSlideDown`.
  * The toggle chevron (`▲`) rotates 180° smoothly without text flickering.

### 5. Desktop Lobby Button Layout & Clipping Elimination
* **The Mandate**: Prevent the "Players in Lobby & Chat" drawer button from being horizontally cut off on desktop displays.
* **Implementation**:
  * Transitioned desktop Lobby layout (`body.lobby-active`) to a dynamic flex column where `#page-lobby` flexes (`flex: 1 1 0%`), cleanly accommodating top navigation, lobby content, and the drawer button.
  * Set `.lobby-chat-drawer` fixed at `38px` above a tight `2px` bottom padding.
  * Button and drawer are 100% visible and un-clipped across all window dimensions.

### 6. Game Room Chatbox Header Removal & Layout Optimization
* **The Mandate**: In game rooms, remove the horizontal divider line above the chatbox and expand the chatbox into the reclaimed space.
* **Implementation**:
  * Removed `#game-chat-header` divider and adjusted vertical flex boundaries so the chat messages area and input fill the container naturally.

### 7. Mobile Words Bar Left Alignment in Game Rooms
* **The Mandate**: Shift "Words" along the bottom of the screen to the left on mobile viewports so the toggle arrow is clearly visible.
* **Implementation**:
  * Adjusted mobile padding, text alignment, and flex positioning for `#game-words-toggle-btn` to ensure the disclosure chevron remains completely unobstructed.

### 8. Word Lists in Tools Description Update
* **The Mandate**: Update the description of Lists in Tools to clarify defaults and navigation.
* **Implementation**:
  * Updated `#tool-lists .tool-header p` to:
    > *"Browse official dictionaries and unique collections. Below, without adjusting the parameters, the first 10,000 words in NWL are on display. To see more, change the parameters using the dropdown menus or select “View Full List”.*

### 9. Tournament Start Date Timezone Alignment & AM/PM Standardization
* **Profile Timezone Alignment**: Replaced browser-default `new Date(data.start_date * 1000).toLocaleString()` with `formatTournamentStartDate(data.start_date)`. The start date is now explicitly computed using the player's selected Profile/Settings timezone (`window.currentUserTimezone` / `morpheme_timezone`), falling back to `Auto (Device)` if unset.
* **Timezone Declaration**: The timezone is clearly declared alongside the formatted date and time (e.g. `CDT (US Central)`, `EDT (US Eastern)`, `UTC`, or `CDT (Device)`).
* **AM/PM Standardization**: Standardized meridiem to uppercase `AM` / `PM` without periods, matching `formatAppDate` across the application.
* **Enrolled State Visibility**: Start date remains displayed both before and after tournament enrollment.
* **Immediate Reactivity**: If a player updates their timezone in Settings or Profile, the tournament start date updates instantly without page reload.

### 10. Tournament Parameter Card Layout & Value Font Sizing
* **Preserved Original Vertical Layout**: Parameter cards (`.param-item`) remain in their vertical tile layout (`flex-direction: column; gap: 4px; padding: 10px 12px; border-radius: 10px;`).
* **Preserved Parameter Titles**: Kept `.param-label` exactly as it was (`font-size: 0.72rem; color: var(--muted-text); text-transform: uppercase; letter-spacing: 0.5px;`).
* **Reduced Value Font Size & Anti-Wrapping**: Reduced `.param-value` font size to `0.88rem` with `white-space: nowrap !important;` (and cleared unscoped `play.css` pill padding `6px 18px` and backgrounds), ensuring values like `"6x8"` and `"10 Letters"` for Bonus Word stay cleanly on a single line under their title without wrapping onto two lines.

### 11. Is Valid Definition Visibility on White & Light Layouts
* **The Mandate**: Fix unreadable white definition text on white/light backgrounds in "Is Valid" in Tools.
* **Implementation**:
  * Eliminated hardcoded `color: #fff;` inline style on `.definition-text` in `static/js/tools.js` and replaced `#valid-definition-display` `color: #ccc;` in `templates/index.html` with `color: var(--text-primary);`.
  * Added `.definition-text` and `#valid-definition-display .definition-text` to `static/css/style.css` under all light/white theme selectors (`theme-white`, `theme-light-*`, `theme-yellow`, `theme-pink`, etc.) with `color: var(--text-primary) !important;`.
  * Definitions now render in crisp dark slate/black against white/light backgrounds and clean white against dark backgrounds.

### 12. Lexicographical Additions & Invariant Enforcement
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
| `static/js/tools.js` | Updated | Added smooth horizontal slide back on mobile Back button click and swipe right (`resetToolsTab(false)`); immediate reset on page entry (`resetToolsTab(true)`); definition color fix. |
| `static/js/settings.js` | Updated | Added smooth horizontal slide back on mobile Back button click (`resetSettingsTab(false)`); immediate reset on entry (`resetSettingsTab(true)`). |
| `static/js/mods.js` | Updated | Added smooth horizontal slide back on mobile Back button click and swipe right (`resetModsTab(false)`); immediate reset on entry (`resetModsTab(true)`). |
| `static/js/app.js` | Updated | Passed `immediate = true` on page navigation to ensure instant menu display when entering Tools, Settings, or Mods. |
| `static/css/play.css` | Updated | Updated mobile bottom navigation panel background to lighter grey (`#e4e7ed -> #d8dce4`); added tactile slide press effect (`translateX(-4px)`). |
| `static/css/forum.css` | Updated | Updated `.forum-bottom-bar` background to matching lighter grey on light and white themes. |
| `templates/index.html` | Updated | Added inline `<style id="mobile-back-nav-theme-override">` for lighter grey bottom panel; restored `<label for="...">` photo attachment triggers; updated cache versions to `v=1789211500`. |
| `static/js/lobby.js` | Updated | Smooth sliding animation for room expansion/retraction; default retracted state on load; arrow rotation. |
| `static/css/lobby.css` | Updated | `@keyframes roomsSlideUp`/`roomsSlideDown`; cubic-bezier transitions on headers and panel; desktop flex drawer placement. |
| `static/js/tournaments.js` | Updated | `formatTournamentStartDate` respecting profile timezone, declared timezone abbreviation/label, uppercase `AM`/`PM`, and instant setting reactivity. |

---

## 🚀 Verification & Synchronization State

| Target | Status | Verification Detail |
| :--- | :--- | :--- |
| **Localhost** | Synchronized | Clean working tree; verified all scripts and styles. |
| **GitHub (`origin/main`)** | Synchronized | Synced on `main` branch. |
| **Live Production (`morpheme.games`)** | Synchronized | PM2 service active, HTTP/1.1 200 OK, verified live asset delivery. |
