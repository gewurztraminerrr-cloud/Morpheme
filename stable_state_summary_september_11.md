# Morpheme Stable State Summary - September 11, 2026

This document records the definitive "Start Over" stable point for Morpheme as of **September 11, 2026**. Full synchronization has been executed and verified across **Localhost**, **GitHub (`origin/main`)**, and the production deployment on **`morpheme.games`**.

---

## 🎯 Executive Summary of Milestones & Features (September 11, 2026)

### 1. View Full List Readability & Contrast on White/Light Layouts
* **The Mandate**: Fix the illegibility of the "View Full List" modal window on the White layout (where black text rendered on a black/dark background), and make the "View Full List" button clearly visible and readable against light backgrounds.
* **Implementation**:
  * **View Full List Trigger Button (`#list-view-full-btn`)**: Styled with solid vibrant royal purple (`#6d28d9`), bold white text (`#ffffff`), border (`#5b21b6`), subtle elevation shadow, and hover effect (`#5b21b6`) under `body.theme-white`, `[class*="theme-white"]`, and all light themes (`[class*="theme-light-"]`, `theme-yellow`, `theme-pink`, `theme-orange`, `theme-gray`, `theme-light-brown`).
  * **Inline Text Link (`.view-full-list-link`)**: Styled list truncation notice link with bold deep purple text (`#6d28d9`, font-weight 700).
  * **Modal Overlay (`#full-list-modal`)**: Configured with a soft dimmed backdrop on desktop (`rgba(15, 23, 42, 0.65)`) with backdrop blur, and pure white background on mobile.
  * **Modal Card (`#full-list-modal-card`)**: Built a clean white surface (`#ffffff`) with subtle border (`#d1d5db`) and soft elevation shadow (`0 20px 45px rgba(0, 0, 0, 0.25)`).
  * **Word Results Grid (`#full-list-modal-results`)**: Rendered in soft light slate (`#f8fafc`) with light border (`#e2e8f0`).
  * **Word Items (`.full-list-item`, `.clickable-word-link`)**: High-contrast dark slate font (`#0f172a`, font-weight 700) providing an exceptional 16.5:1 contrast ratio.
  * **Jump Target Highlight**: Bright yellow highlight (`#fef08a`, border `#d97706`) with high-contrast amber-brown text (`#78350f`).
  * **Word Definition Popover (`.tool-def-popover`)**: Light theme card with deep purple title, dark slate definition body, and purple action links.
  * **Invariants**: Mobile fullscreen exit invariant (`openFullListModal`) strictly preserved.

### 2. Elimination of Temporary Login Page Flash on Gateway Entry
* **The Mandate**: Fix brief flash of `#page-login` when returning users access `morpheme.games`.
* **Implementation**:
  * Preserved `#page-loading` (Gateway Screen) actively until credentials and session are verified, preventing any flash of the Login interface.
  * Sanitized stale `#page-login` location hashes on return.

### 3. Default to Lobby on Gateway Navigation & Profile Page Display Override Fix
* **The Mandate**: When pressing "ENTER LOBBY" from the gateway, ensure the user lands in the Lobby even if they exited previously from Profile, and ensure all top navigation buttons work properly.
* **Implementation**:
  * Set active view explicitly to `#page-lobby` upon clicking ENTER LOBBY and cleared residual hashes.
  * Resolved CSS display override conflict on `#page-profile`, allowing seamless navigation across all top menu tabs.

### 4. Tournament "SIGN UP FOR TOURNAMENT" Button Styling
* **The Mandate**: Make the "SIGN UP FOR TOURNAMENT" button consistently green across all platforms and layout colors/themes in Settings.
* **Implementation**:
  * Added global high-specificity CSS rules and JavaScript hooks guaranteeing the signature green gradient (`#2ecc71` to `#27ae60`) across all themes.

### 5. Profile Page Centering on Desktops & Laptops
* **The Mandate**: Center all content in Profile (including the title and search panel) on desktops and laptops where it previously shifted to the left.
* **Implementation**:
  * Applied `max-width: 960px; margin: 0 auto;` with centered flex layouts to `#page-profile`, `.profile-content-container`, `.profile-page-header`, and `.profile-search-bar`.

### 6. Laptop Header Pronunciation Visibility
* **The Mandate**: Ensure the "MORE-FEEM" pronunciation text stays visible next to "MORPHEME" in the top-left corner on laptop screens.
* **Implementation**:
  * Adjusted responsive breakpoints and font scaling in `style.css` so the pronunciation remains displayed down to 768px viewports.

### 7. Lobby Logo & "No Show Rooms Selected" Separation
* **The Mandate**: Increase separation and sizing between the logo and "No 'Show Rooms' selected" text on laptops and desktops.
* **Implementation**:
  * Enlarged typography and increased margin/padding separation in `lobby.css` to prevent cramping.

### 8. Active Rooms Expand / Retract Toggle & FAQ Updates
* **The Mandate**: Add an Active Rooms expand/collapse toggle with a thumb handle along the bottom of the rooms panel, hide top elements and Create Room panel when expanded, and document in FAQ.
* **Implementation**:
  * Implemented bottom-docked expand/retract button with thumb handle.
  * Dynamically collapses the Create Room panel and top lobby elements on expansion, restoring them on retraction.
  * Updated FAQ with detailed operational guidance.

### 9. Lobby Player Row Hover Glow & Scrollbar Fixes
* **The Mandate**: Fix clipping on player row hover glow and hide thin grey scrollbars on mobile lobby panels.
* **Implementation**:
  * Added horizontal list padding and glowing box-shadows.
  * Suppressed unsightly scrollbars on mobile browsers.

### 10. "New Users" Tab in Tools
* **The Mandate**: Add a "New Users" tab below Personal Timer in Tools with country flags, registration dates, weekly stats, and total user count.
* **Implementation**:
  * Created `/api/tools/new-users` endpoint filtering registered non-guest users with rolling 7-day stats.
  * Implemented `#tool-new-users` pane with responsive stats cards and scrollable table.

### 11. FAQ Dictionary Breakdown & 15-Letter Cap Clarity
* **The Mandate**: Update dictionary descriptions to state 15-letter word caps and remove outdated 16+ letter footnote.
* **Implementation**:
  * Removed footnote and clarified 15-letter word caps for NWL and CSW dictionaries in `#faq-dictionaries`.

### 12. Suggestions Category Header Description & Voting Notice
* **The Mandate**: Explain in the Suggestions category header that user agreements and disagreements count as votes for moderator decisions based on popularity.
* **Implementation**:
  * Updated category description in `app.py` and `forum.js`.

### 13. User Current Time Display Next to Timezone
* **The Mandate**: Display player local time next to their timezone on Profile and mini-profiles.
* **Implementation**:
  * Integrated live localized time strings using `Intl.DateTimeFormat`.

### 14. Profile Metadata Layout & Mathematically Equal Row Spacing
* **The Mandate**: Group Profile metadata into 3 clean semantic flex rows with equal vertical spacing, expand "About Me" height, and reduce excess top padding.
* **Implementation**:
  * Reorganized into 3 distinct flex rows with equal `gap` spacing across desktop and mobile.

### 15. Guest Session Data Auto-Purge & Auth Hardening
* **The Mandate**: Automatically purge guest user data upon logout and prevent numerical collisions.
* **Implementation**:
  * Built `purge_guest_user(username)` in `app.py` for clean database scrubbing.

### 16. Lexicographical Additions & Invariant Enforcement
* **The Mandate**: Add definitions for Added Words (`LENATE`, `LENATES`, `LENATION`, `LENATIONS`, `MALAYOPHOBIA`, `JUFFERS`) strictly adhering to root-sourcing and suffix propagation rules.
* **Implementation**:
  * Sourced definitions adhering to all dictionary rules in `AGENTS.md`.

### 17. Full-Width Category Back Buttons
* **The Mandate**: Extend the "Back to category" / "Back" button across the entire containing panel.
* **Implementation**:
  * Updated container and button styling so navigation spans full width.

### 18. Lobby & Tools Full-Screen Fit Without Scrolling on Desktop
* **The Mandate**: Fit entire Lobby content (including Chat buttons) and Tools menu entirely within the screen on desktops and laptops without slight vertical scrolling.
* **Implementation**:
  * Calibrated container heights and vertical padding in `lobby.css` and `style.css` so that all elements sit comfortably within 100vh.

### 19. "My Rating" Value Separation on Desktop & Laptop
* **The Mandate**: Place an explicit space between "My Rating" and the user's rating value on desktop and laptop lobby views.
* **Implementation**:
  * Updated `#my-rating-btn` markup and dynamic render formatting in `lobby.js`.

### 20. Lobby Title-to-Button Spacing Harmonization
* **The Mandate**: Match the desktop/laptop spacing between game titles (`ACCUMULATIVE`, `FIRST COME FIRST SERVE`, `SPLIT POINTS`) and their buttons to the tight, clean 6px margin used on mobile.
* **Implementation**:
  * Standardized `.game-title` bottom margin to `6px` across all desktop viewports.

### 21. Mobile Menu Navigation Persistence (Tools, Settings, Mods)
* **The Mandate**: Ensure that clicking Tools, Settings, or Mods in the top menu on mobile always lands the user on the main menu hub rather than automatically opening the last-visited sub-tab.
* **Implementation**:
  * Reset active sub-view state upon top navigation clicks on mobile devices in `tools.js`, `mods.js`, and `app.js`.

### 22. Added Words Plural Audit & Dictionary Invariant Enforcement
* **The Mandate**: Resolve missing plurals for singulars in Added Words (specifically `ABLEPSIAS`), audit `-IA`/`-IAS` singular-derived plurals, and verify Latin/Greek `-IUM`/`-IUMS`/`-IA` and `-ION`/`-IONS`/`-IA` nouns.
* **Implementation**:
  * Added `ABLEPSIAS` with root definition propagated from `ABLEPSIA`.
  * Audited and added singulars with `-IAS` plurals (`ACARDIAS`, `AGNOSIAS`, `AKINESIAS`, `APROSOPIAS`, etc.).
  * Audited and added legitimate `-IUMS` and `-IONS` plurals (`COLLOQUIUMS`, `CRITERIONS`, `ELECTRONIUMS`, `PALLADIUMS`, etc.).
  * Strictly adhered to dictionary rules in `AGENTS.md`.

### 23. Mini-Profile Modal Dimensions & "About Me" Expansion
* **The Mandate**: Enlarge the vertical length of mini-profiles and make the horizontal length larger on desktops and laptops so that user information is not cut off, giving significantly more room to the ABOUT ME section.
* **Implementation**:
  * **Horizontal Length (Width)**: Enlarged base desktop and laptop width to **`750px`** (`max-width: min(94vw, 750px)` desktop, `min(96vw, 750px)` laptop), giving **~335px** to each metadata column (Registered, Last Visited, Timezone, Demographics) so full details never truncate.
  * **Vertical Length (Height)**: Enlarged desktop `max-height` to **`min(96vh, 1050px)`** and laptop to **`97vh`**.
  * **ABOUT ME Section (`.mini-profile-description`)**:
    * Desktop: `min-height: 120px; max-height: 400px; padding: 16px 20px; line-height: 1.55;` (allowing 15–20+ lines of text without tiny scroll cutoffs).
    * Laptop: `min-height: 90px !important; max-height: 250px !important; padding: 12px 18px !important;` (optimizing surrounding vertical margins to grant max height to bio).
  * **Visual Calibration**: Avatar (`68px`), username (`1.45rem`), full name (`0.95rem`), stat values (`1.1rem`), metadata items (`0.92rem`), and action buttons (`11px 16px`), plus native hover `title` tooltips for full details.
  * Mobile viewports (`@media (max-width: 600px)`) strictly preserved.

### 24. Lobby Journey Notice Sticky Header Background Matching
* **The Mandate**: In the Lobby with a white layout on mobile, the background behind the "ENTER A ROOM TO CONTINUE YOUR JOURNEY" notice was black along with the message. Keep the message black, but make the space around it match the panel below it.
* **Implementation**:
  * Replaced the erroneous `var(--bg-main, #12121f)` in `.lobby-journey-sticky-header` with `var(--bg-primary, #12121f)` combined with `background-image: linear-gradient(var(--bg-panel), var(--bg-panel))`.
  * Added explicit theme matching for `theme-white`:
    ```css
    [class*="theme-white"] .lobby-journey-sticky-header,
    body.theme-white .lobby-journey-sticky-header {
        background-color: var(--bg-primary, #ffffff) !important;
        background-image: linear-gradient(var(--bg-panel, rgba(0, 0, 0, 0.05)), var(--bg-panel, rgba(0, 0, 0, 0.05))) !important;
    }
    ```
  * Preserved the black obsidian-diamond glass message banner (`.lobby-journey-message`) with pearl lettering.
  * Synchronized across both `static/css/lobby.css` and the embedded styles in `templates/index.html` (`lobby.css?v=1788990000`).

---

## 🔒 Permanent System Invariants (STRICT / PRESERVED)

1. **Mobile Fullscreen Invariant (`openFullListModal`)**:
   * Calling `openFullListModal()` MUST explicitly execute `document.exitFullscreen()` immediately. Never disabled or bypassed.
2. **Android Virtual Keyboard Black Screen Prevention**:
   * Fullscreen is exited upon navigating to non-game utility pages (Tools, Settings, Profile, Forum, How to Play, Donate) or when opening modal dialogs with text inputs.
   * Fullscreen re-engagement is blocked while utility pages or input modals are active.
3. **Gateway Screen (`#page-loading`)**:
   * Gateway screen touch/click requests fullscreen smoothly into the Lobby without layout shifts.
4. **Dictionary Suffix & Plural Sourcing**:
   * Noun plurals and verb conjugations inherit base word definitions; root definitions are fully validated.

---

## 🌐 Synchronization Verification

* **Local Working Directory**: Clean (`git status` clean).
* **GitHub Remote Repository**: `gewurztraminerrr-cloud/Morpheme` on branch `main`.
* **Production Deployment**: `morpheme.games` (IP `132.148.72.249`), PM2 process `0` (`morpheme`) online, serving HTTP 200 OK.
* **Latest Commit ID**: `586df14c193e27b71c93256e4f4b3d657a052c68` (tracked and deployed across all environments).

