# Stable State Summary - July 4, 2026

This document summarizes the stable state of **Morpheme** as of July 4, 2026. All local changes, remote code on GitHub, and the live application running on `morpheme.games` are fully synchronized.

## Latest Commit Information
* **Commit ID**: `e0b0a2c` (full: `e0b0a2c` — see git log for full hash)
* **Branch**: `main`
* **Date**: July 4, 2026

## Server Deployment Notes
* The live server runs at `/home/morpheme/morpheme/` on the remote host.
* To deploy any future changes, SSH into the server and run:
  ```bash
  cd /home/morpheme/morpheme && git pull origin main && pm2 restart all
  ```
* PM2 process name: `morpheme` (id: 0, fork mode)

## Changes Included in this Stable State

1. **Clues Tab "Remaining" Toggle (24H Rooms)**:
   - Added a **"Remaining"** / **"Return to Clues"** toggle button inside the header of the Clues tab in 24H rooms.
   - Swaps the Clues list dynamically with a counts-by-length table showing remaining words, mimicking the standard room's Remaining tab.
   - Overrides default layout constraints to let the remaining words table stretch horizontally across the panel (`display: block` and `width: 100%`) when shown, resetting to default grid styles for standard clues.

2. **Persistent "Score Sum" Leaderboard (24H Rooms)**:
   - Added a new SQL database table `daily_score_sums` to persistently store daily score accumulations.
   - Implemented a backend hook inside `save_round_history()` to accumulate players' round scores at 12 AM (midnight reset transition) when a round ends in a 24H room.
   - Created a backend API endpoint `/api/daily-score-sums` returning the leaderboard sorted descending by total score.
   - Rendered the Score Sum leaderboard with ranking numbers, player search, "Find Me" button, and player count display.

3. **3D Interactive "ENTER LOBBY" Button**:
   - Designed a fully 3D interactive neon button for the gateway screen.
   - Configured custom CSS for depth and hover states with robust mouse and touch event listeners.

4. **Progressive Bounce Multiplier Formats (Bounce 1x, 2x, 3x)**:
   - Updated `spinner_set.py` to randomly sub-select a progressive speed format when a Bounce format is spun.
   - Configured `play.js` to parse the Bounce multiplier and adjust physics ball velocity accordingly.

5. **Split Layout Sidebar Width & Spacing Adjustments**:
   - Reduced the desktop/laptop sidebar width from `400px` to `300px`.

6. **Unified Navigation Button Highlights**:
   - Standardized the active navigation button highlight state across all platforms.

7. **Intermission Points Sum Preservation**:
   - Fixed a backend bug where the total board points count would prematurely change during intermission if a Valued Letters spin occurred.

8. **Mods Dictionary Suffix, Plural, and Conjugation Sourcing Rules**:
   - Added a dynamic recursive resolver `format_resolved_definition` in `app.py`.
   - Auto-saves resolved definitions to `dictionaries/Definitions.txt`.

9. **Moderation Definition Guidelines Card Updates**:
   - Reworded guidelines for adjective inflections, adverb sub-tags, and interjections.

10. **Intermission 10-Second Warning Vibration (Mobile)**:
    - Added synchronization between the 10-second warning beep and device hardware vibration.
    - Included a user toggle in Settings → Audio & Sounds.

11. **PayPal Support Custom Amount Retention**:
    - Intercepts clicks on the PayPal Checkout link to dynamically inject the user's custom amount into the URL.

12. **Mobile Forum Height Constraints & Viewport Lock**:
    - Constrained `#page-forums` to `height: calc(100vh - 120px)` on mobile, preventing whole-screen scroll bounce.

13. **Mobile Layout Locked Header for Store, Donate, Leaderboards, and Tournaments**:
    - Constrained page container heights and enabled internal vertical scrolling on mobile.

14. **Bounce Format Speed Explanations in FAQ**:
    - Expanded FAQ descriptions for Bounce 1x, 2x, and 3x progressive speed multipliers.

15. **Mods Definition Management Rendering Optimization**:
    - Capped rendered items in "Words Without Definitions" list to 100, with a filter hint appended.

16. **Unscramble Tool Snug Stacked Layout & Vertical Scrolling (Laptops/Desktops)**:
    - Redesigned Unscramble tool pane to be vertically stacked with scrolling content panel.

17. **Mobile Carousel Restorations for Settings, Tools, and Mods**:
    - Restored horizontal swipe/scroll snap carousel on mobile for Settings, Tools, and Mods.

18. **Unscramble Tool Mobile Layout and Scrolling Enhancements**:
    - Restructured Unscramble on mobile with scoped scrolling for Session History list.

19. **New FAQ Entries**:
    - Added entries explaining 3D cube sizes on mobile and Spinner Set parameter divergence.

20. **Mobile Leaderboards Table Width Improvements**:
    - Reduced padding on leaderboard containers to maximize horizontal space on mobile.

21. **Added Words List Copy Prevention**:
    - Implemented multi-layered CSS and JS security blocking copy/select on the Added Words list.

22. **Store Item Update (OSPD7)**:
    - Replaced OSPD6 with OSPD7 in the Store page with updated description and Amazon link.

23. **Store & Profile Navigation Restructuring**:
    - Moved User Profile to a dedicated top-level page; moved Store to a Tools sub-tab.

24. **Solid-Background App Icon for PWAs (Mobile)**:
    - Generated solid-background `morpheme.png` for home screen shortcuts with no white letterboxing.
    - Configured `manifest.json` with `"purpose": "any maskable"` and high-resolution sizes.

25. **FAQ Fullscreen Exit Guidance**:
    - Added FAQ entry explaining correct mobile techniques to exit immersive PWA fullscreen mode.

26. **Tools Auto-Scroll on Mobile**:
    - Smooth auto-scroll to bottom of `.tools-content` after word generation or validation on mobile.

27. **Unscramble Keyboard Visibility Overrides on Mobile**:
    - CSS `:focus-within` styles shrink Unscramble header on keyboard open; JS auto-scrolls input into view.

28. **Desktop Mini-Profile Enhancements & Mobile Fixes**:
    - Increased `.mini-profile-card` width to `560px` on desktop.
    - Increased description `max-height` to `180px` on desktop; set `height: 140px` fixed on desktop.
    - On mobile (max-width `600px`): capped description to `90px` height; wrapped action buttons in 2-column grid to prevent "Friends" button overflow.

29. **Round Replay Mobile Layout Restructure**:
    - Restructured the `.section-header` inside Round Replay so that on all screen sizes:
      - **"Timeline of Discovery"** heading sits alone on its own row with a `border-bottom` divider.
      - **"▶ Watch Replay"** button sits on the left and **"Words recorded with millisecond precision"** sits on the right, both inside a padded `.replay-subheader` pill row below the heading.
    - Added a new `.replay-subheader` div in `index.html` and defined styles in `style.css`.
    - Added generous mobile overrides in `play.css` (`@media max-width: 900px`) with `padding: 16px`, outer card border, `gap: 14px`, and divider line.
    - Applied inline `style=""` attributes directly on the HTML elements to guarantee spacing is visible regardless of browser CSS caching.

## CSS Cache-Buster Versions (Current)
* `style.css` → `v=57`
* `play.css` → `v=136`
* `lobby.css` → `v=30`
* `howtoplay.css` → `v=10`
* `forum.css` → `v=23`
* `donate.css` → `v=1`

## Verification
* **Local**: All changes verified in local source files.
* **GitHub**: All commits pushed to `origin/main`.
* **Production**: Server pulled latest via `git pull origin main && pm2 restart all` from `/home/morpheme/morpheme/`. Live on **morpheme.games**.
