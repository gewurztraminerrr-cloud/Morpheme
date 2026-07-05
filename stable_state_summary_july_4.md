# Stable State Summary - July 4, 2026

This document summarizes the stable state of **Morpheme** as of July 4, 2026. All local changes, remote code on GitHub, and the live application running on `morpheme.games` are fully synchronized.

## Latest Commit Information
* **Commit ID**: `651a1768478440026e6ef1a3556ddcf6177a6a43`
* **Branch**: `main`
* **Date**: July 4, 2026

## Changes Included in this Stable State

1. **Clues Tab "Remaining" Toggle (24H Rooms)**:
   - Added a **"Remaining"** / **"Return to Clues"** toggle button inside the header of the Clues tab in 24H rooms.
   - Swaps the Clues list dynamically with a counts-by-length table showing remaining words, mimicking the standard room's Remaining tab.
   - Overrides default layout constraints to let the remaining words table stretch horizontally across the panel (`display: block` and `width: 100%`) when shown, resetting to default grid styles for standard clues.

2. **Persistent "Score Sum" Leaderboard (24H Rooms)**:
   - Added a new SQL database table `daily_score_sums` to persistently store daily score accumulations.
   - Implemented a backend hook inside `save_round_history()` to accumulate players' round scores at 12 AM (midnight reset transition) when a round ends in a 24H room.
   - Created a backend API endpoint `/api/daily-score-sums` returning the leaderboard sorted descending by total score.
   - Rendered the Score Sum leaderboard with:
     - **Ranking Numbers**: Rows ordered by total scores (#1, #2, #3...).
     - **Player Search**: Textbox input for real-time username matching.
     - **"Find Me" Button**: Automatically scrolls to and highlights the current user's entry.
     - **Player Count**: Displays the total number of players in the leaderboard at the top.

3. **3D Interactive "ENTER LOBBY" Button**:
   - Designed a fully 3D interactive neon button for the gateway screen (`#btn-enter-lobby-gateway`).
   - Configured custom CSS for depth (`transform: translateY(-16px)` and solid bottom shadow `0 16px 0 #b30059`) and hover states (`transform: translateY(-20px)` and solid shadow `0 20px 0 #b30059`).
   - Implemented robust event listeners for mouse (`mousedown`, `mouseup`, `mouseleave`, `click`) and touch (`touchstart`, `touchend`, `touchcancel`) to ensure the button flattens instantly to the floor (`transform: translateY(0px)` with no bottom shadow) upon tap/click and remains flattened for 200ms before transition.
   - Handled swipe-to-scroll cancellations by sliding back up if the user drags off the button boundaries.

4. **Progressive Bounce Multiplier Formats (Bounce 1x, 2x, 3x)**:
   - Updated `spinner_set.py` to randomly sub-select a progressive speed format when a `Bounce` format is spun: **Bounce 1x** (33% weight), **Bounce 2x** (33% weight), and **Bounce 3x** (34% weight).
   - Configured `play.js` to parse the Bounce multiplier format name from the room state and adjust the physics ball velocity:
     - **Bounce 1x**: Low speed range (1.5 to 3.5 px/frame).
     - **Big Bounce / Medium**: Medium speed range (4.5 to 7.5 px/frame).
     - **Mega Bounce / High**: High speed range (8.5 to 13.5 px/frame).
     - Allowed Bounce multiplier formats to persist under high-density requirements in the backend.

5. **Split Layout Sidebar Width & Spacing Adjustments**:
   - Reduced the desktop/laptop sidebar width (`.tools-sidebar`) from `400px` to `300px`, recovering 100px of horizontal space.
   - The adjacent content panels (`.tools-content`) automatically expand horizontally to consume the recovered space.

6. **Unified Navigation Button Highlights**:
   - Standardized the active navigation button highlight state in the sidebars across all platforms. Removed the desktop-specific vertical blue left border indicator (`border-left: 3px solid #4facfe`) in favor of a soft rounded outline (`1px solid rgba(79, 172, 254, 0.3)`) and background tint (`rgba(79, 172, 254, 0.15)`) matching mobile platforms.

7. **Intermission Points Sum Preservation**:
   - Fixed a backend bug in `app.py` where the total board points count shown in brackets/parentheses (`(XX total pts)`) would prematurely change to the next round's points count if the upcoming round spun *Valued Letters* during intermission.
   - Intermission states now always return the completed round's total points (`room.previous_total_points`) until the next round starts.

8. **Mods Dictionary Suffix, Plural, and Conjugation Sourcing Rules**:
   - Added a dynamic recursive resolver `format_resolved_definition` in `app.py` for dictionary lookups.
   - If a word's definition reads `"(Noun) plural of [singular]"` or is a verb conjugation (`-S`, `-ED`, `-ING`), the resolver recursively fetches the root word definition and repeats it.
   - Handles alternative spelling forms (like `DIOCK` pointing to alternative form of `DIOCH`) by presenting alternative spelling tags: `{root}, {meaning}. Also {alternative} [{pos}]`.
   - Guessing/healing logic: If a plural or conjugated word is not in the dictionary, it strips standard suffixes (`-S`, `-ES`, `-ED`, `-ING`), looks up the root form online, and dynamically constructs and caches the redirect definition to heal the missing entry.
   - **Auto-Saving to Disk**: Hooked the resolver into Added Words additions, manual definition updates, and `newNWL`/`newCSW` dictionary uploads. Definitions for these new entries are automatically resolved, formatted according to the rules, and saved alphabetically to `dictionaries/Definitions.txt` on disk.

9. **Moderation Definition Guidelines Card Updates**:
   - Reworded guidelines layout inside the Mods Definition Management pane.
   - Included rules and demonstrations for:
     - **Adjectives with inflections (-ER, -EST, -LY)**: e.g., `FAT,FATTER,FATTEST,FATLY` -> `FAT, having an abundance of flesh [adj]`.
     - **Adjectives with implicit adverb sub-tags (-LY)**: e.g., `reflective` -> `capable of reflecting light, images, or sound waves [adj REFLECTIVELY]`.
     - **Interjections ([interj])**: e.g., `ahem` -> `Expr. desire to attract attention, gain time, or show disapproval [interj]`.

10. **Intermission 10-Second Warning Vibration (Mobile)**:
    - Added synchronization between the 10-second warning countdown beep/bell and device hardware vibration during intermission.
    - Triggers a 500ms vibration pulse via standard `navigator.vibrate` for mobile web browsers, alongside native vibration messages sent to the hybrid app bridge (`MorphemeAudioBridge`).
    - **User Configuration Toggle**: Included an "Intermission Vibration Alert" toggle switch in the "Audio & Sounds" section under settings, allowing users to enable or disable the vibration feature (persisted via DB and localStorage).

11. **PayPal Support Custom Amount Retention**:
    - Added an interactive click listener on the PayPal Checkout link inside the Donate tab.
    - Intercepts clicks to dynamically recalculate the URL with the custom amount inputted by the user (`https://paypal.me/jeffbabiak/{amount}`) right before redirecting, guaranteeing the amount is retained without re-typing.

12. **Mobile Forum Height Constraints & Viewport Lock**:
    - Constrained the outer `#page-forums` element to `height: calc(100vh - 120px) !important; overflow-y: hidden !important;` on mobile device screens (`max-width: 900px`).
    - Unified the responsive mobile grid transition breakpoint in `static/css/forum.css` from `820px` to `900px`.
    - This locks the outer page structure exactly in place, preventing whole-screen vertical browser scroll/bouncing while allowing the internal list panels (.forum-sidebar, .forum-main) to scroll their contents independently.

13. **Mobile Layout locked header for Store, Donate, Leaderboards, and Tournaments**:
    - Constrained the heights of `#page-store`, `#page-donate`, `#page-leaderboards`, and `#page-tournaments` to `height: calc(100vh - 120px) !important;` on mobile devices (`max-width: 900px`).
    - Enabled vertical scrolling inside the page containers themselves (`overflow-y: auto !important; -webkit-overflow-scrolling: touch !important;`).
    - This locks the body container viewport exactly in place on mobile, keeping the top branding bar (logo and navigation menu) fixed and visible at all times as players scroll the content.

14. **Bounce Format Speed Explanations in FAQ**:
    - Expanded the description of the **Bounce** format in the FAQ section of `templates/index.html`.
    - Added clear explanations for the **1x** (slow/gentle), **2x** (moderate), and **3x** (extremely fast) progressive speed multipliers to clarify how speed impacts gameplay.

15. **Mods Definition Management Rendering Optimization**:
    - Capped the number of rendered items in the **Words Without Definitions** list to the first 100 entries inside `static/js/mods.js`.
    - Appended an inline info alert at the bottom of the table instructing moderators to type in the search box to filter results when the list exceeds 100 matching items.
    - This drastically reduces DOM complexity (avoiding rendering 200,000+ elements simultaneously), eliminating virtual keyboard resize delays, thread-locking, and 2-3 second black screen freezes on mobile.

16. **Unscramble Tool Snug Stacked Layout & Vertical Scrolling (Laptops/Desktops)**:
    - Redesigned the **Unscramble** tool pane layout in `templates/index.html` to be vertically stacked (`flex-direction: column`) rather than horizontal.
    - Placed the game dashboard (jumbled word, word counts, input, buttons) at the top, and stacked the **Status & Results** panel underneath it.
    - Reduced the large jumbled word size from `5rem` to `3.2rem`, optimized button heights to `45px`, and tightened container paddings/margins.
    - Added vertical scrolling (`overflow-y: auto !important;`) to the main content panel of `#tool-unscramble`, letting containers expand naturally so that players can scroll down the page to view the full Status & Results content.

17. **Mobile Carousel Restorations for Settings, Tools, and Mods**:
    - Restored the working horizontal swipe/scroll snap carousel for Settings, Tools, and Mods on mobile devices.
    - Reverted layout issues where sidebar and content panes overlapped, and established correct scrolling behavior and 100% viewport container heights.

18. **Unscramble Tool Mobile Layout and Scrolling Enhancements**:
    - Restructured the Unscramble tool on mobile to place the "Status & Results" panel directly below the scrambled word game area.
    - Scoped vertical scrolling (`max-height: 260px` with custom thin scrollbar) specifically to the "Session History" list within Status & Results, keeping the "Active" section pinned at the top.
    - Styled the "Reveal" button with a highly readable blue gradient background to align with modern design standards.

19. **New FAQ Entries**:
    - Added entries to the game FAQ explaining why the 3D cube sizes are adjustable on mobile even though mobile displays are limited to 2D boards, and detailed explanations of why Spinner Set parameters might diverge from generated board parameters due to sanitization and timeout fallbacks.

20. **Mobile Leaderboards Table Width Improvements**:
    - Reduced padding on `.leaderboard-container` (to 0px) and `#page-leaderboards` (to 6px) on mobile screens. This spreads out the width of the tables to sit much closer to the vertical screen edges, maximizing horizontal screen space for ranking details.

21. **Added Words List Copy Prevention**:
    - Implemented a robust multi-layered security layer blocking copying or selecting words from the "Added Words" list under Tools -> Lists.
    - Utilizes CSS `user-select: none !important;` to block text highlighting, inline event attributes (`oncopy`, `oncut`, `oncontextmenu`, `ondragstart` returned as `false`) on dynamically rendered items, and event listener overrides in the parent scroll container (`main-list-results`) to block `copy`, `cut`, `contextmenu`, and `selectstart` events whenever the custom Added Words list is loaded.

22. **Store Item Update (Official Scrabble Players Dictionary Seventh Edition)**:
    - Replaced the "Sixth Edition" of the Official Scrabble Player’s Dictionary with the newly released "Seventh Edition" (OSPD7) in the Store page.
    - Updated description, alt tags, and bullet items, and configured the direct external purchase link to point to the Seventh Edition product page on Amazon.

23. **Store & Profile Navigation Restructuring**:
    - Relocated the **User Profile** interface from a sub-tab of the Tools page to a dedicated top-level page (`#page-profile`) accessed directly from the header navigation menu.
    - Repositioned the **Morpheme Store** from a top-level header navigation page to a sub-tab tool (`#tool-store`) inside the Tools sidebar, placed at the very bottom of the tools options.
    - Updated CSS selectors for layout, sizing, and mobile locks to align with the new structural placement, and updated JS routing paths and username click events in `app.js` to target `#page-profile` directly.

24. **Solid-Background App Icon for PWAs (Mobile)**:
    - Generated a solid-background square version of the PWA app icon (`morpheme.png`) on a background matching the app's dark theme color (`#0d1117`). This completely prevents mobile platforms (like iOS and Android) from adding ugly white border letterboxing to transparent home screen shortcuts.
    - Exported the original transparent logo version to `morpheme_transparent.png` and updated header icons and lobby placeholder image tags in `index.html` to point to it, preserving transparent rendering where it belongs.
    - Configured manifest icons in `manifest.json` with `"purpose": "any maskable"` to enable full-bleed adaptive icon rendering on Android devices, stopping the OS from placing the shortcut in a white badge circle.
    - Updated `manifest.json` with high-resolution sizes (`192x192`, `512x512`, `1024x1024`) and bumped client cache-buster versions in `index.html` for immediate application updates.

25. **FAQ Fullscreen Exit Guidance**:
    - Added a new entry to the Quick Navigate FAQ page clarifying how to exit fullscreen mode when playing in PWA/standalone mode on mobile.
    - Noted that native Android overlay prompts (like "drag from the top and touch the back button") do not exit the game's fullscreen state.
    - Instructed players on proper mobile platform techniques: swiping up from the bottom edge to display the device navigation/home bar and swiping up again to minimize or switch the app.

26. **Tools Auto-Scroll on Mobile**:
    - Implemented a smooth auto-scroll to the bottom of the `.tools-content` pane after a word is generated (in the Random Word Generator) or checked (in the Word Validator).
    - Ensures that the complete definition and pronunciation of the word is automatically scrolled into view on smaller mobile screens, removing the need for manual scrolling.

27. **Unscramble Keyboard Visibility Overrides on Mobile**:
    - Added responsive CSS `:focus-within` styles to `.unscramble-game-area` in `play.css`. When the text input is focused (keyboard active) on mobile screens, the header elements (jumbled text) shrink and sub-info is hidden to minimize height.
    - Added a focus event listener in `tools.js` to automatically scroll the active Unscramble container to the top of the viewport when focused, guaranteeing the text field and the primary action buttons ("Check Word" and "Reveal") remain fully visible and above the virtual keyboard.

28. **Desktop Mini-Profile Width Increase**:
    - Increased the width of the `.mini-profile-card` overlay container to `560px` (from `360px`) for desktop environments.
    - Added `max-width: 90%` to ensure the card scales down fluidly and remains responsive on mobile screens.
    - Leveraged the fluid layout configurations of internal components (such as flexbox stats alignments) to stretch details evenly.

## Verification
* **Local**: Verified dictionary test cases (all passed).
* **Production**: Successfully pulled changes, updated cache-busting versions, and verified live on **morpheme.games**.
