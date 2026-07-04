# Stable State Summary - July 4, 2026

This document summarizes the stable state of **Morpheme** as of July 4, 2026. All local changes, remote code on GitHub, and the live application running on `morpheme.games` are fully synchronized.

## Latest Commit Information
* **Commit ID**: `3e30add75f1b9329ec3621574b871ecb0ae73029`
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
   - Added a clear alert bar instructing moderators **NEVER** to use a trailing period at the end of definitions (immediately before square bracket tags). Removed the trailing period from the example noun definition of `arity`.
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

## Verification
* **Local**: Verified dictionary test cases (all passed).
* **Production**: Successfully pulled changes, updated cache-busting versions, and verified live on **morpheme.games**.
