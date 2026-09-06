# Stable State Summary – September 6, 2026

## Latest Commit Information
- **Commit ID**: `040150f7`
- **Commit Message**: `docs: checkpoint September 6 stable start over state`
- **Active Git Tags**:
  - `START_OVER_POINT`
  - `START_OVER_POINT_SEPTEMBER_6`
  - `stable-2026-09-06`
  - `save-point-latest`
  - `start-over`

---

## Synchronization Status
- **Localhost (`/Users/jeffbabiak`)**: Synchronized (`040150f7` / tags updated)
- **GitHub (`origin/main` & Tags)**: Synchronized (`040150f7` / tags updated)
- **Production Server (`132.148.72.249` / `morpheme.games`)**: Synchronized (`040150f7`), PM2 online, HTTP 200 OK
- **Flutter Mobile App (`morpheme_word_game`)**: Synchronized (loads `https://morpheme.games/` with native SoLoud audio bridge)

---

## Changes Implemented & Stable Specifications

1. **Firefox Desktop & Laptop Horizontal Overflow Elimination**:
   - Eliminated page-wide and container horizontal scrolling (`overflow-x`) across Firefox on all desktop and laptop resolutions (`1024x650`, `1200x750`, `1280x720`, `1366x768`, `1440x900`, `1920x1080`) for Mods, Tools, Settings, Lobby, Forums, and Profiles.
   - **Header Navigation**: Added responsive desktop compact navbar rules in `static/css/style.css` for viewports between 901px and 1380px (`.nav-btn { padding: 4px 7px; font-size: 0.73rem; }`, hidden auxiliary pronunciation subtitle, and `overflow-x: hidden`), ensuring all 12–13 buttons fit cleanly without horizontal spill.
   - **Gecko CSS Grid Track Bounds**: Updated `.lobby-grid` in `static/css/lobby.css` across all media queries to use explicit `minmax(0, ...fr)` declarations so grid columns shrink below `min-content`, and enforced `min-width: 0 !important; max-width: 100% !important;` on all lobby panels (`.game-types-panel`, `.solo-friends-panel`, `.active-rooms-panel`).
   - **Flex Min-Width in Split Layouts**: Updated `.tools-content` inside `@media (min-width: 901px)` in `static/css/play.css` to `flex: 1 1 0% !important; min-width: 0 !important; width: calc(100% - 280px) !important; max-width: calc(100% - 280px) !important; overflow-x: hidden !important; overflow-y: auto !important; height: 100% !important;`, preventing tables, forms, and cards from blowing out the layout.
   - **Store Grid Fluidity**: Configured `#tool-store .store-item-image` with `max-width: 260px; min-width: 0` and `.store-item-features` with `grid-template-columns: repeat(auto-fit, minmax(min(120px, 100%), 1fr)) !important;` so items wrap fluidly on 1024px screens.
   - **Universal Root Clipping**: Replaced `max-width: 100vw` with `max-width: 100% !important; width: 100% !important; overflow-x: clip !important; overflow-x: hidden !important;` across `html`, `body`, `#app`, `main.pages`, `.pages`, and `.page`.
   - **Selenium Automated Verification**: Ran 132 automated test combinations in headless Firefox across 6 resolutions and 22 views/tabs, achieving a 100% pass rate with 0 horizontal overflows.

2. **Tools: "View Full List" Word Jump Persistent Highlight Badge**:
   - In `static/js/tools.js`, when jumping to a searched word via **"ENTER WORD TO JUMP TO..."**, the virtualized list centers the target item, triggers an accent pulse animation, and permanently decorates the word with an amber badge (`jump-target-highlight`), ensuring players never lose visual tracking when scrubbing or inspecting adjacent words.

3. **Tools: Lists Filter Note Removal**:
   - In `templates/index.html`, removed the obsolete placeholder text `"Note: Select a word length or starting letter to filter the list."` from the Lists tab header.

4. **Removal of Three Mobile Overlay Buttons Above Keyboard**:
   - Permanently removed the black floating action bar containing the three buttons (bubble with circle, envelope, key) above the virtual keyboard across the mobile interface.

5. **Tournament Countdown Timer Resolution**:
   - In `static/js/tournaments.js`, corrected the tournament countdown timer formatting and state updates so active matches display a live, decrementing countdown instead of a static `"00:03:41"` timestamp.

6. **Added Words (AW) Definition Enrichment & Sourcing Logic**:
   - Reverted AW definitions and dictionary sourcing in `app.py` and `dictionaries/Definitions.txt` to authentic, clean, concise parenthetical specifications:
     - Plurals: `plural of [WORD] (([definition of singular]))` (e.g., `CHAMAS` -> `plural of CHAMA ((East Africa, chiefly Kenya) Any of several types of informal cooperative society.)`).
     - Diminutives & Variants: `Diminutive of [ROOT] (([definition of root]))` (e.g., `POLESTER` -> `(motor racing) Diminutive of polesitter ((motor racing) A driver placed in pole position.)`).
     - Synonyms: `Synonym of [WORD] (([definition of word]))` (e.g., `MALAXER` -> `Synonym of malaxator (A mill designed for malaxation, particularly for softening or mixing a mass.)`).
     - Verb Conjugations: `third-person singular simple present indicative of [VERB] (([definition of verb]))`.
   - Eliminated recursive nested dictionary dumps on compound terms like `MEANGIRLS`.

7. **FAQ Modernization & Comprehensive Documentation**:
   - In `templates/index.html`, added and modernized three major FAQ sections:
     - **Achievements vs. Leaderboards Explained**: Detailed explanations of all 6 Achievement tables and all 10 Leaderboard tables, explicitly identifying shared tracking metrics (Best Scores, Point Efficiency, Best Words, % Words Found, Hard Words Found, Games Played) and global-exclusive metrics (Average Score, Avg % Found, Peak Ratings, Top Rated Active ladder).
     - **"View Full List" Guide**: Clear explanation of the rapid scroll thumb scrubber, adjacent word browsing, instant word jump with amber badge highlight, and on-tap definition card lookup.
     - **Mobile Fullscreen & Gesture Mechanics**: Rewrote the outdated FAQ entry regarding mobile fullscreen behavior, clarifying PWA display modes, Android system notifications, automatic fullscreen exit on text input/modals to prevent keyboard black screens, and home-indicator app switching.

8. **Mobile Fullscreen & Keyboard Invariant Safeguards**:
   - All mobile fullscreen rules remain strictly enforced: `openFullListModal` exits fullscreen immediately, fullscreen is deactivated on non-game utility pages (Tools, Settings, Profile, Forum, How to Play, Donate) to prevent mobile display surface rebuilds, and gateway screen touches seamlessly transition into the Lobby.

9. **Asset & Cache Versioning**:
   - Incremented query parameter versions in `templates/index.html` (`tools.js?v=18989`, `tournaments.js?v=15467`, CSS bundles).
