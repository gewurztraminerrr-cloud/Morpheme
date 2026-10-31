# Stable State Summary — October 30, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_30
```
*(Also tagged as `stable-2026-10-30`)*

## Latest Commit ID
```
2255fa55c37bce9879acd3ed4713f2283fe30c5f
```
*(Short SHA: `2255fa55`)*

## Synchronization Status

| Environment | Status | Commit / Tag | Verification Details |
|:---|:---|:---|:---|
| **Localhost** | ✅ Synchronized | `2255fa55` | Working tree on `main`, zero uncommitted code changes |
| **GitHub (`origin/main`)** | ✅ Synchronized | `2255fa55` | Pushed to `origin/main` & tags `START_OVER_POINT_OCTOBER_30` / `stable-2026-10-30` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | `2255fa55` | Reset to latest origin/main, PM2 online, Waitress bound, live HTTP/2 200 OK |
| **App (Mobile WebView)** | ✅ Synchronized | `https://morpheme.games/` | Flutter app loads live synchronized webview with native audio/haptic bridges |

---

## Session Highlights & Architectural Updates (October 25 – October 30, 2026)

### 1. Dictionary & Lexicographical Synchronization
- **`AEROMONIASES` Addition**:
  - Definition: `plural of aeromoniasis ((pathology) infection with bacteria of the genus Aeromonas.)`
  - Synchronized across `dictionaries/added_words.txt`, `dictionaries/added_words_dates.txt` (date: `25/10/2026`), `dictionaries/added_words_duplicate.txt`, `dictionaries/Definitions.txt`, `dictionaries/wikdefs.txt`, and `dictionaries/wikdefs_duplicate.txt`.
  - Updated `dictionaries/word_stats.json` with latest statistics.
  - Synchronized SQLite database `morpheme.db` (`wiktionary_definitions` table) on both localhost and remote production environments.
  - Preserved alphabetical ordering and index synchronization.

### 2. Client Submission & Optimistic Word Validation
- **Removal of False Optimistic Red Flash (`static/js/play.js`)**:
  - Eliminated erroneous red flash feedback for valid words that temporarily were not in the client-side pre-generated word list.
  - Server-authoritative validation cleanly handles word validation without optimistic client misclassification.

### 3. Bonus Letter Format & Tile Rendering
- **Bonus Cell Guarantee at Round Promotion (`game_room.py`)**:
  - Enforced deterministic bonus cell generation on round promotion in Bonus Letter format games.
- **Intermission Bonus Tile Visibility (`static/js/play.js`)**:
  - Fixed issue where the bonus letter tile was hidden during round intermission.
  - Ensured the bonus tile renders reliably whenever `bonus_cell` matches during both active play and intermission phases.

### 4. Round Timing & Intermission Transition
- **Exact Intermission 0:00 Transition (`game_room.py`, `static/js/play.js`)**:
  - Synchronized round start timer to launch the subsequent round exactly at `0:00` intermission expiration, eliminating phase drift.

### 5. 24h & Non-Grid Room Engine Robustness
- **Midnight Transition Fix (`game_room.py`)**:
  - Resolved midnight `start_next_round` retry loop caused by `UnboundLocalError` on copy operations.
  - Guarded post-promotion failures to ensure unbroken continuity across daily resets.
- **Local Import & Dimension Parsing**:
  - Fixed `calculate_word_score` local import scoping.
  - Implemented safe dimension parsing for non-grid rooms.

### 6. Scoring Pre-Calculation & Word Value Sorting
- **In-Game Room Word Sorting (`static/js/play.js`)**:
  - Accurately pre-calculated scores and sorted found words by descending point value in game rooms.
- **Intermission Bonus Formula Equations**:
  - Guaranteed that all bonus words and bonus letter words display full bonus value equation breakdowns under "All Words" during intermission.

### 7. Definition Lookups & Proper Noun Filtering
- **Curated Dictionary Guard (`app.py`, `word_validator.py`)**:
  - Prevented false-positive proper noun filtering on valid curated dictionary entries such as `FIGURANT`, `DANSEUR`, and `ARMBAND`.

### 8. Game Room Audio & Autoplay Recovery
- **Triple Word Audio Playback (`static/js/play.js`)**:
  - Enabled `triple.mp3` sound playback in game rooms upon finding triple-value or high-tier words.
  - Added user gesture autoplay recovery so audio permissions seamlessly initialize on first interaction across desktop and mobile browsers.

### 9. Intermission Default View & Player Notepads
- **Intermission "VIEW ALL" Default (`static/js/play.js`)**:
  - At the start of intermission in all game rooms, the middle player control button below "Players" automatically defaults to "VIEW ALL".
- **Dictionary Source Tags in Notepads (`templates/index.html`, `static/css/play.css`, `static/js/play.js`)**:
  - In player notepads during intermission, dictionary tags (`NWL`, `CSW`, or stacked vertically if both) are displayed in small, crisp typography directly next to the point value of each word.

### 10. Achievements UI Refinement
- **Exceptional Performances Cleanup (`templates/index.html`)**:
  - Removed the `AVERAGE PERF` metric from the header banner of "EXCEPTIONAL PERFORMANCES" in every user's Achievements window.

### 11. Header Parameters Display
- **Time Unit Formatting & Bold Parameters (`templates/index.html`, `static/css/play.css`)**:
  - Replaced seconds with standard units in room headers: `86400s` -> `24h`, `600s` -> `10m`, and `180s` -> `3m`.
  - Styled parameter values below room type (e.g., `Accumulative`, `4x4`, `45s`) with bold font weight while preserving thin, un-bolded pipe dividers.

### 12. Remaining Tab Formula & Responsive Legend
- **Row Arithmetic Display (`static/js/play.js`, `static/css/play.css`)**:
  - Formatted each row in Remaining (all public and 24h rooms) as:
    `7LW                    51 + 49 = 100  (51%)`
    with left-aligned `.len-cell`, right-aligned `.count-cell`, and two non-breaking spaces before `(51%)`.
- **Responsive Vector Legend (`templates/index.html`, `static/css/play.css`)**:
  - Above the Remaining table, displays the explanatory legend:
    `[found] + [missed] = [total words]  [percent found]`
  - Built as an inline responsive vector SVG (`viewBox="0 0 380 16"` with `width: 100%`) that automatically stretches to match the table margins across all devices.
  - Scaled with safety padding so the beginning `[` and ending `]` brackets never clip or spill outside the panel on any mobile device, tablet, laptop, or desktop display.

### 13. Lobby Sliding Drawer Equal Width Layout
- **Equal Horizontal Width for Players in Lobby & Lobby Chat (`static/css/lobby.css`)**:
  - Updated `.lobby-slide-body` on desktops and laptops to `grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)`.
  - Increased horizontal space for the `PLAYERS IN LOBBY` panel of online users so user chips spread out across 50% of the drawer.
  - Decreased the horizontal width of `Lobby Chat` messages to match, making the horizontal length of both elements exactly equal.

### 14. Closed Rooms Default Sorting & Proximity Activation
- **Closed Rooms Default Order (`static/js/lobby.js`)**:
  - When the **Closed Rooms** button is clicked, it defaults to showing rooms with the **highest average rating first**.
  - If the user explicitly presses **Enter** in the "Sort rooms by proximity..." textbox (or clicks **My Rating**), the closed rooms then sort by proximity to that average rating value.
  - Switching back to the Closed Rooms tab resets the sorting back to highest average rating first.

### 15. Tools: Personal Timer Pause/Resume, Mobile Stacked Buttons & Scroll Stability
- **Pause & Resume Controls (`templates/index.html`, `static/js/tools.js`, `static/css/play.css`)**:
  - Added a Pause/Resume button to Personal Timer styled with the identical blue gradient (`linear-gradient(135deg, #00d2ff, #3a7bd5)`), border radius, box shadow, and typography as Start Timer.
  - On desktop and laptops, Pause and Stop Timer appear side by side.
  - On mobile devices, buttons stack vertically in a single column (`flex-direction: column !important; width: 100% !important;`), placing each button on its own individual row on top of each other with large, readable font.
  - Explicitly styled `#timer-stop-btn` with red gradient (`#ff4d4d` to `#d43f3f`) and Pause/Resume/Start with blue gradient (`#00d2ff` to `#3a7bd5`), preventing generic theme overrides from washing out button colors.
  - Clicking Pause freezes countdown interval, preserves remaining seconds, and toggles button in-place to "Resume".
  - Clicking Resume restarts countdown from the remaining time and toggles button in-place back to "Pause".
  - Clicking Stop Timer resets the timer, hides Pause/Resume and Stop, and reveals Start Timer.
- **Mobile Scroll Stability (`templates/index.html`, `static/css/play.css`)**:
  - Moved `.mobile-bottom-nav.tools-bottom-nav` inside `.tools-content-column` so `.tools-split-layout` contains exactly 2 panels instead of a phantom 3rd pane.
  - Added `touch-action: pan-y !important;` and `overscroll-behavior-x: none !important;` to `.tools-content`, `.tools-content-column`, `.tool-pane`, and `#tool-timer`.
  - Added strict `box-sizing: border-box !important;` and `overflow-x: hidden !important;` across `#tool-timer` and `#timer-display-container`, preventing any horizontal jitter, shift, or wobble when scrolling vertically on mobile devices.

### 16. Tools: Find Count Random Words Table Mobile Layout Fix
- **Mobile Row-by-Row Display & Vertical Elongation Fix (`templates/index.html`, `static/css/play.css`)**:
  - Removed accidental `history-table` class from `.random-words-table` in `templates/index.html`. Previously, `.history-table` forced mobile columns to `width: 22% !important; word-break: break-word !important;`, which caused words to break character-by-character vertically into a single-pixel column, elongating the table by hundreds of pixels vertically.
  - On mobile devices (`@media (max-width: 900px)` and `body.is-mobile`), random words run cleanly row-by-row (`flex-direction: column !important; width: 100% !important;`), with each word centered on its own individual row with a subtle divider line, generous touch target padding (`10px 12px`), and clear font size (`0.95rem`), completely eliminating any vertical squishing or character breaking.
  - On desktops and laptops (`@media (min-width: 901px)` and `body:not(.is-mobile)`), random words continue running horizontally in a single bar side-by-side (`flex-direction: row !important; flex: 1 1 0 !important;`).
  - **Reordered Placement Above Search Input (`templates/index.html`)**: Moved the Random Words table, "More random words" button, and dictionary filter dropdown above the "ENTER WORD..." textbox and Search button for quicker suggestion discovery and immediate tap-to-search interaction.

### 17. Tools: Random Word Generator "Contains All" Letter Filter & Mobile Compact Layout
- **"Contains All" Letter Filter Dropdown (`templates/index.html`, `static/js/tools.js`, `app.py`)**:
  - Added a `<select id="random-contains">` dropdown menu to the Random Word tool positioned between the length filter and dictionary filter.
  - Defaults to `Contains All` (value `"all"`, no letter restriction) and lists letters `A` through `Z`.
  - Frontend `generateRandomWord()` in `static/js/tools.js` retrieves `contains` and passes `&contains=${encodeURIComponent(contains)}` to `/api/tools/random_word`.
  - Backend `tools_random_word()` in `app.py` accepts the `contains` query parameter, validates single alphabet characters, and filters words so that the selected letter is present on every generated word.
- **Mobile Compact Spacing Symmetry with Lists Tool (`static/css/play.css`)**:
  - On mobile devices (`@media (max-width: 900px)` and `body.is-mobile`), `#tool-random .combo-control-panel` adopts a 2-column grid layout matching the Lists tool (`grid-template-columns: 1fr 1fr; gap: 6px;`).
  - Controls are sized compactly with `height: 36px`, `font-size: 0.82rem`, and `border-radius: 6px`.
  - Top row displays `random-length` and `random-contains` side-by-side; `random-dict` and `random-gen-btn` span both columns (`grid-column: 1 / -1`) to maximize tap target clarity and space efficiency.
- **Desktop Single-Row & Responsive Wrap Layout (`static/css/play.css`)**:
  - On desktop and laptop viewports, `#tool-random .combo-control-panel .input-row` displays all four controls side-by-side in a neat horizontal flex row with `gap: 8px` and `flex-wrap: wrap !important;` to ensure responsive resilience without truncation.

### 18. Tools: Sequence Search Mobile Compact Layout & Spacing
- **Compact 2-Column Grid Matching Lists Tool (`static/css/play.css`)**:
  - On mobile devices (`@media (max-width: 900px)` and `body.is-mobile`), `#tool-sequence .combo-control-panel` adopts the space-saving 2-column grid layout matching Lists in Tools (`grid-template-columns: 1fr 1fr; gap: 6px;` with compact `36px` element heights, `0.82rem` font size, and `border-radius: 6px`).
  - Symmetrically sorted elements into:
    - Row 1: Sequence textbox (`seq-input`) and search mode dropdown (`seq-mode`, Contains/Starts/Ends) side-by-side.
    - Row 2: Dictionary dropdown (`seq-dict`) and length dropdown (`seq-length`) side-by-side.
    - Row 3: Search button (`seq-search-btn`) spanning both columns (`grid-column: 1 / -1; width: 100%`).
### 19. Tools: Combo Checker Mobile Compact 2-Column Layout & Slim MP/LIC Tabs
- **Compact 2-Column Input & Dictionary Layout (`static/css/play.css`)**:
  - On mobile devices (`@media (max-width: 900px)` and `body.is-mobile`), `#tool-combo .combo-unified-panel .combo-control-panel .input-row` adopts a space-saving 2-column grid layout (`grid-template-columns: 1fr 1fr; gap: 6px;`).
  - Row 1: `ENTER LETTERS TO CHECK...` input (`#combo-input`) and dictionary dropdown (`#combo-dict`) side-by-side.
  - Row 2: Check button (`#combo-search-btn`) spanning both columns (`grid-column: 1 / -1; width: 100%`).
  - Vertically reduced element heights to `36px` (`min-height: 36px`, `max-height: 36px`), compact padding (`0 8px`), and `border-radius: 6px`.
  - Blue control panel box (`.combo-unified-panel .combo-control-panel`) padding reduced from `12px 14px` to `8px 10px; margin-bottom: 8px; border-radius: 10px;`.
  - Styled `#combo-input::placeholder` with `text-transform: none; font-size: 0.78rem` so "Enter letters to check..." fits cleanly without truncation on small screens, while maintaining `16px` font-size on the active input to prevent iOS Safari auto-zoom.
- **Slim MP and LIC Segmented Tabs (`static/css/play.css`)**:
  - Compacted `.combo-tabs-header`: `padding: 3px; margin: 0 auto 8px auto; max-width: 280px; gap: 4px; border-radius: 8px;`.
  - Compacted `.combo-tab-btn`: height reduced to `30px` (`min-height: 30px`, `max-height: 30px`), `padding: 4px 10px`, `font-size: 0.82rem`, and `border-radius: 6px`.
  - Substantially increases visible screen real estate for the MP and LIC results tables below.
  - Desktop and laptop viewports (`> 900px`) remain completely untouched.

### 20. Mods: Lobby Notice Mobile Screen-Fit Layout
- **Complete Viewport Screen Fit Without Scrolling (`templates/index.html`, `static/css/play.css`)**:
  - Re-engineered the mobile layout for Lobby Notice (`#mod-tab-notice`) under `@media (max-width: 900px)` and `body.is-mobile` to fit all components completely on screen across all mobile devices (from iPhone SE up to large smartphones) without vertical scrolling.
  - **Tool Header Note**: Compacted padding (`4px 8px`), margin (`0 0 6px 0`), and font size (`0.74rem`).
  - **Live Announcement Card**: Slimmed card padding (`8px 10px`), reduced header row spacing, and compact preview box (`min-height: 32px; max-height: 52px; padding: 6px 8px; font-size: 0.78rem`).
  - **Compose Card**: Reduced padding to `8px 10px; margin-bottom: 6px;`, set `#notice-status-area:empty` to `display: none` to eliminate empty dead space, and sized the announcement textarea compactly (`height: 72px; min-height: 68px; max-height: 85px; font-size: 16px; padding: 6px 8px;` with `0.78rem` placeholder) to prevent iOS auto-zoom while fitting gracefully.
  - **Side-by-Side Action Buttons**: Replaced the previous 100% full-width stacked buttons with a sleek single-row flex layout (`.mod-notice-buttons`): `📢 Broadcast Notice` expands flexibly and `Clear Notice` is sized compactly (`flex: 0 0 105px`), both with unified `36px` height matching other mobile tool buttons.
  - Total content height reduced from over 650px down to ~320px, fitting comfortably well within the visible viewport above the bottom navigation bar on all phones. Desktop/laptop displays (`> 900px`) remain 100% untouched.

### 21. Mods: Ban/Timeout User Reduced Description Padding on Mobile & Aesthetic Textboxes on All Platforms
- **Aesthetic Textbox & Control Modernization Across All Platforms (`static/css/play.css`)**:
  - Replaced primitive, unstyled default browser textboxes with cohesive, theme-integrated modern controls across desktop, laptop, tablet, and mobile.
  - `#timeout-username-input` and `#ban-username-input`: Styled with `var(--input-bg)` and `1.5px solid var(--input-border)`, `8px` border-radius (`6px` on mobile), `40px` height (`36px` on mobile), subtle inset shadow, theme-adaptive text color, and bespoke glow outlines on focus (amber `#f59e0b` for Timeout, crimson `#f43f5e` for Ban).
  - `.timeout-duration-box` & `#timeout-hours-input`: Redesigned duration container into an aesthetic pill control with centered typography and an embedded dark-contrast number input (`#f59e0b` bold text, `28px` height, `6px` radius).
  - `#timeout-reason-input` & `#ban-reason-input`: Styled textareas with matching background, borders, `8px` radius (`6px` on mobile), subtle inset shadows, and themed focus ring glows.
  - Buttons (`#timeout-user-btn`, `#check-timeout-btn`, `#lift-timeout-btn`, `#ban-user-btn`): Unified height to `40px` on desktop (`36px` on mobile) with `8px` radius (`6px` on mobile) for flawless horizontal row alignment.
- **Reduced Description Padding & Compact Layout on Mobile (`static/css/play.css`)**:
  - Compacted `#mod-tab-ban .tool-header p`: `padding: 4px 8px; margin: 0 0 4px 0; font-size: 0.74rem;`.
  - Collapsed empty status text area (`#mod-tab-ban #ban-status-area:empty` to `display: none;`), eliminating 35px of dead blank space directly under the top description.
  - Reduced description bottom margins inside cards (`.mod-action-card p: margin-bottom: 6px; font-size: 0.76rem;`).
  - Sized action cards to `padding: 8px 10px; margin-bottom: 8px; border-radius: 8px;`.
  - Significantly compresses vertical spacing on mobile screens, bringing the Permanent Ban & Erase section into full view without extensive scrolling.

### 22. Mods: Moderator Access Reduced Padding Above and Below Add Moderator Panel on Mobile
- **Eliminated Dead Void Above Panel (`static/css/play.css`)**:
  - Overrode `#mod-tab-access .tool-header p` with compact padding (`4px 8px`), tight margin (`0 0 4px 0`), and responsive typography (`0.74rem`).
  - Added `#mod-tab-access #mod-status-area:empty { display: none !important; min-height: 0 !important; margin: 0 !important; padding: 0 !important; }`, eliminating 35px of vacant blank space above the panel when no status message is active.
- **Compacted Add Moderator Panel & Eliminated 55px Void Below Panel (`static/css/play.css`)**:
  - Overrode `.mod-add-section` desktop margin (`margin-bottom: 40px !important;` reduced to `margin-bottom: 8px !important;`).
  - Compacted panel container padding from `20px` / `12px` down to `8px 10px !important;` with a tight `gap: 6px !important;` and `border-radius: 8px !important;`.
  - Unified `#new-mod-username` input height to `36px !important` with `16px` font size (preventing iOS Safari auto-zoom) and `0.78rem` placeholder.
  - Sized `#add-mod-btn` to `36px !important` with `0.82rem` font size and `6px` radius.
  - Compacted `#mod-tab-access .mod-list-title` margins to `margin-top: 6px !important; margin-bottom: 6px !important; font-size: 0.88rem !important;`.
  - Reduced moderator list grid gap (`8px !important;`) and list item padding (`10px 14px !important;`).
- Applied consistently under both `@media (max-width: 900px)` and `body.is-mobile`. Desktop and laptop views (`> 900px`) remain completely untouched.

### 23. Tools: "CC", "FC", and "ADD" Action Buttons in Random Word and Is Valid
- **Bottom Panel Word Actions Matching Definition Pop-ups (`templates/index.html`, `static/js/tools.js`, `static/css/play.css`)**:
  - Integrated the exact three bottom action buttons (**CC**, **FC**, and **ADD**) along the bottom of the word content panels in both **Random Word** (`#random-word-actions`) and **Is Valid** (`#valid-word-actions`).
  - Separated with a subtle top border line matching definition pop-up modals (`border-top: 1px solid rgba(var(--text-primary-rgb), 0.12);`, and light-theme border `#cbd5e1`).
  - **CC (Combo Checker)**: Navigates smoothly to Combo Checker searching the word with its selected dictionary, and tracks navigation stack for instant back-navigation.
  - **FC (Find Count)**: Navigates directly to Find Count with the word entered and history tracking.
  - **ADD / ADDED**: Checks the user's "Words to Work on" list via `getWordsToWorkOn()`. If the word is already on their list, the ADD container is cleanly hidden allowing CC and FC to expand symmetrically across the full width (identical to definition pop-up behavior). If not yet on the list, clicking ADD instantly adds the word to the user's study list, updates the button label to "ADDED", and adds class `.is-added` with emerald green styling (`#10b981`) and pointer events disabled.
  - Sized for responsive touch and desktop clicks: `32px` on desktop and `30px` on mobile with `6px` border-radius.
  - In Is Valid, buttons show when a word is confirmed valid (`IS VALID`) and cleanly hide if the word is invalid (`IS NOT VALID`) or during validation check loading. In Random Word, buttons display upon word generation and clear during loading.

### 24. Desktop: Equalized Left and Right Padding for Vertical Separator Line Between Donate and Logged-in Username
- **Symmetrical Spacing on Desktop & Firefox Normalization (`static/css/style.css`, `templates/index.html`)**:
  - Resolved an asymmetry issue on desktop displays—specifically prominent in desktop Firefox where the distance between the Donate tab and the vertical line (`border-left` of `.user-display`) was compressed to just 4px (or expanded excessively when `margin-left: auto` was active), while the distance to the username on the right side of the line was 10px–15px.
  - Eliminated `margin-left: auto !important` on `.user-display` across desktop media queries (`@media (min-width: 901px) and (max-width: 1380px)`), replacing it with mathematically balanced `margin-left` values that account for the flex container's `gap`.
  - **Firefox Desktop Breakpoints (`html.is-firefox-desktop` & `@supports (-moz-appearance: none)`)**:
    - Large Desktop (`> 1380px`): Sized `.nav` gap to `6px`, `.user-display` `margin-left: 9px !important;` (left spacing = `6px + 9px = 15px`) and `padding-left: 15px !important;` (right spacing = `15px`), producing exact 15px:15px symmetry.
    - Mid-size Desktop & Laptops (`1101px - 1380px`): Sized `.nav` gap to `4px`, `.user-display` `margin-left: 6px !important;` (left spacing = `4px + 6px = 10px`) and `padding-left: 10px !important;` (right spacing = `10px`), producing exact 10px:10px symmetry.
    - Compact Desktop & Smaller Laptop Screens (`901px - 1100px`): Sized `.nav` gap to `3px`, `.user-display` `margin-left: 5px !important;` (left spacing = `3px + 5px = 8px`) and `padding-left: 8px !important;` (right spacing = `8px`), producing exact 8px:8px symmetry.
  - **Base & Standard Desktop Breakpoints (Chrome, Safari, Edge)**:
    - Base desktop rule: `margin-left: 11px;` with `gap: 4px` (left spacing = 15px) and `padding-left: 15px;` (right spacing = 15px).
    - Compact/mid desktop rule (`901px - 1380px`): `margin-left: 5.5px !important;` with `gap: 2.5px` (left spacing = 8px) and `padding-left: 8px !important;` (right spacing = 8px).
### 25. Tools: Random Word "Contains Any" Rename & Find Count Input Spacing Adjustment
- **Random Word "Contains Any" Label Rename (`templates/index.html`)**:
  - Renamed the letter filter dropdown (`#random-contains`) default option text from `"Contains All"` to `"Contains Any"`.
  - Maintained `value="all"` and the underlying API query logic (`/api/tools/random_word?contains=all`) intact across all platforms so words containing any letters remain generated without alteration.
- **Find Count Input Panel Spacing Adjustment (`templates/index.html`, `static/css/play.css`)**:
  - Increased spacing above the word input panel: Increased `.suggested-words-section` bottom margin from `20px` to `32px`, placing the submit textbox visibly further away from the random words table, the "More random words" button, and the dictionary filter select dropdown.
  - Decreased spacing below the word input panel: Reduced `#find-count-results-container` top margin from `25px` down to `10px`, and set mobile `#tool-find-count .combo-control-panel` bottom margin to `0px !important` (overriding the generic mobile `12px !important`), bringing the submit textbox and search button into tight proximity (`10px` gap) with the `"The word 'ALOHA' has been found 0 times..."` results summary message.
  - Bumped stylesheet cache buster in `templates/index.html` for `play.css` to `v=1793420000`.
### 26. Game Header & Lobby: Bold Values for Spinner Set, Multiplayer Subanagrams, and Active Rooms Selected Game Type Panel
- **Spinner Set & Multiplayer Subanagrams Parameter Values (`static/css/play.css`)**:
  - Enforced `font-weight: 700 !important;` on all parameter value spans inside `.game-params`:
    - `#param-diff` (e.g. `Easy`, `Medium`, `Hard`, or letters count in Subanagrams)
    - `#param-min` (e.g. `3L`, `4L`)
    - `#param-dict` (e.g. `CSW`, `TWL`, `ENABLE`)
    - `#param-range` (e.g. `300-400`, `50-100`)
    - `#param-format` (e.g. `Normal`, `Totally Random`)
    - `#param-bonus` (e.g. `8L`, `7L`)
  - Preserved parameter labels (`Diff:`, `Letters:`, `Min:`, `Dict:`, `Words:`, `Format:`, `Bonus:`) and pipe separators (`|`) at `font-weight: 500`, explicitly styling `#param-diff-label { font-weight: 500 !important; }`.
  - Automatically covers standard game header rooms and Multiplayer Subanagrams across both desktop and mobile layouts.
- **Lobby Active Rooms "Select a game type" Panel (`templates/index.html`, `static/js/app.js`, `static/css/lobby.css`)**:
  - In `handleShowRoomsInline` (`templates/index.html`) and `updateActiveRoomsPanel` (`static/js/app.js`), wrapped constant parameter values in `<strong>` tags when a "Show Rooms" button is selected under ACTIVE ROOMS:
    - Displays: `Board: <strong>6x8</strong> | Time: <strong>45s</strong>`
    - Preserves `"Board:"`, `"Time:"`, and the pipe separator `" | "` as regular/unbold (weight 500).
  - Added `.selected-game-params strong { font-size: inherit !important; font-weight: 700 !important; }` in `static/css/lobby.css` to guarantee clean baseline alignment and exact font sizing.
  - Bumped cache busters in `templates/index.html` for `play.css`, `lobby.css`, and `app.js` to `v=1793421000`.
### 27. Round Replay: Mobile 4x4 Tile Sizing, Word Scroller & Desktop/Laptop Bottom Word Visibility Fix
- **Mobile 4x4 Board Demonstration Resizing (`static/js/tools.js`)**:
  - Calibrated 4x4 mobile cell size calculation (`tools.js`) to use 6-column metrics (`effectiveCols = 6`), rendering tiles at the exact same size as the 4x6 demonstration.
  - Aligns 4x4 vertical board height with 4x6, 5x7, 6x8, and 3D Cube demonstrations (~200px–225px), preventing oversized boards from pushing words offscreen.
- **Mobile Walkthrough List Height & Dedicated Words Scroller (`static/css/play.css`, `static/js/mobile_scroller.js`)**:
  - Set `#review-walkthrough-list` height on mobile to `180px` (`height: 180px !important; max-height: 180px !important; min-height: 180px !important; overflow-y: auto !important;`), ensuring exactly 4 found words are visible at a time consistently across all board dimensions (4x4, 4x6, 5x7, 6x8, 3D).
  - Removed whole-page scroller track and thumb on `.history-review-layout` (`padding-right: 0 !important;`).
  - Attached custom vertical scroller track and draggable gradient thumb directly to `#review-walkthrough-list` (`mobile_scroller.js`), positioned over the right edge of the words list (`padding-right: 22px !important;`).
- **Desktop & Laptop Bottom Word Visibility & Draggable Scrollbar Thumb Fix (`static/css/style.css`)**:
  - Resolved parent container clipping where `.walkthrough-list`'s hardcoded `max-height: 500px` exceeded the available modal height, causing the bottom words (last 2 words) and the bottom of the scrollbar thumb to extend outside `.history-review-main`'s `overflow: hidden` boundary.
  - Updated `.history-review-modal` to `height: 85vh; max-height: 90vh;`.
  - Configured flex layout across `.history-review-layout`, `.history-review-main` (`align-items: stretch; min-height: 0;`), `.history-walkthrough-section` (`min-height: 0; flex: 1;`), and `.walkthrough-list` (`flex: 1 1 0; min-height: 0; max-height: none; overflow-y: auto; padding-bottom: 24px;`).
  - Ensured all words—including the very last words found—are fully visible upon scrolling to the bottom, with the scrollbar thumb remaining completely visible and draggable.
- **Cache Busters (`templates/index.html`)**:
  - Bumped `style.css`, `play.css`, `mobile_scroller.js`, and `tools.js` cache busters to `v=1793422000`.

### 28. Round Replay: Mobile 3 Visible Words & Strict Words-Only Scroll Lock
- **3 Visible Words at a Time (`static/css/play.css`)**:
  - Calibrated `#review-walkthrough-list` height on mobile to `132px` (`height: 132px !important; max-height: 132px !important; min-height: 132px !important;`) with each `.walkthrough-item` fixed at `44px` (`height: 44px !important; min-height: 44px !important; max-height: 44px !important; box-sizing: border-box !important; padding: 10px 0 !important;`).
  - Displays exactly 3 words at a time across all board dimensions (4x4, 4x6, 5x7, 6x8, 3D), with additional words smoothly accessible via vertical scrolling.
- **Strict Words-Only Scrolling (`static/css/play.css`, `static/js/tools.js`)**:
  - Ensured the entire Round Replay page / modal is locked and NOT scrollable:
    - `#history-review-overlay`: `overflow: hidden !important; touch-action: none !important;`
    - `.history-review-modal`: `overflow: hidden !important; touch-action: none !important;`
    - `.history-review-layout`: `overflow: hidden !important; overflow-y: hidden !important; touch-action: none !important;`
    - `.history-review-main`: `overflow: hidden !important; overflow-y: hidden !important; touch-action: none !important;`
    - `body.replay-overlay-open`, `html.replay-overlay-open`: `overflow: hidden !important; touch-action: none !important;`
  - In `static/js/tools.js`, added `replay-overlay-open` class toggling to body/html on overlay open/close, and bound a non-passive `touchmove` listener on `#history-review-overlay` that calls `e.preventDefault()` on touches outside `#review-walkthrough-list` and `.m-scroller-track`.
  - Configured `touch-action: pan-y !important;` exclusively on `.walkthrough-list` and `.walkthrough-item`, ensuring only the found words list can scroll.
  - Adjusted `.m-scroller-track-overlay .m-scroller-thumb` with `min-height: 32px !important;` for responsive dragging within the 132px word list.
- **Cache Busters (`templates/index.html`)**:
  - Bumped `play.css` and `tools.js` cache busters to `v=1793423000`.

### 29. Round Replay: Mobile Layout Moved Up Below Header & Dynamic 3-Word Collapse on "Watch Replay"
- **Mobile Layout Moved Up Below Header (`static/css/play.css`)**:
  - Replaced bottom-anchored `column-reverse` flex alignment on `.history-review-main` with top-anchored `display: flex !important; flex-direction: column !important; justify-content: flex-start !important;`.
  - Positioned `#review-board-container` immediately below `.overlay-header` using `order: 1 !important; flex: 0 0 auto !important; margin-bottom: 0 !important;`.
  - Positioned `.history-walkthrough-section` directly under the board using `order: 2 !important; flex: 1 1 0 !important; min-height: 0 !important; display: flex !important; flex-direction: column !important;`.
  - Allowed `#review-walkthrough-list` to expand into all the freed-up vertical space (`flex: 1 1 0 !important; min-height: 0 !important; height: auto !important; max-height: none !important;`), giving users extensive visible scrollable words when browsing round history.
- **Dynamic Collapse to 3 Words on "Watch Replay" (`static/css/play.css`, `static/js/tools.js`)**:
  - When the user clicks "Watch Replay" (`startBtn`), `.history-review-modal` receives `.replay-active` and `#review-walkthrough-list` receives `.collapsed-3`.
  - On mobile, `.replay-active .walkthrough-list` and `.walkthrough-list.collapsed-3` collapse to `height: 132px !important; max-height: 132px !important; min-height: 132px !important; flex: none !important;`, displaying exactly 3 words at a time while the replay animation and progress bar (`#replay-progress-container`) run.
  - When replay finishes or "Show All" (`skipBtn`) is clicked, `.replay-active` and `.collapsed-3` are removed, expanding the words list back into the full vertical space.
  - Calls `window.updateReplayScroller()` to smoothly recompute track height and thumb proportions across expansions and collapses.
- **Cache Busters (`templates/index.html`)**:
  - Bumped `play.css` and `tools.js` cache busters to `v=1793424000`.

### 30. Game Room Chatbox Placement Numbers (`game_room.py`, `app.py`, `static/js/play.js`, `templates/index.html`)
- **Chatbox Placement Prefix**:
  - In `#chat-history`, displays the user's placement from the round using a number sign (`#`) followed by their placement before their username: `#<placement> <username> (<rating>): <message>` (e.g. `"#13 jeffb (1200): Hello, everybody!"`).
  - If a user did not play the round (DNP, spectator, or unranked), the placement number prefix is omitted: `<username> (<rating>): <message>` (e.g. `"jeffb (1200): Hello, everybody!"`).
- **Backend Tracking & Placement Calculation (`game_room.py`, `app.py`)**:
  - In `GameRoom`, added `last_round_placements: Dict[str, int] = field(default_factory=dict)` storing the 1-based placement of all round participants sorted by score descending, breaking ties with rating descending.
  - Excluded players who did not play (DNP with 0 score, 0 words submitted, and 0 invalid words) and spectators from `last_round_placements`.
  - Updated `add_chat_message` to record `'placement': placement` directly in message dicts, resolving automatically from `self.last_round_placements` or `self.winners_history[0]['all_players']`.
  - In `app.py`, added `last_round_placements` to `/api/room/<room_id>/state` snapshot, and updated `/api/room/<room_id>/chat` to accept and record `placement`.
- **Frontend Live & Optimistic Handling (`static/js/play.js`, `templates/index.html`)**:
  - Added `updateRoundPlacementsMap(state)` to populate `window._lastRoundPlacements` dynamically from server state, with client fallbacks for intermission players and `winners_history[0]`.
  - Updated `sendChatMessage()` to inject `placement` into optimistic chat messages for 0ms visual feedback and send it in the chat API payload.
  - Updated `renderChat(messages)` to prefix the username with `#<placement> ` when available, keeping profile clicking intact via `data-username` and clean-name regex stripping.
  - Bumped `play.js` cache buster to `v=1793425000`.

### 31. Tools Unscramble: Mobile Full-Width Dropdown & Desktop Lighter Color Theme Panel Division (`templates/index.html`, `static/css/play.css`, `static/css/style.css`)
- **Mobile Dropdown Full Available Space & "MUST HAVE: ANY" Default Text**:
  - In `templates/index.html`, updated `#unscramble-must-have` options to display `"MUST HAVE: ANY"` for value `""` (selected by default) and `"MUST HAVE: <Letter>"` for letters A-Z (`"MUST HAVE: A"` through `"MUST HAVE: Z"`).
  - Removed the side label `<span class="unscramble-must-have-label">` from the markup and hid any remaining instances via CSS (`display: none !important;`).
  - In `static/css/play.css`, configured `.unscramble-must-have-group` and `#unscramble-must-have` on mobile viewports (`max-width: 650px`) to span 100% of the available row width (`width: 100% !important; max-width: 100% !important; flex: 1 1 100% !important; text-align-last: center !important; height: 40px !important; margin: 0 !important; gap: 0 !important;`), matching the horizontal length and appearance of "Len: 7", "All (Full)", and "New Game".
- **Desktop Light Theme Left and Right Panel Emphasized Division**:
  - In `static/css/style.css`, removed `.unscramble-container` from lines 607-610 where it had previously received a solid white background (`#ffffff !important`) and border (`1px solid #cbd5e1 !important`), which had grouped both panels into a single enclosing white card and eliminated the division between them.
  - Set `#tool-unscramble .unscramble-container` to `background: transparent !important; border: none !important; box-shadow: none !important;` on all white and light themes (`theme-white`, `[class*="theme-white"]`, `[class*="theme-light-"]`).
  - Added Section 16b in `static/css/style.css` styling `.unscramble-game-area` (left panel) and `#unscramble-found-container` (right panel) as independent bounded cards with distinct white/card backgrounds (`#ffffff` on white, `var(--bg-primary)` + `linear-gradient(var(--bg-panel), var(--bg-panel))` on light themes), 1.5px solid contrasting borders (`#cbd5e1` on white, `var(--card-border)` on light), and subtle drop shadows (`0 6px 24px rgba(0,0,0,0.08)` / `0 10px 30px rgba(0,0,0,0.1)`).
  - The 14px gap between the two panels now exposes the page background (the same contrasting color used above them around the control panel), emphasizing the clean division between the left and right panels exactly matching Subanagrams in Tools.
- **Cache Busters (`templates/index.html`)**:
  - Bumped `style.css` and `play.css` cache busters to `v=1793426000`.

---

## Verification & Validation

| Test / Check | Result | Verification Notes |
|:---|:---|:---|
| **Local Working Tree** | ✅ Clean | Clean working tree on `main` |
| **Origin / GitHub** | ✅ Synchronized | Synchronized with `origin/main` |
| **Remote Production** | ✅ Synchronized | Deployed to `132.148.72.249`, PM2 online, HTTP/2 200 OK |
| **Mobile App (WebView)** | ✅ Synchronized | Points directly to `https://morpheme.games/` |
| **Dictionary Line Count & Integrity** | ✅ Verified | Byte-exact synchronization of dictionary text files and SQLite databases |
| **Git Tags** | ✅ Tagged | `START_OVER_POINT_OCTOBER_30` and `stable-2026-10-30` updated to latest commit |


