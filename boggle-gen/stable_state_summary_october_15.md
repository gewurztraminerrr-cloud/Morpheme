# Stable State Summary — October 15, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_15
```
*(Also tagged as `stable-2026-10-15`)*

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| **localhost**      | ✅ Synchronized | Working tree clean, synced with origin/main |
| **GitHub (main)**  | ✅ Synchronized | Pushed to `origin/main` & tags updated on GitHub |
| **morpheme.games** | ✅ Synchronized | Deployed to 132.148.72.249 (PM2 online, verified live) |
| **App / Client**   | ✅ Synchronized | Verified on desktop, laptop, tablet, and mobile layouts across all 39 themes |

---

## Session Highlights & Architectural Updates (October 14–15, 2026)

### 1. Multiplayer Subanagrams UI & Board Enhancements
- **Lobby Stretch on Mobile**:
  - In `static/css/lobby.css`, added a mobile override for `.matrix-subanagrams .matrix-header` and `.matrix-subanagrams .matrix-row` (`grid-template-columns: 45px 1fr !important;`).
  - The multiplayer Subanagrams **Start** button and its header row expand horizontally across the entire available screen width next to the "2m" indicator instead of being confined to a narrow column.
- **Lobby Header & Column Updates**:
  - In `templates/index.html`, updated the Lobby title from **`Subanagrams (Practice)`** to **`Subanagrams`** across all platforms.
  - Updated the column header text directly above the Start button from **`Practice`** to **`Practice / Warmup`**.
- **Restricting the Enter Button Strictly to Multiplayer Subanagrams**:
  - In `static/css/play.css`, hidden `#submit-word-btn` by default across all standard game rooms (4x4, 5x5, 6x8, 3D, tournament, private, 24h) on desktop, laptop, mobile, and tablet (`display: none !important;`).
  - Restored `body.is-desktop .word-input-section` max-width back to `500px !important;` for standard rooms, keeping the word input box, Rotate, and Transpose buttons centered with standard spacing.
  - In multiplayer Subanagrams (`body.is-subanagrams`), displayed the emerald **Enter** button (`#submit-word-btn`) with `display: inline-flex !important;` on both desktop and mobile.
- **Mobile Subanagrams Input Textbox Restored**:
  - Overrode the mobile hiding rule inside `@media (max-width: 992px)` for Subanagrams (`body.is-subanagrams .word-input-section`), displaying `#word-input` with centered uppercase letters (`font-size: 1.1rem; letter-spacing: 2px;`) to display letters in real time as tiles are tapped on mobile devices.
- **Instant Synchronous Loading Feedback on Room Entry**:
  - In `templates/index.html` and `static/js/lobby.js`, eliminated the 2–3 second delay when entering a room from the lobby.
  - `window.showLoadingOverlay('Loading...')` is invoked **synchronously on the first line** of click handlers before any network requests (`checkAccountTimeoutAndAlert`), with forced DOM reflow (`void overlay.offsetHeight;`) ensuring the loading spinner and message render in the exact frame the button is clicked.

### 2. Lexicographical Sourcing & Permanent Restoration of WANGYANITE / WANGYANITES
- **Root Cause Identified**:
  - `WANGYANITE` and `WANGYANITES` were originally added via the Mods interface at `22:50` on the production server.
  - When subsequent deployments ran `git reset --hard origin/main`, git-tracked dictionary files were reset to origin, wiping uncommitted additions.
- **Strict AGENTS.md Rule Compliance**:
  - **`WANGYANITE`**:
    `A rare sulfide mineral with the chemical formula palladium nickel octasulfide (PdNi8S8).`
  - **`WANGYANITES`** *(repeats base singular definition per AGENTS.md Rule 1)*:
    `plural of wangyanite (A rare sulfide mineral with the chemical formula palladium nickel octasulfide (PdNi8S8).)`
  - Added into `dictionaries/added_words.txt`, `dictionaries/added_words_duplicate.txt`, `dictionaries/added_words_dates.txt` (date: `14/10/2026`).
  - Added into `dictionaries/Definitions.txt`, `dictionaries/wikdefs.txt`, `dictionaries/wikdefs_duplicate.txt`, `dictionaries/wikdefs_aw_duplicate.txt`.
  - Added into SQLite database `morpheme.db` (`wiktionary_definitions` table) on both local and production environments.

### 3. Automated Deployment Safety Guard (`scratch/pull_latest.py`)
- Upgraded deployment workflow with a pre-reset safety guard that scans `added_words.txt` and `added_words_dates.txt` on remote for uncommitted additions before performing `git reset --hard`.
- Any words added on production that do not yet exist in `origin/main` are automatically backed up, preserved across the git reset, and seamlessly re-injected so user-added words will never be lost on future deploys.

### 4. Fix for Tools "ALL" Dictionary Loader
- **Bug**: In `app.py`, `load_tools_dictionary('ALL')` previously read word files using `line.strip().upper()` without splitting on whitespace (`.split()[0]`). For date-stamped files like `new_NWL.txt`, words with dates (e.g., `TESTTESTS\t22/05/2026`) failed exact string matching in `ALL`.
- **Fix**: Updated `load_tools_dictionary('ALL')` to parse words using `line.strip().split()[0].upper()`.
- **List Badge Support**: Updated `tools_validate_word` to check membership against cached tools dictionary sets so `NWL` and `CSW` badges display for newly added words.

### 5. Removal of Test Words from New NWL Words & Dictionaries
- Manually purged obsolete test entries `TESTTEST` and `TESTTESTS` from `dictionaries/new_NWL.txt`, `added_words.txt`, `added_words_duplicate.txt`, `Definitions.txt`, and `wikdefs_aw_duplicate.txt` on both localhost and remote.
- Verified that both words now return `is_valid: false` across all dictionaries on production.

### 6. Full Synchronization of October 15 Added Words
- Synchronized newly added mineral words from production into localhost and GitHub:
  - **`ABELLOEMRINGERITE`** / **`ABELLOEMRINGERITES`** (`15/10/2026`)
  - **`ACHAVALITE`** / **`ACHAVALITES`** (`15/10/2026`)
  - **`ACHYROPHANITE`** / **`ACHYROPHANITES`** (`15/10/2026`)
- Sourced and placed definitions across all dictionary files and local/remote SQLite databases.

### 7. Equalized Forum Vertical Spacing Above General & Category Tabs
- **Sidebar Spacing Symmetry**:
  - In `static/css/forum.css`, changed `.forum-nav-list` padding from `padding: 6px 0;` to `padding: 0;` (and `padding: 0 !important;` on mobile).
  - Eliminated the asymmetrical `6px` top gap above **General**, ensuring identical `15px` padding above and below each category item and making item active/hover highlights fit flush between the top and bottom divider lines.
- **Category Heading Vertical Margins**:
  - Added `margin-top: 5px;` to `.header-titles h2` (and `margin-top: 4px !important;` on mobile `#forum-category-title`) to eliminate the browser default `0.83em` (~24px) margin-top, giving equal top and bottom spacing around category titles in the main forum view.

### 8. Category Divider Lines on Mobile in Forum
- In `static/css/forum.css`, added `.forum-nav-list .forum-cat-item:not(:last-child)` with `border-bottom: 1px solid rgba(var(--text-primary-rgb), 0.12) !important;` to the global base rules and the mobile `@media (max-width: 900px)` media query.
- Categories on mobile devices now feature the same clean, subtle horizontal divider lines as on desktop and laptop displays.
- Incremented `forum.css` version in `templates/index.html` to `v=1791130148` for immediate cache busting across client browsers.

### 9. Forum Maximum Image Attachment Limit Increased to 6
- **Backend Enforced Limits**: In `app.py`, updated `create_forum_post` (`POST /api/forum/posts`) and `create_forum_comment` (`POST /api/forum/comments`) from `upload_files[:4]` to `upload_files[:6]`.
- **Client Frontend Validation & UI**:
  - In `static/js/forum.js`, updated `addFiles` target array limit check and alert message from 4 to 6 ("You can attach a maximum of 6 images per post.").
  - Updated `renderImagePreviews` file counter and indicator texts (`(up to 6)`, `${targetArray.length}/6 images`, `Maximum 6 images selected`).
  - In `templates/index.html`, updated all upload input labels and prompts for new threads, inline comment boxes, and floating reply popups to reference 6 images instead of 4.
  - Bumped `forum.js` cache buster version in `templates/index.html` to `v=1791130138`.

### 10. Subanagrams "Word Guaranteed" vs "Totally Random" Invariant Enforcement
- **Root Cause**: In Subanagrams mode, boards generated from the letter bag without a full-length word (such as `FLENONUVE` of length 9) could retain or default to the `"Word Guaranteed"` format label if `mode: 'word'` was requested or defaulted, or if intermission parameters had staged `"Word Guaranteed"` before candidate solving finished.
- **Strict Invariant**: Enforced that a board is labeled `"Word Guaranteed"` **if and only if** there is genuinely at least one valid dictionary word of maximum sequence length (`len(w) == seq_len`) formed by those letters; otherwise it is strictly categorized as `"Totally Random"`.
- **Backend Generator & Promotion Updates**:
  - In `subanagrams_generator.py`, verified that `best_results` contains a word where `len(w) == len(best_candidate)`. If no full-length word exists, `board_format` and `mode` are strictly forced to `'Totally Random'` and `'random'`. Added fallback word recovery from `candidates_by_len` when `mode == 'word'` is requested.
  - In `game_room.py`, added authoritative full-length word verification across all board generation and promotion hooks (`subanagrams_search_worker`, `generate_spinner_params`, `start_next_round` rollover and promotion, and kickstart methods).
  - In `app.py`, added full-length word check during public singleton hub reconstruction.
  - In `static/js/play.js`, ensured Subanagrams format strictly falls back to `'Totally Random'` and never displays `'Normal'`. Bumped `play.js` cache buster in `templates/index.html` to `v=1791130400`.

### 11. MP Word Explorer Feature Architecture
- **Interactive Multi-Length Discovery**:
  - Implemented the MP Word Explorer within Combo Checker, allowing users to explore all words of a given length and view their non-plural, non-inflection MP connections.
  - Dropdown controls for Length (4–15), MP distance (0MP–3MP), Dictionary (NWL, CSW, AW, All, etc.), and ordering modes (A–Z and Random).
  - Implemented length-descending sorting (`b.length - a.length`) as primary tier across matches, displaying longest connections first (e.g. 8-letter connections `WASTERIE` and `WASTRIFE` for `WASTRIE` before 7-letter `ASTRIDE`).
  - Excluded 4-letter words from MP Explorer match results by rule.
  - Interactive clickable word links throughout input words and match results opening the instant lexicographical definition popup (`window.lookupWord`).

### 12. Combo Checker Header Design & Description Box Harmonization
- **Title Backgrounds**:
  - Eliminated the light-grey rectangular backgrounds around all section headings in Combo Checker (`#tool-combo .result-section h3, .combo-results-container .result-section h3, #combo-explorer-section h3 { background: transparent !important; }`) across all desktop, laptop, tablet, and mobile platforms.
- **Harmonized Description Formatting**:
  - Matched design, typography, layout, and light-greyish muted color (`rgba(var(--text-primary-rgb), 0.4)`) across all three Combo Checker descriptions: MP (Microprocedures) v2.0, LIC (Letter Insertion Combinations) v2.0, and MP Word Explorer.
  - Applied the Suggestions/Ideas scrolling-box effect (`height: calc(2.7em + 8px); overflow-y: auto;`) with subtle borders, border-radius, and custom scrollbar thumb styling.
  - Added dedicated theme overrides for white and light themes to ensure consistent contrast and readability across dark and light palettes.

### 13. Mobile Virtual Keyboard Black Screen Elimination & Input Visibility
- **Root Cause Identification**:
  - Investigated the 2-second temporary black screen occurring on mobile devices when tapping "FILTER INPUT WORDS..." (`#combo-explorer-filter`) and "ENTER LETTERS TO CHECK..." (`#combo-input`).
  - Identified two contributing factors:
    1. HTML `autocapitalize="characters"` dynamically forcing Android Chrome IME into all-caps mode, stalling GPU rasterizer buffer swaps.
    2. Software keyboard deployment while HTML5 Fullscreen mode is active on mobile devices triggering Android SurfaceFlinger buffer recreation.
- **Permanent Solution**:
  - Removed `autocapitalize="characters"` from `#combo-input` and `#combo-explorer-filter`, setting `autocapitalize="none" autocorrect="off"`. Visual capitalization is handled cleanly via CSS `text-transform: uppercase` and JS `.toUpperCase()`.
  - Added synchronous `exitFullscreenForKeyboard()` checks on `pointerdown` and `touchstart` for Combo Checker inputs in `static/js/app.js` and `static/js/tools.js`, guaranteeing the soft keyboard launches outside of fullscreen without display surface destruction.
  - Implemented `_scrollComboInputIntoView()` in `static/js/tools.js` triggered on focus, pointerdown, and `visualViewport` resize. When the keyboard deploys, `.tools-content` automatically scrolls the input into clear view above the virtual keyboard without pinning or sticking the control panel to the top during normal scrolling.

### 14. Definition Popup "CC" Navigation & Scroll-to-Top
- **Auto-Navigation & Page Switching**:
  - In `static/js/tools.js`, clicking **“CC”** (`#tool-def-cc-btn`) on any word definition popup now automatically switches to the Tools page (`window.showPage('page-tools')`) if invoked from Play, Lobby, or full list views.
  - Resets saved scroll offsets (`window._savedComboScrollTop = 0` and `_savedToolScrollPositions['combo'] = 0`).
- **Clean Scroll to Top**:
  - Resets `.scrollTop = 0` across `.tools-content`, `#combo-scroll-container`, and `#tool-combo`.
  - Resets horizontal scrollers (`#mp-container` and `#lic-container`) to `.scrollLeft = 0`.
  - On mobile, automatically slides the layout horizontally to the tool content pane.
  - Populates the searched word and triggers `runComboSearch()`, leaving the user at the very top of Combo Checker with the header, description, and control panel directly in view.

### 15. Unified Visible Panel for Combo Checker Search and MP/LIC Tabs
- **Visible Container**:
  - In `templates/index.html` and `static/css/tools.css`, unified the "ENTER LETTERS TO CHECK…" search row and the MP / LIC tab row within a single visible enclosing panel card (`.combo-unified-container` / `.combo-unified-card`).
  - Styled with subtle borders (`1px solid rgba(var(--text-primary-rgb), 0.12)`), rounded corners (`border-radius: 12px`), cohesive background tints, and balanced inner padding.
  - Maintained responsive proportions and compact margins across mobile, tablet, and desktop displays so all contents remain fully visible without clipping or overflow.

### 16. New Word Lists Added to "Words to Work on" Dropdown Menu
- **Expanded Lexicon Lists**:
  - In `templates/index.html`, `static/js/tools.js`, and `app.py`, added the following word lists to the "Words to Work on" tool dropdown menu:
    - **`CSW Only`**: CSW words that are not valid in NWL.
    - **`New NWL Words`**: Newly approved words added to the official NWL dictionary.
    - **`New CSW Words`**: Newly approved words added to the official CSW dictionary.
    - **`New AW Words`**: Words newly added to the Added Words (AW) dictionary.
    - **`All New Words`**: Combined aggregate of all recent additions across NWL, CSW, and AW.
  - Wired backend endpoints and cache indexing to serve random and sorted sub-lists with instantaneous retrieval.

### 17. 39-Theme Visual Contrast, Neatness, and Legibility Audit
- **Full Theme Coverage**:
  - Audited all 39 theme color layouts across Morpheme (Standard, Dark, White, Light Gray/Red/Orange/Yellow/Green/Blue/Purple/Pink, Neon, Cyberpunk, Forest, Sunset, Retro, Slate, Ocean, Midnight, Espresso, Rose, Emerald, Amethyst, and all color variations).
  - Guaranteed high contrast, clean typography, neat element spacing, and appropriate color accents across both **Combo Checker** and **Words to Work on**.
  - In `static/css/tools.css` and `static/css/style.css`, added explicit theme overrides for the unified panel card, dropdown menus, text inputs, action buttons, tab navigation, result cards, and word list items.

### 18. Multiplayer Subanagrams Mobile Button Restructuring & Platform-Wide Clear Removal
- **Clear Button Removal Across All Platforms**:
  - Completely removed / hidden `#clear-word-btn` (`display: none !important;`) on all devices (mobile, tablet, desktop, laptop) in multiplayer Subanagrams rooms.
- **Mobile Button Reordering & Hierarchy**:
  - Traded places between the **Enter** and **Shuffle** buttons on mobile devices.
  - Positioned **Enter** (`#submit-word-btn`) **above** the input row, stretching across the entire width of the container (`width: 100% !important; flex: 1 1 100% !important;`) with an enlarged `48px` height, bold `1.05rem` font, and vivid emerald gradient (`linear-gradient(135deg, #10b981 0%, #059669 100%)`), making it the largest, most prominent, and easiest button to tap.
  - Relocated **Shuffle** (`#shuffle-board-btn`) into the row alongside `#word-input` (`height: 42px`, `padding: 0 16px`, blue gradient).
  - Attached debounced touch and click delegators (`window.submitSubanagramsInput(e)`) for zero-latency tap response.
- **Desktop Subanagrams Layout**:
  - Aligned `#word-input` (order 1, flex 1), `#submit-word-btn` (order 2, 44px), and `#shuffle-board-btn` (order 3, 44px) seamlessly side-by-side in a single unified horizontal row.
- **Rules & Documentation Updates**:
  - Harmonized the FAQ item (`#faq-subanagrams`) and Lobby Guide card to document the new mobile layout and reflect the retirement of the Clear button.
  - Bumped cache query versions in `templates/index.html` (`play.css?v=1791150000`, `play.js?v=1791150000`).

---

## Verification & Health Check

- **Automated Health Check**: `curl -s https://morpheme.games/api/lobby-stats` returns `{"stats":{}}` with HTTP 200 OK.
- **Validation Endpoints Verified Live**:
  - `WANGYANITE` & `WANGYANITES`: `is_valid: true`, `lists: ["AW"]`
  - `ABELLOEMRINGERITE` & `ABELLOEMRINGERITES`: `is_valid: true`, `lists: ["AW"]`
  - `TESTTEST` & `TESTTESTS`: `is_valid: false`
- **PM2 Process**: `morpheme` online (fork mode, memory stable ~185MB initial / ~800MB–1.4GB fully warmed up).
- **Theme Audit**: Verified legible contrast and clean borders across light, dark, and saturated color palettes.
- **Mobile Fullscreen Invariant**: Continuous fullscreen preserved across navigation, with cleanly targeted keyboard exit safeguards for text inputs to eliminate mobile black screens.
