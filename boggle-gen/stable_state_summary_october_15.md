# Stable State Summary — October 15, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_15
```
*(Also tagged as `stable-2026-10-15`)*

## Latest Commit ID
```
6a63a5a62d3ede042a4de1e2a0c3f6f24e58276c
```

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| **localhost**      | ✅ Synchronized | Commit `6a63a5a6`, working tree clean, synced with origin/main |
| **GitHub (main)**  | ✅ Synchronized | Pushed to `origin/main` & tags updated on GitHub (`6a63a5a6`) |
| **morpheme.games** | ✅ Synchronized | Deployed to 132.148.72.249 (Commit `6a63a5a6`, PM2 online, verified live HTTP/2 200 OK) |
| **App / Client**   | ✅ Synchronized | Verified on desktop, laptop, tablet, and mobile layouts |

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
  - In multiplayer Subanagrams (`body.is-subanagrams`), displayed the emerald **Enter** button (`#submit-word-btn`) and red **Clear** button (`#clear-word-btn`) with `display: inline-flex !important;` on both desktop and mobile.
- **Mobile Subanagrams Input Textbox, Clear, and Enter Buttons Restored**:
  - Overrode the mobile hiding rule inside `@media (max-width: 992px)` for Subanagrams (`body.is-subanagrams .word-input-section`), displaying `#word-input` with centered uppercase letters (`font-size: 1.1rem; letter-spacing: 2px;`) to display letters in real time as tiles are tapped on mobile devices, with Enter and Clear buttons alongside.
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

### 4. Definition Synchronization for ABELLOEMRINGERITE & ABELLOEMRINGERITES
- Permanently synchronized definitions and dictionary entries for `ABELLOEMRINGERITE` and `ABELLOEMRINGERITES` (repeating the mineral definition per AGENTS.md Rule 1).

### 5. Removal of Erroneous Scratch Word TESTTEST
- Removed erroneous test word `TESTTEST` and its plural `TESTTESTS` from all local and remote dictionary text files and database tables.

### 6. Subanagrams Intention Documentation Across App
- In `templates/index.html`:
  - Updated Subanagrams Guide / "What am I looking at" modal (`#sub-help-modal`) to explicitly state the intention is to warm up for rated play.
  - In `#faq-subanagrams`, stated that the primary intention of multiplayer Subanagrams is to warm up for rated play.
  - In Lobby Guide Modal (`#lobby-guide-modal`), highlighted warm-up intention.
- **Forum Mobile Original Post Response Badge (`static/js/forum.js`)**:
  - In `showReplyMobileModal(postNumber)`:
    - Replaced the text-based user count label with thread-consistent badge: `💬 ${respCount}`.
    - Added interactive click listener so tapping it opens responses list modal.

### 7. Fix for Temporary Duplicate Word Row Under Words on Submission
- Deduplicated optimistic submission handling in `static/js/play.js` inside `submitWord()`, ensuring server response updates the existing row in place without duplicating or flickering.

### 8. Combo Checker: Non-Plural / Non-Inflection MP Filtering & Interactive MP Word Explorer
- **Non-Plural & Non-Inflection Filtering (`is_plural_or_inflection`)**:
  - Implemented inflection filtering in `app.py`: excludes simple and irregular plural suffixes (`+S`, `+ES`, `+IES`), verb and adjective inflections (`+D`, `+ED`, `+ER`, `+EST`, `+ING`), vowel elision (e.g. drop `E` before `-ED`, `-ING`, `-ER`, `-EST`), and consonant doubling (e.g., `HOP` -> `HOPPED`, `HOPPING`).
  - Example: For input word `WASTRIE` at 1MP, plural `WASTRIES` is excluded, while true Morpheme distance connections such as `WASTRIFE` are included.
- **Dual-Table MP Word Explorer (`#combo-explorer-section`)**:
  - Placed cleanly in Tools -> Combo Checker between the MP horizontal container and the LIC container.
  - **Independent Dictionary Dropdown (`#combo-explorer-dict`)**:
    - Users can select which dictionary to explore independently from the main Combo Checker input:
      `NWL`, `CSW`, `AW`, `All`, `CSW Only`, `NWL Uniques`, `New NWL Words`, `New CSW Words`, `New AW Words`, and `All New Words`.
  - **Word Length Dropdown**: 3 to 10 letters (defaults to 7 Letters).
  - **MP Value Dropdown**: 0 MP, 1 MP, 2 MP, 3 MP (defaults to 1 MP).
  - **Random Order Button**: Enables displaying words and their respective MPs in randomized order rather than alphabetical order.
  - **A–Z Button**: Restores standard ordering.
  - **4-Letter Word Exclusion**: Specifically for the MP Word Explorer, 4-letter words are not captured in results (e.g. for `LINKBOY`, `LINK` is not captured as a 0MP word, so `LINKBOY` is not in the 0MP input words list and `LINK` does not appear as an MP connection).
  - **Output Table Length-Descending Sorting**: In the MP Word Explorer output table, words are sorted by length, largest first (`-len(w)`, with alphabetical secondary sort). E.g. for `WASTRIE` at 1MP, 8-letter connections `WASTERIE` and `WASTRIFE` appear at the very top.
  - **Standard Combo Checker Preserved**: The main Combo Checker search tool retains standard dictionary and procedure rules.
  - **Complete Native Word List Rendering**:
    - Replaced DOM chunking with full string concatenation and container event delegation (`#combo-explorer-input-list`).
    - Eliminates list truncation at 150 words (e.g. `ACKNOWN`), rendering all 27,678+ words instantaneously (~3ms) with smooth continuous native scrolling from A to Z.
  - **Dictionary Dropdown Labeling**:
    - Renamed `"New All Words"` option to `"All New Words"`.
  - **Input Words Table**: Displays all words of the selected length that actually have one or more non-plural/non-inflection MP connections. Includes instant client-side text filtering (`#combo-explorer-filter`) and instantaneous list rendering.
  - **Response Table**: Displays all non-plural, non-inflection words at that MP value away from the selected input word.
  - **Definition Lookup**: Each matching word in the response table is an interactive link that invokes `window.lookupWord(word, event)` for instant lexicographical definition popups.
  - **Mobile Fullscreen Invariant**: Preserved continuously across all dropdown selections, filters, and word selections without triggering fullscreen exits.

### 9. Explorer Length Sorting, Title Grey Background Removal, and Mobile Input Visibility
- **Length-Strict Sorting (e.g. `SCHWAS` vs `WAGES` for `SCHWAGS`)**:
  - In `static/js/tools.js` and `app.py`, created `_sortMatchesByLength(matches, isRandom)` ensuring words are categorized and sorted by length descending (`b.length - a.length`) as the absolute primary sorting tier.
  - 6-letter words (such as `SCHWAS`) always appear before 5-letter words (`WAGES`, `SWAGS`, `WANGS`) regardless of alphabetical order or randomized order. When Random mode is enabled, randomization shuffles strictly within identical length tiers.
  - Fixed button ID listener binding in `static/js/tools.js` to ensure the A–Z toggle button reliably resets sorting across all platforms.
- **Root Backdrop Black Flash Prevention**:
  - In `static/css/play.css`, set `html, html:fullscreen, html:-webkit-full-screen, body, ::backdrop { background: var(--bg-primary, #0f172a) !important; }`.
  - Enforced `font-size: 16px !important;` on `#combo-input` and `#combo-explorer-filter` on mobile viewports (`max-width: 900px`) to eliminate iOS/Android auto-zoom window viewport repainting flashes.
- **Mobile Keyboard Visibility & Smooth Scroll**:
  - In `static/js/tools.js` and `static/css/play.css`, bound focus handlers to `#combo-input` and `#combo-explorer-filter` to dynamically append `.keyboard-open` with `padding-bottom: 350px !important;` to the scrollable container.
  - Automatically invokes smooth scroll centering (`scrollIntoView({ behavior: 'smooth', block: 'center' })`) on focus and visualViewport resize, keeping input boxes visible directly above the software keyboard while strictly adhering to the continuous fullscreen invariant.
- **Removal of Light Grey Rectangular Backgrounds Around Combo Checker Titles**:
  - In `static/css/play.css`, removed the background styling on `.result-section h3` and explicitly enforced `#tool-combo .result-section h3, .combo-results-container .result-section h3, #combo-explorer-section h3 { background: transparent !important; }` across all desktop, laptop, tablet, and mobile platforms.

---

## Verification & Health Check

- **Automated Health Check**: `curl -I https://morpheme.games/` returns `HTTP/1.1 200 OK` / `HTTP/2 200`.
- **Live Combo Explorer Endpoint Verified Across All Dictionaries**:
  - `NWL`, `CSW`, `AW`, `All`, `CSW Only`, `NWL Uniques`, `New NWL Words`, `New CSW Words`, `New AW Words`, `All New Words`: all return `200 OK`.
- **Live Parameter Verification**:
  - `POST https://morpheme.games/api/tools/combo/explorer` with `{"dictionary":"CSW","length":7,"mp":1,"word":"WASTRIE"}`:
    - Status: `200 OK`
    - Count: 37 matches
    - 4-letter words present: `false`
    - Matches sorted by length largest first: `true` (Top matches: `WASTERIE` [8], `WASTRIFE` [8], `ASTRIDE` [7], `STRIAE` [6]...)
    - `WASTRIES` present in matches: `false` (successfully filtered as plural)
  - `POST https://morpheme.games/api/tools/combo/explorer` with `{"dictionary":"CSW","length":7,"mp":0,"word":"LINKBOY"}`:
    - Count: 0 (`LINK` is not captured as 0MP in Explorer)
  - `POST https://morpheme.games/api/tools/combo` with `{"dictionary":"CSW","search_term":"LINKBOY"}`:
    - Main Combo Checker retains `LINK` under 0MP according to standard rules.
- **PM2 Process**: `morpheme` online (fork mode, memory stable ~800MB–1.4GB fully warmed up).
- **Mobile Fullscreen Invariant**: Preserved continuously across all pages, navigation, modals, and tabs.
