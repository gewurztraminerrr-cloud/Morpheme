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

---

## Verification & Health Check

- **Automated Health Check**: `curl -I https://morpheme.games/` returns `HTTP/1.1 200 OK` / `HTTP/2 200`.
- **Validation Endpoints Verified Live**:
  - `WANGYANITE` & `WANGYANITES`: `is_valid: true`, `lists: ["AW"]`
  - `ABELLOEMRINGERITE` & `ABELLOEMRINGERITES`: `is_valid: true`, `lists: ["AW"]`
  - `TESTTEST` & `TESTTESTS`: `is_valid: false`
- **PM2 Process**: `morpheme` online (fork mode, memory stable ~800MB–1.4GB fully warmed up).
- **Mobile Fullscreen Invariant**: Preserved continuously across all pages, navigation, modals, and tabs.
