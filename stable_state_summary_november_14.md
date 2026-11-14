# Stable State Summary — November 14, 2026 (Start Over Point)

## Start Over Point Tag & Commit ID
```
Tag:       START_OVER_POINT_NOVEMBER_14
Commit ID: 9c9c42ac
Short ID:  9c9c42ac
```
*(Also tagged as `START_OVER_POINT_TODAY`, `STABLE_STATE_NOVEMBER_14`, `STABLE_STATE_TODAY`, `stable-2026-11-14`, `start-over`, `latest-stable-state`, and `current-stable-state`)*

## Synchronization Status

| Environment | Status | Verification Details |
|:---|:---|:---|
| **Localhost** | ✅ Synchronized | Working tree clean on `main`, zero uncommitted changes, verified HTTP 200 on `http://localhost:5001/` |
| **GitHub (`origin/main`)** | ✅ Synchronized | Pushed to `origin/main` at `9c9c42ac` & tagged with `START_OVER_POINT_NOVEMBER_14` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | Reset hard to `origin/main` at `9c9c42ac`, PM2 online (`morpheme`), verified HTTP 200 and letter length select verified via headless Chrome on `https://morpheme.games/` |
| **App (Mobile WebView)** | ✅ Synchronized | Mobile WebView loads live production environment with full asset caching (Build `33213`) |

---

## Session Features & Fixes (November 14, 2026)

### 1. Consistent Gateway Presentation ("ENTER LOBBY" Screen on Every Visit)
- **Elimination of Tab Session Bypass (`templates/index.html`, `static/js/app.js`):**
  - Removed `sessionStorage.getItem('morpheme_gateway_passed')` bypass checks that previously skipped the gateway screen on reloads or re-visits in the same tab.
  - The **ENTER LOBBY** screen now reliably and consistently appears whenever the site is opened, refreshed, or visited from an external URL (such as navigating to `google.ca` and then returning to `morpheme.games`).
  - Single-page in-app navigation across top tabs (Lobby, Play, Tools, Mods, Leaderboard, Settings, Profile, Forum) continues smoothly without re-prompting the gateway.
  - Active game sessions (`#page-play`) remain protected so ongoing matches resume seamlessly on refresh.
- **Lobby Initialization & Top Nav Clickability (`static/js/app.js`, `static/js/lobby.js`):**
  - Resolved `SyntaxError: Identifier 'activePageEl' has already been declared` in `app.js` and `ReferenceError: Cannot access 'isFetchingLobbyStats' before initialization` in `lobby.js`.
  - Restored full top menu button responsiveness (`setupNavigation()`) and proper Lobby grid rendering across all devices.

### 2. Reverse Dictionary Tool ("Key Word in Definition" / Find Words by Meaning)
- **Tool Architecture & Interface (`templates/index.html`, `static/js/tools.js`, `static/css/play.css`):**
  - Integrated under the Random Word tab in the Tools menu (`#tool-reverse-dictionary`).
  - Search input with clear button (`✕`), search button (`Search`), and keyboard `Enter` trigger.
  - Dropdown options:
    - **Search Mode**: `Similar to` (default) or `Exact` (inflections).
    - **Dictionaries**: `All Dictionaries` (default), `NWL`, `CSW`, `Added Words (AW)`.
    - **Letter Length**: `All Lengths` (default), or specific letter length from 2 to 28 letters with dynamic word counts.
  - Dedicated custom scrollbar (`.reverse-dict-table-scroll`) strictly containing all table rows with **zero outer page scrolling** (`no-outer-scroll`), fitting 100% within the viewport across desktop and mobile screens.

- **Output Table Selection by Letter Length (`templates/index.html`, `static/js/tools.js`, `app.py`, `static/css/play.css`):**
  - **Letter Length Dropdown Selector (`#reverse-dict-length`)**: Added directly in the controls actions row alongside Match Mode and Dictionary selectors. Allows users to filter table output by exact letter length or view all lengths.
  - **Dynamic Length Aggregation with Counts**: Upon search completion, the dropdown dynamically populates with the available letter lengths and their exact word counts (e.g. `All Lengths (9,151)`, `3 Letters (38)`, `4 Letters (213)`, `5 Letters (545)`).
  - **Instantaneous Zero-Latency Filtering**: Selecting any letter length from the dropdown instantly filters the results table in 0ms directly from client memory without requiring a server re-query.
  - **Interactive Length Badges in Table**: Word length badges in the table are clickable with interactive hover effects; clicking a length badge (e.g. `5`) automatically selects that length in the dropdown and filters the table. Clicking again toggles back to All Lengths.
  - **Dynamic Status Counter & Numbering**: Updates the status text (e.g. `Found 545 words (5 letters) for "bird" (9,151 total)`) and recalculates sequential row numbers (1, 2, 3...) for the filtered view.
  - **Backend API Support**: Enhanced `/api/tools/reverse-dictionary` to accept `length` / `len` parameter, skipping non-matching lengths early before computing token sets.

- **High-Performance Semantic & Taxonomic Engine (`app.py`):**
  - Pre-warmed cache over 827,000+ definitions in `DEFINITIONS_CACHE`.
  - Recursive taxonomic hyponym closure up to depth 6 (`closure(lambda ss: ss.hyponyms(), depth=6)`) and cross-synset sister linking (e.g. `bird` $\leftrightarrow$ `fowl`).
  - Broad category queries expand to thousands of families, sub-species, and compound terms (`peacock`, `seabird`, `waterfowl`, `shorebird`, `songbird`, `chionidid`, `sheathbill`, `pajock`, `bantam`, etc.).
  - Fast token set intersection (`d_tokens & stem_set`) via string translation whenever `stem_set > 30`, allowing 4,000+ expanded terms to scan across 827,000+ cached definitions in ~1.1s.
  - Removed arbitrary limit caps (`limit = 500 / 1000`); all matching definitions are returned and displayed without table truncation (e.g. 9,151 matches for `bird`).

- **Accurate & Uncapped Definition Highlighting (`app.py`, `static/js/tools.js`):**
  - Eliminated the artificial `[:250]` slice on `matched_terms`.
  - Dynamically tracks all tokens that matched definitions during search (`used_highlight_terms.update(inter)`), returning them alongside direct query terms, morphological inflections, and concept expansions.
  - Resolved missing highlights: terms like `birds` (in `RHABDORNIS`) and `ovenbird` / `ovenbirds` (in `RAYADITO`) now highlight yellow with `<mark class="reverse-dict-highlight">` alongside `passerine`.
  - Automatic client-side plural and inflection generation (`-s`, `-es`, singular forms) for direct query words.
  - Pre-compiled single-pass regex (`window.buildReverseDictHighlightRegex`) in `tools.js`, compiling the highlight pattern once per search rather than 9,000+ times inside the rendering loop, delivering near-instant rendering.

- **Strict Monotonic Word Length Sorting (`app.py`):**
  - Updated sort comparator to `matches.sort(key=lambda x: (x['len'], -x['score'], x['word']))`.
  - Word length strictly and monotonically increases from 2 to 24 letters across all 9,151 results with zero resets.
  - Eliminates the previous score-boundary restart where `ILIOTROCHANTERICUSES` (20 letters, rank 2,586) was followed by `ALK` (3 letters, rank 2,587).
  - Positions under `bird`: `ALK` (3 letters) is at Rank #16, `BIRD` (4 letters) is at Rank #47, `BANTAM` (6 letters) is at Rank #1,029, and `ILIOTROCHANTERICUSES` (20 letters) is at Rank #9,136.

- **Mobile Viewport Optimization & Dictionary Stacking (`static/css/play.css`, `templates/index.html`):**
  - **Visible Description Badge**: Unhidden on mobile screens (`display: block !important`) with compact rounded badge styling (`max-height: 44px; padding: 4px 8px; font-size: 0.74rem`).
  - **Auto-Focus Mobile Protection**: Gated `inp.focus()` on tab switch strictly to non-mobile devices (`!isMobile`), preventing the mobile virtual keyboard from opening unprompted and scrolling the header description out of view.
  - **Reduced Top/Bottom Spacing**: Resolved `flex: 1 1 280px` wrapper expansion in column layout, collapsing the search input and controls panel from 330px down to **88px**.
  - **Expanded Table Length**: Increased results table card height from ~217px to **~457px+** (`flex: 1 1 0px`), more than doubling visible rows on mobile.
  - **Vertical Dictionary Badges**: `.reverse-dict-badges-wrapper` configured with `flex-direction: column` on mobile, placing `NWL` directly above `CSW`.
  - **Compressed Dictionary Column**: Narrowed mobile dictionary column from 72px down to **38px** (header displays `Dict` on mobile and `Dictionaries` on desktop).
  - **Expanded Word Column**: Widened `.col-word` from 96px up to **124px**, giving words up to 12 letters and their length pills ample room without truncation.
  - **Status Bar Cleanliness**: Removed the `"Similar Concepts & Synonyms"` and `"All Dictionaries"` meta badges to the right of the counter, keeping the status bar clean.

---

### 2. Previously Established Foundation (November 13 & Earlier)
- **Board & Spinner Exclusivity**:
  - Bonus Letter points (`+3`) and badge (`BL`) award strictly when the format is `Bonus Letter`.
  - Normal format boards never award bonus points. Dual tiles (`/`) and bonus cells (`*`) are mutually exclusive.
- **Truthful Word Count Ranges**:
  - `SpinnerSet.get_factchecked_wc_range` ensures displayed word count ranges truthfully reflect solved words.
- **Zero-Change Equilibrium Method (Unrated Placement)**:
  - 10-round placement progression calculating $R_{\text{round}} = \text{opp\_rating} \cdot (\text{user\_score} / \text{opp\_score})$ for zero-change equilibrium, concluding on game 10 with a 2300 cap.
- **Trie Solver Acceleration**:
  - Removed artificial delays from solver cell loops; boards solve in ~20ms with zero timeout word drops.
- **Synchronized Submission Flash Timing**:
  - Valid (blue) and invalid (red) submissions flash simultaneously at `0ms` for `150ms` across all formats.

---

## Verification & Integrity Check
- Automated test suite passed:
  - Reverse Dictionary queries (`snake`, `fishes`, `mean old woman`, `derogatory`): 200 OK across both Exact and Similar modes.
  - Recursive hyponym queries (`bird`, `fowl`): verified `PAJOCK`, `SHEATHBILL`, and `CHIONIDID` present in full 9,151 results with no table truncation.
  - Highlight accuracy: verified `passerine` and `birds` highlighted in `RHABDORNIS`, and `ovenbirds` highlighted in `RAYADITO`.
  - Monotonic sorting: verified word lengths are strictly non-decreasing across all 9,151 rows (`ALK` at #16, `BANTAM` at #1,029, `ILIOTROCHANTERICUSES` at #9,136).
  - Rating placement contracts (`scratch/test_unrated_placement.py`): 3/3 tests OK.
- Production deployment:
  - Pushed to GitHub `origin/main`.
  - Force-updated tags `START_OVER_POINT_NOVEMBER_14`, `START_OVER_POINT_TODAY`, `STABLE_STATE_NOVEMBER_14`, `STABLE_STATE_TODAY`, `stable-2026-11-14`, `start-over`, `latest-stable-state`, and `current-stable-state`.
  - Pulled to `132.148.72.249`, PM2 restarted, live HTTP 200 OK verified on `https://morpheme.games/`.
  - Local server running cleanly on `http://localhost:5001/`.
  - Build version bumped to `33211` (cache-buster `1794157000`).
