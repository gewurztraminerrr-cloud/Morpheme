# Stable State Summary — October 15, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_15
```
*(Also tagged as `stable-2026-10-15`)*

## Latest Commit ID
```
65f63c084fe9319949d644b14e44c961f0dc9b6c
```

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| **localhost**      | ✅ Synchronized | Commit `65f63c08`, working tree clean, synced with origin/main |
| **GitHub (main)**  | ✅ Synchronized | Pushed to `origin/main` & tags updated on GitHub |
| **morpheme.games** | ✅ Synchronized | Deployed to 132.148.72.249 (PM2 online, verified live HTTP/2 200 OK) |
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

### 11. Multiplayer Subanagrams Board "Shuffle" Button
- **User Request**: Add a button next to the textbox in multiplayer Subanagrams that rearranges the letters in random order reading "Shuffle", completely separate and different from Rotate and Transpose in grid rooms.
- **HTML Layout**:
  - In `templates/index.html`, added `<button id="shuffle-board-btn" class="shuffle-btn timer-btn" type="button">Shuffle</button>` inside `.word-input-section` directly alongside `#word-input`, `#submit-word-btn` (Enter), and `#clear-word-btn` (Clear).
- **CSS Styling & Responsiveness**:
  - In `static/css/play.css`, styled `#shuffle-board-btn` with a vivid blue gradient (`linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)`), bold white font, rounded corners (`border-radius: 8px`), and interactive hover shadow.
  - Hidden by default in standard grid rooms (`display: none !important;`) and only displayed when `body.is-subanagrams` is active (`display: inline-flex !important;`).
  - **Desktop Layout**: Widened the Subanagrams input row container to `680px`, keeping `#word-input`, `#submit-word-btn` (Enter), `#clear-word-btn` (Clear), and `#shuffle-board-btn` (Shuffle) side-by-side in a single row (`flex-wrap: nowrap !important;`).
  - **Mobile Layout**: Configured `flex-wrap: wrap !important;` on `.word-input-section` with `#word-input`, Enter, and Clear on the top row, and relocated `#shuffle-board-btn` to its own dedicated row directly beneath them, stretching horizontally across the full width (`width: 100% !important; flex: 1 1 100% !important;`) matching the combined width of the textbox and the two action buttons.
- **Client Randomization & State Preservation**:
  - In `static/js/play.js`, implemented `window.shuffleSubanagramsBoard()`:
    - Extracts letters and runs a Fisher-Yates shuffle with a guarantee of producing a distinct permutation whenever at least two distinct letters exist.
    - Saves the custom arrangement in `window.subanagramsCustomBoard` alongside the sorted multiset key `window.subanagramsBoardKey`.
    - Updates `window.lastGameState.board = [shuffled]` and triggers `renderBoard([shuffled])` immediately without layout shifts or fullscreen interruptions.
    - Preserves the shuffled order across background server countdown ticks/heartbeats during the round.
    - Automatically remaps `mouseState.selectedPath` and `mouseState.visitedCells` to the new tile coordinates if letters were already entered in `#word-input`, re-triggering highlight pathfinding.
    - Resets cleanly when a new round starts, on round intermission, or upon room exit.
  - Bumped `play.css` cache buster to `v=1791130460` and `play.js` cache buster to `v=1791130450` in `templates/index.html`.

### 12. Subanagrams Concept & Mobile Tile Tapping Documentation (FAQ & Lobby Guide Modal)
- **User Request**:
  - In FAQ, state where applicable that for mobile devices, users must individually tap on each letter to form words in multiplayer Subanagrams.
  - Explain the concept of Subanagrams in the same place.
  - Explain multiplayer Subanagrams in the "WHAT AM I LOOKING AT" popup that explains the lobby features.
- **FAQ Updates (`templates/index.html`)**:
  - In `#faq-strategies`, expanded the list of game arenas from three to four by adding **Subanagrams** with its tactical focus on anagram agility, rack permutation, and vocabulary breadth under a 2-minute timer.
  - Added a dedicated FAQ question `#faq-subanagrams`: *"What is Subanagrams, what is the concept behind it, and how do I enter words on mobile devices?"*:
    - Clearly defined the concept: dealing a single horizontal row of 7, 8, or 9 letters where players race to find all valid dictionary sub-words (subanagrams) ranging from 3 letters up to the full rack length.
    - Highlighted core mechanics: non-adjacent positional freedom (letters can be selected from anywhere on the rack in any order), tile frequency limit (each tile can only be used as many times as dealt), and real-time multiplayer competition on the identical rack.
    - Detailed mobile input: on mobile devices, **users must individually tap each letter tile** to spell out words in the `#word-input` textbox (swiping across tiles is not applicable), followed by tapping the emerald **Enter** button to submit, red **Clear** to reset, and using the blue **Shuffle** button directly beneath to permute the rack.
    - Detailed desktop input: physical keyboard typing (<kbd>Enter</kbd> to submit) or mouse clicking.
  - In `#faq-icons-flags`, updated the mobile device indicator to reflect both swiping on 2D grid boards and individually tapping letter tiles in Subanagrams.
- **Lobby Guide Modal ("WHAT AM I LOOKING AT", `#lobby-guide-modal`)**:
  - Expanded the Subanagrams card to **Multiplayer Subanagrams**:
    - Explains the multiplayer competition, 2-minute round duration, 7–9 letter rack length, and anagram discovery mechanics without grid adjacency constraints.
    - Highlights the mobile device rule: **users must individually tap each letter tile** to form words, submit via Enter, reset via Clear, and utilize the full-width blue Shuffle button.
    - Clarifies the direct entry `Start [N]` button action into live synchronized multiplayer rounds.

### 13. Subanagrams Warm-Up Intention & Forum Mobile Original Post Response Badge
- **User Requests**:
  1. *If a mobile device selects a responding post number to see the original post, reword "2 users responded" to "[speech bubble icon] [number of responses]" in the original message in the same way as you have done from the viewpoint of being in the thread.*
  2. *In all places where you explain multiplayer subanagram ("WHAT AM I LOOKING AT", and FAQ), also include that the intention of it is to warm up for rated play.*
- **Subanagrams Warm-Up Intention (`templates/index.html`)**:
  - In `#faq-strategies`, noted that Subanagrams' intention is to serve as an ideal warm-up for rated play, sharpening anagram agility, pattern recognition, and vocabulary breadth through quick permutations without affecting your rating.
  - In `#faq-subanagrams`, stated that the primary intention of multiplayer Subanagrams is to warm up for rated play, providing a focused, rapid arena to exercise vocabulary recall and anagramming speed before jumping into competitive ladder matches. Added a dedicated rule bullet for "Warm-Up for Rated Play".
  - In the Lobby Guide Modal (`#lobby-guide-modal`), updated the Multiplayer Subanagrams card description and availability note to highlight that its specific intention is serving as a fast-paced warm-up for rated play.
- **Forum Mobile Original Post Response Badge (`static/js/forum.js`)**:
  - In `showReplyMobileModal(postNumber)`:
    - Replaced the text-based user count label (`${userCount} users responded`) with the thread-consistent badge: `💬 ${respCount}` where `respCount = resps.length`.
    - If `respCount > 0`, renders `<span class="reply-preview-resp-count" role="button" tabindex="0">💬 ${respCount}</span>`. If 0, omits the badge, matching the thread view.
    - Added an interactive click listener on the preview response badge so tapping it immediately opens the responses list modal (`showResponsesPopup(postNumber)`).
    - Updated desktop hover preview `showReplyHoverPreview` to use the same `💬 ${respCount}` badge.
  - Bumped `forum.js` cache buster in `templates/index.html` to `v=1791130520`.

### 14. Fix for Temporary Duplicate Word Row Under Words on Submission
- **Bug / Symptoms**:
  - Upon entering a valid word, two identical rows of that word briefly appeared under the "Words" panel.
  - One of them (the unstyled duplicate inserted without checkmarks or click handlers) would disappear a few moments later on the subsequent state poll/refresh, leaving only the single correct find with its checkmark and score breakdown.
- **Root Cause**:
  - In `static/js/play.js` inside `submitWord()`, instant local validation optimistically appended the word to `window._localSubmittedWordsList` and called `updateGameState()`, which immediately rendered the word into `#submitted-words-list` with full formatting (`✓ WORD`, points, click listener).
  - When the HTTP submission network response (`POST /room/${roomId}/submit_word`) subsequently completed, the success handler pushed the word into `window._localSubmittedWordsList` a second time, and unconditionally ran `listEl.insertAdjacentHTML('afterbegin', html)` to insert another row at the top of `#submitted-words-list`.
  - This resulted in two simultaneous rows for the same word. The next state poll or density update re-rendered `#submitted-words-list` from the deduplicated list, causing the duplicate top row to abruptly vanish.
- **Resolution**:
  - **Deduplication in Local State**: In `submitWord()`, checked `window._localSubmittedWordsList` before pushing; if the word already exists from the optimistic check, its points and score details are updated in place with authoritative server values rather than appending a duplicate entry.
  - **DOM Check Prior to Insertion**: In both standard/accumulative/split and FCFS modes, verified whether `#submitted-words-list` already contains a `.word-item` or `.feed-item` for that word. If already rendered by the optimistic check, DOM re-insertion is skipped and points are updated in place on the existing element. If not yet in DOM, prepends with full formatting (`✓` checkmark, definition click handler).
  - Updated `currentState.players` submission list in place to ensure instant local state consistency.
  - Bumped `play.js` cache buster in `templates/index.html` to `v=1791130540`.

---

## Verification & Health Check

- **Automated Health Check**: `curl -I https://morpheme.games/` returns `HTTP/1.1 200 OK` / `HTTP/2 200`.
- **Validation Endpoints Verified Live**:
  - `WANGYANITE` & `WANGYANITES`: `is_valid: true`, `lists: ["AW"]`
  - `ABELLOEMRINGERITE` & `ABELLOEMRINGERITES`: `is_valid: true`, `lists: ["AW"]`
  - `TESTTEST` & `TESTTESTS`: `is_valid: false`
- **PM2 Process**: `morpheme` online (fork mode, memory stable ~800MB–1.4GB fully warmed up).
- **Mobile Fullscreen Invariant**: Preserved continuously across all pages, navigation, modals, and tabs.
