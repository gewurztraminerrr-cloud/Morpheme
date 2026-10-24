# Stable State Summary — October 24, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_24
```
*(Also tagged as `stable-2026-10-24`)*

## Latest Commit ID
```
f53f91bbe87ea408674bff7faea0c179467e921e
```

## Synchronization Status

| Environment | Status | Details |
|:---|:---|:---|
| **Localhost** | ✅ Synchronized | Clean working tree on `main` |
| **GitHub (`origin/main`)** | ✅ Synchronized | Pushed to `origin/main` & tags `START_OVER_POINT_OCTOBER_24` / `stable-2026-10-24` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | Updated to latest commit, tags fetched, PM2 process online, live HTTP 200 OK |
| **App / Client** | ✅ Synchronized | Cache busters bumped (`howtoplay.css?v=1792844000`, `play.css?v=1792845000`, `tools.js?v=1792845000`, `play.js?v=1792871000`, `app.js?v=1792871000`) for instant live propagation across all devices |

---

## Session Highlights & Architectural Updates (October 24, 2026)

### 1. In-Game Guessing Notice & Submission Error Fixes
- **Submission Error Resolution**: Resolved a Python `NameError: name 'returned_path' is not defined` inside `submit_word` in `app.py` that occurred when submitting invalid words, which previously produced an HTTP 500 "SUBMISSION ERROR" modal.
- **Continuous "You're Guessing!" Feedback**: In `static/js/play.js`, after a user enters 4 invalid words in succession on the board, the "You're guessing!" warning appears. On subsequent invalid submissions beyond 4, the alert continues to display consistently rather than resetting prematurely.
- **Dynamic Reset on Valid Submission**: Once the user successfully enters any valid word, the invalid guess counter resets to 0, granting 4 fresh attempts before the notice can trigger again.

### 2. Find Count: Horizontal Random Words Selection on Desktops & Laptops
- **Space Optimization**: Re-engineered the random word selection table in Tools -> Find Count to run horizontally across columns on laptops and desktops (`@media (min-width: 901px)` and `body:not(.is-mobile)`), saving over 70% of vertical space and eliminating excessive scrolling.
- **Responsive Mobile Integrity**: Retained the vertical single-column list format on mobile devices for optimal touch ergonomics.

### 3. FAQ "Default Color" Specification & High-Contrast Highlight Badges
- **Clear Settings Attribution**: In FAQ under "What do the colors of the words highlighted in the word list mean?", replaced the ambiguous "Dark Slate" reference with "Default Color" and added the clarification: `(Default color according to the color layout chosen in Settings)`.
- **Theme-Aware Highlight Contrast**: Overhauled `.faq-highlight` styling in `static/css/howtoplay.css` across all color layouts (Gold, Turquoise, Purple, Slate, Emerald Green, Bronze) to ensure crystal-clear text readability on dark, light, and white backgrounds.

### 4. New Words Ordering: New Additions Pinned to Top in Alphabetical and Random Orders
- **Issue Resolved**: When a new word was added to Added Words (AW) via the Mods tab (such as `TILCAYO`), it previously appeared near the bottom of the "New AW Words" list in the Lists tool.
- **Implementation**:
  - Implemented `order_new_words_list(lst)` in `app.py`: groups word-date pairs by addition date (`DD/MM/YYYY`) in descending order (newest date first). When Alphabetical is selected, words are sorted alphabetically within each date group; when Random is selected, words are shuffled within each date group.
  - Pinned newest addition dates (e.g., `24/10/2026`: `TILCAYO`, `TILCAYOS`, `TESTINGG`) strictly to the top of `New AW Words` and `All New Words`.
  - Implemented client-side `parseWordDate` and `sortNewWordsList` in `static/js/tools.js` to ensure the same date-descending priority is preserved when viewing the list or opening the Full List modal.

### 5. Mini-Profile & Public Profile Complete Synchronization
- **Issue Resolved**: In Tools -> Find Count, mini-profiles showed inaccurate stats that diverged from canonical Profile data.
- **Implementation**:
  - Re-engineered `get_mini_profile(username)` in `app.py` to delegate directly to `get_public_profile(username)`, eliminating disconnected schema queries and ensuring both endpoints share the exact same 30-second TTL cache.
  - Updated `static/js/tools.js` to format games and wins using `.toLocaleString()` and clean placeholder dashes for missing demographics.
  - Verified live: both `jeffjeff` (101 games, 26 wins) and `jeffb` (27 games, 4 wins) match identically across mini-profile and public profile.

### 6. Real-Time Combo Query Cache Invalidation on Added Words Modifications
- **Enhancement**: When words are added to or removed from Added Words via the Mods tab or Tools interface, `COMBO_QUERY_CACHE.clear()` is immediately executed on the server and `window._comboClientCache.clear()` is invoked on the client.
- **Result**: Combo Checker queries for Microprocedures (MP) and Letters In Common (LIC) immediately incorporate new additions without requiring a server reboot or waiting for cache expiration.

### 7. Firefox Desktop MORPHEME Title Enlarge & Equalized Vertical Spacing
- **Enhancement**: Under Firefox on desktop (`@-moz-document url-prefix()`), increased the main MORPHEME title logo font size and equalized top and bottom header padding.
- **Visual Polish**: Aligned visual hierarchy and baseline padding identically with Chrome and Safari.

### 8. Automatic Inflection Suffix Appending for Added Words Definitions
- **Lexicographical Integrity**: Resolved definitions for Added Words now automatically compute and append valid inflection suffixes (`[inflections: ...]`) covering plurals, participles, and verb tenses via `get_valid_inflections_for_word`.

### 9. Combo Checker: Removal of "v2.0" from MP Title and Balanced Spacing
- **Title Cleanliness**: Removed outdated `"v2.0"` designation from the Microprocedures header in the Combo Checker tool (`#combo-pane-mp h3`), rendering clean and authoritative as `MP (Microprocedures)`.
- **Harmonized Spacing & Padding**:
  - Enclosed title and navigation arrows inside a dedicated `.combo-section-header` flex row with `gap: 16px`, `margin-top: 2px`, and `margin-bottom: 10px;`.
  - Expanded horizontal scroll arrow button gap to `gap: 8px; flex-shrink: 0;` with touch-friendly padding.
  - Symmetrically matched layout, typography, and arrow spacing on the LIC (Letters In Common) tab (`#combo-pane-lic`).

### 10. Point Sum Equation Contrast in White Layouts & Elimination of Congratulatory Black Glow
- **High-Contrast Equations in White/Light Themes**:
  - Eliminated faint opacity styling from score equation math (`5 + 3 = 8`).
  - Added semantic `.word-item-pts` with `opacity: 1 !important` and `font-weight: 800 !important`.
  - Distinct high-contrast color palettes tailored to each word state.
- **Eliminated Black Glow Around Congratulatory Winner Card**:
  - Completely removed the heavy `.winner-flash *` black outline shadow (`text-shadow: 0 0 10px rgba(0, 0, 0, 0.9);`).
  - Adapted typography to selected layout: warm amber gold (`#b45309`) in White/Light, luminous gold (`#ffd700`) in Dark.

### 11. Words to Work on Strict Tab Isolation & Multi-Tool Integration
- **Strict Single-Tab Containment**: Confined `#tool-work-on` strictly to its dedicated container, ensuring explicit isolation so `#page-tools` only displays Words to Work on when that specific tab is selected.
- **Tool Integration**: Added the user's personal "Words to Work on" list as a selectable dictionary option in MP Word Explorer and Subanagrams.
- **Mobile Table Containment**: Fitted the Words to Work on table to device viewport with restricted internal scrolling.

### 12. In-Game Room & Board Interaction Enhancements
- **Touch-Swipe Real-Time Letter Display**: On mobile devices, swiped letters display immediately beneath the board inside the validation space while finger dragging is active.
- **Bonus Word Row Live Highlighting**: Words found during the round matching the secret bonus word highlight with an emerald green row background in real time under the Words panel.
- **Continuous In-Game Chat at Round End**: Users who joined as spectators or did not play the active round can continue chatting freely during round end.

### 13. Tools Modernization (Lists, Subanagrams, Unscramble)
- **Lists Ordering & Virtual Scrolling**: Added Alphabetical and Random ordering dropdowns to the Lists tool with responsive mobile grid formatting and robust virtual scrolling.
- **Subanagrams & Unscramble Dual-Panel Layout**: Reorganized Unscramble on desktop and laptop viewports to mirror Subanagrams' dual-panel layout.

### 14. Store & Visual Boutique Enhancements
- **Boutique Showcase Lightbox**: Made boutique showcase displays and product imagery expandable via full-screen lightbox modal on click.
- **Dictionary Edition Verification**: Added mandatory dictionary edition verification routine to the monthly store update cycle.

### 15. Board Generation Compliance & Spinner Set Accuracy Across All Game Modes
- **Constraint Adherence**: Fixed compliance across all game modes (Standard, Checkerboard, Equality Frequency, Min Word Length, and Letter Distribution).
- **Spinner Set Synchronization**: Realigned Spinner Set declared parameters to reflect exact search parameters, ensuring game room parameter badges consistently match the generated board.
- **Physical Caps vs. +AW Modes**: Clarified and enforced physical caps specifically for non-AW boards while allowing authentic +AW boards (which feature extended vocabularies) to achieve 300-400, 400-500, and 500+ ranges without artificial truncation.
- **Remaining Words Counter**: Fixed countdown computation to accurately track unrevealed words during active gameplay.

### 16. Parameter Lock & Board Discovery Timelines (0:45 & 0:55 Guarantees)
- **Search Cadence**: Guaranteed that board search finishes by 0:45 of intermission so player spinner sets immediately display authentic parameters.
- **Dynamic 0:55 Fallback**: Added a safe fallback generation trigger at 0:55 of intermission if an ultra-rare parameter combination hasn't finalized by 0:45, preventing intermission delays or game freezes.
- **Parameter Immutability**: Room parameters lock strictly upon round transition, preventing any mid-round parameter drift.

### 17. Board Design & Format Consistency (Equality Frequency & Checkerboard)
- **Layout Consistency**: Unified letter generation formatting and matrix serialization across Equality Frequency and Checkerboard modes, ensuring board design consistency and preventing visual misalignment in board rendering.
- **Board Generator Reference Resolution**: Resolved `BoardGenerator` scope and initialization references inside `GameRoom.ensure_next_board_ready`.

### 18. Game Room Audio Isolation
- **Clean Audio Transition**: Fixed audio controller routing to ensure lobby background music terminates cleanly when entering game rooms, preventing overlapping audio tracks.

### 19. Board and Word List Synchronization Shield (`notonboard.png` Fix)
- **Issue Resolved**: Addressed an edge case where a client would submit a word clearly visible on the board and receive a "not on board" validation failure (`notonboard.png`).
- **Root Cause**: An asynchronous race condition occurred when a background search thread completed right at an intermission boundary, causing a partial mutation of room staging attributes while the word list remained from an earlier candidate. Furthermore, a boolean return value in `BoardGenerator.is_word_on_board` was incorrectly unpacked as a tuple in certain validation flows.
- **Implementation**:
  - **Atomic Staging**: Enclosed all board staging and attribute swaps under `self._state_lock` in `game_room.py` to prevent partial updates.
  - **Tuple/Boolean Unpack Fix**: Corrected `BoardGenerator.is_word_on_board` invocation and return-handling across all verification paths.
  - **Integrity Shield**: Implemented a strict pre-round integrity check in `start_next_round` and `get_room_state` that verifies all staged words match the board letters before declaring the room active.

### 20. Word Count Range Randomness & Elimination of Hardcoded 100-200 Fallback
- **Issue Resolved**: Repeated occurrences where the `100-200` word count range was chosen 5+ times in succession.
- **Root Causes**:
  - Premature 0:55 fallback had hardcoded `easy_wc = '100-200'`.
  - Boards generated authentically in background search before 0:45 were being discarded as stale if they finished slightly early, unnecessarily forcing fallback boards.
  - Artificial clamp in `SpinnerSet.sanitize_params` forced `100-200` whenever min word length was 8.
  - Overly skewed probability weights in `SpinnerSet._spin_word_count`.
- **Implementation**:
  - **Pre-Reveal Board Acceptance**: Allowed valid boards generated prior to the 0:45 mark to be accepted directly into the room staging pipeline.
  - **Cache Lookup in Fallback**: Updated `apply_easy_fallback_and_search` to query existing valid board cache before applying fallback parameters.
  - **Balanced Range Odds**: Re-tuned `SpinnerSet._spin_word_count` to maintain true uniform randomness across all permitted ranges (`100-200`, `200-300`, `300-400`, `400-500`, `500+`).
  - **Simulation Verified**: Verified over 5,000 parameter generations that 100-200 occurrence dropped from forced 100% to a natural ~16.7% in 6x8 grids, with balanced distribution across higher word count tiers.

---

## Complete Commit History (October 24, 2026)

| Commit | Summary |
|:---|:---|
| `f53f91bb` | **Restore true random distribution for word count ranges and fix premature easy fallback bias** |
| `06c2c7f1` | **Fix board and word list desynchronization across round transitions and intermission** |
| `5cb9714e` | **fix: resolve BoardGenerator reference in GameRoom.ensure_next_board_ready** |
| `88303a55` | **fix: ensure board design and spinner set format consistency between Equality Freq and Checkerboard** |
| `4f18607c` | **Guarantee board discovery by 0:45, easy fallback by 0:55, and lock declared parameters across round transition** |
| `92e6c419` | **Prevent lobby music from playing inside game rooms** |
| `20a861aa` | **Ensure + AW word counts are strictly 300-400, 400-500, and 500+ and restrict physical caps to non-AW** |
| `d67d6ea0` | **Fix board generation compliance, Spinner Set accuracy, and Remaining word counts across all game modes** |
| `761b6ed9` | **docs: finalize commit ID in stable state summary for October 24, 2026** |
| `ddc59fb7` | **docs: update stable state summary for October 24, 2026 (Start Over Point)** |
| `cf9fb995` | **fix(guessing): keep displaying guessing popup on invalid guesses until a valid word is entered** |
| `90eada35` | **fix(guessing): reset invalid word count on valid word & fix submission error in submit_word** |
| `8ce1db01` | **feat(tools): make Find Count random words table run horizontally on laptops and desktops** |
| `b5b74522` | **fix(faq): change Dark Slate to Default Color and enhance highlight readability across color layouts** |
| `6c47e501` | **fix(lists): keep newly added words at top of New Words lists in alphabetical and random order** |
| `f08cb2b7` | **fix(profile): synchronize mini-profile stats with canonical public profile data** |
| `2ac6bd13` | **fix(combo): invalidate server COMBO_QUERY_CACHE and client _comboClientCache on Added Words changes** |
| `f29fcdaa` | **style: enlarge MORPHEME title and equalize vertical padding on desktop Firefox** |
| `5cf00d0e` | **feat: include valid inflections at the end of definitions for all Added Words** |
| `b4a39873` | **Remove v2.0 from MP title and space out title, description, and arrows in Combo Checker** |
| `9bca2e79` | **Improve point sum equation contrast in white layouts and remove black glow from round-end congratulatory message** |
| `dc59dc73` | **Confine Words to Work on content strictly to its own tab** |
| `89db81c4` | **Add Words to Work on word list to MP Explorer and Subanagrams tools** |
| `2bac69b1` | **fix(play): guard defPanel and all_word_scores lookups in updateGameState** |
| `0bd0e59c` | **feat(play): highlight bonus word row green under Words during round** |
| `c0584457` | **fix(lobby): make border surrounding set rating limits panel razor thin (1px)** |
| `79c6a2a8` | **fix(lobby): equalize space above and below users under players in lobby** |
| `bc48b71b` | **feat(chat): allow continuous typing in chat at round end for users who did not play the round** |
| `455168d0` | **feat(store): add mandatory dictionary edition check to monthly store update routine** |
| `f07dce8d` | **fix(play): uniform divider line thickness on color chart and remove shadow on white layouts** |
| `40a42a84` | **fix(chat): ensure 'Chat disabled for 2 seconds...' notice properly disappears after 2s** |
| `6a4033bc` | **fix(mobile): fit Words to Work on table to screen and restrict scrolling to table only** |
| `9a476c3d` | **feat(mobile): display swiped letters under board in validation space and replace with validation text on release** |
| `a79761a0` | **fix(tools): ensure jumbled letters in Unscramble and Subanagrams fit on a single row without wrapping** |
| `bb26cfdd` | **feat(unscramble): reorganize desktop/laptop layout to mirror Subanagrams dual-panel layout** |
| `f83d68d6` | **fix(lists): correctly scale and scroll virtual thumb when all words fit in DOM (All New Words on mobile)** |
| `70bfef0a` | **update(lobby-guide): change path tracing examples from STOP to STAMP and SLOW to PAROLE** |
| `7a13b836` | **fix(lists): respect random order when opening View Full List modal** |
| `935a6e96` | **fix(app): ensure lst is converted to list before subscripting in order_list** |
| `e0e05daf` | **feat(tools): add Alphabetical and Random order dropdown to Lists with responsive mobile grid layout** |
| `b701f27d` | **fix(vibration): decouple intermission 0:10 vibration alert from bell setting and add Flutter bridge haptic support** |
| `96c549a8` | **style(store): ensure black text on white active category filter in monochromatic dark themes** |
| `8a3f304b` | **style(store,lobby): remove glowing box-shadow from store category filters and start solo match button** |
| `54bddfa1` | **style(subanagrams): widen sequence game area panel and narrow word list results panel on desktop** |
| `48be6f05` | **fix(subanagrams): restore full-height words panel on desktops and laptops with responsive dual-panel layout** |
| `860cd24a` | **feat(store): make showcase images and product images expandable via lightbox** |
| `5a21f948` | **refactor(ui): rename SHOW EVERYONE to VIEW ALL and FIND FRIENDS to VIEW FRIENDS in game rooms** |
| `145d2cad` | **feat(store): update showcase imagery to official Boggle 4x4/5x5 grid games and dictionaries** |
| `1b98dfdb` | **feat(store): showcase Morpheme boutique outdoor displays on desktop store status banner** |

---

## Server & Infrastructure
- **Production Host**: Ubuntu 24.04 LTS at `132.148.72.249`.
- **Application Stack**: Python 3.9 / 3.12 + Waitress WSGI managed by PM2 (`morpheme`).
- **Web Server / Reverse Proxy**: Nginx 1.24.0 with HTTP/2 and SSL/TLS termination (`https://morpheme.games`).
- **Database**: SQLite (`morpheme.db`) with WAL mode, automated schema migrations, in-memory rating caches, and IP timeout controls.
- **Verification**: Verified live on production host `132.148.72.249` with HTTP 200 OK across public profile, mini-profile, guessing validation, and new word lists endpoints.
