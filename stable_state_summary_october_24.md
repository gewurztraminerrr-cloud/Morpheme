# Stable State Summary — October 24, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_24
```
*(Also tagged as `stable-2026-10-24`)*

## Latest Commit ID
```
1f3816b6c74dfc872734c6a1d654ec7ddce1ed62
```

## Synchronization Status

| Environment | Status | Details |
|:---|:---|:---|
| **Localhost** | ✅ Synchronized | Clean working tree on `main` |
| **GitHub (`origin/main`)** | ✅ Synchronized | Pushed to `origin/main` & tags `START_OVER_POINT_OCTOBER_24` / `stable-2026-10-24` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | Updated to latest commit, tags fetched, PM2 process online, live HTTP 200 OK |
| **App / Client** | ✅ Synchronized | Cache busters bumped (`howtoplay.css?v=1792844000`, `play.css?v=1792900000`, `tools.js?v=1792902000`, `play.js?v=1792902000`, `app.js?v=1792902000`) for instant live propagation across all devices |

---

## Session Highlights & Architectural Updates (October 24, 2026)

### 0. In-Game Rooms: Split "Add to Words to Work on" with "Search in Combo Checker"
- **Horizontal Split Layout**: In game rooms (`#page-play`), when a word is highlighted/selected in the words panel, `#work-on-game-btn-container` is displayed as a responsive horizontal flex row containing two equal-width action buttons:
  - Button 1: "Add to Words to Work on" (`#add-to-work-on-game-btn`)
  - Button 2: "Search in Combo Checker" (`#search-in-combo-game-btn`)
- **CC Popup Logic Parity**: Clicking "Search in Combo Checker" triggers `window.searchWordInComboChecker(targetWord, targetDict, 'page-play')`, executing the identical logic previously attached to the "CC" button on definition popups:
  - Sets `_navigatingViaDefPopup = true` to preserve back navigation.
  - Pushes current position to `_defPopupNavStack` with `originPage: 'page-play'`.
  - Enables Tools back button (`setDefBackNavVisibility(true)`).
  - Navigates to Tools -> Combo Checker -> Manual Tab -> Microprocedures (MP).
  - Resolves room dictionary: if `+ AW` is active, sets Combo Checker dictionary to `ALL`; otherwise sets `NWL` or `CSW`.
  - Populates input with the selected word and runs search.
  - Scrolls all desktop and mobile containers cleanly to the top.
  - Clicking the Tools Back button seamlessly returns the user back to the active game room.
- **Theme Contrast**: Styled with high contrast across dark themes, light themes, and pure white theme.

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

### 21. Intermission Countdown Loop Fix (`0:02`, `0:05`, `0:17` Reset Fix)
- **Issue Resolved**: During intermission, the timer repeatedly got stuck in a cycle of counting down to 0:00 and snapping back to 0:02, 0:05, or 0:17 over and over again.
- **Root Causes**:
  - `updateLocalTimer()` in `static/js/play.js` had a re-anchor mechanism where if `remaining <= 0` but `window.lastGameState.time_remaining > 0`, it read the stale snapshot `time_remaining` and repeatedly re-anchored `localEndTime = (Date.now() / 1000) + time_remaining`, causing an endless loop.
  - In `game_room.py`, 24-hour room intermission limits were set to 2.0s in one check and 5.0s in another, causing server-side desynchronization between `time_remaining` and `intermission_end_time`.
- **Implementation**:
  - Removed broken re-anchor logic in `updateLocalTimer()` so that when `remaining <= 0`, the timer cleanly clamps to `0:00`, clears the interval, and triggers a transition sync check.
  - Unified 24-hour room intermission limits in `game_room.py` to a consistent 5.0 seconds.

### 22. Remaining Tab Mobile Alignment: Length Labels Aligned Furthest to Left with Matching Padding
- **Issue Resolved**: On mobile devices in game rooms, the Remaining tab (`#remaining-words-table`) inherited `text-align: center !important` from `.words-panel`, causing length labels ("6LW", "7LW", etc.) to render in the middle of each row while word counts were aligned to the right.
- **Implementation**:
  - In `static/css/play.css`, added explicit `text-align: left !important; padding-left: 10px !important;` for `.len-cell` and `text-align: right !important; padding-right: 10px !important;` for `.count-cell` under `@media (max-width: 992px)` and `body.is-mobile`, placing the length labels furthest to the left with padding matching the count numbers on the right.
  - Bumped `play.css?v=1792896000` in `templates/index.html`.

### 23. Rotate & Transpose Buttons Mobile Symmetry: Equal Top, Bottom, and Right Padding
- **Issue Resolved**: On mobile devices in game rooms, the timer panel (`.timer-display`) containing the Rotate and Transpose buttons had 12px right padding while vertical padding around the buttons was 6px (buttons are 26px high within the 38px panel), leaving an asymmetric 12px space to the right of the Transpose button.
- **Implementation**:
  - In `static/css/play.css`, set `.timer-display` padding to `6px 6px 6px 12px !important;` under `@media (max-width: 992px)` and `body.is-mobile`.
  - Shifted the Transpose button 6px to the right so that spacing above (6px), below (6px), and to the right (6px) of the buttons within the panel is perfectly equalized.
  - Bumped `play.css?v=1792897000` in `templates/index.html`.

### 24. Clear Unsubmitted Swiped Letters Below Board at End of Round on Mobile
- **Issue Resolved**: On mobile devices in game rooms, text spelling out the letters the user had swiped below the board in `#word-validation-status` (`.validation-status`, `.status-swiping`) would freeze and remain visible throughout intermission if the round ended before the letters were submitted.
- **Root Causes**:
  - As a user drags across board cells on mobile, `selectCell` actively updates `#word-validation-status` with the uppercase letters of the drag path.
  - If the round expires while the player's finger is on the screen or if the user never submits the word before round end, `finishDragSelection` either does not fire or state transitions to intermission, leaving the unsubmitted letters permanently frozen under the board.
- **Implementation**:
  - In `static/js/play.js`:
    1. In `updateGameState()` upon state transition from `active` to `intermission`, immediately cleared `#word-validation-status` text, class, and cancelled any pending `validationTimeout`.
    2. In `updateGameState()` standing intermission checks (`state.state === 'intermission'`), added safeguard to clear `#word-validation-status` text and styling if any text or `.status-swiping` class is present.
    3. In `updateLocalTimer()` when the timer counts down to 0 (`seconds <= 0 || remaining <= 0`), immediately cleared `#word-validation-status` text, reset `mouseState.isDown = false`, cleared drag selection paths, and removed active cell highlights.
    4. In `showValidationFeedback()`, added an early return guard if `window.lastGameState.state === 'intermission'` or `'waiting'` so late network responses do not re-populate text under the board.
    5. In `clearWordInputAndSelection()`, unconditionally cleared `#word-validation-status` text and styling.
    6. In `finishDragSelection()`, added a guard checking if the round is over (`window.lastGameState.state === 'intermission' || window.lastGameState.state === 'waiting'`) to immediately clear `#word-validation-status` and suppress late submissions.
  - In `templates/index.html`, bumped `play.js?v=1792898000`.

### 25. Lexicographical Verification of Added Words Inflections via Definitions
- **Issue Resolved**: When looking up Added Words such as `ADENIA`, unrelated words like `ADENIUM` (and `ADENIUMS`) were incorrectly listed under `[inflections: ...]`.
- **Root Cause**: An overly broad Latin suffix-chopping rule (`w.endswith('A') -> UM, ON`) in `get_roots_for_word` guessed that `ADENIUM` was the singular root of `ADENIA` without verifying that the definitions of the two words actually linked to each other. Because `ADENIUM` is a valid Scrabble word (a genus of shrubs), it was added as a root and generated further false inflections (`ADENIUMS`).
- **Implementation**:
  - Implemented `extract_root_from_definition(defn)` to extract grammatical root references from headwords (`DOG, ...`) and inflection statements (`plural of ...`, `participle of ...`, `past tense of ...`, `comparative form of ...`, etc.).
  - Implemented `extract_bracket_inflections(defn, root)` using `re.finditer` to parse bracketed inflection lists (e.g. `[n -S]`, `[v -ED, -ING, -S]`, `[n BACTERIA]`, `[n MEDIA]`, `[n CORONAS or CORONAE]`).
  - Implemented `is_verified_inflection(w, cand, def_w, def_cand)`:
    1. Validates that `def_cand` explicitly points to `w` as root or `def_w` explicitly points to `cand` as root.
    2. Validates that `cand` is declared in bracket notations of `def_w` or `w` is declared in bracket notations of `def_cand`.
    3. Rejects candidates that have independent definitions describing unrelated entities (e.g. `ADENIUM` describing a genus of shrubs is rejected as an inflection of `ADENIA`).
    4. For unindexed candidates without definitions, permits inclusion only if they are regular grammatical inflections (`is_plural_or_inflection`).
  - Removed speculative Latin singular string-chopping from `get_roots_for_word` and required all candidate roots and inflections to pass `is_verified_inflection`.
  - Verified: `ADENIA` now correctly lists only `[inflections: ADENIAS]`, `ADENIAS` lists `[inflections: ADENIA]`, and `ADENIUM` lists `[inflections: ADENIUMS]`.

### 26. Strict Added Words (+ AW) Word Count Range Enforcement (300-400, 400-500, 500+) (`100.png` Fix)
- **Issue Resolved**: As documented in `100.png`, an Accumulative 4x4 room with `Dict: CSW + AW` displayed `Words: 100-200` with 188 total words found (`0/188 - 0% (309 total pts)`), violating the ironclad rule that Added Words (`+ AW`) rounds must always have word count ranges of `300-400`, `400-500`, or `500+`.
- **Root Causes**:
  1. `RoomManager._get_factchecked_wc_range(count, use_added_words=False, dictionary=None)` received `use_added_words` as an argument but completely ignored it, calculating ranges solely based on raw word count and mapping 188 words to `'100-200'`.
  2. Fallback rescue routines (`get_emergency_fallback_board`, `ensure_next_board_ready`) had `min_accept = max(100, min_accept)` even when `use_added_words` was True, and did not realign `max_accept` when target ranges were resolved to 300+, causing live generators to sample boards down to 199 words.
  3. Cached board retrieval (`pop_compatible_cached_board`, `pop_cached_board`) did not reject pre-generated boards with `< 300` words or with `word_count_range in ['50-100', '100-200', '200-300']` when `use_added_words` was active.
  4. Client-side rendering in `play.js` (`formatParameters`, `wordsPanelHeader`) and `private_matches.js` (`renderMatchList`, `renderInvites`) did not include safety clamps to prevent displaying contradictory headers.
- **Implementation & Ironclad Fixes**:
  - **`_get_factchecked_wc_range`**: Updated to strictly inspect `is_aw = bool(use_added_words) or '+ AW' in str(dictionary).upper()`. When `is_aw` is True, returns `'500+'` (>=500), `'400-500'` (>=400), or `'300-400'` (<400) — never `< 300`.
  - **Fallback Safeguards**: Updated `get_emergency_fallback_board`, `ensure_next_board_ready`, and `trigger_early_fallback_rescue` to enforce `min_accept = max(300, min_accept)` and `max_accept = max(399, max_accept)` whenever Added Words is active, ensuring candidate order strictly uses `['300-400', '400-500', '500+']`.
  - **Cache Protection**: In `board_generator.py`, updated `pop_compatible_cached_board` and `pop_cached_board` to discard any cached boards with `< 300` words or non-compliant range tags when `use_added_words` is active.
  - **Room Lifecycle Enforcement**: In `game_room.py` (`start_next_round`, `start_board_search`, `kickstart_room`, and existing room wakeup), enforced that `current_word_count_range`, `spinner_params`, and `frozen_revealed_params` are always validated via `SpinnerSet.sanitize_params` and `_get_factchecked_wc_range`.
  - **Client-Side Safeguards**: In `static/js/play.js` and `static/js/private_matches.js`, added fallback clamps ensuring that whenever `+ AW` is active, the displayed word count range is always `300-400`, `400-500`, or `500+`.
  - **Verification**: Verified via automated test suite `test_aw_word_count_compliance.py` (500+ sanitizations, live emergency generator, and room round transition) that `+ AW` rounds NEVER produce ranges below 300 words.

---

## Complete Commit History (October 24, 2026)

| Commit | Summary |
|:---|:---|
| `1f3816b6` | **docs: update stable state summary and walkthrough for October 24** |
| `3107b587` | **fix(tools): prevent resetToolsTab and MutationObserver from wiping active Combo Checker pane during def popup navigation** |
| `0c5cad6b` | **feat(play): split Add to Words to Work on button with Search in Combo Checker in game rooms** |
| `9479ef5b` | **docs: finalize Start Over stable point with Added Words 300+ word count enforcement** |
| `5c8d4dbd` | **Enforce strict 300-400, 400-500, 500+ word count ranges for Added Words (+ AW) across all generation and fallback paths** |
| `10bc3cbc` | **docs: finalize Start Over stable point with definition verification** |
| `3f4ce71e` | **docs: update stable state summary with definition-verified inflections** |
| `ce4a1a53` | **fix(definitions): verify inflection candidates against dictionary definitions before appending** |
| `3aefca06` | **docs: update latest commit ID in stable state summary** |
| `7d734dca` | **docs: finalize Start Over stable point for October 24, 2026** |
| `b0fb3d13` | **docs: finalize commit ID in stable state summary for October 24, 2026** |
| `8f4a780d` | **docs: update stable state summary with unsubmitted swiped letters fix** |
| `c5eca53d` | **fix(mobile): clear swiped unsubmitted letters below board at end of round** |
| `a69f1605` | **docs: finalize commit ID in stable state summary for October 24, 2026** |
| `7a19ad5d` | **fix(game_room): import copy at module level to prevent UnboundLocalError in start_next_round** |
| `eee04f3c` | **style(play): equalize padding above, below, and to the right of Rotate/Transpose buttons in timer panel on mobile** |
| `758400b4` | **docs: update stable state summary with matching Remaining padding** |
| `97fc2ee3` | **style(play): match left padding on len-cell with right padding on count-cell in Remaining tab on mobile** |
| `43923e2d` | **docs: update stable state summary with Remaining tab mobile alignment** |
| `7080f27c` | **style(play): align remaining words length labels furthest to the left on mobile devices** |
| `1b6e0704` | **fix(play): eliminate intermission timer countdown loop and align 24h intermission limits** |
| `a133fbc5` | **docs: update walkthrough and stable state summary with intermission timer fix** |
| `1e070568` | **docs: update stable state summary for October 24, 2026 (Start Over Point)** |
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
