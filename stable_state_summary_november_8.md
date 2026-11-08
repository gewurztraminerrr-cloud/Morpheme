# Stable State Summary — November 8, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_NOVEMBER_8
```
*(Also tagged as `stable-2026-11-08`)*

## Latest Feature Commit ID
```
f4f4c6f6e684b7d7cd39d393b7f10493a61b6f1a
```

## Synchronization Status

| Environment | Status | Commit / Tag | Verification Details |
|:---|:---|:---|:---|
| **Localhost** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_8` | Working tree clean on `main`, zero uncommitted tracked changes |
| **GitHub (`origin/main`)** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_8` | Pushed to `origin/main` & tags `START_OVER_POINT_NOVEMBER_8` / `stable-2026-11-08` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_8` | Fast-forward updated to `origin/main`, PM2 online (process `morpheme`), HTTP 200 OK |
| **App (Mobile WebView)** | ✅ Synchronized | `https://morpheme.games/` | Flutter app loads live synchronized webview with native bridges |

---

## Session Features & Fixes (November 8, 2026)

### 1. Inactive Player Room Roster Eviction at 10-Minute Boundary (`app.py`, `game_room.py`, `static/js/app.js`, `static/js/play.js`)
- **Immediate Roster Eviction Across All Devices**:
  - Addressed multi-browser / multi-platform presence divergence where a player inactive for 10 minutes was kicked to Lobby, but their username lingered in the game room Players list on mobile devices (whose screens went black/asleep) for an additional 10 minutes until screen wake.
  - Enforced instantaneous eviction from room rosters the moment the 10-minute inactivity threshold is reached, regardless of whether the user is on desktop, laptop, or mobile.
  - Implemented proactive backend auto-sweep in `game_room.py` and `app.py` that removes expired sessions directly from the room's active roster and immediately broadcasts updated roster state to all remaining players without waiting for a client-side wake ping.

### 2. Elimination of Empty / Zero-Match Words in Combo Checker MP Explorer (`app.py`, `static/js/tools.js`)
- **Non-Plural, Non-Inflected Verification Prior to Table Insertion**:
  - Resolved issue where input words such as `ABAMPERE` (Length 8, Diff 0, 1 MP) appeared in the left-hand candidate table, but clicking them yielded "No non-plural, non-inflection diff 0 1MP words found for ABAMPERE".
  - Updated `check_word_has_mp` in `app.py` to ensure candidate words are validated against matching non-plural, non-inflected partner words before being rendered in the MP Explorer candidate list.
  - Applied this filtering uniformly across all user configurations (Length, Difference, and MP Value settings) so that words without valid partners are never added to the candidate table.

### 3. Active Round Word Definition Lookups in Game Rooms (`static/js/play.js`)
- **Reliable Definition Lookups During Gameplay**:
  - Fixed an issue where clicking on a submitted word during an active round in game rooms failed to display its definition popup (particularly disruptive in 24h rooms where all words are played).
  - Eliminated conflicting DOM re-renders and inadvertent highlight clearing in `static/js/play.js`.
  - Ensured word definitions consistently load and display in active rounds without race conditions.

### 4. Intermission Word Click Handler Collision Resolution (`static/js/play.js`)
- **Intermission Definition & Board Tracing Integrity**:
  - Resolved an event listener collision where a global `#submitted-words-list` click listener intercepted clicks on intermission words, blocking the definition modal and board path tracing in `displayAllWords()`.
  - Added an intermission guard so clicking any word during intermission reliably renders its full definition, traces letter paths on the grid, and highlights finder indicators.

### 5. Lobby Placeholder Logo Display & Safari/WebKit Geometry Fix (`templates/index.html`, `static/css/lobby.css`, `static/js/lobby.js`)
- **Placeholder Logo Container Wrapping**:
  - Resolved issue where the Morpheme logo in the "No 'Show Rooms' selected" panel in the Lobby was invisible or collapsed.
  - Diagnosed a WebKit/Safari flexbox column bug causing `height: auto` SVGs to collapse to 0 height; resolved by wrapping the SVG in `<div class="placeholder-icon-wrap">` with explicit minimum and proportional dimensions (`min-width: 140px; min-height: 124px;` desktop, `min-width: 130px; min-height: 115px;` mobile).
- **Mobile Viewport Spacing & Drop Shadow Aesthetics**:
  - Tuned `#rooms-placeholder-view` padding on mobile (`padding: 24px 16px 16px`) to eliminate vertical overflow and unwanted scrolling.
  - Enhanced logo styling with dual cyan/teal (`rgba(56, 189, 248, 0.45)`) and white drop shadows on dark themes, with high-contrast shadows on light/white themes.
- **Dynamic Toggle Restoral**:
  - Added toggle-off / deselect support in `handleShowRoomsInline` and exposed `window.renderRoomsPlaceholderView()` so deselecting "Show Rooms" immediately restores the centered logo placeholder view.
  - Bumped asset query strings to `1794100000` to prevent caching issues.

### 6. Toggle / Retract Selected Words in Game Rooms Across Rounds and Intermission (`static/js/play.js`, `templates/index.html`)
- **Word Selection Retraction on Repeated Click**:
  - Implemented `toggleWordSelection(word, clickedElement)` in `static/js/play.js`.
  - When a user selects a word under Words or All Words (as well as FCFS feed, notepads, or previous day list) during active rounds or intermission, clicking that same word again immediately deselects/retracts it.
  - Instantly removes `.finder-active` highlight from word items, clears `.review-highlight` from board cells, removes `.finder-highlight` from player cards in the roster, and calls `clearDefinitionPanel()` to return the definition card back to its clean default state.
  - Bumped asset version query string to `1794101000`.

### 7. Intermission Tile Filtering for Q / QU Tiles and Separate U Tiles (`static/js/play.js`, `static/js/play_tournament.js`, `static/js/tools.js`, `templates/index.html`)
- **Exhaustive Q / QU Match Length Exploration ([2, 1])**:
  - Resolved an issue during intermission where clicking a "QU" tile on the board failed to include words like `EQUINES` in the filtered All Words list, even though `EQUINES` appeared in All Words when no tile was selected.
  - The root cause was greedy matching in `rebuildTileToWordsMap()` (`dfs2D` and `dfs3D`), `findWordPathOnBoard()`, `findWordPathOnCube()`, and tools/tournament pathfinders: when matching letters on a `'Q'` or `'QU'` tile, if the word contained `'QU'`, the search greedily consumed both letters (`matchLength = 2`) on the single tile and never tested `matchLength = 1` (consuming only `'Q'` on the tile and proceeding to an adjacent `'U'` tile).
  - Updated all DFS pathfinding and indexing routines to branch over both `matchLength = 2` and `matchLength = 1` for `'Q'` / `'QU'` tiles.
  - Words formed via `Q + separate adjacent U` tiles (such as `EQUINES`: `E` -> `Q` -> `U` -> `I` -> `N` -> `E` -> `S`) are now properly indexed and displayed when clicking either the `QU` tile or the separate adjacent `U` tile during intermission.
  - Ensured all board words across `all_words_paths`, `lastDisplayAllWordsArgs`, `previous_all_words`, and `all_words` are exhaustively indexed and mapped to their tiles.
  - Bumped asset version query string to `1794102000`.

### 8. Final-Second / Intermission Word Checkmark & State Synchronization (`game_room.py`, `static/js/play.js`, `templates/index.html`)
- **Buzzer-Beater Grace Period on Server (`game_room.py`)**:
  - Implemented a 1.5-second buzzer-beater grace period in `GameRoom.submit_word()` so that words submitted at the final second of the round that arrive over the network right as the server transitions to intermission are validated, credited, and recorded rather than discarded with "Round is not active".
  - Synchronized `winners_history` and player scores for submissions accepted during this grace period.
- **Optimistic Rollback & Checkmark Parity (`static/js/play.js`)**:
  - Resolved state discrepancy where words submitted on the wire were optimistically highlighted blue as found by the player, but failed to show a checkmark (`✓`) in the "All Words" list because `isFoundByAny` only checked server player arrays.
  - In `displayAllWords()`, enforced `isFoundByAny = allFoundSet.has(wordUpper) || isTargetFound`, guaranteeing that any word marked as found by the player (`isTargetFound` / blue) is 100% consistently displayed with a checkmark (`✓`).
  - Merged player found words across `previous_submitted_words` during intermission, `previous_players`, and `window._localSubmittedWords` into `uniqueGlobalFound`.
  - Added optimistic rollback in `submitWord()` so if a submission is genuinely rejected by the server, it is removed from `_localSubmittedWords` and does not linger as a false found word.
  - Bumped asset version query string to `1794103000`.

### 9. Mobile Bottom Panel Navigation Alignment ('Words ➜' and 'Board ➜') (`static/css/play.css`, `templates/index.html`)
- **Right-Aligned Positioning for Mobile Navigation Arrows**:
  - Resolved layout spacing on mobile devices in game rooms where the bottom navigation buttons ("Words ➜" on the board panel and "Board ➜" on the players panel) appeared shifted inward to the left and not far enough to the right.
  - Removed the legacy `margin-right: 25px !important;` offset on `.mobile-nav-indicators .nav-right` and updated it to `margin-right: 0 !important;`, allowing the buttons to sit 25px further right (aligned with the standard 15px container padding).
  - Restored visual balance and symmetry with the left-hand navigation buttons ("➜ Players" and "➜ Board").
  - Bumped `play.css` version query string in `templates/index.html` to `1794090100`.

### 10. Intermission Board Tile Filtering Letter Accuracy (`static/js/play.js`, `templates/index.html`)
- **Accurate Tile Word Filtering During Intermission**:
  - Resolved an issue during intermission where selecting a letter on the board caused words that did not use that letter to appear in the filtered All Words list.
  - Eliminated server path coordinate mismatch on mobile portrait boards by removing untransposed server coordinate injection in `rebuildTileToWordsMap()`, relying authoritatively on `getWordCells()` DFS pathfinder on the actual displayed board (`previous_board`).
  - Added path transposition for `all_words_paths` and `previous_all_words_paths` in `safelyTransposeState()` so client paths match transposed boards.
  - Removed the `all_words_paths` prerequisite check in `displayAllWords()`, ensuring tile filtering reliably executes even if `all_words_paths` is empty.
  - Implemented ironclad secondary letter verification in `displayAllWords()` so that only words containing the selected tile's letter(s) can ever appear.
  - Bumped `play.js` version query string in `templates/index.html` to `1794104000`.

### 11. Constant Parameter Rating Adjustment Tool in Moderator Tab (`app.py`, `templates/index.html`, `static/js/mods.js`)
- **Global & Live Game Room Rating Adjustment**:
  - Created dedicated "Constant Parameter Rating Adjustment" tab in Mods section (`#mod-tab-ratings`) to increase or reduce ratings across all users in specific game room constant parameter configurations (e.g., `ACC 4x4 45s`, `ACC 4x4 3m`, `FCFS 4x4 45s`, `Split 4x4 45s`, `Cube 3x3x3 3m`, etc.) when ratings become skewed.
  - **Strict Authorization**: Access is strictly restricted to root administrator `jeffb` on both frontend and backend (`(session.get('username') or '').strip().lower() == 'jeffb'`), with any non-jeffb request rejected with `403 Forbidden`.
  - **Instantaneous Live Synchronization**: Updates the database (`user_ratings` with `MAX(1, rating + delta)`) and live in-memory room player ratings (`room_manager.rooms`) simultaneously under thread locks, ensuring players active in live rooms see their rating change in under 1 second without waiting for round completion.
  - **Interactive UI & Audit Logging**:
    - Step 1: Point delta input with quick presets (`+10`, `+25`, `+50`, `+100`, `-10`, `-25`, `-50`, `-100`).
    - Step 2: Categorized configuration cards with "Select All", "Deselect All", and "Active Only" filters.
    - Step 3: Apply button with confirmation safety dialog detailing affected configs and users.
    - Audit logging stored in `rating_adjustments` database table and appended to `rating_audit.log`.
  - **Strict Constant Parameter Conformance & Refined Protection Rules**:
    - Strictly restricts selectable configurations to the 31 authentic constant parameter room configurations that exist in Morpheme (ACC: 45s/3m/10m across 4x4, 4x6, 5x7, 6x8 [12 configs]; FCFS: 45s/3m across 4x4, 4x6, 5x7, 6x8 [8 configs]; Split: 45s/3m across 4x4, 4x6, 5x7, 6x8 [8 configs]; Cube: 3m/5m/10m on 3x3x3 [3 configs]), eliminating phantom or non-existent options (e.g. 10s, 24h, 1m, 2m, 4m).
    - **Removal of "Active Only" Button & Clean Dual-Button Alignment**: Removed the redundant "Active Only" button, styling the "Select All" and "Deselect All" buttons with clean padding and borders to fit harmoniously into the configuration selection card header.
    - **Absolute Rating Floor of 1 (`MAX(1, rating + delta)`)**: Ensured rating adjustments clamp to an absolute minimum rating of `1` (not 0 or negative numbers) when negative point adjustments are applied to low-rated users (such as deducting 100 points from a rating of 50).
    - **Provisional/Newcomer Protection Rule**: Excludes users who have a rating of 1200 and have played fewer than 10 games (`games_played < 10` across persistent `users.games_played` and `round_history`), protecting unplayed or barely-played newcomer accounts from mass rating adjustments while continuing to adjust established users with 1200 ratings who have played 10 or more games.
    - **Firmly Established Rating Average & Protection Rule (>10 Games)**: When declaring the average rating for each constant configuration, provisional/unestablished 1200 ratings (users who have played ≤ 10 games) are strictly excluded; only firmly established ratings (any rating != 1200, plus 1200 ratings for users with > 10 games) are included in the average calculation and eligible user counts.
    - **In-Panel Rating Breakdown & Stats Overview Banner**: Integrated a stats banner directly inside the Constant Parameter Rating Adjustment panel displaying total firmly established ratings (18) and 1200 ratings (31 total: 15 provisional ≤10 games, 16 established >10 games); each configuration card now explicitly declares established count, average rating, and 1200 count (e.g. `4 established (avg 1200.0) • 9 @ 1200`).
    - **Unconditional Apply Button Clickability & Direct Event Handling**: Fixed an issue where the "Apply Rating Adjustment" button could become unclickable due to client-side timing checks in `checkRatingAdjAuth()`; button defaults to active with direct `onclick="window.applyRatingAdjustment()"` fallback, verifies authority seamlessly against `/api/mods/status`, and lets backend authoritatively enforce `jeffb` permission.
  - Bumped `mods.js` version query string in `templates/index.html` to `1794128000`.

### 12. 10-Rounds Unrated Placement Policy, Toggle Switches & Zero-Change Equilibrium Rating (`app.py`, `game_room.py`, `rating_logic.py`, `static/js/mods.js`, `static/js/play.js`, `static/js/lobby.js`, `static/js/app.js`, `templates/index.html`)
- **Per-Configuration Toggle Switches & Batch Controls**:
  - Added dedicated animated iOS-style toggle switches (`.mod-toggle-switch`) on each of the 31 constant parameter configuration cards in `#mod-tab-ratings` (`⚡ 10-Rounds Est.` vs `Standard 1200`), plus Step 2 batch buttons (`⚡ Enact 10-Rounds Est. (Selected)` and `Revoke 10-Rounds Est. (Selected)`) and an active configuration counter in the overview banner.
  - **Strict `jeffb` Authorization**: Toggling placement switches is strictly restricted to root administrator `jeffb` on both the backend (`POST /api/mods/toggle-unrated-placement` returning `403 Forbidden` for any other moderator) and the frontend (`disabled` toggle inputs, `cursor: not-allowed`, dimmed opacity, and restricted tooltip).
- **10-Round Unrated Placement & White Square Display**:
  - When enabled for a configuration, all new users henceforth start **Unrated** with a pure white rating square (`#ffffff` with no metallic shine) in both the Lobby (`lobby.js`) and Game Rooms (`play.js`, `app.js`), and their rating remains unchanged (`rating_change = 0`) for their first 10 rounds.
  - Unrated players are protected from mass rating delta adjustments and excluded from round abandonment bounty payouts during their 10 placement rounds.
- **Zero-Change Equilibrium Rating Establishment & Anti-Cheat Cap (2100)**:
  - At the conclusion of each placement round against active human opponents, the server calculates the Zero-Change Equilibrium Rating ($R_{\text{round}} = R_{\text{opponents}} \times \frac{S_{\text{user}}}{S_{\text{opponents}}}$)—the rating at which the player would have had `0` rating change given the round's scores.
  - Upon completing round 10, the user's approximate rating is established as the average of the 10 equilibrium ratings, clamped to a strict ceiling of **2100** (`max(1, min(2100, R_approx))`) in the event of cheating and a floor of `1`. From round 11 onward, the user's rating changes normally with each round.
- **Expanded 3-Tier Card Layout & Configuration-Specific Game Counts**:
  - Redesigned configuration cards (`minmax(250px, 1fr)`) into a 3-row layout providing full, unclipped room for: (1) Checkbox & Configuration Title, (2) Firmly Established Count & Average alongside Accounts at `1200`, and (3) Policy Status & Toggle Switch.
  - Fixed room creation so the room creator is explicitly initialized with `joined_mid_round = False`, ensuring Round 1 participation is recorded in `user_ratings`.
  - Updated `get_rating_configs` and `adjust_ratings_by_config` to count games played strictly **per configuration** (`ur.games_played` and matching `round_history` entries for that mode, board dimensions, and duration) rather than global site-wide games, ensuring unestablished `1200` accounts (`≤ 10` rounds in that specific configuration) are never falsely counted as established.
  - Bumped asset cache busters to `1794128000` (`mods.js`) and `1794125000` (`play.js`, `lobby.js`, `app.js`).
