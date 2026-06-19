# Morpheme Stable State Summary - June 18, 2026

This summary documents the stable state of the Morpheme application as of June 18, 2026. Localhost, GitHub origin, and `morpheme.games` are fully synchronized and verified under Commit ID **`c31693e`**.

---

## 🚀 Key Improvements & Bug Fixes

### 1. Custom Persistent Dictionaries & Git Overwrite Prevention
*   **The Issue**: Adding words directly to Git-tracked files (`NWL.txt`/`CSW.txt`) on the production server caused those changes to be completely wiped out by Git hard-resets during subsequent deployments.
*   **Persistence Fix**: Modified the `/api/mods/dictionary/submit` endpoint in [app.py](file:///Users/jeffbabiak/app.py) to write custom dictionary additions to `custom_nwl.txt` and `custom_csw.txt` instead of standard tracked dictionary files.
*   **Git Tracking Exclusions**: Untracked `new_NWL.txt`/`new_CSW.txt` and added them alongside `custom_nwl.txt`/`custom_csw.txt` to [.gitignore](file:///Users/jeffbabiak/.gitignore), ensuring they are persistent and immune to Git resets.
*   **WordValidator Integration**: Updated `_load_dictionaries` and `ensure_csw_loaded` in [word_validator.py](file:///Users/jeffbabiak/word_validator.py) to automatically load and merge custom dictionaries into the active word sets on startup and lazy-loading.
*   **Added Words Restoration**: Successfully uploaded and merged **1,666 missing words** from `misplaced.txt` back into the production `added_words.txt`, bringing the total count to **9,395** words. Also synchronized this updated file back to the localhost.
*   **Clean Staging Area**: Maintains the staging cleanup logic in [app.py](file:///Users/jeffbabiak/app.py) which removes newly uploaded dictionary words from `added_words.txt`. Since custom additions are saved in the persistent `custom_nwl.txt`/`custom_csw.txt` files, the words are safely transitioned from the staging list to the active dictionary without risk of loss.

### 2. Dimension-Aware Fallback Word Lengths
*   **Watchdog & 6x8 Rescue Fix**: Modified the stuck intermission watchdog and the 6x8 rescue functions in [game_room.py](file:///Users/jeffbabiak/game_room.py) to use `SpinnerSet._spin_min_word_length()` to determine the minimum word length instead of hardcoding `'min_word_length': 3`.
*   **Preventing Low-Length Promoted Play**: This ensures that when a 6x8 or 4x6 room times out and loads the emergency fallback board (which contains columns of 8-letter and 6-letter words respectively), the room correctly maintains its required minimum length (6L–8L for 6x8; 4L–6L for 4x6) and rejects short words like `GIN`.

### 3. Tournament Replay Security & Active Standings Censorship
*   **Standings Visibility**: Modified [tournaments.js](file:///Users/jeffbabiak/static/js/tournaments.js) to hide the **Round Standings** leaderboard card entirely for active participants who have not completed their turn.
*   **Selective Replay Censorship**: Modified [app.py](file:///Users/jeffbabiak/app.py) to censor `board_data` and `submitted_words` of other players for active rounds. 
*   *   Unplayed participants and spectators/guests see censored placeholder results (`board_data: null`, `submitted_words: []`).
*   *   Players who **have finished their turn** can view other players' replays, even while the round remains active.
*   **Winner Route Protection**: Restricted the `/api/tournament/winner-turn/<tid>/<username>` endpoint in [tournament_logic.py](file:///Users/jeffbabiak/tournament_logic.py) to completed tournaments only (`status == 'completed'`), preventing custom queries from harvesting active round details.

### 4. Unified Replay UI & Visual Board Highlight Paths
*   **Consolidation**: Consolidated the redundant "Snapshot" and "Walkthrough" buttons in the tournament standings and lobby lists into a single, clean **▶ Replay** button.
*   **Board Highlights Restored**: Fixed board rendering and path coordinate extraction inside `watchTournamentReplay` and `watchTournamentWinnerReplay` when board data is in the dictionary format.
*   **Timeline Interactive Paths**: Added an interactive path highlight feature. Clicking any word in the "Timeline of Discovery" review list during replay playback instantly highlights its valid path coordinates on the board grid.

### 5. Tournament Mid-Round Leave Protection
*   **Draft Auto-Saves**: Implemented client-side auto-saves (`/api/tournament/save-draft`) on the `/api/tournament/submit` channel after each word discovery to preserve in-progress score state.
*   **Finalization on Re-entry/Exit**: Automatically creates a score row (`submitted_at = NULL`) when starting a turn. If a user tries to re-enter, refreshes the tab, or navigates away, the server finalizes the turn immediately (`submitted_at = current_time`) and returns a 403 error, preventing timers from resetting.
*   **Stale Turn Auto-Finalization**: Added a 15-second grace buffer to auto-finalize stale turns during tournament status updates if browser closures or disconnects left the turn unsubmitted.

### 6. Touch Drag Selections & Mobile Vertical Board Orientation
*   **Swipe Alignment**: Isolated touchscreen and mouse drag selectors using pointer identifiers to prevent touch release failures. Ensure selections submit immediately upon pointer release.
*   **Vertical Mobile Boards**: Solved the double-transposition layout bug. We now transpose the underlying board grid arrays directly in a unified `safelyTransposeState` helper (applied to normal rooms, tournament matches, and private matches) and render the transposed arrays row-by-row naturally without double CSS grid/loop variable transpositions.
*   **Intermission Filter Coordination**: Swapped row and column coordinates when mapping path filters in intermission so they align perfectly with transposed grid elements.

### 7. Win/Loss UI Score Text Cleanup
*   **Duplicate Elimination**: Removed the duplicate text line `You: X | Opponent: Y` under the "YOU WON! ADVANCING..." and "YOU LOST" panels in [tournaments.js](file:///Users/jeffbabiak/static/js/tournaments.js), keeping the layout clean as the detailed styled scores list is already rendered below it.

### 8. Default Settings on Registration
*   **DB Setting Initialization**: Populated `user_settings` table entries during registration to set default configurations:
    *   `board_sizes`: `{"4x4": 82, "4x6": 82, "5x7": 65, "6x8": 54}`
    *   `corner_cutoff` (Tile Selectable Space): `39`
*   **System Fallbacks**: Configured fallback configurations in `settings.js`, `play.js`, and `play.css` to match these default settings.

---

## 🛠 Active Features & Configuration
*   **Board Formats**: Normal, Checkerboard, Double, Triple, Valued Letters, Rotation, Penalty, Mania, Either/Or, Bonus Word, and Density.
*   **Grid Dimensions**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
*   **Dictionaries**: NWL (American) and CSW (International) Tries.
*   **Verification**: Tested local compilation checks and verified word validator tests pass 100% cleanly.

---

**Latest Stable Commit ID**: `a255c20bff4fdaf71c51eb9063f753239cc0a67a`  
**GitHub Tag**: `START_OVER_POINT_JUNE_18`  
**Localhost & GitHub Sameness Status**: Synchronized  
**Production Server Status**: Green / PM2 Online / Live at commit `a255c20`
