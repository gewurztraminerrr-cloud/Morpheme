# Morpheme Stable State Summary - June 18, 2026

This summary documents the stable state of the Morpheme application as of June 18, 2026. Localhost, GitHub origin, and `morpheme.games` are fully synchronized and verified under Commit ID **`3184446`**.

---

## 🚀 Key Improvements & Bug Fixes

### 1. Tournament Replay Security & Active Standings Censorship
*   **The Issue**: Players who had not yet completed their turn in the active tournament round could view the Round Standings and click "Replay" to see other players' words and boards, leaking solutions before they played.
*   **Standings Visibility**: Modified [tournaments.js](file:///Users/jeffbabiak/static/js/tournaments.js) to hide the **Round Standings** leaderboard card entirely for active participants who have not completed their turn.
*   **Selective Replay Censorship**: Modified [app.py](file:///Users/jeffbabiak/app.py) to censor `board_data` and `submitted_words` of other players for active rounds. 
    *   Unplayed participants and spectators/guests see censored placeholder results (`board_data: null`, `submitted_words: []`).
    *   Players who **have finished their turn** can view other players' replays, even while the round remains active, as they can no longer replay their own turn.
*   **Winner Route Protection**: Restricted the `/api/tournament/winner-turn/<tid>/<username>` endpoint in [tournament_logic.py](file:///Users/jeffbabiak/tournament_logic.py) to completed tournaments only (`status == 'completed'`), preventing custom queries from harvesting active round details.

### 2. Unified Replay UI & Visual Board Highlight Paths
*   **Consolidation**: Consolidated the redundant "Snapshot" and "Walkthrough" buttons in the tournament standings and lobby lists into a single, clean **▶ Replay** button.
*   **Board Highlights Restored**: Fixed board rendering and path coordinate extraction inside `watchTournamentReplay` and `watchTournamentWinnerReplay` when board data is in the dictionary format.
*   **Timeline Interactive Paths**: Added an interactive path highlight feature. Clicking any word in the "Timeline of Discovery" review list during replay playback instantly highlights its valid path coordinates on the board grid.

### 3. Tournament Mid-Round Leave Protection
*   **Draft Auto-Saves**: Implemented client-side auto-saves (`/api/tournament/save-draft`) on the `/api/tournament/submit` channel after each word discovery to preserve in-progress score state.
*   **Finalization on Re-entry/Exit**: Automatically creates a score row (`submitted_at = NULL`) when starting a turn. If a user tries to re-enter, refreshes the tab, or navigates away, the server finalizes the turn immediately (`submitted_at = current_time`) and returns a 403 error, preventing timers from resetting.
*   **Stale Turn Auto-Finalization**: Added a 15-second grace buffer to auto-finalize stale turns during tournament status updates if browser closures or disconnects left the turn unsubmitted.

### 4. Touch Drag Selections & Mobile Transposed Coordinates
*   **Swipe Alignment**: Isolated touchscreen and mouse drag selectors using pointer identifiers to prevent touch release failures. Ensure selections submit immediately upon pointer release.
*   **Mobile Transposed Layout Paths**: Fixed word validation highlighting on portrait mobile layout rotations (such as 4x6, 5x7, 6x8) by transposing coordinates.

### 5. Win/Loss UI Score Text Cleanup
*   **Duplicate Elimination**: Removed the duplicate text line `You: X | Opponent: Y` under the "YOU WON! ADVANCING..." and "YOU LOST" panels in [tournaments.js](file:///Users/jeffbabiak/static/js/tournaments.js), keeping the layout clean as the detailed styled scores list is already rendered below it.

---

## 🛠 Active Features & Configuration
*   **Board Formats**: Normal, Checkerboard, Double, Triple, Valued Letters, Rotation, Penalty, Mania, Either/Or, Bonus Word, and Density.
*   **Grid Dimensions**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
*   **Dictionaries**: NWL (American) and CSW (International) Tries.
*   **Verification**: Tested local compilation checks and verified word validator tests pass 100% cleanly.

---

**Latest Stable Commit ID**: `3184446b7a9504af6a31032df4ea8b60fa22513f`  
**GitHub Tag**: `START_OVER_POINT_JUNE_18`  
**Localhost & GitHub Sameness Status**: Synchronized  
**Production Server Status**: Green / PM2 Online / Live at commit `3184446`
