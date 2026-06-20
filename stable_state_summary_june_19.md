# Morpheme Stable State Summary - June 19, 2026

This summary documents the stable state of the Morpheme application as of June 19, 2026. Localhost, GitHub origin, and `morpheme.games` are fully synchronized and verified under Commit ID **`7ca8e67`** (and subsequently tagged as **`START_OVER_POINT_JUNE_19`**).

---

## 🚀 Key Improvements & Bug Fixes

### 1. 24h Room Resets & Ejections
*   **Chicago Midnight Transition**: Modified `check_and_update_state` in [game_room.py](file:///Users/jeffbabiak/game_room.py) to clear the active players and spectators roster (`self.players = []`, `self.spectators = []`) at Chicago's 12 AM (midnight) boundary.
*   **Client Ejection**: Configured `updateGameState` in [play.js](file:///Users/jeffbabiak/static/js/play.js) to detect the daily reset, eject active clients back to the Lobby, clear the saved `last_joined_room` from localStorage, and trigger a custom daily reset information modal.
*   **Intermission Stats Sync**: Reconstructs and preserves yesterday's statistics (`previous_total_words` and `previous_total_points`) during server-side room reconstruction by querying the SQL database.
*   **On-Demand Kickstart**: Modified the API room route in [app.py](file:///Users/jeffbabiak/app.py) to automatically spawn a background thread to generate the daily board if a reconstructed 24h room has no board.

### 2. Active Rooms Rating Filter
*   **Blank Default**: Changed the average rating filter textbox (`#rating-filter`) under the Active Rooms panel to load empty (`""`) by default instead of pre-populating with `1200`.
*   **My Rating Button**: Restored the button's action in [lobby.js](file:///Users/jeffbabiak/static/js/lobby.js) to fill the textbox with the user's config-specific rating on demand.

### 3. Solo Game Setup Options
*   **10 Minutes Option**: Added `<option value="600">10 Minutes</option>` to the Solo game Time Limit selector in [index.html](file:///Users/jeffbabiak/templates/index.html).
*   **Format Label Cleaning**: Removed the parenthetical `(14% special)` from the "Random" format option text in Solo's Board Format dropdown.

### 4. Store Layout Updates
*   **Category Tabs Removal**: Removed the store category navigation tab buttons ("Hardware", "Themes", "Avatars", and "Perks") from the Store page in [index.html](file:///Users/jeffbabiak/templates/index.html) completely.
*   **Immediate Grid Display**: Displays all store products (starting with the Yacig 4-in-1 stylus) directly below the "Morpheme Store" title.
*   **JS Fallback**: Ensures the tab-switching listener in [app.js](file:///Users/jeffbabiak/static/js/app.js) exits gracefully when tabs are absent, maintaining standard flex layouts.

### 5. Instant Valid Word Points Feedback
*   **Client-Side Score Calculator**: Implemented `calculateWordScoreLocally` in [play.js](file:///Users/jeffbabiak/static/js/play.js) replicating base scores, valued letter counts, hidden bonus words, and special tile modifiers.
*   **Instant Message**: Display of validation feedback (e.g. `[word] VALID ([points] PTS)`) is now instantaneous, avoiding the previous quick flash transition from `PASSER VALID` to `PASSER VALID (3 PTS)`.

### 6. Guest Round History and Tally Synchronization
*   **The Issue**: Guest player round results and submissions in standard rooms were previously excluded from the `round_history` database and the `word_tally.log` (only 24h rooms allowed guests). This caused a discrepancy where intermission screens correctly showed all connected players (including guests) who moused a word, but the "Find Count" tool reported fewer finds because it only queried database records.
*   **Resolution**: Updated player snapshot filters in [game_room.py](file:///Users/jeffbabiak/game_room.py) to save and log rounds for guest players (`p.is_registered or p.is_guest`) across all room types.

---

## 🛠 Active Features & Configuration
*   **Board Formats**: Normal, Checkerboard, Double, Triple, Valued Letters, Rotation, Penalty, Mania, Either/Or, Bonus Word, and Density.
*   **Grid Dimensions**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
*   **Dictionaries**: NWL (American) and CSW (International) Tries.

---

**Latest Stable Commit ID**: `7ca8e67` (tagged as `START_OVER_POINT_JUNE_19`)  
**GitHub Tag**: `START_OVER_POINT_JUNE_19`  
**Localhost & GitHub Sameness Status**: Synchronized  
**Production Server Status**: Green / PM2 Online / Live at commit `7ca8e67`
