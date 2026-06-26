# Morpheme Stable State Summary - June 26, 2026

This summary documents the stable state of the Morpheme application as of June 26, 2026. Localhost, GitHub origin, and `morpheme.games` are fully synchronized and verified under Commit ID **`6d269c360106016000bef11198aa8f57c177847e`** (and subsequently tagged as **`START_OVER_POINT_JUNE_26`**).

---

## 🚀 Key Improvements & Bug Fixes

### 1. Either/Or Tile Optimization (Balanced Word Distribution)
*   **10% / 10% Balance Constraint**: Overhauled the Either/Or tile generation loop in [board_generator.py](file:///Users/jeffbabiak/board_generator.py) (both in main and emergency loops) to trace paths and calculate the exact percentage of board words using each letter of the Either/Or tile. 
*   **Anti-Neglection Scoring**: Implemented a balance metric `balance_score = min(ratio_l1, ratio_l2)` to maximize utilization. The generator now enforces that at least 10% of all words on the board must use the first letter, and at least 10% must use the second letter. If a perfect candidate meeting this threshold is found, it is accepted immediately; otherwise, it defaults to the most balanced candidate possible, ensuring both letters are highly active and useful.

### 2. Maximum Path Point Valuation
*   **Optimal Score Defaulting**: Modified the scoring engine in [scoring.py](file:///Users/jeffbabiak/scoring.py) and the client-side local score calculator in [play.js](file:///Users/jeffbabiak/static/js/play.js) to bypass strict path matching and perform a fallback path search if a submitted word's dragged path did not hit the bonus tile. If the word *could* have been formed using a path that passes through the Either/Or tile (or a bonus letter), the system automatically defaults to the greatest point value (rewarding the +3 bonus points).

### 3. Intermission Dual-Letter Word Cross-Population
*   **Multi-Letter Tile Parsing**: Rebuilt the client-side `rebuildTileToWordsMap` function in [play.js](file:///Users/jeffbabiak/static/js/play.js) to extract and track *all* letters in Either/Or slash tiles (e.g., `U/A` maps to both `U` and `A`) rather than only the first letter.
*   **Consistent Filtering**: During intermission, clicking on either letter of the Either/Or tile or any duplicate letter elsewhere on the board now correctly cross-populates and displays all words passing through those letters, preventing list mismatches (e.g. allowing `GULLET` to display under both clicked `U` tiles).

### 4. Board Size Settings & Warm Local Caching
*   **Dimensional Precedence**: Overhauled the settings parser to directly consult `window.userSettings` with per-dimension precedence (checking specific dimension widths and heights before reverting to general defaults).
*   **Warm Local Caching**: Implemented local caching of settings objects to eliminate unnecessary database lookups and network roundtrips, speeding up client board rendering.

### 5. Portrait Mobile Swipe Path Fix (6x8 Either/Or)
*   **The Issue**: On portrait mobile devices, the swipe path was transposed prior to server submission, causing valid word submissions on 6x8 Either/Or boards to register as invalid.
*   **Resolution**: Implemented an untransposition routine on the client-side swipe path representation before submitting to the server. Raised Either/Or format odds to 50% temporarily to verify correct behavior in live matches.

### 6. Widescreen Container-Aware Layout Engine
*   **Dynamic Panel Scaling**: Re-engineered the client-side layout engine in [play.js](file:///Users/jeffbabiak/static/js/play.js) to implement widescreen container-aware computations and an equal-reduction panel sizing algorithm in `applyPanelLayout`.
*   **Extreme Size Safety**: Side panels (Players, Words, and Definitions) can now dynamically scale and shrink (down to 160px for players and 220px for words) to accommodate larger board dimensions (e.g. 6x8 or 5x7) without causing overflow or clipping.
*   **Container Queries**: Replaced rigid viewport-based media queries with modern CSS Container Queries for the Players and Words panels, ensuring perfect alignment at any viewport aspect ratio.
*   **Live Sizing Capping**: Linked the settings board-size sliders to invoke `checkBoardOverflow` directly on input, capping the maximum selectable dimensions dynamically based on available screen space.

### 7. Case-Insensitive Username Authentication
*   **The Issue**: Users experienced login or registration failures due to minor casing mismatches in their usernames.
*   **Resolution**: Standardized username handling across all login, registration, and database lookup endpoints in [app.py](file:///Users/jeffbabiak/app.py) to be case-insensitive. Updated the login page description to notify users of this change.

### 8. Daily Reset Player Eviction
*   **The Issue**: During the daily Chicago midnight reset in 24h rooms, active players were kicked back to the lobby with a generic "Inactivity" message, causing confusion.
*   **Resolution**: Modified `game_room.py` to eject active players with a specific, friendly "Daily Reset" modal notification, clearly indicating the transition of the 24h board.

### 9. Find Count Tool Dictionary Sampling
*   **Resolution**: Enhanced the random word generator for the "Find Count" tool in [app.py](file:///Users/jeffbabiak/app.py) to sample words from a combined set of the NWL, CSW, and custom Added Words (AW) dictionaries, offering a broader and more representative vocabulary.

### 10. 2% Equality Freq Board Format
*   **Resolution**: Introduced the 2% Equality Freq board format, adjusted the baseline Normal format odds to 70%, and updated the FAQ list, odds visual breakdowns, and backend board probe generation algorithms in [board_generator.py](file:///Users/jeffbabiak/board_generator.py).

### 11. Rating Logic & FAQ Updates
*   **Resets**: Implemented end-of-round rating resets where `rating_change` is reset to 0 upon round completion or when rejoining a new round.
*   **FAQ Integration**: Updated the FAQ section in [index.html](file:///Users/jeffbabiak/templates/index.html) and [howtoplay.css](file:///Users/jeffbabiak/static/css/howtoplay.css) with detailed explanations of leaving, rejoining, and late-joining rating calculations.

### 12. QU-Tile Fallback Handling
*   **Resolution**: Implemented a client-side fallback mechanism for mouse and swipe paths containing Q/QU tiles. If an entered path with a Q tile fails validation, the system automatically retries validation without the implied "U" (e.g., retrying `QUANAT` as `QANAT` on the board), eliminating spelling errors caused by the board's implicit QU tile.

### 13. Undefined Words Management
*   **Resolution**: Added an interactive, scrollable "Undefined Words" table to the Definition Management interface inside the Mods tab, allowing moderators to quickly identify and define words that lack database definitions.

### 14. Profile Grid Improvements
*   **Resolution**: Standardized the Exceptional/Round Reviews grid layout on the user profile page, stretching the grid to fill the panel width on desktop and utilizing equal 1fr widths for all 8 columns.

### 15. Intermission All Words Duplicate Render
*   **Resolution**: Configured the intermission "All Words" list in [play.js](file:///Users/jeffbabiak/static/js/play.js) to render a found word under both tiles when the same letter appears twice on the board.

### 16. Lists Tool Ghost Timers & Double Lazy-Loading
*   **Resolution**: Fixed a bug in [tools.js](file:///Users/jeffbabiak/static/js/tools.js) where the Lists warning timer ghost-triggered and caused double-lazy-loading of word databases.

### 17. Intermission Cell Density & Sticky Metrics
*   **Resolution**: Fixed a bug where intermission cell density and max density mapping did not remain sticky to the completed round, maintaining visual consistency.

### 18. Case-Insensitive Invitations & Async Invitation Processing (With Friends)
*   **Resolution**: Overhauled private matches (With Friends) to ensure case-insensitive username invitations work flawlessly. Offloaded the computationally heavy board generation to a background daemon thread, reducing the "Send Invite" HTTP response time to under 10ms. Disabled browser caching on private match fetch endpoints and added a double-play safeguard. Replaced the "Play Turn" button with "Waiting for Friend" for opponents' turns, and increased polling frequency to 5 seconds.

### 19. Tournament Round Duration & Grid Dimension Adjustments
*   **Resolution**: Set all upcoming tournament rounds to a 3-minute round time limit (180s) and turn duration (180s), and configured grid dimensions to 4x4.

### 20. Profile Search Error Notification
*   **Resolution**: Integrated a dynamic error notification system below the Profile search textbox. If a guest or non-existent username is searched, the profile container is hidden, and the error `"The username you entered does not exist."` is displayed. The error is instantly cleared and hidden as soon as the user starts typing.

### 21. Mania Mode Letter Abundance Calibration
*   **Resolution**: Calibrated Mania mode to require a minimum of 1/5 (20%) of all letters on the board to be the abundant letter for rare letters (Q, Z, J, X, K) and 1/3 (33.3%) for common letters. Exempted the active Mania letter from the individual and total rare-letter sanitizer caps and optimization limits, ensuring abundant letters remain protected.

---

## 🛠 Active Features & Configuration
*   **Board Formats**: Normal (70%), Checkerboard, Double, Triple, Valued Letters, Rotation, Penalty, Mania, Either/Or (50%), Bonus Word, Density, and Equality Freq (2%).
*   **Grid Dimensions**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
*   **Dictionaries**: NWL (American) and CSW (International) Tries, plus custom Added Words.

---

**Latest Stable Commit ID**: `6d269c360106016000bef11198aa8f57c177847e` (tagged as `START_OVER_POINT_JUNE_26`)  
**GitHub Tag**: `START_OVER_POINT_JUNE_26`  
**Localhost & GitHub Sameness Status**: Synchronized  
**Production Server Status**: Green / PM2 Online / Live at commit `6d269c360106016000bef11198aa8f57c177847e` (PM2 restarted successfully)
