# Stable State Summary - April 2, 2026

This artifact summarizes the extensive stability work performed on the Morpheme board generation engine and the game UI to ensure consistent density and difficulty.

## 1. Board Generation Engine (board_generator.py)
The core engine has been upgraded to its most intensive "Master Level" search capability to handle extreme density requirements (7L+ and 8L+ words on 6x8 grids).

*   **A/U Dominance & Vowel Glue**:
    *   **'A' weight** increased to **850 (Hard)** and **1500 (Easy)**.
    *   **'U' weight** increased to **600** across all tables.
    *   'E' dominance has been reduced to allow 'A' to serve as the primary long-word connector.
*   **"Master Level" Search Intensity**:
    *   **IO Passes**: Increased to **25 passes** for 8-letter targets and **18 passes** for 7-letter targets.
    *   **Max Attempts**: Increased to **150 attempts** for "Extreme" configurations (8L or 7L Checkerboard).
*   **Structural Logic Fixes**:
    *   **Mania Format Integration**: Moved Mania "Flooding" **before** the optimization passes. This ensures the generator builds word counts *around* the mania letters instead of overwriting the word count after optimization.
    *   **Checkerboard Pattern Awareness**: IO optimization now strictly honors V-C-V-C patterns, testing only valid letter types (Vowels on vowel cells, etc.).
    *   **Best-in-Class Tracking**: Implemented global "Best Board Found" tracking. If a 120-attempt search fails to hit a 100-word target but hits 85 words, it now returns the 85-word board instead of failing back to a 20-word generic board.
    *   **Vowel Ratio Guard**: Increased allowed vowel ratio to **50%** for high-density rounds to provide the "vowel glue" needed for 7-10 letter words.

## 2. API & State Management (app.py)
*   **Uniqueness Integration**: Re-implemented `current_uniqueness` and `next_round_uniqueness` in the API response.
*   **Return Signature Stability**: Ensured `generate_board` always returns exactly 6 values, even in fallback and cube paths, to prevent `ValueError` crashes in Room Manager.

## 3. Game UI (play.js & spinner_set.py)
*   **Header Uniqueness Percentage**: Fixed the field mapping for the `param-diff` ID. The header now correctly displays the calculated difficulty percentage (e.g., `Diff: Hard (72%)`) for both active rounds and intermission reveals.
*   **Spinner Target Capping**: Fixed a logic error in `SpinnerSet.py` to prevent "impossible" targets (like 200+ words of 7-letter minimums) from being generated in new tournaments.
*   **Parameter Reveal Animation**: Verified that the 'Gold Fade' animation correctly triggers at the 45s mark during intermission for all round configurations.

## 4. Current Audit Results (10-Round Stress Test)
While stability has improved, the following baseline has been established for 6x8 grids:
*   **Success (100-200, 6L)**: Consistently producing 150-350 words.
*   **Success (Mania, 7L)**: Improved to ~100 words (now that flooding is pre-optimized).
*   **Challenge (8L, 200+)**: Hit ~66-100 words (Best Found). This remains the extreme edge case of the mathematical grid.

> [!TIP]
> To continue work tomorrow: The **vowel weighting** should be considered the primary handle for connectivity. If 8L words are still sparse, focus on the **horizontal/vertical island seeding** in `_create_normal_board`.

> [!IMPORTANT]
> The engine is currently in its most computationally intensive state. If generation times increase beyond server timeouts, consider lowering `max_attempts` from 150 to 100.
