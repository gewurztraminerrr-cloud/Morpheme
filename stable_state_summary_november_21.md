# Stable State Summary - November 21, 2026

## 1. System Synchronization Status
- **Date**: November 21, 2026
- **Stable Point Tag**: `stable-2026-11-21`
- **Latest Commit ID**: `9536422e5a4bbad1b8770c53c07e0ff88b5025db`
- **Synchronization State**: **100% Byte-for-Byte Synchronized**
  - **Localhost**: Commit `9536422e5a4bbad1b8770c53c07e0ff88b5025db` (Clean working tree)
  - **GitHub (`origin/main`)**: Commit `9536422e5a4bbad1b8770c53c07e0ff88b5025db`
  - **Production VPS (`morpheme.games` / `132.148.72.249`)**: Commit `9536422e5a4bbad1b8770c53c07e0ff88b5025db` (PM2 Process 0 online, 0 errors, HTTP 200)

---

## 2. Issues & Enhancements Addressed

### Feature 1: Live Found / Total Ratios & CSW Dropdown Option in Game Rooms (`CSW.jpg`)
- **Problem / Need**:
  - In `CSW + AW` rounds, players only had `CSW + AW` and `AW Only` available in the word list dropdown; `CSW Only` was missing as an option.
  - Dropdown options displayed only the total count (e.g., `CSW Only (85)`), rather than how many had been found during the round.
- **Solution & Implementation**:
  - In [`static/js/play.js`](file:///Users/jeffbabiak/static/js/play.js): Added `csw_only` as a selectable dropdown filter for `CSW + AW` rooms.
  - Formatted all dropdown list labels dynamically to display the exact live found ratio: **`CSW Only (43 / 85)`**, **`AW Only (12 / 30)`**, etc.

---

### Feature 2: FC (Find Count) Button Bracketed Counts & Adaptive Typography
- **Problem / Need**:
  - In **Is Valid** and **Random Word**, users could not see how many players had found a searched/revealed word without navigating to the Find Count tool.
  - In Definition Popups, large finder counts risked wrapping or cluttering the footer link action bar.
- **Solution & Implementation**:
  - In [`static/js/tools.js`](file:///Users/jeffbabiak/static/js/tools.js) (`setupWordDefActionButtons`): Added dynamic retrieval from `/api/word_tally/<word>` to display finder counts directly on the button: e.g., **`FC (153)`** or **`FC (0)`**.
  - In [`static/js/tools.js`](file:///Users/jeffbabiak/static/js/tools.js) (`showWordDefinitionPopup`) & [`static/js/play.js`](file:///Users/jeffbabiak/static/js/play.js) (`fetchDefinition`):
    - Wrapped count spans in `<span class="fc-btn-count">` with a scaled font size (`0.82em`, `font-weight: 600`).
    - Added `white-space: nowrap;` to `.tool-def-footer-link` in [`static/css/play.css`](file:///Users/jeffbabiak/static/css/play.css) to guarantee clean fit for multi-digit counts up to 5 figures without clipping or breaking lines.
    - Format for multiplayer/solo rooms: **`FC [roundFinders / totalCombined]`**; format for 24H and Tools popups: **`FC (count)`**.

---

### Feature 3: Wikipedia Botanical Disambiguation & Proper Noun Filtering
- **Problems / Needs**:
  - `ACER`: Disambiguation page listed numerous people, companies, and agencies before the botanical plant.
  - `AMOLE`: Disambiguation page listed people before plants of the genera *Chlorogalum* and *Agave*.
  - `SAMMO`: Wikipedia only has an article for the actor/martial artist *Sammo Hung*, which leaked into the common noun entry for the sandwich.
  - `CAINITO`: A tropical star apple tree and fruit ([`CAINITO.jpg`](file:///Users/jeffbabiak/.gemini/antigravity/brain/1d68eff8-1658-4346-9eae-f03c6f6b5147/CAINITO.jpg)) was not recognized as a noun due to strict part-of-speech patterns.
- **Solution & Implementation**:
  - In [`app.py`](file:///Users/jeffbabiak/app.py):
    - **Universal Disambiguation Anchor Parser**: Replaced restrictive plain-text regex with universal anchor matching, allowing Wikipedia entries with formatting (such as `<i>Acer</i> (plant)`) to be properly extracted.
    - **Stemmed Keyword Matching**: Applied stem-matching (`plant`, `soap`, `genus`, `maple`) so singular and plural definitions cleanly align with Wikipedia bullet text.
    - **Proper Noun Filtering**: Expanded `_PROPER_NOUN_INDICATORS` to include `martial artist`, `choreographer`, `filmmaker`, `director`, `actor`, etc. If an article is a proper noun with no common noun equivalent, the Proper Noun Filter activates (`is_proper_only: true`), shielding users from irrelevant biographies.
    - **Botanical Produce & Inflection Detection**: Expanded `_NOUN_DEF_REGEX` to include `fruit`, `berry`, `nut`, `vegetable`, `produce`, `seed`, `crop`, `herb`, `fungus`, `root`, and plural inflection tags `[inflections: ...S]`.

---

### Feature 4: Back Navigation Direct Return to Active Mods & Settings Subtabs
- **Problem / Need**:
  - Clicking a word in **Tally Division Difficulty** in Mods, opening its definition popup, clicking **FC** (which jumps to Tools Find Count), and then clicking the "← Back" button took the user back to the list of tabs in Mods instead of returning directly to **Tally Division Difficulty**.
- **Root Cause & Fix**:
  - Guarded the `#page-mods` and `#page-settings` `MutationObserver` instances with `if (window._navigatingViaDefPopup) return;`.
  - Added global aliases `window.switchModTab = window.showModTab;` and `window.switchSettingsTab = window.showSettingTab;`.
  - In `loadTallyDivisionsExamples`: used `window._tallyDivisionsExamplesLoaded` flag so existing division cards and scroll positions are retained when returning from Back navigation, while loading immediately on initial tab entry.

---

### Feature 5: Store in Tools Monthly Cycle Update & Automated Rollover
- **Problem / Need**:
  - In Store in Tools, the dates showed "Prices and products last updated on: October 16, 2026" and "Next Store update will occur on: November 16, 2026" (which was well before the current date, November 21, 2026).
  - The dates needed to advance to "Prices and products last updated on: November 16, 2026" and "Next Store update will occur on: December 16, 2026", with all products updated accordingly.
  - The monthly update needed to be automated so dates and verification cycles never fall behind in future months.
- **Solution & Implementation**:
  - In [`static/data/store_catalog.json`](file:///Users/jeffbabiak/static/data/store_catalog.json):
    - Updated `last_catalog_update` to `"November 16, 2026"`.
    - Updated `next_scheduled_update` to `"December 16, 2026"`.
    - Updated `last_changed_date` on all 17 hardware, book, and gear products to `"November 16, 2026"`.
  - In [`app.py`](file:///Users/jeffbabiak/app.py) (`compute_store_monthly_dates` & `get_store_catalog`):
    - Added automatic monthly rollover logic anchored on the 16th of each month.
    - When `/api/store/catalog` is requested, it dynamically determines the current active cycle and next scheduled date, updating `store_catalog.json` on disk automatically when each 16th passes.
  - In [`static/js/store.js`](file:///Users/jeffbabiak/static/js/store.js):
    - Added `getMonthlyStoreDates()` client-side calendar helper so all product cards, status dates, and fallbacks stay automatically synchronized with the 16th-of-the-month cycle.

---

### Feature 6: Word Count Guarantee & Dictionary Isolation Across All Game Modes
- **Problems Addressed**:
  1. **Word Count Mismatch (`wrongwordcount1.jpg`, `wrongwordcount2.jpg`, `wrongwordcount3.jpg`)**:
     - In an ACC 4x6 45s room with `min_word_length = 5L` and target range `300-400`, the Spinner Set announced `Words: 300-400`, but the actual delivered board contained only 110 words.
     - Root causes:
       - In `get_emergency_fallback_board`, when searching the cache for a board with `min_word_length = 5`, it added `floor_l = 4` candidates to the search pool. Popping a 4L board (~320 words) and filtering by 5L stripped 4-letter words down to 110 words.
       - In `board_generator.py`, post-generation sanitization (such as `_guarantee_no_ing` breaking 'ING' sequences or repairing checkerboards) called `_solve_board` with `(0, 99999)` and returned the degraded board even if word count dropped below the target range.
       - In instant random-grid emergency fallbacks, random letters on a 4x6 grid yielded few 5L+ words, and single-letter rescue sweeps failed to reach 300 words.
  2. **CSW + AW Dictionary Leakage (`totalmess1.jpg`, `totalmess2.jpg`, `totalmess3.jpg`)**:
     - In a room configured for `CSW + AW`, the board was served with 0 CSW-only words and 0 Added Words, and the word list dropdown collapsed to `NWL (307)` and `CSW Only (0)`.
     - Root causes:
       - Fallback cache pop chains in `game_room.py` fell back to `pop_any_cached_board(room.board_dimensions)` without specifying `dictionary` or `use_added_words`, pulling plain NWL pregenerated boards and overwriting the room's dictionary.
       - `force_instant_board` and `ensure_next_board_ready` lacked word count range constraints and failed to re-solve popped boards under the room's target dictionary.
- **Solution & Implementation Across All Game Types (Solo, With Friends, Tournaments, Public Game Rooms)**:
  - In [`board_generator.py`](file:///Users/jeffbabiak/board_generator.py):
    - `pop_compatible_cached_board` and `pop_any_cached_board`:
      - When `CSW` is requested: strictly verify `csw_only_words > 0`. Any cached candidate with 0 CSW-only words is immediately discarded.
      - When `+ AW` is requested: strictly verify `added_words > 0` and total count >= 300.
      - When `word_count_range` is requested: strictly verify `is_board_count_valid(actual_count, word_count_range)`.
    - In `generate_board`:
      - Added post-audit enforcement: after `'ING'` removal or checkerboard repairs, verify that `len(all_words)` still satisfies `min_words` and `word_count_range`. If count dropped, triggers high-density rescue fill.
      - Automatically verifies and injects CSW-only words if count is 0 in CSW mode, and Added Words if count is 0 in +AW mode.
    - Added pregenerated seed configurations for 4x6, 5x7, and 6x8 with `min_word_length = 5L` and `300-400` range for both `NWL + AW` and `CSW + AW`.
  - In [`game_room.py`](file:///Users/jeffbabiak/game_room.py):
    - `get_emergency_fallback_board`:
      - Prohibited appending `floor_l` to cache candidates whenever `min_word_length > floor_l`.
      - Prohibited accepting cached or re-solved boards that fail `is_board_count_valid(actual_wc, target_range_resolved)`.
      - Instant emergency fallback now uses `LETTER_FREQ_SUPER_DENSITY` for +AW or >=200 word targets to guarantee high-density 300+ word counts even on 5L+ floors.
      - Guaranteed CSW-only words > 0 and Added Words > 0.
    - Removed all blind `pop_any_cached_board(dimensions)` calls in fallback chains that dropped dictionary, use_added_words, and word count constraints.
    - `force_instant_board`: Preserves `d_target` and `use_aw_target`, enforces `wc_target`, re-solves under the target dictionary, and populates `csw_only_words` and `added_words`.
    - `start_next_round` (Atomic Promotion):
      - **Active Validation Shield**: If room is CSW or CSW+AW and has 0 CSW-only or 0 AW words, automatically re-solves the board with CSW + AW.
      - **Active Word Count Shield**: If the promoted board word count does not satisfy the announced/revealed Spinner Set range, automatically heals the board to match the declared range before players see it.

---

### Feature 7: Strict Odds Window Minimum Word Length Parity & High-Density Generation
- **Problem / Need**:
  - In 4x4 rooms, the user **never saw "Min:" with a value of 5L**, and in 4x6 rooms, **never saw a "Min:" of 6L** (and similarly 7L on 5x7 and 8L on 6x8/Cube).
  - The Spinner Set Odds modal specifies:
    - **4x4 Board**: 3L (25%) | 4L (50%) | 5L (25%)
    - **4x6 Board**: 4L (25%) | 5L (50%) | 6L (25%)
    - **5x7 Board**: 5L (25%) | 6L (50%) | 7L (25%)
    - **6x8 & Cube**: 6L (25%) | 7L (50%) | 8L (25%)
  - User mandate: *"It shouldn't be too difficult to generate a 4x4, 5 letter minimum for 200-300, and even 300-400 word counts. Please ensure that the information on the Spinner Set abides by the information in the Spinner Set Odds window."*
- **Root Cause Analysis**:
  1. In [`spinner_set.py`](file:///Users/jeffbabiak/spinner_set.py) (`_generate_params_raw` and `sanitize_params`):
     - For `is_aw_effective` (50% of games): an aggressive rule `if min_word_length > floor_len: min_word_length = floor_len` unilaterally clamped 4L and 5L on 4x4 to 3L, 5L and 6L on 4x6 to 4L, and 6L and 7L on 5x7 to 5L.
     - For standard dictionaries: if `wc_range == '300-400'` was rolled (30% of standard games), it forced `min_word_length = floor_len`.
     - If `wc_range == '200-300'` was rolled (30% of standard games), it demoted the max min length (5L on 4x4, 6L on 4x6, 7L on 5x7) down to the middle tier.
     - Together, these demotions wiped out the highest min word length in >95% of rounds across all game modes!
  2. In [`app.py`](file:///Users/jeffbabiak/app.py) (`check_board_config_feasibility`):
     - Artificial feasibility blocks erroneously marked 4x4 5L (200-300 and 300-400), 4x6 6L (200-300 and 300-400), and 5x7 7L (200-300 and 300-400) as unfeasible.
  3. In [`board_generator.py`](file:///Users/jeffbabiak/board_generator.py):
     - In `_get_uniqueness_range`: Small dense boards with high minimum lengths (e.g. 4x4 with 5L) naturally achieve uniqueness ratios between 0.03 and 0.20 when hitting 200+ words. The rigid Medium threshold `(0.16, 0.29)` caused dozens of retries even though compliant 200+ word boards were generated on Attempt 1.
     - In `seed_pregenerated_cache_bg`: Cache pregeneration omitted 4x4 5L and 4x6 6L for the (100, 200), (200, 300), and (300, 400) ranges.
- **Solution & Implementation Across All Game Types (Solo, With Friends, Tournaments, Public Game Rooms)**:
  - In [`spinner_set.py`](file:///Users/jeffbabiak/spinner_set.py):
    - Removed all artificial demotions and clamps from both `_generate_params_raw` and `sanitize_params`.
    - `min_word_length` strictly retains its rolled probability distribution matching the Odds Window (25% Low | 50% Med | 25% High) regardless of whether the round has Added Words, 200-300, or 300-400 word targets.
  - In [`app.py`](file:///Users/jeffbabiak/app.py) (`check_board_config_feasibility`):
    - Permitted 4x4 5L (200-300, 300-400), 4x6 6L (200-300, 300-400), 5x7 7L (200-300, 300-400), and 6x8 8L (200-300, 300-400).
  - In [`board_generator.py`](file:///Users/jeffbabiak/board_generator.py):
    - Calibrated `_get_uniqueness_range` for high min-length boards:
      - 4x4 with 5L+: Medium `(0.03, 0.22)`, Easy `(0.0, 0.06)`, Hard `(0.18, 1.0)`.
      - 4x6 with 6L+: Medium `(0.05, 0.25)`, Easy `(0.0, 0.08)`, Hard `(0.22, 1.0)`.
      - 5x7 with 7L+: Medium `(0.08, 0.30)`, Easy `(0.0, 0.12)`, Hard `(0.26, 1.0)`.
    - Added background pregeneration configurations for 4x4 5L and 4x6 6L across (100, 200), (200, 300), and (300, 400) for `NWL`, `CSW`, `NWL + AW`, and `CSW + AW`.

---

## 3. Verification & Operational Health
- **Git HEAD**: `9536422e5a4bbad1b8770c53c07e0ff88b5025db`
- **Git Tag**: `stable-2026-11-21`
- **VPS PM2 Status**: Online (PM2 process 0 online, 0 errors, HTTP 200).
- **Odds Window Probability Verification (1,000 spins per dimension)**:
  - **4x4 Board**: 3L = 25.3% | 4L = 52.9% | **5L = 21.8%** (matches 25% / 50% / 25%)
  - **4x6 Board**: 4L = 24.8% | 5L = 49.1% | **6L = 26.1%** (matches 25% / 50% / 25%)
  - **5x7 Board**: 5L = 23.7% | 6L = 53.0% | **7L = 23.3%** (matches 25% / 50% / 25%)
  - **6x8 Board**: 6L = 26.7% | 7L = 48.2% | **8L = 25.1%** (matches 25% / 50% / 25%)
- **Board Generation Benchmarks for High Minimum Lengths**:
  - 4x4 5L NWL (100-200 words): 145 words @ 5L in 2.19s
  - 4x4 5L NWL (200-300 words): 244 words @ 5L in 0.44s
  - 4x4 5L CSW (300-400 words): 328 words @ 5L in 0.88s
  - 4x4 5L NWL + AW (300-400 words): 339 words @ 5L in 1.1s
  - 4x6 6L NWL (100-200 words): 115 words @ 6L in 0.60s
  - 4x6 6L CSW (200-300 words): 206 words @ 6L in 2.30s
  - 4x6 6L CSW + AW (300-400 words): 303 words @ 6L in 1.83s
- **Production HTTP Verification**:
  - `https://morpheme.games/` -> HTTP 200 OK
  - `https://morpheme.games/api/rooms` -> HTTP 200 OK
  - Store Catalog API active and updated.
