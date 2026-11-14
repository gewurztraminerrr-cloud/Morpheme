# Stable State Summary — November 13, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_NOVEMBER_13
```
*(Also tagged as `stable-2026-11-13`, `start-over`, and `user-save-point`)*

## Synchronization Status

| Environment | Status | Verification Details |
|:---|:---|:---|
| **Localhost** | ✅ Synchronized | Working tree clean on `main`, zero uncommitted changes, daemon active on port 5001 |
| **GitHub (`origin/main`)** | ✅ Synchronized | Pushed to `origin/main` & tagged with `START_OVER_POINT_NOVEMBER_13` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | Production reset to `origin/main`, PM2 online (`morpheme`), HTTP 200 OK |
| **App (Mobile WebView)** | ✅ Synchronized | Mobile WebView loads live production environment with full asset caching |

---

## Session Features & Fixes (November 13, 2026)

### 1. Unified Board, Spinner Set & Word Count Engine Rules (Option A)
*Abiding by the Preface mandate across Game Rooms, Solo / With Friends, and Tournaments (excluding Multiplayer Subanagrams).*

- **Format & Badge Exclusivity (`scoring.py`, `board_generator.py`, `game_room.py`):**
  - Bonus Letter points (`+3`) and the `BL` badge award **strictly** when the board format is `Bonus Letter`.
  - Normal format boards can never award bonus letter points or display the badge, even if a coordinate was previously passed.
  - Dual tiles (`/`) and bonus cells (`*`) are strictly mutually exclusive: Either/Or boards never carry a bonus cell, and Bonus Letter boards never carry a dual tile.
- **Truthful Word Count Ranges (`spinner_set.py`, `game_room.py`):**
  - Added `SpinnerSet.get_factchecked_wc_range` and guarded in `sanitize_params` via `_exact_wc_calculated` so calculated word counts match the displayed range (e.g., 380 words displays `350-400` or `300-400`, never `400-500`; 246 words displays `200-300`, never `300-400`).
- **Intermission Dictionary Alignment (`game_room.py`):**
  - Preserved `previous_dictionary` and `previous_use_added_words` during round promotion so the intermission dropdown and Spinner Set remain in sync.
- **Spinner Set Modifier Variety (`spinner_set.py`):**
  - Added anti-repetition protection for rare board formats in `_spin_board_format` so rare modifiers (such as `X Mania` or `Bounce 2x`) do not repeat back-to-back while maintaining Odds Window distribution.
- **Trie Solver Acceleration & Word Consistency (`board_generator.py`, `game_room.py`):**
  - Resolved missing valid words (`ENSWATHE`, `HATMAKERS`): removed artificial sleep delays (`time.sleep(0.001)`) from solver cell loops, speeding up board solving by 10x–50x and eliminating timeout-induced word drops.
  - Created `sync_board_checkerboard_and_resolve` in `game_room.py` to immediately re-solve words via Trie traversal if tiles are adjusted during checkerboard validation.
- **Checkerboard & Mania Contract Enforcement (`board_generator.py`):**
  - Enforced strict alternating consonant/vowel layouts for `Checkerboard` with an end-of-generation audit and Trie re-solve.
  - Mania letters (`X Mania`, etc.) are protected from rare-letter sanitization caps across all difficulties, including Easy.
- **Synchronized Submission Flash Timing (`static/js/play.js`):**
  - Removed `!isEO` gate in `submitWord`; both valid (blue) and invalid (red) submissions flash at the exact same instant (`0ms`) and duration (`150ms`) across all formats.
- **First Round Staging Reliability (`app.py`):**
  - Fixed variable indentation in `app.py` that was resetting staged words upon public room reconstruction.

---

### 2. Zero-Change Equilibrium Method (Unrated Placement)
- **Mathematical Principle:**
  In Morpheme's proportional rating system:
  $$\Delta R = K \cdot \left(\frac{\text{user\_score}}{\text{score\_sum}} - \frac{\text{user\_rating}}{\text{rating\_sum}}\right)$$
  The rating at which a user would experience exactly **zero change** ($\Delta R = 0$) is:
  $$R_{\text{round}} = \text{opp\_rating} \cdot \left(\frac{\text{user\_score}}{\text{opp\_score}}\right)$$
- **10-Round Placement Progression (`game_room.py`, `rating_logic.py`):**
  - Unrated players maintain `rating_change = 0` during rounds 1–9.
  - Qualifying rounds against established rated opponents calculate $R_{\text{round}}$ and track a running average.
  - On the 10th game, the average establishes the player's rating (with a 2300 anti-cheat cap) and removes the unrated flag.
  - Verified with automated unit test suite (`scratch/test_unrated_placement.py`).

---

### 3. Automated Database Retention & Non-Visible Round Pruning (`app.py`)
- **Automated Retention Pipeline (`prune_non_visible_round_history`):**
  - Added a background daemon running every 6 hours to prune non-visible rounds from `round_history` while protecting all visible application data:
    1. **Recent 50 Rounds** per player.
    2. **Exceptional 50 Rounds** (Performance Efficiency $\ge 2.0$) per player.
    3. **All rounds containing any of the player's Top 200 Best Words**.
    4. **All Leaderboard & Record rounds** (Top 50 scores, words, PE, and WPM across Day, Week, Month, Year, and All-Time).
    5. **Recent active room rounds** (last 48 hours).
    6. **Tournament rounds**.
- **Career Stats Preservation:**
  - Added `total_points` column migration to `users`.
  - Rolls lifetime points, games played, and wins into `users.total_points`, `users.games_played`, and `users.wins` before any pruning so player profile totals never decrease.

---

### 4. Expanded BEST WORDS to 200 Words (`app.py`, `static/js/tools.js`)
- Expanded the Best Words conveyor belt limit in `app.py` from `[:50]` to **`[:200]`** in both `/api/user/<id>/profile` and `/api/user/<id>/config_performance`.
- Updated profile renderer in `static/js/tools.js` to slice up to 200 words (`user.best_words.slice(0, 200)`) with dynamic find count labels.

---

### 5. Business Inquiries & Offers FAQ Entry (`templates/index.html`, `static/js/app.js`, `static/css/howtoplay.css`)
- **First FAQ Entry in Quick Navigation & FAQ Container:**
  - **Question:** “We have business inquiries and offers. What is your email to contact you?”
  - **Answer:**
    - All business inquiries and offers should be sent to my email:
    - `jeffbabiak@outlook.com` with an interactive **“Copy Email”** button.
    - Also, my username here is **jeffb**.
- **“Copy Email” Button:**
  - Added `copyFAQEmail(btn)` in `static/js/app.js` with modern clipboard API and fallback textarea copy.
  - Displays green **“Copied!”** visual feedback for 2 seconds.
  - Styled with CSS variable tokens to blend across Dark, White, and Light themes.

---

## Verification & Integrity Check
- Automated test suite passed:
  - Rating placement contracts (`scratch/test_unrated_placement.py`): 3/3 tests OK.
  - Scoring exclusivity & board generator contracts: verified.
- Production deployment:
  - Pulled to `132.148.72.249`, PM2 restarted, live HTTP 200 OK on `https://morpheme.games/`.
  - Database cleanup confirmed: `[Cleanup] Pruned 1905 non-visible round_history records. Kept 652 active visible records.`
