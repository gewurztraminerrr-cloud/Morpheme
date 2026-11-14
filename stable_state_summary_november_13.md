# Stable State Summary — November 13, 2026 (Start Over Point)

## Start Over Point Tag & Commit ID
```
Tag:       START_OVER_POINT_NOVEMBER_13
Commit ID: 841f4d3cd251296fe7d47225c567844aa1d25c60
Short ID:  841f4d3c
```
*(Also tagged as `stable-2026-11-13`, `start-over`, `latest-stable-state`, and `current-stable-state`)*

## Synchronization Status

| Environment | Status | Verification Details |
|:---|:---|:---|
| **Localhost** | ✅ Synchronized | Working tree clean on `main`, zero uncommitted changes, verified HTTP 200 on `http://localhost:5001/` |
| **GitHub (`origin/main`)** | ✅ Synchronized | Pushed to `origin/main` at `841f4d3c` & tagged with `START_OVER_POINT_NOVEMBER_13` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | Reset hard to `origin/main` at `841f4d3c`, PM2 online (`morpheme`), verified HTTP 200 on `https://morpheme.games/` |
| **App (Mobile WebView)** | ✅ Synchronized | Mobile WebView loads live production environment with full asset caching (Build `33204`) |

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

### 6. Intermission 0:10 Vibration & Bell Alert Reliability (`static/js/play.js`, `static/js/settings.js`)
- **Intermission Alert Threshold Reset Architecture:**
  - Previously, `hasPlayedIntermissionBell` and `hasPlayedIntermissionVibration` flags set at 0:10 were not reset if the local timer skipped or if the next intermission was entered with `remaining > 10.5` seconds, leaving the flags permanently `true`.
  - Added `checkAndTriggerIntermissionAlerts(remaining)` in `static/js/play.js` that explicitly resets both flags whenever `remaining > 10.5` seconds or when not in intermission, guaranteeing every round starts with fresh flags.
  - Integrated `checkAndTriggerIntermissionAlerts(rem)` into `syncTimerWithServer` so mobile background/timer throttling cannot skip the 0:10 trigger window.
- **Triple-Redundant Mobile Audio Pipeline:**
  - Added Web Audio API `AudioBuffer` caching and decoding to `BoardAudio` (`preloadBell`, `playBellBuffer`) to play bell notifications via the unblocked `AudioContext` keep-alive stream.
  - Primed HTML5 `Audio` element (`window.intermissionBellAudio`) during initial user interactions (`unlockAudio`).
  - Native fallback via `MorphemeAudioBridge.postMessage` for Flutter / mobile wrappers.
- **Enhanced Mobile Vibration:**
  - Robust vibration fallback (`navigator.vibrate([400, 150, 400])` -> single pulse `500ms` / `300ms` -> `MorphemeAudioBridge`).

---

### 7. Definition Popup for Found Words in Intermission Notepad (`static/js/play.js`, `static/js/tools.js`, `static/css/play.css`)
- **Interactive Notepad Word Definitions:**
  - When opening a player's notepad modal during intermission (by clicking their rectangle under Players), clicking on any found word line now triggers the global definition popup (`showWordDefinitionPopup` / `lookupWord`) with full word metadata (length badge, NWL/CSW/AW dictionary membership badges, pronunciation, clickable inflection links, image illustrations, Combo Checker, Find Count, and Words to Work On actions).
  - Previously, clicking a word line called `closeNotepadPopup()` immediately. Now, the notepad stays open and smoothly highlights the selected word (`.notepad-word-line.selected`).
- **Responsive Adjacent / Overlay Positioning (`tools.js`):**
  - Enhanced `showWordDefinitionPopup` smart positioning: when triggered from inside `.notepad-card`, it detects available viewport space on desktop/tablet (`window.innerWidth >= 768px`) and places the popup adjacent to the notepad card (to the right if space permits, or to the left), keeping the notepad of words and the definition completely unobstructed. On mobile / narrow screens, it centers and clamps gracefully over the notepad.
- **Robust Outside-Click & Lifecycle Handling:**
  - Added `.notepad-word-line` and `.notepad-scroll-btn` exclusions to the popover document click listener so tapping another word or scrolling lines in the notepad smoothly transitions definitions without dismissing the popup.
  - Calling `closeNotepadPopup()` or transitioning into an active round (`state.state === 'active'`) automatically cleans up both the notepad modal and the definition popup.
- **Styling & Layout Streamlining (`play.js`, `play.css`):**
  - Removed redundant dictionary tags (`NWL`, `CSW`, `AW`) to the right of point values in notepad word lines, keeping each notepad line uncluttered (`1. WORD  12 pts`) while the definition popup provides the complete dictionary and definition metadata.
  - Added subtle hover (`rgba(181, 137, 0, 0.15)`) and active selection (`rgba(42, 161, 152, 0.25)`) backgrounds with smooth transitions and rounded corners on `.notepad-word-line`.

---

### 8. All Words List Point Value Sort & Score Consistency (`static/js/play.js`, `static/js/app.js`, `templates/index.html`)
- **Authoritative Intermission Scores Delivery:**
  - Resolved root cause where `state.all_word_scores` (which can be empty or staged during intermission) was passed directly to `displayAllWords`, leaving unfound words without server-side score maps.
  - Now systematically passes `(state.state === 'intermission' ? (state.previous_all_word_scores || state.all_word_scores) : state.all_word_scores) || {}`, and incorporates a resilient fallback inside `displayAllWords` from `window.lastGameState`.
- **Strict Format Gating for Bonus Letter (+3) & Dual Tiles:**
  - Gated bonus cell coordinate detection strictly to `isBonusLetterFormat = ('bonus letter' in fmt || fmt === 'bonus')`.
  - Prevented leftover/stale bonus coordinates from previous rounds or state defaults on Normal or Either/Or boards from spuriously awarding +3 points and `BL` tags to unfound words, which previously created score discrepancies between found and unfound words.
  - Similarly gated `eoPts` strictly to `isEitherOrFormat`.
- **Mathematical Consistency between Breakdown and Total:**
  - Reconciled `totalPts` and `basePts` so `totalPts = basePts + bonusWordPts + bonusLetterPts + eoPts` is guaranteed to be mathematically exact and identical across objects, numbers, and fallback calculations.
  - Guaranteed `entry.totalPts` is strictly a finite number (never `NaN` or `undefined`).
- **Deterministic Multi-Tier Sort Comparator:**
  - When sorting by **Point Value**:
    1. Primary: `(Number(b.totalPts) || 0) - (Number(a.totalPts) || 0)` (Descending)
    2. Secondary tie-break: Secret bonus word priority (`a.isBonus` vs `b.isBonus`)
    3. Tertiary tie-break: Word length `b.len - a.len` (Descending)
    4. Quaternary tie-break: Alphabetical `a.wordUpper < b.wordUpper` (Ascending)
  - When sorting by **Letter Length**:
    1. Secret bonus word priority
    2. Length (Descending)
    3. Point Value (Descending)
    4. Alphabetical (Ascending)
  - Completely eliminates comparator `NaN` returns that previously destabilized `Array.prototype.sort()`.

---

### 9. Player Activity & Stats Tool Architecture (`app.py`, `templates/index.html`, `static/js/tools.js`, `static/js/app.js`, `static/css/play.css`)
- **FAQ Streamlining:**
  - Removed registered and online player counts from FAQ (both quick navigation list item and content accordion card).
- **Tool Renaming & Re-purposing:**
  - Renamed the "New Users" tool in Tools to **Player Activity & Stats** (`<div class="tool-btn-title">Player Activity & Stats</div>`) with description: `<div class="tool-btn-desc">Live counts, daily logins & new members</div>`.
  - Added support for both `new-users` and `player-stats` identifiers across tool navigation.
- **Two Clean Subtabs Inside the Tool:**
  - **Activity & Stats Subtab (`#subtab-pane-activity`)**:
    - **Streamlined Live Metrics Presentation**: Removed the separate text statement card in favor of the clean, prominent 5-card metric grid:
      1. `ONLINE NOW` (pulsing green dot indicator)
      2. `REGISTERED USERS` (total non-guest accounts)
      3. `USERS TODAY` (unique registered users active today)
      4. `GUESTS TODAY` (unique guest accounts active today)
      5. `REGISTRATIONS THIS WEEK` (rolling 7 days)
    - **Mobile Spacing Optimization**: Added slight top padding (`10px`) and margin (`2px`) above the tabs (`.player-stats-subtabs`) on mobile devices to cleanly separate the tool description box from the "Activity & Stats" tab.
  - **New Users Subtab (`#subtab-pane-new-users`)**:
    - Displays the full chronological member table with Date of Registration, flag badges, clickable mini-profile popups, custom scrollbars, and `[X] this week` pill badge.
- **Persistent Daily Activity Tracking Engine (`app.py`):**
  - Added `daily_user_logins (date_str, user_id, is_guest, username, last_seen)` SQLite table and date index.
  - Added thread-safe `track_daily_login(user_id, username, is_guest)` with zero-overhead in-memory daily cache.
  - Automatically captures logins from `/api/login`, `/api/auth/auto-login`, `/api/guest-login`, `/api/session`, `touch_user_last_visited`, and live `room_manager.user_presence`.
  - Endpoint `/api/tools/new-users` (and `/api/tools/player-stats`) returns complete atomic payload containing `total_users`, `registered_users`, `online_count`, `unique_users_today`, `unique_guests_today`, `registrations_this_week`, and `users`.

---

## Verification & Integrity Check
- Automated test suite passed:
  - Rating placement contracts (`scratch/test_unrated_placement.py`): 3/3 tests OK.
  - Scoring exclusivity & board generator contracts: verified.
  - Sort comparator logic & tie-break contracts: verified across all test cases.
  - Daily login tracking & activity stats contracts: verified on both local and production.
- Production deployment:
  - Commit `841f4d3c` pushed to GitHub `origin/main`.
  - Pulled to `132.148.72.249`, PM2 restarted, live HTTP 200 OK on `https://morpheme.games/`.
  - Local server verified HTTP 200 OK on `http://localhost:5001/`.
  - Build version bumped to `33204` (cache-buster `1794150000`).
