# Stable State Summary — September 13, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of September 13, 2026. The codebase, databases, and lexicon across Localhost, GitHub (`main`), and Production (`morpheme.games` / `132.148.72.249`) are 100% synchronized and verified.

---

## 1. Repository & Deployment Synchronization

| Environment | Host / Path | Status |
| :--- | :--- | :--- |
| **Localhost** | `/Users/jeffbabiak` | ✅ Clean & Synchronized |
| **GitHub** | `https://github.com/gewurztraminerrr-cloud/Morpheme` (`main`) | ✅ Clean & Synchronized |
| **Production Server** | `132.148.72.249` (`morpheme.games`) | ✅ Deployed & Online (PM2 healthy) |

- **Date**: September 13, 2026
- **Latest Commit ID**: See Section 4 below

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Tournament System Modernization & Bug Fixes
1. **Strict 2-Day Round Schedule Enforcement (`tournament_logic.py`, `app.py`)**:
   - Eliminated early round advancement. Previously, when all paired players in a round finished their turns ahead of time, the engine automatically advanced the tournament to the next round immediately.
   - The tournament engine now strictly respects the 2-day round duration (`turn_duration = 2 days`), holding the round until the official 48-hour countdown timer reaches zero before advancing to the next round.
   - Updated post-match status messaging for winners: *"Match result finalized. You won this match! Round [X+1] will begin when Round [X] ends."*
   - Reconciled Tournament 18 in the live production database back to Round 1 with preserved scores and timers expiring September 15, 2026.
2. **All Tournament Pairings Round Dropdown Menu (`static/js/tournaments.js`)**:
   - Added a `Round:` dropdown menu along the top of the "All Tournament Pairings" modal window.
   - Displays round numbers with the active/current round at the very top (`Round X (Current Round)`), preceding rounds decreasing downward (`Round 2`, `Round 1`), and an `All Rounds` option when multiple rounds exist.
   - Dynamically filters the displayed pairings to the selected round and defaults to the current round.
3. **Dedicated Round Victory Card with Proceed Button (`static/js/tournaments.js`)**:
   - When advancing between rounds, winning players are presented with a dedicated celebratory victory card (opponent defeat notice, final scores, and score breakdown) with an explicit `PROCEED TO ROUND X ➔` action button before displaying the new round turn.
   - Added `sessionStorage` tracking per user and round, plus a toggle button to review previous round victory results anytime.
4. **Tournament Match Result Evaluation & User ID Propagation (`static/js/tournaments.js`, `app.py`, `static/js/app.js`)**:
   - Resolved a bug where winning players saw *"YOU LOST THIS MATCH"* because `window.currentUserId` was undefined in the client.
   - Updated matchup winner evaluation to check `matchup.user_id`, `matchup.opponent_id`, and `window.currentUserId`.
   - Exposed `user_id` across `/api/session`, `/api/login`, and `/api/auth/auto-login`.
5. **Round UI Cleanliness during Tournament Matches (`static/css/play.css`, `static/js/play.js`, `static/js/app.js`)**:
   - Automatically hide "FIND ME", "FIND FRIENDS", "SHOW EVERYONE" buttons and the rating color bar (`#game-color-bar`) during active tournament match rounds via `body.is-tournament-round`.
6. **All Tournament Pairings Mobile Fit (`static/js/tournaments.js`, `static/css/style.css`)**:
   - Resolved horizontal overflow on mobile viewports by setting proportional flex shrinking (`min-width: 0 !important; flex: 1 1 0%`), text truncation with ellipsis on usernames, compact badge padding, and optimized modal card dimensions.
7. **Mobile Tap Highlight Removal (`static/css/style.css`)**:
   - Removed native blue square tap outline and focus flashing on mobile devices when tapping usernames in tournaments (`-webkit-tap-highlight-color: transparent !important; outline: none !important;`).

---

### B. Board & Gameplay Experience
1. **Board Panel Red Warning Pulse for White Layouts (`static/css/play.css`, `static/js/play.js`)**:
   - Designed dedicated high-contrast keyframes (`low-time-pulse-white` and `mobile-low-time-pulse-white`) for white and light layouts (`[class*="theme-white"]`, `body.theme-white`, `[class*="theme-light-"]`, yellow, pink, orange, gray, light-brown).
   - Delivers a vibrant red glow with unmistakable outer aura and bold red border when 10 seconds or less remain.
   - Removed mobile box-shadow suppression and updated countdown timer text to high-contrast red (`#ef4444`).
2. **Desktop & Laptop Word Input Capitalization (`static/css/play.css`, `static/js/play.js`)**:
   - Enforced uppercase character transformation in the "Enter word" input (`#word-input`) on desktops and laptops via CSS `text-transform: uppercase` and input event handlers while keeping the placeholder lowercase.

---

### C. Added Words (AW) Lexicon & Definition Pipeline
1. **Immediate Synchronous Disk & Dictionary Writes (`app.py`)**:
   - Replaced deferred asynchronous thread file saving with immediate synchronous writes on both addition and removal under file locks.
   - Synchronously updates `added_words.txt`, `added_words_duplicate.txt`, `wikdefs.txt`, `wikdefs_duplicate.txt`, `Definitions.txt`, `word_stats.json`, and SQLite DB `wiktionary_definitions` before returning the response.
2. **Elimination of Custom Word Added Placeholders & Prefix-Root Decomposition (`app.py`)**:
   - Permanently eradicated generic strings like `"(noun) A custom word added to the dictionary."` across all dictionaries and database tables.
   - Built a prefix-root decomposition engine for standard prefixes (`DIS-`, `DE-`, `UN-`, `RE-`, `MIS-`, `OVER-`, `OUT-`, `PRE-`, `POST-`, `NON-`, `SUB-`, `INTER-`), searching base roots against the 827,000-word definition database and synthesizing rich, authentic lexicographical definitions.
   - Enforced `AGENTS.md` morphological suffix rules for noun plurals (`-IES`, `-ES`, `-S`), verb conjugations (`-ING`, `-ED`, `-S`), agent nouns (`-ER`, `-ERS`), and derived forms (`-NESS`, `-LY`).
3. **Red Text Notification for Missing Sequences in Added Words (`app.py`, `static/js/mods.js`)**:
   - Made `"The sequence '[sequence]' is not present in AW"` display with bold red text (`#f43f5e`), matching standard error and warning indicators in the Mods `#added-word-status-area`.
4. **AW Deletion Validation (`app.py`)**:
   - Validates that words exist in AW before attempting deletion, returning clear feedback if a sequence is not present.
5. **Chronological Persistence Across Deployments (`dictionaries/added_words.txt`, `app.py`)**:
   - Preserved `added_words.txt` with user additions ordered newest-first at index 0, ensuring deployments retain chronological tracking permanently.

---

### D. Navigation, Authentication & UI Stability
1. **Tools & Mods Blank Content Panel Default on Laptops & Desktops (`static/js/app.js`, `static/js/tools.js`, `static/js/mods.js`)**:
   - Synchronously clears and resets the right-side content panel on all desktop and laptop viewports when switching away from or returning to Tools or Mods, preventing the previous tab from briefly flashing before selection.
2. **Mobile Lobby Slider Stability (`static/js/lobby.js`)**:
   - Fixed an issue where tapping the rating filter panel or room tabs on mobile viewports caused the viewport slider to inadvertently slide back to the main lobby panel.
3. **Registration Username Limit Clarification (`templates/index.html`, `app.py`)**:
   - Updated username input placeholder to `(max 16 characters)` to prevent truncation on mobile devices.
4. **Resend Email Verification API Renewal (`app.py`)**:
   - Migrated to `RESEND_API_KEY` loaded securely from `.env`. Replaced shell `curl` subprocess with robust Python `requests.post` call.
5. **Registration Verification Notice Wording (`app.py`, `static/js/app.js`)**:
   - Changed email verification notice to: *"Please check your Junk mail in 1 or 2 minutes if you do not see it."*
6. **Ghost Session & Registration Session Desync Fix (`static/js/app.js`)**:
   - Atomically updates `localStorage.morpheme_username` and clears `morpheme_logged_out` upon registration to prevent previously logged-out users from reappearing on app reopen.
7. **Disable Top Menu Navigation on Login Page (`static/js/app.js`, `static/css/style.css`)**:
   - Enforced `login-active` mode disabling top menu tabs with `pointer-events: none` and `opacity: 0.4` while on the Login page.
8. **3D Tactile Flattening for Gateway LOGIN Button (`static/js/app.js`, `static/css/style.css`, `templates/index.html`)**:
   - Added instant mechanical socket sinking and flattening on touch/pointer down identical to the ENTER LOBBY button.
9. **Private Message Invitation & Unread Persistence (`static/js/tools.js`, `static/css/style.css`, `templates/index.html`)**:
   - Maintained PM invitation toast visibility until explicitly dismissed or read, added unread notification badge to the Profile nav button, and elevated toast z-index above all overlays.
10. **Settings Appearance Reorganization (`templates/index.html`)**:
    - Moved Cube Scale (3D) directly below Board Size.
    - Relocated Synesthesia to the very bottom of the Appearance tab.
11. **Profile Metadata Dedicated Rows (`templates/index.html`, `static/css/style.css`)**:
    - Structured metadata into 3 clean rows: Row 1 (Real Name, Age, Gender), Row 2 (Flag/Country, Timezone), Row 3 (Registered, Last Visited, Proof).
12. **Mobile Title Header Swipe-Up Collapse (`static/js/tools.js`, `static/js/settings.js`, `static/js/mods.js`, `static/css/play.css`)**:
    - Enabled swipe-up collapse and swipe-down reveal for Tools, Settings, and Mods headers on mobile devices for full-screen content focus.
13. **Lobby Panel Ordering (`templates/index.html`, `static/css/lobby.css`)**:
    - Swapped positions so the "+ Create Room" panel sits directly above the Rating Filter panel.

---

## 3. Permanent Invariants Maintained (AGENTS.md)
1. **Full List Modal (`openFullListModal`)**: Explicitly exits fullscreen (`document.exitFullscreen()`) immediately upon invocation.
2. **Android Virtual Keyboard Black Screen Prevention**: Fullscreen is strictly exited when navigating to utility pages or opening modal dialogs with text inputs.
3. **Gateway Screen**: Fullscreen requested on initial tap and preserved continuously into the Lobby without layout shifts.
4. **Dictionary Definitions**: Adheres strictly to noun plural and verb conjugation root pointer sourcing rules, as well as prefix-root authentic definitions without placeholder text.

---

## 4. Verification & Commit Identification

- **Local Python Compilation**: 100% pass (`python3 -m py_compile app.py`).
- **Strict 2-Day Tournament Schedule Verification**: Passed via automated test suite.
- **Production Server Health**: PM2 process `morpheme` online (PID 0), zero errors.
- **Commit ID**: **`d0a0cce078c187a505e6fcfe1dae8a75e3328e96`** (`d0a0cce0`)
