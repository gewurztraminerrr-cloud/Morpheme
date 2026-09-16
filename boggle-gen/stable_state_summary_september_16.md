# Stable State Summary — September 16, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of **September 16, 2026**. The codebase, databases, assets, and styling across **Localhost**, **GitHub (`main`)**, and **Production (`morpheme.games` / `132.148.72.249`)** are 100% synchronized, verified, and active.

---

## 1. Repository & Deployment Synchronization

| Environment | Host / Branch | Latest Commit ID | Status |
| :--- | :--- | :--- | :--- |
| **Localhost** | `/Users/jeffbabiak` (`main`) | `4bb66c0fde3d3f53b152c5b22941b0213aec3fbc` | ✅ Clean & Synchronized |
| **GitHub** | `origin/main` | `4bb66c0fde3d3f53b152c5b22941b0213aec3fbc` | ✅ Clean & Synchronized |
| **Production Server** | `132.148.72.249` (`/home/morpheme/morpheme`) | `4bb66c0fde3d3f53b152c5b22941b0213aec3fbc` | ✅ Deployed & Online (`HTTP/2 200 OK`) |
| **PM2 Process** | `morpheme` (PID 0) | `4bb66c0fde3d3f53b152c5b22941b0213aec3fbc` | ✅ Healthy (`online`, uptime active) |

- **Date**: September 16, 2026
- **Latest Commit ID**: **`4bb66c0fde3d3f53b152c5b22941b0213aec3fbc`** (`4bb66c0f`)
- **Current App Build**: `'33146'`
- **Cache-Buster Version**: `?v=1789331000`

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Tournament UI, Pairings & Standings
1. **Round Standings & Hall of Fame Side-by-Side Layout (`static/css/style.css`)**:
   - On desktop and laptop viewports (`min-width: 900px`), when **ROUND STANDINGS** (`#tournament-leaderboard-card`) is present, it is displayed side-by-side with **TOURNAMENT HALL OF FAME** (`.history-panel`) (Round Standings on the left at `grid-column: 1`, Hall of Fame on the right at `grid-column: 2`).
   - When Round Standings is hidden/absent, Tournament Hall of Fame spans across both columns (`grid-column: span 2`).
   - The vertical length of Round Standings (`.t-leaderboard-list`) is set to `height: 300px; max-height: 300px`, precisely matching the vertical length of the Tournament Hall of Fame table wrapper (`.history-table-wrapper`), with both cards aligned as flex column containers.
2. **Flag Displays Across All Tournament Sections (`static/js/tournaments.js`)**:
   - **Current Pairings**: Country flag rendered directly to the left of each player's username.
   - **View All Pairings**: Country flag displayed to the left of both usernames across all historical and current matchups.
   - **Round Standings**: Country flag rendered to the left of every user in the standings list.
   - **Championship Bracket**: Country flag rendered to the left of every user in the bracket ladder.
   - **Scores Breakdown**: In win/loss cards ("YOU WON", "YOU LOST", "YOU ADVANCED"), each player's flag is displayed to the left of their username.
   - **Tournament Hall of Fame**: Champion's country flag rendered directly beside their username.
3. **Current Pairings & All Pairings Layout Redesign (`static/js/tournaments.js`, `tournament_logic.py`)**:
   - Trophies (`🏆`) placed adjacent to the winning user's username.
   - Scores positioned beside `"vs"` (e.g. `jeffjeff 75 vs 103 🏆 jeffles`), with all trailing periods removed after the second score.
4. **Desktop Long Username & Score Separation (`static/js/tournaments.js`)**:
   - Under `"SCORES:"` in match completion cards, long usernames are protected with `min-width: 250px`, `gap: 20px`, `margin-right: 12px` on the player container and `white-space: nowrap; flex-shrink: 0;` on the score span, preventing scores from colliding with usernames (e.g. `username12 31 pts` rather than `username1231 pts`).
5. **Mobile Current Round Header Spacing (`static/js/tournaments.js`)**:
   - Separated the blue `"Current Round"` indicator badge from the `"Round [X]"` heading with ample margin spacing (`margin-top: 10px; margin-bottom: 4px;`).
6. **Eliminated Lingering Round Banners (`static/js/tournaments.js`)**:
   - Removed obsolete "Won Round" banners and stale "View Results" buttons from lingering in completed states.

---

### B. App-Wide Toggle Switch System
1. **Global High-Visibility Toggle Switch Colors (`static/css/play.css`, `static/css/style.css`)**:
   - Updated toggle switches throughout the entire application (Settings, Mods Added Words, and all utility panes):
     - **Toggled ON**: Vibrant Green (`#2ecc71 !important`) with green border (`rgba(46, 204, 113, 0.4)`) and neon green glowing box shadow (`0 0 12px rgba(46, 204, 113, 0.5)`).
     - **Toggled OFF**: Vivid Red (`#ef4444 !important`) with red border (`rgba(239, 68, 68, 0.4)`) and subtle red glowing box shadow (`0 0 8px rgba(239, 68, 68, 0.3)`).
   - Applied directly in `play.css` and `style.css` so stylesheets loaded late never revert to room theme accent colors.

---

### C. White & Light Layout Contrast & Readability
1. **Flag Selection Modal on White Themes (`static/css/style.css`)**:
   - Ensured high-contrast dark text (`#0f172a`), deep slate headings (`#334155`), solid borders, and clear white background tiles across `#flag-picker-modal` and `#flag-picker-grid` on all white, yellow, pink, orange, gray, and light-brown themes.
2. **Change Password / Email & Subanagrams in Tools (`static/css/style.css`)**:
   - Changed all form labels, input borders, and generated subanagram word counts, group headers, and clickable word links from faint light tones to deep high-contrast text (`#0f172a` / `#334155`).
3. **Versus Opponent Username Readability (`static/css/style.css`)**:
   - Formatted `.t-versus-opponent-name` to bold dark slate (`#0f172a !important`) across all white and light layout themes.
4. **Low-Time Red Board Pulse in Tournaments (`static/js/play.js`, `static/css/play.css`)**:
   - When 10 seconds or less remain in a tournament round, the board panel (`#play-panel-board`) triggers `.low-time-warning`, pulsating with bold red flashing keyframes across white and light themes.

---

### D. Mods & Added Words (AW) Placement
1. **Validation Message Relocation (`templates/index.html`)**:
   - Relocated the Added Words status notification area (`#added-word-status-area`) directly below the word input and action button panel for immediate visual confirmation without layout shifts.

---

## 3. Permanent Invariant Rules Enforced (`AGENTS.md`)

1. **Full List Modal (`openFullListModal`)**: Immediately and explicitly exits fullscreen (`document.exitFullscreen()`) upon modal invocation in `static/js/tools.js`.
2. **Android Virtual Keyboard Black Screen Prevention**:
   - Fullscreen is strictly exited when navigating to utility pages (Tools, Settings, Profile, Forum, How to Play, Donate) or opening modal dialogs with text inputs.
   - Automatic fullscreen re-engagement never triggers while on non-game utility pages or when any modal/input is active.
3. **Gateway Screen (`#page-loading`)**: Fullscreen is requested immediately on initial tap/press across the initial ENTER LOBBY screen so the Android system prompt appears on the gateway screen, and fullscreen is preserved continuously into the Lobby without layout shifting.
4. **Dictionary Suffix and Plural Sourcing Rules**: Plurals repeat the singular root definition; conjugated verbs (`-S`, `-ED`, `-ING`) repeat the base root verb definition; prefix decomposition produces rich lexicographical definitions.

---

## 4. Verification Checkpoint

- **Local Compilation**: Clean Python compile on `app.py` and all modules.
- **Git State**: Local repository `/Users/jeffbabiak` is clean, with all commits pushed to GitHub `origin/main`.
- **Production Server (`132.148.72.249`)**: Working directory `/home/morpheme/morpheme` is at commit `4bb66c0fde3d3f53b152c5b22941b0213aec3fbc`.
- **PM2 Daemon**: Process `morpheme` running `online` with zero errors.
- **Live Endpoint Verification**: `curl -sI https://morpheme.games` returns `HTTP/2 200 OK`.
- **Latest Commit ID**:
  ```
  4bb66c0fde3d3f53b152c5b22941b0213aec3fbc
  ```
