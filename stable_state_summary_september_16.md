# Stable State Summary — September 16, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of **September 16, 2026**. The codebase, databases, assets, and styling across **Localhost**, **GitHub (`main`)**, and **Production (`morpheme.games` / `132.148.72.249`)** are 100% synchronized, verified, and active.

---

## 1. Repository & Deployment Synchronization

| Environment | Host / Branch | Latest Commit ID | Status |
| :--- | :--- | :--- | :--- |
| **Localhost** | `/Users/jeffbabiak` (`main`) | `4cd273c91390c94305a91b4c7396d1d902c5cb45` | ✅ Clean & Synchronized |
| **GitHub** | `origin/main` | `4cd273c91390c94305a91b4c7396d1d902c5cb45` | ✅ Clean & Synchronized |
| **Production Server** | `132.148.72.249` (`/home/morpheme/morpheme`) | `4cd273c91390c94305a91b4c7396d1d902c5cb45` | ✅ Deployed & Online (`HTTP/2 200 OK`) |
| **PM2 Process** | `morpheme` (PID 0) | `4cd273c91390c94305a91b4c7396d1d902c5cb45` | ✅ Healthy (`online`, uptime active) |
| **Flutter Mobile App** | `morpheme_word_game` | `4cd273c91390c94305a91b4c7396d1d902c5cb45` | ✅ Synchronized (`https://morpheme.games/` audio bridge) |

- **Stable Save Point Date**: September 16, 2026
- **Latest Commit ID**: `4cd273c91390c94305a91b4c7396d1d902c5cb45` (`4cd273c9`)
- **Active Git Tags**:
  - `START_OVER_POINT_SEPTEMBER_16`
  - `stable-2026-09-16`
  - `START_OVER_POINT`
  - `save-point-latest`
  - `start-over`
  - *(Historic reference preserved: `START_OVER_POINT_SEPTEMBER_13`, `START_OVER_POINT_SEPTEMBER_10`)*
- **Active Cache-Buster Versions**:
  - `style.css?v=1789334000`
  - `lobby.css?v=1789337000`
  - `play.css?v=1789341000`
  - `howtoplay.css?v=1789330000`
  - `forum.css?v=1789336000`
  - `donate.css?v=1789324000`

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Lobby & Navigation
1. **Lobby Journey Banner Glow Removal (`templates/index.html`, `static/css/lobby.css`)**:
   - Removed the outer blurred box shadow (`0 4px 14px rgba(0, 0, 0, 0.45)`) around the **"ENTER A ROOM TO CONTINUE YOUR JOURNEY"** banner (`.lobby-journey-message`), eliminating the fuzzy, dark glowing halo around the bottom of the banner on light themes.
   - Preserved all inner text formatting, pearl-white gradient typography, top specular bevel highlight (`::after`), and diamond shimmer sweep (`::before`) intact.
2. **Tournaments Top Menu Button Highlight (`templates/index.html`, `static/js/tournaments.js`)**:
   - Highlighted the top navigation **Tournaments** menu button in active blue during open tournament registration until the current user joins or registration closes.
3. **Mobile Back Button High-Visibility Blue Highlight (`templates/index.html`)**:
   - Changed the pressed/hovered/focused state of the bottom Back navigation button (`.bottom-back-btn`, `.forum-back-btn`) from red to high-visibility blue (`#38bdf8` / `#0284c7`) across all themes.

---

### B. Game Rooms & Gameplay UI
1. **Mobile Player List Username Positioning (`templates/index.html`, `static/css/play.css`)**:
   - Under **PLAYERS** in game rooms, moved the username to the immediate right side of the colored square representing rating (separated by a clean gap), replacing the previous centered positioning.
2. **Mobile Room Spacing & Padding Tightening (`templates/index.html`, `static/css/play.css`)**:
   - Tightened vertical spacing on mobile: reduced padding above the timer, board, and word validation message.
   - Added appropriate breathing room between the Spinner Set and the timer panel, ensuring comfortable layout proportions without awkward vertical gaps.
3. **Mobile Chat Textbox Clearance (`templates/index.html`, `static/css/play.css`)**:
   - Reserved proper bottom clearance for the mobile chat textbox above the bottom Board navigation button when the keyboard/panel is expanded, preventing UI overlaps.
4. **Desktop Round Replay Alignment (`templates/index.html`, `static/css/style.css`, `static/css/play.css`)**:
   - On desktops in Round Replay (`#history-review-overlay`), aligned the **Watch Replay** button to the left side of the containing panel (`.replay-subheader`), with the microsecond precision hint text placed immediately to the right (`gap: 16px;`).
   - Updated precision hint text to read **"Words recorded with microsecond precision"**.

---

### C. Tournaments Engine & UI
1. **Tournament BYE Logic on Odd Player Counts (`tournament_logic.py`)**:
   - When an odd number of players advance to a new round, the system awards a BYE to the player with the highest score from the preceding round.
2. **Round Standings & Hall of Fame Side-by-Side Layout (`static/css/style.css`)**:
   - On desktop/laptop viewports, Round Standings and Tournament Hall of Fame are positioned side by side with matching card heights (`300px`), maximizing screen space efficiency.
3. **App-Wide Flag Displays (`static/js/tournaments.js`)**:
   - Integrated country flags across Current Pairings, View All Pairings, Round Standings, Championship Bracket, Scores breakdown, and Hall of Fame.
4. **Pairings & Standings Typography Separation (`static/js/tournaments.js`)**:
   - Prevented long usernames from overlapping with score values under match completion cards with dedicated minimum widths and nowrap layout rules.

---

### D. App-Wide Controls & Settings
1. **Global High-Visibility Toggle Switch Colors (`static/css/play.css`, `static/css/style.css`)**:
   - Enforced bright green (`#2ecc71`) with neon glow for ON states and bright red (`#ef4444`) with soft glow for OFF states across all toggle switches throughout the app.
2. **Added Words Status Area Placement (`templates/index.html`)**:
   - Fixed the Added Words status notification area directly above/below the input with reserved layout dimensions, eliminating UI jitter.

---

## 3. Permanent Invariant Rules Enforced (`AGENTS.md`)

1. **Full List Modal (`openFullListModal`)**: Immediately and explicitly exits fullscreen (`document.exitFullscreen()`) upon modal invocation in `static/js/tools.js`. Never remove or disable.
2. **Android Virtual Keyboard Black Screen Prevention**:
   - Fullscreen is strictly exited when navigating to utility pages (Tools, Settings, Profile, Forum, How to Play, Donate) or opening modal dialogs with text inputs.
   - Automatic fullscreen re-engagement never triggers while on non-game utility pages or when any modal/input is active.
3. **Gateway Screen (`#page-loading`)**: Fullscreen is requested immediately on initial tap/press across the initial ENTER LOBBY screen so the Android system prompt appears on the gateway screen, and fullscreen is preserved continuously into the Lobby without layout shifting.
4. **Dictionary Suffix and Plural Sourcing Rules**: Plurals repeat the singular root definition; conjugated verbs (`-S`, `-ED`, `-ING`) repeat the base root verb definition; prefix decomposition produces rich lexicographical definitions.

---

## 4. Verification Checkpoint

- **Local Compilation**: Clean Python compile on `app.py` and all modules.
- **Git State**: Local repository `/Users/jeffbabiak` is clean, with all commits pushed to GitHub `origin/main`.
- **Production Server (`132.148.72.249`)**: Working directory `/home/morpheme/morpheme` is at commit `4cd273c91390c94305a91b4c7396d1d902c5cb45`.
- **PM2 Daemon**: Process `morpheme` running `online` with zero errors.
- **Live Endpoint Verification**: `curl -sI https://morpheme.games` returns `HTTP/2 200 OK`.
- **Latest Commit ID**:
  ```
  4cd273c91390c94305a91b4c7396d1d902c5cb45
  ```
