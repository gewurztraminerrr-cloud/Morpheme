# Morpheme Stable State Summary — November 22, 2026

## 1. System Synchronization State
- **Date**: Sunday, November 22, 2026
- **Latest Commit ID**: `b321367dfe8e9df7c2a1b121a57ec0eccd5bc358`
- **Git Tag**: `stable-2026-11-22`
- **GitHub Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme` (`main` branch)
- **Live Production URL**: `https://morpheme.games/` (HTTP 200 OK)
- **VPS PM2 Process**: Online (Process ID 0, 0% CPU, normal memory)
- **Byte-for-Byte Synchronization**:
  - **Localhost**: Commit `b321367dfe8e9df7c2a1b121a57ec0eccd5bc358`
  - **GitHub (`main`)**: Commit `b321367dfe8e9df7c2a1b121a57ec0eccd5bc358`
  - **Production VPS (`morpheme.games`)**: Commit `b321367dfe8e9df7c2a1b121a57ec0eccd5bc358`

---

## 2. Key Changes & Features Completed in this Stable State

### A. Multiplayer Subanagrams "Remaining" Tab (`static/css/play.css`, `static/js/play.js`)
- **Restored Tab Bar at Identical Location**:
  - Removed `body.is-subanagrams #words-tabs-container, body.is-subanagrams .words-tabs { display: none !important; }` in `static/css/play.css`.
  - Enabled the `.words-tabs` bar in `.words-table-box` at the exact same location as public rooms.
- **Configured Subanagrams Word Tabs**:
  - Displays **`Words`** (`data-tab="found"`) and **`Remaining`** (`data-tab="remaining"`).
  - Hides non-applicable tabs (`clues`, `previous`, `score-sum`, `history`).
- **Dynamic Length-Based Remaining Breakdown & Instant Live Re-render**:
  - Calculates `maxLen` dynamically for Subanagrams based on the board's sequence length (e.g. 6 to 10 letters), avoiding blank rows beyond the board sequence.
  - Leaves `maxLen = 30` strictly intact for all standard public rooms.
  - Renders the exact same `#remaining-words-table` with `.len-cell` and `.count-cell` (`[found] + [missed] = [total] ([pct]%)`) and legend.
  - **Dynamic Real-Time Re-rendering on Submission**: Connected `submitWord` (both instant 0ms optimistic path and server confirmation callback) directly to `updateGameState(currentState)`, allowing `#remaining-words-table` counts and percentages to update dynamically with every word entered.
  - Preserves round persistence during intermission.

### B. Multiplayer Subanagrams Dynamic Spinner Set Randomization (`game_room.py`, `spinner_set.py`)
- **Joint Odds Window Enforcement**:
  - Sequence lengths 6, 7, 8, 9, 10 (20% probability each).
  - Word count ranges 50–100, 100–200, 200–300, 300–400, 500+ (20% probability each).
  - Dictionaries NWL, CSW, NWL + AW, CSW + AW (25% probability each).
- **Anti-Repetition Across Consecutive Rounds**:
  - Sequence length, dictionary, and word count range are guaranteed never to repeat back-to-back across consecutive rounds.
- **Fixed Intermission Watchdog Fallback**:
  - Removed logic in `ensure_next_board_ready` and `generate_spinner_params` that previously fell back to reusing `self.spinner_params` (the completed round's parameters).
  - If next round parameters are missing or repeat the completed round, fresh distinct parameters are generated automatically.
- **Instant Next-Round Pre-Generation**:
  - Subanagrams next-round generation (~5ms) is immediately dispatched at round transition via `subanagrams_instant_pregen`, staging the upcoming board and parameters well in advance of the 0:45 reveal.

### C. Subanagrams Word Count Range & Added Words Isolation (`static/js/play.js`, `game_room.py`)
- **Resolved "300-400 words displayed for 122 word board" Bug**:
  - Standard Boggle public room rules enforce `300-400` words whenever Added Words (`+ AW`) is active.
  - In `static/js/play.js`, guarded the Added Words override with `if (useAW && !isSubanagrams)` so Subanagrams displays its true selected word count range (e.g. 50–100, 100–200, 200–300, 300–400, 500+).
  - In `game_room.py`, isolated `generate_spinner_params` so Subanagrams does not run `_get_factchecked_wc_range` with the Boggle AW override.
- **Strict Compliance Guard**:
  - Guaranteed that boards generated in Subanagrams strictly fall within the chosen word count range, and the 500+ range requires at least 500 words.

### D. Moderator Tools Enhancements (`templates/moderation.html`, `app.py`)
- **Definition Management**:
  - Re-positioned the table of words without definitions below the input controls.
  - Added a dedicated Refresh button above the table that re-queries words missing definitions.
- **Pronunciation Management**:
  - Added a table of words without pronunciations below the input controls.
  - Added a dedicated Refresh button above the table.
  - Defaulted the initial view to 5-letter words so moderators start reviewing at 5L words immediately.
- **Theme Support**:
  - Ensured all moderation tables are readable and styled cleanly across all color themes: Default, Dark, Light, and White.

### E. Automatic Lobby Player Count Updates (`static/js/lobby.js`, `templates/index.html`)
- **Full Real-Time Polling for Every Game Button**:
  - Updated `startStatsPolling` in `static/js/lobby.js` to invoke `fetchLobbyStats('all')` continuously on every 1-second interval tick when viewing the Lobby (`isOnLobby()`).
  - Removed mode restrictions (`accumulative_only` and `fcfs_sp_only`) in `updateLobbyButtons`, allowing every game button (`accumulative`, `fcfs`, `split`, `3d`, `subanagrams`) across all times and board dimensions to update its player count in brackets (e.g. `[2]`, `[0]`) dynamically in real-time without requiring manual page reloads or clicking Refresh.
  - Retained DOM update diffing (`rawText.trim() !== targetText.trim()`) to prevent unnecessary DOM reflows when player counts remain unchanged.
  - Bumped the cache buster for `static/js/lobby.js` in `templates/index.html` to `?v=1795222000`.

### F. Architectural Integrity & Strict Boundaries
- **Public Room Isolation**:
  - Standard public rooms (`accumulative`, 4x4, 4x6, 6x8, Split, 3D, Daily) remain completely untouched and unmodified.
- **Zero Server Drift**:
  - Localhost, GitHub `main`, and production VPS (`morpheme.games`) remain 100% byte-for-byte in sync.
