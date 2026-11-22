# Morpheme Stable State Summary — November 22, 2026

## 1. System Synchronization State
- **Date**: Sunday, November 22, 2026
- **Latest Commit ID**: `c763b9f74e1d94a531aca08b2b2e611e8648277a`
- **Git Tag**: `stable-2026-11-22`
- **GitHub Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme` (`main` branch)
- **Live Production URL**: `https://morpheme.games/` (HTTP 200 OK)
- **VPS PM2 Process**: Online (Process ID 0, 0% CPU, normal memory)
- **Byte-for-Byte Synchronization**:
  - **Localhost**: Commit `c763b9f74e1d94a531aca08b2b2e611e8648277a`
  - **GitHub (`main`)**: Commit `c763b9f74e1d94a531aca08b2b2e611e8648277a`
  - **Production VPS (`morpheme.games`)**: Commit `c763b9f74e1d94a531aca08b2b2e611e8648277a`

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

### F. Tools: Word Charts Desktop & Laptop Screen Fit, Scrollers, and Light Theme Styling (`templates/index.html`, `static/css/play.css`, `static/js/tools.js`)
- **Fit All Content Onto Screen**:
  - Enforced full-height flex column layout on desktops and laptops (`@media (min-width: 901px)`) for `#tool-word-chart` across all 3 sub-tabs:
    1. Sub-tab 1: Point Values & Divisions (`#subtab-pane-wc-points`)
    2. Sub-tab 2: Dictionary Word Counts (`#subtab-pane-wc-dicts`)
    3. Sub-tab 3: Dynamic Word Lookup (`#subtab-pane-wc-lookup`)
  - Guaranteed `bodyScrollHeight == bodyClientHeight`, `tcScrollHeight == tcClientHeight`, and `paneScrollHeight == paneClientHeight` so outer pages/panes do not scroll.
- **Custom Scrollers & Sticky Headers**:
  - Wrapped each sub-tab's table/content in custom scrollable wrappers (`.word-chart-table-container.list-scroll-area-wrapper`, `.word-chart-table-scroll`, `.wc-lookup-scroll-wrapper`, `.wc-lookup-scroll-area`) with custom draggable scrollbar tracks and gradient thumbs.
  - Initialized via `initCustomScrollbarForElement` and refreshed on every sub-tab switch, data load, and search via `window.updateWordChartScrollbars()`.
  - Configured sticky table headers (`position: sticky; top: 0; z-index: 10`) with 100% solid, fully opaque backgrounds (`var(--bg-primary, #181920)`) and bottom box-shadows, completely eliminating scrolled row text bleed-through across all themes.
- **Pure White Tables & Cards for Light-Colored Layouts**:
  - Configured pure white backgrounds (`#ffffff !important;`) and light subtle borders (`#cbd5e1` / `#e2e8f0`) across all 3 sub-tabs for light themes (`theme-white` and all `theme-light-*`), replacing dark gray:
    1. Sub-tab 1: Table container, scroll wrapper, table body, sticky headers, and rows are crisp white with high-contrast text columns and light blue info banner.
    2. Sub-tab 2: Stat boxes at the top and dictionary word count table are pure white with distinct colored numbers and clear column headers.
    3. Sub-tab 3: Search bar input, live lookup result card, metric cards, division example cards, and division pill boxes are pure white and soft light gray, eliminating all dark gray styling in light themes.
  - Inactive sub-tab buttons in light layouts are styled with white backgrounds and crisp borders.
  - Custom scrollbar tracks and blue gradient thumbs styled cleanly for light layouts.
  - Bumped cache busters: `play.css?v=1795224000` and `tools.js?v=1795224000`.

### G. Multiplayer Subanagrams Spinner Set Parameter Preservation at Round Transition (`game_room.py`, `subanagrams_generator.py`)
- **Root Cause Eliminated**:
  - Previously, `pick_distinct_bonus_word` in `game_room.py` was unconditionally called during `start_next_round` for all game types. Because `pick_distinct_bonus_word` was designed for Boggle boards (hard-clamped target lengths to [6..10], rejected words ending in `ING`/`INGS`, and filtered out recent history), it rejected the pre-generated Subanagrams bonus word staged in `room.next_round_bonus`. It then selected a different word of a different length from the board, and overwrote `room.bonus_word` and `bonus_word_length` in both `room.spinner_params` and `room.frozen_revealed_params`.
  - When the round transitioned from intermission to active (`!isIntermission`), the client UI in `static/js/play.js` read `state.bonus_word.length`, abruptly changing the Bonus Word letter length on the Spinner Set at `0:00`.
- **Subanagrams Bonus Word Isolation**:
  - In `game_room.py`, isolated `start_next_round` so Subanagrams does NOT call `pick_distinct_bonus_word`. It strictly preserves `room.bonus_word` from `room.next_round_bonus` (or `frozen_revealed_params['bonus_word']`).
  - Guarantees that `room.bonus_word` and `bonus_word_length` in `room.spinner_params`, `room.frozen_revealed_params`, and `room.next_spinner_params` strictly retain the exact declared bonus word length revealed at 0:45.
- **Bonus Word Length Honoring in Generator**:
  - Updated `subanagrams_generator.py` so `generate_subanagrams_board_and_words` respects `params.get('bonus_word_length')`. It filters candidate words for the target length first, ensuring generated bonus words match the spun target length from inception.
- **Complete Parameter Immutability**:
  - Ensured all other parameters revealed at 0:45 intermission (**Sequence Length**, **Dictionary**, **Target Word Count Range**, **Minimum Word Length**, and **Board Format / Mode**) remain 100% frozen and immutable through the `0:00` round transition.
  - Aligned `is_count_in_range` in `subanagrams_generator.py` to strictly use `< 100`, `< 200`, `< 300`, `< 400` so range boundary counts never trigger range re-classification.
  - Added guard in `generate_spinner_params` so any background calls after 0:45 reveal cannot overwrite frozen parameters.

### H. Multiplayer Subanagrams Real-Time Typing Letter Highlighting (`static/css/play.css`, `static/js/play.js`, `templates/index.html`)
- **Real-Time 0ms Tile Highlighting As User Types**:
  - In `static/js/play.js`, implemented `highlightSubanagramsInput(typedText)` and `isSubanagramsActive()` with instant 0ms latency.
  - Matches typed characters directly against the rack's DOM cells (`.board-cell[data-letter]`), immediately applying `.typing-highlight` and `.selected` as the player types into `#word-input`.
  - Backspacing, editing, or clearing the input immediately synchronizes or clears highlights across all tiles.
  - Supports duplicate letters (e.g. rack with multiple 'A's accurately highlights distinct physical tiles) and 'QU'/'Q' sequence equivalence.
  - Handles letters not on the rack gracefully without throwing or clearing matching rack letters.
  - Preserves highlights and board mapping accurately across board tile shuffling (`window.shuffleSubanagramsBoard`).
- **Eliminated `hasBoardSelection` Block**:
  - Fixed root cause where `hasBoardSelection` in `play.js` detected `mouseState.selectedPath.length > 0` after the first typed letter and prematurely aborted/cleared highlights for all subsequent typed letters.
  - Delegated Subanagrams typing directly to `highlightSubanagramsInput(word)` in the `word-input` input and keyup event listeners.
- **Dedicated Highlighting Styles**:
  - In `static/css/play.css`, added dedicated styles for `.board-cell.typing-highlight` in Subanagrams with high-contrast text and custom highlight colors (`var(--highlight-typing-color, #ffcc00)`).
  - Maintained separation between mouse/touch selection (`.selected`, `.current`) and typing highlights (`.typing-highlight`).
- **Cache Busters Bumped**:
  - Bumped `play.css` and `play.js` to `?v=1795225000` in `templates/index.html`.

### I. Architectural Integrity & Strict Boundaries
- **Public Room Isolation**:
  - Standard public rooms (`accumulative`, 4x4, 4x6, 6x8, Split, 3D, Daily) remain completely untouched and unmodified.
- **Zero Server Drift**:
  - Localhost, GitHub `main`, and production VPS (`morpheme.games`) remain 100% byte-for-byte in sync.
