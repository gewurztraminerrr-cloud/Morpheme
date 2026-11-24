# Morpheme Stable State Summary – November 23, 2026

**Synchronization Checkpoint & Stable Anchor**  
- **Git Branch**: `main` (Synchronized: Localhost == GitHub `origin/main` == VPS `132.148.72.249`)
- **Production Server**: `132.148.72.249` ([morpheme.games](https://morpheme.games)) – Live & Healthy (HTTP 200 OK)
- **Localhost Development**: Port `5001` – Running & Healthy (HTTP 200 OK)
- **Date/Time**: November 23, 2026 (22:28 EST / 03:28 UTC)

---

## 1. Key Achievements & Updates in this Stable Point

### A. Lobby Room Entry Freeze & 45-Second Timeout Fix (`loadfail2.jpg`)
- **Root Cause Resolution**:
  - Investigated `loadfail2.jpg` displaying `"Joining room took longer than expected. Please try again."` when entering ACC 4x4 45s from the Lobby.
  - In commit `e73000da`, `needs_wakeup` in `game_room.py` required `existing_room.time_remaining <= 0` across all game modes. When an empty standard boggle room was in intermission with time remaining, it returned in an `intermission` state rather than immediately waking up to Active Round 1.
  - Returning the room state triggered the 0:50 watchdog fallback in `get_next_round_milestone()`, which ran `self.ensure_next_board_ready()` synchronously on the Waitress request thread.
  - On a 4x4 grid with `min_word_length = 5`, generating 300–400 words is physically impossible. The generator cycled through 5 compliance attempts, Stage 2 optimization, and rescue sweeps, tying up the request thread for > 40 seconds until the client's `AbortController` timed out at 45 seconds.
- **Three-Tier Architectural Fix**:
  - **Immediate Wakeup for Empty Boggle Rooms (`game_room.py:4424`)**: Restricted `time_remaining <= 0` strictly to Subanagrams. Empty competitive boggle rooms (ACC, FCFS, Split) now wake up immediately to Active Round 1 in **< 5ms**.
  - **Asynchronous 0:50 Watchdog (`game_room.py:1739-1746`)**: Wrapped `ensure_next_board_ready()` in a background daemon thread so watchdog checks never block an HTTP request thread or Waitress worker.
  - **Target Word Range Clamping (`game_room.py:3654-3665`)**: Clamped 4x4 Added Words emergency fallback targets to realistic ranges when `min_word_length` is 4 (`200-300`) or 5+ (`100-200`), eliminating unachievable generation loops.

---

### B. Multiplayer Subanagrams Intermission Loading
- **Intermission Lifecycle & Tab State Preservation**:
  - Fixed issue where the Multiplayer Subanagrams intermission failed to display after round completion.
  - Preserved the full 60-second intermission display so players can review scores, placements, and all discoverable words even if they refresh or if the roster momentarily empties.
  - Ensured the intermission panel defaults to the "Found Words" tab upon transition.

---

### C. Clean Baseline for Word Tallies & Dynamic Divisions (0 Finds, Div Max of Max)
- **All Words Start at 0 Finds**:
  - Purged artificial seed benchmark counts from `dictionaries/word_stats.json` and `user_word_tallies`.
  - Every word throughout Morpheme starts at **0 finds** with its baseline maximum rarity division:
    - **7+ Letter Words**: `Division 5 of 5` (maximum points, e.g., 9 of 9 pts for 7L, 13 of 13 pts for 8L).
    - **6-Letter Words**: `Division 3 of 3` (5 of 5 pts).
    - **5-Letter Words**: `Division 2 of 2` (3 of 3 pts).
  - Perfect consistency between **Dynamic Word Lookup** (Word Charts) and **Find Count**: both tools report identical find counts, rarity tiers, and point values.
- **Unified Single Source of Truth Across All Tally Systems**:
  - `sync_word_stats_from_db` calls `rebuild_word_stats_from_db` directly, locking `word_stats.json`, Word Charts Dynamic Word Lookup, Find Count, and game room scoring into the exact same authenticated database source (`user_word_tallies` and `word_finds` in `morpheme.db`, excluding Guest and 24h room finds).
  - During live gameplay, round completions update `user_word_tallies`, `word_finds`, and `word_stats.json` in locked synchronization.
- **Protection Against Database Table Bloat**:
  - Added a strict ceiling of at most 10 recent finders per word in `word_finds` during round history pruning, ensuring the database remains permanently compact and fast.

---

### D. Dynamic Word Lookup Length Badge in Tools -> Word Charts
- **Visual & Layout Alignment**:
  - In Word Charts -> Dynamic Word Lookup: moved `"5 letters"` next to the word name down to the dictionaries row below it.
  - Formatted the length badge as `<span class="tool-def-len-badge">${wordLen}L</span>`, perfectly matching the badge design in the universal Word Definition Popup.
  - Bumped cache buster in `templates/index.html` (`tools.js?v=1795380001`).

---

### E. Universal Word Definition Popup Auto-Repositioning (No Bottom Cut-Off)
- **Cross-Platform Viewport Edge Detection**:
  - Whenever any word is clicked/selected throughout the app (Tools tabs, Game rooms, Notepad, Round Replay, Mods lists, etc.) and `#tool-word-def-popover` appears:
  - If the bottom of the popover extends past the visible viewport bottom (`rect.bottom > viewportHeight - 12`), the system smoothly shifts `top` higher with a clean 12px safety margin.
- **Dynamic Asynchronous Height Handling**:
  - Attached a `ResizeObserver` on `#tool-word-def-popover` to auto-detect content expansion when definitions, inflections, Wikipedia links, and images load asynchronously.
  - Added image `load` and `error` event handlers inside `renderDefData(data)` to re-calculate popover positioning upon image rendering.
- **Small Screen & Mobile Landscape Fallback**:
  - Ensured shifting `top` higher never pushes the popover above the top screen edge (`top >= 12px`).
  - Capped `maxHeight` to `viewportHeight - 24px` with smooth touch scrolling on `.tool-def-content` for constrained screen heights.

---

### F. Selected Word Highlight & Dark Blue <-> White Breathing Pulse
- **High-Contrast Dark Blue to White Flash on Default Layouts**:
  - Implemented the user-requested breathing pulse on default color layouts (dark themes), flashing smoothly from **dark blue (`#0f2b5c`)** to **pure luminous white (`#ffffff`)**, mirroring the high contrast seen on light-colored layouts.
  - Utilized CSS `@property --selected-cell-bg`, `--selected-border-color`, and `--selected-text-color` for hardware-accelerated color interpolation without DOM churn or hitching.
  - Dynamically flips text color from crisp white to deep dark blue (`#0f2b5c`) during the white crest phase, ensuring maximum AAA contrast (13.5:1) throughout the entire animation cycle.
  - Preserved distinct category glows on borders (blue for unfound/default, lavender for AW, green for bonus, gold for CSW, sky blue for player, rose for opponent).
  - Bumped cache buster in `templates/index.html` (`play.css?v=1795370003`).

---

## 2. Infrastructure, Server Health & Resource Usage
- **Production Server**: GoDaddy VPS (`132.148.72.249`, Ubuntu 24.04 LTS).
- **Process Management**: PM2 process `0` (`morpheme`) managing Waitress WSGI server.
- **Task Queue & GIL**: Waitress worker queue cleared and operating normally; no blocking synchronous board generation on request threads.
- **Localhost Development**: Port `5001` running identically configured daemon (`app.py`).

---

## 3. Verification Summary
- **Live Room Transition Test**: Repeated automated cycles entering Multiplayer Subanagrams, returning to Lobby, and immediately entering ACC 4x4 45s executed in **0.14s – 0.15s** with 0 errors or timeouts.
- **Browser UI Test**: Verified guest login, lobby navigation, room start, and definition popovers across desktop and mobile viewports.
- **Health Check**: `https://morpheme.games/` (HTTP 200 OK), `http://localhost:5001/` (HTTP 200 OK).
