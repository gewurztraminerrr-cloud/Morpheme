# Morpheme Stable State Summary – November 24, 2026

**Synchronization Checkpoint & Stable Anchor**  
- **Git Branch**: `main` (Synchronized: Localhost == GitHub `origin/main` == VPS `132.148.72.249`)
- **Production Server**: `132.148.72.249` ([morpheme.games](https://morpheme.games)) – Live & Healthy (HTTP 200 OK)
- **Localhost Development**: Port `5001` – Running & Healthy (HTTP 200 OK)
- **Date/Time**: November 24, 2026 (01:03 EST / 06:03 UTC)
- **Anchor Commit ID**: `62b8c47b` ("Implement Concept 1: Word Length Radar for desktop and laptop game rooms")

---

## 1. Key Achievements & Consolidated Features

### A. Intermission Words Filter Dropdowns (Order & Dynamic Counts)
- **Swapped Dropdown Hierarchy**:
  - Reordered the filter row controls above the intermission words table:
    1. **Word List Dropdown** (`#dictionary-filter-dropdown`): `CSW + AW`, `CSW`, `CSW Only`, `NWL`, `AW`
    2. **Length Dropdown** (`#length-filter-dropdown`): `All (found/total)`, `4LW (found/total)`, `5LW`, etc.
    3. **Order Dropdown** (`#order-filter-dropdown`): `Alphabetical`, `Random`
  - Perfectly reflects the logical hierarchy: select dictionary list first, then filter by word length within that category.
- **Dynamic Found / Total Word Counts**:
  - Length filter dropdown dynamically computes word counts against the active dictionary selection (`dictFilteredWords`):
    - Example: `All (34/58)`, `4LW (12/18)`, `5LW (15/22)`.
  - Switching between dictionary categories immediately recalculates counts and updates option labels in real time.
- **Commit History**: `115dec4e` (Dynamic counts), `f1f32a6f` (Swapped positions).

---

### B. 24-Hour Room 10-Minute Inactivity Kick Exemption
- **Root Cause Resolution**:
  - `static/js/app.js` contained duplicate global listeners (`DOMContentLoaded` and `visibilitychange`) that evicted players after 10 minutes of tab inactivity without verifying if the active room was a 24h daily room.
- **Implementation**:
  - Added `is24h` checks in `app.js` (line 650 and line 4504) and strengthened checks in `play.js` (lines 1429 and 1828) against room IDs containing `24h` / `86400` or `time_limit >= 7200`.
  - Confirmed 24h room finds remain properly isolated from leaderboards, find counts, and player ratings.
- **Commit History**: `be3ff605`.

---

### C. Selected Word Cell Visual Elevation
- **High-Contrast Dark Blue to White Breathing Pulse**:
  - Selected word card pulses from dark blue (`#0f2b5c`) to pure white (`#ffffff`) using CSS `@property` color interpolation.
  - Dynamically flips text color between white and `#0f2b5c` for AAA contrast compliance throughout the cycle.
- **Glowing Indicator Pin (`▶`)**:
  - Active pointer beacon at `left: -20px` bouncing smoothly with a 1.2s spring animation (`@keyframes pin-bounce`) toward the selected card.
  - Multi-tier neon drop-shadow glow matching the word category (Sky blue, Lavender for AW, Green for bonus, Gold for CSW, Rose for opponent).
- **Diagonal Light Sheen Sweep**:
  - 115-degree linear gradient sweep every 2.6 seconds giving the card a polished reflective sheen.
- **Commit History**: `95ffe0ac`, `b41691a4`.

---

### D. Universal Word Definition Popup Edge Detection
- **Auto-Repositioning**:
  - Viewport edge detection prevents popover cut-off when clicking words near the bottom or right of the screen.
  - Flips popover upward or adjusts coordinate offsets automatically across desktop and mobile devices.

---

### E. Dynamic Word Lookup Length Badge
- **Layout Alignment**:
  - Formatted word length as `<span class="tool-def-len-badge">${wordLen}L</span>` next to dictionary tags in Tools -> Word Charts -> Dynamic Word Lookup, matching the universal popup styling.

---

### F. Clean Baseline for Word Tallies & Dynamic Divisions
- **Zero Authentic Baseline**:
  - Purged artificial seed benchmark tallies from `word_stats.json` and `user_word_tallies`.
  - All words initialize at 0 finds with max rarity division (7+L: Div 5 of 5, 6L: Div 3 of 3, 5L: Div 2 of 2).
  - Word Charts Dynamic Word Lookup, Find Count, and live game scoring are locked to the exact same authenticated database source.
  - `word_finds` capped at 10 recent finders per word to protect SQLite database compactness.

---

### G. Lobby Room Entry Freeze & 45-Second Timeout Resolution
- **Immediate Wakeup**: Empty competitive boggle rooms wake up to Round 1 in < 5ms (`game_room.py:4424`).
- **Asynchronous Watchdog**: Wrapped `ensure_next_board_ready()` in a background daemon thread (`game_room.py:1739-1746`) so Waitress request threads are never blocked.
- **Board Generator Clamping**: Clamped 4x4 Added Words emergency target word ranges to realistic bounds.
- **Commit History**: `3f3b9c7b`, `e73000da`.

---

### H. Concept 1: Word Length Radar (Desktop & Laptop Game Rooms)
- **Problem**:
  - In `hmm.png`, `.board-panel` stretched vertically to match the height of the left and right panels, leaving a 250px–350px empty field below Rotate and Transpose buttons.
- **Implementation**:
  - Created a responsive, live-updating **Word Length Radar** (`#word-length-radar`) directly below the word input and rotate/transpose row.
  - Dynamically calculates active board letter lengths (`totalByLen` vs `foundByLen`) and renders animated progress bars with vibrant bracket colors (4L: Emerald, 5L: Amber, 6L: Purple, 7L: Pink, 8L+: Indigo).
  - Displays instant optimistic completion updates as the player submits words, plus completion checkmarks (`✓`) when a letter bracket is fully cleared.
  - In intermission, switches seamlessly to a round completion debrief (`Word Length Breakdown`).
- **Strict Desktop / Laptop Only Constraint**:
  - Hidden unconditionally on mobile devices and tablets via `body.is-mobile #word-length-radar { display: none !important; }` and `@media (max-width: 992px) { display: none !important; }`.
  - In JS, skips execution if `window.innerWidth <= 992` or mobile user agent is detected.
- **Cache Busters**: `play.css?v=1795370006`, `play.js?v=1795390003`.
- **Commit History**: `62b8c47b`.

---

## 2. Infrastructure, Server Health & Resource Usage

| Node / Environment | Host / Port | Git Revision | Status |
| :--- | :--- | :--- | :--- |
| **Production VPS** | GoDaddy `132.148.72.249` | `62b8c47b` | **Online (HTTP 200)** |
| **GitHub Remote** | `origin/main` | `62b8c47b` | **Synchronized** |
| **Local Development** | Port `5001` | `62b8c47b` | **Online (HTTP 200)** |
| **PM2 Process ID** | `0` (`morpheme`) | Fork Mode | **Online (0% CPU, 4.5MB init)** |

---

## 3. Verification & Validation Summary
1. **HTTP Health Checks**:
   - `https://morpheme.games/` -> **200 OK**
   - `http://localhost:5001/` -> **200 OK**
2. **HTML Template Serving**:
   - Production `https://morpheme.games/` verified serving `dictionary-filter-dropdown` before `length-filter-dropdown`.
3. **Dropdown Interaction & Selenium Validation**:
   - Headless browser validation confirmed dynamic counts update properly across dictionary filter changes.
   - Screenshot visual artifact: [`preview_swapped_dropdowns.png`](file:///Users/jeffbabiak/.gemini/antigravity/brain/1d68eff8-1658-4346-9eae-f03c6f6b5147/preview_swapped_dropdowns.png).
