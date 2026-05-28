# Morpheme Stable State Summary - May 28, 2026

This summary documents the stable state of the Morpheme application as of May 28, 2026. The focus of recent work has been on board generation robustness (watchdogs and re-roll healing), high-fidelity CSS layout refinements, and animation synchronization.

## 🚀 Key Improvements & Bug Fixes

### 1. Interactive How to Play Swipe Animation (Middle Card)
- **Fluid 30-Second Loop**: Expanded the second demo board ("Length Matters") animation under the How to Play modal to cycle through **10 specific words** over 30 seconds (3 seconds per word) in the exact requested order: `PLAY`, `PALY`, `PAL`, `ALP`, `LAP`, `YAP`, `PYA`, `PAY`, `PLY`, `LAY`.
- **Perfect Highlight Synchronization**: Fixed a keyframe synchronization bug in `highlightMulti3` (the `A` tile) where it incorrectly lit up at `86% - 89%` during the swipe preview of the 9th word `PLY` (which contains no `A` letter). The highlights are now perfectly synchronized with the swiping cursor tracer.

### 2. Snug Clues Tab Layout (24h Rooms)
- **High-Fidelity Grid Layout**: Cleaned up the Clues tab in 24-hour accumulative rooms to render exactly **2 columns per row** across mobile, laptop, and desktop views.
- **Removed Overrides**: Eliminated hardcoded inline-styled overrides inside `play.js` so that the clues panel respects clean, fluid CSS grid rules.
- **Snug Sizing Polish**: Tightly optimized `.clue-item` cards, reducing padding (from `20px 15px` to `10px 8px`), margins, and minimum height (from `100px` to `65px`) to make all letter/stat contents fit snugly and elegantly inside the container.

### 3. Spinner Set Odds & 3D Cube Difficulty Alignment
- **Parameter Parity**: Updated `_get_uniqueness_range` and `get_difficulty_label` in `board_generator.py` for the 3D Cube board format (`is_cube`) to match the `is_6x8` uniqueness thresholds exactly:
  - **Easy**: `(0.0, 0.34)`
  - **Medium**: `(0.35, 0.49)`
  - **Hard**: `(0.50, 1.0)`
- **UI Synchronization**: Updated the Spinner Set Odds modal text inside `index.html` to correctly display the aligned ranges for the Cube board format.

### 4. Login Page Case-Sensitivity Warning
- **Snug Warnings**: Added a stylish warning notice indicating that both Username and Password are case sensitive.
- **Clean Layout Flow**: Placed the note elegantly inside both the Login and Register forms, sitting directly below the Password inputs and above the CAPTCHAs for high visibility and clean vertical alignment.

### 5. Self-Healing Stuck Board Watchdog
- **Stuck Intermission Recovery**: Implemented a 10-second stuck board watchdog at `0:00:00` (timer at zero during intermission).
- **Auto Re-spin & Clear**: If the promo stage is blocked for more than 10.0 seconds trying to generate a board (often due to strict 5x7 or 6x8 Hard uniqueness parameters), it automatically re-spins the Spinner Set to a new set of parameters (downgrading any 'Hard' difficulty to 'Medium' to guarantee rapid completion), clears stale staging variables, and triggers a fresh background search.
- **Abort Guard**: Added a parameter-change abort guard inside `start_next_round` emergency generation to safely terminate stale threads.

---

## 🛠 Active Features & Configuration
- **Board Dimensions**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
- **Dictionaries**: NWL (American) and CSW (International) Tries.
- **Difficulty Tiers**: Easy, Medium, Hard, and Expert.
- **Game Modes**: Standard, Accumulative (24h Rooms with midnight boundary resets), FCFS, Split, and Private Matches.

---

**Stable Point Tag (snapshot-current)**: a34b2c9  
**Start Over Tag (START_OVER_POINT_MAY_28)**: a34b2c9  
**GitHub Push**: Completed  
**Status**: Stable / Production Ready / Synchronized
