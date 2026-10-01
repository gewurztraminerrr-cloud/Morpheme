# Stable State Summary — September 30, 2026

## Latest Feature Commit ID
```
4ddd5b56d711b61b7d37604cc54ac2237941252b
```

## Start Over Point Tag
```
START_OVER_POINT_SEPTEMBER_30
```

## Synchronization Status

| Environment    | Status |
|----------------|--------|
| localhost      | ✅ Synchronized |
| GitHub (main)  | ✅ Synchronized |
| morpheme.games | ✅ Synchronized (HTTP 200 OK) |

---

## Session Features & Fixes (September 30, 2026)

### 1. New Users Lobby Guide ("What am I looking at?")
- **Header Button & Sticky Banner**:
  - Replaced `"ENTER A ROOM TO CONTINUE YOUR JOURNEY"` in the Lobby header with an interactive button:
    `<span>For New Users Asking: “What am I looking at?”</span>`
  - Styled with Morpheme’s obsidian-diamond shimmer glass aesthetic and purple neon glow.
  - Implemented instant local storage check (`morpheme_lobby_guide_dismissed`) inline on page load to prevent visual flash or layout shift.
- **"Welcome to Morpheme!" Comprehensive Modal**:
  - Full-featured glassmorphic modal window with purple border glow.
  - Designed with **no 'X' buttons** anywhere on the window.
  - Structured in the exact requested order:
    1. **Part 1: Main Lobby (Center Panel)**
    2. **Part 2: ACTIVE ROOMS (Right Panel)**
    3. **Part 3: PLAY SOLO OR WITH FRIENDS (Left panel on mobile devices, and below multiplayer Subanagrams on laptops and desktops)**
    4. **Part 4: Live Match Example & Word Pathing**
  - Two persistent action buttons in the sticky footer:
    - `"Show the link to this explanation again next time I enter Lobby"` (keeps the guide button active).
    - `"Remove the link and do not show this message again"` (restores original journey banner permanently).

### 2. Matchmaking Array: Constant Parameters (Dimensions & Time Limits)
- **Concept Definition**:
  - Explains that the **Game Type**, **Round Time**, and **Board Dimensions** are **constant parameters: they never change in game rooms**.
  - Once a room is opened or created in any public matrix, these three rules are locked for the lifetime of that room across all consecutive rounds.
- **Horizontal & Vertical Array Mechanics**:
  - **Top Horizontal Constant Parameter (Board Dimensions)**: Column headers (`4x4`, `4x6`, `5x7`, `6x8`) define the board size. All buttons below a constant parameter in the array strictly abide by that parameter.
  - **Left-Side Vertical Constant Parameter (Round Time)**: Row headers (`45s`, `3m`, `10m`, `24h`) define the round duration. All buttons across that row strictly abide by that time limit.
  - **Array Intersection Point**: Shows how the column and row intersect (e.g. column `4x6` and row `3m` meet at the button for a 3-minute round on a 4x6 board).
- **High-Fidelity Visual Matrix Diagram**:
  - Embedded an authentic mockup of the FCFS matrix with cyan top callout (`↓ Top Constant: All below are 4x6`), amber left-side callout (`3m`), and a highlighted gold intersection button (`Show Rooms [2]`).
  - Included three explanatory footnote cards detailing the parameters and intersection rules.

### 3. Accurate Definition of Closed Rooms & Inert Example Buttons
- **Corrected Closed Rooms Definition**:
  - Strictly updated to: *"A room is closed if there are **8 people playing in a room** or if the **rating of the user is outside the rating range given to the room**."*
  - Clarified that an active round in progress does not close a room.
- **Authentic Active Room Visual Cards (Open vs. Closed)**:
  - Replicated actual Morpheme room cards with player pills, rating badges, and exact button behavior.
  - Open room card displays `Spectate` vertically stacked above `Join`.
  - Closed room card displays full capacity (8/8), rating requirement, and only a full-width `Spectate` button.
  - All example buttons made completely inert (`pointer-events: none !important`) so clicks never trigger navigation.

### 4. Custom Draggable Scrollbar & Hidden Native Scrollbar
- Hidden native browser scrollbars cross-browser (`scrollbar-width: none; -ms-overflow-style: none; ::-webkit-scrollbar { display: none }`).
- Luminous purple thumb on dedicated track smoothly synchronizes with content scrolling.
- Fixed thumb movement by removing `top: 0 !important;` from CSS and updating `initLobbyGuideScrollbar()` with `thumb.style.setProperty('top', ...)` and `requestAnimationFrame`.
- Added listeners for pointer drag capture, track clicks, touch gestures, and mouse wheel input.

### 5. Selected Room Gameplay Showcase & Interactive Word Pathing ("How to Play")
- **Shrunken 4x4 Board**:
  - Rendered a compact 4x4 dice grid (~190px total width, 44px cells) styled with 3D dice appearance that fits cleanly on mobile and desktop without overflowing.
- **Interactive Word Pathing**:
  - Interactive pills (`STREAM`, `STOP`, `SLOW`, `TRAP`) demonstrating:
    - **`STREAM` (Multi-Directional Zigzag Curve)**: `(0,0)S → (0,1)T → (1,1)R → (2,1)E → (1,2)A → (1,3)M`
    - **`STOP` (Orthogonal Horizontal Line)**: `(0,0)S → (0,1)T → (0,2)O → (0,3)P`
    - **`SLOW` (Orthogonal Vertical Column)**: `(0,0)S → (1,0)L → (2,0)O → (3,0)W`
    - **`TRAP` (Diagonal Angle Step)**: `(0,1)T → (1,1)R → (1,2)A → (0,3)P`
  - Active cells light up in cyan with sequential step badges (`1`, `2`, `3`...).
  - Dynamic explanation card updates with word name, points awarded, movement classification, and coordinate path.
  - Simulated word input box shows submitted word and NWL validation status.
- **Core Rules Cards**:
  1. Adjacent Steps (King's Move)
  2. No Tile Reuse
  3. Length & Point Scaling
  4. Formats & Tactical Depth

### 6. AW Dictionary FAQ Guidance on Checking Is Valid with "All"
- Updated Added Words FAQ entry to advise searching in Tools -> Is Valid using the `"All"` word list first before suggesting missing words in the Forum.

### 7. Word Lists Added to Tools Dropdowns & "All" Standardization
- Added `CSW Only`, `NWL Uniques`, `New NWL Words`, `New CSW Words`, `New AW Words`, and `All New Words` across dictionary dropdowns.
- Standardized all dropdown `"ALL"` labels to mixed-case `"All"`.
- Backend dictionary loader restricted `16plus.txt` merging exclusively to full dictionaries.

### 8. Mobile Personal Quote Left-Alignment in Profile
- Left-aligned `"PERSONAL QUOTE"` and its body text on mobile devices in Profile matching `"ABOUT ME"`.

### 9. Asymmetric History Panel Padding Fix in Unscramble
- Fixed asymmetric left/right gutters under Status & Results in Tools -> Unscramble.

### 10. Store Magnetic Letters Copy Polish
- Updated JoyCat Silicone Magnetic Letters text: *"Then, when you open the fridge, you are reminded of them!"*

### 11. Word Pronunciation & Definition Hierarchy
- Standardized headword layout: word title on top, pronunciation immediately below the word title, and definition immediately below the pronunciation.

### 12. "Set to Default Sizes" Button in Settings
- Added a one-click button in Settings -> Appearance -> Board Size to reset 4x4 (82px), 4x6 (82px), 5x7 (65px), and 6x8 (54px) back to canonical defaults.

### 13. "Reset to Defaults" on Sound Theme in Sound Settings
- Added instant sound theme reset to Default Sound Theme.

### 14. Subtle Desktop Firefox Top Menu & Header Sizing (Desktops ONLY)
- Applied exclusively to Firefox on desktop viewports (`min-width: 901px`, `body:not(.is-mobile)`) to align its visual metrics with Microsoft Edge and Google Chrome:
  - **Slightly More Padding Above & Below Buttons**: `.nav-btn` vertical padding gently increased by ~1–2px (`8px 13px` / `6px 9px` / `5px 7px`).
  - **Slightly More Padding Between Buttons**: Inter-button gap expanded (`6px` / `4px` / `3px`).
  - **Slightly Larger MORPHEME MORE-FEEM & Top Menu**: Logo text (`1.92rem` / `1.28rem` / `1.16rem`), pronunciation (`0.88rem` / `0.77rem` / `0.72rem`), logo icon (`38px` / `30px` / `29px`), and nav font sizes subtly enlarged.
  - **Slightly Increased Overall Vertical Length**: Header `min-height` increased to `60px` / `48px` / `44px` with dynamic spacer synchronization.
  - All non-Firefox browsers (Edge, Chrome, Safari) and mobile devices remain 100% untouched.

### 15. Added Words Definition Engine Overhaul (Web Search & Non-Proper-Noun Sourcing)
- Programmatically fetches authentic dictionary definitions from Wordnik Century/Webster dictionaries and web search when words are added via the Added Words tab in Mods.
- Filters out proper nouns (cities, municipalities, ports, capitals, provinces, rivers, islands, personal names, surnames, and living people).
- Replaces trivial fallback strings with authentic lexicographical concepts.
- Integrates seamlessly into real-time Added Words pre-check (`get_word_definitions_for_aw_check`).

### 16. Definition Engine Logic Hardening Against SEO Titles & Marketing Headlines
- Solved greedy colon matching that previously captured website title tags and marketing slogans (e.g. `grandeval: Explore its Definition & Usage | RedKiwi Words`).
- Added Phrontistery rare/classical English dictionary query to directly extract genuine definitions for rare, literary, and archaic words (e.g., `GRANDEVAL` -> `(adjective) Of great age; ancient.`).
- Automatically strips website branding and page title trailers by splitting on pipes (`|`).
- Added strict junk patterns rejecting SEO marketing phrases (`Explore its Definition`, `Definition & Usage`, `Definition & Meaning`, `Usage & Examples`, `RedKiwi`, `WordHippo`, `Factsheet`, etc.).
- Enforced strict lexicographical grammar starters on colon-separated matches, guaranteeing that CTA titles, headers, and marketing slogans are completely rejected.

---

## How to Roll Back to This Point

```bash
# On localhost:
git checkout START_OVER_POINT_SEPTEMBER_30

# On production (SSH to server):
git fetch --tags -f origin
git checkout START_OVER_POINT_SEPTEMBER_30
pm2 reload morpheme
```
