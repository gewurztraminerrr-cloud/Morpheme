# Morpheme Stable State Summary — November 27, 2026 (Comprehensive)

**Stable Sync Point Commit ID**: `849b35b1b7105e83e4606b1f67d1b9c0211b5255` (`849b35b1`)  
**Timestamp**: November 27, 2026 ~ 13:50 EST  
**Environments Synchronized**: Localhost (`http://127.0.0.1:5001`), GitHub (`gewurztraminerrr-cloud/Morpheme:main`), Production VPS (`https://morpheme.games`), and App Client.

---

## 1. System Synchronization State

| Environment | Status | Commit ID | Branch | Verification Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Local Workspace** | Clean | `849b35b1` | `main` | All working tree code synchronized |
| **Local Server** | Running | `849b35b1` | `main` | Waitress daemon online on port 5001 (`HTTP/1.1 200 OK`, 890,268 bytes) |
| **GitHub Remote** | Clean | `849b35b1` | `main` | Pushed to `origin/main` |
| **Production VPS** | Online | `849b35b1` | `main` | PM2 process 0 online, Nginx `https://morpheme.games` (`HTTP/2 200 OK`, 890,268 bytes) |
| **Android / App** | Synchronized | `849b35b1` | `main` | WebView & web assets synchronized with production |

---

## 2. Key Features, Improvements & Bug Fixes (November 27, 2026)

### A. Reverse Dictionary Layout & Controls (Desktop & Mobile)
- **Desktop & Laptop Layout**:
  - Reorganized all input controls onto a single horizontal row.
  - Positioned the **Enter word, phrase, or concept** search textbox on the leftmost section, with match mode, dictionary, length filter, and search button aligned alongside it.
  - Generous whitespace and padding surround all controls and buttons.
- **Mobile Devices Layout (Mimicking Lists in Tools)**:
  - Structured the controls into an exact 2×2 grid mimicking Lists in Tools:
    - **Row 1**: Word Entry Textbox (left, 50% width) and "Similar to" Dropdown (right, 50% width).
    - **Row 2**: "Dictionary" Dropdown (left, 50% width) and "Length" Dropdown (right, 50% width).
    - **Row 3**: Full-width "Search" Button (100% width, matching Lists "Update" button).
  - Enforced `min-width: 0 !important; max-width: 100% !important; height: 34px !important;` across `@media (max-width: 900px)`, `(pointer: coarse)`, and `body.is-mobile` so that **all 4 controls (the textbox and all 3 dropdown menus) have strictly equal length (width)**.
  - Standardized container padding to `6px 8px` and margin to `6px` matching Lists in Tools.

### B. Subanagrams & Unscramble Light-Color Layout Support & True Neutral Light Grey
- **Color-Specific Light Layouts**:
  - Subanagrams words panel (`#sub-results-container`) and Unscramble history panel (`#unscramble-found-container`) now dynamically reflect all 10 theme color families:
    - **Green Themes** (`theme-light-green`, `theme-green`, etc.): Light green background (`#dcfce7`) with `#86efac` border.
    - **Blue Themes**: Light blue background (`#e0f2fe`) with `#93c5fd` border.
    - **Red Themes**: Light red background (`#fee2e2`) with `#fca5a5` border.
    - **Purple Themes**: Light purple background (`#f3e8ff`) with `#d8b4fe` border.
    - **Yellow Themes**: Light yellow background (`#fef9c3`) with `#fde047` border.
    - **Orange Themes**: Light orange background (`#ffedd5`) with `#fdba74` border.
    - **Pink Themes**: Light pink background (`#fce7f3`) with `#f472b6` border.
    - **Brown Themes**: Light brown background (`#f5ebe0`) with `#d7c4b7` border.
    - **Gray Themes**: Light gray background (`#ebebeb`) with `#cccccc` border.
    - **Dark/Black Themes**: Preserved dark glass background (`rgba(30, 30, 50, 0.6)`).
- **Pure Neutral Light Grey for White Layouts (`theme-white`)**:
  - Replaced the cool slate tone (`#f1f5f9` / `#cbd5e1`, which carried a slight icy blue hue) with pure, neutral light grey:
    - **Background**: `#ebebeb` (equal RGB `rgb(235, 235, 235)`).
    - **Border**: `#cccccc` (equal RGB `rgb(204, 204, 204)`).
  - High-contrast text colors (`#0f172a`, `#334155`) and row dividers adapt automatically across all themes.

### C. Rounded Edges on All Panels (Tools, Mods, Settings)
- Enforced unified rounded borders (`border-radius: 12px` to `14px`) across all panel surfaces in Tools, Mods, and Settings, ensuring no sharp rectangular edges remain.

### D. Settings Content Panels Contrast Elevation
- **Light Layouts**: Substantially brightened the content panels in the following Settings cards:
  - Synesthesia, Lobby Music, Triple Format Music, Board Sound Effects, Intermission Vibration Alert, Allow Private Messages, Allow Game Invitations, Game Rooms Chat, and Lobby Guide ("What am I looking at?").
- **Dark/Default Layouts**: Brightened the same content panels slightly closer to white, making them pop distinctly against surrounding container backgrounds.

### E. Sequence Tool Polish
- Output table in Sequence Tool brightened to `#ffffff` with a clean border on light-colored layouts.
- Added `padding-left: 12px !important;` to the "Please enter a sequence" placeholder text so it no longer touches the left edge.

### F. Tool Help/Guide "?" Buttons
- Ensured the "?" guide button is visible directly adjacent to tool titles (e.g., `Tools - Combo Checker ?`).

### G. Profile & User Experience Stability (from Morning Checkpoint)
- **About Me vs Personal Quote Textbox Sizing**: Locked both `#profile-quote-val` and `#profile-description-val` to an identical 110px height with smooth vertical scrolling.
- **PT SUM Title & Stat Alignment**: Fixed container height and alignment in `.stat-box` so PT SUM remains sub-pixel aligned with all other profile statistics even when exceeding 5+ digits.
- **Universal Description Scrolling**: Unified header descriptions across all 17 tools, Settings, Mods, and Forums to allow smooth touch dragging and scrolling.
- **Checkerboard Format Integrity**: Validated generation pipelines so Checkerboard board format always renders authentic checkerboard geometry.
- **Spinner Set Randomness Fidelity**: Aligned generation weights strictly with in-game published odds, eliminating premature 0:50 fallback overrides.
- **Strict + AW Word Count Enforcement**: Enforced minimum 300–400 word count tiers whenever Added Words (+ AW) are present.
- **Server Room-Manager Concurrency**: Isolated board generation from global manager locks, preventing logo-screen stalls.

---

## 3. Operational Instructions for "Start Over" Stable Point

If you ever need to reset or synchronize any machine or environment to this exact stable point:

### Git / Localhost:
```bash
git fetch origin main
git reset --hard 849b35b1b7105e83e4606b1f67d1b9c0211b5255
```

### Production VPS (`morpheme.games`):
```bash
ssh morpheme@132.148.72.249
cd /home/morpheme/morpheme
git fetch origin main
git reset --hard 849b35b1b7105e83e4606b1f67d1b9c0211b5255
pm2 restart 0
```

### Local Dev Server:
```bash
cd /Users/jeffbabiak
python3 app.py
```
