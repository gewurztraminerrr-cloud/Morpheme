# Morpheme Stable State Summary — November 27, 2026 (Evening Sync)

**Stable Sync Point Branch**: `main`  
**Timestamp**: November 27, 2026 ~ 19:20 EST  
**Environments Synchronized**: Localhost (`http://127.0.0.1:5001`), GitHub (`gewurztraminerrr-cloud/Morpheme:main`), Production VPS (`https://morpheme.games`), and App Client.

---

## 1. System Synchronization State

| Environment | Status | Verification Notes |
| :--- | :--- | :--- |
| **Local Workspace** | Clean | Working tree clean, all changes committed and synchronized |
| **Local Server** | Running | Waitress server online on port 5001 (`HTTP/1.1 200 OK`) |
| **GitHub Remote** | Clean | Pushed to `origin/main` |
| **Production VPS** | Online | PM2 process 0 online, Nginx `https://morpheme.games` (`HTTP/2 200 OK`) |
| **Android / App** | Synchronized | WebView & web assets synchronized with production |

---

## 2. Key Features, Improvements & Bug Fixes (November 27, 2026)

### A. Subanagrams & Unscramble Mobile Focus & Border Feedback
- **Persistent Textbox Focus**:
  - Ensured that on mobile devices, typing focus remains inside the submission textbox immediately after submitting a word or sequence in Subanagrams and Unscramble.
- **Dynamic Border Flash Feedback**:
  - Border highlights green briefly (`.valid-flash`) with a subtle glow when entering a valid word.
  - Border highlights red briefly (`.invalid-flash`) with a subtle glow when entering an invalid word or sequence.
  - Smooth CSS animation and transition across all devices.

### B. Help / Guide Modal Dialog Polish
- **Centered "Got it" Button**:
  - Centered the blue "Got it" confirmation button inside the description modal opened via the `?` button across all tools, settings, and mods.

### C. Offensive Words Management & Mod Controls
- **Word Extraction & Definitions**:
  - Added offensive words from `removed.txt` to Added Words (AW) staging table.
  - Incorporated full definitions from `removeddefs.txt` into the definitions database table.
- **Mod Toggle Switch**:
  - Added an "Include Offensive Words in Game" toggle switch panel inside `Added Words` in Mods directly below "Include Added Words in Game".
  - Perfectly styled to match the "Include Added Words in Game" panel across all dark and light themes (`theme-white`, `theme-light-gray`, `theme-light-blue`, etc.).
  - When OFF (default), offensive words are completely excluded from board solving, Lists in Tools, Word Charts, Reverse Dictionary, and lookups throughout the application.

### D. Database Submission in Mods & Word Processing Logic
- **Smart Duplicate Handling**:
  - Replaced legacy batch overwrite behavior with individual word validation.
  - When uploading any text file (`newNWL.txt` or `newCSW.txt`), each word is verified against the respective dictionary.
  - Words already present are recognized as valid and skipped without error.
  - Words not present are added, recorded in the New Words tracking table, and automatically promoted (removed) from Added Words.
- **Prominent Explanation Panel**:
  - Positioned an informative `.mod-action-card` explanation panel prominently above the upload status and action buttons (`Select .txt File` and `Submit to Database`).
  - Clearly explains what happens when uploading files containing only new words vs. files containing both existing and new words.
  - Moved out of the hidden `.tool-header` container to ensure immediate visibility in all themes.

### E. Dynamic Word Lookup in Word Charts
- **Streamlined Metric Cards**:
  - Removed the redundant "Division Details" container with definition (the definition remains easily accessible by clicking the word name).
  - Formatted Dynamic Value card to `${data.points} pts of ${data.max_points} pts` (e.g. `9 pts of 9 pts`), matching the `Div X of Y` division status formatting.

### F. Reverse Dictionary Layout & Controls (Desktop & Mobile)
- **Desktop & Laptop Layout**:
  - Reorganized all input controls onto a single horizontal row with generous padding and whitespace.
- **Mobile Equal Control Lengths**:
  - Structured controls into a 2×2 grid mimicking Lists in Tools:
    - Row 1: Word Entry Textbox (50%) and "Similar to" Dropdown (50%).
    - Row 2: "Dictionary" Dropdown (50%) and "Length" Dropdown (50%).
    - Row 3: Full-width "Search" Button (100%).
  - Enforced strictly equal length/width across all 4 controls on mobile.

### G. Subanagrams & Unscramble Light-Color Layout Support & True Neutral Light Grey
- **Color-Specific Light Layouts**:
  - Subanagrams and Unscramble panels dynamically adapt to all 10 theme color families (green, blue, red, purple, yellow, orange, pink, brown, gray, dark glass).
- **True Neutral Light Grey for `theme-white`**:
  - Replaced cool slate tones with pure neutral light grey (`#ebebeb` background, `#cccccc` border).

### H. Universal Panel Edge Rounding & Contrast Elevation
- Unified `border-radius: 12px` to `14px` across all panel surfaces in Tools, Mods, and Settings.
- Brightened content panels in Settings (Synesthesia, Audio, Chat, etc.) for crisp contrast across all light and dark themes.

### I. Word Charts — Point Values & Divisions (Subtab 1)
- **Tight Table Wrap Without Empty Space**:
  - Set `#tool-word-chart #subtab-pane-wc-points .word-chart-table-container` and its scroll area to `flex: 0 1 auto !important; height: auto !important; max-height: 100% !important;` so the container wraps tightly around only the rows with data, eliminating empty space below the `16+ Letters` row.
  - Removed the bottom border on the last row (`16+ Letters`) and added `7px` bottom corner radii for seamless integration with the container border.

### J. Word Charts — Dictionary Word Counts (Subtab 2)
- **Tight Table Wrap & Seamless Total Row**:
  - Set `#tool-word-chart #subtab-pane-wc-dicts .word-chart-table-container` and its scroll area to `flex: 0 1 auto !important; height: auto !important; max-height: 100% !important;`, wrapping tightly around table rows with zero empty space inside the container, matching Subtab 1.
  - Lowered and integrated the `Total` row to sit flush at the bottom edge with `7px` bottom corner radii, zero bottom border, and no whitespace.
- **Cache Invalidation**:
  - Bumped `play.css` and `tools.js` to `?v=1795770010` in `templates/index.html` to eliminate stale cached CSS in Chrome, Safari, Firefox, and mobile.

### K. Words User is Good at — Light Theme Pagination Buttons
- **High-Contrast "Previous" & "Next" Buttons**:
  - Replaced transparent/low-contrast buttons in light themes (`theme-white`, `theme-light-gray`, `theme-light-light-gray`, etc.) with pure white background (`#ffffff`), 1px slate border (`#cbd5e1`), dark slate text (`#1e293b`), and subtle drop shadow (`box-shadow: 0 1px 3px rgba(0,0,0,0.08)`).
  - Hover state darkens border to `#94a3b8` with subtle background tint `#f8fafc`.
  - Disabled state gracefully dims to `rgba(255, 255, 255, 0.6)` with `#94a3b8` text and `0.5` opacity.

---

## 3. Operational Instructions for "Start Over" Stable Point

If you ever need to reset or synchronize any machine or environment to this exact stable point:

### Git / Localhost:
```bash
git fetch origin main
git reset --hard HEAD
```

### Production VPS (`morpheme.games`):
```bash
ssh morpheme@132.148.72.249
cd /home/morpheme/morpheme
git fetch origin main
git reset --hard origin/main
pm2 restart 0
```
