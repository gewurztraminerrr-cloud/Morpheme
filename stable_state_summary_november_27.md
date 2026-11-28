# Morpheme Stable State Summary — November 27, 2026 (End of Day / Start Over Point)

**Stable Sync Point Branch**: `main`  
**Latest Feature Commit ID**: `e71c810cea2c327b16a803be0e8bdf7ca0b9c580`  
**Start Over Point Tags**: `START_OVER_POINT_NOVEMBER_27`, `START_OVER_POINT_TODAY`, `stable-2026-11-27`  
**Timestamp**: November 27, 2026 ~ 23:45 EST  
**Environments Synchronized**: Localhost (`http://127.0.0.1:5001`), GitHub (`gewurztraminerrr-cloud/Morpheme:main`), Production VPS (`https://morpheme.games`), and App Client.

---

## 1. System Synchronization State

| Environment | Status | Verification Notes |
| :--- | :--- | :--- |
| **Local Workspace** | ✅ Synchronized | Working tree clean, all files tracked and committed at `e71c810c` |
| **Local Server** | ✅ Synchronized | Running on port 5001 (`HTTP/1.1 200 OK`), verified via Selenium |
| **GitHub Remote** | ✅ Synchronized | Pushed to `origin/main` (`gewurztraminerrr-cloud/Morpheme`) |
| **Production VPS** | ✅ Synchronized | Deployed to 132.148.72.249, PM2 online (`HTTP/2 200 OK` on `https://morpheme.games/`) |
| **App / Client** | ✅ Synchronized | Mobile and desktop layouts tested and verified across all themes |

---

## 2. Key Features, Improvements & Bug Fixes (November 27, 2026)

### A. Dynamic Word Lookup & Tally Divisions Bottom Refresh Buttons
- **Full-Width Persistent Refresh Button in Tools**:
  - In Tools → Word Charts → Dynamic Word Lookup (`#subtab-pane-wc-lookup`), the "🔄 Refresh Tally Data" button (`#wc-refresh-tally-btn`) extends 100% horizontally across the screen along the bottom.
  - Sited in `.wc-lookup-bottom-bar`, remaining pinned along the bottom even when the user scrolls through the rarity division cards above it.
- **Mobile Side-by-Side Search Bar**:
  - On mobile devices (`body.is-mobile` / `@media (max-width: 900px)`), the word input field (`#wc-word-input`) and the "Check Point Value" button (`#wc-word-search-btn`) are arranged side-by-side (`flex-direction: row; flex-wrap: nowrap;`).
  - Text input handles flexible shrinking/expansion cleanly without overflow; 16px font size preserves iOS Safari auto-zoom prevention.
- **Full-Width Sticky Refresh Button in Mods**:
  - In Mods → Tally Divisions Difficulty (`#mod-tab-tally-divisions`), the "🔄 Repopulate Examples" button (`#tally-repopulate-examples-btn`) is positioned in a sticky bottom bar (`.mod-tally-bottom-bar`) across the bottom of the screen.
  - Remains persistent along the bottom while scrolling through division threshold dropdowns and example cards.
- **Bonus Word Scoring Calculation**:
  - Fixed baseline bonus word point calculation in Dynamic Word Lookup so words begin at the default maximum division baseline (9 points) rather than minimum (5 points), giving correct 16 pts (9 + 7) for 7-letter words like `TESTING`.

### B. Mods: Lobby Notice & Forum Maintenance Mobile Optimization
- **Lobby Notice Textbox Full-Screen Extension**:
  - Extended the vertical length of the "Current Live Announcement" and "Compose Announcement" textboxes to fit the full mobile viewport comfortably without outer panel scrolling.
  - Internal textarea scrolling is preserved for long announcements.
- **Forum Maintenance Nested Double Boxes**:
  - For each Category section within Forum Maintenance on mobile devices, wrapped the card in an outer colored container with a distinct inner colored container, creating an unmistakable nested visual hierarchy showing that threads and actions belong together.
- **Mod Tab Display Isolation**:
  - Scoped `#mod-tab-notice` styling to `#mod-tab-notice.active` on mobile, eliminating an issue where the Lobby Notice panel remained forced visible over other mod tabs (such as Tally Divisions Difficulty and Forum Maintenance).

### C. Profile & Aesthetics
- **Profile User Photo Glow Removal**:
  - Removed the continuous glowing box-shadow animation around user photos on Profile across all user accounts, creating a cleaner, modern portrait presentation.

### D. Forum Enhancements
- **Category Description Inside '?' Help Button**:
  - Moved Forum category descriptions (such as in Suggestions/Ideas) into an interactive `?` help button immediately to the right of the category title, matching the universal help button pattern across Tools, Mods, and Settings.
  - Category titles and help popups render consistently across all themes.

### E. Tournament Round Standings & Replay
- **Round Replay "All" Tab Word Distinction**:
  - Fixed Round Replays in Tournament Round Standings so when selecting "All", found words are highlighted in accent blue while missed words render in muted text, restoring consistent behavior with other Round Replays.

### F. Subanagrams & Unscramble Mobile Focus & Border Feedback
- **Persistent Textbox Focus**:
  - Ensured that on mobile devices, typing focus remains inside the submission textbox immediately after submitting a word or sequence in Subanagrams and Unscramble.
- **Dynamic Border Flash Feedback**:
  - Border highlights green briefly (`.valid-flash`) with a subtle glow when entering a valid word.
  - Border highlights red briefly (`.invalid-flash`) with a subtle glow when entering an invalid word or sequence.

### G. Help / Guide Modal Dialog Polish
- **Centered "Got it" Button**:
  - Centered the blue "Got it" confirmation button inside the description modal opened via the `?` button across all tools, settings, and mods.

### H. Offensive Words Management & Mod Controls
- **Word Extraction & Definitions**:
  - Added offensive words from `removed.txt` to Added Words (AW) staging table.
  - Incorporated full definitions from `removeddefs.txt` into the definitions database table.
- **Mod Toggle Switch**:
  - Added an "Include Offensive Words in Game" toggle switch panel inside `Added Words` in Mods directly below "Include Added Words in Game".
  - Styled to match the "Include Added Words in Game" panel across all dark and light themes (`theme-white`, `theme-light-gray`, `theme-light-blue`, etc.).
  - When OFF (default), offensive words are completely excluded from board solving, Lists in Tools, Word Charts, Reverse Dictionary, and lookups throughout the application.

### I. Database Submission in Mods & Word Processing Logic
- **Smart Duplicate Handling**:
  - Replaced legacy batch overwrite behavior with individual word validation.
  - When uploading any text file (`newNWL.txt` or `newCSW.txt`), each word is verified against the respective dictionary.
  - Words already present are recognized as valid and skipped without error.
  - Words not present are added, recorded in the New Words tracking table, and automatically promoted (removed) from Added Words.
- **Prominent Explanation Panel**:
  - Positioned an informative `.mod-action-card` explanation panel prominently above the upload status and action buttons (`Select .txt File` and `Submit to Database`).
  - Moved out of the hidden `.tool-header` container to ensure immediate visibility in all themes.

### J. Reverse Dictionary Layout & Controls (Desktop & Mobile)
- **Desktop & Laptop Layout**:
  - Reorganized all input controls onto a single horizontal row with generous padding and whitespace.
- **Mobile Equal Control Lengths**:
  - Structured controls into a 2×2 grid mimicking Lists in Tools:
    - Row 1: Word Entry Textbox (50%) and "Similar to" Dropdown (50%).
    - Row 2: "Dictionary" Dropdown (50%) and "Length" Dropdown (50%).
    - Row 3: Full-width "Search" Button (100%).
  - Enforced strictly equal length/width across all 4 controls on mobile.

### K. Subanagrams & Unscramble Light-Color Layout Support & True Neutral Light Grey
- **Color-Specific Light Layouts**:
  - Subanagrams and Unscramble panels dynamically adapt to all 10 theme color families (green, blue, red, purple, yellow, orange, pink, brown, gray, dark glass).
- **True Neutral Light Grey for `theme-white`**:
  - Replaced cool slate tones with pure neutral light grey (`#ebebeb` background, `#cccccc` border).

### L. Universal Panel Edge Rounding & Contrast Elevation
- Unified `border-radius: 12px` to `14px` across all panel surfaces in Tools, Mods, and Settings.
- Brightened content panels in Settings (Synesthesia, Audio, Chat, etc.) for crisp contrast across all light and dark themes.

### M. Word Charts — Point Values & Divisions (Subtab 1) & Dictionary Word Counts (Subtab 2)
- **Tight Table Wrap Without Empty Space**:
  - Set table containers and scroll areas to `flex: 0 1 auto !important; height: auto !important; max-height: 100% !important;` so the containers wrap tightly around only the rows with data, eliminating empty space below the `16+ Letters` row and `Total` row.
  - Seamless bottom corner radii (`7px`) for flawless container integration.
- **Cache Invalidation**:
  - Bumped cache busters across CSS and JS files (`play.css?v=1795770016`, `tools.js?v=1795770013`, `forum.css?v=1795770020`, `tournaments.js?v=1795770011`).

### N. Words User is Good at — Light Theme Pagination Buttons
- **High-Contrast "Previous" & "Next" Buttons**:
  - Pure white background (`#ffffff`), 1px slate border (`#cbd5e1`), dark slate text (`#1e293b`), and subtle drop shadow in all light themes.

---

## 3. Operational Instructions for "Start Over" Stable Point

If you ever need to reset or synchronize any machine or environment to this exact stable point:

### Git / Localhost:
```bash
git fetch origin main
git checkout main
git reset --hard START_OVER_POINT_NOVEMBER_27
```

### Production VPS (`morpheme.games`):
```bash
ssh morpheme@132.148.72.249
cd /home/morpheme/morpheme
git fetch origin main
git checkout main
git reset --hard START_OVER_POINT_NOVEMBER_27
pm2 restart 0
```
