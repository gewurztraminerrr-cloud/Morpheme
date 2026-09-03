# Morpheme Stable State Summary — September 2, 2026

**Synchronization Checkpoint**: Complete alignment across GitHub (`origin/main`), Production (`morpheme.games`), Localhost, and Mobile App.
**Latest Commit ID**: `0da98ef9`
**Timestamp**: September 2, 2026

---

## 1. Summary of Work & Fixes Consolidated in this Baseline

### A. Lobby Layout, drawer, and Scrollbars
1. **Lobby Players & Chat Drawer Default Closed**:
   - Initialized `closeLobbyChatDrawer()` to ensure the bottom drawer is collapsed on initial page load, login, registration, and view transitions.
2. **Constant Height During Chat Typing**:
   - Removed mobile keyboard shrink CSS and JavaScript viewport-resizing hacks so the open chatbox maintains a smooth, constant height when typing messages.
3. **Themed Cyan/Blue Scrollbar Thumbs**:
   - Replaced magenta/red scrollbars with the themed cyan/blue gradient (`rgba(0, 240, 255, 0.7)` and `linear-gradient(...)`) across the Game Types panel, Solo/Friends panel, and Active Rooms list on mobile.
4. **Calibrated Mobile Viewport**:
   - Aligned the bottom boundary of the Active Rooms panel (`calc(100dvh - 166px)`) to sit 6px directly above the fixed 38px "Players in Lobby & Chat" drawer button.
5. **Mobile Morpheme Logo Placeholder**:
   - Scaled placeholder logo image inside `#rooms-list` to 190px (`drop-shadow(0 0 15px rgba(255, 255, 255, 0.2))`).
6. **Desktop Drawer Proportions**:
   - Configured `.lobby-slide-body` on desktop to `grid-template-columns: 1fr 2fr;` (Players box is half the width of the chatbox).

---

### B. Tools & Subanagrams
1. **Option Label Clarification**:
   - Renamed `"Word Guaranteed"` to **`"Word of Max Length Guaranteed"`** in dropdowns and toast feedback messages.
2. **Mobile Subanagrams Layout**:
   - Moved the green **Submit** button to its own full-width row (`.sub-input-row`) directly below the input textbox and above the action buttons.
   - Positioned "Reveal All Subanagrams" and "Clear Found" side-by-side in `.sub-action-buttons-row`.
   - Tuned bottom padding (`padding: 12px 14px 14px 14px !important;`) and collapsed empty feedback messages on `.subanagrams-game-area` so the card has neat, proportional spacing without wasted space.

---

### C. Leaderboards
1. **Mobile Selection & Tap Highlight Suppression**:
   - Added `-webkit-tap-highlight-color: transparent !important;`, `user-select: none !important;`, `-webkit-touch-callout: none !important;`, and `outline: none !important;` to `.lb-user-cell`, `.rating-square`, and their child elements.
   - Added selection clearing `window.getSelection()?.removeAllRanges()` on click to prevent blue highlight artifacts.

---

### D. Lexicon & Dictionary Database Integrity
1. **Purged Synthetic / Vague Definitions**:
   - Cleaned `dictionaries/Definitions.txt` of all synthetic definitions.
   - Preserved **677,150 authentic definitions**.
   - Exported missing word lists for offline reference:
     - [`/Users/jeffbabiak/dictionaries/words_missing_nwl_csw.txt`](file:///Users/jeffbabiak/dictionaries/words_missing_nwl_csw.txt)
     - [`/Users/jeffbabiak/dictionaries/words_missing_definitions.txt`](file:///Users/jeffbabiak/dictionaries/words_missing_definitions.txt)

---

## 2. Synchronization Status

| Target | Status | Commit / Version |
| :--- | :--- | :--- |
| **Local Repository** | Clean (`main`) | `0da98ef9` |
| **GitHub Remote** | Synchronized (`origin/main`) | `0da98ef9` |
| **Production Server** | Synchronized (`132.148.72.249`) | `0da98ef9` |
| **PM2 Process** | Online (`morpheme` id: 0) | Restarted & Verified |
