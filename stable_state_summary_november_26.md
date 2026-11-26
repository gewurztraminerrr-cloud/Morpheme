# Morpheme Stable State Summary — November 26, 2026

**Stable Sync Point Commit ID**: `607095f082e171b3e70cf2771d9e26e3869269d0` (`607095f0`)  
**Timestamp**: November 26, 2026 ~ 14:15 EST  
**Environments Synchronized**: Localhost (`http://127.0.0.1:5001`), GitHub (`gewurztraminerrr-cloud/Morpheme:main`), Production VPS (`https://morpheme.games`).

---

## 1. System Synchronization State

| Environment | Status | Commit ID | Branch | Verification Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Local Workspace** | Clean | `607095f0` | `main` | Zero uncommitted tracked code changes |
| **Local Server** | Running | `607095f0` | `main` | Waitress server daemon active on port 5001 (`HTTP 200 OK`) |
| **GitHub Remote** | Clean | `607095f0` | `main` | Fully pushed to `origin/main` |
| **Production VPS** | Online | `607095f0` | `main` | PM2 process 0 online, `https://morpheme.games` (`HTTP/2 200`) |

---

## 2. Key Features & Enhancements Included in this Stable Point

### A. Desktop/Laptop Game Room Return Navigation (FC / CC "Back" Auto-Scroll)
- **Problem**: When a user on a desktop or laptop was playing in a game room, clicked a word to view its definition, and selected "FC" (Find Count) or "CC" (Combo Checker), clicking "Back" to return to the game room reset the viewport scroll to `(0, 0)`. This left the static top menu bar in view and cut off the bottom of the game room, forcing the user to manually scroll down every time.
- **Fix Applied**:
  - Implemented `scrollGameRoomToBottomDesktop` in `static/js/play.js`. It measures the top header/menu bar height and total document height, smoothly or immediately scrolling the viewport down to the bottom so the top menu bar is tucked out of sight and the full room layout is visible.
  - In `static/js/tools.js`, enhanced `toolsGoBack` when `navEntry.originPage === 'page-play'` on desktop/laptop devices to execute `scrollGameRoomToBottomDesktop` across render frames (0ms, rAF, 50ms, 120ms, 250ms, 450ms).
  - In `static/js/app.js`, updated `showPage` so returning from definition navigation to `page-play` does not reset the desktop scroll to `(0, 0)`.
  - In `showWordDefinitionPopup`, guaranteed that `originPage` is recorded as `'page-play'` whenever the definition popup is invoked from a game room.

### B. Validation Timing & Flashing Effect Synchronization (VALID vs INVALID)
- **Problem**: In 24h rooms and penalty modes, invalid word submissions had a noticeable lag before red validation text and red letter flashing appeared (waiting 200–400ms for a round-trip network response), whereas valid words validated and flashed instantly at 0ms.
- **Fix Applied**:
  - In `static/js/play.js` (`submitWord`), removed the artificial `!isPenaltyMode && !is24H` restriction on local optimistic invalid validation. Now, all game modes (timed rooms, 24h rooms, Solo/Private matches, and penalty formats) trigger local optimistic feedback and board tile flashes at the exact same instant (0ms) as valid words.
  - Synchronized duration: Letter tile flashes on the board (`tile-flash-blue` vs `tile-flash-red`) both run for **150ms**; input box background flashes run for **150ms**; validation status messages reset after **3000ms** (3 seconds).
  - Added board path resolution for typed invalid words (`findWordPathOnBoard` / `findWordPathOnCube`) in `showValidationFeedback` so that any invalid word traceable on the board flashes its constituent letter tiles identically to valid words.
  - Guarded tile flash (`_tileFlashTimeout`) and input background flash (`_inputFlashTimeout`) timeouts so rapid subsequent submissions never cut short the snappy 150ms flash duration.

### C. All Words Table Selected Word Pointer Arrow (`▶`)
- **Problem**: In the All Words table during intermission/round review, the pointer arrow (`▶`) to the left of the selected word was partially clipped/hidden on the left margin.
- **Fix Applied**:
  - Adjusted container padding to 20px on both sides (`padding: 4px 20px !important`).
  - Positioned `.all-words-pointer` cleanly with `left: -16px` and boosted the `pin-bounce` keyframe animation for an eye-catching, fully visible bouncing indicator.

### D. Intermission Dropdowns Percentage Display
- **Enhancement**: In intermission dropdown menus above the table of words:
  - Letter length dropdown options now indicate percentages found formatted as `(12/34)  35%` (with two spaces between bracket and percentage).
  - The main word lists selector dropdown next to the letter lengths dropdown also features percentages found.

### E. N Mania Board Generation (33% Common Letter Rule)
- **Enhancement**: Enforced that on letter-mania boards (such as N Mania in Solo 4x4 and other grid dimensions), 33% of a common letter appears on the board (e.g. for N Mania on 4x4, 33% of 16 cells is 5.28 -> 5 N's total, fixing the previous occurrence of only 4 N's).

### F. 24h Rooms Safari Scroll Stability
- **Problem**: In 24h rooms on Safari on laptops, scrolling down the found words list caused the list to twitch/hiccup back to the top on every 1-second timer tick.
- **Fix Applied**: Stabilized scroll anchoring during 1-second live state polling so the user's scroll position in the list remains steady on all devices and browsers.

---

## 3. Operational Instructions for "Start Over" Stable Point

If you ever need to reset or sync any machine to this exact point:
```bash
git fetch origin main
git reset --hard 607095f082e171b3e70cf2771d9e26e3869269d0
```
Production VPS reload:
```bash
cd /home/morpheme/morpheme && git fetch origin main && git reset --hard origin/main && pm2 restart 0
```
Local server restart:
```bash
cd /Users/jeffbabiak && python3 app.py
```
