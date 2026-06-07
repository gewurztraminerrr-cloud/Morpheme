# Stable State Summary — June 6, 2026 (Start Over Point)

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `48cf725` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `48cf725` / `snapshot-current` / `START_OVER_POINT_JUNE_6` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `48cf725` / `snapshot-current` | ✅ Fully Deployed & PM2 Restarted |

**All environments are 100% synchronized at the latest commit 48cf725.**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py`.
The active recovery points `START_OVER_POINT_JUNE_6` and `snapshot-current` tags have been successfully updated and pushed to GitHub.

---

## Serving Versions (cache-busted)

| File / Style | Version | Description |
|--------------|---------|-------------|
| `/css/play.css` | `v=86` | Added overrides for `.bonus-highlight` elements when actively selected or highlighted to hide the lime green background and glow. |
| `/css/howtoplay.css` | `v=10` | Implemented visible list grid and button styles for FAQ quick navigation. |
| `/js/app.js` | `v=41` | Implemented scroll navigation and pulse highlight logic for FAQ links. |
| `/js/play.js` | `v=142` | Disabled definitions panel gold flashing animation at round complete in 24h rooms. |
| `templates/index.html` | *Dynamic* | Replaced the quick-nav dropdown selector with the visible question link button grid, bumped howtoplay.css cache-buster to `v=10`, app.js cache-buster to `v=41`, play.js cache-buster to `v=142`, and play.css cache-buster to `v=86`. |

---

## Work Completed on June 6, 2026 (and follow-ups on June 7)

### 1. Visible Link Navigation Grid for FAQ
* **Goal achieved:** Allow players to select/click visible questions under "How to Play" but above the FAQ section, taking them directly to the corresponding answer in the FAQ with smooth scrolling and micro-animations.
* **Implementation (`templates/index.html` & `static/css/howtoplay.css` & `static/js/app.js`):**
  * Removed the select dropdown widget.
  * Replaced it with a visible question button grid (`.faq-questions-grid`).
  * Styled each link button with a glassmorphic background, star bullet indicator, and hover slide animations.
  * Configured click event listeners to smoothly scroll to the correct FAQ block and trigger a pulsing glow animation.

### 2. Disable Definitions Panel Gold Flashing in 24h Rooms
* **Goal achieved:** Prevent the definitions panel container (`.definitions-panel`) from slowly flashing gold at 12 AM (intermission start) in 24h rooms while keeping the winner announcement text visible.
* **Implementation (`static/js/play.js` & `templates/index.html`):**
  * Modified the intermission winner announcement logic in `play.js` to conditionally skip adding the `.winner-flash` class if the room is a 24h room (identified by `state.time_limit >= 7200`).
  * Incremented the cache-buster version parameter for `/js/play.js` in `templates/index.html` from `?v=141` to `?v=142`.

### 3. Either/Or Word Involvement Optimization (Target 1/3)
* **Goal achieved:** Optimize Either/Or tile letter pairing and placement so that approximately 1/3 (33%) of the valid words generated on the board involve the Either/Or tile.
* **Implementation (`board_generator.py`):**
  * Updated the E/O tile selection loops (both standard and emergency paths) to temporarily apply each candidate cell and letter partner, solve the board using `_solve_board(store_paths=True)`, and calculate the exact percentage of words passing through the tile.
  * Checks for ratio difference from `0.333`. If a candidate cell is within 5% of `1/3` (28.3% to 38.3%), it is accepted immediately. Otherwise, it searches and selects the candidate with the smallest absolute difference from 1/3.
  * Corrected path membership lookup (`cell in temp_words[w]`) in `_solve_board` result parsing.

### 4. Highlight Selection Override for Either/Or and Bonus Letter Tiles
* **Goal achieved:** Ensure that when a player highlights or selects an Either/Or tile or a Bonus Letter tile, the lime green styling (background, border, text color, box-shadow, and pulsing animation) is completely overridden by the standard selection or typing highlight colors.
* **Implementation (`static/css/play.css` & `templates/index.html`):**
  * Added CSS rules to override `.board-cell.selected.bonus-highlight`, `.board-cell.current.bonus-highlight`, `.board-cell.typing-highlight.bonus-highlight`, and `.board-cell.review-highlight.bonus-highlight`.
  * Set these classes to use the standard theme accent colors (`--highlight-mouse-color` and `--highlight-typing-color`), reset text color to `#000`, and set `animation: none !important` to stop the pulsing lime green box-shadow animation when selected.
  * Injected these overrides both in the external stylesheet and the templates inline stylesheet for maximum safety and cache independence, and bumped the stylesheet version parameter in `templates/index.html` from `?v=85` to `?v=86`.
