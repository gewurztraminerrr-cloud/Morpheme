# Stable State Summary — June 7, 2026 (Start Over Point)

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `7721533` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `7721533` / `snapshot-current` / `START_OVER_POINT_JUNE_7` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `7721533` / `snapshot-current` | ✅ Fully Deployed & PM2 Restarted |

**All environments are 100% synchronized at the latest commit 7721533.**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py`.
The active recovery points `START_OVER_POINT_JUNE_7` and `snapshot-current` tags have been successfully updated and pushed to GitHub.

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

## Work Completed on June 7, 2026

### 1. Either/Or Word Involvement Optimization (Target 1/3)
* **Goal achieved:** Optimize Either/Or tile letter pairing and placement so that approximately 1/3 (33%) of the valid words generated on the board involve the Either/Or tile.
* **Implementation (`board_generator.py`):**
  * Updated the E/O tile selection loops (both standard and emergency paths) to temporarily apply each candidate cell and letter partner, solve the board using `_solve_board(store_paths=True)`, and calculate the exact percentage of words passing through the tile.
  * Checks for ratio difference from `0.333`. If a candidate cell is within 5% of `1/3` (28.3% to 38.3%), it is accepted immediately. Otherwise, it searches and selects the candidate with the smallest absolute difference from 1/3.
  * Corrected path membership lookup (`cell in temp_words[w]`) in `_solve_board` result parsing.

### 2. Highlight Selection Override for Either/Or and Bonus Letter Tiles
* **Goal achieved:** Ensure that when a player highlights or selects an Either/Or tile or a Bonus Letter tile, the lime green styling (background, border, text color, box-shadow, and pulsing animation) is completely overridden by the standard selection or typing highlight colors.
* **Implementation (`static/css/play.css` & `templates/index.html`):**
  * Added CSS rules to override `.board-cell.selected.bonus-highlight`, `.board-cell.current.bonus-highlight`, `.board-cell.typing-highlight.bonus-highlight`, and `.board-cell.review-highlight.bonus-highlight`.
  * Set these classes to use the standard theme accent colors (`--highlight-mouse-color` and `--highlight-typing-color`), reset text color to `#000`, and set `animation: none !important` to stop the pulsing lime green box-shadow animation when selected.
  * Injected these overrides both in the external stylesheet and the templates inline stylesheet for maximum safety and cache independence, and bumped the stylesheet version parameter in `templates/index.html` from `?v=85` to `?v=86`.

### 3. FAQ Question Hidden Features Update
* **Goal achieved:** Add information to the FAQ regarding clicking on the Spinner Set to see more information.
* **Implementation (`templates/index.html`):**
  * Added item 6 to the "Are there any additional features in game rooms that aren’t obvious?" FAQ answer: `<li><strong>Clicking on the Spinner Set</strong> displays more information about the likelihoods and meaning of each Spinner.</li>`.
