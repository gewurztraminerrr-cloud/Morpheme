# Stable State Summary — June 10, 2026 (Start Over Point)

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `5ea2383017d2a5a70aacb52cc219d5a0eb38f714` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `5ea2383017d2a5a70aacb52cc219d5a0eb38f714` / `snapshot-current` / `START_OVER_POINT_JUNE_10` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `5ea2383017d2a5a70aacb52cc219d5a0eb38f714` / `snapshot-current` | ✅ Fully Deployed & PM2 Restarted |

**All environments are 100% synchronized at the latest commit 5ea2383017d2a5a70aacb52cc219d5a0eb38f714.**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py`.
The active recovery points `START_OVER_POINT_JUNE_10` and `snapshot-current` tags have been successfully updated and pushed to GitHub.

---

## Serving Versions (cache-busted)

| File / Style | Version | Description |
|--------------|---------|-------------|
| `/css/style.css` | `v=28` | Confined the Tournaments Championship Bracket standings list to a compact max-height of 200px and added customized thin scrollbars. |
| `/css/lobby.css` | `v=25` | Implemented mobile-first grid layout for the rating filter to guarantee buttons wrap below the input box, and flex side-by-side override for desktop viewports. |
| `/js/play.js` | `v=146` | Implemented lowest-RTT clock offset synchronization, local countdown timer clamping to prevent display anomalies (like starting at 0:47), and rapid 300ms transition polling at 0:00 intermission to render the new board instantly. |
| `templates/index.html` | *Dynamic* | Bumps cache-busters for style.css (v=28), lobby.css (v=25), and play.js (v=146). |

---

## Work Completed on June 10, 2026

### 1. Lobby Rating Filter Sizing & Mobile-First CSS Grid Layout
* **Goal achieved:** Return rating filter buttons side-by-side on desktop, but guarantee they stack below the textbox on mobile screens without truncation.
* **Implementation (`static/css/lobby.css` & `templates/index.html`):**
  * Restructured styling to be mobile-first: configured `.rating-filter-container` default rules to use CSS Grid (`display: grid; grid-template-columns: 1fr 1fr; gap: 6px;`).
  * Placed `#rating-filter` input on row 1 (`grid-column: span 2; width: 100%;`) and the two buttons on row 2 (`grid-column: span 1; width: 100%;`), guaranteeing they stack vertically.
  * Overrode the grid layout for screens above 900px (`@media (min-width: 901px)`) to use flexbox (`display: flex; flex-wrap: nowrap; gap: 8px;`) with `#rating-filter` at a min-width of 185px (removing duplicate `min-width: 0` overrides) to show the full placeholder text.
  * Widened `.lobby-grid` desktop columns to `1.7fr 1.3fr` to accommodate the side-by-side flex layout without text truncation.
  * Bumped `lobby.css` cache-buster to `v=25` inside `templates/index.html`.

### 2. Confine Tournaments Championship Bracket to Scrollable Area
* **Goal achieved:** Confine the list of users in the Championship Bracket to a small, scrollable space if there are many participants.
* **Implementation (`static/css/style.css` & `templates/index.html`):**
  * Updated `.t-standings-list` to reduce the `max-height` parameter from `400px` to `200px`.
  * Added styling rules for thin webkit-based scrollbars using `scrollbar-width: thin` and scrollbar colors matched to the accent color (`var(--accent-color)`).
  * Bumped `style.css` cache-buster to `v=28` in `templates/index.html`.

### 3. Forum Thread Bumping on Reply
* **Goal achieved:** Automatically bump a forum thread to the top of its category whenever a new comment/reply is posted.
* **Implementation (`app.py`):**
  * Updated `get_forum_posts(category_id)` to select a dynamic `last_activity` timestamp using `COALESCE((SELECT MAX(timestamp) FROM forum_comments WHERE post_id = p.id), p.timestamp)`.
  * Sorted results by `last_activity DESC` to order posts chronologically by the latest activity (either creation or reply comment).

### 4. Timer Sync and Board Transition Optimization
* **Goal achieved:** Prevent premature round starts at `0:02` intermission, ensure the active round starts precisely at `0:45` (preventing `0:47`), and display the board instantly when the timer hits `0:00`.
* **Implementation (`static/js/play.js` & `templates/index.html`):**
  * Implemented an SNTP-style RTT-based offset calculation: `offset = server_time - (tBefore + tAfter) / 2`, and tracked the offset associated with the lowest RTT (`bestServerTimeRTT`) to eliminate network transmission time skew.
  * Added countdown clamping in `updateLocalTimer` to guarantee the displayed timer value never exceeds the room's limits (active limit or intermission limit).
  * Configured rapid polling (`300`ms interval) at `0:00` intermission to query the server and detect the `active` transition immediately, reverting back to the standard `1000`ms interval on transition.
  * Bumped `play.js` cache-buster to `v=146` in `templates/index.html`.
