# Stable State Summary — June 10, 2026 (Start Over Point)

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `74096b026dfd150244f7724269cd2fef119be955` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `74096b026dfd150244f7724269cd2fef119be955` / `snapshot-current` / `START_OVER_POINT_JUNE_10` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `74096b026dfd150244f7724269cd2fef119be955` / `snapshot-current` | ✅ Fully Deployed & PM2 Restarted |

**All environments are 100% synchronized at the latest commit 74096b026dfd150244f7724269cd2fef119be955.**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py`.
The active recovery points `START_OVER_POINT_JUNE_10` and `snapshot-current` tags have been successfully updated and pushed to GitHub.

---

## Serving Versions (cache-busted)

| File / Style | Version | Description |
|--------------|---------|-------------|
| `/css/style.css` | `v=28` | Confined the Tournaments Championship Bracket standings list to a compact max-height of 200px and added customized thin scrollbars. |
| `/css/lobby.css` | `v=25` | Implemented mobile-first grid layout for the rating filter to guarantee buttons wrap below the input box, and flex side-by-side override for desktop viewports. |
| `/css/forum.css` | `v=2` | Styled `.file-input-wrapper` and `input[type="file"]` to transparently overlay the dummy select box, resolving mobile photo chooser opening bugs. |
| `/js/play.js` | `v=146` | Implemented lowest-RTT clock offset synchronization, local countdown timer clamping to prevent display anomalies, and rapid 300ms transition polling at 0:00 intermission to render the new board instantly. Added global touch and wheel scroll prevention when the "You're guessing!" popup is active, and bypass logic for 24h rooms. |
| `templates/index.html` | *Dynamic* | Bumps cache-busters for style.css (v=28), lobby.css (v=25), play.js (v=146), and forum.css (v=2). |

---

## Work Completed on June 10, 2026

### 1. Lobby Rating Filter Sizing & Mobile-First CSS Grid Layout
* **Goal achieved:** Return rating filter buttons side-by-side on desktop, but guarantee they stack below the textbox on mobile screens without truncation.
* **Implementation (`static/css/lobby.css` & `templates/index.html`):**
  * Restructured styling to be mobile-first: configured `.rating-filter-container` default rules to use CSS Grid (`display: grid; grid-template-columns: 1fr 1fr; gap: 6px;`).
  * Placed `#rating-filter` input on row 1 (`grid-column: span 2; width: 100%;`) and the two buttons on row 2 (`grid-column: span 1; width: 100%;`), guaranteeing they stack vertically.
  * Overrode the grid layout for screens above 900px (`@media (min-width: 901px)`) to use flexbox (`display: flex; flex-wrap: nowrap; gap: 8px;`) with `#rating-filter` at a min-width of 185px to show the full placeholder text.
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

### 5. 6x8 Board Rescue and Dynamic Fallback
* **Goal achieved:** Prevent 6x8 rooms from hanging during intermission due to board generation delays, and ensure the board contains valid words matching the board coordinates.
* **Implementation (`game_room.py` & `app.py`):**
  * Added `check_6x8_rescue(room)` in `RoomManager` to detect when a 6x8 room has <= 10 seconds remaining in intermission without a generated board.
  * When triggered, it overrides the room's spinner configuration with fast parameters (Normal format, Medium difficulty, NWL dictionary, min length 3, bonus length 6) and restarts the generation thread.
  * Implemented a robust `get_emergency_fallback_board(dimensions, board_format, time_limit)` helper to construct compliant boards dynamically with correct words matching the grid coordinates (both 2D and 3D).

### 6. Viewport Scroll Locking on "You're guessing!" Popup
* **Goal achieved:** Keep the board in place and do not allow scrolling or movement of the background page while the guessing popup is shown.
* **Implementation (`static/js/play.js`):**
  * Configured `showGuessingPopup()` to set `document.body.style.overflow = 'hidden'` on popup show and restore it on close.
  * Registered document-level passive-false listeners for `wheel` and `touchmove` events that intercept and invoke `e.preventDefault()`, fully freezing scroll and swipe interactions on all devices (mobile and desktop) while guessing.

### 7. Guessing Popup Bypass for 24h Rooms
* **Goal achieved:** Do not display the "You're guessing!" window if the user is in a 24-hour room.
* **Implementation (`static/js/play.js`):**
  * Added a check at the beginning of `showGuessingPopup()` to return early and skip showing the modal entirely if the room's `time_limit` is >= 7200 seconds.

### 8. FAQ Multiplier Descriptions Update (Double/Triple)
* **Goal achieved:** Document the precise rating change limits for Double and Triple multiplier formats in the FAQ.
* **Implementation (`templates/index.html`):**
  * Updated FAQ descriptions to mention:
    * Double: "The highest and lowest rating change possible is +32 and -32."
    * Triple: "The highest and lowest rating change possible is +48 and -48."

### 9. FCFS and Split Room Creation Isolation
* **Goal achieved:** Open a brand-new room with only the creator inside whenever a user creates an FCFS or SP room, bypassing any room-reuse (singleton) behavior.
* **Implementation (`app.py` & `game_room.py`):**
  * Modified the Flask `/api/room/create` endpoint to assign a random UUID instead of a stable `pub_v2_...` ID when public rooms are created for FCFS or Split.
  * Updated `create_room()` in `RoomManager` to skip the singleton lookup logic for `fcfs` and `split` formats, ensuring they are always spawned as fresh instances.

### 10. Forum File Upload Mobile Tap Fix
* **Goal achieved:** Ensure the "Choose file" area in the forums successfully opens the photo library / file selection on all mobile devices (iOS/Android).
* **Implementation (`static/css/forum.css` & `templates/index.html`)**:
  * Configured `.file-input-wrapper` with `position: relative; cursor: pointer;`.
  * Positioned the `<input type="file">` absolutely at `top: 0; left: 0; width: 100%; height: 100%;` and set `opacity: 0; z-index: 2; cursor: pointer;`.
  * This transparently stretches the real click area to cover the entire custom `.file-dummy` upload box so that taps anywhere on the dummy elements trigger the native browser photo selector.
  * Added hover transitions and color changes on `.file-input-wrapper:hover .file-dummy` for enhanced visual feedback.
  * Bumped `forum.css` version to `v=2` in `templates/index.html` to force immediately refreshed mobile styles.
