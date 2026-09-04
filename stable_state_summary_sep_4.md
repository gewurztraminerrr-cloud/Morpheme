# Stable State Summary – September 4, 2026

## Latest Commit Information
- **Commit ID**: `03831315`
- **Commit Message**: `fix(desktop): hide game room on desktops and laptops unless active`
- **Active Git Tags**:
  - `START_OVER_POINT`
  - `START_OVER_POINT_SEPTEMBER_4`
  - `stable-2026-09-04`
  - `save-point-latest`
  - `start-over`

---

## Synchronization Status
- **Localhost**: Synchronized (`03831315` / tags updated)
- **GitHub (`origin/main` & Tags)**: Synchronized (`03831315` / tags updated)
- **Production Server (`132.148.72.249` / `morpheme.games`)**: Synchronized (`03831315`), PM2 online, HTTP 200 OK

---

## Changes Implemented & Stable Specifications

1. **Desktop / Laptop Game Room Isolation & Visibility**:
   - Fixed a CSS selector issue in `static/css/play.css` where `body.is-desktop #page-play` previously forced `display: flex !important` on the game room container regardless of whether the page was active.
   - Constrained the rule strictly to `body.is-desktop #page-play.active`, ensuring the game room is completely hidden when viewing the Lobby, Login, Settings, Tools, or Leaderboards, and only rendered when a room button or match is selected.

2. **Mobile Low-Time Red Slow-Flash Pulse Warning (<= 10s)**:
   - When the round timer reaches 10 seconds or less on mobile devices, the entire board window—including the content above the light grey timer (the Spinner Set and game parameters in `.play-header`) and below the timer (the game board in `.board-panel`)—slow-pulses red seamlessly and in unison (`@keyframes mobile-low-time-pulse`).
   - Removed any red border outline or box-shadow on the Spinner Set or board panel, making the background pulse completely seamless and unified across the entire mobile screen.
   - Cleanly activates when `remaining <= 10 && remaining > 0 && currentState === 'active'` on mobile viewports/devices and immediately clears when the timer expires or on game state reset.

3. **Desktop / Laptop Game Header & Grid Layout**:
   - On desktops and laptops (viewports > 992px), the game header (`.play-header`) containing the game title, mode parameters (Diff, Min, Dict, Words, Format, Bonus), and `Return to Lobby` button spans cleanly across the top of `#page-play` directly above the 3-column grid (`Players`, `Board`, and `Words` panels), preserving complete layout symmetry.
   - Preserved in static markup at `#page-play > .play-header` to ensure instantaneous server-side / page-load rendering without layout shifts.

4. **Desktop / Laptop Word Input & Controls Placement**:
   - On laptops and desktops (and **ONLY** laptops and desktops), the `Rotate` and `Transpose` buttons are positioned immediately to the right of the word input textbox (`#word-input`) inside `.word-input-section`.
   - Styled with matching capsule aesthetics, hover states, border radiuses, and height (`44px`).

5. **Desktop / Laptop Timer Padding & Spacing Reduction**:
   - Reduced the internal vertical padding on `body.is-desktop .board-panel .timer-display` to `3px 20px` (`border-radius: 8px`), creating a slim, snug capsule around `Time: 0:00`.
   - Decreased the board panel top padding to `8px` and the timer bottom margin to `3px` (`margin: 0 auto 3px auto !important;`), reducing excess space above and below the timer and above the game board on laptops and desktops.

6. **Mobile Device Board Window Isolation**:
   - On mobile devices (`is-mobile` / `max-width: 992px`), `.play-header` (Spinner Set & constant parameters) is dynamically and strictly isolated inside the **Board window** (`.board-panel`) directly above the timer (`.timer-display`).
   - When swiping horizontally to the **Players panel** (which sits below the Color Chart) or to the **Words panel**, the parameters and Spinner Set are **not displayed**.
   - On mobile devices, `Rotate` and `Transpose` buttons remain positioned on the right side of the timer display bar (`.timer-display`) directly above the board.

7. **Mobile Viewport Snap & Round Navigation**:
   - Reduced bottom margin of the light grey timer panel (`.timer-display`) to `3px` on mobile screens.
   - Smoothly slides mobile users to the **Board window** upon round start and slides the top navigation menu up and out of view.

8. **Cache Busting & Versioning**:
   - Incremented query parameter versions for `style.css`, `play.css`, `app.js`, and `play.js` to `v=33864` in `templates/index.html`.
