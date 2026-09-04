# Stable State Summary – September 4, 2026

## Latest Commit Information
- **Commit Message**: `docs: update Start Over Stable State Summary for September 4, 2026`
- **Active Git Tags**:
  - `START_OVER_POINT`
  - `START_OVER_POINT_SEPTEMBER_4`
  - `stable-2026-09-04`
  - `save-point-latest`
  - `start-over`

---

## Synchronization Status
- **Localhost**: Synchronized
- **GitHub (`origin/main` & Tags)**: Synchronized
- **Production Server (`132.148.72.249` / `morpheme.games`)**: Synchronized, PM2 online, HTTP 200 OK

---

## Changes Implemented & Stable Specifications

1. **Desktop / Laptop Game Header & Grid Isolation**:
   - On desktops and laptops (viewports > 992px), the game header (`.play-header`) containing the game title, mode parameters (Diff, Min, Dict, Words, Format, Bonus), and `Return to Lobby` button spans cleanly across the top of `#page-play` directly above the 3-column grid (`Players`, `Board`, and `Words` panels), preserving complete layout symmetry.
   - Preserved in static markup at `#page-play > .play-header` to ensure instantaneous server-side / page-load rendering without layout shifts.

2. **Desktop / Laptop Word Input & Controls Placement**:
   - On laptops and desktops (and **ONLY** laptops and desktops), the `Rotate` and `Transpose` buttons are positioned immediately to the right of the word input textbox (`#word-input`) inside `.word-input-section`.
   - Styled with matching capsule aesthetics, hover states, border radiuses, and height (`44px`).

3. **Mobile Device Board Window Isolation**:
   - On mobile devices (`is-mobile` / `max-width: 992px`), `.play-header` (Spinner Set & constant parameters) is dynamically and strictly isolated inside the **Board window** (`.board-panel`) directly above the timer (`.timer-display`).
   - When swiping horizontally to the **Players panel** (which sits below the Color Chart) or to the **Words panel**, the parameters and Spinner Set are **not displayed**.
   - On mobile devices, `Rotate` and `Transpose` buttons remain positioned on the right side of the timer display bar (`.timer-display`) directly above the board.

4. **Mobile Low-Time Red Pulse Warning (<= 10s)**:
   - On mobile devices during low-time warnings (`<= 10s`), the entire Board view—including space above and below the light grey timer panel—glows and pulses red continuously in unison (`mobile-low-time-pulse`).
   - On mobile, `.play-header` remains transparent with no separate card backgrounds or border outlines.

5. **Mobile Viewport Snap & Round Navigation**:
   - Reduced bottom margin of the light grey timer panel (`.timer-display`) to `3px` on mobile screens.
   - Smoothly slides mobile users to the **Board window** upon round start and slides the top navigation menu up and out of view.

6. **Cache Busting & Versioning**:
   - Incremented query parameter versions for `style.css`, `play.css`, `app.js`, and `play.js` to `v=33860` in `templates/index.html`.
