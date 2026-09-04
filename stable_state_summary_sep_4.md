# Stable State Summary – September 4, 2026

## Latest Commit Information
- **Commit ID**: `d1a5e540` (`d1a5e5406d4e5f72e79e6027a4d46f5e278f30aa`)
- **Commit Message**: `docs: add Start Over Stable State Summary for September 4, 2026`
- **Active Git Tags**:
  - `START_OVER_POINT`
  - `START_OVER_POINT_SEPTEMBER_4`
  - `stable-2026-09-04`
  - `save-point-latest`

---

## Synchronization Status
- **Localhost**: Synchronized at `7ac832a6`
- **GitHub (`origin/main`)**: Synchronized at `7ac832a6`
- **Production Server (`132.148.72.249` / `morpheme.games`)**: Synchronized at `7ac832a6`, PM2 online, HTTP 200 OK

---

## Changes Implemented

1. **Desktop / Laptop Rotate & Transpose Button Placement**:
   - On laptops and desktops (and **ONLY** laptops and desktops), moved the connected `Rotate` and `Transpose` buttons directly to the right of the word input textbox (`#word-input`) inside `.word-input-section`.
   - Maintained the exact same aesthetic capsule styling, hover states, border radiuses, and height (`44px` matching `#word-input`).
   - On mobile devices, the `Rotate` and `Transpose` buttons remain positioned on the right side of the timer display bar (`.timer-display`) above the board.

2. **Mobile Low-Time Red Pulse Behavior (<= 10s)**:
   - On mobile devices during low-time warnings (`<= 10s`), the entire Board view—including all space above and below the light grey timer panel—glows and pulses red continuously in unison (`mobile-low-time-pulse`).
   - On mobile, `.play-header` remains transparent with no separate card backgrounds or border outlines.

3. **Board & Timer Panel Spacing on Mobile**:
   - Decreased the bottom margin of the light grey timer panel (`.timer-display`) to `3px` on mobile screens (`max-width: 992px`) in `static/css/play.css`.

4. **Mobile Round Start Sliding & Navigation**:
   - Smoothly slides mobile users to the **Board window** upon round start and slides the top navigation menu up and out of view.

5. **Cache Busting**:
   - Bumped `style.css`, `play.css`, `app.js`, and `play.js` cache query versions to `v=33858` in `templates/index.html`.
