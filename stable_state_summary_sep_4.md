# Stable State Summary – September 4, 2026

## Latest Commit Information
- **Commit ID**: `1c670f66`
- **Commit Message**: `style(forum): move Back to Category button higher on mobile devices`
- **Active Git Tags**:
  - `START_OVER_POINT`
  - `START_OVER_POINT_SEPTEMBER_4`
  - `stable-2026-09-04`
  - `save-point-latest`
  - `start-over`

---

## Synchronization Status
- **Localhost**: Synchronized (`1c670f66` / tags updated)
- **GitHub (`origin/main` & Tags)**: Synchronized (`1c670f66` / tags updated)
- **Production Server (`132.148.72.249` / `morpheme.games`)**: Synchronized (`1c670f66`), PM2 online, HTTP 200 OK

---

## Changes Implemented & Stable Specifications

1. **Mobile Forum "Back to Category" Higher Elevation**:
   - Adjusted `.forum-sub-header` and `.forum-back-btn` (`#forum-back-to-list`) on mobile devices with `margin: -15px -15px 12px -15px !important;` and `padding: 4px 15px 6px 15px !important;`, raising the button higher to the top edge of the single post view container with compact padding (`padding: 8px 12px !important;`).

2. **Mobile Lobby Game Type Title Spacing & Active Rooms Parity**:
   - Adjusted the spacing below each game type title (`ACCUMULATIVE`, `FIRST COME FIRST SERVE`, and `SPLIT POINTS`) in the mobile Lobby to `gap: 6px !important;` and `margin: 0 !important; padding: 0 !important;` on `.game-title`, matching the snug spacing of `ACTIVE ROOMS` above its content.
   - Increased top breathing room above each game section (`margin-bottom: 22px !important;` on `.game-types-panel .game-section` and `margin-bottom: 16px–18px !important;` below the Journey banner), clearly grouping each title with its own game configuration matrix below it.

1. **Lobby Drawer Players Count Bracket Format**:
   - Changed the player count display inside the opened "Players in Lobby & Chat" sliding panel header from parentheses `(1)` to square brackets `[1]` (`👥 Players in Lobby [<span id="lobby-players-count-header">0</span>]`), matching the outer drawer button notation.

2. **Mobile Game Board Perfect Centering & Equal Side Margins**:
   - Updated mobile `.board-panel` to use 0 side padding (`padding: 10px 0 0 0 !important;`) so that the game board (`#game-board`) centers mathematically to the exact pixel midpoint of the device screen with 100% equal empty space on the left and right sides.
   - Configured `#game-board` on mobile to `width: max-content !important; max-width: 100vw !important; margin: 0 auto !important; align-self: center !important; justify-self: center !important; justify-content: center !important; overflow: visible !important;`.
   - Constrained non-board controls (the timer, input textbox, play-header, and status alerts) to centered capsules (`width: calc(100% - 24px) !important; max-width: 480px !important; margin: 0 auto 3px auto !important;`).
   - Side panels (`Players` and `Words/Definitions`) retain their comfortable `15px` side padding.

3. **Mobile Chatbox Left-Aligned Message Display**:
   - Resolved an issue where mobile chat messages inside the in-game chat box and lobby chat drawer were inadvertently centering due to column flex alignment.
   - Enforced explicit `text-align: left !important;` and `align-items: stretch !important;` across `.chat-panel`, `#chat-history`, `.chat-message`, `.chat-sys`, `.chat-user`, `.chat-text`, and placeholder messages.
   - Messages are now anchored neatly along the left edge of the chat box on all mobile devices and desktop views.

3. **Obsidian-Diamond Frosted Glass Journey Banner (Mobile, Laptops, Desktops)**:
   - Completely replaced the pink-to-black gradient design with an ultra-sleek, prestigious **Obsidian-Diamond Frosted Glass** container (`linear-gradient(180deg, rgba(255, 255, 255, 0.08) 0%, rgba(18, 18, 30, 0.78) 40%, rgba(10, 10, 18, 0.88) 100%)`).
   - Added a crisp top specular bevel highlight (`.lobby-journey-message::after`) with a diamond-bright sheen.
   - Implemented a smooth, luminous diamond light shimmer sweep (`.lobby-journey-message::before` with `@keyframes journey-diamond-shimmer`) that traverses the banner every 4.8s.
   - Styled the text with pure, high-contrast metallic pearl-white typography (`linear-gradient(180deg, #ffffff 0%, #f1f5f9 60%, #cbd5e1 100%)` via `-webkit-background-clip: text`) with subtle back-glow drop shadow.
   - Applied responsive font sizing and letter-spacing for standard phones, small displays (<= 360px), tablets, laptops, and wide monitors.

3. **Elimination of Dual Music Echo & Single-Stream Lock**:
   - Completely eliminated the slight millisecond echo/double-playing issue where the HTML5 `<audio id="lobby-music">` fallback element and Web Audio API `AudioBufferSourceNode` could trigger concurrently.
   - Enforced strict single-engine playback: whenever Web Audio buffer playback initiates, the HTML5 audio element is immediately paused, reset (`currentTime = 0`), and muted.
   - Added a trigger latch (`_gatewayAudioTriggered`) and singleton guard (`isPlaying` / `this.source` verification) so rapid pointer/touch/click events cannot create overlapping audio streams.

4. **Tournament Current Pairings Resolution**:
   - Fixed an issue where the "Current Pairings" card on the Tournaments page scanned from Round 1 and showed an earlier round's pairing rather than the user's latest/final round matchup.
   - Updated both backend `/api/tournament/status` (to include the final round's matchups in `all_matchups` when completed) and frontend `static/js/tournaments.js` (to prioritize the current/final round and search backwards for latest active pairings).

2. **Desktop / Laptop Game Room Isolation & Visibility**:
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
