# Stable State Summary — October 8, 2026

## Latest Feature Commit ID
```
740c5fcb2f9195b672f9e7da3a57b56752e48383
```

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_8
```
*(Also tagged as `stable-2026-10-08`)*

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| localhost      | ✅ Synchronized | Working tree clean, HEAD at `782d2ba3` |
| GitHub (main)  | ✅ Synchronized | Pushed to origin/main & tags updated |
| morpheme.games | ✅ Synchronized | Deployed to 132.148.72.249 (PM2 online) |
| App / Client   | ✅ Synchronized | Web client, game rooms, lobby, and holding pages verified |

---

## Session Features & Fixes (October 8, 2026)

### 1. Lobby Player Count Display for Accumulative Rooms
- **Scope & Fix (`app.py` & `lobby.js`)**:
  - Aligned the real-time room inactivity filter in `get_lobby_stats()` from 60 seconds to 600 seconds (10 minutes) to match the actual game room inactivity eviction threshold. Previously, active players reviewing round scores, waiting between rounds, or briefly backgrounded were dropped from lobby counts after just 60 seconds, displaying `[0]`.
  - Added `room.spectators` to the candidate player pool in `get_lobby_stats()` so spectators or auto-synced players are correctly counted.
  - Ensured public singleton hubs (`pub_...`) are never skipped by `room.is_solo`, and normalized `room.game_type` (`.replace('solo_', '')`) for consistent button key matching.
  - Fixed mobile wake/unminimize in `lobby.js` (`handleLobbyVisibilityChange`, `pageshow`, `resume`, `focus`) to clear `isDeviceSuspended = false` upon wake and immediately call `handleUserReturnToLobby()` and `fetchLobbyStats('all')`. Previously, `!isDeviceSuspended` check was permanently false on wake, freezing buttons on stale `[0]`.
  - Added an in-flight concurrency lock (`isFetchingLobbyStats`) in `fetchLobbyStats` to prevent duplicate network calls piling up on mobile browsers.
  - Throttled `syncLobbyTimeoutState()` from running every 1 second to once every 30 seconds, relieving pressure on the browser connection pipeline.

### 2. In-Game 10-Minute Idle Room Eviction & Session Expired Notice
- **Scope & Fix (`play.js` & `app.js`)**:
  - Fixed handling when a player idles in a room or minimizes and returns after exceeding the 10-minute idle threshold.
  - Removed stuck "WAIT..." display upon return and ensured players who exceeded the idle window are automatically evicted back to the Lobby.
  - Displayed the "Session Expired" alert modal clearly explaining room inactivity eviction.

### 3. Set Rating Limits Textboxes 5-Digit Maximum & Numeric Guard
- **Scope & Fix (`templates/index.html` & `lobby.js`)**:
  - Enforced a strict maximum length of **at most 5 digits** (0–99999) on both "Min Rating" and "Max Rating" textboxes (`.rating-input`) across all static and dynamic templates.
  - Added `maxlength="5"` attribute to all input elements.
  - Added inline `.slice(0, 5)` sanitization on `oninput` handlers (`this.value = this.value.replace(/\D/g, '').slice(0, 5)`).
  - Enforced 5-digit limit in `keydown` handler (ignoring extra keystrokes if length is already 5 and no text is selected).
  - Capped `ArrowUp` stepper to `99999` so arrow stepping cannot exceed 5 digits.
  - Added length checking in `beforeinput` handler to reject data insertions exceeding 5 digits.
  - Added `.slice(0, 5)` capping in `input`, `paste`, and `drop` event listeners.
  - Configured virtual keyboards on mobile devices to default to 10-key numeric keypad (`inputmode="numeric" pattern="[0-9]*"`).
  - Blocked all letters, symbols, negative signs, decimals, 'e'/'E', whitespace, and punctuation.
  - Bumped `lobby.js` cache version in `templates/index.html`.

### 4. Fix for Recurring "Session Expired" Notice Loop
- **Scope & Fix (`play.js` & `app.js`)**:
  - Eliminated the recurring "Session Expired" popup loop that triggered whenever users reopened the app from minimizing or navigating pages.
  - Fixed `ejectToLobby()` to explicitly clear `window.lastGameState = null;`, `window.currentRoomId = null;`, and reset `lastGameInteractionTime = Date.now();` so subsequent checks do not inherit stale room states and immediately re-trigger eviction.
  - Restricted the 5-second in-game idle checker to only run when actively on `page-play`, with the document visible (not minimized/hidden), and with an active `window.currentRoomId`.
  - Added a 5-minute cooldown deduplication guard (`_lastInactivityNoticeTime`) preventing repeated notifications.
  - Updated absence threshold from 1 hour to 10 minutes in `app.js` and `play.js` so returning from background after the 10-minute server room expiration quietly clears the room session and routes to the lobby without showing modal errors.
  - Refreshed idle timestamps on visibility resume and during all app interactions across all pages.
  - Bumped `app.js` and `play.js` cache query parameters in `templates/index.html`.

### 5. In-Game Room Chat Submission
- **Scope & Fix (`play.js`)**:
  - Restored `let isSendingChat = false;` and `let lastChatSendTime = 0;` at file scope in `static/js/play.js`.
  - Resolved `ReferenceError: isSendingChat is not defined` runtime exception that had prevented chat message submissions on both Enter key and Send button.
  - Added a 250ms debounce guard to prevent double-submitting across pointerdown/click events.
  - Implemented immediate toast alerts if a chat message is rejected by content moderation or user timeout.
  - Triggered an immediate `updateGameState()` call upon successful submission so messages render instantly without awaiting poll intervals.
  - Wrapped Play DOM initialization with `document.readyState` support and registered delegated event listeners on `document` as a fallback.
  - Bumped `play.js` cache query parameter in `templates/index.html`.

### 6. Maintenance & 502 Holding Page Selection Prevention
- **Nginx Holding Page (`502.html`)**:
  - Implemented complete text selection prevention (`user-select: none !important`, `-webkit-user-select: none !important`, `-moz-user-select: none !important`, `-ms-user-select: none !important`, `-webkit-touch-callout: none !important`) across all elements on the "Morpheme is updating..." screen.
  - Set `::selection` to transparent so clicking, double-clicking, or dragging cursor/touch cannot highlight text.
  - Synchronized `/var/www/html/502.html` on the production server.

### 7. Round Replay Scrollbar & Draggable Thumb (Desktop & Laptops)
- **Word Timeline Scrollbar Styling (`play.css`)**:
  - Upgraded native browser scrollbars on desktop and laptop displays to use the luminous blue gradient thumb matching the app design (`linear-gradient(180deg, #409cff, #2980b9)` with border glow and hover states).
  - Added custom rounded track (`rgba(255, 255, 255, 0.10)` with `rgba(255, 255, 255, 0.18)` border).
  - Adjusted right padding (`14px`) so word entries never overlap the custom thumb.

### 8. Round Replay Mobile Scroller
- **Dynamic Mobile Scroller (`mobile_scroller.js` & `play.css`)**:
  - Connected the body-attached mobile vertical scroller track and thumb directly to Round Replay's container (`#history-review-overlay .history-review-layout`).
  - Elevated `.m-scroller-track-overlay` to `z-index: 2147483647 !important` so it stays topmost above all overlay backdrops and modals.
  - Suppressed default browser scrollbars in the modal on mobile for clean touch interactions.
  - Added automated scroll and resize observers plus explicit lifecycle update triggers upon modal open.

### 9. Lobby Guide Modal
- **Indicator Enhancement**:
  - Added `"All across are 3 mins"` label with an arrow pointing to the right under `"3m"` in the *"WHAT AM I LOOKING AT?"* diagram, harmonized with the existing 4x6 indicator format.
