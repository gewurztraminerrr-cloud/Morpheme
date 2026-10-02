# Stable State Summary — October 2, 2026

## Latest Feature Commit ID
```
3cf6434b5f022255f14e413e94e719017dede9b0
```

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_2
```
*(Also tagged as `stable-2026-10-02`)*

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| localhost      | ✅ Synchronized | Working tree clean, HEAD at `3cf6434b` |
| GitHub (main)  | ✅ Synchronized | Pushed to origin/main (`3cf6434b`) & tags synchronized |
| morpheme.games | ✅ Synchronized | Deployed to 132.148.72.249, HTTP 200 OK, PM2 Online |
| App / Client   | ✅ Synchronized | Web client & backend endpoints verified |

---

## Session Features & Fixes (October 2, 2026)

### 1. Leaderboard Mobile Layout & Centering
- **Balanced Side Padding**:
  - Replaced the asymmetric mobile padding (`padding: 10px 22px 10px 6px`) on `#page-leaderboards` with balanced, equalized padding (`padding: 10px 10px !important`).
  - Completely resolved the visual shift where content leaned to the left and left excess whitespace on the right.
- **Centered Header & Period Navigation Tabs**:
  - Under mobile viewport widths ($\le 992\text{px}$), `.lb-header`, `.lb-header h2`, `.lb-tabs` (DAY, WEEK, MONTH, YEAR, ALL-TIME), and `.lb-attribution` are centered horizontally with `justify-content: center` and `text-align: center`.

### 2. Tournament Rounds: Chatbox Suppressed
- **Clean Focus During Competitive Play**:
  - Removed the in-game chatbox during Tournament rounds across all platforms.
  - Added CSS rule `body.is-tournament-round .chat-panel, body.is-tournament-round .left-panel-container .chat-panel { display: none !important; }`.
  - In `initTournamentPlay()`, explicitly hides `.chat-panel`; in `exitTournamentPlay()`, cleanly restores the chatbox display property upon exiting back to regular rooms or lobby.
  - Device layout helper (`adjustPlayHeaderForDevice()`) also suppresses the chatbox whenever an active tournament round is detected.

### 3. Settings & Synesthesia: Spacing & Flexbox Structure Fix
- **Structural Container Encapsulation**:
  - Identified root cause of unyielding vertical gap above the colored letters: `.setting-panel` enforces `display: flex; flex-direction: column; gap: 20px;`, which was injecting an unyielding 20px flex gap between the "Reset All to Black" button and the letter grid.
  - Wrapped both elements into a dedicated `.synesthesia-section` with internal `gap: 4px`, mirroring the Board Size sliders container.
  - The "Reset All to Black" button now sits snugly right above the letter palette with minimal, balanced padding.
- **Theme Divider Padding Polish**:
  - Reduced vertical margin below the theme selector divider so theme layout options sit closer to their header.

### 4. Color Palette Expansion: 9 "Light Light" Layout Themes
- **Ultra-Light Layout Themes Added Across All Platforms**:
  - Added 9 new lighter themes: Light Light Grey, Light Light Red, Light Light Orange, Light Light Yellow, Light Light Green, Light Light Cyan, Light Light Blue, Light Light Purple, and Light Light Magenta.
  - Available identically across desktop, laptop, tablet, and mobile devices in **Settings $\rightarrow$ Layout / App Theme**.
  - Formatted with softer pastel hues and calibrated panel contrast tiers (`var(--bg-primary)` and `var(--card-border)`) matching the dark/light design system rules.

### 5. Color Chart: White Flashing Indicator for High Ratings
- **High-Rating Visibility**:
  - For high rating tiers ($\ge 2000$: Overlord, Conqueror, Warlord, Juggernaut, Apex, The Void tiers, and Singularity), the active tier indicator on `#game-color-bar` flashes **white** (`@keyframes rating-flash-white`) via an animated luminous inset box shadow and subtle glow.
  - Solved the contrast issue where dark red, dark maroon, and pitch-black (`#000000`) high-tier segments flashing to black were previously invisible.
  - Lower rating tiers (< 2000: Greens, Blues, Yellows, Light Oranges) continue to pulse to black for contrast against light backgrounds.

### 6. Game Room Chatbox Expansion Behavior
- **Isolated Textbox Interaction**:
  - Clicking or focusing the chat input textbox (`#chat-input`) to send a message in game rooms no longer expands the entire chatbox.
  - Added click event stop-propagation and touchstart/touchmove input guards so interacting with the textbox keeps the chatbox in its normal compact state.
- **Dedicated Chatbox Expansion**:
  - The chatbox expands (`expandChat()`) exclusively when the user clicks or taps on the chatbox itself (`.chat-panel` / `#chat-history`).
  - Mobile keyboard position adjustments are guarded to active expanded state only.

### 7. Audio & Sounds: Bell Sound #2 Redesign
- **Melodic Ascending Two-Tone Chime**:
  - Replaced the harsh monotone 880 Hz strike for Bell Sound #2 in **Settings $\rightarrow$ Audio & Sounds** with an acoustically modeled melodic ascending two-tone chime ($\text{F}_5 \rightarrow \text{C}_6$, a cheerful, harmonious perfect fifth).
  - Built with authentic physical bell acoustics: fundamental strike tones, $+0.2\%$ detuned beating shimmer ($1.5\text{ Hz}$), harmonic overtones, warm $0.5\times$ subharmonic hum, a soft $4\text{ms}$ raised-cosine mallet attack, and a $15\%$ hall diffusion reverb tail.
  - Synchronized across web and Flutter mobile.

### 8. Desktop & Laptop Game Room Header Glow Removal (White Layout)
- **Flat Header Styling on White Layout**:
  - Removed glowing effect / box shadow (`box-shadow: none !important;`) around `.play-header:not(.low-time-warning)` when using the White theme (`body.theme-white`, `[class*="theme-white"]`) and light theme layouts on desktops and laptops.
  - Aligns the header perimeter with the flat, clean styling of the other game room panels (players, board, words).
  - Preserved critical low-time warning countdown pulses.

### 9. Guide Matchmaking Grid Mobile Optimization
- **Vertical Orientation on Mobile Devices**:
  - The matchmaking grid diagram renders with its longest side oriented vertically on mobile screens (`isMobile`), optimizing screen real estate and legibility without horizontal crushing.
- **Tap-to-Enlarge Modal**:
  - Tapping the diagram opens a high-resolution, full-screen inspection modal with pinch/zoom and fluid scrolling.

### 10. Mobile Fullscreen & Continuous Navigation Invariants
- **Permanent Fullscreen Invariants Preserved (`.agents/AGENTS.md`)**:
  - Continuous fullscreen is preserved across all top menu tabs (Lobby, Play, Tools, Mods, Leaderboard, Settings, Profile, Forum, How to Play, Donate).
  - Fullscreen is never exited on textbox focus/blur, dropdown interactions, tab switching, or room entry/exit.
  - Strict isolation of `document.exitFullscreen()` to the Full List Modal (`openFullListModal`).
- **Synchronous Gateway Interaction**:
  - Gateway screen cleanly initiates fullscreen and lobby music on direct user gesture.
