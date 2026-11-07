# Stable State Summary — November 6, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_NOVEMBER_6
```
*(Also tagged as `stable-2026-11-06`)*

## Latest Commit ID
```
Resolved with: git rev-parse START_OVER_POINT_NOVEMBER_6
```

## Synchronization Status

| Environment | Status | Commit / Tag | Verification Details |
|:---|:---|:---|:---|
| **Localhost** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_6` | Working tree on `main`, zero uncommitted tracked changes |
| **GitHub (`origin/main`)** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_6` | Pushed to `origin/main` & tags `START_OVER_POINT_NOVEMBER_6` / `stable-2026-11-06` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_6` | Fast-forward pulled to origin/main, PM2 online (process `morpheme`), HTTP/2 200 |
| **App (Mobile WebView)** | ✅ Synchronized | `https://morpheme.games/` | Flutter app loads live synchronized webview with native bridges |

---

## Session Highlights & Architectural Updates (November 6, 2026)

### 1. Global Timezone Adherence & Abbreviation Standardization (`app.js`, `tools.js`, `settings.js`, `tournaments.js`, `index.html`, `app.py`, `game_room.py`)
- **Strict Timezone Code Mapping**:
  - Replaced browser seasonal standard/daylight strings (`CST`/`CDT`, `EST`/`EDT`, `PST`/`PDT`, `MST`/`MDT`) with unified abbreviations adhering to user selection:
    - **CT** for Central Time
    - **ET** for Eastern Time
    - **PT** for Pacific Time
    - **MT** for Mountain Time
    - **AKT** for Alaska Time
    - **HST** for Hawaii Time
    - **UTC** for Universal Time
  - Centralized in `window.TIMEZONE_ABBR_MAP` and `window.getAppTimezoneAbbr(tz, rawAbbr)`.
- **Canada Central Option**:
  - Added **Canada Central (Winnipeg - CT)** (`America/Winnipeg`) to both Settings and Profile timezone selectors, alongside Canada Eastern (Toronto - ET) and Canada Pacific (Vancouver - PT), honoring local Manitoba year-round offset rules.
- **Tools Timezone Alignment**:
  - **Tools -> New Users**: Member registration dates (e.g. for user *Ffoet*) now format with the viewer's selected timezone abbreviation (`... CT`).
  - **Tools -> Find Count**: Found-at timestamps (e.g. for *POLYENIC*, *CREOLIST*) now format with the viewer's selected timezone abbreviation (`... CT`).
- **Dynamic Tool Date Refresh**:
  - Implemented `window.refreshCurrentToolDates()` to instantly re-render open tool panes (New Users, Find Count, Profile Round History) whenever the user changes their timezone in Settings or Profile.
- **Backend Sync & Profile Cache Invalidation**:
  - Synchronized `users.timezone` and `user_settings.setting_value` in `/api/settings/update`, `/api/profile/update`, and `get_profile`.
  - Added cache invalidation for `_PROFILE_CACHE` so saved preferences are never wiped on page refresh.
  - Replaced static offset in `format_chicago_to_utc` with `zoneinfo.ZoneInfo("America/Chicago")` for accurate date-specific daylight saving evaluation.

### 2. "WHAT AM I LOOKING AT?" Demonstration Board & Intuition Quote (`lobby.js`, `index.html`)
- **Core Skill Quote**:
  - Added the intuition and pattern flow quote directly adjacent to word demonstrations in the guide:
    > *"This game requires a strong intuitive sense of probability that pieces together almost instantaneously on first glance of a board. The more you play, the stronger this ability will get. If you commit to learning the words, the pieces come together and the patterns flow like liquid."*
- **Example Word Replacement**:
  - Replaced demonstration word `"TRAP"` with `"WONDERS"` across the animated guide demo board in both HTML structure and interactive path scripting.

### 3. Forum Category Navigation Scroll Position Retention (`forum.js`)
- **Scroll Memory**:
  - Implemented category scroll memory in `categoryScrollPositions` dictionary.
  - When browsing threads in any forum category and clicking "Back to Category", the container scroll position is restored cleanly rather than snapping to the top.

### 4. Instant "VIEW ALL" Transition for Game Room Players List (`play.js`)
- **Eliminated Round-Boundary Lag**:
  - Implemented `resetPlayersToViewAll()` in `static/js/play.js`.
  - Triggers immediately at the exact instant `round_start` or `round_over` packets arrive, resetting the players list filter to "VIEW ALL" with zero perceptible delay.

### 5. Wiktionary Inflection Cross-Reference for Definition Popups (`app.py`)
- **Comprehensive Fallback**:
  - Enhanced `is_verified_inflection` in `app.py` with an automated fallback to Wiktionary HTML extraction when WordNet lacks grammatical inflections (e.g. `LEGERING` and `LEGERED` from base verb `leger`).
  - Resolves base lemmas and displays complete definitions in game room and tool definition popups.

### 6. Scoreboard Sorting: Active 0-Point Players Ahead of DNP (`game_room.py`)
- **Player Status Hierarchy**:
  - Refined scoreboard sorting logic: players with 0 points who actively played are ranked strictly above Did Not Play (DNP) players.

### 7. Multiplayer Subanagrams Mobile & Desktop Layout Refinements (`index.html`, `style.css`)
- **Random Tab "MUST HAVE" Dropdown**:
  - Integrated "MUST HAVE" letter selector into the Random tab controls for Subanagrams.
  - Compacted desktop controls into a single row and paired inputs cleanly on mobile devices.

### 8. Tools & Store Mobile UI Densification (`style.css`, `index.html`, `tools.js`)
- **2-Column Tool Grid**:
  - Refactored mobile controls for Unscramble, Is Valid, and Lists tools into compact 2-column layouts matching unified design language.
  - Extended Unscramble "must-have" input to full width.
  - Added dedicated custom scrollbar thumb to the desktop Morpheme Store.

### 9. High-Contrast Podium Styling for Light Themes (`style.css`)
- **1st Place Contrast**:
  - Styled 1st place in light themes with vibrant gold/yellow `#facc15` and crisp dark outline `#1f2937`.
  - Styled 4th+ places in distinct sky blue `#0284c7`, preventing visual confusion with 2nd place silver.

### 10. 24h Rooms Clues Tab Isolation (`style.css`, `game_room.py`)
- **Clues Tab Overflow Fix**:
  - Enforced strict `.tab-content` display rules for 24h room Clues tab, eliminating title and content bleed into adjacent game tabs.

### 11. Settings Drawer Navigation Hierarchy (`style.css`, `settings.js`)
- **Arrow Streamlining**:
  - Removed navigation arrows from sidebar category tabs, preserving arrows strictly on interactive action buttons within subpages.
  - Right-aligned navigation chevrons consistently across desktop and laptop screens.

### 12. Personal Timer Audial Gong & UI Consolidation (`tools.js`, `index.html`, `style.css`)
- **Authentic Synthesized Gong**:
  - Built Web Audio gong synthesizer with inharmonic detuned partials, harmonic bloom, and exponential decay reverb.
  - Added preview audio button in Personal Timer tool and consolidated alarm preview controls.
  - Added pulsating red flash animation across Definitions panel upon timer expiry.

### 13. Retention of Word Finds in Find Count / Tools (`game_room.py`, `app.py`)
- **Prevented Word Find Drops on Mid-Round Join**:
  - Removed `or u_mid` from `save_round_history` skip condition (`if u_id != -1 and u_name != 'System' and u_score <= 0 and not u_submitted: continue`).
  - Rounds where players entered mid-round and found words are now fully saved to `round_history`, ensuring all discovered words are permanently retained in Find Count across round transitions and server restarts.
- **Preserved 500+ Board Finds**:
  - Removed early return for 500+ word count range boards in `save_round_history`. While 500+ boards remain unranked for player ratings, word finds are permanently saved to history.
- **Global Module State & Cache Safety**:
  - Aliased `sys.modules['__main__']` to `'app'` at the entry point of `app.py` so background threads and modules importing `app` share the running `_WORD_FINDS_CACHE`.
  - Reduced `_WORD_FINDS_CACHE` TTL to 5.0 seconds and improved in-memory room deduplication.

### 14. Lobby Music FAQ Entry (`index.html`)
- **FAQ Quick Navigate & Answer**:
  - Added dedicated button in Quick Navigate FAQ grid and corresponding item in the FAQ container:
    - **Question**: *What music is playing in the Lobby?*
    - **Answer**: *Simon Patterson - Thump*

### 15. Game Room Chat Setting in Gameplay & Preferences (`index.html`, `play.css`, `settings.js`, `play.js`)
- **Gameplay & Preferences Toggle**:
  - Added "Game Room Chat" sub-navigation button and dedicated subpage (`#setting-subpage-game-room-chat`) under Gameplay & Preferences in Settings.
  - Provided a clean switch toggle ("Enable Game Room Chat") allowing players to easily enable or disable the live chat box inside game rooms.
- **Client & Server Persistence**:
  - Setting key `game_room_chat` is stored in `window.userSettings`, cached in `localStorage` (`morpheme_game_room_chat`), and persisted server-side via `/api/settings/update` into `user_settings`.
- **Dynamic In-Room Application**:
  - When disabled, `body.game-chat-disabled` is attached and `.chat-panel` receives `display: none !important;`. The Players panel automatically expands to utilize the full height of the left column.
  - `applyGameRoomChatSetting()` coordinates with `adjustPlayHeaderForDevice()`, room initialization, and tournament exit flows to guarantee the chat panel stays hidden when disabled and restores seamlessly when re-enabled.

### 16. High-Resolution & Crisp Image Uploads for Profile and Forum (`tools.js`, `forum.js`, `style.css`, `forum.css`)
- **Elimination of Aggressive Compression**:
  - Previously, avatar uploads in `tools.js` were forced down to a tiny `300px` max dimension with lossy JPEG `0.8` quality, causing extreme blur on high-DPI/Retina screens and when enlarged in the lightbox. Forum uploads in `forum.js` were forced down to `1200px` at `0.8` JPEG quality.
- **Original File Preservation**:
  - Avatars up to `2048px` and Forum images up to `2560px` that are under `8MB` bypass canvas re-encoding completely, maintaining pristine bit-for-bit sharpness.
  - Preserves original PNG (`image/png`) and WebP formats without converting them to lossy JPEG.
- **High-Quality Scaling Engine**:
  - When downscaling images exceeding 2048px/2560px, HTML5 Canvas smoothing is set to high (`ctx.imageSmoothingQuality = 'high'`) with `0.95` quality encoding.
- **Crisp CSS Rendering**:
  - Added `image-rendering: -webkit-optimize-contrast; image-rendering: auto;` and backface stabilization to `.image-lightbox-content img`, `.profile-avatar.large`, `.mini-avatar`, `.post-image-item img`, `.comment-image-item img`, `.preview-thumb`, and `.reply-preview-thumb` to prevent subpixel blurring in both thumbnail and enlarged modal views.

### 17. Game Room Chat Disabled Crossed-Out Horn Icon (`game_room.py`, `app.py`, `play.js`, `play.css`)
- **Icon Presence & Seamless Coexistence**:
  - Added a crossed-out horn icon (`🔇`, title `"Chat disabled (cannot read chats)"`) to the bottom-left corner of player cards (`.player-row-bottom`) for any user who has disabled game room chat.
  - Sits cleanly alongside the country flag and device input method icon (`[Flag] [Input] [🏆] [🔇]`).
  - **Zero Interference with Trophy Icon**: The presence logic and position of the 🏆 trophy icon for exceptional performances during intermission are completely unaffected. When a player with disabled chat earns an exceptional performance, both the trophy and crossed-out horn icons render side-by-side with clean flex spacing, while the flexible spacer `<div style="flex:1;"></div>` pushes the words count and score cleanly to the right.
- **Real-Time Client & Server Synchronization**:
  - Added `game_room_chat` to the `Player` dataclass in `game_room.py` and serialized it in `/api/room/<room_id>/state`.
  - Toggling the setting in Settings immediately updates the player's active room state in memory and calls `applyGameRoomChatSetting()`, which instantly refreshes the player list without waiting for polling.

### 18. Forum "Start a New Thread" Page: Mobile Zero-Scroll Fit & Desktop Full Page Extension (`templates/index.html`, `static/css/forum.css`)
- **Mobile Viewport Fit Without Vertical Scrolling**:
  - On mobile devices (`@media (max-width: 820px)` and `.is-mobile`), the entire "Start a New Thread" page (`#forum-view-create`), the card container (`.forum-create-card`), title, "What's on your mind?" input, "Details" textarea, "Attach Images" group, "Publish Thread" action button, and "← Cancel" bottom bar fit completely within the screen height with zero vertical scrolling (`scrollHeight == clientHeight`).
  - Added streamlined mobile padding (`10px 12px`), compact form gaps (`6px`), optimized input heights, flexible textarea min-height (`60px`), and full-width touch-friendly buttons.
- **Desktop & Laptop Full Page Extension**:
  - On desktop and laptop screens (>= 821px), the `.forum-create-card` and form flex layout stretch to occupy 100% of the available vertical height in `.forum-main`.
  - Completely eliminates the empty void below the form card.
  - The Details `textarea` (`#forum-post-content`) flexibly expands (`flex: 1 1 auto; height: 100%; min-height: 140px;`) while anchoring the "Publish Thread" button and "← Cancel" bottom bar cleanly at the bottom.

---

## Full Commit History (November 6, 2026)

| Commit ID | Message |
|:---|:---|
| `e66082a2` | Adhere to user selected timezone throughout application and standardize abbreviations |
| `c18c0e5f` | Update WHAT AM I LOOKING AT guide: replace TRAP with WONDERS and add pattern flow quote |
| `1fd551c7` | Keep scroll position when returning from thread to category list in forum |
| `9c57bd65` | Instantly change Players list to VIEW ALL at start and end of round |
| `c29ed66b` | Allow Wiktionary inflection cross-reference in is_verified_inflection for LEGERING and related words |
| `a6df2055` | Place users with 0 pts ahead of DNP users in game rooms |
| `8f9b8477` | Fit Subanagrams desktop random controls on single row and pair MUST HAVE with Random on mobile |
| `8f24e3f9` | Organize Subanagrams mobile controls and add desktop Store custom scroller with thumb |
| `7e651362` | Subanagrams: add MUST HAVE dropdown to Random tab and reorganize mobile layout |
| `0c91ea48` | Tools: extend Unscramble must-have full-width, compact Subanagrams mobile controls, and add desktop Store scroller thumb |
| `79e376cb` | Tools mobile: compact 2-column layout and shrunk controls for Unscramble and Is Valid matching Lists in Tools |
| `fc09a21b` | Light themes: style #1 place with vibrant yellow and crisp black outline |
| `83bf346a` | Players list & score sum: style 4th+ places in distinct sky blue, distinct from 2nd silver |
| `9d3561c8` | 24h rooms: fix Clues tab layout by enforcing strict tab-content visibility and removing title bleed |
| `11a21449` | Settings: remove arrows from sidebar category tabs, keeping arrows only on settings buttons within them |
| `19673cf1` | Settings: move navigation arrows to right side across tabs and buttons on desktop and laptop |
| `f126e947` | Personal Timer: add authentic gong audio, green start button, compact mobile layout, and relocate demo box from settings |
| `b66c42ad` | docs: establish Start Over stable point for November 6 (START_OVER_POINT_NOVEMBER_6) |
| `d48be28d` | docs: update Stable State Summary for November 6 (START_OVER_POINT_NOVEMBER_6) |
| `ccbcba5e` | Fix Find Count word retention: preserve history for mid-round joins and 500+ boards |
| `c5689261` | docs(faq): add What music is playing in the Lobby? (Simon Patterson - Thump) |
| `8ab0e881` | feat(settings): allow user to disable game room chat in Gameplay & Preferences |
| `36ecea8f` | fix(settings): correct game room chat toggle display state in desktop view |
| `d9eecdc9` | fix(images): preserve high resolution and crisp rendering for profile and forum images |
| `2884f3d6` | feat(play): add crossed-out horn icon in game room for users with chat disabled |
| `44fe2da9` | feat(forum): fit create thread page on mobile without vertical scrolling and extend to full page on desktop |

---

## Verification & Validation

1. **Timezone Adherence**:
   - Automated Selenium test (`verify_timezone.py` / `test_timezone.html`) verified `formatAppDate`, `formatTournamentStartDate`, `formatTimeoutTimestamp`, and `getAppTimezoneAbbr` across CT, ET, PT, MT, AKT, HST, and UTC.
   - Tested live production at `https://morpheme.games` with headless Chrome confirming `05/11/2026 5:12 PM CT` (Chicago) and `05/11/2026 6:12 PM CT` (Winnipeg) with zero CST/EST bleed.
2. **WHAT AM I LOOKING AT Guide**:
   - Verified demonstration board rendering word `WONDERS` and intuition quote live on production.
3. **Forum Category Scroll Retention**:
   - Verified category list scroll position retention across navigation transitions.
4. **Instant VIEW ALL Transition**:
   - Verified instant player list state change on round start and round end.
5. **Wiktionary Fallback**:
   - Verified live on `/api/definition?word=LEGERED` returning complete grammatical definition.
6. **Game Room Chat Setting**:
   - Verified toggle in Gameplay & Preferences hides chat panel and restores it immediately, persisting in `user_settings` and `localStorage`.
7. **Image Clarity & High-Resolution Uploads**:
   - Automated Selenium test (`scratch/test_image_clarity.py`) verified:
     - CSS `image-rendering` styling on `#image-lightbox-img` and `.profile-avatar.large`.
     - Preserving PNG original files untouched without downscaling or lossy compression.
     - Preserving aspect ratio and high 0.95 quality when scaling oversized 4000x3000 images down to 2560px max dimension.
     - Global availability of `window.compressImage` and `window.Forum`.
8. **Chat Disabled Crossed-Out Horn Icon**:
   - Automated Selenium test (`scratch/test_chat_disabled_icon.py`) verified:
     - Presence of crossed-out horn `🔇` icon when chat is disabled, with tooltip "Chat disabled (cannot read chats)".
     - Non-interference and harmonious coexistence with exceptional performance 🏆 trophy icon in intermission.
     - Disappearance of trophy icon during active rounds while crossed-out horn remains persistent.
     - Instant DOM re-rendering upon calling `applyGameRoomChatSetting()`.
9. **Forum Start a New Thread Mobile Fit & Desktop Full Page Extension**:
   - Automated Selenium test suite (`scratch/test_forum_create_layout.py`) verified:
     - **Mobile Viewports (390x844, 375x667, 412x915)**: `scrollBodyHasScroll == False` (`scrollHeight == clientHeight`), all elements including form card, title, title input, details textarea, image upload section, Publish Thread button, and Cancel bottom bar fit within the viewport with zero vertical scrolling.
     - **Desktop & Laptop Viewports (1280x850, 1440x900)**: `scrollBodyHasScroll == False`, form card fills 100% of available height in `.forum-main`, empty space below the form is eliminated, and Details textarea dynamically expands to fill available height.

