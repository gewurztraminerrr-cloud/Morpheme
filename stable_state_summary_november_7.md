# Stable State Summary — November 7, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_NOVEMBER_7
```
*(Also tagged as `stable-2026-11-07`)*

## Latest Commit ID
```
Resolved with: git rev-parse START_OVER_POINT_NOVEMBER_7
```

## Synchronization Status

| Environment | Status | Commit / Tag | Verification Details |
|:---|:---|:---|:---|
| **Localhost** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_7` | Working tree on `main`, zero uncommitted tracked changes |
| **GitHub (`origin/main`)** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_7` | Pushed to `origin/main` & tags `START_OVER_POINT_NOVEMBER_7` / `stable-2026-11-07` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | `START_OVER_POINT_NOVEMBER_7` | Fast-forward pulled to origin/main, PM2 online (process `morpheme`), HTTP/2 200 |
| **App (Mobile WebView)** | ✅ Synchronized | `https://morpheme.games/` | Flutter app loads live synchronized webview with native bridges |

---

## Session Highlights & Architectural Updates

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

### 4. Instant Players List VIEW ALL Mode Transition (`play.js`)
- **Real-Time Responsiveness**:
  - Automatically switches the Players list view mode to `VIEW ALL` immediately upon round start (`START_OF_ROUND`) and round end (`END_OF_ROUND`) without waiting for periodic polling intervals.

### 5. Wiktionary Suffix-Drop Verification Fallback (`definition_enrichment.py`)
- **Grammatical Rule & Inflection Fallback**:
  - When enriching definitions for inflected words (e.g. `LEGERED`, `LEGERING`), Wiktionary suffix-dropping fallback extracts base lemmas (`LEGER`) to ensure accurate definitions are returned even if direct word-form entries are unlisted.

### 6. Game Room Scores: 0 pts Placed Ahead of DNP Users (`play.js`)
- **Leaderboard Accuracy**:
  - Users who played a round but scored 0 points (`0 pts`) are ordered strictly ahead of Did Not Play (`DNP`) players in room standings and end-of-round score summaries.

### 7. Subanagrams & Tools Mobile Control Optimization (`tools.js`, `style.css`)
- **Layout & Usability**:
  - Compacted two-column layouts and unified button heights on mobile devices across Subanagrams, Unscramble, and Is Valid tools.
  - Positioned "MUST HAVE" dropdown alongside Random generation controls on mobile devices.

### 8. Desktop Store Custom Scrollbar with Draggable Thumb (`store.js`, `style.css`)
- **Enhanced Desktop Navigation**:
  - Designed and deployed custom scrollbar thumb with drag-and-scroll functionality for the Store page on desktop and laptop displays.

### 9. Light Theme 1st Place High-Contrast Yellow Styling (`play.css`)
- **Visual Distinction**:
  - Styled 1st-place ranks in light and white themes with a vibrant gold-yellow fill (`#ffd700`) framed by a crisp outline for maximum readability.

### 10. Players List & Score Sum 4th+ Place Sky Blue (`play.css`)
- **Rank Color Palette**:
  - Styled 4th place and below in a distinct soft sky blue (`#60a5fa`) across players list and score sums, creating clean visual separation from 2nd-place silver.

### 11. 24h Rooms Clues Tab Visibility (`play.js`, `play.css`)
- **Tab Layout Integrity**:
  - Enforced strict tab-content scoping for Clues tabs in 24h puzzle rooms, preventing title bleed and unrendered containers.

### 12. Settings Navigation Arrows Refinement (`settings.js`, `settings.css`)
- **UI Consistency**:
  - Removed redundant arrows from sidebar category tabs, preserving right-aligned navigation arrows exclusively on action buttons within settings panels across desktop and laptop viewports.

### 13. Personal Timer Authentic Gong Audio & UI (`tools.js`, `index.html`)
- **Feature Refinement**:
  - Integrated authentic high-fidelity gong audio chime on timer completion, added vibrant green start button, and streamlined mobile layout while relocating demonstration panels out of settings.

### 14. Find Count Board Word Retention for Mid-Round Joins & 500+ Boards (`play.js`, `game_room.py`)
- **Persistence Across Rounds**:
  - Preserved found-word history in Find Count across mid-round joins and large 500+ board cycles without clearing active session progress.

### 15. Lobby Music FAQ Documentation (`templates/index.html`)
- **Music Attribution**:
  - Documented background music track title and artist (*Simon Patterson - Thump*) in the Lobby & FAQ guide.

### 16. Game Room Chat Disabling Preference (`settings.js`, `play.js`, `play.css`, `game_room.py`, `app.py`)
- **User Preference Control**:
  - Added "Game Room Chat" toggle under Settings -> Gameplay & Preferences allowing players to disable in-game chat panels.
  - Instantly hides the chat panel and persists preference to backend database (`user_settings`) and `localStorage`.

### 17. Game Room Chat Disabled Crossed-Out Horn Icon (`game_room.py`, `app.py`, `play.js`, `play.css`, `templates/index.html`)
- **Icon Presence & Seamless Coexistence**:
  - Added a crossed-out horn icon (`🔇`, title `"Chat disabled (cannot read chats)"`) to the bottom-left corner of player cards (`.player-row-bottom`) for any user who has disabled game room chat.
  - Sits cleanly alongside the country flag and device input method icon (`[Flag] [Input] [🏆] [🔇]`).
  - **Zero Interference with Trophy Icon**: The presence logic and position of the 🏆 trophy icon for exceptional performances during intermission are completely unaffected. When a player with disabled chat earns an exceptional performance, both the trophy and crossed-out horn icons render side-by-side with clean flex spacing, while the flexible spacer `<div style="flex:1;"></div>` pushes the words count and score cleanly to the right.
- **Cross-Browser & Platform Rendering (Circle with Cross Through It)**:
  - Styled `.chat-disabled-icon` with explicit font stack `font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif` and inline-flex vertical centering, ensuring the Apple-designed speaker horn with the red circle with a cross through it renders consistently across Chrome, Safari, and Firefox.
  - Bumped asset cache busters (`play.js?v=1794030000`, `play.css?v=1794030000`, `forum.css?v=1794030000`) in `templates/index.html` to invalidate cached assets in Google Chrome so the icon appears immediately on all laptops and desktops without requiring manual cache clearing.
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

## Full Commit History (Leading up to November 7, 2026)

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
| `f6dd995f` | feat(forum): fit create thread page on mobile without vertical scrolling and extend to full page on desktop |
| `8252dce0` | fix(play): ensure chat disabled icon appears on Google Chrome and all desktop browsers with circle and cross |
| `1d31c4f9` | docs: establish Start Over stable point for November 7 (START_OVER_POINT_NOVEMBER_7) |

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
     - Cross-browser font stack rendering with `Apple Color Emoji` across Chrome, Safari, and Firefox.
9. **Forum Start a New Thread Mobile Fit & Desktop Full Page Extension**:
   - Automated Selenium test suite (`scratch/test_forum_create_layout.py`) verified:
     - **Mobile Viewports (390x844, 375x667, 412x915)**: `scrollBodyHasScroll == False` (`scrollHeight == clientHeight`), all elements including form card, title, title input, details textarea, image upload section, Publish Thread button, and Cancel bottom bar fit within the viewport with zero vertical scrolling.
     - **Desktop & Laptop Viewports (1280x850, 1440x900)**: `scrollBodyHasScroll == False`, form card fills 100% of available height in `.forum-main`, empty space below the form is eliminated, and Details textarea dynamically expands to fill available height.
