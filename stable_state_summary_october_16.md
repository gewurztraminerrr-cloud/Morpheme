# Stable State Summary — October 16, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_16
```
*(Also tagged as `stable-2026-10-16`)*

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| **localhost**      | ✅ Synchronized | Working tree clean, synced with origin/main |
| **GitHub (main)**  | ✅ Synchronized | Pushed to `origin/main` & tags updated on GitHub |
| **morpheme.games** | ✅ Synchronized | Deployed to 132.148.72.249 (PM2 online, verified live HTTP 200 OK) |
| **App / Client**   | ✅ Synchronized | Verified on desktop, laptop, tablet, and mobile layouts across all 39 themes |

---

## Session Highlights & Architectural Updates (October 15–16, 2026)

### 1. Forum Popups & Forms Readability Across All 39 Color Layouts
- **Universal High-Contrast Theming**:
  - Full theme-adaptive styling overhaul for all modal dialogs, popups, response windows, and composition forms in the Forum across all 39 Morpheme color layouts (dark themes, pure white, and all 18 light pastel themes).
- **Floating Response Popup Window (`.forum-reply-popup`)**:
  - **White & Light Layouts**: Pure `#ffffff` card with `#cbd5e1` borders, `#f8fafc` header, and high-contrast `#0f172a` text.
  - **Window Controls (Minimize & Close)**: Replaced low-contrast gray text with `#f1f5f9` backgrounds, `#cbd5e1` borders, and `#1e293b` text (with red hover state on close).
  - **Textarea & Placeholder**: `#ffffff` background with `1.5px solid #cbd5e1` border, `#0f172a` text, legible `#64748b` placeholders, and an active blue focus ring.
  - **Counter & File Upload**: High-contrast `#475569` character counter; file upload box styled with clean dashed borders and `#0f172a` label/icon text.
  - **Buttons**: Secondary "Cancel" button styled with neutral borders and `#1e293b` text; "Post Reply" retains the vibrant signature blue gradient with `#ffffff` text.
  - **Dark Themes**: Clean `#f8fafc` text on deep translucent slate (`rgba(0, 0, 0, 0.35)`), legible `#94a3b8` placeholders and counters, and clear window control buttons.
- **Original Post / New Thread Form (`.forum-create-card`)**:
  - Full white and light layout overrides ensuring title input, content textarea, labels, counters, and attachment boxes are rendered in clean white with crisp borders and dark text.
- **In-Thread Discussion Form (`.forum-comment-form`)**:
  - Extended full white and light theme adaptability to discussion textareas, counters, and attachment boxes.
- **Responses List Popup (`.forum-responses-popup`)**:
  - Background, header, close button, and response item cards adapt across white, light, and dark layouts.
  - Added deep violet `#6d28d9` on `rgba(109, 40, 217, 0.1)` for `.forum-post-number-badge` on light themes to prevent washed-out text.
- **Hover Preview & Mobile Preview Modals (`.forum-reply-hover-card`, `.forum-reply-mobile-card`)**:
  - Author name, date, post badge, response counter badge, snippet text, and the "Take Me There" navigation button maintain crisp, legible contrast across all themes.
- **External Post Navigation Confirmation Modal (`#generic-confirm-modal`)**:
  - Styled modal card, header, body text, Cancel, and Confirm buttons with theme-adaptive contrast.

### 2. Comprehensive AI Content Moderation & Anti-Spam Zero-Loophole Engine
- **Multi-Touchpoint Coverage**:
  - Automatically screens and moderates user inputs across all platform entry points:
    - **User Registration**: Real-time screening of usernames for obscenities, slurs, threats, and explicit terms with zero exceptions.
    - **User Profiles**: Profiles, bios, custom quotes, full names, locations, and avatar picture uploads.
    - **Lobby & Game Room Chat**: Real-time screening of chat messages, image attachments, rapid flooding, and duplicate message spam.
    - **Community Forum**: Real-time screening of thread titles, thread content, replies/comments, image uploads, and commercial advertisement links.
- **Zero-Latency Local Heuristic Guarantee**:
  - Built-in comprehensive regular expression and heuristic engine in `app.py:check_local_moderation` guaranteeing instant penalty enforcement without relying on external API availability or network latency.
  - Catches directed abuse (`fuck you`, `fuck off`, `f*ck you`, `go fuck yourself`, `what the fuck`, `stfu`, `shut the fuck up`, `gtfo`).
  - Catches severe harassment, self-harm, and violent threats (`kys`, `kill yourself`, `go die`, `get cancer`, `die in a fire`, `hope you die`).
  - Catches compound insults & profanities (`piece of shit`, `sack of shit`, `full of shit`, `load of shit`, `crock of shit`, `shithead`, `shitface`, `bullshit`, `shitty`, `dumbass`, `jackass`, `asshole`, `asswipe`, `dickhead`, `dipshit`, `douchebag`).
  - Catches hostile personal attacks (`you suck`, `u suck`, `you're trash`, `you are garbage`, `you're pathetic`, `you're worthless`).
  - Catches hate speech and slurs across racial, ethnic, religious, and sexual identities.
  - Catches commercial spam links (Telegram `t.me/`, WhatsApp `chat.whatsapp.com/`, Discord invites `discord.gg/`, crypto scams, and solicitation).
- **Competitive Scrabble/Boggle Dictionary Immunity**:
  - Morpheme is a competitive word game platform where valid dictionary words like *ASS*, *BITCH*, *TIT*, *CRAP*, *POOP*, *FART*, *DAMN*, and *SHIT* are routinely found and played on the board.
  - The moderation engine validates whether words are used in a legitimate gameplay context (e.g. *"I played ASS for 12 points"*, *"is BITCH valid in CSW?"*, *"I found SHIT on the board"*).
  - However, if **any** targeted pronoun or insult word (`you`, `you're`, `u`, `ur`, `piece of`, `sack of`, `dumb`, `stupid`, `ugly`) is present, game-context immunity is instantly denied and the message is blocked.

### 3. Active Multimodal Google Gemini Vision Pipeline
- **AI Computer Vision for Images**:
  - Integrated Google Gemini 1.5/3.5 Vision via `/v1beta/models/gemini-3.5-flash-lite:generateContent` to inspect binary image bytes on all photo upload routes.
  - Robust JSON response extraction with automatic markdown code-fence stripping (` ```json ... ``` `).
  - Configured active `GEMINI_API_KEY` on the production server (`/home/morpheme/morpheme/.env`) and verified with live multimodal test payloads.

### 4. Immediate Game Room UI Lockdown
- **Automatic & Instantaneous Lockdown**:
  - When an account is placed on timeout or ban, `window.lockAllGameRoomsUI(timeoutData)` executes immediately without waiting for user action.
  - Evicts the penalized user from any active game room back to the main lobby.
  - All game rooms across the application immediately render visually locked (`opacity: 0.45 !important; filter: grayscale(0.85) !important; cursor: not-allowed !important;`).
  - Injects a high-contrast glowing red warning banner at the top of the rooms list displaying the active restriction and remaining duration.
  - Background polling via `/api/rooms` and `/api/lobby-stats` synchronizes the locked state across all open tabs.

### 5. Lobby Button Click Modal Trigger for Banned & Timed-Out Users
- **Clickable Locked Buttons with Explanation Modals**:
  - Removed `pointer-events: none` and HTML `disabled` attributes from locked buttons, using `pointer-events: auto !important;` so browser click events register.
  - Global capture-phase click interceptor intercepts clicks on **any** game button (`.game-btn`, `.acc-btn`, `.fcfs-btn`, `.split-btn`), room card, room list item, Join/Locked button, Spectate button, Create Room button, or the warning banner.
  - **If Banned**: Pressing any button halts entry and launches the styled **Account Permanently Banned** modal explaining why the account was terminated.
  - **If Timed Out**: Pressing any button halts entry and launches the styled **Account Timed Out** modal with a real-time countdown timer and the specific violation reason.
  - When the timeout expires, background polling automatically unlocks the UI and restores normal gameplay buttons without requiring a full page reload.

### 6. Uniform Timeout Enforcement (Removal of Hardcoded Bypasses)
- **Eliminated Exemption Loops**:
  - Removed hardcoded developer exemptions in `app.py:check_user_timeout()` that previously caused `jeffb` to bypass shared IP timeout checks.
  - All users sharing an IP address now receive identical, uniform timeout and ban enforcement.

### 7. Exponential Penalty Ladder with 24-Hour Forgiveness Decay
- **Escalating Penalty Ladder**:
  - Automatic doubling formula: $\text{duration} = 10 \times 2^{(\text{offenses} - 1)}$ minutes.
  - Offense 1: **10 minutes** | Offense 2: **20 minutes** | Offense 3: **40 minutes** | Offense 4: **80 minutes** | Offense 5: **160 minutes** | Offense 6: **320 minutes** (capped at 7 days).
- **24-Hour Forgiveness Decay**:
  - For every 24 hours of clean, offense-free play, the user's recorded offense level decays by 1, allowing reformed players to heal their standing over time.
- **Dual-Layer Account & IP Tracking**:
  - Penalties are recorded to both the user account record and the `ip_timeouts` database table to prevent evasion via logout or account switching.

### 8. Morpheme Store Modernization & Dynamic Multi-Currency Engine
- **Profile Flag Currency Detection**:
  - In `static/js/store.js`, automatic currency mapping based on the user's selected profile country flag across 50+ world currencies with real-time conversion rates.
- **Centralized Catalog (`static/data/store_catalog.json`)**:
  - Structured, versioned catalog containing 17 curated products across 5 categories.
  - Automatic metadata tracking for `last_catalog_update` and `next_scheduled_update`.
- **Competitive Tournament Hardware**:
  - Added ASUS ROG Zephyrus G16, Custom Tournament PC Rig, 16" Apple MacBook Pro (M4 Max), and Apple Mac Studio + Studio Display 5K alongside tablets, phones, and official CSW/NWL dictionaries.
- **Mobile Store UI Overhaul**:
  - Dual-card status banner (`.store-status-dates-card`), currency badge pill (`.store-currency-card`), and touch-friendly pill category filters.

### 9. Mobile Game Room Chatbox Polish & Transition Stability
- Standardized mobile game room chat drawer behavior to match Lobby Chat with visual viewport tracking (`window.visualViewport`) to eliminate virtual keyboard layout jumps.

### 10. Forum Enhancements & Visual Restyling
- **Self-Reaction Prevention**: Prohibited users from liking or disliking their own threads, posts, or comments, with user-friendly toast feedback.
- **Restored Reaction Pill Borders**: Full high-contrast styling preserved on reaction pills with `cursor: not-allowed` for author's own posts.
- **500-Character Truncation**: Automatic body truncation with smooth interactive `"View Whole Post"` expansion.

### 11. Tools & Combo Checker Navigation
- **MP Word Explorer Mobile Horizontal Extension**: Full-width responsive controls and fluid layout on mobile devices.
- **Desktop/Laptop "← Back" Navigation**: Multi-hop stack-based navigation (`window._defPopupNavStack`) to seamlessly navigate between definition popups, CC, and FC.
- **Tools Tab Screen Freeze Resolution**: MutationObserver guard ensuring tool switching never resets the active tab or triggers lockups.

### 12. Cache Busting & Asset Versions
- In `templates/index.html`:
  - `lobby.css?v=1791173000`
  - `app.js?v=1791173000`
  - `lobby.js?v=1791173000`
  - `play.css?v=1791164500`
  - `play.js?v=1791172000`
  - `store.js?v=1791164500`
  - `forum.css?v=1792196000`
  - `forum.js?v=1791160800`
  - `tools.js?v=1791150200`

---

## Complete Core System Architecture & Operational Reference

### 1. Game Rooms & Board Generation
- **Board Geometries**: Standard 4x4 (Boggle), 5x5 (Big Boggle), 6x8 (Super Boggle), 3D Cube (4x4x4), and Subanagrams.
- **Validation Engine**: Dual-dictionary validation (NWL & CSW) with Added Words (AW) tries and optimized C engine (`morpheme_metric.c`).
- **24-Hour Daily Challenge**: Automated daily boards pre-generated at midnight with persistent player rankings.

### 2. Tools Architecture
- **Combo Checker (CC)**: Microprocedures (MP) v2.0, Letters In Common (LIC), and MP Word Explorer.
- **Find Count (FC)**: Discovery leaderboards and lifetime word frequency statistics.
- **Lists & Study Sets**: High-probability word lists, subanagrams, CSW Only, New NWL, and custom study sets.

### 3. Server & Infrastructure
- **Remote Production Host**: Ubuntu 24.04 LTS at `132.148.72.249`.
- **Application Stack**: Python 3.9 + Waitress WSGI managed by PM2 (`morpheme`).
- **Database**: SQLite (`morpheme.db`) with WAL mode, automated migrations, IP ban cache, and timeout tables.
