# Stable State Summary — October 14, 2026

## Latest Feature Commit ID
```
b1dbc9d12ae43ffdb6b1e5be0233b8ef16fef464
```

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_14
```
*(Also tagged as `stable-2026-10-14`)*

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| localhost      | ✅ Synchronized | Working tree clean, HEAD at `b1dbc9d1` |
| GitHub (main)  | ✅ Synchronized | Pushed to `origin/main` & tags updated on GitHub |
| morpheme.games | ✅ Synchronized | Deployed to 132.148.72.249 (PM2 online, HTTP 200 OK) |
| App / Client   | ✅ Synchronized | Verified on desktop, laptop, mobile layouts |

---

## Session Features & Enhancements (October 14, 2026)

### 1. Forum Real-Time Viewers Count Badge
- Added a live badge beside the Forum title indicating active users viewing the forums (e.g. `1 viewing`).
- Matched the exact visual styling, sizing, typography, and rounded badge treatment as the `[number] posts today` counter beside category titles.
- Backed by real-time presence tracking and periodic heartbeat polling.

### 2. Standalone Mentions (`@username`) & Autocomplete Friends Popup
- Supported tagging users with `@username` without requiring a post number.
- Tagged recipients have their "View Responses" tab and the corresponding post card highlighted gold until clicked.
- Added an interactive autocomplete popup window that activates upon typing `"@"` in thread titles, details, or reply comments.
- Dynamically filters the list to friends matching the query starting character, with keyboard selection (`ArrowUp`/`ArrowDown`/`Enter`/`Tab`) and touch-safe scrolling.
- Automatically hides the popup once no matching friends remain, allowing the user to tag anyone not on their friends list.
- Comprehensive mobile virtual keyboard integration:
  - Hoisted the popup element directly to `document.body` to prevent container clipping from `#page-forums` on mobile viewports.
  - Integrated `window.visualViewport` to dynamically position the popup above the on-screen keyboard when lower screen space is constrained.
  - Added async session recovery so friends load reliably regardless of mobile connection speed.

### 3. Profile Character Limit Constraints
- Reduced maximum allowable character limits for user profile fields:
  - **Name**: 50 characters (reduced from 100).
  - **Age**: 3 characters (reduced from 100).
  - **Gender**: 25 characters (reduced from 100).
- Applied coordinated validation across frontend inputs and backend API handlers.

### 4. Appearance Settings: Highlight Colors & Black Contrast Inversion
- Added **Black** (`#000000`) and three shades of grey (**Light Grey** `#d4d4d4`, **Medium Grey** `#888888`, **Dark Grey** `#4a4a4a`) in place of the single White option for both:
  - **Highlight as you Type**
  - **Highlight as you Mouse/Swipe**
- Capitalized "Type" in the setting title "Highlight as you Type".
- Implemented automatic contrast text color inversion: when a black or dark grey highlight color is selected, tile letters invert to white (`#ffffff`) to maintain crisp, readable contrast across all game modes and themes.

### 5. Lobby Chatbox Placeholder Update
- Changed placeholder message in the Lobby chatbox from `"Type a message to the lobby..."` to **`"Type Lobby message..."`**.

### 6. Profile & Mini-Profile: Lifetime Forum Post Count & Interactive Search
- Added `💬 Forum Posts: [number]` to the profile metadata row immediately to the left of **Proof** on desktop and laptop displays, and ordered directly above Proof on mobile.
- Added matching `💬 Forum Posts: [number]` to **Mini-Profiles** directly above Proof.
- Backed by a permanent lifetime post counter in `users.total_forum_posts` that never decreases when older threads fall off active feeds.
- Clicking the Forum Posts chip opens the Forum page, populates user search, and displays all posts and replies by that user.

### 7. Forum Card Borders, Delineations & Response Viewer Popup
- Thread replies and opening posts styled with distinct card borders and backgrounds matching category thread cards across both dark and light/white themes.
- Displayed interactive response count chips (e.g., `#2 💬 3`) directly on post numbers that have replies.
- Clicking a post badge opens a dedicated popup listing all direct responses, with smooth scrolling and gold focus highlighting.
- Thread title positioned cleanly below username/author box with post number and responses badge to the left of the title.

### 8. Forum Thread Bookmarks & 500-Post Limit Lock
- Added thread bookmarking to save and jump to the user's last-read reply.
- Enforced a 500-post limit per thread with a notification banner when closed.
- Added Scroll to Bottom button and icon-based thread refresh button.

---

## Deployment & Verification Log
- **Git Commit**: `b1dbc9d12ae43ffdb6b1e5be0233b8ef16fef464`
- **GitHub Tags**: `START_OVER_POINT_OCTOBER_14`, `stable-2026-10-14`
- **Remote Production Host**: `132.148.72.249`
- **PM2 Service**: `morpheme` online (pid 2259, status online)
- **HTTP Health**: `HTTP/1.1 200 OK` on `https://morpheme.games/`
- **Cache Busters**:
  - `forum.js?v=1791130020`
  - `forum.css?v=1791130020`
  - `app.js?v=1791130020`
  - `play.css?v=1791130010`
  - `settings.js?v=1791130010`
