# Stable State Summary — October 14, 2026 (End of Day)

## Latest Feature Commit ID
```
9cbfdd2c1f73752e505cead10aee79b90eb11c6c
```

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_14
```
*(Also tagged as `stable-2026-10-14`)*

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| localhost      | ✅ Synchronized | Working tree clean, synchronized with origin/main |
| GitHub (main)  | ✅ Synchronized | Pushed to `origin/main` & tags updated on GitHub |
| morpheme.games | ✅ Synchronized | Deployed to 132.148.72.249 (PM2 online, HTTP 200 OK) |
| App / Client   | ✅ Synchronized | Verified on desktop, laptop, tablet, and mobile layouts |

---

## Session Features & Enhancements (October 14, 2026)

### 1. Forum: Post Likes & Dislikes Across All Categories
- **Thumbs Up / Thumbs Down Voting**:
  - Implemented interactive Likes (👍) and Dislikes (👎) on thread opening posts and reply comments across all forum categories.
  - Positioned directly to the left of the Bookmark button on post toolbars.
  - Backed by `forum_reactions` table in SQLite (`morpheme.db`) with `(user_id, post_id, comment_id)` constraints and fast indexed lookups.
  - Interactive backend endpoint `/api/forum/post/<id>/react` supporting instant toggling and switching reactions.
- **Suggestions/Ideas Category Focus**:
  - Emphasized voting functionality in the Suggestions/Ideas header description to highlight how community favorability informs developer and moderator prioritization.
  - Maintained the clean short description in the sidebar tab while expanding the detailed voting explanation in the category header banner.

### 2. Forum: Mobile Layout & Centered Action Toolbar
- **Full-Width Text Below Avatar & Actions**:
  - Restructured post detail and reply comment cards so that post text begins full-width beneath the user's thumbnail avatar and action buttons.
  - Eliminated the wide mobile left-indentation that previously trapped text beside the avatar thumbnail.
- **Centered Action Row**:
  - On mobile devices, structured the action buttons row (`👍`, `👎`, `🔖 Bookmark`, `↩ Reply`) into a full-width container (`width: 100%`) centered horizontally (`justify-content: center`).
  - Enforced `flex-wrap: nowrap` and `white-space: nowrap` across all action buttons, ensuring the Reply button stays on the same row without awkward wrapping on narrow screens.

### 3. Forum: Real-Time Viewers Count & UI Refinements
- **Real-Time Active Viewers Badge**:
  - Added a live badge beside the Forum title indicating active users viewing the forums (e.g., `1 viewing`).
  - Matched the styling, sizing, typography, and pill treatment of the `[number] posts today` badge.
  - Powered by real-time presence tracking and heartbeat polling.
- **Header Metadata Placement**:
  - In Forum thread headers, placed the post number badge and responses count directly to the right of the username.
  - Tab divider lines added between tabs in Tools, Settings, Mods, and Forum on desktop and laptop displays.

### 4. Forum: Autocomplete Mentions (`@username`) & Virtual Keyboard Integration
- Supported tagging users with `@username` without requiring a post number.
- Added an interactive autocomplete popup window that activates upon typing `"@"` in thread titles, details, or reply comments.
- Dynamically filters the list to friends matching the query starting character, with keyboard selection (`ArrowUp`/`ArrowDown`/`Enter`/`Tab`) and touch-safe scrolling.
- Automatically hides the popup once no matching friends remain, allowing the user to tag anyone not on their friends list.
- Comprehensive mobile virtual keyboard integration:
  - Hoisted the popup element directly to `document.body` to prevent container clipping from `#page-forums` on mobile viewports.
  - Integrated `window.visualViewport` to dynamically position the popup above the on-screen keyboard when lower screen space is constrained.
  - Added async session recovery so friends load reliably regardless of mobile connection speed.

### 5. Profile & Mini-Profile Enhancements
- **Character Limits**:
  - **Name**: 50 characters (reduced from 100).
  - **Age**: 3 characters (reduced from 100).
  - **Gender**: 25 characters (reduced from 100).
  - Coordinated across frontend input attributes and backend API validation.
- **Lifetime Forum Post Count**:
  - Added `💬 Forum Posts: [number]` to the profile metadata row immediately to the left of **Proof** on desktop/laptop, and directly above Proof on mobile.
  - Added matching `💬 Forum Posts: [number]` to **Mini-Profiles** directly above Proof.
  - Backed by permanent lifetime counter `users.total_forum_posts`.
  - Clicking the Forum Posts chip opens Forum search for that user's posts.

### 6. Appearance Settings & Game Contrast
- Added **Black** (`#000000`) and three shades of grey (**Light Grey** `#d4d4d4`, **Medium Grey** `#888888`, **Dark Grey** `#4a4a4a`) in place of the single White option for:
  - **Highlight as you Type**
  - **Highlight as you Mouse/Swipe**
- Implemented automatic contrast text color inversion: when black or dark grey highlights are active, tile letters invert to white (`#ffffff`) for readability.
- Capitalized "Type" in "Highlight as you Type".

### 7. Lobby & Registration Improvements
- Changed Lobby chatbox placeholder from `"Type a message to the lobby..."` to **`"Type Lobby message..."`**.
- Fixed clickability on Lobby Journey banner and added 3D popping effect to lobby black messages.
- Polished lobby button styling: removed outer glow while preserving 3D extrusion and diamond shimmer.
- Prominent 3D obsidian depth: upgraded the top gradient highlight (`#383a48`), specular inset top edge, and physical extruded side bevels for both the Lobby Journey banner and the New User Guide button.
- Active rooms border: added black border on desktop for white layouts, thicker 2px borders for default layout.
- Registration validation: restricted username input to letters, numbers, and underscores, and relocated validation notices immediately below their respective containers.

### 8. Forum Thread Bookmarks & 500-Post Limit Lock
- Added thread bookmarking to save and jump to the user's last-read reply.
- Enforced a 500-post limit per thread with a notification banner when closed.
- Added Scroll to Bottom button and icon-based thread refresh button.

---

## Deployment & Verification Log
- **Git Commit**: `9cbfdd2c1f73752e505cead10aee79b90eb11c6c`
- **GitHub Tags**: `START_OVER_POINT_OCTOBER_14`, `stable-2026-10-14`
- **Remote Production Host**: `132.148.72.249`
- **PM2 Service**: `morpheme` online (status online, memory healthy)
- **HTTP Health**: `HTTP/2 200 OK` on `https://morpheme.games/`
- **Cache Busters**:
  - `forum.css?v=1791130140`
  - `forum.js?v=1791130130`
  - `app.js?v=1791130100`
  - `play.css?v=1791130090`
  - `settings.js?v=1791130010`
  - `lobby.css?v=1791130080`
