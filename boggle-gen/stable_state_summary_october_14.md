# Stable State Summary — October 14, 2026 (End of Day / Start Over Point)

## Latest Commit ID
```
ff1fcecbd9c026ca012714a2ff7b0febe4acb7f2
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

### 1. Forum: Dual Reply Interfaces (Stationary Textbox & Floating Reply Popup)
- **Coexisting Stationary & Floating Modes**:
  - Preserved the fixed, docked comment textbox at the bottom of every thread for continuous desktop and mobile typing.
  - Clicking the "Reply" button on any post or comment opens a dedicated, movable floating reply popup window pre-filled with `@Username #NUM`.
  - Both composers feature independent rich-text formatting toolbars, live 2,000-character counters, image attachment pickers, and posting spinners.
- **Mobile Reply Continuity**:
  - On mobile devices and touchscreens, tapping "Reply" opens the floating popup cleanly without viewport jumping, ensuring full compliance with the Mobile Fullscreen & Virtual Keyboard Invariant Rules.

### 2. Sequence Tool: Scroll Position & Target Word Restoration on "Back" Navigation
- **Live Scroll Tracking**:
  - Added continuous scroll tracking on `#seq-list-results` and `.tools-content` so navigation preserves exact scroll offsets.
- **Exact Selection Anchor**:
  - Clicking any word in the Sequence tool records the selected word and scroll offsets before opening the definition popover or navigating to Combo Checker ("CC") / Find Count ("FC").
- **Multi-Frame Restoration**:
  - Pressing "Back" (`toolsGoBack` or switching back to Sequence) restores scroll position across multiple rendering frames (`requestAnimationFrame`, 30ms, 100ms, 250ms, 450ms).
  - Automatically scrolls the selected word into view (`block: 'center'`) and triggers the glowing target animation (`jumpTargetGlow` / `.jump-target-pulse`).

### 3. Forum: Desktop Reply Link Hover Post Previews & Unique Responder Counts
- **Floating Hover Preview Card**:
  - Hovering over any post link (e.g., `#14` or `@User #14`) on desktop/laptop displays a floating glassmorphic preview card.
  - Displays author avatar initial, country flag, username, formatted timestamp, post number badge (`#14`), full post content, and attachment thumbnails.
  - Smoothly fades and dismisses when the mouse cursor leaves the link or preview card.
- **Unique User Response Counter**:
  - Positioned directly adjacent to the post number badge in the preview card, displays how many unique users replied to that specific post (e.g., `💬 3 users responded` or `💬 1 user responded`).

### 4. Forum: Mobile Dedicated Preview Modal & "Take Me There" Navigation
- **Dedicated Bottom Modal Preview**:
  - On mobile devices and touchscreens, tapping `#NUM` opens an overlay card (`#forum-reply-mobile-overlay`) displaying the replied-to post, author metadata, full content, and responder count.
- **"Take Me There" Interactive Navigation**:
  - Features a prominent **"Take Me There →"** button.
  - Tapping it dismisses the modal, smoothly scrolls the viewport straight to the target post in the thread, and illuminates it with a glowing gold highlight pulse (`forum-item-highlight-gold`).
  - Preserves continuous mobile fullscreen without unwanted exit triggers.

### 5. How to Play & FAQ: Comprehensive Forum Features & Tools Guide
- **Quick Navigation Link**:
  - Added a direct button: **"How does the Forum work and what are all of its features?"** to the Quick Navigate FAQ grid with smooth scrolling and highlight pulse.
- **Exhaustive Documentation (`#faq-forum-guide`)**:
  - **Access & Permissions**: Guest read-only access vs. registered member actions (creating threads, replies, mentions, uploads, bookmarks).
  - **Categories & Real-Time Presence**: Channel categories, live active viewers counter with 20s heartbeats and exit beacons (`navigator.sendBeacon`), 30s auto-refresh, and mobile drawer touch swipe gestures.
  - **Thread Lists & Status Badges**: Thread cards with author avatars, flags, view counts (👁️), reply counts (💬), last responder info, Pinned 📌 threads, Locked 🔒 threads, gold unread glow, and instant title/author search.
  - **Creating Threads & Rich Formatting**: 100-character titles, 2,000-character content with live counters, formatting bar (bold, italic, strikethrough, code, quote, spoiler), and smart `@` autocomplete for Friends and players.
  - **Image Attachments & Lightbox**: Multi-image uploads, client-side auto-compression (max 1200px, 0.8 quality) for zero-lag transmission, 2MB GIF support, removable preview thumbnails, and full-resolution edge-to-edge Lightbox modals.
  - **Thread View & Bookmarks**: Authoritative post numbering (`#1`, `#2`, `#14`), personal bookmarking (🔖), "Jump to Bookmark", and "Scroll to Bottom".
  - **Dual Reply Interfaces**: Docked stationary textbox vs. floating popup window.
  - **Interactive Reply Links & Previews**: Hover previews on desktop, mobile tap modal with "Take Me There →", responder tags, and cross-thread references.
  - **Moderation & Fullscreen Invariant**: Post editing/deletion, moderator controls, reporting in Complaints, and 100% adherence to mobile continuous fullscreen.

### 6. Forum: Post Likes & Dislikes Across All Categories
- **Thumbs Up / Thumbs Down Voting**:
  - Interactive Likes (👍) and Dislikes (👎) on thread opening posts and comments across all categories.
  - Backed by SQLite table `forum_reactions` with `(user_id, post_id, comment_id)` constraints and fast indexed lookups.
  - Endpoint `/api/forum/post/<id>/react` supporting instant toggling and switching.
- **Suggestions/Ideas Category Focus**:
  - Emphasized voting in the Suggestions/Ideas header banner to reflect community prioritization.

### 7. Forum: Mobile Layout & Centered Action Toolbar
- **Full-Width Text Below Avatar & Actions**:
  - Restructured post cards so text begins full-width beneath the user's thumbnail avatar and action buttons, eliminating left-indentation on mobile.
- **Centered Action Row**:
  - Structured action buttons (`👍`, `👎`, `🔖 Bookmark`, `↩ Reply`) into a centered full-width row with `flex-wrap: nowrap` and `white-space: nowrap`.

### 8. Forum: Real-Time Viewers Count & Autocomplete Mentions (`@username`)
- **Active Viewers Badge**: Live badge beside Forum title indicating active users viewing forums (e.g., `1 viewing`).
- **Autocomplete Mentions**: Typing `@` opens an interactive popup filtering Friends and active players with arrow/touch selection and virtual keyboard awareness via `window.visualViewport`.

### 9. Profile & Mini-Profile Enhancements
- **Field Character Limits**: Name restricted to 50 chars, Age to 3 chars, Gender to 25 chars.
- **Lifetime Forum Post Count**: Added `💬 Forum Posts: [number]` to profiles and mini-profiles backed by `users.total_forum_posts`, with click-to-search integration.

### 10. Appearance Settings & Game Contrast
- Added **Black** (`#000000`) and three shades of grey (**Light Grey** `#d4d4d4`, **Medium Grey** `#888888`, **Dark Grey** `#4a4a4a`) for Highlight as you Type / Mouse/Swipe with automatic letter contrast inversion.

### 11. Word Definition Popup: Dictionary Badges & Color Coding
- Added colored dictionary badges (`NWL` in blue, `CSW` in green, `AW` in amber) adjacent to the word length tag in the definition popup header, consistent with Words to Work on.

### 12. Lobby Polish & Registration Validation
- Changed Lobby chatbox placeholder to **`"Type Lobby message..."`**.
- Upgraded 3D obsidian depth on Lobby Journey and New User Guide buttons.
- Restricted username registration to letters, numbers, and underscores with clear inline validation notices.
