# Stable State Summary — October 14, 2026 (End of Day / Start Over Point)

## Latest Commit ID
```
f7b8964c45d0316e2a8261d0c62642f5cfcccdf4
```

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_14
```
*(Also tagged as `stable-2026-10-14`)*

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| localhost      | ✅ Synchronized | Commit `f7b8964c`, working tree clean, synced with origin/main |
| GitHub (main)  | ✅ Synchronized | Pushed to `origin/main` & tags updated on GitHub |
| morpheme.games | ✅ Synchronized | Deployed to 132.148.72.249 (PM2 online, HTTP 200 OK) |
| App / Client   | ✅ Synchronized | Verified on desktop, laptop, tablet, and mobile layouts |

---

## Session Features & Enhancements (October 14, 2026)

### 1. Forum: Continuous Top Menu & Restored Bottom Action Bar (Back, Return to Category, Cancel)
- **Top Menu Bar Continuous Visibility**:
  - Replaced destructive `element.scrollIntoView({ block: 'start' })` window scrolling calls with horizontal container scrolling (`container.scrollTo({ left: ..., behavior: 'smooth' })`).
  - Purged duplicate header padding and selector overrides from `static/css/forum.css` so that the base `static/css/style.css` rules govern header sizing, fonts, padding, and alignments uniformly across Lobby and Forum with zero content shift.
- **Restoration of Forum Bottom Bar & Navigation Buttons (`← Back`, `← Back to category`, `← Cancel`)**:
  - **Root Cause**: The mobile viewport lock had targeted a non-existent `#app` element for flex column layout, leaving `body` in default block display. Consequently, `.pages` expanded to 100% of body height below the ~120px top menu bar, pushing the bottom 120px of the forum container (which holds `.forum-bottom-bar`) completely off the bottom edge of the screen and clipping it under `overflow: hidden`.
  - **Fix Applied**:
    - Applied `display: flex !important; flex-direction: column !important; box-sizing: border-box !important;` directly to `html:has(#page-forums.active), body:has(#page-forums.active), body.forums-active`.
    - Made `.top-menu-bar` and `.separator` strictly non-shrinking (`flex: 0 0 auto !important; flex-shrink: 0 !important;`).
    - Configured `.pages` to fill the exact remainder of the viewport (`flex: 1 1 0% !important; min-height: 0 !important; height: auto !important;`).
    - Configured `.forum-view-scroll-body` to flex and scroll internally (`flex: 1 1 0px !important; min-height: 0 !important; overflow-y: auto !important;`).
    - Locked `.forum-bottom-bar` firmly at the bottom of the viewport (`flex: 0 0 auto !important; flex-shrink: 0 !important; z-index: 50 !important;`) with iPhone safe-area padding (`padding: 8px 14px max(8px, env(safe-area-inset-bottom)) 14px !important;`).
    - Removed conflicting height and overflow overrides on `#page-forums` in `static/css/play.css`.
  - **Result**:
    - `← Back` (`#forum-category-back-btn`) is 100% visible and anchored at the bottom of the thread list view on mobile.
    - `← Back to category` / Return to Category (`#forum-back-to-list`) is 100% visible and anchored at the bottom of the single thread view.
    - `← Cancel` (`#forum-cancel-create`) is 100% visible and anchored at the bottom of the create post view.

### 2. Forum: Dual Reply Interfaces (Stationary Textbox & Floating Reply Popup)
- **Coexisting Stationary & Floating Modes**:
  - Preserved the fixed, docked comment textbox at the bottom of every thread for continuous desktop and mobile typing.
  - Clicking the "Reply" button on any post or comment opens a dedicated, movable floating reply popup window pre-filled with `@Username #NUM`.
  - Both composers feature independent rich-text formatting toolbars, live 2,000-character counters, image attachment pickers, and posting spinners.
- **Mobile Reply Continuity**:
  - On mobile devices and touchscreens, tapping "Reply" opens the floating popup cleanly without viewport jumping, ensuring full compliance with the Mobile Fullscreen & Virtual Keyboard Invariant Rules.

### 3. Sequence Tool: Scroll Position & Target Word Restoration on "Back" Navigation
- **Live Scroll Tracking**:
  - Added continuous scroll tracking on `#seq-list-results` and `.tools-content` so navigation preserves exact scroll offsets.
- **Exact Selection Anchor**:
  - Clicking any word in the Sequence tool records the selected word and scroll offsets before opening the definition popover or navigating to Combo Checker ("CC") / Find Count ("FC").
- **Multi-Frame Restoration**:
  - Pressing "Back" (`toolsGoBack` or switching back to Sequence) restores scroll position across multiple rendering frames (`requestAnimationFrame`, 30ms, 100ms, 250ms, 450ms).
  - Automatically scrolls the selected word into view (`block: 'center'`) and triggers the glowing target animation (`jumpTargetGlow` / `.jump-target-pulse`).

### 4. Forum: Desktop Reply Link Hover Post Previews & Unique Responder Counts
- **Floating Hover Preview Card**:
  - Hovering over any post link (e.g., `#14` or `@User #14`) on desktop/laptop displays a floating glassmorphic preview card.
  - Displays author avatar initial, country flag, username, formatted timestamp, post number badge (`#14`), full post content, and attachment thumbnails.
  - Smoothly fades and dismisses when the mouse cursor leaves the link or preview card.
- **Unique User Response Counter**:
  - Positioned directly adjacent to the post number badge in the preview card, displays how many unique users replied to that specific post (e.g., `💬 3 users responded` or `💬 1 user responded`).

### 5. Forum: Mobile Dedicated Preview Modal & "Take Me There" Navigation
- **Dedicated Bottom Modal Preview**:
  - On mobile devices and touchscreens, tapping `#NUM` opens an overlay card (`#forum-reply-mobile-overlay`) displaying the replied-to post, author metadata, full content, and responder count.
- **"Take Me There" Interactive Navigation**:
  - Features a prominent **"Take Me There →"** button.
  - Tapping it dismisses the modal, smoothly scrolls the viewport straight to the target post in the thread, and illuminates it with a glowing gold highlight pulse (`forum-item-highlight-gold`).
  - Preserves continuous mobile fullscreen without unwanted exit triggers.

### 6. How to Play & FAQ: Comprehensive Forum Features & Tools Guide
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

### 7. Forum: Post Likes & Dislikes Across All Categories
- **Thumbs Up / Thumbs Down Voting**:
  - Interactive Likes (👍) and Dislikes (👎) on thread opening posts and comments across all categories.
  - Backed by SQLite table `forum_reactions` with `(user_id, post_id, comment_id)` constraints and fast indexed lookups.
  - Endpoint `/api/forum/post/<id>/react` supporting instant toggling and switching.
- **Suggestions/Ideas Category Focus**:
  - Emphasized voting in the Suggestions/Ideas header banner to reflect community prioritization.

### 8. Forum: Mobile Layout & Centered Action Toolbar
- **Full-Width Text Below Avatar & Actions**:
  - Restructured post cards so text begins full-width beneath the user's thumbnail avatar and action buttons, eliminating left-indentation on mobile.
- **Centered Action Row**:
  - Structured action buttons (`👍`, `👎`, `🔖 Bookmark`, `↩ Reply`) into a centered full-width row with `flex-wrap: nowrap` and `white-space: nowrap`.

### 9. Forum: Real-Time Viewers Count & Autocomplete Mentions (`@username`)
- **Active Viewers Badge**: Live badge beside Forum title indicating active users viewing forums (e.g., `1 viewing`).
- **Autocomplete Mentions**: Typing `@` opens an interactive popup filtering Friends and active players with arrow/touch selection and virtual keyboard awareness via `window.visualViewport`.

### 10. Profile & Mini-Profile Enhancements
- **Field Character Limits**: Name restricted to 50 chars, Age to 3 chars, Gender to 25 chars.
- **Lifetime Forum Post Count**: Added `💬 Forum Posts: [number]` to profiles and mini-profiles backed by `users.total_forum_posts`, with click-to-search integration.

### 11. Appearance Settings & Game Contrast
- Added **Black** (`#000000`) and three shades of grey (**Light Grey** `#d4d4d4`, **Medium Grey** `#888888`, **Dark Grey** `#4a4a4a`) for Highlight as you Type / Mouse/Swipe with automatic letter contrast inversion.

### 12. Word Definition Popup & Words to Work on: Word Lists Color Coding
- **Adhering to All Words in Game Rooms**:
  - **NWL**: Blue (`#409cff` border, `rgba(64, 156, 255, 0.22)` background, `#93c5fd` text in dark mode; `#1e40af` text, `#dbeafe` background, `#3b82f6` border in light mode).
  - **CSW**: Solid Gold (`#ffd700` border, `rgba(255, 215, 0, 0.22)` background, `#ffd700` text in dark mode; `#713f12` text, `rgba(255, 215, 0, 0.35)` background, `#eab308` border in light mode).
  - **AW**: Royal Purple (`#9333ea` border, `rgba(147, 51, 234, 0.25)` background, `#d8b4fe` text in dark mode; `#3b0764` text, `#f3e8ff` background, `#9333ea` border in light mode).
  - Displayed consistently across definition popups (`#tool-word-def-popover`) adjacent to the word length badge, and across every word row under "Word lists valid in" in the Words to Work on tool.

### 13. Lobby Polish & Registration Validation
- Changed Lobby chatbox placeholder to **`"Type Lobby message..."`**.
- Upgraded 3D obsidian depth on Lobby Journey and New User Guide buttons.
- Restricted username registration to letters, numbers, and underscores with clear inline validation notices.

### 14. Added Words & Lexicographical Engine: Strict Proper Noun & Fandom Filtering (ANGELITE / ANGELITES Fix)
- **Proper Noun & Fandom Filtering (`_AW_DISALLOWED_PROPER_NOUN_REGEX`)**:
  - Implemented strict filtering across `get_word_definitions_for_aw_check`, `lookup_raw_definition_online`, `format_resolved_definition`, and `lookup_web_search_definition` to reject proper nouns, television/film fandom terms (e.g. "fan of the television series..."), eponyms, and media titles.
  - Resolved the singular/plural discrepancy where capitalized Wiktionary titles cataloged fandoms as "Noun", ensuring only authentic common noun lexicographical definitions are accepted.
- **Authoritative Definitions for ANGELITE and ANGELITES**:
  - **ANGELITE**: `A sky-blue or lilac-blue semiprecious variety of anhydrite (anhydrous calcium sulfate), used as an ornamental gemstone.`
  - **ANGELITES**: `plural of angelite (A sky-blue or lilac-blue semiprecious variety of anhydrite (anhydrous calcium sulfate), used as an ornamental gemstone.)`
  - Propagated to `wiktionary_definitions` table in SQLite (`morpheme.db`) and all dictionary files (`Definitions.txt`, `wikdefs.txt`, `wikdefs_duplicate.txt`, `wikdefs_aw_duplicate.txt`).
- **New AW Words & Added Words Restoration**:
  - Added `ANGELITE` and `ANGELITES` to `added_words.txt`, `added_words_duplicate.txt`, and `added_words_dates.txt` with their submission date (`10/10/2026`).
  - Restored them to the top of **New AW Words** in the Tools tab, appearing above previously added words from `30/09/2026`.

### 15. Forum: User Profile Avatar Image Thumbnail Display
- **Issue Resolved**: When users (such as `jeffles`) with an uploaded/selected profile image posted threads or comments in the Forum, the circular avatar space displayed only a fallback text initial (e.g. "J") instead of their chosen profile picture, even though the thumbnail rendered correctly in the mini-profile.
- **Implementation**:
  - Implemented `buildAvatarHtml(username, avatarUrl, className)` in `static/js/forum.js` to inspect `avatar_url`.
  - When `avatar_url` is present, renders a circular thumbnail with `background-image: url(...)`, `background-size: cover`, `background-position: center`, and `flex-shrink: 0`, leaving the inner text empty.
  - When `avatar_url` is absent, cleanly falls back to the uppercase initial letter (e.g., `"J"`).
  - Integrated across:
    - Thread opening post header (`.author-avatar`)
    - Comment replies list (`.comment-avatar`)
    - Desktop reply hover preview cards (`.reply-preview-avatar`)
    - Mobile reply preview dialogs (`.reply-preview-avatar`)
  - Added `.has-avatar` styling in `static/css/forum.css` ensuring crisp cropping and alignment with zero layout shift.

### 16. Forum: Thread Cards Like & Dislike Info Positioned at Bottom Right Corner
- **Layout Enhancement**: In the Forum category thread list (`renderPosts`), like and dislike counts and icons (`👍 [N]` and `👎 [N]`) are now grouped into `.post-stats-right` with `margin-left: auto;`, positioning them firmly at the bottom right corner of each thread card.
- **Left Meta Preserved**: Thread comments count (`💬 [N] comments`) and attachment indicator (`🖼️ Includes images`) remain neatly aligned on the bottom left in `.post-stats-left`.
- **Thread Detail Integrity**: Strictly isolated to thread cards in category/search listings; nothing inside the thread detail view (`renderPostDetail` / thread comments / discussion reaction buttons) was touched or altered.

### 17. Forum: Thread Discussion Header — Refresh Button on Same Row as Bookmark & Scroll to Bottom on Mobile
- **Issue**: On mobile viewports, the comments header (`.comments-header`) wraps into two lines. The Refresh button (`🔄`) previously sat on line 1 beside "Discussion" and the comment count badge, while "Bookmark" and "Scroll to Bottom" occupied line 2.
- **Enhancement**:
  - Moved `#forum-refresh-comments-btn` into `.comments-header-actions-box` in `templates/index.html`.
  - On mobile (`@media (max-width: 820px)` and `body.is-mobile`), `.comments-header-actions-box` forms a single, unified row with `[ 🔄 ]` (`flex: 0 0 32px`), `[ 🔖 Bookmark ]` (`flex: 1 1 0`), and `[ ⬇ Scroll to Bottom ]` (`flex: 1 1 0`).
  - Matches the 27px height and border-radius across all three controls for a balanced, space-efficient mobile header with no tap-target truncation or text wrapping.
  - On desktop (`min-width: 821px`), preserves seamless single-line horizontal alignment alongside "Discussion" and comment count.

### 18. Forum: Category Threads List — 100 Character Content Limit
- **Excerpts Clamped**: In `renderPosts` ([forum.js](file:///Users/jeffbabiak/static/js/forum.js)), thread cards displayed when viewing threads for a Category now slice and limit `post.content` to the first 100 characters (`post.content.slice(0, 100).trimEnd() + '...'`).
- **Full Thread Content Preserved**: The full content text is displayed without restriction when clicking into and viewing the thread (`renderPostDetail`).

### 19. Mobile Tactile Flattening for 'Cancel' and 'Return to Category' Buttons
- **Equalized Tactile Press & Slide**: Gave `#forum-cancel-create` (`← Cancel`) and `#forum-back-to-list` (`← Back to category` / Return to Category) the exact tactile slide and flattening effect previously on the mobile "Back" buttons (`#forum-category-back-btn`, `#tools-mobile-back-btn`, `#mods-mobile-back-btn`, `#settings-mobile-back-btn`).
- **CSS Transformation & Elevation Dynamics**:
  - Configured `transition: all 0.2s ease, transform 0.1s ease !important;` on all forum back buttons.
  - On active / pressed (`:active`, `.active`), applied `transform: translateX(-4px) scale(0.98) !important;` and reduced `box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3) !important;` across dark, light, and white themes.
- **Mobile Touch Animation Window**:
  - In `static/js/forum.js`, added a 120ms animation window on mobile devices for `backToListBtn` and `cancelCreateBtn`, allowing the physical 3D flattening and sliding animation to complete visibly before switching views back to the list.

### 20. Multiplayer Subanagrams Mobile Support, Single-Letter Tap Construction, Backtrack Undo, Enter Submission, Clear Button & Appearance Slider
- **Mobile Access**:
  - Removed mobile User-Agent HTTP 403 rejections in `create_room` and `join_room` in `app.py`.
  - Removed mobile display hiding of `.matrix-subanagrams` in `static/css/lobby.css` and `static/js/app.js` (`filterCubeOnMobile`), allowing full visibility and room creation/joining on mobile.
- **Letter Submission & Board Interaction**:
  - Drag/swipe auto-submission disabled in Subanagrams (`finishDragSelection`, `handleCellMouseOver`, `handleCellMouseMove`, `handleCellTouchMove`).
  - Individual letter press/tap constructs words in `#word-input` (`handleSubanagramsTileClick`).
  - **Backtracking Undo**: Pressing the most recent letter entered removes it from the sequence, unhighlights it, and marks the preceding letter as `current`. Pressing that letter removes it as well, repeating down the stack.
  - **Enter Submission**: Submission occurs via the "Enter" button (`#submit-word-btn` renamed to "Enter" in Subanagrams) or physical Enter key.
  - **Clear Button**: Added `#clear-word-btn` to `.word-input-section` (`templates/index.html`) to wipe the textbox and reset all selected tile states and highlights.
- **Appearance Settings**:
  - Moved `#setting-panel-subanagrams-tile-size` directly below `#setting-panel-cube-scale` in Tools -> Appearance (`templates/index.html`).
  - Removed `desktop-only-setting` class and updated description, enabling tile sizing controls on mobile.
  - Updated `body.is-mobile .game-board.is-subanagrams-board .board-cell` in `static/css/play.css` to use `var(--subanagrams-cell-size, 64px)` dynamically.

### 21. Multiplayer Subanagrams Lobby Layout, Restricting Enter Button to Subanagrams, Mobile Textbox & Instant Loading Feedback
- **Subanagrams Lobby Button Horizontal Stretch on Mobile**:
  - In `static/css/lobby.css`, added a targeted mobile override for `.matrix-subanagrams .matrix-header` and `.matrix-subanagrams .matrix-row` (`grid-template-columns: 45px 1fr !important;`).
  - The multiplayer Subanagrams **Start** button and its header row now expand across the entire available screen width next to the 45px "2m" indicator instead of being restricted to a quarter-width column.
- **Lobby Header Title & Column Header Update**:
  - In `templates/index.html`, updated the Lobby title from **`Subanagrams (Practice)`** to **`Subanagrams`** across all platforms. Synchronized the Lobby Guide modal entry to reflect **`Subanagrams`**.
  - Updated the column header text directly above the Start button from **`Practice`** to **`Practice / Warmup`**.
- **Restricting Enter Button Strictly to Multiplayer Subanagrams**:
  - In `static/css/play.css`, hidden `#submit-word-btn` by default across all standard game rooms (4x4, 5x5, 6x8, 3D, tournament, private, 24h) on desktop, laptop, mobile, and tablet (`display: none !important;`).
  - Restored `body.is-desktop .word-input-section` max-width back to `500px !important;` for standard rooms, keeping the word input box, Rotate, and Transpose buttons centered with original spacing.
  - In multiplayer Subanagrams (`body.is-subanagrams`), displayed the emerald/green **Enter** button (`#submit-word-btn`) and red **Clear** button (`#clear-word-btn`) with `display: inline-flex !important;` on both desktop and mobile.
- **Mobile Subanagrams Input Textbox, Clear, and Enter Buttons Restored**:
  - Overrode the mobile hiding rule inside `@media (max-width: 992px)` for Subanagrams (`body.is-subanagrams .word-input-section`), displaying `#word-input` with centered uppercase letters (`font-size: 1.1rem; letter-spacing: 2px;`) to display letters in real time as tiles are tapped on mobile devices, with the Enter and Clear buttons alongside.
- **Instant Synchronous Loading Notice on Button Press**:
  - In `templates/index.html` and `static/js/lobby.js`, eliminated the 2–3 second delay when entering a room from the lobby.
  - `window.showLoadingOverlay('Loading...')` is now invoked **synchronously on the first line** of click handlers before any network requests (`checkAccountTimeoutAndAlert`), with forced DOM reflow (`void overlay.offsetHeight;`) ensuring the loading spinner and message render in the exact frame the button is clicked.
  - Avoided dropping clicks during the gateway transition cooldown (`_lobbyEnterCooldown`), immediately showing the loading overlay and proceeding seamlessly into the room.

---

## Verification & Health Check

- **Automated Health Check**: `curl -I https://morpheme.games/` returns `HTTP/1.1 200 OK` / `HTTP/2 200`.
- **Static Assets Sync**: `lobby.css`, `play.css`, `app.js`, and `play.js` verified live on production.
- **PM2 Process**: `morpheme` process online with dictionaries fully loaded.
- **Mobile Fullscreen Rules**: Fullscreen invariant strictly maintained across all navigation, modals, and tabs.

