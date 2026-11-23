# Morpheme Stable State Summary – November 23, 2026

**Synchronization Checkpoint & Stable Anchor**  
- **Canonical Git Commit ID**: `3f723a73204416fdd671cfe57512eeaa3c73f818` (`3f723a73`)
- **Git Branch**: `main` (synchronized: local `main` == `origin/main` == VPS `main`)
- **Production Server**: `132.148.72.249` ([morpheme.games](https://morpheme.games)) – Live (HTTP 200 OK)
- **Localhost Development**: Port `5001` – Running & healthy (Tally Hardened version)
- **Date/Time**: November 23, 2026 (18:35 UTC / 13:35 EST)

---

## 1. Key Achievements & Updates in this Stable Point

### A. Universal Word Definition Popup Auto-Repositioning (No Bottom Cut-Off)
- **Cross-Platform Viewport Edge Detection**:
  - Whenever any word is clicked/selected throughout the entire app (Tools tabs: Lists, Words User is Good at, Combo Checker, Is Valid, Sequence, Subanagrams, Unscramble, Reverse Dictionary, Word Chart, Words to Work on, Find Count; Game rooms: Notepad, Round Replay; Mods word lists, etc.) and the definition popup (`#tool-word-def-popover`) appears:
  - If the bottom of the popover extends past the visible viewport bottom (`rect.bottom > viewportHeight - 12`), the system immediately and smoothly shifts `top` higher so the entire window is 100% visible with a clean 12px safety margin.
- **Dynamic Asynchronous Height Handling**:
  - Attached a `ResizeObserver` on `#tool-word-def-popover` that automatically detects when content expands (such as when multi-line definitions, division badges, inflections, Wikipedia links, and images load asynchronously from `/api/tools/validate`).
  - Added direct image `load` and `error` event handlers inside `renderDefData(data)` to re-calculate popover positioning as soon as definition images finish rendering.
  - Bound window `resize` and `orientationchange` listeners so orientation changes on mobile and window resizes on desktop/laptop keep the popover fully in view.
- **Top Edge & Small Screen Fallback (Constrained Max-Height)**:
  - Ensured shifting `top` higher never pushes the popover off the top of the screen (`top >= 12px`).
  - If the viewport height is very small (such as mobile landscape orientation), capped `maxHeight` to `viewportHeight - 24px` and enabled smooth touch scrolling (`-webkit-overflow-scrolling: touch`) on `.tool-def-content`, ensuring header titles and footer action buttons (CC, FC, ADD) remain visible and accessible.
  - Reset inline `maxHeight` and scroll states upon popover close so subsequent lookups measure naturally.
- **Cache Buster**:
  - Bumped `tools.js?v=1795370000` to `v=1795380000` in `templates/index.html`.

---

### B. Empty Default Player in "Words User is Good at" (No `jeffb` Default)
- **Blank Starting State**:
  - The "Player" textbox (`#good-at-user-input`) now starts off completely blank with no initial text.
  - Eliminated the backend fallback to `jeffb` or top recorded player when no username is provided: `/api/tools/words-good-at` returns an empty successful dataset (`username: ''`, `words: []`, `total_count: 0`) when no username query parameter is present.
  - Frontend prompts the user with: `"Enter a player's username above or select from your friends list to view the words they are good at."`
  - Neutral placeholder updated to `placeholder="Player..."` (removing `"e.g. jeffb"`).
  - Preserved full search and friend selection functionality: typing a username or choosing a friend from the popup immediately fetches and renders that player's mastered words.

### C. Unified Blue "Update" and Primary Action Buttons Across All Themes & Layouts
- **All Layout Colors & Themes (Light & Dark)**:
  - Standardized the "Update" button in **Words User is Good at** (`#good-at-search-btn`) and **Lists** (`#list-update-btn`) to use the vibrant blue action button styling (`linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important; color: #ffffff !important; border: none !important; font-weight: 700 !important;`) matching the other Tools tabs across all layout colors (default dark, black, gray, dark-gray, dark-red, dark-blue, white, light-gray, light-light-gray, light-red, light-orange, light-green, light-blue, etc.).
  - Fixed mobile selector conflict in `play.css` where `.combo-control-panel .input-row > button` had previously been bundled with inputs and assigned `background: var(--input-bg)` (causing light theme Update buttons on mobile to render light gray/white instead of blue).
  - Explicitly unified hover (`linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)`) and active states across desktop, laptop, and mobile viewports.
  - Extended unified blue styling across all tool action buttons (`#combo-search-btn`, `#valid-check-btn`, `#seq-search-btn`, `#find-count-btn`, `#random-gen-btn`, `#unscramble-gen-btn`, `#wc-word-search-btn`) in light-colored layouts.

---

### D. Friends Dropdown Popup in "Words User is Good at"
- **Interactive Friends Popup Aligned with With Friends Invitations**:
  - Implemented an autocomplete dropdown for the "Player" input (`#good-at-user-input`) in **Tools $\to$ Words User is Good at**, mimicking the exact visual style and behavior of the **With Friends** game room invitation dropdown.
  - **Popup Structure & Content**:
    - **Header**: Shows `FRIENDS (<count>)` count matching `/api/friends/list`.
    - **Items**: Each friend row features a colored circular avatar with their initial letter, optional country flag emoji (`🇨🇦`), friend username (with typed prefix matching highlighted in bold), and an online/offline status indicator dot.
    - **Empty & Non-Logged In States**: Displays clear messages if no friends match the typed filter or if the user is not currently logged in.
  - **Prefix Filtering & Keyboard/Touch Navigation**:
    - Typing into the Player textbox filters the friends list by username prefix in real-time (`startsWith`).
    - Full keyboard accessibility: ArrowDown/ArrowUp moves selection, Enter selects friend and automatically loads their mastered words tool data, Escape hides the popup, and Tab selects or closes.
    - Seamless touch support: tapping on mobile or desktop item selects the friend, updates the textbox value, hides the popup, and automatically invokes `window.loadWordsGoodAtTool(friend.username, true)`.
    - Outside clicks or tool switching automatically closes the dropdown.
  - **Stacking Context & Overflow Resolution**:
    - Wrapped `#good-at-user-input` in `.good-at-user-input-wrap` (`position: relative; display: flex; flex-direction: column;`).
    - Fixed `.combo-control-panel` stacking bug by explicitly overriding `overflow: visible !important; z-index: 100 !important;` so the dropdown floats cleanly above the summary banner and results table on desktop, laptop, and mobile viewports.
    - Replaced `scrollIntoView()` with inner `dropdown.scrollTop` calculations so ancestor containers are never unintentionally shifted.

---

### E. Words User is Good at (Tools Tab) Mobile & Desktop Overhaul
- **Control Organization Aligned with Lists**:
  - Restructured the controls in `#tool-good-at` into an `.input-row` identical in organization to `#tool-lists`.
  - **Mobile 2×2 Grid + Full-Width Button**:
    - **Row 1**: Player textbox (`#good-at-user-input`) & Word Filter textbox (`#good-at-word-filter`)
    - **Row 2**: Length dropdown (`#good-at-length-select`) & Sort dropdown (`#good-at-sort-select`)
    - **Row 3**: Full-width **Update** button spanning both columns (`grid-column: 1 / -1`)
  - **Desktop Single Horizontal Row**: Aligned inputs, dropdowns, and Update button in a clean horizontal flex line matching Lists.
- **Un-Crammed Horizontal Touch Scrolling on Mobile**:
  - Eliminated cramped column percentages and abbreviated cell values.
  - Set comfortable table dimensions (`min-width: 540px`) with natural padding (`8px 12px` to `10px 14px`).
  - Rendered full readable values: Word text with definition popup triggers, length indicator (`3L`), bold find counts, full division badges (`Division 1 (1 pt)`, `Division 2 (3 pts)`, `Division 5 (9 pts)`), and full dates (`Nov 23, 2026`).
  - Enabled smooth horizontal swiping on mobile viewports via `-webkit-overflow-scrolling: touch` with subtle native scroll indicators.
- **Universal Internal Vertical Scrolling with Sticky Headers**:
  - **Mobile**: Configured `#good-at-results-wrapper` to fill all remaining vertical viewport space (`flex: 1 1 0px !important; min-height: 280px !important; height: 100% !important; overflow-y: auto !important;`). Sticky header (`position: sticky; top: 0; z-index: 10`) keeps column titles anchored while words scroll beneath.
  - **Desktop**: Isolated vertical flex sizing (`min-height: 380px; height: 100%`) with internal vertical scroll and custom thin scrollbars, preventing outer page displacement.
- **Removed Featured Players & Compensated Table Height**:
  - Removed the `FEATURED PLAYERS` container (`#good-at-top-players`) and chips from `#tool-good-at`.
  - Reallocated the vertical space directly to the results table: table vertical height expanded from 299px to 334px on mobile and from 297px to 380px (+28%) on desktop, showing significantly more word rows without scrolling.
- **Centered Pagination & Full-Width Stretched Buttons Across All Platforms**:
  - Positioned "Page 1 of 3 (122 words)" indicator text at the exact horizontal center directly above the Previous and Next buttons across mobile, laptop, and desktop viewports.
  - Hardened page indicator contrast and visibility: styled with `color: var(--text-primary) !important`, bold `font-weight: 700`, `opacity: 1`, and active JS rendering across single-page and multi-page results.
  - Placed Previous and Next buttons directly beneath the page indicator, stretching across the entire horizontal length of the container (`flex: 1 1 0px` each), with clean disabled states and hover transitions.
  - Added explicit `@media (min-width: 901px)` desktop/laptop styling (`height: 38px`, `font-size: 0.88rem`, `padding: 8px 16px`) alongside mobile rules (`height: 32px`, `font-size: 0.8rem`).
  - Equalized the bottom padding/spacing of the results table on desktops and laptops to 14px to match the 14px left padding.
- **Granular Word Length Filter (3 through 15 Letters & 16+ Letters)**:
  - Expanded the Length dropdown in "Words User is Good at" to include discrete options: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 Letters, and 16+ Letters.
  - Updated dropdown option label to explicitly read `16+ Letters` (matching the length dropdown style).
  - Backend endpoint `/api/tools/words-good-at` performs exact length matching for lengths 3-15 and `>= 16` for `16+`.
- **Streamlined 3-Panel Summary Banner**:
  - Removed the redundant "PLAYER" panel to the left of "WORDS" (the searched player name is already clearly visible in the input box).
  - Stretched the remaining 3 panels horizontally across the full container (`repeat(3, 1fr)`): Mastered Words (`#good-at-stat-unique`), Total Finds (`#good-at-stat-finds`), and Rarest Mastered (`#good-at-stat-rarest`).
  - Balanced padding and gap spacing across both mobile (6px gap) and desktop (12px gap) layouts.
- **Description & Featured Players Spacing**:
  - Cleaned up top player chip pills with dynamic toggle (hidden when empty).
  - Adjusted header description text padding and max-height to ensure text remains fully legible without clipping descenders.

---

### F. Moderation Policy, Hierarchy & Permanent Ban Protocol
- **Permanent Ban Enforcement & Removal of "Lift Ban"**:
  - Fully eliminated the "Lift Ban" button and associated frontend/backend handlers from **Mods $\to$ Ban/Timeout User**.
  - Enforced a strictly permanent ban model: when a user account is banned, the record is permanently eradicated, active sessions are terminated, and their IP address is blacklisted.
- **Active Room Ban Chatbox Notice**:
  - If a banned user is currently active in a game room (as a player or spectator), the system immediately broadcasts a system message to the room's chatbox (`<username> has been banned from all rooms.`), matching the exact behavior and presentation format used for Timeouts, before authoritatively evicting the user.
- **Hardened Root Administrator Protection (`jeffb`)**:
  - Client-side and server-side hard checks prevent any moderator from timing out, banning, or IP-banning user `jeffb`.
  - Rejection with explicit authorization error (`403 Forbidden: Protected user cannot be timed out or banned`) if attempted.
- **Moderator Hierarchy Rules**:
  - Only root administrator `jeffb` possesses the privilege to timeout or ban fellow moderators.
  - Standard moderators are prevented from taking moderation actions against other moderators.
- **Moderator Record & Privilege Purging**:
  - If a moderator account is banned, all associated moderator permissions, credentials, and cache entries are completely purged from the system.

---

## 2. Infrastructure & System Status

| Component | Status | Details |
|---|---|---|
| **Production VPS** | `ONLINE` | `132.148.72.249` ([morpheme.games](https://morpheme.games)), Ubuntu 24.04 LTS, PM2 Process ID 0 |
| **Localhost** | `ONLINE` | Port `5001`, Python 3 Flask / SocketIO daemon (Tally Hardened) |
| **Git Canonical Branch** | `SYNCHRONIZED` | `origin/main` == `HEAD` (`f29e01e3`) |
| **Asset Cache Buster** | `1795380000` | Applied across `play.css` and `tools.js` in `templates/index.html` |
| **RAM Utilization** | `OPTIMAL` | SQLite definitions proxy + LRU cache maintaining low ~1.1 GB memory profile |

---

## 3. Verification & Test Evidence
- **Mobile Emulation (390×844 iPhone Viewport)**:
  - Verified 2×2 control grid + full-width Update button matches Lists.
  - Verified horizontal swiping (`scrollLeft`) reveals all columns comfortably without clipping.
  - Verified vertical scrolling (`scrollTop`) with sticky headers anchored at table top.
  - Verified removal of Featured Players container.
  - Verified centered "Page 1 of 3 (122 words)" with horizontal full-width buttons below.
- **Laptop & Desktop (1280×800 & 1440×900 Viewports)**:
  - Verified horizontal input controls row.
  - Verified centered page indicator and full-width pagination buttons stretching across the horizontal container (`899px` on laptop, `1049px` on desktop).
  - Verified matching 14px bottom padding/margin on table to equal left padding.
  - Verified discrete length filters (8, 9, 10, 11, 12, 13, 14, 15, and 16+ Letters).
  - Verified vertical table scrolling within `#good-at-results-wrapper` with sticky headers.
  - Verified zero outer page layout displacement.
- **Live Production HTTP**:
  - `curl -sI https://morpheme.games/` confirmed HTTP/2 200 OK serving cache buster `v=1795380000`.
  - PM2 logs confirmed clean process restart with zero errors.
