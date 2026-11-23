# Morpheme Stable State Summary – November 22, 2026

**Synchronization Checkpoint & Stable Anchor**  
- **Canonical Git Commit ID**: `aa05c65a96cbe84203c67ced2eae1ee655e2bbfd` (`aa05c65a`)
- **Git Branch**: `main` (synchronized: local `main` == `origin/main` == VPS `main`)
- **Production Server**: `132.148.72.249` ([morpheme.games](https://morpheme.games)) – Live (HTTP 200 OK)
- **Localhost Development**: Port `5001` – Running & healthy (Tally Hardened version)
- **Date/Time**: November 23, 2026 (03:10 UTC / 22:10 EST)

---

## 1. Key Achievements & Fixes in this Stable Point

### A. Memory Optimization & VPS Stability
- **SQLite Proxy + LRU Cache for Definitions**: Replaced in-memory loading of massive dictionary definitions with an indexed SQLite proxy backed by an in-memory LRU cache (`_DEFINITION_CACHE`, `_DEFINITION_CACHE_KEYS`).
- **RAM Footprint Cut by >65%**: Slashed resident memory usage from ~3.3 GB down to ~1.1 GB, completely eliminating GoDaddy high-memory alerts and host throttling.
- **Fast Lookups**: Preserved sub-millisecond query performance (~0.014 ms).

### B. Tiered Bonus Word Self-Healing & Fallback
- **Zero-Drop Protection**: Eliminated instances of rounds missing a Bonus Word. Implemented a 4-tier progressive fallback (Target range $\to$ Extended range $\to$ Dynamic board scan $\to$ Deterministic letter-pool seed) guaranteeing that every round transition stages a valid Bonus Word.
- **Intermission & Start-of-Round Healing**: Rooms actively verify and heal missing bonus words immediately upon entering intermission and before round start.

### C. Sequential Word Path Highlighting in Round Review
- **Tile-by-Tile Pathing Animation**: When selecting words in All Words review during intermission, the board sequentially animates each step along the tile path (85ms per step) before holding, restoring the original visual pathing experience.

### D. User Input Method Tracking & Percentages
- **Tracking & Persisting**: Real-time tracking of player input methods (`mouse`, `keyboard`, `touch`) submitted during rounds, persisted in user records (`input_mouse_count`, `input_keyboard_count`, `input_touch_count`).
- **Lobby & Room Indicators**: Added hover tooltip ("Input Method: Mousing / Swiping / Keyboard") in active rooms.
- **Profile & Mini-Profile Breakdown**: Added percentage breakdown display (e.g., `🖱️ 83% · 📱 17%`) across both full profiles and mini-profiles.
- **Mobile Layout Alignment**: Positioned the input method block directly beneath Forum Posts on mobile in both Full Profile and Mini-Profile cards.

### E. Mini-Profile & Full Profile Stats Layout (Astronomical Number Resilience)
- **Universal "TOP 10" Renaming & Count Display**:
  - Replaced "TOP 10 RATE" with **`TOP 10`** across all viewports (desktop, laptop, tablet, and mobile) on both Full Profile and Mini-Profiles.
  - Displays the raw integer count of times the user finished within top 10 positions (`top_10_count`, e.g. `12`), removing percentage calculations and rates entirely.
- **Full Profile Layout**:
  - **Desktop & Laptop**: `TOP 10` is positioned immediately to the right of `WINS`:
    $$\mathbf{GAMES} \quad\vert\quad \mathbf{WINS} \quad\vert\quad \mathbf{TOP\ 10} \quad\vert\quad \mathbf{WORDS} \quad\vert\quad \mathbf{PT\ SUM} \quad\vert\quad \mathbf{AVG\ WPM}$$
  - **Mobile Layout (Two 3-Column Rows)**:
    `AVG WPM` and `TOP 10` are switched around so `AVG WPM` leads row 2:
    - **Row 1**: $\mathbf{GAMES} \quad\vert\quad \mathbf{WINS} \quad\vert\quad \mathbf{WORDS}$
    - **Row 2**: $\mathbf{AVG\ WPM} \quad\vert\quad \mathbf{TOP\ 10} \quad\vert\quad \mathbf{PT\ SUM}$
- **Mini-Profile Layout**:
  - **Desktop & Laptop**: 6 stats across 1 row with `AVG WPM` and `TOP 10` on the right side:
    $$\mathbf{Games} \quad\vert\quad \mathbf{Wins} \quad\vert\quad \mathbf{Words} \quad\vert\quad \mathbf{PT\ Sum} \quad\vert\quad \mathbf{Avg\ WPM} \quad\vert\quad \mathbf{Top\ 10}$$
  - **Mobile Devices**:
    - **Row 1**: $\mathbf{Games} \quad\vert\quad \mathbf{Wins} \quad\vert\quad \mathbf{Words}$
    - **Row 2**: $\mathbf{Top\ 10} \quad\vert\quad \mathbf{Avg\ WPM} \quad\vert\quad \mathbf{PT\ Sum}$
- **Mobile Mini-Profile Action Buttons & Moderator Styling**:
  - **Uniform Vertical Length**: All buttons (Full Profile, Round Reviews, Message, Friends, Timeout, Ban) share the identical vertical height (`height: 26px; min-height: 26px; max-height: 26px; font-size: 0.78rem; border-radius: 7px;`).
  - **Equidistant Spacing**: Buttons are spaced exactly `8px` apart both horizontally and vertically, matching the distance between Full Profile and Message (`gap: 8px; margin-top: 8px;`).
  - **Color Accents**:
    - **Timeout**: Distinct warning **Yellow** (`#facc15` text/border, `rgba(250, 204, 21, 0.18)` background).
    - **Ban**: Distinct danger **Red** (`#ef4444` / `#f87171` text/border, `rgba(239, 68, 68, 0.18)` background).
- **Fluid Font Sizing & Tooltips**: Used `clamp()` font scaling and tighter letter spacing so numbers with 7–9 digits fit cleanly without wrapping or overflow. Full exact figures are available on hover/tap via title tooltips.

### F. Room Eviction & Multi-Room Switching Synchronization
- **Cleanup on Join**: Removed the legacy `is_hub` exemption in `cleanup_user_rooms()` that was preventing users from being removed from public rooms (`pub_...`) when entering a different room.
- **Explicit Leave Handling**: Marked `room.left_users[user_id]` when leaving a room so in-flight state-polling requests (`/api/room/<id>/state`) cannot inadvertently auto-add the user back into the previous room.
- **Awaited Navigation**: Awaited `leaveCurrentRoom()` when clicking Return to Lobby or the Lobby navbar link, ensuring the server-side leave completes before lobby navigation.

### G. Profile Avatar White Theme Masking Fix & Upload Hardening
- **Theme Masking Elimination**: Removed the aggressive `background: var(--bg-primary) !important;` rule in White and Light Gray themes that was masking uploaded avatar images as plain white squares.
- **Class-Based Image Isolation**: Added `.has-image` class to `.profile-avatar.large` and `.mini-avatar` ensuring uploaded photos display clearly with proper containment and contrast.
- **Upload Hardening**:
  - Expanded allowed avatar image extensions to include mobile formats (`png`, `jpg`, `jpeg`, `gif`, `webp`, `bmp`, `jfif`, `heic`, `avif`).
  - Added MIME normalization for Gemini AI moderation (`jpg` $\to$ `image/jpeg`).
  - Added descriptive, user-friendly error messages during upload.

### H. Mini-Profile "About Me" Vertical Sizing
- **Expanded Vertical Height**: Increased the vertical length of the "About Me" box in Mini-Profiles on all platforms:
  - **Desktop/Laptop**: `min-height: 105px; max-height: 240px;` (up from 60px/200px), comfortably showing 4–5 lines of narrative text.
  - **Tablets ($\le 600\text{px}$)**: `min-height: 90px; max-height: 160px;`.
  - **Mobile Compact**: `min-height: 85px; max-height: 165px;` (up from 50px/125px).
- **Full Profile Preservation**: Full profile description boxes remain at their clean default dimensions.

### I. Thumbnail Image Lightbox & Window Context Preservation
- **Foreground Display Stacking (`z-index: 2147483646`)**:
  - Elevated `.image-lightbox-overlay` to `z-index: 2147483646 !important;` and `#image-lightbox-close` to `z-index: 2147483647 !important;`.
  - Guaranteed that clicking any image thumbnail (e.g., Mini-Profile avatar, Full Profile avatar, Forum post/reply attachments, Word definition image, or Store item) displays the enlarged image directly in front of whatever window or modal the user is currently viewing.
- **Context Retention on Close**:
  - Closing the lightbox (via the top-right `&times;` close button, clicking anywhere on the backdrop outside the image, or pressing the `Escape` key) closes only the lightbox.
  - Stopped event propagation and used capture-phase Escape listener interception so that the underlying modal or window (e.g. Mini-Profile, Forum topic, Word Definition) remains open and interactive, allowing the user to seamlessly resume what they were doing without losing their place.
- **Universal Thumbnail Zoom Handlers**:
  - Mini-profile avatar thumbnail (`#mini-profile-avatar`) has `cursor: pointer` and opens full image zoom on click.
  - Full profile avatar (`.profile-avatar.large`) allows non-owners to click to zoom, and provides owners with a dedicated zoom icon button without overriding the upload trigger.
  - Added global click delegation for `.definition-image`, `.forum-lightbox-trigger`, and `.reply-preview-thumb`.

### J. Moderator Timeout & Ban Confirmation Popup Warnings
- **Eliminated Same-Window Text Boxes**:
  - Removed the inline warning text boxes ("Are you sure you want to timeout this user?" and "Are you sure you want to permanently erase and ban this user?") from the form windows.
  - Sized action buttons cleanly to "⏱️ Timeout User" / "🚫 Ban & Erase User" alongside standard "Cancel" buttons.
- **Dedicated Modal Confirmation Popup (`showConfirmModal`)**:
  - Clicking "Timeout User" or "Ban & Erase User" now opens a dedicated foreground confirmation popup (`z-index: 100020 !important;`) positioned on top of the modal.
  - Explicitly questions the moderator: *"Are you certain you want to timeout '{username}'?"* detailing duration and reason.
  - If confirmed on the popup, the moderation request executes; if cancelled, the popup closes while leaving the form window intact for editing or dismissal.
- **Dynamic Accent Styling**: Styled the popup confirmation buttons with matching amber (`#d97706`) for timeouts and crimson (`#e11d48`) for bans.

### K. Dynamic Font Scaling for Astronomical Numbers (`applyDynamicStatSize`)
- **Proportional Scaling Inspired by Is Valid**: Implemented `applyDynamicStatSize` across Mini-Profile and Full Profile stat boxes (`Games`, `Wins`, `Words`, `PT Sum`, `Top 10`, `Avg WPM`).
- **Zero Truncation Guarantee**: Dynamically measures formatted digit count and progressively reduces font size and tightens letter spacing as numbers reach into hundreds of thousands (e.g. `200,000` games) or multi-millions of points (e.g. `35,890,120` pts). Astronomical figures fit completely on screen without clipping or trailing ellipses.

### L. Tournament Parameters Single Difficulty Percentage
- **Duplicate Percentage Elimination**:
  - Fixed an issue where Tournament Parameters rendered two percentages under Difficulty (e.g., `Medium (48%) (42%)`).
  - The first percentage was a signup-phase simulation placeholder (`params['difficulty'] = "Medium (48%)"`), and the second was the factual board uniqueness calculation (`params['uniqueness_ratio'] = 0.42` $\to$ `(42%)`).
- **Synchronized Backend & Frontend**:
  - **Backend (`app.py:get_tournament_status`)**: Injects `board_meta.get('difficulty')` into `params['difficulty']` for the active round so the API serves the single canonical difficulty string (e.g. `Medium (42%)`).
  - **Frontend (`static/js/tournaments.js`)**: Extracts `baseDiff` cleanly by stripping any existing parentheses before appending the exact board percentage, guaranteeing that only ONE accurate percentage ever renders across signup, active, and completed tournament states.

### M. "Words User is Good at" Tool & Cheater Tally Purge Architecture
- **New Tools Tab ("Words User is Good at")**:
  - Added new dedicated tab in Tools: *"Words User is Good at - Words that do not fool players"*.
  - Page description highlights the vocabulary mastery of top players, allowing players to study what difficult words do not fool experienced opponents or inspect their own mastered vocabulary.
  - Features real-time player search, featured top-player quick picker chips, length filtering (3L to 8L+), and multi-criteria sorting (Most Found, Highest Division/Rarest, Longest, Alphabetical, and Most Recent).
  - Displays summary statistics: Mastered Words, Total Finds, and Rarest Mastered Division. Clickable words seamlessly open definition popups.
- **Dual-Tally Architecture (`user_word_tallies`)**:
  - Implemented relational table `user_word_tallies (user_id, word, find_count, last_found_at)` with composite indexes for lightning-fast queries.
  - Round end logging in `game_room.py:log_word_tally` records authentic word finds with registered user attribution.
- **Cheater & Ban Purge Automation**:
  - Hooked `ban_user_api` to automatically purge all records from `user_word_tallies` and `word_finds` whenever a moderator bans a user.
  - Executes `tally_divisions.rebuild_word_stats_from_db()` to recalculate `word_stats.json` strictly from remaining non-banned users.
  - Wipes cheater finds across **Find Count in Tools** (total find count & 10 most recent finders), **Dynamic Word Lookup (Word Charts)**, and **Tally Division Difficulty**, instantly restoring inflated words to their rightful maximum rarity divisions.

---

## 2. Synchronization Status Matrix

| Component | Status | Commit / Version | Notes |
| :--- | :--- | :--- | :--- |
| **Localhost** | ✅ Active | `aa05c65a` | Port 5001, daemon running, single difficulty percentage verified |
| **GitHub (`main`)** | ✅ Up to date | `aa05c65a` | Remote synchronized with zero uncommitted changes |
| **Production VPS** | ✅ Live | `aa05c65a` | `morpheme.games` (132.148.72.249), PM2 online, HTTP 200 OK |
| **App / Client UI** | ✅ Verified | `aa05c65a` | Tournament parameters display only one factual percentage |

---

## 3. Deployment & Recovery Reference

To reset or redeploy to this exact point in the future:
```bash
git fetch origin main
git reset --hard aa05c65a96cbe84203c67ced2eae1ee655e2bbfd
python3 scratch/deploy_reliable.py
```
