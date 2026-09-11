# Morpheme Stable State Summary - September 11, 2026

This summary documents the stable state of the Morpheme application as of September 11, 2026. Complete synchronization across Localhost, GitHub `origin/main`, and `morpheme.games` has been completed and verified under Commit ID **`a455e178`** (and its tagged stable point).

---

## 🚀 Key Improvements & Features (September 11, 2026)

### 1. "New Users" Tab in Tools
* **The Mandate**: Create a new tab below Personal Timer in Tools called "New Users" displaying country flags and registration dates, ordered with the newest members at the top, along with total registered users and weekly registrations.
* **Implementation**:
  * **Backend API (`/api/tools/new-users`)**: Queries non-guest registered accounts (`username NOT LIKE 'Guest_%'`), calculating total users and rolling 7-day registrations (`created_at >= datetime('now', '-7 days')`), ordered with newest accounts at the top (`ORDER BY (created_at IS NULL), created_at DESC, id DESC`).
  * **Tools Navigation & UI**: Added `data-tool="new-users"` directly below Personal Timer in `.tools-sidebar` and created `#tool-new-users` pane in `templates/index.html`.
  * **Stat Cards & Table**: Included responsive `.stat-box` cards for "Registrations This Week" and "Total Users", plus a scrollable table with country flags (via `window.getFlagHtml`), localized registration timestamps (via `window.formatAppDate`), and click handlers to view mini-profiles.

### 2. FAQ Dictionary Descriptions & 16+ Words Note Removal
* **The Mandate**: Remove the `* Note on 16+ Letter Words` paragraph under the dictionary breakdown table in FAQ, and update NWL and CSW descriptions to state they cap word lengths at 15 letters respectively.
* **Implementation**:
  * Removed the footnote paragraph from `#faq-dictionaries` in `templates/index.html`.
  * Updated NWL description: explicitly notes that NWL caps word lengths at 15 letters.
  * Updated CSW description: explicitly notes that CSW caps word lengths at 15 letters.

### 3. Suggestions Category Description Voting Notice
* **The Mandate**: In Forum (specifically in the Suggestions category header, not in the side navigation menu), mention that user agreements count as votes, disagreements count as votes, and moderator decisions are based on popularity.
* **Implementation**:
  * **`app.py` (`get_forum_categories`)**: Injected `header_description` for the Suggestions / Suggestions/Ideas category containing the full notice, while preserving `d['description']` as the concise string for the side navigation menu.
  * **`static/js/forum.js` (`selectCategory`)**: Updated `#forum-category-desc` to display the expanded voting notice in the main category view while the side menu items remain clean and compact.

### 4. User Current Time Display next to Timezone
* **The Mandate**: Display the user's current local time next to their timezone on Profile and on their mini-profile directly below Last Visited.
* **Implementation**:
  * Added `getUserCurrentTimeString(tz)` in `static/js/tools.js` using `Intl.DateTimeFormat`.
  * Updated `#profile-timezone-val` (for searched profiles) and `#profile-timezone-owner-time` (for own profile) to show live local time.
  * Added `#mini-profile-timezone` in `templates/index.html` displaying the player's timezone and current local time directly under Last Visited on mini-profiles.

### 5. Profile Metadata Layout & Equal Row Spacing
* **The Mandate**: Rearrange Profile metadata so "Registered" and "Last Visited" are side-by-side, "Timezone" and "Proof" are on the row below them, expand "About Me" height, reduce excess top padding, and equalize vertical row spacing between Name, Registered, and Timezone.
* **Implementation**:
  * Grouped profile metadata items in `templates/index.html` into three clean semantic flex rows (`.profile-meta-line`):
    * Row 1: Name, Age, Gender, Location
    * Row 2: Registered, Last Visited
    * Row 3: Timezone, Proof
  * Configured `.profile-metadata-row` as flex column with a constant `gap: 12px` (desktop) and `gap: 8px` (mobile), guaranteeing 100% mathematically equal spacing between all rows.
  * Expanded `.description-text` `min-height` to 110px and reduced padding above "About Me".
  * Equalized spacing between flag and country name on mini-profiles via `.mini-meta-icon`.

### 6. Guest Session Data Purge & Auth Hardening
* **The Mandate**: Automatically purge guest user data on logout and prevent collision with existing numbers.
* **Implementation**:
  * Added `purge_guest_user(username)` in `app.py` to scrub guest records, ratings, settings, and presences from SQLite upon logout.
  * Guaranteed unique random allocation for guest suffixes.

---

## 🛠 Active Features & Configuration
* **Board Formats**: Normal (72%), Checkerboard (12%), Double (1%), Triple (1%), Valued Letters, Rotation, Penalty, Mania, Either/Or, Bonus Word, and Density.
* **Grid Dimensions**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
* **Dictionaries**: NWL (American), CSW (International), Added Words (AW), and 16+ Supplementary List.
* **Lexicographical Invariants**: Plural and suffix propagation preserved across dictionary entries.
* **Mobile Invariants**: Fullscreen exiting invariants preserved on Tools, Settings, Profile, Forum, and modal dialogs to prevent Android keyboard display rebuild black screens.

---

**Localhost Status**: Synchronized  
**GitHub Repository (`origin/main`)**: Synchronized  
**Production Server (`morpheme.games`)**: Green / PM2 Online / Synchronized  
