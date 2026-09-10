# Stable State Summary – September 10, 2026

## Latest Commit Information
- **Stable Save Point**: September 10, 2026
- **Active Git Tags**:
  - `START_OVER_POINT_SEPTEMBER_10`
  - `stable-2026-09-10`
  - `START_OVER_POINT`
  - `save-point-latest`
  - `start-over`
  - *(Historic reference preserved: `START_OVER_POINT_SEPTEMBER_8`, `stable-2026-09-08`)*

---

## Synchronization Status
- **Localhost (`/Users/jeffbabiak`)**: Synchronized
- **GitHub (`origin/main` & Tags)**: Synchronized
- **Production Server (`132.148.72.249` / `morpheme.games`)**: Synchronized, PM2 online, HTTP 200 OK
- **Flutter Mobile App (`morpheme_word_game`)**: Synchronized (loads `https://morpheme.games/` with native audio bridge)

---

## Features & Stable Specifications (September 10 Checkpoint)

### 1. Complete Enrichment of Spaced Definitions in Added Words (AW)
- **Authentic Wiktionary Definition Resolution**: Scanned the complete Added Words list for words defined as alternative spellings or forms of spaced/multi-word terms (e.g. `SOWTHISTLE` -> `"Alternative spelling of sow thistle"`). Queried and resolved definitions for **3,751 unique spaced terms** (97.3% coverage) using Wiktionary REST API with Wikipedia summary fallback.
- **Singular Words Formatted**: **3,855 singular words** now append the authentic definition of their spaced target in parentheses:
  `Alternative spelling of sow thistle (Any of the thistles in one of the genera Cicerbita and Sonchus.)`
- **Plural Words Formatted**: **2,908 plural words** referencing singular words now include the complete root definition in standard AW plural format:
  `plural of sowthistle (Alternative spelling of sow thistle (Any of the thistles in one of the genera Cicerbita and Sonchus.))`
- **Comprehensive Coverage**: A total of **6,763 words** were enriched across `wikdefs.txt`, `Definitions.txt`, `morpheme.db` (`wiktionary_definitions` table), and backup files.
- **Production Deployed**: Synchronized database and files to `132.148.72.249`, verified live via `/api/tools/validate`.

### 2. Omission of Archaic 2nd/3rd Person Verb Inflections from AW
- **Words Removed**: Removed **4,961** archaic verb inflections across three distinct inflection groups:
  1. *Third-person singular simple present indicative ending in `-eth`*: **2,489 words** (e.g., `ABANDONETH`, `ABASETH`).
  2. *Second-person singular simple past indicative ending in `-st`*: **552 words** (e.g., `ABANDONEDST`, `ABASEDST`).
  3. *Second-person singular simple present indicative ending in `-st`*: **1,920 words** (e.g., `ABANDONEST`, `ABASEST`).
- **Dictionary File Cleanup**:
  - `added_words.txt`: Updated from 474,064 to **469,103 words**.
  - `wikdefs.txt`: 720,678 entries.
  - `Definitions.txt`: 755,121 entries.
  - Purged from `morpheme.db` (`wiktionary_definitions` table).
- **FAQ Entry**: Updated the Added Words FAQ section in `templates/index.html` to document the omission of archaic verb inflections ending in `-st` and `-eth`.

### 3. File Maintenance & Duplicate Snapshots
- **Removed Deprecated Files**: Deleted legacy information-retrieval and backup files containing `ABASTARDIZE`.
- **Clean Snapshots**: Generated fresh duplicate snapshots (`added_words_duplicate.txt`, `wikdefs_duplicate.txt`) and verified 0 occurrences of `ABASTARDIZE` across all dictionaries.
- **Backup Synchronization**: Maintained fully up-to-date backup files (`added_words_backup.txt`, `wikdefs_backup.txt`, `Definitions_backup.txt`).

### 4. Word of the Day (WOTD) Mobile Single-Line Dynamic Sizing
- **Binary-Search Auto-Fitting**: Sized `#wotd-display` inside `.wotd-container` using the binary-search font-fitting algorithm (`applyDynamicValidationStyle`), mirroring the single-line behavior of Is Valid.
- **Single-Line Guarantee**: Long words fit on a single line (`white-space: nowrap !important; word-break: keep-all !important; overflow-wrap: normal !important;`) without wrapping or clipping, retaining the gold gradient styling and animations.

### 5. Mobile Fixed Bottom Back Buttons (Tools, Mods, Settings, Forum)
- **Architectural Edge Docking**: Restructured Tools, Mods, Settings, and Forum views so that fixed bottom navigation bars (`.mobile-bottom-nav`, `.forum-bottom-bar`) are direct docked flex children (`flex: 0 0 auto; margin: 0; padding: 7px 14px 4px 14px; border-top: 1px solid var(--input-border); z-index: 100`) anchored at the absolute bottom edge with no empty space underneath.
- **Independent Scrolling**: Content scrolls smoothly in its respective container above the docked bar. Tapping Back smoothly returns to the category or menu list.
- **Reduced Bottom Padding**: Reduced padding below the fixed buttons to `7px 14px 4px 14px` on mobile for a tight, elegant docked appearance.

### 6. Menu Tabs Display Fix (Mods & Settings)
- **DOM Hierarchy Integrity**: Fixed the container hierarchy in `templates/index.html` so that `#page-mods` and `#page-settings` remain top-level sibling views at depth 3 and are never inadvertently hidden when `#page-tools` is toggled.

### 7. Permanent Mobile Fullscreen & Virtual Keyboard Invariants
- **Full List Modal Invariant**: `document.exitFullscreen()` is explicitly invoked whenever the full list modal is opened (`static/js/tools.js`).
- **Android Virtual Keyboard Black Screen Prevention**: Fullscreen is unconditionally exited when navigating to utility pages (Tools, Settings, Profile, Forum, How to Play, Donate) or when opening text inputs/modals. Automatic fullscreen re-engagement is blocked on utility pages.
- **Gateway Screen Invariant**: Fullscreen is requested immediately on initial tap across `#page-loading` (ENTER LOBBY) and continuously preserved into `#page-lobby` without layout shifting.

### 8. Tile Selectable Space (Corner Cutoff) 39% Default Consistency
- **Initial HTML & CSS**: Initialized at 39% in `templates/index.html` with `--corner-cutoff: 39%` and clip-path polygon using 39%/61%.
- **Client Fallback**: Sized to 39% in `static/js/settings.js` whenever user settings are uninitialized.
- **Guest Initialization**: Initialized in `guest_login()` in `app.py` with `('corner_cutoff', '39')`.

### 9. Per-User Settings & Synesthesia Isolation
- **Per-User Namespacing**: Uses `morpheme_settings_<username>` localStorage keys.
- **DOM Cleanup on Logout**: Strips all 26 CSS custom properties (`--letter-*-color`) from `document.documentElement` upon logout.

### 10. Store in Tools Dual Currency & Collins Official Book
- **Green Book**: Features Collins Official Scrabble Words (2-15 All Words) edition.
- **Dual Currency**: Dual CAD and USD pricing across all 13 items.
