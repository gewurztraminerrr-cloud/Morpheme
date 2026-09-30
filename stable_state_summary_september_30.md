# Stable State Summary — September 30, 2026

## Latest Commit ID
```
f583530bba3b5168d601a91e56b4aece72fb7ba6
```

## Tags
- `START_OVER_POINT_SEPTEMBER_30`
- `stable-2026-09-30`

## Synchronization Status

| Environment    | Commit                                     | Status |
|----------------|--------------------------------------------|--------|
| localhost      | `f583530bba3b5168d601a91e56b4aece72fb7ba6` | ✅ Synchronized |
| GitHub (main)  | `f583530bba3b5168d601a91e56b4aece72fb7ba6` | ✅ Synchronized |
| morpheme.games | `f583530bba3b5168d601a91e56b4aece72fb7ba6` | ✅ Synchronized (HTTP 200 OK) |

---

## Session Features & Fixes (September 30, 2026)

### 1. AW Dictionary FAQ Guidance on Checking Is Valid with "All"
- In the FAQ entry for Added Words (AW), updated the ending guidance to:
  > *"If you find a word that is not present in AW, CSW, and NWL, but should be (search any word in Is Valid in Tools using “All” word list first: If it’s not a valid word, it is not a word used in Morpheme), mention that in the designated thread in the Suggestions category in the Forum."*
- **Files**: `templates/index.html`.

### 2. Word Lists Added to Tools Dropdowns & "All" Standardization
- **6 Word Lists Added**: Added `CSW Only` (85,587 words), `NWL Uniques` (90,300 words), `New NWL Words` (2 words), `New CSW Words` (0 words), `New AW Words` (40 words with recorded dates), and `All New Words` (42 words) across dictionary dropdown menus in:
  - Combo Checker (`#combo-dict`)
  - Sequence (`#seq-dict`)
  - Subanagrams Manual (`#sub-dict`) & Random (`#sub-dict-random`)
  - Random Word (`#random-dict`)
  - Unscramble (`#unscramble-dict`)
  - Find Count (`#random-words-dict-select`)
- **Unscramble Renaming**: Renamed `"Uniques"` option in Unscramble to `"NWL Uniques"`.
- **"ALL" to "All" Standardization**: Changed all visible `"ALL"` labels across every dictionary dropdown in the application to mixed-case `"All"` (e.g., `"All (Full)"`, `"All"`, `"All Likelihood"`).
- **Backend Dictionary Loader Guard**:
  - In `load_tools_dictionary(dict_name)` in `app.py`, restricted merging of `16plus.txt` exclusively to full dictionaries (`ALL`, `All`, `NWL`, `CSW`). Subset and new word lists (`csw_only`, `uniqueNWL`, `new_nwl`, `new_csw`, `new_added`, `all_new`) now preserve their exact list integrity without injecting 9,227 unrelated 16+ words.
  - Pre-warmed all subset and new word lists in `warm_up_server_resources()`.
  - Added `all_new` to `/api/tools/lists` and frontend Lists helpers with date metadata sorted newest first.
  - Updated `/api/tools/random-words` (Find Count) to dynamically support all dictionaries.
- **Files**: `app.py`, `templates/index.html`, `static/js/tools.js`.

### 2. Left-Align Personal Quote on Mobile Devices in Profile
- **Requirement**: On mobile devices on Profile, align "PERSONAL QUOTE" to the left side in the same way as "ABOUT ME".
- **Implementation**:
  - In `static/css/style.css`: updated mobile breakpoint styles for `.profile-quote-box .profile-quote-minimal p`, `.profile-quote-box .meta-label`, and `.profile-quote-minimal` from `text-align: center !important;` to `text-align: left !important;`.
  - In `static/css/play.css`: updated `.profile-quote-minimal` from `text-align: center !important;` to `text-align: left !important;`.
  - In `templates/index.html`: bumped cache busters for `style.css` and `play.css` to `v=1790628200`.
- **Files**: `templates/index.html`, `static/css/style.css`, `static/css/play.css`.

### 2. Equal Left and Right Padding for Panels under "Status & Results" in Unscramble
- **Issue**: Under "Status & Results" in Tools -> Unscramble, the panel containing words had noticeably more empty space on the right side than on the left side due to asymmetric list padding (`padding-right: 22px !important;` with `0` left padding in the history scroller to leave space for the scrollbar track, and `padding: 5px 10px 5px 5px;` on the outer `#unscramble-found-list`).
- **Fix**:
  - In `templates/index.html`: changed `#unscramble-found-list` inline padding from `5px 10px 5px 5px` to equal `5px 0`.
  - In `static/css/play.css`:
    - Added global `#unscramble-found-list { padding: 0 !important; }` to eliminate asymmetric list gutters.
    - Updated `.unscramble-history-list` across mobile and desktop breakpoints to use balanced padding: `padding-left: 20px !important; padding-right: 20px !important; padding-bottom: 6px !important;`.
  - In `static/js/tools.js`: updated `#unscramble-history-scroll` inline style to use `padding: 0 20px 6px 20px;`.
  - Bumped cache busters for `play.css` and `tools.js` to `v=1790628100`.
- **Files**: `templates/index.html`, `static/css/play.css`, `static/js/tools.js`.

### 3. Store Magnetic Letters Copy Rewrite
- In Tools -> Store, updated the copy under **JoyCat Silicone Uppercase Magnetic Letters**:
  - Changed: *"Then, every time you open the fridge, you are reminded of it!"*
  - To: *"Then, when you open the fridge, you are reminded of them!"*
- **Files**: `templates/index.html`.

### 4. Pronunciation Immediately Under Word Declaration/Title & Definition Immediately Under Pronunciation
- **Core Requirement**: Across word declarations, dictionary popovers, and definition cards, position the pronunciation of a word immediately under the word declaration/title, and place the definition text immediately under the pronunciation.
- **Gameplay Definition Panel (`.definitions-panel`)**:
  - Integrated `#definition-pronunciation` directly into `#definition-header`, placed immediately below the word title (`#definition-word`).
  - Positioned `#definition-content` (definition text) immediately below the pronunciation.
  - Eliminated the separating border divider between the word title and pronunciation so the headword and its pronunciation sit tightly and cleanly together.
  - Adjusted alignment across breakpoints: on mobile (`<= 900px`), both the word title and its pronunciation are centered, with definition text left-aligned below; on desktop, both title and pronunciation are left-aligned above definition text.
  - Updated `fetchDefinition` in `play.js` to render `data.pronunciation` into `#definition-pronunciation` immediately below the word title, and cleared/hidden it when changing views or resetting rounds.
- **In-Place Tool Definition Popover (`tool-def-popover`)**:
  - Moved `#tool-def-pronunciation` inside `.tool-def-header-left` directly under the word title (`#tool-def-word-text`) and badge, before the close button and definition content.
  - Positioned `#tool-def-content` immediately below the pronunciation.
- **Moderator "Declare Word & Definition" Card (`#mod-tab-defs`)**:
  - Added an optional Pronunciation input field (`#def-pron-input`) directly below the Word(s) declaration input (`#def-word-input`) and directly above the Definition textarea (`#def-text-input`).
  - Added keyboard Enter handling for fluid navigation between fields.
  - Updated `/api/mods/definitions/add` in `app.py` and `mods.js` to save pronunciations to `PRONUNCIATIONS_CACHE` and append to `pronunciations.txt`.
- **Files**: `templates/index.html`, `static/css/play.css`, `static/js/play.js`, `static/js/tools.js`, `static/js/mods.js`, `app.py`.

### 5. "Set to Default Sizes" Button in Board Size Settings
- **Feature**: Added an actionable button labeled **"Set to Default Sizes"** positioned directly below the configurable dimension sliders and above the 2D example preview board in Settings -> Appearance -> Board Size.
- **Functionality**:
  - Immediately resets all 4 dimension-specific sliders to their canonical defaults:
    - 4x4 Grid: 82px
    - 4x6 Grid: 82px
    - 5x7 Grid: 65px
    - 6x8 Grid: 54px
  - Updates the 2D example preview board to default (82px).
  - Persists the reset sizes to `localStorage` and syncs with the server via `saveSettingDebounced`.
  - Dynamically updates the active board on `#page-play` and recalculates panel layout/overflow if the user is currently in a room matching one of the board dimensions.
  - Provides instant tactile button feedback ("Reset to Defaults!").
- **Files**: `templates/index.html`, `static/js/settings.js`.

### 6. Settings -> Appearance -> Board Size Layout Hierarchy
- Reordered the Board Size tab inside Settings -> Appearance so that the interactive dimension sliders appear above the 2D example preview board, rather than below it.
- **Files**: `templates/index.html`.

---

## Tournament BYE Rules Reference (Summary for User Question)
- **Round 1 (First Round)**: The BYE recipient is chosen **completely at random** (or through seeded bracket position if a pre-existing seeding ranking is established).
- **Subsequent Rounds (Swiss System Standard)**:
  - In a standard Swiss-system tournament, BYEs are assigned to the **lowest-performing player** (lowest score / lowest match points) who has **not yet received a BYE**, ensuring top performers play each other for the championship and no player receives more than one BYE.
  - Awarding a BYE to the top performer would unfairly advance the leader without requiring them to play, which contradicts Swiss pairing principles.
