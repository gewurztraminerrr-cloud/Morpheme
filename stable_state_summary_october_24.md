# Stable State Summary — October 24, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_24
```
*(Also tagged as `stable-2026-10-24`)*

## Latest Commit ID
```
0fc18dc26ab9c724ef067fb648708608ca4fbeb6
```

## Synchronization Status

| Environment | Status | Details |
|:---|:---|:---|
| **Localhost** | ✅ Synchronized | Clean working tree on `main` |
| **GitHub (`origin/main`)** | ✅ Synchronized | Pushed to `origin/main` & tags `START_OVER_POINT_OCTOBER_24` / `stable-2026-10-24` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | Updated to latest commit, tags fetched, PM2 process online, live HTTP 200 OK |
| **App / Client** | ✅ Synchronized | Cache busters bumped (`play.css?v=1792830000`, `play.js?v=1792826000`, `tools.js?v=1792842000`) for instant live propagation across all devices |

---

## Session Highlights & Architectural Updates (October 24, 2026)

### 1. New Words Ordering: New Additions Pinned to Top in Alphabetical and Random Orders
- **Issue Resolved**: When a new word was added to Added Words (AW) via the Mods tab (such as `TILCAYO`), it erroneously appeared third from last at the bottom of the "New AW Words" list in the Lists tool.
- **Root Cause**:
  - `tools_get_lists()` in `app.py` was iterating through all 538,420 words in `added_words.txt` instead of loading specifically from `added_words_dates.txt`.
  - Both backend `order_list` and client `sortWordsAlphabetical` / `shuffleWordsArray` sorted words purely A–Z by name across all dates, pushing `TILCAYO` (starting with T) to the bottom near `WANGYANITE`.
- **Implementation**:
  - Implemented `order_new_words_list(lst)` in `app.py`: groups word-date pairs by addition date (`DD/MM/YYYY`) in descending order (newest date first). When Alphabetical is selected, words are sorted alphabetically within each date group; when Random is selected, words are shuffled within each date group.
  - Pinned newest addition dates (e.g., `24/10/2026`: `TILCAYO`, `TILCAYOS`, `TESTINGG`) strictly to the top of `New AW Words` and `All New Words`.
  - Implemented client-side `parseWordDate` and `sortNewWordsList` in `static/js/tools.js` to ensure the same date-descending priority is preserved when viewing the list or opening the Full List modal.
  - Bumped `tools.js` cache buster to `?v=1792842000`.

### 2. Mini-Profile & Public Profile Complete Synchronization
- **Issue Resolved**: In Tools -> Find Count, the mini-profile for `jeffjeff` displayed `1` under "GAMES" while their Profile displayed `101`; for `jeffb` it displayed `138` while Profile displayed `27`, and "WINS" read `0`.
- **Implementation**:
  - Re-engineered `get_mini_profile(username)` in `app.py` to delegate directly to `get_public_profile(username)`, eliminating disconnected schema queries and ensuring both endpoints share the exact same 30-second TTL cache.
  - Updated `static/js/tools.js` to format games and wins using `.toLocaleString()` and clean placeholder dashes for missing demographics.
  - Verified live: both `jeffjeff` (101 games, 26 wins) and `jeffb` (27 games, 4 wins) now match identically across mini-profile and public profile.

### 3. Real-Time Combo Query Cache Invalidation on Added Words Modifications
- **Enhancement**: When words are added to or removed from Added Words via the Mods tab or Tools interface, `COMBO_QUERY_CACHE.clear()` is immediately executed on the server and `window._comboClientCache.clear()` is invoked on the client.
- **Result**: Combo Checker queries for Microprocedures (MP) and Letters In Common (LIC) immediately incorporate new additions without requiring a server reboot or waiting for cache expiration.

### 4. Firefox Desktop MORPHEME Title Enlarge & Equalized Vertical Spacing
- **Enhancement**: Under Firefox on desktop (`@-moz-document url-prefix()`), increased the main MORPHEME title logo font size and equalized top and bottom header padding.
- **Visual Polish**: Aligned visual hierarchy and baseline padding identically with Chrome and Safari.

### 5. Automatic Inflection Suffix Appending for Added Words Definitions
- **Lexicographical Integrity**: Resolved definitions for Added Words now automatically compute and append valid inflection suffixes (`[inflections: ...]`) covering plurals, participles, and verb tenses via `get_valid_inflections_for_word`.

### 6. Combo Checker: Removal of "v2.0" from MP Title and Balanced Spacing
- **Title Cleanliness**: Removed outdated `"v2.0"` designation from the Microprocedures header in the Combo Checker tool (`#combo-pane-mp h3`), rendering clean and authoritative as `MP (Microprocedures)`.
- **Harmonized Spacing & Padding**:
  - Enclosed title and navigation arrows inside a dedicated `.combo-section-header` flex row with `gap: 16px`, `margin-top: 2px`, and `margin-bottom: 10px;`.
  - Expanded horizontal scroll arrow button gap to `gap: 8px; flex-shrink: 0;` with touch-friendly padding.
  - Symmetrically matched layout, typography, and arrow spacing on the LIC (Letters In Common) tab (`#combo-pane-lic`).

### 7. Point Sum Equation Contrast in White Layouts & Elimination of Congratulatory Black Glow
- **High-Contrast Equations in White/Light Themes**:
  - Eliminated faint opacity styling from score equation math (`5 + 3 = 8`).
  - Added semantic `.word-item-pts` with `opacity: 1 !important` and `font-weight: 800 !important`.
  - Distinct high-contrast color palettes tailored to each word state.
- **Eliminated Black Glow Around Congratulatory Winner Card**:
  - Completely removed the heavy `.winner-flash *` black outline shadow (`text-shadow: 0 0 10px rgba(0, 0, 0, 0.9);`).
  - Adapted typography to selected layout: warm amber gold (`#b45309`) in White/Light, luminous gold (`#ffd700`) in Dark.

### 8. Words to Work on Strict Tab Isolation & Multi-Tool Integration
- **Strict Single-Tab Containment**: Confined `#tool-work-on` strictly to its dedicated container, ensuring explicit isolation so `#page-tools` only displays Words to Work on when that specific tab is selected.
- **Tool Integration**: Added the user's personal "Words to Work on" list as a selectable dictionary option in MP Word Explorer and Subanagrams.
- **Mobile Table Containment**: Fitted the Words to Work on table to device viewport with restricted internal scrolling.

### 9. In-Game Room & Board Interaction Enhancements
- **Touch-Swipe Real-Time Letter Display**: On mobile devices, swiped letters display immediately beneath the board inside the validation space while finger dragging is active.
- **Bonus Word Row Live Highlighting**: Words found during the round matching the secret bonus word highlight with an emerald green row background in real time under the Words panel.
- **Continuous In-Game Chat at Round End**: Users who joined as spectators or did not play the active round can continue chatting freely during round end.

### 10. Tools Modernization (Lists, Subanagrams, Unscramble)
- **Lists Ordering & Virtual Scrolling**: Added Alphabetical and Random ordering dropdowns to the Lists tool with responsive mobile grid formatting and robust virtual scrolling.
- **Subanagrams & Unscramble Dual-Panel Layout**: Reorganized Unscramble on desktop and laptop viewports to mirror Subanagrams' dual-panel layout.

### 11. Store & Visual Boutique Enhancements
- **Boutique Showcase Lightbox**: Made boutique showcase displays and product imagery expandable via full-screen lightbox modal on click.
- **Dictionary Edition Verification**: Added mandatory dictionary edition verification routine to the monthly store update cycle.

---

## Complete Commit History (October 24, 2026)

| Commit | Summary |
|:---|:---|
| `6c47e501` | **fix(lists): keep newly added words at top of New Words lists in alphabetical and random order** |
| `f08cb2b7` | **fix(profile): synchronize mini-profile stats with canonical public profile data** |
| `2ac6bd13` | **fix(combo): invalidate server COMBO_QUERY_CACHE and client _comboClientCache on Added Words changes** |
| `f29fcdaa` | **style: enlarge MORPHEME title and equalize vertical padding on desktop Firefox** |
| `5cf00d0e` | **feat: include valid inflections at the end of definitions for all Added Words** |
| `b4a39873` | **Remove v2.0 from MP title and space out title, description, and arrows in Combo Checker** |
| `9bca2e79` | **Improve point sum equation contrast in white layouts and remove black glow from round-end congratulatory message** |
| `dc59dc73` | **Confine Words to Work on content strictly to its own tab** |
| `89db81c4` | **Add Words to Work on word list to MP Explorer and Subanagrams tools** |
| `2bac69b1` | **fix(play): guard defPanel and all_word_scores lookups in updateGameState** |
| `0bd0e59c` | **feat(play): highlight bonus word row green under Words during round** |
| `c0584457` | **fix(lobby): make border surrounding set rating limits panel razor thin (1px)** |
| `79c6a2a8` | **fix(lobby): equalize space above and below users under players in lobby** |
| `bc48b71b` | **feat(chat): allow continuous typing in chat at round end for users who did not play the round** |
| `455168d0` | **feat(store): add mandatory dictionary edition check to monthly store update routine** |
| `f07dce8d` | **fix(play): uniform divider line thickness on color chart and remove shadow on white layouts** |
| `40a42a84` | **fix(chat): ensure 'Chat disabled for 2 seconds...' notice properly disappears after 2s** |
| `6a4033bc` | **fix(mobile): fit Words to Work on table to screen and restrict scrolling to table only** |
| `9a476c3d` | **feat(mobile): display swiped letters under board in validation space and replace with validation text on release** |
| `a79761a0` | **fix(tools): ensure jumbled letters in Unscramble and Subanagrams fit on a single row without wrapping** |
| `bb26cfdd` | **feat(unscramble): reorganize desktop/laptop layout to mirror Subanagrams dual-panel layout** |
| `f83d68d6` | **fix(lists): correctly scale and scroll virtual thumb when all words fit in DOM (All New Words on mobile)** |
| `70bfef0a` | **update(lobby-guide): change path tracing examples from STOP to STAMP and SLOW to PAROLE** |
| `7a13b836` | **fix(lists): respect random order when opening View Full List modal** |
| `935a6e96` | **fix(app): ensure lst is converted to list before subscripting in order_list** |
| `e0e05daf` | **feat(tools): add Alphabetical and Random order dropdown to Lists with responsive mobile grid layout** |
| `b701f27d` | **fix(vibration): decouple intermission 0:10 vibration alert from bell setting and add Flutter bridge haptic support** |
| `96c549a8` | **style(store): ensure black text on white active category filter in monochromatic dark themes** |
| `8a3f304b` | **style(store,lobby): remove glowing box-shadow from store category filters and start solo match button** |
| `54bddfa1` | **style(subanagrams): widen sequence game area panel and narrow word list results panel on desktop** |
| `48be6f05` | **fix(subanagrams): restore full-height words panel on desktops and laptops with responsive dual-panel layout** |
| `860cd24a` | **feat(store): make showcase images and product images expandable via lightbox** |
| `5a21f948` | **refactor(ui): rename SHOW EVERYONE to VIEW ALL and FIND FRIENDS to VIEW FRIENDS in game rooms** |
| `145d2cad` | **feat(store): update showcase imagery to official Boggle 4x4/5x5 grid games and dictionaries** |
| `1b98dfdb` | **feat(store): showcase Morpheme boutique outdoor displays on desktop store status banner** |

---

## Server & Infrastructure
- **Production Host**: Ubuntu 24.04 LTS at `132.148.72.249`.
- **Application Stack**: Python 3.9 / 3.12 + Waitress WSGI managed by PM2 (`morpheme`).
- **Web Server / Reverse Proxy**: Nginx 1.24.0 with HTTP/2 and SSL/TLS termination (`https://morpheme.games`).
- **Database**: SQLite (`morpheme.db`) with WAL mode, automated schema migrations, in-memory rating caches, and IP timeout controls.
- **Verification**: Verified live on production host `132.148.72.249` with HTTP 200 OK across public profile, mini-profile, and new word lists endpoints.
