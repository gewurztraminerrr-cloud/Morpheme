# Stable State Summary — October 24, 2026 (Start Over Point)

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_24
```
*(Also tagged as `stable-2026-10-24`)*

## Latest Commit ID
```
391fe514aaef237cb610f2c2f3d3a30b63291e81
```

## Synchronization Status

| Environment | Status | Details |
|:---|:---|:---|
| **Localhost** | ✅ Synchronized | Clean working tree on `main` |
| **GitHub (`origin/main`)** | ✅ Synchronized | Pushed to `origin/main` & tags `START_OVER_POINT_OCTOBER_24` / `stable-2026-10-24` |
| **morpheme.games (`132.148.72.249`)** | ✅ Synchronized | Updated to latest commit, tags fetched, PM2 process online, live HTTP 200 OK |
| **App / Client** | ✅ Synchronized | Cache busters bumped (`play.css?v=1792830000`, `play.js?v=1792826000`, `tools.js?v=1792825000`) for instant live propagation across all devices |

---

## Session Highlights & Architectural Updates (October 24, 2026)

### 1. Combo Checker: Removal of "v2.0" from MP Title and Balanced Spacing
- **Title Cleanliness**: Removed outdated `"v2.0"` designation from the Microprocedures header in the Combo Checker tool (`#combo-pane-mp h3`), rendering clean and authoritative as `MP (Microprocedures)`.
- **Harmonized Spacing & Padding**:
  - Enclosed title and navigation arrows inside a dedicated `.combo-section-header` flex row with `gap: 16px`, `margin-top: 2px`, and `margin-bottom: 10px;`.
  - Expanded horizontal scroll arrow button gap to `gap: 8px; flex-shrink: 0;` with touch-friendly padding.
  - Relaxed description box margins and padding (`margin-bottom: 10px !important; padding: 5px 12px !important;`) across desktop and mobile, eliminating the previous tight 4px/6px cramped stack.
  - Symmetrically matched layout, typography, and arrow spacing on the LIC (Letters In Common) tab (`#combo-pane-lic`).

### 2. Point Sum Equation Contrast in White Layouts & Elimination of Congratulatory Black Glow
- **High-Contrast Equations in White/Light Themes**:
  - Eliminated faint `opacity: 0.8` / `0.6` styling from score equation math (`5 + 3 = 8`).
  - Added semantic `.word-item-pts` with `opacity: 1 !important` and `font-weight: 800 !important`.
  - Engineered distinct, high-contrast text color palettes tailored to each word state:
    - Player found words: `#0a1e3f !important` (deep navy on blue).
    - Bonus words: `#052e16 !important` (forest green on lime green).
    - Missed words: `#881337 !important` (deep maroon on soft rose).
    - CSW-only words: `#000000 !important` (crisp black on gold).
    - Added words: `#3b0764 !important` (royal purple on lavender).
- **Eliminated Black Glow Around Congratulatory Winner Card**:
  - Completely removed the heavy `.winner-flash *` black outline shadow (`text-shadow: 0 0 10px rgba(0, 0, 0, 0.9);`).
  - Set `text-shadow: none !important;` across all winner card elements.
  - Adapted typography to selected appearance layout: rich warm amber gold (`#b45309`) in White and Light layouts, and luminous pure gold (`#ffd700`) in dark layouts.

### 3. Words to Work on Strict Tab Isolation & Multi-Tool Integration
- **Strict Single-Tab Containment**:
  - Resolved issue where Words to Work on content erroneously persisted or leaked into other tool panes when switching tabs.
  - Confined `#tool-work-on` strictly to its dedicated container, ensuring explicit isolation so `#page-tools` only displays Words to Work on when that specific tab is selected.
- **Tool Integration**:
  - Added the user's personal "Words to Work on" list as a selectable dictionary option in MP Word Explorer and Subanagrams, enabling focused study on missed or starred words.
  - Mobile table containment: Fitted the Words to Work on table to the device viewport with restricted internal scrolling and pinned action controls.

### 4. In-Game Room & Board Interaction Enhancements
- **Touch-Swipe Real-Time Letter Display**:
  - On mobile devices, swiped letters now display immediately beneath the board inside the validation space while finger dragging is active, providing tactile visual feedback before swapping to validation status upon finger release.
- **Bonus Word Row Live Highlighting**:
  - Words found during the round that match the secret bonus word now highlight with an emerald green row background in real time under the Words panel.
- **Continuous In-Game Chat at Round End**:
  - Users who joined as spectators or did not play the active round can now continue chatting freely during round end without arbitrary input freezes.
  - Guaranteed the "Chat disabled for 2 seconds..." rate-limiting notice reliably dismisses after 2 seconds.
- **Haptic Vibration Decoupling**:
  - Decoupled the 10-second intermission countdown vibration alert from the auditory bell setting, allowing silent haptic alerts, and added Flutter bridge native haptic support.

### 5. Tools Modernization (Lists, Subanagrams, Unscramble)
- **Lists Ordering & Virtual Scrolling**:
  - Added Alphabetical and Random ordering dropdowns to the Lists tool with responsive mobile grid formatting.
  - Ensured randomized seed order is strictly respected when viewing full lists in modal view.
  - Fixed virtual scroll thumb scaling when all items fit within the viewport without overflowing.
- **Subanagrams & Unscramble Dual-Panel Layout**:
  - Reorganized Unscramble on desktop and laptop viewports to mirror Subanagrams' dual-panel layout (game controls on the left, word lists on the right).
  - Widened sequence game area panel and guaranteed jumbled letter tiles fit on a single row without line wraps.

### 6. Store & Visual Boutique Enhancements
- **Boutique Showcase Lightbox**:
  - Made boutique showcase displays and product imagery expandable via full-screen lightbox modal on click.
  - Updated showcase imagery with official Boggle 4x4/5x5 grid games and dictionaries.
  - Guaranteed black text on active white category filter pills in monochromatic dark themes.
  - Added mandatory dictionary edition verification routine to the monthly store update cycle.

---

## Complete Commit History (October 24, 2026)

| Commit | Summary |
|:---|:---|
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
- **Verification**: Verified live on production host `132.148.72.249` with HTTP 200 OK and Chrome CDP headless verification.
