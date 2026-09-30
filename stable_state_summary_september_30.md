# Stable State Summary — September 30, 2026

## Latest Commit ID
```
83824cc9042af20fd2d446566b8ca95177a43078
```

## Tags
- `START_OVER_POINT_SEPTEMBER_30`
- `stable-2026-09-30`

## Synchronization Status

| Environment    | Commit                                     | Status |
|----------------|--------------------------------------------|--------|
| localhost      | `83824cc9042af20fd2d446566b8ca95177a43078` | ✅ Synchronized |
| GitHub (main)  | `83824cc9042af20fd2d446566b8ca95177a43078` | ✅ Synchronized |
| morpheme.games | `83824cc9042af20fd2d446566b8ca95177a43078` | ✅ Synchronized (HTTP 200 OK) |

---

## Session Features & Fixes (September 30, 2026)

### 1. Board Size Example Board Below Configurable Sliders
- **Feature**: In Settings -> Appearance -> Board Size, rearranged the layout so the 2D example preview board is placed directly **below** the configurable dimension sliders (4x4, 4x6, 5x7, 6x8).
- **Files**: `templates/index.html`.

### 2. Clues Tab Letter Length Filter Tabs (24H Rooms)
- **Feature**: Added dynamic letter length filter buttons (e.g., `ALL`, `7LW`, `8LW`, `9LW`, `10LW`) directly below the "Remaining" toggle button in the Clues tab (`#tab-content-clues`) in 24-hour rooms.
- **Dynamic Board Lengths**: Dynamically inspects all words on the current board; word lengths not present on the board are completely omitted (e.g., no `10LW` button if there are no 10-letter words on the board).
- **Filtering**: Clicking any length button filters the clue list to display only unfound clues matching that length. Clicking `ALL` repopulates all clues.
- **Display Behavior**: Tabs display when in Clues mode in 24H rooms and hide when toggled to Remaining counts mode or outside 24H rooms.
- **Files**: `templates/index.html`, `static/css/play.css`, `static/js/play.js`, `game_room.py`.

### 3. Mods Tab Visibility & Moderator Access (`jeffy`, etc.)
- **Root Cause**: `setCurrentUser(username, ..., isMod = false)` defaulted `isMod` to `false`, overwriting mod status during gateway "ENTER LOBBY" navigation.
- **Fix**: Changed default to `isMod = null`, persisted `morpheme_is_mod` in `localStorage`, and updated `updateAuthUI()` and gateway transition to preserve moderator status.
- **Database Sync**: Added auto-sync from `dictionaries/mods.txt` into SQLite `moderators` table in `morpheme.db`.
- **Files**: `app.py`, `static/js/app.js`, `static/js/mods.js`, `templates/index.html`.

### 4. New AW Words Date Integrity & Historical Dates
- Fixed bug where viewing "New AW Words" caused older words below newly added words to display the current date instead of their original addition date.
- Added `DOCKSMAN` and `DOCKSMEN` with authentic historical dates and definitions according to Dictionary Suffix and Plural Sourcing Rules.
- **Files**: `app.py`, `dictionaries/Definitions.txt`, `dictionaries/added_words.txt`, `dictionaries/added_words_dates.txt`, `dictionaries/added_words_duplicate.txt`, `dictionaries/wikdefs.txt`, `dictionaries/wikdefs_aw_duplicate.txt`, `dictionaries/wikdefs_duplicate.txt`, `dictionaries/word_stats.json`.

### 5. Multiplayer Subanagrams Header Title Mixed Case
- Changed multiplayer Subanagrams header title and meta text from all-caps "SUBANAGRAMS PRACTICE | 2m" to mixed-case "Subanagrams Practice | 2m".
- **Files**: `static/js/play.js`, `templates/index.html`.

### 6. AW Dictionary FAQ Forum Guidelines
- Updated FAQ for Added Words (AW) to specify:
  - If a word does not belong in AW or a definition is incorrect, mention it in the designated thread in Complaints in the Forum.
  - If a word is not present in AW, CSW, and NWL but should be, mention it in the designated thread in Suggestions in the Forum.
- **Files**: `templates/index.html`.

### 7. Desktop & Laptop Lobby Players Horizontal Wrap
- In "Players in Lobby & Chat", changed desktop/laptop layout from full-width single-user rows to horizontally wrapped player badge chips matching the mobile layout.
- **Files**: `static/css/lobby.css`.

### 8. Mini-Profile About Me Vertical Height Halved
- Decreased vertical length of the About Me scroll box by half on desktop and laptop screens when content requires a scrollbar/thumb.
- **Files**: `static/css/lobby.css`.

### 9. Subanagrams Tile Size Setting for Desktop & Laptop
- Added desktop and laptop tile size selector (Normal vs Compact/Mini) to game settings.
- **Files**: `static/js/settings.js`, `static/css/play.css`, `templates/index.html`.

### 10. Subanagrams Spinner Set Odds Modal
- Added interactive modal displaying detailed spinner odds breakdown when clicking the spinner badge in Subanagrams.
- **Files**: `static/js/play.js`, `static/css/play.css`, `templates/index.html`.

---

## How to Roll Back to This Point

```bash
# On localhost:
git checkout START_OVER_POINT_SEPTEMBER_30

# On morpheme.games (production server):
cd ~/morpheme
git fetch origin main
git reset --hard START_OVER_POINT_SEPTEMBER_30
pm2 restart morpheme
```
