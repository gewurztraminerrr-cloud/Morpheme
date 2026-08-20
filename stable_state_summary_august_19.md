# Stable State Summary — August 19, 2026

This document records the official **'Start Over'** stable point for **Morpheme** as of August 19, 2026. The codebase across localhost, GitHub (`main`), and `morpheme.games` is synchronized.

---

## 1. Repository & Deployment Information

* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Branch**: `main`
* **Date**: August 19, 2026
* **Commit ID**: `81e7ae3e24115ad3cd43bc7d2c97f02043a69d4b` (`81e7ae3`)
* **Asset Version**: `v=33070`

---

## 2. Key Features, Improvements & Fixes in This Stable State

### A. Instant 24h Midnight Rollover & Elimination of Double Eviction (`game_room.py`, `static/js/play.js`)
- **2-Second Midnight Transition**: Reduced the midnight rollover intermission in 24h rooms from 60 seconds down to **2 seconds**, pre-staging the new day's board instantly.
- **Protected Re-Entry**: Modified eviction logic in `play.js` so that only actively established players present during the round's concluding moment receive the end-of-day modal. Re-entering a 24h room immediately from the lobby will never trigger a second kick.

### B. Automatic Root Word Definition Lookup & Bracket Appending (`app.py`)
- **Recursive Root Resolution**: For any word defined with a pointer pattern (e.g. `third-person singular simple present indicative of [word]`, `plural of [word]`, `diminutive of [word]`, `synonym of [word]`, `alternative form of [word]`, `conjugation of [word]`, `comparative of [word]`, etc.), the definition engine automatically retrieves the full lexicographical definition of the referenced root word and appends it directly inside parentheses/brackets next to the root word.
- **Verified Examples**:
  - `BEHEDGES` $\rightarrow$ `third-person singular simple present indicative of behedge ((transitive) To hedge about; surround with or as with a hedge.)`
  - `MALAXER` $\rightarrow$ `Synonym of malaxator (one who, or that which, malaxates; esp. a machine for grinding, kneading, or stirring into a pasty or doughy mass [n -S])`
  - `MALAXERS` $\rightarrow$ `plural of malaxer (Synonym of malaxator (one who, or that which, malaxates; esp. a machine for grinding, kneading, or stirring into a pasty or doughy mass [n -S]))`
  - `POLESTER` $\rightarrow$ `(motor racing) Diminutive of polesitter ((motor racing) A driver placed in pole position.)`

### C. Clean Definition Formatting (Removed Leading `(noun)`) (`app.py`)
- Removed `(noun)` / `(Noun)` from the start of definitions across the entire dictionary lookup and resolution pipeline. Noun entries now start cleanly with their direct definition or root reference (e.g. `a horseman, also CABALLERO [n -S]` or `APPLE, the firm round edible fruit of the apple tree`). All other language origins (`(Hawaiian)`, `(French)`) and non-noun tags (`(verb)`, `(adjective)`) remain preserved.

### D. Dictionary Cleanup: Obsolete Words, Abbreviations & Misspellings Removed (`dictionaries/`)
- **Protected Standard Words**: NWL (199,429 words), CSW (281,598 words), and 16+ supplementary words (9,227 words) are completely preserved and locked.
- **Removed Flagged Added Words & Inflections**: Removed **30,644** obsolete words, abbreviations, and misspellings along with all their derived conjugations, plurals, and participles (e.g. `ABASTARDIZE`, `ABASTARDIZED`, `ABASTARDIZES`, `ABASTARDIZING`, `ABBERANT`, `ABDOM`, etc.).
- **Updated Lexicon Counts**:
  - **Truly Added Words**: **`469,764`** *(down from 500,408)*.
  - **Raw Added Words File (`added_words.txt`)**: **`479,310`** *(down from 509,954)*.
- **Untouched Duplicate Backups**: `dictionaries/added_words_backup.txt`, `dictionaries/Definitions_backup.txt`, and `dictionaries/wikdefs_backup.txt` remain permanently preserved.
- **Flushed Pregenerated Boards**: Flushed and refreshed `pregenerated_boards` and `used_boards` in SQLite so all board parameters align strictly with the cleaned dictionary.

### E. 24h Room Score Sum 0-Score Exclusion (`app.py`, `static/js/play.js`)
- Players with an overall total score of 0 are completely excluded from the Score Sum table across all four 24h rooms (`24h_4x4`, `24h_4x6`, `24h_5x7`, and `24h_6x8`).
- Backfill queries, SQL aggregation (`HAVING MAX(d.score_sum) > 0`), in-memory room scans, and frontend render logic only display and count players who have earned a score of 1 or greater.

### F. Guaranteed Session Expired Notice Suppression on $\ge$ 1 Hour Return (`templates/index.html`, `static/js/app.js`, `static/js/play.js`)
- Timestamps track strictly on physical human interactions (`mousedown`, `keydown`, `touchstart`, `pointerdown`, `scroll`), removing false-active background heartbeat intervals.
- Dual-layer storage & memory verification ensures returning after $\ge 1$ hour of absence silently returns to the lobby with zero popup modal.

### G. In-Place Word Definition Popover across Tools (`static/js/tools.js`, `static/css/play.css`, `templates/index.html`)
- Clicking words in **Combo Checker**, **Sequence**, **Subanagrams**, **Lists**, and **View Full List** displays a sleek in-place definition popover card directly next to the word without navigating the user away to the "Is Valid" tool.

### H. 170× C-Accelerated Morpheme Metric & High-Speed Combo Checker (`app.py`, `morpheme_metric.c`, `static/js/tools.js`)
- Bare-metal C engine running LCS and bitmask backtracking in CPU registers, taking search times from ~45 seconds down to **0.06s – 0.25s**.
- Guaranteed 0MP subword extraction and uncapped results tables.

### I. Accumulative Lobby Real-Time Auto-Polling & Live Count Synchronization (`static/js/lobby.js`)
- 2-second background auto-polling on the lobby page ensuring active player counts (`Start [0]` $\rightarrow$ `Start [1]`) synchronize across all connected computers.

### J. Unscramble Tool Desktop & Laptop Full-Width Panel Expansion (`templates/index.html`, `static/css/play.css`)
- Expanded to `1200px` max-width with responsive font clamping so all 21-letter jumbled strings fit on a single line.

---

## 3. Production Deployment Instructions

To synchronize the live server on `morpheme.games` with this exact commit:

```bash
cd /home/morpheme/morpheme && git pull origin main && pm2 restart all
```
