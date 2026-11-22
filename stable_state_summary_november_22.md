# Morpheme Stable State Summary — November 22, 2026

## 1. System Synchronization State
- **Date**: Sunday, November 22, 2026
- **Latest Commit ID**: `44d25270a26ac01ef15f94d19ea1693d6734af9d`
- **Git Tag**: `stable-2026-11-22` (and `stable-2026-11-21` updated)
- **GitHub Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme` (`main` branch)
- **Live Production URL**: `https://morpheme.games/` (HTTP 200 OK)
- **VPS PM2 Process**: Online (Process ID 0, 0% CPU, normal memory)
- **Byte-for-Byte Synchronization**:
  - **Localhost**: Commit `44d25270a26ac01ef15f94d19ea1693d6734af9d`
  - **GitHub (`main`)**: Commit `44d25270a26ac01ef15f94d19ea1693d6734af9d`
  - **Production VPS (`morpheme.games`)**: Commit `44d25270a26ac01ef15f94d19ea1693d6734af9d`

---

## 2. Key Changes & Features Completed in this Stable State

### A. Multiplayer Subanagrams Dynamic Spinner Set Randomization (`game_room.py`, `spinner_set.py`)
- **Joint Odds Window Enforcement**:
  - Sequence lengths 6, 7, 8, 9, 10 (20% probability each).
  - Word count ranges 50–100, 100–200, 200–300, 300–400, 500+ (20% probability each).
  - Dictionaries NWL, CSW, NWL + AW, CSW + AW (25% probability each).
- **Anti-Repetition Across Consecutive Rounds**:
  - Sequence length, dictionary, and word count range are guaranteed never to repeat back-to-back across consecutive rounds.
- **Fixed Intermission Watchdog Fallback**:
  - Removed logic in `ensure_next_board_ready` and `generate_spinner_params` that previously fell back to reusing `self.spinner_params` (the completed round's parameters).
  - If next round parameters are missing or repeat the completed round, fresh distinct parameters are generated automatically.
- **Instant Next-Round Pre-Generation**:
  - Subanagrams next-round generation (~5ms) is immediately dispatched at round transition via `subanagrams_instant_pregen`, staging the upcoming board and parameters well in advance of the 0:45 reveal.

### B. Subanagrams Word Count Range & Added Words Isolation (`static/js/play.js`, `game_room.py`)
- **Resolved "300-400 words displayed for 122 word board" Bug**:
  - Standard Boggle public room rules enforce `300-400` words whenever Added Words (`+ AW`) is active.
  - In `static/js/play.js`, guarded the Added Words override with `if (useAW && !isSubanagrams)` so Subanagrams displays its true selected word count range (e.g. 50–100, 100–200, 200–300, 300–400, 500+).
  - In `game_room.py`, isolated `generate_spinner_params` so Subanagrams does not run `_get_factchecked_wc_range` with the Boggle AW override.
- **Strict Compliance Guard**:
  - Guaranteed that boards generated in Subanagrams strictly fall within the chosen word count range, and the 500+ range requires at least 500 words.

### C. Moderator Tools Enhancements (`templates/moderation.html`, `app.py`)
- **Definition Management**:
  - Re-positioned the table of words without definitions below the input controls.
  - Added a dedicated Refresh button above the table that re-queries words missing definitions.
- **Pronunciation Management**:
  - Added a table of words without pronunciations below the input controls.
  - Added a dedicated Refresh button above the table.
  - Defaulted the initial view to 5-letter words so moderators start reviewing at 5L words immediately.
- **Theme Support**:
  - Ensured all moderation tables are readable and styled cleanly across all color themes: Default, Dark, Light, and White.

### D. Architectural Integrity & Strict Boundaries
- **Multiplayer Subanagrams Isolation**:
  - All Subanagrams changes are strictly conditioned on `game_type == 'subanagrams'`. Standard public rooms (`accumulative`, 4x4, 4x6, 6x8, Split, 3D, Daily) remain completely untouched and unmodified.
- **Server Performance & Resource Footprint**:
  - CPU usage on the VPS remains at 0–1% at idle.
  - Clean PM2 process management without hanging threads or memory leaks.

---

## 3. Verification & Live Status
- **20-Round Simulation Test**:
  - Executed a 20-round consecutive simulation test locally; verified 0 parameter repeats across rounds and 100% word count range compliance.
- **Live Production Room State (`pub_v2_subanagrams_subanagrams_120`)**:
  - **Round 1**: 7 letters (`ESBOTLA`), 207 words, Range: `200-300`, NWL, Totally Random.
  - **Round 2 (Transitioned)**: 10 letters (`RNTAASIEPG`), 653 words, Range: `500+`, NWL, Word Guaranteed.
  - Verified live via API and browser rendering.
