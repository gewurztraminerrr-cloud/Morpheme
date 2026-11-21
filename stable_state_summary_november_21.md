# Stable State Summary - November 21, 2026

## 1. System Synchronization Status
- **Date**: November 21, 2026
- **Stable Point Tag**: `stable-2026-11-21`
- **Latest Commit ID**: `12cd9b5c33f194b744334a9a8a0e1427944d1380`
- **Synchronization State**: **100% Byte-for-Byte Synchronized**
  - **Localhost**: Commit `12cd9b5c33f194b744334a9a8a0e1427944d1380` (Clean working tree)
  - **GitHub (`origin/main`)**: Commit `12cd9b5c33f194b744334a9a8a0e1427944d1380`
  - **Production VPS (`morpheme.games` / `132.148.72.249`)**: Commit `12cd9b5c33f194b744334a9a8a0e1427944d1380` (PM2 Process 0 online, 0 restarts)

---

## 2. Issues & Enhancements Addressed

### Feature 1: Live Found / Total Ratios & CSW Dropdown Option in Game Rooms (`CSW.jpg`)
- **Problem / Need**:
  - In `CSW + AW` rounds, players only had `CSW + AW` and `AW Only` available in the word list dropdown; `CSW Only` was missing as an option.
  - Dropdown options displayed only the total count (e.g., `CSW Only (85)`), rather than how many had been found during the round.
- **Solution & Implementation**:
  - In [`static/js/play.js`](file:///Users/jeffbabiak/static/js/play.js#L6870-L6930): Added `csw_only` as a selectable dropdown filter for `CSW + AW` rooms.
  - Formatted all dropdown list labels dynamically to display the exact live found ratio: **`CSW Only (43 / 85)`**, **`AW Only (12 / 30)`**, etc.

---

### Feature 2: FC (Find Count) Button Bracketed Counts & Adaptive Typography
- **Problem / Need**:
  - In **Is Valid** and **Random Word**, users could not see how many players had found a searched/revealed word without navigating to the Find Count tool.
  - In Definition Popups, large finder counts risked wrapping or cluttering the footer link action bar.
- **Solution & Implementation**:
  - In [`static/js/tools.js`](file:///Users/jeffbabiak/static/js/tools.js#L8382-L8418) (`setupWordDefActionButtons`): Added dynamic retrieval from `/api/word_tally/<word>` to display finder counts directly on the button: e.g., **`FC (153)`** or **`FC (0)`**.
  - In [`static/js/tools.js`](file:///Users/jeffbabiak/static/js/tools.js#L11790-L11825) (`showWordDefinitionPopup`) & [`static/js/play.js`](file:///Users/jeffbabiak/static/js/play.js#L12780-L12855) (`fetchDefinition`):
    - Wrapped count spans in `<span class="fc-btn-count">` with a scaled font size (`0.82em`, `font-weight: 600`).
    - Added `white-space: nowrap;` to `.tool-def-footer-link` in [`static/css/play.css`](file:///Users/jeffbabiak/static/css/play.css#L18257-L18265) to guarantee clean fit for multi-digit counts up to 5 figures without clipping or breaking lines.
    - Format for multiplayer/solo rooms: **`FC [roundFinders / totalCombined]`**; format for 24H and Tools popups: **`FC (count)`**.

---

### Feature 3: Wikipedia Botanical Disambiguation & Proper Noun Filtering
- **Problems / Needs**:
  - `ACER`: Disambiguation page listed numerous people, companies, and agencies before the botanical plant.
  - `AMOLE`: Disambiguation page listed people before plants of the genera *Chlorogalum* and *Agave*.
  - `SAMMO`: Wikipedia only has an article for the actor/martial artist *Sammo Hung*, which leaked into the common noun entry for the sandwich.
  - `CAINITO`: A tropical star apple tree and fruit ([`CAINITO.jpg`](file:///Users/jeffbabiak/.gemini/antigravity/brain/1d68eff8-1658-4346-9eae-f03c6f6b5147/CAINITO.jpg)) was not recognized as a noun due to strict part-of-speech patterns.
- **Solution & Implementation**:
  - In [`app.py`](file:///Users/jeffbabiak/app.py#L11175-L11235):
    - **Universal Disambiguation Anchor Parser**: Replaced restrictive plain-text regex with universal anchor matching, allowing Wikipedia entries with formatting (such as `<i>Acer</i> (plant)`) to be properly extracted.
    - **Stemmed Keyword Matching**: Applied stem-matching (`plant`, `soap`, `genus`, `maple`) so singular and plural definitions cleanly align with Wikipedia bullet text.
    - **Proper Noun Filtering**: Expanded `_PROPER_NOUN_INDICATORS` to include `martial artist`, `choreographer`, `filmmaker`, `director`, `actor`, etc. If an article is a proper noun with no common noun equivalent, the Proper Noun Filter activates (`is_proper_only: true`), shielding users from irrelevant biographies.
    - **Botanical Produce & Inflection Detection**: Expanded `_NOUN_DEF_REGEX` to include `fruit`, `berry`, `nut`, `vegetable`, `produce`, `seed`, `crop`, `herb`, `fungus`, `root`, and plural inflection tags `[inflections: ...S]`.
    - **Multi-Word Cross-Reference Pointers**: Updated `_CROSS_REF_POINTER_REGEX` to resolve multi-word targets like `star apple`.
  - **Live Verification**:
    - **`ACER`**: Resolves directly to **`Acer (plant)`** ([`Maple`](https://en.wikipedia.org/wiki/Maple)) with description *"Genus of flowering plants"* and photos of maple leaves and samaras.
    - **`AMOLE`**: Resolves directly to **[`Chlorogalum`](https://en.wikipedia.org/wiki/Chlorogalum)** with description *"Genus of flowering plants"* explaining its use as soap with photos of the soaproot.
    - **`SAMMO`**: Activates the Proper Noun Filter, cleanly stating that Wikipedia articles refer to proper nouns rather than the informal sandwich noun.
    - **`CAINITO`**: Resolves to **[`Chrysophyllum cainito`](https://en.wikipedia.org/wiki/Chrysophyllum_cainito)** matching `CAINITO.jpg`, showing the tropical star apple tree and fruit.

---

### Feature 4: Tally Reset & Single Source of Truth Hardening
- **Problems / Needs**:
  - Legacy test strings (`DOTRS`, `PORET`, `BAFEF`, `LESEA`, `AOSYR`, etc.) and desynced counts (`MONTE (1)`, `BANDA (2)` vs 0 in FC) caused discrepancies between Find Count and Tally Divisions.
- **Solution & Implementation**:
  - Preserved all code logic and created full timestamped backups (`.backup_20261121_154438`) on localhost and VPS.
  - Cleared database records in `word_finds`, `round_history`, `tournament_scores`, `private_match_turns`, and `daily_score_sums` while preserving all 24,354 user accounts, forum categories, private messages, and settings.
  - Disabled legacy snapshot scripts (`scratch/backfill_all_word_finds.py`) to prevent resurrection of unvalidated words.
  - Established `word_finds` and `word_stats.json` as the unified source of truth for both Find Count (Tools) and Tally Divisions (Mods).

---

## 3. Server Health & Environment
- **Server**: `morpheme.games` (`132.148.72.249`)
- **RAM**: Upgraded to 4 GB (`3,916 MB` total, ~1.9 GB RSS allocated, 0.00 load average)
- **Process Manager**: PM2 Process 0 (`morpheme`), online, 0 unexpected restarts.
