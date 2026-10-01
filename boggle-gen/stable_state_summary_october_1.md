# Stable State Summary — October 1, 2026

## Latest Feature Commit ID
```
747e8b5e9e03d49b2c3c6f4974fdbfe457f920f7
```

## Start Over Point Tag
```
START_OVER_POINT_OCTOBER_1
```

## Synchronization Status

| Environment    | Status | Commit / Notes |
|----------------|--------|----------------|
| localhost      | ✅ Synchronized | `747e8b5e` |
| GitHub (main)  | ✅ Synchronized | `747e8b5e` |
| morpheme.games | ✅ Synchronized | `747e8b5e` (HTTP 200 OK, PM2 Online) |
| App / Client   | ✅ Synchronized | Web client & backend endpoints verified |

---

## Session Features & Fixes (October 1, 2026)

### 1. Hardened Added Words Definition Engine for Regular Plurals (`GUTTUSES`)
- **Loose Web Search Regex Elimination**:
  - Previously, `lookup_web_search_definition` matched unrelated text when querying words ending in `-es` or other suffixes (e.g. matching an eaves trough definition from "gutter" when querying "GUTTUSES").
  - Hardened Match 1 and Match 5 in `lookup_web_search_definition` to enforce `\bThe meaning of [word] is\b` and require the target word to directly precede or accompany the noun definition snippet.
- **Plural & Suffix Resolution Priority in Added Words**:
  - In `ensure_aw_definitions_for_words`, morphological suffix resolution (`-IES`, `-ES`, `-S`, `-ED`, `-ING`, `-ERS`) now runs **first**, guaranteeing that regular plural forms inherit their root definitions directly without unverified web lookups polluting cache or database tables.
- **Validation Pre-check Database Safeguards**:
  - In `get_word_definitions_for_aw_check`, unverified web search results are no longer written to `morpheme.db` during pre-validation checks.

### 2. Comprehensive Classical & Irregular Plural Resolution (`LOCHI`)
- **Classical Latin, Greek, and Irregular Plural Patterns**:
  - Expanded `ensure_aw_definitions_for_words` and `get_definition_cached_or_online_with_guess` to support classical declensions so any irregular or classical plural entered in Added Words inherits its root definition:
    - **`-I` → `-US`**: (e.g. `LOCHI` → `LOCHUS`, `CACTI` → `CACTUS`, `ALUMNI` → `ALUMNUS`, `FUNGI` → `FUNGUS`, `SYLLABI` → `SYLLABUS`)
    - **`-AE` → `-A`**: (e.g. `LARVAE` → `LARVA`, `ALGAE` → `ALGA`, `NEBULAE` → `NEBULA`)
    - **`-A` → `-UM` / `-ON`**: (e.g. `STRATA` → `STRATUM`, `BACTERIA` → `BACTERIUM`, `CRITERIA` → `CRITERION`, `PHENOMENA` → `PHENOMENON`)
    - **`-ICES` → `-EX` / `-IX`**: (e.g. `VERTICES` → `VERTEX`, `MATRICES` → `MATRIX`, `INDICES` → `INDEX`)
    - **`-ES` → `-IS`**: (e.g. `CRISES` → `CRISIS`, `THESES` → `THESIS`, `SYNOPSES` → `SYNOPSIS`, `AXES` → `AXIS`)
- **Pointer Snippet Capture**:
  - Added regex pattern capture in `lookup_web_search_definition` to recognize dictionary pointer snippets (`[word] definition: plural of [target]`).
- **Guaranteed Target Definition Expansion**:
  - In `ensure_aw_definitions_for_words`, if any definition is a pointer (`plural of [target]`, etc.) without an existing parenthetical definition, the engine extracts `[target]`, resolves its definition, and appends it within `(...)` per `AGENTS.md` rules.

### 3. Synchronized Dictionary & Database Entries
- **`GUTTUS` & `GUTTUSES`**:
  - `GUTTUS`: `(noun) A type of ancient Greek and Roman vessel designed for pouring liquids.`
  - `GUTTUSES`: `plural of guttus (A type of ancient Greek and Roman vessel designed for pouring liquids.)`
- **`LOCHUS` & `LOCHI`**:
  - `LOCHUS`: `(noun) In ancient Greece, a body of infantry; in Sparta, one of the larger divisions in which able-bodied men were grouped.`
  - `LOCHI`: `plural of lochus (In ancient Greece, a body of infantry; in Sparta, one of the larger divisions in which able-bodied men were grouped.)`
- Synchronized across `Definitions.txt`, `wikdefs.txt`, `wikdefs_duplicate.txt`, and production `morpheme.db`.

---

## Verification & Health Check

1. **`GUTTUSES` Live Endpoint Verification**:
   - `GET https://morpheme.games/api/definition?word=GUTTUSES`
   - Returns: `"plural of guttus (A type of ancient Greek and Roman vessel designed for pouring liquids.)"`
2. **`LOCHI` Live Endpoint Verification**:
   - `GET https://morpheme.games/api/definition?word=LOCHI`
   - Returns: `"plural of lochus (In ancient Greece, a body of infantry; in Sparta, one of the larger divisions in which able-bodied men were grouped.)"`
3. **Plural Resolution Engine**:
   - Tested across Latin/Greek and irregular plural forms (`ALUMNI` → `ALUMNUS`, `CACTI` → `CACTUS`, `LARVAE` → `LARVA`, `CRISES` → `CRISIS`, `VERTICES` → `VERTEX`).
4. **Room States & Server Health**:
   - `GET https://morpheme.games/api/lobby-stats` → HTTP 200 OK.
   - `GET https://morpheme.games/api/room/pub_v2_accumulative_5x7_86400/state` → HTTP 200 OK.
   - PM2 Process `morpheme` online and responsive.
