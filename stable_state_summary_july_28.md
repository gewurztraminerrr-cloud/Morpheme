# Morpheme Engine — Stable State Summary (July 28, 2026)

This document records the official **July 28, 2026 Save Point / Start Over Snapshot** for Morpheme (`morpheme.games`). All changes, fixes, and invariants are fully synchronized across localhost, GitHub (`origin/main`), and production servers.

---

## Key Highlights & Resolved Issues

### 1. Spinner Set Odds Window Alignment & Probability Engine
- **Odds Window Percentages**: Re-aligned `spinner_set.py` weights and array configurations to match the in-game **Spinner Set Odds Window** with 100% precision:
  - **Normal**: 66%
  - **Checkerboard**: 12%
  - **Equality Freq**: 4%
  - **Bounce**: 2%
  - **Density**: 2%
  - **[Letter] Mania**: 2% (draws uniformly across vowels [33%] and consonants [67%])
  - **Penalty**: 2%
  - **Either/Or**: 2%
  - **Bonus Letter**: 2%
  - **Valued Letters**: 2%
  - **Rotation**: 2%
  - **Double**: 1%
  - **Triple**: 1%
- **Difficulty Distribution**: Easy (25%), Medium (50%), Hard (25%).
- **Dictionary Distribution**: NWL (25%), CSW (25%), NWL + AW (25%), CSW + AW (25%).
- **Word Count Range**: Standard (50-100: 9%, 100-200: 30%, 200-300: 30%, 300-400: 30%, 500+: 1%) | + AW (300-400: 33%, 400-500: 33%, 500+: 34%).

### 2. Parameter Locking & Single Update Per Intermission Rule
- **Single Reveal Lock**: At 0:45 intermission timer, parameter generation runs **EXACTLY ONCE** and locks `_spinner_params_locked = True`.
- **Zero Mid-Intermission or Mid-Round Shifts**: Guarded all `room.spinner_params` assignments behind `if not getattr(room, '_spinner_params_locked', False):`.
- **Immutability Guarantee**: Once revealed at 0:45, `room.spinner_params` remains 100% frozen through the end of the round. No parameter shifts occur mid-intermission, at 0:00 round start, or 5 seconds into active play.

### 3. Spun Board Format Preservation
- **No Format Overwrites**: Replaced fallback poppers with format-preserving emergency board resolution.
- If a spun format (e.g. `Mania`, `Bounce`, `Checkerboard`, `Density`, `CSW + AW`) is not pre-cached in SQLite, the engine dynamically generates a fresh board matching that **exact format** on the fly instead of downgrading to `Normal`.

### 4. Bonus Word Length Exact Synchronization & Validation
- **Exact Staging**: Bonus word length (`bonus_word_length`) is automatically set to `len(final_bonus_word)` at 0:45 reveal time and room creation.
- **Round Start Validation**: Updated `start_next_round()` to validate `next_round_bonus` against `room.next_round_words` (the new board's words) instead of the old round's word set. This prevents valid bonus words (like `PACKAGE` 7L) from being discarded at 0:00, guaranteeing that the bonus word revealed at 0:45 **never changes** when the round starts.
- **Strict 6-10L Minimum Range**: Enforced `6 <= len(w) <= 10` for all bonus word candidates, preventing 5-letter words (e.g. `MOODS`) from ever becoming bonus words.

### 5. Intermission "All Words" List Complete Preservation
- **Payload & Display Integrity**: Updated `app.py` and `static/js/play.js` so that during intermission, `previous_all_words` and `words_to_return` directly return the completed round's full word list (`room.previous_all_words`) without secondary min-length filtering.
- **No Empty Word Lists**: Eliminates `"No words found"` placeholding messages at the end of rounds, ensuring the completed round's full word list is **always rendered 100% accurately**.

### 6. Active Round Factual Display Policy
- **Factual Active Headers**: Updated `static/js/play.js` so that `preferSp` is `false` during active play (`!isIntermission`). The header bar during active play strictly displays the **factual ground-truth parameters** of the board being played (`state.current_word_count_range`, `state.current_min_length`, `state.current_dictionary`), ensuring header labels and grid contents match with 100% precision.

---

## Synchronization Verification Command

To update `morpheme.games` or any remote environment to this exact stable state point:

```bash
git stash && git pull origin main && pm2 restart all
```

---
*Snapshot Created: July 28, 2026*
