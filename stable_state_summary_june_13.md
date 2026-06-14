# Stable State Summary — June 13, 2026 (Night)

**Commit:** `6cf6233`
**Tags:** `START_OVER_POINT_JUNE_13` / `snapshot-current`
**Date:** June 13, 2026 (22:11 CT)
**Status:** All platforms synchronized — localhost, GitHub, morpheme.games

---

## Changes Since Last Stable Point (June 13 Evening)

### 1. Leaver Rating — Score-Based Proportional System
- **Files:** `app.py`, `game_room.py`
- Replaced flat **-16 abandonment penalty** with a fair proportional system:
  - Player's score at the moment they leave is preserved in `round_quitters`
  - At round end, that score drives their proportional rating change (same formula as stayers)
  - A small **+8 disruption bounty** is added to the pool for remaining players
- **If the player returns before the round ends:**
  - Their full score and submitted words are restored — they pick up exactly where they left off
  - Their `joined_mid_round` flag stays `False` (they're still a starter)
  - The +8 bounty is reversed (stayers don't get a windfall for someone who came back)

### 2. Critical Bug Fixed: round_quitters Cleared Before Async Rating Ran
- **File:** `game_room.py`
- `self.round_quitters = []` was executed inside the state lock BEFORE `process_results_async()` launched
- Quitters were never actually included in the rating calculation — this bug predated this session
- **Fix:** Snapshot `quitters_snapshot = list(self.round_quitters)` before clearing, pass snapshot to async thread

### 3. Abandonment Bounty Cap
- **File:** `game_room.py`
- Bounty was being added on top of the proportional change with no ceiling, causing +22 instead of max +16
- **Fix:** Before distributing bounty, compute headroom = `cap - current_rating_change`; bonus = `min(bonus, headroom)`
- Caps enforced: Normal ±16 · Double ±32 · Triple ±48

---

## All Changes This Session (Full June 13)

| # | Change | Files |
|---|--------|-------|
| 1 | Transition latency fix (fallback boards) | `game_room.py` |
| 2 | Mobile Forum auto-scroll | `static/js/forum.js` |
| 3 | ENTER LOBBY button tactile press-down | `static/css/style.css`, `static/js/app.js` |
| 4 | Tool loading text 1-3 minutes | `static/js/tools.js` |
| 5 | Mid-round join rating protection (root cause fixed) | `game_room.py` |
| 6 | Leaver proportional rating + rejoin score restore | `game_room.py`, `app.py` |
| 7 | Bounty cap ≤ ±16/32/48 | `game_room.py` |

---

## Platform Sync Status
| Platform | Commit |
|----------|--------|
| localhost | `6cf6233` |
| GitHub (origin/main) | `6cf6233` |
| morpheme.games (132.148.72.249) | `6cf6233` |

---

## Infrastructure
- Server: Ubuntu 24.04 @ 132.148.72.249
- Process manager: PM2 (`morpheme` process, fork mode)
- Deploy script: `boggle-gen/scratch/deploy.py`
