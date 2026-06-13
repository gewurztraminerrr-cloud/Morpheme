# Stable State Summary — June 13, 2026

**Commit:** `3c52486`
**Tag:** `START_OVER_POINT_JUNE_13` / `snapshot-current`
**Date:** June 13, 2026
**Status:** All platforms synchronized — localhost, GitHub, morpheme.games

---

## Changes Since Last Stable Point (June 13 AM)

### 1. Transition Latency Fix
- **File:** `game_room.py`
- Eliminated synchronous `generate_board` calls during round transitions
- Replaced with `get_emergency_fallback_board` for instant board delivery
- Fixes: "WAITING..." delay and "Generating Next Board" loading lag at round start

### 2. Mobile Forum Auto-Scroll
- **File:** `static/js/forum.js`
- On mobile (≤820px), when a category is selected the page auto-scrolls
  so the user sees the category title and thread list immediately
- Scroll triggered in `showListView`, `showPostView`, and `showCreateView`

### 3. ENTER LOBBY Button — Blue Tap Highlight Fix
- **Files:** `static/css/style.css`, `static/js/app.js`
- Added `-webkit-tap-highlight-color: transparent` to kill iOS blue flash
- Added `outline: none` to remove focus ring on tap
- Added `touchstart`/`touchend`/`touchcancel` JS listeners to toggle `.pressed` class
  (bypasses iOS Safari's unreliable CSS `:active` without a touch listener)
- `.pressed` CSS class sets `animation: none` so `pulseGlow` fully collapses on press
- Result: clean physical press-down effect — glow collapses, button dims, sinks

### 4. Tool Loading Description Update
- **Files:** `templates/index.html`, `static/js/tools.js`
- Changed "2-4 minutes" → "1-3 minutes" for list load time
- Lowered automated timeout warning threshold to 3 minutes (180s)

### 5. Mid-Round Join Rating Protection — Root Cause Fixed
- **File:** `game_room.py`
- **Root cause found and fixed:** `joined_mid_round` flag was not being set
  correctly because `last_active` was overwritten with `time.time()` BEFORE
  the round-start comparison read it — so the check always saw "now" and
  concluded the player was present at round start
- **Fix:** Snapshot `prior_last_active` before updating `last_active`, then
  compare snapshot against `round_start_time`
- Also fixed: 15-second refresh grace period loophole (removed entirely)
- Also fixed: Quitter rejoin path never re-set `joined_mid_round = True`
- Result: Players who join mid-round truly get `rating_change = 0` and have
  zero effect on other players' ratings

---

## File Versions (Cache Busters)
| File | Version |
|------|---------|
| `style.css` | v45 |
| `app.js` | v43 |
| `forum.js` | v52 |

---

## Platform Sync Status
| Platform | Commit |
|----------|--------|
| localhost | `3c52486` |
| GitHub (origin/main) | `3c52486` |
| morpheme.games (132.148.72.249) | `3c52486` |

---

## Infrastructure
- Server: Ubuntu 24.04 @ 132.148.72.249
- Process manager: PM2 (`morpheme` process, fork mode)
- Deploy script: `boggle-gen/scratch/deploy.py`
