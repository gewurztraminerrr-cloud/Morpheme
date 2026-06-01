# Stable State Summary — June 1, 2026 (Updated Save Point)

## Snapshot Commit & Save Point

| Environment | Commit | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `948f38d` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `948f38d` | ✅ Pushed & Synchronized |
| **morpheme.games** (production) | `948f38d` | ✅ Fully Deployed & PM2 Reloaded |

**All environments are 100% synchronized at the latest stable commit `948f38d`.**
The local modifications have been committed, pushed to remote, and successfully deployed to the remote production environment via `deploy.py` (which completed a repository hard reset and successfully restarted all PM2 processes).

---

## Serving Versions (cache-busted)

| File / Style | Version | Description |
|--------------|---------|-------------|
| `/js/play.js` | `v=113` | Instant tab swapping, zero-hesitation local validation (authoritative State bypass) |
| `/js/lobby.js` | `v=5` | Rating filter proximity sorting and search listener registrations |
| `/css/lobby.css` | `v=17` | Premium flex rating search bar layout and Find button style |
| Inline `<style>` block | *Dynamic* | Mobile WebView cache bypass styling in `index.html` |

---

## Work Completed This Session (June 1 Fixes)

### 1. Active Gameplay Validation Flashing Fix
* **Goal achieved:** Completely resolved active gameplay double-flashing. When a new round starts, a valid word immediately flashes only the correct color (blue/green) with zero delay or temporary red flashes.
* **Authoritative Checking (`play.js`):**
  * Sourced active validation directly from the authoritative `preState.all_words` array rather than relying on the stale intermission render cache (`window.lastDisplayAllWordsArgs`). 
  * This guarantees that new round valid words are immediately evaluated correctly locally without any interference from the previous round's finished words.

### 2. Backend Guest Player History Persistence (24h Rooms)
* **Goal achieved:** Ensures that guest accounts participating in 24-hour rooms have their yesterday's found words successfully stored in history and retrieved under the "Previous Day" tab, instead of seeing `FOUND (0)` and `None`.
* **Snapshot Retention (`game_room.py`):**
  * Updated `start_next_round` ([game_room.py:L3622](file:///Users/jeffbabiak/game_room.py#L3622)) to capture snapshots for both registered players AND guest players (`p.is_guest`) if the room is a 24-hour room (`room.time_limit >= 7200`).
* **Round History Persistence (`game_room.py`):**
  * Updated `save_round_history` ([game_room.py:L4288](file:///Users/jeffbabiak/game_room.py#L4288)) to include guest players (`p.is_guest`) in the participating players list when saving historical records for 24-hour rooms. This ensures guest players' found words are written to the database `round_history` table at the end of a round.
* **Username Mapping Recovery (`game_room.py`):**
  * Added a robust check in `get_yesterdays_history` ([game_room.py:L2375](file:///Users/jeffbabiak/game_room.py#L2375)) for negative guest IDs (`uid < 0`) to reconstruct guest usernames as `Guest_{abs(uid)}`. This guarantees that client-side session-to-history matching succeeds even for guest users who do not reside in the permanent `users` database table.

---

## Key Files Modified

| File | Location | Purpose |
|------|----------|---------|
| `game_room.py` | Production + GitHub | Include guest players in snapshots and complete round saves for 24h rooms, and correctly map guest usernames |
| `static/js/play.js` | Production + GitHub | Sourced active validation check from `preState.all_words` directly to resolve validation double-flashing |
| `templates/index.html` | Production + GitHub | Bump play.js version parameter from `112` to `113` for WebView cache busting |

---

## Previous Save Points

* [May 31 Stable State Summary](file:///Users/jeffbabiak/.gemini/antigravity/brain/3cc91699-cf5c-405f-9bbd-1934305e8305/stable_state_summary_may_31.md)
* [May 30 Stable State Summary](file:///Users/jeffbabiak/.gemini/antigravity/brain/3cc91699-cf5c-405f-9bbd-1934305e8305/stable_state_summary_may_30.md)
* [May 29 Stable State Summary](file:///Users/jeffbabiak/.gemini/antigravity/brain/3cc91699-cf5c-405f-9bbd-1934305e8305/stable_state_summary_may_29.md)
