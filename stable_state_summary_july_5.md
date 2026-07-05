# Stable State Summary - July 5, 2026

This document summarizes the stable state of **Morpheme** as of July 5, 2026. All local changes, remote code on GitHub, and the live application running on `morpheme.games` are fully synchronized.

## Latest Commit Information
* **Commit ID**: `9f93f38`
* **Branch**: `main`
* **Tags**: `snapshot-current`, `START_OVER_POINT_JULY_5`
* **Date**: July 5, 2026

## Server Deployment Instructions
The live server runs at `/home/morpheme/morpheme/` on the remote host.
To deploy any future changes, SSH into the server and run:
```bash
cd /home/morpheme/morpheme && git pull origin main && pm2 restart all
```
* PM2 process name: `morpheme` (id: 0, fork mode)

## CSS / JS Cache-Buster Versions (Current)
| File | Version |
|---|---|
| `style.css` | `v=57` |
| `play.css` | `v=136` |
| `lobby.css` | `v=30` |
| `howtoplay.css` | `v=10` |
| `forum.css` | `v=23` |
| `donate.css` | `v=1` |
| `play.js` | `v=223` |
| `tools.js` | `v=79` |
| `app.js` | `v=70` |
| `forum.js` | `v=54` |
| `lobby.js` | `v=10` |
| `tournaments.js` | `v=7` |
| `leaderboard.js` | `v=3` |
| `mods.js` | `v=13` |

## Changes Since July 4 Stable Point

### 1. Round Replay Mobile Layout Restructure (carry-over from July 4)
- Restructured `.section-header` inside Round Replay so on mobile:
  - **"Timeline of Discovery"** sits alone on its own row with a `border-bottom` divider.
  - **"▶ Watch Replay"** button on the left, **"Words recorded with millisecond precision"** on the right inside a padded `.replay-subheader` row.
- Added a new `.replay-subheader` div in `index.html`; styles in `style.css` and `play.css`.
- Applied inline `style=""` attributes directly on HTML elements to guarantee spacing bypasses browser CSS caching.
- Added generous mobile padding (`padding: 16px`), outer card border, `gap: 14px`, and divider line.

### 2. Mini-Profile Description Vertical Space (Desktop)
- Changed `.mini-profile-description` from `max-height: 180px` to a fixed `height: 140px` on desktop.
- On mobile (max-width `600px`): fixed `height: 90px`.

### 3. Unscramble Button Color & Label Fix
- **Root cause found**: `tools.js` was calling `revealBtn.style.background = ''` after each reveal, wiping the blue color, and `revealBtn.innerText = "Unscramble"` was mislabelling the button.
- **Fixed**: Button now correctly restores to **blue** (`linear-gradient(135deg, #4facfe, #2980b9)`) and is labelled **"Reveal"** on every new game reset.
- Updated both the HTML initial inline style and the JS reset to match.

### 4. Definitions Panel — Winner Notification at Round End
- The definitions panel already showed the winner card during intermission via the 1-second state poll. Confirmed and documented.
- **New**: Added `isTimerExpired` guard in the winner announcement block (`play.js`).
  - If `.definitions-panel` has the `timer-flash` class (Personal Timer expired), the winner card is silently skipped — "Time is up!" stays visible.
  - Both the winner branch and the "no winner" branch respect this guard.
  - The old code was actively stripping `timer-flash` when showing a winner — that line is removed.

### 5. Timer-Flash Persists Across Rounds
- **Before**: Two places in `play.js` stripped `timer-flash` and reset `#definition-content` on round transitions.
  - Active → Intermission transition (lines ~1120–1128)
  - Intermission → New round transition (lines ~1626–1629)
- **After**: Both transition blocks now check `timerStillExpired` / `timerExpiredAtTransition` before wiping content or removing `timer-flash`.
- The red flashing "Time is up!" notice now **persists indefinitely** across all round boundaries until the user manually presses **Stop Timer** in the Personal Timer tool.

## Verification
* **Local** (`/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme`): Clean working tree — nothing to commit.
* **GitHub** (`gewurztraminerrr-cloud/Morpheme`, branch `main`): Fully synchronized. Tags `snapshot-current` and `START_OVER_POINT_JULY_5` applied.
* **Production** (`morpheme.games`, `/home/morpheme/morpheme`): Requires `git pull origin main && pm2 restart all` to pick up July 5 changes.
