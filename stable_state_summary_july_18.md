# Stable State Summary - July 18, 2026

This document summarizes the stable state of **Morpheme** as of July 18, 2026. All local changes, remote code on GitHub, and the live application running on `morpheme.games` are fully synchronized.

## Latest Commit Information
* **Commit ID**: `09479e5`
* **Branch**: `main`
* **Tags**: `snapshot-current`, `START_OVER_POINT_JULY_18`
* **Date**: July 18, 2026

## Server Deployment Instructions
The live server runs at `/home/morpheme/morpheme/` on the remote host.
To deploy any future changes, SSH into the server and run:
```bash
cd /home/morpheme/morpheme && git pull origin main && pm2 restart all
```
* PM2 process name: `morpheme` (id: 0, fork mode)

## Changes Since July 5 Stable Point

### 1. Cache Pop Word Count Validation Check (Issues 1 & 4)
- **Problem**: When entering a room or transitioning to intermission, the server would stage cached boards that had sparse filtered word lists (e.g. only 19 or 37 words matching the minimum length floor), leading to empty/very sparse rounds while the UI claimed `400-500` or `500+` targets.
- **Solution**: Added strict validation check loops across all cache popping pathways (lobby creation kickstart, generate_spinner_params, and ensure_next_board_ready). The server now iterates through candidates and discards any popped cached board whose actual scorable count (after minimum length filtering) does not meet the target word count floor.

### 2. Strict Word Count Floor for `+ AW` Configurations (Issue 3)
- **Problem**: Added Words configurations were spinning or falling back to low density ranges (e.g. `50-100` or `100-200`) which is not allowed.
- **Solution**: Raised the minimum acceptable words (`min_accept`) for any round using a `+ AW` dictionary configuration from 100 to 300 across all validation checks. Corrected the `SpinnerSet` range spinner mapping so `+ AW` automatically resolves `50-100` ranges to `300-400`.

### 3. Conditional Reveal Timing in `ensure_next_board_ready` (Issue 2)
- **Problem**: At the transition from active to intermission (`1:00` remaining), the watchdog fallback `ensure_next_board_ready()` was unconditionally setting `spinner_params_revealed = True`. This caused the client UI parameters to change prematurely at `1:00` instead of the designed `0:45` mark.
- **Solution**: Updated `ensure_next_board_ready()` to only set `spinner_params_revealed = True` if the current elapsed time is actually past the reveal threshold (15s elapsed, or `0:45` remaining). Otherwise, it keeps the flag `False` and allows the normal proactive reveal workflow to execute the change exactly at `0:45`.

### 4. Constant Parameters Loop Fix
- **Problem**: Every intermission transition immediately triggered `ensure_next_board_ready()` because `next_round_board` was `None`. This popped a random board, overrode the spinner parameters, and set `spinner_params_generated = True` before proactive generation or search could even start, forcing the parameters to stay stuck on the same cache-refilled configuration.
- **Solution**:
  - Removed the premature `ensure_next_board_ready()` call from the ACTIVE -> INTERMISSION transition, so the room enters intermission correctly and runs the normal generation and search workflow.
  - Removed the `pop_any_cached_board` fallback from the proactive `generate_spinner_params` method. If there is a cache miss for the spun parameters, the room preserves the spun parameters, and `start_board_search` runs in the background to search/generate a matching board. This restores full random variation to the Spinner Set.

### 5. Intermission Rescue Watchdog Disablement
- **Problem**: The intermission rescue watchdog was triggering at `15s` remaining (45s elapsed in intermission) when the background board generator thread had not finished generating a matching board. When it triggered, it forcefully popped a fallback board from the cache and updated the spinner parameters (e.g. word count range) to match it. This caused the parameters on the client to change and highlight gold at `0:15` remaining.
- **Solution**: We completely disabled the intermission watchdog's `15s` remaining check. The background generator is now allowed to run for the entire intermission. If it still hasn't completed by `0s` remaining, `start_next_round` seamlessly handles it via its own last-chance fallback cache pop. This completely prevents mid-intermission parameter changes, keeping them constant after the `0:45` mark.

### 6. Instantaneous Transition (Removed Wait Loop)
- **Problem**: At the end of intermission (`0:00` remaining), `start_next_round` was entering a loop that slept/waited for up to `1.5` seconds if the background generator had not yet finished staging the next board. Combined with the fallback cache pop lookup and startup delay, this created a noticeable 2-3 second delay reading "WAIT..." before the new round started.
- **Solution**: We removed the `1.5` second sleep/wait loop from `start_next_round`. Since the background generator has the entire 60-second intermission to build the board, if it isn't ready at the exact `0:00` mark, we instantly proceed to the cache pop fallback. The sqlite cache query executes in a fraction of a millisecond, starting the round instantly.

## Verification
* **Local** (`/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme`): Fully clean working tree. Unit test suite passes completely.
* **GitHub** (`gewurztraminerrr-cloud/Morpheme`, branch `main`): Fully synchronized up to commit `09479e5`. Tags `snapshot-current` and `START_OVER_POINT_JULY_18` successfully pushed.
* **Production** (`morpheme.games`, `/home/morpheme/morpheme`): Fully updated and restarted under PM2.
