# Morpheme Project Checkpoint - March 24, 2026 (5:10 PM)

This checkpoint represents the successfully restored and stabilized state of the Morpheme application after a series of significant UI and logic refinements on March 24th.

## Core Features Captured:

### 1. Game Fairness & Scoring Refinements (Commit f93afa4)
- **Mid-Round Joiner Exclusion**: Dynamic logic ensures that players joining after the round start do not impact the ratings of original participants.
- **Selective Reward/Penalty Pools**: Scoring calculations now distinguish between full-participation "competitive" players and "casual" mid-round joiners.
- **Abandonment Detection**: Implemented soft abandonment detection to ensure fair bounty distribution when players exit mid-round.

### 2. UI & Aesthetic Excellence (Commit 658c4b1)
- **Spinner Set Window Congratulatory Style**: Improved accessibility for the "CONGRATULATIONS" message with a dark, high-contrast background and gold text flash (visible across all themes).
- **Profile UI Background Contrast**: Enhanced profile cards for improved readability on diverse user-selected color palettes.
- **Achievement Limits**: All achievement/ranking tables are now capped at 50 rows to ensuring UI stability and performance.
- **Simplified Profile Filters**: Redundant time-period filters (Day/Week/Month/Year) have been removed from Round Reviews and Exceptional Rounds to declutter the interface.

### 3. Server Stability & AI Support
- **Stable Flask Environment**: Debug mode (`debug=False`) and the reloader (`use_reloader=False`) have been disabled to prevent port conflicts and improve startup reliability.
- **Port 3000 Standard**: The server is cleanly bound to port 3000 and verified to be responding.
- **500+ Word Count Logic**: Iterative board optimization for high-word-count rounds (500+) is active.

### 4. Multiplayer Bug Fixes
- **Private Match Initialization**: Fixed a critical variable initialization bug in `private_match_logic.py` that caused "With Friends" matches to fail creation silently and become stuck at `current_round = 0`. Users can now successfully invite friends and start private matches.

### 5. Database Persistence
- **Snapshot Created**: The current `morpheme.db` has been snapshotted to `morpheme.db.save_point_2026-03-24_final`.

### 5. Start Over Workflow (Hardened)
- **Automated Restoration**: The `.agent/workflows/start_over.md` has been updated to point to this March 24th baseline.
- **Database Injection**: The workflow now automatically restores the Synesthesia database snapshot, ensuring personal colors are never lost during a reset.

## Restore Procedure:
To return to this EXACT state, run:
1. `git reset --hard save_point_2026-03-24_STABLE`
2. `git clean -fd`
3. `cp morpheme.db.save_point_2026-03-24_final morpheme.db`
4. `./run_morpheme.sh`
