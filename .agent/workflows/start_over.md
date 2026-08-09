---
description: Revert the project to the August 9, 2026 save point. Use this when the user says "Start Over".
---

### Revert Procedure

1.  **Reset Repository to Save Point:**
    - `git reset --hard 2598183`

2.  **Clean Up Untracked Files:**
// turbo
    - `git clean -fd`

3.  **Restore Database Snapshot (Synesthesia & User Settings):**
// turbo
    - `cp morpheme.db.save_point_2026-03-26_final morpheme.db`

4.  **Restart Server:**
// turbo
    - `nohup python3 app.py > server.log 2>&1 &`

5.  **Verify Status:**
    - Navigate to `http://localhost:3000` and confirm the UI is at the August 9 stable state.
    - Check: Delete Post button appears above thread title in Forum (static HTML), About Me scrollbar visible on full Profile page and mini-profile modal (all browsers), Forum JS/CSS at v66/v42.

### August 9 Save Point Details
- **Commit**: `2598183`
- **Branch**: `main`
- **GitHub**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
- **Stable State Doc**: `stable_state_august_9_2026.md` in Antigravity brain
