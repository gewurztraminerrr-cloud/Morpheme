---
description: Revert the project to the August 2, 2026 save point. Use this when the user says "Start Over".
---

### Revert Procedure

1.  **Reset Repository to Save Point:**
    - `git reset --hard save_point_august_2_2026`

3.  **Clean Up Untracked Files:**
// turbo
    - `git clean -fd`

4.  **Restore Database Snapshot (Synesthesia & User Settings):**
// turbo
    - `cp morpheme.db.save_point_2026-03-26_final morpheme.db`

5.  **Restart Server:**
// turbo
    - `nohup python3 app.py > server.log 2>&1 &`

6.  **Verify Status:**
    - Navigate to `http://localhost:3000` and confirm the UI organization, Synesthesia settings for 'jeffy', 50 row limits on achievement tables, and Spinner Set congratulatory color are restored.
