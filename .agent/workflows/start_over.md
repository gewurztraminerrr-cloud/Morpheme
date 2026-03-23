---
description: Revert the project to the March 23, 2026 save point. Use this when the user says "Start Over".
---

### Revert Procedure

1.  **Stop Server Process:**
// turbo
    - `lsof -t -i :3000 | xargs kill -9 || true`

2.  **Reset Repository to Save Point:**
// turbo
    - `git reset --hard PERMANENT_RESTORATION_SAVE_POINT_DO_NOT_DELETE`

3.  **Clean Up Untracked Files:**
// turbo
    - `git clean -fd`

4.  **Restore Database Snapshot (Synesthesia & User Settings):**
// turbo
    - `cp morpheme.db.save_point_2026-03-23 morpheme.db`

5.  **Restart Server:**
// turbo
    - `nohup python3 app.py > server.log 2>&1 &`

6.  **Verify Status:**
    - Navigate to `http://localhost:3000` and confirm the UI organization, Synesthesia settings for 'jeffy', and 500+ rounds are restored.

