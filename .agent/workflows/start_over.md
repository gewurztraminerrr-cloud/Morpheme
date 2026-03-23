---
description: Revert the project to the March 22, 2026 save point. Use this when the user says "Start Over".
---

### Revert Procedure

1.  **Stop Server Process:**
// turbo
    - `pkill -f "python3 app.py"`

2.  **Reset Repository to Save Point:**
// turbo
    - `git reset --hard save_point_2026-03-22`

3.  **Clean Up Untracked Files:**
// turbo
    - `git clean -fd`

4.  **Restart Server:**
// turbo
    - `./run_morpheme.sh`

5.  **Verify Status:**
    - Navigate to `http://localhost:3000` and confirm the UI organization and settings are restored.
