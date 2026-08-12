---
description: Revert the project to the August 11, 2026 save point. Use this when the user says "Start Over".
---

### Revert Procedure

1.  **Reset Repository to Save Point:**
    - `git reset --hard 2ea239c`

2.  **Clean Up Untracked Files:**
// turbo
    - `git clean -fd`

3.  **Restart Server / Sync:**
// turbo
    - `python3 scratch/deploy_all_fixes.py`

4.  **Verify Status:**
    - Navigate to `morpheme.games` and confirm UI is at the August 11 stable state.

### August 11 Save Point Details
- **Commit**: `2ea239c`
- **Branch**: `main`
- **GitHub**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
- **Stable State Doc**: `stable_state_august_11_2026.md` in Antigravity brain
