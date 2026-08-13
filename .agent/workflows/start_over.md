---
description: Revert the project to the August 12, 2026 save point. Use this when the user says "Start Over".
---

### Revert Procedure

1.  **Reset Repository to Save Point:**
    - `git reset --hard 8ccad54`

2.  **Clean Up Untracked Files:**
    - `git clean -fd`

3.  **Restart Server / Sync:**
    - `python3 scratch/deploy_remote.py`

4.  **Verify Status:**
    - Navigate to `morpheme.games` and confirm UI is at the August 12 stable state.

### August 12 Save Point Details
- **Commit**: `8ccad54`
- **Branch**: `main`
- **GitHub**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
- **Stable State Doc**: `stable_state_summary_august_12.md` in repository root

