---
description: Revert to the latest saved stable state
---
// turbo-all
If the user asks to "Start over" or revert to the saved state, follow these steps:

1. Kill any running server: `kill $(lsof -t -i:3000) || true`
2. Revert files: `git reset --hard latest-stable-state`
3. Clean untracked: `git clean -fd`
4. Restore databases: `cp morpheme.db.snapshot morpheme.db && cp developer_messages.db.snapshot developer_messages.db`
5. Start server: `python3 app.py`
6. Inform the user that the application has been successfully reverted to the saved state (from Feb 5, 2026, including Forum and Performance updates).
