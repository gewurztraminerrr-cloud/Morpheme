---
description: Revert to the state saved on Feb 5, 2026 (Commit afdcf53)
---
If the user asks to "Start Over" or revert to the saved state, follow these steps:

1. Stop any running server processes.
2. Run `git reset --hard afdcf53` to revert all files to the saved state.
3. Run `git clean -fd` to remove any untracked files.
4. Restore the database: `cp morpheme.db.snapshot morpheme.db` and `cp developer_messages.db.snapshot developer_messages.db`.
5. Restart the server using `./run_morpheme.sh`.
6. Inform the user that the application has been successfully reverted to the state from Feb 5, 2026.
