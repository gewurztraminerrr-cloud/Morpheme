---
description: Revert the application to the state saved on 2026-02-04 (Commit da27513)
---
If the user asks to "Start Over" or revert to the saved state, follow these steps:

1. Stop any running server processes.
2. Run `git reset --hard da27513` to revert all files to the saved state.
3. Run `git clean -fd` to remove any untracked files.
4. Restart the server using `./run_morpheme.sh`.
5. Inform the user that the application has been successfully reverted to the state from Feb 4, 2026.
