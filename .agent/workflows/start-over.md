---
description: Revert to Scrollable Unscramble Checkpoint (2026-02-10)
---

This workflow resets the codebase to the state saved after the Unscramble tool scrollable list fix.

// turbo
1. Reset the codebase to the checkpoint commit
```
git reset --hard f35b9fe1b29d9e8ffd175fe685c9ba4d102ffcac
```

2. Clean any untracked files
```
git clean -fd
```

3. Restart the server if necessary
```
python3 app.py
```
