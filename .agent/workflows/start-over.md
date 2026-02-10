---
description: Revert the application to the last saved checkpoint (Achievement Refinement)
---

This workflow resets the codebase to the state saved at the end of the Achievement Modal refinement session.

// turbo
1. Reset the codebase to the checkpoint commit
```
git reset --hard 78ca2f7e94763f95c37a94d362d53ab86803baaa
```

2. Clean any untracked files
```
git clean -fd
```

3. Restart the server if necessary
```
python3 app.py
```
