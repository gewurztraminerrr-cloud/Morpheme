---
description: Reset the Morpheme app to the saved state (save-point)
---
# Start Over (Restore to Save Point)

This workflow will reset the application code to the state saved at the `save-point` tag.
**WARNING**: This will discard any changes made since the last "Save Current State" action.

1. Reset git to the save-point tag.
// turbo
```bash
git reset --hard save-point
```

2. Clean untracked files.
// turbo
```bash
git clean -fd
```

3. Restart the server.
// turbo
```bash
ps aux | grep "python3 app.py" | grep -v grep | awk '{print $2}' | xargs kill -9 && nohup python3 app.py > server.log 2>&1 &
```

4. Notify user.
   - The application has been restored to the saved state.
