---
description: Revert the application to the 'snapshot-current' state
---
// turbo-all
1. Revert to the tagged snapshot:
```bash
git reset --hard snapshot-current
```
2. Force pull from origin just in case:
```bash
git pull origin main --force
```
3. Restart the server:
```bash
lsof -i :3000 -t | xargs kill && python3 app.py &
```
