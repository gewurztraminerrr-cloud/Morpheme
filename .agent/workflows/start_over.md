---
description: How to revert to the saved state
---

To revert the application to the saved "Stable Layout" state:

1. Stop any running servers.
2. Run the following command:
```bash
git reset --hard user-save-point
git clean -fd
```
3. Restart the server.
