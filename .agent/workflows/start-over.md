---
description: How to revert the application to the stable state saved on 2026-01-30
---

To revert the application to the state saved on January 30th, 2026, follow these steps:

// turbo
1. Kill any running server processes:
   `lsof -ti:3000 | xargs kill -9`

2. Copy all files from the backup back to the root directory:
   `cp -r .backups/stable_2026-01-30/* .`

3. Your application is now restored to its state at the end of the session where Clues tab and 7-minute cleanup were implemented.
