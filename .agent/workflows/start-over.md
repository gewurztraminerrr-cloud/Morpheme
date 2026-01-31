---
description: How to revert the application to the current stable state (Combo Checker + LIC + Sensitive Cleanup) saved on 2026-01-31
---

To revert the application to the stable state saved on January 31st, 2026 (which includes the refined Combo Checker with MP/LIC grids and the sanitized GitHub history), follow these steps:

### Method 1: Restore from GitHub (Recommended)

// turbo
1. Kill any running server processes:
   `lsof -ti:3000 | xargs kill -9`

2. Reset the local repository to match the GitHub stable state:
   `git fetch origin && git reset --hard origin/main`

3. Your application is now restored to its stable state.

### Method 2: Restore from Local Backup

// turbo
1. Kill any running server processes:
   `lsof -ti:3000 | xargs kill -9`

2. Copy files from the local backup:
   `cp -r .backups/stable_2026-01-31/* .`
