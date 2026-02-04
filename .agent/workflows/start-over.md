description: How to revert the application to the current stable state (Profile Refinement + Skills Grid + Flags) saved on 2026-02-03
---

To revert the application to the stable state saved on February 3rd, 2026 (which includes the full Profile Refinement, Skill Rankings Grid, and Country Flag implementation), follow these steps:

### Method 1: Restore from GitHub (Recommended)

// turbo
1. Kill any running server processes:
   `lsof -ti:3000 | xargs kill -9`

2. Reset the local repository to the snapshot commit (38b3b32):
   `git reset --hard 38b3b32`

3. Your application is now restored to its stable state.

### Method 2: Restore from Local Backup

// turbo
1. Kill any running server processes:
   `lsof -ti:3000 | xargs kill -9`

2. Copy files from the local backup:
   `rsync -av --exclude='.git' --exclude='.backups' .backups/stable_profile_refinement_2026_02_03/ .`
