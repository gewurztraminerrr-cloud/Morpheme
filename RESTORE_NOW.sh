#!/bin/bash
# RESTORE_NOW.sh
# Emergency recovery to March 23 Stable State

echo "=== EMERGENCY RESTORATION STARTING ==="

# 1. Kill any existing server
echo "Stopping server on Port 3000..."
lsof -t -i :3000 | xargs kill -9 || true

# 2. Reset the code
echo "Reverting all code to PERMANENT tag..."
git reset --hard PERMANENT_RESTORATION_SAVE_POINT_DO_NOT_DELETE

# 3. Clean untracked files
echo "Cleaning working directory..."
git clean -fd

# 4. Inject the stable database snapshot (User settings, Synesthesia, etc)
echo "Restoring database from stable backup..."
cp morpheme.db.save_point_2026-03-23 morpheme.db

# 5. Restart the server
echo "Restarting Morpheme server (Single Process Mode)..."
nohup python3 app.py > server.log 2>&1 &

echo "=== RESTORATION COMPLETE ==="
echo "Check your browser at http://localhost:3000"
