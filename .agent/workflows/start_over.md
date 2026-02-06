description: Revert to the stable state saved on Feb 6, 2026 (Achievements Popup, 10-row limit, and History limit)
---

To revert the application to the state saved on February 6th, 2026, follow these steps:

### Revert Workflow

// turbo
1. Stop the server and discard current changes:
   `git reset --hard latest-stable-state && git clean -fd`

// turbo
2. Restore the database from snapshot:
   `cp morpheme.db.snapshot morpheme.db && cp developer_messages.db.snapshot developer_messages.db`

// turbo
3. Restart the server:
   `./run_morpheme.sh`

4. Refresh your browser to see the restored state.
