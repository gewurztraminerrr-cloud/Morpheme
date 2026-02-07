---
description: Revert to the latest saved stable state
---
// turbo-all
If the user asks to "Start over" or revert to the saved state, follow these steps:

1. Kill any running server: `kill $(lsof -t -i:3000) || true`
2. Revert files: `git reset --hard latest-stable-state`
3. Clean untracked: `git clean -fd`
4. Restore databases: `cp morpheme.db.snapshot morpheme.db && cp developer_messages.db.snapshot developer_messages.db`
5. Start server: `python3 app.py`
6. Inform the user that the application has been successfully reverted to the saved state (Late Feb 6, 2026).

**Saved Features (as of this checkpoint):**
- **Manual Tool Restriction**: Disabled in rooms, enabled in lobby only.
- **Fair Play Enforcement**: Detects Manual tool usage and prevents score/rating changes for that round.
- **Skill Level Legend**: Professional and clever names assigned to all rating brackets (Initiate -> Infinite).
- **Legibility Overhaul**: Opaque tooltips, massive 1.5rem bold text, and 90px clearance margin for rating charts.
- **Achievements & Stats**: Including the Achievements popup, scrollable history, and condensed board views.
