description: Revert to the stable state with Bell Notification and Winners History (Commit 7983794)
---

To revert the application to the state saved on February 4th, 2026 (including Bell Notification and Winners History), follow these steps:

### Revert Workflow

// turbo
1. Commit or stash any current experimental changes:
   `git add . && git stash`

// turbo
2. Reset the repository to the saved 'latest-stable-state' tag:
   `git reset --hard latest-stable-state`

3. Refresh your browser to see the restored state.
