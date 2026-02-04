description: Revert to the stable state with Finder Highlighting and Bonus Word Priority
---

To revert the application to the state saved on February 3rd, 2026 (including Finder Highlighting, Word Checkmarks, and Bonus Word Prioritization), follow these steps:

### Revert Workflow

// turbo
1. Commit or stash any current experimental changes:
   `git add . && git stash`

// turbo
2. Reset the repository to the saved 'latest-stable-state' tag:
   `git reset --hard latest-stable-state`

3. Refresh your browser to see the restored state.
