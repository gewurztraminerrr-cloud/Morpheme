---
description: Custom workflow to revert the project to the pre-Tournament state with the specific Aesthetic Layout restored from git. Use this when the user says "Start Over" or wants to undo Tournament changes while keeping their UI mods.
---

### Revert Procedure

1.  **Stop Server Process:**
    - Find the PID listening on port 3000 (usually `python3 app.py`).
    - Use `kill -9` or `pkill` to terminate it forcefully.

2.  **Delete Tournament Files:**
    - Remove the backend file: `rm tournament_manager.py` (if present).
    - Remove frontend templates: `rm templates/play_tournament.html templates/tournaments.html` (if present).

3.  **Clean Up `app.py`:**
    - Remove the Tournament Code Block (Imports, Routes, Logic) usually at the end of the file (lines ~2814-2930).
    - Ensure the root route (`@app.route('/')`) uses `return render_template('index.html')` instead of `send_from_directory('static', 'index.html')`. This is critical for template rendering.

4.  **Restore Aesthetic Layout (Static HTML):**
    - Assuming the "Aesthetic" version was committed to git as `static/index.html`:
      `git checkout static/index.html`
    - Copy this recovered file to be the active template:
      `cp static/index.html templates/index.html`
    - Remove the static copy to prevent confusion and force template serving:
      `rm static/index.html`

5.  **Remove "Tournaments" Button:**
    - Edit `templates/index.html` to remove the line containing the "Tournaments" navigation button (e.g., `<button ... data-page="tournaments">`).

6.  **Restart Server:**
    - Run `./run_morpheme.sh` or `python3 app.py`.
    - Verify layout and functionality via browser.
