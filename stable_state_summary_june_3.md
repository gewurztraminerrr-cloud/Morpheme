# Stable State Summary — June 3, 2026

## Snapshot Commit & Save Point

| Environment | Commit / Tag | Status |
|-------------|--------------|--------|
| **localhost** (`/Users/jeffbabiak/`) | `e7a3740` | ✅ Clean & Synchronized |
| **GitHub** (`origin/main`) | `e7a3740` / `snapshot-current` | ✅ Pushed & Tagged |
| **morpheme.games** (production) | `e7a3740` / `snapshot-current` | ✅ Fully Deployed & PM2 Reloaded |

**All environments are 100% synchronized at the latest commit `e7a3740`.**
The local modifications to `board_generator.py` have been committed, pushed to GitHub, and successfully deployed to the remote production environment via the `deploy.py` script. The active production tag `snapshot-current` has been updated and pushed to remote.

---

## Serving Versions

| File | Version / State | Description |
|------|-----------------|-------------|
| `board_generator.py` | Commit `e7a3740` | Resolved 3D cube neighbor transitions inside `_has_ing_sequence`, `_sanitize_forbidden_sequences`, and `_guarantee_no_ing`. Enforced ING sequence verification on both target and achieved difficulties. Supported 3D Either/Or layouts and added early-break optimization for protected tiles. |

---

## Work Completed Up To June 3, 2026

### 1. Fix 3D Cube Neighbor Resolution inside Sequence Checkers
* **Goal achieved:** The "ING" sequence checkers (`_has_ing_sequence`, `_sanitize_forbidden_sequences`, and `_guarantee_no_ing`) now correctly use `_get_cube_neighbors(f, r, c)` when checking 3D cube layouts.
* **Why it was a bug:** Previously, standard coordinate offsets (`df`, `dr`, `dc`) were used for 3D boards, which are invalid on a 6-face cube surface where cells wrap around edges and corners. This blind spot allowed "ING" sequences wrapping around edges to leak to the game.

### 2. Enforce ING Check on Promoted Boards
* **Goal achieved:** If a board generated for an "Easy" target difficulty achieves a uniqueness ratio that places it in the "Medium" or "Hard" difficulty label range, the generator now detects the achieved difficulty and runs "ING" checks/sanitization. This closes the uniqueness promotion loophole.

### 3. Support 3D Either/Or Layouts
* **Goal achieved:** Rewrote the Either/Or tile application block inside both `generate_board` and `_generate_emergency_compliant_board` to support 3D coordinates `(f, r, c)`, eliminating a pre-existing 3D Either/Or comparison bug (`TypeError`).

### 4. Optimize Sequence Sanitizer Looping
* **Goal achieved:** Introduced `made_progress` tracking inside `_sanitize_forbidden_sequences` to break out of the attempts loop immediately if all found sequences are fully protected (e.g. they form the bonus word), eliminating log spams and wasted CPU cycles.

---

## Key Files Tracked

| File | Location | Purpose |
|------|----------|---------|
| `board_generator.py` | Production + GitHub | Resolved 3D neighbor transition logic, enforced target/achieved difficulty checks, and supported 3D Either/Or layouts. |
| `scratch/test_ing_prevention.py` | Local / Scratch | Verification script testing all 48 combinations of dimensions, formats, and difficulties. |
