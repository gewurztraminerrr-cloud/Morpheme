# Stable State Summary - July 1, 2026

This document summarizes the stable state of **Morpheme** as of July 1, 2026. All local changes, remote code on GitHub, and the live application running on morpheme.games are fully synchronized.

## Latest Commit Information
* **Commit ID**: `50835eb` (and subsequent documentation commits)
* **Branch**: `main`
* **Date**: July 1, 2026

## Changes Included in this Stable State

1. **PWA Immersive Fullscreen Mode for Mobile Devices**:
   * **Manifest Configuration**: Updated `static/manifest.json` display mode from `"standalone"` to `"fullscreen"`.
   * **Programmatic Fallback**: Injected an inline JavaScript IIFE inside `templates/index.html` that triggers `document.documentElement.requestFullscreen({ navigationUI: 'hide' })` on the first touch or click interaction. This bypasses the mobile browser's aggressive local manifest caching and forces the bottom system navigation bar (`|||`, `O`, `<`) and top status bar to hide immediately on launch.
   * **Syntax Cleanup**: Restored two missing closing braces in the hybrid file-picker click listener to resolve a script block syntax error, ensuring the entire HTML head script block parses and runs correctly on all devices.
   * **Cache Busting**: Appended `?v=2` to the manifest link inside the HTML head to force mobile Chrome to detect a version change and pull the updated display specifications immediately.

2. **Tools Validator ("Is Valid") Keyboard Focus Fix**:
   * **Dynamic Keyboard Hiding**: Modified the validation submit flow in `static/js/tools.js` to run a mobile device regex check:
     * **On Mobile**: Calls `inputEl.blur()` to release input focus, which slides the virtual mobile keyboard out of view automatically after checking a sequence, letting players read definitions/pronunciations clearly without the keyboard blocking the viewport.
     * **On Desktop**: Continues to run `inputEl.focus()` so that players can perform rapid, successive lookups using their physical keyboards without re-selecting the textbox.
   * **Cache Busting**: Incremented the script import link for `tools.js` from `v=71` to `v=72` in `templates/index.html` to guarantee client devices load the updated blur logic instantly.

3. **PM2 Server Daemon and Process Recovery**:
   * **Process/Thread Leak Cleanup**: Diagnosed and resolved a PM2 crash-restart loop on `morpheme.games` (which had accumulated 5,261 restarts and hit the OS user processes limit `ulimit -u` of 7718, triggering `RuntimeError: can't start new thread`).
   * **Correct Interpreter Settings**: Terminated all defunct processes (`pm2 kill`) and restarted the server with the correct virtual environment path: `pm2 start app.py --name morpheme --interpreter venv/bin/python3` followed by `pm2 save` to ensure stability across server reboots.

4. **Mobile Navigation Back Buttons Removal**:
   * **Button Removal**: Removed the mobile back buttons (`forum-mobile-back-btn` for categories list, `tools-mobile-back-btn` for tools list, `mods-mobile-back-btn` for mods list, and `settings-mobile-back-btn` for settings categories) from `templates/index.html` completely.
   * **Swipe Navigation**: Declutters the mobile UI, allowing users to return to category list menus seamlessly using standard swipe gestures.

5. **Mods Tab Mobile Layout Alignment**:
   * **Layout Synchronization**: Added `#page-mods` to the mobile media queries in `static/css/play.css`, aligning the split-layout rules of the Mods tab layout exactly with the Tools and Settings pages on mobile viewports.
   * **Sidebar Width Constraint**: Ensures that the Mods sidebar list spans exactly 100% of the screen width on mobile, and the moderation panels remain hidden off-screen to the right until selected, maintaining consistency across all dashboards.
   * **Cache Busting**: Incremented the import link of `play.css` to `v=105` in `templates/index.html` to force mobile devices to fetch the updated styles immediately.

6. **Mobile Touch-Dragging Navigation Lock & Lobby Height Lock**:
   * **Touch Action Prevention**: Added `touch-action: none;` to the `.header` (logo, "MORPHEME MORE-FEEM", and top navigation tabs container) and `.separator` elements in `static/css/style.css` (cache-busted to `v=49` in `templates/index.html`).
   * **Lobby Page Viewport Lock**: Added `#page-lobby` and `.lobby-grid` to the mobile media queries in `static/css/play.css`, setting a fixed height of `calc(100vh - 120px)` and `overflow-y: hidden` (cache-busted to `v=106` in `templates/index.html`).
   * **No Viewport Bouncing**: Prevents touch-dragging gestures on the logo, tabs, and page background from causing the whole lobby screen to bounce or move slightly. It locks the header and outer page layout in place on mobile, leaving only the inner panels scrollable vertically.

7. **FAQ Format Description Updates**:
   * **Checkerboard**: Reworded the Checkerboard format explanation in `templates/index.html` to clearly detail its alternating row/column structure and consonant-vowel rhythmic patterning.
   * **[Letter] Mania**: Reworded the [Letter] Mania description to detail the specific board-wide occurrences threshold rules (33% or more for common letters, 20% or more for rare consonants).

## Verification
* **Local**: Verified on Safari and Chrome.
* **Production**: Pulled changes and verified live on **morpheme.games** (all services running with 0 restarts and healthy CPU/Memory footprints).
