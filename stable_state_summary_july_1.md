# Stable State Summary - July 1, 2026

This document summarizes the stable state of **Morpheme** as of July 1, 2026. All local changes, remote code on GitHub, and the live application running on morpheme.games are fully synchronized.

## Latest Commit Information
* **Commit ID**: `717090f` (and subsequent documentation commits)
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

## Verification
* **Local**: Verified on Safari and Chrome.
* **Production**: Pulled changes and verified live on **morpheme.games** (all services running with 0 restarts and healthy CPU/Memory footprints).
