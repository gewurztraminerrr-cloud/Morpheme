# Stable State Summary - August 12, 2026

This document summarizes the stable state of **Morpheme** as of August 12, 2026. All local changes, remote code on GitHub, and the live application running on morpheme.games are fully synchronized.

## Latest Commit Information
* **Commit ID**: `72eb355`
* **Branch**: `main`
* **Date**: August 12, 2026
* **Repository**: `https://github.com/gewurztraminerrr-cloud/Morpheme`
* **Live Deployment**: `https://morpheme.games/`

## Features & Improvements Included in this Stable State

1. **0:45 Intermission Parameter Reveal & Slow Gold Flash**:
   - Built `@keyframes slowGoldPulse` in `play.css` with 4s slow double gold flash animation and gold glow (`#ffd700`).
   - Tied exact parameter update and reveal trigger to `Math.ceil(remaining) <= 45` in `updateLocalTimer()` and `updateParameters()` in `play.js`.
   - Updated `.play-header`, `.game-params`, `.spinner-set-label`, `.header-meta`, and `.spinner-modal-card` to pulse gold smoothly at 0:45 intermission time.

2. **View Full List Large Custom Scrollbar Thumb (`tools.js`, `index.html`, `play.css`)**:
   - Equipped `#full-list-modal` with `.custom-scrollbar-track` and `.custom-scrollbar-thumb`.
   - Styled a 20px wide track with a 14px–16px wide, purple/violet gradient thumb (`#a78bfa` to `#7c3aed`), glowing border (`rgba(167, 139, 250, 0.8)`), and 40px minimum height.
   - Suppressed native browser scrollbars (`::-webkit-scrollbar` and `scrollbar-width: none`), keeping ONLY the single larger custom scrollbar thumb visible.
   - Linked mouse dragging, touch dragging, and track clicking directly to `full-list-modal-results`.

3. **Mobile Smooth Sliding Tab Navigation for Mods and Settings (`settings.js`, `mods.js`)**:
   - Updated `showSettingTab()` in `settings.js` and `showModTab()` in `mods.js` to use `layout.scrollTo({ left: layout.clientWidth || layout.scrollWidth, behavior: 'smooth' });` on mobile.
   - Tapping a tab button in Mods or Settings on mobile smoothly slides the view horizontally across the screen to reveal the selected tab's content, matching Tools.

4. **Combo Checker Mobile Table Touch Scrolling & Navigation Controls (`tools.js`, `index.html`, `play.css`)**:
   - Added active horizontal touch dragging on `.horizontal-scroll-container` (`#mp-container` and `#lic-container`).
   - Added `◄` and `►` scroll buttons to MP and LIC section headers for instant 1-tap navigation across `0MP` through `6MP` tables on mobile.

5. **Strict Exclusion of 24h Rooms & Valued Letters Format from Leaderboards (`app.py`)**:
   - Enforced `rh.round_duration < 7200` and `LOWER(COALESCE(rh.board_format, '')) NOT LIKE '%valued%'` in `get_leaderboard_data()` base filters across all 8 Leaderboard views (*Best Scores*, *Best Words*, *Best PE*, *Best Pct Found*, *Best Ratings*, *Avg Score*, *Obscure Words*, *Most Games*).

6. **Gameplay & Settings Customizations**:
   - Added **Visual Settings** tab to Appearance settings.
   - Added **Private Message Blocking** and **With Friends Invitation Blocking** toggles in Settings ("Gameplay & Preferences").
   - Added popup notice when attempting to invite players who have With Friends invitations disabled.
