# Morpheme Stable State Summary - June 13, 2026

This summary documents the stable state of the Morpheme application as of June 13, 2026. The synchronization across Localhost, GitHub origin, and `morpheme.games` is fully completed and verified.

---

## 🚀 Key Improvements & Bug Fixes

### 1. Low-Latency Audio Engine & Flutter Wrapper Upgrades
*   **Flutter `flutter_soloud` Audio Engine Migration**:
    *   Migrated the mobile hybrid app audio system from the `audioplayers` package to the lower-latency `flutter_soloud` engine.
    *   Preloads sound assets natively on application startup, allowing instantaneous playback.
*   **Native Sound Bridge (`MorphemeAudioBridge`)**:
    *   Enabled the mobile WebView to route audio events (like tile selection clicks and success sounds) natively through the Flutter container.
    *   Bypasses WebKit/WebView-induced audio latency, delivering click feedback with zero delay.
*   **Dynamic Audio Session Routing & Bluetooth HFP**:
    *   Configured the mobile audio session to dynamically activate Hands-Free Profile (HFP) categories (`playAndRecord` / `voiceChat`) when Bluetooth earpieces/AirPods are connected.
    *   Automatically reverts to `playback` when routing sound via speakers or wired devices, ensuring high-fidelity sound.
*   **Intermission Bell & Beep Native Warning Routing**:
    *   Removed subsonic keep-alive oscillators and silent audio loops from the standard web client's `play.js`.
    *   Wired the intermission countdown warnings (the `0:10` remaining bell/beeper alert) to trigger native Flutter chimes dynamically based on the user's selected bell type.

### 2. Word Lists Performance & Rendering Overhaul
*   **Gzip Compression Middleware**:
    *   Implemented a global `@app.after_request` gzip compression middleware.
    *   Automatically compresses responses larger than 500 bytes, reducing the uncompressed NWL list payload size from **2.4MB to ~544KB (a 4.42x size reduction)** and saving significant mobile data.
*   **In-Memory Response Caching**:
    *   Added memory-based caching (`ENDPOINT_LISTS_CACHE`) in `tools_get_lists` (`app.py`), decreasing subsequent list query response times from up to 2.6 seconds to **sub-20ms**.
    *   Bypasses caching for the mutable moderator `'added'` word list to keep moderator modifications real-time.
    *   Automatically clears the cache when an administrator uploads a new dictionary file.
*   **Progressive Background Rendering**:
    *   Replaced the infinite scroll pagination trigger with a smooth progressive background renderer.
    *   Renders the first 2,000 words instantly for immediate view, then renders the rest of the dictionary in small chunks of 2,000 words every 5ms.
    *   Adjusts the scrollbar height dynamically, enabling users to drag the custom scrollbar thumb smoothly from 'A' to 'Z' without freezing or lagging the browser layout.
*   **Extended Timeout & Descriptions**:
    *   Extended client-side timeouts in `tools.js` to 4 minutes to allow slow mobile data connections to finish downloading.
    *   Appended warnings in the Lists description: *"Some lists may take 2-4 minutes to load, especially if you are using data, and not wi-fi."*

### 3. Forum, Lightbox, and Image Upload Upgrades
*   **SQLite Database Schema Migration for Comment Images**:
    *   Added an `image_url` column to the `forum_comments` table via startup schema migrations, enabling image attachments on comment replies.
*   **Client-Side Image Compression**:
    *   Implemented client-side image resizing and quality compression before upload, converting files to lightweight JPEGs.
    *   Bypasses Nginx and Flask request payload limit constraints (with `MAX_CONTENT_LENGTH` raised to `16MB`).
    *   Added visual loading states and disabled upload/submit buttons during active network requests.
*   **Responsive Image Lightbox**:
    *   Enabled expanding forum post and comment images into a responsive full-screen overlay lightbox on mobile and desktop devices.

### 4. Find Count Tool Enhancements
*   **Invalid Word Indicators**:
    *   Correctly flags invalid dictionary words inside the Find Count query results rather than showing "0 times found" for them, improving clarity.

---

## 🛠 Active Features & Configuration
*   **Tools Suite**: Word of the Day, Random Word, Unscramble, Find Count, Word Lists, and Personal Timer.
*   **Board Formats**: Normal (72%), Checkerboard (12%), Double (1%), Triple (1%), Valued Letters, Rotation, Penalty, Mania, Either/Or, Bonus Word, and Density.
*   **Grid Dimensions**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
*   **Verification**: All integration, caching, Gzip compression, and database persistence tests pass 100% cleanly. Synchronized on the live production server `morpheme.games` on port 443 (via PM2).

---

**Latest Stable Commit ID**: `b3184c5a4646e34a8ad4927edf22a27b2c112059`  
**Localhost & GitHub Sameness Status**: Synchronized  
**Production Server Status**: Green / PM2 Online / Live at commit `b3184c5`
