# Morpheme Stable State Summary - June 13, 2026

This summary documents the stable state of the Morpheme application as of June 13, 2026. The synchronization across Localhost, GitHub origin, and `morpheme.games` is fully completed and verified.

---

## 🚀 Key Improvements & Bug Fixes

### 1. Public Room On-Demand Reconstruction Optimizations
*   **The Issue**: When public singleton rooms/hubs (e.g. `pub_v2_...`) were reconstructed on-demand (such as after a server restart or deployment), the intermission state was initialized with a tight 15-second timer. If the server was cold or loading dictionaries lazily (like the 2.6-second CSW trie build time on first access), the background board search would not complete before the countdown reached 0:00. This caused players to experience a delayed "GENERATING NEXT BOARD..." loading message.
*   **The Fix**:
    *   **Background Search Kickstart**: Modified the public singleton reconstruction logic in `app.py` to immediately launch a background thread to generate spinner parameters and kickstart the board generator search before the player's web page even completes loading.
    *   **Extended Intermission Buffer**: Increased the initial intermission countdown buffer for reconstructed public rooms from 15 seconds to **25 seconds** (`time.time() - 35`), giving the server ample lead-time to pre-generate and score the board.

### 2. Low-Latency Audio Engine & Flutter Wrapper Upgrades
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

### 3. Word Lists Performance & Rendering Overhaul
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
    *   Extended client-side timeouts in `tools.js` to 3 minutes to allow slow mobile data connections to finish downloading.
    *   Appended warnings in the Lists description: *"Some lists may take 1-3 minutes to fully load, especially if you are using data, and not wi-fi."*

### 4. Forum, Lightbox, and Image Upload Upgrades
*   **SQLite Database Schema Migration for Comment Images**:
    *   Added an `image_url` column to the `forum_comments` table via startup schema migrations, enabling image attachments on comment replies.
*   **Client-Side Image Compression**:
    *   Implemented client-side image resizing and quality compression before upload, converting files to lightweight JPEGs.
    *   Bypasses Nginx and Flask request payload limit constraints (with `MAX_CONTENT_LENGTH` raised to `16MB`).
    *   Added visual loading states and disabled upload/submit buttons during active network requests.
*   **Responsive Image Lightbox**:
    *   Enabled expanding forum post and comment images into a responsive full-screen overlay lightbox on mobile and desktop devices.
*   **Mobile Auto-Scroll transitions**:
    *   Implemented automatic viewport scrolling to active content on mobile devices (screen width <= 820px) when changing forum views. This scrolls the view smoothly to the category title/threads, post details, or create post inputs, avoiding manual scroll requirements after selection.

### 5. Find Count Tool Enhancements
*   **Invalid Word Indicators**:
    *   Correctly flags invalid dictionary words inside the Find Count query results rather than showing "0 times found" for them, improving clarity.

### 6. Transition Latency & Worker Exhaustion Elimination
*   **The Issue**: During round transition countdowns reaching 0:00 (especially after a deploy or server restart where singleton rooms are reconstructed on the fly), if the background board generation thread did not complete in time, the room would attempt to generate a board synchronously inside the HTTP/WS request thread. This blocked the Gunicorn worker thread for 5–8 seconds, freezing other network requests and showing "WAITING..." to players.
*   **The Fix**: Modified `start_next_round` in `game_room.py` to immediately utilize the instant `get_emergency_fallback_board(...)` function (which runs in <1ms) rather than executing a synchronous CPU-bound board search. It pre-calculates, validates, and initializes the board density maps instantly.

---

## 🛠 Active Features & Configuration
*   **Tools Suite**: Word of the Day, Random Word, Unscramble, Find Count, Word Lists, and Personal Timer.
*   **Board Formats**: Normal (72%), Checkerboard (12%), Double (1%), Triple (1%), Valued Letters, Rotation, Penalty, Mania, Either/Or, Bonus Word, and Density.
*   **Grid Dimensions**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
*   **Verification**: All integration, caching, Gzip compression, and database persistence tests pass 100% cleanly. Synchronized on the live production server `morpheme.games` on port 443 (via PM2).

---

**Latest Stable Commit ID**: `06449b734ea4a8dea9b39f95f493b44690df241d`  
**Localhost & GitHub Sameness Status**: Synchronized  
**Production Server Status**: Green / PM2 Online / Live at commit `06449b7`
