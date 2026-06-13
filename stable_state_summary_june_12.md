# Morpheme Stable State Summary - June 12, 2026

This summary documents the stable state of the Morpheme application as of June 12, 2026. The synchronization across Localhost, GitHub origin, and `morpheme.games` is fully completed and verified.

---

## 🚀 Key Improvements & Bug Fixes

### 1. Bluetooth Audio Latency Fix (Mobile Earpieces)
*   **The Issue**: When playing with Bluetooth earpieces (e.g. AirPods, Galaxy Buds), the sound effects of letter selection and word validation had a tremendous delay or were cut off. This occurred because Bluetooth hardware aggressive noise-gating put the earpieces' DAC/amplifiers to sleep during moments of silence. Playback of short game sounds (50ms) had to wake the hardware up, causing 200ms–500ms startup delays.
*   **The Fix**:
    *   Configured the Web Audio API keep-alive oscillator to output a subsonic **5Hz sine wave** (completely inaudible and below the physical reproduction range of mobile speakers) at a volume of **0.01** (sufficiently above the noise-gate threshold to keep the DAC active and awake).
    *   Added listeners to automatically call `ctx.suspend()` when the browser tab goes to the background (saving mobile battery) and `ctx.resume()` upon tab focus/visibility change (instantly warming up the Bluetooth stream for zero-delay tap response).
    *   **Cache Busting**: Bumped `play.js` query version string in `templates/index.html` to `v=154`.

### 2. Mobile File Pickers & Photo Library Triggers Fix (Standard Web & Hybrid App)
*   **The Issue**: WebKit/Mobile Safari and Android Chrome ignored or blocked `<label>` click forwarding to hidden file input elements when the inputs were styled with offscreen layout or zero sizing. Furthermore, calling `e.preventDefault()` on the click events canceled the user's active touch/click session, causing browsers to block programmatic `.click()` actions as "not user-initiated".
*   **The Fix**:
    *   Converted uploader container tags (`#forum-comment-image-wrapper`, `#forum-post-image-wrapper`, `#profile-avatar-trigger`, `#dict-upload-wrapper`) from `<label>` elements to standard `<div>` elements in [templates/index.html](file:///Users/jeffbabiak/templates/index.html) to eliminate flaky native label click forwarding and prevent double-triggering on desktop.
    *   Refactored the global click capturing listener on the document. For the hybrid mobile app, it blocks propagation and calls the native Flutter uploader bridge. For standard web browsers (desktop & mobile Chrome/Safari), it programmatically invokes `input.click()` synchronously without calling `e.preventDefault()`, allowing standard browsers to open the photo library/file selector dialog natively.
    *   Verified and successfully tested in mobile Chrome, Safari, and the hybrid mobile app on Android.
    *   **Cache Busting**: Bumped version strings for static assets (`forum.js?v=48`, `tools.js?v=54`, `mods.js?v=9`) to ensure users load the latest script implementations.

### 3. Find Count Tool
*   **Feature**: Integrated a new "Find Count" tool inside the Tools sidebar navigation, positioned immediately before the **Personal Timer**.
*   **Functionality**:
    *   Allows users to search for any word to retrieve its total find count across all dictionaries since Morpheme began.
    *   Renders a table of the 10 most recent users who found the word, showing the finder's country flag (rendered as an image) and username, and the formatted local/UTC date it was found.
    *   Binds click events to the finder rows to display their mini-profile overlay (`window.showMiniProfile(username)`).
    *   Fully optimized for responsive viewports on desktop, laptop, and mobile screens.

### 4. Country Flag Image Rendering (Windows/Desktop Fix)
*   **The Issue**: Windows desktop browsers do not support flag emojis natively, falling back to rendering plain two-letter abbreviations (e.g. "CA" instead of the Canadian flag).
*   **The Fix**:
    *   Defined global helpers `window.emojiToCountryCode` and `window.getFlagHtml` at the top of the frontend code.
    *   Instead of raw emoji characters, the helper dynamically renders a crisp flag image hosted on `flagcdn.com` using inline styles that align and scale perfectly with text.
    *   Integrated flag image rendering globally across all key views: Forum, Leaderboards, Gameplay Player List, Tools/Profiles, and the country selection dropdown lists.

### 5. Registration Flag Selection Requirement
*   **Feature**: Required flag selection during user registration to ensure every new registered user has a location representing where they live.
*   **Implementation**:
    *   Added a styled country flag selection select dropdown in the registration popup form (`#signup-form` in `index.html`), matched with input styling in `style.css`.
    *   Populated the dropdown options dynamically from the globally exposed `ALL_FLAGS` country catalog.
    *   Validated country selection in `handleSignUp()` and submitted the flag parameter in the register payload.
    *   Updated the backend `/api/register` endpoint in `app.py` to validate flag selection and write the user flag directly to the database.

---

## 🛠 Active Features & Configuration
*   **Tools Suite**: Word of the Day, Random Word, Unscramble, Find Count, and Personal Timer.
*   **Board Formats**: Normal (72%), Checkerboard (12%), Double (1%), Triple (1%), and other special variants.
*   **Grid Dimensions**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
*   **Verification**: Tested and deployed to the live production server `morpheme.games` on port 443 (via PM2 process `0`).

---

**Latest Stable Commit ID**: `e3806022cd27ab1032f13ac81b33b5cf7739676f`  
**Localhost & GitHub Sameness Status**: Synchronized  
**Production Server Status**: Green / PM2 Online / Live  
