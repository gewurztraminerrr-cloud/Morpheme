# Morpheme Stable State Summary - June 12, 2026

This summary documents the stable state of the Morpheme application as of June 12, 2026. The synchronization across Localhost, GitHub origin, and `morpheme.games` is fully completed and verified.

---

## 🚀 Key Improvements & Bug Fixes

### 1. Find Count Tool
*   **Feature**: Integrated a new "Find Count" tool inside the Tools sidebar navigation, positioned immediately before the **Personal Timer**.
*   **Functionality**:
    *   Allows users to search for any word to retrieve its total find count across all dictionaries since Morpheme began.
    *   Renders a table of the 10 most recent users who found the word, showing the finder's country flag (rendered as an image) and username, and the formatted local/UTC date it was found.
    *   Binds click events to the finder rows to display their mini-profile overlay (`window.showMiniProfile(username)`).
    *   Fully optimized for responsive viewports on desktop, laptop, and mobile screens.
*   **Asset Cache Busting**: Bushed static script and style caches to force client reloading.

### 2. Country Flag Image Rendering (Windows/Desktop Fix)
*   **The Issue**: Windows desktop browsers do not support flag emojis natively, falling back to rendering plain two-letter abbreviations (e.g. "CA" instead of the Canadian flag).
*   **The Fix**:
    *   Defined global helpers `window.emojiToCountryCode` and `window.getFlagHtml` at the top of the frontend code.
    *   Instead of raw emoji characters, the helper dynamically renders a crisp flag image hosted on `flagcdn.com` using inline styles that align and scale perfectly with text.
    *   Integrated flag image rendering globally across all key views:
        *   **Forum**: In post list card metadata, thread details author info, and comment header author names.
        *   **Leaderboards**: Next to usernames in the ranking rows.
        *   **Gameplay (Player List)**: Under player names in the active list.
        *   **Tools/Profiles**: In the mini-profile popup, the main user profile view, the optimistic profile flag update UI, the mini friend cards, and the Find Count search results table.
        *   **Dropdown Selector**: In the profile country selection list dropdown items.
    *   **Cache Busting**: Incremented the style and script versions (`style.css?v=29`, `forum.js?v=39`, `app.js?v=42`, `tools.js?v=44`) to ensure instant loading.

### 3. Registration Flag Selection Requirement
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

**Latest Stable Commit ID**: `[TO_BE_REPLACED]`  
**Localhost & GitHub Sameness Status**: Synchronized  
**Production Server Status**: Green / PM2 Online / Live  
