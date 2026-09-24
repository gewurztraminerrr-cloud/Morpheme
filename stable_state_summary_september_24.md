# Stable State Summary — September 24, 2026

> **Start Over Point**: All environments (localhost, GitHub, production `morpheme.games`, and mobile web app) are 100% synchronized and verified.  
> **Commit**: `5bd456ae6ce08461888e6a634132792c2b4fd715` (short: `5bd456ae`)  
> **Tags**: `START_OVER_POINT_SEPTEMBER_24`, `stable-2026-09-24`  
> **Branch**: `main` — `origin/main`  
> **Server**: `132.148.72.249` (`morpheme.games`) — PM2 process `morpheme` (ID: 0), online ✅  
> **Date/Time**: 2026-09-24 13:04 CDT  

---

## 1. Repository & Deployment Synchronization

| Environment | Host / Path | Status | Tags |
| :--- | :--- | :--- | :--- |
| **Localhost** | `/Users/jeffbabiak/` | ✅ Clean & Synchronized (`5bd456ae`) | `START_OVER_POINT_SEPTEMBER_24`, `stable-2026-09-24` |
| **GitHub** | `https://github.com/gewurztraminerrr-cloud/Morpheme` (`main`) | ✅ Pushed & Synchronized (`5bd456ae`) | `START_OVER_POINT_SEPTEMBER_24`, `stable-2026-09-24` |
| **Production Server** | `132.148.72.249` (`morpheme.games`) | ✅ Deployed & Online (PM2 process 0 healthy) | `5bd456ae` Synchronized |
| **Mobile Web / App** | All client platforms (iOS / Android / Desktop) | ✅ Verified (0ms gateway hydration, audio engine) | Synchronized |

---

## 2. Key Features, Improvements & Fixes (September 24, 2026)

### A. Word Lists Tool — Expanded "Likelihood" Dropdown Suite
1. **Renamed & Grouped Likelihood Options (`templates/index.html`, `static/js/tools.js`)**:
   - Renamed existing `Likelihood` to **`NWL Likelihood`** (`value="nwl_likelihood"`).
   - Added **`CSW Likelihood`** (`value="csw_likelihood"`).
   - Added **`CSW Only Likelihood`** (`value="csw_only_likelihood"`).
   - Added **`AW Likelihood`** (`value="added_likelihood"`).
   - Added **`ALL Likelihood`** (`value="all_likelihood"`).
   - Placed all five Likelihood options consecutively together in the dropdown selector.
   - Expanded select element width from `130px` to `190px` to ensure full label visibility without clipping.
2. **Backend Scrabble Likelihood Calculations (`app.py`)**:
   - Implemented `build_likelihood_list()` helper to compute Scrabble tile-frequency likelihood scores and sort descending by score, then alphabetically by word.
   - Handled `nwl_likelihood` (with backward compatibility alias for `likelihood`), `csw_likelihood`, `csw_only_likelihood`, `added_likelihood`, and `all_likelihood`.
   - Included lazy-loading for CSW dictionary when querying CSW-dependent likelihood lists.
3. **Frontend List & Modal Display Compatibility (`static/js/tools.js`)**:
   - Updated `currentWordsType` and `wordType` checks to test `includes('likelihood')` so score badges render cleanly across all five likelihood types.
   - Updated full list modal word search and jumping logic to index and jump through likelihood objects properly.
   - Updated `typeMap` to display exact titles for each likelihood list.

### B. White Layout Readability & Contrast Polish
1. **Numerical Value Readability in Likelihood Lists (`static/css/style.css`)**:
   - Styled `.likelihood-score` across `theme-white`, `theme-light-*`, and all light variants with high-contrast `#6d28d9` (dark violet/purple), `font-weight: 700`, and full `opacity: 1 !important` (remedying the washed-out light blue color on white backgrounds).
   - Applied to both `#main-list-results .list-item .likelihood-score` (the main scroll area) and `#full-list-modal-results .full-list-item .likelihood-score` (the full list modal overlay).
   - Added hover state color (`#5b21b6`).
2. **Word Count Indicator Readability (`static/css/style.css`)**:
   - Styled `#main-list-count` and `.list-count` (e.g. `(10,000)`) on white and light layouts with slate grey `#475569`, `opacity: 1 !important`, and `font-weight: 600 !important` so count indicators are crisp and legible.
3. **Pronunciation Contrast & Formatting Fixes**:
   - Preserved fix for pronunciation contrast and extra spacing in definition popup cards.

---

## 3. Invariants & Rules Preserved
- **Mobile Fullscreen & Soft Keyboard Rule**: Fullscreen exits cleanly when opening non-game utility pages and modal dialogs with inputs (`static/js/tools.js`), preventing Android Chrome black-screen display rebuilds.
- **Gateway Screen Integrity**: Immediate fullscreen engagement preserved continuously into the lobby without viewport dimension shift.
- **Dictionary Rule Compliance**: Plural noun and verb conjugation sourcing integrity preserved across word lists.

---

## 4. Verification Check
- **API Tests**:
  - `GET /api/tools/lists?list_type=nwl_likelihood&length=4`: OK (4,247 results)
  - `GET /api/tools/lists?list_type=csw_likelihood&length=4`: OK (5,662 results)
  - `GET /api/tools/lists?list_type=csw_only_likelihood&length=4`: OK (1,420 results)
  - `GET /api/tools/lists?list_type=added_likelihood&length=4`: OK (2,914 results)
  - `GET /api/tools/lists?list_type=all_likelihood&length=4`: OK (8,581 results)
- **Live Assets Verification**:
  - Checked `/static/css/style.css?v=1790271000` on production server `132.148.72.249`: verified `#main-list-count` and `.likelihood-score` high-contrast rules are served live.
