# Morpheme Stable State Summary - April 11, 2026

This summary documents the stable state of the Morpheme application as of April 11, 2026. The focus of recent work has been on UI polish, navigation robustness, and fixing persistent modal issues.

## 🚀 Key Improvements & Bug Fixes

### 1. Navigation & UI Polish
- **Dynamic Highlights**: Top navigation buttons (Lobby, Play, Store, etc.) now correctly highlight based on the active page.
- **Lobby Persistence**: Fixed the issue where the Lobby button would stay purple even when on other pages.
- **Modal Stability**: Standardized the use of `.forced-show` and `.hidden` classes. All modals (Mini Profile, Image Lightbox, History Review, How to Play) now close reliably via their "X" buttons and escape key listeners.

### 2. Game Format Logic
- **Normal Format Frequency**: Restored the intended **80% frequency** for the "Normal" board format. Fixed a logic bug that was preventing Normal boards from repeating, which had slashed their actual appearance rate to ~44%.
- **Variety Enforcement**: Maintained variety for special formats (Density, Penalty, Mania, etc.) by ensuring they don't appear back-to-back, while allowing "Normal" to serve as the consistent baseline.

### 3. FAQ Updates
- **Density Format**: Added a detailed explanation of the "Density" format to the FAQ. 
    - *Definition*: Uses high-contrast grayscale shading to represent "lexical weight."
    - *Visuals*: Darker tiles indicate higher word-usage frequency (common words), while lighter tiles indicate rarer vocabulary.

### 4. Technical Stability
- **Round Transitions**: Hardened the atomic transition logic to ensure 0:00 transitions are instantaneous.
- **Error Handling**: Removed legacy technical notices like "Checking dictionary accuracy..." to provide a cleaner user experience.
- **Session Restoration**: Implemented "Self-Healing" for public hub sessions to prevent accidental inactivity kicks during server reconnections.

## 🛠 Active Features & Configuration
- **Board Sizes**: 4x4, 4x6, 5x7, 6x8, and 3x3x3 Cube.
- **Dictionaries**: NWL (American) and CSW (International).
- **Difficulty Tiers**: Easy, Medium, Hard, and Expert (based on uniqueness ratio).
- **Game Modes**: Standard, Accumulative, FCFS, Split, and Private Matches.

## 📌 Next Steps
- Continue monitoring tournament fairness and rating redistribution.
- Enhance mobile responsiveness for larger 6x8 and 3D Cube formats.
- Finalize the achievement and trophy system integration.

---
**Stable Point Created**: April 11, 2026  
**GitHub Push**: Completed  
**Status**: Stable / Production Ready
