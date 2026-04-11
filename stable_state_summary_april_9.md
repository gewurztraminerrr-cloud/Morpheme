# Stable State Summary - April 9, 2026

## Overview
As of April 9, 2026, the Morpheme project has reached a high level of stability and performance. Key systems for round transitions, scoring, and board generation have been optimized and synchronized between the backend and frontend.

## Key Achievements

### 1. Proactive Search System
- **Lead-Time Optimization:** The system now begins generating parameters and searching for the *next* round's board immediately (5 seconds into the *current* active round).
- **Result:** Provides the board generator with the maximum possible lead time (Round Duration + Intermission), completely eliminating stalls at 0:00, even for complex 500+ word boards.

### 2. Scoring Breakdown UI
- **Detailed Math:** Updated the "All Words" list in the intermission to show full scoring breakdowns (e.g., "2 + 3 = 5") when words match the hidden bonus word or utilize special bonus tiles.
- **Backend Sync:** Integrated `return_details=True` in `scoring.py` across all generation paths, including the emergency fallback, to ensure breakdown data is always available.

### 3. Difficulty Distribution & Labeling
- **Strict Weights:** Enforced a distribution of **25% Easy (1-35%)**, **50% Medium (36-54%)**, and **25% Hard (55%+)** across all rooms.
- **Label Synchronization:** Corrected `board_generator.py` thresholds to ensure "Hard" mode boards are accurately identified and displayed in the UI, particularly for 4x4 grids.

### 4. Transition Reliability
- **Atomic State Promotion:** Hardened `RoomManager` to ensure all metadata (words, scores, paths, bonus cells) is swapped at the same instant as the board.
- **UI Persistence:** Refined the Spinner Set display to maintain visibility of the "Word Count Range" target during active rounds, preventing UI flickering.

## Current Configuration
- **Port:** 5001
- **Management:** `run_morpheme.sh`
- **Primary Files:**
    - `game_room.py`: Authoritative state machine and proactive milestone logic.
    - `spinner_set.py`: Parameter generation and difficulty weighting.
    - `board_generator.py`: Board layout and uniqueness calibration.
    - `static/js/play.js`: Frontend state polling and word list rendering.
    - `scoring.py`: Detailed point calculation logic.

## GitHub
- **Main Branch:** Pushed to `main` on `https://github.com/gewurztraminerrr-cloud/Morpheme.git`.
- **Commit Message:** "Stable State - April 9: Proactive Search, Scoring Breakdown, and Difficulty Refinement"
