# Morpheme Stable State Summary - June 28, 2026

This document summarizes the stable state of **Morpheme** as of June 28, 2026. All local changes, remote code on GitHub, and the live application running on morpheme.games are fully synchronized.

## Latest Commit Information
* **Commit ID**: `21f2a056cd0b3de6aa189301d71b31cb05ee018e` (Short: `21f2a05`)
* **Branch**: `main`
* **Commit Message**: "fix: revert to pure forward evaluation in Combo Checker to align with game moves and fix AGRESTIAL cost"
* **Date**: June 28, 2026

## Changes Included in this Stable State

1. **Pure Forward Morpheme Metric & Performance Optimization in Combo Checker**:
   * Reverted the evaluation back to the **pure forward direction** (`search_term -> candidate`). This aligns with the one-way nature of the game moves and ensures that when you search for `AGRESTIC`, `AGRESTIAL` is correctly listed under **2MP** (not 1MP).
   * Restrained candidate matching to a maximum absolute length difference of **3** (`abs(len(A) - len(B)) <= 3`), which is the maximum allowed move length in the game of Morpheme.
   * Extended the LIC (Letters in Common) limit: if the search term has length $x$, the LIC tables can include words with lengths up to **$x + 4$** (as long as `target_len <= count + 4`).
   * **Performance Optimizations for Long Words (like `AGRESTIC`)**:
     * Implemented **dynamic max_mp limits** based on search term length to prevent combinatorial explosion: `max_mp = 6` for length $\le 5$, `max_mp = 4` for length 6, and `max_mp = 3` for length $\ge 7$.
     * Implemented **subsequence/substring short-circuiting** in `calculate_morpheme_metric` (avoiding backtracking entirely if a substring match is found).
     * Implemented **early backtracking pruning** by passing `limit = max_mp` to the backtracking engine, causing it to exit immediately when the cost exceeds the limit.
     * Implemented **linearity-based early exit**: since `MP >= target_len - linearity`, the function exits immediately if `linearity < target_len - limit`, bypassing backtracking for 95% of candidates.
     * These optimizations reduce the search time for an 8-letter word like `AGRESTIC` from **85 seconds (which timed out) to under 5.5 seconds** (and cut the loop time in half compared to the bidirectional version)!
   * For example:
     * Searching `CITER` shows `RETICULA` under **3MP** (since `CITER -> RETICULA` is 3).
     * Searching `AGRESTIC` shows `CITER` under **1MP** (since `AGRESTIC -> CITER` is 1).
     * Searching `RETICULA` shows `CITER` under **0MP** (since `RETICULA -> CITER` is 0).
     * Searching `GLOTTAL` shows `EPIGLOTTAL` under **3MP** (since `GLOTTAL -> EPIGLOTTAL` is 3).
     * Searching `EPIGLOTTAL` shows `GLOTTAL` under **0MP** (since `EPIGLOTTAL -> GLOTTAL` is 0).
     * Searching `CITER` (5) includes `RETICULA` (8) and `RETICULE` (8) in the **5LIC** table.

2. **Lobby Music Touch Device Autoplay & Stability**:
   * Unified `playLobbyMusicHelper` to use a play-then-seek pattern on all devices.
   * Ensured that on touch devices (iOS/Android), music autoplays seamlessly upon any initial user interaction (click/touch) without breaking.
   * Removed the debug text `"TEST"` that was appearing in front of `"JOURNEY"` in the Lobby.
