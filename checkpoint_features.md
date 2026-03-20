# Morpheme Application Checkpoint - March 20, 2026

This document records the state of the "Morpheme" application at the time of the "Start Over" save point request.

## Core Features Saved

### 1. Game Mechanics & Formats
- **Standard Word Game**: Core logic for finding words on a board.
- **Penalty Format**: Invalid words that are present on the board incur a -3 point penalty instead of being counted as valid.
- **Either/Or Format**: Supports tiles with two possible letters (e.g., "C/H"). Using these tiles grants a +3 bonus.
- **Dynamic Board Generation**: Configurable board dimensions and letter distributions.

### 2. Room & Match Management
- **Public & Private Rooms**: Create and join matches with various configurations (Time limit, board size, format).
- **24-Hour Persistent Rooms**: Long-running rooms that persist beyond a single session.
- **Tournament System**: Logic for managing structured tournaments and round-based play.
- **Private Matchmaking**: Dedicated logic for private games.

### 3. User & Social Features
- **Authentication**: Full registration and login system, including Guest login support.
- **Profiles & Avatars**: User profile pages with custom avatar uploads and persistence.
- **Lobby**: Real-time lobby showing active rooms and player counts.
- **Skill Rankings**: Leaderboards for "Day", "Week", and "All-time" across "Best Scores" and "Best Word Counts".
- **Achievement System**: Tracking personal bests and room-specific achievements.

### 4. Interactive Tools
- **"Is Valid" Tool**: Validates if a word exists in the dictionary.
- **Pronunciation Lookup**: Integrated phonetic spelling and pronunciation help within the validation tool.
- **Presence Tracking**: Real-time tracking of users entering/leaving the application.

### 5. UI Organization & UX Improvements (NEW)
- **Organized Settings UI**: "Synesthesia" and "Next Round Notification" settings are now in their own dedicated panels for better organization.
- **Improved Settings Interactivity**: Restored visual indicator for bell/beep sound selection when "Next Round Notification" is enabled.
- **De-cluttered Tool Sidebar**: Simplified tool selection by prioritizing core features and removing secondary tools.
- **Consistent Theme Support**: Optimized CSS for Light and Dark modes with readability fixes for popups and overlay text.

### 6. Platform Integrity
- **Inactivity/Idle Timeout**: Automatic removal from rooms after 10 minutes of inactivity to maintain game flow.
- **Leave Penalty**: Rating penalties for prematurely leaving active matches.
- **Database Snapshots**: Mechanism for state persistence and stability backups.

---
**Checkpoint Git Tag**: `save_point_2026-03-20`
**Current Commit**: `b5f7b706bccb597008edad6ea0eff8255a442c5a`
