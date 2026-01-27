// Lobby functions

// Global state for polling
let lobbyPollInterval = null;
let lobbyStatsInterval = null;
let currentLobbyConfig = null;

// Use event delegation on the lobby page container
// This ensures clicks work even after navigating away and back
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded - setting up Lobby event delegation');
    const lobbyPage = document.getElementById('page-lobby');

    if (!lobbyPage) {
        console.error('Lobby page not found!');
        return;
    }

    // Event delegation for all button clicks in the lobby
    lobbyPage.addEventListener('click', async (e) => {
        const target = e.target;

        // Handle Accumulative button clicks - create room and go to Play
        const accBtn = target.closest('.acc-btn');
        if (accBtn) {
            // Stop any existing polling since we are leaving the lobby flow (or entering a non-polled flow)
            stopLobbyPolling();

            // If already in a room, leave it first
            if (window.currentRoomId) {
                console.log('Leaving current room before creating new one...');
                if (window.leaveCurrentRoom) {
                    await window.leaveCurrentRoom();
                }
            }

            const gameType = accBtn.dataset.game;
            const timeLimit = parseInt(accBtn.dataset.time);
            const boardDimensions = accBtn.dataset.board;

            console.log('Accumulative button clicked! Creating room:', { gameType, timeLimit, boardDimensions });

            try {
                console.log('Sending fetch to /api/room/create...');
                const response = await fetch('/api/room/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        game_type: gameType,
                        time_limit: timeLimit,
                        board_dimensions: boardDimensions
                    })
                });

                console.log('Response received, status:', response.status);
                const data = await response.json();
                console.log('Room creation response data:', data);

                if (data.success) {
                    console.log('SUCCESS! Room ID:', data.room_id);
                    // Store room ID globally for play.js
                    window.currentRoomId = data.room_id;

                    // Enable Play button
                    document.getElementById('play-btn').disabled = false;

                    // Navigate to Play
                    console.log('Navigating to Play page...');
                    showPage('page-play');

                    // FORCE FOCUS: Focus input immediately after navigation (User Click Context)
                    // This is the most reliable way to handle "Join Game" focus
                    setTimeout(() => {
                        const input = document.getElementById('word-input');
                        if (input) {
                            input.disabled = false; // Ensure enabled
                            input.focus();
                            console.log('Lobby: Focused word-input');
                        }
                    }, 100);

                    // Start polling if play.js is loaded
                    if (window.startGamePolling) {
                        console.log('Starting game polling...');
                        window.startGamePolling();
                    } else {
                        console.error('window.startGamePolling not found!');
                    }
                } else {
                    console.error('Room creation failed:', data.error);
                    alert('Failed to create room: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                console.error('Error creating room:', error);
                alert('Network error: ' + error.message);
            }
            return; // Handled
        }

        // Handle FCFS and Split buttons - update Active Rooms panel
        const listBtn = target.closest('.fcfs-btn, .split-btn');
        if (listBtn) {
            const gameType = listBtn.dataset.game;
            const timeLimit = parseInt(listBtn.dataset.time);
            const boardDimensions = listBtn.dataset.board;

            // Update global config
            currentLobbyConfig = { gameType, timeLimit, boardDimensions };

            const gameNames = {
                'fcfs': 'First Come First Serve',
                'split': 'Split Points'
            };

            const infoEl = document.getElementById('selected-game-info');
            infoEl.innerHTML = `
                <strong>${gameNames[gameType]}</strong><br>
                Board: ${boardDimensions} | Time: ${formatTime(timeLimit)}
            `;

            const roomsList = document.getElementById('rooms-list');
            roomsList.innerHTML = '<p class="placeholder">Loading rooms...</p>';

            // Fetch immediately WITHOUT auto-create (just show list)
            await fetchAndRenderRooms(gameType, timeLimit, boardDimensions, false);

            // Start polling
            startLobbyPolling();
            return; // Handled
        }

        // Handle "Create Room" button click (inside the panel)
        const createBtn = target.closest('.confirm-create-room-btn');
        if (createBtn && currentLobbyConfig) {
            // Prompt for Rating limits
            let minRating = 0;
            let maxRating = 9999;

            const rangeStr = prompt("Enter Rating Range (Min-Max)\nExample: 1000-1500\nLeave empty for no limit", "");
            if (rangeStr) {
                const parts = rangeStr.split('-');
                if (parts.length === 2) {
                    minRating = parseInt(parts[0].trim()) || 0;
                    maxRating = parseInt(parts[1].trim()) || 9999;
                } else if (parts.length === 1 && parts[0].trim() !== '') {
                    // Assume single number is MIN rating
                    minRating = parseInt(parts[0].trim()) || 0;
                }
            }

            // Trigger creation via fetchAndRenderRooms
            const roomsList = document.getElementById('rooms-list');
            roomsList.innerHTML = '<p class="placeholder">Creating room...</p>';

            await fetchAndRenderRooms(
                currentLobbyConfig.gameType,
                currentLobbyConfig.timeLimit,
                currentLobbyConfig.boardDimensions,
                true, // ALLOW auto-create now
                minRating,
                maxRating
            );
            return;
        }

        // Handle Join Room logic (dynamic button inside rooms-list)
        const joinBtn = target.closest('.join-room-btn');
        if (joinBtn) {
            const roomId = joinBtn.dataset.room;
            // SPECIAL CASE: Return to current room
            if (window.currentRoomId && roomId === window.currentRoomId) {
                console.log('Returning to current room (no re-join needed)');
                showPage('page-play');
                // Ensure polling restarts
                if (window.startGamePolling) window.startGamePolling();
                return;
            }

            // Stop lobby polling
            stopLobbyPolling();

            // If already in a OTHER room, leave it first
            if (window.currentRoomId) {
                if (window.leaveCurrentRoom) {
                    await window.leaveCurrentRoom();
                }
            }
            const isSpectator = joinBtn.dataset.spectator === 'true';
            console.log(`Joining room: ${roomId} (Spectator: ${isSpectator})`);

            try {
                const response = await fetch(`/api/room/${roomId}/join`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ as_spectator: isSpectator })
                });
                const data = await response.json();

                if (data.success) {
                    window.currentRoomId = roomId;
                    if (data.role === 'spectator') {
                        window.isSpectatorMode = true;
                    } else {
                        window.isSpectatorMode = false;
                    }

                    showPage('page-play');
                    // FORCE FOCUS
                    setTimeout(() => {
                        const input = document.getElementById('word-input');
                        if (input && !window.isSpectatorMode) {
                            input.disabled = false;
                            input.focus();
                        }
                    }, 100);

                    if (window.startGamePolling) window.startGamePolling();
                } else {
                    alert('Failed to join room: ' + data.error);
                    startLobbyPolling(); // Resume polling
                }
            } catch (error) {
                console.error('Error joining room:', error);
                alert('Network error joining room');
                startLobbyPolling();
            }
            return; // Handled
        }
    });

    console.log('Lobby event delegation setup complete');
});

// Helper function to fetch and render rooms
async function fetchAndRenderRooms(gameType, timeLimit, boardDimensions, allowAutoCreate = false, minRating = 0, maxRating = 9999) {
    const roomsList = document.getElementById('rooms-list');

    // Safety check if we navigated away
    if (!isOnLobby()) {
        console.log('fetchAndRenderRooms called but not on lobby, ignoring');
        return;
    }

    try {
        // Fetch active rooms for this configuration
        const url = `/api/rooms?game_type=${gameType}&board_dimensions=${boardDimensions}&time_limit=${timeLimit}`;
        const response = await fetch(url);
        const data = await response.json();

        let rooms = (data && data.rooms) ? data.rooms : [];

        // AUTO-CREATE: If no rooms exist, AND we are allowed to auto-create (i.e. user action)
        if (rooms.length === 0 && allowAutoCreate) {
            console.log('No rooms found. Auto-creating a new room...');

            const createResp = await fetch('/api/room/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    game_type: gameType,
                    time_limit: timeLimit,
                    board_dimensions: boardDimensions,
                    min_rating: minRating,
                    max_rating: maxRating
                })
            });
            const createData = await createResp.json();

            if (createData.success && createData.room_id) {
                // Manually add the new room to the list so it appears immediately
                rooms.push({
                    room_id: createData.room_id,
                    player_count: 1,
                    max_players: 8,
                    min_rating: minRating,
                    max_rating: maxRating,
                    combined_rating: 1000,
                    state: 'waiting',
                    current_round: 0,
                    players: [{
                        username: window.currentUser || 'You',
                        rating: 1000
                    }]
                });
            } else {
                roomsList.innerHTML = '<p class="placeholder">Error creating room</p>';
                return;
            }
        }

        // Render rooms list logic with Create Button always available
        const createButtonHtml = `
            <div style="margin-bottom: 15px; text-align: center;">
                <button class="confirm-create-room-btn" style="
                    background: #2ecc71; 
                    color: white; 
                    border: none; 
                    padding: 8px 16px; 
                    border-radius: 4px; 
                    cursor: pointer; 
                    font-weight: bold;
                    width: 100%;
                ">+ Create Room</button>
            </div>
        `;

        if (rooms.length === 0) {
            roomsList.innerHTML = createButtonHtml + '<p class="placeholder">No active rooms found</p>';
        } else {
            const html = rooms.map(room => {
                const playersHtml = room.players.map(p => {
                    const ratingColor = window.getRatingColor ? window.getRatingColor(p.rating) : '#fff';
                    return `<span class="room-player-pill" title="Rating: ${p.rating}" style="display:inline-flex; align-items:center;">
                        <span style="background-color: ${ratingColor}; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; display:inline-block;"></span>    
                        ${p.username} (${p.rating})
                    </span>`;
                }).join('');

                // Logic for Join vs Spectate
                const isFull = room.player_count >= (room.max_players || 8);
                const roomMin = room.min_rating || 0;
                const roomMax = room.max_rating || 9999;

                let actionButtons = '';

                if (room.room_id === window.currentRoomId) {
                    actionButtons = `<button class="join-room-btn return-mode" data-room="${room.room_id}" style="background: #e67e22;">Return to Game</button>`;
                } else {
                    // Spectate Button
                    actionButtons += `<button class="join-room-btn watch-mode" data-room="${room.room_id}" data-spectator="true" style="background: #34495e; margin-right: 5px;">Spectate</button>`;

                    let ratingText = '';
                    if (roomMin > 0 || roomMax < 9999) {
                        ratingText = `(${roomMin}-${roomMax < 9999 ? roomMax : '∞'})`;
                    }

                    if (!isFull) {
                        actionButtons += `<button class="join-room-btn" data-room="${room.room_id}" data-min-rating="${roomMin}">
                            Join ${ratingText}
                        </button>`;
                    } else {
                        actionButtons += `<button class="join-room-btn disabled" disabled style="opacity:0.5; cursor:not-allowed;">Full</button>`;
                    }
                }

                return `
                <div class="room-item">
                    <div class="room-header-row">
                        <div class="room-status ${room.state}">${room.state.toUpperCase()}</div>
                        <div class="room-meta">
                            Rating: ${room.combined_rating} 
                            ${(roomMin > 0 || roomMax < 9999) ? `<span style="color:#e74c3c; margin-left:5px; font-size:0.9em;">Req: ${roomMin}-${roomMax < 9999 ? roomMax : '∞'}</span>` : ''}
                        </div>
                    </div>
                    <div class="room-players-row">
                        ${playersHtml}
                    </div>
                    ${actionButtons}
                </div>
                `;
            }).join('');

            // Put Create button at the top
            roomsList.innerHTML = createButtonHtml + html;
        }

    } catch (error) {
        console.error('Error fetching rooms:', error);
        roomsList.innerHTML = '<p class="placeholder">Error loading rooms</p>';
    }
}

function startLobbyPolling() {
    stopLobbyPolling(); // Clear existing
    console.log('Starting lobby polling...');

    lobbyPollInterval = setInterval(() => {
        if (isOnLobby() && currentLobbyConfig) {
            // Poll without auto-create
            fetchAndRenderRooms(
                currentLobbyConfig.gameType,
                currentLobbyConfig.timeLimit,
                currentLobbyConfig.boardDimensions,
                false
            );
        } else {
            // If we're not on lobby or no config, stop polling to save resources
            console.log('Stopping lobby polling (not on lobby or no config)');
            stopLobbyPolling();
        }
    }, 3000); // 3 seconds
}

function stopLobbyPolling() {
    if (lobbyPollInterval) {
        clearInterval(lobbyPollInterval);
        lobbyPollInterval = null;
        console.log('Lobby polling stopped.');
    }
}

// Stats Polling
function startStatsPolling() {
    stopStatsPolling();
    console.log('Starting lobby stats polling...');

    // Initial fetch
    fetchLobbyStats();

    lobbyStatsInterval = setInterval(() => {
        if (isOnLobby()) {
            fetchLobbyStats();
        } else {
            stopStatsPolling();
        }
    }, 5000); // 5 seconds
}

function stopStatsPolling() {
    if (lobbyStatsInterval) {
        clearInterval(lobbyStatsInterval);
        lobbyStatsInterval = null;
        console.log('Stats polling stopped.');
    }
}

async function fetchLobbyStats() {
    try {
        const response = await fetch('/api/lobby-stats');
        const data = await response.json();

        if (data.stats) {
            updateLobbyButtons(data.stats);
        }
    } catch (error) {
        console.error('Error fetching lobby stats:', error);
    }
}

function updateLobbyButtons(stats) {
    // Stats format: "game_type|board|time": count

    // Reset all buttons to [0] first? Or just update known ones?
    // Safer to just update based on keys match.
    // Actually, if a count goes to 0, it might disappear from stats (if map is sparse).
    // If no rooms exist for a config, key won't be in stats.

    const buttons = document.querySelectorAll('.game-btn');
    buttons.forEach(btn => {
        const game = btn.dataset.game;
        const board = btn.dataset.board;
        const time = btn.dataset.time;

        const key = `${game}|${board}|${time}`;
        const count = stats[key] || 0;

        // Update text: "Start [N]" or "Show Rooms [N]"
        // Preserve prefix
        const currentText = btn.textContent;
        // Regex to find [N]
        const newText = currentText.replace(/\[\d+\]/, `[${count}]`);

        if (currentText !== newText) {
            btn.textContent = newText;
        }
    });
}

function formatTime(seconds) {
    const s = parseInt(seconds);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m`;
    if (s < 86400) return `${Math.floor(s / 3600)}h`;
    return `${Math.floor(s / 86400)}d`;
}

// Lobby music is handled by app.js to ensure consistent state management

// Observe page changes to handle polling
const lobbyPage = document.getElementById('page-lobby');
if (lobbyPage) {
    const observer = new MutationObserver(() => {
        if (isOnLobby()) {
            // Ensure we resume polling if coming back to lobby
            // If we navigate in app, 'isOnLobby' changes.
            if (currentLobbyConfig && !lobbyPollInterval) {
                startLobbyPolling();
            }

            // Always start stats polling when entering lobby
            if (!lobbyStatsInterval) {
                startStatsPolling();
            }

        } else {
            stopLobbyPolling();
            stopStatsPolling();
        }
    });

    observer.observe(lobbyPage, {
        attributes: true,
        attributeFilter: ['class']
    });
}

function isOnLobby() {
    const lobbyPage = document.getElementById('page-lobby');
    return lobbyPage && lobbyPage.classList.contains('active');
}

console.log('lobby.js fully loaded - version with polling');
