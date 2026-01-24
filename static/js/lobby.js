// Lobby functions

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
        if (target.classList.contains('acc-btn')) {
            const gameType = target.dataset.game;
            const timeLimit = parseInt(target.dataset.time);
            const boardDimensions = target.dataset.board;

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
        }

        // Handle FCFS and Split buttons - update Active Rooms panel
        else if (target.classList.contains('fcfs-btn') || target.classList.contains('split-btn')) {
            const game = target.dataset.game;
            const time = target.dataset.time;
            const board = target.dataset.board;

            const gameNames = {
                'fcfs': 'First Come First Serve',
                'split': 'Split Points'
            };

            const infoEl = document.getElementById('selected-game-info');
            infoEl.innerHTML = `
                <strong>${gameNames[game]}</strong><br>
                Board: ${board} | Time: ${formatTime(time)}
            `;

            const roomsList = document.getElementById('rooms-list');
            roomsList.innerHTML = '<p class="placeholder">No active rooms</p>';
        }
    });

    console.log('Lobby event delegation setup complete');
});

function formatTime(seconds) {
    const s = parseInt(seconds);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m`;
    if (s < 86400) return `${Math.floor(s / 3600)}h`;
    return `${Math.floor(s / 86400)}d`;
}

// Lobby music playback (3:25 to 4:55 loop)
const lobbyMusic = document.getElementById('lobby-music');
if (lobbyMusic) {
    const startTime = 205; // 3:25
    const endTime = 295;   // 4:55

    lobbyMusic.currentTime = startTime;

    lobbyMusic.addEventListener('timeupdate', () => {
        if (lobbyMusic.currentTime >= endTime) {
            lobbyMusic.currentTime = startTime;
        }
    });

    // Play on first click
    document.addEventListener('click', () => {
        if (lobbyMusic.paused && isOnLobby()) {
            lobbyMusic.play().catch(err => console.log('Playback failed:', err));
        }
    }, { once: true });

    // Observe page changes
    const observer = new MutationObserver(() => {
        if (isOnLobby()) {
            if (lobbyMusic.paused) {
                lobbyMusic.play().catch(err => console.log('Playback failed:', err));
            }
        } else {
            if (!lobbyMusic.paused) {
                lobbyMusic.pause();
            }
        }
    });

    const lobbyPage = document.getElementById('page-lobby');
    if (lobbyPage) {
        observer.observe(lobbyPage, {
            attributes: true,
            attributeFilter: ['class']
        });
    }
}

function isOnLobby() {
    const lobbyPage = document.getElementById('page-lobby');
    return lobbyPage && lobbyPage.classList.contains('active');
}

console.log('lobby.js fully loaded - version with detailed logging');
