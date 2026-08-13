// Lobby functions

// Global state for polling
let lobbyPollInterval = null;
let lobbyStatsInterval = null;
let currentLobbyConfig = null;
window.activeRatingFilterValue = null;

function formatLobbyTime(seconds) {
    const s = parseInt(seconds);
    if (isNaN(s)) return '3m';
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m`;
    if (s < 86400) return `${Math.floor(s / 3600)}h`;
    return `${Math.floor(s / 86400)}d`;
}

function showLobbyToast(message, type = 'info') {
    let toast = document.getElementById('lobby-action-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'lobby-action-toast';
        toast.style.cssText = `
            position: fixed;
            top: 75px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(15, 23, 42, 0.95);
            color: #ffffff;
            border: 2px solid #38bdf8;
            padding: 12px 24px;
            border-radius: 30px;
            font-weight: 700;
            font-size: 0.95rem;
            z-index: 100000;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
            pointer-events: none;
            text-align: center;
        `;
        document.body.appendChild(toast);
    }
    toast.style.borderColor = (type === 'error') ? '#ef4444' : (type === 'success') ? '#22c55e' : '#38bdf8';
    toast.textContent = message;
    toast.style.opacity = '1';
    toast.style.display = 'block';

    if (window._lobbyToastTimeout) clearTimeout(window._lobbyToastTimeout);
    window._lobbyToastTimeout = setTimeout(() => {
        if (toast) {
            toast.style.opacity = '0';
            setTimeout(() => { toast.style.display = 'none'; }, 300);
        }
    }, 3500);
}
window.showLobbyToast = showLobbyToast;

async function enterLobbyRoom(rawBtn) {
    if (!rawBtn) return;
    const btn = (typeof rawBtn.closest === 'function') ? (rawBtn.closest('.game-btn, button') || rawBtn) : rawBtn;
    try {
        if (typeof stopLobbyPolling === 'function') stopLobbyPolling();
        const gameType = (btn.dataset && btn.dataset.game) ? btn.dataset.game : 'accumulative';
        const timeLimit = (btn.dataset && btn.dataset.time) ? (parseInt(btn.dataset.time) || 45) : 45;
        const boardDimensions = (btn.dataset && btn.dataset.board) ? btn.dataset.board : '4x4';

        showLobbyToast(`Entering ${gameType.toUpperCase()} (${boardDimensions}, ${formatLobbyTime(timeLimit)})...`);

        if (window.currentRoomId && window.leaveCurrentRoom) {
            try { await window.leaveCurrentRoom(); } catch (e) {}
        }
        window.currentRoomId = null;

        if (window.showLoadingOverlay) window.showLoadingOverlay('Entering Room...');
        localStorage.removeItem('tournament_play_active');
        localStorage.removeItem('private_match_active');

        let joinedId = null;
        const listResp = await fetch(`/api/rooms?game_type=${gameType}&board_dimensions=${boardDimensions}&time_limit=${timeLimit}&_t=${Date.now()}`, { cache: 'no-store' });
        if (listResp.ok) {
            const listData = await listResp.json();
            if (listData.rooms && listData.rooms.length > 0) {
                const existingId = listData.rooms[0].room_id;
                console.log('Found existing room, joining:', existingId);
                const joinResp = await fetch(`/api/room/${existingId}/join`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ as_spectator: false })
                });

                if (joinResp.ok) {
                    const joinData = await joinResp.json();
                    if (joinData.success) {
                        joinedId = existingId;
                    }
                }
            }
        }

        if (!joinedId) {
            console.log('Creating new room for config:', { gameType, timeLimit, boardDimensions });
            const createResp = await fetch('/api/room/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    game_type: gameType,
                    time_limit: timeLimit,
                    board_dimensions: boardDimensions
                })
            });

            if (createResp.ok) {
                const data = await createResp.json();
                if (data.success && data.room_id) {
                    joinedId = data.room_id;
                } else {
                    showLobbyToast('Failed to create room: ' + (data.error || 'Unknown error'), 'error');
                }
            } else {
                showLobbyToast('Server error creating room.', 'error');
            }
        }

        if (joinedId) {
            window.currentRoomId = joinedId;
            localStorage.setItem('last_joined_room', joinedId);
            window.isSpectatorMode = false;

            const playBtn = document.getElementById('play-btn');
            if (playBtn) {
                playBtn.disabled = false;
                playBtn.title = "";
            }
            if (window.updateManualToolState) window.updateManualToolState();
            
            if (typeof window.showPage === 'function') {
                window.showPage('page-play');
            } else if (typeof showPage === 'function') {
                showPage('page-play');
            }

            setTimeout(() => {
                const input = document.getElementById('word-input');
                if (input) {
                    input.disabled = false;
                    input.focus();
                }
            }, 100);

            if (window.startGamePolling) window.startGamePolling();
            showLobbyToast('Room joined successfully!', 'success');
        } else {
            startLobbyPolling();
        }
    } catch (error) {
        console.error('Error entering room:', error);
        showLobbyToast('Network error: ' + error.message, 'error');
        startLobbyPolling();
    } finally {
        if (window.hideLoadingOverlay) window.hideLoadingOverlay();
    }
}

async function handleAccumulativeClick(accBtn) {
    return enterLobbyRoom(accBtn);
}
window.handleAccumulativeClick = handleAccumulativeClick;

async function handleShowRoomsClick(listBtn) {
    if (typeof window.handleShowRoomsInline === 'function') {
        return window.handleShowRoomsInline(listBtn);
    }
    return enterLobbyRoom(listBtn);
}
window.handleShowRoomsClick = handleShowRoomsClick;

function handleLobbyButtonClickCore(btn, evt) {
    if (!btn) return;
    const realBtn = (typeof btn.closest === 'function') ? (btn.closest('.game-btn, button') || btn) : btn;
    const gameType = (realBtn && realBtn.dataset && realBtn.dataset.game) ? realBtn.dataset.game : 'accumulative';
    if (realBtn && (realBtn.classList.contains('acc-btn') || gameType === 'accumulative')) {
        enterLobbyRoom(realBtn);
    } else {
        handleShowRoomsClick(realBtn);
    }
}
window.handleLobbyButtonClickCore = handleLobbyButtonClickCore;
window.handleLobbyButtonClick = handleLobbyButtonClickCore;

// Use event delegation on document as fallback
function setupLobbyEvents() {
    console.log('Setting up Lobby event delegation');
    document.addEventListener('click', async (e) => {
        const rawTarget = e.target;
        if (!rawTarget) return;
        const target = rawTarget.nodeType === 3 ? rawTarget.parentElement : rawTarget;
        if (!target || typeof target.closest !== 'function') return;

        const accBtn = target.closest('.acc-btn');
        if (accBtn) {
            handleAccumulativeClick(accBtn);
            return;
        }

        const listBtn = target.closest('.fcfs-btn, .split-btn');
        if (listBtn) {
            handleShowRoomsClick(listBtn);
            return;
        }

        // Handle "Create Room" button click (inside the panel)
        const createBtn = target.closest('.confirm-create-room-btn');
        if (createBtn) {
            if (window.showLoadingOverlay) window.showLoadingOverlay('Creating Room...');
            console.log('Create Room clicked. Config:', currentLobbyConfig);

            if (!currentLobbyConfig) {
                console.error('Create Room failed: Missing lobby config');
                alert('Error: Game configuration not found. Please select a game type again.');
                return;
            }

            // Read from embedded inputs
            const panel = createBtn.closest('.create-room-panel');
            const minInput = panel.querySelector('.min-rating-input');
            const maxInput = panel.querySelector('.max-rating-input');

            let minRating = 0;
            let maxRating = 9999;

            if (minInput && minInput.value.trim() !== '') {
                minRating = parseInt(minInput.value);
            }
            if (maxInput && maxInput.value.trim() !== '') {
                maxRating = parseInt(maxInput.value);
            }

            if (isNaN(minRating)) minRating = 0;
            if (isNaN(maxRating)) maxRating = 9999;

            createRoom(minRating, maxRating);
            return;
        }

        async function createRoom(minRating, maxRating) {
            // CLEAR SPECIAL MODES: We are entering a normal room
            localStorage.removeItem('tournament_play_active');
            localStorage.removeItem('private_match_active');

            try {
                const createResp = await fetch('/api/room/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        game_type: currentLobbyConfig.gameType,
                        time_limit: currentLobbyConfig.timeLimit,
                        board_dimensions: currentLobbyConfig.boardDimensions,
                        min_rating: minRating,
                        max_rating: maxRating
                    })
                });
                
                if (!createResp.ok) {
                    const createErr = await createResp.text();
                    throw new Error(`Creation failed (${createResp.status}): ${createErr}`);
                }

                const data = await createResp.json();

                if (data.success) {
                    console.log('Room Created, Joining:', data.room_id);
                    // Join and go to play page
                    window.currentRoomId = data.room_id;
                    localStorage.setItem('last_joined_room', data.room_id);
                    window.isSpectatorMode = false; // Creator is always player
                    stopLobbyPolling();
                    const playBtn = document.getElementById('play-btn');
                    if (playBtn) {
                        playBtn.disabled = false;
                        playBtn.title = "";
                    }
                    if (window.updateManualToolState) window.updateManualToolState();
                    showPage('page-play');

                    // Force focus
                    setTimeout(() => {
                        const input = document.getElementById('word-input');
                        if (input) {
                            input.disabled = false;
                            input.focus();
                        }
                    }, 100);

                    if (window.startGamePolling) window.startGamePolling();
                } else {
                    alert('Failed to create room: ' + data.error);
                }
            } catch (e) {
                console.error('Creation error', e);
                alert('Error creating room: ' + e.message);
            }
        }


        // Handle Join Room logic (dynamic button inside rooms-list)
        const joinBtn = target.closest('.join-room-btn');
        if (joinBtn) {
            if (window.showLoadingOverlay) window.showLoadingOverlay('Joining Room...');
            joinBtn.style.opacity = '0.5';
            joinBtn.style.pointerEvents = 'none';
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

            // CLEAR SPECIAL MODES: We are entering a normal room
            localStorage.removeItem('tournament_play_active');
            localStorage.removeItem('private_match_active');

            try {
                const response = await fetch(`/api/room/${roomId}/join`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ as_spectator: isSpectator })
                });
                
                if (!response.ok) {
                    const err = await response.text();
                    throw new Error(`Join failed (${response.status}): ${err}`);
                }
                
                const data = await response.json();

                if (data.success) {
                    window.currentRoomId = roomId;
                    localStorage.setItem('last_joined_room', roomId);
                    if (data.role === 'spectator') {
                        window.isSpectatorMode = true;
                    } else {
                        window.isSpectatorMode = false;
                    }

                    const playBtn = document.getElementById('play-btn');
                    if (playBtn) {
                        playBtn.disabled = false;
                        playBtn.title = "";
                    }
                    if (window.updateManualToolState) window.updateManualToolState();



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
                alert('Network error joining room: ' + error.message);
                startLobbyPolling();
            }
            return; // Handled
        }
    });

    console.log('Lobby event delegation setup complete');

    // Setup Rating Filter Handlers (Find button and Enter key)
    function handleRatingFilterSearch() {
        const input = document.getElementById('rating-filter');
        if (input) {
            const val = input.value.trim();
            if (val === '') {
                window.activeRatingFilterValue = null;
            } else {
                const parsed = parseInt(val);
                window.activeRatingFilterValue = isNaN(parsed) ? null : parsed;
            }
            
            console.log('[Lobby] Rating filter search executed. Value:', window.activeRatingFilterValue);
            
            // Trigger render/update of rooms immediately using current configuration
            if (currentLobbyConfig) {
                fetchAndRenderRooms(
                    currentLobbyConfig.gameType,
                    currentLobbyConfig.timeLimit,
                    currentLobbyConfig.boardDimensions,
                    false
                );
            }
        }
    }

    const ratingFilterInput = document.getElementById('rating-filter');
    if (ratingFilterInput) {
        ratingFilterInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.keyCode === 13) {
                e.preventDefault();
                handleRatingFilterSearch();
            }
        });
        ratingFilterInput.addEventListener('input', () => {
            if (isOnLobby() && currentLobbyConfig) {
                const val = ratingFilterInput.value.trim();
                if (val === '') {
                    window.activeRatingFilterValue = null;
                } else {
                    const parsed = parseInt(val);
                    window.activeRatingFilterValue = isNaN(parsed) ? null : parsed;
                }
                fetchAndRenderRooms(
                    currentLobbyConfig.gameType,
                    currentLobbyConfig.timeLimit,
                    currentLobbyConfig.boardDimensions,
                    false
                );
            }
        });
    }

    const findRatingBtn = document.getElementById('find-rating-btn');
    if (findRatingBtn) {
        findRatingBtn.addEventListener('click', (e) => {
            e.preventDefault();
            handleRatingFilterSearch();
        });
    }

    const myRatingBtn = document.getElementById('my-rating-btn');
    if (myRatingBtn) {
        myRatingBtn.addEventListener('click', (e) => {
            e.preventDefault();
            if (!currentLobbyConfig) {
                console.log('[Lobby] My Rating clicked, but no active config. Doing nothing.');
                return;
            }
            let userRating = 1200;
            const gameType = currentLobbyConfig.gameType;
            const board = currentLobbyConfig.boardDimensions;
            const time = currentLobbyConfig.timeLimit;
            const configKey = `${gameType}|${board}|${time}`;
            const ratings = window.currentUserConfigRatings || {};
            const ratingObj = ratings[configKey];
            if (ratingObj && ratingObj.rating !== undefined) {
                userRating = ratingObj.rating;
            } else {
                if (gameType === 'fcfs' || gameType === 'split' || gameType === '3d') {
                    userRating = 1200;
                } else {
                    userRating = window.lastPlayerRating || 1200;
                }
            }
            const input = document.getElementById('rating-filter');
            if (input) {
                input.value = userRating;
            }
            window.activeRatingFilterValue = userRating;
            console.log('[Lobby] My Rating clicked. Value:', userRating);
            fetchAndRenderRooms(
                currentLobbyConfig.gameType,
                currentLobbyConfig.timeLimit,
                currentLobbyConfig.boardDimensions,
                false
            );
        });
    }

    // Start stats polling if we land on lobby
    if (isOnLobby()) {
        startStatsPolling();

        // Mobile layout: Snap to center panel on load
        const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
        if (isMobile) {
            setTimeout(() => {
                const mainPanel = document.getElementById('mobile-panel-main');
                if (mainPanel) {
                    const lobbyGrid = mainPanel.closest('.lobby-grid') || mainPanel.parentElement;
                    if (lobbyGrid) {
                        lobbyGrid.scrollLeft = mainPanel.offsetLeft;
                    }
                }
            }, 100);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupLobbyEvents);
} else {
    setupLobbyEvents();
}

// Helper function to fetch and render rooms
async function fetchAndRenderRooms(gameType, timeLimit, boardDimensions, allowAutoCreate = false, minRating = 0, maxRating = 9999, force = false) {
    const roomsList = document.getElementById('rooms-list');

    // Safety check if we navigated away (only check during background poll, bypass when force=true)
    if (!force && !isOnLobby()) {
        console.log('fetchAndRenderRooms called but not on lobby, ignoring');
        return;
    }

    // Ensure persistent structure for inputs so they aren't wiped on poll
    let roomsContainer = document.getElementById('dynamic-rooms-container');
    if (!roomsContainer) {
        const isGuest = window.currentUser && window.currentUser.startsWith('Guest_');
        const createButtonHtml = `
            <div class="create-room-panel" ${isGuest ? 'style="filter: grayscale(1); opacity: 0.7;"' : ''}>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9em; margin-bottom: 8px; text-align: center;">
                    ${isGuest ? 'Register to Create Custom Rooms' : 'Set Rating Limits (Optional)'}
                </div>
                <div class="rating-inputs-row">
                    <input type="number" class="rating-input min-rating-input" placeholder="Min Rating" min="0" step="100" ${isGuest ? 'disabled' : ''}>
                    <input type="number" class="rating-input max-rating-input" placeholder="Max Rating" min="0" step="100" ${isGuest ? 'disabled' : ''}>
                </div>
                <button class="confirm-create-room-btn" ${isGuest ? 'disabled style="cursor:not-allowed;"' : ''}>
                    ${isGuest ? 'Registered Only' : '+ Create Room'}
                </button>
            </div>
            <div id="dynamic-rooms-container" style="display: flex; flex-direction: column; gap: 12px;">
                <p class="placeholder">Loading rooms...</p>
            </div>
        `;
        roomsList.innerHTML = createButtonHtml;
        roomsContainer = document.getElementById('dynamic-rooms-container');
    }

    try {
        // Fetch active rooms for this configuration with cache busting
        const url = `/api/rooms?game_type=${gameType}&board_dimensions=${boardDimensions}&time_limit=${timeLimit}&_t=${Date.now()}`;
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
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
                        rating: (window.currentUser && window.currentUser.startsWith('Guest_')) ? 0 : 1000
                    }]
                });
            } else {
                roomsContainer.innerHTML = '<p class="placeholder">Error creating room</p>';
                return;
            }
        }

        // Recalculate average ratings
        rooms.forEach(room => {
            let totalRating = 0;
            let pCount = 0;
            if (room.players && Array.isArray(room.players)) {
                pCount = room.players.length;
                totalRating = room.players.reduce((sum, p) => {
                    const isGuest = p.username.startsWith('Guest_');
                    const rating = isGuest ? 0 : (p.rating || 0);
                    return sum + rating;
                }, 0);
            }
            const avgRating = pCount > 0 ? Math.round(totalRating / pCount) : 0;
            room.display_average_rating = avgRating;
        });

        // Apply Authoritative Rating Proximity Filter/Sort if "Find" or "Enter" has been executed
        let filteredRooms = [...rooms];
        const targetRating = window.activeRatingFilterValue;
        if (targetRating !== null && !isNaN(targetRating)) {
            // Sort rooms by closeness to entered average value (closest first)
            filteredRooms.sort((a, b) => {
                const diffA = Math.abs(a.display_average_rating - targetRating);
                const diffB = Math.abs(b.display_average_rating - targetRating);
                return diffA - diffB;
            });
        }

        if (filteredRooms.length === 0) {
            roomsContainer.innerHTML = '<p class="placeholder">No active rooms found matching criteria</p>';
        } else {
            const html = filteredRooms.map(room => {
                const playersHtml = room.players.map(p => {
                    // Override rating for Guest users
                    const isGuest = p.username.startsWith('Guest_');
                    const displayRating = isGuest ? 0 : p.rating;
                    const ratingColor = window.getRatingColor ? window.getRatingColor(displayRating) : '#fff';

                    return `<span class="room-player-pill" title="Rating: ${displayRating}" style="display:inline-flex; align-items:center;">
                        <span onclick="window.showMiniProfile('${p.username}')" style="background-color: ${ratingColor}; width: 11px; height: 11px; border-radius: 3px; margin-right: 5px; display:inline-block; cursor: pointer;"></span>    
                        ${p.username} (${displayRating})
                    </span>`;
                }).join('');

                // Logic for Join vs Spectate
                const isFull = room.player_count >= (room.max_players || 8);
                const roomMin = room.min_rating || 0;
                const roomMax = room.max_rating || 9999;
                const hasLimits = roomMin > 0 || roomMax < 9999;

                // Determine user's rating for this room configuration
                const userRating = getUserConfigRating(room.game_type, room.board_dimensions, room.time_limit);

                // Check rating restrictions for FCFS and SP (Split) rooms
                const gameTypeLower = String(room.game_type || '').toLowerCase();
                const isFcfsOrSp = gameTypeLower === 'fcfs' || gameTypeLower === 'split' || gameTypeLower === 'sp' || (room.room_id && (room.room_id.includes('fcfs') || room.room_id.includes('split')));
                const currentUser = window.currentUser || '';
                const isCurrentUserGuest = !currentUser || currentUser.startsWith('Guest_') || Boolean(window.currentUserIsGuest);
                const isRatingOutOfRange = hasLimits && (userRating < roomMin || userRating > roomMax || isCurrentUserGuest);

                let actionButtons = '';

                if (room.room_id === window.currentRoomId) {
                    actionButtons = `<button class="join-room-btn return-mode" data-room="${room.room_id}" style="background: #e67e22;">Return to Game</button>`;
                } else {
                    // Spectate Button - Always allowed for public rooms
                    actionButtons += `<button class="join-room-btn watch-mode" data-room="${room.room_id}" data-spectator="true" style="background: #34495e; margin-right: 5px;">Spectate</button>`;

                    let ratingText = '';
                    if (hasLimits) {
                        ratingText = `(${roomMin}-${roomMax < 9999 ? roomMax : '∞'})`;
                    }

                    // For FCFS and SP rooms (or any room with rating limits), if user rating is outside limit, ONLY allow spectate (Remove Join button)
                    if (isRatingOutOfRange) {
                        // Join green button is completely removed
                    } else if (!isFull) {
                        actionButtons += `<button class="join-room-btn" data-room="${room.room_id}" data-min-rating="${roomMin}">
                            Join ${ratingText}
                        </button>`;
                    } else {
                        actionButtons += `<button class="join-room-btn disabled" disabled style="opacity:0.5; cursor:not-allowed;">Full</button>`;
                    }
                }

                // Display limitations
                const ratingRangeText = `${roomMin} - ${roomMax < 9999 ? roomMax : '∞'}`;
                const hasLimits = roomMin > 0 || roomMax < 9999;

                return `
                <div class="room-item">
                    <div class="room-header-row">
                        <div class="room-status ${room.state}">${room.state.toUpperCase()}</div>
                        <div class="room-meta">
                            <span class="room-avg-rating">Avg Rating: ${room.display_average_rating}</span> 
                            ${hasLimits ? `<span class="rating-req-badge">Req: ${ratingRangeText}</span>` : ''}
                        </div>
                    </div>
                    <div class="room-players-row">
                        ${playersHtml}
                    </div>
                    ${actionButtons}
                </div>
                `;
            }).join('');

            if (roomsContainer.innerHTML !== html) {
                roomsContainer.innerHTML = html;
            }
        }

    } catch (error) {
        console.error('Error fetching rooms:', error);
        if (roomsContainer) {
            roomsContainer.innerHTML = '<p class="placeholder">Error loading rooms</p>';
        } else if (roomsList) {
            roomsList.innerHTML = '<p class="placeholder">Error loading rooms</p>';
        }
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
        const response = await fetch(`/api/lobby-stats?_t=${Date.now()}`, { cache: 'no-store' });
        const data = await response.json();

        if (data.stats) {
            updateLobbyButtons(data.stats);
        }
    } catch (error) {
        console.error('Error fetching lobby stats:', error);
    }
}
window.fetchLobbyStats = fetchLobbyStats;

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
            if (window.resetLobbyButtons) window.resetLobbyButtons();
            
            if (currentLobbyConfig) {
                if (typeof updateMyRatingButton === 'function') {
                    updateMyRatingButton(currentLobbyConfig.gameType, currentLobbyConfig.boardDimensions, currentLobbyConfig.timeLimit);
                }
                if (!lobbyPollInterval) {
                    startLobbyPolling();
                }
            }

            // Always start stats polling when entering lobby
            if (!lobbyStatsInterval) {
                startStatsPolling();
            }

            // Mobile layout: Snap to center panel on load
            const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
            if (isMobile) {
                setTimeout(() => {
                    const mainPanel = document.getElementById('mobile-panel-main');
                    if (mainPanel) {
                        const lobbyGrid = mainPanel.closest('.lobby-grid') || mainPanel.parentElement;
                        if (lobbyGrid) {
                            lobbyGrid.scrollLeft = mainPanel.offsetLeft;
                        }
                    }
                }, 100);
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
    const el = lobbyPage || document.getElementById('page-lobby');
    return el && el.classList.contains('active');
}

function getUserConfigRating(gameType, board, time) {
    if (!window.currentUser || window.currentUserIsGuest) return 0;
    const cleanType = String(gameType || '').replace('solo_', '');
    const configKey = `${cleanType}|${board || '4x4'}|${time || 180}`;
    const ratings = window.currentUserConfigRatings || {};
    const val = ratings[configKey];
    if (val !== undefined && val !== null) {
        if (typeof val === 'object' && val.rating !== undefined) {
            return Number(val.rating) || 1200;
        }
        if (typeof val === 'number') {
            return val;
        }
    }
    if (window.currentUserRating !== undefined && window.currentUserRating !== null) {
        return Number(window.currentUserRating) || 1200;
    }
    if (window.lastPlayerRating !== undefined && window.lastPlayerRating !== null) {
        return Number(window.lastPlayerRating) || 1200;
    }
    return 1200;
}
window.getUserConfigRating = getUserConfigRating;

function resetLobbyButtons() {
    const gameButtons = document.querySelectorAll('.game-btn, .acc-btn, .fcfs-btn, .split-btn');
    gameButtons.forEach(btn => {
        btn.disabled = false;
        btn.style.opacity = '1';
        btn.style.pointerEvents = 'auto';
    });
    const joinButtons = document.querySelectorAll('.join-room-btn, .confirm-create-room-btn');
    joinButtons.forEach(btn => {
        btn.disabled = false;
        btn.style.opacity = '1';
        btn.style.pointerEvents = 'auto';
    });
    const myRatingBtn = document.getElementById('my-rating-btn');
    if (myRatingBtn) {
        if (!window.currentUser || window.currentUserIsGuest) {
            myRatingBtn.style.display = 'none';
        } else {
            myRatingBtn.style.display = 'inline-block';
            if (window.currentLobbyConfig) {
                const rating = getUserConfigRating(
                    window.currentLobbyConfig.gameType,
                    window.currentLobbyConfig.boardDimensions,
                    window.currentLobbyConfig.timeLimit
                );
                myRatingBtn.textContent = `My Rating (${rating})`;
                myRatingBtn.dataset.rating = rating;
            } else {
                myRatingBtn.textContent = 'My Rating';
                myRatingBtn.removeAttribute('data-rating');
            }
        }
    }
}
window.resetLobbyButtons = resetLobbyButtons;

async function updateMyRatingButton(gameType, board, time) {
    const btn = document.getElementById('my-rating-btn');
    if (!btn) return;

    if (!window.currentUser || window.currentUserIsGuest) {
        btn.style.display = 'none';
        return;
    }

    // Set initial text synchronously
    const initialRating = getUserConfigRating(gameType, board, time);
    btn.textContent = `My Rating (${initialRating})`;
    btn.dataset.rating = initialRating;
    btn.style.display = 'inline-block';

    if (!window.currentUserConfigRatings || Object.keys(window.currentUserConfigRatings).length === 0) {
        if (typeof window.loadCurrentUserConfigRatings === 'function') {
            try {
                await window.loadCurrentUserConfigRatings();
                const updatedRating = getUserConfigRating(gameType, board, time);
                btn.textContent = `My Rating (${updatedRating})`;
                btn.dataset.rating = updatedRating;
            } catch (e) {
                console.error('[Lobby] Error loading config ratings:', e);
            }
        }
    }
}
window.updateMyRatingButton = updateMyRatingButton;

console.log('lobby.js fully loaded - version with polling');
