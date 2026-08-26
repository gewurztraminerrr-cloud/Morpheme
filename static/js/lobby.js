// Lobby functions

// Global state for polling
let lobbyPollInterval = null;
let lobbyStatsInterval = null;
let currentLobbyConfig = null;
window.currentLobbyConfig = null;
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
    const duration = (type === 'error') ? 3000 : 1500;
    window._lobbyToastTimeout = setTimeout(() => {
        if (toast) {
            toast.style.opacity = '0';
            setTimeout(() => { toast.style.display = 'none'; }, 200);
        }
    }, duration);
}
window.showLobbyToast = showLobbyToast;

async function enterLobbyRoom(rawBtn) {
    if (!rawBtn) return;
    if (typeof window.checkAccountTimeoutAndAlert === 'function' && await window.checkAccountTimeoutAndAlert()) {
        return;
    }
    const btn = (typeof rawBtn.closest === 'function') ? (rawBtn.closest('.game-btn, button') || rawBtn) : rawBtn;
    try {
        if (typeof stopLobbyPolling === 'function') stopLobbyPolling();
        const gameType = (btn.dataset && btn.dataset.game) ? btn.dataset.game : 'accumulative';
        const timeLimit = (btn.dataset && btn.dataset.time) ? (parseInt(btn.dataset.time) || 45) : 45;
        const boardDimensions = (btn.dataset && btn.dataset.board) ? btn.dataset.board : '4x4';

        // 1. INSTANT NAVIGATION: Switch directly to the game room page immediately!
        window._isEnteringRoom = true;
        if (typeof window.clearGameUIAndCache === 'function') {
            window.clearGameUIAndCache();
        }
        if (typeof window.showPage === 'function') {
            window.showPage('page-play');
        } else if (typeof showPage === 'function') {
            showPage('page-play');
        }
        if (typeof window.switchPlayPanel === 'function') {
            window.switchPlayPanel('board', false);
        }

        const playBtn = document.getElementById('play-btn');
        if (playBtn) {
            playBtn.disabled = false;
            playBtn.title = "";
        }
        if (window.updateManualToolState) window.updateManualToolState();

        if (window.currentRoomId && window.leaveCurrentRoom) {
            try { window.leaveCurrentRoom().catch(function() {}); } catch (e) {}
        }
        window.currentRoomId = null;

        localStorage.removeItem('tournament_play_active');
        localStorage.removeItem('private_match_active');

        // 2. Perform room creation and loading inside the game room in the background
        const createResp = await fetch('/api/room/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                game_type: gameType,
                time_limit: timeLimit,
                board_dimensions: boardDimensions
            })
        });

        let data = null;
        try {
            data = await createResp.json();
        } catch(e) {
            const txt = await createResp.text().catch(() => '');
            data = { error: txt };
        }

        window._isEnteringRoom = false;

        if (createResp.status === 403 && data && data.timed_out) {
            const rText = data.timeout_reason || 'Moderator timeout';
            const dText = data.remaining || 'a temporary timeout';
            const msg = `You are currently placed on a temporary timeout from all game rooms.<br><br><strong>Reason:</strong> <span style="color: var(--text-primary); font-weight: 600;">${rText}</span><br><br><strong>Time Remaining:</strong> <span style="color: #f59e0b; font-size: 1.15rem; font-weight: 700;">${dText}</span><br><br>To keep matches fair and respectful for all players, room access is temporarily restricted during a timeout period.<br><br>Please wait until your timeout expires before joining another match!`;
            if (typeof window.showPage === 'function') window.showPage('page-lobby');
            if (window.showAlertModal) {
                window.showAlertModal('Account Timed Out', msg, true);
            } else {
                alert('Account Timed Out\n\nReason: ' + rText + '\nTime Remaining: ' + dText);
            }
            return;
        }

        if (createResp.ok && data && data.success && data.room_id) {
            window.currentRoomId = data.room_id;
            localStorage.setItem('last_joined_room', data.room_id);
            window.isSpectatorMode = false;

            if (data.state && typeof window.updateGameState === 'function') {
                window.updateGameState(data.state);
            }

            setTimeout(() => {
                const input = document.getElementById('word-input');
                if (input) {
                    input.disabled = false;
                    input.focus();
                }
            }, 50);

            if (window.startGamePolling) window.startGamePolling();
        } else {
            const errMsg = (data && data.error) ? data.error : 'Server error entering room.';
            if (typeof window.showPage === 'function') window.showPage('page-lobby');
            if (data && (data.timed_out || errMsg.toLowerCase().includes('timed out'))) {
                const durationText = data.remaining || "your timeout expires";
                const reasonText = data.timeout_reason || "Moderator timeout";
                const msg = `You are currently placed on a temporary timeout from all game rooms.<br><br><strong>Reason:</strong> <span style="color: var(--text-primary); font-weight: 600;">${reasonText}</span><br><br><strong>Time Remaining:</strong> <span style="color: #f59e0b; font-size: 1.15rem; font-weight: 700;">${durationText}</span><br><br>To keep matches fair and respectful for all players, room access is temporarily restricted during a timeout period.<br><br>Please wait until your timeout expires before joining another match!`;
                if (window.showAlertModal) {
                    window.showAlertModal('Account Timed Out', msg, true);
                } else {
                    alert(errMsg);
                }
            } else {
                if (window.showAlertModal) {
                    window.showAlertModal('Unable to Enter Room', errMsg);
                } else {
                    alert(errMsg);
                }
            }
        }
    } catch (error) {
        window._isEnteringRoom = false;
        console.error('Error entering room:', error);
        if (typeof window.showPage === 'function') window.showPage('page-lobby');
        showLobbyToast('Network error: ' + error.message, 'error');
    } finally {
        if (window.hideLoadingOverlay) window.hideLoadingOverlay();
    }
}

async function handleAccumulativeClick(accBtn) {
    return enterLobbyRoom(accBtn);
}
window.handleAccumulativeClick = handleAccumulativeClick;

async function handleShowRoomsClick(listBtn) {
    if (typeof window.checkAccountTimeoutAndAlert === 'function' && await window.checkAccountTimeoutAndAlert()) {
        return;
    }
    if (typeof window.handleShowRoomsInline === 'function') {
        return window.handleShowRoomsInline(listBtn);
    }
    return enterLobbyRoom(listBtn);
}
window.handleShowRoomsClick = handleShowRoomsClick;

async function handleLobbyButtonClickCore(btn, evt) {
    if (!btn) return;
    if (typeof window.checkAccountTimeoutAndAlert === 'function' && await window.checkAccountTimeoutAndAlert()) {
        return;
    }
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

async function createRoom(config, minRating, maxRating) {
    if (typeof window.checkAccountTimeoutAndAlert === 'function' && await window.checkAccountTimeoutAndAlert()) {
        return;
    }
    if (window._isCreatingRoom) {
        console.warn('[createRoom] Room creation already in progress, ignoring duplicate call');
        return;
    }
    window._isCreatingRoom = true;

    // CLEAR SPECIAL MODES: We are entering a normal room
    localStorage.removeItem('tournament_play_active');
    localStorage.removeItem('private_match_active');

    // 1. INSTANT NAVIGATION: Switch directly to the game room page immediately!
    window._isEnteringRoom = true;
    if (typeof window.clearGameUIAndCache === 'function') {
        window.clearGameUIAndCache();
    }
    if (typeof window.showPage === 'function') {
        window.showPage('page-play');
    } else if (typeof showPage === 'function') {
        showPage('page-play');
    }
    if (typeof window.switchPlayPanel === 'function') {
        window.switchPlayPanel('board', false);
    }

    const playBtn = document.getElementById('play-btn');
    if (playBtn) {
        playBtn.disabled = false;
        playBtn.title = "";
    }
    if (window.updateManualToolState) window.updateManualToolState();

    try {
        const createResp = await fetch('/api/room/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                game_type: config.gameType,
                time_limit: config.timeLimit,
                board_dimensions: config.boardDimensions,
                min_rating: minRating,
                max_rating: maxRating
            })
        });
        
        window._isEnteringRoom = false;

        if (!createResp.ok) {
            const createErr = await createResp.text();
            if (typeof window.showPage === 'function') window.showPage('page-lobby');
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
            
            if (data.state && typeof window.updateGameState === 'function') {
                window.updateGameState(data.state);
            }

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
            if (typeof window.showPage === 'function') window.showPage('page-lobby');
            alert(data.error || 'Unable to create room');
        }
    } catch (err) {
        window._isEnteringRoom = false;
        if (typeof window.showPage === 'function') window.showPage('page-lobby');
        console.error('Error creating room:', err);
        alert(err.message || 'Error creating room');
    } finally {
        window._isCreatingRoom = false;
        if (window.hideLoadingOverlay) window.hideLoadingOverlay();
        document.querySelectorAll('.confirm-create-room-btn').forEach(btn => {
            btn.disabled = false;
        });
    }
}
window.createRoom = createRoom;

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
            // Handled directly by inline onclick window.handleCreateRoomButtonClick
            return;
        }


        // Handle Join Room logic (dynamic button inside rooms-list)
        const joinBtn = target.closest('.join-room-btn');
        if (joinBtn) {
            if (typeof window.checkAccountTimeoutAndAlert === 'function' && await window.checkAccountTimeoutAndAlert()) {
                return;
            }
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

            // 1. INSTANT NAVIGATION: Switch directly to the game room page immediately!
            window._isEnteringRoom = true;
            if (typeof window.clearGameUIAndCache === 'function') {
                window.clearGameUIAndCache();
            }
            if (typeof window.showPage === 'function') {
                window.showPage('page-play');
            } else if (typeof showPage === 'function') {
                showPage('page-play');
            }
            if (typeof window.switchPlayPanel === 'function') {
                window.switchPlayPanel('board', false);
            }

            const playBtn = document.getElementById('play-btn');
            if (playBtn) {
                playBtn.disabled = false;
                playBtn.title = "";
            }
            if (window.updateManualToolState) window.updateManualToolState();

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
                
                let data = null;
                try {
                    data = await response.json();
                } catch(e) {
                    const txt = await response.text().catch(() => '');
                    data = { error: txt };
                }

                window._isEnteringRoom = false;

                if (!response.ok) {
                    if (typeof window.showPage === 'function') window.showPage('page-lobby');
                    const errMsg = (data && data.error) ? data.error : `Join failed (${response.status})`;
                    if (data && (data.timed_out || errMsg.toLowerCase().includes('timed out'))) {
                        const durationText = data.remaining || "your timeout expires";
                        const reasonText = data.timeout_reason || "Moderator timeout";
                        const msg = `You are currently placed on a temporary timeout from all game rooms.<br><br><strong>Reason:</strong> <span style="color: var(--text-primary); font-weight: 600;">${reasonText}</span><br><br><strong>Time Remaining:</strong> <span style="color: #f59e0b; font-size: 1.15rem; font-weight: 700;">${durationText}</span><br><br>To keep matches fair and respectful for all players, room access is temporarily restricted during a timeout period.<br><br>Please wait until your timeout expires before joining another match!`;
                        if (window.showAlertModal) {
                            window.showAlertModal('Account Timed Out', msg, true);
                        } else {
                            alert(errMsg);
                        }
                    } else if (errMsg.toLowerCase().includes('full')) {
                        if (window.showAlertModal) {
                            window.showAlertModal('Room Full', 'This room is now full.<br><br>Please press the Refresh button to update the list of Open Rooms and Closed Rooms.');
                        } else {
                            alert('This room is now full. Please press the Refresh button to update the list of Open Rooms and Closed Rooms.');
                        }
                        if (currentLobbyConfig) {
                            fetchAndRenderRooms(currentLobbyConfig.gameType, currentLobbyConfig.timeLimit, currentLobbyConfig.boardDimensions, false);
                        }
                    } else if (errMsg.toLowerCase().includes('not found')) {
                        if (window.showAlertModal) {
                            window.showAlertModal('Room Closed', 'This room has ended or is no longer active.');
                        } else {
                            alert('This room has ended or is no longer active.');
                        }
                        if (window.currentLobbyConfig && typeof window.fetchAndRenderRooms === 'function') {
                            window.fetchAndRenderRooms(window.currentLobbyConfig.gameType, window.currentLobbyConfig.timeLimit, window.currentLobbyConfig.boardDimensions);
                        }
                    } else {
                        if (window.showAlertModal) {
                            window.showAlertModal('Unable to Join', errMsg);
                        } else {
                            alert(errMsg);
                        }
                    }
                    return;
                }

                if (data.success) {
                    if (!isSpectator && data.role === 'spectator') {
                        if (typeof window.showPage === 'function') window.showPage('page-lobby');
                        if (window.showAlertModal) {
                            window.showAlertModal('Room Full', 'This room is now full.<br><br>Please press the Refresh button to update the list of Open Rooms and Closed Rooms.');
                        } else {
                            alert('This room is now full. Please press the Refresh button to update the list of Open Rooms and Closed Rooms.');
                        }
                        if (currentLobbyConfig) {
                            fetchAndRenderRooms(currentLobbyConfig.gameType, currentLobbyConfig.timeLimit, currentLobbyConfig.boardDimensions, false);
                        }
                        return;
                    }
                    window.currentRoomId = roomId;
                    localStorage.setItem('last_joined_room', roomId);
                    if (data.role === 'spectator') {
                        window.isSpectatorMode = true;
                    } else {
                        window.isSpectatorMode = false;
                    }

                    if (data.state && typeof window.updateGameState === 'function') {
                        window.updateGameState(data.state);
                    }

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
                    if (typeof window.showPage === 'function') window.showPage('page-lobby');
                    const errMsg = data.error || 'Unknown error';
                    if (errMsg.toLowerCase().includes('full')) {
                        if (window.showAlertModal) {
                            window.showAlertModal('Room Full', 'This room is now full.<br><br>Please press the Refresh button to update the list of Open Rooms and Closed Rooms.');
                        } else {
                            alert('This room is now full. Please press the Refresh button to update the list of Open Rooms and Closed Rooms.');
                        }
                        if (currentLobbyConfig) {
                            fetchAndRenderRooms(currentLobbyConfig.gameType, currentLobbyConfig.timeLimit, currentLobbyConfig.boardDimensions, false);
                        }
                    } else if (errMsg.toLowerCase().includes('not found')) {
                        if (window.showAlertModal) {
                            window.showAlertModal('Room Closed', 'This room has ended or is no longer active.');
                        } else {
                            alert('This room has ended or is no longer active.');
                        }
                        if (window.currentLobbyConfig && typeof window.fetchAndRenderRooms === 'function') {
                            window.fetchAndRenderRooms(window.currentLobbyConfig.gameType, window.currentLobbyConfig.timeLimit, window.currentLobbyConfig.boardDimensions);
                        }
                    } else {
                        if (window.showAlertModal) {
                            window.showAlertModal('Unable to Join', 'Failed to join room: ' + errMsg);
                        } else {
                            alert('Failed to join room: ' + errMsg);
                        }
                    }
                }
            } catch (err) {
                window._isEnteringRoom = false;
                if (typeof window.showPage === 'function') window.showPage('page-lobby');
                console.error('Join error:', err);
                if (window.showAlertModal) {
                    window.showAlertModal('Error', 'Failed to join room: ' + err.message);
                } else {
                    alert('Failed to join room: ' + err.message);
                }
            } finally {
                joinBtn.style.opacity = '1';
                joinBtn.style.pointerEvents = 'auto';
                if (window.hideLoadingOverlay) window.hideLoadingOverlay();
            }    }
            return; // Handled
        }
    });

    console.log('Lobby event delegation setup complete');

    // Setup Open Rooms vs Closed Rooms tab selection
    window.currentRoomFilterTab = window.currentRoomFilterTab || 'open';
    window.setRoomFilterTab = function(tab) {
        window.currentRoomFilterTab = (tab === 'closed') ? 'closed' : 'open';
        const openBtn = document.getElementById('open-rooms-filter-btn');
        const closedBtn = document.getElementById('closed-rooms-filter-btn');
        if (openBtn) {
            if (window.currentRoomFilterTab === 'open') openBtn.classList.add('active');
            else openBtn.classList.remove('active');
        }
        if (closedBtn) {
            if (window.currentRoomFilterTab === 'closed') closedBtn.classList.add('active');
            else closedBtn.classList.remove('active');
        }
        console.log('[Lobby] Room filter tab switched to:', window.currentRoomFilterTab);
        if (currentLobbyConfig) {
            fetchAndRenderRooms(
                currentLobbyConfig.gameType,
                currentLobbyConfig.timeLimit,
                currentLobbyConfig.boardDimensions,
                false
            );
        }
    };

    const openRoomsBtn = document.getElementById('open-rooms-filter-btn');
    if (openRoomsBtn) {
        openRoomsBtn.addEventListener('click', (e) => {
            e.preventDefault();
            window.setRoomFilterTab('open');
        });
    }

    const closedRoomsBtn = document.getElementById('closed-rooms-filter-btn');
    if (closedRoomsBtn) {
        closedRoomsBtn.addEventListener('click', (e) => {
            e.preventDefault();
            window.setRoomFilterTab('closed');
        });
    }

    // Setup Rating Filter Handlers (Enter key and input)
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

    const myRatingBtn = document.getElementById('my-rating-btn');
    if (myRatingBtn) {
        myRatingBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const activeConfig = currentLobbyConfig || window.currentLobbyConfig;
            if (!activeConfig) {
                console.log('[Lobby] My Rating clicked, but no active config. Doing nothing.');
                return;
            }
            const userRating = getUserConfigRating(
                activeConfig.gameType,
                activeConfig.boardDimensions,
                activeConfig.timeLimit
            );
            const input = document.getElementById('rating-filter');
            if (input) {
                input.value = userRating;
            }
            window.activeRatingFilterValue = userRating;
            console.log('[Lobby] My Rating clicked. Value:', userRating);
            fetchAndRenderRooms(
                activeConfig.gameType,
                activeConfig.timeLimit,
                activeConfig.boardDimensions,
                false
            );
        });
    }

    // Set up MutationObserver here, inside setupLobbyEvents, where DOM is guaranteed ready.
    // (The top-level getElementById runs before DOM and returns null, so the observer
    //  must be set up here instead.)
    const lobbyEl = document.getElementById('page-lobby');
    if (lobbyEl) {
        const lobbyObserver = new MutationObserver(() => {
            if (isOnLobby()) {
                // ALWAYS do an immediate full stats update on every lobby entry.
                // This is the primary mechanism for showing correct counts on entry.
                fetchLobbyStats('all');

                if (window.resetLobbyButtons) window.resetLobbyButtons();

                if (currentLobbyConfig) {
                    if (typeof updateMyRatingButton === 'function') {
                        updateMyRatingButton(currentLobbyConfig.gameType, currentLobbyConfig.boardDimensions, currentLobbyConfig.timeLimit);
                    }
                }

                // Start accumulative auto-poll interval if not already running
                if (!lobbyStatsInterval) startStatsPolling();

                // Mobile: snap to main panel
                const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
                if (isMobile) {
                    const scrollToMain = () => {
                        const mainPanel = document.getElementById('mobile-panel-main');
                        if (mainPanel) {
                            const lobbyGrid = mainPanel.closest('.lobby-grid') || mainPanel.parentElement;
                            if (lobbyGrid) lobbyGrid.scrollLeft = mainPanel.offsetLeft;
                        }
                    };
                    scrollToMain();
                    requestAnimationFrame(scrollToMain);
                    setTimeout(scrollToMain, 50);
                    setTimeout(scrollToMain, 150);
                }
            } else {
                stopLobbyPolling();
                stopStatsPolling();
            }
        });
        lobbyObserver.observe(lobbyEl, { attributes: true, attributeFilter: ['class'] });
    }

    // Start stats polling if we land on lobby at page-load time
    if (isOnLobby()) {
        startStatsPolling();

        // Mobile layout: Snap to center main lobby panel on load
        const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
        if (isMobile) {
            const scrollToMain = () => {
                const mainPanel = document.getElementById('mobile-panel-main');
                if (mainPanel) {
                    const lobbyGrid = mainPanel.closest('.lobby-grid') || mainPanel.parentElement;
                    if (lobbyGrid) {
                        lobbyGrid.scrollLeft = mainPanel.offsetLeft;
                    }
                }
            };
            scrollToMain();
            requestAnimationFrame(scrollToMain);
            setTimeout(scrollToMain, 50);
            setTimeout(scrollToMain, 150);
        }
    } // end if (isOnLobby())
} // end setupLobbyEvents

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupLobbyEvents);
} else {
    setupLobbyEvents();
}

// Helper function to fetch and render rooms
async function fetchAndRenderRooms(gameType, timeLimit, boardDimensions, allowAutoCreate = false, minRating = 0, maxRating = 9999, force = false) {
    window.fetchAndRenderRooms = fetchAndRenderRooms;
    currentLobbyConfig = { gameType, timeLimit: parseInt(timeLimit) || 45, boardDimensions: boardDimensions || '4x4' };
    window.currentLobbyConfig = currentLobbyConfig;
    const roomsList = document.getElementById('rooms-list');

    // Ensure persistent structure for inputs so they aren't wiped on poll
    let roomsContainer = document.getElementById('dynamic-rooms-container');
    const isLoadingText = roomsContainer && roomsContainer.innerHTML.includes('Loading active rooms');

    // Safety check if we navigated away (bypass when force=true or when container is currently showing loading indicator)
    if (!force && !isLoadingText && !isOnLobby()) {
        console.log('fetchAndRenderRooms called but not on lobby, ignoring');
        return;
    }

    if (!roomsContainer && roomsList) {
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
                <p class="placeholder" style="padding: 16px; text-align: center; color: rgba(255,255,255,0.7); font-size: 0.95rem;">Loading active rooms...</p>
            </div>
        `;
        roomsList.innerHTML = createButtonHtml;
        roomsContainer = document.getElementById('dynamic-rooms-container');
    }

    try {
        // Fetch active rooms for this configuration with cache busting and 4-second timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 4000);
        const url = `/api/rooms?game_type=${gameType}&board_dimensions=${boardDimensions}&time_limit=${timeLimit}&_t=${Date.now()}`;
        const response = await fetch(url, { cache: 'no-store', signal: controller.signal });
        clearTimeout(timeoutId);
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        const data = await response.json();

        let rooms = (data && data.rooms) ? data.rooms : [];

        // Recalculate average ratings
        rooms.forEach(room => {
            let totalRating = 0;
            let pCount = 0;
            if (room.players && Array.isArray(room.players)) {
                pCount = room.players.length;
                totalRating = room.players.reduce((sum, p) => {
                    const uname = (p && p.username) ? String(p.username) : '';
                    const isGuest = uname.startsWith('Guest_');
                    const rating = isGuest ? 0 : (p && p.rating ? p.rating : 0);
                    return sum + rating;
                }, 0);
            }
            const avgRating = pCount > 0 ? Math.round(totalRating / pCount) : 0;
            room.display_average_rating = avgRating;
        });

        // Client-side safety filter: never display rooms with no players, regardless of what the server returned.
        // This guards against any race conditions between room creation and the lobby poll cycle.
        rooms = rooms.filter(room => room.players && room.players.length > 0);

        // Directly update the Show Rooms [N] button for this config from the rooms data we already have.
        // This is more reliable than a separate stats polling interval.
        const totalPlayers = rooms.reduce((sum, r) => sum + (r.players ? r.players.length : 0), 0);
        document.querySelectorAll(
            `.game-btn[data-game="${gameType}"][data-board="${boardDimensions}"][data-time="${timeLimit}"]`
        ).forEach(btn => {
            const rawText = btn.textContent;
            const normalizedText = rawText.replace(/\s+/g, ' ').trim();
            const newNormalized = normalizedText.replace(/\[\d+\]/, `[${totalPlayers}]`);
            if (normalizedText !== newNormalized) btn.textContent = newNormalized;
        });

        // Ensure we target the fresh active container element right before DOM mutation
        roomsContainer = document.getElementById('dynamic-rooms-container') || document.getElementById('rooms-list');
        const scrollParent = document.getElementById('rooms-list');
        const savedListScroll = scrollParent ? scrollParent.scrollTop : 0;
        const savedContainerScroll = roomsContainer ? roomsContainer.scrollTop : 0;

        // Classify each room as Open or Closed for the current user
        const currentUser = window.currentUser || '';
        const currentUserId = window.currentUserId ? String(window.currentUserId) : '';
        const savedRoomId = localStorage.getItem('last_joined_room') || '';

        rooms.forEach(room => {
            const rId = String(room.room_id || '');
            const pCountVal = Number(room.player_count) || (room.players ? room.players.length : 0);
            const isFull = pCountVal >= (Number(room.max_players) || 8);
            const roomMin = Number(room.min_rating) || 0;
            const roomMax = Number(room.max_rating) || 9999;
            const hasLimits = roomMin > 0 || roomMax < 9999;
            const userRating = getUserConfigRating(room.game_type, room.board_dimensions, room.time_limit);
            const isCurrentUserGuest = !currentUser || currentUser.startsWith('Guest_') || Boolean(window.currentUserIsGuest);
            const isRatingOutOfRange = hasLimits && (userRating < roomMin || userRating > roomMax || isCurrentUserGuest);

            // Check if current user is an active player inside this room
            const isUserInRoom = (room.players && Array.isArray(room.players) && room.players.some(p => {
                const uname = (p && p.username) ? String(p.username).trim().toLowerCase() : '';
                const myUname = (currentUser || '').trim().toLowerCase();
                const uid = (p && p.user_id) ? String(p.user_id) : '';
                return (uname && myUname && uname === myUname) || (uid && currentUserId && uid === currentUserId);
            })) || (window.currentRoomId && rId === String(window.currentRoomId)) || (savedRoomId && rId === String(savedRoomId));

            room._isUserInRoom = isUserInRoom;

            if (isUserInRoom) {
                // Instantly synchronize current room state in memory and storage
                window.currentRoomId = rId;
                localStorage.setItem('last_joined_room', rId);
                room._isOpen = true; // User's active room is ALWAYS accessible and open to them
            } else {
                // Open Room: within rating limits AND less than 8 players
                room._isOpen = !isRatingOutOfRange && !isFull;
            }
        });

        // Filter based on active tab ('open' vs 'closed') - user's own room is always kept
        const activeTab = window.currentRoomFilterTab || 'open';
        let filteredRooms = rooms.filter(room => {
            if (room._isUserInRoom) return true;
            return activeTab === 'open' ? room._isOpen : !room._isOpen;
        });

        // Sort: Place active room with current user AT THE TOP, followed by rating filter/proximity
        filteredRooms.sort((a, b) => {
            if (a._isUserInRoom && !b._isUserInRoom) return -1;
            if (!a._isUserInRoom && b._isUserInRoom) return 1;

            const targetRating = window.activeRatingFilterValue;
            if (targetRating !== null && !isNaN(targetRating)) {
                const diffA = Math.abs(a.display_average_rating - targetRating);
                const diffB = Math.abs(b.display_average_rating - targetRating);
                return diffA - diffB;
            }
            return 0;
        });

        if (filteredRooms.length === 0) {
            if (roomsContainer) {
                if (activeTab === 'open' && rooms.length > 0) {
                    roomsContainer.innerHTML = `<p class="placeholder" style="padding: 16px; text-align: center; color: rgba(255,255,255,0.7); font-size: 0.95rem;">No open rooms currently match your rating.<br><span style="color: #38bdf8; cursor: pointer; text-decoration: underline; font-weight: 700; display: inline-block; margin-top: 8px;" onclick="window.setRoomFilterTab('closed')">View ${rooms.length} Closed / Rated Room(s)</span></p>`;
                } else {
                    const emptyMsg = (activeTab === 'open')
                        ? 'No open rooms currently available for your rating. Click <strong>+ Create Room</strong> above to start one!'
                        : 'No closed or full rooms currently active in this configuration.';
                    roomsContainer.innerHTML = `<p class="placeholder" style="padding: 16px; text-align: center; color: rgba(255,255,255,0.7); font-size: 0.95rem;">${emptyMsg}</p>`;
                }
            }
        } else {
            const html = filteredRooms.map(room => {
                let playersHtml = (room.players || []).map(p => {
                    // Override rating for Guest users
                    const uname = (p && p.username) ? String(p.username) : 'Player';
                    const isGuest = uname.startsWith('Guest_');
                    const displayRating = isGuest ? 0 : (p && p.rating !== undefined ? p.rating : 0);
                    const ratingColor = window.getRatingColor ? window.getRatingColor(displayRating) : '#fff';

                    return `<span class="room-player-pill" title="Rating: ${displayRating}" style="display:inline-flex; align-items:center;">
                        <span onclick="window.showMiniProfile('${uname}')" style="background-color: ${ratingColor}; width: 11px; height: 11px; border-radius: 3px; margin-right: 5px; display:inline-block; cursor: pointer;"></span>    
                        ${uname} (${displayRating})
                    </span>`;
                }).join('');

                if (!playersHtml) {
                    playersHtml = '<span style="color: rgba(255,255,255,0.5); font-size: 0.85rem; font-style: italic;">No active players currently in room — Click Join to start!</span>';
                }

                const roomMin = Number(room.min_rating) || 0;
                const roomMax = Number(room.max_rating) || 9999;
                const hasLimits = roomMin > 0 || roomMax < 9999;
                const rId = String(room.room_id || '');
                const rState = String(room.state || 'active');

                let actionButtons = '';

                if (room._isUserInRoom || (rId && (rId === window.currentRoomId || rId === savedRoomId))) {
                    actionButtons = `<button class="join-room-btn return-mode" data-room="${rId}" onclick="handleJoinRoomInline(this)" style="background: #e67e22; color: #ffffff; font-weight: 700;">Return to Game</button>`;
                } else if (activeTab === 'closed' || !room._isOpen) {
                    // Closed Rooms: ONLY allow Spectate button to be visible
                    actionButtons = `<button class="join-room-btn watch-mode" data-room="${rId}" data-spectator="true" onclick="handleJoinRoomInline(this)" style="background: #34495e;">Spectate</button>`;
                } else {
                    // Open Rooms: Spectate & Join
                    actionButtons += `<button class="join-room-btn watch-mode" data-room="${rId}" data-spectator="true" onclick="handleJoinRoomInline(this)" style="background: #34495e; margin-right: 5px;">Spectate</button>`;

                    let ratingText = '';
                    if (hasLimits) {
                        ratingText = `(${roomMin}-${roomMax < 9999 ? roomMax : '∞'})`;
                    }
                    actionButtons += `<button class="join-room-btn" data-room="${rId}" data-min-rating="${roomMin}" onclick="handleJoinRoomInline(this)">
                        Join ${ratingText}
                    </button>`;
                }

                // Display limitations
                const ratingRangeText = `${roomMin} - ${roomMax < 9999 ? roomMax : '∞'}`;

                return `
                <div class="room-item">
                    <div class="room-header-row">
                        <div class="room-status ${rState}">${rState.toUpperCase()}</div>
                        <div class="room-meta">
                            ${hasLimits ? `<span class="rating-req-badge">Req: ${ratingRangeText}</span>` : ''}
                            <span class="room-avg-rating">Avg Rating: ${room.display_average_rating || 0}</span> 
                        </div>
                    </div>
                    <div class="room-players-row">
                        ${playersHtml}
                    </div>
                    ${actionButtons}
                </div>
                `;
            }).join('');

            if (roomsContainer) {
                roomsContainer.innerHTML = html;
            }
        }

        // Restore scroll positions after render
        if (scrollParent && savedListScroll > 0) {
            try { scrollParent.scrollTop = savedListScroll; } catch(e) {}
        }
        if (roomsContainer && savedContainerScroll > 0) {
            try { roomsContainer.scrollTop = savedContainerScroll; } catch(e) {}
        }

        if (typeof updateMyRatingButton === 'function') {
            updateMyRatingButton(gameType, boardDimensions, timeLimit);
        }

    } catch (error) {
        console.error('Error fetching rooms:', error);
        if (roomsContainer) {
            roomsContainer.innerHTML = '<p class="placeholder" style="padding: 16px; text-align: center; color: rgba(255,255,255,0.7); font-size: 0.95rem;">No active rooms currently open. Click <strong>+ Create Room</strong> above to start one!</p>';
        }
    }
}

function startLobbyPolling() {
    // No auto-polling for FCFS/SP room lists.
    // Active rooms update only on user actions: clicking 'Show Rooms', pressing 🔄 Refresh, or creating a room.
}

function stopLobbyPolling() {
    if (lobbyPollInterval) {
        clearInterval(lobbyPollInterval);
        lobbyPollInterval = null;
    }
}

// Stats Polling
function startStatsPolling() {
    stopStatsPolling();

    // Immediately do a FULL update (all buttons)
    fetchLobbyStats('all');

    // Auto-poll every 1 second — continuously updates Accumulative buttons
    lobbyStatsInterval = setInterval(() => {
        if (isOnLobby()) {
            fetchLobbyStats('accumulative_only');
        }
    }, 1000);
}
window.startStatsPolling = startStatsPolling;

// Immediately trigger full stats fetch & continuous polling
startStatsPolling();
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        startStatsPolling();
    });
}

function stopStatsPolling() {
    if (lobbyStatsInterval) {
        clearInterval(lobbyStatsInterval);
        lobbyStatsInterval = null;
    }
}
window.stopStatsPolling = stopStatsPolling;

window.syncLobbyTimeoutState = async function() {
    try {
        const toResp = await fetch(`/api/user/my_timeout_status?_t=${Date.now()}`, { cache: 'no-store' });
        const toData = await toResp.json();
        const isTimedOut = !!(toData && toData.timed_out);
        window._userTimeoutInfo = isTimedOut ? toData : null;
        window._userIsTimedOut = isTimedOut;
        
        const lobbyButtons = document.querySelectorAll('.game-btn, .confirm-create-room-btn, .join-room-btn, .nav-btn[data-page="play"]');
        lobbyButtons.forEach(btn => {
            if (isTimedOut) {
                btn.classList.add('timeout-locked');
                btn.title = "Account Timed Out: Click for details";
            } else {
                btn.classList.remove('timeout-locked');
                if (btn.title === "Account Timed Out: Click for details") btn.title = "";
            }
        });
        return isTimedOut;
    } catch(e) {
        return false;
    }
};

// mode: 'all' (initial/entry) | 'accumulative_only' (auto-poll) | 'fcfs_sp_only' (Refresh click)
async function fetchLobbyStats(mode = 'all') {
    try {
        if (typeof window.syncLobbyTimeoutState === 'function') {
            window.syncLobbyTimeoutState().catch(() => {});
        }
        const response = await fetch(`/api/lobby-stats?_t=${Date.now()}`, { cache: 'no-store' });
        const data = await response.json();
        if (data.stats) {
            updateLobbyButtons(data.stats, mode);
        }
    } catch (error) {
        console.error('Error fetching lobby stats:', error);
    }
}
window.fetchLobbyStats = fetchLobbyStats;

// Lobby Refresh button handler
async function handleLobbyRefresh(btn) {
    const refreshBtn = btn || document.getElementById('lobby-refresh-btn');
    if (refreshBtn) {
        refreshBtn.classList.add('refreshing');
        refreshBtn.disabled = true;
    }

    try {
        let activeConfig = window.currentLobbyConfig || currentLobbyConfig;
        if (!activeConfig) {
            const activeBtn = document.querySelector('.game-btn.active') || document.querySelector('.fcfs-btn.active, .split-btn.active');
            if (activeBtn && activeBtn.dataset) {
                activeConfig = {
                    gameType: activeBtn.dataset.game || activeBtn.dataset.gameType || 'fcfs',
                    timeLimit: parseInt(activeBtn.dataset.time || activeBtn.dataset.timeLimit || '45'),
                    boardDimensions: activeBtn.dataset.board || activeBtn.dataset.boardDimensions || '4x4'
                };
                window.currentLobbyConfig = activeConfig;
                currentLobbyConfig = activeConfig;
            }
        }

        const statsPromise = fetchLobbyStats('all');
        let roomsPromise = null;
        if (activeConfig && typeof window.fetchAndRenderRooms === 'function') {
            const { gameType, timeLimit, boardDimensions } = activeConfig;
            roomsPromise = window.fetchAndRenderRooms(gameType, timeLimit, boardDimensions, false, 0, 9999, true);
        }

        await Promise.all([statsPromise, roomsPromise].filter(Boolean));
    } catch (err) {
        console.error('Error in handleLobbyRefresh:', err);
    } finally {
        if (refreshBtn) {
            setTimeout(() => {
                refreshBtn.classList.remove('refreshing');
                refreshBtn.disabled = false;
            }, 200);
        }
    }
}
window.handleLobbyRefresh = handleLobbyRefresh;
window.handleLobbyRefresh = handleLobbyRefresh;

function updateLobbyButtons(stats, mode = 'all') {
    // Stats format: "game_type|board|time": count
    // mode = 'all'               → update all buttons (FCFS, SP, Accumulative)
    // mode = 'accumulative_only' → only update Accumulative buttons (auto 1s poll)
    // mode = 'fcfs_sp_only'      → only update FCFS and SP buttons (Refresh button click)
    const buttons = document.querySelectorAll('.game-btn, .acc-btn, .fcfs-btn, .split-btn');
    buttons.forEach(btn => {
        const game = (btn.dataset.game || '').toLowerCase();
        const board = (btn.dataset.board || '').toLowerCase();
        const time = btn.dataset.time;
        if (!game || !board || !time) return;

        // Skip filtering based on mode
        if (mode === 'accumulative_only' && game !== 'accumulative') return;
        if (mode === 'fcfs_sp_only' && game === 'accumulative') return;

        const key = `${game}|${board}|${time}`;
        const count = (stats && stats[key] !== undefined) ? stats[key] : 0;

        const rawText = btn.textContent || '';
        const baseText = rawText.replace(/\s*\[\d+\]\s*$/, '').trim();
        const targetText = `${baseText} [${count}]`;

        if (rawText.trim() !== targetText) {
            btn.textContent = targetText;
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

function isOnLobby() {
    if (window.currentPageId === 'page-lobby' || window.currentPageId === 'lobby') return true;
    const el = document.getElementById('page-lobby');
    if (!el) return false;
    if (el.classList.contains('active')) return true;
    if (el.style.display && el.style.display !== 'none') return true;
    return window.location.hash === '#page-lobby' || window.location.hash === '#lobby';
}
window.isOnLobby = isOnLobby;

function getUserConfigRating(gameType, board, time) {
    const cleanType = String(gameType || '').replace('solo_', '');
    const cleanBoard = String(board || '4x4');
    const cleanTime = parseInt(time) || 45;
    const configKey = `${cleanType}|${cleanBoard}|${cleanTime}`;
    const ratings = window.currentUserConfigRatings || {};
    const val = ratings[configKey];
    if (val !== undefined && val !== null) {
        if (typeof val === 'object' && val.rating !== undefined && val.rating !== null) {
            return Number(val.rating) || 1200;
        }
        if (typeof val === 'number' && val > 0) {
            return val;
        }
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

    if (!window.currentLobbyConfig) {
        btn.textContent = 'My Rating';
        btn.removeAttribute('data-rating');
        btn.style.display = (window.currentUser && !window.currentUserIsGuest) ? 'inline-block' : 'none';
        return;
    }

    const initialRating = getUserConfigRating(gameType, board, time);
    btn.textContent = `My Rating (${initialRating})`;
    btn.dataset.rating = initialRating;
    btn.style.display = (window.currentUser && !window.currentUserIsGuest) ? 'inline-block' : 'none';

    if (window.currentUser && !window.currentUserIsGuest) {
        if (!window.currentUserConfigRatings || Object.keys(window.currentUserConfigRatings).length === 0) {
            if (typeof window.loadCurrentUserConfigRatings === 'function') {
                try {
                    await window.loadCurrentUserConfigRatings();
                    if (window.currentLobbyConfig) {
                        const updatedRating = getUserConfigRating(gameType, board, time);
                        btn.textContent = `My Rating (${updatedRating})`;
                        btn.dataset.rating = updatedRating;
                    }
                } catch (e) {
                    console.error('[Lobby] Error loading config ratings:', e);
                }
            }
        }
    }
}
window.updateMyRatingButton = updateMyRatingButton;
window.fetchAndRenderRooms = fetchAndRenderRooms;

console.log('lobby.js fully loaded - version with polling');
