let isTournamentPlay = false;
let isPrivateMatchPlay = false;
let privateMatchWords = [];
let privateMatchScore = 0;
let privateMatchParams = null;
let tournamentWords = [];
let pollInterval = null;
let timerInterval = null;
let lastServerUpdate = Date.now();  // Track last server response for freeze detection
let selectedPlayerUsername = null; // Track selected player for filtering/highlighting
let cachedTimerValueEl = null;    // Cache for high-frequency updates
let cachedBoardPanelEl = null;
let lastPlayersHtml = null;       // Cache for renderPlayers
const playerRatingCache = new Map(); // Cache for Chat Colors
/* --- Distributed Board Generation (DBG) --- */
const DICE_CONFIG_4x4 = [
    "AAEEGN", "ABBJOO", "ACHOPS", "AFFKPS",
    "AOOTTW", "CIMOTU", "DEILRX", "DELRVY",
    "DISTTY", "EEGHNW", "EEINSU", "EHRTVW",
    "EIOSST", "ELRTTY", "HIMNQU", "HLNNRZ"
];
let lastProbeTime = 0;
const PROBE_INTERVAL = 4000; // 4 seconds between probes
let tournamentScore = 0;
let tournamentStartTime = 0;
let localEndTime = 0;
let stableServerTimeOffset = null; // Persistent offset to prevent jitter
let timerFormatIs24h = false;     // Cached format to prevent flashing between HH:MM:SS and M:SS

// Mouse selection state
let mouseState = {
    isDown: false,
    selectedPath: [],       // Array of {row, col, letter}
    visitedCells: new Set() // Set of "row,col" strings
};

// Split Points UI State
let splitNotepadState = {}; // { username: 'unique' | 'split' | 'invalid' }
let showBoardInSplitIntermission = false;

// Board Rotation State
let isBoardRotated = false;
let activeWordsTab = 'found'; // 'found' or 'remaining'
let validationTimeout = null;
let highlightedSplitWord = null; // Track word for shared highlighting in Split Points
let highlightedFoundWord = null; // Track word from All Words list to highlight finders
let lastRenderedBoardJSON = null;
let lastRenderedGrayed = null;
let lastRenderedRotation = null;
let lastRenderedDensityJSON = null;
let hasPlayedIntermissionBell = false; // Flag for next round notification
let findFriendsMode = false;
let userFriendsCache = [];
let lastSpinnerDataJSON = null; // Detect if parameters have actually changed

// Input Method Tracking
let currentInputMethod = 'mouse';
function updateInputMethod(method) {
    // Only apply/track input method changes DURING an active round
    if (window.lastGameState && window.lastGameState.state !== 'active') return;

    if (currentInputMethod === method) return;
    currentInputMethod = method;
    const roomId = getCurrentRoomId();
    if (roomId) {
        fetch(`/room/${roomId}/update_input_method`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_method: method })
        }).catch(err => console.error('Failed to update input method:', err));
    }
}

// --- EVICTION / KICK LOGIC ---
let lastActivityTime = Date.now();
function resetIdleTimer() {
    lastActivityTime = Date.now();
}

// Global activity listeners
window.addEventListener('mousemove', resetIdleTimer, { passive: true });
window.addEventListener('mousedown', resetIdleTimer, { passive: true });
window.addEventListener('keydown', resetIdleTimer, { passive: true });
window.addEventListener('touchstart', resetIdleTimer, { passive: true });
window.addEventListener('scroll', resetIdleTimer, { passive: true });

async function ejectToLobby(reason = "inactivity") {
    console.warn(`[play.js] EVICTING USER. Reason: ${reason}`);
    
    // 1. Notify server immediately so lobby counts decrease
    if (window.leaveCurrentRoom) {
        try {
            // Fire-and-forget: Do NOT await this so the client UI redirects instantaneously
            // and is never blocked by a slow/throttled network request
            window.leaveCurrentRoom();
        } catch (e) {
            console.error('[play.js] Failed to notify server of leave during ejection:', e);
        }
    }

    // 2. Stop poll and clear state
    stopPolling();
    window.currentRoomId = null;
    localStorage.removeItem('last_joined_room');

    // 3. Clear ANY other overlays that might block the explanation
    document.querySelectorAll('.modal-overlay, .board-overlay, .results-overlay').forEach(ov => {
        ov.classList.add('hidden');
        ov.style.display = 'none';
    });

    // 4. Redirect to Lobby
    if (window.navigateToPage) window.navigateToPage('lobby');
    else if (window.showPage) window.showPage('page-lobby');
    else window.location.href = '#page-lobby';

    // 6. DELAYED SHOW: Stagger the modal so it appears AFTER the page switch
    setTimeout(() => {
        let title = "Session Expired";
        let message = `
            You have been returned to the lobby due to 10 minutes of inactivity. 
            <br><br>
            To keep room slots open for active players, matches are automatically cleared after prolonged idle periods. 
            Feel free to join a new match when you're ready!
        `;
        
        if (reason === "mobile-cube-restriction") {
            title = "Unsupported Device";
            message = `
                3D Cube mode is not supported on mobile devices. 
                <br><br>
                Please play on a desktop or laptop to enjoy Cube rooms! 
                Feel free to join any other 2D match on your current device.
            `;
        }
        
        if (reason === "daily_reset") {
            title = "Daily Reset";
            message = `
                The 24-hour Daily Room has reset for the new day!
                <br><br>
                A fresh daily board has been generated. Head back in to start finding words and climb the leaderboard!
            `;
        }
        
        if (window.showAlertModal) {
            window.showAlertModal(title, message, true); // priority=true ensures it isn't overwritten by lobby notices
            console.log('[play.js] Displayed modal via showAlertModal.');
        } else {
            // Fallback for extreme cases
            const modal = document.getElementById('generic-info-modal');
            const titleEl = document.getElementById('generic-modal-title');
            const bodyEl = document.getElementById('generic-modal-body');
            if (modal && titleEl && bodyEl) {
                titleEl.textContent = title;
                bodyEl.innerHTML = `<p style="padding: 30px; text-align: center; color: var(--text-primary);">${message}</p>`;
                modal.classList.remove('hidden');
                modal.style.display = 'flex';
                modal.style.zIndex = '100001';
            }
        }
    }, 800); // Increased delay to 800ms for more stability across all devices
}

// Check for idle logout every 5 seconds
setInterval(() => {
    // Optimization: Skip idle check if tab is hidden to save battery
    if (document.hidden) return;

    const roomId = getCurrentRoomId();
    if (roomId) {
        // EXEMPTION: No idle limit for 24h rooms
        const is24h = window.lastGameState && window.lastGameState.game_type === 'accumulative' && window.lastGameState.time_limit >= 7200;
        if (is24h) return;

        const idleTime = Date.now() - lastActivityTime;
        if (idleTime > 10 * 60 * 1000) { // 10 minutes
            console.warn('[play.js] 10m Idle reached. EVICTING.');
            ejectToLobby("inactivity");
        }
    }
}, 5000);

// Add global listeners for input detection
let lastTouchTime = 0;

document.addEventListener('keydown', () => {
    updateInputMethod('keyboard');
    resetIdleTimer();
}, true);

document.addEventListener('mousedown', () => {
    // If a touch event was fired in the last 1500ms, ignore this mousedown as it is a simulated mobile event
    if (Date.now() - lastTouchTime < 1500) {
        return;
    }
    updateInputMethod('mouse');
    resetIdleTimer();
}, true);

document.addEventListener('touchstart', () => {
    lastTouchTime = Date.now();
    updateInputMethod('touch');
    resetIdleTimer();
}, true);

function getCurrentRoomId() {
    let rid = window.currentRoomId;
    if (!rid) {
        // Fallback 1: URL path /room/xyz
        const m = window.location.pathname.match(/\/room\/([^\/]+)/);
        if (m) rid = m[1];
    }
    if (!rid) {
        // Fallback 2: localStorage
        rid = localStorage.getItem('last_joined_room');
    }
    return rid || null;
}

// Expose for lobby.js to call
window.startGamePolling = function () {
    console.log('[play.js] window.startGamePolling called');
    
    const isTournamentActive = !!localStorage.getItem('tournament_play_active');
    const isPrivateActive = !!localStorage.getItem('private_match_active');
    
    if (isTournamentActive) {
        if (!isTournamentPlay || !window.lastGameState) {
            initTournamentPlay();
        } else {
            console.log('[play.js] Returning to active tournament round. Preserving board and state.');
            setTimeout(checkBoardOverflow, 50);
        }
        return;
    }
    
    if (isPrivateActive) {
        if (!isPrivateMatchPlay || !window.lastGameState) {
            initPrivateMatchPlay();
        } else {
            console.log('[play.js] Returning to active private match. Preserving board and state.');
            setTimeout(checkBoardOverflow, 50);
        }
        return;
    }
    
    isTournamentPlay = false;
    isPrivateMatchPlay = false;
    startPolling();
};

let joinAttemptCount = 0; // Track initial join attempts to avoid race-condition kickouts

window.stopGamePolling = function () {
    stopPolling();
};

function clearGameUIAndCache() {
    console.log('[play.js] Clearing Game UI and Caches from previous match');
    
    // 1. Reset Global state caches
    window.lastGameState = null;
    window.lastRenderedStateJSON = null;
    window.lastRenderedBoardJSON = null;
    window.lastPlayersHtml = null;
    window._wasEverInRoster = false;
    
    // Reset all round-specific and mode-specific word/score lists
    privateMatchWords = [];
    privateMatchScore = 0;
    tournamentWords = [];
    tournamentScore = 0;
    selectedPlayerUsername = null;
    
    // 2. Reset DOM elements to clean placeholders
    const wordsList = document.getElementById('submitted-words-list');
    if (wordsList) {
        wordsList.innerHTML = '<p class="placeholder">Game active - Waiting for words...</p>';
    }
    
    const wordsStats = document.getElementById('words-stats');
    if (wordsStats) {
        wordsStats.textContent = '';
    }
    
    const wordInput = document.getElementById('word-input');
    if (wordInput) {
        wordInput.value = '';
        wordInput.disabled = false;
        wordInput.style.backgroundColor = '';
    }
    
    const defContent = document.getElementById('definition-content');
    if (defContent) {
        defContent.innerHTML = '<p class="placeholder">Select a word to see its definition.</p>';
    }
    
    const boardEl = document.getElementById('game-board');
    if (boardEl) {
        boardEl.innerHTML = `
            <div class="board-loader-container">
                <div class="board-loader-spinner"></div>
                <div class="board-loader-text">CONNECTING TO MATRIX...</div>
            </div>
        `;
    }
    
    const chatBox = document.getElementById('chat-messages');
    if (chatBox) {
        chatBox.innerHTML = '';
    }
}
window.clearGameUIAndCache = clearGameUIAndCache;

function startPolling() {
    console.log('[play.js] Starting Game Polling - Resetting Join Counter');
    joinAttemptCount = 0; // Reset counter for new room entry
    resetIdleTimer();
    
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    // ONLY clear UI and cache if we are actually switching rooms or if we don't have a lastGameState!
    const activeRoomId = getCurrentRoomId();
    const isSameRoom = window.lastGameState && (window.lastGameState.room_id === activeRoomId);

    if (!isSameRoom) {
        clearGameUIAndCache();
    } else {
        console.log('[play.js] Returning to same active room. Preserving board and state.');
        if (window.lastGameState) {
            updateGameState(window.lastGameState);
        }
    }
    
    isPrivateMatchPlay = false;
    isTournamentPlay = false;
    
    // Initial fetch to ensure we have state immediately
    updateGameState();
    
    // Setup pulse for subsequent polls (Dynamic based on visibility)
    refreshPollInterval();
}

function refreshPollInterval() {
    if (pollInterval) clearInterval(pollInterval);
    
    let delay = document.hidden ? 15000 : 1000;
    
    // User Request: Automatic/instant transition. 
    // Speed up polling significantly when intermission is about to end (last 2s)
    if (window.lastGameState && window.lastGameState.state === 'intermission' && !document.hidden) {
        const tr = window.lastGameState.time_remaining;
        if (tr < 2.5) {
             delay = 500; // Poll twice as fast at the very end
        }
    }

    pollInterval = setInterval(updateGameState, delay);
}

// Global Visibility Listener to handle battery management
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        // Tab became visible: Update immediately and restore fast polling
        console.log('[play.js] Tab visible: Restoring fast polling.');
        updateGameState();
        refreshPollInterval();
        
        // Re-sync timer if needed
        if (window.lastGameState) {
            syncTimerWithServer(window.lastGameState);
        }
    } else {
        // Tab hidden: Enter battery-saving mode
        console.log('[play.js] Tab hidden: Entering battery-saving mode.');
        refreshPollInterval();
        
        // Pause the high-frequency timer interval while hidden
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }
});

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

async function updateGameState(incomingState = null) {
    const roomId = getCurrentRoomId();
    if (!roomId) {
        return;
    }

    try {
        let state;
        if (incomingState) {
            state = incomingState;
        } else {
            // Optimization: Skip fetching if tab has been hidden for a while but not yet reached the 15s pulse
            // This is just extra safety, the refreshPollInterval handles the bulk of it.
            const response = await fetch(`/api/room/${roomId}/state`, { cache: 'no-store' });
            if (!response.ok) {
                if (response.status === 404 || response.status === 403 || response.status === 401) {
                    let errorMsg = "";
                    try {
                        const errData = await response.json();
                        errorMsg = errData.error || "";
                    } catch(e) {}

                    const isDescriptiveInactivity = errorMsg.toLowerCase().includes('inactivity') || 
                                                    errorMsg.toLowerCase().includes('removed') || 
                                                    errorMsg.toLowerCase().includes('idle') ||
                                                    errorMsg.toLowerCase().includes('expired');

                    if (window._wasEverInRoster || isDescriptiveInactivity) {
                        ejectToLobby("inactivity");
                    } else {
                        // RACE CONDITION PROTECTION: If we just joined, wait a few polls for server to sync
                        if (!window.lastGameState && joinAttemptCount < 5) {
                            joinAttemptCount++;
                            console.log(`[play.js] Poll failed (403/404) during join sequence. Attempt ${joinAttemptCount}/5. Silently retrying...`);
                            return; 
                        }
                        
                        // Generic failure (only if we are established or exhausted retries)
                        stopPolling();
                        window.currentRoomId = null;
                        if (window.showPage) window.showPage('page-lobby');
                    }
                }
                return;
            }
            state = await response.json();
        }

        if (!state) return;

        // Mobile Board Transposition: Turn landscape flat boards (rows < cols) into portrait (longest side runs vertically)
        try {
            if (window.innerWidth <= 900 && state.board && state.board.length > 0 && Array.isArray(state.board[0])) {
                const isBoard3D = state.board_dimensions === '3x3x3' || state.board.length === 6;
                if (!isBoard3D) {
                    const rows = state.board.length;
                    const cols = state.board[0].length;
                    if (rows < cols) {
                        // Transpose Board letters array safely
                        const transposedBoard = [];
                        for (let c = 0; c < cols; c++) {
                            transposedBoard[c] = [];
                            for (let r = 0; r < rows; r++) {
                                transposedBoard[c][r] = (state.board[r] && state.board[r][c] !== undefined) ? state.board[r][c] : '';
                            }
                        }
                        state.board = transposedBoard;

                        // Transpose Cell Density grid array safely
                        if (state.cell_density && state.cell_density.length > 0 && Array.isArray(state.cell_density[0])) {
                            const transposedDensity = [];
                            for (let c = 0; c < cols; c++) {
                                transposedDensity[c] = [];
                                for (let r = 0; r < rows; r++) {
                                    if (state.cell_density[r] && state.cell_density[r][c] !== undefined) {
                                        transposedDensity[c][r] = state.cell_density[r][c];
                                    } else {
                                        transposedDensity[c][r] = 0;
                                    }
                                }
                            }
                            state.cell_density = transposedDensity;
                        }
                    }
                }
            }
        } catch (transpositionError) {
            console.error("[Mobile] Transposition failed safely:", transpositionError);
        }

        // Mobile Device Restriction: Cube is not allowed on mobile!
        const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const is3D = state.board_dimensions === '3x3x3' || (state.board && state.board.length === 6 && Array.isArray(state.board[0]) && Array.isArray(state.board[0][0]));
        if (isMobile && is3D) {
            console.log('[Mobile] Cube rooms are not permitted on mobile devices. Kicking player to lobby.');
            ejectToLobby("mobile-cube-restriction");
            return;
        }

        // --- BATTERY OPTIMIZATION: STATE CHANGE CHECK ---
        // If the state hasn't changed (excluding jittery fields like server_time/time_remaining), skip rendering
        const stateToCompare = { ...state };
        delete stateToCompare.server_time;
        delete stateToCompare.time_remaining;
        
        const stateJSON = JSON.stringify(stateToCompare);
        if (window.lastRenderedStateJSON === stateJSON && !incomingState) {
            // Optimization: Update server heartbeat timestamp even if state is identical
            lastServerUpdate = Date.now();
            
            // If the tab is hidden, we definitely don't need to do anything else
            if (document.hidden) return;
            
            // If visible, we still need to sync the timer reference, but we can skip heavy DOM re-renders
            syncTimerWithServer(state);
            return;
        }
        window.lastRenderedStateJSON = stateJSON;
        
        // --- Distributed Board Generation (DBG) ---
        // If the server is searching, we help by probing random boards.
        // Lead-Time Optimization: Use next_spinner_params if they exist (Proactive search)
        const canProbe = state.board_search_started && !state.solving_complete && !isTournamentPlay;
        if (canProbe && (state.spinner_params_revealed || state.next_spinner_params)) {
            runBoardProbe();
        }

        // Capture previous state for transition logic (e.g. daily reset kick)
        const previousState = window.lastGameState;
        
        // RESILIENCE FIX: If heartbeat response is missing density/counts (common during round start
        // or server-side lag), preserve the previous values to prevent UI flickering/0-point states.
        if (previousState) {
            if (!state.cell_density && previousState.cell_density) {
                state.cell_density = previousState.cell_density;
                state.max_cell_density = previousState.max_cell_density;
            }
            if (state.total_points_count === 0 && previousState.total_points_count > 0) {
                state.total_points_count = previousState.total_points_count;
                state.total_words_count = previousState.total_words_count;
            }
        }
        
        window.lastGameState = state;  // Store for optimistic updates
        
        // Ensure polling interval is fresh (switches to high-frequency if near intermission end)
        if (state.state === 'intermission') {
            refreshPollInterval();
        }

        // Detect transition to intermission (round end)
        if (previousState && previousState.state === 'active' && state.state === 'intermission') {
            const wordInput = document.getElementById('word-input');
            const chatInput = document.getElementById('chat-input');
            
            // User Request: Prevent ALL users from spillover chatting for 2s
            if (chatInput) {
                chatInput.disabled = true;
                const originalPlaceholder = chatInput.placeholder;
                chatInput.placeholder = "Chat disabled for 2s...";
                
                // Clear word input immediately so they see it's over
                if (wordInput) {
                    wordInput.value = '';
                    wordInput.blur();
                }

                // Reset mouse selection state if it was active
                if (typeof mouseState !== 'undefined') {
                    mouseState.isDown = false;
                    mouseState.selectedPath = [];
                    if (mouseState.visitedCells) mouseState.visitedCells.clear();
                }
                
                if (window.chatFocusTimeout) clearTimeout(window.chatFocusTimeout);
                window.chatFocusTimeout = setTimeout(() => {
                    chatInput.disabled = false;
                    chatInput.placeholder = originalPlaceholder || "Type message...";
                    // Ensure we are still in intermission before stealing focus from potentially new round input
                    if (window.lastGameState && window.lastGameState.state === 'intermission') {
                        chatInput.focus();
                    }
                }, 2000);
            }


            // AUTO-SCROLL TO BOTTOM ON INTERMISSION START (Only if currently on play page)
            const playPage = document.getElementById('page-play');
            // FIX: Use classList.contains('active') because visibility is controlled by class in style.css
            const isShowingPlay = playPage && playPage.classList.contains('active') && window.location.hash === '#page-play';
            
            if (isShowingPlay) {
                requestAnimationFrame(() => {
                    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                });
            }

            // FORCE TAB TO 'WORDS' ON INTERMISSION START (Only if not already viewing something else)
            if (activeWordsTab !== 'history' && activeWordsTab !== 'remaining') {
                activeWordsTab = 'found';
            }
            window.userViewingDefinitionIntermission = false;
            
            // CLEAR STALE DEFINITIONS (Ensure winner announcement triggers)
            const defContent = document.getElementById('definition-content');
            if (defContent) {
                defContent.innerHTML = '<p class="placeholder">Select a word to see its definition</p>';
            }
            const defHeader = document.getElementById('definition-header');
            if (defHeader) defHeader.style.display = 'none';

            console.log('[play.js] Transition to Intermission: Forcing Words tab and resetting view state.');
        }

        // WINNER ANNOUNCEMENT LOGIC (Persistent during intermission)
        const defContent = document.getElementById('definition-content');
        const defPanel = document.querySelector('.definitions-panel');
        const defHeader = document.getElementById('definition-header');
        const isViewingDefinition = window.userViewingDefinitionIntermission === true;
        let hasActualWinner = false;

        if (state.state === 'intermission' && state.winners_history) {
            const latest = state.winners_history[0];
            // MANDATE: Only show if the record is for the round that JUST finished AND someone scored
            const isForCurrentRound = latest && latest.round === state.current_round;
            hasActualWinner = isForCurrentRound && (latest.score || 0) > 0;

            const bonusText = state.previous_bonus_word || state.bonus_word;
            const bonusHtml = bonusText ? `<div style="font-size: 0.85rem; color: #fff; opacity: 0.8; margin: 4px 0;">Bonus Word: <span style="color: #ffd700; font-weight: 800; letter-spacing: 1px;">${bonusText.toUpperCase()}</span></div>` : '';

            if (hasActualWinner && defContent && !isViewingDefinition) {
                // If not already showing this round's winner
                const winnerTextIdentifier = `WINNER_R${latest.round}`;
                if (!defContent.innerHTML.includes(winnerTextIdentifier) || defContent.innerHTML.includes('placeholder')) {
                    const winnersList = latest.winners.map(w => w.username).join(' & ');
                    
                    if (defPanel) {
                        defPanel.classList.add('winner-flash');
                        defPanel.classList.remove('timer-flash');
                    }
                    if (defHeader) defHeader.style.display = 'none';
                    
                    const me = (state.your_username || window.currentUser || document.getElementById('current-username')?.innerText || '').toLowerCase().trim();
                    const amIPlayerInRoom = state.players.some(p => p.username.toLowerCase().trim() === me);
                    const playerCount = (state.players && Array.isArray(state.players)) ? state.players.length : 0;
                    const maxPlayers = state.max_players || 8;
                    const canJoinInThisRoom = (playerCount < maxPlayers) || state.game_type === 'accumulative';

                    const joinButtonHtml = (!amIPlayerInRoom && canJoinInThisRoom) 
                        ? `<button id="winner-spec-join-btn" class="spectator-join-btn premium-btn" style="margin-top: 10px; width: auto; font-size: 0.9rem; padding: 10px 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 2px solid #ffd700; z-index: 10; flex-shrink: 0;">Join Match</button>` 
                        : '';

                    defContent.innerHTML = `
                        <div id="${winnerTextIdentifier}" style="display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 100%; text-align: center; padding: 5px; box-sizing: border-box; background: rgba(255, 215, 0, 0.05);">
                            <div style="font-size: 0.75rem; color: #ffd700; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 850; margin-bottom: 2px; animation: textPulse 1.5s infinite; flex-shrink: 0;">🏆 Round Complete 🏆</div>
                            <h2 style="font-size: 1.2rem; color: #fff; text-shadow: 0 0 10px rgba(255,215,0,0.5); font-weight: 950; margin: 3px 0; line-height: 1; flex-shrink: 0;">CONGRATULATIONS</h2>
                            <div style="font-size: 1.15rem; color: #ffd700; font-weight: 800; text-shadow: 0 0 8px rgba(0,0,0,0.7); max-width: 95%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex-shrink: 0;">${winnersList.toUpperCase()}</div>
                            <div style="font-size: 0.75rem; opacity: 0.95; margin-top: 3px; font-style: italic; flex-shrink: 0;">1st Place with ${latest.score || 0} pts</div>
                            ${bonusHtml}
                            ${joinButtonHtml}
                        </div>
                    `;

                    // Handle Join Button inside winner announcement
                    if (joinButtonHtml) {
                        setTimeout(() => {
                            const winJoinBtn = document.getElementById('winner-spec-join-btn');
                            if (winJoinBtn) {
                                winJoinBtn.onclick = async () => {
                                    winJoinBtn.textContent = 'Joining...';
                                    winJoinBtn.disabled = true;
                                    try {
                                        const roomId = window.currentRoomId;
                                        const resp = await fetch(`/api/room/${roomId}/join`, {
                                            method: 'POST',
                                            headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify({ as_spectator: false })
                                        });
                                        const data = await resp.json();
                                        if (data.success) {
                                            window.isSpectatorMode = false;
                                            setTimeout(updateGameState, 100);
                                        } else {
                                            alert(data.error);
                                            winJoinBtn.textContent = 'Join Match';
                                            winJoinBtn.disabled = false;
                                        }
                                    } catch (e) {
                                        console.error('Spec Join Error:', e);
                                        winJoinBtn.textContent = 'Error';
                                        winJoinBtn.disabled = false;
                                    }
                                };
                            }
                        }, 50);
                    }
                }
            } else if (!hasActualWinner && defContent && !isViewingDefinition && (defContent.innerHTML.includes('CONGRATULATIONS') || defContent.innerHTML.includes('placeholder'))) {
                // CLEAR the previous winner announcement if current round had no winner
                defContent.innerHTML = `
                    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 100%; text-align: center; opacity: 0.9;">
                        <div style="font-size: 0.8rem; color: var(--text-primary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">Round Ended</div>
                        <div style="font-size: 1.1rem; color: #fff; font-weight: 700; margin-bottom: 5px;">No Scoring Words Found</div>
                        ${bonusHtml}
                    </div>
                `;
                if (defPanel) defPanel.classList.remove('winner-flash');
                if (defHeader) defHeader.style.display = 'flex';
            } else if (isViewingDefinition) {
                // IF viewing a definition: Clean up celebratory effects and restore study header
                if (defPanel && defPanel.classList.contains('winner-flash')) {
                    defPanel.classList.remove('winner-flash');
                }
                if (defHeader && defHeader.style.display === 'none') {
                    defHeader.style.display = 'flex';
                }
            }
        }

        // update global room id if needed
        window.currentRoomId = state.room_id || roomId;

        const boardPanel = document.querySelector('.board-panel');
        const wordInputSection = document.querySelector('.word-input-section');
                const currentUsername = state.your_username || window.currentUser || localStorage.getItem('morpheme_username');
        window.currentUser = currentUsername ? currentUsername.trim() : null;

        // Cache for recovery after reset
        if (currentUsername) {
            localStorage.setItem('last_morpheme_user', currentUsername);
        }

        const amIPlayer = state.players.some(p => {
            const match = p.username.toLowerCase() === (currentUsername ? currentUsername.toLowerCase().trim() : '');
            return match;
        });

        const amISpectator = (state.spectators || []).some(p => {
            const match = p.username.toLowerCase() === (currentUsername ? currentUsername.toLowerCase().trim() : '');
            return match;
        });

        if (amIPlayer || amISpectator) {
            window._wasEverInRoster = true;
        }

        const is24H = (state.time_limit >= 7200);

        // COMBINED EVICTION / 24H RESET LOGIC
        if (!amIPlayer && !amISpectator && currentUsername) {
            const wasInBefore = previousState && (
                previousState.players.some(p => p.username.toLowerCase() === currentUsername.toLowerCase()) ||
                (previousState.spectators || []).some(s => s.username.toLowerCase() === currentUsername.toLowerCase())
            );

            // INSTANTANEOUS EVICTION: If we were previously established in the roster, kick immediately
            if (window._wasEverInRoster) {
                console.warn('[play.js] Authoritative eviction detected: User missing from roster. Ejecting instantaneously.');
                const roundChanged = previousState && state.current_round > previousState.current_round;

                if (is24H && wasInBefore && roundChanged) {
                     ejectToLobby("daily_reset");
                     return;
                }

                ejectToLobby("inactivity");
                return;
            }

            // SILENT AUTO-REJOIN HANDSHAKE: Before counting an eviction, try to rejoin ONLY if we haven't been successfully established in the room roster before
            if (!window._wasEverInRoster) {
                const roomId = window.currentRoomId || state.room_id;
                if (roomId && !window._isRejoiningRoom) {
                    window._isRejoiningRoom = true;
                    console.warn(`[play.js] Silent auto-rejoin triggered for room ${roomId}`);
                    fetch(`/api/room/${roomId}/join`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ as_spectator: !!window.isSpectatorMode })
                    })
                    .then(resp => resp.json())
                    .then(data => {
                        window._isRejoiningRoom = false;
                        if (data.success) {
                            console.log(`[play.js] Silent auto-rejoin successful! Role: ${data.role}`);
                            // Force a state update to immediately fetch updated roster
                            setTimeout(updateGameState, 100);
                        } else {
                            console.error(`[play.js] Silent auto-rejoin failed: ${data.error}`);
                        }
                    })
                    .catch(err => {
                        window._isRejoiningRoom = false;
                        console.error('[play.js] Silent auto-rejoin error:', err);
                    });
                }
            }

            window._emptyPlayersPollCount = (window._emptyPlayersPollCount || 0) + 1;
            console.warn(`[play.js] EVICTION WARNING: User not found in state.players or state.spectators! Count: ${window._emptyPlayersPollCount}/3`);
            
            if (window._emptyPlayersPollCount >= 3) {
                // 24H Reset Logic: Only auto-rejoin IF the round has actually changed (e.g. midnight flip)
                // If the round is the same, then this was likely an individual inactivity kick.
                const roundChanged = previousState && state.current_round > previousState.current_round;

                if (is24H && wasInBefore && roundChanged) {
                     ejectToLobby("daily_reset");
                     return;
                }

                // Normal Eviction
                ejectToLobby("inactivity");
                return;
            }
        } else {
            // Reset counter if player is found
            window._emptyPlayersPollCount = 0;
        }

        if (window.isSpectatorMode !== !amIPlayer) {
            console.log(`[play.js] Role Sync: amIPlayer=${amIPlayer}, currentUsername=${currentUsername}`);
            window.isSpectatorMode = !amIPlayer;
        }

        if (!amIPlayer) {
            // I am a spectator
            if (wordInputSection) wordInputSection.style.display = 'none';

            // Ensure spectator panel exists
            let spectatorPanel = document.getElementById('spectator-status-panel');
            if (!spectatorPanel) {
                spectatorPanel = createSpectatorPanel();
            }

            const defContent = document.getElementById('definition-content');
            
            // PRIORITY: During intermission, show the winner announcement for everyone (even spectators)
            if (defContent) {
                if (state.state === 'intermission') {
                    defContent.style.display = 'block';
                    if (spectatorPanel) spectatorPanel.style.display = 'none';
                } else {
                    defContent.style.display = 'none';
                    if (spectatorPanel) spectatorPanel.style.display = 'flex';
                }
            }
            
            // Re-render spectator content only if not in intermission (if in intermission, defContent wins)
            if (state.state !== 'intermission' && spectatorPanel) {
                spectatorPanel.style.display = 'flex';

                // Check if there is space to join
                const playerCount = (state.players && Array.isArray(state.players)) ? state.players.length : 0;
                const maxPlayers = state.max_players || 8;
                const isAccumulative = state.game_type === 'accumulative';
                const canJoin = isAccumulative || (playerCount < maxPlayers);

            // Render Content
                spectatorPanel.innerHTML = `
                    <div class="spectator-title">SPECTATING</div>
                    <div class="spectator-actions">
                        ${canJoin ?
                        `<button id="spec-join-btn" class="spectator-join-btn premium-btn">Join Game</button>` :
                        `<div class="spectator-full-badge">Full Room</div>`
                    }
                    </div>
                `;

                // Re-attach event listener
                setTimeout(() => {
                    const joinBtn = document.getElementById('spec-join-btn');
                    if (joinBtn) {
                        joinBtn.onclick = async () => {
                            console.log('[SpecJoin] Join clicked');
                            joinBtn.textContent = 'Joining...';
                            joinBtn.disabled = true;

                            try {
                                const resp = await fetch(`/api/room/${window.currentRoomId}/join`, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ as_spectator: false })
                                });
                                const data = await resp.json();

                                if (data.success) {
                                    window.isSpectatorMode = false;
                                    const playBtn = document.getElementById('play-btn');
                                    if (playBtn) {
                                        playBtn.disabled = false;
                                        playBtn.title = "";
                                    }
                                    if (window.updateManualToolState) window.updateManualToolState();
                                    setTimeout(updateGameState, 100);
                                } else {
                                    alert(data.error);
                                    joinBtn.textContent = 'Join Game';
                                    joinBtn.disabled = false;
                                }
                            } catch (e) {
                                console.error('Spec Join Error:', e);
                                joinBtn.textContent = 'Error';
                                joinBtn.disabled = false;
                            }
                        };
                    }
                }, 50);
            }
        } else {
            // I am a player
            if (wordInputSection) wordInputSection.style.display = ''; // Restore flex/block
            const specPanel = document.getElementById('spectator-status-panel');
            if (specPanel) specPanel.style.display = 'none';
            const defContent = document.getElementById('definition-content');
            if (defContent) defContent.style.display = '';
        }

        if (state.error) {
            console.error('Room error:', state.error);
            stopPolling();
            return;
        }

        // Update parameters
        updateParameters(state);

        // [DYNAMIC RATING] Emphasize the user's current rating on the color bar
        if (state.players && typeof window.updateUserRatingHighlight === 'function') {
            const me = state.players.find(p => p.username.toLowerCase().trim() === (currentUsername ? currentUsername.toLowerCase().trim() : ''));
            if (me) {
                window.updateUserRatingHighlight(me.rating);
            }
        }

        // Sync local timer with server
        syncTimerWithServer(state);

        // Track last successful server update
        lastServerUpdate = Date.now();

        // Render board
        // Render board
        const isSplitIntermission = (state.game_type === 'split' && state.state === 'intermission');
        const isFCFSIntermission = (state.game_type === 'fcfs' && state.state === 'intermission');

        if (isSplitIntermission && !showBoardInSplitIntermission) {
            renderSplitNotepads(state.players, state);
        } else if (isFCFSIntermission && !showBoardInSplitIntermission) {
            renderFCFSNotepads(state.players, state);
        } else {
            // ONLY gray out if we are specifically in intermission
            const isIntermission = state.state === 'intermission';
            const is3D = state.game_type === '3d' || (state.board && state.board.length === 6 && Array.isArray(state.board[0]) && Array.isArray(state.board[0][0]));
            renderBoard(state.board, isIntermission, is3D, state);
        }

        // Update players (pass full state for context if needed)
        renderPlayers(state.players, currentUsername, state);

        // Update chat
        if (state.chat_messages) {
            renderChat(state.chat_messages);
        }

        // Enable/disable input and buttons
        const isActive = state.state === 'active';
        const inputEl = document.getElementById('word-input');
        const submitBtnEl = document.getElementById('submit-word-btn');
        const rotateBtn = document.getElementById('rotate-board-btn');

        if ((isSplitIntermission || isFCFSIntermission) && !showBoardInSplitIntermission) {
            // Hide controls when board is hidden (Split/FCFS intermission)
            if (inputEl) inputEl.style.display = 'none';
            if (submitBtnEl) submitBtnEl.style.display = 'none';
            if (rotateBtn) rotateBtn.style.display = 'none';
        } else {
            // Show otherwise
            if (inputEl) inputEl.style.display = '';
            if (submitBtnEl) submitBtnEl.style.display = '';
            if (rotateBtn) rotateBtn.style.display = '';

            if (inputEl && inputEl.disabled !== !isActive) {
                inputEl.disabled = !isActive;
            }
            if (submitBtnEl && submitBtnEl.disabled !== !isActive) {
                submitBtnEl.disabled = !isActive;
            }
        }

        const lastStateStr = previousState ? previousState.state : null;
        
        // --- POST-ROUND RESULTS RENDERING (Heartbeat Sync) ---
        if (state.state === 'intermission') {
            const currentRoundId = `${state.room_id}_${state.current_round}`;
            const wordsListEl = document.getElementById('submitted-words-list');
            
            // OPTIMIZATION: Only render the massive word list once per round transition or when solving finishes.
            // Rendering 2000+ words on every pulse (1s) can cause the browser main thread to hang.
            const isTransition = !window.lastRenderedIntermissionWords || window.lastRenderedIntermissionWords !== currentRoundId;
            const isEmpty = wordsListEl && (wordsListEl.innerHTML.includes('placeholder') || wordsListEl.children.length === 0);
            const newlySolved = state.solving_complete && !window.lastSolvingComplete;
            
            if (isTransition || isEmpty || newlySolved) {
                if (state.all_words) {
                    const myWords = state.players.find(p => p.username.toLowerCase().trim() === (state.your_username || currentUsername || '').toLowerCase().trim())?.submitted_words || [];
                    const allFound = state.players.reduce((acc, p) => acc.concat(p.submitted_words || []), []);
                    
                    displayAllWords(
                        state.all_words, 
                        state.bonus_word, 
                        myWords, 
                        allFound, 
                        state.all_word_scores || {}, 
                        state.csw_only_words || [], 
                        state.added_words || []
                    );
                    window.lastRenderedIntermissionWords = currentRoundId;
                    window.lastSolvingComplete = state.solving_complete;
                }
            }

        } else if (state.state !== 'intermission') {
            window.lastRenderedIntermissionWords = null;
            window.lastSolvingComplete = false;
        }

        // --- HEADER PARAMETER REVEAL ANIMATION (Gold Fade) ---
        // Triggered only when the NEW parameters for the upcoming round are revealed (at 45s remaining)
        if (state.state === 'intermission') {
            const hasSpinner = !!(state.spinner_params && state.spinner_params.word_count_range);
            const currentSpinnerJSON = hasSpinner ? JSON.stringify({
                rid: state.room_id,
                rnd: state.current_round,
                params: state.spinner_params
            }) : null;
            
            // FIX: Only trigger reveal transition if we have reached the 0:45 remaining threshold
            // This prevents the fade effect from happening at the very beginning of intermission.
            const isParamRevealTransition = (hasSpinner && state.time_remaining <= 45 && lastSpinnerDataJSON !== currentSpinnerJSON);

            if (isParamRevealTransition) {
                lastSpinnerDataJSON = currentSpinnerJSON;
                
                // Trigger the CSS animation
                const container = document.querySelector('.game-params');
                if (container) {
                    container.classList.remove('reveal-new');
                    void container.offsetWidth; // Force reflow to allow re-triggering the animation
                    container.classList.add('reveal-new');
                }

            }
        } else if (state.state !== 'intermission') {
            lastSpinnerDataJSON = null;
        }
        
        // Check for state transitions (Cleanup/Misc)
        const isNewRound = (state.state === 'active' && (lastStateStr !== 'active' || (previousState && state.current_round !== previousState.current_round)));
        if (lastStateStr !== state.state || isNewRound) {
            if (isNewRound) {
                // Clear any winner announcement from Definitions Panel
                const defContent = document.getElementById('definition-content');
                const defHeader = document.getElementById('definition-header');
                const defPanel = document.querySelector('.definitions-panel');
                if (defContent) defContent.innerHTML = '<p class="placeholder">Select a word to see its definition</p>';
                if (defHeader) defHeader.style.display = 'none';
                if (defPanel) {
                    defPanel.classList.remove('timer-flash');
                    defPanel.classList.remove('winner-flash');
                }
                window.userViewingDefinitionIntermission = false;

                const wordsList = document.getElementById('submitted-words-list');
                if (wordsList) {
                    wordsList.innerHTML = '<p class="placeholder">Game active - Waiting for words...</p>';
                }

                // Reset/clear player submission caches during transition to avoid frame leaks
                if (state.players && Array.isArray(state.players)) {
                    state.players.forEach(p => {
                        p.submitted_words = [];
                        p.score = 0;
                    });
                }
                state.fcfs_found_words = [];

                // Reset Highlighting
                highlightedSplitWord = null;
                highlightedFoundWord = null;
                selectedPlayerUsername = null; // Reset player selection

                // DATA SYNC FIX: Explicitly clear Remaining list to prevent crossover
                const remainingList = document.getElementById('remaining-words-list');
                if (remainingList) remainingList.innerHTML = '';

                // Reset Find Friends mode
                findFriendsMode = false;
                const findFriendsBtn = document.getElementById('find-friends-btn');
                if (findFriendsBtn) findFriendsBtn.classList.remove('active');
                const showEveryoneBtn = document.getElementById('show-everyone-btn');
                if (showEveryoneBtn) showEveryoneBtn.classList.remove('active');

                // Clear and Focus Word Input on Game Start
                if (window.chatFocusTimeout) {
                    clearTimeout(window.chatFocusTimeout);
                    window.chatFocusTimeout = null;
                }
                
                setTimeout(() => {
                    const wordInput = document.getElementById('word-input');
                    if (wordInput) {
                        wordInput.value = '';
                    }

                    const isMobile = window.innerWidth <= 900;
                    if (!isMobile) {
                        if (wordInput) wordInput.focus();
                    } else {
                        // Mobile Device: Do NOT auto-focus the textbox (prevents keyboard from popping up and blocking board).
                        // Instead, scroll the viewport instantly/smoothly so that the board and timer are fully visible.
                        setTimeout(() => {
                            const timerDisplay = document.querySelector('.timer-display');
                            if (timerDisplay) {
                                const rect = timerDisplay.getBoundingClientRect();
                                const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                                const targetY = Math.max(0, rect.top + scrollTop - 15);
                                // 1. Immediate scroll jump to prevent layout lag
                                window.scrollTo(0, targetY);
                                
                                // 2. Smooth correction scroll after browser layout fully settles
                                setTimeout(() => {
                                    const rectFresh = timerDisplay.getBoundingClientRect();
                                    const scrollTopFresh = window.pageYOffset || document.documentElement.scrollTop;
                                    const targetYFresh = Math.max(0, rectFresh.top + scrollTopFresh - 15);
                                    window.scrollTo({ top: targetYFresh, behavior: 'smooth' });
                                }, 100);
                            }
                        }, 50); // Small delay to let board rendering settle for accurate coordinates
                    }
                }, 150);
            }
            // NO assignment to previousState constant. window.lastGameState update at top handles tracking.
        }

        // Update words panel based on state
        const wordsPanelHeader = document.getElementById('words-panel-title');
        const wordsStats = document.getElementById('words-stats');
        const tabsContainer = document.getElementById('words-tabs-container');

        // Show tabs in ALL rooms
        if (tabsContainer) {
            tabsContainer.style.display = 'flex';
        }

        // words-panel-title
        if (wordsPanelHeader) {
            let headerText = 'Words';
            if (state.game_type === 'fcfs') headerText = 'Live Feed';

            const factRange = (state.spinner_params && state.spinner_params.word_count_range) || state.current_word_count_range || 'Random';
            if (activeWordsTab === 'remaining') headerText = `Remaining (${factRange})`;
            if (activeWordsTab === 'clues') headerText = `Clues (${factRange})`;
            if (activeWordsTab === 'previous') headerText = 'Previous Day';
            if (activeWordsTab === 'history') headerText = 'Past Winners';
            if (state.state === 'intermission' && activeWordsTab === 'found') headerText = 'All Words';
            wordsPanelHeader.textContent = headerText;
        }

        // Update Tab Text dynamically
        const foundTabBtn = document.querySelector('.word-tab[data-tab="found"]');
        if (foundTabBtn) {
            foundTabBtn.textContent = (state.game_type === 'fcfs') ? 'Live Feed' : 'Words';
        }

        let currentUser = null;
        try {
            currentUser = state.your_username || window.currentUser || localStorage.getItem('morpheme_username') || null;
            if (currentUser) currentUser = currentUser.trim();
        } catch (e) { console.warn('No currentUser', e); }



        // 1. Configure Tab Buttons Visibility & Labels
        // const is24H = ... (already declared above)
        const tabBtns = document.querySelectorAll('.word-tab');

        tabBtns.forEach(btn => {
            const tab = btn.dataset.tab;
            if (is24H) {
                // 24H: Found, Clues, Previous
                if (tab === 'found') {
                    btn.textContent = 'Found';
                    btn.style.display = 'block';
                } else if (tab === 'clues' || tab === 'previous') {
                    btn.style.display = 'block';
                } else {
                    btn.style.display = 'none'; // Hide Remaining, History
                }
            } else {
                // Standard/FCFS: Words, Remaining, History
                if (tab === 'found') {
                    btn.textContent = 'Words';
                    btn.style.display = 'block';
                } else if (tab === 'remaining' || tab === 'history') {
                    btn.style.display = 'block';
                } else {
                    btn.style.display = 'none'; // Hide Clues/Previous
                }
            }
        });

        // Ensure activeWordsTab is valid for current room type
        if (is24H && (activeWordsTab === 'remaining' || activeWordsTab === 'history')) {
            activeWordsTab = 'found';
        }
        if (!is24H && (activeWordsTab === 'clues' || activeWordsTab === 'previous')) {
            activeWordsTab = 'found';
        }

        // Apply Tab Visibility logic
        const tabContents = document.querySelectorAll('.tab-content');
        tabContents.forEach(content => {
            const tabId = content.id.replace('tab-content-', '');
            content.classList.toggle('active', activeWordsTab === tabId);
        });

        tabBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === activeWordsTab);
        });

        // Toggle Stats Visibility (Only in Found/Words tab)
        if (wordsStats) {
            wordsStats.style.display = activeWordsTab === 'found' ? 'block' : 'none';
        }

        // 2. Render Tab Contents

        // --- Shared Found Words strings (for counting and highlighting) ---
        let allPlayerFoundStrs = [];
        state.players.forEach(p => {
            if (p.submitted_words) {
                p.submitted_words.forEach(w => {
                    const str = typeof w === 'string' ? w : w.word;
                    allPlayerFoundStrs.push(str.toUpperCase());
                });
            }
        });

        const allWords = state.all_words || [];

        // --- SUBMITTED WORDS LIST (Standard/Found tab) ---
        const listEl = document.getElementById('submitted-words-list');
        if (listEl && activeWordsTab === 'found') {
            if (state.state === 'intermission') {
                // INTERMISSION: Show ALL words

                // 1. Calculate Global Stats (All players)
                const totalWords = state.initial_total_words || state.total_words_count || allWords.length;
                const globalUnique = new Set(allPlayerFoundStrs).size;
                const globalPercentage = totalWords > 0 ? Math.round((globalUnique / totalWords) * 100) : 0;

                // 2. Calculate Personal Stats (Current User)
                const myPlayer = state.players.find(p => p.username.toLowerCase() === (currentUser || "").toLowerCase());
                const myWords = myPlayer ? (myPlayer.submitted_words || []) : [];
                const personalUnique = new Set(myWords.map(w => (typeof w === 'string' ? w : w.word).toUpperCase())).size;
                const personalPercentage = totalWords > 0 ? Math.round((personalUnique / totalWords) * 100) : 0;

                // Display Both
                let finderButtonHtml = '';
                if (highlightedFoundWord) {
                    const finders = state.players.filter(p =>
                        p.submitted_words && p.submitted_words.some(sw =>
                            (typeof sw === 'object' ? sw.word : sw).toUpperCase() === highlightedFoundWord
                        )
                    );
                    if (finders.length > 0) {
                        finderButtonHtml = `
                            <button id="view-finders-btn" class="secondary" onclick="window.showFinderModal('${highlightedFoundWord}')" style="margin-top: 8px; font-size: 0.75rem; padding: 4px 10px; width: 100%; border-radius: 6px; border: 1px solid var(--input-border); background: var(--input-bg); color: var(--text-primary); cursor: pointer; transition: all 0.2s;">
                                View all finders [${finders.length}]
                            </button>
                        `;
                    }
                }

                let totalPointsValue = state.total_points_count || 0;
                // Cache from all_word_scores during intermission (populated server-side)
                if (totalPointsValue === 0 && state.all_word_scores) {
                    const computed = Object.values(state.all_word_scores).reduce((s, v) =>
                        s + (typeof v === 'number' ? v : (v && v.total) || 0), 0);
                    if (computed > 0) totalPointsValue = computed;
                }
                if (totalPointsValue === 0 && window.lastValidTotalPoints) {
                    totalPointsValue = window.lastValidTotalPoints;
                }
                if (totalPointsValue > 0) window.lastValidTotalPoints = totalPointsValue;

                wordsStats.innerHTML = `
                    <div style="line-height: 1.2;" class="stats-text-primary">
                        ${personalUnique}/${totalWords} - ${personalPercentage}% (${totalPointsValue} total pts)
                        <div class="stats-text-secondary" style="font-size: 0.75em; margin-top: 2px;">
                            Collective Percentage: ${globalPercentage}%
                        </div>
                        ${finderButtonHtml}
                    </div>`;

                const targetUsername = selectedPlayerUsername || currentUser;
                let targetWords = [];
                if (targetUsername) {
                    const targetPlayer = state.players.find(p => p.username === targetUsername);
                    if (targetPlayer && targetPlayer.submitted_words) {
                        targetWords = targetPlayer.submitted_words.map(w => typeof w === 'string' ? w : w.word);
                    }
                }

                const uniqueGlobalFound = [...new Set(allPlayerFoundStrs)];
                const bonusForList = state.state === 'intermission' ? (state.previous_bonus_word || state.bonus_word) : state.bonus_word;
                const cswForList = state.state === 'intermission' ? (state.previous_csw_only_words || state.csw_only_words) : state.csw_only_words;
                
                displayAllWords(allWords, bonusForList, targetWords, uniqueGlobalFound, state.all_word_scores, cswForList, state.added_words);
                if (state.game_type === 'split' || state.game_type === 'fcfs') addSplitViewBoardToggle();

            } else if (state.game_type !== 'fcfs') {
                // ACTIVE STATE (Not Intermission) & Not FCFS
                // Personal List for Standard, Split, AND Accumulative
                const myPlayer = state.players.find(p => (p.username || "").toLowerCase().trim() === (currentUser || "").toLowerCase().trim());
                const myWords = myPlayer ? (myPlayer.submitted_words || []) : [];


                // 2. Personal Stats Only (Active)
                const totalWords = state.initial_total_words || state.total_words_count || allWords.length;
                const uniqueFound = new Set(myWords.map(w => (typeof w === 'string' ? w : w.word).toUpperCase())).size;
                const percentage = totalWords > 0 ? Math.round((uniqueFound / totalWords) * 100) : 0;

                // Total points — use server value, fall back to last known good
                let totalPoints = state.total_points_count || 0;
                if (totalPoints === 0 && window.lastValidTotalPoints) totalPoints = window.lastValidTotalPoints;
                if (totalPoints === 0 && state.all_word_scores) {
                    totalPoints = Object.values(state.all_word_scores).reduce((s, v) =>
                        s + (typeof v === 'number' ? v : (v && v.total) || 0), 0);
                }
                if (totalPoints > 0) window.lastValidTotalPoints = totalPoints;

                const safeWordsStats = document.getElementById('words-stats') || wordsStats;
                if (safeWordsStats) {
                    safeWordsStats.textContent = `${uniqueFound}/${totalWords} - ${percentage}% (${totalPoints} total pts)`;
                    safeWordsStats.style.display = 'block';
                }

                const sortedWords = [...myWords].sort((a, b) => (b.time || 0) - (a.time || 0));
                if (sortedWords.length === 0) {
                    listEl.innerHTML = '<p class="placeholder">Find words on the board!</p>';
                } else {
                    listEl.innerHTML = sortedWords.map(w => {
                        const word = typeof w === 'string' ? w : w.word;
                        const wordUpper = word.toUpperCase();
                        const points = typeof w === 'string' ? (state.all_word_scores[wordUpper] || 0) : (w.points || 0);
                        const isBonus = state.bonus_word && wordUpper === state.bonus_word.toUpperCase();
                        const isCSWOnly = state.csw_only_words && state.csw_only_words.some(csw => csw.toUpperCase() === wordUpper);

                        const isAddedWord = state.added_words && state.added_words.some(aw => aw.toUpperCase() === wordUpper);

                        let className = 'word-item player-word';
                        if (isBonus) {
                            className += ' bonus-word';
                        } else if (isAddedWord) {
                            className += ' added-word';
                        } else if (isCSWOnly) {
                            className += ' csw-only';
                        }
                        if (points < 0) className += ' penalty-word';
                        if (highlightedFoundWord === wordUpper) className += ' finder-active';

                        // All words in this list ARE found by user
                        const indicator = '<span class="found-indicator present">✓</span>';

                        return `<div class="${className}" data-word="${word}" style="display:flex; justify-content:space-between; cursor:pointer;">
                            <span>${indicator}${word}</span>
                            <span style="opacity:0.8">${points}</span>
                        </div>`;
                    }).join('');

                    // Add click listeners
                    listEl.querySelectorAll('.word-item').forEach(item => {
                        item.onclick = () => {
                            const word = item.dataset.word.toUpperCase();
                            highlightedFoundWord = (highlightedFoundWord === word) ? null : word;
                            updateGameState();
                            window.fetchDefinition(item.dataset.word);
                        };
                    });
                }
            } else {
                // FCFS: Shared Live Feed
                // Use dedicated shared list from server (The Authoritative Source)
                let allFoundWords = [...(state.fcfs_found_words || [])];
                
                // If the server list is missing entirely (undefined) but we have player words, use those as fallback
                if (state.fcfs_found_words === undefined && state.players && Array.isArray(state.players)) {
                    state.players.forEach(p => {
                        const words = p.submitted_words || [];
                        words.forEach(w => {
                            const wordStr = (typeof w === 'string' ? w : w.word) || '';
                            const wordUpper = wordStr.toUpperCase();
                            if (!wordUpper) return;

                            const alreadyIn = allFoundWords.some(afw => (afw.word || '').toUpperCase() === wordUpper);
                            if (!alreadyIn) {
                                let wObj = (typeof w === 'string') ? { word: w, points: '?', time: 0 } : { ...w };
                                wObj.finder = p.username;
                                wObj.is_ai = p.is_ai || (p.username && p.username.toLowerCase().includes('bot'));
                                allFoundWords.push(wObj);
                            }
                        });
                    });
                }

                const totalWords = state.initial_total_words || state.total_words_count || allWords.length;
                const uniqueFound = new Set(allFoundWords.map(w => (typeof w === 'string' ? w : w.word).toUpperCase())).size;
                const percentage = totalWords > 0 ? Math.round((uniqueFound / totalWords) * 100) : 0;
                let totalPoints = state.total_points_count || 0;
                if (state.state === 'intermission' && totalPoints === 0 && window.lastValidTotalPoints) {
                    totalPoints = window.lastValidTotalPoints;
                }
                if (totalPoints > 0) window.lastValidTotalPoints = totalPoints;

                const safeWordsStats = document.getElementById('words-stats');
                if (safeWordsStats) safeWordsStats.textContent = `${uniqueFound}/${totalWords} - ${percentage}% (${totalPoints} total pts)`;

                const sortedWords = allFoundWords.sort((a, b) => (b.time || 0) - (a.time || 0));
                const threshold = 150;
                const isAtBottom = (listEl.scrollTop + listEl.clientHeight >= listEl.scrollHeight - threshold);
                const wasEmpty = listEl.innerHTML.includes('placeholder') || listEl.children.length === 0;

                if (sortedWords.length === 0) {
                    listEl.innerHTML = '<p class="placeholder">Find words!</p>';
                } else {
                    const html = sortedWords.map(wObj => {
                        const word = wObj.word;
                        const wordUpper = word.toUpperCase();
                        const points = wObj.points;
                        const finder = wObj.finder || (wObj.is_ai ? 'AI' : 'Someone');
                        const isMe = finder && currentUser && finder.toLowerCase().trim() === currentUser.toLowerCase().trim();
                        const indicator = isMe ? '<span class="found-indicator present">✓</span>' : (wObj.is_ai ? '<span>🤖</span>' : '<span>🔸</span>');

                        const isBonus = state.bonus_word && wordUpper === state.bonus_word.toUpperCase();
                        const isAddedWord = state.added_words && state.added_words.some(aw => aw.toUpperCase() === wordUpper);
                        const isCSWOnly = state.csw_only_words && state.csw_only_words.some(csw => csw.toUpperCase() === wordUpper);

                        let className = 'word-item';
                        if (isMe) className += ' player-word';
                        else className += ' opponent-word';

                        if (isBonus) className += ' bonus-word';
                        else if (isAddedWord) className += ' added-word';
                        else if (isCSWOnly) className += ' csw-only';
                        
                        if (highlightedFoundWord === wordUpper) className += ' finder-active';

                        return `<div class="${className}" data-word="${word}" style="display:flex; justify-content:space-between; align-items:center; cursor:pointer;">
                            <div>${indicator}<span style="font-weight:bold;">${word}</span><span style="font-size:0.8em; opacity:0.7; margin-left:6px;">(${finder})</span></div>
                            <span style="opacity:0.8">${points}</span>
                        </div>`;
                    }).join('');

                    if (listEl.innerHTML !== html) {
                        listEl.innerHTML = html;
                        // Add click listeners
                        listEl.querySelectorAll('.word-item').forEach(item => {
                            item.onclick = () => {
                                const word = item.dataset.word.toUpperCase();
                                highlightedFoundWord = (highlightedFoundWord === word) ? null : word;
                                updateGameState();
                                window.fetchDefinition(item.dataset.word);
                            };
                        });

                        // For FCFS Feed (Newest at top), we don't automatically scroll to bottom.
                        // If it was empty, ensure we are at top.
                        if (wasEmpty) {
                            requestAnimationFrame(() => { listEl.scrollTop = 0; });
                        }
                    }
                }
            }
        }

        // --- REMAINING TAB ---
        const remainingListEl = document.getElementById('remaining-words-list');
        if (remainingListEl && activeWordsTab === 'remaining') {
            let myFoundStrs = [];
            if (state.game_type === 'fcfs') {
                myFoundStrs = allPlayerFoundStrs;
            } else if (currentUser) {
                const me = state.players.find(p => p.username.toLowerCase() === (currentUser || "").toLowerCase());
                if (me && me.submitted_words) {
                    myFoundStrs = me.submitted_words.map(w => (typeof w === 'string' ? w : w.word).toUpperCase());
                }
            }

            // User Request Fix: Persistence - Keep showing the previous round's remaining words 
            // for the ENTIRE intermission. We only switch to the new round data when state is 'active'.
            const isIntermission = state.state === 'intermission';
            const showRealtimeCounts = !isIntermission && state.total_counts_by_len;
            
            let countsByLen = {};
            if (showRealtimeCounts) {
                // Real-time mode: calculate remaining from server totals minus found words
                const totalByLen = state.total_counts_by_len || {};
                
                // ROUND SYNC: If the server provided a round tag, verify it matches the current round.
                const expectedRound = state.current_round;
                
                if (totalByLen._round !== undefined && totalByLen._round !== expectedRound) {
                    console.warn(`[Remaining-Sync] Mismatch (Counts Round: ${totalByLen._round}, Expected: ${expectedRound}).`);
                    remainingListEl.innerHTML = '<p class="placeholder" style="opacity: 0.6; font-style: italic;">Syncing word counts...</p>';
                    return; 
                }
                
                const foundByLen = {};
                
                const isFCFS = state.game_type === 'fcfs';
                const sourceWords = isFCFS ? (state.fcfs_found_words || []) : myFoundStrs;
                
                // CRITICAL SYNC: Only subtract VALID words from the board totals.
                // Subtracting penalties or duplicates makes the Remaining tab inaccurate.
                const validFound = sourceWords.filter(w => {
                    if (typeof w === 'string') return true; // Standard word
                    return (w.points > 0 || (w.score_details && w.score_details.total > 0));
                });
                
                validFound.forEach(w => {
                    const wordStr = (typeof w === 'string' ? w : w.word);
                    const l = wordStr ? wordStr.length : 0;
                    if (l > 0) foundByLen[l] = (foundByLen[l] || 0) + 1;
                });
                
                for (let i = 1; i <= 30; i++) {
                    countsByLen[i] = Math.max(0, (totalByLen[i] || 0) - (foundByLen[i] || 0));
                }

                // HEADER SYNC is handled by updateGameStatsHeader to avoid fighting/duplicate labels
            } else {
                // Intermission/Clues fallback: calculate from explicit list
                const remainingWords = allWords.filter(w => !myFoundStrs.includes(w.toUpperCase()));
                for (let i = 3; i <= 30; i++) countsByLen[i] = 0;
                remainingWords.forEach(w => {
                    const len = w.length;
                    if (len >= 3 && len <= 30) countsByLen[len]++;
                });
            }

            let html = '<table id="remaining-words-table">';
            const minLen = state.current_min_length || 3;
            for (let i = minLen; i <= 30; i++) {
                // Show rows from minLen to 20
                html += `<tr><td class="len-cell">${i}LW</td><td class="count-cell">${countsByLen[i] || 0}</td></tr>`;
            }
            html += '</table>';
            remainingListEl.innerHTML = html;
        }

        // --- CLUES TAB (24H Only) ---
        const cluesListEl = document.getElementById('clues-list');
        if (cluesListEl && activeWordsTab === 'clues') {
            const myPlayer = state.players.find(p => p.username.toLowerCase() === (currentUser || "").toLowerCase());
            const myWords = myPlayer ? myPlayer.submitted_words : [];
            const foundSet = new Set(myWords.map(w => (typeof w === 'string' ? w : w.word).toUpperCase()));
            const unfoundWords = allWords.filter(w => !foundSet.has(w.toUpperCase()));
            console.log('[CluesDebug] allWords:', allWords.length, 'foundSet:', foundSet.size, 'unfound:', unfoundWords.length);

            if (unfoundWords.length === 0) {
                cluesListEl.innerHTML = '<p class="placeholder">All words found!</p>';
            } else {
                // Sort Clues Alpha for better searching
                unfoundWords.sort((a, b) => a.length - b.length || a.localeCompare(b));
                const clueListHtml = unfoundWords.map(w => {
                    const prefix = w.substring(0, 2);
                    let sum = 0;
                    for (let char of w.toUpperCase()) {
                        sum += (window.LETTER_VALUES || LETTER_VALUES)[char] || 1;
                    }
                    return `
                        <div class="clue-item">
                            <span class="clue-prefix">${prefix}..</span>
                            <div class="clue-divider"></div>
                            <span class="clue-stats">${w.length} Letters &bull; ${sum} pts</span>
                        </div>
                    `;
                }).join('');
                
                cluesListEl.innerHTML = '<div class="clues-grid" style="grid-template-columns: repeat(2, 1fr); gap: 10px; padding: 10px;">' + clueListHtml + '</div>';
            }
        }

        // --- PREVIOUS DAY TAB (24H Only) ---
        const prevListEl = document.getElementById('previous-words-list');
        if (prevListEl && activeWordsTab === 'previous') {
            let prevAll = [];
            if (state.previous_all_words) {
                if (Array.isArray(state.previous_all_words)) {
                    prevAll = state.previous_all_words;
                } else {
                    prevAll = Object.keys(state.previous_all_words);
                }
            }

            if (prevAll.length === 0) {
                prevListEl.innerHTML = '<p class="placeholder">No previous data.</p>';
            } else {
                // PERSONAL HISTORY: Use my restored player's previous words OR persisted history
                // Note: state.players might be empty if wiped by 24h reset!
                const myPlayer = (state.players || []).find(p => p.username.toLowerCase() === (currentUser || "").toLowerCase());
                let myPrevWords = myPlayer ? (myPlayer.previous_submitted_words || []) : [];

                // BACKUP: If player was wiped (24h daily reset), check history
                console.log('[PreviousTab] Checking history. MyPrev:', myPrevWords.length, 'Hist:', !!state.previous_day_history);
                if (myPrevWords.length === 0 && state.previous_day_history) {
                    const normalizedCurrent = currentUser ? currentUser.trim().toLowerCase() : '';
                    // retrieve locally saved username as fallback (for Guests who get reset)
                    const localUser = localStorage.getItem('last_morpheme_user');
                    const normalizedLocal = localUser ? localUser.trim().toLowerCase() : '';

                    console.log('[PreviousTab] Searching for:', normalizedCurrent, 'or Local:', normalizedLocal);

                    Object.values(state.previous_day_history).forEach(record => {
                        if (record.username) {
                            const recName = record.username.trim().toLowerCase();
                            // Match either current session name OR locally saved name
                            if ((normalizedCurrent && recName === normalizedCurrent) ||
                                (normalizedLocal && recName === normalizedLocal)) {
                                myPrevWords = record.found_words || [];
                                console.log('[PreviousTab] Restored from HISTORY (Match found for:', record.username, ') Words:', myPrevWords.length);
                            }
                        }
                    });
                }

                console.log('[PreviousTab] Rendering. PrevAll:', prevAll.length, 'MyPrev:', myPrevWords.length);

                const foundSet = new Set(myPrevWords.map(w => (typeof w === 'string' ? w : w.word).toUpperCase()));

                const foundList = [];
                const missedList = [];

                prevAll.forEach(w => {
                    if (foundSet.has(w.toUpperCase())) {
                        foundList.push(w);
                    } else {
                        missedList.push(w);
                    }
                });

                // Sort function: Length desc, then Alpha
                const sortFn = (a, b) => {
                    if (a.length !== b.length) return b.length - a.length;
                    return a.toUpperCase().localeCompare(b.toUpperCase());
                };

                foundList.sort(sortFn);
                missedList.sort(sortFn);

                const prevCSWOnly = (state.previous_csw_only_words || []).map(w => w.toUpperCase());

                // Render Helper
                const renderRow = (w, isFound) => {
                    const isCSWOnly = prevCSWOnly.includes(w.toUpperCase());
                    let statusClass = isFound ? 'player-word' : 'missed';
                    if (!isFound && isCSWOnly) statusClass += ' csw-only';
                    const icon = isFound ? '✓' : '✗';
                    const details = state.previous_all_word_scores && state.previous_all_word_scores[w];
                    let ptsDisplay = '';
                    if (details) {
                        const total = details.total || 0;
                        const base = details.base || 0;
                        const bonus = (details.bonus_word_points || 0) + (details.bonus_letter_points || 0) + (details.either_or_points || 0);
                        if (bonus > 0) {
                            ptsDisplay = `${base} + ${bonus} = ${total}`;
                        } else {
                            ptsDisplay = total;
                        }
                    }

                    return `<div class="word-item ${statusClass}" data-word="${w}" style="display:flex; justify-content:space-between; cursor:pointer;">
                        <span>${w}</span>
                        <span style="opacity:0.6; font-size:0.85em;">${ptsDisplay} ${icon}</span>
                    </div>`;
                };

                // Render Sections
                let html = '';

                // BOARD DISPLAY
                const boardHtml = state.previous_board ? `<div id="prev-board-container"></div>` : '';
                html += boardHtml;

                // Found Section
                html += `<div style="padding:10px; background:rgba(0,0,0,0.1); font-weight:bold; color:#4a90e2;">FOUND (${foundList.length})</div>`;
                if (foundList.length > 0) {
                    html += foundList.map(w => renderRow(w, true)).join('');
                } else {
                    html += `<div style="padding:15px; text-align:center; font-style:italic; opacity:0.6;">None</div>`;
                }

                // Missed Section
                html += `<div style="padding:10px; background:rgba(0,0,0,0.1); font-weight:bold; margin-top:10px; color:#888;">MISSED (${missedList.length})</div>`;
                if (missedList.length > 0) {
                    html += missedList.map(w => renderRow(w, false)).join('');
                } else {
                    html += `<div style="padding:15px; text-align:center; font-style:italic; opacity:0.6;">None</div>`;
                }

                prevListEl.innerHTML = html;

                // Render the board after HTML is injected
                if (state.previous_board) {
                    const boardCont = document.getElementById('prev-board-container');
                    renderPreviousBoard(state.previous_board, boardCont);
                }

                // Add click listeners for definitions
                prevListEl.querySelectorAll('.word-item').forEach(item => {
                    item.addEventListener('click', (e) => {
                        const word = item.dataset.word;
                        if (window.fetchDefinition) {
                            window.fetchDefinition(word);
                        }
                    });
                });
            }
        }

        // --- ROOM HISTORY TAB ---
        const historyListEl = document.getElementById('winners-list');
        if (historyListEl && activeWordsTab === 'history') {
            const history = state.winners_history || [];
            if (history.length === 0) {
                historyListEl.innerHTML = '<p class="placeholder" style="text-align:center; margin-top:20px;">No winners recorded yet.</p>';
            } else {
                historyListEl.innerHTML = history.map(h => {
                    const winnersHtml = h.winners.map(w => {
                        const name = typeof w === 'string' ? w : w.username;
                        const rating = typeof w === 'string' ? 0 : (w.rating || 0);
                        const rColor = window.getRatingColor ? window.getRatingColor(rating) : '#f1f1f1';

                        return `
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <div style="width: 12px; height: 12px; background: ${rColor}; border-radius: 2px; box-shadow: 0 0 5px ${rColor}44;"></div>
                                <span style="font-weight: 700; color: var(--text-primary); font-size: 0.95rem;">${name}</span>
                                <span style="font-size: 0.8rem; opacity: 0.5; font-weight: 600;">(${rating})</span>
                            </div>
                        `;
                    }).join('');

                    return `
                        <div class="history-item" style="padding: 12px 15px; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between; align-items: center; transition: background 0.2s;">
                            <div style="display: flex; flex-direction: column; gap: 4px;">
                                ${winnersHtml}
                                <span style="font-size: 0.7rem; opacity: 0.4; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">Round ${h.round}</span>
                            </div>
                            <div style="background: rgba(255,215, 0, 0.1); border: 1px solid rgba(255,215, 0, 0.2); padding: 5px 10px; border-radius: 8px; font-weight: 900; color: #ffd700; font-size: 1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.2); display: flex; align-items: center; gap: 10px;">
                                <span>${h.score}<span style="font-size: 0.65rem; opacity: 0.8; font-weight: 800; margin-left: 3px;">PTS</span></span>
                                <button title="Watch Replay" onclick="event.stopPropagation(); watchRoundHistory('${state.room_id}', ${h.round}, false)" style="background:none; border:none; color:#ffd700; cursor:pointer; font-size:1.3rem; padding:0; display:flex; align-items:center; margin-right: 5px;">▶</button>
                                <button title="View Snapshot" onclick="event.stopPropagation(); watchRoundHistory('${state.room_id}', ${h.round}, true)" style="background:none; border:none; color:#ffd700; cursor:pointer; font-size:1.1rem; padding:0; display:flex; align-items:center;">📷</button>
                            </div>
                        </div>
                    `;
                }).join('');
            }
        }


        // Auto-focus check
        if (isActive && !previousState) {
            const inputField = document.getElementById('word-input');
            if (inputField) {
                setTimeout(() => inputField.focus(), 50);
            }
        }

    } catch (error) {
        console.error('Error updating game state:', error);
    }
}

function renderPlayers(players, currentUser = null, state = null) {
    console.log('[RenderPlayers] Players:', players && players.length);
    const listEl = document.getElementById('players-list');
    const headingEl = document.getElementById('players-heading');
    const findMeBtn = document.getElementById('find-me-btn');
    const findFriendsBtn = document.getElementById('find-friends-btn');
    const showEveryoneBtn = document.getElementById('show-everyone-btn');

    if (state && state.game_type === 'accumulative') {
        const totalPeople = (players ? players.length : 0) + (state.spectators ? state.spectators.length : 0);
        if (headingEl) headingEl.textContent = `Players [${totalPeople}]`;
        if (findMeBtn) findMeBtn.style.display = 'block';
        if (findFriendsBtn) findFriendsBtn.style.display = 'block';
        if (showEveryoneBtn) showEveryoneBtn.style.display = 'block';
    } else {
        if (headingEl) headingEl.textContent = `Players`;
        if (findMeBtn) findMeBtn.style.display = 'none';
        if (findFriendsBtn) findFriendsBtn.style.display = 'none';
        if (showEveryoneBtn) showEveryoneBtn.style.display = 'none';
    }

    if (!players || players.length === 0) {
        listEl.innerHTML = '<p class="placeholder">No players</p>';
        return;
    }

    // Sort players by score (Highest First), break ties with Rating
    const sortedPlayers = [...players].sort((a, b) => (b.score - a.score) || (b.rating - a.rating));

    // Attach click handler via event delegation parent or recreate list
    // Recreating list is fine here

    // Top 3 + Nearby Logic (simplified for brevity: show top 20)
    // For now, render ALL players (scrollable if > 8)
    let itemsToRender = sortedPlayers.map((p, idx) => ({ ...p, originalRank: idx + 1 }));

    if (findFriendsMode && currentUser) {
        itemsToRender = itemsToRender.filter(p =>
            p.username.toLowerCase() === (currentUser || "").toLowerCase() ||
            userFriendsCache.some(f => f.username.toLowerCase() === p.username.toLowerCase())
        );
    }

    const html = itemsToRender.map((p) => {
        const index = p.originalRank - 1; // Use original rank for visuals if needed
        const rank = p.originalRank;

        // Override rating for Guest users
        const isGuest = p.username.startsWith('Guest_');
        const displayRating = isGuest ? 0 : p.rating;

        let changeTxt = '0';
        let changeClass = 'change-neutral';
        if (p.rating_change > 0) {
            changeTxt = `+${p.rating_change}`;
            changeClass = 'change-positive';
        } else if (p.rating_change < 0) {
            changeTxt = `${p.rating_change}`;
            changeClass = 'change-negative';
        }


        
        // Final display string (User request: in brackets next to rating)
        const ratingDisplayStr = `${displayRating} <span class="${changeClass}">(${changeTxt})</span>`;
        const ratingDisplay = (isGuest && displayRating === 0 && p.rating_change === 0) ? 'Guest' : ratingDisplayStr;

        const bonusClass = p.found_bonus_word ? ' bonus-finder' : '';
        const userClass = (p.username.toLowerCase() === (currentUser || "").toLowerCase()) ? ' current-user' : '';

        // Highlight if selected
        const selectedClass = (p.username === selectedPlayerUsername) ? ' selected-player' : '';

        // Highlight if found selected word (Golden Highlight)
        let finderClass = '';
        if (highlightedFoundWord) {
            const hasChosenWord = p.submitted_words && p.submitted_words.some(sw => {
                const w = (typeof sw === 'object' ? sw.word : sw) || '';
                return w.toUpperCase() === highlightedFoundWord;
            });
            if (hasChosenWord) {
                finderClass = ' finder-highlight';
            }
        }

        // Calculate rating color
        let ratingColor = window.getRatingColor ? window.getRatingColor(displayRating) : '#fff';
        if (!isGuest && p.games_played === 0) {
            ratingColor = '#0044ff';
        }

        // Input Method Icon
        let inputIcon = '🖱️';
        if (p.input_method === 'keyboard') inputIcon = '⌨️';
        if (p.input_method === 'touch') inputIcon = '📱';

        // Trophy Logic (Exceptional Performance)
        const peVal = p.performance_efficiency || 1.0;
        const trophyHtml = (p.has_exceptional_round && players.length > 1 && !p.is_ai) ? `<span title="Exceptional Performance (PE: ${peVal.toFixed(2)}x)" class="trophy-icon">🏆</span>` : '';

        return `
        <div class="player-item${bonusClass}${userClass}${selectedClass}${finderClass}" data-username="${p.username}">
            <div class="player-row-top">
                <span class="player-rank">#${rank}</span>
                <span class="rating-square" onclick="window.showMiniProfile('${p.username}'); event.stopPropagation();" style="background-color: ${ratingColor}; cursor: pointer;"></span>
                <span class="player-username">${p.username}</span>
                <span class="player-rating-val">${ratingDisplay}</span>
            </div>
            <div class="player-row-bottom">
                <span class="player-flag">${p.country_flag || '🏳️'}</span>
                <span class="player-input-icon">${inputIcon}</span>
                ${trophyHtml}
                <div style="flex:1;"></div>
                <span class="player-words-count">${p.words_count} words</span>
                <span class="player-score-val">${(p.score === 0 && p.words_count === 0 && (!p.invalid_words || p.invalid_words.length === 0) && (state && state.state === 'intermission')) ? 'DNP' : Math.max(0, p.score) + ' pts'}</span>
            </div>
        </div>
        `;
    }).join('');

    if (html === lastPlayersHtml) return;
    
    // SAVE SCROLL POSITION
    const oldScrollTop = listEl.scrollTop;
    
    lastPlayersHtml = html;
    listEl.innerHTML = html;

    // RESTORE SCROLL POSITION (Prevents jumping on score updates)
    listEl.scrollTop = oldScrollTop;

    // Add click listeners for selection
    const items = listEl.querySelectorAll('.player-item');
    items.forEach(item => {
        item.addEventListener('click', () => {
            const username = item.dataset.username;
            // Toggle selection
            if (selectedPlayerUsername === username) {
                selectedPlayerUsername = null; // Deselect
            } else {
                selectedPlayerUsername = username;
            }
            // Trigger immediate update (or wait for next poll)
            // waiting for next poll (1s) is fine, or we can force it
            updateGameState();
        });
    });
}

// Chat Logic
let lastChatTimestamp = 0;

function resetChat() {
    console.log('[play.js] resetting chat state');
    lastChatTimestamp = 0;
    const listEl = document.getElementById('chat-history');
    if (listEl) {
        listEl.innerHTML = '<p class="placeholder">No messages yet</p>';
    }
}

function renderChat(messages) {
    const listEl = document.getElementById('chat-history');
    if (!listEl) return;

    if (!messages || messages.length === 0) {
        if (listEl.innerHTML.trim() === '') {
            listEl.innerHTML = '<p class="placeholder">No messages yet</p>';
        }
        return;
    }

    // Only update if new messages arrived (using timestamp of last message for precision)
    const latestMsg = messages[messages.length - 1];
    const latestTimestamp = latestMsg ? (latestMsg.time || 0) : 0;
    
    if (latestTimestamp === lastChatTimestamp && messages.length > 0) return;
    lastChatTimestamp = latestTimestamp;

    // Remove placeholder
    const placeholder = listEl.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    // Populate rating cache from current state
    if (window.lastGameState) {
        if (window.lastGameState.players) {
            window.lastGameState.players.forEach(p => {
                if (p.username && p.rating !== undefined) playerRatingCache.set(p.username, p.rating);
            });
        }
        if (window.lastGameState.spectators) {
            window.lastGameState.spectators.forEach(s => {
                if (s.username && s.rating !== undefined) playerRatingCache.set(s.username, s.rating);
            });
        }
    }

    // Render all messages (simple rebuild for now to ensure order)
    const html = messages.map(msg => {
        const username = msg.username;
        const text = msg.message;
        const isSystem = msg.is_system;

        // Escape HTML to prevent XSS
        const safeText = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

        if (isSystem || (username && username.toUpperCase() === 'SYSTEM')) {
            const lowerText = safeText.toLowerCase();
            
            // SPECIAL: Apply Premium Golden styling to winner announcement messages
            const isWinner = msg.is_winner || 
                            lowerText.includes('🏆') || 
                            lowerText.includes('winner') || 
                            lowerText.includes('congratulations') || 
                            lowerText.includes('winning the round');

            const isGoldColor = (msg.color && (msg.color.toLowerCase() === '#ffd700' || msg.color.toLowerCase() === 'gold'));

            if (isWinner || isGoldColor) {
                // High-visibility premium styling
                const winnerStyles = `
                    color: #ffd700 !important; 
                    font-weight: 950 !important; 
                    text-shadow: 0 0 10px rgba(255, 215, 0, 0.7), 0 0 2px rgba(0,0,0,0.9) !important;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                `;
                const containerStyles = `
                    border-left: 4px solid #ffd700;
                    background: rgba(255, 215, 0, 0.08);
                    padding: 8px 12px;
                    margin: 6px 0;
                    border-radius: 0 8px 8px 0;
                `;
                
                return `
                <div class="chat-message chat-sys" style="${containerStyles}">
                    <span class="chat-text gold-status" style="${winnerStyles}">${safeText}</span>
                </div>`;
            }

            return `
            <div class="chat-message chat-sys">
                <span class="chat-text">${safeText}</span>
            </div>`;
        }

        // Determine User Color from Rating Cache
        let userColor = '#a8d5ff'; // Default blue-ish
        if (window.getRatingColor) {
            const rating = playerRatingCache.get(username);
            if (rating !== undefined) {
                userColor = window.getRatingColor(rating);
            }
        }

        return `
        <div class="chat-message">
            <span class="chat-user" style="color: ${userColor};">${username}:</span>
            <span class="chat-text">${safeText}</span>
        </div>`;
    }).join('');

    listEl.innerHTML = html;

    // Add click listeners to usernames
    listEl.querySelectorAll('.chat-user').forEach(userEl => {
        userEl.style.cursor = 'pointer';
        userEl.title = "View profile";
        userEl.onclick = () => {
            const rawName = userEl.innerText.trim();
            const cleanName = rawName.endsWith(':') ? rawName.slice(0, -1) : rawName;
            if (window.showMiniProfile) window.showMiniProfile(cleanName);
        };
    });

    // Scroll to bottom
    listEl.scrollTop = listEl.scrollHeight;
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    const roomId = getCurrentRoomId();

    if (!message || !roomId) return;

    try {
        await fetch(`/api/room/${roomId}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        input.value = ''; // Clear input
        // updateGameState will pick it up on next poll
    } catch (e) {
        console.error('Failed to send chat:', e);
    }
}

// Add event listeners for chat
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat-input');
    const chatSend = document.getElementById('chat-send-btn');

    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendChatMessage();
        });
    }

    if (chatSend) {
        chatSend.addEventListener('click', sendChatMessage);
    }

    // Find Me button logic
    const findMeBtn = document.getElementById('find-me-btn');
    if (findMeBtn) {
        findMeBtn.addEventListener('click', () => {
            const listEl = document.getElementById('players-list');
            const myCard = listEl.querySelector('.current-user');
            if (myCard) {
                // Precise scroll-to-card in container
                const containerRect = listEl.getBoundingClientRect();
                const cardRect = myCard.getBoundingClientRect();
                const relativeTop = cardRect.top - containerRect.top;
                const scrollTarget = listEl.scrollTop + relativeTop - (containerRect.height / 2) + (cardRect.height / 2);
                
                listEl.scrollTo({ 
                    top: scrollTarget, 
                    behavior: 'smooth' 
                });

                // Subtle highlight animation
                myCard.style.transition = 'background 0.3s ease';
                const originalBg = myCard.style.background;
                myCard.style.background = 'rgba(64, 156, 255, 0.6)';
                setTimeout(() => {
                    myCard.style.background = originalBg;
                }, 800);
            }
        });
    }

    const findFriendsBtn = document.getElementById('find-friends-btn');
    if (findFriendsBtn) {
        findFriendsBtn.addEventListener('click', async () => {
            if (!findFriendsMode) {
                // Fetch friends list
                try {
                    const resp = await fetch('/api/friends/list');
                    if (resp.ok) {
                        const data = await resp.json();
                        if (data.friends) {
                            userFriendsCache = data.friends;
                        }
                    }
                } catch (e) {
                    console.error('Failed to fetch friends for filtering:', e);
                }
            }
            findFriendsMode = !findFriendsMode;
            findFriendsBtn.classList.toggle('active', findFriendsMode);
            // Re-render players immediately
            if (window.lastGameState) {
                const currentUsername = window.lastGameState.your_username || window.currentUser || localStorage.getItem('morpheme_username');
                renderPlayers(window.lastGameState.players, currentUsername, window.lastGameState);
            }
        });
    }

    const showEveryoneBtn = document.getElementById('show-everyone-btn');
    if (showEveryoneBtn) {
        showEveryoneBtn.addEventListener('click', () => {
            // Disable Find Friends Mode
            findFriendsMode = false;
            const ffBtn = document.getElementById('find-friends-btn');
            if (ffBtn) ffBtn.classList.remove('active');

            // Re-render players
            if (window.lastGameState) {
                const currentUsername = window.lastGameState.your_username || window.currentUser || localStorage.getItem('morpheme_username');
                renderPlayers(window.lastGameState.players, currentUsername, window.lastGameState);
            }

            // Scroll to top
            const playersListEl = document.getElementById('players-list');
            if (playersListEl) {
                playersListEl.scrollTop = 0;
            }
        });
    }
});

function displayAllWords(allWords, bonusWord, targetUserWords = [], allFoundWords = [], allWordScores = {}, cswOnlyWords = [], addedWords = []) {
    console.log(`[displayAllWords] RENDERING. BonusWord: "${bonusWord}" | Words count: ${allWords ? allWords.length : 0}`);
    const listEl = document.getElementById('submitted-words-list');
    const titleEl = document.getElementById('words-panel-title');
    if (titleEl && activeWordsTab === 'found') {
        titleEl.textContent = 'All Words';
    }

    if (!allWords || allWords.length === 0) {
        listEl.innerHTML = '<p class="placeholder">No words found</p>';
        return;
    }

    const targetWordsUpper = targetUserWords.map(w => (typeof w === 'object' ? (w.word || '') : w).toUpperCase());
    const allFoundUpper = allFoundWords.map(w => (typeof w === 'object' ? (w.word || '') : w).toUpperCase());
    const cswOnlyUpper = (cswOnlyWords || []).map(w => w.toUpperCase());
    const addedUpper = (addedWords || []).map(w => w.toUpperCase());

    // Sort: Length desc, then Alpha
    // Sort: Color Priority, then Length desc, then Alpha
    // Priority: Red/Orange (Bonus) > Purple (Added) > Blue (Found) > Gold (CSW) > Black/Gray (Missed/Unfound)
    if (bonusWord && allWords) {
        const bonusIdx = allWords.findIndex(w => (typeof w === 'object' ? (w.word || '') : w).toUpperCase() === bonusWord.toUpperCase());
        console.log(`[displayAllWords] BonusWord search: "${bonusWord}" found at index: ${bonusIdx}`);
    }
    console.log(`[displayAllWords] Added words for highlight:`, addedUpper);
    const sortedWords = [...allWords].sort((a, b) => {
        const wordA = (typeof a === 'object' ? (a.word || '') : a).toUpperCase();
        const wordB = (typeof b === 'object' ? (b.word || '') : b).toUpperCase();
        const bonusUpper = bonusWord ? bonusWord.toUpperCase().trim() : null;

        // 0. Bonus Word (Absolute Top Priority)
        if (bonusUpper) {
            const wordAUpper = wordA.trim();
            const wordBUpper = wordB.trim();
            if (wordAUpper === bonusUpper) return -1;
            if (wordBUpper === bonusUpper) return 1;
        }

        // 1. Length (Desc) - Primary sort
        const lenA = wordA.length;
        const lenB = wordB.length;
        if (lenA !== lenB) return lenB - lenA;

        // 2. Alphabetical (Asc) - Secondary sort
        return wordA.localeCompare(wordB);
    });

    console.log('[renderWordsList] Rendering words:', sortedWords.length);

    // PERFORMANCE: Use event delegation instead of 2000 individual listeners
    if (listEl && !listEl.hasScoringListener) {
        listEl.addEventListener('click', (e) => {
            const item = e.target.closest('.word-item');
            if (!item || !item.dataset.word) return;
            
            const word = item.dataset.word.toUpperCase();
            
            // Mark that the user is explicitly viewing a definition (so winner announcement doesn't overwrite it)
            if (window.lastGameState && window.lastGameState.state === 'intermission') {
                window.userViewingDefinitionIntermission = true;
            }

            if (highlightedFoundWord === word) {
                highlightedFoundWord = null;
            } else {
                highlightedFoundWord = word;
            }

            // Highlighting update - fast approach using classes on the parent
            const allItems = listEl.querySelectorAll('.word-item');
            allItems.forEach(el => el.classList.remove('finder-active'));
            if (highlightedFoundWord) item.classList.add('finder-active');

            // Optionally call updateGameState to sync other UI parts, but keep it minimal
            if (window.lastGameState) {
                updateGameState(window.lastGameState);
            }
            
            window.fetchDefinition(item.dataset.word);
        });
        listEl.hasScoringListener = true;
    }

    listEl.innerHTML = sortedWords.map(entry => {
        const word = (typeof entry === 'object' ? (entry.word || '') : entry);
        const wordUpper = word.toUpperCase();
        const bonusUpper = bonusWord ? bonusWord.toUpperCase().trim() : null;
        const isBonus = bonusUpper && wordUpper.trim() === bonusUpper;
        const isCSWOnly = cswOnlyUpper.includes(wordUpper);
        const isAddedWord = addedUpper.includes(wordUpper);
        const isTargetFound = targetWordsUpper.includes(wordUpper);
        const isFoundByAny = allFoundUpper.includes(wordUpper);
        const pointsData = allWordScores[word] || allWordScores[wordUpper] || 0;
        let pointsText = pointsData;

        if (typeof pointsData === 'object' && pointsData !== null) {
            const hasBonusLetter = (pointsData.bonus_letter_points || 0) > 0;
            const hasBonusWord = (pointsData.bonus_word_points || 0) > 0;
            const hasEO = (pointsData.either_or_points || 0) > 0;
            
            if (hasBonusLetter || hasBonusWord || hasEO) {
                let parts = [];
                // Base
                parts.push(pointsData.base);
                
                if (hasBonusWord) parts.push(pointsData.bonus_word_points);
                if (hasBonusLetter) parts.push(pointsData.bonus_letter_points);
                if (hasEO) parts.push(pointsData.either_or_points);
                
                if (parts.length > 1) {
                    pointsText = `${parts.join(' + ')} = ${pointsData.total}`;
                } else {
                    pointsText = pointsData.total;
                }
            } else {
                pointsText = pointsData.total;
            }
        }

        let className = 'word-item';

        // Highlighting for finder feature
        if (highlightedFoundWord === wordUpper) {
            className += ' finder-active';
        }

        // Priority classes
        if (isBonus) className += ' bonus-word';
        else if (isAddedWord) className += ' added-word';
        else if (isTargetFound) className += ' player-word';
        else if (isCSWOnly) className += ' csw-only';
        else if (isFoundByAny) className += ' found-by-other missed';
        else className += ' unfound missed';

        const indicatorClass = isFoundByAny ? 'found-indicator present' : 'found-indicator empty';
        const indicatorIcon = isFoundByAny ? '✓' : '';
        
        // Bonus indicator (Left side)
        let bonusIndicator = '';
        if (typeof pointsData === 'object' && pointsData !== null) {
            if ((pointsData.bonus_letter_points || 0) > 0) {
                 // Try to get the actual bonus letter from the board
                 const s = window.lastGameState;
                 let bChar = '★';
                 if (s && s.bonus_cell) {
                     const r = s.bonus_cell.r, c = s.bonus_cell.c;
                     if (s.board && s.board[r] && s.board[r][c]) bChar = s.board[r][c];
                 }
                 bonusIndicator = `<span class="list-bonus-tag bl-tag" title="Used Bonus Letter">BL</span>`;
            } else if ((pointsData.bonus_word_points || 0) > 0) {
                 bonusIndicator = `<span class="list-bonus-tag bonus-word-tag">★</span>`;
            } else if ((pointsData.either_or_points || 0) > 0) {
                 bonusIndicator = `<span class="list-bonus-tag eo-tag">EO</span>`;
            }
        }

        const indicator = `<span class="${indicatorClass}">${indicatorIcon}</span>`;

        return `<div class="${className}" data-word="${word}">
            <div class="word-left">
                ${indicator}
                ${bonusIndicator}
                <span class="word-text">${wordUpper}</span>
            </div>
            <span class="points-math">${pointsText}</span>
        </div>`;
    }).join('');
}

// ... existing functions (updateParameters, renderBoard, etc) ...
// Copying existing helper functions to ensure file completion

function updateParameters(state) {
    if (!state) return;
    try {
        // Clear stale local flags if we are receiving a legitimate room state
        if (state.room_id && !state.room_id.includes('tournament')) {
            isTournamentPlay = false;
        }
        
        // Display mappings
    const typeMap = {
        'accumulative': 'Accumulative',
        'fcfs': 'First Come First Serve',
        'split': 'Split Points',
        '3d': 'Cube',
        'private': 'With Friends',
        'tournament': 'Tournament',
        'solo_accumulative': 'Solo'
    };

    const timerVal = document.getElementById('timer-value');
    // Timer display is handled by syncTimerWithServer and updateLocalTimer for smoothness

    // Update Title
    const titleEl = document.getElementById('game-title');
    if (titleEl) {
        titleEl.textContent = typeMap[state.game_type] || 'Morpheme';
    }

    // Update Header Title (legacy, if still present)
    const headerTitle = document.querySelector('.play-header h2');
    if (headerTitle) {
        headerTitle.textContent = typeMap[state.game_type] || 'Game';
    }

    const sp = state.spinner_params || {};
    // --- STICKY PARAMETERS LOGIC ---
    // User Request: Only change labels at the same time as the gold-to-black reveal.
    // We maintain a local cache of what's currently shown.
    if (!window._displayedParams) {
        window._displayedParams = { dims: '', time: '', bonus: '', diff: '', uniq: '' };
        window._lastRevealedRoundCount = -1;
    }

    const currentRound = state.current_round || 0;
    const isRevealed = !!(state.spinner_params && state.spinner_params_revealed);
    const wasRevealed = !!(window._lastRevealedState);
    const isIntermission = state.intermission === true || state.state === 'intermission';
    const now = Date.now();

    // Determine current fact-checked labels
    // INTERMISSION REVEAL: During intermission, we prefer the spinner_params (sp) if they are the target of the reveal.
    // Otherwise, we prefer the state's authoritative ground-truth labels.
    const factBoardDims = sp.board_dimensions || state.board_dimensions || '4x4';
    const factTimeLimit = sp.time_limit || state.time_limit || 60;
    const preferSp = isIntermission && isRevealed;

    const factFmt = (preferSp ? (sp.board_format || state.current_board_format) : (state.current_board_format || sp.board_format)) || 'Normal';
    const factDiff = (preferSp ? (sp.difficulty || state.current_difficulty) : (state.current_difficulty || sp.difficulty)) || 'Medium';
    const factBonus = (preferSp ? (sp.bonus_word_length || state.current_bonus_word_length) : (state.current_bonus_word_length || sp.bonus_word_length)) || (state.bonus_word ? state.bonus_word.length : 0);
    const factMinLen = (preferSp ? (sp.min_word_length || state.current_min_length) : (state.current_min_length || sp.min_word_length)) || 3;
    const factDict = (preferSp ? (sp.dictionary || state.current_dictionary) : (state.current_dictionary || sp.dictionary)) || 'NWL';
    const factWordRange = (preferSp ? (sp.word_count_range || state.current_word_count_range) : (state.current_word_count_range || sp.word_count_range)) || 'Random';
    
    let factUniq = 0; 
    if (preferSp) {
        factUniq = sp.uniqueness || 0;
    } else {
        factUniq = (state.current_uniqueness !== undefined && state.current_uniqueness !== null) ? state.current_uniqueness : 0;
    }

    // UPDATE POLICY:
    // 1. If Active Round: Update immediately to match facts.
    // 2. If Intermission & NOT Revealed: Stay sticky to previous round facts.
    // 3. If Intermission & JUST Revealed: Update NOW and trigger animation.
    
    let shouldUpdateLabels = false;
    let triggerAnimation = false;

    if (!isIntermission) {
        shouldUpdateLabels = true;
        window._lastRevealedRoundCount = currentRound; 
    } else {
        if (isRevealed && !wasRevealed && (currentRound !== window._animTriggeredForRound)) {
            shouldUpdateLabels = true;
            triggerAnimation = true;
            window._animTriggeredForRound = currentRound;
            console.log("[play.js] REVEAL ANIMATION + LABEL UPDATE for round:", currentRound);
        } else if (isRevealed) {
            shouldUpdateLabels = true;
        } else {
            shouldUpdateLabels = false;
        }
    }

    if (shouldUpdateLabels) {
        const newUniq = factUniq;
        const newDiff = factDiff;
        const stringified = JSON.stringify([factBoardDims, factTimeLimit, factMinLen, factDict, factWordRange, factFmt, newDiff, newUniq, factBonus]);
        
        if (window._lastParamString !== stringified) {
            window._lastParamString = stringified;
            
            window._displayedParams.dims = factBoardDims;
            window._displayedParams.time = factTimeLimit + 's';
            window._displayedParams.min = factMinLen + 'L';
            window._displayedParams.dict = factDict;
            window._displayedParams.range = factWordRange;
            window._displayedParams.fmt = factFmt;
            window._displayedParams.bonus = factBonus + 'L';
            
            let diffLabel = factDiff;
            if (diffLabel === 'Varying...') diffLabel = 'Random';
            else if (diffLabel === 'Normal') diffLabel = 'Medium';
            else if (diffLabel === 'Expert' || diffLabel === 'Difficult') diffLabel = 'Hard';
            else if (diffLabel === 'Beginner') diffLabel = 'Easy';
            
            const uniquePct = (newUniq > 0 && !diffLabel.includes('(')) ? ` (${Math.round(newUniq * 100)}%)` : "";
            window._displayedParams.diff = diffLabel + uniquePct;
            
            if (typeof updateColorBarHighlight === 'function') {
                updateColorBarHighlight(diffLabel, newUniq);
            }

            // Apply to DOM
            if (document.getElementById('param-board')) document.getElementById('param-board').textContent = window._displayedParams.dims;
            if (document.getElementById('param-time')) document.getElementById('param-time').textContent = window._displayedParams.time;
            if (document.getElementById('param-diff')) document.getElementById('param-diff').textContent = window._displayedParams.diff;
            if (document.getElementById('param-min')) document.getElementById('param-min').textContent = window._displayedParams.min;
            if (document.getElementById('param-dict')) document.getElementById('param-dict').textContent = window._displayedParams.dict;
            if (document.getElementById('param-range')) document.getElementById('param-range').textContent = window._displayedParams.range;
            if (document.getElementById('param-bonus')) document.getElementById('param-bonus').textContent = window._displayedParams.bonus;
        }
    }

    if (triggerAnimation) {
        const paramContainer = document.querySelector('.game-params');
        if (paramContainer) {
            paramContainer.classList.remove('reveal-new');
            void paramContainer.offsetWidth; // Force reflow
            paramContainer.classList.add('reveal-new');
        }
    }
    
    window._lastRevealedState = isRevealed;

    const format = document.getElementById('param-format');
    if (format && (shouldUpdateLabels || !format.textContent)) {
        format.textContent = window._displayedParams.fmt;
    }
    const pBonus = document.getElementById('param-bonus');
    if (pBonus && (shouldUpdateLabels || !pBonus.textContent)) {
        pBonus.textContent = window._displayedParams.bonus;
    }
    } catch (err) {
        console.error('[play.js] Error in updateParameters:', err);
    }
}

function updateTimer(seconds) {
    // Legacy local timer update (called by interval)
    // see updateLocalTimer
}

function syncTimerWithServer(state) {
    const clientTime = Date.now() / 1000;
    const serverTime = state.server_time;
    if (serverTime) {
        const currentOffset = serverTime - clientTime;
        // Establish once, then only update if drifting by > 3s
        if (stableServerTimeOffset === null || Math.abs(currentOffset - stableServerTimeOffset) > 3) {
            stableServerTimeOffset = currentOffset;
        }
    }

    let endTime = 0;
    if (state.state === 'active') {
        endTime = state.round_end_time || (state.server_time + state.time_remaining);
    } else if (state.state === 'intermission') {
        endTime = state.intermission_end_time || (state.server_time + state.time_remaining);
    }

    if (endTime && stableServerTimeOffset !== null) {
        localEndTime = endTime - stableServerTimeOffset;
    }

    // SPECIAL CASE: 24h Rooms align to LOCAL MIDNIGHT for display
    if (state.time_limit >= 7200) {
        const now = new Date();
        const tomorrow = new Date(now);
        tomorrow.setDate(now.getDate() + 1);
        tomorrow.setHours(0, 0, 0, 0);
        localEndTime = tomorrow.getTime() / 1000;
        timerFormatIs24h = true;
    }

    if (!timerInterval && localEndTime > 0 && !document.hidden) {
        timerInterval = setInterval(updateLocalTimer, 500); // Optimized from 100ms to 500ms
    } else if ((localEndTime <= 0 || document.hidden) && timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function updateLocalTimer() {
    if (!localEndTime) return;

    if (!cachedTimerValueEl) cachedTimerValueEl = document.getElementById('timer-value');
    if (!cachedBoardPanelEl) cachedBoardPanelEl = document.querySelector('.board-panel');

    const now = Date.now() / 1000;
    const remaining = Math.max(0, localEndTime - now);
    const seconds = Math.ceil(remaining);

    // Format determination: Stick with it once detected
    if (window.lastGameState && window.lastGameState.time_limit >= 3600) {
        timerFormatIs24h = true;
    }

    let display;
    if (timerFormatIs24h) {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        // Use H:MM:SS format as requested (no leading zero on hours if < 10)
        display = `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        display = `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    if (cachedTimerValueEl) {
        if (cachedTimerValueEl.textContent !== display) {
            cachedTimerValueEl.textContent = display;
            cachedTimerValueEl.style.fontVariantNumeric = 'tabular-nums';
        }

        // Freeze detection
        if (Date.now() - lastServerUpdate > 5000) {
            cachedTimerValueEl.style.color = '#ff6b6b';
        } else {
            cachedTimerValueEl.style.color = '';
        }
    }

    // Low time visual
    if (cachedBoardPanelEl) {
        const currentState = (window.lastGameState && window.lastGameState.state) || 'active';
        if (remaining <= 10 && remaining > 0 && currentState === 'active') {
            cachedBoardPanelEl.classList.add('low-time-warning');
        } else {
            cachedBoardPanelEl.classList.remove('low-time-warning');
        }
    }

    if (remaining <= 0 && timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        if (cachedBoardPanelEl) cachedBoardPanelEl.classList.remove('low-time-warning');

        // User Request: Automatic/instant transition at 0:00
        const currentState = (window.lastGameState && window.lastGameState.state) || 'active';
        if (currentState === 'intermission') {
            console.log('[play.js] Intermission local timer reached 0:00 - Triggering immediate server poll for next round.');
            updateGameState();
        }
    }

    // -- Next Round Bell Logic --
    if (window.lastGameState && window.lastGameState.state === 'intermission') {
        const isEnabled = window.userSettings && (window.userSettings.next_round_bell_enabled === true || window.userSettings.next_round_bell_enabled === 'true');
        const bellType = (window.userSettings && window.userSettings.next_round_bell_type) || 'bell1';

        if (isEnabled && seconds === 10 && !hasPlayedIntermissionBell) {
            console.log(`[play.js] Playing intermission bell: ${bellType}`);
            const audio = new Audio(`/static/audio/${bellType}.wav`);
            audio.play().catch(e => console.warn('Bell audio failed:', e));
            hasPlayedIntermissionBell = true;
        }
    } else {
        // Reset flag when not in intermission
        hasPlayedIntermissionBell = false;
    }
}

// Helper for special match timers (Tournament, Private Match)
function updateSpecialMatchTimer(seconds) {
    const timerEl = document.getElementById('timer-value');
    if (timerEl) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        timerEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    }
}

function renderBoard(board, grayed = false, is3D = false, state = null) {
    if (window.hideLoadingOverlay) window.hideLoadingOverlay();
    const boardEl = document.getElementById('game-board');
    if (!boardEl || !board) return;

    // Determine dimensions early
    let rows = 0;
    let cols = 0;
    if (board && board.length > 0) {
        rows = (is3D && Array.isArray(board[0])) ? board[0].length : board.length;
        cols = (is3D && Array.isArray(board[0])) ? board[0][0].length : (board[0] ? board[0].length : 0);
    }

    if (cols > 0 && rows > 0) {
        boardEl.style.setProperty('--board-cols', cols);
        boardEl.style.setProperty('--board-rows', rows);
    }

    const boardPanel = boardEl.closest('.board-panel');
    if (boardPanel) {
        if (is3D || !board || board.length === 0) {
            boardPanel.classList.remove('full-bleed-mobile');
        } else {
            boardPanel.classList.add('full-bleed-mobile');
        }
    }

    // Toggle 3D hint below word input
    const rotateHint = document.getElementById('cube-rotate-hint');
    if (rotateHint) {
        const showHint = is3D && window.lastGameState && window.lastGameState.state === 'active';
        if (showHint) rotateHint.classList.remove('hidden');
        else rotateHint.classList.add('hidden');
    }

    const rotateBtn = document.getElementById('rotate-board-btn');
    if (rotateBtn) {
        if (is3D) rotateBtn.classList.add('hidden');
        else rotateBtn.classList.remove('hidden');
    }

    // Handle Empty/Loading State
    // Check if board has any actual letter content
    let hasLetters = false;
    if (is3D) {
        if (Array.isArray(board) && board.length > 0) {
            hasLetters = board.some(f => Array.isArray(f) && f.some(r => Array.isArray(r) && r.some(c => c && typeof c === 'string' && c.trim() !== '')));
        }
    } else {
        if (Array.isArray(board) && board.length > 0) {
            hasLetters = board.some(row => Array.isArray(row) && row.some(cell => cell && typeof cell === 'string' && cell.trim() !== ''));
        }
    }
    
    // IF board is empty OR has no letters, show loading spinner
    if (board.length === 0 || !hasLetters) {
        const loadingMsg = "";
        const subMsg = "";
        
        boardEl.innerHTML = `
            <style>
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                @keyframes pulse { 0%, 100% { opacity: 0.6; transform: scale(0.9); } 50% { opacity: 1; transform: scale(1.1); } }
                .loading-spinner { animation: spin 0.8s linear infinite !important; }
                .optimizing-text { animation: pulse 2s ease-in-out infinite; }
            </style>
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; width:100%; height:300px; color:var(--text-primary);">
                <div class="loading-spinner" style="margin-bottom:15px; width:45px; height:45px; border:4px solid rgba(var(--text-primary-rgb),0.1); border-top:4px solid var(--accent-color); border-radius:50%;"></div>
                <div class="optimizing-text" style="font-weight:700; font-size:1.2rem; text-transform:uppercase; letter-spacing:1px; color:var(--accent-color);">${loadingMsg}</div>
                <div style="font-size:0.9rem; opacity:0.7; margin-top:10px; text-align:center; max-width:80%;">${subMsg}</div>
            </div>
        `;
        boardEl.className = 'game-board';
        boardEl.style.display = 'flex';
        boardEl.style.justifyContent = 'center';
        boardEl.style.alignItems = 'center';
        // CRITICAL: Clear grid styles so the spinner isn't constrained
        boardEl.style.gridTemplateColumns = '';
        boardEl.style.gridTemplateRows = '';
        return;
    }

    // Optimization: Skip if board hasn't changed
    const densityJSON = JSON.stringify((window.lastGameState && window.lastGameState.cell_density) || []);
    const boardJSON = JSON.stringify(board);
    if (boardJSON === lastRenderedBoardJSON && 
        densityJSON === lastRenderedDensityJSON &&
        grayed === lastRenderedGrayed && 
        isBoardRotated === lastRenderedRotation && 
        boardEl.classList.contains('game-board')) {
        reapplyBoardHighlights();
        return;
    }
    lastRenderedBoardJSON = boardJSON;
    lastRenderedDensityJSON = densityJSON;
    lastRenderedGrayed = grayed;
    lastRenderedRotation = isBoardRotated;

    boardEl.className = 'game-board';
    boardEl.style.display = '';
    
    // FORTE: Enforce grid sizing BEFORE children are added to prevent "vertical column" flickering
    if (!is3D && cols > 0 && rows > 0) {
        boardEl.style.gridTemplateColumns = `repeat(${cols}, var(--cell-size, 60px))`;
        boardEl.style.gridTemplateRows = `repeat(${rows}, var(--cell-size, 60px))`;
    } else {
        boardEl.style.gridTemplateColumns = '';
        boardEl.style.gridTemplateRows = '';
    }
    
    if (is3D) {
        boardEl.classList.add('is-3d-view');
        boardEl.style.display = 'block';
        
        let html = `
            <div class="cube-container">
                <div class="cube" id="game-cube" style="transform: rotateX(${window.cubeRotationX !== undefined ? window.cubeRotationX : -30}deg) rotateY(${window.cubeRotationY !== undefined ? window.cubeRotationY : 45}deg);">
        `;
        const faceClasses = ['face-front', 'face-back', 'face-left', 'face-right', 'face-top', 'face-bottom'];
        board.forEach((face, f) => {
            html += `<div class="cube-face ${faceClasses[f]}">`;
            face.forEach((row, r) => {
                row.forEach((cell, c) => {
                    const char = (cell || "").trim();
                    let tileHtml = "";
                    if (char.includes('/')) {
                        const [top, bottom] = char.split('/');
                        const topDisp = top === 'Q' ? 'QU' : top;
                        const bottomDisp = bottom === 'Q' ? 'QU' : bottom;
                        tileHtml = `
                            <div class="dual-letter-container" style="display:flex; flex-direction:column; justify-content:center; height:100%; width:100%; align-items:center; gap: 2px;">
                                <span style="font-size:0.52em; line-height:0.9; font-weight:900;">${topDisp}</span>
                                <div class="dual-divider" style="width:65%; height:1px; background:rgba(0,0,0,0.3); margin:1px 0;"></div>
                                <span style="font-size:0.52em; line-height:0.9; font-weight:900;">${bottomDisp}</span>
                            </div>
                        `;
                    } else {
                        const displayL = char === 'Q' ? 'QU' : char;
                        tileHtml = displayL;
                    }

                    // 3D Bonus Highlight Detection
                    let bonusClass = "";
                        const bc = (state && state.bonus_cell) ? state.bonus_cell : (window.lastGameState ? window.lastGameState.bonus_cell : null);
                        if (Array.isArray(bc) && bc.length === 3) {
                            if (Number(bc[0]) === f && Number(bc[1]) === r && Number(bc[2]) === c) {
                                bonusClass = " bonus-highlight";
                                tileHtml += `<span class="bonus-star">★</span>`;
                            }
                        }
                    
                    // Valued Letters support for 3D
                    let tileValue = "";
                    let bFormat = (state && state.current_board_format) ? state.current_board_format : ((window.lastGameState && window.lastGameState.current_board_format) ? window.lastGameState.current_board_format : 'Normal');
                    if (bFormat.toLowerCase().includes('valued') && !char.includes('/')) {
                        const val = LETTER_VALUES[char.toUpperCase()] || 1;
                        tileValue = `<span class="tile-value">${val}</span>`;
                    }

                    // Density Format Support for 3D
                    let densityStyle = "";
                    if (window.lastGameState && window.lastGameState.cell_density && bFormat && bFormat.toLowerCase().includes('density')) {
                         const grid = window.lastGameState.cell_density;
                         const maxD = window.lastGameState.max_cell_density || 1;
                         if (grid && grid[f]) {
                             const cur = grid[f][r][c];
                             if (cur === 0) {
                                 densityStyle = "background: #ffffff !important; color: #000000 !important; box-shadow: inset 0 0 10px rgba(0,0,0,0.1) !important;";
                             } else {
                                 // HIGH CONTRAST GRAYSCALE: Normalizing by GLOBAL MAX density
                                 const ratio = Math.max(0, Math.min(1, cur / maxD));
                                 // 100% (white) to 0% (black)
                                 const grayVal = Math.round(100 - (ratio * 100));
                                 const textColor = (grayVal < 50) ? '#ffffff' : '#000000';
                                 densityStyle = `background: hsl(0, 0%, ${grayVal}%) !important; color: ${textColor} !important; transition: background 0.4s ease, color 0.4s ease;`;
                             }
                         }
                    }
                    
                    // USER REQUEST: If this is the bonus cell, do NOT apply density shading to the background.
                    if (bonusClass.includes('bonus-highlight')) {
                        densityStyle = '';
                    }

                    html += `<div class="cube-cell board-cell tile-cell${bonusClass}" data-f="${f}" data-r="${r}" data-c="${c}" data-letter="${char}" style="${densityStyle}">${tileHtml}${tileValue}</div>`;
                });
            });
            html += `</div>`;
        });
        html += `
                </div>
            </div>
        `;
        boardEl.innerHTML = html;
        setupCubeRotation();
        reapplyBoardHighlights();
        return;
    }

    // rows and cols are already defined in the scope from lines 2199-2200
    if (cols === 0 || rows === 0) return;

    // OPTIMIZATION: In Density mode, we want SMOOTH transitions.
    // Wiping innerHTML destroys elements and breaks CSS transitions. 
    // We only wipe if the board dimensions have changed.
    const expectedCount = rows * cols;
    const currentCells = boardEl.querySelectorAll('.board-cell:not(.cube-cell)'); // Only 2D cells
    
    if (currentCells.length === expectedCount && !is3D) {
        // SYNC EXISTING CELLS (Supports Dynamic Shading Transitions)
        let idx = 0;
        if (isBoardRotated) {
            for (let r = rows - 1; r >= 0; r--) {
                for (let c = cols - 1; c >= 0; c--) {
                    const existing = currentCells[idx++];
                    // Update only if necessary
                    updateBoardCell(existing, r, c, board[r][c], grayed, undefined, state);
                }
            }
        } else {
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const existing = currentCells[idx++];
                    updateBoardCell(existing, r, c, board[r][c], grayed, undefined, state);
                }
            }
        }
    } else {
        // FULL RERENDER
        boardEl.innerHTML = '';
        if (isBoardRotated) {
            for (let r = rows - 1; r >= 0; r--) {
                for (let c = cols - 1; c >= 0; c--) {
                    const cell = createBoardCell(r, c, board[r][c], grayed, undefined, state);
                    boardEl.appendChild(cell);
                }
            }
        } else {
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const cell = createBoardCell(r, c, board[r][c], grayed, undefined, state);
                    boardEl.appendChild(cell);
                }
            }
        }
    }

    // Trigger overflow check but don't clear grid styles anymore (to prevent flickering)
    setTimeout(checkBoardOverflow, 50);
    reapplyBoardHighlights();
}


// Helper: Check if board panel needs vertical scrolling
// Continuous Adaptive Layout Engine (formerly checkBoardOverflow)
// Emergency Shim: Replaced older logic with confirmed strict capped logic
// Helper: Check if board panel needs vertical scrolling
// Continuous Adaptive Layout Engine (Board-First + Capped)
// Emergency Shim: Replaced older logic with confirmed strict capped logic
function checkBoardOverflow() {
    const playPage = document.getElementById('page-play');
    const boardPanel = document.querySelector('.board-panel');
    const boardEl = document.getElementById('game-board');
    if (!playPage || !boardPanel || !boardEl) return;

    // 1. Get Board Dimensions (Rows & Cols)
    let cols = 0;
    let rows = 0;
    if (window.lastGameState && window.lastGameState.board && window.lastGameState.board[0]) {
        cols = window.lastGameState.board[0].length;
        rows = window.lastGameState.board.length;
    } else {
        const gridCols = boardEl.style.gridTemplateColumns;
        if (gridCols && gridCols.includes('repeat')) {
            const match = gridCols.match(/repeat\((\d+)/);
            if (match) cols = parseInt(match[1]);
        }
        const gridRows = boardEl.style.gridTemplateRows;
        if (gridRows && gridRows.includes('repeat')) {
            const match = gridRows.match(/repeat\((\d+)/);
            if (match) rows = parseInt(match[1]);
        }
    }
    // FALLBACKS: Try to get from last rendered state OR global state
    if (!cols || !rows) {
        if (window.lastGameState && window.lastGameState.board_dimensions) {
             const parts2 = window.lastGameState.board_dimensions.split('x');
             if (parts2.length >= 2) {
                 // Format is usually [Rows]x[Cols] or [Face]x[Rows]x[Cols]
                 if (parts2.length === 3) {
                     rows = parseInt(parts2[1]);
                     cols = parseInt(parts2[2]);
                 } else {
                     rows = parseInt(parts2[0]);
                     cols = parseInt(parts2[1]);
                 }
             }
        }
    }

    if (cols === 0 || rows === 0) return;
    
    if (!cols) cols = 4;
    if (!rows) rows = 4;

    const isSixByEight = (cols === 6 && rows === 8) || (cols === 8 && rows === 6);
    console.log(`[LayoutCheck] Dimensions: ${cols}x${rows}. Is 6x8 target? ${isSixByEight}`);

    // Get Cell Size (Always fetch to ensure sync with User Settings)
    const computedStyle = getComputedStyle(document.documentElement);
    const cellSizeVar = computedStyle.getPropertyValue('--cell-size').trim();
    let baseCellSize = parseInt(cellSizeVar) || 60;
    
    // User Request: Settings-true board size (Remove 6x8 override)
    let cellSize = baseCellSize;

    // 2. Calculate Required Width for Board
    // Width = (Cols * Size) + Gap + Padding + Scrollbar
    const boardGap = 4 * (cols - 1);
    const isMobileView = window.innerWidth <= 900;
    const boardPadding = isMobileView ? 16 : 40; // Full-bleed padding on mobile vs standard
    let requiredBoardWidth = (cols * cellSize) + boardGap + boardPadding;

    // Add Scrollbar Width if present
    const scrollbarWidth = boardPanel.offsetWidth - boardPanel.clientWidth;
    requiredBoardWidth += scrollbarWidth;

    // Constrain cell size on small screens so the board ALWAYS fits snugly without horizontal overflow
    const maxAllowedWidth = window.innerWidth - 20; // 10px margin on each side
    if (requiredBoardWidth > maxAllowedWidth) {
        const targetCellSize = Math.floor((maxAllowedWidth - boardGap - boardPadding - scrollbarWidth) / cols);
        cellSize = Math.max(25, targetCellSize); // Prevent shrinking below readable 25px
        requiredBoardWidth = (cols * cellSize) + boardGap + boardPadding + scrollbarWidth;
    }

    playPage.style.setProperty('--cell-size', `${cellSize}px`);
    window.cachedCellSize = cellSize; // Store for other listeners if needed

    // 3. Calculate Available Space
    const windowWidth = window.innerWidth;
    // Safety Margin: Standard
    const safetyMargin = 120;

    // The key difference: We start with Window and subtract Board
    const availableForPanels = windowWidth - requiredBoardWidth - safetyMargin;

    // 4. Distribute Remaining Space Dynamically to fit the board snug
    // Fetch adaptive default limits based on screen size (matching media queries)
    let maxLeft = isSixByEight ? 350 : 340;
    let maxRight = isSixByEight ? 330 : 320;

    if (windowWidth >= 1920) {
        maxLeft = 520;
        maxRight = 500;
    } else if (windowWidth >= 1600) {
        maxLeft = 480;
        maxRight = 460;
    } else if (windowWidth >= 1400) {
        maxLeft = 440;
        maxRight = 420;
    } else if (windowWidth < 1200) {
        maxLeft = 320;
        maxRight = 300;
    }
    
    // Set a reasonable minimum boundary so sidebars remain readable/functional
    const minLeft = 200;
    const minRight = 200;

    let newLeft = maxLeft;
    let newRight = maxRight;

    // Total space requested by default maximum panels
    const defaultTotalPanels = maxLeft + maxRight;

    if (availableForPanels >= defaultTotalPanels) {
        // Plenty of room! Keep panels at their full gorgeous sizes
        newLeft = maxLeft;
        newRight = maxRight;
    } else {
        // The board is very large! We must shrink the side panels horizontally
        // to make sure the board fits snugly in the middle.
        if (availableForPanels >= (minLeft + minRight)) {
            // We can distribute the constrained space proportionally!
            const ratio = availableForPanels / defaultTotalPanels;
            newLeft = Math.floor(maxLeft * ratio);
            newRight = Math.floor(maxRight * ratio);
            
            // Enforce minimum limits
            newLeft = Math.max(newLeft, minLeft);
            newRight = Math.max(newRight, minRight);
        } else {
            // Under extreme constraints, use absolute minimal sizes
            newLeft = minLeft;
            newRight = minRight;
        }
    }

    // 5. Apply
    playPage.style.setProperty('--left-panel-w', `${newLeft}px`);
    playPage.style.setProperty('--right-panel-w', `${newRight}px`);

    // CRITICAL: Apply directly to .play-grid to override CSS media queries specifying these vars on .play-grid
    const playGrid = document.querySelector('.play-grid');
    if (playGrid) {
        playGrid.style.setProperty('--left-panel-w', `${newLeft}px`);
        playGrid.style.setProperty('--right-panel-w', `${newRight}px`);
    }

    // Maintain vertical scroll class
    if (boardPanel.scrollHeight > boardPanel.clientHeight) {
        playPage.classList.add('has-vertical-scroll');
    } else {
        playPage.classList.remove('has-vertical-scroll');
    }
}

// Deprecated old function (renamed to avoid conflict)
function checkBoardOverflow_OLD() {
    const playPage = document.getElementById('page-play');
    const boardPanel = document.querySelector('.board-panel');
    const boardEl = document.getElementById('game-board');
    if (!playPage || !boardPanel || !boardEl) return;

    // 1. Get Board Dimensions
    let cols = 0;
    if (window.lastGameState && window.lastGameState.board && window.lastGameState.board[0]) {
        cols = window.lastGameState.board[0].length;
    } else {
        const gridCols = boardEl.style.gridTemplateColumns;
        if (gridCols && gridCols.includes('repeat')) {
            const match = gridCols.match(/repeat\((\d+)/);
            if (match) cols = parseInt(match[1]);
        }
    }
    if (!cols) cols = 8;

    // Get Cell Size
    const computedStyle = getComputedStyle(document.documentElement);
    const cellSizeVar = computedStyle.getPropertyValue('--cell-size').trim();
    const cellSize = parseInt(cellSizeVar) || 60;

    // 2. Calculate Required Width
    const boardGap = 4 * (cols - 1);
    const boardPadding = 24;
    let requiredBoardWidth = (cols * cellSize) + boardGap + boardPadding;

    // Add Scrollbar Width if present
    const scrollbarWidth = boardPanel.offsetWidth - boardPanel.clientWidth;
    requiredBoardWidth += scrollbarWidth;

    // 3. Calculate Available Space
    const windowWidth = window.innerWidth;
    // 80px base + 80px safety buffer to prevent scrolling and add breathability
    const layoutGaps = 160;
    const availableForPanels = windowWidth - requiredBoardWidth - layoutGaps;

    // 4. Calculate Panel Widths
    // "Fit the size" -> Fill available space
    // Base proportions
    const baseLeft = 330;
    const baseRight = 310;
    const totalBase = baseLeft + baseRight;

    // Distribute ALL available space proportionally
    // We can cap it if it gets absurdly large, but "no empty space" implies filling.
    // Let's cap strictly to ensure it doesn't break UI internals (e.g. > 600px might be ugly)
    const scale = availableForPanels / totalBase;

    // Allow expansion up to a reasonable limit (e.g. 1.5x) or full fill?
    // User said "horizontal length ... fit the size of the board ... no empty unused space"
    // I will let it fill completely.

    let newLeft = Math.floor(baseLeft * scale);
    let newRight = Math.floor(baseRight * scale);

    // Safety check: don't go below 0
    newLeft = Math.max(0, newLeft);
    newRight = Math.max(0, newRight);

    // 5. Apply
    playPage.style.setProperty('--left-panel-w', `${newLeft}px`);
    playPage.style.setProperty('--right-panel-w', `${newRight}px`);

    const playGrid = document.querySelector('.play-grid');
    if (playGrid) {
        playGrid.style.setProperty('--left-panel-w', `${newLeft}px`);
        playGrid.style.setProperty('--right-panel-w', `${newRight}px`);
    }

    // Maintain vertical scroll class for potential other uses
    if (boardPanel.scrollHeight > boardPanel.clientHeight) {
        playPage.classList.add('has-vertical-scroll');
    } else {
        playPage.classList.remove('has-vertical-scroll');
    }
}

// Expose globally for app.js navigation and resize triggers
window.checkBoardOverflow = checkBoardOverflow;

// Initial Listener for Resize
window.addEventListener('resize', () => {
    if (window.checkBoardOverflow) checkBoardOverflow();
});

// Also observe panel resize
if (window.ResizeObserver) {
    const resizeObserver = new ResizeObserver(entries => {
        if (window.checkBoardOverflow) checkBoardOverflow();
    });
    // Wait for DOM
    setTimeout(() => {
        const bp = document.querySelector('.board-panel');
        if (bp) resizeObserver.observe(bp);
    }, 1000);
}

// Letter values for "Valued Letters" format
const LETTER_VALUES = {
    'A': 2, 'B': 4, 'C': 4, 'D': 3, 'E': 1, 'F': 5, 'G': 3, 'H': 5, 'I': 2, 'J': 10, 'K': 6, 'L': 3, 'M': 4, 'N': 2, 'O': 2, 'P': 4, 'Q': 10, 'R': 2, 'S': 2, 'T': 2, 'U': 4, 'V': 5, 'W': 5, 'X': 10, 'Y': 5, 'Z': 10
};

function updateBoardCell(cell, r, c, letter, grayed, f, state = null) {
    if (!cell) return;
    
    // Update basic classes
    cell.className = 'board-cell' + (grayed ? ' grayed' : '');
    
    // Update dataset for identification
    cell.dataset.r = r;
    cell.dataset.c = c;
    if (typeof f !== 'undefined') cell.dataset.f = f;

    // Check if letter OR format changed (to ensure point badges are cleared/added correctly)
    const currentLetter = cell.dataset.letter;
    const currentFormat = cell.dataset.renderedFormat;
    const boardFormat = (state && state.current_board_format) ? state.current_board_format : ((window.lastGameState && window.lastGameState.current_board_format) ? window.lastGameState.current_board_format : 'Normal');
    
    if (currentLetter !== letter || currentFormat !== boardFormat || cell.children.length <= 1) {
        cell.dataset.letter = letter;
        cell.dataset.renderedFormat = boardFormat;
        cell.innerHTML = ''; // Fresh start
        
        if (letter.includes('/')) {
            cell.classList.add('dual-letter');
            const [top, bottom] = letter.split('/');
            const container = document.createElement('div');
            container.className = 'dual-letter-container';
            const topEl = document.createElement('span');
            topEl.textContent = (top === 'Q' ? 'QU' : top);
            container.appendChild(topEl);
            const divider = document.createElement('div');
            divider.className = 'dual-divider';
            container.appendChild(divider);
            const bottomEl = document.createElement('span');
            bottomEl.textContent = (bottom === 'Q' ? 'QU' : bottom);
            container.appendChild(bottomEl);
            cell.appendChild(container);
        } else {
            const letterSpan = document.createElement('span');
            letterSpan.textContent = letter === 'Q' ? 'QU' : letter;
            cell.appendChild(letterSpan);
        }
        
        // Valued Letters support (Points in corner)
        if (boardFormat.toLowerCase().includes('valued') && !letter.includes('/')) {
            const valSpan = document.createElement('span');
            valSpan.className = 'tile-value';
            valSpan.textContent = LETTER_VALUES[letter.toUpperCase()] || 1;
            cell.appendChild(valSpan);
        }

        // Re-inject hitbox
        const hitbox = document.createElement('div');
        hitbox.className = 'cell-hitbox';
        cell.appendChild(hitbox);
        
        // USER REQUEST: If this is the bonus cell, add a persistent star icon to make it "Appear"
        const bonusCell = (state && state.bonus_cell) ? state.bonus_cell : (window.lastGameState ? window.lastGameState.bonus_cell : null);
        let isBonusMatch = false;
        if (bonusCell) {
            if (Array.isArray(bonusCell)) {
                if (bonusCell.length === 3) {
                    if (typeof f !== 'undefined' && Number(bonusCell[0]) === f && Number(bonusCell[1]) === r && Number(bonusCell[2]) === c) isBonusMatch = true;
                } else if (bonusCell.length === 2) {
                    if (Number(bonusCell[0]) === r && Number(bonusCell[1]) === c) isBonusMatch = true;
                }
            } else if (typeof bonusCell === 'object') {
                if (bonusCell.f !== undefined) {
                    if (typeof f !== 'undefined' && Number(bonusCell.f) === f && Number(bonusCell.r) === r && Number(bonusCell.c) === c) isBonusMatch = true;
                } else if (bonusCell.r !== undefined) {
                    if (Number(bonusCell.r) === r && Number(bonusCell.c) === c) isBonusMatch = true;
                }
            }
        }

        // Tooltip suppression fix
        cell.removeAttribute('title');
        hitbox.removeAttribute('title');
    }

    // Update Special Highlights (Bonus Cell)
    const activeBonusCell = (state && state.bonus_cell) ? state.bonus_cell : (window.lastGameState ? window.lastGameState.bonus_cell : null);
    let isMatch = false;
    
    if (activeBonusCell) {
        if (Array.isArray(activeBonusCell)) {
            if (activeBonusCell.length === 3) {
                if (typeof f !== 'undefined' && Number(activeBonusCell[0]) === f && Number(activeBonusCell[1]) === r && Number(activeBonusCell[2]) === c) {
                    isMatch = true;
                }
            } else if (activeBonusCell.length === 2) {
                if (Number(activeBonusCell[0]) === r && Number(activeBonusCell[1]) === c) {
                    isMatch = true;
                }
            }
        } else if (typeof activeBonusCell === 'object') {
            if (activeBonusCell.f !== undefined) {
                if (typeof f !== 'undefined' && Number(activeBonusCell.f) === f && Number(activeBonusCell.r) === r && Number(activeBonusCell.c) === c) isMatch = true;
            } else if (activeBonusCell.r !== undefined) {
                if (Number(activeBonusCell.r) === r && Number(activeBonusCell.c) === c) isMatch = true;
            }
        }
    }

    // STAR MANAGEMENT: Ensure star is present/absent based on live isMatch
    const existingStar = cell.querySelector('.bonus-star');
    if (isMatch && !existingStar) {
        const star = document.createElement('span');
        star.className = 'bonus-star';
        star.textContent = '★';
        cell.appendChild(star);
    } else if (!isMatch && existingStar) {
        existingStar.remove();
    }
    
    if (isMatch || (boardFormat.toLowerCase().includes('either') && letter.includes('/'))) {
        cell.classList.add('bonus-highlight');
    } else {
        cell.classList.remove('bonus-highlight');
    }

    // Apply Density (This is the DYNAMIC part!)
    applyDensityToCell(cell, r, c, f, state);
}

function applyDensityToCell(cell, r, c, f, state = null) {
    const densityData = (state && state.cell_density) ? state.cell_density : (window.lastGameState && window.lastGameState.cell_density);
    const hasDensityData = densityData && Array.isArray(densityData) && densityData.length > 0;
    const boardFormat = (state && state.current_board_format) ? state.current_board_format : (window.lastGameState && window.lastGameState.current_board_format) || 'Normal';
    
    // USER REQUEST: If this is the bonus cell, do NOT apply density shading to the background.
    // The green .bonus-highlight must take absolute precedence.
    if (cell.classList.contains('bonus-highlight')) {
        cell.style.background = '';
        cell.style.backgroundColor = '';
        cell.style.color = '';
        cell.style.boxShadow = '';
        return;
    }

    if (hasDensityData && boardFormat.toLowerCase().includes('density')) {
         const grid = densityData;
         const maxD = Math.max(1, window.lastGameState.max_cell_density || 1);
         let cur = -1;
         
         if (typeof f !== 'undefined' && grid[f] && grid[f][r]) {
             cur = grid[f][r][c];
         } else if (grid[r]) {
             cur = grid[r][c];
         }
         
         if (cur === 0) {
             cell.style.setProperty('background', '#ffffff', 'important');
             cell.style.setProperty('background-color', '#ffffff', 'important');
             cell.style.setProperty('color', '#000000', 'important');
             cell.style.setProperty('box-shadow', 'inset 0 0 10px rgba(0,0,0,0.1)', 'important');
         } else if (cur > 0) {
             // HIGH CONTRAST GRAYSCALE: Linear ratio matching 3D implementation
             const ratio = Math.max(0, Math.min(1, cur / maxD));
             const grayLightness = Math.round(100 - (ratio * 100)); 
             const hslColor = `hsl(0, 0%, ${grayLightness}%)`;
             
             // Trigger animation if density changed
             const lastD = cell.dataset.lastD;
             if (lastD !== undefined && Number(lastD) !== cur) {
                 cell.classList.remove('cell-lightened');
                 void cell.offsetWidth; // Trigger reflow
                 cell.classList.add('cell-lightened');
             }
             cell.dataset.lastD = cur;

             cell.style.setProperty('background', hslColor, 'important');
             cell.style.setProperty('background-color', hslColor, 'important');
             cell.style.setProperty('transition', 'background 0.4s ease, color 0.4s ease', 'important');
             
             const textColor = (grayLightness < 50) ? '#ffffff' : '#000000';
             cell.style.setProperty('color', textColor, 'important');
         }
    } else if (boardFormat.toLowerCase().includes('density')) {
        // We have no density data BUT we are in density mode. 
        // Do NOT clear the style yet - wait for a state that has it.
        // This prevents the "brief flickers" during heartbeats.
    } else {
        // Reset ONLY if we are truly no longer in density format
        cell.style.background = '';
        cell.style.backgroundColor = '';
        cell.style.color = '';
        cell.style.boxShadow = '';
        delete cell.dataset.lastD;
    }
}

function createBoardCell(r, c, letter, grayed, f, state = null) {
    const cell = document.createElement('div');
    cell.dataset.letter = letter;
    updateBoardCell(cell, r, c, letter, grayed, f, state);
    return cell;
}

/**
 * Finds if a word can be formed on the board and returns the path of coordinates.
 * Supports the "Q" tile representing "QU".
 */
function findWordPathOnBoard(word, board, targetCoord = null) {
    if (!word || !board) return null;
    const rows = board.length;
    if (rows === 0) return null;
    const cols = board[0].length;
    const upperWord = word.toUpperCase();

    function dfs(r, c, index, currentPath, visited, hasHitTarget) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) return null;
        if (visited.has(`${r},${c}`)) return null;

        const cellValue = board[r][c].toUpperCase();
        const letters = cellValue.includes('/') ? cellValue.split('/') : [cellValue];
        let foundMatch = false;
        let matchLength = 0;

        for (const char of letters) {
            if (char === 'Q') {
                if (upperWord.substring(index, index + 2) === 'QU') {
                    matchLength = 2;
                    foundMatch = true;
                    break;
                } else if (upperWord[index] === 'Q') {
                    matchLength = 1;
                    foundMatch = true;
                    break;
                }
            } else if (upperWord[index] === char) {
                matchLength = 1;
                foundMatch = true;
                break;
            }
        }

        if (!foundMatch) return null;

        const newVisited = new Set(visited);
        newVisited.add(`${r},${c}`);
        const newPath = [...currentPath, { r, c }];
        
        // Track hit
        let nowHit = hasHitTarget;
        if (targetCoord) {
            if (Array.isArray(targetCoord)) {
                if (r === targetCoord[0] && c === targetCoord[1]) nowHit = true;
            } else if (typeof targetCoord === 'object') {
                if (r === targetCoord.r && c === targetCoord.c) nowHit = true;
            }
        }

        const nextIndex = index + matchLength;
        if (nextIndex >= upperWord.length) {
            // If we are searching for a specific target cell, enforce the hit.
            // Otherwise (standard typing), return the path.
            if (targetCoord && !nowHit) return null;
            return newPath;
        }

        // Try directions
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const result = dfs(r + dr, c + dc, nextIndex, newPath, newVisited, nowHit);
                if (result) return result;
            }
        }
        return null;
    }

    // Try starting from all possible cells to find a path that hits the target
    if (targetCoord) {
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const path = dfs(r, c, 0, [], new Set(), false);
                if (path) return path;
            }
        }
    }
    
    // If no bonus-hitting path found or no target, return any valid path
    function dfsBasic(r, c, index, currentPath, visited) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) return null;
        if (visited.has(`${r},${c}`)) return null;
        const cellValue = board[r][c].toUpperCase();
        const letters = cellValue.includes('/') ? cellValue.split('/') : [cellValue];
        let foundMatch = false;
        let matchLength = 0;
        for (const char of letters) {
            if (char === 'Q') {
                if (upperWord.substring(index, index + 2) === 'QU') { matchLength = 2; foundMatch = true; break; }
                else if (upperWord[index] === 'Q') { matchLength = 1; foundMatch = true; break; }
            } else if (upperWord[index] === char) { matchLength = 1; foundMatch = true; break; }
        }
        if (!foundMatch) return null;
        const newVisited = new Set(visited);
        newVisited.add(`${r},${c}`);
        const newPath = [...currentPath, { r, c }];
        const nextIndex = index + matchLength;
        if (nextIndex >= upperWord.length) return newPath;
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const result = dfsBasic(r + dr, c + dc, nextIndex, newPath, newVisited);
                if (result) return result;
            }
        }
        return null;
    }
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const result = dfsBasic(r, c, 0, [], new Set());
            if (result) return result;
        }
    }
    return null;
}

window.findWordPathOnCube = function(word, board) {
    if (!word || !board || board.length !== 6) return null;
    const upperWord = word.toUpperCase();

    function getCubeNeighbors(f, r, c) {
        const res = [];
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const nr = r + dr, nc = c + dc;
                if (nr >= 0 && nr < 3 && nc >= 0 && nc < 3) res.push({ f, r: nr, c: nc });
            }
        }
        // Simplifiedインターフェース neighbors (matching board_generator.py logic)
        if (f === 0) {
            if (r === 0) res.push({ f: 4, r: 2, c }, { f: 4, r: 2, c: c - 1 }, { f: 4, r: 2, c: c + 1 });
            if (r === 2) res.push({ f: 5, r: 0, c }, { f: 5, r: 0, c: c - 1 }, { f: 5, r: 0, c: c + 1 });
            if (c === 0) res.push({ f: 2, r, c: 2 }, { f: 2, r: r - 1, c: 2 }, { f: 2, r: r + 1, c: 2 });
            if (c === 2) res.push({ f: 3, r, c: 0 }, { f: 3, r: r - 1, c: 0 }, { f: 3, r: r + 1, c: 0 });
        } else if (f === 1) {
            if (r === 0) res.push({ f: 4, r: 0, c: 2 - c }, { f: 4, r: 0, c: 2 - (c - 1) }, { f: 4, r: 0, c: 2 - (c + 1) });
            if (r === 2) res.push({ f: 5, r: 2, c: 2 - c }, { f: 5, r: 2, c: 2 - (c - 1) }, { f: 5, r: 2, c: 2 - (c + 1) });
            if (c === 0) res.push({ f: 3, r, c: 2 }, { f: 3, r: r - 1, c: 2 }, { f: 3, r: r + 1, c: 2 });
            if (c === 2) res.push({ f: 2, r, c: 0 }, { f: 2, r: r - 1, c: 0 }, { f: 2, r: r + 1, c: 0 });
        } else if (f === 2) {
            if (r === 0) res.push({ f: 4, r: c, c: 0 }, { f: 4, r: c - 1, c: 0 }, { f: 4, r: c + 1, c: 0 });
            if (r === 2) res.push({ f: 5, r: 2 - c, c: 0 }, { f: 5, r: 2 - (c - 1), c: 0 }, { f: 5, r: 2 - (c + 1), c: 0 });
            if (c === 0) res.push({ f: 1, r, c: 2 }, { f: 1, r: r - 1, c: 2 }, { f: 1, r: r + 1, c: 2 });
            if (c === 2) res.push({ f: 0, r, c: 0 }, { f: 0, r: r - 1, c: 0 }, { f: 0, r: r + 1, c: 0 });
        } else if (f === 3) {
            if (r === 0) res.push({ f: 4, r: 2 - c, c: 2 }, { f: 4, r: 2 - (c - 1), c: 2 }, { f: 4, r: 2 - (c + 1), c: 2 });
            if (r === 2) res.push({ f: 5, r: c, c: 2 }, { f: 5, r: c - 1, c: 2 }, { f: 5, r: c + 1, c: 2 });
            if (c === 0) res.push({ f: 0, r, c: 2 }, { f: 0, r: r - 1, c: 2 }, { f: 0, r: r + 1, c: 2 });
            if (c === 2) res.push({ f: 1, r, c: 0 }, { f: 1, r: r - 1, c: 0 }, { f: 1, r: r + 1, c: 0 });
        } else if (f === 4) {
            if (r === 0) res.push({ f: 1, r: 0, c: 2 - c }, { f: 1, r: 0, c: 2 - (c - 1) }, { f: 1, r: 0, c: 2 - (c + 1) });
            if (r === 2) res.push({ f: 0, r: 0, c }, { f: 0, r: 0, c: c - 1 }, { f: 0, r: 0, c: c + 1 });
            if (c === 0) res.push({ f: 2, r: 0, c: r }, { f: 2, r: 0, c: r - 1 }, { f: 2, r: 0, c: r + 1 });
            if (c === 2) res.push({ f: 3, r: 0, c: 2 - r }, { f: 3, r: 0, c: 2 - (r - 1) }, { f: 3, r: 0, c: 2 - (r + 1) });
        } else if (f === 5) {
            if (r === 0) res.push({ f: 0, r: 2, c }, { f: 0, r: 2, c: c - 1 }, { f: 0, r: 2, c: c + 1 });
            if (r === 2) res.push({ f: 1, r: 2, c: 2 - c }, { f: 1, r: 2, c: 2 - (c - 1) }, { f: 1, r: 2, c: 2 - (c + 1) });
            if (c === 0) res.push({ f: 2, r: 2, c: 2 - r }, { f: 2, r: 2, c: 2 - (r - 1) }, { f: 2, r: 2, c: 2 - (r + 1) });
            if (c === 2) res.push({ f: 3, r: 2, c: r }, { f: 3, r: 2, c: r - 1 }, { f: 3, r: 2, c: r + 1 });
        }
        return res.filter(n => n.f >= 0 && n.f < 6 && n.r >= 0 && n.r < 3 && n.c >= 0 && n.c < 3);
    }

    function dfs(f, r, c, index, currentPath, visited) {
        if (index >= upperWord.length) return currentPath;
        if (visited.has(`${f},${r},${c}`)) return null;

        const cellValue = board[f][r][c].toUpperCase();
        let matchLength = 0;
        if (cellValue === 'Q') {
            if (upperWord.substring(index, index + 2) === 'QU') matchLength = 2;
            else if (upperWord[index] === 'Q') matchLength = 1;
        } else if (upperWord[index] === cellValue) {
            matchLength = 1;
        }

        if (matchLength === 0) return null;

        const newVisited = new Set(visited);
        newVisited.add(`${f},${r},${c}`);
        const newPath = [...currentPath, { f, r, c }];

        const nextIndex = index + matchLength;
        if (nextIndex >= upperWord.length) return newPath;

        for (const n of getCubeNeighbors(f, r, c)) {
            const result = dfs(n.f, n.r, n.c, nextIndex, newPath, newVisited);
            if (result) return result;
        }
        return null;
    }

    for (let f = 0; f < 6; f++) {
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                const path = dfs(f, r, c, 0, [], new Set());
                if (path) return path;
            }
        }
    }
    return null;
}

/**
 * Reapplies visual highlights (typing and mouse selection) to the board.
 * Useful after the board DOM has been rebuilt.
 */
function reapplyBoardHighlights() {
    const board = window.lastGameState ? window.lastGameState.board : null;
    if (!board) return;

    // Clear PREVIOUS highlights of ALL types to avoid stale visuals
    document.querySelectorAll('.board-cell').forEach(cell => {
        cell.classList.remove('selected', 'current', 'typing-highlight', 'review-highlight');
    });

    // 1. Reapply mouse selection highlights (drag)
    if (mouseState && mouseState.selectedPath && mouseState.selectedPath.length > 0) {
        const isEnabled = !(window.userSettings && window.userSettings.highlight_mouse === false);
        if (isEnabled) {
            mouseState.selectedPath.forEach((p, index) => {
                const isCurrent = (index === mouseState.selectedPath.length - 1);
                let selector = `.board-cell[data-r="${p.row}"][data-c="${p.col}"]`;
                if (p.face !== undefined && p.face !== null) {
                    selector = `.board-cell[data-f="${p.face}"][data-r="${p.row}"][data-c="${p.col}"]`;
                }
                const cell = document.querySelector(selector);
                if (cell) {
                    cell.classList.add('selected');
                    if (isCurrent) {
                        cell.classList.add('current');
                    }
                }
            });
        }
    }

    // 2. Reapply typing highlights (input box) - SKIP IF MOUSING to avoid "double-highlighting" the board
    const wordInputEl = document.getElementById('word-input');
    const is3D = board.length === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]);

    if (wordInputEl && wordInputEl.value.trim() && !(mouseState && mouseState.isDown)) {
        const isEnabled = window.userSettings && window.userSettings.highlight_typing !== false;
        if (isEnabled) {
            const word = wordInputEl.value.trim();
            const path = is3D ? findWordPathOnCube(word, board) : findWordPathOnBoard(word, board);
            if (path) {
                path.forEach(coord => {
                    let selector = `.board-cell[data-r="${coord.r}"][data-c="${coord.c}"]`;
                    if (coord.f !== undefined) selector = `.board-cell[data-f="${coord.f}"][data-r="${coord.r}"][data-c="${coord.c}"]`;
                    const cell = document.querySelector(selector);
                    if (cell) cell.classList.add('typing-highlight');
                });
            }
        }
    }

    // 3. Reapply review highlights (All Words / Finder list)
    if (typeof highlightedFoundWord !== 'undefined' && highlightedFoundWord) {
        const path = is3D ? findWordPathOnCube(highlightedFoundWord, board) : findWordPathOnBoard(highlightedFoundWord, board);
        if (path) {
            // Check if we need to animate (new selection) or just show (board refresh)
            const isNewSelection = window._lastAnimatedReviewWord !== highlightedFoundWord;
            
            // CAPTURE the current word for the closure to avoid race conditions with window._lastAnimatedReviewWord
            const currentHighlightedWord = highlightedFoundWord;

            path.forEach((coord, index) => {
                let selector = `.board-cell[data-r="${coord.r}"][data-c="${coord.c}"]`;
                if (coord.f !== undefined) selector = `.board-cell[data-f="${coord.f}"][data-r="${coord.r}"][data-c="${coord.c}"]`;
                const cell = document.querySelector(selector);
                if (cell) {
                    if (isNewSelection) {
                        setTimeout(() => {
                            if (highlightedFoundWord === currentHighlightedWord) {
                                cell.classList.add('review-highlight');
                            }
                        }, index * 60);
                    } else {
                        cell.classList.add('review-highlight');
                    }
                }
            });
            window._lastAnimatedReviewWord = highlightedFoundWord;
        }
    } else {
        window._lastAnimatedReviewWord = null;
    }

    // 4. Reapply density shading (Density format) — must run after every highlight pass
    // because density inline styles can be overwritten by class-based highlight rules.
    const densityData = window.lastGameState && window.lastGameState.cell_density;
    const bFormat = (window.lastGameState && window.lastGameState.current_board_format) || '';
    if (densityData && Array.isArray(densityData) && densityData.length > 0 && bFormat.toLowerCase().includes('density')) {
        document.querySelectorAll('.board-cell').forEach(cell => {
            const r = parseInt(cell.dataset.r);
            const c = parseInt(cell.dataset.c);
            const f = cell.dataset.f !== undefined ? parseInt(cell.dataset.f) : undefined;
            applyDensityToCell(cell, r, c, f);
        });
    }
}

function renderSplitNotepads(players, state) {
    const boardEl = document.getElementById('game-board');
    if (!boardEl) return;

    // Capture scroll positions before clearing
    const scrollMap = {};
    document.querySelectorAll('.user-notepad').forEach(el => {
        const username = el.dataset.username;
        const list = el.querySelector('.notepad-list');
        if (username && list) {
            scrollMap[username] = list.scrollTop;
        }
    });

    // Use grid layout for notepads? Or flex?
    // User requested: "Each user has its own square, resembling a notepad... scrollable"
    // "highest scoring user listed first"

    // Sort players by score
    const sortedPlayers = [...players].sort((a, b) => b.score - a.score);

    // Set container style for notepads
    boardEl.className = 'split-notepads-container';
    // Clear inline grid styles from renderBoard
    boardEl.style.gridTemplateColumns = '';
    boardEl.style.gridTemplateRows = '';
    boardEl.innerHTML = '';

    sortedPlayers.forEach(p => {
        // Init state if needed
        if (!splitNotepadState[p.username]) {
            splitNotepadState[p.username] = 'unique'; // Default tab
        }
        const currentTab = splitNotepadState[p.username];

        const notepad = document.createElement('div');
        notepad.className = 'user-notepad';
        notepad.dataset.username = p.username; // Store username for scroll tracking

        // highlight selected user
        if (p.username === selectedPlayerUsername) {
            notepad.classList.add('selected');
        }

        // Shared word highlighting logic
        if (highlightedSplitWord) {
            const hWord = highlightedSplitWord.trim().toUpperCase();
            const hasHighlightedWord = p.submitted_words && p.submitted_words.some(sw => {
                const wStr = (typeof sw === 'object' ? sw.word : sw) || '';
                return wStr.trim().toUpperCase() === hWord;
            });

            if (hasHighlightedWord) {
                notepad.classList.add('highlight-shared');
            }
        }

        // Add click to select user (for highlighting words on right panel)
        notepad.onclick = (e) => {
            // Avoid triggering if clicking tabs or items
            if (e.target.classList.contains('notepad-tab') || e.target.closest('.notepad-item')) return;

            if (selectedPlayerUsername === p.username) {
                selectedPlayerUsername = null;
            } else {
                selectedPlayerUsername = p.username;
            }
            updateGameState();
        };

        // Header
        const header = document.createElement('div');
        header.className = 'notepad-header';
        header.innerHTML = `<strong>${p.username}</strong> <span>${(p.score === 0 && (!p.submitted_words || p.submitted_words.length === 0) && (state && state.state === 'intermission')) ? 'DNP' : p.score + ' pts'}</span>`;
        notepad.appendChild(header);

        // Tabs
        const tabs = document.createElement('div');
        tabs.className = 'notepad-tabs';

        ['unique', 'split', 'invalid'].forEach(tabName => {
            const btn = document.createElement('button');
            btn.className = `notepad-tab ${currentTab === tabName ? 'active' : ''}`;
            btn.textContent = tabName.charAt(0).toUpperCase() + tabName.slice(1);
            btn.onclick = (e) => {
                e.stopPropagation(); // Prevent notepad selection logic
                splitNotepadState[p.username] = tabName;
                updateGameState(); // Re-render
            };
            tabs.appendChild(btn);
        });
        notepad.appendChild(tabs);

        // Content area with scroll buttons
        const mainContent = document.createElement('div');
        mainContent.className = 'notepad-main-content';

        // List
        const list = document.createElement('div');
        list.className = 'notepad-list';

        // Scroll Controls
        const scrollControls = document.createElement('div');
        scrollControls.className = 'notepad-scroll-controls';

        const btnUp = document.createElement('button');
        btnUp.className = 'notepad-scroll-btn up';
        btnUp.innerHTML = '▲';
        btnUp.onclick = (e) => {
            e.stopPropagation();
            const firstItem = list.querySelector('.notepad-item');
            const rowHeight = firstItem ? firstItem.offsetHeight + 4 : 29; // height + 4px gap
            list.scrollBy({ top: -rowHeight, behavior: 'smooth' });
        };

        const btnDown = document.createElement('button');
        btnDown.className = 'notepad-scroll-btn down';
        btnDown.innerHTML = '▼';
        btnDown.onclick = (e) => {
            e.stopPropagation();
            const firstItem = list.querySelector('.notepad-item');
            const rowHeight = firstItem ? firstItem.offsetHeight + 4 : 29; // height + 4px gap
            list.scrollBy({ top: rowHeight, behavior: 'smooth' });
        };

        scrollControls.appendChild(btnUp);
        scrollControls.appendChild(btnDown);

        // ... wordsToShow loop logic follows ...
        // (I will include the wordsToShow logic in the next chunk or merge)


        // Filter words based on tab
        let wordsToShow = [];
        if (currentTab === 'invalid') {
            // Use invalid_words from backend
            if (p.invalid_words && p.invalid_words.length > 0) {
                p.invalid_words.forEach(w => {
                    wordsToShow.push({ word: w, points: 0, is_invalid: true });
                });
            }
        } else {
            // Process valid words
            if (p.submitted_words) {
                p.submitted_words.forEach(w => {
                    const isUnique = w.is_unique;
                    if (currentTab === 'unique' && isUnique) {
                        wordsToShow.push(w);
                    } else if (currentTab === 'split' && !isUnique) {
                        wordsToShow.push(w);
                    }
                });

                // Sort words: Longest first, then alphabetical
                wordsToShow.sort((a, b) => {
                    const wordA = a.word || '';
                    const wordB = b.word || '';
                    if (wordB.length !== wordA.length) {
                        return wordB.length - wordA.length;
                    }
                    return wordA.localeCompare(wordB);
                });
            }
        }

        // Render words
        if (wordsToShow.length === 0) {
            list.innerHTML = '<div style="color:var(--text-primary);font-style:italic;padding:10px;text-align:center;">None</div>';
        } else {
            wordsToShow.forEach(w => {
                const row = document.createElement('div');
                row.className = 'notepad-item';
                if (w.is_invalid) row.classList.add('invalid');

                // Active highlighting for the word itself
                if (highlightedSplitWord) {
                    if (w.word.trim().toUpperCase() === highlightedSplitWord.trim().toUpperCase()) {
                        row.classList.add('active');
                    }
                }

                // Definition handler
                row.dataset.word = w.word;
                row.style.cursor = 'pointer';
                row.onclick = (e) => {
                    e.stopPropagation();
                    // Split highlighting logic
                    if (currentTab === 'split') {
                        const clickedWord = w.word.trim().toUpperCase();
                        if (highlightedSplitWord && highlightedSplitWord.trim().toUpperCase() === clickedWord) {
                            highlightedSplitWord = null;
                        } else {
                            highlightedSplitWord = clickedWord;
                        }
                        updateGameState();
                    }
                    window.fetchDefinition(w.word);
                };

                let ptsDisplay = w.points;
                if (w.score_details) {
                    const bonusWordPts = w.score_details.bonus_word_points || 0;
                    const bonusLetterPts = w.score_details.bonus_letter_points || 0;
                    
                    if (bonusLetterPts > 0 || (w.score_details.either_or_points || 0) > 0) {
                        const extra = bonusLetterPts + (w.score_details.either_or_points || 0);
                        const originalValue = (w.score_details.base || 0) + bonusWordPts;
                        ptsDisplay = `${originalValue} + ${extra} = ${w.points}`;
                    } else if (bonusWordPts > 0) {
                        ptsDisplay = `${(w.score_details.base || 0)} + ${bonusWordPts} = ${w.points}`;
                    }
                }
                
                // Add split multiplier indicator for shared words
                if (w.shared_count > 1) {
                    ptsDisplay += ` <small style="opacity:0.7;">(÷${w.shared_count})</small>`;
                }

                const isBonusWord = w.is_bonus || (state.bonus_word && w.word.toUpperCase() === state.bonus_word.toUpperCase());
                if (isBonusWord) row.classList.add('bonus-word');

                row.innerHTML = `<span>${w.word}</span> <span style="font-size:0.85em; opacity:0.9;">${ptsDisplay}</span>`;
                list.appendChild(row);
            });
        }

        mainContent.appendChild(list);
        mainContent.appendChild(scrollControls);
        notepad.appendChild(mainContent);
        boardEl.appendChild(notepad);

        // Restore scroll position
        if (scrollMap[p.username]) {
            list.scrollTop = scrollMap[p.username];
        }
    });
}

function renderFCFSNotepads(players, state) {
    const boardEl = document.getElementById('game-board');
    if (!boardEl) return;

    // Capture scroll positions
    const scrollMap = {};
    document.querySelectorAll('.user-notepad').forEach(el => {
        const username = el.dataset.username;
        const list = el.querySelector('.notepad-list');
        if (username && list) {
            scrollMap[username] = list.scrollTop;
        }
    });

    // Sort players by score
    const sortedPlayers = [...players].sort((a, b) => b.score - a.score);

    // Reuse Split Points container styles
    boardEl.className = 'split-notepads-container';
    // Clear inline grid styles from renderBoard
    boardEl.style.gridTemplateColumns = '';
    boardEl.style.gridTemplateRows = '';
    boardEl.innerHTML = '';

    sortedPlayers.forEach(p => {
        const notepad = document.createElement('div');
        notepad.className = 'user-notepad';
        notepad.dataset.username = p.username;

        if (p.username === selectedPlayerUsername) {
            notepad.classList.add('selected');
        }

        notepad.onclick = (e) => {
            if (selectedPlayerUsername === p.username) {
                selectedPlayerUsername = null;
            } else {
                selectedPlayerUsername = p.username;
            }
            updateGameState();
        };

        // Header
        const header = document.createElement('div');
        header.className = 'notepad-header';
        header.innerHTML = `<strong>${p.username}</strong> <span>${(p.score === 0 && (!p.submitted_words || p.submitted_words.length === 0) && (state && state.state === 'intermission')) ? 'DNP' : p.score + ' pts'}</span>`;
        notepad.appendChild(header);

        // No Tabs for FCFS

        // Content area with scroll buttons
        const mainContent = document.createElement('div');
        mainContent.className = 'notepad-main-content';

        // List
        const list = document.createElement('div');
        list.className = 'notepad-list';
        list.style.marginTop = '10px';
        list.style.height = '100%'; // Fill parent

        // Scroll Controls
        const scrollControls = document.createElement('div');
        scrollControls.className = 'notepad-scroll-controls';

        const btnUp = document.createElement('button');
        btnUp.className = 'notepad-scroll-btn up';
        btnUp.innerHTML = '▲';
        btnUp.onclick = (e) => {
            e.stopPropagation();
            const firstItem = list.querySelector('.notepad-item');
            const rowHeight = firstItem ? firstItem.offsetHeight + 4 : 29; // height + 4px gap
            list.scrollBy({ top: -rowHeight, behavior: 'smooth' });
        };

        const btnDown = document.createElement('button');
        btnDown.className = 'notepad-scroll-btn down';
        btnDown.innerHTML = '▼';
        btnDown.onclick = (e) => {
            e.stopPropagation();
            const firstItem = list.querySelector('.notepad-item');
            const rowHeight = firstItem ? firstItem.offsetHeight + 4 : 29; // height + 4px gap
            list.scrollBy({ top: rowHeight, behavior: 'smooth' });
        };

        scrollControls.appendChild(btnUp);
        scrollControls.appendChild(btnDown);

        if (!p.submitted_words || p.submitted_words.length === 0) {
            list.innerHTML = '<div style="color:var(--text-primary);font-style:italic;padding:10px;text-align:center;">None</div>';
        } else {
            // Sort words: Longest first, then alphabetical
            const wordsToShow = [...p.submitted_words];
            wordsToShow.sort((a, b) => {
                // Handle objects vs strings
                const wordA = (typeof a === 'string' ? a : a.word) || '';
                const wordB = (typeof b === 'string' ? b : b.word) || '';
                if (wordB.length !== wordA.length) {
                    return wordB.length - wordA.length;
                }
                return wordA.localeCompare(wordB);
            });

            wordsToShow.forEach(wObj => {
                const w = (typeof wObj === 'string' ? wObj : wObj.word);
                let ptsNum = (typeof wObj === 'string' ? 0 : wObj.points);
                let ptsDisplay = (typeof wObj === 'string' ? '?' : wObj.points);

                if (wObj.score_details && (wObj.score_details.bonus_letter_points > 0 || (wObj.score_details.either_or_points || 0) > 0)) {
                    const extra = (wObj.score_details.bonus_letter_points || 0) + (wObj.score_details.either_or_points || 0);
                    const originalValue = (wObj.score_details.base || 0) + (wObj.score_details.bonus_word_points || 0);
                    ptsDisplay = `${originalValue} + ${extra} = ${wObj.points}`;
                } else if (wObj.score_details && (wObj.score_details.bonus_word_points || 0) > 0) {
                    ptsDisplay = `${(wObj.score_details.base || 0)} + ${wObj.score_details.bonus_word_points} = ${wObj.points}`;
                }

                const row = document.createElement('div'); // Define row here
                row.className = 'notepad-item'; // Assign class here
                if (ptsNum < 0) {
                    row.classList.add('penalty-word'); // Use classList.add
                    row.style.color = '#ff3333';
                    row.style.fontWeight = 'bold';
                }
                row.dataset.word = w;
                row.onclick = () => window.fetchDefinition(w); // Direct handler
                row.innerHTML = `<span>${w}</span> <span style="font-size:0.85em; opacity:0.9;">${ptsDisplay}</span>`;
                list.appendChild(row);
            });
        }

        mainContent.appendChild(list);
        mainContent.appendChild(scrollControls);
        notepad.appendChild(mainContent);
        boardEl.appendChild(notepad);


        // Restore scroll
        if (scrollMap[p.username]) {
            list.scrollTop = scrollMap[p.username];
        }
    });
}


function addSplitViewBoardToggle() {
    const panelHeader = document.querySelector('.words-panel h3');
    if (!panelHeader) return;

    // Check if button already exists
    if (document.getElementById('toggle-board-btn')) return;

    const btn = document.createElement('button');
    btn.id = 'toggle-board-btn';
    btn.textContent = showBoardInSplitIntermission ? 'Show Notepads' : 'Show Board';
    btn.className = 'active-room-btn'; // Re-use a style class
    btn.style.fontSize = '0.7rem';
    btn.style.marginLeft = '10px';
    btn.style.padding = '2px 8px';

    btn.onclick = () => {
        showBoardInSplitIntermission = !showBoardInSplitIntermission;
        updateGameState();
    };

    panelHeader.appendChild(btn);
}

// Spinner Logic
// Word Submission
// Initialize Word Submission Listeners
function initWordSubmission() {
    const submitBtn = document.getElementById('submit-word-btn');
    const wordInputEl = document.getElementById('word-input');
    
    if (!wordInputEl) {
        console.warn('[play.js] word-input element not found during initWordSubmission');
        return;
    }

    if (submitBtn) {
        submitBtn.onclick = () => {
            const val = wordInputEl.value;
            console.log('[play.js] submission triggered via button click:', val);
            submitWord(val);
        };
    }

// DELEGATED GLOBAL LISTENER for Enter-key submission
// This ensures it works even if the input element is re-created or swapped.
document.addEventListener('keydown', (e) => {
    if (e.target && e.target.id === 'word-input') {
        if (e.key === 'Enter' || e.key === 'Return') {
            e.preventDefault();
            const wordToSubmit = e.target.value;
            console.log('[play.js] delegated submission triggered:', wordToSubmit);
            submitWord(wordToSubmit);
        }
    }
});

    // Real-time highlighting and "word declaration" while typing
    let typingHighlightTimeout = null;

    wordInputEl.addEventListener('input', () => {
        const word = wordInputEl.value.trim();
        
        // UX: If round ended and we just cleared the input manually, refocus chat
        if (window.refocusChatPending && word.length === 0 && (!mouseState || !mouseState.isDown)) {
            window.refocusChatPending = false;
            const chatInput = document.getElementById('chat-input');
            if (chatInput) setTimeout(() => chatInput.focus(), 150);
        }

        // Fast path: if empty or too short, clear highlight instantly and cancel any pending search
        if (!word || word.length < 3) {
            if (typingHighlightTimeout) {
                clearTimeout(typingHighlightTimeout);
                typingHighlightTimeout = null;
            }
            document.querySelectorAll('.board-cell.typing-highlight').forEach(c => c.classList.remove('typing-highlight'));
            return;
        }

        // Debounce pathfinding logic by 30ms so that typing characters renders with zero-latency
        if (typingHighlightTimeout) {
            clearTimeout(typingHighlightTimeout);
        }

        typingHighlightTimeout = setTimeout(() => {
            const board = window.lastGameState ? window.lastGameState.board : null;
            const isEnabled = window.userSettings && window.userSettings.highlight_typing !== false;
            if (!isEnabled) {
                document.querySelectorAll('.board-cell.typing-highlight').forEach(c => c.classList.remove('typing-highlight'));
                return;
            }

            document.querySelectorAll('.board-cell.typing-highlight').forEach(c => c.classList.remove('typing-highlight'));
            if (!board || (mouseState && mouseState.isDown)) return;

            const is3D = board.length === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]);
            const path = is3D ? findWordPathOnCube(word, board) : findWordPathOnBoard(word, board);
            if (path) {
                path.forEach(coord => {
                    let selector = `.board-cell[data-r="${coord.r}"][data-c="${coord.c}"]`;
                    if (coord.f !== undefined) selector = `.board-cell[data-f="${coord.f}"][data-r="${coord.r}"][data-c="${coord.c}"]`;
                    const cell = document.querySelector(selector);
                    if (cell) cell.classList.add('typing-highlight');
                });
            }
        }, 30);
    });
}
initWordSubmission();


async function submitWord(wordParam = null, pathParam = null) {
    const input = document.getElementById('word-input');
    const word = wordParam ? wordParam.toUpperCase() : (input ? input.value.trim().toUpperCase() : '');
    const roomId = getCurrentRoomId();
    
    // Visual Debug / Clear immediately
    if (input) {
        input.value = '';
        input.style.backgroundColor = 'rgba(255, 255, 0, 0.1)'; // Yellow tint for "pending"
        input.dispatchEvent(new Event('input'));
    }

    console.log('[play.js] submitWord entering:', word, 'Room:', roomId);
    if (!word) {
        console.warn('[play.js] Empty word submission ignored');
        return;
    }

    // 1. PATH RESOLUTION
    let finalPath = pathParam;
    const board = window.lastGameState ? window.lastGameState.board : null;
    if (!finalPath && word && board) {
        const is3D = board.length === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]);
        finalPath = (is3D && typeof findWordPathOnCube === 'function') 
            ? findWordPathOnCube(word, board) 
            : (typeof findWordPathOnBoard === 'function' ? findWordPathOnBoard(word, board) : null);
        if (finalPath) {
            finalPath = finalPath.map(p => {
                if (typeof p.f !== 'undefined') return [p.f, p.r, p.c];
                return [p.r, p.c];
            });
        }
    }

    // Define currentUser for consistency in local updates
    let currentUser = window.currentUser || (window.lastGameState && window.lastGameState.your_username) || localStorage.getItem('morpheme_username') || '';
    currentUser = currentUser.trim();

    console.log(`[play.js] Attempting submission: "${word}" (Path: ${finalPath ? 'Yes' : 'No'})`);

    if (isTournamentPlay) {
        handleTournamentWord(word);
        return;
    }

    if (isPrivateMatchPlay) {
        await handlePrivateMatchWord(word);
        return;
    }

    if (!word) return;
    
    if (!roomId) {
        console.warn('[play.js] Submission failed: No active Room ID found');
        showValidationFeedback('Not in a game room', false);
        return;
    }

    console.log(`[play.js] Submitting word "${word}" to room ${roomId} via ${currentInputMethod}`);


    try {
        const response = await fetch(`/room/${roomId}/submit_word`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                word: word,
                input_method: currentInputMethod,
                path: finalPath
            })
        });
        const data = await response.json();

        // Show validation feedback
        showValidationFeedback(data.message || (data.success ? 'Valid Word' : 'Invalid Word'), data.success);

        if (input) {
            input.style.backgroundColor = data.success ? 'rgba(0, 255, 0, 0.1)' : 'rgba(255, 0, 0, 0.1)';
            setTimeout(() => { if (input) input.style.backgroundColor = ''; }, 1000);
        }

        if (data.success) {
            const currentState = window.lastGameState;
            if (currentState) {
                // USER: INSTANT DENSITY UPDATE
                if (data.cell_density) {
                    currentState.cell_density = data.cell_density;
                    currentState.max_cell_density = data.max_cell_density;
                    updateGameState(currentState);
                }

                const listEl = document.getElementById('submitted-words-list');
                const wordsStats = document.getElementById('words-stats');

                // Local UI Update: Add word immediately to the list for instant feedback
                const isFcfs = (currentState.game_type === 'fcfs');
                
                if (!isFcfs) {
                    // Standard, Split, Accumulative, 3D, and Solo modes
                    if (listEl) {
                        const placeholder = listEl.querySelector('.placeholder');
                        if (placeholder) placeholder.remove();

                        const isBonus = data.word === currentState.bonus_word;
                        let className = 'word-item player-word' + (isBonus ? ' bonus-word' : '') + (data.points < 0 ? ' penalty-word' : '');
                        const html = `<div class="${className}" style="display:flex; justify-content:space-between; animation: slideIn 0.3s ease;">
                            <span>${data.word}</span>
                            <span style="opacity:0.8">${data.points}</span>
                        </div>`;
                        listEl.insertAdjacentHTML('afterbegin', html);

                        const me = currentState.players.find(p => p.username.toLowerCase() === (currentUser || "").toLowerCase().trim());
                        if (me) {
                            me.score = Math.max(0, data.new_score);
                            renderPlayers(currentState.players, currentUser, currentState);
                        }
                    }

                    if (wordsStats && data.points > 0) {
                        // Update the "Found/Total" counter immediately
                        const parts = wordsStats.textContent.match(/(\d+)\/(\d+) - (\d+)% \(([\d,]+) total pts\)/);
                        if (parts) {
                            let found = parseInt(parts[1]) + 1;
                            const total = parseInt(parts[2]);
                            const percent = total > 0 ? Math.round((found / total) * 100) : 0;
                            const totalPoints = parts[4];
                            wordsStats.textContent = `${found}/${total} - ${percent}% (${totalPoints} total pts)`;
                        } else {
                            const partsSimple = wordsStats.textContent.match(/(\d+)\/(\d+) - (\d+)%/);
                            if (partsSimple) {
                                let found = parseInt(partsSimple[1]) + 1;
                                const total = parseInt(partsSimple[2]);
                                const percent = total > 0 ? Math.round((found / total) * 100) : 0;
                                wordsStats.textContent = `${found}/${total} - ${percent}%`;
                            }
                        }
                    }
                } else {
                    // FCFS Mode: Shared Feed update
                    if (listEl) {
                        const placeholder = listEl.querySelector('.placeholder');
                        if (placeholder) placeholder.remove();

                        const itemId = `feed-${currentUser}-${data.word}`.replace(/\s+/g, '');
                        const html = `<div id="${itemId}" class="feed-item myself" style="animation: slideInRight 0.3s ease;">
                            <span class="feed-word">${data.word}</span>
                            <span class="feed-info">${currentUser} • ${data.points}pts</span>
                        </div>`;
                        listEl.insertAdjacentHTML('afterbegin', html);
                        
                        const panelEl = listEl.parentElement;
                        requestAnimationFrame(() => { if (panelEl) panelEl.scrollTop = 0; });
                    }

                    if (wordsStats) {
                        const parts = wordsStats.textContent.match(/(\d+)\/(\d+) - (\d+)% \(([\d,]+) total pts\)/);
                        if (parts) {
                            let found = parseInt(parts[1]) + 1;
                            const total = parseInt(parts[2]);
                            const percent = total > 0 ? Math.round((found / total) * 100) : 0;
                            const totalPoints = parts[4];
                            wordsStats.textContent = `${found}/${total} - ${percent}% (${totalPoints} total pts)`;
                        } else {
                            const partsSimple = wordsStats.textContent.match(/(\d+)\/(\d+) - (\d+)%/);
                            if (partsSimple) {
                                let found = parseInt(partsSimple[1]) + 1;
                                const total = parseInt(partsSimple[2]);
                                const percent = total > 0 ? Math.round((found / total) * 100) : 0;
                                wordsStats.textContent = `${found}/${total} - ${percent}%`;
                            }
                        }
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error submitting word:', error);
        showValidationFeedback('Submission Error', false);
    }
    // input.value = ''; // MOVED TO TOP OF FUNCTION
    // Clear typing highlights and declaration after submission
    document.querySelectorAll('.board-cell.typing-highlight').forEach(c => c.classList.remove('typing-highlight'));
    // ONLY clear panel and header if we're not celebrating the winner!
    const isIntermission = window.lastGameState && window.lastGameState.state === 'intermission';
    if (!isIntermission) {
        const defHeader = document.getElementById('definition-header');
        if (defHeader) defHeader.style.display = 'none';
        const defContent = document.getElementById('definition-content');
        if (defContent) defContent.innerHTML = '<p class="placeholder">Select a word to see its definition</p>';
    }
}

function showValidationFeedback(message, isValid) {
    const statusEl = document.getElementById('word-validation-status');
    if (!statusEl) return;

    // Clear existing timeout
    if (validationTimeout) clearTimeout(validationTimeout);

    // Set text and class
    statusEl.textContent = message;
    statusEl.className = 'validation-status ' + (isValid ? 'status-valid' : 'status-invalid');

    // Reset after 3 seconds
    validationTimeout = setTimeout(() => {
        statusEl.textContent = '';
        statusEl.className = 'validation-status';
        validationTimeout = null;
    }, 3000);
}

function leaveCurrentRoom() {
    if (isTournamentPlay) {
        // We don't necessarily want to force forfeit on EVERY leave (e.g. browser refresh handles itself better)
        // but for the "Leave" button it is handled in the listener.
        // This is a backup.
        exitTournamentPlay();
        return;
    }
    const roomId = getCurrentRoomId();
    if (!roomId) return;

    // 1. Clear local state and stop polling immediately (synchronous)
    // This guarantees the UI updates/redirects and stops checking room state instantly
    stopPolling();
    window.currentRoomId = null;
    localStorage.removeItem('last_joined_room');
    const playBtn = document.getElementById('play-btn');
    if (playBtn) {
        playBtn.disabled = true;
        playBtn.classList.remove('active');
        playBtn.title = "Join a room to play.";
    }
    if (window.updateManualToolState) window.updateManualToolState();
    clearGameUIAndCache();

    // 2. Fire-and-forget network notification
    // Use keepalive: true or sendBeacon so the browser does not cancel the request on unload/redirect,
    // and do not block the thread awaiting it!
    const url = `/api/room/${roomId}/leave`;
    if (navigator.sendBeacon) {
        navigator.sendBeacon(url);
    } else {
        fetch(url, { method: 'POST', keepalive: true }).catch(() => {});
    }
}
window.leaveCurrentRoom = leaveCurrentRoom;

const returnBtnEl = document.getElementById('return-lobby-btn');
if (returnBtnEl) {
    returnBtnEl.addEventListener('click', async () => {
        if (isTournamentPlay) {
            // User leaving tournament mid-round = forfeit
            const confirmLeave = confirm("Leaving mid-round will end your tournament turn and record a score of 0. Are you sure?");
            if (!confirmLeave) return;
            try { await fetch('/api/tournament/forfeit', { method: 'POST' }); } catch (e) { }
            exitTournamentPlay();
            return;
        }
        await leaveCurrentRoom();
        showPage('page-lobby');
    });
}

const rotateBtnEl = document.getElementById('rotate-board-btn');
if (rotateBtnEl) {
    rotateBtnEl.addEventListener('click', () => {
        isBoardRotated = !isBoardRotated;
        console.log('[play.js] Board rotation toggled. Rotated:', isBoardRotated);
        // Force re-render if we have state
        if (window.lastGameState && window.lastGameState.board) {
            // Consistent with updateGameState: Only gray in intermission
            const isIntermission = window.lastGameState.state === 'intermission';
            const is3D = window.lastGameState.game_type === '3d' || (window.lastGameState.board && window.lastGameState.board.length === 6 && Array.isArray(window.lastGameState.board[0]) && Array.isArray(window.lastGameState.board[0][0]));
            renderBoard(window.lastGameState.board, isIntermission, is3D);
        }
    });
}

// Definition Logic
async function fetchDefinition(word) {
    if (!word) return;
    const defContent = document.getElementById('definition-content');
    const defWord = document.getElementById('definition-word');
    const defHeader = document.getElementById('definition-header');
    if (!defContent) return;

    // Show word immediately in dedicated header
    if (defWord && defHeader) {
        defWord.textContent = word.toUpperCase();
        defHeader.style.display = 'block';
    }

    defContent.innerHTML = '<p class="placeholder">Loading definition...</p>';

    try {
        const resp = await fetch(`/api/definition?word=${encodeURIComponent(word)}`);
        const data = await resp.json();

        if (data.definition || data.pronunciation) {
            let html = '';
            if (data.pronunciation) {
                html += `<div class="pronunciation">${data.pronunciation}</div>`;
            }
            if (data.definition) {
                html += `<span class="definition-text">${data.definition}</span>`;
            }
            defContent.innerHTML = html;
        } else {
            defContent.innerHTML = `<p class="placeholder">Definition not found.</p>`;
        }
    } catch (e) {
        console.error('Definition error:', e);
        defContent.innerHTML = `<p class="placeholder">Error: ${e.message}</p>`;
    }
}

const submittedWordsListEl = document.getElementById('submitted-words-list');
if (submittedWordsListEl) {
    submittedWordsListEl.addEventListener('click', (e) => {
        // Mark that the user is explicitly viewing a definition (so winner announcement doesn't overwrite it)
        if (window.lastGameState && window.lastGameState.state === 'intermission') {
            window.userViewingDefinitionIntermission = true;
        }

        // Handle clicks on feed items, notepads, etc.
        const item = e.target.closest('.word-item') ||
            e.target.closest('.feed-item') ||
            e.target.closest('.clue-item') ||
            e.target.closest('.notepad-item'); // Added intermission items
        if (item) {
            let word = item.dataset.word;

            // Fallback parsing if data-word is missing (Legacy or dynamic)
            if (!word) {
                if (item.classList.contains('feed-item')) {
                    const wEl = item.querySelector('.feed-word');
                    if (wEl) word = wEl.textContent.trim();
                } else {
                    // Try first span or bold text
                    const bold = item.querySelector('span[style*="bold"]');
                    const firstSpan = item.querySelector('span');
                    if (bold) word = bold.textContent.trim();
                    else if (firstSpan) word = firstSpan.textContent.trim();
                }
            }

            if (word) fetchDefinition(word);
        }
    });
}

window.addEventListener('beforeunload', () => {
    if (window.currentRoomId) {
        const url = '/api/room/' + window.currentRoomId + '/leave';
        navigator.sendBeacon(url);
    }
});

function createSpectatorPanel() {
    const panel = document.createElement('div');
    panel.id = 'spectator-status-panel';
    panel.className = 'spectator-status-panel';

    // Append to DEFINITIONS PANEL (Right Column)
    // Sacrifice definition space for spectator info
    const defPanel = document.querySelector('.definitions-panel');
    if (defPanel) {
        defPanel.appendChild(panel);
    } else {
        // Fallback to board panel if definitions panel missing
        const boardPanel = document.querySelector('.board-panel');
        if (boardPanel) boardPanel.appendChild(panel);
    }

    return panel;
}

// ─────────── Mouse & Touch Board Interaction ───────────

function selectCell(row, col, letter, cellEl, face = null) {
    const key = face !== null ? `${face},${row},${col}` : `${row},${col}`;
    const pathLen = mouseState.selectedPath.length;

    // Check for backtracking
    if (pathLen >= 2) {
        const secondToLast = mouseState.selectedPath[pathLen - 2];
        const match = face !== null ?
            (secondToLast.face === face && secondToLast.row === row && secondToLast.col === col) :
            (secondToLast.row === row && secondToLast.col === col);

        if (match) {
            const lastCell = mouseState.selectedPath.pop();
            const lastKey = lastCell.face !== null ? `${lastCell.face},${lastCell.row},${lastCell.col}` : `${lastCell.row},${lastCell.col}`;
            mouseState.visitedCells.delete(lastKey);

            let oldSelector = `.board-cell[data-r="${lastCell.row}"][data-c="${lastCell.col}"]`;
            if (lastCell.face !== null && lastCell.face !== undefined) {
                oldSelector = `.board-cell[data-f="${lastCell.face}"][data-r="${lastCell.row}"][data-c="${lastCell.col}"]`;
            }
            const oldCellEl = document.querySelector(oldSelector);
            if (oldCellEl) oldCellEl.classList.remove('selected', 'current');

            if (cellEl) {
                document.querySelectorAll('.board-cell.current').forEach(c => c.classList.remove('current'));
                cellEl.classList.add('current');
            }

            updateWordInputFromPath();
            return;
        }
    }

    if (mouseState.visitedCells.has(key)) return;

    mouseState.visitedCells.add(key);
    mouseState.selectedPath.push({ row, col, letter, face });

    if (cellEl) {
        document.querySelectorAll('.board-cell.current').forEach(c => c.classList.remove('current'));
        cellEl.classList.add('selected', 'current');
    }

    updateWordInputFromPath();
}

function updateWordInputFromPath() {
    const wordInputEl = document.getElementById('word-input');
    if (wordInputEl) {
        wordInputEl.value = mouseState.selectedPath.map(p => {
            const L = p.letter; // Show full letter string (e.g. "L/T")
            return L === 'Q' ? 'QU' : L;
        }).join('');
    }
}

function handleCellMouseDown(e) {
    if (e.button !== 0) return; // Only left click
    if (window.isSpectatorMode) return;
    if (Date.now() - lastTouchTime < 1500) return; // Ignore simulated mouse events on touch devices

    const cell = e.target.closest('.board-cell');
    if (!cell || cell.classList.contains('grayed')) return;

    // Prevent native browser drag/selection behavior from interrupting our swipe
    e.preventDefault();

    // Reset path
    mouseState.isDown = true;
    mouseState.selectedPath = [];
    mouseState.visitedCells = new Set();
    document.querySelectorAll('.board-cell.selected, .board-cell.current').forEach(c => {
        c.classList.remove('selected', 'current');
    });

    const f = cell.dataset.f !== undefined ? parseInt(cell.dataset.f) : null;
    const r = parseInt(cell.dataset.r || cell.dataset.row);
    const c = parseInt(cell.dataset.c || cell.dataset.col);
    const letter = getLetterFromCellAndEvent(cell, e);
    selectCell(r, c, letter, cell, f);
}

function getLetterFromCellAndEvent(cell, e) {
    const letter = cell.dataset.letter;
    if (letter && letter.includes('/')) {
        const rect = cell.getBoundingClientRect();
        const centerY = rect.top + rect.height / 2;
        let clientY = null;
        if (e) {
            if (e.touches && e.touches.length > 0) {
                clientY = e.touches[0].clientY;
            } else if (e.changedTouches && e.changedTouches.length > 0) {
                clientY = e.changedTouches[0].clientY;
            } else if (typeof e.clientY === 'number') {
                clientY = e.clientY;
            }
        }
        if (clientY !== null) {
            const [top, bottom] = letter.split('/');
            const selectedLetter = (clientY < centerY) ? top : bottom;
            console.log(`[EitherOr] Resolved coords split: clientY=${clientY} centerY=${centerY} -> selected: ${selectedLetter}`);
            return selectedLetter;
        }
    }
    return letter;
}

function handleCellMouseOver(e) {
    if (!mouseState.isDown) return;
    if (window.isSpectatorMode) return;

    const cell = e.target.closest('.board-cell');
    if (!cell || cell.classList.contains('grayed')) return;

    const f = cell.dataset.f !== undefined ? parseInt(cell.dataset.f) : null;
    const r = parseInt(cell.dataset.r || cell.dataset.row);
    const c = parseInt(cell.dataset.c || cell.dataset.col);
    const letter = getLetterFromCellAndEvent(cell, e);
    selectCell(r, c, letter, cell, f);
}

function handleCellTouchStart(e) {
    if (window.isSpectatorMode) return;
    if (mouseState.isDown) return; // Prevent double touch/accidental brushes from erasing path

    const touch = e.touches[0];
    const target = document.elementFromPoint(touch.clientX, touch.clientY);
    const cell = target && target.closest('.board-cell');

    if (cell && !cell.classList.contains('grayed')) {
        e.preventDefault();

        mouseState.isDown = true;
        mouseState.selectedPath = [];
        mouseState.visitedCells = new Set();
        document.querySelectorAll('.board-cell.selected, .board-cell.current').forEach(c => {
            c.classList.remove('selected', 'current');
        });

        const f = cell.dataset.f !== undefined ? parseInt(cell.dataset.f) : null;
        const r = parseInt(cell.dataset.r || cell.dataset.row);
        const c = parseInt(cell.dataset.c || cell.dataset.col);
        const letter = getLetterFromCellAndEvent(cell, e);
        selectCell(r, c, letter, cell, f);
    }
}

function handleCellTouchMove(e) {
    if (!mouseState.isDown) return;
    if (window.isSpectatorMode) return;

    // Prevent mobile scrolling unconditionally during an active board swipe
    e.preventDefault();

    const touch = e.touches[0];
    const target = document.elementFromPoint(touch.clientX, touch.clientY);
    const cell = target && target.closest('.board-cell');

    if (cell && !cell.classList.contains('grayed')) {
        const f = cell.dataset.f !== undefined ? parseInt(cell.dataset.f) : null;
        const r = parseInt(cell.dataset.r || cell.dataset.row);
        const c = parseInt(cell.dataset.c || cell.dataset.col);
        const letter = getLetterFromCellAndEvent(cell, e);
        selectCell(r, c, letter, cell, f);
    }
}

function finishDragSelection() {
    if (!mouseState.isDown) return;
    mouseState.isDown = false;

    const path = mouseState.selectedPath;
    if (path.length >= 1) {
        const word = path.map(p => {
            const L = p.letter.includes('/') ? p.letter.split('/')[0] : p.letter;
            return L === 'Q' ? 'QU' : L;
        }).join('');
        const serverPath = path.map(p => (p.face !== null && p.face !== undefined) ? [p.face, p.row, p.col] : [p.row, p.col]);

        if (word.length >= 3) {
            submitWord(word, serverPath);
        }
    }

    // Clear visual state
    document.querySelectorAll('.board-cell.selected, .board-cell.current').forEach(c => {
        c.classList.remove('selected', 'current');
    });
    mouseState.selectedPath = [];
    mouseState.visitedCells = new Set();

    // UX: If round ended while we were dragging, refocus chat now that we're released
    const inputEl = document.getElementById('word-input');
    if (window.refocusChatPending) {
        const typingInProgress = (inputEl && inputEl.value.trim().length > 0);
        
        if (!typingInProgress) {
            window.refocusChatPending = false;
            const chatInput = document.getElementById('chat-input');
            if (chatInput) setTimeout(() => chatInput.focus(), 150);
        }
    }

    // ALWAYS clear live input display after selection is committed
    if (inputEl) inputEl.value = '';
}

// Wire board events via delegation on the static board wrapper
(function initBoardInteraction() {
    const boardEl = document.getElementById('game-board');
    if (!boardEl) return;

    boardEl.addEventListener('mousedown', handleCellMouseDown);
    boardEl.addEventListener('mouseover', handleCellMouseOver);
    boardEl.addEventListener('touchstart', handleCellTouchStart, { passive: false });
    boardEl.addEventListener('touchmove', handleCellTouchMove, { passive: false });

    // Release: commit word
    document.addEventListener('mouseup', finishDragSelection);
    document.addEventListener('touchend', finishDragSelection);
})();

// Word Tabs Switching
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('word-tab')) {
        activeWordsTab = e.target.dataset.tab;

        // Update UI immediately for responsiveness
        document.querySelectorAll('.word-tab').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === activeWordsTab);
        });
        // Update visibility of ALL tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            const tabId = content.id.replace('tab-content-', '');
            content.classList.toggle('active', activeWordsTab === tabId);
        });

        // Refresh state visualization
        if (window.lastGameState) {
            updateGameState();
        }
    }
});

// --- TOURNAMENT PLAY LOGIC ---

// Helper to ensure UI is correctly reset for active play (escapes spectator mode)
function resetPlayUI() {
    console.log('[play.js] resetPlayUI() called for active play session');
    const wordInputSection = document.querySelector('.word-input-section');
    const wordInput = document.getElementById('word-input');
    const submitBtn = document.getElementById('submit-word-btn');
    const defContent = document.getElementById('definition-content');
    const specPanel = document.getElementById('spectator-status-panel');
    const boardPanel = document.querySelector('.board-panel');

    if (wordInputSection) wordInputSection.style.display = 'flex';
    if (wordInput) {
        wordInput.disabled = false;
        wordInput.value = '';
        wordInput.style.display = '';
        setTimeout(() => wordInput.focus(), 150);
    }
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.style.display = '';
    }
    if (defContent) defContent.style.display = 'block';
    if (specPanel) specPanel.style.display = 'none';
    if (boardPanel) {
        boardPanel.style.pointerEvents = 'auto';
        boardPanel.style.opacity = '1';
    }

    // Ensure spectator mode flag is reset
    window.isSpectatorMode = false;

    // Refresh submission listeners to ensure DOM sync
    if (typeof initWordSubmission === 'function') {
        initWordSubmission();
    }

    // Refresh layout sizing
    setTimeout(checkBoardOverflow, 100);
}

async function initTournamentPlay() {
    const activeData = JSON.parse(localStorage.getItem('tournament_play_active'));
    if (!activeData) return;

    console.log('[Tournament] Initializing turn session:', activeData);
    isTournamentPlay = true;
    window.isTournamentPlay = true;
    tournamentWords = []; // Will now store {word, points, timestamp}
    tournamentScore = 0;
    tournamentStartTime = Date.now() / 1000;

    // Stop any standard polling
    stopPolling();

    // Clear UI
    resetChat();
    const wordsList = document.getElementById('submitted-words-list');
    if (wordsList) wordsList.innerHTML = '';
    const wordsStats = document.getElementById('words-stats');
    if (wordsStats) wordsStats.textContent = '';

    try {
        const response = await fetch('/api/tournament/game-state');
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            exitTournamentPlay();
            return;
        }

        // Mobile Device Restriction: Cube is not allowed on mobile!
        const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const is3D = data.params.board_dimensions === '3x3x3' || (data.board && data.board.length === 6 && Array.isArray(data.board[0]) && Array.isArray(data.board[0][0]));
        if (isMobile && is3D) {
            console.log('[Mobile] Cube tournament matches are not permitted on mobile devices. Kicking player.');
            localStorage.removeItem('tournament_play_active');
            ejectToLobby("mobile-cube-restriction");
            return;
        }

        window.tournamentParams = data.params;
        const tournamentGameState = {
            board: data.board,
            state: 'active',
            timer: data.params.time_limit,
            game_type: 'tournament',
            status: 'active',
            board_dimensions: data.params.board_dimensions,
            time_limit: data.params.time_limit,
            current_board_format: data.params.board_format || 'Normal',
            spinner_params: data.params,
            bonus_word: data.bonus_word
        };
        window.lastGameState = tournamentGameState;
        lastRenderedBoardJSON = null; // Force re-render

        // Render Board
        console.log('[Tournament] Rendering tournament board. Format:', (data.params.board_format || 'Normal'));
        renderBoard(data.board, false, is3D);
        updateParameters(tournamentGameState);
        resetPlayUI();

        // Timer Setup: The tournament game has its OWN local timer starting from the moment they click play
        localEndTime = (Date.now() / 1000) + data.params.time_limit;

        if (timerInterval) clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            const current = Date.now() / 1000;
            const diff = Math.max(0, Math.ceil(localEndTime - current));
            updateSpecialMatchTimer(diff); // Use the new helper

            if (diff <= 0) {
                clearInterval(timerInterval);
                finishTournamentTurn();
            }
        }, 1000);

        // UI Adjustments - Show Opponent
        const playerList = document.getElementById('players-list');
        if (playerList) {
            let oppName = "None (Bye)";
            try {
                const statusResp = await fetch('/api/tournament/status');
                const statusData = await statusResp.json();
                if (statusData.user_status && statusData.user_status.matchup) {
                    oppName = statusData.user_status.matchup.opponent_name || oppName;
                }
            } catch (e) { console.error("Failed to fetch matchup for display:", e); }

            playerList.innerHTML = `
                <div class="player-card active" style="border-left: 4px solid #2ecc71;">
                    <div class="player-info">
                        <div class="username">TOURNAMENT TURN</div>
                        <div class="score">Versus: <span style="color:#2ecc71">${oppName}</span></div>
                    </div>
                </div>
            `;
        }

    } catch (e) {
        console.error("Failed to load tournament game:", e);
        exitTournamentPlay();
    }
}

async function handleTournamentWord(word) {
    if (!word) return;

    if (tournamentWords.find(w => w.word === word)) {
        showValidationFeedback('Already found!', false);
        return;
    }

    // Check if word is on board
    const board = window.lastGameState ? window.lastGameState.board : null;
    if (!board) {
        console.error('[Tournament] No board found in lastGameState');
        return;
    }

    const is3D = board.length === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]);
    const path = is3D ? findWordPathOnCube(word, board) : findWordPathOnBoard(word, board);
    if (!path) {
        showValidationFeedback(`${word} is invalid.`, false);
        return;
    }

    // Check dictionary
    const dict = window.tournamentParams ? window.tournamentParams.dictionary : 'NWL';
    let is_valid_dict = false;
    try {
        const resp = await fetch('/api/tools/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word, dictionary: dict })
        });
        const data = await resp.json();
        is_valid_dict = data.is_valid;
    } catch (e) {
        console.error('Validation error', e);
    }

    // Check length and dictionary validity
    const minLen = window.tournamentParams ? window.tournamentParams.min_word_length : 3;
    if (word.length < minLen && !is_valid_dict) {
        showValidationFeedback("Sequence is not a word and is too small", false);
        return;
    }

    if (!is_valid_dict) {
        showValidationFeedback(`${word} is invalid.`, false);
        return;
    }

    if (word.length < minLen) {
        showValidationFeedback(`${word} is invalid.`, false);
        return;
    }

    // Use tournament-specific scoring (1=1, 2=2, 3=3, 4=4, 5=5, 6=10, 7=15, 8=25)
    let pts = word.length;
    if (word.length === 6) pts = 10;
    else if (word.length === 7) pts = 15;
    else if (word.length >= 8) pts = 25;

    // Bonus Word
    let isBonus = false;
    if (window.lastGameState && window.lastGameState.bonus_word && word === window.lastGameState.bonus_word.toUpperCase()) {
        pts += word.length;
        isBonus = true;
    }

    tournamentWords.push({
        word: word,
        points: pts,
        timestamp: Date.now() / 1000,
        is_bonus: isBonus
    });
    tournamentScore += pts;

    // Show success feedback
    showValidationFeedback(isBonus ? 'BONUS WORD!' : 'Valid Word', true);

    // Update Score UI
    const scoreEl = document.querySelector('.player-card .score');
    if (scoreEl) scoreEl.textContent = `Score: ${tournamentScore}`;

    // Update Found List
    const list = document.getElementById('submitted-words-list');
    if (list) {
        // Remove placeholder if it exists
        const placeholder = list.querySelector('.placeholder');
        if (placeholder) placeholder.remove();

        const item = document.createElement('div');
        item.className = 'word-item player-word' + (isBonus ? ' bonus-word' : '');
        item.style.display = 'flex';
        item.style.justifyContent = 'space-between';
        item.style.animation = 'slideIn 0.3s ease';
        item.innerHTML = `<span>${word}</span> <span style="opacity:0.8">${pts}</span>`;
        list.prepend(item);
    }

    // Flash Highlight
    reapplyBoardHighlights();
}

async function finishTournamentTurn() {
    console.log('[Tournament] Finish Turn. Final Score:', tournamentScore);
    const activeData = JSON.parse(localStorage.getItem('tournament_play_active'));

    try {
        const response = await fetch('/api/tournament/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tournament_id: activeData.tid,
                round_number: activeData.round,
                words: tournamentWords,
                score: Math.floor(tournamentScore),
                round_start_time: tournamentStartTime
            })
        });

        const res = await response.json();
        if (res.success) {
            alert(`Turn Complete! You scored ${tournamentScore} points.`);
        } else {
            alert("Error submitting score: " + res.error);
        }
    } catch (e) {
        console.error("Submit error:", e);
    }

    exitTournamentPlay();
}

function exitTournamentPlay() {
    localStorage.removeItem('tournament_play_active');
    isTournamentPlay = false;
    window.isTournamentPlay = false;
    clearGameUIAndCache();
    if (window.navigateToPage) {
        window.navigateToPage('tournaments');
    } else {
        window.location.href = '#page-tournaments';
    }
}

// --- PRIVATE MATCH PLAY LOGIC ---
window.initPrivateMatchPlay = function () {
    console.log('[play.js] initPrivateMatchPlay() START');
    isPrivateMatchPlay = true;
    isBoardRotated = false; // RESET: Ensure board isn't flipped 180 from a previous public game
    isTournamentPlay = false;
    privateMatchWords = [];
    privateMatchScore = 0;

    const activeMatch = JSON.parse(localStorage.getItem('private_match_active'));
    if (!activeMatch) {
        exitPrivateMatchPlay();
        return;
    }

    // Mobile Device Restriction: Cube is not allowed on mobile!
    const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const is3D = activeMatch.parameters.board_dimensions === '3x3x3' || (activeMatch.board && activeMatch.board.length === 6 && Array.isArray(activeMatch.board[0]) && Array.isArray(activeMatch.board[0][0]));
    if (isMobile && is3D) {
        console.log('[Mobile] Cube rooms are not permitted on mobile devices. Kicking player.');
        localStorage.removeItem('private_match_active');
        ejectToLobby("mobile-cube-restriction");
        return;
    }

    privateMatchParams = activeMatch.parameters;

    const mockState = {
        board: activeMatch.board,
        state: 'active',
        round: activeMatch.round,
        game_type: 'private',
        board_dimensions: activeMatch.parameters.board_dimensions,
        time_limit: activeMatch.parameters.time_limit,
        current_board_format: activeMatch.parameters.board_format || 'Normal',
        spinner_params: activeMatch.parameters
    };
    window.lastGameState = mockState;
    lastRenderedBoardJSON = null; // Force re-render

    console.log('[play.js] Rendering private match board:', activeMatch.board);
    renderBoard(activeMatch.board, false, is3D);

    updateParameters(mockState);
    resetPlayUI();

    // Clear UI
    resetChat();
    const wordsList = document.getElementById('submitted-words-list');
    if (wordsList) wordsList.innerHTML = '';
    const wordsStats = document.getElementById('words-stats');
    if (wordsStats) wordsStats.textContent = '';

    // Stop random multiplayer polling
    stopPolling();

    // Start local timer
    // Calculate remaining time based on end_time stored in localStorage
    // If we just launched, end_time was set.
    // If we reloaded page, it persists.
    let endTime = activeMatch.end_time;

    // Fix: If endTime is too far in past/future or invalid, reset it? 
    // Actually server doesn't enforce real-time for private turns as strictly 
    // unless we validate start_time there. 
    // For now trust client storage or reset if expired.

    if (endTime < Date.now() / 1000) {
        // Already expired? Or maybe just started and time offset issue?
        // Let's assume user just clicked Play if it's super old.
        // But launchPrivateMatch sets it to now + limit.
        // So play execution is correct.
    }

    // UI Adjustments
    const playerList = document.getElementById('players-list');
    if (playerList) {
        playerList.innerHTML = `
            <div class="player-card active" style="border-left: 4px solid var(--accent-color);">
                <div class="player-info">
                    <div class="username">PRIVATE MATCH</div>
                    <div class="score">Round ${activeMatch.round}</div>
                </div>
            </div>
        `;
    }

    startPrivateMatchTimer(endTime);
};

function startPrivateMatchTimer(endTime) {
    if (timerInterval) clearInterval(timerInterval);

    const timerEl = document.getElementById('timer-value');
    console.log('[play.js] Starting private match timer for endTime:', endTime);

    timerInterval = setInterval(() => {
        const now = Date.now() / 1000;
        const remaining = Math.max(0, Math.floor(endTime - now));
        const mins = Math.floor(remaining / 60);
        const secs = remaining % 60;
        if (timerEl) {
            timerEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
        }

        if (remaining <= 0) {
            console.log('[play.js] Private match timer reached 0! Triggering auto-finish.');
            clearInterval(timerInterval);

            // Tiny delay to ensure user actually sees the 0:00
            setTimeout(() => {
                finishPrivateMatchTurn();
            }, 500);
        }
    }, 1000);
}

async function handlePrivateMatchWord(word) {
    if (!word) return;

    if (privateMatchWords.find(w => w.word === word)) {
        showValidationFeedback('Already found!', false);
        return;
    }

    // Check if word is on board
    const board = window.lastGameState ? window.lastGameState.board : null;
    if (!board) {
        console.error('[Private Match] No board found in lastGameState');
        return;
    }

    const is3D = board.length === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]);
    const path = is3D ? findWordPathOnCube(word, board) : findWordPathOnBoard(word, board);
    if (!path) {
        showValidationFeedback('Not on board!', false);
        return;
    }

    // 1. Initial Checks (Dictionary & Min Length)
    const dict = privateMatchParams ? privateMatchParams.dictionary : 'NWL';
    let isDictionaryValid = false;
    try {
        const resp = await fetch('/api/tools/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word, dictionary: dict })
        });
        const data = await resp.json();
        isDictionaryValid = data.is_valid;
    } catch (e) {
        console.error('Validation error', e);
    }

    const minLen = privateMatchParams ? privateMatchParams.min_word_length : 3;
    if (word.length < minLen) {
        if (!isDictionaryValid) {
            showValidationFeedback("Sequence is not a word and is too small", false);
        } else {
            showValidationFeedback(`Too short (min ${minLen})`, false);
        }
        return;
    }

    // 2. Format & Bonus Info
    const fmt = privateMatchParams ? privateMatchParams.board_format : 'Normal';
    const fmtLower = fmt.toLowerCase();
    const activeMatch = JSON.parse(localStorage.getItem('private_match_active'));
    const bonusCell = activeMatch ? activeMatch.bonus_cell : null;
    let pts = 0;
    let isPenalty = false;

    // 3. Scoring Logic
    if (isDictionaryValid) {
        // Valid Word Scoring
        if (fmtLower.includes('valued') || fmtLower.includes('value')) {
            const letterValues = {
                'A': 2, 'B': 4, 'C': 4, 'D': 3, 'E': 1, 'F': 5, 'G': 3, 'H': 5, 'I': 2, 'J': 10,
                'K': 6, 'L': 3, 'M': 4, 'N': 2, 'O': 2, 'P': 4, 'Q': 10, 'R': 2, 'S': 2, 'T': 2,
                'U': 4, 'V': 5, 'W': 5, 'X': 9, 'Y': 5, 'Z': 9
            };
            for (let char of word.toUpperCase()) {
                pts += letterValues[char] || 1;
            }
        } else {
            // Standard length-based scoring (Fix: 5-letter words = 2 points)
            const L = word.length;
            if (L <= 2) pts = 0;
            else if (L <= 4) pts = 1;
            else if (L === 5) pts = 2;
            else if (L === 6) pts = 3;
            else if (L === 7) pts = 5;
            else pts = 11;
        }

        // Hidden Bonus Word (+Length)
        if (activeMatch && activeMatch.bonus_word && activeMatch.bonus_word.toUpperCase() === word) {
            pts += word.length;
            showValidationFeedback('BONUS WORD FOUND!', true);
        } else {
            showValidationFeedback('Valid Word', true);
        }

        // Format Bonus (+3 points for Either/Or or Bonus Letter tile)
        // User requested: No bonuses for Checkerboard
        if (bonusCell && !fmtLower.includes('checkerboard')) {
            const wordPath = is3D ? findWordPathOnCube(word, board) : findWordPathOnBoard(word, board);
            if (wordPath) {
                const hitsBonus = wordPath.some(coord => {
                    const cell = is3D ? board[coord.f][coord.r][coord.c] : board[coord.r][coord.c];
                    const isEitherOr = typeof cell === 'string' && cell.includes('/');
                    
                    if (isEitherOr) return true; // Any Either/Or tile usage triggers bonus
                    
                    if (!bonusCell) return false;

                    if (Array.isArray(bonusCell)) {
                        if (is3D && bonusCell.length === 3) return coord.f === bonusCell[0] && coord.r === bonusCell[1] && coord.c === bonusCell[2];
                        return coord.r === bonusCell[0] && coord.c === bonusCell[1];
                    } else if (typeof bonusCell === 'object') {
                        if (is3D && bonusCell.f !== undefined) return coord.f === bonusCell.f && coord.r === bonusCell.r && coord.c === bonusCell.c;
                        return coord.r === bonusCell.r && coord.c === bonusCell.c;
                    }
                    return false;
                });
                if (hitsBonus) {
                    pts += 3;
                    console.log('[Private Match] Awarded +3 Bonus for Special Tile');
                }
            }
        }
    } else {
        // Invalid Word Check for Penalty
        if (fmtLower.includes('penalty')) {
            const wordPath = is3D ? findWordPathOnCube(word, board) : findWordPathOnBoard(word, board);
            if (wordPath) {
                pts = -3;
                isPenalty = true;
                showValidationFeedback('INVALID (PENALTY -3)', false);
            } else {
                showValidationFeedback('Not in dictionary!', false);
                return;
            }
        } else {
            showValidationFeedback('Not in dictionary!', false);
            return;
        }
    }

    privateMatchWords.push({
        word: word,
        points: pts,
        timestamp: Date.now() / 1000,
        is_penalty: isPenalty
    });
    privateMatchScore += pts;
    if (privateMatchScore < 0) privateMatchScore = 0;

    const scoreEl = document.querySelector('.player-card .score');
    if (scoreEl) scoreEl.textContent = `Score: ${privateMatchScore}`;

    const list = document.getElementById('submitted-words-list');
    if (list) {
        // Remove placeholder if it exists
        const placeholder = list.querySelector('.placeholder');
        if (placeholder) placeholder.remove();

        const item = document.createElement('div');
        item.className = 'word-item player-word' + (isPenalty ? ' penalty-word' : '');
        item.style.display = 'flex';
        item.style.justifyContent = 'space-between';
        item.style.animation = 'slideIn 0.3s ease';
        item.innerHTML = `<span>${word}</span> <span style="opacity:0.8">${pts}</span>`;
        list.prepend(item);
    }

    // Flash Highlight
    reapplyBoardHighlights();
}

async function finishPrivateMatchTurn() {
    console.log('[play.js] finishPrivateMatchTurn() called');
    const activeMatch = JSON.parse(localStorage.getItem('private_match_active'));
    if (!activeMatch) {
        console.warn('[play.js] finishPrivateMatchTurn: No activeMatch found');
        exitPrivateMatchPlay();
        return;
    }

    try {
        console.log('[play.js] Submitting turn for match:', activeMatch.mid, 'Round:', activeMatch.round);
        const resp = await fetch('/api/private-match/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                match_id: activeMatch.mid,
                round_number: activeMatch.round,
                words: privateMatchWords,
                score: privateMatchScore
            })
        });

        if (!resp.ok) {
            throw new Error('Server returned ' + resp.status);
        }

        const data = await resp.json();
        // Assuming success if we got here

        alert("Turn Submitted!");
    } catch (e) {
        console.error(e);
        alert("Error submitting turn: " + e.message);
        // Do not exit if error, so user can try again?
        // Or exit anyway to avoid stuck state?
        // Let's exit for now to avoid state corruption, but user might lose turn data if not saved.
        // Actually, if it failed, we should probably keep them on screen?
        // But the user requested "Instant Lobby Loading", so let's retry navigating.
    }

    exitPrivateMatchPlay();
}

function exitPrivateMatchPlay() {
    isPrivateMatchPlay = false;
    localStorage.removeItem('private_match_active');
    clearGameUIAndCache();

    // Clean up timers
    if (window.privateMatchInterval) clearInterval(window.privateMatchInterval);

    if (window.navigateToPage) {
        window.navigateToPage('lobby');
        // Force refresh of match lists immediately
        if (window.loadPrivateMatches) setTimeout(window.loadPrivateMatches, 100);
    }
}

// User Request: Render a smaller, static board for history review
function renderPreviousBoard(board, container) {
    if (!board || board.length === 0 || !container) return;

    container.innerHTML = '';
    
    // Check if 3D (3 layers of arrays) or 2D (2 layers)
    const is3D = board.length === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]);

    if (is3D) {
        // Render 3D Cube Faces (2x3 grid)
        container.style.display = 'grid';
        container.style.gridTemplateColumns = 'repeat(3, 1fr)';
        container.style.gap = '15px';
        container.style.padding = '15px';
        container.style.background = 'rgba(0,0,0,0.3)';
        container.style.borderRadius = '16px';
        container.style.marginBottom = '20px';

        board.forEach((face, idx) => {
            const faceCont = document.createElement('div');
            faceCont.style.display = 'grid';
            faceCont.style.gap = '2px';
            faceCont.style.padding = '5px';
            faceCont.style.background = 'rgba(0,0,0,0.2)';
            faceCont.style.borderRadius = '8px';
            
            const rows = face.length;
            const cols = face[0].length;
            const fCellSize = rows > 4 ? '18px' : '22px';
            
            faceCont.style.gridTemplateColumns = `repeat(${cols}, ${fCellSize})`;
            faceCont.style.gridTemplateRows = `repeat(${rows}, ${fCellSize})`;
            faceCont.style.justifyContent = 'center';

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const cell = document.createElement('div');
                    cell.style.width = fCellSize;
                    cell.style.height = fCellSize;
                    cell.style.fontSize = '0.7rem';
                    cell.style.display = 'flex';
                    cell.style.justifyContent = 'center';
                    cell.style.alignItems = 'center';
                    cell.style.fontWeight = '800';
                    cell.style.background = 'var(--input-bg)';
                    cell.style.borderRadius = '3px';
                    cell.style.color = 'var(--text-primary)';
                    cell.textContent = face[r][c];
                    faceCont.appendChild(cell);
                }
            }
            container.appendChild(faceCont);
        });
    } else {
        // Render 2D Grid
        container.style.display = 'grid';
        container.style.gap = '4px';
        container.style.padding = '10px';
        container.style.background = 'rgba(0,0,0,0.2)';
        container.style.borderRadius = '12px';
        container.style.marginBottom = '20px';

        const rows = board.length;
        const cols = board[0].length;

        // Use smaller cell size for previous board
        const cellSize = rows > 4 ? '35px' : '45px';
        container.style.gridTemplateColumns = `repeat(${cols}, ${cellSize})`;
        container.style.gridTemplateRows = `repeat(${rows}, ${cellSize})`;
        container.style.justifyContent = 'center';

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell';
                cell.style.width = cellSize;
                cell.style.height = cellSize;
                cell.style.fontSize = rows > 4 ? '0.9rem' : '1.1rem';
                cell.style.display = 'flex';
                cell.style.justifyContent = 'center';
                cell.style.alignItems = 'center';
                cell.style.fontWeight = '700';
                cell.style.background = 'var(--input-bg)';
                cell.style.border = '1px solid var(--input-border)';
                cell.style.borderRadius = '6px';
                cell.style.color = 'var(--text-primary)';
                cell.textContent = board[r][c];
                container.appendChild(cell);
            }
        }
    }
}

// --- 3D Morpheme Cube Rotation ---
window.cubeRotationX = -25;
window.cubeRotationY = 45;

function setupCubeRotation() {
    // We bind a global keydown for arrows to rotate the current active cube
    // This listener is idempotent; only one global listener is needed.
    if (window.cubeListenerAdded) return;
    window.cubeListenerAdded = true;

    document.addEventListener('keydown', (e) => {
        const cube = document.getElementById('game-cube');
        if (!cube) return;

        const step = 90;
        let changed = false;

        // Initialize if first time
        if (window.cubeRotationX === undefined) window.cubeRotationX = -30;
        if (window.cubeRotationY === undefined) window.cubeRotationY = 45;

        if (e.key === 'ArrowUp') { 
            // Up Arrow -> Tilted Top View
            window.cubeRotationX = -30;
            changed = true;
        }
        else if (e.key === 'ArrowDown') { 
            // Down Arrow -> Tilted Bottom View
            window.cubeRotationX = 30;
            changed = true;
        }
        else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { 
            // Left/Right -> Rotate sides while maintaining tilt
            const dir = (e.key === 'ArrowLeft') ? -1 : 1;
            window.cubeRotationY += (dir * step);
            changed = true; 
        }

        if (changed) {
            e.preventDefault();
            cube.style.transform = `rotateX(${window.cubeRotationX}deg) rotateY(${window.cubeRotationY}deg)`;
        }
    });
}

// --- Distributed Board Generation (DBG) Methods ---
function runBoardProbe() {
    const s = window.lastGameState;
    if (!s) return;
    
    // Choose params: Prefer next_spinner_params for proactive search, fallback to current
    const params = s.next_spinner_params || s.spinner_params;
    if (!params) return;
    
    // Safety: Only probe if we are in intermission or a search is explicitly active
    const now = Date.now();
    if (now - lastProbeTime < PROBE_INTERVAL) return;
    lastProbeTime = now;

    console.log('[DBG] Client Probing...');
    const board = generateProbeBoard(s.spinner_params);
    if (!board) return;

    fetch(`/api/room/${s.room_id}/propose-board`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board: board })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            console.log(`[DBG] SUCCESS! Server accepted our board (${data.words_found} words, ${Math.round(data.uniqueness*100)}% unique)`);
        }
    })
    .catch(e => {}); // Fail silently
}

function generateProbeBoard(params) {
    const dims = params.dimensions || '4x4';
    const format = params.format || 'Normal';
    const bonusWord = params.bonus_word || '';
    const parts = dims.split('x');
    if (parts.length !== 2) return null;
    const rows = parseInt(parts[0]);
    const cols = parseInt(parts[1]);

    let letters = [];
    if (rows === 4 && cols === 4 && format !== 'Checkerboard') {
         const dice = [...DICE_CONFIG_4x4];
         for (let i = dice.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [dice[i], dice[j]] = [dice[j], dice[i]];
         }
         letters = dice.map(d => d[Math.floor(Math.random() * 6)]);
    } else {
         const alpha = "EEEEEEEEEEERRRRRRRRRTTTTTTTTTAAAAAAAAAIIIIIIIINNNNNNNNSSSSSSSSOOOOOOOLLLLLLDDDDDDUUUUUCCCGGGMMMYYYPPPBKBVWFZKQXJ";
         for(let i=0; i<rows*cols; i++) letters.push(alpha[Math.floor(Math.random()*alpha.length)]);
    }

    let board = [];
    for(let r=0; r<rows; r++) board.push(letters.slice(r*cols, (r+1)*cols));

    if (format === 'Checkerboard') {
         const vowels = "AEIOU";
         const cons = "BCDFGHJKLMNPQRSTVWXYZ";
         for(let r=0; r<rows; r++) {
             for(let c=0; c<cols; c++) {
                 const isVowelSpace = (r + c) % 2 === 0;
                 board[r][c] = isVowelSpace ? vowels[Math.floor(Math.random()*vowels.length)] : cons[Math.floor(Math.random()*cons.length)];
             }
         }
    }

    if (bonusWord) {
        board = embedBonusWordForProbe(board, bonusWord);
    }

    return board;
}

function embedBonusWordForProbe(board, word) {
    if (!word) return board;
    const rows = board.length;
    const cols = board[0].length;
    const pWord = word.toUpperCase().replace(/QU/g, 'Q').split('');
    if (pWord.length > rows * cols) return board;

    for (let attempt = 0; attempt < 10; attempt++) {
        let r = Math.floor(Math.random() * rows);
        let c = Math.floor(Math.random() * cols);
        let path = [[r, c]];
        let currentBoard = board.map(row => [...row]);
        currentBoard[r][c] = pWord[0];

        let success = true;
        for (let i = 1; i < pWord.length; i++) {
            let neighbors = [];
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    if (dr === 0 && dc === 0) continue;
                    let nr = r + dr, nc = c + dc;
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !path.some(p => p[0] === nr && p[1] === nc)) {
                        neighbors.push([nr, nc]);
                    }
                }
            }
            if (neighbors.length === 0) { success = false; break; }
            let next = neighbors[Math.floor(Math.random() * neighbors.length)];
            r = next[0]; c = next[1];
            path.push([r, c]);
            currentBoard[r][c] = pWord[i];
        }
        if (success) return currentBoard;
    }
    return board;
}

/**
 * Global function for opening the finders modal (Shared across play.js logic)
 */
window.showFinderModal = function (word) {
    const modal = document.getElementById('generic-info-modal');
    const title = document.getElementById('generic-modal-title');
    const body = document.getElementById('generic-modal-body');

    if (modal && title && body) {
        const wordUpper = word.toUpperCase();
        title.textContent = `Who found "${wordUpper}"?`;

        // Use current game state to find players
        if (!window.lastGameState || !window.lastGameState.players) {
            console.warn('[showFinderModal] No lastGameState available');
            return;
        }

        const finders = window.lastGameState.players.filter(p =>
            p.submitted_words && p.submitted_words.some(sw =>
                (typeof sw === 'object' ? sw.word : sw).toUpperCase() === wordUpper
            )
        );

        if (finders.length === 0) {
            body.innerHTML = '<p class="placeholder" style="padding: 20px; text-align: center;">No one has found this word yet.</p>';
        } else {
            body.innerHTML = finders.map(p => {
                const rating = p.rating || 0;
                const rColor = window.getRatingColor ? window.getRatingColor(rating) : '#fff';
                return `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); background: rgba(255,255,255,0.02); margin-bottom: 4px; border-radius: 6px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div style="width: 14px; height: 14px; background: ${rColor}; border-radius: 3px; box-shadow: 0 0 10px ${rColor}22;"></div>
                            <span style="font-weight: 700; font-size: 0.95rem;">${p.username}</span>
                        </div>
                        <span style="opacity: 0.5; font-size: 0.8rem; font-weight: 600;">${rating}</span>
                    </div>
                `;
            }).join('');
        }

        modal.classList.remove('hidden');
        modal.style.display = 'flex'; // Ensure it's visible despite any other classes
    }
};
