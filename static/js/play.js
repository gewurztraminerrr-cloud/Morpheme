// === Game Room Panel Navigation (Mobile) ===
// Fixes the "both panels visible simultaneously" split-screen bug.
//
// Root cause: scrollIntoView({behavior:'smooth'}) + scroll-behavior:smooth in CSS
// lets iOS viewport-resize events (e.g. "exit full screen" banner) interrupt the
// scroll animation midway, freezing the .play-grid between two snap points.
//
// Fix: direct scrollLeft assignment + strict snap enforcement on touchend/resize/visibility.
// =============================================================================

const _PLAY_PANELS = ['players', 'board', 'words'];
window._currentPlayPanel = 'board'; // tracks which panel is currently in view

window.switchPlayPanel = function(panelId) {
    window._currentPlayPanel = panelId || 'board';
    const playGrid = document.querySelector('.play-grid');
    if (!playGrid) return;
    const idx = _PLAY_PANELS.indexOf(window._currentPlayPanel);
    if (idx === -1) return;
    const targetLeft = idx * playGrid.clientWidth;
    playGrid.scrollLeft = targetLeft;
    requestAnimationFrame(() => { playGrid.scrollLeft = targetLeft; });
    setTimeout(() => { playGrid.scrollLeft = targetLeft; }, 50);
};

// Track which panel the user swiped to and enforce clean snapping when scrolling/swiping finishes.
(function _setupPlayGridScrollTracker() {
    function enforceSnap() {
        const playGrid = document.querySelector('.play-grid');
        if (!playGrid) return;
        const panelWidth = playGrid.clientWidth;
        if (!panelWidth || panelWidth <= 0) return;
        
        const idx = Math.round(playGrid.scrollLeft / panelWidth);
        const targetIdx = Math.max(0, Math.min(idx, _PLAY_PANELS.length - 1));
        window._currentPlayPanel = _PLAY_PANELS[targetIdx] || 'board';
        const targetLeft = targetIdx * panelWidth;
        
        if (Math.abs(playGrid.scrollLeft - targetLeft) > 1) {
            playGrid.scrollLeft = targetLeft;
        }
    }

    function _attachTracker() {
        const playGrid = document.querySelector('.play-grid');
        if (!playGrid) return;
        let _scrollTimeout;
        
        playGrid.addEventListener('scroll', () => {
            clearTimeout(_scrollTimeout);
            _scrollTimeout = setTimeout(enforceSnap, 80);
        }, { passive: true });

        // Enforce snap as soon as finger lifts off screen
        ['touchend', 'touchcancel', 'pointerup', 'pointercancel'].forEach(evt => {
            document.addEventListener(evt, () => {
                const playPage = document.getElementById('page-play');
                if (playPage && playPage.classList.contains('active')) {
                    setTimeout(enforceSnap, 50);
                    setTimeout(enforceSnap, 250);
                }
            }, { passive: true });
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _attachTracker);
    } else {
        _attachTracker();
    }
})();

// Re-snap the game panels when the viewport changes (iOS "exit full screen" banner or orientation change).
window.addEventListener('resize', () => {
    const isMobile = window.innerWidth <= 900 || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    if (!isMobile) return;
    if (typeof window.switchPlayPanel === 'function') {
        requestAnimationFrame(() => {
            window.switchPlayPanel(window._currentPlayPanel || 'board');
        });
    }
});

// =============================================================================

let isTournamentPlay = false;

let isPrivateMatchPlay = false;
let wrongGuessesOnBoardCount = 0;
window.isPopupVisible = false;
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
let bestServerTimeRTT = Infinity; // Track the best RTT seen so far
let timerFormatIs24h = false;     // Cached format to prevent flashing between HH:MM:SS and M:SS

// Global audio object to bypass mobile autoplay restrictions
window.intermissionBellAudio = new Audio();

window.updateIntermissionBellSource = function() {
    if (window.intermissionBellAudio) {
        const bellType = (window.userSettings && window.userSettings.next_round_bell_type) || 'bell1';
        window.intermissionBellAudio.src = `/static/audio/${bellType}.wav`;
        try {
            window.intermissionBellAudio.load();
        } catch (e) {
            console.warn('[play.js] Failed to load intermission bell audio:', e);
        }
    }
};

const unlockAudio = () => {
    window.updateIntermissionBellSource();
    const tripleMusic = document.getElementById('triple-music');
    if (tripleMusic) {
        try {
            tripleMusic.load();
        } catch (e) {}
    }
};

document.addEventListener('click', unlockAudio, { once: true });
document.addEventListener('touchstart', unlockAudio, { once: true });

// Sound effects system using Web Audio API
const BoardAudio = {
    ctx: null,
    
    init() {
        if (this.ctx) return;
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) {
            this.ctx = new AudioContext();
        }
    },
    
    playTileSound(pathLen = 1) {
        if (window.userSettings && window.userSettings.board_sounds === false) return;
        if (typeof MorphemeAudioBridge !== 'undefined') {
            try {
                MorphemeAudioBridge.postMessage(JSON.stringify({ sound: 'tile', pathLen: pathLen }));
            } catch (e) {
                console.error("MorphemeAudioBridge error:", e);
            }
            return;
        }
        this.init();
        if (!this.ctx) return;
        
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
        
        try {
            const osc = this.ctx.createOscillator();
            const gainNode = this.ctx.createGain();
            
            osc.connect(gainNode);
            gainNode.connect(this.ctx.destination);
            
            osc.type = 'sine';
            
            // Ascending pitch scaling based on path length (50Hz steps from 400Hz)
            const baseFreq = 400;
            const step = 50;
            const freq = Math.min(1200, baseFreq + (pathLen * step));
            
            osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
            
            gainNode.gain.setValueAtTime(0.08, this.ctx.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.05);
            
            osc.start();
            osc.stop(this.ctx.currentTime + 0.05);
        } catch (e) {
            console.warn('Failed to play tile sound:', e);
        }
    },
    
    playSuccessSound() {
        if (window.userSettings && window.userSettings.board_sounds === false) return;
        if (typeof MorphemeAudioBridge !== 'undefined') {
            try {
                MorphemeAudioBridge.postMessage(JSON.stringify({ sound: 'success' }));
            } catch (e) {
                console.error("MorphemeAudioBridge error:", e);
            }
            return;
        }
        this.init();
        if (!this.ctx) return;
        
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
        
        try {
            const playBeep = (freq, startTime, duration, volume) => {
                const osc = this.ctx.createOscillator();
                const gainNode = this.ctx.createGain();
                
                osc.connect(gainNode);
                gainNode.connect(this.ctx.destination);
                
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(freq, startTime);
                
                gainNode.gain.setValueAtTime(volume, startTime);
                gainNode.gain.exponentialRampToValueAtTime(0.001, startTime + duration);
                
                osc.start(startTime);
                osc.stop(startTime + duration);
            };
            
            const now = this.ctx.currentTime;
            playBeep(523.25, now, 0.08, 0.15); // C5
            playBeep(783.99, now + 0.06, 0.15, 0.15); // G5
        } catch (e) {
            console.warn('Failed to play success sound:', e);
        }
    },
    
    playFailureSound() {
        if (window.userSettings && window.userSettings.board_sounds === false) return;
        if (typeof MorphemeAudioBridge !== 'undefined') {
            try {
                MorphemeAudioBridge.postMessage(JSON.stringify({ sound: 'failure' }));
            } catch (e) {
                console.error("MorphemeAudioBridge error:", e);
            }
            return;
        }
        this.init();
        if (!this.ctx) return;
        
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
        
        try {
            const osc = this.ctx.createOscillator();
            const gainNode = this.ctx.createGain();
            
            osc.connect(gainNode);
            gainNode.connect(this.ctx.destination);
            
            osc.type = 'sawtooth';
            
            osc.frequency.setValueAtTime(150, this.ctx.currentTime);
            osc.frequency.linearRampToValueAtTime(100, this.ctx.currentTime + 0.18);
            
            gainNode.gain.setValueAtTime(0.12, this.ctx.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.18);
            
            osc.start();
            osc.stop(this.ctx.currentTime + 0.18);
        } catch (e) {
            console.warn('Failed to play failure sound:', e);
        }
    }
};

// Warm up and resume AudioContext on user interaction
const initAudioOnUserInteraction = () => {
    BoardAudio.init();
    if (BoardAudio.ctx && BoardAudio.ctx.state === 'suspended') {
        BoardAudio.ctx.resume();
    }
};
document.addEventListener('click', initAudioOnUserInteraction);
document.addEventListener('touchstart', initAudioOnUserInteraction);

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
// Board Transposition: true = 90° portrait rotation for mobile (swaps display cols/rows + iterates [c][r])
let isBoardTransposed = false;
Object.defineProperty(window, 'isBoardTransposed', {
    get: () => isBoardTransposed,
    set: (v) => { isBoardTransposed = v; },
    configurable: true
});

function safelyTransposeState(state) {
    if (!state) return;
    if (state._isAlreadyTransposed) {
        window.isBoardTransposed = !!state._isBoardTransposedValue;
        return;
    }
    
    window.isBoardTransposed = false;
    // Transpose whenever the display is in portrait mode (height > width) — regardless of device type.
    // This ensures the longest board dimension always runs vertically on any portrait screen.
    const isPortraitMode = window.innerHeight > window.innerWidth;
    
    try {
        if (isPortraitMode && state.board && state.board.length > 0 && Array.isArray(state.board[0])) {
            const isBoard3D = state.board_dimensions === '3x3x3' || (state.board.length === 6 && Array.isArray(state.board[0]) && Array.isArray(state.board[0][0]));
            if (!isBoard3D) {
                const rows = state.board.length;
                const cols = state.board[0].length;
                if (rows < cols) {
                    window.isBoardTransposed = true;
                    // Transpose Board letters array safely
                    const transposedBoard = [];
                    for (let c = 0; c < cols; c++) {
                        transposedBoard[c] = [];
                        for (let r = 0; r < rows; r++) {
                            transposedBoard[c][r] = (state.board[r] && state.board[r][c] !== undefined) ? state.board[r][c] : '';
                        }
                    }
                    state.board = transposedBoard;

                    // Transpose previous_board too — during intermission renderBoard is called with
                    // state.previous_board directly, so it must be transposed alongside state.board.
                    // Without this, the board snaps to landscape orientation when the round ends.
                    if (state.previous_board && state.previous_board.length > 0 && Array.isArray(state.previous_board[0])) {
                        const pRows = state.previous_board.length;
                        const pCols = state.previous_board[0].length;
                        if (pRows < pCols) {
                            const transposedPrev = [];
                            for (let c = 0; c < pCols; c++) {
                                transposedPrev[c] = [];
                                for (let r = 0; r < pRows; r++) {
                                    transposedPrev[c][r] = (state.previous_board[r] && state.previous_board[r][c] !== undefined) ? state.previous_board[r][c] : '';
                                }
                            }
                            state.previous_board = transposedPrev;
                        }
                    }

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

                    // Transpose Bonus Cell coordinate safely
                    if (state.bonus_cell) {
                        if (Array.isArray(state.bonus_cell) && state.bonus_cell.length === 2) {
                            state.bonus_cell = [state.bonus_cell[1], state.bonus_cell[0]];
                        } else if (typeof state.bonus_cell === 'object') {
                            if (state.bonus_cell.r !== undefined && state.bonus_cell.c !== undefined) {
                                state.bonus_cell = { r: state.bonus_cell.c, c: state.bonus_cell.r };
                            }
                        }
                    }
                }
            }
        }
    } catch (transpositionError) {
        console.error("[Mobile] Transposition failed safely:", transpositionError);
    }
    state._isAlreadyTransposed = true;
    state._isBoardTransposedValue = window.isBoardTransposed;
}


let activeWordsTab = 'found'; // 'found' or 'remaining'
window._cluesShowRemaining = false;
let validationTimeout = null;
let highlightedSplitWord = null; // Track word for shared highlighting in Split Points
let highlightedFoundWord = null; // Track word from All Words list to highlight finders
window.intermissionTileFilter = null;
let lastRenderedBoardJSON = null;
let lastRenderedGrayed = null;
let lastRenderedRotation = null;
let lastRenderedUserTranspose = null;
let lastRenderedDensityJSON = null;
let hasPlayedIntermissionBell = false; // Flag for next round notification
let playersFilterMode = 'everyone'; // 'everyone', 'friends', 'me'
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

// --- IDLE TRACKING ---
// Reset only on play-page interactions. mousemove excluded. Other tabs do not count.
let lastGameInteractionTime = Date.now();

function resetIdleTimer() {
    lastGameInteractionTime = Date.now();
}

function isOnPlayPage() {
    const playPage = document.getElementById('page-play');
    if (!playPage) return false;
    const style = window.getComputedStyle(playPage);
    return style.display !== 'none' && !playPage.classList.contains('hidden');
}

async function ejectToLobby(reason = "inactivity") {
    console.warn(`[play.js] EVICTING USER. Reason: ${reason}`);

    // 1. Notify server immediately so lobby counts decrease
    if (window.leaveCurrentRoom) {
        try {
            await window.leaveCurrentRoom();
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

    // 4. Build message
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
            A fresh daily board has been generated. Head back in to start finding words!
        `;
    }

    // 5. SHOW MODAL FIRST — over whatever page the user is currently on (Tools, Profile, etc.)
    //    so they always see the notice regardless of which tab they were in.
    if (window.showAlertModal) {
        window.showAlertModal(title, message, true);
        console.log('[play.js] Displayed eviction modal before redirect.');
    } else {
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

    // 6. THEN navigate to lobby after a short delay so the modal is visible first
    setTimeout(() => {
        if (window.navigateToPage) window.navigateToPage('lobby');
        else if (window.showPage) window.showPage('page-lobby');
        else window.location.href = '#page-lobby';
    }, 400);
}

// Reset idle timer only on interactions that happen WHILE on the Play page.
// Nav button clicks and interactions on other tabs (Tools, Profile, etc.) do NOT count.
// mousemove intentionally excluded — fires constantly on desktop, masking true idle.
let lastTouchTime = 0;

document.addEventListener('mousedown', (e) => {
    if (Date.now() - lastTouchTime < 1500) return; // ignore ghost mouse from touch
    updateInputMethod('mouse');
    if (isOnPlayPage()) resetIdleTimer(); // only count play-page clicks
}, true);

document.addEventListener('touchstart', () => {
    lastTouchTime = Date.now();
    updateInputMethod('touch');
    if (isOnPlayPage()) resetIdleTimer(); // only count play-page taps
}, true);

document.addEventListener('keydown', (e) => {
    if (window.isPopupVisible) {
        e.preventDefault();
        e.stopPropagation();
        return;
    }
    // If the event target is an INPUT or TEXTAREA, only toggle to keyboard if it's the game word-input!
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) {
        if (e.target.id !== 'word-input') {
            return;
        }
    }
    updateInputMethod('keyboard');
    if (isOnPlayPage()) resetIdleTimer(); // only count play-page keystrokes
}, true);

// Check for idle logout every 5 seconds.
// Single clock: no play-page interaction in 10 min = eject, regardless of which tab they are on.
setInterval(() => {
    const roomId = getCurrentRoomId();
    if (!roomId) return;

    // EXEMPTION: No idle limit for 24h rooms
    const is24h = window.lastGameState && window.lastGameState.game_type === 'accumulative' && window.lastGameState.time_limit >= 7200;
    if (is24h) return;

    const idleMs = Date.now() - lastGameInteractionTime;
    if (idleMs > 10 * 60 * 1000) { // 10 minutes
        console.warn('[play.js] 10m idle from Play page (' + Math.round(idleMs/1000) + 's). EVICTING.');
        ejectToLobby("inactivity");
    }
}, 5000);

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

/**
 * ensureLoadingCardStyles() — all styles are now in play.css as static rules.
 * Kept as a no-op stub so existing call sites don't break.
 */
function ensureLoadingCardStyles() {
    const boardEl = document.getElementById('game-board');
    if (boardEl) {
        boardEl.style.display = 'flex';
        boardEl.style.flexDirection = 'column';
        boardEl.style.alignItems = 'center';
        boardEl.style.justifyContent = 'center';
        boardEl.style.gridTemplateColumns = 'none';
        boardEl.style.gridTemplateRows = 'none';
        boardEl.style.width = '100%';
    }
}

function clearGameUIAndCache() {
    console.log('[play.js] Clearing Game UI and Caches from previous match');
    
    const tripleMusic = document.getElementById('triple-music');
    if (tripleMusic && !tripleMusic.paused) {
        tripleMusic.pause();
    }
    window.lastGameState = null;
    window.lastDisplayAllWordsArgs = null;
    window.lastRenderedStateJSON = null;
    window.lastRenderedBoardJSON = null;
    window.lastPlayersHtml = null;
    window._wasEverInRoster = false;
    stopBounceFormat();
    stopRotatingLetters();
    
    // Reset local module render caches to force re-render on next entry
    lastRenderedBoardJSON = null;
    lastRenderedGrayed = null;
    lastRenderedRotation = null;
    lastRenderedUserTranspose = null;
    lastRenderedDensityJSON = null;
    
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
        ensureLoadingCardStyles();
        boardEl.className = 'game-board-loading';
        boardEl.removeAttribute('style');
        boardEl.innerHTML = `
            <div class="loading-container">
                <div class="glow-spinner"></div>
                <div class="glow-title">Establishing Server Connection…</div>
                <div class="status-ticker">[PROCESSING] Connecting to Morpheme server…</div>
                <div class="why-text">Preparing your game room</div>
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
    window._wasEverInRoster = false; // Reset roster flag to prevent instantaneous eviction on re-entry
    resetIdleTimer();
    
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    // ONLY clear UI and cache if we are actually switching rooms or if we don't have a lastGameState!
    const activeRoomId = getCurrentRoomId();
    const isSameRoom = window.lastGameState && (window.lastGameState.room_id === activeRoomId);

    if (!isSameRoom) {
        clearGameUIAndCache();
        if (activeRoomId && activeRoomId.startsWith('practice_')) {
            window.isSpectatorMode = false;
        }
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
    // Speed up polling significantly when transitioning or when intermission is about to end
    if (!document.hidden) {
        if (window._rapidTransitionPolling) {
             delay = 100; // High-frequency polling at 0:00 transition
        } else if (window.lastGameState && window.lastGameState.state === 'intermission') {
             const tr = window.lastGameState.time_remaining;
             if (tr < 2.5) {
                  delay = 500; // Poll twice as fast at the very end
             }
        }
    }

    pollInterval = setInterval(updateGameState, delay);
}

function setTimerWaitingState(isWaiting) {
    const timerVal = document.getElementById('timer-value');
    const timerLabel = document.querySelector('.timer-label');
    if (isWaiting) {
        if (timerVal) {
            timerVal.innerHTML = '';
            void timerVal.offsetHeight; // Force reflow
            timerVal.innerHTML = 'WAIT<span class="wait-dot">.</span><span class="wait-dot">.</span><span class="wait-dot">.</span>';
            
            // Force animation restart on next frame to ensure it runs on mobile wake-up
            requestAnimationFrame(() => {
                timerVal.querySelectorAll('.wait-dot').forEach((el, index) => {
                    el.style.animation = 'none';
                    void el.offsetHeight; // trigger reflow
                    el.style.animation = `wait-bounce 1.4s infinite ease-in-out`;
                    el.style.animationDelay = `${index * 0.2}s`;
                });
            });
        }
        if (timerLabel) timerLabel.textContent = "";
    } else {
        if (timerLabel && timerLabel.textContent !== "Time:") {
            timerLabel.textContent = "Time:";
        }
    }
}

// Global Visibility Listener to handle battery management
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        // Tab became visible: Update immediately and restore fast polling
        console.log('[play.js] Tab visible: Restoring fast polling.');

        // Force-abort any stale/stuck in-flight fetch and release lock to prevent queue clogging
        if (window._activeStateFetchController) {
            console.log('[play.js] Tab visible: Aborting stale in-flight state fetch.');
            try { window._activeStateFetchController.abort(); } catch(e) {}
            window._activeStateFetchController = null;
        }
        isFetchingState = false;

        // TOURNAMENT MODE: Handle resume separately — don't poll public room endpoints
        if (isTournamentPlay) {
            const now = Date.now() / 1000;
            if (localEndTime && localEndTime > now) {
                // Time remains — restart the local countdown
                if (timerInterval) clearInterval(timerInterval);
                timerInterval = setInterval(() => {
                    const diff = Math.max(0, Math.ceil(localEndTime - (Date.now() / 1000)));
                    updateSpecialMatchTimer(diff);
                    if (diff <= 0) {
                        clearInterval(timerInterval);
                        finishTournamentTurn();
                    }
                }, 1000);
                const initialDiff = Math.max(0, Math.ceil(localEndTime - now));
                updateSpecialMatchTimer(initialDiff);
            } else if (localEndTime) {
                // Timer already expired while tab was hidden — finish the turn now
                console.log('[play.js] Tournament timer expired while hidden. Finishing turn.');
                if (timerInterval) clearInterval(timerInterval);
                finishTournamentTurn();
            } else {
                setTimerWaitingState(true);
            }
            return; // Do NOT call updateGameState/refreshPollInterval for tournament
        }

        // PRIVATE MATCH MODE: Handle resume separately — don't poll public room endpoints
        if (isPrivateMatchPlay) {
            const activeMatch = JSON.parse(localStorage.getItem('private_match_active'));
            const endTime = activeMatch ? activeMatch.end_time : null;
            const now = Date.now() / 1000;
            if (endTime && endTime > now) {
                startPrivateMatchTimer(endTime);
            } else if (endTime) {
                console.log('[play.js] Private match timer expired while hidden. Finishing turn.');
                if (timerInterval) clearInterval(timerInterval);
                finishPrivateMatchTurn();
            } else {
                setTimerWaitingState(true);
            }
            return; // Do NOT call updateGameState/refreshPollInterval for private match
        }

        // Instant Feedback: Show ticking countdown if active, otherwise show "WAIT..."
        if (localEndTime && localEndTime > (Date.now() / 1000)) {
            if (!timerInterval) {
                timerInterval = setInterval(updateLocalTimer, 500);
            }
            updateLocalTimer();
        } else {
            setTimerWaitingState(true);
        }

        // Add a small 80ms delay before fetching to let the mobile OS restore cellular/Wi-Fi connectivity
        setTimeout(() => {
            if (!document.hidden) {
                updateGameState();
                refreshPollInterval();
            }
        }, 80);

        // Re-sync timer if needed (ONLY if lastGameState is fresh, within 5 seconds)
        if (window.lastGameState && window._lastGameStateFetchedTime && (Date.now() - window._lastGameStateFetchedTime < 5000)) {
            syncTimerWithServer(window.lastGameState);
            updateLocalTimer(); // Instantly update timer display
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

// Window Focus Listener: Provides robust mobile wake-up when focus is gained
window.addEventListener('focus', () => {
    if (!document.hidden) {
        console.log('[play.js] Window focus gained: Checking wake-up update.');

        // Force-abort any stale/stuck in-flight fetch and release lock to prevent queue clogging
        if (window._activeStateFetchController) {
            console.log('[play.js] Focus: Aborting stale in-flight state fetch.');
            try { window._activeStateFetchController.abort(); } catch(e) {}
            window._activeStateFetchController = null;
        }
        isFetchingState = false;

        // TOURNAMENT MODE: Handle resume separately — don't poll public room endpoints
        if (isTournamentPlay) {
            const now = Date.now() / 1000;
            if (localEndTime && localEndTime > now) {
                if (timerInterval) clearInterval(timerInterval);
                timerInterval = setInterval(() => {
                    const diff = Math.max(0, Math.ceil(localEndTime - (Date.now() / 1000)));
                    updateSpecialMatchTimer(diff);
                    if (diff <= 0) {
                        clearInterval(timerInterval);
                        finishTournamentTurn();
                    }
                }, 1000);
                const initialDiff = Math.max(0, Math.ceil(localEndTime - now));
                updateSpecialMatchTimer(initialDiff);
            } else if (localEndTime) {
                console.log('[play.js] Tournament timer expired while hidden (focus). Finishing turn.');
                if (timerInterval) clearInterval(timerInterval);
                finishTournamentTurn();
            } else {
                setTimerWaitingState(true);
            }
            return;
        }

        // PRIVATE MATCH MODE: Handle resume separately — don't poll public room endpoints
        if (isPrivateMatchPlay) {
            const activeMatch = JSON.parse(localStorage.getItem('private_match_active'));
            const endTime = activeMatch ? activeMatch.end_time : null;
            const now = Date.now() / 1000;
            if (endTime && endTime > now) {
                startPrivateMatchTimer(endTime);
            } else if (endTime) {
                console.log('[play.js] Private match timer expired while hidden (focus). Finishing turn.');
                if (timerInterval) clearInterval(timerInterval);
                finishPrivateMatchTurn();
            } else {
                setTimerWaitingState(true);
            }
            return;
        }

        // Instant Feedback: Show ticking countdown if active, otherwise show "WAIT..."
        if (localEndTime && localEndTime > (Date.now() / 1000)) {
            if (!timerInterval) {
                timerInterval = setInterval(updateLocalTimer, 500);
            }
            updateLocalTimer();
        } else {
            setTimerWaitingState(true);
        }

        // Add a small 80ms delay to let the network stack settle
        setTimeout(() => {
            if (!document.hidden) {
                updateGameState();
                refreshPollInterval();
            }
        }, 80);

        if (window.lastGameState && window._lastGameStateFetchedTime && (Date.now() - window._lastGameStateFetchedTime < 5000)) {
            syncTimerWithServer(window.lastGameState);
            updateLocalTimer();
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

// Helper to fetch with timeout, allowing rapid recovery from dead socket connections
async function fetchWithTimeout(url, options = {}, timeoutMs = 1200) {
    if (window._activeStateFetchController) {
        try {
            window._activeStateFetchController.abort();
        } catch (e) {}
    }
    const controller = new AbortController();
    window._activeStateFetchController = controller;
    const id = setTimeout(() => {
        try {
            controller.abort();
        } catch (e) {}
    }, timeoutMs);
    try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        clearTimeout(id);
        if (window._activeStateFetchController === controller) {
            window._activeStateFetchController = null;
        }
        return response;
    } catch (error) {
        clearTimeout(id);
        if (window._activeStateFetchController === controller) {
            window._activeStateFetchController = null;
        }
        throw error;
    }
}

let lastStateFetchTime = 0;
let isFetchingState = false;
window._activeStateFetchController = null;

async function updateGameState(incomingState = null) {
    if (isTournamentPlay || localStorage.getItem('tournament_play_active') || isPrivateMatchPlay || localStorage.getItem('private_match_active')) {
        console.log('[play.js] updateGameState: Discarding poll update because a match session is active');
        return;
    }

    const roomId = getCurrentRoomId();
    if (!roomId) {
        return;
    }

    // Micro-debounce to prevent duplicate parallel fetches within 150ms (common on simultaneous focus/visibility events)
    const now = Date.now();
    if (!incomingState && now - lastStateFetchTime < 150) {
        console.log('[play.js] updateGameState: Suppressing duplicate concurrent state fetch.');
        return;
    }
    if (!incomingState) {
        if (isFetchingState) {
            console.log('[play.js] updateGameState: Fetch already in progress. Skipping.');
            return;
        }
        isFetchingState = true;
        lastStateFetchTime = now;
    }

    try {
        let state;
        let tBefore = null;
        let tAfter = null;
        if (incomingState) {
            state = incomingState;
            window._lastGameStateFetchedTime = Date.now();
        } else {
            // Optimization: Skip fetching if tab has been hidden for a while but not yet reached the 15s pulse
            // This is just extra safety, the refreshPollInterval handles the bulk of it.
            
            // Mobile/Wake-up instant response: If a request takes >600ms (common on suspended mobile tabs due to dead HTTP sockets),
            // abort it and retry immediately. The retry forces a fresh TCP/SSL socket and completes instantly!
            let response;
            try {
                tBefore = Date.now() / 1000;
                response = await fetchWithTimeout(`/api/room/${roomId}/state?_t=${Date.now()}`, { cache: 'no-store' }, 600);
            } catch (err) {
                console.log(`[play.js] Wake-up state fetch timed out or failed, retrying immediately with fresh socket...`);
                try {
                    tBefore = Date.now() / 1000;
                    // Try again with a slightly longer timeout (1800ms) to ensure it doesn't block indefinitely
                    response = await fetchWithTimeout(`/api/room/${roomId}/state?_t=${Date.now()}&retry=true`, { cache: 'no-store' }, 1800);
                } catch (retryErr) {
                    console.log(`[play.js] Wake-up retry fetch also failed:`, retryErr);
                    return; // Let the next scheduled poll handle it
                }
            }
            
            // Check if user left or switched rooms while the network fetch was in-flight
            let activeRoomId = getCurrentRoomId();
            if (activeRoomId !== roomId) {
                console.log(`[play.js] updateGameState: User left or switched rooms (current: ${activeRoomId}, target: ${roomId}) while fetch was in-flight. Ignoring response.`);
                return;
            }

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
            tAfter = Date.now() / 1000;
            window._lastGameStateFetchedTime = Date.now();

            // Check again after parsing the json response
            activeRoomId = getCurrentRoomId();
            if (activeRoomId !== roomId) {
                console.log(`[play.js] updateGameState: User left or switched rooms during JSON parsing. Ignoring state update.`);
                return;
            }
        }

        if (!state) return;

        // Store raw state BEFORE transposition so orientation changes can re-evaluate correctly.
        // window.lastGameState is mutated in-place by safelyTransposeState; if we cloned from it
        // on orientation change, a portrait→landscape rotation would leave the board transposed.
        window.lastRawGameState = JSON.parse(JSON.stringify(state));

        safelyTransposeState(state);

        // Mobile Device Restriction: Cube is not allowed on mobile!
        const isMobile = (window.innerWidth <= 992) || /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
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
        if (window.lastRenderedStateJSON === stateJSON && window.lastRenderedTab === activeWordsTab && !incomingState) {
            // Optimization: Update server heartbeat timestamp even if state is identical
            lastServerUpdate = Date.now();
            
            // If the tab is hidden, we definitely don't need to do anything else
            if (document.hidden) return;
            
            // If visible, we still need to sync the timer reference, but we can skip heavy DOM re-renders
            syncTimerWithServer(state, tBefore, tAfter);
            updateLocalTimer(); // Instantly update the timer display on identical-state resume!
            return;
        }
        window.lastRenderedStateJSON = stateJSON;
        window.lastRenderedTab = activeWordsTab;
        
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
        
        // INTERMISSION BOARD NORMALIZATION:
        // During intermission, renderBoard() uses state.previous_board to display the completed
        // round's board, while reapplyBoardHighlights() uses window.lastGameState.board for DFS
        // path searching. If these two differ even briefly, word highlight clicks will silently
        // fail (path found on wrong board → no cells match). Normalize here so they are identical.
        if (state.state === 'intermission' && state.previous_board && state.previous_board.length > 0) {
            state.board = state.previous_board;
        }

        window.lastGameState = state;  // Store for optimistic updates

        
        // LOADING STATE: Room was just created and board is being generated async
        if (state.state === 'loading') {
            console.log('[play.js] Room is in loading state. Showing loading indicator and fast-polling...');
            const timerVal = document.getElementById('timer-value');
            const timerLabel = document.querySelector('.timer-label');
            if (timerVal) timerVal.textContent = '...';
            if (timerLabel && timerLabel.textContent !== "Time:") timerLabel.textContent = 'Time:';

            const boardEl = document.getElementById('game-board');
            if (boardEl) {
                ensureLoadingCardStyles();
                let loadingMsg = "GENERATING NEXT BOARD…";
                if (state.current_board_format) {
                    loadingMsg = `GENERATING ${state.current_board_format.toUpperCase()}…`;
                }
                boardEl.className = 'game-board-loading';
                ensureLoadingCardStyles();
                boardEl.innerHTML = `
                    <div class="loading-container">
                        <div class="glow-spinner"></div>
                        <div class="glow-title">${loadingMsg}</div>
                        <div class="status-ticker">[PROCESSING] Building grid layout and solver trie…</div>
                        <div class="why-text">Preparing Morpheme game board</div>
                    </div>
                `;
            }
            // Fast-poll until board is ready
            if (pollInterval) clearInterval(pollInterval);
            pollInterval = setInterval(updateGameState, 500);
            return;
        }
        
        // Clear rapid transition polling if state changed
        if (previousState && previousState.state !== state.state) {
            window._rapidTransitionPolling = false;
            refreshPollInterval();
        } else if (state.state === 'intermission') {
            refreshPollInterval();
        }

        // Detect transition to intermission (round end)
        if (previousState && previousState.state === 'active' && state.state === 'intermission') {
            stopRotatingLetters();
            const wordInput = document.getElementById('word-input');
            const chatInput = document.getElementById('chat-input');
            const isMobile = window.innerWidth <= 992;
            
            // User Request: Prevent ALL users from spillover chatting for 2s (Skip for mobile!)
            if (chatInput && !isMobile) {
                chatInput.disabled = true;
                const originalPlaceholder = chatInput.placeholder;
                chatInput.placeholder = "Chat disabled for 2s...";
                
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


            // AUTO-SCROLL TO BOTTOM ON INTERMISSION START (Only if currently on play page)
            const playPage = document.getElementById('page-play');
            // FIX: Use classList.contains('active') because visibility is controlled by class in style.css
            const isShowingPlay = playPage && playPage.classList.contains('active') && window.location.hash === '#page-play';
            
            if (isShowingPlay) {
                requestAnimationFrame(() => {
                    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
                    window.scrollTo({ top: document.body.scrollHeight, left: scrollLeft, behavior: 'smooth' });
                });
            }

            // FORCE TAB TO 'WORDS' ON INTERMISSION START (Only if not already viewing something else)
            if (activeWordsTab !== 'history' && activeWordsTab !== 'remaining') {
                activeWordsTab = 'found';
            }
            window.userViewingDefinitionIntermission = false;
            
            // CLEAR STALE DEFINITIONS (Ensure winner announcement triggers)
            // But don't overwrite if Personal Timer has expired — timer-flash takes priority
            const defPanelCheck = document.querySelector('.definitions-panel');
            const timerExpiredAtTransition = defPanelCheck && defPanelCheck.classList.contains('timer-flash');
            if (!timerExpiredAtTransition) {
                const defContent = document.getElementById('definition-content');
                if (defContent) {
                    defContent.innerHTML = '<p class="placeholder">Select a word to see its definition</p>';
                }
                const defHeader = document.getElementById('definition-header');
                if (defHeader) defHeader.style.display = 'none';
            }

            console.log('[play.js] Transition to Intermission: Forcing Words tab and resetting view state.');
        }

        // WINNER ANNOUNCEMENT LOGIC (Persistent during intermission)
        const defContent = document.getElementById('definition-content');
        const defPanel = document.querySelector('.definitions-panel');
        const defHeader = document.getElementById('definition-header');
        const isViewingDefinition = window.userViewingDefinitionIntermission === true;
        // Personal Timer expiry takes priority — do not overwrite "Time is up!" notice
        const isTimerExpired = defPanel ? defPanel.classList.contains('timer-flash') : false;
        let hasActualWinner = false;

        if (state.state === 'intermission' && state.winners_history) {
            const latest = state.winners_history[0];
            // MANDATE: Only show if the record is for the round that JUST finished AND someone scored
            const isForCurrentRound = latest && latest.round === state.current_round;
            hasActualWinner = isForCurrentRound && (latest.score || 0) > 0;

            const bonusText = (state.bonus_word && String(state.bonus_word).toUpperCase() !== 'NONE') ? state.bonus_word : '';
            const bonusHtml = bonusText ? `<div style="font-size: 0.85rem; color: #fff; opacity: 0.8; margin: 4px 0;">Bonus Word: <span style="color: #ffd700; font-weight: 800; letter-spacing: 1px;">${bonusText.toUpperCase()}</span></div>` : '';

            if (hasActualWinner && defContent && !isViewingDefinition && !isTimerExpired) {
                // If not already showing this round's winner
                const winnerTextIdentifier = `WINNER_R${latest.round}`;
                if (!defContent.innerHTML.includes(winnerTextIdentifier) || defContent.innerHTML.includes('placeholder')) {
                    const winnersList = latest.winners.map(w => w.username).join(' & ');
                    
                    if (defPanel) {
                        const is24H = (state.time_limit >= 7200);
                        if (!is24H) {
                            defPanel.classList.add('winner-flash');
                        }
                        // NOTE: do NOT remove timer-flash here — Personal Timer expiry takes priority
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
            } else if (!hasActualWinner && defContent && !isViewingDefinition && !isTimerExpired && (defContent.innerHTML.includes('CONGRATULATIONS') || defContent.innerHTML.includes('placeholder'))) {
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

            // 24H RESET EVICTION: If we are in a 24H room and midnight reset occurred, eject the user to lobby!
            const midnightReset = state.midnight_reset_occurred;
            if (is24H && midnightReset) {
                console.warn(`[play.js] Midnight daily reset detected in 24h room (midnightReset: ${midnightReset}). Ejecting to lobby.`);
                ejectToLobby("daily_reset");
                return;
            }

            // INSTANTANEOUS EVICTION: If we were previously established in the roster in a non-24h room, kick immediately
            if (window._wasEverInRoster) {
                console.warn('[play.js] Authoritative eviction detected: User missing from roster. Ejecting instantaneously.');
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

            // Show winner announcement to spectators during intermission, otherwise show normal spectator panel
            const showWinnerAnnouncement = (state.state === 'intermission' && hasActualWinner);

            if (showWinnerAnnouncement) {
                if (defContent) defContent.style.display = '';
                if (defHeader) defHeader.style.display = 'none';
                if (spectatorPanel) spectatorPanel.style.display = 'none';
            } else {
                if (defContent) defContent.style.display = 'none';
                if (defHeader) defHeader.style.display = 'none';
                if (spectatorPanel) spectatorPanel.style.display = 'flex';
            }

            // Check if there is space to join
            const playerCount = (state.players && Array.isArray(state.players)) ? state.players.length : 0;
            const maxPlayers = state.max_players || 8;
            const isAccumulative = state.game_type === 'accumulative';
            const canJoin = isAccumulative || (playerCount < maxPlayers);

            // Determine spectator rating limits
            const currentUsername = state.your_username || window.currentUser || localStorage.getItem('morpheme_username');
            let myRating = window.lastPlayerRating;
            if (myRating === undefined || myRating === null || isNaN(myRating)) {
                myRating = (currentUsername && currentUsername.startsWith('Guest_')) ? 0 : 1000;
            }
            if (state.spectators && Array.isArray(state.spectators) && currentUsername) {
                const meSpec = state.spectators.find(s => s.username && s.username.toLowerCase() === currentUsername.toLowerCase());
                if (meSpec && meSpec.rating !== undefined && meSpec.rating !== null) {
                    myRating = meSpec.rating;
                }
            }

            const minRating = state.min_rating !== undefined ? state.min_rating : 0;
            const maxRating = state.max_rating !== undefined ? state.max_rating : 9999;
            const isWithinLimits = (myRating >= minRating && myRating <= maxRating);

            // Render Content
            if (spectatorPanel) {
                if (isWithinLimits) {
                    const slotOpen = playerCount < 8;
                    spectatorPanel.innerHTML = `
                        <div class="spectator-title">SPECTATING</div>
                        <div class="spectator-actions" style="flex-direction: column; align-items: center; gap: 8px;">
                            ${slotOpen ?
                            `<button id="spec-join-btn" class="spectator-join-btn premium-btn">Join Game</button>
                             <div class="spectator-slot-open" style="font-size: 0.8rem; color: #10b981; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; display: flex; align-items: center; gap: 4px; animation: pulse 2s infinite;">
                                <span style="display: inline-block; width: 6px; height: 6px; background-color: #10b981; border-radius: 50%;"></span>
                                Slot Open - Join When Ready
                             </div>` :
                            `<div class="spectator-full-badge">Full Room</div>`
                        }
                        </div>
                    `;
                } else {
                    spectatorPanel.innerHTML = `
                        <div class="spectator-title">SPECTATING</div>
                    `;
                }

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
        syncTimerWithServer(state, tBefore, tAfter);
        updateLocalTimer();

        // Track last successful server update
        lastServerUpdate = Date.now();

        // Render board
        // Render board
        const isSplitIntermission = (state.game_type === 'split' && state.state === 'intermission');
        const isFCFSIntermission = (state.game_type === 'fcfs' && state.state === 'intermission');

        // Cleanup split/FCFS board toggle button when not in split/FCFS intermission
        if (!(isSplitIntermission || isFCFSIntermission)) {
            const existingBtn = document.getElementById('toggle-board-btn');
            if (existingBtn) {
                existingBtn.remove();
            }
        }

        if (isSplitIntermission && !showBoardInSplitIntermission) {
            renderSplitNotepads(state.players, state);
        } else if (isFCFSIntermission && !showBoardInSplitIntermission) {
            renderFCFSNotepads(state.players, state);
        } else {
            // Render previous day's board if the user is on the previous tab
            let grayed = state.state === 'intermission';
            let boardToRender = state.board;
            if (state.state === 'intermission' && state.previous_board && state.previous_board.length > 0) {
                boardToRender = state.previous_board;
            } else if (activeWordsTab === 'previous' && state.previous_board && state.previous_board.length > 0) {
                boardToRender = state.previous_board;
                grayed = true;
            }
            if ((!boardToRender || boardToRender.length === 0) && window.lastGameState && window.lastGameState.board) {
                boardToRender = window.lastGameState.board;
            }
            const is3D = state.game_type === '3d' || (boardToRender && boardToRender.length === 6 && Array.isArray(boardToRender[0]) && Array.isArray(boardToRender[0][0]));
            renderBoard(boardToRender, grayed, is3D, state);
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
            if (rotateBtn) {
                if (isSplitIntermission || isFCFSIntermission) {
                    rotateBtn.style.display = 'none';
                } else {
                    rotateBtn.style.display = '';
                }
            }

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
            window.lastRenderedIntermissionWords = currentRoundId;
            window.lastSolvingComplete = state.solving_complete;
        } else if (state.state !== 'intermission') {
            window.lastRenderedIntermissionWords = null;
            window.lastSolvingComplete = false;
            window.lastRenderedIntermissionKey = null; // Clear render cache key outside intermission
            const filterContainer = document.getElementById('length-filter-container');
            if (filterContainer) filterContainer.style.display = 'none';
            const findersContainer = document.getElementById('finders-button-container');
            if (findersContainer) findersContainer.style.display = 'none';
        }


        
        // Check for state transitions (Cleanup/Misc)
        const roomChanged = previousState && state.room_id !== previousState.room_id;

        // Issue 8: Detect rejoin (previousState is null but we have a stored round from sessionStorage)
        // If round changed since we last played this room, wipe local submitted words
        const _sessionKey = `morpheme_last_round_${state.room_id}`;
        const _storedRound = parseInt(sessionStorage.getItem(_sessionKey) || '0', 10);
        const _isRejoin = !previousState && state.state === 'active';
        if (_isRejoin && _storedRound > 0 && state.current_round !== _storedRound) {
            console.log(`[play.js] Rejoin round mismatch: was ${_storedRound}, now ${state.current_round}. Clearing local submitted words.`);
            window._localSubmittedWords = new Set();
            window._localSubmittedWordsList = [];
            window.lastDisplayAllWordsArgs = null;
        }
        if (state.state === 'active') {
            sessionStorage.setItem(_sessionKey, state.current_round);
        }

        const isNewRound = (state.state === 'active' && (lastStateStr !== 'active' || roomChanged || (previousState && state.current_round !== previousState.current_round)));
        if (lastStateStr !== state.state || isNewRound) {
            if (isNewRound) {
                wrongGuessesOnBoardCount = 0;
                window._localSubmittedWords = new Set();
                window._localSubmittedWordsList = [];
                // Clear word panel render cache for new round to prevent stale pre-validation
                window.lastDisplayAllWordsArgs = null;

                // Handle Rotation format
                const bFormat = state.current_board_format || '';
                if (bFormat.toLowerCase().includes('rotat')) {
                    startRotatingLetters();
                } else {
                    stopRotatingLetters();
                }
                
                // Handle Bounce format
                const is3D = state.board_dimensions && state.board_dimensions.includes('3x3x3');
                if (bFormat.toLowerCase().includes('bounce') && !is3D) {
                    setTimeout(startBounceFormat, 300);
                } else {
                    stopBounceFormat();
                }

                // Clear any winner announcement from Definitions Panel
                const defContent = document.getElementById('definition-content');
                const defHeader = document.getElementById('definition-header');
                const defPanel = document.querySelector('.definitions-panel');
                const timerStillExpired = defPanel && defPanel.classList.contains('timer-flash');
                // Only reset definition content if Personal Timer is NOT in expired state
                if (!timerStillExpired) {
                    if (defContent) defContent.innerHTML = '<p class="placeholder">Select a word to see its definition</p>';
                    if (defHeader) defHeader.style.display = 'none';
                }
                if (defPanel) {
                    // Keep timer-flash alive across rounds — Personal Timer expiry persists until user stops it
                    defPanel.classList.remove('winner-flash');
                }
                window.userViewingDefinitionIntermission = false;

                // Reset mouse selection state on new round to prevent stale swipes
                if (typeof mouseState !== 'undefined') {
                    mouseState.isDown = false;
                    mouseState.selectedPath = [];
                    if (mouseState.visitedCells) mouseState.visitedCells.clear();
                }

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

                // Reset Players Filter mode
                playersFilterMode = 'everyone';

                window.intermissionTileFilter = null;
                document.querySelectorAll('.board-cell.intermission-highlight').forEach(el => {
                    el.classList.remove('intermission-highlight');
                });
                const existingFilterBtn = document.getElementById('intermission-filter-btn-container');
                if (existingFilterBtn) {
                    existingFilterBtn.remove();
                }

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

                    const isMobile = window.innerWidth <= 992;
                    if (!isMobile) {
                        if (wordInput) wordInput.focus();
                    } else {
                        // Mobile Device: Do NOT auto-focus the textbox (prevents keyboard from popping up).
                        // Instead, scroll the timer into view at the very top of the panel.
                        setTimeout(() => {
                            const timerDisplay = document.querySelector('.timer-display');
                            if (timerDisplay) {
                                timerDisplay.scrollIntoView({ behavior: 'smooth', block: 'start' });
                            }
                        }, 100); // Small delay to let board rendering settle
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
                // 24H: Found, Clues, Previous, Score Sum
                if (tab === 'found') {
                    btn.textContent = 'Found';
                    btn.style.display = 'block';
                } else if (tab === 'clues' || tab === 'previous' || tab === 'score-sum') {
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
        if (!is24H && (activeWordsTab === 'clues' || activeWordsTab === 'previous' || activeWordsTab === 'score-sum')) {
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

        // Toggle Definitions Panel Visibility (Confined to Found/Words tab on all devices)
        if (defPanel) {
            defPanel.style.display = activeWordsTab === 'found' ? '' : 'none';
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

        const allWords = state.state === 'intermission' ? ((state.previous_all_words && state.previous_all_words.length > 0) ? state.previous_all_words : (state.all_words || [])) : (state.all_words || []);
        const cswForList = state.state === 'intermission' ? ((state.previous_csw_only_words && state.previous_csw_only_words.length > 0) ? state.previous_csw_only_words : state.csw_only_words) : state.csw_only_words;
        const addedForList = state.state === 'intermission' ? ((state.previous_added_words && state.previous_added_words.length > 0) ? state.previous_added_words : state.added_words) : state.added_words;

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
                // Old finders button removed per user request

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

                const targetUsernameStr = (selectedPlayerUsername || currentUser || '').toLowerCase().trim();
                const currentUserIdStr = String(window.currentUserId || '');
                let targetPlayer = state.players ? state.players.find(p => 
                    (p.username && p.username.toLowerCase().trim() === targetUsernameStr) ||
                    (p.user_id && String(p.user_id) === currentUserIdStr)
                ) : null;
                if (!targetPlayer && state.previous_players) {
                    targetPlayer = state.previous_players.find(p => 
                        (p.username && p.username.toLowerCase().trim() === targetUsernameStr) ||
                        (p.user_id && String(p.user_id) === currentUserIdStr)
                    );
                }
                let targetWords = targetPlayer && targetPlayer.submitted_words ? targetPlayer.submitted_words.map(w => typeof w === 'string' ? w : w.word) : [];
                
                // Merge locally submitted words so all found words (including CSW & 5x7 words) are highlighted in blue
                if (!selectedPlayerUsername && window._localSubmittedWords) {
                    const existingSet = new Set(targetWords.map(w => w.toUpperCase()));
                    for (const lw of window._localSubmittedWords) {
                        const lwUpper = lw.toUpperCase();
                        if (!existingSet.has(lwUpper)) {
                            targetWords.push(lw);
                            existingSet.add(lwUpper);
                        }
                    }
                }

                const uniqueGlobalFound = [...new Set(allPlayerFoundStrs)];
                const bonusForList = state.previous_bonus_word || state.bonus_word;


                const roundId = `${state.room_id}_${state.current_round}`;
                const filterJSON = JSON.stringify(window.intermissionTileFilter || null);
                const selectedLen = window.selectedAllWordsLength || 'all';
                const currentRenderKey = `${roundId}_${activeWordsTab}_${state.solving_complete}_${filterJSON}_${selectedLen}_${selectedPlayerUsername || ''}_${highlightedFoundWord || ''}`;

                if (window.lastRenderedIntermissionKey !== currentRenderKey) {
                    displayAllWords(allWords, bonusForList, targetWords, uniqueGlobalFound, state.all_word_scores, cswForList, addedForList);
                    window.lastRenderedIntermissionKey = currentRenderKey;
                }
                if (state.game_type === 'split' || state.game_type === 'fcfs') addSplitViewBoardToggle();

            } else if (state.game_type !== 'fcfs') {
                // ACTIVE STATE (Not Intermission) & Not FCFS
                // Personal List for Standard, Split, AND Accumulative
                const currentUserIdStr = String(window.currentUserId || currentUser || '');
                const myPlayer = state.players.find(p => 
                    (p.username && p.username.toLowerCase().trim() === (currentUser || "").toLowerCase().trim()) ||
                    (p.user_id && String(p.user_id) === currentUserIdStr)
                );
                
                let myWords = myPlayer ? [...(myPlayer.submitted_words || [])] : [];
                
                // SAFEGUARD: Preserve locally submitted words to prevent polls from wiping active words
                if (window._localSubmittedWordsList && window._localSubmittedWordsList.length > 0) {
                    const serverWordSet = new Set();
                    myWords.forEach(w => {
                        const str = (typeof w === 'string' ? w : (w && w.word) || '').toUpperCase().trim();
                        if (str) serverWordSet.add(str);
                    });
                    
                    for (const localObj of window._localSubmittedWordsList) {
                        const lWord = (typeof localObj === 'string' ? localObj : (localObj && localObj.word) || '').toUpperCase().trim();
                        if (lWord && !serverWordSet.has(lWord)) {
                            myWords.push(localObj);
                            serverWordSet.add(lWord);
                        }
                    }
                }


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
                        const pointsObj = (typeof w === 'object' && w !== null && w.score_details)
                            ? w.score_details
                            : (state.all_word_scores && state.all_word_scores[wordUpper]);
                        
                        let ptsNum = typeof w === 'string' ? 0 : (w.points || 0);
                        if (typeof pointsObj === 'object' && pointsObj !== null && pointsObj.total !== undefined) {
                            ptsNum = pointsObj.total;
                        } else if (typeof state.all_word_scores[wordUpper] === 'number') {
                            ptsNum = state.all_word_scores[wordUpper];
                        }
                        
                        let ptsDisplay = ptsNum;

                        if (typeof pointsObj === 'object' && pointsObj !== null) {
                            const bonusWordPts = pointsObj.bonus_word_points || 0;
                            const bonusLetterPts = pointsObj.bonus_letter_points || 0;
                            const eoPts = pointsObj.either_or_points || 0;
                            const basePts = pointsObj.base !== undefined ? pointsObj.base : (ptsNum - bonusWordPts - bonusLetterPts - eoPts);

                            let parts = [basePts];
                            if (bonusWordPts > 0) parts.push(bonusWordPts);
                            if (bonusLetterPts > 0) parts.push(bonusLetterPts);
                            if (eoPts > 0) parts.push(eoPts);

                            if (parts.length > 1) {
                                ptsDisplay = `${parts.join(' + ')} = ${ptsNum}`;
                            } else {
                                ptsDisplay = ptsNum;
                            }
                        }

                        const isBonus = state.bonus_word && wordUpper === state.bonus_word.toUpperCase();
                        const isCSWOnly = cswForList && cswForList.some(csw => csw.toUpperCase() === wordUpper);
                        const isAddedWord = addedForList && addedForList.some(aw => aw.toUpperCase() === wordUpper);

                        let className = 'word-item player-word';
                        if (isBonus) {
                            className += ' bonus-word';
                        } else if (isAddedWord) {
                            className += ' added-word';
                        } else if (isCSWOnly) {
                            className += ' csw-only';
                        }
                        if (ptsNum < 0) className += ' penalty-word';
                        if (highlightedFoundWord === wordUpper) className += ' finder-active';

                        // All words in this list ARE found by user
                        const indicator = '<span class="found-indicator present">✓</span>';

                        return `<div class="${className}" data-word="${word}" style="display:flex; justify-content:space-between; cursor:pointer;">
                            <span>${indicator}${word}</span>
                            <span style="opacity:0.8">${ptsDisplay}</span>
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
                        const isAddedWord = addedForList && addedForList.some(aw => aw.toUpperCase() === wordUpper);
                        const isCSWOnly = cswForList && cswForList.some(csw => csw.toUpperCase() === wordUpper);

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
        const cluesListEl = document.getElementById('clues-list');
        const remainingListEl = document.getElementById('remaining-words-list');
        const showRemainingInClues = is24H && activeWordsTab === 'clues' && window._cluesShowRemaining;
        if ((remainingListEl && activeWordsTab === 'remaining') || showRemainingInClues) {
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
                // Allow ±1 tolerance: at the exact transition moment the tag may lag by 1 poll cycle.
                const expectedRound = state.current_round;
                const roundDiff = totalByLen._round !== undefined ? Math.abs(totalByLen._round - expectedRound) : 0;
                
                if (totalByLen._round !== undefined && roundDiff > 1) {
                    console.warn(`[Remaining-Sync] Mismatch (Counts Round: ${totalByLen._round}, Expected: ${expectedRound}).`);
                    const targetEl = showRemainingInClues ? cluesListEl : remainingListEl;
                    if (targetEl) {
                        targetEl.innerHTML = '<p class="placeholder" style="opacity: 0.6; font-style: italic;">Syncing word counts...</p>';
                    }
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
            
            const targetEl = showRemainingInClues ? cluesListEl : remainingListEl;
            if (targetEl) {
                targetEl.innerHTML = html;
                if (showRemainingInClues) {
                    targetEl.style.display = 'block';
                    targetEl.style.width = '100%';
                }
            }
        }

        // --- CLUES TAB (24H Only) ---
        const cluesToggleBtn = document.getElementById('clues-toggle-remaining-btn');
        if (cluesToggleBtn) {
            cluesToggleBtn.style.display = is24H ? 'block' : 'none';
            cluesToggleBtn.textContent = window._cluesShowRemaining ? 'Return to Clues' : 'Remaining';
        }

        if (cluesListEl && activeWordsTab === 'clues') {
            if (!window._cluesShowRemaining) {
                cluesListEl.style.display = '';
                cluesListEl.style.width = '';
                const oldScrollTop = cluesListEl.scrollTop;
                
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
                    
                    cluesListEl.innerHTML = clueListHtml;
                }
                
                cluesListEl.scrollTop = oldScrollTop;
            }
        }

        // --- PREVIOUS DAY TAB (24H Only) ---
        const prevListEl = document.getElementById('previous-words-list');
        if (prevListEl && activeWordsTab === 'previous') {
            const oldScrollTop = prevListEl.scrollTop;
            
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
                    
                    const playerWordObj = myPrevWords.find(pw => (typeof pw === 'string' ? pw : pw.word).toUpperCase() === w.toUpperCase());
                    const details = (playerWordObj && playerWordObj.score_details)
                        ? playerWordObj.score_details
                        : (state.previous_all_word_scores && state.previous_all_word_scores[w]);
                    const totalPoints = (playerWordObj && playerWordObj.points !== undefined)
                        ? playerWordObj.points
                        : (details ? (details.total || 0) : 0);

                    let ptsDisplay = '';
                    if (details) {
                        const total = totalPoints;
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

                const isDaily = state.time_limit >= 7200;

                if (!isDaily) {
                    // Found Section (only in standard/tournament non-24h rooms)
                    html += `<div style="padding:10px; background:rgba(0,0,0,0.1); font-weight:bold; color:#4a90e2;">FOUND (${foundList.length})</div>`;
                    if (foundList.length > 0) {
                        html += foundList.map(w => renderRow(w, true)).join('');
                    } else {
                        html += `<div style="padding:15px; text-align:center; font-style:italic; opacity:0.6;">None</div>`;
                    }
                }

                // Missed Section
                const missedMargin = isDaily ? '' : ' margin-top:10px;';
                html += `<div style="padding:10px; background:rgba(0,0,0,0.1); font-weight:bold;${missedMargin} color:#888;">MISSED (${missedList.length})</div>`;
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
            
            prevListEl.scrollTop = oldScrollTop;
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
                                <button title="Watch Replay" onclick="event.stopPropagation(); watchRoundHistory('${state.room_id}', ${h.round}, false)" style="background:none; border:none; color:#ffd700; cursor:pointer; font-size:1.3rem; padding:0; display:flex; align-items:center;">▶</button>
                            </div>
                        </div>
                    `;
                }).join('');
            }
        }

        // --- SCORE SUM TAB (24H Only) ---
        const scoreSumListEl = document.getElementById('score-sum-list');
        if (scoreSumListEl && activeWordsTab === 'score-sum') {
            if (!window._dailyScoreSumsData) {
                fetchDailyScoreSums();
            } else {
                renderDailyScoreSums();
            }
        }


        // Auto-focus check
        if (isActive && !previousState) {
            const inputField = document.getElementById('word-input');
            const isMobile = window.innerWidth <= 992;
            if (inputField && !isMobile) {
                setTimeout(() => inputField.focus(), 50);
            }
        }

        // Rapid retry if server is lagged (stayed in active or intermission state but time is up)
        if (state && (state.state === 'active' || state.state === 'intermission') && state.time_remaining <= 0) {
            console.warn(`[play.js] Server state is ${state.state} with 0s left. Scheduling rapid state check in 200ms...`);
            if (!window._rapidPollCount || window._rapidPollCount < 20) {
                window._rapidPollCount = (window._rapidPollCount || 0) + 1;
                setTimeout(() => updateGameState(), 100);
            } else {
                window._rapidPollCount = 0;
            }
        } else {
            window._rapidPollCount = 0;
        }

    } catch (error) {
        console.error('Error updating game state:', error);
    } finally {
        if (!incomingState) {
            isFetchingState = false;
        }
    }
}

function fetchDailyScoreSums() {
    const listEl = document.getElementById('score-sum-list');
    if (listEl) {
        listEl.innerHTML = '<p class="placeholder">Loading rankings...</p>';
    }
    
    fetch('/api/daily-score-sums')
        .then(res => res.json())
        .then(data => {
            window._dailyScoreSumsData = data.players || [];
            renderDailyScoreSums();
        })
        .catch(err => {
            console.error('Error fetching score sums:', err);
            if (listEl) {
                listEl.innerHTML = '<p class="placeholder" style="color: var(--theme-danger);">Failed to load rankings</p>';
            }
        });
}

function renderDailyScoreSums() {
    const listEl = document.getElementById('score-sum-list');
    if (!listEl) return;
    
    const players = window._dailyScoreSumsData || [];
    const searchInput = document.getElementById('score-sum-search');
    const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
    
    const filteredPlayers = players.filter(p => p.username.toLowerCase().includes(query));
    
    // Update player count at the top next to "Find Me"
    const countEl = document.getElementById('score-sum-player-count');
    if (countEl) {
        countEl.textContent = `Players: ${players.length}`;
    }
    
    if (filteredPlayers.length === 0) {
        listEl.innerHTML = '<p class="placeholder">No players found</p>';
        return;
    }
    
    const currentUserNameNormalized = window.currentUser ? window.currentUser.toLowerCase().trim() : '';
    
    let html = '<div class="score-sum-table">';
    filteredPlayers.forEach(p => {
        // Find the absolute rank of this player in the main unfiltered list
        const absRank = players.findIndex(orig => orig.username === p.username) + 1;
        const isMe = p.username.toLowerCase().trim() === currentUserNameNormalized;
        
        html += `
            <div class="score-sum-row ${isMe ? 'is-me-row' : ''}" data-username="${p.username.toLowerCase().trim()}" style="display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; border-radius: 6px; margin-bottom: 4px; background: ${isMe ? 'rgba(var(--text-primary-rgb), 0.15)' : 'var(--input-bg)'}; border: 1px solid ${isMe ? 'var(--text-primary)' : 'var(--input-border)'};">
                <span class="player-rank" style="font-weight: bold; width: 45px; opacity: 0.8;">#${absRank}</span>
                <span class="player-name" style="flex: 1; text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; ${isMe ? 'font-weight: bold;' : ''}">${p.username}</span>
                <span class="player-score" style="font-weight: bold;">${p.score_sum} pts</span>
            </div>
        `;
    });
    html += '</div>';
    
    listEl.innerHTML = html;
}

function renderPlayers(players, currentUser = null, state = null) {
    console.log('[RenderPlayers] Players:', players && players.length);
    const listEl = document.getElementById('players-list');
    const headingEl = document.getElementById('players-heading');
    const findMeBtn = document.getElementById('find-me-btn');
    const findFriendsBtn = document.getElementById('find-friends-btn');
    const showEveryoneBtn = document.getElementById('show-everyone-btn');

    if (state && state.game_type === 'accumulative') {
        let activePlayerCount = 0;
        if (state.state === 'intermission' || state.intermission === true) {
            // Intermission: Count only players who actively participated (did not DNP)
            activePlayerCount = players ? players.filter(p => 
                p.words_count > 0 || p.score > 0 || (p.invalid_words && p.invalid_words.length > 0)
            ).length : 0;
        } else {
            // Active: Count all active players currently in the room (excluding spectators)
            activePlayerCount = players ? players.length : 0;
        }
        if (headingEl) headingEl.textContent = `Players [${activePlayerCount}]`;
        if (findMeBtn) findMeBtn.style.display = 'block';
        if (findFriendsBtn) findFriendsBtn.style.display = 'block';
        if (showEveryoneBtn) showEveryoneBtn.style.display = 'block';

        // Update active highlight classes dynamically
        if (findMeBtn) findMeBtn.classList.toggle('active', playersFilterMode === 'me');
        if (findFriendsBtn) findFriendsBtn.classList.toggle('active', playersFilterMode === 'friends');
        if (showEveryoneBtn) showEveryoneBtn.classList.toggle('active', playersFilterMode === 'everyone');
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

    if (playersFilterMode === 'friends' && currentUser) {
        itemsToRender = itemsToRender.filter(p =>
            p.username.toLowerCase() === (currentUser || "").toLowerCase() ||
            userFriendsCache.some(f => f.username.toLowerCase() === p.username.toLowerCase())
        );
    } else if (playersFilterMode === 'me' && currentUser) {
        const myIdx = itemsToRender.findIndex(p => p.username.toLowerCase() === (currentUser || "").toLowerCase());
        if (myIdx !== -1) {
            const start = Math.max(0, myIdx - 2);
            const end = Math.min(itemsToRender.length, myIdx + 3);
            itemsToRender = itemsToRender.slice(start, end);
        }
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
                <span class="player-flag">${window.getFlagHtml ? window.getFlagHtml(p.country_flag) : (p.country_flag || '🏳️')}</span>
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
            
            // If in intermission state (end of round), open the notepad popup!
            if (window.lastGameState && window.lastGameState.state === 'intermission') {
                if (typeof window.showNotepadPopup === 'function') {
                    window.showNotepadPopup(username);
                }
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

        // Determine User Color from Rating Cache or message property
        let rating = msg.rating;
        if (rating === undefined || rating === null) {
            rating = playerRatingCache.get(username);
        }

        let userColor = '#a8d5ff'; // Default blue-ish
        if (window.getRatingColor) {
            if (rating !== undefined && rating !== null) {
                userColor = window.getRatingColor(rating);
            }
        }

        let ratingSuffix = '';
        if (rating !== undefined && rating !== null) {
            ratingSuffix = ` (${rating})`;
        }

        return `
        <div class="chat-message">
            <span class="chat-user" data-username="${username}" style="color: ${userColor};">${username}${ratingSuffix}:</span>
            <span class="chat-text">${safeText}</span>
        </div>`;
    }).join('');

    listEl.innerHTML = html;

    // Add click listeners to usernames
    listEl.querySelectorAll('.chat-user').forEach(userEl => {
        userEl.style.cursor = 'pointer';
        userEl.title = "View profile";
        userEl.onclick = () => {
            const username = userEl.getAttribute('data-username');
            if (window.showMiniProfile && username) {
                window.showMiniProfile(username);
            } else {
                const rawName = userEl.innerText.trim();
                const cleanName = rawName.endsWith(':') ? rawName.slice(0, -1) : rawName;
                if (window.showMiniProfile) window.showMiniProfile(cleanName);
            }
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

    // --- CHAT PANEL EXPAND / COLLAPSE SYSTEM ---
    const chatPanel = document.querySelector('.chat-panel');
    const leftPanelContainer = document.querySelector('.left-panel-container');
    const collapseBtn = document.getElementById('chat-collapse-btn');

    if (chatPanel) {
        chatPanel.addEventListener('click', (e) => {
            // Do not expand if the user clicked the text input or input section controls
            if (e.target.closest('.chat-input-section')) {
                return;
            }

            // Avoid re-expanding or stealing focus if already expanded
            if (chatPanel.classList.contains('expanded')) {
                return;
            }

            chatPanel.classList.add('expanded');
            if (leftPanelContainer) {
                leftPanelContainer.classList.add('chat-expanded');
            }
            if (collapseBtn) {
                collapseBtn.style.display = 'block';
            }

            // Automatically focus input on expand if we didn't click inside it
            if (chatInput && e.target !== chatInput) {
                chatInput.focus();
            }
        });
    }

    if (collapseBtn) {
        collapseBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent re-expanding
            chatPanel.classList.remove('expanded');
            if (leftPanelContainer) {
                leftPanelContainer.classList.remove('chat-expanded');
            }
            collapseBtn.style.display = 'none';
        });
    }

    // Collapse when clicking outside the chatbox
    document.addEventListener('click', (e) => {
        if (chatPanel && chatPanel.classList.contains('expanded')) {
            if (!chatPanel.contains(e.target)) {
                chatPanel.classList.remove('expanded');
                if (leftPanelContainer) {
                    leftPanelContainer.classList.remove('chat-expanded');
                }
                if (collapseBtn) {
                    collapseBtn.style.display = 'none';
                }
            }
        }
    });

    // Escape key listener to close chat panel
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && chatPanel && chatPanel.classList.contains('expanded')) {
            chatPanel.classList.remove('expanded');
            if (leftPanelContainer) {
                leftPanelContainer.classList.remove('chat-expanded');
            }
            if (collapseBtn) {
                collapseBtn.style.display = 'none';
            }
            if (chatInput) {
                chatInput.blur();
            }
        }
    });

    // Find Me button logic
    const findMeBtn = document.getElementById('find-me-btn');
    if (findMeBtn) {
        findMeBtn.addEventListener('click', () => {
            playersFilterMode = 'me';
            
            // Re-render players
            if (window.lastGameState) {
                const currentUsername = window.lastGameState.your_username || window.currentUser || localStorage.getItem('morpheme_username');
                renderPlayers(window.lastGameState.players, currentUsername, window.lastGameState);
            }

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
                    behavior: 'instant' 
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
            playersFilterMode = 'friends';
            
            // Fetch friends list if not loaded
            if (userFriendsCache.length === 0) {
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
            playersFilterMode = 'everyone';

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

function rebuildTileToWordsMap() {
    window.tileToWordsMap = {};
    window.tileToWordsMapCacheKey = null;

    if (!window.lastGameState || !window.lastGameState.board) return;

    const board = window.lastGameState.board;
    const isTransposed = !!window.isBoardTransposed;
    const is3D = board.length === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]);
    const roundId = `${window.lastGameState.room_id}_${window.lastGameState.current_round}`;
    const cacheKey = `${roundId}_${isTransposed}`;

    const paths = window.lastGameState.all_words_paths || {};

    // 3D Neighbors Helper matching board_generator.py logic
    function getCubeNeighbors(f, r, c) {
        const res = [];
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const nr = r + dr, nc = c + dc;
                if (nr >= 0 && nr < 3 && nc >= 0 && nc < 3) res.push({ f, r: nr, c: nc });
            }
        }
        if (f === 0) {
            res.push({ f: 4, r: 2, c }, { f: 4, r: 2, c: c - 1 }, { f: 4, r: 2, c: c + 1 });
            res.push({ f: 5, r: 0, c }, { f: 5, r: 0, c: c - 1 }, { f: 5, r: 0, c: c + 1 });
            res.push({ f: 2, r, c: 2 }, { f: 2, r: r - 1, c: 2 }, { f: 2, r: r + 1, c: 2 });
            res.push({ f: 3, r, c: 0 }, { f: 3, r: r - 1, c: 0 }, { f: 3, r: r + 1, c: 0 });
        } else if (f === 1) {
            res.push({ f: 4, r: 0, c: 2 - c }, { f: 4, r: 0, c: 2 - (c - 1) }, { f: 4, r: 0, c: 2 - (c + 1) });
            res.push({ f: 5, r: 2, c: 2 - c }, { f: 5, r: 2, c: 2 - (c - 1) }, { f: 5, r: 2, c: 2 - (c + 1) });
            res.push({ f: 3, r, c: 2 }, { f: 3, r: r - 1, c: 2 }, { f: 3, r: r + 1, c: 2 });
            res.push({ f: 2, r, c: 0 }, { f: 2, r: r - 1, c: 0 }, { f: 2, r: r + 1, c: 0 });
        } else if (f === 2) {
            res.push({ f: 4, r: c, c: 0 }, { f: 4, r: c - 1, c: 0 }, { f: 4, r: c + 1, c: 0 });
            res.push({ f: 5, r: 2 - c, c: 0 }, { f: 5, r: 2 - (c - 1), c: 0 }, { f: 5, r: 2 - (c + 1), c: 0 });
            res.push({ f: 0, r, c: 0 }, { f: 0, r: r - 1, c: 0 }, { f: 0, r: r + 1, c: 0 });
            res.push({ f: 1, r, c: 2 }, { f: 1, r: r - 1, c: 2 }, { f: 1, r: r + 1, c: 2 });
        } else if (f === 3) {
            res.push({ f: 4, r: 2 - c, c: 2 }, { f: 4, r: 2 - (c - 1), c: 2 }, { f: 4, r: 2 - (c + 1), c: 2 });
            res.push({ f: 5, r: c, c: 2 }, { f: 5, r: c - 1, c: 2 }, { f: 5, r: c + 1, c: 2 });
            res.push({ f: 1, r, c: 0 }, { f: 1, r: r - 1, c: 0 }, { f: 1, r: r + 1, c: 0 });
            res.push({ f: 0, r, c: 2 }, { f: 0, r: r - 1, c: 2 }, { f: 0, r: r + 1, c: 2 });
        } else if (f === 4) {
            res.push({ f: 1, r: 0, c: 2 - r }, { f: 1, r: 0, c: 2 - (r - 1) }, { f: 1, r: 0, c: 2 - (r + 1) });
            res.push({ f: 0, r: 0, c }, { f: 0, r: 0, c: c - 1 }, { f: 0, r: 0, c: c + 1 });
            res.push({ f: 2, r: 0, c: r }, { f: 2, r: 0, c: r - 1 }, { f: 2, r: 0, c: r + 1 });
            res.push({ f: 3, r: 0, c: 2 - r }, { f: 3, r: 0, c: 2 - (r - 1) }, { f: 3, r: 0, c: 2 - (r + 1) });
        } else if (f === 5) {
            res.push({ f: 0, r: 2, c }, { f: 0, r: 2, c: c - 1 }, { f: 0, r: 2, c: c + 1 });
            res.push({ f: 1, r: 2, c: 2 - r }, { f: 1, r: 2, c: 2 - (r - 1) }, { f: 1, r: 2, c: 2 - (r + 1) });
            res.push({ f: 2, r: 2, c: 2 - r }, { f: 2, r: 2, c: 2 - (r - 1) }, { f: 2, r: 2, c: 2 - (r + 1) });
            res.push({ f: 3, r: 2, c: r }, { f: 3, r: 2, c: r - 1 }, { f: 3, r: 2, c: r + 1 });
        }
        return res.filter(n => n.f >= 0 && n.f < 6 && n.r >= 0 && n.r < 3 && n.c >= 0 && n.c < 3);
    }

    // High-performance DFS pathfinder to find ALL cells that can form a word
    function getWordCells(word) {
        const upperWord = word.toUpperCase();
        const wordLen = upperWord.length;
        const matchingKeys = new Set(); // Stores "f,r,c" or "r,c" keys

        if (is3D) {
            function dfs3D(f, r, c, index, visited) {
                const cellValue = board[f][r][c].toUpperCase();
                const letters = cellValue.includes('/') ? cellValue.split('/') : [cellValue];
                let foundMatch = false;
                let matchLength = 0;
                for (const char of letters) {
                    if (char === 'Q') {
                        if (upperWord.substring(index, index + 2) === 'QU') { matchLength = 2; foundMatch = true; break; }
                        else if (upperWord[index] === 'Q') { matchLength = 1; foundMatch = true; break; }
                    } else if (upperWord[index] === char) { matchLength = 1; foundMatch = true; break; }
                }
                if (!foundMatch) return;

                const nextIndex = index + matchLength;
                const visitedKey = `${f},${r},${c}`;
                visited.add(visitedKey);

                if (nextIndex >= wordLen) {
                    for (const k of visited) matchingKeys.add(k);
                    visited.delete(visitedKey);
                    return;
                }

                const neighbors = getCubeNeighbors(f, r, c);
                for (const n of neighbors) {
                    const nKey = `${n.f},${n.r},${n.c}`;
                    if (!visited.has(nKey)) {
                        dfs3D(n.f, n.r, n.c, nextIndex, visited);
                    }
                }
                visited.delete(visitedKey);
            }

            for (let f = 0; f < 6; f++) {
                for (let r = 0; r < 3; r++) {
                    for (let c = 0; c < 3; c++) {
                        dfs3D(f, r, c, 0, new Set());
                    }
                }
            }
        } else {
            const rows = board.length;
            const cols = board[0].length;

            function dfs2D(r, c, index, visited) {
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
                if (!foundMatch) return;

                const nextIndex = index + matchLength;
                const visitedKey = `${r},${c}`;
                visited.add(visitedKey);

                if (nextIndex >= wordLen) {
                    for (const k of visited) {
                        const parts = k.split(',');
                        const itemR = parseInt(parts[0]);
                        const itemC = parseInt(parts[1]);
                        matchingKeys.add(`${itemR},${itemC}`);
                    }
                    visited.delete(visitedKey);
                    return;
                }

                for (let dr = -1; dr <= 1; dr++) {
                    for (let dc = -1; dc <= 1; dc++) {
                        if (dr === 0 && dc === 0) continue;
                        const nr = r + dr;
                        const nc = c + dc;
                        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                            const nKey = `${nr},${nc}`;
                            if (!visited.has(nKey)) {
                                dfs2D(nr, nc, nextIndex, visited);
                            }
                        }
                    }
                }
                visited.delete(visitedKey);
            }

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    dfs2D(r, c, 0, new Set());
                }
            }
        }

        return matchingKeys;
    }

    // Populate tileToWordsMap by finding ALL cells for every word
    for (const wordUpper of Object.keys(paths)) {
        const matchingKeys = getWordCells(wordUpper);
        for (const key of matchingKeys) {
            if (!window.tileToWordsMap[key]) {
                window.tileToWordsMap[key] = new Set();
            }
            window.tileToWordsMap[key].add(wordUpper);
        }
    }

    window.tileToWordsMapCacheKey = cacheKey;
    console.log(`[rebuildTileToWordsMap] Pre-mapped ${Object.keys(window.tileToWordsMap).length} tiles for round ${roundId} (exhaustive paths)`);
}


function updateIntermissionRenderKey() {
    if (!window.lastGameState) return;
    const state = window.lastGameState;
    const roundId = `${state.room_id}_${state.current_round}`;
    const filterJSON = JSON.stringify(window.intermissionTileFilter || null);
    const selectedLen = window.selectedAllWordsLength || 'all';
    window.lastRenderedIntermissionKey = `${roundId}_${activeWordsTab}_${state.solving_complete}_${filterJSON}_${selectedLen}_${selectedPlayerUsername || ''}_${highlightedFoundWord || ''}`;
}

function displayAllWords(allWords, bonusWord, targetUserWords = [], allFoundWords = [], allWordScores = {}, cswOnlyWords = [], addedWords = []) {
    console.log(`[displayAllWords] RENDERING. BonusWord: "${bonusWord}" | Words count: ${allWords ? allWords.length : 0}`);
    const listEl = document.getElementById('submitted-words-list');
    const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    const prevScrollTop = (listEl && !isMobile) ? listEl.scrollTop : 0;
    const titleEl = document.getElementById('words-panel-title');
    if (titleEl && activeWordsTab === 'found') {
        titleEl.textContent = 'All Words';
    }

    if (!allWords || allWords.length === 0) {
        listEl.innerHTML = '<p class="placeholder">No words found</p>';
        return;
    }

    // Save arguments for re-rendering on filter change
    window.lastDisplayAllWordsArgs = [allWords, bonusWord, targetUserWords, allFoundWords, allWordScores, cswOnlyWords, addedWords];

    // Normalize input words to avoid type and case transformations inside loops
    const bonusUpper = bonusWord ? bonusWord.toUpperCase().trim() : null;
    const cleanWords = allWords.map(entry => {
        const w = typeof entry === 'object' ? (entry.word || '') : entry;
        const wordUpper = w.toUpperCase();
        return {
            original: entry,
            word: w,
            wordUpper: wordUpper,
            len: w.length,
            isBonus: bonusUpper && (wordUpper.trim() === bonusUpper)
        };
    });

    // Calculate available lengths
    const lengthsSet = new Set();
    for (let i = 0; i < cleanWords.length; i++) {
        lengthsSet.add(cleanWords[i].len);
    }
    const availableLengths = [...lengthsSet].sort((a, b) => a - b);
    
    const findersContainer = document.getElementById('finders-button-container');
    const findersBtn = document.getElementById('view-finders-btn-top');
    
    if (findersContainer && findersBtn) {
        if (highlightedFoundWord) {
            findersContainer.style.display = 'block';
            
            const s = window.lastGameState;
            let finders = [];
            if (s && s.players) {
                const sortedAll = [...s.players].sort((a, b) => (b.score - a.score) || (b.rating - a.rating));
                const rankMap = new Map();
                sortedAll.forEach((p, idx) => rankMap.set(p.username, idx + 1));

                finders = s.players.filter(p =>
                    p.submitted_words && p.submitted_words.some(sw =>
                        (typeof sw === 'object' ? sw.word : sw).toUpperCase() === highlightedFoundWord
                    )
                ).sort((a, b) => {
                    const rA = rankMap.get(a.username) || 999;
                    const rB = rankMap.get(b.username) || 999;
                    return rA - rB;
                });
            }
            
            const findersNames = finders.map(p => p.username).join(', ') || 'None';
            const findersCount = finders.length;
            
            findersBtn.innerHTML = `
                <span>Finders: ${findersNames}</span>
                <span style="font-weight: bold;">[${findersCount}]</span>
            `;
            
            findersBtn.onclick = () => {
                window.showFinderModal(highlightedFoundWord);
            };
        } else {
            findersContainer.style.display = 'none';
        }
    }

    const filterContainer = document.getElementById('length-filter-container');
    const filterDropdown = document.getElementById('length-filter-dropdown');
    
    if (filterContainer && filterDropdown) {
        filterContainer.style.display = 'block';
        
        // Save current selection
        const prevSelection = window.selectedAllWordsLength || 'all';
        
        // Populate dropdown only if lengths have changed
        const currentLengthsStr = availableLengths.join(',');
        if (filterDropdown.dataset.currentLengths !== currentLengthsStr) {
            let optionsHtml = '<option value="all">All Lengths</option>';
            availableLengths.forEach(len => {
                optionsHtml += `<option value="${len}">${len}LW</option>`;
            });
            filterDropdown.innerHTML = optionsHtml;
            filterDropdown.dataset.currentLengths = currentLengthsStr;
        }
        
        // Restore selection if it's still available
        if (availableLengths.includes(parseInt(prevSelection)) || prevSelection === 'all') {
            filterDropdown.value = prevSelection;
            window.selectedAllWordsLength = prevSelection;
        } else {
            filterDropdown.value = 'all';
            window.selectedAllWordsLength = 'all';
        }
        
        // Add change listener (if not already added)
        if (!filterDropdown.dataset.listenerAdded) {
            filterDropdown.addEventListener('change', (e) => {
                window.selectedAllWordsLength = e.target.value;
                displayAllWords(...window.lastDisplayAllWordsArgs);
            });
            filterDropdown.dataset.listenerAdded = 'true';
        }
    }

    const targetWordsUpper = targetUserWords.map(w => (typeof w === 'object' ? (w.word || '') : w).toUpperCase());
    const allFoundUpper = allFoundWords.map(w => (typeof w === 'object' ? (w.word || '') : w).toUpperCase());
    const cswOnlyUpper = (cswOnlyWords || []).map(w => w.toUpperCase());
    const addedUpper = (addedWords || []).map(w => w.toUpperCase());

    const targetWordsSet = new Set(targetWordsUpper);
    const allFoundSet = new Set(allFoundUpper);
    const cswOnlySet = new Set(cswOnlyUpper);
    const addedSet = new Set(addedUpper);

    if (bonusWord && allWords) {
        const bonusIdx = allWords.findIndex(w => (typeof w === 'object' ? (w.word || '') : w).toUpperCase() === bonusWord.toUpperCase());
        console.log(`[displayAllWords] BonusWord search: "${bonusWord}" found at index: ${bonusIdx}`);
    }
    console.log(`[displayAllWords] Added words for highlight:`, addedUpper);

    // Filter first (by intermission tile filter)
    let displayWords = cleanWords;
    if (window.intermissionTileFilter && window.lastGameState && window.lastGameState.all_words_paths) {
        const { r, c, f } = window.intermissionTileFilter;
        const roundId = window.lastGameState ? `${window.lastGameState.room_id}_${window.lastGameState.current_round}` : null;
        const isTransposed = !!window.isBoardTransposed;
        const cacheKey = `${roundId}_${isTransposed}`;

        if (!window.tileToWordsMap || window.tileToWordsMapCacheKey !== cacheKey) {
            rebuildTileToWordsMap();
        }

        const filterKey = (f !== null && f !== undefined && f !== -1) ? `${f},${r},${c}` : `${r},${c}`;
        const matchingWords = window.tileToWordsMap[filterKey] || new Set();
        displayWords = displayWords.filter(entry => matchingWords.has(entry.wordUpper));
    }

    // Filter by length
    const selectedLength = window.selectedAllWordsLength || 'all';
    const filteredWords = selectedLength === 'all' 
        ? displayWords 
        : displayWords.filter(entry => entry.len.toString() === selectedLength);

    // Sort the filtered subset only (using optimized comparator)
    filteredWords.sort((a, b) => {
        // 0. Bonus Word (Absolute Top Priority)
        if (a.isBonus) return -1;
        if (b.isBonus) return 1;

        // 1. Length (Desc) - Primary sort
        if (a.len !== b.len) return b.len - a.len;

        // 2. Alphabetical (Asc) - Secondary sort (ASCII operators are 10-20x faster than localeCompare)
        if (a.wordUpper < b.wordUpper) return -1;
        if (a.wordUpper > b.wordUpper) return 1;
        return 0;
    });

    console.log('[renderWordsList] Rendering words:', filteredWords.length);

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

    // Filter clear button handling
    const existingFilterBtn = document.getElementById('intermission-filter-btn-container');
    if (existingFilterBtn) {
        existingFilterBtn.remove();
    }

    if (window.intermissionTileFilter) {
        const filterBtnContainer = document.createElement('div');
        filterBtnContainer.id = 'intermission-filter-btn-container';
        filterBtnContainer.style.margin = '5px 0';
        filterBtnContainer.style.width = '100%';
        
        filterBtnContainer.innerHTML = `
            <button id="clear-intermission-filter-btn" style="width: 100%; padding: 8px 10px; font-size: 0.75rem; font-weight: 600; border-radius: 6px; background: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid rgba(46, 204, 113, 0.3); cursor: pointer; display: flex; justify-content: center; align-items: center; gap: 6px; transition: all 0.2s ease;">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
                Show Full List Again (Filtered by Tile "${window.intermissionTileFilter.letter}")
            </button>
        `;
        
        listEl.parentNode.insertBefore(filterBtnContainer, listEl);
        
        const clearBtn = document.getElementById('clear-intermission-filter-btn');
        clearBtn.addEventListener('click', () => {
            window.intermissionTileFilter = null;
            document.querySelectorAll('.board-cell.intermission-highlight').forEach(el => {
                el.classList.remove('intermission-highlight');
            });
            displayAllWords(...window.lastDisplayAllWordsArgs);
        });
    }

    listEl.innerHTML = filteredWords.map(entry => {
        const word = entry.word;
        const wordUpper = entry.wordUpper;
        const isBonus = entry.isBonus;
        const isCSWOnly = cswOnlySet.has(wordUpper);
        const isAddedWord = addedSet.has(wordUpper) || (window.globalAddedWordsSet && window.globalAddedWordsSet.has(wordUpper));
        const isTargetFound = targetWordsSet.has(wordUpper);
        const isFoundByAny = allFoundSet.has(wordUpper);
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
        // Color priority for FOUND BY PLAYER:
        //   Blue (player-word) ALWAYS takes absolute priority.
        //   Never add a dictionary-tier class when isTargetFound — blue must not be overridden by gold (csw-only).
        // Color priority for other words:
        //   1. NWL words → dark red (nwl-word)
        //   2. CSW-only words (in CSW, not in NWL) → gold (csw-only)
        //   3. AW-only words (not in NWL or CSW) → purple (added-word)
        if (isBonus) {
            className += ' bonus-word';
        } else if (isTargetFound) {
            // FIX: Player's own found words are ALWAYS blue. Never add dictionary-tier colors.
            // csw-only (gold) must NOT override player-word (blue) for words found by the player.
            className += ' player-word';
        } else {
            // For words found by others or unfound: apply found-status + dictionary-tier color
            if (isFoundByAny) {
                className += ' found-by-other';
            } else {
                className += ' unfound';
            }

            let baseDict = 'NWL'; // The base standard dictionary for this round
            let useAW = false;
            const s = window.lastGameState;
            if (s) {
                const dictStr = String(s.previous_dictionary || s.current_dictionary || 'NWL').toUpperCase();
                useAW = (
                    s.previous_use_added_words === true ||
                    s.use_added_words === true ||
                    dictStr.includes('+ AW') ||
                    dictStr.includes('+AW') ||
                    dictStr.includes('ADDED') ||
                    (addedSet && addedSet.size > 0)
                );
                baseDict = dictStr.replace(/\s*\+\s*AW/i, '').replace(/\s*\+AW/i, '').trim();
            }

            const isCSWMode = (
                baseDict === 'CSW' ||
                baseDict.includes('CSW') ||
                baseDict.includes('COLLINS') ||
                (cswOnlySet && cswOnlySet.size > 0)
            );

            if (useAW || (addedSet && addedSet.size > 0)) {
                // NWL+AW or CSW+AW — three tiers
                if (isAddedWord) {
                    className += ' added-word'; // Purple: AW-only (not in NWL or CSW)
                } else if (isCSWOnly) {
                    className += ' csw-only';   // Gold: in CSW but not NWL
                } else {
                    className += ' nwl-word';   // Sleek dark slate: in NWL
                }
            } else if (isCSWMode || isCSWOnly) {
                // Pure CSW round or any CSW-only words present — two tiers
                if (isCSWOnly) {
                    className += ' csw-only';   // Gold: in CSW but not NWL
                } else {
                    className += ' nwl-word';   // Sleek dark slate: in NWL (also in CSW)
                }
            } else {
                // Pure NWL or other fallback
                if (isCSWOnly) {
                    className += ' csw-only';   // Gold: CSW-only word
                } else {
                    className += ' nwl-word';   // Sleek dark slate for standard NWL words
                }
            }
        }

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

    if (listEl && !isMobile) {
        listEl.scrollTop = prevScrollTop;
    }

    // Synchronize render cache key to prevent redundant heartbeat re-renders
    updateIntermissionRenderKey();
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
    const preferSp = (isIntermission && isRevealed);

    const factFmt = sp.board_format || state.current_board_format || 'Normal';
    const factDiff = sp.difficulty || state.current_difficulty || 'Medium';
    const factBonus = sp.bonus_word_length || (state.bonus_word ? state.bonus_word.length : state.current_bonus_word_length) || 0;
    const factMinLen = sp.min_word_length || state.current_min_length || 3;
    const rawDict = (sp && sp.dictionary) ? sp.dictionary : (state.current_dictionary || 'NWL');
    const useAW = (sp && sp.use_added_words !== undefined) 
        ? (sp.use_added_words === true || /\+\s*AW/i.test(sp.dictionary || ''))
        : (state.use_added_words === true || /\+\s*AW/i.test(state.current_dictionary || ''));
    const cleanBaseDict = rawDict.replace(/\s*\+\s*AW/i, '').replace(/\s*\+AW/i, '').replace(/ADDED_WORDS/i, '').trim() || 'NWL';
    const factDict = useAW ? `${cleanBaseDict} + AW` : cleanBaseDict;
    let rawWordRange = sp.word_count_range || state.current_word_count_range || 'Random';
    if (Array.isArray(rawWordRange)) {
        rawWordRange = rawWordRange.join('-');
    } else {
        rawWordRange = String(rawWordRange).replace(',', '-');
    }
    const factWordRange = rawWordRange;
    
    let factUniq = 0;
    if (sp && sp.uniqueness !== undefined && sp.uniqueness !== null && sp.uniqueness > 0) {
        factUniq = sp.uniqueness;
    } else if (state.current_uniqueness !== undefined && state.current_uniqueness !== null) {
        factUniq = state.current_uniqueness;
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
        window._animTriggeredForRound = -1;
    } else {
        if (isRevealed && !wasRevealed && (window._animTriggeredForRound !== currentRound)) {
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
            let bonusVal = parseInt(factBonus);
            if (isNaN(bonusVal) || bonusVal <= 0) {
                if (state.bonus_word && typeof state.bonus_word === 'string' && state.bonus_word.length >= 3 && String(state.bonus_word).toUpperCase() !== 'NONE') {
                    bonusVal = state.bonus_word.length;
                } else {
                    const dStr = String(factBoardDims || '');
                    bonusVal = dStr.includes('6x8') ? 9 : (dStr.includes('5x7') ? 8 : (dStr.includes('4x6') ? 7 : 8));
                }
            }
            window._displayedParams.bonus = bonusVal + 'L';
            
            let diffLabel = factDiff;
            if (newUniq > 0) {
                const uVal = newUniq > 1 ? newUniq / 100.0 : newUniq;
                const dStr = String(factBoardDims || '').toLowerCase();
                let easyMax = 0.15;
                let medMax = 0.29;
                if (dStr.includes('4x6')) { easyMax = 0.25; medMax = 0.39; }
                else if (dStr.includes('5x7')) { easyMax = 0.29; medMax = 0.44; }
                else if (dStr.includes('6x8') || dStr.includes('cube') || dStr.includes('3x3x3')) { easyMax = 0.34; medMax = 0.49; }
                else { easyMax = 0.15; medMax = 0.29; }

                if (uVal <= easyMax) diffLabel = 'Easy';
                else if (uVal <= medMax) diffLabel = 'Medium';
                else diffLabel = 'Hard';
            } else {
                if (diffLabel === 'Varying...') diffLabel = 'Random';
                else if (diffLabel === 'Normal') diffLabel = 'Medium';
                else if (diffLabel === 'Expert' || diffLabel === 'Difficult') diffLabel = 'Hard';
                else if (diffLabel === 'Beginner') diffLabel = 'Easy';
            }
            
            const uniquePct = (newUniq > 0 && !diffLabel.includes('(')) ? ` (${Math.round(newUniq * 100)}%)` : "";
            window._displayedParams.diff = diffLabel + uniquePct;
            
            if (typeof updateColorBarHighlight === 'function') {
                updateColorBarHighlight(diffLabel, newUniq);
            }

            // Apply to DOM
            if (document.getElementById('param-board')) document.getElementById('param-board').textContent = window._displayedParams.dims;
            if (document.getElementById('param-time')) document.getElementById('param-time').textContent = window._displayedParams.time;
            
            // Populate the new label above Spinner Set
            if (document.getElementById('label-game-type')) document.getElementById('label-game-type').textContent = document.getElementById('game-title').textContent;
            if (document.getElementById('label-board')) document.getElementById('label-board').textContent = window._displayedParams.dims;
            if (document.getElementById('label-time')) document.getElementById('label-time').textContent = window._displayedParams.time;

            const diffEl = document.getElementById('param-diff');
            if (diffEl) {
                diffEl.textContent = window._displayedParams.diff;
                // Dynamically apply color based on difficulty (Easy -> emerald, Medium -> golden, Hard -> red)
                const lowerDiff = diffLabel.toLowerCase();
                if (lowerDiff.includes('easy') || lowerDiff.includes('beginner')) {
                    diffEl.style.color = '#2ecc71'; // Dark green / emerald
                } else if (lowerDiff.includes('medium') || lowerDiff.includes('normal')) {
                    diffEl.style.color = '#60a5fa'; // Blue (same as FAQ)
                } else if (lowerDiff.includes('hard') || lowerDiff.includes('expert') || lowerDiff.includes('difficult')) {
                    diffEl.style.color = '#ff4d4d'; // Red
                } else {
                    diffEl.style.color = ''; // Reset/Default
                }
            }
            if (document.getElementById('param-min')) document.getElementById('param-min').textContent = window._displayedParams.min;
            if (document.getElementById('param-dict')) document.getElementById('param-dict').textContent = window._displayedParams.dict;
            if (document.getElementById('param-range')) document.getElementById('param-range').textContent = window._displayedParams.range;
            if (document.getElementById('param-bonus')) document.getElementById('param-bonus').textContent = window._displayedParams.bonus;
        }
    }

    if (triggerAnimation) {
        const paramContainer = document.querySelector('.game-params');
        const labelContainer = document.querySelector('.spinner-set-label');
        if (paramContainer) {
            paramContainer.classList.remove('reveal-new');
            void paramContainer.offsetWidth; // Force reflow
            paramContainer.classList.add('reveal-new');
        }
        if (labelContainer) {
            labelContainer.classList.remove('reveal-new');
            void labelContainer.offsetWidth; // Force reflow
            labelContainer.classList.add('reveal-new');
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

    // Dynamic Spinner Set font autoscaler
    if (typeof window.adjustSpinnerSetFontSize === 'function') {
        window.adjustSpinnerSetFontSize();
    }

    } catch (err) {
        console.error('[play.js] Error in updateParameters:', err);
    }
}

window.adjustSpinnerSetFontSize = function() {
    const gameParams = document.querySelector('.game-params');
    if (!gameParams) return;

    if (window.innerWidth > 992) {
        gameParams.style.fontSize = ''; // Reset to default on desktop
        return;
    }

    // Clean up text content for accurate character length comparison
    const txt = gameParams.textContent || "";
    const cleanTxt = txt.replace(/\s+/g, ' ').trim();

    // If the text length is long (80 characters or more), shrink font size
    if (cleanTxt.length >= 80) {
        gameParams.style.fontSize = '0.735rem'; // Slightly smaller font size
    } else {
        gameParams.style.fontSize = '0.85rem'; // Keep default mobile size
    }
};

window.addEventListener('resize', () => {
    if (typeof window.adjustSpinnerSetFontSize === 'function') {
        window.adjustSpinnerSetFontSize();
    }
});

// Orientation change: force board re-transposition when device is rotated.
// Clears the battery-optimization cache so the next heartbeat triggers a full re-render
// with the correct portrait/landscape transposition applied.
(function () {
    let _lastOrientIsPortrait = window.innerHeight > window.innerWidth;
    function _onOrientationChange() {
        const nowPortrait = window.innerHeight > window.innerWidth;
        if (nowPortrait !== _lastOrientIsPortrait) {
            _lastOrientIsPortrait = nowPortrait;
            // Bust the render cache so the next state update re-transposes and re-renders.
            window.lastRenderedStateJSON = null;
            // If we have a current state, immediately re-process it with the new orientation.
            if (window.lastGameState) {
                // Use the raw (pre-transpose) state so we can re-transpose from scratch
                // for the new orientation. Cloning lastGameState (post-transpose) would
                // leave wide boards transposed when rotating to landscape.
                const sourceState = window.lastRawGameState || window.lastGameState;
                const freshState = JSON.parse(JSON.stringify(sourceState));
                // Strip transposition cache so safelyTransposeState re-evaluates.
                delete freshState._isAlreadyTransposed;
                delete freshState._isBoardTransposedValue;
                lastRenderedBoardJSON = null; // Force board re-render even if letters unchanged
                if (typeof updateGameState === 'function') updateGameState(freshState);
            }
        }
    }
    window.addEventListener('orientationchange', () => setTimeout(_onOrientationChange, 200));
    window.addEventListener('resize', _onOrientationChange);
})();

function updateTimer(seconds) {
    // Legacy local timer update (called by interval)
    // see updateLocalTimer
}

function syncTimerWithServer(state, tBefore = null, tAfter = null) {
    const clientTime = Date.now() / 1000;
    const serverTime = state.server_time;
    if (serverTime) {
        let currentOffset;
        let rtt = null;
        if (tBefore !== null && tAfter !== null) {
            rtt = tAfter - tBefore;
            currentOffset = serverTime - (tBefore + tAfter) / 2;
        } else {
            currentOffset = serverTime - clientTime;
        }

        // Reset our tracker if a large step-change (> 3 seconds) occurs, indicating a clock change
        if (stableServerTimeOffset === null || Math.abs(currentOffset - stableServerTimeOffset) > 3) {
            bestServerTimeRTT = Infinity;
        }

        // Update offset if we have a better (lower RTT) sample, or if it is uninitialized
        if (stableServerTimeOffset === null || (rtt !== null && rtt < bestServerTimeRTT)) {
            stableServerTimeOffset = currentOffset;
            if (rtt !== null) {
                bestServerTimeRTT = rtt;
            }
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

    // SPECIAL CASE: 24h Rooms align to authoritative server midnight boundary
    if (state.time_limit >= 7200) {
        timerFormatIs24h = true;
    }

    if (!timerInterval && localEndTime > 0 && !document.hidden) {
        timerInterval = setInterval(updateLocalTimer, 100); // 100ms for near-instant 0:00 detection
    } else if ((localEndTime <= 0 || document.hidden) && timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function updateLocalTimer() {
    // Thread Freeze Detection: If the JavaScript thread locks/freezes (e.g. mobile lock screen/app minimized)
    // and thaws later without clean visibility events, instantly trigger a state update.
    const tickTime = Date.now();
    if (window._lastLocalTimerTickTime) {
        const gap = tickTime - window._lastLocalTimerTickTime;
        if (gap > 2200) {
            console.log(`[play.js] updateLocalTimer: Thawed after freeze gap of ${gap}ms. Requesting delayed wake-up update.`);
            if (!localEndTime || localEndTime <= (Date.now() / 1000)) {
                setTimerWaitingState(true);
            }
            setTimeout(() => {
                updateGameState();
                refreshPollInterval();
            }, 80);
        }
    }
    window._lastLocalTimerTickTime = tickTime;

    if (!localEndTime) return;

    if (!cachedTimerValueEl) cachedTimerValueEl = document.getElementById('timer-value');
    if (!cachedBoardPanelEl) cachedBoardPanelEl = document.querySelector('.board-panel');

    const now = Date.now() / 1000;
    let remaining = Math.max(0, localEndTime - now);

    // Clamp remaining time to room or intermission limit to prevent values like 0:47 or 1:02
    if (window.lastGameState) {
        const currentState = window.lastGameState.state;
        if (currentState === 'active') {
            const limit = window.lastGameState.time_limit || 45;
            if (remaining > limit) {
                remaining = limit;
            }
        } else if (currentState === 'intermission') {
            const limit = (window.lastGameState.time_limit >= 7200) ? 5 : 60;
            if (remaining > limit) {
                remaining = limit;
            }
        }
    }

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
        setTimerWaitingState(false);

        // Low time visual for text (User Request)
        if (remaining <= 10 && remaining > 0) {
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
        console.log(`[play.js] Local timer reached 0:00 in ${currentState} state - Scheduling rapid server poll.`);
        
        // Instant client-side transition feedback!
        setTimerWaitingState(true);
        if (currentState === 'active') {
            const wordInput = document.getElementById('word-input');
            if (wordInput) {
                wordInput.value = '';
                wordInput.disabled = true;
                wordInput.blur();
            }
            // Reset mouse selection state if it was active
            if (typeof mouseState !== 'undefined') {
                mouseState.isDown = false;
                mouseState.selectedPath = [];
                if (mouseState.visitedCells) mouseState.visitedCells.clear();
            }
        } else if (currentState === 'intermission') {
            // Board comes from cache (instantaneous) — no overlay needed.
            // Just call updateGameState immediately to get the new board.
        }

        // Force immediate reset of fetching guard to ensure 0:00 transition fetch is NEVER suppressed!
        isFetchingState = false;
        lastStateFetchTime = 0;
        updateGameState();

        // Rapid 300ms polling loop until server state transition is confirmed
        let transitionAttempts = 0;
        const expectedNextState = (currentState === 'active') ? 'intermission' : 'active';
        if (window._transitionPollTimer) clearInterval(window._transitionPollTimer);
        window._transitionPollTimer = setInterval(() => {
            transitionAttempts++;
            if (window.lastGameState && window.lastGameState.state === expectedNextState) {
                clearInterval(window._transitionPollTimer);
                window._transitionPollTimer = null;
                return;
            }
            if (transitionAttempts >= 15) { // 4.5 seconds max safety timeout
                clearInterval(window._transitionPollTimer);
                window._transitionPollTimer = null;
                return;
            }
            isFetchingState = false;
            lastStateFetchTime = 0;
            updateGameState();
        }, 300);
    }

    // -- Update Triple Format Music State --
    updateTripleMusicState(remaining);

    // -- Next Round Bell Logic --
    if (window.lastGameState && window.lastGameState.state === 'intermission') {
        const isEnabled = window.userSettings && (window.userSettings.next_round_bell_enabled === true || window.userSettings.next_round_bell_enabled === 'true');
        const bellType = (window.userSettings && window.userSettings.next_round_bell_type) || 'bell1';

        if (isEnabled && remaining <= 10.0 && remaining > 1.0 && !hasPlayedIntermissionBell) {
            console.log(`[play.js] Playing intermission bell: ${bellType}`);
            
            const vibrationEnabled = !window.userSettings || window.userSettings.vibration_alert === true || window.userSettings.vibration_alert === 'true';
            
            // Trigger vibration on devices supporting it (mobile web browsers)
            if (vibrationEnabled && navigator.vibrate) {
                try {
                    navigator.vibrate(500); // Vibrate for 500ms
                } catch (err) {
                    console.warn('[play.js] navigator.vibrate failed:', err);
                }
            }

            if (typeof MorphemeAudioBridge !== 'undefined') {
                try {
                    // Send bell chime natively for mobile app wrappers (bypasses webview audio block)
                    MorphemeAudioBridge.postMessage(JSON.stringify({ sound: 'bell', type: bellType }));
                    // Trigger native app wrapper vibration if enabled
                    if (vibrationEnabled) {
                        MorphemeAudioBridge.postMessage(JSON.stringify({ action: 'vibrate', duration: 500 }));
                    }
                } catch (e) {
                    console.error("MorphemeAudioBridge error:", e);
                }
                hasPlayedIntermissionBell = true;
                return;
            }
            const audio = window.intermissionBellAudio;
            if (audio) {
                if (!audio.src || !audio.src.includes(bellType)) {
                    audio.src = `/static/audio/${bellType}.wav`;
                    try {
                        audio.load();
                    } catch (e) {
                        console.warn('[play.js] Failed to load audio src on trigger:', e);
                    }
                }
                audio.play().catch(e => {
                    console.warn('[play.js] Intermission audio failed, playing fallback:', e);
                    const fallbackAudio = new Audio(`/static/audio/${bellType}.wav`);
                    fallbackAudio.play().catch(fe => console.warn('Fallback audio failed:', fe));
                });
            } else {
                const fallbackAudio = new Audio(`/static/audio/${bellType}.wav`);
                fallbackAudio.play().catch(e => console.warn('Fallback audio failed:', e));
            }
            hasPlayedIntermissionBell = true;
        }
    } else {
        // Reset flag when not in intermission
        hasPlayedIntermissionBell = false;
    }
}

function updateTripleMusicState(remaining) {
    const audio = document.getElementById('triple-music');
    if (!audio) return;

    // Respect user triple music preferences
    const musicEnabled = (!window.userSettings || (window.userSettings.triple_music !== false && window.userSettings.triple_music !== 'false'));
    if (!musicEnabled) {
        if (!audio.paused) {
            audio.pause();
        }
        return;
    }

    const state = window.lastGameState;
    if (!state) {
        if (!audio.paused) {
            audio.pause();
        }
        return;
    }

    const sp = state.spinner_params || {};
    const isIntermission = state.state === 'intermission';
    const isRevealed = state.spinner_params_revealed === true || state.spinner_params_revealed === 'true';
    const preferSp = isIntermission && isRevealed;

    const factFmt = (preferSp ? (sp.board_format || state.current_board_format) : (state.current_board_format || sp.board_format)) || 'Normal';
    const isTriple = (factFmt && factFmt.toString().toLowerCase().trim() === 'triple');

    if (state.state === 'active') {
        window._lastActiveRoundFormat = state.current_board_format || 'Normal';
    }
    const wasTriplePrevious = (window._lastActiveRoundFormat && window._lastActiveRoundFormat.toString().toLowerCase().trim() === 'triple');

    if (state.state === 'active') {
        if (isTriple) {
            audio.volume = 1.0;
            if (audio.paused) {
                audio.currentTime = 0;
                audio.play().catch(e => console.warn('Failed to play triple music:', e));
            }
        } else {
            if (!audio.paused) {
                audio.pause();
            }
        }
    } else if (isIntermission) {
        if (wasTriplePrevious && remaining >= 50.0) {
            // Fade-out: 10s from start of intermission (remaining: 60.0 -> 50.0)
            const fadeOutVolume = Math.max(0.0, Math.min(1.0, (remaining - 50.0) / 10.0));
            audio.volume = fadeOutVolume;
            if (fadeOutVolume > 0) {
                if (audio.paused) {
                    audio.play().catch(e => console.warn('Failed to play triple music during fade-out:', e));
                }
            } else {
                if (!audio.paused) {
                    audio.pause();
                }
            }
        } else if (isTriple && remaining >= 0 && remaining <= 10.0) {
            // Fade-in: 10s before round starts (remaining: 10.0 -> 0.0)
            const fadeInVolume = Math.max(0.0, Math.min(1.0, (10.0 - remaining) / 10.0));
            audio.volume = fadeInVolume;
            if (fadeInVolume > 0) {
                if (audio.paused) {
                    audio.currentTime = 0;
                    audio.play().catch(e => console.warn('Failed to play triple music during fade-in:', e));
                }
            } else {
                if (!audio.paused) {
                    audio.pause();
                }
            }
        } else {
            if (!audio.paused) {
                audio.pause();
            }
        }
    } else {
        if (!audio.paused) {
            audio.pause();
        }
    }
}

// Helper for special match timers (Tournament, Private Match)
function updateSpecialMatchTimer(seconds) {
    const timerEl = document.getElementById('timer-value');
    if (timerEl) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        timerEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
        setTimerWaitingState(false);
    }
}

function renderBoard(board, grayed = false, is3D = false, state = null) {
    if (!grayed) {
        window.intermissionTileFilter = null;
        const existingFilterBtn = document.getElementById('intermission-filter-btn-container');
        if (existingFilterBtn) {
            existingFilterBtn.remove();
        }
    }
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
    const transposeBtn = document.getElementById('transpose-board-btn');
    if (rotateBtn) {
        if (is3D) rotateBtn.classList.add('hidden');
        else rotateBtn.classList.remove('hidden');
    }
    if (transposeBtn) {
        if (is3D) transposeBtn.classList.add('hidden');
        else transposeBtn.classList.remove('hidden');
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
    
    // Clear loading interval if board has content
    if (hasLetters && window.boardLoadingInterval) {
        clearInterval(window.boardLoadingInterval);
        window.boardLoadingInterval = null;
    }
    
    // IF board is empty OR has no letters (and NOT in intermission), show loading spinner
    if (!hasLetters && (!state || state.state !== 'intermission')) {
        let loadingMsg = "GENERATING NEXT BOARD...";
        if (state && state.state === 'loading') {
            loadingMsg = "GENERATING NEXT BOARD...";
        } else if (state && state.current_board_format) {
            loadingMsg = `GENERATING ${state.current_board_format.toUpperCase()}...`;
        } else if (window.lastGameState && window.lastGameState.current_board_format) {
            loadingMsg = `GENERATING ${window.lastGameState.current_board_format.toUpperCase()}...`;
        }
        
        // Clear any existing interval to prevent leaks
        if (window.boardLoadingInterval) {
            clearInterval(window.boardLoadingInterval);
        }
        
        const statuses = [
            "Rolling letter frequencies...",
            "Attempting to embed bonus word...",
            "Running DFS solver to find all possible words...",
            "Checking word count against target range...",
            "Validating uniqueness ratio for difficulty...",
            "Enforcing rare letter caps (Max 1 of Q, Z, J, X, K)...",
            "Verifying checkerboard alternation...",
            "Sanitizing layout for playability...",
            "Calibrating board density...",
            "Finalizing ironclad compliance check..."
        ];
        let statusIdx = 0;
        
        // Start the interval to rotate status messages
        window.boardLoadingInterval = setInterval(() => {
            const el = document.getElementById('board-loading-status');
            if (el) {
                statusIdx = (statusIdx + 1) % statuses.length;
                el.innerText = `[PROCESSING] ${statuses[statusIdx]}`;
            } else {
                clearInterval(window.boardLoadingInterval);
            }
        }, 1200);
        
        ensureLoadingCardStyles();
        boardEl.className = 'game-board-loading';
        boardEl.removeAttribute('style');
        boardEl.innerHTML = `
            <div class="loading-container">
                <div class="glow-spinner"></div>
                <div class="glow-title">${loadingMsg}</div>
                <div id="board-loading-status" class="status-ticker">[PROCESSING] ${statuses[0]}</div>
                <div class="why-text">
                    Morpheme solves and calibrates every board in real-time. We run hundreds of simulations in the background to guarantee your target word count, board difficulty, and letter distribution.
                </div>
            </div>
        `;
        return;
    }

    // Optimization: Skip if board hasn't changed
    const densityJSON = JSON.stringify((window.lastGameState && window.lastGameState.cell_density) || []);
    const boardJSON = JSON.stringify(board);
    const isUserTransposed = !!window.isUserBoardTransposed;
    if (boardJSON === lastRenderedBoardJSON && 
        densityJSON === lastRenderedDensityJSON &&
        grayed === lastRenderedGrayed && 
        isBoardRotated === lastRenderedRotation && 
        isUserTransposed === lastRenderedUserTranspose &&
        boardEl.classList.contains('game-board')) {
        reapplyBoardHighlights();
        if (typeof checkBoardOverflow === 'function') checkBoardOverflow();
        return;
    }
    lastRenderedBoardJSON = boardJSON;
    lastRenderedDensityJSON = densityJSON;
    lastRenderedGrayed = grayed;
    lastRenderedRotation = isBoardRotated;
    lastRenderedUserTranspose = isUserTransposed;

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
                                 let textColor;
                                 if (grayVal <= 60) {
                                     textColor = '#ffffff';
                                 } else {
                                     const textL = Math.round(20 * (100 - grayVal) / 40);
                                     textColor = `hsl(0, 0%, ${textL}%)`;
                                 }
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
    boardEl.innerHTML = '';
    for (let rOut = 0; rOut < rows; rOut++) {
        for (let cOut = 0; cOut < cols; cOut++) {
            let r1 = isBoardRotated ? (rows - 1 - rOut) : rOut;
            let c1 = isBoardRotated ? (cols - 1 - cOut) : cOut;

            let origR = r1;
            let origC = window.isUserBoardTransposed ? (cols - 1 - c1) : c1;

            const cellChar = (board && board[origR] && board[origR][origC] !== undefined) ? board[origR][origC] : '';
            const cell = createBoardCell(origR, origC, cellChar, grayed, undefined, state);
            boardEl.appendChild(cell);
        }
    }

    // Handle format specific animations (Rotation and Bounce)
    const bFormat = (state && state.current_board_format) ? state.current_board_format : ((window.lastGameState && window.lastGameState.current_board_format) ? window.lastGameState.current_board_format : 'Normal');
    if (bFormat.toLowerCase().includes('bounce') && !is3D) {
        setTimeout(startBounceFormat, 300);
    } else {
        stopBounceFormat();
    }
    if (bFormat.toLowerCase().includes('rotat')) {
        startRotatingLetters();
    } else {
        stopRotatingLetters();
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
    if (!playPage || !playPage.classList.contains('active')) return;
    const boardPanel = document.querySelector('.board-panel');
    const boardEl = document.getElementById('game-board');
    if (!boardPanel || !boardEl) return;

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
    const displayCols = cols;
    const displayRows = rows;
    console.log(`[LayoutCheck] Raw/Display: ${cols}x${rows}. Transposed: ${isBoardTransposed}`);
    boardEl.style.setProperty('--board-cols', displayCols);
    boardEl.style.setProperty('--board-rows', displayRows);

    // Get Cell Size (Always fetch to ensure sync with User Settings)
    const minDim = Math.min(cols, rows);
    const maxDim = Math.max(cols, rows);
    const currentDim = `${minDim}x${maxDim}`;
    let savedSettingSize = null;
    const storedSettingsObj = window.userSettings || {};

    // Determine the base cell size using a strict precedence:
    // 1. Session override for this specific dimension (dim slider adjusted in this session)
    // 2. Session override for global board size (main slider adjusted in this session)
    // 3. Saved setting for this specific dimension (from user profile / database)
    // 4. Hardcoded default for this specific dimension (dimension-specific defaults take precedence over main slider)
    // 5. Saved setting for global board size (from user profile / database)
    if (!window.cachedCellSizes) window.cachedCellSizes = {};

    const defaultForDim = currentDim === '6x8' ? 54 : (currentDim === '5x7' ? 65 : (currentDim === '4x6' ? 82 : 82));

    if (window.cachedCellSizes[currentDim]) {
        savedSettingSize = parseInt(window.cachedCellSizes[currentDim]);
    } else if (storedSettingsObj.board_sizes && storedSettingsObj.board_sizes[currentDim]) {
        savedSettingSize = parseInt(storedSettingsObj.board_sizes[currentDim]);
    } else {
        savedSettingSize = defaultForDim;
    }

    let cellSize = savedSettingSize;

    document.documentElement.style.setProperty('--cell-size', `${cellSize}px`);
    playPage.style.setProperty('--cell-size', `${cellSize}px`);
    boardEl.style.setProperty('--cell-size', `${cellSize}px`);
    const previewBoard = document.getElementById('preview-board');
    if (previewBoard) previewBoard.style.setProperty('--cell-size', `${cellSize}px`);

    applyPanelLayout(cellSize, cols);

    // Maintain vertical scroll class
    if (boardPanel.scrollHeight > boardPanel.clientHeight) {
        playPage.classList.add('has-vertical-scroll');
    } else {
        playPage.classList.remove('has-vertical-scroll');
    }
}

// Standalone panel layout calculator — can be called directly from Settings sliders
// cellSize: current tile px size, cols: number of board columns
function applyPanelLayout(cellSize, cols) {
    console.log('[play.js] applyPanelLayout called with cellSize:', cellSize, 'cols:', cols);
    const playPage = document.getElementById('page-play');
    const playGrid = document.querySelector('.play-grid');
    console.log('[play.js] playPage:', !!playPage, 'playGrid:', !!playGrid);
    if (!playPage && !playGrid) return;

    const boardGap = 4 * (cols - 1);
    const boardPadding = 40; // desktop only
    const requiredBoardWidth = (cols * cellSize) + boardGap + boardPadding;

    let gridGap = 12;
    if (window.innerWidth >= 1920) {
        gridGap = 24;
    } else if (window.innerWidth >= 1600) {
        gridGap = 20;
    } else if (window.innerWidth >= 1400) {
        gridGap = 16;
    }

    let gridWidth = window.innerWidth;
    if (playGrid && playGrid.clientWidth > 0) {
        gridWidth = playGrid.clientWidth;
    } else {
        let pagesWidth = window.innerWidth;
        if (window.innerWidth >= 1920) {
            pagesWidth = Math.floor(window.innerWidth * 0.92);
        } else if (window.innerWidth >= 1440) {
            pagesWidth = Math.floor(window.innerWidth * 0.95);
        }
        gridWidth = pagesWidth - 20;
    }

    const availableForPanels = gridWidth - requiredBoardWidth - (gridGap * 2);

    const minLeft = 160;
    const minRight = 220;
    let newLeft, newRight;

    console.log('[play.js] availableForPanels:', availableForPanels, 'requiredBoardWidth:', requiredBoardWidth, 'gridWidth:', gridWidth);

    const maxRight = window.innerWidth >= 1920 ? 500 : window.innerWidth >= 1600 ? 460 : window.innerWidth >= 1400 ? 420 : 380;
    const maxLeft  = window.innerWidth >= 1920 ? 460 : window.innerWidth >= 1600 ? 420 : window.innerWidth >= 1400 ? 380 : 340;

    const totalMax = maxLeft + maxRight;
    if (availableForPanels >= totalMax) {
        newLeft = maxLeft;
        newRight = maxRight;
    } else if (availableForPanels <= (minLeft + minRight)) {
        newLeft = minLeft;
        newRight = minRight;
    } else {
        // Distribute the reduction equally between both panels
        const deficit = totalMax - availableForPanels;
        const reduction = Math.floor(deficit / 2);

        newLeft = maxLeft - reduction;
        newRight = maxRight - reduction;

        // Enforce minimum constraints and transfer excess deficit to the other panel if needed
        if (newLeft < minLeft) {
            newLeft = minLeft;
            newRight = availableForPanels - minLeft;
        } else if (newRight < minRight) {
            newRight = minRight;
            newLeft = availableForPanels - minRight;
        }
    }

    console.log('[play.js] setting widths: newLeft:', newLeft, 'newRight:', newRight);
    if (playPage)  { playPage.style.setProperty('--left-panel-w',  `${newLeft}px`);  playPage.style.setProperty('--right-panel-w',  `${newRight}px`); }
    if (playGrid)  {
        playGrid.style.setProperty('--left-panel-w',  `${newLeft}px`);
        playGrid.style.setProperty('--right-panel-w',  `${newRight}px`);
        const isMobileView = window.innerWidth <= 992;
        if (isMobileView) {
            playGrid.style.gridTemplateColumns = '';
        } else {
            playGrid.style.gridTemplateColumns = `${newLeft}px 1fr ${newRight}px`;
        }
    }
}
window.applyPanelLayout = applyPanelLayout;

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

// Removed ResizeObserver to prevent infinite rendering loops and layout thrashing.
// Window resize listener is sufficient for board overflow checks.

// Letter values for "Valued Letters" format
const LETTER_VALUES = {
    'A': 2, 'B': 4, 'C': 4, 'D': 3, 'E': 1, 'F': 5, 'G': 3, 'H': 5, 'I': 2, 'J': 10, 'K': 6, 'L': 3, 'M': 4, 'N': 2, 'O': 2, 'P': 4, 'Q': 10, 'R': 2, 'S': 2, 'T': 2, 'U': 4, 'V': 5, 'W': 5, 'X': 8, 'Y': 5, 'Z': 8
};

function updateBoardCell(cell, r, c, letter, grayed, f, state = null) {
    if (!cell) return;
    
    // Update basic classes
    cell.className = 'board-cell' + (grayed ? ' grayed' : '');
    
    // Update dataset for identification
    const prevR = cell.dataset.r;
    const prevC = cell.dataset.c;
    cell.dataset.r = r;
    cell.dataset.c = c;
    if (typeof f !== 'undefined') cell.dataset.f = f;

    // Check if letter OR coordinates OR format changed (to ensure point badges are cleared/added correctly)
    const currentLetter = cell.dataset.letter;
    const currentFormat = cell.dataset.renderedFormat;
    const boardFormat = (state && state.current_board_format) ? state.current_board_format : ((window.lastGameState && window.lastGameState.current_board_format) ? window.lastGameState.current_board_format : 'Normal');
    
    if (currentLetter !== letter || prevR !== String(r) || prevC !== String(c) || currentFormat !== boardFormat || cell.children.length <= 1) {
        cell.dataset.letter = letter;
        cell.dataset.renderedFormat = boardFormat;
        cell.innerHTML = ''; // Fresh start
        
        if (letter.includes('/')) {
            cell.classList.add('dual-letter');
            const [top, bottom] = letter.split('/');
            const container = document.createElement('div');
            container.className = 'dual-letter-container';
            const topEl = document.createElement('span');
            topEl.className = 'letter-content';
            topEl.textContent = (top === 'Q' ? 'QU' : top);
            container.appendChild(topEl);
            const divider = document.createElement('div');
            divider.className = 'dual-divider';
            container.appendChild(divider);
            const bottomEl = document.createElement('span');
            bottomEl.className = 'letter-content';
            bottomEl.textContent = (bottom === 'Q' ? 'QU' : bottom);
            container.appendChild(bottomEl);
            cell.appendChild(container);
        } else {
            const letterSpan = document.createElement('span');
            letterSpan.className = 'letter-content';
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
        const bonusCell = (state && typeof state.bonus_cell !== 'undefined') ? state.bonus_cell : (window.lastGameState ? window.lastGameState.bonus_cell : null);
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
    const activeBonusCell = (state && typeof state.bonus_cell !== 'undefined') ? state.bonus_cell : (window.lastGameState ? window.lastGameState.bonus_cell : null);
    let isMatch = false;
    
    if (activeBonusCell) {
        if (Array.isArray(activeBonusCell)) {
            if (activeBonusCell.length === 3) {
                if (typeof f !== 'undefined' && Number(activeBonusCell[0]) === f && Number(activeBonusCell[1]) === r && Number(activeBonusCell[2]) === c) {
                    isMatch = true;
                }
            } else if (activeBonusCell.length === 2) {
                const targetR = Number(activeBonusCell[0]);
                const targetC = Number(activeBonusCell[1]);
                if (targetR === r && targetC === c) {
                    isMatch = true;
                }
            }
        } else if (typeof activeBonusCell === 'object') {
            if (activeBonusCell.f !== undefined) {
                if (typeof f !== 'undefined' && Number(activeBonusCell.f) === f && Number(activeBonusCell.r) === r && Number(activeBonusCell.c) === c) isMatch = true;
            } else if (activeBonusCell.r !== undefined) {
                const targetR = Number(activeBonusCell.r);
                const targetC = Number(activeBonusCell.c);
                if (targetR === r && targetC === c) {
                    isMatch = true;
                }
            }
        }
    }

    // STAR MANAGEMENT: Match the server's is_spec_bonus_fmt rule exactly.
    // scoring.py line 87: is_spec_bonus_fmt = ('bonus' in fmt_lower or 'either' in fmt_lower)
    // Only show the bonus star and highlight when the board format actually awards bonus points.
    // Normal format → no star/highlight (no +3 scored server-side either).
    const existingStar = cell.querySelector('.bonus-star');
    const fmtLower = (boardFormat || '').toLowerCase();
    const isBonusLetterFormat = fmtLower.includes('bonus') || fmtLower.includes('either');
    if (isMatch && isBonusLetterFormat && !existingStar) {
        const star = document.createElement('span');
        star.className = 'bonus-star';
        star.textContent = '★';
        cell.appendChild(star);
    } else if ((!isMatch || !isBonusLetterFormat) && existingStar) {
        existingStar.remove();
    }

    // Apply bonus-highlight background only when the format awards bonus points.
    // Also highlight split-letter cells in "either" format.
    if (isMatch && isBonusLetterFormat) {
        cell.classList.add('bonus-highlight');
    } else if (fmtLower.includes('either') && letter.includes('/')) {
        cell.classList.add('bonus-highlight');
    } else {
        cell.classList.remove('bonus-highlight');
    }

    // Apply Density (This is the DYNAMIC part!)
    applyDensityToCell(cell, r, c, f, state);
}

function applyDensityToCell(cell, r = null, c = null, f = null, state = null) {
    if (r === null || r === undefined) r = parseInt(cell.dataset.r);
    if (c === null || c === undefined) c = parseInt(cell.dataset.c);
    if (f === null || f === undefined) f = cell.dataset.f !== undefined ? parseInt(cell.dataset.f) : undefined;
    const densityData = (state && state.cell_density) ? state.cell_density : (window.lastGameState && window.lastGameState.cell_density);
    const hasDensityData = densityData && Array.isArray(densityData) && densityData.length > 0;
    const boardFormat = (state && state.current_board_format) ? state.current_board_format : (window.lastGameState && window.lastGameState.current_board_format) || 'Normal';
    
    // USER REQUEST: If this is the bonus cell or has selection/flash highlights,
    // do NOT apply density shading to the background.
    // The highlight styles must take absolute precedence.
    const isHighlighted = cell.classList.contains('selected') || 
                          cell.classList.contains('current') || 
                          cell.classList.contains('typing-highlight') || 
                          cell.classList.contains('review-highlight') || 
                          cell.classList.contains('intermission-highlight') || 
                          cell.classList.contains('bonus-highlight') ||
                          cell.classList.contains('tile-flash-blue') ||
                          cell.classList.contains('tile-flash-green') ||
                          cell.classList.contains('tile-flash-red') ||
                          cell.classList.contains('tile-flash-purple');
                          
    if (isHighlighted) {
        cell.style.removeProperty('background');
        cell.style.removeProperty('background-color');
        cell.style.removeProperty('color');
        cell.style.removeProperty('box-shadow');
        cell.style.removeProperty('transition');
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
             // Issue 6: Clamp to minimum 20% lightness so tiles are never pure black (always readable)
             const grayLightness = Math.max(20, Math.round(100 - (ratio * 100)));
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
              
              let textColor;
              if (grayLightness <= 60) {
                  textColor = '#ffffff';
              } else {
                  const textL = Math.round(20 * (100 - grayLightness) / 40);
                  textColor = `hsl(0, 0%, ${textL}%)`;
              }
              cell.style.setProperty('color', textColor, 'important');
         }
    } else if (boardFormat.toLowerCase().includes('density')) {
        // We have no density data BUT we are in density mode. 
        // Do NOT clear the style yet - wait for a state that has it.
        // This prevents the "brief flickers" during heartbeats.
    } else {
        // Reset ONLY if we are truly no longer in density format
        cell.style.removeProperty('background');
        cell.style.removeProperty('background-color');
        cell.style.removeProperty('color');
        cell.style.removeProperty('box-shadow');
        cell.style.removeProperty('transition');
        delete cell.dataset.lastD;
    }
}

function createBoardCell(r, c, letter, grayed, f, state = null) {
    const cell = document.createElement('div');
    cell.dataset.letter = letter;
    updateBoardCell(cell, r, c, letter, grayed, f, state);
    return cell;
}

let bounceBalls = [];
let bounceAnimationId = null;

function startBounceFormat() {
    stopBounceFormat();
    
    const boardEl = document.getElementById('game-board');
    if (!boardEl) return;
    
    // Ensure boardEl has position: relative
    boardEl.style.position = 'relative';
    
    // Determine cell size
    const cell = boardEl.querySelector('.board-cell');
    const cellSize = cell ? cell.offsetWidth : 60;
    
    // Number of balls based on board parameters: exactly 10 balls per 16 tiles
    const rows = window.lastGameState && window.lastGameState.board ? window.lastGameState.board.length : 4;
    const cols = window.lastGameState && window.lastGameState.board && window.lastGameState.board[0] ? window.lastGameState.board[0].length : 4;
    const count = Math.round((rows * cols * 10) / 16);
    
    const colors = [
        'radial-gradient(circle at 30% 30%, #a855f7 0%, #7e22ce 60%, #581c87 100%)', // Glossy Purple
        'radial-gradient(circle at 30% 30%, #22d3ee 0%, #0891b2 60%, #155e75 100%)', // Glossy Cyan
        'radial-gradient(circle at 30% 30%, #f472b6 0%, #db2777 60%, #9d174d 100%)', // Glossy Pink
        'radial-gradient(circle at 30% 30%, #fde047 0%, #ca8a04 60%, #854d0e 100%)', // Glossy Yellow
        'radial-gradient(circle at 30% 30%, #34d399 0%, #059669 60%, #064e3b 100%)', // Glossy Emerald
    ];
    
    // Create balls
    for (let i = 0; i < count; i++) {
        const ball = document.createElement('div');
        ball.className = 'bounce-ball';
        
        // Size with respect to board parameters (cell size)
        // Let's make size random between 1.0x and 1.4x the cell size!
        const sizeMultiplier = 1.0 + Math.random() * 0.4;
        const size = cellSize * sizeMultiplier;
        
        ball.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: ${size}px;
            height: ${size}px;
            border-radius: 50%;
            background: ${colors[i % colors.length]};
            pointer-events: none;
            z-index: 10;
            will-change: transform;
            filter: drop-shadow(0 6px 12px rgba(0,0,0,0.45));
        `;
        
        boardEl.appendChild(ball);
        
        // Random initial position inside board
        // Use offsetWidth/offsetHeight (intrinsic size) not getBoundingClientRect
        // because getBoundingClientRect returns viewport-relative coords which change with scroll
        const boardW = boardEl.offsetWidth;
        const boardH = boardEl.offsetHeight;
        const maxX = Math.max(20, boardW - size);
        const maxY = Math.max(20, boardH - size);
        const x = Math.random() * maxX;
        const y = Math.random() * maxY;
        
        // Progressive speeds based on Bounce multiplier:
        // 1x: low (1.5 to 3.5), 2x: medium (4.5 to 7.5), 3x: high (8.5 to 13.5)
        const bFormat = (window.lastGameState && window.lastGameState.current_board_format) || 'Bounce 1x';
        let speedMin = 1.5;
        let speedRange = 2.0;
        if (bFormat.toLowerCase().includes('3x')) {
            speedMin = 8.5;
            speedRange = 5.0;
        } else if (bFormat.toLowerCase().includes('2x')) {
            speedMin = 4.5;
            speedRange = 3.0;
        }

        const angle = Math.random() * Math.PI * 2;
        const speed = speedMin + Math.random() * speedRange;
        const vx = Math.cos(angle) * speed;
        const vy = Math.sin(angle) * speed;
        
        bounceBalls.push({ el: ball, x, y, vx, vy, size });
    }
    
    // Animation loop
    function updatePhysics() {
        const boardEl = document.getElementById('game-board');
        if (!boardEl) return;
        
        // Use offsetWidth/offsetHeight — these reflect the element's intrinsic layout size
        // and are unaffected by scroll position, unlike getBoundingClientRect()
        const boardW = boardEl.offsetWidth;
        const boardH = boardEl.offsetHeight;
        if (boardW === 0 || boardH === 0) {
            bounceAnimationId = requestAnimationFrame(updatePhysics);
            return;
        }
        
        bounceBalls.forEach(b => {
            b.x += b.vx;
            b.y += b.vy;
            
            const maxX = boardW - b.size;
            const maxY = boardH - b.size;
            
            // Bounce off left/right
            if (b.x <= 0) {
                b.x = 0;
                b.vx = Math.abs(b.vx);
            } else if (b.x >= maxX) {
                b.x = maxX;
                b.vx = -Math.abs(b.vx);
            }
            
            // Bounce off top/bottom
            if (b.y <= 0) {
                b.y = 0;
                b.vy = Math.abs(b.vy);
            } else if (b.y >= maxY) {
                b.y = maxY;
                b.vy = -Math.abs(b.vy);
            }
            
            // Apply transform for performance
            b.el.style.transform = `translate3d(${b.x}px, ${b.y}px, 0)`;
        });
        
        bounceAnimationId = requestAnimationFrame(updatePhysics);
    }
    
    bounceAnimationId = requestAnimationFrame(updatePhysics);
}

function stopBounceFormat() {
    if (bounceAnimationId) {
        cancelAnimationFrame(bounceAnimationId);
        bounceAnimationId = null;
    }
    
    const existingBalls = document.querySelectorAll('.bounce-ball');
    existingBalls.forEach(el => el.remove());
    bounceBalls = [];
}

function startRotatingLetters() {
    if (window.rotatingLettersInterval) clearInterval(window.rotatingLettersInterval);
    
    // Initial rotation
    rotateLettersRandomly();
    
    window.rotatingLettersInterval = setInterval(() => {
        rotateLettersRandomly();
    }, 4000); // Rotate every 4 seconds
}

function rotateLettersRandomly() {
    const letters = document.querySelectorAll('.board-cell .letter-content');
    letters.forEach(el => {
        const angles = [0, 90, 180, 270];
        const angle = angles[Math.floor(Math.random() * angles.length)];
        el.style.transform = `rotate(${angle}deg)`;
        el.style.transition = 'transform 0.5s ease'; // Smooth rotation
        el.style.display = 'inline-block'; // Ensure transform works on spans!
    });
}

function stopRotatingLetters() {
    if (window.rotatingLettersInterval) {
        clearInterval(window.rotatingLettersInterval);
        window.rotatingLettersInterval = null;
    }
    // Reset rotations
    const letters = document.querySelectorAll('.board-cell .letter-content');
    letters.forEach(el => {
        el.style.transform = '';
        el.style.transition = '';
    });
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

    // Identify all potential bonus coordinates
    const specialCoords = new Set();
    
    // 1. Add Either/Or tiles directly from the current board
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            if (board[r][c] && board[r][c].includes('/')) {
                specialCoords.add(`${r},${c}`);
            }
        }
    }

    // 2. Add transposed/untransposed bonus letter coordinate
    let bc = targetCoord;
    if (!bc) {
        bc = (window.lastGameState && window.lastGameState.bonus_cell) ? window.lastGameState.bonus_cell : null;
    }
    if (bc) {
        if (Array.isArray(bc)) {
            if (bc.length === 2) {
                const targetR = Number(bc[0]);
                const targetC = Number(bc[1]);
                specialCoords.add(`${targetR},${targetC}`);
            }
        } else if (typeof bc === 'object') {
            if (bc.r !== undefined) {
                const targetR = Number(bc.r);
                const targetC = Number(bc.c);
                specialCoords.add(`${targetR},${targetC}`);
            }
        }
    }

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
        
        let nowHit = hasHitTarget || specialCoords.has(`${r},${c}`);

        const nextIndex = index + matchLength;
        if (nextIndex >= upperWord.length) {
            if (specialCoords.size > 0 && !nowHit) return null;
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

    // Try starting from all possible cells to find a path that hits a special tile
    if (specialCoords.size > 0) {
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const path = dfs(r, c, 0, [], new Set(), false);
                if (path) return path;
            }
        }
    }
    
    // If no bonus-hitting path found or no special tiles, fallback to finding ANY valid path
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

window.findWordPathOnCube = function(word, board, targetCoord = null) {
    if (!word || !board || board.length !== 6) return null;
    const upperWord = word.toUpperCase();

    // Identify all potential bonus coordinates on this 3D board
    const specialCoords = new Set();
    
    // 1. Add Either/Or tiles directly from the current board
    for (let f = 0; f < 6; f++) {
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                if (board[f][r][c] && board[f][r][c].includes('/')) {
                    specialCoords.add(`${f},${r},${c}`);
                }
            }
        }
    }

    // 2. Add bonus letter coordinate
    let bc = targetCoord;
    if (!bc) {
        bc = (window.lastGameState && window.lastGameState.bonus_cell) ? window.lastGameState.bonus_cell : null;
    }
    if (bc && Array.isArray(bc) && bc.length === 3) {
        specialCoords.add(`${bc[0]},${bc[1]},${bc[2]}`);
    } else if (bc && typeof bc === 'object' && bc.f !== undefined) {
        specialCoords.add(`${bc.f},${bc.r},${bc.c}`);
    }

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

    function dfs(f, r, c, index, currentPath, visited, hasHitTarget) {
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

        let nowHit = hasHitTarget || specialCoords.has(`${f},${r},${c}`);

        const nextIndex = index + matchLength;
        if (nextIndex >= upperWord.length) {
            if (specialCoords.size > 0 && !nowHit) return null;
            return newPath;
        }

        for (const n of getCubeNeighbors(f, r, c)) {
            const result = dfs(n.f, n.r, n.c, nextIndex, newPath, newVisited, nowHit);
            if (result) return result;
        }
        return null;
    }

    // Try starting from all possible cells to find a path that hits a special coordinate
    if (specialCoords.size > 0) {
        for (let f = 0; f < 6; f++) {
            for (let r = 0; r < 3; r++) {
                for (let c = 0; c < 3; c++) {
                    const path = dfs(f, r, c, 0, [], new Set(), false);
                    if (path) return path;
                }
            }
        }
    }

    // Fallback to standard DFS
    function dfsBasic(f, r, c, index, currentPath, visited) {
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
            const result = dfsBasic(n.f, n.r, n.c, nextIndex, newPath, newVisited);
            if (result) return result;
        }
        return null;
    }

    for (let f = 0; f < 6; f++) {
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                const path = dfsBasic(f, r, c, 0, [], new Set());
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
    let board = window.lastGameState ? window.lastGameState.board : null;
    if (activeWordsTab === 'previous' && window.lastGameState && window.lastGameState.previous_board && window.lastGameState.previous_board.length > 0) {
        board = window.lastGameState.previous_board;
    }
    if (!board) return;

    // Clear PREVIOUS highlights of ALL types to avoid stale visuals
    document.querySelectorAll('.board-cell').forEach(cell => {
        cell.classList.remove('selected', 'current', 'typing-highlight', 'review-highlight', 'intermission-highlight');
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

    // 5. Reapply intermission tile click filter indicator
    if (window.lastGameState && window.lastGameState.state === 'intermission' && window.intermissionTileFilter) {
        const { r, c, f } = window.intermissionTileFilter;
        let selector = `.board-cell[data-r="${r}"][data-c="${c}"]`;
        if (f !== undefined && f !== null) {
            selector = `.board-cell[data-f="${f}"][data-r="${r}"][data-c="${c}"]`;
        }
        const cell = document.querySelector(selector);
        if (cell) {
            cell.classList.add('intermission-highlight');
        }
    }
}

function renderSplitNotepads(players, state) {
    const boardEl = document.getElementById('game-board');
    if (!boardEl) return;

    // Capture scroll positions before clearing
    const scrollMap = {};
    const containerScrollTop = boardEl.scrollTop;
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
            btn.ontouchstart = (e) => {
                e.preventDefault();
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

        // Add touch listener to manually scroll container
        let touchStartY = 0;
        list.addEventListener('touchstart', (e) => {
            touchStartY = e.touches[0].clientY;
        }, { passive: true });

        list.addEventListener('touchmove', (e) => {
            const touchEndY = e.touches[0].clientY;
            const deltaY = touchStartY - touchEndY;
            if (Math.abs(deltaY) > 5) {
                const container = document.querySelector('.split-notepads-container');
                if (container) {
                    container.scrollTop += deltaY;
                    touchStartY = touchEndY;
                }
            }
        }, { passive: true });

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
            list.scrollTop -= rowHeight;
        };
        btnUp.ontouchstart = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const firstItem = list.querySelector('.notepad-item');
            const rowHeight = firstItem ? firstItem.offsetHeight + 4 : 29; // height + 4px gap
            list.scrollTop -= rowHeight;
        };

        const btnDown = document.createElement('button');
        btnDown.className = 'notepad-scroll-btn down';
        btnDown.innerHTML = '▼';
        btnDown.onclick = (e) => {
            e.stopPropagation();
            const firstItem = list.querySelector('.notepad-item');
            const rowHeight = firstItem ? firstItem.offsetHeight + 4 : 29; // height + 4px gap
            list.scrollTop += rowHeight;
        };
        btnDown.ontouchstart = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const firstItem = list.querySelector('.notepad-item');
            const rowHeight = firstItem ? firstItem.offsetHeight + 4 : 29; // height + 4px gap
            list.scrollTop += rowHeight;
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
                    ptsDisplay += ` <small style="opacity:0.7;">(${w.shared_count})</small>`;
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

    // Restore container scroll position
    boardEl.scrollTop = containerScrollTop;
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

    // Capture container scroll position before clearing
    const containerScrollTop = boardEl.scrollTop;

    // Sort players by score
    const sortedPlayers = [...players].sort((a, b) => b.score - a.score);

    // Reuse Split Points container styles
    boardEl.className = 'split-notepads-container fcfs-mode';
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

        // Add touch listener to manually scroll container
        let touchStartY = 0;
        list.addEventListener('touchstart', (e) => {
            touchStartY = e.touches[0].clientY;
        }, { passive: true });

        list.addEventListener('touchmove', (e) => {
            const touchEndY = e.touches[0].clientY;
            const deltaY = touchStartY - touchEndY;
            if (Math.abs(deltaY) > 5) {
                const container = document.querySelector('.split-notepads-container');
                if (container) {
                    container.scrollTop += deltaY;
                    touchStartY = touchEndY;
                }
            }
        }, { passive: true });

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
            list.scrollTop -= rowHeight;
        };
        btnUp.ontouchstart = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const firstItem = list.querySelector('.notepad-item');
            const rowHeight = firstItem ? firstItem.offsetHeight + 4 : 29; // height + 4px gap
            list.scrollTop -= rowHeight;
        };

        const btnDown = document.createElement('button');
        btnDown.className = 'notepad-scroll-btn down';
        btnDown.innerHTML = '▼';
        btnDown.onclick = (e) => {
            e.stopPropagation();
            const firstItem = list.querySelector('.notepad-item');
            const rowHeight = firstItem ? firstItem.offsetHeight + 4 : 29; // height + 4px gap
            list.scrollTop += rowHeight;
        };
        btnDown.ontouchstart = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const firstItem = list.querySelector('.notepad-item');
            const rowHeight = firstItem ? firstItem.offsetHeight + 4 : 29; // height + 4px gap
            list.scrollTop += rowHeight;
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

    // Restore container scroll position
    boardEl.scrollTop = containerScrollTop;
}


function addSplitViewBoardToggle() {
    const isMobile = window.innerWidth <= 992;
    const timerDisplay = document.querySelector('.timer-display');
    const panelHeader = document.querySelector('.words-panel h3');

    // Check if button already exists
    let btn = document.getElementById('toggle-board-btn');
    if (!btn) {
        btn = document.createElement('button');
        btn.id = 'toggle-board-btn';
        btn.onclick = () => {
            showBoardInSplitIntermission = !showBoardInSplitIntermission;
            updateGameState();
        };
    }

    btn.textContent = showBoardInSplitIntermission ? 'Show Notepads' : 'Show Board';

    if (isMobile && timerDisplay) {
        // Mobile style and placement inside timer display (same as rotate button position)
        btn.className = 'rotate-btn'; // Matches .timer-display .rotate-btn CSS rule
        btn.style.fontSize = ''; // Clear desktop styling overrides
        btn.style.marginLeft = '';
        btn.style.padding = '';
        if (btn.parentElement !== timerDisplay) {
            timerDisplay.appendChild(btn);
        }
    } else if (panelHeader) {
        // Desktop style and placement
        btn.className = 'active-room-btn';
        btn.style.fontSize = '0.7rem';
        btn.style.marginLeft = '10px';
        btn.style.padding = '2px 8px';
        if (btn.parentElement !== panelHeader) {
            panelHeader.appendChild(btn);
        }
    }
}

// Spinner Logic
// Word Submission
// Initialize Word Submission Listeners
function initWordSubmission() {
    if (window.hasInitWordSubmission) return;
    window.hasInitWordSubmission = true;

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
    if (window.isPopupVisible) {
        e.preventDefault();
        e.stopPropagation();
        return;
    }
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
            const isMobile = window.innerWidth <= 992;
            if (chatInput && !isMobile) setTimeout(() => chatInput.focus(), 150);
        }

        // Fast path: if empty or too short, clear highlight instantly and cancel any pending search
        if (!word || word.length < 3) {
            if (typingHighlightTimeout) {
                clearTimeout(typingHighlightTimeout);
                typingHighlightTimeout = null;
            }
            if (!window.isProgrammaticClear) {
                document.querySelectorAll('.board-cell.typing-highlight').forEach(c => {
                    c.classList.remove('typing-highlight');
                    applyDensityToCell(c);
                });
            }
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
                document.querySelectorAll('.board-cell.typing-highlight').forEach(c => {
                    c.classList.remove('typing-highlight');
                    applyDensityToCell(c);
                });
                return;
            }

            const cellsToUpdate = new Set();
            document.querySelectorAll('.board-cell.typing-highlight').forEach(c => {
                c.classList.remove('typing-highlight');
                cellsToUpdate.add(c);
            });
            if (!board || (mouseState && mouseState.isDown)) {
                cellsToUpdate.forEach(c => applyDensityToCell(c));
                return;
            }

            const is3D = board.length === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]);
            const path = is3D ? findWordPathOnCube(word, board) : findWordPathOnBoard(word, board);
            if (path) {
                path.forEach(coord => {
                    let selector = `.board-cell[data-r="${coord.r}"][data-c="${coord.c}"]`;
                    if (coord.f !== undefined) selector = `.board-cell[data-f="${coord.f}"][data-r="${coord.r}"][data-c="${coord.c}"]`;
                    const cell = document.querySelector(selector);
                    if (cell) {
                        cell.classList.add('typing-highlight');
                        cellsToUpdate.add(cell);
                    }
                });
            }
            cellsToUpdate.forEach(c => applyDensityToCell(c));
        }, 30);
    });
}
initWordSubmission();


function clearSubmissionVisuals() {
    document.querySelectorAll('.board-cell.typing-highlight').forEach(c => {
        c.classList.remove('typing-highlight');
        applyDensityToCell(c);
    });
    const isIntermission = window.lastGameState && window.lastGameState.state === 'intermission';
    if (!isIntermission) {
        const defHeader = document.getElementById('definition-header');
        if (defHeader) defHeader.style.display = 'none';
        const defContent = document.getElementById('definition-content');
        if (defContent) defContent.innerHTML = '<p class="placeholder">Select a word to see its definition</p>';
    }
}

function calculateWordScoreLocally(word, path) {
    if (!word) return 0;
    const preState = window.lastGameState;
    const boardFormat = (preState && preState.current_board_format) ? preState.current_board_format : 'Normal';
    const fmtLower = boardFormat.toLowerCase();
    const isValuedFormat = (fmtLower.includes('valued') || fmtLower.includes('value'));
    const length = word.length;
    let score = 0;

    // 1. Base Score
    if (isValuedFormat) {
        const letterValues = {
            'A': 2, 'B': 4, 'C': 4, 'D': 3, 'E': 1, 'F': 5, 'G': 3, 'H': 5, 'I': 2, 'J': 10,
            'K': 6, 'L': 3, 'M': 4, 'N': 2, 'O': 2, 'P': 4, 'Q': 10, 'R': 2, 'S': 2, 'T': 2,
            'U': 4, 'V': 5, 'W': 5, 'X': 8, 'Y': 5, 'Z': 8
        };
        const chars = word.toUpperCase().split('');
        let i = 0;
        while (i < chars.length) {
            const char = chars[i];
            if (char === 'Q' && i + 1 < chars.length && chars[i + 1] === 'U') {
                score += letterValues['Q'] || 10;
                i += 2;
            } else {
                score += letterValues[char] || 1;
                i += 1;
            }
        }
    } else {
        if (length <= 2) score = 0;
        else if (length <= 4) score = 1;
        else if (length === 5) score = 2;
        else if (length === 6) score = 3;
        else if (length === 7) score = 5;
        else score = 11;
    }

    // 2. Hidden Bonus Word (+Length)
    if (preState && preState.bonus_word && word.toUpperCase() === preState.bonus_word.toUpperCase()) {
        score += length;
    }

    // 3. Format Bonus (+3 points for Either/Or or Bonus Letter tile)
    const isSpecBonusFmt = (fmtLower.includes('bonus letter') || fmtLower.includes('either'));
    if (preState && preState.board && isSpecBonusFmt && !fmtLower.includes('checkerboard')) {
        let usedBonus = false;
        const bonusCell = preState.bonus_cell;
        const board = preState.board;
        const is3D = (board.length === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]));

        if (path && Array.isArray(path)) {
            for (const node of path) {
                let f = -1, r = -1, c = -1;
                if (Array.isArray(node)) {
                    if (node.length === 3) {
                        f = Number(node[0]);
                        r = Number(node[1]);
                        c = Number(node[2]);
                    } else if (node.length === 2) {
                        r = Number(node[0]);
                        c = Number(node[1]);
                    }
                } else if (node && typeof node === 'object') {
                    f = node.f !== undefined ? Number(node.f) : -1;
                    r = node.r !== undefined ? Number(node.r) : -1;
                    c = node.c !== undefined ? Number(node.c) : -1;
                }

                if (fmtLower.includes('bonus letter') && bonusCell) {
                    let targetF = -1, targetR = -1, targetC = -1;
                    if (Array.isArray(bonusCell)) {
                        if (bonusCell.length === 3) {
                            targetF = Number(bonusCell[0]);
                            targetR = Number(bonusCell[1]);
                            targetC = Number(bonusCell[2]);
                        } else {
                            targetR = Number(bonusCell[0]);
                            targetC = Number(bonusCell[1]);
                        }
                    } else if (typeof bonusCell === 'object') {
                        targetF = bonusCell.f !== undefined ? Number(bonusCell.f) : -1;
                        targetR = bonusCell.r !== undefined ? Number(bonusCell.r) : -1;
                        targetC = bonusCell.c !== undefined ? Number(bonusCell.c) : -1;
                    }

                    if (is3D) {
                        if (f === targetF && r === targetR && c === targetC) {
                            usedBonus = true;
                            break;
                        }
                    } else {
                        if (r === targetR && c === targetC) {
                            usedBonus = true;
                            break;
                        }
                    }
                }

                let cellVal = '';
                if (is3D) {
                    if (f >= 0 && f < board.length && r >= 0 && r < board[f].length && c >= 0 && c < board[f][r].length) {
                        cellVal = String(board[f][r][c]);
                    }
                } else {
                    if (r >= 0 && r < board.length && c >= 0 && c < board[0].length) {
                        cellVal = String(board[r][c]);
                    }
                }

                if (fmtLower.includes('either') && cellVal && cellVal.includes('/')) {
                    usedBonus = true;
                    break;
                }
            }
        }
        if (!usedBonus) {
            const fallbackPath = is3D 
                ? (typeof findWordPathOnCube === 'function' ? findWordPathOnCube(word, board) : null)
                : (typeof findWordPathOnBoard === 'function' ? findWordPathOnBoard(word, board) : null);
            
            if (fallbackPath) {
                let fallbackHitsBonus = false;
                const specialCoords = new Set();
                
                if (fmtLower.includes('either')) {
                    const rows = board.length;
                    const cols = is3D ? 0 : board[0].length;
                    if (is3D) {
                        for (let f = 0; f < 6; f++) {
                            for (let r = 0; r < board[f].length; r++) {
                                for (let c = 0; c < board[f][r].length; c++) {
                                    if (board[f][r][c] && String(board[f][r][c]).includes('/')) {
                                        specialCoords.add(`${f},${r},${c}`);
                                    }
                                }
                            }
                        }
                    } else {
                        for (let r = 0; r < rows; r++) {
                            for (let c = 0; c < cols; c++) {
                                if (board[r][c] && String(board[r][c]).includes('/')) {
                                    specialCoords.add(`${r},${c}`);
                                }
                            }
                        }
                    }
                }
                
                if (fmtLower.includes('bonus letter') && bonusCell) {
                    let targetF = -1, targetR = -1, targetC = -1;
                    if (Array.isArray(bonusCell)) {
                        if (bonusCell.length === 3) {
                            targetF = Number(bonusCell[0]); targetR = Number(bonusCell[1]); targetC = Number(bonusCell[2]);
                            specialCoords.add(`${targetF},${targetR},${targetC}`);
                        } else {
                            targetR = Number(bonusCell[0]); targetC = Number(bonusCell[1]);
                            specialCoords.add(`${targetR},${targetC}`);
                        }
                    } else if (typeof bonusCell === 'object') {
                        targetF = bonusCell.f !== undefined ? Number(bonusCell.f) : -1;
                        targetR = bonusCell.r !== undefined ? Number(bonusCell.r) : -1;
                        targetC = bonusCell.c !== undefined ? Number(bonusCell.c) : -1;
                        if (is3D) specialCoords.add(`${targetF},${targetR},${targetC}`);
                        else specialCoords.add(`${targetR},${targetC}`);
                    }
                }
                
                for (const node of fallbackPath) {
                    let key = '';
                    if (Array.isArray(node)) {
                        if (node.length === 3) key = `${node[0]},${node[1]},${node[2]}`;
                        else key = `${node[0]},${node[1]}`;
                    } else if (node && typeof node === 'object') {
                        if (node.f !== undefined) key = `${node.f},${node.r},${node.c}`;
                        else key = `${node.r},${node.c}`;
                    }
                    if (specialCoords.has(key)) {
                        fallbackHitsBonus = true;
                        break;
                    }
                }
                if (fallbackHitsBonus) {
                    usedBonus = true;
                }
            }
        }
        if (usedBonus) {
            score += 3;
        }
    }

    return score;
}

async function submitWord(wordParam = null, pathParam = null, _quFallback = false) {
    try {
        const input = document.getElementById('word-input');
    let word = wordParam ? wordParam.toUpperCase() : (input ? input.value.trim().toUpperCase() : '');
    const roomId = getCurrentRoomId();
    
    // Clear input immediately (no yellow tint — it's visual noise before validation color)
    if (input) {
        window.isProgrammaticClear = true;
        input.value = '';
        input.dispatchEvent(new Event('input'));
        window.isProgrammaticClear = false;
    }

    console.log('[play.js] submitWord entering:', word, 'Room:', roomId);
    if (!word) {
        console.warn('[play.js] Empty word submission ignored');
        return;
    }



    // 1. PATH RESOLUTION
    let finalPath = pathParam;
    let usesEitherOrTile = false;
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
    
    // --- EITHER/OR PATH RESOLUTION ---
    const boardFormat = (window.lastGameState && window.lastGameState.current_board_format) ? window.lastGameState.current_board_format : 'Normal';
    const isEO = boardFormat.toLowerCase().includes('either');
    if (finalPath && isEO && board) {
        let possibleWords = [''];
        let validPath = true;

        for (const node of finalPath) {
            let cellVal = '';
            if (node.length === 3) {
                const [f, r, c] = node;
                if (f >= 0 && f < board.length && r >= 0 && r < board[f].length && c >= 0 && c < board[f][r].length) {
                    cellVal = String(board[f][r][c]);
                } else {
                    validPath = false;
                    break;
                }
            } else {
                const [r, c] = node;
                if (r >= 0 && r < board.length && c >= 0 && c < board[0].length) {
                    cellVal = String(board[r][c]);
                } else {
                    validPath = false;
                    break;
                }
            }

            if (cellVal.includes('/')) {
                usesEitherOrTile = true;
                const options = cellVal.split('/');
                let newWords = [];
                for (const prefix of possibleWords) {
                    for (const opt of options) {
                        newWords.push(prefix + opt);
                    }
                }
                possibleWords = newWords;
            } else {
                for (let i = 0; i < possibleWords.length; i++) {
                    possibleWords[i] += cellVal;
                }
            }
        }

        if (validPath) {
            let validOptions = [];
            let wordList = [];
            if (window.lastGameState && window.lastGameState.all_words) {
                const allWordsState = window.lastGameState.all_words;
                if (Array.isArray(allWordsState)) {
                    wordList = allWordsState;
                } else if (allWordsState && typeof allWordsState === 'object') {
                    wordList = Object.keys(allWordsState);
                }
            }
            for (const w of possibleWords) {
                const isVal = wordList.some(item => {
                    const wText = (typeof item === 'string' ? item : (item.word || '')) || '';
                    return wText.toUpperCase() === w.toUpperCase();
                });
                if (isVal) {
                    validOptions.push(w);
                }
            }

            let submittedClean = word.replace(/\//g, '');
            if (validOptions.includes(submittedClean)) {
                word = submittedClean;
            } else if (validOptions.length >= 1) {
                word = validOptions[0];
            } else if (possibleWords.includes(submittedClean)) {
                word = submittedClean;
            } else if (possibleWords.length > 0) {
                word = possibleWords[0];
            }
        }
    }

    // Define currentUser for consistency in local updates
    let currentUser = window.currentUser || (window.lastGameState && window.lastGameState.your_username) || localStorage.getItem('morpheme_username') || '';
    currentUser = currentUser.trim();

    console.log(`[play.js] Attempting submission: "${word}" (Path: ${finalPath ? 'Yes' : 'No'})`);

        if (isTournamentPlay) {
            handleTournamentWord(word, finalPath);
            return;
        }

        if (isPrivateMatchPlay) {
            await handlePrivateMatchWord(word, finalPath);
            return;
        }

    if (!word) return;
    
    if (!roomId) {
        console.warn('[play.js] Submission failed: No active Room ID found');
        showValidationFeedback('Not in a game room', false);
        return;
    }

    console.log(`[play.js] Submitting word "${word}" to room ${roomId} via ${currentInputMethod}`);

    // --- INSTANT LOCAL VALIDATION (zero hesitation) ---
    let optimisticColor = null; // 'red' | 'blue' | 'green' | 'purple'
    let optimisticIsDefinitive = false;
    const preState = window.lastGameState;
    if (preState) {
        const minLen = preState.current_min_length || (preState.spinner_params ? preState.spinner_params.min_word_length : 3) || 3;
        const effectiveLen = word.replace(/Q(?!U)/g, 'QU').length;

        let alreadyFound = false;
        if (window._localSubmittedWords && window._localSubmittedWords.has(word)) {
            alreadyFound = true;
        } else if (preState.game_type === 'fcfs') {
            alreadyFound = preState.players && preState.players.some(p =>
                p.submitted_words && p.submitted_words.some(w => (typeof w === 'object' ? (w.word || '') : w).toUpperCase() === word)
            );
        } else {
            const myPlayer = preState.players && preState.players.find(p =>
                p.username && currentUser && p.username.toLowerCase() === currentUser.toLowerCase()
            );
            alreadyFound = myPlayer && myPlayer.submitted_words &&
                myPlayer.submitted_words.some(w => (typeof w === 'object' ? (w.word || '') : w).toUpperCase() === word);
        }

        if (alreadyFound) {
            // Definitively already found — flash purple immediately and return
            showValidationFeedback('Already found!', false, false, finalPath);
            clearSubmissionVisuals();
            return;
        } else if (effectiveLen < minLen) {
            // Too short — flash TOO SHORT immediately and return
            showValidationFeedback(`${word.toUpperCase()} IS TOO SHORT (MIN: ${minLen}L)`, false, false, finalPath);
            clearSubmissionVisuals();
            return;
        } else {
            // Check dictionary validity locally using the authoritative all_words list from the game state
            const allWordsState = preState.all_words || [];
            let wordList = [];
            if (Array.isArray(allWordsState)) {
                wordList = allWordsState;
            } else if (allWordsState && typeof allWordsState === 'object') {
                wordList = Object.keys(allWordsState);
            }

            const curFmtLow = String(preState.board_format || '').toLowerCase();
            const isPenaltyMode = curFmtLow.includes('penalty');

            if (wordList.length > 0) {
                // Word list is populated and trustworthy — check locally for instant flash.
                const isInWordList = wordList.some(w =>
                    (typeof w === 'string' ? w : (w.word || '')).toUpperCase() === word
                );

                if (isInWordList) {
                    // Confirmed valid — check if it's Bonus Word OR uses Bonus Letter tile
                    const isBonusWord = preState.bonus_word && word === preState.bonus_word.toUpperCase();
                    let usesBonusLetterTile = false;
                    const _bonusFmtLow = (preState.current_board_format || '').toLowerCase();
                    const _isBonusFmt = _bonusFmtLow.includes('bonus') || _bonusFmtLow.includes('either');
                    if (_isBonusFmt && preState.bonus_cell && finalPath) {
                        const bc = preState.bonus_cell;
                        let targetF = -1, targetR = -1, targetC = -1;
                        if (Array.isArray(bc)) {
                            if (bc.length === 3) { targetF = Number(bc[0]); targetR = Number(bc[1]); targetC = Number(bc[2]); }
                            else if (bc.length === 2) { targetR = Number(bc[0]); targetC = Number(bc[1]); }
                        } else if (typeof bc === 'object') {
                            targetF = bc.f !== undefined ? Number(bc.f) : -1;
                            targetR = bc.r !== undefined ? Number(bc.r) : -1;
                            targetC = bc.c !== undefined ? Number(bc.c) : -1;
                        }
                        for (const node of finalPath) {
                            let f = -1, r = -1, c = -1;
                            if (Array.isArray(node)) {
                                if (node.length === 3) { f = Number(node[0]); r = Number(node[1]); c = Number(node[2]); }
                                else if (node.length === 2) { r = Number(node[0]); c = Number(node[1]); }
                            } else if (node && typeof node === 'object') {
                                f = node.f !== undefined ? Number(node.f) : -1;
                                r = node.r !== undefined ? Number(node.r) : -1;
                                c = node.c !== undefined ? Number(node.c) : -1;
                            }
                            if (targetF !== -1 ? (f === targetF && r === targetR && c === targetC) : (r === targetR && c === targetC)) {
                                usesBonusLetterTile = true;
                                break;
                            }
                        }
                    }

                    const isBonus = isBonusWord;
                    const showBonusMsg = isBonusWord && !usesEitherOrTile;
                    const localPts = calculateWordScoreLocally(word, finalPath);
                    const msg = showBonusMsg 
                        ? `BONUS WORD! (${localPts} PTS)`
                        : (usesBonusLetterTile ? `${word.toUpperCase()} VALID (+3 BONUS LTR! ${localPts} PTS)` : `${word.toUpperCase()} VALID (${localPts} PTS)`);
                    
                    showValidationFeedback(msg, true, isBonusWord, finalPath, true);
                    optimisticColor = isBonusWord ? 'green' : 'blue';
                    optimisticIsDefinitive = true;

                    // Optimistic Instant UI Update: Add word to local lists immediately
                    window._localSubmittedWords = window._localSubmittedWords || new Set();
                    if (!window._localSubmittedWords.has(word)) {
                        window._localSubmittedWords.add(word);
                        window._localSubmittedWordsList = window._localSubmittedWordsList || [];
                        window._localSubmittedWordsList.push({
                            word: word,
                            points: localPts,
                            score_details: {},
                            time: Date.now() / 1000
                        });

                        if (typeof updateGameState === 'function') {
                            updateGameState();
                        }
                    }
                } else if (!isPenaltyMode) {
                    // Confirmed invalid locally in non-penalty mode — flash red and return immediately
                    showValidationFeedback(`${word.toUpperCase()} INVALID`, false, false, finalPath);
                    clearSubmissionVisuals();
                    return;
                }
            }
        }
    }

    let serverPath = finalPath;
    if (serverPath && window.isBoardTransposed) {
        serverPath = serverPath.map(node => {
            if (Array.isArray(node) && node.length === 2) {
                return [node[1], node[0]]; // Swapping r and c back to untranspose for server
            }
            return node;
        });
    }

    try {
        // Use a timeout + keepalive to handle mobile network stalls gracefully.
        // keepalive ensures the request completes even if the page visibility changes mid-flight.
        // Timeout is 8s — fast enough to surface real errors, long enough for normal server load.
        const submitBodyJson = JSON.stringify({
            word: word,
            input_method: currentInputMethod,
            path: serverPath
        });

        let response;
        for (let attempt = 0; attempt < 2; attempt++) {
            const controller = new AbortController();
            const fetchTimeout = setTimeout(() => controller.abort(), 8000);
            try {
                response = await fetch(`/room/${roomId}/submit_word`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    keepalive: true,
                    signal: controller.signal,
                    body: submitBodyJson
                });
            } finally {
                clearTimeout(fetchTimeout);
            }
            // 503 = server lock busy (board generation in progress). Retry once after 300ms.
            if (response.status === 503 && attempt === 0) {
                console.warn(`[play.js] Submit got 503 (server busy) for "${word}" — retrying in 300ms`);
                await new Promise(r => setTimeout(r, 300));
                continue;
            }
            break;
        }
        const data = await response.json();

        // QU-TILE FALLBACK (mouse/swipe only): If the word failed and it contains QU
        // from a Q tile, strip the U and resubmit. E.g. QUANAT → QANAT.
        // Only fires once (_quFallback guard) and only for mouse/swipe (pathParam non-null).
        if (!data.success && !_quFallback && pathParam !== null && /QU/.test(word)) {
            const strippedWord = word.replace(/QU/g, 'Q');
            if (strippedWord !== word) {
                console.log(`[play.js] QU-tile fallback: retrying "${word}" as "${strippedWord}"`);
                submitWord(strippedWord, pathParam, true);
                return;
            }
        }

        // Determine the server's actual color result.
        const currentState = window.lastGameState;
        const isBonusWord = data.success && currentState && currentState.bonus_word && data.word && data.word.toUpperCase() === currentState.bonus_word.toUpperCase();
        const hasBonusLetter = data.score_details ? ((data.score_details.bonus_letter_points || 0) > 0) : false;
        const isBonus = isBonusWord;
        const serverIsPenalty = data.message && data.message.toUpperCase().includes('PENALTY');
        const serverIsActuallyValid = data.success && !serverIsPenalty;
        const serverIsAlreadyFound = data.message && data.message.toUpperCase().includes('ALREADY FOUND');
        let serverColor = 'red';
        if (serverIsAlreadyFound) {
            serverColor = 'purple';
        } else if (serverIsActuallyValid) {
            serverColor = isBonusWord ? 'green' : 'blue';
        }

        if (!finalPath && data.path) {
            finalPath = data.path;
        }

        let msg;
        if (data.success) {
            msg = isBonusWord 
                ? `BONUS WORD! (${data.points} PTS)`
                : (hasBonusLetter ? `${(data.word || word).toUpperCase()} VALID (+3 BONUS LTR! ${data.points} PTS)` : `${(data.word || word).toUpperCase()} VALID (${data.points} PTS)`);
        } else {
            msg = data.message || `${word.toUpperCase()} INVALID`;
        }

        if (optimisticColor !== null && optimisticColor === serverColor) {
            // Local optimistic check ALREADY flashed tiles/input and showed feedback!
            // Skip redundant second flash on server response.
        } else {
            // Local check didn't run or server result differs from local check — trigger feedback flash
            const shouldPlayServerSound = (optimisticColor === null);
            showValidationFeedback(msg, data.success, isBonusWord, finalPath, shouldPlayServerSound);
        }



        let isSpecialSkip = false;
        const preState = window.lastGameState;
        if (preState) {
            const minLen = preState.current_min_length || (preState.spinner_params ? preState.spinner_params.min_word_length : 3) || 3;
            const effectiveLen = word.replace(/Q(?!U)/g, 'QU').length;
            
            let alreadyFound = false;
            if (window._localSubmittedWords && window._localSubmittedWords.has(word)) {
                alreadyFound = true;
            } else if (preState.game_type === 'fcfs') {
                alreadyFound = preState.players && preState.players.some(p =>
                    p.submitted_words && p.submitted_words.some(w => (typeof w === 'object' ? (w.word || '') : w).toUpperCase() === word)
                );
            } else {
                const myPlayer = preState.players && preState.players.find(p =>
                    p.username && currentUser && p.username.toLowerCase() === currentUser.toLowerCase()
                );
                alreadyFound = myPlayer && myPlayer.submitted_words &&
                    myPlayer.submitted_words.some(w => (typeof w === 'object' ? (w.word || '') : w).toUpperCase() === word);
            }
            
            if (alreadyFound || effectiveLen < minLen) {
                isSpecialSkip = true;
            }
        }
        if (serverIsAlreadyFound) {
            isSpecialSkip = true;
        }

        if (data.success) {
            window._localSubmittedWords = window._localSubmittedWords || new Set();
            window._localSubmittedWords.add(word);
            window._localSubmittedWordsList = window._localSubmittedWordsList || [];
            window._localSubmittedWordsList.push({
                word: data.word || word,
                points: data.points || 0,
                score_details: data.score_details || {},
                time: Date.now() / 1000
            });
            recordGuessResult(true, true, isSpecialSkip);
        } else {
            recordGuessResult(false, finalPath && finalPath.length > 0, isSpecialSkip);
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
                    }

                    const me = currentState.players.find(p => p.username.toLowerCase() === (currentUser || "").toLowerCase().trim());
                    if (me) {
                        me.score = Math.max(0, data.new_score);
                        renderPlayers(currentState.players, currentUser, currentState);
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
    } finally {
        clearSubmissionVisuals();
    }
}

function recordGuessResult(isValid, isOnBoard, isSpecialSkip = false) {
    if (isSpecialSkip) return;
    if (isValid) {
        wrongGuessesOnBoardCount = 0;
    } else {
        if (isOnBoard) {
            wrongGuessesOnBoardCount = (wrongGuessesOnBoardCount || 0) + 1;
            if (wrongGuessesOnBoardCount >= 4) {
                showGuessingPopup();
            }
        }
    }
}

let activeGuessingScrollPrevent = null;

function showGuessingPopup() {
    if (window.lastGameState && window.lastGameState.time_limit >= 7200) return;
    if (document.getElementById('guessing-popup')) return;

    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    // Global scroll/swipe lock for the entire document
    activeGuessingScrollPrevent = function(e) {
        if (e.cancelable !== false) {
            e.preventDefault();
        }
    };
    document.addEventListener('wheel', activeGuessingScrollPrevent, { passive: false });
    document.addEventListener('touchmove', activeGuessingScrollPrevent, { passive: false });

    const popup = document.createElement('div');
    popup.id = 'guessing-popup';
    popup.innerHTML = `
        <div class="guessing-popup-card">
            <div class="guessing-popup-icon">⚠️</div>
            <div class="guessing-popup-text">You're guessing!</div>
        </div>
    `;

    popup.style.position = 'fixed';
    popup.style.top = '0';
    popup.style.left = '0';
    popup.style.width = '100vw';
    popup.style.height = '100vh';
    popup.style.display = 'flex';
    popup.style.justifyContent = 'center';
    popup.style.alignItems = 'center';
    popup.style.zIndex = '999999';
    popup.style.backdropFilter = 'blur(8px)';
    popup.style.webkitBackdropFilter = 'blur(8px)';
    popup.style.background = 'rgba(0, 0, 0, 0.4)';
    popup.style.opacity = '0';
    popup.style.transition = 'opacity 0.25s ease';

    if (!document.getElementById('guessing-popup-styles')) {
        const style = document.createElement('style');
        style.id = 'guessing-popup-styles';
        style.textContent = `
            .guessing-popup-card {
                background: var(--card-bg, rgba(30, 30, 30, 0.85));
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border-radius: 16px;
                padding: 30px 50px;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 15px;
                transform: scale(0.9);
                transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
                color: var(--text-primary, #ffffff);
            }
            .guessing-popup-icon {
                font-size: 3rem;
                animation: popup-wiggle 1s ease infinite alternate;
            }
            .guessing-popup-text {
                font-size: 1.6rem;
                font-weight: 800;
                letter-spacing: 0.5px;
                text-align: center;
                background: linear-gradient(135deg, #ff4e50, #f9d423);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: text-pulse 1.5s ease-in-out infinite alternate;
            }
            @keyframes popup-wiggle {
                0% { transform: rotate(-5deg); }
                100% { transform: rotate(5deg); }
            }
            @keyframes text-pulse {
                0% { opacity: 0.85; filter: drop-shadow(0 0 2px rgba(249, 212, 35, 0.3)); }
                100% { opacity: 1; filter: drop-shadow(0 0 10px rgba(255, 78, 80, 0.5)); }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(popup);

    popup.offsetHeight;
    popup.style.opacity = '1';
    popup.querySelector('.guessing-popup-card').style.transform = 'scale(1)';

    window.isPopupVisible = true;

    if (mouseState) {
        mouseState.isDown = false;
        mouseState.selectedPath = [];
        mouseState.visitedCells = new Set();
    }
    document.querySelectorAll('.board-cell.selected, .board-cell.current, .board-cell.typing-highlight').forEach(c => {
        c.classList.remove('selected', 'current', 'typing-highlight');
        applyDensityToCell(c);
    });

    const wordInput = document.getElementById('word-input');
    if (wordInput) {
        wordInput.disabled = true;
        wordInput.blur();
    }
    const boardPanel = document.querySelector('.board-panel') || document.getElementById('game-board');
    if (boardPanel) {
        boardPanel.style.pointerEvents = 'none';
        boardPanel.style.userSelect = 'none';
        boardPanel.style.webkitUserSelect = 'none';
    }

    setTimeout(() => {
        popup.style.opacity = '0';
        popup.querySelector('.guessing-popup-card').style.transform = 'scale(0.9)';
        setTimeout(() => {
            popup.remove();
            document.body.style.overflow = originalOverflow;
            
            // Release global scroll/swipe lock
            if (activeGuessingScrollPrevent) {
                document.removeEventListener('wheel', activeGuessingScrollPrevent, { passive: false });
                document.removeEventListener('touchmove', activeGuessingScrollPrevent, { passive: false });
                activeGuessingScrollPrevent = null;
            }
            
            window.isPopupVisible = false;
            if (wordInput) {
                wordInput.disabled = false;
                const isMobile = window.innerWidth <= 992;
                if (!isMobile) wordInput.focus();
            }
            if (boardPanel) {
                boardPanel.style.pointerEvents = 'auto';
            }
        }, 250);
    }, 3000);
}

function showValidationFeedback(message, isValid, isBonus = false, path = null, playSound = true) {
    const statusEl = document.getElementById('word-validation-status');
    if (!statusEl) return;

    // Clear highlights, but keep active highlights if the user has already started a new swipe or typing sequence
    const isSwipingNewWord = mouseState && mouseState.isDown;
    const inputEl = document.getElementById('word-input');
    const isUserTypingNewWord = inputEl && inputEl.value.trim().length > 0;

    if (!isSwipingNewWord) {
        document.querySelectorAll('.board-cell.selected, .board-cell.current').forEach(c => {
            c.classList.remove('selected', 'current');
            applyDensityToCell(c);
        });
    }
    if (!isUserTypingNewWord) {
        document.querySelectorAll('.board-cell.typing-highlight').forEach(c => {
            c.classList.remove('typing-highlight');
            applyDensityToCell(c);
        });
    }

    // Clear existing timeout
    if (validationTimeout) clearTimeout(validationTimeout);

    const isPenalty = message && message.toUpperCase().includes('PENALTY');
    const isActuallyValid = isValid && !isPenalty;
    const isAlreadyFound = message && (message === 'Already found!' || message.toUpperCase().includes('ALREADY FOUND'));

    // Play validation sound only if requested (prevents double beep when optimistic check already played it)
    if (playSound) {
        if (isActuallyValid) {
            BoardAudio.playSuccessSound();
        } else {
            BoardAudio.playFailureSound();
        }
    }

    // Set text and class
    statusEl.textContent = message;
    if (isAlreadyFound) {
        statusEl.className = 'validation-status status-already-found';
        statusEl.style.color = '#a855f7';
    } else if (isActuallyValid) {
        statusEl.className = 'validation-status status-valid';
        statusEl.style.color = isBonus ? '#22c55e' : '#48bb78';
    } else {
        statusEl.className = 'validation-status status-invalid';
        statusEl.style.color = '#f56565';
    }

    // Flash highlighted tiles that traced the word (instead of full-screen flash)
    const shouldFlash = window.userSettings ? (window.userSettings.word_flash !== false) : true;
    let targetPath = path;
    if ((!targetPath || targetPath.length === 0) && message) {
        const wordMatch = message.match(/^([A-Z]+)\s+/i);
        if (wordMatch && window.lastGameState && window.lastGameState.all_words_paths) {
            const wUpper = wordMatch[1].toUpperCase();
            targetPath = window.lastGameState.all_words_paths[wUpper];
        }
    }

    if (shouldFlash && targetPath && targetPath.length > 0) {
        let tileFlashClass = 'tile-flash-red';
        if (isAlreadyFound) {
            tileFlashClass = 'tile-flash-purple';
        } else if (isActuallyValid) {
            tileFlashClass = isBonus ? 'tile-flash-green' : 'tile-flash-blue';
        }
        
        targetPath.forEach(coord => {
            let r, c, f;
            if (Array.isArray(coord)) {
                if (coord.length === 3) {
                    f = coord[0];
                    r = coord[1];
                    c = coord[2];
                } else if (coord.length === 2) {
                    r = coord[0];
                    c = coord[1];
                }
            } else if (coord && typeof coord === 'object') {
                r = coord.r;
                c = coord.c;
                f = coord.f;
            }
            
            if (r !== undefined && c !== undefined) {
                let selector = `.board-cell[data-r="${r}"][data-c="${c}"]`;
                if (f !== undefined) {
                    selector = `.board-cell[data-f="${f}"][data-r="${r}"][data-c="${c}"]`;
                }
                const cell = document.querySelector(selector);
                if (cell) {
                    cell.classList.remove('tile-flash-blue', 'tile-flash-green', 'tile-flash-red', 'tile-flash-purple');
                    // Trigger reflow to restart animation if already playing
                    void cell.offsetWidth; 
                    cell.classList.add(tileFlashClass);
                    applyDensityToCell(cell);
                    // Rapid, snappy flash effect: 150ms
                    const flashMs = 150;
                    setTimeout(() => {
                        cell.classList.remove(tileFlashClass);
                        applyDensityToCell(cell);
                    }, flashMs);
                }
            }
        });
    }

    // Flash word input background (150ms fast response)
    const input = document.getElementById('word-input');
    if (input) {
        let bgColor = 'rgba(239, 68, 68, 0.35)'; // default invalid red
        if (isAlreadyFound) {
            bgColor = 'rgba(168, 85, 247, 0.35)'; // purple
        } else if (isActuallyValid) {
            bgColor = isBonus ? 'rgba(34, 197, 94, 0.35)' : 'rgba(59, 130, 246, 0.35)'; // green or blue
        }
        input.style.backgroundColor = bgColor;
        input.style.transition = 'background-color 0.08s ease-in-out';
        setTimeout(() => {
            if (input) {
                input.style.backgroundColor = '';
            }
        }, 150);
    }

    // Reset after 3 seconds
    validationTimeout = setTimeout(() => {
        statusEl.textContent = '';
        statusEl.className = 'validation-status';
        validationTimeout = null;
    }, 3000);
}

async function leaveCurrentRoom() {
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
    stopPolling();
    window.currentRoomId = null;
    localStorage.removeItem('last_joined_room');
    window._localSubmittedWords = new Set();
    window._localSubmittedWordsList = [];
    
    const playBtn = document.getElementById('play-btn');
    if (playBtn) {
        playBtn.disabled = true;
        playBtn.classList.remove('active');
        playBtn.title = "Join a room to play.";
    }
    if (window.updateManualToolState) window.updateManualToolState();

    // 2. Non-blocking beacon/fetch to notify server of leave
    const url = `/api/room/${roomId}/leave`;
    if (navigator.sendBeacon) {
        navigator.sendBeacon(url);
    } else {
        fetch(url, { method: 'POST', keepalive: true }).catch(() => {});
    }

    // 3. Refresh lobby stats in background without blocking
    if (typeof window.fetchLobbyStats === 'function') {
        window.fetchLobbyStats().catch(() => {});
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
        if (isPrivateMatchPlay) {
            const confirmLeave = confirm("Leaving mid-round will end your turn and submit your current words. Are you sure?");
            if (!confirmLeave) return;
            await finishPrivateMatchTurn();
            return;
        }
        showPage('page-lobby');
        leaveCurrentRoom();
    });
}

window.isUserBoardTransposed = false;

document.addEventListener('click', (e) => {
    const transposeBtn = e.target ? e.target.closest('#transpose-board-btn') : null;
    if (transposeBtn) {
        e.preventDefault();
        e.stopPropagation();
        window.isUserBoardTransposed = !window.isUserBoardTransposed;
        console.log('[play.js] Transpose clicked! isUserBoardTransposed:', window.isUserBoardTransposed);
        if (window.lastGameState && window.lastGameState.board) {
            let boardToRender = window.lastGameState.board;
            let isIntermission = window.lastGameState.state === 'intermission';
            if (activeWordsTab === 'previous' && window.lastGameState.previous_board && window.lastGameState.previous_board.length > 0) {
                boardToRender = window.lastGameState.previous_board;
                isIntermission = true;
            }
            const is3D = window.lastGameState.game_type === '3d' || (boardToRender && boardToRender.length === 6 && Array.isArray(boardToRender[0]) && Array.isArray(boardToRender[0][0]));
            renderBoard(boardToRender, isIntermission, is3D);
        }
        return;
    }

    const rotateBtn = e.target ? e.target.closest('#rotate-board-btn') : null;
    if (rotateBtn) {
        e.preventDefault();
        e.stopPropagation();
        isBoardRotated = !isBoardRotated;
        console.log('[play.js] Rotate clicked! isBoardRotated:', isBoardRotated);
        if (window.lastGameState && window.lastGameState.board) {
            let boardToRender = window.lastGameState.board;
            let isIntermission = window.lastGameState.state === 'intermission';
            if (activeWordsTab === 'previous' && window.lastGameState.previous_board && window.lastGameState.previous_board.length > 0) {
                boardToRender = window.lastGameState.previous_board;
                isIntermission = true;
            }
            const is3D = window.lastGameState.game_type === '3d' || (boardToRender && boardToRender.length === 6 && Array.isArray(boardToRender[0]) && Array.isArray(boardToRender[0][0]));
            renderBoard(boardToRender, isIntermission, is3D);
        }
        return;
    }
});

// Definition Logic
async function fetchDefinition(word) {
    if (!word) return;
    if (window.isSpectatorMode) return; // Block spectators from overwriting definition panel
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

        if (data.definition || data.pronunciation || data.image_url) {
            let html = '';
            if (data.pronunciation) {
                html += `<div class="pronunciation">${data.pronunciation}</div>`;
            }
            if (data.definition) {
                html += `<span class="definition-text">${data.definition}</span>`;
            }
            if (data.image_url) {
                html += `<div class="definition-image-container" style="margin-top: 15px; text-align: center;"><img src="${data.image_url}" class="definition-image" style="max-width: 100%; max-height: 180px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.4);" /></div>`;
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
    if (window.isPopupVisible) return;
    const key = face !== null ? `${face},${row},${col}` : `${row},${col}`;
    const pathLen = mouseState.selectedPath.length;

    // Check for backtracking (mousing back to the second-to-last letter)
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
            if (oldCellEl) {
                oldCellEl.classList.remove('selected', 'current');
                applyDensityToCell(oldCellEl);
            }

            if (cellEl) {
                document.querySelectorAll('.board-cell.current').forEach(c => {
                    c.classList.remove('current');
                    applyDensityToCell(c);
                });
                cellEl.classList.add('selected', 'current');
                applyDensityToCell(cellEl);
            }

            updateWordInputFromPath();
            BoardAudio.playTileSound(mouseState.selectedPath.length);
            return;
        }
    }

    if (mouseState.visitedCells.has(key)) return;

    // Enforce strict grid adjacency during drag/mouse selection
    if (pathLen > 0) {
        const lastCell = mouseState.selectedPath[pathLen - 1];
        const isAdjacent = (face !== null && lastCell.face !== null) ?
            (Math.abs(lastCell.row - row) <= 1 && Math.abs(lastCell.col - col) <= 1) :
            (Math.abs(lastCell.row - row) <= 1 && Math.abs(lastCell.col - col) <= 1);

        if (!isAdjacent) {
            console.log(`[play.js] Rejecting non-adjacent tile selection: (${row},${col}) from (${lastCell.row},${lastCell.col})`);
            return;
        }
    }

    mouseState.visitedCells.add(key);
    mouseState.selectedPath.push({ row, col, letter, face });

    if (cellEl) {
        document.querySelectorAll('.board-cell.current').forEach(c => {
            c.classList.remove('current');
            applyDensityToCell(c);
        });
        cellEl.classList.add('selected', 'current');
        applyDensityToCell(cellEl);
    }

    updateWordInputFromPath();
    BoardAudio.playTileSound(mouseState.selectedPath.length);
}

function updateWordInputFromPath() {
    const wordInputEl = document.getElementById('word-input');
    if (wordInputEl) {
        wordInputEl.value = mouseState.selectedPath.map(p => {
            const L = p.letter; // Show full letter string (e.g. "L/T")
            if (L && L.includes('/')) {
                return `[${L}]`;
            }
            return L === 'Q' ? 'QU' : L;
        }).join('');
    }
}

function handleIntermissionTilePress(cell, r, c, f, letter) {
    // Do not allow a letter to be pressed on the board during intermission until 5 seconds into the intermission have elapsed
    if (window.lastGameState && window.lastGameState.state === 'intermission') {
        const intermission_duration = (window.lastGameState.time_limit >= 7200) ? 5 : 60;
        const now = Date.now() / 1000;
        const remaining = localEndTime ? Math.max(0, localEndTime - now) : (window.lastGameState.time_remaining || 0);
        const elapsed = intermission_duration - remaining;
        
        if (elapsed < 5) {
            console.log(`[IntermissionPress] Press ignored. Only ${elapsed.toFixed(1)}s elapsed in intermission (requires 5.0s).`);
            return;
        }
    }

    console.log(`[IntermissionPress] Pressed tile at row=${r}, col=${c}, face=${f}, letter=${letter}`);
    
    let tabSwitched = false;
    
    // Toggle/Set the filter
    if (window.intermissionTileFilter && 
        window.intermissionTileFilter.r === r && 
        window.intermissionTileFilter.c === c && 
        window.intermissionTileFilter.f === f) {
        // Clicking again clears the filter!
        window.intermissionTileFilter = null;
        cell.classList.remove('intermission-highlight');
    } else {
        // Remove highlight from any other cell
        document.querySelectorAll('.board-cell.intermission-highlight').forEach(el => {
            el.classList.remove('intermission-highlight');
        });
        
        // Set the new filter
        window.intermissionTileFilter = { r, c, f, letter };
        cell.classList.add('intermission-highlight');
        
        // Automatically switch to 'found' (All Words) tab if not already on it
        if (activeWordsTab !== 'found') {
            const foundTabBtn = document.querySelector('.word-tab[data-tab="found"]');
            if (foundTabBtn) {
                tabSwitched = true;
                foundTabBtn.click();
            }
        }
    }
    
    // Re-display all words (Render instantly BEFORE scrolling, unless tab click already did it)
    if (!tabSwitched && window.lastDisplayAllWordsArgs) {
        displayAllWords(...window.lastDisplayAllWordsArgs);
    }

    // Scroll to the Words panel instantly (Only on mobile devices to prevent viewport jumps on desktop)
    const isMobile = window.innerWidth <= 992;
    if (isMobile && window.intermissionTileFilter) {
        if (typeof window.switchPlayPanel === 'function') {
            window.switchPlayPanel('words');
        }
    }
}

function handleCellMouseDown(e) {
    if (window.isPopupVisible) return;
    if (e.button !== 0) return; // Only left click
    if (Date.now() - lastTouchTime < 1500) return; // Ignore simulated mouse events on touch devices
    initAudioOnUserInteraction();

    const cell = e.target.closest('.board-cell');
    if (!cell) return;

    // INTERMISSION TILE PRESS FILTERS
    if (window.lastGameState && window.lastGameState.state === 'intermission') {
        e.preventDefault();
        const f = cell.dataset.f !== undefined ? parseInt(cell.dataset.f) : null;
        const r = parseInt(cell.dataset.r || cell.dataset.row);
        const c = parseInt(cell.dataset.c || cell.dataset.col);
        const letter = cell.dataset.letter;
        handleIntermissionTilePress(cell, r, c, f, letter);
        return;
    }

    if (window.isSpectatorMode) return;

    if (cell.classList.contains('grayed')) return;

    // Prevent native browser drag/selection behavior from interrupting our swipe
    e.preventDefault();

    const f = cell.dataset.f !== undefined ? parseInt(cell.dataset.f) : null;
    const r = parseInt(cell.dataset.r || cell.dataset.row);
    const c = parseInt(cell.dataset.c || cell.dataset.col);
    const letter = getLetterFromCellAndEvent(cell, e);
    const key = f !== null ? `${f},${r},${c}` : `${r},${c}`;

    // If we already have an active path started (e.g. trackpad relaxed pressure momentarily)
    if (mouseState.selectedPath.length > 0) {
        const lastCell = mouseState.selectedPath[mouseState.selectedPath.length - 1];
        const lastKey = lastCell.face !== null ? `${lastCell.face},${lastCell.row},${lastCell.col}` : `${lastCell.row},${lastCell.col}`;
        
        // If clicking exact same cell, keep state active
        if (lastKey === key) {
            mouseState.isDown = true;
            return;
        }

        // Check if adjacent to lastCell
        const isAdjacent = (f !== null && lastCell.face !== null) ?
            (Math.abs(lastCell.row - r) <= 1 && Math.abs(lastCell.col - c) <= 1) :
            (Math.abs(lastCell.row - r) <= 1 && Math.abs(lastCell.col - c) <= 1);

        if (isAdjacent && !mouseState.visitedCells.has(key)) {
            // Flawless continuation!
            mouseState.isDown = true;
            selectCell(r, c, letter, cell, f);
            return;
        }
    }

    // Otherwise (brand new sequence or clicked non-adjacent starting tile): reset path
    mouseState.isDown = true;
    mouseState.selectedPath = [];
    mouseState.visitedCells = new Set();
    document.querySelectorAll('.board-cell.selected, .board-cell.current').forEach(c => {
        c.classList.remove('selected', 'current');
        applyDensityToCell(c);
    });

    selectCell(r, c, letter, cell, f);
}

function getLetterFromCellAndEvent(cell, e) {
    const letter = cell.dataset.letter;
    // USER REQUEST: Do not split Either/Or letters based on touch position.
    // Return the full string (e.g. "G/O") and let the server resolve the valid word from the path.
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

function handleCellMouseMove(e) {
    if (!mouseState.isDown) return;
    if (window.isSpectatorMode) return;

    const target = document.elementFromPoint(e.clientX, e.clientY);
    const cell = target && target.closest('.board-cell');
    if (!cell || cell.classList.contains('grayed')) return;

    const f = cell.dataset.f !== undefined ? parseInt(cell.dataset.f) : null;
    const r = parseInt(cell.dataset.r || cell.dataset.row);
    const c = parseInt(cell.dataset.c || cell.dataset.col);
    const letter = getLetterFromCellAndEvent(cell, e);
    selectCell(r, c, letter, cell, f);
}
// High-performance touch tracking state to eliminate mobile swiping lag
let lastTouchX = -1;
let lastTouchY = -1;

function handleCellTouchStart(e) {
    if (window.isPopupVisible) return;
    initAudioOnUserInteraction();
    
    // Unconditionally prevent default scroll/gestures immediately on any board touch start
    if (e.cancelable !== false) {
        e.preventDefault();
    }

    const touch = e.changedTouches ? e.changedTouches[0] : e.touches[0];
    let cell = e.target.closest('.board-cell');
    if (!cell) {
        const target = touch && document.elementFromPoint(touch.clientX, touch.clientY);
        cell = target && target.closest('.board-cell');
    }
    if (!cell) return;

    // INTERMISSION TILE PRESS FILTERS
    if (window.lastGameState && window.lastGameState.state === 'intermission') {
        const f = cell.dataset.f !== undefined ? parseInt(cell.dataset.f) : null;
        const r = parseInt(cell.dataset.r || cell.dataset.row);
        const c = parseInt(cell.dataset.c || cell.dataset.col);
        const letter = cell.dataset.letter;
        handleIntermissionTilePress(cell, r, c, f, letter);
        return;
    }

    if (window.isSpectatorMode) return;

    if (mouseState.isDown) return; // Prevent double touch/accidental brushes from erasing path

    lastTouchX = touch.clientX;
    lastTouchY = touch.clientY;
    
    if (cell && !cell.classList.contains('grayed')) {
        mouseState.isDown = true;
        window.activeTouchIdentifier = touch ? touch.identifier : undefined;
        mouseState.selectedPath = [];
        mouseState.visitedCells = new Set();
        document.querySelectorAll('.board-cell.selected, .board-cell.current').forEach(c => {
            c.classList.remove('selected', 'current');
            applyDensityToCell(c);
        });

        const f = cell.dataset.f !== undefined ? parseInt(cell.dataset.f) : null;
        const r = parseInt(cell.dataset.r || cell.dataset.row);
        const c = parseInt(cell.dataset.c || cell.dataset.col);
        const letter = getLetterFromCellAndEvent(cell, e);
        selectCell(r, c, letter, cell, f);
    }
}

function handleCellTouchMove(e) {
    // Prevent mobile scrolling/gestures unconditionally during any board swipe interaction
    if (e.cancelable !== false) {
        e.preventDefault();
    }

    if (!mouseState.isDown) return;
    if (window.isSpectatorMode) return;

    let touch = e.touches[0];
    if (window.activeTouchIdentifier !== undefined && e.touches) {
        const match = Array.from(e.touches).find(t => t.identifier === window.activeTouchIdentifier);
        if (match) touch = match;
    }
    if (!touch) return;
    
    // PERFORMANCE THROTTLE: Skip expensive DOM calculation if the finger hasn't moved a meaningful distance
    if (Math.abs(touch.clientX - lastTouchX) < 10 && Math.abs(touch.clientY - lastTouchY) < 10) {
        return;
    }
    lastTouchX = touch.clientX;
    lastTouchY = touch.clientY;

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

function finishDragSelection(e) {
    if (!mouseState.isDown) return;
    
    // If this was triggered by a touchend/touchcancel but the user's swiping touch has not ended yet, do NOT terminate the swipe!
    if (e && e.changedTouches && window.activeTouchIdentifier !== undefined) {
        // FAIL-SAFE: If no active touch points remain on the screen, always terminate the swipe!
        if (e.touches && e.touches.length === 0) {
            // Proceed to terminate
        } else {
            let hasEnded = false;
            for (let i = 0; i < e.changedTouches.length; i++) {
                if (e.changedTouches[i].identifier === window.activeTouchIdentifier) {
                    hasEnded = true;
                    break;
                }
            }
            if (!hasEnded) {
                return;
            }
        }
    }

    mouseState.isDown = false;

    const path = mouseState.selectedPath;
    
    // Clear selection state immediately so that synchronous validation calls don't see it as active!
    mouseState.selectedPath = [];
    mouseState.visitedCells = new Set();
    window.activeTouchIdentifier = undefined;

    // Clear visual selection and typing highlights from DOM immediately
    document.querySelectorAll('.board-cell.selected, .board-cell.current, .board-cell.typing-highlight').forEach(c => {
        c.classList.remove('selected', 'current', 'typing-highlight');
        applyDensityToCell(c);
    });

    if (path.length >= 1) {
        try {
            const word = path.map(p => {
                const L = p.letter;
                // For Either/Or slash letters, use the first character option as a clean representation.
                // The server's path reconstruction will auto-correct to the valid dictionary option.
                const cleanL = (L && L.includes('/')) ? L.split('/')[0] : L;
                return cleanL === 'Q' ? 'QU' : cleanL;
            }).join('');
            const serverPath = path.map(p => {
                if (p.face !== null && p.face !== undefined) {
                    return [p.face, p.row, p.col];
                }
                return [p.row, p.col];
            });

            // Unconditionally submit word
            submitWord(word, serverPath);
        } catch (err) {
            console.error('[play.js] Error in finishDragSelection submission:', err);
        }
    }

    // UX: Refocus chat if pending
    const inputEl = document.getElementById('word-input');
    if (window.refocusChatPending) {
        window.refocusChatPending = false;
        const chatInput = document.getElementById('chat-input');
        const isMobile = window.innerWidth <= 992;
        if (chatInput && !isMobile) setTimeout(() => chatInput.focus(), 150);
    }

    if (inputEl) {
        inputEl.value = '';
    }
}

// Wire board events via delegation on the static board wrapper
(function initBoardInteraction() {
    const boardEl = document.getElementById('game-board');
    if (!boardEl) return;

    boardEl.addEventListener('dragstart', (e) => e.preventDefault());
    boardEl.addEventListener('mousedown', handleCellMouseDown);

    boardEl.addEventListener('mouseover', handleCellMouseOver);
    document.addEventListener('mousemove', handleCellMouseMove, { passive: true });
    // Unconditionally prevent board touches from starting page/panel horizontal scrolling
    boardEl.addEventListener('touchstart', (e) => {
        if (e.cancelable !== false) {
            e.preventDefault();
        }
        handleCellTouchStart(e);
    }, { passive: false });

    boardEl.addEventListener('touchmove', (e) => {
        if (e.cancelable !== false) {
            e.preventDefault();
        }
        handleCellTouchMove(e);
    }, { passive: false });

    // Release: commit word
    document.addEventListener('mouseup', finishDragSelection);
    document.addEventListener('touchend', finishDragSelection);
    document.addEventListener('touchcancel', finishDragSelection);
})();

// Word Tabs Switching
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('word-tab')) {
        activeWordsTab = e.target.dataset.tab;

        if (activeWordsTab === 'score-sum') {
            fetchDailyScoreSums();
        }

        // Update UI immediately for responsiveness
        document.querySelectorAll('.word-tab').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === activeWordsTab);
        });
        // Update visibility of ALL tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            const tabId = content.id.replace('tab-content-', '');
            content.classList.toggle('active', activeWordsTab === tabId);
        });

        // Toggle Definitions Panel Visibility (Confined to Found/Words tab on all devices)
        const defPanel = document.querySelector('.definitions-panel');
        if (defPanel) {
            defPanel.style.display = activeWordsTab === 'found' ? '' : 'none';
        }

        // Refresh state visualization
        if (window.lastGameState) {
            updateGameState(window.lastGameState);
        }
    }
});

// Clues toggle button click listener
document.addEventListener('click', (e) => {
    if (e.target.id === 'clues-toggle-remaining-btn') {
        window._cluesShowRemaining = !window._cluesShowRemaining;
        if (window.lastGameState) {
            updateGameState(window.lastGameState);
        }
    }
});

// Score sum search and find-me input listeners
document.addEventListener('input', (e) => {
    if (e.target.id === 'score-sum-search') {
        renderDailyScoreSums();
    }
});

document.addEventListener('click', (e) => {
    if (e.target.id === 'score-sum-find-me-btn') {
        const currentUserNameNormalized = window.currentUser ? window.currentUser.toLowerCase().trim() : '';
        if (!currentUserNameNormalized) return;
        
        // Find if user exists in the list
        const searchInput = document.getElementById('score-sum-search');
        if (searchInput && searchInput.value !== '') {
            searchInput.value = '';
            renderDailyScoreSums();
        }
        
        const row = document.querySelector(`.score-sum-row[data-username="${currentUserNameNormalized}"]`);
        if (row) {
            const listEl = document.getElementById('score-sum-list');
            if (listEl) {
                const containerRect = listEl.getBoundingClientRect();
                const rowRect = row.getBoundingClientRect();
                const relativeTop = rowRect.top - containerRect.top;
                const scrollTarget = listEl.scrollTop + relativeTop - (containerRect.height / 2) + (rowRect.height / 2);
                listEl.scrollTo({
                    top: scrollTarget,
                    behavior: 'instant'
                });
            } else {
                row.scrollIntoView({ behavior: 'instant', block: 'center' });
            }
            // Flash effect for visibility
            row.style.transition = 'background-color 0.3s ease';
            const originalBg = row.style.background;
            row.style.background = 'rgba(var(--text-primary-rgb), 0.35)';
            setTimeout(() => {
                row.style.background = originalBg;
            }, 1000);
        } else {
            alert("You are not on the Score Sum list yet! Play in the daily 24h room to get ranked.");
        }
    }
});

// --- TOURNAMENT PLAY LOGIC ---

// Helper to ensure UI is correctly reset for active play (escapes spectator mode)
function resetPlayUI() {
    console.log('[play.js] resetPlayUI() called for active play session');
    wrongGuessesOnBoardCount = 0;
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
        const isMobile = window.innerWidth <= 992;
        if (!isMobile) {
            setTimeout(() => wordInput.focus(), 150);
        }
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
            bonus_word: data.bonus_word,
            all_words: data.all_words || [],
            bonus_cell: data.bonus_cell
        };
        window.lastGameState = tournamentGameState;
        lastRenderedBoardJSON = null; // Force re-render

        // On mobile, if the board data is wider than tall (landscape), transpose it to
        // portrait so the long dimension runs vertically and fills the phone screen naturally.
        isBoardTransposed = false; // reset
        isBoardRotated = false;    // reset
        safelyTransposeState(tournamentGameState);
        data.board = tournamentGameState.board;
        data.bonus_cell = tournamentGameState.bonus_cell;

        // Render Board
        console.log('[Tournament] Rendering tournament board. Format:', (data.params.board_format || 'Normal'));
        renderBoard(data.board, false, is3D);
        updateParameters(tournamentGameState);
        resetPlayUI();

        // Timer Setup: The tournament game has its OWN local timer starting from the moment they click play
        localEndTime = (Date.now() / 1000) + data.params.time_limit;

        if (timerInterval) clearInterval(timerInterval);
        
        // Update immediately to prevent initial visual delay of 1s (showing stale values)
        const initialDiff = Math.max(0, Math.ceil(localEndTime - (Date.now() / 1000)));
        updateSpecialMatchTimer(initialDiff);

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

async function handleTournamentWord(word, path = null) {
    if (!word) return;

    // Check if word is on board
    const board = window.lastGameState ? window.lastGameState.board : null;
    if (!board) {
        console.error('[Tournament] No board found in lastGameState');
        return;
    }

    const is3D = board.length === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]);
    const fmt = (window.tournamentParams && typeof window.tournamentParams.board_format === 'string') ? window.tournamentParams.board_format : 'Normal';
    const isEO = fmt && fmt.toLowerCase().includes('either');

    let resolvedWord = word.toUpperCase();
    let resolvedPath = path;

    if (resolvedPath && isEO) {
        // Reconstruct all possible words from the path
        let possibleWords = [''];
        let validPath = true;

        for (const node of resolvedPath) {
            let cellVal = '';
            let f = undefined, r = undefined, c = undefined;
            if (Array.isArray(node)) {
                if (node.length === 3) {
                    [f, r, c] = node;
                } else if (node.length === 2) {
                    [r, c] = node;
                }
            } else if (node && typeof node === 'object') {
                r = node.r;
                c = node.c;
                f = node.f;
            }

            if (f !== undefined && f !== null) {
                if (f >= 0 && f < board.length && r >= 0 && r < board[f].length && c >= 0 && c < board[f][r].length) {
                    cellVal = String(board[f][r][c]);
                } else {
                    validPath = false;
                    break;
                }
            } else if (r !== undefined && c !== undefined) {
                if (r >= 0 && r < board.length && c >= 0 && c < board[0].length) {
                    cellVal = String(board[r][c]);
                } else {
                    validPath = false;
                    break;
                }
            } else {
                validPath = false;
                break;
            }

            if (cellVal.includes('/')) {
                const options = cellVal.split('/');
                let newWords = [];
                for (const prefix of possibleWords) {
                    for (const opt of options) {
                        newWords.push(prefix + opt);
                    }
                }
                possibleWords = newWords;
            } else {
                for (let i = 0; i < possibleWords.length; i++) {
                    possibleWords[i] += cellVal;
                }
            }
        }

        if (validPath) {
            let validOptions = [];
            let wordList = [];
            if (window.lastGameState && window.lastGameState.all_words) {
                const allWordsState = window.lastGameState.all_words;
                if (Array.isArray(allWordsState)) {
                    wordList = allWordsState;
                } else if (allWordsState && typeof allWordsState === 'object') {
                    wordList = Object.keys(allWordsState);
                }
            }

            // Check dictionary validity for each possible word locally
            for (const w of possibleWords) {
                const isVal = wordList.some(item => {
                    const wText = (typeof item === 'string' ? item : (item.word || '')) || '';
                    return wText.toUpperCase() === w.toUpperCase();
                });
                if (isVal) {
                    validOptions.push(w);
                }
            }

            // Find which option to select
            let submittedClean = resolvedWord.replace(/\//g, '');
            if (validOptions.includes(submittedClean)) {
                resolvedWord = submittedClean;
            } else if (validOptions.length >= 1) {
                resolvedWord = validOptions[0];
            } else if (possibleWords.includes(submittedClean)) {
                resolvedWord = submittedClean;
            } else if (possibleWords.length > 0) {
                resolvedWord = possibleWords[0];
            }
        }
    } else if (!resolvedPath) {
        // If not Either/Or or no path, validate the path using findWordPath
        const p = is3D ? findWordPathOnCube(resolvedWord, board) : findWordPathOnBoard(resolvedWord, board);
        if (!p) {
            showValidationFeedback(`${resolvedWord} is invalid.`, false);
            recordGuessResult(false, false);
            return;
        }
        resolvedPath = p;
    }

    if (tournamentWords.find(w => w.word === resolvedWord)) {
        showValidationFeedback('Already found!', false, false, resolvedPath);
        recordGuessResult(false, resolvedPath && resolvedPath.length > 0, true);
        return;
    }

    word = resolvedWord;
    path = resolvedPath;

    // Check dictionary locally using all_words list
    let is_valid_dict = false;
    if (window.lastGameState && window.lastGameState.all_words) {
        const allWordsState = window.lastGameState.all_words;
        let wordList = [];
        if (Array.isArray(allWordsState)) {
            wordList = allWordsState;
        } else if (allWordsState && typeof allWordsState === 'object') {
            wordList = Object.keys(allWordsState);
        }
        is_valid_dict = wordList.some(w => {
            const wText = (typeof w === 'string' ? w : (w.word || '')) || '';
            return wText.toUpperCase() === word.toUpperCase();
        });
    }

    // Check length and dictionary validity
    const minLen = window.tournamentParams ? window.tournamentParams.min_word_length : 3;
    const effectiveLen = word.replace(/Q(?!U)/g, 'QU').length;
    if (effectiveLen < minLen) {
        showValidationFeedback(`${word.toUpperCase()} IS TOO SHORT (MIN: ${minLen}L)`, false, false, path);
        recordGuessResult(false, path && path.length > 0, true);
        return;
    }

    if (!is_valid_dict) {
        showValidationFeedback(`${word.toUpperCase()} INVALID`, false, false, path);
        recordGuessResult(false, path && path.length > 0);
        return;
    }

    // Use the same scoring rules as public rooms (3 and 4 letter words = 1pt, 5L = 2pts, 6L = 3pts, 7L = 5pts, and 8L+ = 11pts)
    let pts = calculateWordScoreLocally(word, path);

    // Bonus Word
    let isBonus = false;
    if (window.lastGameState && window.lastGameState.bonus_word && typeof window.lastGameState.bonus_word === 'string' && word === window.lastGameState.bonus_word.toUpperCase()) {
        isBonus = true;
    }

    tournamentWords.push({
        word: word,
        points: pts,
        timestamp: Date.now() / 1000,
        is_bonus: isBonus
    });
    tournamentScore += pts;

    // Check if the path uses Either/Or tile
    let usesEitherOrTile = false;
    if (path && board) {
        for (const node of path) {
            let cellVal = '';
            let f = undefined, r = undefined, c = undefined;
            if (Array.isArray(node)) {
                if (node.length === 3) {
                    [f, r, c] = node;
                } else if (node.length === 2) {
                    [r, c] = node;
                }
            } else if (node && typeof node === 'object') {
                r = node.r;
                c = node.c;
                f = node.f;
            }

            if (f !== undefined && f !== null) {
                if (f >= 0 && f < board.length && r >= 0 && r < board[f].length && c >= 0 && c < board[f][r].length) {
                    cellVal = String(board[f][r][c]);
                }
            } else if (r !== undefined && c !== undefined) {
                if (r >= 0 && r < board.length && c >= 0 && c < board[0].length) {
                    cellVal = String(board[r][c]);
                }
            }

            if (cellVal && cellVal.includes('/')) {
                usesEitherOrTile = true;
                break;
            }
        }
    }

    // Show success feedback
    const showBonusMsg = isBonus && !usesEitherOrTile;
    showValidationFeedback(showBonusMsg ? `BONUS WORD! (${pts} PTS)` : `${word.toUpperCase()} VALID (${pts} PTS)`, true, isBonus, path);
    recordGuessResult(true, true);

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

    // Save Draft to Server
    saveTournamentDraft();
}

function saveTournamentDraft() {
    const activeData = JSON.parse(localStorage.getItem('tournament_play_active'));
    if (!activeData) return;

    fetch('/api/tournament/save-draft', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            tournament_id: activeData.tid,
            round_number: activeData.round,
            words: tournamentWords,
            score: Math.floor(tournamentScore),
            round_start_time: tournamentStartTime
        })
    }).catch(err => console.warn('[Tournament] Failed to save draft:', err));
}

async function finishTournamentTurn(targetPage = 'tournaments') {
    console.log('[Tournament] Finish Turn. Final Score:', tournamentScore);
    const activeData = JSON.parse(localStorage.getItem('tournament_play_active'));
    if (!activeData) {
        exitTournamentPlay(targetPage);
        return;
    }

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

    exitTournamentPlay(targetPage);
}

function exitTournamentPlay(targetPage = 'tournaments') {
    localStorage.removeItem('tournament_play_active');
    isTournamentPlay = false;
    window.isTournamentPlay = false;
    isBoardTransposed = false; // RESET: clear portrait transposition set for tournament mobile
    isBoardRotated = false;    // RESET: ensure board isn't flipped from previous game
    clearGameUIAndCache();
    if (window.navigateToPage) {
        window.navigateToPage(targetPage);
    } else {
        window.location.href = '#page-' + targetPage;
    }
}

window.saveTournamentDraft = saveTournamentDraft;
window.finishTournamentTurn = finishTournamentTurn;
window.exitTournamentPlay = exitTournamentPlay;
window.finishPrivateMatchTurn = finishPrivateMatchTurn;
window.exitPrivateMatchPlay = exitPrivateMatchPlay;

// --- PRIVATE MATCH PLAY LOGIC ---
window.initPrivateMatchPlay = function () {
    console.log('[play.js] initPrivateMatchPlay() START');
    isPrivateMatchPlay = true;
    window.isPrivateMatchPlay = true;
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
        current_difficulty: activeMatch.parameters.difficulty || 'Medium',
        current_dictionary: activeMatch.parameters.dictionary || 'NWL',
        current_word_count_range: activeMatch.parameters.word_count_range || '100-200',
        spinner_params: activeMatch.parameters,
        bonus_word: activeMatch.bonus_word,
        all_words: activeMatch.all_words || [],
        bonus_cell: activeMatch.bonus_cell
    };
    safelyTransposeState(mockState);
    activeMatch.board = mockState.board;
    activeMatch.bonus_cell = mockState.bonus_cell;
    
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

    // Update immediately to prevent initial visual delay of 1s
    const updateImmediate = () => {
        const now = Date.now() / 1000;
        const remaining = Math.max(0, Math.floor(endTime - now));
        const mins = Math.floor(remaining / 60);
        const secs = remaining % 60;
        if (timerEl) {
            timerEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
            setTimerWaitingState(false);
        }
        return remaining;
    };

    const initialRemaining = updateImmediate();
    if (initialRemaining <= 0) {
        console.log('[play.js] Private match timer already expired! Triggering auto-finish immediately.');
        setTimeout(() => {
            finishPrivateMatchTurn();
        }, 500);
        return;
    }

    timerInterval = setInterval(() => {
        const remaining = updateImmediate();

        if (remaining <= 0) {
            console.log('[play.js] Private match timer reached 0! Triggering auto-finish.');
            clearInterval(timerInterval);
            timerInterval = null;

            // Tiny delay to ensure user actually sees the 0:00
            setTimeout(() => {
                finishPrivateMatchTurn();
            }, 500);
        }
    }, 1000);
}

async function handlePrivateMatchWord(word, path = null) {
    if (!word) return;

    // Check if word is on board
    const board = window.lastGameState ? window.lastGameState.board : null;
    if (!board) {
        console.error('[Private Match] No board found in lastGameState');
        return;
    }

    const is3D = board.length === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]);
    const fmt = privateMatchParams ? privateMatchParams.board_format : 'Normal';
    const isEO = fmt.toLowerCase().includes('either');

    let resolvedWord = word.toUpperCase();
    let resolvedPath = path;

    let usesEitherOrTile = false;
    if (resolvedPath && isEO) {
        // Reconstruct all possible words from the path
        let possibleWords = [''];
        let validPath = true;

        for (const node of resolvedPath) {
            let cellVal = '';
            let f = undefined, r = undefined, c = undefined;
            if (Array.isArray(node)) {
                if (node.length === 3) {
                    [f, r, c] = node;
                } else if (node.length === 2) {
                    [r, c] = node;
                }
            } else if (node && typeof node === 'object') {
                r = node.r;
                c = node.c;
                f = node.f;
            }

            if (f !== undefined && f !== null) {
                if (f >= 0 && f < board.length && r >= 0 && r < board[f].length && c >= 0 && c < board[f][r].length) {
                    cellVal = String(board[f][r][c]);
                } else {
                    validPath = false;
                    break;
                }
            } else if (r !== undefined && c !== undefined) {
                if (r >= 0 && r < board.length && c >= 0 && c < board[0].length) {
                    cellVal = String(board[r][c]);
                } else {
                    validPath = false;
                    break;
                }
            } else {
                validPath = false;
                break;
            }

            if (cellVal.includes('/')) {
                usesEitherOrTile = true;
                const options = cellVal.split('/');
                let newWords = [];
                for (const prefix of possibleWords) {
                    for (const opt of options) {
                        newWords.push(prefix + opt);
                    }
                }
                possibleWords = newWords;
            } else {
                for (let i = 0; i < possibleWords.length; i++) {
                    possibleWords[i] += cellVal;
                }
            }
        }

        if (validPath) {
            const dict = privateMatchParams ? privateMatchParams.dictionary : 'NWL';
            let validOptions = [];

            // Check dictionary validity for each possible word locally
            let pmWordList = [];
            if (window.lastGameState && window.lastGameState.all_words) {
                const allWordsState = window.lastGameState.all_words;
                if (Array.isArray(allWordsState)) {
                    pmWordList = allWordsState;
                } else if (allWordsState && typeof allWordsState === 'object') {
                    pmWordList = Object.keys(allWordsState);
                }
            }
            for (const w of possibleWords) {
                const isVal = pmWordList.some(item => {
                    const wText = (typeof item === 'string' ? item : (item.word || '')) || '';
                    return wText.toUpperCase() === w.toUpperCase();
                });
                if (isVal) {
                    validOptions.push(w);
                }
            }

            // Find which option to select
            // Check if the submitted word matches one of the valid options (clean version)
            let submittedClean = resolvedWord.replace(/\//g, '');
            if (validOptions.includes(submittedClean)) {
                resolvedWord = submittedClean;
            } else if (validOptions.length >= 1) {
                resolvedWord = validOptions[0];
            } else if (possibleWords.includes(submittedClean)) {
                resolvedWord = submittedClean;
            } else if (possibleWords.length > 0) {
                resolvedWord = possibleWords[0];
            }
        }
    } else if (!resolvedPath) {
        // If not Either/Or or no path, validate the path using findWordPath
        const p = is3D ? findWordPathOnCube(resolvedWord, board) : findWordPathOnBoard(resolvedWord, board);
        if (!p) {
            showValidationFeedback('Not on board!', false);
            recordGuessResult(false, false);
            return;
        }
        resolvedPath = p;
    }

    if (privateMatchWords.find(w => w.word === resolvedWord)) {
        showValidationFeedback('Already found!', false, false, resolvedPath);
        recordGuessResult(false, resolvedPath && resolvedPath.length > 0, true);
        return;
    }

    // Set word to the resolved clean word for dictionary check and subsequent scoring
    word = resolvedWord;

    // 1. Initial Checks (Dictionary & Min Length) - Checked locally
    let isDictionaryValid = false;
    if (window.lastGameState && window.lastGameState.all_words) {
        const allWordsState = window.lastGameState.all_words;
        let pmWordList = [];
        if (Array.isArray(allWordsState)) {
            pmWordList = allWordsState;
        } else if (allWordsState && typeof allWordsState === 'object') {
            pmWordList = Object.keys(allWordsState);
        }
        isDictionaryValid = pmWordList.some(item => {
            const wText = (typeof item === 'string' ? item : (item.word || '')) || '';
            return wText.toUpperCase() === word.toUpperCase();
        });
    }

    const minLen = privateMatchParams ? privateMatchParams.min_word_length : 3;
    const effectiveLen = word.replace(/Q(?!U)/g, 'QU').length;
    if (effectiveLen < minLen) {
        showValidationFeedback(`${word.toUpperCase()} IS TOO SHORT (MIN: ${minLen}L)`, false, false, resolvedPath);
        recordGuessResult(false, resolvedPath && resolvedPath.length > 0, true);
        return;
    }

    // 2. Format & Bonus Info
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
                'U': 4, 'V': 5, 'W': 5, 'X': 8, 'Y': 5, 'Z': 8
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
        let isBonus = false;
        if (activeMatch && activeMatch.bonus_word && activeMatch.bonus_word.toUpperCase() === word) {
            pts += word.length;
            isBonus = true;
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

                    let targetR, targetC;
                    if (Array.isArray(bonusCell)) {
                        if (is3D && bonusCell.length === 3) return coord.f === bonusCell[0] && coord.r === bonusCell[1] && coord.c === bonusCell[2];
                        targetR = bonusCell[0];
                        targetC = bonusCell[1];
                    } else if (typeof bonusCell === 'object') {
                        if (is3D && bonusCell.f !== undefined) return coord.f === bonusCell.f && coord.r === bonusCell.r && coord.c === bonusCell.c;
                        targetR = bonusCell.r;
                        targetC = bonusCell.c;
                    }
                    return coord.r === Number(targetR) && coord.c === Number(targetC);
                });
                if (hitsBonus) {
                    pts += 3;
                    console.log('[Private Match] Awarded +3 Bonus for Special Tile');
                }
            }
        }

        if (isBonus) {
            if (usesEitherOrTile) {
                showValidationFeedback(`${word.toUpperCase()} VALID (${pts} PTS)`, true, true, resolvedPath);
            } else {
                showValidationFeedback(`BONUS WORD FOUND! (${pts} PTS)`, true, true, resolvedPath);
            }
        } else {
            showValidationFeedback(`${word.toUpperCase()} VALID (${pts} PTS)`, true, false, resolvedPath);
        }
    } else {
        // Invalid Word Check for Penalty
        if (fmtLower.includes('penalty')) {
            const wordPath = is3D ? findWordPathOnCube(word, board) : findWordPathOnBoard(word, board);
            if (wordPath) {
                pts = -3;
                isPenalty = true;
                showValidationFeedback('INVALID (PENALTY -3)', false, false, wordPath);
            } else {
                showValidationFeedback('Not in dictionary!', false, false, resolvedPath);
                recordGuessResult(false, resolvedPath && resolvedPath.length > 0);
                return;
            }
        } else {
            showValidationFeedback('Not in dictionary!', false, false, resolvedPath);
            recordGuessResult(false, resolvedPath && resolvedPath.length > 0);
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

    if (isDictionaryValid) {
        recordGuessResult(true, true);
    } else {
        recordGuessResult(false, resolvedPath && resolvedPath.length > 0);
    }
}

async function finishPrivateMatchTurn(targetPage = 'lobby') {
    console.log('[play.js] finishPrivateMatchTurn() called');
    const activeMatch = JSON.parse(localStorage.getItem('private_match_active'));
    if (!activeMatch) {
        console.warn('[play.js] finishPrivateMatchTurn: No activeMatch found');
        exitPrivateMatchPlay(targetPage);
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

    exitPrivateMatchPlay(targetPage);
}

function exitPrivateMatchPlay(targetPage = 'lobby') {
    isPrivateMatchPlay = false;
    window.isPrivateMatchPlay = false;
    localStorage.removeItem('private_match_active');
    clearGameUIAndCache();

    // Clean up timers
    if (window.privateMatchInterval) clearInterval(window.privateMatchInterval);

    if (window.navigateToPage) {
        window.navigateToPage(targetPage);
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
    const board = generateProbeBoard(params);
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
    const format = params.board_format || params.format || 'Normal';
    const bonusWord = params.bonus_word || '';
    const parts = dims.split('x');
    if (parts.length !== 2) return null;
    const rows = parseInt(parts[0]);
    const cols = parseInt(parts[1]);

    let letters = [];
    if (format === 'Equality Freq') {
         const freq = {
             'A': 700, 'B': 413, 'C': 413, 'D': 413, 'E': 700, 'F': 413, 'G': 413, 'H': 413, 'I': 700, 'J': 25,
             'K': 413, 'L': 413, 'M': 413, 'N': 413, 'O': 700, 'P': 413, 'Q': 25, 'R': 413, 'S': 413, 'T': 413,
             'U': 700, 'V': 413, 'W': 413, 'X': 25, 'Y': 413, 'Z': 25
         };
         let pool = [];
         for (let char in freq) {
             for (let i = 0; i < freq[char]; i++) pool.push(char);
         }
         for (let i = 0; i < rows * cols; i++) {
             letters.push(pool[Math.floor(Math.random() * pool.length)]);
         }
    } else if (rows === 4 && cols === 4 && format !== 'Checkerboard') {
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
        
        // Calculate base word points
        let defaultWordVal = 0;
        const len = wordUpper.length;
        if (len === 3 || len === 4) defaultWordVal = 1;
        else if (len === 5) defaultWordVal = 2;
        else if (len === 6) defaultWordVal = 3;
        else if (len === 7) defaultWordVal = 5;
        else if (len === 8) defaultWordVal = 8;
        else if (len === 9) defaultWordVal = 11;
        else if (len >= 10) defaultWordVal = 15;
        
        if (window.lastGameState && window.lastGameState.bonus_word && wordUpper === String(window.lastGameState.bonus_word).toUpperCase()) {
            defaultWordVal += len;
        }

        title.textContent = `Who found "${wordUpper}"? (${defaultWordVal} pts)`;

        // Use current game state to find players
        if (!window.lastGameState || !window.lastGameState.players) {
            console.warn('[showFinderModal] No lastGameState available');
            return;
        }

        // Sort all players to establish round placement
        const sortedAllPlayers = [...window.lastGameState.players].sort((a, b) => (b.score - a.score) || (b.rating - a.rating));
        const playerRankMap = new Map();
        sortedAllPlayers.forEach((p, idx) => {
            playerRankMap.set(p.username, idx + 1);
        });

        const finders = window.lastGameState.players.filter(p =>
            p.submitted_words && p.submitted_words.some(sw =>
                (typeof sw === 'object' ? sw.word : sw).toUpperCase() === wordUpper
            )
        ).sort((a, b) => {
            const rankA = playerRankMap.get(a.username) || 999;
            const rankB = playerRankMap.get(b.username) || 999;
            return rankA - rankB;
        });

        const findersCount = finders.length;

        // Fetch tally
        body.innerHTML = '<p class="placeholder" style="padding: 20px; text-align: center;">Loading tally...</p>';
        
        fetch(`/api/word_tally/${wordUpper}`)
            .then(res => res.json())
            .then(data => {
                const totalTally = data.count || 0;
                const totalCombined = totalTally;
                
                let html = `
                    <div style="padding: 14px 16px; font-size: 0.95rem; color: var(--text-secondary); border-bottom: 1px solid rgba(255,255,255,0.15); margin-bottom: 15px; text-align: center; width: 100%; box-sizing: border-box;">
                        <strong>${wordUpper}</strong> (${defaultWordVal} pts) has been found <strong>${totalCombined}</strong> times total since Morpheme began.
                    </div>
                `;
                
                if (findersCount === 0) {
                    html += '<p class="placeholder" style="padding: 20px; text-align: center;">No one found this word on this round.</p>';
                } else {
                    html += finders.map(p => {
                        const rating = p.rating || 0;
                        const score = p.score || 0;
                        const rank = playerRankMap.get(p.username) || 999;
                        const rColor = window.getRatingColor ? window.getRatingColor(rating) : '#fff';
                        
                        const swObj = p.submitted_words ? p.submitted_words.find(sw =>
                            (typeof sw === 'object' ? sw.word : sw).toUpperCase() === wordUpper
                        ) : null;
                        const wordPts = (swObj && typeof swObj === 'object' && typeof swObj.points === 'number') ? swObj.points : defaultWordVal;
                        
                        // Placement Badge Styling
                        let badgeBg = 'rgba(255,255,255,0.05)';
                        let badgeColor = 'var(--text-secondary)';
                        let rankText = `#${rank}`;
                        if (rank === 1) {
                            badgeBg = 'rgba(255, 215, 0, 0.15)';
                            badgeColor = '#FFD700';
                            rankText = '1st';
                        } else if (rank === 2) {
                            badgeBg = 'rgba(192, 192, 192, 0.15)';
                            badgeColor = '#E0E0E0';
                            rankText = '2nd';
                        } else if (rank === 3) {
                            badgeBg = 'rgba(205, 127, 50, 0.15)';
                            badgeColor = '#CD7F32';
                            rankText = '3rd';
                        } else {
                            const j = rank % 10, k = rank % 100;
                            let suffix = "th";
                            if (j === 1 && k !== 11) suffix = "st";
                            else if (j === 2 && k !== 12) suffix = "nd";
                            else if (j === 3 && k !== 13) suffix = "rd";
                            rankText = rank + suffix;
                        }
                        
                        return `
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); background: rgba(255,255,255,0.02); margin-bottom: 4px; border-radius: 6px;">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <div style="display: flex; align-items: center; justify-content: center; width: 36px; height: 22px; background: ${badgeBg}; color: ${badgeColor}; border-radius: 4px; font-size: 0.75rem; font-weight: 800;">
                                        ${rankText}
                                    </div>
                                    <div style="width: 14px; height: 14px; background: ${rColor}; border-radius: 3px; box-shadow: 0 0 10px ${rColor}22;"></div>
                                    <span style="font-weight: 700; font-size: 0.95rem;">${p.username}</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <span style="font-size: 0.85rem; font-weight: 700; color: var(--accent-color, #a855f7);">${wordPts} pts</span>
                                    <span style="opacity: 0.4; font-size: 0.8rem; font-weight: 600;">⭐ ${rating}</span>
                                </div>
                            </div>
                        `;
                    }).join('');
                }
                
                body.innerHTML = html;
            })
            .catch(err => {
                console.error('[showFinderModal] Error fetching tally:', err);
                body.innerHTML = '<p class="placeholder" style="padding: 20px; text-align: center;">Error loading tally.</p>';
            });

        modal.classList.remove('hidden');
        modal.style.display = 'flex';
    }
};

/* ==========================================================================
   Tactile Yellow Lined Notepad Modal for Round-End Word View
   ========================================================================== */

window.showNotepadPopup = function(username) {
    console.log('[Notepad] Requesting notepad for:', username);

    const state = window.lastGameState;
    if (!state || !state.players) return;

    const player = state.players.find(p => p.username === username);
    if (!player) return;

    // Get all valid submitted words and sort by length descending (largest words first)
    const rawWords = player.submitted_words || [];
    const words = [...rawWords].sort((a, b) => {
        const wordA = (typeof a === 'object' ? a.word : a) || '';
        const wordB = (typeof b === 'object' ? b.word : b) || '';
        return (wordB.length - wordA.length) || wordA.localeCompare(wordB);
    });

    // Create notepad modal overlay if it doesn't already exist
    let modal = document.getElementById('notepad-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'notepad-modal';
        modal.className = 'notepad-overlay hidden';
        document.body.appendChild(modal);
    }

    // Render beautiful lined legal pad structure
    modal.innerHTML = `
        <div class="notepad-card">
            <div class="notepad-binder-rings">
                <div class="binder-ring"></div>
                <div class="binder-ring"></div>
                <div class="binder-ring"></div>
                <div class="binder-ring"></div>
                <div class="binder-ring"></div>
            </div>
            <div class="notepad-header">
                <span class="notepad-title">${username}'s Notepad</span>
                <button class="notepad-close" title="Close Notepad">&times;</button>
            </div>
            <div class="notepad-body">
                <div class="notepad-margin-line"></div>
                <div class="notepad-words-container" id="notepad-words-scroller">
                    ${words.length === 0 ? `
                        <div class="notepad-word-line" style="font-style: italic; color: #93a1a1; font-size: 0.95rem; font-weight: normal;">
                            <span>No words found</span>
                        </div>
                    ` : words.map((w, idx) => {
                        const wordText = (typeof w === 'object' ? w.word : w) || '';
                        const pts = (typeof w === 'object' ? w.points : 0) || 0;
                        return `
                            <div class="notepad-word-line">
                                <span class="notepad-word"><span class="notepad-word-num">${idx + 1}.</span>${wordText}</span>
                                <span class="notepad-word-points">${pts} pts</span>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
            <div class="notepad-controls">
                <button class="notepad-scroll-btn" id="notepad-scroll-up" title="Scroll Up">&#9650;</button>
                <button class="notepad-scroll-btn" id="notepad-scroll-down" title="Scroll Down">&#9660;</button>
            </div>
        </div>
    `;

    // Click outside or close button to close modal
    const closeBtn = modal.querySelector('.notepad-close');
    closeBtn.addEventListener('click', window.closeNotepadPopup);
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            window.closeNotepadPopup();
        }
    });

    // Touch/click scrolling mechanics
    const scroller = modal.querySelector('#notepad-words-scroller');
    const scrollUp = modal.querySelector('#notepad-scroll-up');
    const scrollDown = modal.querySelector('#notepad-scroll-down');

    const updateScrollButtonsState = () => {
        if (!scroller) return;
        const isAtTop = scroller.scrollTop <= 1;
        const isAtBottom = (scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight) <= 1.5;

        if (isAtTop) {
            scrollUp.classList.add('disabled');
        } else {
            scrollUp.classList.remove('disabled');
        }

        if (isAtBottom) {
            scrollDown.classList.add('disabled');
        } else {
            scrollDown.classList.remove('disabled');
        }
    };

    if (words.length <= 15) {
        scrollUp.classList.add('disabled');
        scrollDown.classList.add('disabled');
    } else {
        scrollUp.addEventListener('click', () => {
            scroller.scrollBy({ top: -84, behavior: 'smooth' }); // Scroll by exactly 3 lines (28px * 3)
        });

        scrollDown.addEventListener('click', () => {
            scroller.scrollBy({ top: 84, behavior: 'smooth' });
        });

        scroller.addEventListener('scroll', updateScrollButtonsState);
        // Delay slightly for render layouts
        setTimeout(updateScrollButtonsState, 80);
    }

    // Reveal Notepad with smooth transition
    modal.classList.remove('hidden');
};

window.closeNotepadPopup = function() {
    const modal = document.getElementById('notepad-modal');
    if (modal) {
        modal.classList.add('hidden');
    }
};

