// Play Page JavaScript
let pollInterval = null;
let timerInterval = null;  // Separate interval for smooth timer updates
let previousState = null;
let localEndTime = null;  // End time in local clock terms
let lastServerUpdate = Date.now();  // Track last server response for freeze detection

function getCurrentRoomId() {
    return window.currentRoomId || null;
}

// Expose for lobby.js to call
window.startGamePolling = function () {
    startPolling();
};

function startPolling() {
    console.log('[play.js] startPolling() called');
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    console.log('[play.js] Setting up interval to call updateGameState');
    pollInterval = setInterval(updateGameState, 1000);
    console.log('[play.js] Calling updateGameState() immediately');
    updateGameState(); // Initial call
}

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

async function updateGameState() {
    const roomId = getCurrentRoomId();
    console.log('[play.js] updateGameState() called, roomId:', roomId);
    if (!roomId) {
        console.warn('[play.js] No roomId found, exiting updateGameState');
        return;
    }

    try {
        console.log(`[play.js] Fetchingstate from /api/room/${roomId}/state`);
        const response = await fetch(`/api/room/${roomId}/state`);
        console.log('[play.js] Fetch response received, status:', response.status);
        const state = await response.json();
        console.log('[play.js] State data:', state);

        if (state.error) {
            console.error('Room error:', state.error);
            stopPolling();
            return;
        }

        // Update parameters
        updateParameters(state);

        // Sync local timer with server
        syncTimerWithServer(state);

        // Track last successful server update
        lastServerUpdate = Date.now();

        // Render board
        renderBoard(state.board, state.state === 'intermission');

        // Update players
        renderPlayers(state.players);

        // Check for state transitions and show/hide spinner
        if (previousState !== state.state) {
            if (state.state === 'intermission' && previousState === 'active') {
                // Transitioning to intermission - show spinner overlay
                showSpinnerOverlay(state.spinner_params);
            } else if (state.state === 'active' && previousState === 'intermission') {
                // Transitioning to active (new round starting) - hide spinner and clear word list
                hideSpinnerOverlay();

                // Clear the submitted words list for new round
                const wordsList = document.getElementById('submitted-words-list');
                wordsList.innerHTML = '<p class="placeholder">No words yet</p>';

                // Auto-focus input field so player can start typing immediately
                const inputField = document.getElementById('word-input');
                if (inputField) {
                    inputField.focus();
                }
            }
            previousState = state.state;
        }

        // Update words panel based on state
        const wordsPanel = document.querySelector('.words-panel h3');
        const wordsStats = document.getElementById('words-stats');
        const wordsList = document.getElementById('submitted-words-list');

        if (state.state === 'intermission') {
            // During intermission, show ALL words with statistics

            // Find current player's submitted words for blue highlighting
            let playerWords = [];
            // Collect ALL submitted words from ALL players
            let allFoundWords = [];

            // currentUser is defined in app.js as a module variable
            try {
                // Try to get from window first, fallback to importing from app.js scope
                const username = window.currentUser || (typeof currentUser !== 'undefined' ? currentUser : null);
                if (username) {
                    const currentPlayer = state.players.find(p => p.username === username);
                    if (currentPlayer && currentPlayer.submitted_words) {
                        playerWords = currentPlayer.submitted_words;
                    }
                }

                // Collect all words from all players
                state.players.forEach(p => {
                    if (p.submitted_words && Array.isArray(p.submitted_words)) {
                        allFoundWords.push(...p.submitted_words);
                    }
                });
            } catch (e) {
                console.warn('Could not find currentUser:', e);
            }

            // Calculate statistics
            const totalWords = state.all_words ? state.all_words.length : 0;
            // Get unique found words (remove duplicates)
            const uniqueFoundWords = [...new Set(allFoundWords.map(w => w.toUpperCase()))];
            const foundCount = uniqueFoundWords.length;
            const percentage = totalWords > 0 ? Math.round((foundCount / totalWords) * 100) : 0;

            // Update heading and statistics
            wordsPanel.textContent = 'All Words';
            wordsStats.textContent = `${foundCount}/${totalWords} - ${percentage}%`;

            displayAllWords(state.all_words, state.bonus_word, playerWords, allFoundWords);
        } else {
            // During active play, show submitted words with statistics

            // Get current player's word count
            let playerWordCount = 0;
            try {
                const username = window.currentUser || (typeof currentUser !== 'undefined' ? currentUser : null);
                if (username) {
                    const currentPlayer = state.players.find(p => p.username === username);
                    if (currentPlayer && currentPlayer.submitted_words) {
                        playerWordCount = currentPlayer.submitted_words.length;
                    }
                }
            } catch (e) {
                console.warn('Could not find currentUser:', e);
            }

            // Calculate statistics
            const totalWords = state.all_words ? state.all_words.length : 0;
            const percentage = totalWords > 0 ? Math.round((playerWordCount / totalWords) * 100) : 0;

            // Update heading and statistics
            wordsPanel.textContent = 'Your Words';
            wordsStats.textContent = `${playerWordCount}/${totalWords} - ${percentage}%`;
            // Keep existing submitted words
        }

        // Enable/disable input
        const isActive = state.state === 'active';
        document.getElementById('word-input').disabled = !isActive;
        document.getElementById('submit-word-btn').disabled = !isActive;

        // Auto-focus input when round becomes active (handles both initial start and post-intermission)
        if (isActive && !previousState) {
            // Initial round start - auto-focus
            const inputField = document.getElementById('word-input');
            if (inputField) {
                inputField.focus();
            }
        }

    } catch (error) {
        console.error('Error updating game state:', error);
    }
}

function updateParameters(state) {
    document.getElementById('param-board').textContent = state.board_dimensions;
    document.getElementById('param-time').textContent = state.time_limit + 's';

    const sp = state.spinner_params;
    if (sp) {
        document.getElementById('param-bonus').textContent = sp.bonus_word_length + ' letters';
        document.getElementById('param-min').textContent = sp.min_word_length;
        document.getElementById('param-dict').textContent = sp.dictionary;

        const wr = sp.word_count_range;
        document.getElementById('param-words').textContent = `${wr[0]}-${wr[1]}`;
        document.getElementById('param-format').textContent = sp.board_format;
    }
}

function updateTimer(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    const display = `${mins}:${secs.toString().padStart(2, '0')}`;
    document.getElementById('timer-value').textContent = display;
}

function syncTimerWithServer(state) {
    // Calculate server time offset
    const clientTime = Date.now() / 1000;  // Convert to seconds
    const serverTime = state.server_time;
    const serverTimeOffset = serverTime - clientTime;

    // Determine which end time to use
    let endTime = 0;
    if (state.state === 'active' && state.round_end_time) {
        endTime = state.round_end_time;
    } else if (state.state === 'intermission' && state.intermission_end_time) {
        endTime = state.intermission_end_time;
    }

    // Convert server end time to local clock terms
    localEndTime = endTime - serverTimeOffset;

    // Start local timer interval if not already running
    if (!timerInterval && localEndTime > 0) {
        timerInterval = setInterval(updateLocalTimer, 100);  // Update every 100ms for smoothness
    } else if (localEndTime <= 0 && timerInterval) {
        // Stop timer if time is up
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function updateLocalTimer() {
    if (!localEndTime) return;

    const now = Date.now() / 1000;
    const remaining = Math.max(0, localEndTime - now);
    const seconds = Math.ceil(remaining);  // Round up to nearest second

    // Update display
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    const display = `${mins}:${secs.toString().padStart(2, '0')}`;
    document.getElementById('timer-value').textContent = display;

    // Freeze detection: if no server update in 5+ seconds, show warning
    const timeSinceLastUpdate = Date.now() - lastServerUpdate;
    if (timeSinceLastUpdate > 5000) {
        // Timer might be frozen - show visual indicator
        document.getElementById('timer-value').style.color = '#ff6b6b';
        document.getElementById('timer-value').title = 'Connection lost - timer may be outdated';
    } else {
        // Normal - reset styling
        document.getElementById('timer-value').style.color = '';
        document.getElementById('timer-value').title = '';
    }

    // Stop interval if time is up
    if (remaining <= 0 && timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function renderBoard(board, grayed) {
    if (!board || board.length === 0) return;

    const boardEl = document.getElementById('game-board');
    const rows = board.length;
    const cols = board[0].length;

    // Set grid layout
    boardEl.style.gridTemplateColumns = `repeat(${cols}, 60px)`;
    boardEl.style.gridTemplateRows = `repeat(${rows}, 60px)`;

    // Clear and render
    boardEl.innerHTML = '';
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const cell = document.createElement('div');
            cell.className = 'board-cell' + (grayed ? ' grayed' : '');
            cell.textContent = board[r][c];
            boardEl.appendChild(cell);
        }
    }
}

function renderPlayers(players) {
    const listEl = document.getElementById('players-list');

    if (!players || players.length === 0) {
        listEl.innerHTML = '<p class="placeholder">No players</p>';
        return;
    }

    listEl.innerHTML = players.map(p => {
        // Format rating change: show +5 or -3, hide if 0
        const ratingChange = p.rating_change ? ` ${p.rating_change > 0 ? '+' : ''}${p.rating_change}` : '';
        // Add bonus-finder class if player found the bonus word
        const bonusClass = p.found_bonus_word ? ' bonus-finder' : '';
        return `
        <div class="player-item${bonusClass}">
            <div class="player-name">${p.username}</div>
            <div class="player-rating">(${p.rating}${ratingChange})</div>
            <div class="player-stats">${p.words_count} words • ${p.score} pts</div>
        </div>
    `;
    }).join('');
}

function displayAllWords(words, bonusWord, playerWords = [], allFoundWords = []) {
    const listEl = document.getElementById('submitted-words-list');

    if (!words || words.length === 0) {
        listEl.innerHTML = '<p class="placeholder">No words found</p>';
        return;
    }

    // Convert to uppercase for case-insensitive comparison
    const playerWordsUpper = playerWords.map(w => w.toUpperCase());
    const allFoundWordsUpper = allFoundWords.map(w => w.toUpperCase());

    // Sort words: by length (largest first), then alphabetically
    const sortedWords = [...words].sort((a, b) => {
        if (a.length !== b.length) {
            return b.length - a.length; // Descending order by length
        }
        return a.toUpperCase().localeCompare(b.toUpperCase()); // Alphabetically
    });

    // Display sorted words with appropriate highlighting
    listEl.innerHTML = sortedWords.map(word => {
        const wordUpper = word.toUpperCase();
        const isBonus = wordUpper === bonusWord.toUpperCase();
        const isPlayerWord = playerWordsUpper.includes(wordUpper);
        const isFound = allFoundWordsUpper.includes(wordUpper);

        // Priority: bonus word > player word > unfound word
        let className = 'word-item';
        if (isBonus) {
            className += ' bonus-word';
        } else if (isPlayerWord) {
            className += ' player-word';
        } else if (!isFound) {
            className += ' unfound';
        }

        return `<div class="${className}">${word}</div>`;
    }).join('');
}

// Spinner Overlay Functions
function showSpinnerOverlay(spinnerParams) {
    if (!spinnerParams) return;

    console.log('[play.js] Showing spinner overlay with params:', spinnerParams);

    // Populate spinner parameters
    document.getElementById('spinner-bonus-length').textContent = spinnerParams.bonus_word_length + ' letters';
    document.getElementById('spinner-min-length').textContent = spinnerParams.min_word_length;
    document.getElementById('spinner-difficulty').textContent = spinnerParams.difficulty;

    const [minWords, maxWords] = spinnerParams.word_count_range;
    document.getElementById('spinner-word-count').textContent = `${minWords}-${maxWords}`;

    document.getElementById('spinner-dictionary').textContent = spinnerParams.dictionary;
    document.getElementById('spinner-format').textContent = spinnerParams.board_format;

    // Show overlay
    const overlay = document.getElementById('spinner-overlay');
    overlay.classList.remove('hidden');
}

function hideSpinnerOverlay() {
    console.log('[play.js] Hiding spinner overlay');
    const overlay = document.getElementById('spinner-overlay');
    overlay.classList.add('hidden');
}

// Close button handler (add only if element exists)
const closeSpinnerBtn = document.getElementById('close-spinner-btn');
if (closeSpinnerBtn) {
    closeSpinnerBtn.addEventListener('click', () => {
        console.log('[play.js] Close spinner button clicked');
        hideSpinnerOverlay();
    });
}

// Word submission (add only if elements exist)
const submitBtn = document.getElementById('submit-word-btn');
const wordInput = document.getElementById('word-input');
if (submitBtn && wordInput) {
    submitBtn.addEventListener('click', submitWord);
    wordInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') submitWord();
    });

    // Auto-clear input when focused
    wordInput.addEventListener('focus', () => {
        wordInput.value = '';
    });
}

async function submitWord() {
    const input = document.getElementById('word-input');
    const word = input.value.trim().toUpperCase();
    const roomId = getCurrentRoomId();

    if (!word || !roomId) return;

    try {
        const response = await fetch(`/api/room/${roomId}/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word })
        });

        const data = await response.json();

        if (data.success) {
            // Add to submitted words list
            const listEl = document.getElementById('submitted-words-list');
            if (listEl.querySelector('.placeholder')) {
                listEl.innerHTML = '';
            }
            const wordItem = document.createElement('div');
            wordItem.className = 'word-item';
            wordItem.textContent = word;
            listEl.insertBefore(wordItem, listEl.firstChild);
        }
        // Invalid words fail silently - no popup alert

        // Always clear input after submission, valid or invalid
        input.value = '';
    } catch (error) {
        console.error('Error submitting word:', error);
        // Clear input even on error
        input.value = '';
    }
}

// Return to lobby (add only if element exists)
const returnBtn = document.getElementById('return-lobby-btn');
if (returnBtn) {
    returnBtn.addEventListener('click', async () => {
        const roomId = getCurrentRoomId();
        if (roomId) {
            try {
                await fetch(`/api/room/${roomId}/leave`, { method: 'POST' });
            } catch (error) {
                console.error('Error leaving room:', error);
            }

            window.currentRoomId = null;
            stopPolling();
        }

        const playBtn = document.getElementById('play-btn');
        if (playBtn) playBtn.disabled = true;
        showPage('lobby');
    });
}
