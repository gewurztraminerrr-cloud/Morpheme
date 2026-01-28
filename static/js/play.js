// Play Page JavaScript
let pollInterval = null;
let timerInterval = null;  // Separate interval for smooth timer updates
let previousState = null;
let localEndTime = null;  // End time in local clock terms
let lastServerUpdate = Date.now();  // Track last server response for freeze detection
let selectedPlayerUsername = null; // Track selected player for filtering/highlighting

// Mouse selection state
let mouseState = {
    isDown: false,
    selectedPath: [],       // Array of {row, col, letter}
    visitedCells: new Set() // Set of "row,col" strings
};

// Split Points UI State
let splitNotepadState = {}; // { username: 'unique' | 'split' | 'invalid' }
let showBoardInSplitIntermission = false;

function getCurrentRoomId() {
    return window.currentRoomId || null;
}

// Expose for lobby.js to call
window.startGamePolling = function () {
    startPolling();
};

window.stopGamePolling = function () {
    stopPolling();
};

function startPolling() {
    console.log('[play.js] startPolling() called');
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    console.log('[play.js] Setting up interval to call updateGameState');
    pollInterval = setInterval(updateGameState, 1000); // 1 second polling
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
    // console.log('[play.js] updateGameState() called, roomId:', roomId);
    if (!roomId) {
        console.warn('[play.js] No roomId found, exiting updateGameState');
        return;
    }

    try {
        // console.log(`[play.js] Fetching state from /api/room/${roomId}/state`);
        const response = await fetch(`/api/room/${roomId}/state`, { cache: 'no-store' });
        // console.log('[play.js] Fetch response received, status:', response.status);

        if (!response.ok) {
            console.error('[play.js] Fetch failed:', response.status, response.statusText);
            return;
        }

        const state = await response.json();
        console.log('[play.js] State received:', state); // DEBUG LOG
        window.lastGameState = state;  // Store for optimistic updates

        // update global room id if needed
        window.currentRoomId = state.room_id || roomId;

        const boardPanel = document.querySelector('.board-panel');
        const wordInputSection = document.querySelector('.word-input-section');
        const currentUsername = window.currentUser || localStorage.getItem('morpheme_username');
        const amIPlayer = state.players.some(p =>
            p.username.toLowerCase() === (currentUsername ? currentUsername.toLowerCase() : '')
        );

        if (!amIPlayer) {
            // I am a spectator
            if (wordInputSection) wordInputSection.style.display = 'none';

            // Ensure spectator panel exists
            let spectatorPanel = document.getElementById('spectator-status-panel');
            if (!spectatorPanel) {
                spectatorPanel = createSpectatorPanel();
            }

            spectatorPanel.style.display = 'flex';

            // Check if there is space to join
            const playerCount = (state.players && Array.isArray(state.players)) ? state.players.length : 0;
            const maxPlayers = state.max_players || 8;
            const canJoin = playerCount < maxPlayers;

            // Render Content
            spectatorPanel.innerHTML = `
                <div class="spectator-content-wrapper">
                    <div class="spectator-header">
                        <span class="spec-icon">👁️</span>
                        <h2 class="spectator-title">Spectating Mode</h2>
                    </div>
                    <p class="spectator-text">You are watching the game live.</p>
                    ${canJoin ?
                    `<button id="spec-join-btn" class="spectator-join-btn">Join Game</button>` :
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
                                setTimeout(updateGameState, 100);
                            } else {
                                alert(data.error);
                                joinBtn.textContent = 'Join Game';
                                joinBtn.disabled = false;
                            }
                        } catch (e) {
                            console.error(e);
                            alert('Network error');
                            joinBtn.textContent = 'Join Game';
                            joinBtn.disabled = false;
                        }
                    };
                }
            }, 0);

        } else {
            // I am a player
            if (wordInputSection) wordInputSection.style.display = ''; // Restore flex/block
            const specPanel = document.getElementById('spectator-status-panel');
            if (specPanel) specPanel.style.display = 'none';
        }

        if (state.error) {
            console.error('Room error:', state.error);
            stopPolling();
            return;
        }

        console.log('[play.js] Updating parameters...');
        // Update parameters
        updateParameters(state);

        // Sync local timer with server
        syncTimerWithServer(state);

        // Track last successful server update
        lastServerUpdate = Date.now();

        // Render board
        console.log('[play.js] Rendering board...');
        const isSplitIntermission = (state.game_type === 'split' && state.state === 'intermission');
        const isFCFSIntermission = (state.game_type === 'fcfs' && state.state === 'intermission');

        if (isSplitIntermission && !showBoardInSplitIntermission) {
            renderSplitNotepads(state.players);
        } else if (isFCFSIntermission && !showBoardInSplitIntermission) {
            renderFCFSNotepads(state.players);
        } else {
            renderBoard(state.board, state.state === 'intermission');
        }

        // Update players (pass full state for context if needed)
        renderPlayers(state.players);

        // Update chat
        if (state.chat_messages) {
            renderChat(state.chat_messages);
        }

        // Enable/disable input
        const isActive = state.state === 'active';
        const inputEl = document.getElementById('word-input');
        const submitBtn = document.getElementById('submit-word-btn');

        if (state.game_type === 'split' && state.state === 'intermission') {
            // Hide for Split Points intermission
            inputEl.style.display = 'none';
            submitBtn.style.display = 'none';
        } else {
            // Show otherwise
            inputEl.style.display = ''; // Reset to default (block/flex)
            submitBtn.style.display = '';

            if (inputEl.disabled === isActive) {
                inputEl.disabled = !isActive;
            }
            if (submitBtn.disabled === isActive) {
                submitBtn.disabled = !isActive;
            }
        }

        // Check for state transitions
        if (previousState !== state.state) {
            if (state.state === 'intermission' && previousState === 'active') {
                showSpinnerOverlay(state.spinner_params);

                // Focus Chat on Intermission
                setTimeout(() => {
                    const chatInput = document.getElementById('chat-input');
                    if (chatInput) chatInput.focus();
                }, 100);

                // Reset scroll position
                const listEl = document.getElementById('submitted-words-list');
                if (listEl && listEl.parentElement) {
                    listEl.parentElement.scrollTop = 0;
                }
            } else if (state.state === 'active' && previousState !== 'active') {
                hideSpinnerOverlay();
                const wordsList = document.getElementById('submitted-words-list');
                wordsList.innerHTML = '<p class="placeholder">Game active - Waiting for words...</p>';

                // Focus Word Input on Game Start
                setTimeout(() => {
                    const wordInput = document.getElementById('word-input');
                    if (wordInput) wordInput.focus();
                }, 100);
            }
            previousState = state.state;
        }

        // Update words panel based on state
        const wordsPanelHeader = document.querySelector('.words-panel h3');
        const wordsStats = document.getElementById('words-stats');

        // Identify current user
        let currentUser = null;
        try {
            currentUser = window.currentUser || (typeof currentUser !== 'undefined' ? currentUser : null);
        } catch (e) { console.warn('No currentUser', e); }

        if (state.state === 'intermission') {
            // INTERMISSION: Show ALL words, highlight selected player's words
            wordsPanelHeader.textContent = 'All Words';

            // Collect all player words to calculate stats
            let allPlayerWords = [];
            state.players.forEach(p => {
                if (p.submitted_words) {
                    // Handle both object and legacy string format
                    p.submitted_words.forEach(w => {
                        const str = typeof w === 'string' ? w : w.word;
                        allPlayerWords.push(str);
                    });
                }
            });

            const totalWords = state.all_words ? state.all_words.length : 0;
            const uniqueFound = new Set(allPlayerWords.map(w => w.toUpperCase())).size;
            const percentage = totalWords > 0 ? Math.round((uniqueFound / totalWords) * 100) : 0;
            wordsStats.textContent = `${uniqueFound}/${totalWords} - ${percentage}%`;

            // Determine which player to highlight
            // Default to current user if no manual selection, or stick to manual selection
            const targetUsername = selectedPlayerUsername || currentUser;

            // Get target player's words
            let targetWords = [];
            if (targetUsername) {
                const targetPlayer = state.players.find(p => p.username === targetUsername);
                if (targetPlayer && targetPlayer.submitted_words) {
                    targetWords = targetPlayer.submitted_words.map(w => typeof w === 'string' ? w : w.word);
                }
            }

            // Also collect ALL found words (strings) for styling
            const allFoundStrs = allPlayerWords.map(w => w.toUpperCase());

            // For FCFS and others, show All Words
            wordsPanelHeader.textContent = 'All Words';

            // FCFS or Standard Intermission:
            // Highlighting "targetWords" covers user-found words.
            // Highlighting "allFoundStrs" covers words found by ANYONE (if we want that).
            // User request: "highlights all words the user found under All Words"

            displayAllWords(state.all_words, state.bonus_word, targetWords, allFoundStrs);

            // For Split Points OR FCFS: Add "View Board" toggle button if needed
            if (state.game_type === 'split' || state.game_type === 'fcfs') {
                addSplitViewBoardToggle();
            }

        } else {
            // ACTIVE STATE
            // ACTIVE STATE
            if (state.game_type === 'accumulative' || state.game_type === 'split') {
                // ACCUMULATIVE / SPLIT: Show only MY words
                wordsPanelHeader.textContent = 'Your Words';

                // Find my words
                // console.log('[play.js] CurrentUser:', currentUser);
                const myPlayer = state.players.find(p => p.username === currentUser);
                // console.log('[play.js] MyPlayer:', myPlayer);
                const myWords = myPlayer ? myPlayer.submitted_words : [];
                // console.log('[play.js] MyWords:', myWords);

                // Calculate stats
                const totalWords = state.all_words ? state.all_words.length : 0;
                const uniqueFound = new Set(myWords.map(w => (typeof w === 'string' ? w : w.word).toUpperCase())).size;
                const percentage = totalWords > 0 ? Math.round((uniqueFound / totalWords) * 100) : 0;
                wordsStats.textContent = `${uniqueFound}/${totalWords} - ${percentage}%`;

                // Render list (Simple Rebuild)
                const listEl = document.getElementById('submitted-words-list');

                // Sort by time descending (newest first)
                // Handle legacy string format just in case
                const sortedWords = [...myWords].sort((a, b) => {
                    const tA = a.time || 0;
                    const tB = b.time || 0;
                    return tB - tA;
                });

                if (sortedWords.length === 0) {
                    listEl.innerHTML = '<p class="placeholder">Find words!</p>';
                } else {
                    const html = sortedWords.map(wObj => {
                        const word = typeof wObj === 'string' ? wObj : wObj.word;
                        const points = typeof wObj === 'object' ? wObj.points : '?';
                        // Check bonus
                        const isBonus = word === state.bonus_word;

                        let className = 'word-item player-word';
                        if (isBonus) className += ' bonus-word';

                        return `<div class="${className}" style="display:flex; justify-content:space-between;">
                <span>${word}</span>
                <span style="opacity:0.8">${points}</span>
            </div>`;
                    }).join('');
                    listEl.innerHTML = html;
                }

            } else {
                // FCFS / Other: ALSO Show "Your Words", but sorted differently
                // Originally this was a live feed, but user requested: "simply lists the words one got"

                wordsPanelHeader.textContent = 'Your Words';

                // Find my words
                const myPlayer = state.players.find(p => p.username === currentUser);
                const myWords = myPlayer ? myPlayer.submitted_words : [];

                // Calculate stats
                const totalWords = state.all_words ? state.all_words.length : 0;
                const uniqueFound = new Set(myWords.map(w => (typeof w === 'string' ? w : w.word).toUpperCase())).size;
                const percentage = totalWords > 0 ? Math.round((uniqueFound / totalWords) * 100) : 0;
                wordsStats.textContent = `${uniqueFound}/${totalWords} - ${percentage}%`;

                // Render list
                const listEl = document.getElementById('submitted-words-list');

                // FCFS Sort: Time Ascending (Newest at Bottom)
                const sortedWords = [...myWords].sort((a, b) => {
                    const tA = a.time || 0;
                    const tB = b.time || 0;
                    return tA - tB; // Ascending
                });

                // Capture scroll state BEFORE render (if we are appending, though here we rebuild)
                // Relaxed threshold to 150px to prevent "halting" if user is mostly at bottom
                const threshold = 150;
                const isAtBottom = (listEl.scrollTop + listEl.clientHeight >= listEl.scrollHeight - threshold);
                const wasEmpty = listEl.innerHTML.includes('placeholder') || listEl.children.length === 0;

                if (sortedWords.length === 0) {
                    listEl.innerHTML = '<p class="placeholder">Find words!</p>';
                } else {
                    // Check if content changed to avoid flicker? 
                    // For now, full rebuild is safer for consistency
                    const html = sortedWords.map(wObj => {
                        const word = typeof wObj === 'string' ? wObj : wObj.word;
                        const points = typeof wObj === 'object' ? wObj.points : '?';
                        const isBonus = word === state.bonus_word;

                        let className = 'word-item player-word';
                        if (isBonus) className += ' bonus-word';

                        return `<div class="${className}" style="display:flex; justify-content:space-between;">
                        <span>${word}</span>
                        <span style="opacity:0.8">${points}</span>
                    </div>`;
                    }).join('');

                    if (listEl.innerHTML !== html) {
                        listEl.innerHTML = html;
                        // Smart Scroll:
                        // If we were at bottom, OR if it's the first render (wasEmpty), scroll to bottom
                        if (isAtBottom || wasEmpty) {
                            requestAnimationFrame(() => {
                                listEl.scrollTop = listEl.scrollHeight;
                            });
                        }
                    }
                }
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

function renderPlayers(players) {
    const listEl = document.getElementById('players-list');
    if (!players || players.length === 0) {
        listEl.innerHTML = '<p class="placeholder">No players</p>';
        return;
    }

    // Sort players by score (Highest First), break ties with Rating
    const sortedPlayers = [...players].sort((a, b) => (b.score - a.score) || (b.rating - a.rating));

    // Get current user
    let currentUser = null;
    try {
        currentUser = window.currentUser || (typeof currentUser !== 'undefined' ? currentUser : null);
    } catch (e) { }

    // Attach click handler via event delegation parent or recreate list
    // Recreating list is fine here

    // Top 3 + Nearby Logic (simplified for brevity: show top 20)
    // For now, just render top 20 to keep it simple and scrollable
    const itemsToRender = sortedPlayers.slice(0, 20);

    const html = itemsToRender.map((p, index) => {
        const ratingChange = p.rating_change ? ` ${p.rating_change > 0 ? '+' : ''}${p.rating_change}` : '';
        const bonusClass = p.found_bonus_word ? ' bonus-finder' : '';
        const userClass = (p.username === currentUser) ? ' current-user' : '';
        const rank = index + 1;

        // Highlight if selected
        const selectedClass = (p.username === selectedPlayerUsername) ? ' selected-player' : '';

        // Override rating for Guest users
        const isGuest = p.username.startsWith('Guest_');
        const displayRating = isGuest ? 0 : p.rating;

        // Calculate rating color
        const ratingColor = window.getRatingColor ? window.getRatingColor(displayRating) : '#fff';

        return `
        <div class="player-item${bonusClass}${userClass}${selectedClass}" data-username="${p.username}">
            <div class="player-name">
                #${rank} 
                <span class="rating-indicator" style="background-color: ${ratingColor}; margin-left: 5px;"></span>
                ${p.username}
            </div>
            <div class="player-rating">(${displayRating}${ratingChange})</div>
            <div class="player-stats">${p.words_count} words • ${p.score} pts</div>
        </div>
        `;
    }).join('');

    listEl.innerHTML = html;

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
let lastChatCount = 0;

function renderChat(messages) {
    const listEl = document.getElementById('chat-history');
    if (!listEl) return;

    if (!messages || messages.length === 0) {
        if (listEl.innerHTML.trim() === '') {
            listEl.innerHTML = '<p class="placeholder">No messages yet</p>';
        }
        return;
    }

    // Only update if new messages arrived
    if (messages.length === lastChatCount) return;
    lastChatCount = messages.length;

    // Remove placeholder
    const placeholder = listEl.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    // Render all messages (simple rebuild for now to ensure order)
    const html = messages.map(msg => {
        const username = msg.username;
        const text = msg.message;
        // Escape HTML to prevent XSS
        const safeText = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

        return `
        <div class="chat-message">
            <span class="chat-user">${username}:</span>
            <span class="chat-text">${safeText}</span>
        </div>`;
    }).join('');

    listEl.innerHTML = html;

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
});

function displayAllWords(allWords, bonusWord, targetUserWords = [], allFoundWords = []) {
    const listEl = document.getElementById('submitted-words-list');
    if (!allWords || allWords.length === 0) {
        listEl.innerHTML = '<p class="placeholder">No words found</p>';
        return;
    }

    const targetWordsUpper = targetUserWords.map(w => w.toUpperCase());
    const allFoundUpper = allFoundWords.map(w => w.toUpperCase());

    // Sort: Length desc, then Alpha
    const sortedWords = [...allWords].sort((a, b) => {
        if (a.length !== b.length) return b.length - a.length;
        return a.toUpperCase().localeCompare(b.toUpperCase());
    });

    listEl.innerHTML = sortedWords.map(word => {
        const wordUpper = word.toUpperCase();
        const isBonus = wordUpper === bonusWord.toUpperCase();

        let className = 'word-item';
        // Logic for highlighting:
        // User requested: "bluish color on the list highlighting each word the user that was selected... found"
        // So check if targetUserWords has it.
        const isTargetFound = targetWordsUpper.includes(wordUpper);

        if (isBonus) {
            className += ' bonus-word';
        }

        // If selected user found it -> Blue
        if (isTargetFound) {
            className += ' player-word'; // This class is styled blue/green usually
        } else if (allFoundUpper.includes(wordUpper)) {
            // Found by someone else -> maybe white?
            // className += ' found-by-other';
            // Default styling is white for found?
        } else {
            // Not found by anyone -> Gray
            className += ' unfound';
        }

        return `<div class="${className}">${word}</div>`;
    }).join('');
}

// ... existing functions (updateParameters, renderBoard, etc) ...
// Copying existing helper functions to ensure file completion

function updateParameters(state) {
    // Display mappings
    const typeMap = {
        'accumulative': 'Accumulative',
        'fcfs': 'First Come First Serve',
        'split': 'Split Points'
    };

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

    document.getElementById('param-board').textContent = state.board_dimensions;
    document.getElementById('param-time').textContent = state.time_limit + 's';

    const sp = state.spinner_params;
    if (sp) {
        document.getElementById('param-bonus').textContent = sp.bonus_word_length + ' letters';
        document.getElementById('param-diff').textContent = sp.difficulty;
        document.getElementById('param-min').textContent = sp.min_word_length;
        document.getElementById('param-dict').textContent = sp.dictionary;

        const wr = sp.word_count_range;
        document.getElementById('param-words').textContent = `${wr[0]}-${wr[1]}`;
        document.getElementById('param-format').textContent = sp.board_format;
    }
}

function updateTimer(seconds) {
    // Legacy local timer update (called by interval)
    // see updateLocalTimer
}

function syncTimerWithServer(state) {
    const clientTime = Date.now() / 1000;
    const serverTime = state.server_time;
    const serverTimeOffset = serverTime - clientTime;

    let endTime = 0;
    if (state.state === 'active' && state.round_end_time) {
        endTime = state.round_end_time;
    } else if (state.state === 'intermission' && state.intermission_end_time) {
        endTime = state.intermission_end_time;
    }

    localEndTime = endTime - serverTimeOffset;

    if (!timerInterval && localEndTime > 0) {
        timerInterval = setInterval(updateLocalTimer, 100);
    } else if (localEndTime <= 0 && timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function updateLocalTimer() {
    if (!localEndTime) return;

    const now = Date.now() / 1000;
    const remaining = Math.max(0, localEndTime - now);
    const seconds = Math.ceil(remaining);

    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    const display = `${mins}:${secs.toString().padStart(2, '0')}`;
    document.getElementById('timer-value').textContent = display;

    // Freeze detection
    if (Date.now() - lastServerUpdate > 5000) {
        document.getElementById('timer-value').style.color = '#ff6b6b';
    } else {
        document.getElementById('timer-value').style.color = '';
    }

    // Low time visual
    const boardPanel = document.querySelector('.board-panel');
    if (boardPanel) {
        if (remaining <= 10 && remaining > 0 && previousState === 'active') {
            boardPanel.classList.add('low-time-warning');
        } else {
            boardPanel.classList.remove('low-time-warning');
        }
    }

    if (remaining <= 0 && timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        if (boardPanel) boardPanel.classList.remove('low-time-warning');
    }
}

function renderBoard(board, grayed) {
    if (!board || board.length === 0) return;
    const boardEl = document.getElementById('game-board');
    if (!boardEl) return; // play page might not be active

    // Reset container style
    boardEl.className = 'game-board';
    // Clear styles set by renderSplitNotepads
    boardEl.style.display = '';
    boardEl.style.gap = '';
    boardEl.style.overflowY = '';
    boardEl.style.alignItems = '';
    // Ensure grid is active (class has it, but just in case of overrides)
    // boardEl.style.display = 'inline-grid'; 

    const rows = board.length;
    const cols = board[0].length;

    boardEl.style.gridTemplateColumns = `repeat(${cols}, 60px)`;
    boardEl.style.gridTemplateRows = `repeat(${rows}, 60px)`;
    boardEl.innerHTML = '';

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const cell = document.createElement('div');
            cell.className = 'board-cell' + (grayed ? ' grayed' : '');
            const letter = board[r][c];
            cell.textContent = letter === 'Q' ? 'QU' : letter;
            cell.dataset.row = r;
            cell.dataset.col = c;
            boardEl.appendChild(cell);
        }
    }
}

function renderSplitNotepads(players) {
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
    boardEl.style.display = 'grid';
    boardEl.style.gridTemplateColumns = 'repeat(auto-fit, minmax(200px, 1fr))';
    boardEl.style.gridTemplateRows = 'auto'; // allow expansion
    boardEl.style.gap = '1rem';
    boardEl.style.overflowY = 'auto';
    boardEl.style.alignItems = 'start'; // Align items to top
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
        // Add click to select user (for highlighting words on right panel)
        notepad.onclick = (e) => {
            // Avoid triggering if clicking tabs
            if (e.target.classList.contains('notepad-tab')) return;

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
        header.innerHTML = `<strong>${p.username}</strong> <span>${p.score} pts</span>`;
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

        // List
        const list = document.createElement('div');
        list.className = 'notepad-list';

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
                    // Metadata added by backend: is_unique, split_points
                    console.log('Split Word Obj:', w);
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
            list.innerHTML = '<div style="color:#000;font-style:italic;padding:10px;text-align:center;">None</div>';
        } else {
            wordsToShow.forEach(w => {
                const row = document.createElement('div');
                row.className = 'notepad-item';
                if (w.is_invalid) row.classList.add('invalid');

                row.innerHTML = `<span>${w.word}</span> <span>${w.points}</span>`;
                list.appendChild(row);
            });
        }

        notepad.appendChild(list);
        boardEl.appendChild(notepad);

        // Restore scroll position
        if (scrollMap[p.username]) {
            list.scrollTop = scrollMap[p.username];
        }
    });
}

function renderFCFSNotepads(players) {
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
    boardEl.style.display = 'grid';
    boardEl.style.gridTemplateColumns = 'repeat(auto-fit, minmax(200px, 1fr))';
    boardEl.style.gridTemplateRows = 'auto';
    boardEl.style.gap = '1rem';
    boardEl.style.overflowY = 'auto';
    boardEl.style.alignItems = 'start';
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
        header.innerHTML = `<strong>${p.username}</strong> <span>${p.score} pts</span>`;
        notepad.appendChild(header);

        // No Tabs for FCFS

        // List
        const list = document.createElement('div');
        list.className = 'notepad-list';
        // Add extra padding since no tabs
        list.style.marginTop = '10px';
        // Adjust height to fill space better without tabs
        list.style.height = 'calc(100% - 40px)';

        if (!p.submitted_words || p.submitted_words.length === 0) {
            list.innerHTML = '<div style="color:#000;font-style:italic;padding:10px;text-align:center;">None</div>';
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
                const pts = (typeof wObj === 'string' ? '?' : wObj.points);

                const row = document.createElement('div');
                row.className = 'notepad-item';
                row.innerHTML = `<span>${w}</span> <span>${pts}</span>`;
                list.appendChild(row);
            });
        }

        notepad.appendChild(list);
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
function showSpinnerOverlay(spinnerParams) {
    if (!spinnerParams) return;
    document.getElementById('spinner-bonus-length').textContent = spinnerParams.bonus_word_length + ' letters';
    document.getElementById('spinner-min-length').textContent = spinnerParams.min_word_length;
    document.getElementById('spinner-difficulty').textContent = spinnerParams.difficulty;
    const [minWords, maxWords] = spinnerParams.word_count_range;
    document.getElementById('spinner-word-count').textContent = `${minWords}-${maxWords}`;
    document.getElementById('spinner-dictionary').textContent = spinnerParams.dictionary;
    document.getElementById('spinner-format').textContent = spinnerParams.board_format;

    document.getElementById('spinner-overlay').classList.remove('hidden');
}

function hideSpinnerOverlay() {
    document.getElementById('spinner-overlay').classList.add('hidden');

    // If closing spinner during intermission, likely want to chat
    if (window.lastGameState && window.lastGameState.state === 'intermission') {
        const chatInput = document.getElementById('chat-input');
        if (chatInput) {
            setTimeout(() => chatInput.focus(), 50);
        }
    }
}

const closeSpinnerBtn = document.getElementById('close-spinner-btn');
if (closeSpinnerBtn) {
    closeSpinnerBtn.addEventListener('click', hideSpinnerOverlay);
}

// Word Submission
const submitBtn = document.getElementById('submit-word-btn');
const wordInputEl = document.getElementById('word-input');
if (submitBtn && wordInputEl) {
    submitBtn.addEventListener('click', () => submitWord());
    wordInputEl.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') submitWord();
    });
}

async function submitWord(wordParam = null) {
    const input = document.getElementById('word-input');
    const word = wordParam ? wordParam.toUpperCase() : input.value.trim().toUpperCase();
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
            // Optimistic Update for Accumulative Mode
            const currentState = window.lastGameState;
            if (currentState && currentState.game_type === 'accumulative') {
                const listEl = document.getElementById('submitted-words-list');
                const wordsStats = document.getElementById('words-stats');

                // Remove placeholder
                const placeholder = listEl.querySelector('.placeholder');
                if (placeholder) placeholder.remove();

                // Create HTML
                const isBonus = data.word === currentState.bonus_word;
                let className = 'word-item player-word';
                if (isBonus) className += ' bonus-word';

                const html = `<div class="${className}" style="display:flex; justify-content:space-between; animation: slideIn 0.3s ease;">
                    <span>${data.word}</span>
                    <span style="opacity:0.8">${data.points}</span>
                </div>`;

                // Prepend to top (since we sort by newest)
                listEl.insertAdjacentHTML('afterbegin', html);

                // Update stats text optimistically
                if (wordsStats) {
                    const parts = wordsStats.textContent.match(/(\d+)\/(\d+) - (\d+)%/);
                    if (parts) {
                        let found = parseInt(parts[1]);
                        const total = parseInt(parts[2]);
                        found++;
                        const percent = total > 0 ? Math.round((found / total) * 100) : 0;
                        wordsStats.textContent = `${found}/${total} - ${percent}%`;
                    } else if (currentState.all_words) {
                        const total = currentState.all_words.length;
                        const found = 1;
                        const percent = total > 0 ? Math.round((found / total) * 100) : 0;
                        wordsStats.textContent = `${found}/${total} - ${percent}%`;
                    }
                }
            } else if (currentState && currentState.game_type === 'fcfs') {
                // FCFS Optimistic Update (Live Feed)
                const listEl = document.getElementById('submitted-words-list');
                const wordsStats = document.getElementById('words-stats');

                // Remove placeholder
                const placeholder = listEl.querySelector('.placeholder');
                if (placeholder) placeholder.remove();

                // Create Feed Item HTML
                // Structure: <div id="feed-USER-WORD" class="feed-item myself">...</div>
                const currentUser = window.currentUser || 'You';
                const itemId = `feed-${currentUser}-${data.word}`.replace(/\s+/g, '');

                // Avoid checking if exists because success=true implies it's new/valid

                const html = `
                <div id="${itemId}" class="feed-item myself" style="animation: slideInRight 0.3s ease;">
                    <span class="feed-word">${data.word}</span>
                    <span class="feed-info">${currentUser} • ${data.points}pts</span>
                </div>`;

                // Append to BOTTOM (Live Feed order is oldest -> newest, or actually newest at bottom usually for chat-like)
                // In render implementation: feedItems.sort((a, b) => a.time - b.time); (Oldest first)
                // So new items go to bottom.
                listEl.insertAdjacentHTML('beforeend', html);

                // Scroll to bottom
                const panelEl = listEl.parentElement; // or closest('.words-panel')
                requestAnimationFrame(() => {
                    if (panelEl) panelEl.scrollTop = panelEl.scrollHeight;
                });

                // Update stats text optimistically
                if (wordsStats) {
                    const parts = wordsStats.textContent.match(/(\d+)\/(\d+) - (\d+)%/);
                    if (parts) {
                        let found = parseInt(parts[1]);
                        const total = parseInt(parts[2]);
                        found++;
                        const percent = total > 0 ? Math.round((found / total) * 100) : 0;
                        wordsStats.textContent = `${found}/${total} - ${percent}%`;
                    } else if (currentState.all_words) {
                        const total = currentState.all_words.length;
                        const found = 1;
                        const percent = total > 0 ? Math.round((found / total) * 100) : 0;
                        wordsStats.textContent = `${found}/${total} - ${percent}%`;
                    }
                }
            }

            // Still trigger poll to sync everything else (User list scores, etc)
            // But immediate feedback is done
        }
        input.value = '';
    } catch (error) {
        console.error('Error submitting word:', error);
        input.value = '';
    }
}

async function leaveCurrentRoom() {
    const roomId = getCurrentRoomId();
    if (!roomId) return;
    try { await fetch(`/api/room/${roomId}/leave`, { method: 'POST' }); } catch (e) { }
    window.currentRoomId = null;
    stopPolling();
    // clear UI
}
window.leaveCurrentRoom = leaveCurrentRoom;

const returnBtnEl = document.getElementById('return-lobby-btn');
if (returnBtnEl) {
    returnBtnEl.addEventListener('click', async () => {
        await leaveCurrentRoom();
        showPage('page-lobby');
    });
}

// Definition Logic
async function fetchDefinition(word) {
    if (!word) return;
    const defContent = document.getElementById('definition-content');
    if (!defContent) return;
    defContent.innerHTML = '<p class="placeholder">Loading definition...</p>';
    try {
        const resp = await fetch(`/api/definition?word=${encodeURIComponent(word)}`);
        const data = await resp.json();
        if (data.definition) {
            defContent.innerHTML = `<span class="definition-word">${data.word}</span><span class="definition-text">${data.definition}</span>`;
        } else {
            defContent.innerHTML = `<p class="placeholder">Definition not found.</p>`;
        }
    } catch (e) { defContent.innerHTML = '<p class="placeholder">Error loading.</p>'; }
}

const submittedWordsListEl = document.getElementById('submitted-words-list');
if (submittedWordsListEl) {
    submittedWordsListEl.addEventListener('click', (e) => {
        // Handle clicks on feed items too
        // Check for word-item or feed-word
        if (e.target.classList.contains('word-item') || e.target.classList.contains('feed-word')) {
            const word = e.target.textContent.trim();
            fetchDefinition(word);
        } else if (e.target.closest('.feed-item')) {
            const word = e.target.closest('.feed-item').querySelector('.feed-word').textContent.trim();
            fetchDefinition(word);
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

    // Append to BOARD PANEL (Center)
    // It should replace the word-input-section area visually
    const boardPanel = document.querySelector('.board-panel');
    if (boardPanel) {
        boardPanel.appendChild(panel);
    } else {
        // Fallback
        document.body.appendChild(panel);
    }

    return panel;
}

// Interaction Handlers (Override)
function handleCellMouseDown(e) {
    if (e.button !== 0) return; // Only left click

    // Spectator Check
    if (window.isSpectatorMode) return;

    const cell = e.target.closest('.board-cell');
    if (!cell) return;

    mouseState.isDown = true;

    const row = parseInt(cell.dataset.row);
    const col = parseInt(cell.dataset.col);
    const letter = cell.dataset.letter;

    selectCell(row, col, letter, cell);
}

function handleCellTouchStart(e) {
    if (window.isSpectatorMode) return;

    const touch = e.touches[0];
    const cell = document.elementFromPoint(touch.clientX, touch.clientY);

    if (cell && cell.closest('.board-cell')) {
        e.preventDefault(); // Prevent scrolling
        mouseState.isDown = true;

        const target = cell.closest('.board-cell');
        const row = parseInt(target.dataset.row);
        const col = parseInt(target.dataset.col);
        const letter = target.dataset.letter;

        selectCell(row, col, letter, target);
    }
}
