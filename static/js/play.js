// Play Page JavaScript
let pollInterval = null;
let timerInterval = null;  // Separate interval for smooth timer updates
let previousState = null;
let localEndTime = null;  // End time in local clock terms
let lastServerUpdate = Date.now();  // Track last server response for freeze detection
let selectedPlayerUsername = null; // Track selected player for filtering/highlighting
let cachedTimerValueEl = null;    // Cache for high-frequency updates
let cachedBoardPanelEl = null;
let lastPlayersHtml = null;       // Cache for renderPlayers

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
let hasPlayedIntermissionBell = false; // Flag for next round notification

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

// Add global listeners for input detection
document.addEventListener('keydown', () => updateInputMethod('keyboard'), true);
document.addEventListener('mousedown', () => updateInputMethod('mouse'), true);
document.addEventListener('touchstart', () => updateInputMethod('touch'), true);

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

    // Reset Chat for new room
    resetChat();

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

        // Capture previous state for transition logic (e.g. daily reset kick)
        const previousState = window.lastGameState;
        window.lastGameState = state;  // Store for optimistic updates

        // Detect transition to intermission (round end)
        if (previousState && previousState.state === 'active' && state.state === 'intermission') {
            const wordInput = document.getElementById('word-input');
            if (wordInput) {
                wordInput.value = '';
                wordInput.blur();
            }
            if (typeof mouseState !== 'undefined') {
                mouseState.isDown = false;
                mouseState.selectedPath = [];
                if (mouseState.visitedCells) mouseState.visitedCells.clear();
            }
        }

        // update global room id if needed
        window.currentRoomId = state.room_id || roomId;

        const boardPanel = document.querySelector('.board-panel');
        const wordInputSection = document.querySelector('.word-input-section');
        const currentUsername = state.your_username || window.currentUser || localStorage.getItem('morpheme_username');

        // Cache for recovery after reset
        if (currentUsername) {
            localStorage.setItem('last_morpheme_user', currentUsername);
        }

        const amIPlayer = state.players.some(p => {
            const match = p.username.toLowerCase() === (currentUsername ? currentUsername.toLowerCase().trim() : '');
            return match;
        });

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
            if (defContent) defContent.style.display = 'none';
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
            const defContent = document.getElementById('definition-content');
            if (defContent) defContent.style.display = '';
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
        renderPlayers(state.players, currentUsername, state);

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
        const lastStateStr = previousState ? previousState.state : null;
        if (lastStateStr !== state.state) {
            if (state.state === 'intermission' && lastStateStr === 'active') {
                showSpinnerOverlay(state.spinner_params, state.players);

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
            } else if (state.state === 'active' && lastStateStr !== 'active') {
                hideSpinnerOverlay();
                const wordsList = document.getElementById('submitted-words-list');
                // Only reset if we actually transitioned FROM something else (avoid initial reset if page load)
                if (lastStateStr) {
                    wordsList.innerHTML = '<p class="placeholder">Game active - Waiting for words...</p>';
                }

                // Reset Highlighting
                highlightedSplitWord = null;
                highlightedFoundWord = null;
                selectedPlayerUsername = null; // Reset player selection

                // Reset Player List Scroll (Undo "Find Me")
                const playersListEl = document.getElementById('players-list');
                if (playersListEl) {
                    playersListEl.scrollTop = 0;
                    // And potentially the parent container if needed
                    if (playersListEl.parentElement) playersListEl.parentElement.scrollTop = 0;
                }

                // DATA SYNC FIX: Explicitly clear Remaining list to prevent crossover
                const remainingList = document.getElementById('remaining-words-list');
                if (remainingList) remainingList.innerHTML = '';

                // Focus Word Input on Game Start
                setTimeout(() => {
                    const wordInput = document.getElementById('word-input');
                    if (wordInput) wordInput.focus();
                }, 100);
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

            if (activeWordsTab === 'remaining') headerText = 'Remaining';
            if (activeWordsTab === 'clues') headerText = 'Clues';
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

        // Identify current user
        let currentUser = null;
        try {
            currentUser = state.your_username || window.currentUser || (typeof currentUser !== 'undefined' ? currentUser : null);
        } catch (e) { console.warn('No currentUser', e); }

        // KICK LOGIC FOR 24H RESET
        // If we were in the room in the previous state, but are gone now (and it's a 24h room),
        // it means the daily reset wiped us. Redirect to lobby.
        const is24H = (state.game_type === 'accumulative' && state.time_limit >= 7200);

        if (is24H && previousState && previousState.players && currentUser) {
            const prevState = previousState;

            // Was I in the room before?
            const wasIn = prevState.players.some(p => p.username === currentUser) ||
                (prevState.spectators || []).some(s => s.username === currentUser);

            // Am I in the room now?
            const isIn = state.players.some(p => p.username === currentUser) ||
                (state.spectators || []).some(s => s.username === currentUser);

            if (wasIn && !isIn) {
                // Only kick if round changed or time jumped (Reset likely)
                // Actually, in 24h room, removal ONLY happens on reset.
                console.log("User removed from 24H room - Attempting Auto-Rejoin");

                // Attempt to join immediately
                fetch(`/api/room/${roomId}/join`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ as_spectator: false })
                })
                    .then(resp => resp.json())
                    .then(data => {
                        if (data.success) {
                            console.log("Auto-rejoin successful!");
                            // Toast or small notification could go here
                        } else {
                            console.error("Auto-rejoin failed:", data.error);
                            alert("Daily Reset! You have been moved to the lobby.");
                            window.location.href = '/';
                        }
                    })
                    .catch(err => {
                        console.error("Auto-rejoin error:", err);
                        window.location.href = '/';
                    });

                return; // Wait for join response
            }
        }

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
                const totalWords = allWords.length;
                const globalUnique = new Set(allPlayerFoundStrs).size;
                const globalPercentage = totalWords > 0 ? Math.round((globalUnique / totalWords) * 100) : 0;

                // 2. Calculate Personal Stats (Current User)
                const myPlayer = state.players.find(p => p.username === currentUser);
                const myWords = myPlayer ? (myPlayer.submitted_words || []) : [];
                const personalUnique = new Set(myWords.map(w => (typeof w === 'string' ? w : w.word).toUpperCase())).size;
                const personalPercentage = totalWords > 0 ? Math.round((personalUnique / totalWords) * 100) : 0;

                // Display Both
                wordsStats.innerHTML = `
                    <div style="line-height: 1.2;">
                        ${personalUnique}/${totalWords} - ${personalPercentage}%
                        <div style="font-size: 0.75em; opacity: 0.7; margin-top: 2px;">
                            Total Found Percentage: ${globalPercentage}%
                        </div>
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
                displayAllWords(allWords, state.bonus_word, targetWords, uniqueGlobalFound, state.all_word_scores, state.csw_only_words);
                if (state.game_type === 'split' || state.game_type === 'fcfs') addSplitViewBoardToggle();

            } else if (state.game_type !== 'fcfs') {
                // ACTIVE STATE (Not Intermission) & Not FCFS
                // Personal List for Standard, Split, AND Accumulative
                const myPlayer = state.players.find(p => p.username === currentUser);
                const myWords = myPlayer ? (myPlayer.submitted_words || []) : [];


                // 2. Personal Stats Only (Active)
                const totalWords = allWords.length;
                const uniqueFound = new Set(myWords.map(w => (typeof w === 'string' ? w : w.word).toUpperCase())).size;
                const percentage = totalWords > 0 ? Math.round((uniqueFound / totalWords) * 100) : 0;

                // Single line display
                wordsStats.textContent = `${uniqueFound}/${totalWords} - ${percentage}%`;

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

                        let className = 'word-item player-word';
                        if (isBonus) className += ' bonus-word';
                        if (isCSWOnly) className += ' csw-only';
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
                let allFoundWords = [];
                state.players.forEach(p => {
                    if (p.submitted_words) {
                        p.submitted_words.forEach(w => {
                            let wObj = (typeof w === 'string') ? { word: w, points: '?', time: 0 } : { ...w };
                            wObj.finder = p.username;
                            allFoundWords.push(wObj);
                        });
                    }
                });

                const totalWords = allWords.length;
                const uniqueFound = new Set(allFoundWords.map(w => w.word.toUpperCase())).size;
                const percentage = totalWords > 0 ? Math.round((uniqueFound / totalWords) * 100) : 0;
                wordsStats.textContent = `${uniqueFound}/${totalWords} - ${percentage}%`;

                const sortedWords = allFoundWords.sort((a, b) => (a.time || 0) - (b.time || 0));
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
                        const finder = wObj.finder;
                        const isMe = finder === currentUser;
                        const isBonus = state.bonus_word && wordUpper === state.bonus_word.toUpperCase();

                        let className = 'word-item' + (isMe ? ' player-word' : ' opponent-word') + (isBonus ? ' bonus-word' : '');
                        if (highlightedFoundWord === wordUpper) className += ' finder-active';

                        const indicator = '<span class="found-indicator present">✓</span>';

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

                        if (isAtBottom || wasEmpty) {
                            requestAnimationFrame(() => { listEl.scrollTop = listEl.scrollHeight; });
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
                const me = state.players.find(p => p.username === currentUser);
                if (me && me.submitted_words) {
                    myFoundStrs = me.submitted_words.map(w => (typeof w === 'string' ? w : w.word).toUpperCase());
                }
            }

            const remainingWords = allWords.filter(w => !myFoundStrs.includes(w.toUpperCase()));
            const countsByLen = {};
            for (let i = 3; i <= 20; i++) countsByLen[i] = 0;
            remainingWords.forEach(w => {
                const len = w.length;
                if (len >= 3 && len <= 20) countsByLen[len]++;
            });

            let html = '<table id="remaining-words-table">';
            for (let i = 3; i <= 20; i++) {
                // Show all lengths from 3 to 20 as requested for scrollability
                html += `<tr><td class="len-cell">${i}LW</td><td class="count-cell">${countsByLen[i] || 0}</td></tr>`;
            }
            html += '</table>';
            remainingListEl.innerHTML = html;
        }

        // --- CLUES TAB (24H Only) ---
        const cluesListEl = document.getElementById('clues-list');
        if (cluesListEl && activeWordsTab === 'clues') {
            const myPlayer = state.players.find(p => p.username === currentUser);
            const myWords = myPlayer ? myPlayer.submitted_words : [];
            const foundSet = new Set(myWords.map(w => (typeof w === 'string' ? w : w.word).toUpperCase()));
            const unfoundWords = allWords.filter(w => !foundSet.has(w.toUpperCase()));

            if (unfoundWords.length === 0) {
                cluesListEl.innerHTML = '<p class="placeholder">All words found!</p>';
            } else {
                unfoundWords.sort((a, b) => a.length - b.length || a.localeCompare(b));
                const html = unfoundWords.map(w => {
                    const prefix = w.substring(0, 2);
                    return `<div class="clue-item">${prefix}.. (${w.length})</div>`;
                }).join('');
                cluesListEl.innerHTML = html;
            }
        }

        // --- PREVIOUS DAY TAB (24H Only) ---
        const prevListEl = document.getElementById('previous-words-list');
        if (prevListEl && activeWordsTab === 'previous') {
            const prevAll = state.previous_all_words || [];

            if (prevAll.length === 0) {
                prevListEl.innerHTML = '<p class="placeholder">No previous data.</p>';
            } else {
                // PERSONAL HISTORY: Use my restored player's previous words OR persisted history
                // Note: state.players might be empty if wiped by 24h reset!
                const myPlayer = (state.players || []).find(p => p.username === currentUser);
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

                // Render Helper
                const renderRow = (w, isFound) => {
                    const statusClass = isFound ? 'player-word' : 'missed';
                    const icon = isFound ? '✓' : '✗';
                    return `<div class="word-item ${statusClass}" data-word="${w}" style="display:flex; justify-content:space-between; cursor:pointer;">
                        <span>${w}</span>
                        <span style="opacity:0.6">${icon}</span>
                    </div>`;
                };

                let html = '';

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
                                <span style="font-weight: 700; color: #fff; font-size: 0.95rem;">${name}</span>
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

    if (state && state.game_type === 'accumulative') {
        const totalPeople = (players ? players.length : 0) + (state.spectators ? state.spectators.length : 0);
        if (headingEl) headingEl.textContent = `Players [${totalPeople}]`;
        if (findMeBtn) findMeBtn.style.display = 'block';
    } else {
        if (headingEl) headingEl.textContent = `Players`;
        if (findMeBtn) findMeBtn.style.display = 'none';
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
    const itemsToRender = sortedPlayers;

    const html = itemsToRender.map((p, index) => {
        // Override rating for Guest users
        const isGuest = p.username.startsWith('Guest_');
        const displayRating = isGuest ? 0 : p.rating;

        let ratingChange = p.rating_change ? `${p.rating_change > 0 ? ' +' : ' '}${p.rating_change}` : '0';
        if (p.joined_mid_round) ratingChange = '🛡️';
        const ratingDisplay = `${displayRating} (${ratingChange.trim()})`;
        const bonusClass = p.found_bonus_word ? ' bonus-finder' : '';
        const userClass = (p.username === currentUser) ? ' current-user' : '';
        const rank = index + 1;

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
        if (p.input_method === 'touch') inputIcon = '👆';

        // Trophy Logic
        const trophyHtml = p.has_exceptional_round ? '<span title="Exceptional Performer" style="font-size: 0.8rem; margin-left: 4px;">🏆</span>' : '';

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
                <span class="player-words-count">${p.words_count} words</span>
                <span class="player-score-val">${p.score} pts</span>
            </div>
        </div>
        `;
    }).join('');

    if (html === lastPlayersHtml) return;
    lastPlayersHtml = html;
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

function resetChat() {
    console.log('[play.js] resetting chat state');
    lastChatCount = 0;
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
        const isSystem = msg.is_system;

        // Escape HTML to prevent XSS
        const safeText = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

        if (isSystem) {
            return `
            <div class="chat-message chat-sys">
                <span class="chat-text">${safeText}</span>
            </div>`;
        }

        return `
        <div class="chat-message">
            <span class="chat-user">${username}:</span>
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
                myCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
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
});

function displayAllWords(allWords, bonusWord, targetUserWords = [], allFoundWords = [], allWordScores = {}, cswOnlyWords = []) {
    const listEl = document.getElementById('submitted-words-list');
    if (!allWords || allWords.length === 0) {
        listEl.innerHTML = '<p class="placeholder">No words found</p>';
        return;
    }

    const targetWordsUpper = targetUserWords.map(w => w.toUpperCase());
    const allFoundUpper = allFoundWords.map(w => w.toUpperCase());
    const cswOnlyUpper = (cswOnlyWords || []).map(w => w.toUpperCase());

    // Sort: Length desc, then Alpha
    // Sort: Color Priority, then Length desc, then Alpha
    // Priority: Green (Bonus) > Blue (Found) > Gold (CSW) > Black/Gray (Missed/Unfound)
    const sortedWords = [...allWords].sort((a, b) => {
        const wordA = a.toUpperCase();
        const wordB = b.toUpperCase();
        const bonusUpper = bonusWord ? bonusWord.toUpperCase() : null;

        // 0. Bonus Word (Absolute Top Priority)
        if (bonusUpper) {
            if (wordA === bonusUpper) return -1;
            if (wordB === bonusUpper) return 1;
        }

        // 1. Length (Desc)
        if (a.length !== b.length) return b.length - a.length;

        // 2. Score (Desc)
        const scoreA = allWordScores[wordA] || 0;
        const scoreB = allWordScores[wordB] || 0;
        if (scoreA !== scoreB) return scoreB - scoreA;

        // 3. Alpha
        return wordA.localeCompare(wordB);
    });

    console.log('[renderWordsList] Rendering words:', sortedWords.length);

    listEl.innerHTML = sortedWords.map(word => {
        const wordUpper = word.toUpperCase();
        const isBonus = bonusWord && wordUpper === bonusWord.toUpperCase();
        const isCSWOnly = cswOnlyUpper.includes(wordUpper);
        const isTargetFound = targetWordsUpper.includes(wordUpper);
        const isFoundByAny = allFoundUpper.includes(wordUpper);
        const points = allWordScores[word] || allWordScores[wordUpper] || 0;

        let className = 'word-item';

        // Highlighting for finder feature
        if (highlightedFoundWord === wordUpper) {
            className += ' finder-active';
        }

        // Priority 1: Bonus Word (Green) - Top Priority
        if (isBonus) {
            className += ' bonus-word';
        }
        // Priority 2: Word Found by Me/Selected Player (Blue)
        else if (isTargetFound) {
            className += ' player-word';
        }
        // Priority 3: CSW-Only Word (Yellow/Gold)
        else if (isCSWOnly) {
            className += ' csw-only';
        }
        // Priority 4: Found by others but not me (Neutral/Missed)
        else if (isFoundByAny) {
            className += ' found-by-other missed';
        }
        // Priority 5: Not found by anyone (Gray/Missed)
        else {
            className += ' unfound missed';
        }

        const indicatorClass = isFoundByAny ? 'found-indicator present' : 'found-indicator empty';
        const indicatorIcon = isFoundByAny ? '✓' : '';
        const indicator = `<span class="${indicatorClass}">${indicatorIcon}</span>`;

        return `<div class="${className}" data-word="${word}" style="display:flex; justify-content:space-between; cursor:pointer;">
            <span>${indicator}${word}</span>
            <span style="opacity:0.8">${points}</span>
        </div>`;
    }).join('');

    // Add click listeners for finder highlighting
    const wordItems = listEl.querySelectorAll('.word-item');
    wordItems.forEach(item => {
        item.addEventListener('click', () => {
            const word = item.dataset.word.toUpperCase();
            if (highlightedFoundWord === word) {
                highlightedFoundWord = null;
            } else {
                highlightedFoundWord = word;
            }
            updateGameState();
            window.fetchDefinition(item.dataset.word);
        });
    });
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
    if (sp && sp.word_count_range) {
        const bonusLen = document.getElementById('param-bonus');
        if (bonusLen) bonusLen.textContent = (sp.bonus_word_length || '?') + ' letters';

        const diff = document.getElementById('param-diff');
        if (diff) diff.textContent = sp.difficulty || '?';

        const minL = document.getElementById('param-min');
        if (minL) minL.textContent = sp.min_word_length || '?';

        const dict = document.getElementById('param-dict');
        if (dict) dict.textContent = sp.dictionary || '?';

        const wr = sp.word_count_range;
        const words = document.getElementById('param-words');
        if (words && Array.isArray(wr) && wr.length >= 2) {
            words.textContent = `${wr[0]}-${wr[1]}`;
        }

        const format = document.getElementById('param-format');
        if (format) format.textContent = sp.board_format || '?';
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

    if (!cachedTimerValueEl) cachedTimerValueEl = document.getElementById('timer-value');
    if (!cachedBoardPanelEl) cachedBoardPanelEl = document.querySelector('.board-panel');

    const now = Date.now() / 1000;
    const remaining = Math.max(0, localEndTime - now);
    const seconds = Math.ceil(remaining);

    // Format determination
    let is24h = false;
    if (window.lastGameState && window.lastGameState.time_limit >= 120) {
        is24h = true;
    }

    let display;
    if (is24h) {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        display = `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        display = `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    if (cachedTimerValueEl) {
        if (cachedTimerValueEl.textContent !== display) {
            cachedTimerValueEl.textContent = display;
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

function renderBoard(board, grayed) {
    if (!board || board.length === 0) return;
    const boardEl = document.getElementById('game-board');
    if (!boardEl) return; // play page might not be active

    // Reset container style
    const boardJSON = JSON.stringify(board);
    if (boardJSON === lastRenderedBoardJSON && grayed === lastRenderedGrayed && isBoardRotated === lastRenderedRotation && boardEl.classList.contains('game-board')) {
        reapplyBoardHighlights(); // Still update highlights if board didn't change!
        return;
    }

    lastRenderedBoardJSON = boardJSON;
    lastRenderedGrayed = grayed;
    lastRenderedRotation = isBoardRotated;

    boardEl.className = 'game-board';
    // Clear styles set by renderSplitNotepads
    boardEl.style.display = '';
    boardEl.style.gap = '';
    boardEl.style.overflowY = '';
    boardEl.style.alignItems = '';

    const rows = board.length;
    const cols = board[0].length;

    boardEl.style.gridTemplateColumns = `repeat(${cols}, var(--cell-size, 60px))`;
    boardEl.style.gridTemplateRows = `repeat(${rows}, var(--cell-size, 60px))`;
    boardEl.innerHTML = '';

    // Handle Rotation: 180 degrees flip
    if (isBoardRotated) {
        // Bottom-up, Right-to-left
        for (let r = rows - 1; r >= 0; r--) {
            for (let c = cols - 1; c >= 0; c--) {
                const cell = createBoardCell(r, c, board[r][c], grayed);
                boardEl.appendChild(cell);
            }
        }
    } else {
        // Normal: Top-down, Left-to-right
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const cell = createBoardCell(r, c, board[r][c], grayed);
                boardEl.appendChild(cell);
            }
        }
    }

    // Check for overflow after render
    setTimeout(checkBoardOverflow, 50);

    // Reapply Highlights that were wiped by innerHTML = ''
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
    // Fallbacks
    if (!cols) cols = 8;
    if (!rows) rows = 8; // Default is usually square-ish if unknown

    // Condition: Is this strictly a 6x8 board? (Or 8x6)
    // User requested changes ONLY for 6x8.
    const isSixByEight = (cols === 6 && rows === 8) || (cols === 8 && rows === 6);
    console.log(`[Layout] Board: ${cols}x${rows}. Is 6x8 target? ${isSixByEight}`);

    // Get Cell Size (Cache this based on window size to avoid repeated getComputedStyle)
    if (!window.cachedCellSize || window.lastW !== window.innerWidth) {
        const computedStyle = getComputedStyle(document.documentElement);
        const cellSizeVar = computedStyle.getPropertyValue('--cell-size').trim();
        window.cachedCellSize = parseInt(cellSizeVar) || 60;
        window.lastW = window.innerWidth;
    }
    const cellSize = window.cachedCellSize;

    // 2. Calculate Required Width for Board
    // Width = (Cols * Size) + Gap + Padding + Scrollbar
    const boardGap = 4 * (cols - 1);
    const boardPadding = 40; // 20px * 2 (CSS matches this)
    let requiredBoardWidth = (cols * cellSize) + boardGap + boardPadding;

    // Add Scrollbar Width if present
    const scrollbarWidth = boardPanel.offsetWidth - boardPanel.clientWidth;
    requiredBoardWidth += scrollbarWidth;

    // 3. Calculate Available Space
    const windowWidth = window.innerWidth;
    // Safety Margin: 40px padding + 40px gaps + 40px buffer = 120px Total.
    const safetyMargin = 120;

    // The key difference: We start with Window and subtract Board
    const availableForPanels = windowWidth - requiredBoardWidth - safetyMargin;

    // 4. Distribute Remaining Space
    let newLeft = 0;
    let newRight = 0;

    if (availableForPanels > 0) {
        // Calculate proportional shares
        const calculatedLeft = Math.floor(availableForPanels * 0.52);
        const calculatedRight = Math.floor(availableForPanels * 0.48);

        // CONDITIONAL CAPS
        let maxLeft, maxRight;

        if (isSixByEight) {
            // "Apply the size changes" -> Smaller
            maxLeft = 260;
            maxRight = 240;
        } else {
            // "Keep what they previously were" -> Standard/Larger
            // Default CSS implies 340/320 base.
            maxLeft = 340;
            maxRight = 320;
        }

        newLeft = Math.min(calculatedLeft, maxLeft);
        newRight = Math.min(calculatedRight, maxRight);
    } else {
        newLeft = 0;
        newRight = 0;
    }

    // 5. Apply
    playPage.style.setProperty('--left-panel-w', `${newLeft}px`);
    playPage.style.setProperty('--right-panel-w', `${newRight}px`);

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

    // Maintain vertical scroll class for potential other uses
    if (boardPanel.scrollHeight > boardPanel.clientHeight) {
        playPage.classList.add('has-vertical-scroll');
    } else {
        playPage.classList.remove('has-vertical-scroll');
    }
}

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

// Helper to create a board cell
function createBoardCell(r, c, letter, grayed) {
    const cell = document.createElement('div');
    cell.className = 'board-cell' + (grayed ? ' grayed' : '');
    cell.textContent = letter === 'Q' ? 'QU' : letter;
    cell.dataset.row = r;
    cell.dataset.col = c;
    cell.dataset.letter = letter; // Original letter
    return cell;
}

/**
 * Finds if a word can be formed on the board and returns the path of coordinates.
 * Supports the "Q" tile representing "QU".
 */
function findWordPathOnBoard(word, board) {
    if (!word || !board) return null;
    const rows = board.length;
    if (rows === 0) return null;
    const cols = board[0].length;
    const upperWord = word.toUpperCase();

    function dfs(r, c, index, currentPath, visited) {
        if (index >= upperWord.length) return currentPath;

        if (r < 0 || r >= rows || c < 0 || c >= cols) return null;
        if (visited.has(`${r},${c}`)) return null;

        const cellChar = board[r][c].toUpperCase();
        let matchLength = 0;

        if (cellChar === 'Q') {
            // "Q" tile matches "QU" in the word, or just "Q" if it's the only thing typed
            if (upperWord.substring(index, index + 2) === 'QU') {
                matchLength = 2;
            } else if (upperWord[index] === 'Q') {
                matchLength = 1;
            } else {
                return null;
            }
        } else {
            if (upperWord[index] === cellChar) {
                matchLength = 1;
            } else {
                return null;
            }
        }

        const newVisited = new Set(visited);
        newVisited.add(`${r},${c}`);
        const newPath = [...currentPath, { r, c }];

        const nextIndex = index + matchLength;
        if (nextIndex >= upperWord.length) return newPath;

        // Try directions
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const result = dfs(r + dr, c + dc, nextIndex, newPath, newVisited);
                if (result) return result;
            }
        }
        return null;
    }

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const path = dfs(r, c, 0, [], new Set());
            if (path) return path;
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
        const isEnabled = window.userSettings && window.userSettings.highlight_mouse !== false;
        if (isEnabled) {
            mouseState.selectedPath.forEach((p, index) => {
                const cell = document.querySelector(`.board-cell[data-row="${p.row}"][data-col="${p.col}"]`);
                if (cell) {
                    cell.classList.add('selected');
                    if (index === mouseState.selectedPath.length - 1) {
                        cell.classList.add('current');
                    }
                }
            });
        }
    }

    // 2. Reapply typing highlights (input box)
    const wordInputEl = document.getElementById('word-input');
    if (wordInputEl && wordInputEl.value.trim()) {
        const isEnabled = window.userSettings && window.userSettings.highlight_typing !== false;
        if (isEnabled) {
            const word = wordInputEl.value.trim();
            const path = findWordPathOnBoard(word, board);
            if (path) {
                path.forEach(coord => {
                    const cell = document.querySelector(`.board-cell[data-row="${coord.r}"][data-col="${coord.c}"]`);
                    if (cell) cell.classList.add('typing-highlight');
                });
            }
        }
    }

    // 3. Reapply review highlights (All Words / Finder list)
    if (typeof highlightedFoundWord !== 'undefined' && highlightedFoundWord) {
        const path = findWordPathOnBoard(highlightedFoundWord, board);
        if (path) {
            // Check if we need to animate (new selection) or just show (board refresh)
            const isNewSelection = window._lastAnimatedReviewWord !== highlightedFoundWord;

            path.forEach((coord, index) => {
                const cell = document.querySelector(`.board-cell[data-row="${coord.r}"][data-col="${coord.c}"]`);
                if (cell) {
                    if (isNewSelection) {
                        // Sequential tracing effect (similar to replay)
                        setTimeout(() => {
                            // Ensure the word is still the one we want to highlight
                            if (highlightedFoundWord === window._lastAnimatedReviewWord) {
                                cell.classList.add('review-highlight');
                            }
                        }, index * 60); // 60ms delay per letter
                    } else {
                        // Instant display for static refreshes
                        cell.classList.add('review-highlight');
                    }
                }
            });

            window._lastAnimatedReviewWord = highlightedFoundWord;
        }
    } else {
        window._lastAnimatedReviewWord = null;
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
                if (pts < 0) {
                    row.className += ' penalty-word';
                    row.style.color = '#ff3333';
                    row.style.fontWeight = 'bold';
                }
                row.dataset.word = w;
                row.onclick = () => window.fetchDefinition(w); // Direct handler
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
function showSpinnerOverlay(spinnerParams, players = []) {
    if (!spinnerParams || !spinnerParams.word_count_range) {
        console.warn('[play.js] showSpinnerOverlay called with incomplete spinnerParams:', spinnerParams);
        if (!spinnerParams) return;
    }

    // WINNER LOGIC
    const winnerAnnounceEl = document.getElementById('spinner-winner-announcement');
    const winnerTextEl = document.getElementById('spinner-winner-text');

    if (winnerAnnounceEl && winnerTextEl) {
        // Find players with score > 0
        const activePlayers = (players || []).filter(p => p.score > 0);

        if (activePlayers.length === 0) {
            winnerAnnounceEl.classList.add('hidden');
        } else {
            // Find max score
            const maxScore = Math.max(...activePlayers.map(p => p.score));
            const winners = activePlayers.filter(p => p.score === maxScore).map(p => p.username);

            if (winners.length === 1) {
                winnerTextEl.textContent = `CONGRATULATIONS to ${winners[0]} for scoring 1st place with ${maxScore} points!`;
            } else {
                // Formatting for multiple winners: "User1, User2 and User3"
                let winnerStr = winners.slice(0, -1).join(', ') + ' and ' + winners.slice(-1);
                winnerTextEl.textContent = `CONGRATULATIONS to ${winnerStr} for scoring 1st place with ${maxScore} points!`;
            }

            winnerAnnounceEl.classList.remove('hidden');
        }
    }

    // Defensive updates
    const safeSetText = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };

    safeSetText('spinner-bonus-length', (spinnerParams.bonus_word_length || '?') + ' letters');
    safeSetText('spinner-min-length', spinnerParams.min_word_length || '?');
    safeSetText('spinner-difficulty', spinnerParams.difficulty || '?');

    const wr = spinnerParams.word_count_range;
    if (wr && Array.isArray(wr) && wr.length >= 2) {
        safeSetText('spinner-word-count', `${wr[0]}-${wr[1]}`);
    } else {
        safeSetText('spinner-word-count', '?-?');
    }

    safeSetText('spinner-dictionary', spinnerParams.dictionary || 'Unknown');
    safeSetText('spinner-format', spinnerParams.board_format || 'Standard');

    const overlay = document.getElementById('spinner-overlay');
    if (overlay) overlay.classList.remove('hidden');
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

    // Real-time highlighting and "word declaration" while typing
    wordInputEl.addEventListener('input', () => {
        const word = wordInputEl.value.trim();
        const board = window.lastGameState ? window.lastGameState.board : null;

        // Dedicated "Word Declaration" area in bottom right
        const defWordEl = document.getElementById('definition-word');
        const defHeaderEl = document.getElementById('definition-header');
        const defContentEl = document.getElementById('definition-content');
        if (defWordEl && defHeaderEl) {
            if (word) {
                defWordEl.textContent = word.toUpperCase();
                defHeaderEl.style.display = 'block';
                // Only show "Typing..." if no existing definition is being looked at
                if (defContentEl && (defContentEl.querySelector('.placeholder') || !defContentEl.innerHTML.trim())) {
                    defContentEl.innerHTML = '<p class="placeholder" style="font-size: 0.8rem; margin-top: 5px;">Typing...</p>';
                }
            } else {
                // If input cleared, and show placeholder if no word selected
                defHeaderEl.style.display = 'none';
                if (defContentEl) {
                    defContentEl.innerHTML = '<p class="placeholder">Select a word to see its definition</p>';
                }
            }
        }

        const isEnabled = window.userSettings && window.userSettings.highlight_typing !== false;
        if (!isEnabled) {
            document.querySelectorAll('.board-cell.typing-highlight').forEach(c => c.classList.remove('typing-highlight'));
            return;
        }

        document.querySelectorAll('.board-cell.typing-highlight').forEach(c => c.classList.remove('typing-highlight'));
        if (!word || !board) return;

        const path = findWordPathOnBoard(word, board);
        if (path) {
            path.forEach(coord => {
                const cell = document.querySelector(`.board-cell[data-row="${coord.r}"][data-col="${coord.c}"]`);
                if (cell) cell.classList.add('typing-highlight');
            });
        }
    });
}

async function submitWord(wordParam = null) {
    const input = document.getElementById('word-input');
    const word = wordParam ? wordParam.toUpperCase() : input.value.trim().toUpperCase();
    const roomId = getCurrentRoomId();

    if (!word || !roomId) return;

    try {
        const response = await fetch(`/room/${roomId}/submit_word`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word, input_method: currentInputMethod })
        });
        const data = await response.json();

        // Show validation feedback
        showValidationFeedback(data.message || (data.success ? 'Valid Word' : 'Invalid Word'), data.success);

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
                if (data.points < 0) className += ' penalty-word';

                const html = `<div class="${className}" style="display:flex; justify-content:space-between; animation: slideIn 0.3s ease;">
                    <span>${data.word}</span>
                    <span style="opacity:0.8">${data.points}</span>
                </div>`;

                // Prepend to top (since we sort by newest)
                listEl.insertAdjacentHTML('afterbegin', html);

                // Update total score and re-render players list
                const me = currentState.players.find(p => p.username === currentUser);
                if (me) {
                    me.score = data.new_score;
                    // Re-render the left panel players list with new score/rank
                    renderPlayers(currentState.players, currentUser, currentState);
                }

                // Update stats text optimistically (only for genuine dictionary words)
                if (wordsStats && data.points > 0) {
                    const parts = wordsStats.textContent.match(/(\d+)\/(\d+) - (\d+)%/);
                    if (parts) {
                        let found = parseInt(parts[1]);
                        const total = parseInt(parts[2]);
                        found++;
                        const percent = total > 0 ? Math.round((found / total) * 100) : 0;
                        wordsStats.textContent = `${found}/${total} - ${percent}%`;
                    }
                    else if (currentState.all_words) {
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
    } catch (error) {
        console.error('Error submitting word:', error);
        showValidationFeedback('Submission Error', false);
    }
    input.value = '';
    // Clear typing highlights and declaration after submission
    document.querySelectorAll('.board-cell.typing-highlight').forEach(c => c.classList.remove('typing-highlight'));
    const defHeader = document.getElementById('definition-header');
    if (defHeader) defHeader.style.display = 'none';
    const defContent = document.getElementById('definition-content');
    if (defContent) defContent.innerHTML = '<p class="placeholder">Select a word to see its definition</p>';
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

async function leaveCurrentRoom() {
    const roomId = getCurrentRoomId();
    if (!roomId) return;
    try { await fetch(`/api/room/${roomId}/leave`, { method: 'POST' }); } catch (e) { }
    window.currentRoomId = null;
    const playBtn = document.getElementById('play-btn');
    if (playBtn) {
        playBtn.disabled = true;
        playBtn.classList.remove('active');
        playBtn.title = "Join a room to play.";
    }
    if (window.updateManualToolState) window.updateManualToolState();
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

const rotateBtnEl = document.getElementById('rotate-board-btn');
if (rotateBtnEl) {
    rotateBtnEl.addEventListener('click', () => {
        isBoardRotated = !isBoardRotated;
        console.log('[play.js] Board rotation toggled. Rotated:', isBoardRotated);
        // Force re-render if we have state
        if (window.lastGameState && window.lastGameState.board) {
            renderBoard(window.lastGameState.board, window.lastGameState.state !== 'active');
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

        if (data.definition) {
            // No longer injecting word span here as it's in the header
            defContent.innerHTML = `<span class="definition-text">${data.definition}</span>`;
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
