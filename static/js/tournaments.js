
window.initTournamentsPage = function () {
    console.log('Initializing Tournaments Page');
    fetchTournamentStatus();

    // Set up polling while on the page
    if (!window.tournamentPollInterval) {
        window.tournamentPollInterval = setInterval(() => {
            const page = document.getElementById('page-tournaments');
            if (page && page.classList.contains('active')) {
                fetchTournamentStatus();
            }
        }, 30000); // 30s poll
    }
};

let currentTournamentState = null;

async function fetchTournamentStatus() {
    try {
        const response = await fetch('/api/tournament/status');
        if (!response.ok) throw new Error("Failed to fetch status");

        const data = await response.json();
        currentTournamentState = data;
        renderTournament(data);
        updateNavHighlight(data.user_status?.has_turn);
    } catch (e) {
        console.error("Error loading tournament:", e);
        const actionArea = document.getElementById('tournament-action-area');
        if (actionArea) {
            actionArea.innerHTML = `<p style="color:var(--accent-color)">Error loading tournament info. Please refresh.</p>`;
        }
    }
}

function updateNavHighlight(hasTurn) {
    const btn = document.getElementById('nav-tournaments-btn');
    if (btn) {
        if (hasTurn) {
            btn.classList.add('has-turn');
        } else {
            btn.classList.remove('has-turn');
        }
    }
}

function renderTournament(data) {
    // Guest Handling
    const guestBlock = document.getElementById('tournament-guest-block');
    const mainContent = document.getElementById('tournament-main-content');

    if (data.is_guest) {
        guestBlock.classList.remove('hidden');
        mainContent.classList.add('hidden');
        return;
    } else {
        guestBlock.classList.add('hidden');
        mainContent.classList.remove('hidden');
    }

    const status = data.status || 'none';
    const params = data.parameters || {};
    const history = data.history || [];
    const userStatus = data.user_status || {};

    // 1. Status Badge
    const badge = document.getElementById('tournament-status-badge');
    let label = status.toUpperCase();
    if (status === 'signup') label = 'REGISTRATION';
    if (status === 'active') label = 'LIVE';
    if (badge) {
        badge.textContent = label;
        badge.className = `tournament-status-badge status-${status}`;
    }

    // 2. Countdown
    updateCountdown(data);

    // 3. Parameters (Spinner Set)
    const pContainer = document.getElementById('tournament-params-container');
    if (pContainer) {
        if (params.board_dimensions) {
            const wordRange = params.word_count_range || [0, 0];
            const wordsStr = wordRange[1] > 50000 ? `${wordRange[0]}+` : `${wordRange[0]}-${wordRange[1]}`;

            pContainer.innerHTML = `
                <div class="param-item">
                    <span class="param-label">Board Size</span>
                    <span class="param-value">${params.board_dimensions}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Time Limit</span>
                    <span class="param-value">${params.time_limit}s</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Difficulty</span>
                    <span class="param-value">${params.difficulty || 'Medium'}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Target Words</span>
                    <span class="param-value">${wordsStr}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Board Format</span>
                    <span class="param-value">${params.board_format}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Min Word Length</span>
                    <span class="param-value">${params.min_word_length} Letters</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Dictionary</span>
                    <span class="param-value">${params.dictionary}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Bonus Word</span>
                    <span class="param-value">${params.bonus_word_length || '?'} Letters</span>
                </div>
            `;
        } else {
            pContainer.innerHTML = '<p class="placeholder">Parameters pending...</p>';
        }
    }

    // 4. Action Area
    const actionArea = document.getElementById('tournament-action-area');
    const actionTitle = document.getElementById('action-title');

    if (actionArea) {
        actionArea.innerHTML = '';
        if (status === 'signup') {
            if (actionTitle) actionTitle.textContent = "Registration Open";
            renderSignupState(actionArea, data, userStatus);
        } else if (status === 'active') {
            if (actionTitle) actionTitle.textContent = "Tournament Active";
            renderActiveState(actionArea, data, userStatus);
        } else if (status === 'completed') {
            if (actionTitle) actionTitle.textContent = "Tournament Ended";
            renderCompletedState(actionArea, data);
        }
    }

    // 4.5 Leaderboard (Active and Completed)
    const lbCard = document.getElementById('tournament-leaderboard-card');
    const lbList = document.getElementById('tournament-leaderboard-list');
    if (lbCard && lbList) {
        if (data.round_scores && data.round_scores.length > 0) {
            lbCard.classList.remove('hidden');
            lbList.innerHTML = data.round_scores.map((s, idx) => {
                const isMe = s.username === window.currentUser;
                const highlight = isMe ? 'border: 1px solid rgba(46, 204, 113, 0.3); background: rgba(46, 204, 113, 0.05);' : '';

                return `
                    <div class="t-leaderboard-item" style="${highlight}">
                        <div class="user-info">
                            <span class="rank">#${idx + 1}</span>
                            <span class="username">${s.username} ${isMe ? '(You)' : ''}</span>
                        </div>
                        <div class="score-group">
                            <span class="score">${s.score} <small style="font-size:0.6rem; opacity:0.6;">PTS</small></span>
                            <div style="display:flex; gap:8px;">
                                <button class="replay-btn-small" title="Watch Walkthrough" onclick="watchTournamentReplay(${data.id}, ${data.current_round}, '${s.username}', false)">▶ Walkthrough</button>
                                <button class="replay-btn-small" title="View Snapshot" onclick="watchTournamentReplay(${data.id}, ${data.current_round}, '${s.username}', true)">📷 Snapshot</button>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            lbCard.classList.add('hidden');
        }
    }

    // 5. History
    const hBody = document.getElementById('tournament-history-body');
    if (hBody) {
        if (history.length === 0) {
            hBody.innerHTML = '<tr><td colspan="4" style="text-align:center; opacity:0.5; padding:40px;">No Hall of Fame records yet. Be the first champion!</td></tr>';
        } else {
            hBody.innerHTML = history.map(h => {
                const date = new Date(h.completed_at * 1000).toLocaleDateString();
                return `
                    <tr>
                        <td>${date}</td>
                        <td style="font-weight:700; color:var(--accent-color);">${h.username}</td>
                        <td>Championship Edition</td>
                        <td>Winner (Rank 1)</td>
                    </tr>
                `;
            }).join('');
        }
    }
}

function renderSignupState(container, data, userStatus) {
    if (userStatus.status !== 'not_joined') {
        container.innerHTML = `
            <div style="font-size:1.4rem; color:#2ecc71; margin-bottom:15px; font-weight:700;">✅ SUCCESSFULLY ENROLLED</div>
            <p style="opacity:0.8;">You are registered for this event. Prepare yourself! The tournament begins soon.</p>
        `;
    } else {
        const btn = document.createElement('button');
        btn.className = 'primary-action';
        btn.style.width = '100%';
        btn.style.padding = '18px';
        btn.style.fontSize = '1.1rem';
        btn.textContent = 'SIGN UP FOR TOURNAMENT';
        btn.onclick = joinTournament;
        container.appendChild(btn);

        const info = document.createElement('p');
        info.style.marginTop = '20px';
        info.style.opacity = '0.6';
        info.innerHTML = `Start Date: ${new Date(data.start_date * 1000).toLocaleString()}`;
        container.appendChild(info);
    }
}

function renderActiveState(container, data, userStatus) {
    if (userStatus.status === 'eliminated') {
        container.innerHTML = `
            <div style="font-size:1.5rem; color:#e74c3c; font-weight:700; margin-bottom:10px;">ELIMINATED</div>
            <p style="opacity:0.8;">You fought well, but have been eliminated from this tournament. Keep practicing for the next one!</p>
        `;
        return;
    }

    if (userStatus.status === 'not_joined') {
        container.innerHTML = `
            <div style="font-size:1.2rem; opacity:0.7; margin-bottom:10px;">Ongoing Championship</div>
            <p>Registration is closed for this event. Wait for the next signup period to compete.</p>
        `;
        return;
    }

    // Active User
    container.innerHTML = `<h2 style="margin-bottom:20px; text-align:left;">Round ${data.current_round}</h2>`;

    if (userStatus.has_turn) {
        const btn = document.createElement('button');
        btn.className = 'primary-action';
        btn.style.width = '100%';
        btn.style.padding = '20px';
        btn.style.fontSize = '1.2rem';
        btn.style.background = '#2ecc71';
        btn.textContent = 'PLAY YOUR TURN';
        btn.onclick = () => {
            // We'll use a special flag or route to launch the tournament game
            launchTournamentGame(data.id, data.current_round);
        };
        container.appendChild(btn);

        const note = document.createElement('p');
        note.style.marginTop = '15px';
        note.style.color = '#2ecc71';
        note.style.fontWeight = '700';
        note.textContent = "CRITICAL: It is your turn! Do not miss the deadline.";
        container.appendChild(note);
    } else {
        container.innerHTML += `
            <div style="font-size:1.2rem; color:#f39c12; font-weight:700; margin-bottom:10px;">SCORE SUBMITTED</div>
            <p style="opacity:0.8;">You have completed your turn for this round. Stay tuned! Results will be processed when the round ends.</p>
        `;
    }
}

function renderCompletedState(container, data) {
    container.innerHTML = `
        <h2 style="color:#9b59b6; margin-bottom:15px; text-align:left;">Tournament Finalized</h2>
        <p style="opacity:0.8;">The champions have been crowned! The next tournament signup period will begin shortly.</p>
    `;
}

async function joinTournament() {
    try {
        const btn = document.querySelector('.primary-action');
        if (btn) btn.disabled = true;

        const response = await fetch('/api/tournament/join', { method: 'POST' });
        const res = await response.json();

        if (res.success) {
            fetchTournamentStatus(); // Refresh
        } else {
            alert(res.error || "Failed to join tournament");
            if (btn) btn.disabled = false;
        }
    } catch (e) {
        console.error("Join error:", e);
        alert("Error connecting to server");
    }
}

function launchTournamentGame(tid, round) {
    // Show a loading overlay or just switch to play page with tournament mode
    const playBtn = document.getElementById('play-btn');
    if (playBtn) {
        // Set a session variable or global flag to denote tournament play
        localStorage.setItem('tournament_play_active', JSON.stringify({ tid, round }));
        window.location.href = '#page-play';
        // In app.js or play.js, we should check this flag and load tournament state
        // For simplicity, we can trigger a page reload or a specific init function.
        location.reload();
    }
}

let countdownInterval;
function updateCountdown(data) {
    if (countdownInterval) clearInterval(countdownInterval);

    let targetTime = 0;
    if (data.status === 'signup') targetTime = data.start_date;
    else if (data.status === 'active') targetTime = data.round_end_time;
    else if (data.status === 'completed') targetTime = data.completed_at + 604800; // 1 week

    if (!targetTime) return;

    const tick = () => {
        const current = Date.now() / 1000;
        let diff = targetTime - current;

        if (diff < 0) {
            diff = 0;
            if (currentTournamentState && currentTournamentState.status !== 'completed') {
                // Potential state change triggered by time
                fetchTournamentStatus();
            }
        }

        const d = Math.floor(diff / 86400);
        const h = Math.floor((diff % 86400) / 3600);
        const m = Math.floor((diff % 3600) / 60);
        const s = Math.floor(diff % 60);

        const str = diff > 0 ? `${d}d ${h}h ${m}m ${s}s` : "00:00:00";
        const el = document.getElementById('tournament-countdown');
        if (el) el.textContent = str;
    };

    tick();
    countdownInterval = setInterval(tick, 1000);
}
window.watchTournamentReplay = function (tid, roundNum, targetUsername, isSnapshot) {
    if (!currentTournamentState || !currentTournamentState.round_scores) return;

    const scoreData = currentTournamentState.round_scores.find(s => s.username === targetUsername);
    if (!scoreData) {
        alert("Replay data not found.");
        return;
    }

    // Prepare a mock "round" object that watchRoundHistory can use
    const mockRound = {
        room_id: `tournament_${tid}`,
        round_number: roundNum,
        board: JSON.parse(scoreData.board_data),
        words: scoreData.submitted_words, // Already an array of objects
        total_score: scoreData.score,
        round_duration: currentTournamentState.parameters.time_limit || 60,
        round_start_time: scoreData.round_start_time || (Date.now() / 1000 - 60),
        timestamp: scoreData.submitted_at * 1000,
        game_type: 'tournament'
    };

    // Temporarily inject into window.lastRenderedRounds so tools.js finds it
    if (!window.lastRenderedRounds) window.lastRenderedRounds = [];

    // Remove old tournament entries for this tid/round/user to avoid bloat
    window.lastRenderedRounds = window.lastRenderedRounds.filter(r =>
        !(r.room_id === mockRound.room_id && r.round_number === mockRound.round_number && r.username === targetUsername)
    );

    mockRound.username = targetUsername; // tag it
    window.lastTournamentReplay = mockRound;

    if (window.watchRoundHistory) {
        window.watchRoundHistory(`tournament_${tid}`, roundNum, isSnapshot);
    }
};
