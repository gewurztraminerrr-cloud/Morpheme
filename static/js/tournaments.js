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

// Also trigger if we are directly on /tournaments (for backward compatibility if needed)
if (window.location.pathname === '/tournaments') {
    document.addEventListener('DOMContentLoaded', window.initTournamentsPage);
}

let currentTournamentState = null;

async function fetchTournamentStatus() {
    try {
        const response = await fetch('/api/tournament/status');
        if (!response.ok) throw new Error("Failed to fetch status");

        const data = await response.json();
        currentTournamentState = data;
        renderTournament(data);
    } catch (e) {
        console.error("Error loading tournament:", e);
        document.getElementById('action-area').innerHTML = `<p style="color:red">Error loading tournament info. Please refresh.</p>`;
    }
}

function renderTournament(data) {
    const status = data.status || 'none';
    const params = data.parameters || {};
    const history = data.history || [];
    const userStatus = data.user_status || {};

    // 1. Status Badge
    const badge = document.getElementById('status-badge');
    let label = status.toUpperCase();
    if (status === 'signup') label = 'REGISTRATION';
    if (status === 'active') label = 'LIVE';
    badge.textContent = label;
    badge.className = `tournament-status-badge status-${status}`;

    // 2. Countdown
    updateCountdown(data);

    // 3. Parameters (Spinner Set)
    const pContainer = document.getElementById('params-container');
    if (params.board_dimensions) {
        const wordRange = params.word_count_range || [0, 0];
        const wordsStr = wordRange[1] > 50000 ? `${wordRange[0]}+` : `${wordRange[0]}-${wordRange[1]}`;

        pContainer.innerHTML = `
            <div class="param-card">
                <div class="param-label">Board Size</div>
                <div class="param-value">${params.board_dimensions}</div>
            </div>
            <div class="param-card">
                <div class="param-label">Time Limit</div>
                <div class="param-value">${params.time_limit}s/game</div>
            </div>
            <div class="param-card">
                <div class="param-label">Difficulty</div>
                <div class="param-value">${params.difficulty || 'Medium'}</div>
            </div>
            <div class="param-card">
                <div class="param-label">Target Words</div>
                <div class="param-value">${wordsStr}</div>
            </div>
            <div class="param-card">
                <div class="param-label">Format</div>
                <div class="param-value">${params.board_format}</div>
            </div>
            <div class="param-card">
                <div class="param-label">Min Word</div>
                <div class="param-value">${params.min_word_length}</div>
            </div>
            <div class="param-card">
                <div class="param-label">Dictionary</div>
                <div class="param-value">${params.dictionary}</div>
            </div>
            <div class="param-card">
                <div class="param-label">Bonus Word</div>
                <div class="param-value">${params.bonus_word_length || '?'} chars</div>
            </div>
        `;
    } else {
        pContainer.innerHTML = '<p style="text-align:center;width:100%;opacity:0.5;">Parameters pending...</p>';
    }

    // 4. Action Area
    const actionArea = document.getElementById('action-area');
    actionArea.innerHTML = ''; // Clear

    if (status === 'signup') {
        renderSignupState(actionArea, data, userStatus);
    } else if (status === 'active') {
        renderActiveState(actionArea, data, userStatus);
    } else if (status === 'completed') {
        renderCompletedState(actionArea, data);
    }

    // 5. History
    const hBody = document.getElementById('history-body');
    if (history.length === 0) {
        hBody.innerHTML = '<tr><td colspan="3" style="text-align:center;opacity:0.5;padding:20px;">No history yet. Be the first!</td></tr>';
    } else {
        hBody.innerHTML = history.map(h => {
            // Date formatting
            const date = new Date(h.completed_at * 1000).toLocaleDateString();
            return `
                <tr>
                    <td>${date}</td>
                    <td>${h.username}</td>
                    <td>#${h.final_rank}</td>
                </tr>
             `;
        }).join('');
    }
}

function renderSignupState(container, data, userStatus) {
    if (userStatus.status !== 'not_joined') {
        container.innerHTML = `
            <div style="font-size:1.2rem; color:#2ecc71; margin-bottom:10px;">✅ You are signed up!</div>
            <p>Tournament starts in <span id="start-limit-timer">...</span></p>
        `;
    } else {
        const btn = document.createElement('button');
        btn.className = 'join-btn';
        btn.textContent = 'Sign Up for Tournament';
        btn.onclick = joinTournament;
        container.appendChild(btn);

        const info = document.createElement('p');
        info.style.marginTop = '15px';
        info.innerHTML = `Starts on: <strong>${new Date(data.start_date * 1000).toLocaleString()}</strong>`;
        container.appendChild(info);
    }
}

function renderActiveState(container, data, userStatus) {
    if (userStatus.status === 'eliminated') {
        container.innerHTML = `
            <div style="font-size:1.5rem; color:#e74c3c;">🚫 Eliminated</div>
            <p>Better luck next time! You can verify past rounds in the history once completed.</p>
        `;
        return;
    }

    if (userStatus.status === 'not_joined') {
        container.innerHTML = `
            <div style="font-size:1.2rem; opacity:0.7;">Tournament in Progress</div>
            <p>You are not participating in this one.</p>
        `;
        return;
    }

    // Active User
    const round = userStatus.round || data.current_round;

    container.innerHTML = `<h2 style="margin-bottom:20px;">Round ${round}</h2>`;

    if (userStatus.has_turn) {
        const btn = document.createElement('button');
        btn.className = 'play-turn-btn'; // Lime green styles
        btn.textContent = 'PLAY YOUR TURN';
        btn.onclick = () => {
            // Redirect to the Game Page specifically for this tournament turn
            window.location.href = `/tournament/game`;
        };
        container.appendChild(btn);

        const note = document.createElement('p');
        note.style.marginTop = '15px';
        note.style.color = '#2ecc71';
        note.textContent = "It's your turn! Good luck.";
        container.appendChild(note);
    } else {
        container.innerHTML += `
            <div style="font-size:1.2rem; color:#f39c12;">⏳ Waiting for Round Results...</div>
            <p>You have submitted your score. Results will be processed when the round ends.</p>
            <p style="margin-top:10px; font-size:0.9rem; opacity:0.6;">Round ends in: <span id="round-limit-timer">...</span></p>
        `;
    }
}

function renderCompletedState(container, data) {
    container.innerHTML = `
        <h2 style="color:#9b59b6; margin-bottom:15px;">Tournament Completed</h2>
        <p>The next tournament is being prepared.</p>
        <p style="margin-top:10px;">Signups open in: <span id="cooldown-limit-timer">...</span></p>
    `;
}

async function joinTournament() {
    try {
        const btn = document.querySelector('.join-btn');
        if (btn) btn.disabled = true;

        const response = await fetch('/api/tournament/join', { method: 'POST' });
        const res = await response.json();

        if (res.success) {
            fetchTournamentStatus(); // Refresh
        } else {
            alert(res.message || "Failed to join");
            if (btn) btn.disabled = false;
        }
    } catch (e) {
        console.error("Join error:", e);
        alert("Error joining tournament");
    }
}

let countdownInterval;
function updateCountdown(data) {
    if (countdownInterval) clearInterval(countdownInterval);

    const now = Date.now() / 1000;
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
            // Optionally refresh state if timer hits 0
            if (diff <= 0) fetchTournamentStatus();
        }

        const d = Math.floor(diff / 86400);
        const h = Math.floor((diff % 86400) / 3600);
        const m = Math.floor((diff % 3600) / 60);
        const s = Math.floor(diff % 60);

        const str = `${d}d ${h}h ${m}m ${s}s`;

        // Update various potential timer elements
        const els = [
            document.getElementById('countdown-timer'),
            document.getElementById('start-limit-timer'),
            document.getElementById('round-limit-timer'),
            document.getElementById('cooldown-limit-timer')
        ];

        els.forEach(el => {
            if (el) el.textContent = str;
        });
    };

    tick();
    countdownInterval = setInterval(tick, 1000);
}
