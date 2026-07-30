
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
        const response = await fetch('/api/tournament/status', { cache: 'no-store' });
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
            const wordRange = params.word_count_range || '...';
            let wordsStr = wordRange;
            if (Array.isArray(wordRange)) {
                wordsStr = wordRange[1] > 50000 ? `${wordRange[0]}+` : `${wordRange[0]}-${wordRange[1]}`;
            }

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
                    <span class="param-value">${params.difficulty || 'Medium'}${params.uniqueness_ratio ? ` (${Math.round(params.uniqueness_ratio * 100)}%)` : ''}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Word Count</span>
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
                    <span class="param-value">${String(params.dictionary || 'NWL').replace(/\+\s*AW/gi, '').replace(/\+\s*ADDED_WORDS/gi, '').trim()}${(params.use_added_words || String(params.dictionary || '').toUpperCase().includes('AW')) ? ' + AW' : ''}</span>
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
        const hasNotPlayedYet = userStatus.status === 'active' && userStatus.has_turn;
        if (data.round_scores && data.round_scores.length > 0 && !hasNotPlayedYet) {
            lbCard.classList.remove('hidden');
            lbList.innerHTML = data.round_scores.map((s, idx) => {
                const isMe = s.username === window.currentUser;
                const highlight = isMe ? 'border: 1px solid rgba(46, 204, 113, 0.3); background: rgba(46, 204, 113, 0.05);' : '';

                // Show replay button ONLY if board data and submitted words are returned by the backend
                const canReplay = s.board_data && s.submitted_words && s.submitted_words.length > 0;
                const replayBtnHtml = canReplay
                    ? `<button class="replay-btn-small" title="Watch Replay" onclick="watchTournamentReplay(${data.id}, ${data.current_round}, '${s.username}', false)">▶ Replay</button>`
                    : '';

                return `
                    <div class="t-leaderboard-item" style="${highlight}">
                        <div class="user-info">
                            <span class="rank">#${idx + 1}</span>
                            <span class="username">${s.username} ${isMe ? '(You)' : ''}</span>
                        </div>
                        <div class="score-group">
                            <span class="score">${s.score} <small style="font-size:0.6rem; opacity:0.6;">PTS</small></span>
                            <div style="display:flex; gap:8px;">
                                ${replayBtnHtml}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            lbCard.classList.add('hidden');
        }
    }

    // 4.6 Matchups (Pairings)
    const mCard = document.getElementById('tournament-matchups-card');
    const mList = document.getElementById('tournament-matchups-list');
    const viewAllBtn = document.getElementById('view-all-pairings-btn');

    if (mCard && mList) {
        if (data.all_matchups && data.all_matchups.length > 0) {
            mCard.classList.remove('hidden');

            // USER REQUEST: Display the pairing for yourself only by default
            const myMatchup = data.all_matchups.find(m => 
                m.u1_name === window.currentUser || m.u2_name === window.currentUser
            );
            
            if (myMatchup) {
                mList.innerHTML = renderMatchupItemHTML(myMatchup);
            } else {
                mList.innerHTML = `<p class="placeholder">You have been eliminated or are not in this round.</p>`;
            }

            if (viewAllBtn) {
                viewAllBtn.onclick = () => {
                    showAllPairingsModal(data.all_matchups);
                };
            }
        } else {
            mCard.classList.add('hidden');
        }
    }

    // 4.7 Championship Standings (The Bracket / Ladder)
    const stdCard = document.getElementById('tournament-standings-card');
    const stdList = document.getElementById('tournament-standings-list');
    if (stdCard && stdList) {
        if (data.standings && data.standings.length > 0) {
            stdCard.classList.remove('hidden');
            stdList.innerHTML = data.standings.map(s => {
                const isMe = s.username === window.currentUser;
                const statusClass = s.status; // 'active', 'eliminated', 'completed'
                const rankInfo = s.final_rank ? `<small style="margin-left:5px; opacity:0.7">Rank #${s.final_rank}</small>` : '';

                return `
                    <div class="t-standing-item ${statusClass}" title="${s.status}">
                        <span class="dot"></span>
                        ${s.username} ${isMe ? '(You)' : ''}
                        ${rankInfo}
                    </div>
                `;
            }).join('');
        } else {
            stdCard.classList.add('hidden');
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
                        <td>
                            <div style="display:flex; align-items:center; gap:10px;">
                                <span>${h.winning_score || 0} pts (R${h.current_round})</span>
                                <button class="replay-btn-small" title="Replay Winning Round" onclick="watchTournamentWinnerReplay(${h.id}, '${h.username}')">▶</button>
                            </div>
                        </td>
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

function renderScoresListHTML(myScore, oppScore, oppName) {
    const myName = window.currentUser || "You";
    const p1 = myScore >= oppScore ? { name: myName, score: myScore } : { name: oppName, score: oppScore };
    const p2 = myScore >= oppScore ? { name: oppName, score: oppScore } : { name: myName, score: myScore };
    
    const color1 = myScore === oppScore ? '#ffd700' : '#2ecc71';
    const color2 = myScore === oppScore ? '#ffd700' : '#e74c3c';
    
    return `
        <div style="margin-top: 15px; font-size: 1.1rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px; text-align: left; display: inline-block; min-width: 220px;">
            <div style="font-weight:700; opacity:0.9; margin-bottom: 8px; text-align: center; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Scores:</div>
            <div style="display:flex; justify-content:space-between; margin-bottom: 6px; color: ${color1}; font-weight: 700;">
                <span>${p1.name}</span>
                <span>${p1.score} pts</span>
            </div>
            <div style="display:flex; justify-content:space-between; color: ${color2}; opacity: ${myScore === oppScore ? '1' : '0.9'}; font-weight: ${myScore === oppScore ? '700' : 'normal'};">
                <span>${p2.name}</span>
                <span>${p2.score} pts</span>
            </div>
        </div>
    `;
}

function renderActiveState(container, data, userStatus) {
    const matchup = userStatus.matchup;
    const total = data.total_participants || 0;

    if (userStatus.status === 'eliminated') {
        const rank = userStatus.final_rank || total;
        let scoresHtml = "";
        if (matchup) {
            const myScore = matchup.my_score || 0;
            const oppScore = matchup.opponent_score || 0;
            scoresHtml = renderScoresListHTML(myScore, oppScore, matchup.opponent_name);
        }
        
        container.innerHTML = `
            <div style="background: rgba(231, 76, 60, 0.1); border: 2px solid #e74c3c; border-radius: 15px; padding: 25px; text-align: center; max-width: 500px; margin: 0 auto;">
                <div style="font-size:3rem; color:#e74c3c; font-weight:900; margin-bottom:10px; text-shadow: 0 0 20px rgba(231, 76, 60, 0.4);">YOU LOST</div>
                <div style="font-size:1.5rem; opacity:0.8; margin-bottom:20px;">Rank: #${rank} of ${total} players</div>
                ${scoresHtml}
                <p style="font-size:1.05rem; opacity:0.7; line-height:1.6; margin-top: 20px; text-align: center;">You fought well, but have been eliminated from this tournament. Keep practicing for the next one!</p>
            </div>
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

    // Header with Round and Matchup
    let matchupHtml = "";
    if (matchup) {
        if (matchup.opponent_id === -1) {
            matchupHtml = `<div style="font-size:1.2rem; color:var(--accent-color); font-weight:600; margin-bottom:20px;">VERSUS: BYE (Auto-advancing)</div>`;
        } else {
            matchupHtml = `<div style="font-size:1.2rem; color:var(--accent-color); font-weight:600; margin-bottom:20px;">VERSUS: <span style="color:#fff">${matchup.opponent_name}</span></div>`;
        }
    }

    container.innerHTML = `
        <h2 style="margin-bottom:5px; text-align:left;">Round ${data.current_round}</h2>
        ${matchupHtml}
    `;

    // Status / result messaging
    if (!userStatus.has_turn) {
        // I have played.
        if (matchup && matchup.opponent_id !== -1) {
            const myScore = matchup.my_score || 0;
            const oppScore = matchup.opponent_score; // null if not played

            if (oppScore === null) {
                // Opponent hasn't played
                container.innerHTML += `
                    <div style="background: rgba(243, 156, 18, 0.1); border: 1px solid #f39c12; border-radius: 12px; padding: 25px; text-align: center; max-width: 500px; margin: 0 auto;">
                        <div style="font-size:1.8rem; color:#f39c12; font-weight:800; margin-bottom:10px;">WAITING FOR OPPONENT</div>
                        <p style="opacity:0.9;">You finished with <strong>${myScore} pts</strong>. We'll show the result once ${matchup.opponent_name} finishes.</p>
                    </div>
                `;
            } else {
                // Both played!
                const scoresHtml = renderScoresListHTML(myScore, oppScore, matchup.opponent_name);
                if (myScore > oppScore) {
                    container.innerHTML += `
                        <div style="background: rgba(46, 204, 113, 0.1); border: 2px solid #2ecc71; border-radius: 15px; padding: 25px; text-align: center; animation: pulse 2s infinite; max-width: 500px; margin: 0 auto;">
                            <div style="font-size:1.8rem; color:#2ecc71; font-weight:800; margin-bottom:15px;">YOU WON! ADVANCING...</div>
                            ${scoresHtml}
                        </div>
                    `;
                } else if (oppScore > myScore) {
                    container.innerHTML += `
                        <div style="background: rgba(231, 76, 60, 0.1); border: 2px solid #e74c3c; border-radius: 15px; padding: 25px; text-align: center; max-width: 500px; margin: 0 auto;">
                            <div style="font-size:1.8rem; color:#e74c3c; font-weight:800; margin-bottom:15px;">YOU LOST</div>
                            ${scoresHtml}
                        </div>
                    `;
                } else {
                    // Tie
                    container.innerHTML += `
                        <div style="background: rgba(243, 156, 18, 0.1); border: 2px solid #f39c12; border-radius: 15px; padding: 25px; text-align: center; max-width: 500px; margin: 0 auto;">
                            <div style="font-size:1.8rem; color:#f39c12; font-weight:800; margin-bottom:5px;">IT'S A TIE!</div>
                            <div style="font-size:1.1rem; opacity:0.8; margin-bottom:15px;">Both of you got ${myScore} pts. Random winner selection pending round end.</div>
                            ${scoresHtml}
                        </div>
                    `;
                }
            }
        } else if (matchup && matchup.opponent_id === -1) {
            // Bye
            container.innerHTML += `
                <div style="background: rgba(46, 204, 113, 0.1); border: 2px solid #2ecc71; border-radius: 15px; padding: 25px; text-align: center; animation: pulse 2s infinite; max-width: 500px; margin: 0 auto;">
                    <div style="font-size:1.8rem; color:#2ecc71; font-weight:800; margin-bottom:5px;">YOU WON! ADVANCING...</div>
                    <div style="font-size:1.1rem; opacity:0.8;">Automatic win this round.</div>
                </div>
            `;
        }
    } else {
        // I have NOT played.
        const btn = document.createElement('button');
        btn.className = 'primary-action';
        btn.style.width = '100%';
        btn.style.padding = '20px';
        btn.style.fontSize = '1.2rem';
        btn.style.background = '#2ecc71';
        btn.textContent = 'PLAY YOUR TURN';
        btn.onclick = () => {
            launchTournamentGame(data.id, data.current_round);
        };
        container.appendChild(btn);

        const note = document.createElement('p');
        note.style.marginTop = '15px';
        note.style.color = '#2ecc71';
        note.style.fontWeight = '700';
        note.textContent = "CRITICAL: It is your turn! Do not miss the deadline.";
        container.appendChild(note);
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
    localStorage.setItem('tournament_play_active', JSON.stringify({ tid, round }));
    if (window.navigateToPage) {
        window.navigateToPage('play');
    } else {
        window.location.href = '#page-play';
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
        board: (() => {
            const parsed = JSON.parse(scoreData.board_data);
            return (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) ? parsed.board : parsed;
        })(),
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
window.watchTournamentWinnerReplay = async function (tid, username) {
    try {
        const res = await fetch(`/api/tournament/winner-turn/${tid}/${username}`);
        const data = await res.json();

        if (data.error) {
            alert("Could not load winner's replay: " + data.error);
            return;
        }

        const mockRound = {
            room_id: `tournament_hist_${tid}`,
            round_number: data.current_round || 0,
            board: (data.board_data && typeof data.board_data === 'object' && !Array.isArray(data.board_data)) ? data.board_data.board : data.board_data,
            words: data.submitted_words,
            total_score: data.score,
            round_duration: data.parameters?.time_limit || 60,
            round_start_time: data.round_start_time || (data.submitted_at - 60),
            timestamp: data.submitted_at * 1000,
            username: data.username,
            game_type: 'tournament'
        };

        window.lastTournamentReplay = mockRound;
        if (window.watchRoundHistory) {
            window.watchRoundHistory(`tournament_hist_${tid}`, mockRound.round_number);
        }
    } catch (e) {
        console.error(e);
        alert("Error connecting to server.");
    }
};
function renderMatchupItemHTML(m) {
    const u1_isMe = m.u1_name === window.currentUser;
    const u2_isMe = m.u2_name === window.currentUser;
    const highlight = (u1_isMe || u2_isMe) ? 'border: 1px solid var(--accent-color); background: rgba(144, 12, 63, 0.1);' : '';

    const s1 = m.u1_score === null ? '...' : m.u1_score;
    const s2 = m.u2_score === null ? '...' : m.u2_score;

    return `
        <div class="t-matchup-item" style="${highlight}">
            <div class="participant">
                <span class="username ${u1_isMe ? 'me' : ''}">${m.u1_name}</span> <span class="pts">${s1}</span>
            </div>
            <div class="vs">vs</div>
            <div class="participant">
                ${m.user2_id === -1
            ? `<span style="opacity:0.5">BYE</span>`
            : `<span class="username ${u2_isMe ? 'me' : ''}">${m.u2_name}</span> <span class="pts">${s2}</span>`
        }
            </div>
        </div>
    `;
}

function showAllPairingsModal(matchups) {
    console.log("[Tournament] Opening All Pairings Modal, count:", matchups.length);
    const modal = document.getElementById('generic-info-modal');
    const titleEl = document.getElementById('generic-modal-title');
    const bodyEl = document.getElementById('generic-modal-body');

    if (!modal || !titleEl || !bodyEl) {
        console.error("[Tournament] Modal elements not found:", { modal, titleEl, bodyEl });
        return;
    }

    titleEl.textContent = "All Tournament Pairings";
    
    const curUser = window.currentUser || localStorage.getItem('morpheme_username');
    
    // Sort matchups such that user's pair is near top for convenience
    const sorted = [...matchups].sort((a,b) => {
        const aHasMe = a.u1_name === curUser || a.u2_name === curUser;
        const bHasMe = b.u1_name === curUser || b.u2_name === curUser;
        if (aHasMe) return -1;
        if (bHasMe) return 1;
        return 0;
    });

    bodyEl.innerHTML = `
        <div class="t-matchups-list" style="padding: 10px 0; max-height: 500px; overflow-y: auto;">
            ${sorted.map(m => renderMatchupItemHTML(m)).join('')}
        </div>
    `;

    modal.classList.remove('hidden');
    modal.style.display = 'flex';
}

// Add close listener if not added globally elsewhere (usually handled by app.js)
const closeGenModal = document.getElementById('close-generic-modal');
if (closeGenModal) {
    closeGenModal.onclick = () => {
        const modal = document.getElementById('generic-info-modal');
        if (modal) {
            modal.classList.add('hidden');
            modal.style.display = 'none';
        }
    };
}
