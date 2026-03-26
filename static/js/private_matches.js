(function () {
    let soloBots = [];

    function init() {
        // Tab switching
        document.querySelectorAll('.sf-tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.sf-tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const tab = btn.dataset.sfTab;
                document.querySelectorAll('.sf-tab-content').forEach(c => c.classList.add('hidden'));
                document.getElementById('sf-tab-' + tab).classList.remove('hidden');
            });
        });

        // Subtab switching
        document.querySelectorAll('.sf-subtab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.sf-subtab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const subtab = btn.dataset.friendsSubtab;
                document.querySelectorAll('.friends-matches-list').forEach(c => c.classList.add('hidden'));
                document.getElementById('friends-list-' + subtab).classList.remove('hidden');
            });
        });

        // Add Bot
        const addBotBtn = document.getElementById('add-bot-btn');
        if (addBotBtn) {
            addBotBtn.addEventListener('click', () => {
                const botId = Date.now();
                const names = ['AlphaBot', 'BetaBot', 'GammaBot', 'ZetaBot', 'BoggleMaster', 'WordWiz'];
                const botName = names[Math.floor(Math.random() * names.length)] + '_' + Math.floor(Math.random() * 999);
                soloBots.push({ id: botId, username: botName, rating: 1200 });
                renderBots();
            });
        }

        // Start Solo
        const startSoloBtn = document.getElementById('start-solo-btn');
        if (startSoloBtn) {
            startSoloBtn.addEventListener('click', startSoloMatch);
        }

        // Send Invite
        const inviteBtn = document.getElementById('invite-friends-btn');
        if (inviteBtn) {
            inviteBtn.addEventListener('click', sendFriendsInvite);
        }

        // Initial Load
        loadPrivateMatches();
        setInterval(loadPrivateMatches, 30000); // Polling for invites/turns

        // Dynamic Min Word Len options
        const soloDims = document.getElementById('sf-config-dims');
        if (soloDims) {
            soloDims.addEventListener('change', () => {
                updateMinWordLenOptions();
                updateFormatOptions();
            });
            updateMinWordLenOptions();
            updateFormatOptions();
        }
    }

    function updateMinWordLenOptions() {
        const soloDims = document.getElementById('sf-config-dims');
        const soloMinLen = document.getElementById('sf-config-min-len');
        if (!soloDims || !soloMinLen) return;

        const dim = soloDims.value;
        const config = {
            '4x4': [3, 4, 5],
            '4x6': [4, 5, 6],
            '5x7': [5, 6, 7],
            '6x8': [6, 7, 8],
            '3x3x3': [6, 7, 8]
        };

        const options = config[dim] || [3, 4, 5];
        const currentVal = parseInt(soloMinLen.value);

        soloMinLen.innerHTML = options.map(opt =>
            `<option value="${opt}" ${opt === currentVal ? 'selected' : ''}>${opt} Letters</option>`
        ).join('');

        // Ensure the selected value is valid for the new options
        if (!options.includes(parseInt(soloMinLen.value))) {
            soloMinLen.value = options[0];
        }
    }

    function updateFormatOptions() {
        const soloDims = document.getElementById('sf-config-dims');
        const soloFormat = document.getElementById('sf-config-format');
        if (!soloDims || !soloFormat) return;

        if (soloDims.value === '3x3x3') {
            // Force Normal
            soloFormat.value = 'Normal';
            // Disable other options to prevent selection
            Array.from(soloFormat.options).forEach(opt => {
                if (opt.value !== 'Normal') {
                    opt.disabled = true;
                }
            });
        } else {
            // Re-enable options for 2D boards
            Array.from(soloFormat.options).forEach(opt => {
                opt.disabled = false;
            });
        }
    }

    function renderBots() {
        const container = document.getElementById('solo-bots-list');
        if (!container) return;
        container.innerHTML = soloBots.map(bot => `
            <div class="bot-entry" data-bot-id="${bot.id}">
                <span>${bot.username}</span>
                <input type="number" value="${bot.rating}" min="400" max="3000" step="100" onchange="window.updateBotRating(${bot.id}, this.value)">
                <button class="bot-remove-btn" onclick="window.removeBot(${bot.id})">×</button>
            </div>
        `).join('');
    }

    window.updateBotRating = (id, rating) => {
        const bot = soloBots.find(b => b.id === id);
        if (bot) bot.rating = parseInt(rating);
    };

    window.removeBot = (id) => {
        soloBots = soloBots.filter(b => b.id !== id);
        renderBots();
    };

    async function startSoloMatch() {
        const params = {
            board_dimensions: document.getElementById('sf-config-dims').value,
            time_limit: parseInt(document.getElementById('sf-config-time').value),
            dictionary: document.getElementById('sf-config-dict').value,
            min_word_length: parseInt(document.getElementById('sf-config-min-len').value),
            bonus_word_length: parseInt(document.getElementById('sf-config-bonus').value),
            difficulty: document.getElementById('sf-config-difficulty').value,
            board_format: document.getElementById('sf-config-format').value,
            word_count_range: document.getElementById('sf-config-range').value
        };

        const participants = soloBots.map(b => ({ username: b.username, is_ai: true, ai_rating: b.rating }));

        try {
            const res = await fetch('/api/solo-match/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ parameters: params, participants })
            });
            const data = await res.json();
            if (data.success) {
                // Important: Clear any special play mode tokens so standard polling takes over
                localStorage.removeItem('private_match_active');
                localStorage.removeItem('tournament_play_active');

                window.currentRoomId = data.room_id;
                if (window.navigateToPage) {
                    window.navigateToPage('play');
                }
            } else {
                alert(data.error);
            }
        } catch (e) {
            console.error(e);
        }
    }

    async function sendFriendsInvite() {
        const input = document.getElementById('friends-invite-input');
        const usernames = input.value.split(',').map(u => u.trim()).filter(u => u);
        if (usernames.length === 0) return;

        const params = {
            board_dimensions: document.getElementById('sf-config-dims').value, // Use same config as solo for simplicity
            time_limit: parseInt(document.getElementById('sf-config-time').value),
            dictionary: document.getElementById('sf-config-dict').value,
            min_word_length: parseInt(document.getElementById('sf-config-min-len').value),
            bonus_word_length: parseInt(document.getElementById('sf-config-bonus').value),
            difficulty: document.getElementById('sf-config-difficulty').value,
            board_format: document.getElementById('sf-config-format').value,
            word_count_range: document.getElementById('sf-config-range').value
        };

        const participants = usernames.map(u => ({ username: u, is_ai: false }));
        // Include bots if any
        soloBots.forEach(b => {
            participants.push({ username: b.username, is_ai: true, ai_rating: b.rating });
        });

        try {
            const res = await fetch('/api/private-match/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ match_type: 'with_friends', parameters: params, participants })
            });
            const data = await res.json();
            if (data.success) {
                input.value = "";
                loadPrivateMatches();
                showInviteSentToast();
            } else {
                if (window.showAlertModal) window.showAlertModal('Error', data.error || 'Failed to send invitation.');
                else alert(data.error);
            }
        } catch (e) {
            console.error(e);
        }
    }

    async function loadPrivateMatches() {
        if (!window.currentUser || window.currentUserIsGuest) return;
        try {
            const res = await fetch('/api/private-match/list');
            const data = await res.json();
            renderMatchList('your-turn', data.your_turn);
            renderMatchList('their-turn', data.their_turn);
            renderMatchList('history', data.history, true);

            // Fetch invites too, to combine badges
            const invRes = await fetch('/api/private-match/invites');
            const invData = await invRes.json();
            renderInvites(invData);

            // --- NOTIFICATION LOGIC ---
            if (invData && invData.length > 0) {
                const notifiedInvites = JSON.parse(localStorage.getItem('morpheme_notified_invites') || '[]');
                let hasNew = false;
                let latestSender = '';

                invData.forEach(inv => {
                    if (!notifiedInvites.includes(inv.id)) {
                        hasNew = true;
                        latestSender = inv.sender_name;
                        notifiedInvites.push(inv.id);
                    }
                });

                if (hasNew) {
                    localStorage.setItem('morpheme_notified_invites', JSON.stringify(notifiedInvites));
                    showInviteNotification(latestSender, invData.length);
                }
            }

            // --- BADGING LOGIC ---
            const turnCount = data.your_turn ? data.your_turn.length : 0;
            const inviteCount = invData ? invData.length : 0;
            const totalActionCount = turnCount + inviteCount;

            // ... rest of badge logic ...
            // 1. Friends Tab Badge (inside Lobby - Total turns + invites)
            const tabBadge = document.getElementById('friends-tab-badge');
            if (tabBadge) {
                tabBadge.textContent = totalActionCount > 0 ? totalActionCount : '';
                tabBadge.classList.toggle('hidden', totalActionCount === 0);
            }

            // 2. Friends Subtab Badges
            // Your Turn
            const turnBadge = document.getElementById('your-turn-badge');
            if (turnBadge) {
                turnBadge.textContent = turnCount > 0 ? turnCount : '';
                turnBadge.classList.toggle('hidden', turnCount === 0);
            }
            // Invites
            const subBadge = document.getElementById('invite-count-badge');
            if (subBadge) {
                subBadge.textContent = inviteCount > 0 ? inviteCount : '';
                subBadge.classList.toggle('hidden', inviteCount === 0);
            }

            // 3. Global Lobby Badge (Nav)
            const lobbyBadge = document.getElementById('lobby-badge');
            if (lobbyBadge) {
                if (totalActionCount > 0) {
                    lobbyBadge.textContent = totalActionCount;
                    lobbyBadge.classList.remove('hidden');
                } else {
                    lobbyBadge.classList.add('hidden');
                    lobbyBadge.textContent = '';
                }
            }

        } catch (e) { console.error('Error loading private matches:', e); }
    }
    window.loadPrivateMatches = loadPrivateMatches;

    function showInviteSentToast() {
        const existing = document.getElementById('invite-sent-toast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.id = 'invite-sent-toast';
        toast.className = 'pm-toast-notification';
        toast.innerHTML = `
            <div class="pm-toast-content" style="border-left: 4px solid #a78bfa;">
                <div class="pm-toast-icon" style="background: rgba(167,139,250,0.2); font-size:1.5rem;">✉️</div>
                <div class="pm-toast-details">
                    <div class="pm-toast-title" style="color:#a78bfa;">Invitation Sent!</div>
                    <div class="pm-toast-text">Your friend will see the invite in their lobby.</div>
                </div>
                <div class="pm-toast-actions">
                    <button class="pm-toast-btn close" onclick="this.closest('#invite-sent-toast').remove()">Dismiss</button>
                </div>
            </div>
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }

    function showInviteNotification(sender, count) {
        const existing = document.getElementById('invite-toast');
        if (existing) {
            existing.remove();
        }

        const toast = document.createElement('div');
        toast.id = 'invite-toast';
        toast.className = 'pm-toast-notification'; // Reuse PM styles for consistency
        toast.innerHTML = `
            <div class="pm-toast-content" style="border-left: 4px solid #4ade80;">
                <div class="pm-toast-icon" style="background: rgba(74, 222, 128, 0.2);">🤝</div>
                <div class="pm-toast-details">
                    <div class="pm-toast-title">Game Invitation</div>
                    <div class="pm-toast-text"><strong>${sender}</strong> invited you to play!</div>
                </div>
                <div class="pm-toast-actions">
                    <button class="pm-toast-btn respond" style="background: #4ade80; color: #000;" onclick="handleInviteToastClick()">View</button>
                    <button class="pm-toast-btn close" onclick="this.closest('.pm-toast-notification').remove()">Dismiss</button>
                </div>
            </div>
        `;
        document.body.appendChild(toast);

        // Auto-remove after 10s
        setTimeout(() => toast.remove(), 10000);
    }

    window.handleInviteToastClick = () => {
        document.getElementById('invite-toast')?.remove();
        if (!window.navigateToPage) return;

        // 1. Navigate to lobby
        window.navigateToPage('lobby');

        // 2. Give the lobby time to render, then click through tabs + scroll
        setTimeout(() => {
            // Click "With Friends" sf-tab
            const friendsSfTab = document.querySelector('.sf-tab-btn[data-sf-tab="friends"]');
            if (friendsSfTab) friendsSfTab.click();

            setTimeout(() => {
                // Click "Invites" subtab
                const inviteSubtab = document.querySelector('[data-friends-subtab="invites"]');
                if (inviteSubtab) inviteSubtab.click();

                // Reload invites so they are fresh, then scroll + highlight
                loadPrivateMatches().then(() => {
                    setTimeout(() => {
                        const invitesList = document.getElementById('friends-list-invites');
                        if (invitesList) {
                            // Scroll the invite panel into view
                            invitesList.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

                            // Pulse-highlight each invite card so it's obvious
                            const panels = invitesList.querySelectorAll('.invite-panel');
                            panels.forEach(panel => {
                                panel.style.transition = 'box-shadow 0.3s ease, transform 0.3s ease';
                                panel.style.boxShadow = '0 0 24px 6px rgba(74, 222, 128, 0.7)';
                                panel.style.transform = 'scale(1.02)';
                                setTimeout(() => {
                                    panel.style.boxShadow = '';
                                    panel.style.transform = '';
                                }, 1800);
                            });
                        }
                    }, 150);
                }).catch(() => {
                    // Fallback: just scroll without reload
                    const invitesList = document.getElementById('friends-list-invites');
                    if (invitesList) invitesList.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                });
            }, 100);
        }, 150);
    };

    function renderMatchList(type, matches, isHistory = false) {
        const container = document.getElementById('friends-list-' + type);
        if (!container) return;
        if (matches.length === 0) {
            container.innerHTML = `<p class="placeholder">No matches in this category.</p>`;
            return;
        }

        container.innerHTML = matches.map(m => {
            const formattedDate = m.last_activity ? new Date(m.last_activity * 1000).toLocaleDateString() : '';
            return `
            <div class="friends-match-panel">
                <div class="match-info">
                    <h4>With Friends ${m.current_round > 1 ? `(Round ${m.current_round})` : ''}</h4>
                    <p style="font-size:0.85em; opacity:0.85; line-height:1.4;">
                        <strong>Board:</strong> ${m.parameters.board_dimensions || '4x4'} | <strong>Time:</strong> ${m.parameters.time_limit || 60}s | <strong>Dict:</strong> ${m.parameters.dictionary || 'NWL'}<br>
                        <strong>Rules:</strong> Min ${m.parameters.min_word_length || 3}L | Bonus ${m.parameters.bonus_word_length || 'None'}<br>
                        <strong>Style:</strong> ${m.parameters.difficulty === 'Normal' ? 'Medium' : (m.parameters.difficulty || 'Medium')} | ${m.parameters.board_format || 'Normal'} | Range: ${(() => {
                    let wr = m.parameters.word_count_range;
                    if (Array.isArray(wr)) {
                        if (wr[1] > 900) return wr[0] + '+';
                        return wr[0] + '-' + wr[1];
                    }
                    return wr === 'random' ? '50-100/100-200/200+' : (wr || '50-100');
                })()}
                        ${isHistory && formattedDate ? `<br><strong>Completed:</strong> ${formattedDate}` : ''}
                    </p>
                    <p style="margin-top:5px;">Players: ${m.players.map(p => `
                        <span class="player-pill ${p.status}">
                            ${p.username}${p.status === 'pending' ? ' (Pending)' : ''}
                        </span>
                    `).join('')}</p>
                </div>
                <div class="match-actions">
                    ${!isHistory ? `<button class="sf-action-btn" onclick="window.launchPrivateMatch(${m.id})">Play Turn</button>` : ''}
                    ${isHistory ? `<button class="rematch-btn" onclick="window.rematchPrivate(${m.id})">Rematch</button>` : ''}
                    ${isHistory ? `<button class="replay-btn-friends" onclick="window.showPrivateHistory(${m.id})">View History</button>` : ''}
                </div>
            </div>
        `;
        }).join('');
    }

    function renderInvites(invites) {
        const container = document.getElementById('friends-list-invites');
        if (!container) return;
        if (invites.length === 0) {
            container.innerHTML = '<p class="placeholder">No pending invitations.</p>';
            return;
        }

        container.innerHTML = invites.map(inv => `
            <div class="friends-match-panel invite-panel">
                <div class="match-info">
                    <h4>Invite from ${inv.sender_name}</h4>
                    <p style="font-size:0.85em; opacity:0.85; line-height:1.4;">
                        <strong>Board:</strong> ${inv.parameters.board_dimensions || '4x4'} | <strong>Time:</strong> ${inv.parameters.time_limit || 60}s | <strong>Dict:</strong> ${inv.parameters.dictionary || 'NWL'}<br>
                        <strong>Rules:</strong> Min ${inv.parameters.min_word_length || 3}L | Bonus ${inv.parameters.bonus_word_length || 'None'}<br>
                        <strong>Style:</strong> ${inv.parameters.difficulty === 'Normal' ? 'Medium' : (inv.parameters.difficulty || 'Medium')} | ${inv.parameters.board_format || 'Normal'} | Range: ${(() => {
                let wr = inv.parameters.word_count_range;
                if (Array.isArray(wr)) {
                    if (wr[1] > 900) return wr[0] + '+';
                    return wr[0] + '-' + wr[1];
                }
                return wr === 'random' ? '50-100/100-200/200+' : (wr || '50-100');
            })()}
                    </p>
                </div>
                <div class="match-actions">
                    <button class="sf-primary-btn" onclick="window.acceptInvite(${inv.id})">Accept</button>
                    <button class="sf-action-btn" style="background:#444;" onclick="window.declineInvite(${inv.id})">Decline</button>
                </div>
            </div>
        `).join('');
    }

    window.acceptInvite = async (inviteId) => {
        try {
            const res = await fetch('/api/private-match/invite/accept', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ invite_id: inviteId })
            });
            const data = await res.json();
            if (data.success) {
                loadPrivateMatches();
                // Switch to Your Turn subtab
                document.querySelector('[data-friends-subtab="your-turn"]').click();
            }
        } catch (e) { }
    };

    window.declineInvite = async (inviteId) => {
        // Just delete for now, maybe add explicit decline later
        try {
            await fetch('/api/private-match/invite/accept', { // We can reuse or add a delete endpoint
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ invite_id: inviteId, action: 'decline' })
            });
            loadPrivateMatches();
        } catch (e) { }
    };

    window.launchPrivateMatch = async (matchId) => {
        try {
            const res = await fetch('/api/private-match/status/' + matchId);
            const data = await res.json();
            if (data.error) {
                alert(data.error);
                return;
            }

            // Setup local state for play.js
            localStorage.setItem('private_match_active', JSON.stringify({
                mid: matchId,
                round: data.current_round,
                parameters: data.parameters,
                board: data.board,
                bonus_word: data.bonus_word,
                end_time: (data.time_remaining !== undefined) ? (Date.now() / 1000 + data.time_remaining) : (data.end_time || (Date.now() / 1000 + data.parameters.time_limit))
            }));

            // Navigate to play page
            if (window.navigateToPage) window.navigateToPage('play');

            // Start game
            if (window.startGamePolling) window.startGamePolling();
        } catch (e) {
            console.error(e);
        }
    };

    window.rematchPrivate = (matchId) => {
        if (window.showConfirmModal) {
            window.showConfirmModal("Rematch", "Start a new match with the same players and settings?", async () => {
                await executeRematch(matchId);
            });
        } else {
            if (confirm("Start a new match with the same players and settings?")) {
                executeRematch(matchId);
            }
        }
    };

    async function executeRematch(matchId) {
        try {
            const res = await fetch('/api/private-match/rematch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ match_id: matchId })
            });
            const data = await res.json();

            if (data.success) {
                // Fetch matches again immediately without alerting first so UI updates safely
                loadPrivateMatches();
                if (window.showAlertModal) {
                    window.showAlertModal("Success", "New match created! Check 'Your Turn'.");
                } else {
                    alert("New match created! Check 'Your Turn'.");
                }
            } else {
                if (window.showAlertModal) window.showAlertModal("Error", "Error starting rematch: " + (data.error || 'Unknown'));
                else alert("Error starting rematch: " + (data.error || 'Unknown'));
            }
        } catch (e) {
            console.error(e);
            if (window.showAlertModal) window.showAlertModal("Error", "Error connecting to server.");
            else alert("Error connecting to server.");
        }
    }

    window.showPrivateHistory = async (matchId) => {
        try {
            const res = await fetch('/api/private-match/history/' + matchId);
            const data = await res.json();
            // Show a simple modal or list with results and replay buttons
            let html = '<div style="padding:20px; color:#fff; background:var(--bg-panel); border-radius:12px; min-width:400px; max-height:80vh; overflow-y:auto; text-align:center;"><h3>Match Results</h3>';

            if (!data || data.length === 0) {
                html += '<div style="padding:40px; opacity:0.6; font-style:italic;">No turns recorded for this match yet.</div>';
            } else {
                data.slice(0, 25).forEach(t => {
                    let wordListHtml = '';
                    try {
                        let words = t.submitted_words;
                        if (typeof words === 'string') {
                            words = JSON.parse(words);
                        }

                        if (Array.isArray(words) && words.length > 0) {
                            // Sort by length (desc) then alphabetically (asc)
                            words.sort((a, b) => b.word.length - a.word.length || a.word.localeCompare(b.word));
                            const wordStrs = words.map(w => `<span style="background:rgba(255,255,255,0.1); padding:2px 6px; border-radius:4px; margin:2px; display:inline-block; font-size:0.8em;">${w.word} (${w.points})</span>`).join('');
                            wordListHtml = `<div style="margin-top:5px; text-align:left; opacity:0.8;">${wordStrs}</div>`;
                        } else {
                            wordListHtml = `<div style="margin-top:5px; text-align:left; font-style:italic; opacity:0.6;">No words found.</div>`;
                        }
                    } catch (e) {
                        console.error("Error parsing words:", e);
                        wordListHtml = `<div style="margin-top:5px; text-align:left; color:red; font-size:0.8em;">Error loading words</div>`;
                    }

                    // Prepare a safe subset of turn data for the replay button to avoid huge HTML attributes
                    const safeTurn = {
                        match_id: t.match_id,
                        round_number: t.round_number,
                        board: t.board,
                        submitted_words: t.submitted_words,
                        score: t.score,
                        submitted_at: t.submitted_at,
                        username: t.username
                    };

                    html += `
                        <div style="background:rgba(255,255,255,0.05); padding:15px; margin-bottom:15px; border-radius:10px; display:flex; flex-direction:column; gap:10px; text-align:left;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <strong style="font-size:1.1em; color:var(--accent-color);">${t.username}</strong>
                                    <div style="font-size:0.9em; opacity:0.7;">Round ${t.round_number} • ${t.score} pts</div>
                                </div>
                                <button class="sf-action-btn" onclick="window.watchPrivateReplay(${JSON.stringify(safeTurn).replace(/"/g, '&quot;')})">Replay</button>
                            </div>
                            ${wordListHtml}
                        </div>
                    `;
                });
            }
            html += '<button class="sf-primary-btn" onclick="document.getElementById(\'private-history-modal\').remove()">Close</button></div>';

            const modal = document.createElement('div');
            modal.id = 'private-history-modal';
            modal.style.position = 'fixed';
            modal.style.inset = '0';
            modal.style.background = 'rgba(0,0,0,0.8)';
            modal.style.zIndex = '9999';
            modal.style.display = 'flex';
            modal.style.alignItems = 'center';
            modal.style.justifyContent = 'center';
            modal.innerHTML = html;

            // Close on background click
            modal.onclick = (e) => {
                if (e.target === modal) modal.remove();
            };

            document.body.appendChild(modal);
        } catch (e) { }
    };

    window.watchPrivateReplay = (turnData) => {
        // TurnData has board, words, score, etc.
        // Convert to format watchRoundHistory expects
        const mockRound = {
            room_id: 'private_' + turnData.match_id,
            round_number: turnData.round_number,
            board: turnData.board,
            words: turnData.submitted_words,
            total_score: turnData.score,
            round_duration: 60, // Fallback
            round_start_time: turnData.submitted_at - 60,
            timestamp: turnData.submitted_at * 1000,
            username: turnData.username
        };

        window.lastTournamentReplay = mockRound;
        if (window.watchRoundHistory) {
            window.watchRoundHistory('private_' + turnData.match_id, turnData.round_number);
        }
    };

    document.addEventListener('DOMContentLoaded', init);
})();
