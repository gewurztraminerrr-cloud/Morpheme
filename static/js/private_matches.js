(function() {
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
            board_dimensions: document.getElementById('solo-dims').value,
            time_limit: parseInt(document.getElementById('solo-time').value),
            dictionary: document.getElementById('solo-dict').value,
            min_word_length: parseInt(document.getElementById('solo-min-len').value),
            bonus_word_length: parseInt(document.getElementById('solo-bonus').value)
        };

        const participants = soloBots.map(b => ({ username: b.username, is_ai: true, ai_rating: b.rating }));

        try {
            const res = await fetch('/api/private-match/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ match_type: 'solo', parameters: params, participants })
            });
            const data = await res.json();
            if (data.success) {
                launchPrivateMatch(data.match_id);
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
            board_dimensions: document.getElementById('solo-dims').value, // Use same config as solo for simplicity
            time_limit: parseInt(document.getElementById('solo-time').value),
            dictionary: document.getElementById('solo-dict').value,
            min_word_length: parseInt(document.getElementById('solo-min-len').value),
            bonus_word_length: parseInt(document.getElementById('solo-bonus').value)
        };

        const participants = usernames.map(u => ({ username: u, is_ai: false }));

        try {
            const res = await fetch('/api/private-match/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ match_type: 'with_friends', parameters: params, participants })
            });
            const data = await res.json();
            if (data.success) {
                alert("Invites sent!");
                input.value = "";
                loadPrivateMatches();
            } else {
                alert(data.error);
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
        } catch (e) {}
    }

    function renderMatchList(type, matches, isHistory = false) {
        const container = document.getElementById('friends-list-' + type);
        if (!container) return;
        if (matches.length === 0) {
            container.innerHTML = '<p class="placeholder">No matches in this category.</p>';
            return;
        }

        container.innerHTML = matches.map(m => `
            <div class="friends-match-panel">
                <div class="match-info">
                    <h4>${m.match_type === 'solo' ? 'Solo Practice' : 'Match vs Friends'}</h4>
                    <p>${m.parameters.board_dimensions} | ${m.parameters.time_limit}s | Round ${m.current_round}</p>
                    <p>Players: ${m.players.map(p => p.username).join(', ')}</p>
                </div>
                <div class="match-actions">
                    ${!isHistory ? `<button class="sf-action-btn" onclick="window.launchPrivateMatch(${m.id})">Play Turn</button>` : ''}
                    ${isHistory ? `<button class="rematch-btn" onclick="window.rematchPrivate(${m.id})">Rematch</button>` : ''}
                    ${isHistory ? `<button class="replay-btn-friends" onclick="window.showPrivateHistory(${m.id})">View History</button>` : ''}
                </div>
            </div>
        `).join('');
    }

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
                end_time: data.end_time || (Date.now() / 1000 + data.parameters.time_limit)
            }));

            // Navigate to play page
            if (window.navigateToPage) window.navigateToPage('play');
            
            // Start game
            if (window.initPrivateMatchPlay) window.initPrivateMatchPlay();
        } catch (e) {
            console.error(e);
        }
    };

    window.rematchPrivate = async (matchId) => {
        // Find existing match info to reuse params/players
        // For now, just alert
        alert("Rematch feature coming soon (creates new match with same settings)");
    };

    window.showPrivateHistory = async (matchId) => {
        try {
            const res = await fetch('/api/private-match/history/' + matchId);
            const data = await res.json();
            // Show a simple modal or list with results and replay buttons
            let html = '<div style="padding:20px; color:#fff;"><h3>Match Results</h3>';
            data.forEach(t => {
                html += `
                    <div style="background:rgba(255,255,255,0.05); padding:10px; margin-bottom:10px; border-radius:10px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong>${t.username}</strong>: ${t.score} pts (Round ${t.round_number})
                        </div>
                        <button class="sf-action-btn" onclick="window.watchPrivateReplay(${JSON.stringify(t).replace(/"/g, '&quot;')})">Watch Replay</button>
                    </div>
                `;
            });
            html += '<button class="sf-primary-btn" onclick="this.parentElement.remove()">Close</button></div>';
            
            const modal = document.createElement('div');
            modal.style.position = 'fixed';
            modal.style.inset = '0';
            modal.style.background = 'rgba(0,0,0,0.8)';
            modal.style.zIndex = '9999';
            modal.style.display = 'flex';
            modal.style.alignItems = 'center';
            modal.style.justifyContent = 'center';
            modal.innerHTML = html;
            document.body.appendChild(modal);
        } catch (e) {}
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
