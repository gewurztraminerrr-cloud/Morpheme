// Leaderboard Logic

document.addEventListener('DOMContentLoaded', () => {
    // Only init if leaderboard page exists
    if (!document.getElementById('page-leaderboards')) return;

    // Default State
    let currentPeriod = 'day';
    let currentFilters = {
        game_type: 'all',
        board_dimensions: 'all',
        time_limit: 'all'
    };
    let leaderboardData = null; // Cache

    // -- HTML Structure Injection --
    // We will inject the UI structure here to keep index.html cleaner or just for modularity
    const leaderboardPage = document.getElementById('page-leaderboards');
    leaderboardPage.innerHTML = `
        <div class="leaderboard-container">
            <div class="lb-header">
                <h2>LEADERBOARDS</h2>
                <div class="lb-tabs">
                    <button class="lb-tab active" data-period="day">DAY</button>
                    <button class="lb-tab" data-period="week">WEEK</button>
                    <button class="lb-tab" data-period="month">MONTH</button>
                    <button class="lb-tab" data-period="year">YEAR</button>
                    <button class="lb-tab" data-period="all">ALL-TIME</button>
                </div>
            </div>

            <div class="lb-controls-row">
                <div class="lb-filters">
                    <select id="lb-filter-game">
                        <option value="all">All Game Types</option>
                        <option value="accumulative">Accumulative</option>
                        <option value="3d">Cube</option>
                        <option value="fcfs">First Come First Serve</option>
                        <option value="split">Split Points</option>
                    </select>
                    <select id="lb-filter-dims">
                        <option value="all">All Sizes</option>
                        <option value="4x4">4x4</option>
                        <option value="4x6">4x6</option>
                        <option value="5x7">5x7</option>
                        <option value="6x8">6x8</option>
                        <option value="3x3x3">3x3x3 Cube</option>
                    </select>
                    <select id="lb-filter-time">
                        <option value="all">All Time Lengths</option>
                        <option value="45">45s Blitz</option>
                        <option value="180">3m Standard</option>
                        <option value="300">5m Speed</option>
                        <option value="600">10m Relaxed</option>
                    </select>
                </div>
                
                <div class="lb-search">
                    <input type="text" id="lb-search-input" placeholder="Search username..." autocomplete="off">
                    <button type="button" id="lb-find-user-btn" class="lb-search-btn">FIND USER</button>
                    <button type="button" id="lb-find-me-btn" class="lb-search-btn">FIND ME</button>
                </div>
            </div>

            <div id="lb-content-area" class="lb-grid">
                <!-- Grid of Cards (Best Score, Best Word, etc.) -->
                <div class="lb-loading">Loading rankings...</div>
            </div>
            
            <div class="lb-attribution">
                * Leaderboards update in real-time. Only non-24h rooms are tracked.
            </div>

            <!-- Mobile Vertical Scroller Track & Thumb -->
            <div id="lb-mobile-scrollbar-track" class="lb-mobile-scrollbar-track">
                <div id="lb-mobile-scrollbar-thumb" class="lb-mobile-scrollbar-thumb"></div>
            </div>
        </div>
    `;

    // Filter Configurations and Cascading Logic
    const allGameTypes = [
        { value: 'all', text: 'All Game Types' },
        { value: 'accumulative', text: 'Accumulative' },
        { value: '3d', text: 'Cube' },
        { value: 'fcfs', text: 'First Come First Serve' },
        { value: 'split', text: 'Split Points' }
    ];

    const allDims = [
        { value: 'all', text: 'All Sizes' },
        { value: '4x4', text: '4x4' },
        { value: '4x6', text: '4x6' },
        { value: '5x7', text: '5x7' },
        { value: '6x8', text: '6x8' },
        { value: '3x3x3', text: '3x3x3 Cube' }
    ];

    const allTimes = [
        { value: 'all', text: 'All Time Lengths' },
        { value: '45', text: '45s Blitz' },
        { value: '180', text: '3m Standard' },
        { value: '300', text: '5m Speed' },
        { value: '600', text: '10m Relaxed' }
    ];

    const modeConfig = {
        'all': {
            dims: ['4x4', '4x6', '5x7', '6x8', '3x3x3'],
            times: ['45', '180', '300', '600']
        },
        'accumulative': {
            dims: ['4x4', '4x6', '5x7', '6x8'],
            times: ['45', '180', '600']
        },
        'fcfs': {
            dims: ['4x4', '4x6', '5x7', '6x8'],
            times: ['45', '180']
        },
        'split': {
            dims: ['4x4', '4x6', '5x7', '6x8'],
            times: ['45', '180']
        },
        '3d': {
            dims: ['3x3x3'],
            times: ['180', '300', '600']
        }
    };

    function updateFilterDropdowns(source = 'game') {
        const gameSelect = document.getElementById('lb-filter-game');
        const dimsSelect = document.getElementById('lb-filter-dims');
        const timeSelect = document.getElementById('lb-filter-time');
        if (!gameSelect || !dimsSelect || !timeSelect) return;

        let selectedGame = gameSelect.value;
        let selectedDim = dimsSelect.value;
        let selectedTime = timeSelect.value;

        if (source === 'dims') {
            if (selectedDim === '3x3x3') {
                if (selectedGame !== '3d' && selectedGame !== 'all') {
                    selectedGame = '3d';
                    gameSelect.value = '3d';
                }
            } else if (selectedDim !== 'all') {
                if (selectedGame === '3d') {
                    selectedGame = 'all';
                    gameSelect.value = 'all';
                }
            }
        } else if (source === 'time') {
            if (selectedTime === '300') {
                if (selectedGame !== '3d' && selectedGame !== 'all') {
                    selectedGame = '3d';
                    gameSelect.value = '3d';
                }
            } else if (selectedTime === '45') {
                if (selectedGame === '3d') {
                    selectedGame = 'all';
                    gameSelect.value = 'all';
                }
            }
        }

        const cfg = modeConfig[selectedGame] || modeConfig['all'];

        // Determine allowed game modes if dimension or time was changed
        let allowedGames = ['all', 'accumulative', '3d', 'fcfs', 'split'];
        if (selectedDim === '3x3x3' || selectedTime === '300') {
            allowedGames = ['all', '3d'];
        } else if (selectedDim !== 'all') {
            allowedGames = ['all', 'accumulative', 'fcfs', 'split'];
        }
        if (selectedTime === '45') {
            allowedGames = allowedGames.filter(g => g !== '3d');
        } else if (selectedTime === '600') {
            allowedGames = allowedGames.filter(g => g !== 'fcfs' && g !== 'split');
        }

        if (source !== 'game') {
            const currentG = gameSelect.value;
            gameSelect.innerHTML = allGameTypes
                .filter(g => g.value === 'all' || allowedGames.includes(g.value))
                .map(g => `<option value="${g.value}">${g.text}</option>`)
                .join('');
            if (allowedGames.includes(currentG) || currentG === 'all') {
                gameSelect.value = currentG;
            } else {
                gameSelect.value = 'all';
            }
        }

        // Rebuild Dims select
        const allowedDims = cfg.dims;
        const currentDim = dimsSelect.value;
        dimsSelect.innerHTML = '<option value="all">All Sizes</option>' + 
            allDims
                .filter(d => d.value !== 'all' && allowedDims.includes(d.value))
                .map(d => `<option value="${d.value}">${d.text}</option>`)
                .join('');
        if (allowedDims.includes(currentDim) || currentDim === 'all') {
            dimsSelect.value = currentDim;
        } else {
            dimsSelect.value = 'all';
        }

        // Rebuild Time select
        const allowedTimes = cfg.times;
        const currentTime = timeSelect.value;
        timeSelect.innerHTML = '<option value="all">All Time Lengths</option>' + 
            allTimes
                .filter(t => t.value !== 'all' && allowedTimes.includes(t.value))
                .map(t => `<option value="${t.value}">${t.text}</option>`)
                .join('');
        if (allowedTimes.includes(currentTime) || currentTime === 'all') {
            timeSelect.value = currentTime;
        } else {
            timeSelect.value = 'all';
        }

        currentFilters.game_type = gameSelect.value;
        currentFilters.board_dimensions = dimsSelect.value;
        currentFilters.time_limit = timeSelect.value;
    }

    // -- Event Listeners --

    // Tabs
    const tabs = leaderboardPage.querySelectorAll('.lb-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentPeriod = tab.dataset.period;
            fetchLeaderboard();
        });
    });

    // Filters
    document.getElementById('lb-filter-game').addEventListener('change', () => {
        updateFilterDropdowns('game');
        fetchLeaderboard();
    });
    document.getElementById('lb-filter-dims').addEventListener('change', () => {
        updateFilterDropdowns('dims');
        fetchLeaderboard();
    });
    document.getElementById('lb-filter-time').addEventListener('change', () => {
        updateFilterDropdowns('time');
        fetchLeaderboard();
    });

    // Search Controls: only trigger on FIND USER / FIND ME / Enter key
    const searchInput = document.getElementById('lb-search-input');
    const findUserBtn = document.getElementById('lb-find-user-btn');
    const findMeBtn = document.getElementById('lb-find-me-btn');

    function preserveScreenPosition(callback) {
        const pageLb = document.getElementById('page-leaderboards');
        const prevPageScroll = pageLb ? pageLb.scrollTop : 0;
        const prevWinScroll = window.scrollY || document.documentElement.scrollTop || 0;

        callback();

        // Lock scroll position on mobile / desktop
        if (pageLb) pageLb.scrollTop = prevPageScroll;
        window.scrollTo(0, prevWinScroll);
        requestAnimationFrame(() => {
            if (pageLb) pageLb.scrollTop = prevPageScroll;
            window.scrollTo(0, prevWinScroll);
        });
    }

    function executeFindUser(e) {
        if (e) {
            e.preventDefault();
            e.stopPropagation();
        }
        preserveScreenPosition(() => {
            const query = searchInput ? searchInput.value.trim().toLowerCase() : '';
            highlightRows(query);
        });
    }

    function executeFindMe(e) {
        if (e) {
            e.preventDefault();
            e.stopPropagation();
        }
        let user = '';
        if (window.currentUser) {
            if (typeof window.currentUser === 'object' && window.currentUser.username) {
                user = window.currentUser.username;
            } else if (typeof window.currentUser === 'string') {
                user = window.currentUser;
            }
        }
        if (!user && window.user && window.user.username) {
            user = window.user.username;
        }
        if (!user) {
            try {
                const stored = localStorage.getItem('morpheme_username') || sessionStorage.getItem('username');
                if (stored) user = stored;
            } catch (e) {}
        }

        if (user) {
            if (searchInput) searchInput.value = user;
            preserveScreenPosition(() => {
                highlightRows(user.toLowerCase());
            });
        } else {
            alert("Please log in to use FIND ME.");
        }
    }

    if (findUserBtn) {
        findUserBtn.addEventListener('click', executeFindUser);
    }
    if (findMeBtn) {
        findMeBtn.addEventListener('click', executeFindMe);
    }
    if (searchInput) {
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                executeFindUser(e);
            }
        });
    }

    // -- Functions --

    async function fetchLeaderboard() {
        const contentArea = document.getElementById('lb-content-area');
        contentArea.innerHTML = '<div class="lb-loading">Fetching data...</div>';

        try {
            const params = new URLSearchParams({
                period: currentPeriod,
                ...currentFilters
            });

            const response = await fetch(`/api/leaderboard?${params}&t=${Date.now()}`);
            const data = await response.json();

            // Process JSON strings into objects so Replay tool can read them
            const processList = (list) => {
                if (!list) return [];
                return list.map(row => {
                    const r = { ...row };
                    if (r.words_json) {
                        try { r.words = JSON.parse(r.words_json); } catch (e) { }
                    }
                    if (r.board_json) {
                        try { r.board = JSON.parse(r.board_json); } catch (e) { }
                    }
                    // watchRoundHistory expects round_number and game_id
                    // but DB sometimes says round_num or id
                    if (!r.round_number && r.round_num) r.round_number = r.round_num;
                    if (!r.game_id && r.id) r.game_id = r.id;
                    return r;
                });
            };

            data.best_scores = processList(data.best_scores);
            data.best_words = processList(data.best_words);
            data.best_pes = processList(data.best_pes);
            data.best_pcts = processList(data.best_pcts);
            data.best_obscure = processList(data.best_obscure);

            leaderboardData = data;

            // SYNC with global replay system
            window.lastRenderedRounds = [
                ...(data.best_scores || []),
                ...(data.best_words || []),
                ...(data.best_pes || []),
                ...(data.best_pcts || []),
                ...(data.best_obscure || [])
            ];

            renderLeaderboard(data);

        } catch (error) {
            console.error("Leaderboard fetch error:", error);
            contentArea.innerHTML = '<div class="error-msg">Failed to load leaderboards.</div>';
        }
    }

    function renderLeaderboard(data) {
        const contentArea = document.getElementById('lb-content-area');
        contentArea.innerHTML = '';

        const filterGame = document.getElementById('lb-filter-game')?.value || 'all';
        const showType = filterGame === 'all';

        const formatDuration = (sec) => {
            if (!sec && sec !== 0) return '';
            const s = parseInt(sec, 10);
            if (isNaN(s)) return sec;
            return s < 60 ? `${s}s` : (s % 60 === 0 ? `${s / 60}m` : `${(s / 60).toFixed(1)}m`);
        };

        const formatConfigMeta = (dims, sec) => {
            const dur = formatDuration(sec);
            if (dims && dur) return `${dims} · ${dur}`;
            if (dims) return dims;
            if (dur) return dur;
            return '';
        };

        const renderTypeBadge = (type) => {
            if (!showType || !type) return '';
            const label = type === '3d' ? 'Cube' :
                          type === 'accumulative' ? 'Acc' :
                          type === 'fcfs' ? 'FCFS' :
                          type === 'split' ? 'Split' : type;
            const bg = type === '3d' ? 'rgba(235, 68, 90, 0.2)' :
                       type === 'accumulative' ? 'rgba(56, 128, 255, 0.2)' :
                       type === 'fcfs' ? 'rgba(45, 211, 111, 0.2)' :
                       'rgba(152, 116, 248, 0.2)';
            const color = type === '3d' ? '#eb445a' :
                          type === 'accumulative' ? '#3880ff' :
                          type === 'fcfs' ? '#2dd36f' :
                          '#9874f8';
            return `<span style="background:${bg}; color:${color}; padding: 2px 6px; border-radius: 4px; font-size: 0.6rem; font-weight: 800; text-transform: uppercase; margin-left: 8px; vertical-align: middle;">${label}</span>`;
        };

        createTableCard(contentArea, "Highest Single Round Scores", data.best_scores, (row, i) => {
            return `
                <td class="col-rank">#${i + 1}</td>
                <td class="col-user">
                    ${renderUserLink(row)}
                    ${renderTypeBadge(row.game_type)}
                </td>
                <td class="col-val highlight">${row.total_score} pts</td>
                <td class="col-meta">${formatConfigMeta(row.board_dimensions, row.round_duration) || '-'}</td>
                <td class="col-date">${formatDate(row.timestamp)}</td>
                <td class="col-action">
                    ${renderReplayBtn(row)}
                </td>
            `;
        });

        createTableCard(contentArea, "Highest Average Score (Min 3 Games)", data.avg_scores, (row, i) => {
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                 </td>
                 <td class="col-val highlight">${Math.round(row.avg_score)}</td>
                 <td class="col-meta">${row.games} games</td>
                 <td class="col-date">${formatDate(row.last_active)}</td>
                 <td class="col-action"></td>
             `;
        });

        createTableCard(contentArea, "Best Words Played", data.best_words, (row, i) => {
            const metaStr = formatConfigMeta(row.board_dimensions, row.round_duration);
            return `
                <td class="col-rank">#${i + 1}</td>
                <td class="col-user">
                     ${renderUserLink(row)}
                     ${renderTypeBadge(row.game_type)}
                </td>
                <td class="col-val highlight">${row.best_word}</td>
                <td class="col-meta" style="color: #ffd700;">${row.best_word_score} pts${metaStr ? `<br><span style="font-size:0.75rem; color:rgba(var(--text-primary-rgb),0.6);">${metaStr}</span>` : ''}</td>
                <td class="col-date">${formatDate(row.timestamp)}</td>
                 <td class="col-action">
                    ${renderReplayBtn(row)}
                </td>
            `;
        });

        createTableCard(contentArea, "Highest Efficiency (PE)", data.best_pes, (row, i) => {
            const metaStr = formatConfigMeta(row.board_dimensions, row.round_duration);
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                      ${renderTypeBadge(row.game_type)}
                 </td>
                 <td class="col-val highlight">${parseFloat(row.performance_ratio).toFixed(2)}x</td>
                 <td class="col-meta">Pts: ${row.total_score}${metaStr ? ` · ${metaStr}` : ''}<br>Words Found: <span style="${row.pct_found > 50 ? 'color: #ff4a4a; font-weight: 800;' : ''}">${row.pct_found || 0}%</span></td>
                 <td class="col-date">${formatDate(row.timestamp)}</td>
                  <td class="col-action">
                    ${renderReplayBtn(row)}
                </td>
            `;
        });

        createTableCard(contentArea, "Highest Percentage of Words Found", data.best_pcts, (row, i) => {
            const metaStr = formatConfigMeta(row.board_dimensions, row.round_duration);
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                      ${renderTypeBadge(row.game_type)}
                 </td>
                 <td class="col-val highlight" style="${row.pct_found > 50 ? 'color: #ff4a4a;' : ''}">${row.pct_found || 0}%</td>
                 <td class="col-meta">Pts: ${row.total_score}${metaStr ? ` · ${metaStr}` : ''}<br>User Avg: ${row.avg_pct}%</td>
                 <td class="col-date">${formatDate(row.timestamp)}</td>
                  <td class="col-action">
                     ${renderReplayBtn(row)}
                 </td>
            `;
        });

        createTableCard(contentArea, "Highest Avg Percentage of Words Found", data.best_avg_pcts, (row, i) => {
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                 </td>
                 <td class="col-val highlight">${row.avg_pct}%</td>
                 <td class="col-meta">${row.games} games</td>
                 <td class="col-date">-</td>
                 <td class="col-action"></td>
             `;
        });

        createTableCard(contentArea, "Highest number of Hard words found", data.best_obscure, (row, i) => {
            const metaStr = formatConfigMeta(row.board_dimensions, row.round_duration);
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                      ${renderTypeBadge(row.game_type)}
                 </td>
                 <td class="col-val highlight">${row.obscure_count}</td>
                 <td class="col-meta">Pts: ${row.total_score}${metaStr ? ` · ${metaStr}` : ''}</td>
                 <td class="col-date">${formatDate(row.timestamp)}</td>
                  <td class="col-action">
                     ${renderReplayBtn(row)}
                 </td>
            `;
        });

        createTableCard(contentArea, "Peak Ratings Achieved", data.best_ratings, (row, i) => {
            const metaStr = formatConfigMeta(row.board_dimensions, row.round_duration);
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                      ${renderTypeBadge(row.game_type)}
                 </td>
                 <td class="col-val highlight" style="color: #409cff;">${row.max_rating}</td> 
                 <td class="col-meta">${metaStr || 'Peak'}</td>
                 <td class="col-date">${formatDate(row.timestamp)}</td>
                 <td class="col-action"></td>
             `;
        }, false);

        createTableCard(contentArea, "Most Games Played", data.most_games, (row, i) => {
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                      ${renderTypeBadge(row.game_type)}
                 </td>
                 <td class="col-val highlight">${row.game_count}</td> 
                 <td class="col-meta">Games</td>
                 <td class="col-date">${formatDate(row.last_active)}</td>
                 <td class="col-action"></td>
             `;
        });

        createTableCard(contentArea, "Current Top Rated Active Players", data.current_ratings, (row, i) => {
            const metaStr = formatConfigMeta(row.board_dimensions, row.round_duration);
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                      ${renderTypeBadge(row.game_type)}
                 </td>
                 <td class="col-val highlight">${row.rating}</td>
                 <td class="col-meta">${metaStr || 'Current'}</td>
                 <td class="col-date">${formatDate(row.last_active)}</td>
                 <td class="col-action"></td>
             `;
        });

        // Re-apply search if it exists
        const searchInput = document.getElementById('lb-search-input');
        if (searchInput && searchInput.value.trim()) {
            highlightRows(searchInput.value.trim().toLowerCase());
        }
    }

    function createTableCard(container, title, rows, rowRenderer, customClass = '') {
        if (!rows || rows.length === 0) return;

        const card = document.createElement('div');
        card.className = `lb-card ${customClass}`;

        // Generate Rows
        const tableRows = rows.map((row, index) => {
            return `<tr class="lb-row lb-row-${index}" data-username="${(row.username || '').toLowerCase()}">
                ${rowRenderer(row, index)}
            </tr>`;
        }).join('');

        card.innerHTML = `
            <div class="lb-card-header" style="display: flex; justify-content: space-between; align-items: center;">
                <div class="lb-card-header-text">${title}</div>
            </div>
            <div class="lb-table-wrapper">
                <table class="lb-table">
                    <tbody>${tableRows}</tbody>
                </table>
            </div>
        `;

        container.appendChild(card);
    }

    function formatDate(isoStr) {
        if (!isoStr) return '-';
        if (typeof window.formatAppDate === 'function') {
            return window.formatAppDate(isoStr);
        }
        const dateStr = isoStr.includes('Z') || isoStr.includes('+') ? isoStr.replace(' ', 'T') : isoStr.replace(' ', 'T') + 'Z';
        const d = new Date(dateStr);
        if (isNaN(d.getTime())) return isoStr;
        const day = String(d.getDate()).padStart(2, '0');
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const year = d.getFullYear();
        return `${day}/${month}/${year}`;
    }

    function renderUserLink(row) {
        // Colored square logic: use user's rating
        const rating = (row.user_rating !== undefined && row.user_rating !== null && row.user_rating > 0)
            ? row.user_rating
            : ((row.rating !== undefined && row.rating !== null && row.rating > 0) ? row.rating : 1200);

        let color = '#0044ff'; // blue default (1200)
        if (window.getRatingColor) {
            color = window.getRatingColor(rating);
        } else {
            // Fallback if getRatingColor not avail globally yet
            if (rating >= 2400) color = '#e74c3c'; // red
            else if (rating >= 2000) color = '#e67e22'; // orange
            else if (rating >= 1800) color = '#f1c40f'; // yellow
            else if (rating >= 1500) color = '#2ecc71'; // green
            else if (rating >= 1200) color = '#0044ff'; // blue
            else color = '#9b59b6'; // purple/basic
        }

        const flagHtml = window.getFlagHtml ? window.getFlagHtml(row.country_flag) : (row.country_flag || '');

        return `
            <div class="lb-user-cell" onclick="window.showMiniProfile('${row.username}')">
                <div class="rating-square" style="background-color: ${color};"></div>
                <span class="user-flag">${flagHtml}</span>
                <span class="username">${row.username}</span>
            </div>
        `;
    }

    function renderReplayBtn(row) {
        // Assuming we have window.watchRoundHistory(roomId, roundNum)
        if (!row.room_id || !row.round_number) return '';
        const gameId = row.game_id || row.id || 'null';
        return `
            <button class="lb-replay-btn" title="Watch Replay" 
                onclick="window.openReplayModal('${row.room_id}', ${row.round_number}, ${gameId})">
                ▶
            </button>
        `;
    }

    // Helper to open replay modal specifically (linking to existing logic)
    // We attach this to window so button onclick works
    window.openReplayModal = function (roomId, roundNum, gameId = null) {
        console.log(`Opening replay for ${roomId} - Round ${roundNum} (GameID: ${gameId})`);
        if (window.watchRoundHistory) {
            // Default to interactive replay (false)
            window.watchRoundHistory(roomId, roundNum, false, gameId);
        } else {
            alert("Replay viewer context not loaded. Please ensure you are logged in.");
        }
    };

    function highlightRows(query) {
        const allRows = document.querySelectorAll('.lb-row');
        if (!query) {
            allRows.forEach(row => row.classList.remove('highlight-search'));
            return;
        }

        // 1. Mark matching rows across all tables
        allRows.forEach(row => {
            const u = (row.dataset.username || '').toLowerCase();
            if (u === query) {
                row.classList.add('highlight-search');
            } else {
                row.classList.remove('highlight-search');
            }
        });

        // 2. Position the matching row in the middle of each table card
        const scrollMatchingRows = () => {
            const cards = document.querySelectorAll('.lb-card');
            cards.forEach(card => {
                const matchingRow = card.querySelector('.lb-row.highlight-search');
                const wrapper = card.querySelector('.lb-table-wrapper');
                if (matchingRow && wrapper) {
                    const rowRect = matchingRow.getBoundingClientRect();
                    const wrapRect = wrapper.getBoundingClientRect();
                    const currentScroll = wrapper.scrollTop;
                    const delta = (rowRect.top + (rowRect.height / 2)) - (wrapRect.top + (wrapRect.height / 2));
                    const targetScroll = Math.max(0, currentScroll + delta);
                    wrapper.scrollTop = targetScroll;
                    wrapper.scrollTo({
                        top: targetScroll,
                        behavior: 'smooth'
                    });
                }
            });
        };

        scrollMatchingRows();
        requestAnimationFrame(scrollMatchingRows);
        setTimeout(scrollMatchingRows, 50);
        setTimeout(scrollMatchingRows, 150);
    }

    function initMobileScrollbar() {
        const page = document.getElementById('page-leaderboards');
        const track = document.getElementById('lb-mobile-scrollbar-track');
        const thumb = document.getElementById('lb-mobile-scrollbar-thumb');
        if (!page || !track || !thumb) return;

        let isDragging = false;
        let startY = 0;
        let startThumbTop = 0;
        let rafId = null;

        function updateThumb() {
            if (isDragging) return;
            if (window.innerWidth > 992) {
                track.style.display = 'none';
                return;
            }

            const scrollHeight = page.scrollHeight;
            const clientHeight = page.clientHeight;
            const scrollTop = page.scrollTop;

            if (scrollHeight <= clientHeight + 15 || clientHeight <= 0) {
                track.style.display = 'none';
                return;
            }
            track.style.display = 'block';

            const trackHeight = track.clientHeight;
            const ratio = Math.min(1, clientHeight / scrollHeight);
            const thumbHeight = Math.max(35, Math.min(trackHeight, trackHeight * ratio));
            thumb.style.height = `${thumbHeight}px`;

            const maxScrollTop = scrollHeight - clientHeight;
            const maxThumbTop = Math.max(0, trackHeight - thumbHeight);
            const thumbTop = maxScrollTop > 0 ? (scrollTop / maxScrollTop) * maxThumbTop : 0;
            thumb.style.top = `${thumbTop}px`;
        }

        function scheduleUpdate() {
            if (rafId) return;
            rafId = requestAnimationFrame(() => {
                rafId = null;
                updateThumb();
            });
        }

        page.addEventListener('scroll', scheduleUpdate, { passive: true });
        window.addEventListener('resize', scheduleUpdate, { passive: true });

        if (window.ResizeObserver) {
            const ro = new ResizeObserver(scheduleUpdate);
            ro.observe(page);
        }
        if (window.MutationObserver) {
            const mo = new MutationObserver(scheduleUpdate);
            mo.observe(page, { childList: true, subtree: true });
        }

        function onDragStart(e) {
            isDragging = true;
            thumb.classList.add('dragging');
            document.body.style.userSelect = 'none';
            startY = e.touches ? e.touches[0].clientY : e.clientY;
            startThumbTop = parseFloat(thumb.style.top) || 0;

            document.addEventListener('mousemove', onDragMove, { passive: false });
            document.addEventListener('mouseup', onDragEnd);
            document.addEventListener('touchmove', onDragMove, { passive: false });
            document.addEventListener('touchend', onDragEnd);
            document.addEventListener('touchcancel', onDragEnd);

            if (e.cancelable !== false) e.preventDefault();
        }

        function onDragMove(e) {
            if (!isDragging) return;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            const deltaY = clientY - startY;

            const trackHeight = track.clientHeight;
            const thumbHeight = thumb.offsetHeight;
            const maxThumbTop = Math.max(0, trackHeight - thumbHeight);

            let newThumbTop = Math.max(0, Math.min(maxThumbTop, startThumbTop + deltaY));
            thumb.style.top = `${newThumbTop}px`;

            const scrollHeight = page.scrollHeight;
            const clientHeight = page.clientHeight;
            const maxScrollTop = scrollHeight - clientHeight;
            if (maxThumbTop > 0) {
                page.scrollTop = (newThumbTop / maxThumbTop) * maxScrollTop;
            }

            if (e.cancelable !== false) e.preventDefault();
        }

        function onDragEnd() {
            if (isDragging) {
                isDragging = false;
                thumb.classList.remove('dragging');
                document.body.style.userSelect = '';

                document.removeEventListener('mousemove', onDragMove);
                document.removeEventListener('mouseup', onDragEnd);
                document.removeEventListener('touchmove', onDragMove);
                document.removeEventListener('touchend', onDragEnd);
                document.removeEventListener('touchcancel', onDragEnd);

                scheduleUpdate();
            }
        }

        thumb.addEventListener('mousedown', onDragStart);
        thumb.addEventListener('touchstart', onDragStart, { passive: false });

        track.addEventListener('click', (e) => {
            if (e.target === thumb) return;
            const rect = track.getBoundingClientRect();
            const clickY = e.clientY - rect.top;
            const trackHeight = track.clientHeight;
            const thumbHeight = thumb.offsetHeight;
            const maxThumbTop = Math.max(0, trackHeight - thumbHeight);
            const targetThumbTop = Math.max(0, Math.min(maxThumbTop, clickY - (thumbHeight / 2)));
            
            const scrollHeight = page.scrollHeight;
            const clientHeight = page.clientHeight;
            const maxScrollTop = scrollHeight - clientHeight;
            if (maxThumbTop > 0) {
                page.scrollTo({
                    top: (targetThumbTop / maxThumbTop) * maxScrollTop,
                    behavior: 'smooth'
                });
            }
        });

        scheduleUpdate();
        setTimeout(scheduleUpdate, 200);
        setTimeout(scheduleUpdate, 600);
    }

    initMobileScrollbar();

    // Initial Fetch
    // We delay slightly to ensure global styles/scripts are ready
    setTimeout(() => {
        // Check if we are already viewing the page?
        // The script loads once. We should fetch if the page is active.
        const page = document.getElementById('page-leaderboards');
        if (page && page.classList.contains('active')) {
            fetchLeaderboard();
        }

        // Also hook into navigation changes to refresh
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.target.classList.contains('active')) {
                    fetchLeaderboard();
                    initMobileScrollbar();
                }
            });
        });
        observer.observe(page, { attributes: true, attributeFilter: ['class'] });

    }, 100);

});
