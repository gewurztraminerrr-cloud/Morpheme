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
                        <option value="all">All Speeds</option>
                        <option value="45">45s Blitz</option>
                        <option value="180">3m Standard</option>
                        <option value="300">5m Speed</option>
                        <option value="600">10m Relaxed</option>
                    </select>
                </div>
                
                <div class="lb-search">
                    <input type="text" id="lb-search-input" placeholder="Find yourself..." autocomplete="off">
                </div>
            </div>

            <div id="lb-content-area" class="lb-grid">
                <!-- Grid of Cards (Best Score, Best Word, etc.) -->
                <div class="lb-loading">Loading rankings...</div>
            </div>
            
            <div class="lb-attribution">
                * Leaderboards update in real-time. Only non-24h rooms are tracked.
            </div>
        </div>
    `;

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
    document.getElementById('lb-filter-game').addEventListener('change', (e) => {
        currentFilters.game_type = e.target.value;
        fetchLeaderboard();
    });
    document.getElementById('lb-filter-dims').addEventListener('change', (e) => {
        currentFilters.board_dimensions = e.target.value;
        fetchLeaderboard();
    });
    document.getElementById('lb-filter-time').addEventListener('change', (e) => {
        currentFilters.time_limit = e.target.value;
        fetchLeaderboard();
    });

    // Search
    const searchInput = document.getElementById('lb-search-input');
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        highlightRows(query);
    });

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

        const filterGame = document.getElementById('lb-filter-game')?.value;
        const showType = filterGame === 'all';

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
                <td class="col-meta">${row.round_duration < 60 ? row.round_duration + 's' : (row.round_duration / 60) + 'm'}</td>
                <td class="col-date">${formatDate(row.timestamp)}</td>
                <td class="col-action">
                    ${renderReplayBtn(row)}
                </td>
            `;
        });

        createTableCard(contentArea, "Best Words Played", data.best_words, (row, i) => {
            return `
                <td class="col-rank">#${i + 1}</td>
                <td class="col-user">
                     ${renderUserLink(row)}
                     ${renderTypeBadge(row.game_type)}
                </td>
                <td class="col-val highlight">${row.best_word}</td>
                <td class="col-meta" style="color: #ffd700;">${row.best_word_score} pts</td>
                <td class="col-date">${formatDate(row.timestamp)}</td>
                 <td class="col-action">
                    ${renderReplayBtn(row)}
                </td>
            `;
        });

        createTableCard(contentArea, "Highest Efficiency (PE)", data.best_pes, (row, i) => {
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                      ${renderTypeBadge(row.game_type)}
                 </td>
                 <td class="col-val highlight">${parseFloat(row.performance_ratio).toFixed(2)}x</td>
                 <td class="col-meta">Pts: ${row.total_score}<br>Words Found: <span style="${row.pct_found > 50 ? 'color: #ff4a4a; font-weight: 800;' : ''}">${row.pct_found || 0}%</span></td>
                 <td class="col-date">${formatDate(row.timestamp)}</td>
                  <td class="col-action">
                    ${renderReplayBtn(row)}
                </td>
            `;
        });

        createTableCard(contentArea, "Highest Percentage of Words Found", data.best_pcts, (row, i) => {
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                      ${renderTypeBadge(row.game_type)}
                 </td>
                 <td class="col-val highlight" style="${row.pct_found > 50 ? 'color: #ff4a4a;' : ''}">${row.pct_found || 0}%</td>
                 <td class="col-meta">Pts: ${row.total_score}</td>
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

        createTableCard(contentArea, "Highest number of Obscure words found", data.best_obscure, (row, i) => {
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                      ${renderTypeBadge(row.game_type)}
                 </td>
                 <td class="col-val highlight">${row.obscure_count}</td>
                 <td class="col-meta">Pts: ${row.total_score}</td>
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

        createTableCard(contentArea, "Peak Ratings Achieved", data.best_ratings, (row, i) => {
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                      ${renderTypeBadge(row.game_type)}
                 </td>
                 <td class="col-val highlight" style="color: #409cff;">${row.max_rating}</td> 
                 <td class="col-meta">Peak</td>
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
            return `
                 <td class="col-rank">#${i + 1}</td>
                 <td class="col-user">
                      ${renderUserLink(row)}
                      ${renderTypeBadge(row.game_type)}
                 </td>
                 <td class="col-val highlight">${row.rating}</td>
                 <td class="col-meta">Current</td>
                 <td class="col-date">${formatDate(row.last_active)}</td>
                 <td class="col-action"></td>
             `;
        }, true); // Enable local search

        // Re-apply search if it exists
        const searchInput = document.getElementById('lb-search-input');
        if (searchInput.value) {
            highlightRows(searchInput.value.toLowerCase());
        }
    }

    function createTableCard(container, title, rows, rowRenderer, includeSearch = false, customClass = '') {
        if (!rows || rows.length === 0) return;

        const card = document.createElement('div');
        card.className = `lb-card ${customClass}`;

        // Generate Rows
        const tableRows = rows.map((row, index) => {
            return `<tr class="lb-row lb-row-${index}" data-username="${(row.username || '').toLowerCase()}">
                ${rowRenderer(row, index)}
            </tr>`;
        }).join('');

        // Header Structure
        let headerHTML = `<div class="lb-card-header-text">${title}</div>`;
        if (includeSearch) {
            headerHTML += `
                <div class="lb-card-search">
                    <input type="text" placeholder="Username" class="lb-local-input">
                    <button class="lb-local-btn">FIND ME</button>
                    <span class="lb-local-msg"></span>
                </div>
             `;
        }

        card.innerHTML = `
            <div class="lb-card-header" style="display: flex; justify-content: space-between; align-items: center;">
                ${headerHTML}
            </div>
            <div class="lb-table-wrapper">
                <table class="lb-table">
                    <tbody>${tableRows}</tbody>
                </table>
            </div>
        `;

        // Attach Search Logic
        if (includeSearch) {
            const btn = card.querySelector('.lb-local-btn');
            const input = card.querySelector('.lb-local-input');
            const msg = card.querySelector('.lb-local-msg');

            // Pre-fill if logged in (handle cases where window.currentUser is user object or username string)
            if (window.currentUser) {
                if (typeof window.currentUser === 'object' && window.currentUser.username) {
                    input.value = window.currentUser.username;
                } else if (typeof window.currentUser === 'string') {
                    input.value = window.currentUser;
                }
            }

            const performSearch = () => {
                const query = input.value.trim().toLowerCase();
                if (!query) return;

                const targetRow = card.querySelector(`.lb-row[data-username="${query}"]`);
                if (targetRow) {
                    // Highlight
                    card.querySelectorAll('.lb-row').forEach(r => r.classList.remove('highlight-search'));
                    targetRow.classList.add('highlight-search');
                    targetRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    msg.textContent = '';
                } else {
                    msg.textContent = 'Not found (in top 1000)';
                    msg.style.color = '#e74c3c';
                    setTimeout(() => msg.textContent = '', 2000);
                }
            };

            btn.addEventListener('click', performSearch);
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') performSearch();
            });
        }

        container.appendChild(card);
    }

    function formatDate(isoStr) {
        if (!isoStr) return '-';
        // Handle "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS"
        const dateStr = isoStr.replace(' ', 'T');
        const d = new Date(dateStr);
        if (isNaN(d.getTime())) return isoStr; // Fallback

        // Check if it's today
        const now = new Date();
        const isToday = d.getDate() === now.getDate() &&
            d.getMonth() === now.getMonth() &&
            d.getFullYear() === now.getFullYear();

        if (isToday) {
            return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }

        // Otherwise "Mon DD"
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    }

    function renderUserLink(row) {
        // Colored square logic
        const rating = row.user_rating || row.rating || 1200;
        let color = '#95a5a6'; // gray default
        if (window.getRatingColor) {
            color = window.getRatingColor(rating);
        } else {
            // Fallback if getRatingColor not avail globally yet
            if (rating >= 2400) color = '#e74c3c'; // red
            else if (rating >= 2000) color = '#e67e22'; // orange
            else if (rating >= 1800) color = '#f1c40f'; // yellow
            else if (rating >= 1500) color = '#2ecc71'; // green
            else if (rating >= 1200) color = '#3498db'; // blue
            else color = '#9b59b6'; // purple/basic
        }

        const flag = row.country_flag ? row.country_flag : '';
        // If avatar isn't supported yet in API response fully, fallback to flag/square

        return `
            <div class="lb-user-cell" onclick="window.showMiniProfile('${row.username}')">
                <div class="rating-square" style="background-color: ${color};"></div>
                <span class="user-flag">${flag}</span>
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
        const rows = document.querySelectorAll('.lb-row');
        rows.forEach(row => {
            if (row.dataset.username.includes(query)) {
                row.classList.add('highlight-search');
                row.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } else {
                row.classList.remove('highlight-search');
            }
        });
    }

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
                }
            });
        });
        observer.observe(page, { attributes: true, attributeFilter: ['class'] });

    }, 100);

});
