var _steadyLoaderRafId = null;
var _fullListAllWords = [];
var _fullListRenderedStart = 0;
var _fullListRenderedEnd = 0;
var _virtualProgressCount = 0;
window._cachedFullWordLists = window._cachedFullWordLists || {};

function initToolsModules() {
    const inits = [
        setupToolsNavigation, setupProfileTool, setupComboChecker, setupListsTool,
        setupSequenceTool, setupManualTool, setupRandomWordTool, setupWotdTool,
        setupSubanagramsTool, setupIsValidTool, setupPrivateMessaging, setupMiniProfileModal,
        setupImageLightbox, setupUnscrambleTool, setupFindCountTool, setupPersonalTimer
    ];
    inits.forEach(fn => {
        try {
            if (typeof fn === 'function') fn();
        } catch (e) {
            console.error('[Tools Init Error]', fn.name, e);
        }
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initToolsModules);
} else {
    initToolsModules();
}

// NEW: Global UTC Timestamp Parser to prevent local timezone offsets
window.parseUTCTimestamp = function(isoStr) {
    if (!isoStr) return new Date();
    if (typeof isoStr === 'number') return new Date(isoStr);
    const dateStr = isoStr.includes('Z') || isoStr.includes('+') ? isoStr.replace(' ', 'T') : isoStr.replace(' ', 'T') + 'Z';
    return new Date(dateStr);
};

// NEW: Global Tool Switcher Helper
window.showTool = function(toolId) {
    // Save lists scroll position before navigating away from Lists tool
    const currentActivePane = document.querySelector('.tool-pane.active');
    if (currentActivePane && currentActivePane.id === 'tool-lists') {
        const listScrollArea = document.getElementById('main-list-results');
        if (listScrollArea) {
            window._savedListsScrollTop = listScrollArea.scrollTop;
        }
    }

    const sidebar = document.querySelector('#page-tools .tools-sidebar');
    const content = document.querySelector('#page-tools .tools-content');
    
    // Update active class on buttons
    if (sidebar) {
        sidebar.querySelectorAll('.tool-nav-btn').forEach(b => {
            if (b.dataset.tool === toolId) b.classList.add('active');
            else b.classList.remove('active');
        });
    }

    // Update active class on panes
    if (content) {
        content.querySelectorAll('.tool-pane').forEach(p => {
            if (p.id === `tool-${toolId}`) p.classList.add('active');
            else p.classList.remove('active');
        });
        if (toolId === 'sequence' || toolId === 'subanagrams' || toolId === 'lists') {
            content.classList.add('no-outer-scroll');
        } else {
            content.classList.remove('no-outer-scroll');
        }
    }

    // Trigger lazy loads
    if (toolId === 'profile') {
        if (typeof refreshProfileTool === 'function') refreshProfileTool();
    }
    if (toolId === 'lists') {
        if (!listsDataLoaded) {
            if (typeof fetchListsData === 'function') fetchListsData();
        } else {
            // Keep the loaded list present and restore saved scroll position
            const listScrollArea = document.getElementById('main-list-results');
            if (listScrollArea && typeof window._savedListsScrollTop === 'number') {
                setTimeout(() => {
                    listScrollArea.scrollTop = window._savedListsScrollTop;
                }, 30);
            }
        }
    }
    if (toolId === 'wotd') {
        if (typeof updateWotd === 'function') updateWotd();
    }
    if (toolId === 'manual') {
        fetch('/api/tools/flag_manual', { method: 'POST' }).catch(e => console.error(e));
    }
    if (toolId === 'find-count') {
        if (typeof loadRandomSuggestedWords === 'function') loadRandomSuggestedWords(false);
    }
    if (toolId === 'change-account') {
        if (typeof loadAccountCredentialsInfo === 'function') loadAccountCredentialsInfo();
    }
    if (toolId === 'unscramble') {
        unscrambleState.history = [];
        try { localStorage.removeItem('morpheme_unscramble_history'); } catch(e) {}
        if (!unscrambleState.jumbled && !unscrambleState.isLoading) {
            if (typeof startNewUnscramble === 'function') {
                startNewUnscramble();
            }
        } else {
            renderUnscrambleFound();
        }
    }

    // Scroll tools content into view horizontally to the right pane on mobile with smooth sliding
    const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    if (isMobile) {
        const layoutEl = document.querySelector('#page-tools .tools-split-layout');
        if (layoutEl) {
            layoutEl.scrollTo({ left: layoutEl.clientWidth || layoutEl.scrollWidth, behavior: 'smooth' });
        }
    }
};

function setupToolsNavigation() {
    const sidebar = document.querySelector('#page-tools .tools-sidebar');
    if (!sidebar) return;

    sidebar.addEventListener('click', (e) => {
        const btn = e.target.closest('.tool-nav-btn');
        if (!btn) return;

        const toolId = btn.dataset.tool;
        if (toolId) {
            window.showTool(toolId);
        }
    });

    // Mobile Layout snapping on navigation to Tools page
    const toolsPage = document.getElementById('page-tools');
    if (toolsPage) {
        const observer = new MutationObserver(() => {
            if (toolsPage.classList.contains('active')) {
                const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
                if (isMobile) {
                    setTimeout(() => {
                        const layoutEl = document.querySelector('#page-tools .tools-split-layout');
                        if (layoutEl) layoutEl.scrollTo({ left: 0, behavior: 'smooth' });
                    }, 100);
                }
            }
        });
        observer.observe(toolsPage, {
            attributes: true,
            attributeFilter: ['class']
        });
    }

    // Mobile touch swipe handling for sliding back to tools list
    const toolsContent = document.querySelector('#page-tools .tools-content');
    const toolsSidebar = document.querySelector('#page-tools .tools-sidebar');
    if (toolsContent && toolsSidebar) {
        let touchStartX = 0;
        let touchStartY = 0;
        let startedInTable = false;
        
        toolsContent.addEventListener('touchstart', (e) => {
            if (e.changedTouches.length > 0) {
                touchStartX = e.changedTouches[0].screenX;
                touchStartY = e.changedTouches[0].screenY;
                startedInTable = !!e.target.closest('.horizontal-scroll-container');
            }
        }, { passive: true });
        
        toolsContent.addEventListener('touchend', (e) => {
            if (startedInTable || e.target.closest('.horizontal-scroll-container')) {
                return; // Ignore swipes that start or end inside the scrollable tables
            }
            
            const touchEndX = e.changedTouches[0].screenX;
            const touchEndY = e.changedTouches[0].screenY;
            const diffX = touchEndX - touchStartX;
            const diffY = touchEndY - touchStartY;
            
            // If swiped right (diffX > 80) and horizontal movement was dominant
            if (diffX > 80 && Math.abs(diffX) > Math.abs(diffY)) {
                const layoutEl = document.querySelector('#page-tools .tools-split-layout');
                if (layoutEl) {
                    layoutEl.scrollTo({ left: 0, behavior: 'smooth' });
                }
            }
        }, { passive: true });
    }
}

// Global snap enforcement on resize for all tools/settings/mods split layouts
(function _setupToolsSplitSnapTracker() {
    function enforceSnapForLayout(layoutEl) {
        if (!layoutEl) return;
        const w = layoutEl.clientWidth;
        if (!w || w <= 0) return;
        const currentLeft = layoutEl.scrollLeft;
        const activePane = layoutEl.querySelector('.tools-content .tool-pane.active');
        const targetLeft = (activePane && currentLeft > w * 0.25) ? w : (currentLeft >= w * 0.5 ? w : 0);
        if (Math.abs(currentLeft - targetLeft) > 1) {
            layoutEl.scrollLeft = targetLeft;
        }
    }

    function _attachSnapListeners() {
        window.addEventListener('resize', () => {
            if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA')) return;
            const isMobile = window.innerWidth <= 900 || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
            if (!isMobile) return;
            document.querySelectorAll('.tools-split-layout').forEach(layoutEl => {
                requestAnimationFrame(() => enforceSnapForLayout(layoutEl));
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _attachSnapListeners);
    } else {
        _attachSnapListeners();
    }
})();

// High-sensitivity kinetic scroll for mobile Tools category sidebars
(function setupSidebarFlickSensitivity() {
    function initSidebarSensitivity(sidebar) {
        if (!sidebar || sidebar._flickSensitivityInit) return;
        sidebar._flickSensitivityInit = true;

        let startY = 0;
        let lastY = 0;
        let lastTime = 0;
        let velocity = 0;
        let isTouching = false;

        sidebar.addEventListener('touchstart', (e) => {
            if (e.touches.length !== 1) return;
            isTouching = true;
            startY = e.touches[0].clientY;
            lastY = startY;
            lastTime = performance.now();
            velocity = 0;
        }, { passive: true });

        sidebar.addEventListener('touchmove', (e) => {
            if (!isTouching || e.touches.length !== 1) return;
            const currentY = e.touches[0].clientY;
            const currentTime = performance.now();
            const dt = currentTime - lastTime;
            if (dt > 8) {
                const dy = currentY - lastY;
                velocity = dy / dt; // px/ms
                lastY = currentY;
                lastTime = currentTime;
            }
        }, { passive: true });

        sidebar.addEventListener('touchend', () => {
            if (!isTouching) return;
            isTouching = false;
            
            // If user flicked with noticeable velocity (> 0.15 px/ms)
            if (Math.abs(velocity) > 0.15) {
                // Generous momentum multiplier so light flick glides to bottom
                const boost = velocity * 650;
                const targetScroll = Math.max(0, Math.min(sidebar.scrollHeight - sidebar.clientHeight, sidebar.scrollTop - boost));
                
                sidebar.scrollTo({
                    top: targetScroll,
                    behavior: 'smooth'
                });
            }
        }, { passive: true });
    }

    function _initAllSidebars() {
        document.querySelectorAll('.tools-sidebar').forEach(initSidebarSensitivity);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _initAllSidebars);
    } else {
        _initAllSidebars();
    }
})();


function setupComboChecker() {
    const searchBtn = document.getElementById('combo-search-btn');
    const input = document.getElementById('combo-input');

    if (searchBtn) {
        searchBtn.addEventListener('click', runComboSearch);
    }

    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runComboSearch();
        });
    }

    // Prevent horizontal scroll/swipe chaining to the parent .tools-split-layout and enable direct touch dragging across tables
    const containers = [document.getElementById('mp-container'), document.getElementById('lic-container')];
    containers.forEach(container => {
        if (!container) return;

        let startX = 0;
        let startY = 0;
        let startScrollLeft = 0;
        let isHorizontalSwipe = false;
        let touchIdentified = false;

        container.addEventListener('touchstart', (e) => {
            if (e.touches.length > 0) {
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                startScrollLeft = container.scrollLeft;
                isHorizontalSwipe = false;
                touchIdentified = false;
            }
        }, { passive: true });

        container.addEventListener('touchmove', (e) => {
            if (e.touches.length > 0) {
                const currentX = e.touches[0].clientX;
                const currentY = e.touches[0].clientY;
                const diffX = startX - currentX;
                const diffY = Math.abs(currentY - startY);

                if (!touchIdentified) {
                    if (Math.abs(diffX) > 5 || diffY > 5) {
                        isHorizontalSwipe = Math.abs(diffX) > diffY;
                        touchIdentified = true;
                    }
                }

                if (isHorizontalSwipe) {
                    container.scrollLeft = startScrollLeft + diffX;
                    e.stopPropagation();
                    if (e.cancelable) e.preventDefault();
                }
            }
        }, { passive: false });

        container.addEventListener('wheel', (e) => {
            if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
                e.stopPropagation();
            }
        }, { passive: true });
    });
}

window.scrollContainerLeft = function(containerId) {
    const el = document.getElementById(containerId);
    if (el) {
        el.scrollBy({ left: -220, behavior: 'smooth' });
    }
};

window.scrollContainerRight = function(containerId) {
    const el = document.getElementById(containerId);
    if (el) {
        el.scrollBy({ left: 220, behavior: 'smooth' });
    }
};

window._comboClientCache = window._comboClientCache || new Map();

async function runComboSearch() {
    const inputEl = document.getElementById('combo-input');
    const dictEl = document.getElementById('combo-dict');
    const resultsContainer = document.getElementById('combo-results');

    if (inputEl) inputEl.blur(); // Dismiss virtual keyboard before re-rendering results
    const searchTerm = inputEl.value.trim().toUpperCase();
    if (!searchTerm || searchTerm.length < 3) return;

    const mpContainer = document.getElementById('mp-container');
    const licContainer = document.getElementById('lic-container');

    const cacheKey = `${searchTerm}_${dictEl.value}`;
    if (window._comboClientCache.has(cacheKey)) {
        const cachedData = window._comboClientCache.get(cacheKey);
        resultsContainer.classList.remove('hidden');
        if (mpContainer) mpContainer.innerHTML = '';
        if (licContainer) licContainer.innerHTML = '';
        renderGroups(cachedData.mp_groups, 'mp-container', 'MP');
        renderGroups(cachedData.lic_groups, 'lic-container', 'LIC');
        return;
    }

    if (mpContainer) mpContainer.innerHTML = '<div class="loading-spinner">Searching...</div>';
    if (licContainer) licContainer.innerHTML = '';

    resultsContainer.classList.remove('hidden');

    try {
        const response = await fetch('/api/tools/combo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ search_term: searchTerm, dictionary: dictEl.value })
        });

        const data = await response.json();
        if (data.error) {
            alert(data.error);
            return;
        }

        window._comboClientCache.set(cacheKey, data);

        if (mpContainer) mpContainer.innerHTML = '';
        renderGroups(data.mp_groups, 'mp-container', 'MP');
        renderGroups(data.lic_groups, 'lic-container', 'LIC');

    } catch (error) {
        console.error('Combo Search Error:', error);
        if (mpContainer) mpContainer.innerHTML = '<div class="error-msg">Search failed.</div>';
    }
}

function renderGroups(groupsData, containerId, type) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const keys = Object.keys(groupsData).map(Number).sort((a, b) => a - b);
    let foundAny = false;

    keys.forEach(key => {
        const words = groupsData[key];
        if (!words || words.length === 0) return;
        foundAny = true;

        const label = type === 'MP' ? `${key}MP` : `${key}LIC`;
        const colId = `combo-${type.toLowerCase()}-${key}`;
        
        const initialBatch = words.slice(0, 100);
        let renderedCount = initialBatch.length;

        const colDiv = document.createElement('div');
        colDiv.className = 'group-column';
        colDiv.innerHTML = `
            <div class="group-header">${label}</div>
            <div class="list-scroll-area-wrapper" style="position: relative; flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; width: 100%;">
                <div class="list-scroll-area group-table-container" id="${colId}-scroll" style="height: 100%; overflow-y: auto; padding: 5px 10px;">
                    <div class="group-word-list" id="${colId}-list">
                        ${initialBatch.map(w => `<div class="group-row"><span class="clickable-word-link" onclick="window.lookupWord('${w}', event)">${w}</span></div>`).join('')}
                    </div>
                </div>
                <div class="custom-scrollbar-track" id="${colId}-track">
                    <div class="custom-scrollbar-thumb" id="${colId}-thumb"></div>
                </div>
            </div>
        `;
        container.appendChild(colDiv);

        const scrollEl = colDiv.querySelector(`#${colId}-scroll`);
        const listEl = colDiv.querySelector(`#${colId}-list`);

        if (scrollEl && listEl && words.length > renderedCount) {
            scrollEl.addEventListener('scroll', () => {
                if (renderedCount < words.length && scrollEl.scrollTop + scrollEl.clientHeight >= scrollEl.scrollHeight - 350) {
                    const nextBatch = words.slice(renderedCount, renderedCount + 100);
                    renderedCount += nextBatch.length;
                    const html = nextBatch.map(w => `<div class="group-row"><span class="clickable-word-link" onclick="window.lookupWord('${w}', event)">${w}</span></div>`).join('');
                    listEl.insertAdjacentHTML('beforeend', html);
                }
            }, { passive: true });
        }

        initCustomScrollbarForElement(`${colId}-scroll`, `${colId}-track`, `${colId}-thumb`);
    });

    if (!foundAny) {
        container.innerHTML = `<div class="no-results">No ${type === 'MP' ? 'connections within 6MP' : 'letters in common'} found.</div>`;
    }
}

// --- Mini Profile Logic ---

function setupMiniProfileModal() {
    const modal = document.getElementById('mini-profile-modal');
    const closeBtn = document.getElementById('mini-profile-close');

    if (modal && closeBtn) {
        const closeModal = () => {
            modal.classList.add('hidden');
            modal.classList.remove('forced-show');
        };
        closeBtn.onclick = closeModal;
        modal.onclick = (e) => {
            if (e.target === modal) closeModal();
        };
    }
}

function formatLastVisited(lastVisitedStr, isOnline) {
    if (isOnline) {
        return 'Currently Online';
    }
    if (!lastVisitedStr) return '-';
    const visitedDate = new Date(lastVisitedStr.endsWith('Z') ? lastVisitedStr : lastVisitedStr + 'Z');
    if (isNaN(visitedDate.getTime())) return '-';
    const now = new Date();
    const diffMs = Math.max(0, now - visitedDate);
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    let durationStr = '';
    if (diffMins < 1) {
        durationStr = '< 1m';
    } else if (diffMins < 60) {
        durationStr = `${diffMins}m`;
    } else if (diffHours < 24) {
        durationStr = `${diffHours}h`;
    } else if (diffDays < 30) {
        durationStr = `${diffDays}d`;
    } else {
        const years = now.getFullYear() - visitedDate.getFullYear();
        const months = (now.getMonth() + 1) - (visitedDate.getMonth() + 1) + (years * 12);
        if (months >= 12) {
            const y = Math.floor(months / 12);
            const m = months % 12;
            durationStr = `${y}y${m > 0 ? ` ${m}m` : ''}`;
        } else {
            durationStr = months > 0 ? `${months}m` : `${diffDays}d`;
        }
    }

    const formattedDate = typeof window.formatAppDate === 'function' ? window.formatAppDate(visitedDate) : visitedDate.toLocaleDateString();
    return `${formattedDate} (${durationStr})`;
}

window.showMiniProfile = async function (username) {
    if (!username) return;
    console.log(`[showMiniProfile] Attempting to open profile for: ${username}`);
    const modal = document.getElementById('mini-profile-modal');
    if (!modal) {
        console.error('[showMiniProfile] mini-profile-modal NOT FOUND in DOM!');
        return;
    }

    try {
        const response = await fetch(`/api/profile/${encodeURIComponent(username)}`);
        const data = await response.json();
        console.log('[showMiniProfile] Data received:', data);

        if (data.error) {
            console.error('[showMiniProfile] API error:', data.error);
            return;
        }

        // Populate Modal Basic Info
        const nameEl = document.getElementById('mini-profile-username');
        if (nameEl) nameEl.innerText = data.username;
        const fullNEl = document.getElementById('mini-profile-fullname');
        if (fullNEl) fullNEl.innerText = data.full_name || '-';

        // Stats: Games, Wins, Win Rate
        const gamesEl = document.getElementById('mini-profile-games');
        if (gamesEl) gamesEl.innerText = data.games_played || 0;
        const winsEl = document.getElementById('mini-profile-wins');
        if (winsEl) winsEl.innerText = data.wins || 0;

        const winRateEl = document.getElementById('mini-profile-winrate');
        if (winRateEl) {
            const games = data.games_played || 0;
            const wins = data.wins || 0;
            let rate = games > 0 ? ((wins / games) * 100) : 0;
            if (rate > 100) rate = 100;
            winRateEl.innerText = `${rate.toFixed(1)}%`;
        }

        const ptSumEl = document.getElementById('mini-profile-pt-sum');
        if (ptSumEl) ptSumEl.innerText = (data.pt_sum || 0).toLocaleString();

        // Demographics: Age and Gender
        const demoEl = document.getElementById('mini-profile-demographics');
        if (demoEl) {
            const age = data.age || '-';
            const gender = data.gender || '-';
            demoEl.innerText = `Age: ${age}, Gender: ${gender}`;
        }

        // Render Joined Date with Duration
        const joinedEl = document.getElementById('mini-profile-joined');
        if (joinedEl && data.created_at) {
            const joinedDate = new Date(data.created_at);
            const now = new Date();
            const years = now.getFullYear() - joinedDate.getFullYear();
            const months = (now.getMonth() + 1) - (joinedDate.getMonth() + 1) + (years * 12);

            let durationStr = '';
            if (months >= 12) {
                const y = Math.floor(months / 12);
                const m = months % 12;
                durationStr = `${y}y${m > 0 ? ` ${m}m` : ''}`;
            } else {
                durationStr = months > 0 ? `${months}m` : '< 1m';
            }

            const formattedJoined = typeof window.formatAppDate === 'function' ? window.formatAppDate(joinedDate) : joinedDate.toLocaleDateString();
            joinedEl.innerText = `Registered: ${formattedJoined} (${durationStr})`;
        } else if (joinedEl) {
            joinedEl.innerText = "Registered: -";
        }

        // Render Last Visited
        const lastVisitedEl = document.getElementById('mini-profile-last-visited');
        if (lastVisitedEl) {
            const isOnline = data.status && data.status.is_online;
            const lvStr = formatLastVisited(data.last_visited, isOnline);
            lastVisitedEl.innerText = `Last Visited: ${lvStr}`;
        }

        // Flag and Meta
        const flagEl = document.getElementById('mini-profile-flag');
        if (flagEl) flagEl.innerHTML = window.getFlagHtml ? window.getFlagHtml(data.country_flag) : (data.country_flag || '🏳️');

        // Country Name Lookup
        const flagEmoji = data.country_flag || '🏳️';
        const countryLookup = typeof ALL_FLAGS !== 'undefined' ? ALL_FLAGS.find(f => f.flag === flagEmoji) : null;
        const countryNameEl = document.getElementById('mini-profile-country-name');
        if (countryNameEl) countryNameEl.innerText = countryLookup ? countryLookup.name : 'International';

        // Proof
        const proofEl = document.getElementById('mini-profile-proof');
        if (proofEl) {
            if (data.proof_url) {
                proofEl.innerHTML = `<a href="${data.proof_url}" target="_blank" style="color: #4facfe; text-decoration: none;">View Proof</a>`;
            } else {
                proofEl.innerText = 'Proof: -';
            }
        }

        // Description and Quote
        const quoteEl = document.getElementById('mini-profile-quote');
        const descEl = document.getElementById('mini-profile-description');
        if (quoteEl) quoteEl.innerText = data.quote ? `"${data.quote}"` : 'No personal quote available.';
        if (descEl) {
            descEl.innerText = data.description || 'No description provided.';
            setTimeout(() => {
                if (typeof initCustomScrollbarForElement === 'function') {
                    initCustomScrollbarForElement('mini-profile-description', 'mini-desc-scrollbar-track', 'mini-desc-scrollbar-thumb');
                }
            }, 50);
        }

        // Avatar
        const avatar = document.getElementById('mini-profile-avatar');
        const rating = data.rating || 0;
        if (avatar) {
            if (data.avatar_url) {
                avatar.style.background = 'none';
                avatar.style.backgroundImage = `url('${data.avatar_url}')`;
                avatar.style.backgroundSize = 'cover';
                avatar.style.backgroundPosition = 'center';
                avatar.style.backgroundColor = 'rgba(0,0,0,0.3)';
                avatar.innerText = '';
                avatar.style.cursor = 'pointer';
                avatar.onclick = () => showImageLightbox(data.avatar_url, `${data.username}'s Profile Image`);
            } else {
                avatar.style.cursor = 'default';
                avatar.onclick = null;
                avatar.style.backgroundImage = 'none';
                const rColor = window.getRatingColor ? window.getRatingColor(rating) : '#fff';
                avatar.style.background = `linear-gradient(135deg, ${rColor}, #444)`;
                avatar.innerText = data.username.charAt(0).toUpperCase();
            }
        }

        // Setup Buttons: Navigation and Search
        const viewFullBtn = document.getElementById('mini-profile-view-full');
        if (viewFullBtn) {
            viewFullBtn.onclick = () => {
                modal.classList.add('hidden');
                modal.classList.remove('forced-show');
                const profileNavBtn = document.getElementById('nav-profile-btn');
                if (profileNavBtn) profileNavBtn.click();
                setTimeout(() => window.performProfileSearch(data.username), 50);
            };
        }

        const roundReviewsBtn = document.getElementById('mini-profile-round-reviews');
        if (roundReviewsBtn) {
            roundReviewsBtn.onclick = () => {
                modal.classList.add('hidden');
                modal.classList.remove('forced-show');
                const profileNavBtn = document.getElementById('nav-profile-btn');
                if (profileNavBtn) profileNavBtn.click();
                setTimeout(() => window.performProfileSearch(data.username, 'history'), 50);
            };
        }

        const msgBtn = document.getElementById('mini-profile-message');
        const globalUser = window.currentUser || (typeof currentUser !== 'undefined' ? currentUser : null);
        const currentName = (typeof globalUser === 'object') ? globalUser.username : globalUser;

        if (msgBtn) {
            const allowPm = data.allow_pm !== false;
            if (currentName && currentName.toLowerCase() !== data.username.toLowerCase() && allowPm) {
                msgBtn.classList.remove('hidden');
                msgBtn.onclick = () => {
                    modal.classList.add('hidden');
                    modal.classList.remove('forced-show');
                    window.openPrivateChat(data.username, false);
                };
                const friendBtn = document.getElementById('mini-profile-friend');
                if (friendBtn) {
                    friendBtn.classList.remove('hidden');
                    if (window.updateFriendButtonStatus) {
                        await window.updateFriendButtonStatus(data.username, friendBtn);
                    }
                    friendBtn.onclick = () => {
                        if (window.handleFriendAction) {
                            window.handleFriendAction(data.username, friendBtn);
                        }
                    };
                }
            } else {
                msgBtn.classList.add('hidden');
                const friendBtn = document.getElementById('mini-profile-friend');
                if (friendBtn) friendBtn.classList.add('hidden');
            }
        }

        // Setup Moderator Action Buttons (Visible strictly for authorized moderators, non-self)
        const isMod = Boolean(window.currentUserIsMod);
        const modActions = document.getElementById('mini-profile-mod-actions');
        const isSelf = currentName && currentName.toLowerCase() === data.username.toLowerCase();

        if (modActions) {
            if (isMod && !isSelf) {
                modActions.classList.remove('hidden');
                const timeoutBtn = document.getElementById('mini-profile-timeout-btn');
                if (timeoutBtn) {
                    timeoutBtn.onclick = () => window.openModTimeoutModal(data.username);
                }
                const banBtn = document.getElementById('mini-profile-ban-btn');
                if (banBtn) {
                    banBtn.onclick = () => window.openModBanModal(data.username);
                }
            } else {
                modActions.classList.add('hidden');
            }
        }

        // Finally Show
        if (modal) {
            modal.classList.add('forced-show');
            modal.classList.remove('hidden');
        }

    } catch (err) {
        console.error("Mini profile fetch error:", err);
    }
};

window.openModTimeoutModal = function(username) {
    if (!username) return;
    const modal = document.getElementById('mod-timeout-modal');
    const uEl = document.getElementById('mod-timeout-modal-username');
    const hInput = document.getElementById('mod-timeout-modal-hours');
    const rInput = document.getElementById('mod-timeout-modal-reason');
    const confirmBtn = document.getElementById('mod-timeout-modal-confirm-btn');

    if (uEl) uEl.textContent = username;
    if (hInput) hInput.value = '';
    if (rInput) rInput.value = '';

    if (confirmBtn) {
        confirmBtn.disabled = false;
        confirmBtn.textContent = 'Yes, Timeout User';
        confirmBtn.onclick = async () => {
            if (['jeffbabiak', 'jeffb', 'system'].includes(username.toLowerCase())) {
                alert(`Action Prohibited: User '${username}' cannot be timed out.`);
                return;
            }
            const hoursVal = hInput ? hInput.value.trim() : '';
            const reasonVal = rInput ? rInput.value.trim() : '';

            confirmBtn.disabled = true;
            confirmBtn.textContent = 'Processing...';

            try {
                const response = await fetch('/api/mods/timeout_user', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: username,
                        reason: reasonVal || 'Moderator timeout',
                        hours: hoursVal || null
                    })
                });
                const data = await response.json();
                if (data.success) {
                    window.closeModTimeoutModal();
                    if (window.showAlertModal) {
                        window.showAlertModal('User Timed Out', `User "${username}" has been timed out for ${data.duration}.<br>Until: ${data.timeout_until} UTC`);
                    } else {
                        alert(`User "${username}" has been timed out for ${data.duration}.`);
                    }
                } else {
                    alert("Error: " + (data.error || "Failed to timeout user."));
                }
            } catch (err) {
                console.error("Error timing out user:", err);
                alert("Network error timing out user.");
            } finally {
                confirmBtn.disabled = false;
                confirmBtn.textContent = 'Yes, Timeout User';
            }
        };
    }

    if (modal) {
        modal.classList.add('forced-show');
        modal.classList.remove('hidden');
    }
};

window.closeModTimeoutModal = function() {
    const modal = document.getElementById('mod-timeout-modal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('forced-show');
    }
};

window.openModBanModal = function(username) {
    if (!username) return;
    const modal = document.getElementById('mod-ban-modal');
    const uEl = document.getElementById('mod-ban-modal-username');
    const rInput = document.getElementById('mod-ban-modal-reason');
    const confirmBtn = document.getElementById('mod-ban-modal-confirm-btn');

    if (uEl) uEl.textContent = username;
    if (rInput) rInput.value = '';

    if (confirmBtn) {
        confirmBtn.disabled = false;
        confirmBtn.textContent = 'Yes, Ban & Erase';
        confirmBtn.onclick = async () => {
            if (['jeffbabiak', 'jeffb', 'system'].includes(username.toLowerCase())) {
                alert(`Action Prohibited: User '${username}' cannot be banned.`);
                return;
            }
            const reasonVal = rInput ? rInput.value.trim() : '';

            confirmBtn.disabled = true;
            confirmBtn.textContent = 'Erasing & Banning...';

            try {
                const response = await fetch('/api/mods/ban_user', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: username,
                        reason: reasonVal || 'Permanent ban'
                    })
                });
                const data = await response.json();
                if (data.success) {
                    window.closeModBanModal();
                    const miniModal = document.getElementById('mini-profile-modal');
                    if (miniModal) {
                        miniModal.classList.add('hidden');
                        miniModal.classList.remove('forced-show');
                    }
                    if (typeof window.loadIpBans === 'function') {
                        window.loadIpBans();
                    }
                    if (window.showAlertModal) {
                        window.showAlertModal('User Erased & Banned', data.message || `User "${username}" has been permanently erased from the database.`);
                    } else {
                        alert(data.message || `User "${username}" has been permanently erased from the database.`);
                    }
                } else {
                    alert("Error: " + (data.error || "Failed to ban user."));
                }
            } catch (err) {
                console.error("Error banning user:", err);
                alert("Network error banning user.");
            } finally {
                confirmBtn.disabled = false;
                confirmBtn.textContent = 'Yes, Ban & Erase';
            }
        };
    }

    if (modal) {
        modal.classList.add('forced-show');
        modal.classList.remove('hidden');
    }
};

window.closeModBanModal = function() {
    const modal = document.getElementById('mod-ban-modal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('forced-show');
    }
};

// --- Profile Tool Logic ---

function setupProfileTool() {
    const searchBtn = document.getElementById('profile-search-btn');
    const myProfileBtn = document.getElementById('profile-my-profile-btn');
    const input = document.getElementById('profile-search-input');

    if (searchBtn) {
        searchBtn.addEventListener('click', () => {
            if (input) performProfileSearch(input.value);
        });
    }

    if (myProfileBtn) {
        myProfileBtn.addEventListener('click', () => {
            const globalUser = window.currentUser || (typeof currentUser !== 'undefined' ? currentUser : null);
            if (globalUser) {
                const name = (typeof globalUser === 'object') ? globalUser.username : globalUser;
                if (name && !name.startsWith('Guest_')) {
                    if (input) input.value = name;
                    performProfileSearch(name);
                    return;
                }
            }
            if (typeof window.refreshProfileTool === 'function') {
                window.refreshProfileTool(true);
            }
        });
    }

    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performProfileSearch(input.value);
        });
        input.addEventListener('input', () => {
            const errorEl = document.getElementById('profile-search-error');
            if (errorEl) {
                errorEl.style.display = 'none';
                errorEl.innerText = '';
            }
        });
    }

    // Expose refresh function globally so app.js can trigger it
    window.refreshProfileTool = (force = false) => {
        const input = document.getElementById('profile-search-input');
        const globalUser = window.currentUser || (typeof currentUser !== 'undefined' ? currentUser : null);

        if (globalUser) {
            const name = (typeof globalUser === 'object') ? globalUser.username : globalUser;

            // Guests do not have profiles
            if (name && name.startsWith('Guest_')) {
                if (input) {
                    input.value = '';
                    input.placeholder = "Log in to view profiles...";
                }
                document.getElementById('profile-display-container').classList.add('hidden');
                return;
            }

            if (name && input) {
                const currentDisplay = document.getElementById('profile-username').innerText;

                if (force || !currentDisplay || currentDisplay === 'Player' || currentDisplay !== name) {
                    input.value = name;
                    input.placeholder = "Enter username...";
                    performProfileSearch(name);
                }
            }
        }
    };

    // Initial load attempts
    setTimeout(window.refreshProfileTool, 800);
    setTimeout(window.refreshProfileTool, 3000);

    // Avatar Upload Logic
    const avatarInput = document.getElementById('profile-avatar-input');
    const avatarTrigger = document.getElementById('profile-avatar-trigger');

    if (avatarInput) {
        avatarInput.addEventListener('change', async (e) => {
            if (e.target.files && e.target.files[0]) {
                await uploadAvatar(e.target.files[0]);
            }
        });
    }



    // Flag Selection Logic (Dropdown)
    const flagTrigger = document.getElementById('profile-flag');
    const flagDropdown = document.getElementById('flag-dropdown');
    const flagDropdownSearch = document.getElementById('flag-dropdown-search');

    if (flagTrigger && flagDropdown) {
        flagTrigger.addEventListener('click', (e) => {
            console.log("Flag clicked!");
            e.stopPropagation();

            const isActive = flagDropdown.classList.contains('active');

            // Close all other dropdowns
            document.querySelectorAll('.dropdown-menu').forEach(d => d.classList.remove('active'));

            if (!isActive) {
                console.log("Opening dropdown...");
                flagDropdown.classList.add('active');
                renderFlagDropdown();
                if (flagDropdownSearch) {
                    flagDropdownSearch.value = '';
                    if (window.innerWidth > 600) {
                        setTimeout(() => flagDropdownSearch.focus({ preventScroll: true }), 50);
                    }
                }
            } else {
                console.log("Closing dropdown...");
                flagDropdown.classList.remove('active');
            }
        });

        if (flagDropdownSearch) {
            flagDropdownSearch.addEventListener('input', (e) => {
                renderFlagDropdown(e.target.value);
            });
            // Stop propagation so clicking search doesn't close dropdown
            flagDropdownSearch.addEventListener('click', (e) => e.stopPropagation());
        }

        // Global click to close
        window.addEventListener('click', () => {
            flagDropdown.classList.remove('active');
        });
    }

    // Profile Tab Logic
    const tabToggles = document.querySelectorAll('.profile-tab-toggle');
    const tabPanes = document.querySelectorAll('.profile-tab-pane');

    tabToggles.forEach(toggle => {
        toggle.addEventListener('click', () => {
            const targetTab = toggle.dataset.tab;

            tabToggles.forEach(t => t.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            toggle.classList.add('active');
            const targetPane = document.getElementById(`profile-tab-${targetTab}`);
            if (targetPane) targetPane.classList.add('active');

            // Re-fetch friends if switching to friends tab
            if (targetTab === 'friends') {
                if (typeof fetchAndRenderFriends === 'function') {
                    fetchAndRenderFriends();
                }
            }
        });
    });

    const refreshBtn = document.getElementById('profile-refresh-btn');
    if (refreshBtn) {
        refreshBtn.onclick = async () => {
            if (refreshBtn.classList.contains('refreshing')) return;
            refreshBtn.classList.add('refreshing');

            const displayedUsername = document.getElementById('profile-username')?.innerText?.trim();
            const targetUser = (displayedUsername && displayedUsername !== 'Player')
                ? displayedUsername
                : null;

            try {
                if (!targetUser) {
                    await window.refreshProfileTool(true);
                } else {
                    console.log(`[Profile] Manual refresh for: ${targetUser}`);
                    await performProfileSearch(targetUser);
                }
            } catch (err) {
                console.error("[Profile] Error refreshing profile:", err);
            } finally {
                setTimeout(() => refreshBtn.classList.remove('refreshing'), 600);
            }
        };
    }
}

// Full Country Flag List (ISO 3166-1)
const ALL_FLAGS = window.ALL_FLAGS || [];

function renderFlagDropdown(filter = '') {
    const list = document.getElementById('flag-dropdown-list');
    if (!list) return;
    list.innerHTML = '';

    const term = filter.toLowerCase().trim();

    const filtered = ALL_FLAGS.filter(f =>
        f.name.toLowerCase().includes(term) || f.code.toLowerCase().includes(term)
    );

    filtered.forEach(item => {
        const div = document.createElement('div');
        div.className = 'dropdown-item';

        div.innerHTML = `
            <span class="dropdown-item-flag">${window.getFlagHtml ? window.getFlagHtml(item.flag) : item.flag}</span>
            <span class="dropdown-item-text">${item.name}</span>
        `;

        div.onclick = (e) => {
            e.stopPropagation();
            updateFlag(item.flag);
        };

        list.appendChild(div);
    });
}

async function updateFlag(flag) {
    const displayedName = document.getElementById('profile-username').innerText;
    const globalUser = window.currentUser || currentUser;
    const currentName = (typeof globalUser === 'object') ? globalUser.username : globalUser;

    // Only allow saving if it matches current user (case-insensitive)
    if (!currentName || currentName.toLowerCase() !== displayedName.toLowerCase()) {
        alert("You can only change the flag on your own profile.");
        const dropdown = document.getElementById('flag-dropdown');
        if (dropdown) dropdown.classList.remove('active');
        return;
    }

    const flagEl = document.getElementById('profile-flag');
    const flagNameEl = document.getElementById('profile-flag-name');
    const dropdown = document.getElementById('flag-dropdown');

    // Optimistic UI
    flagEl.innerHTML = window.getFlagHtml ? window.getFlagHtml(flag) : flag;

    // Find country name
    const country = typeof ALL_FLAGS !== 'undefined' ? ALL_FLAGS.find(f => f.flag === flag) : null;
    if (country) {
        flagEl.title = country.name;
        if (flagNameEl) flagNameEl.innerText = country.name;
    } else {
        flagEl.title = "Unknown Location";
        if (flagNameEl) flagNameEl.innerText = "";
    }

    if (dropdown) dropdown.classList.remove('active');

    try {
        const response = await fetch('/api/profile/update_flag', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ flag: flag })
        });

        const data = await response.json();
        if (data.error) {
            alert("Failed to update flag: " + data.error);
        }
    } catch (err) {
        console.error("Flag update failed:", err);
        alert("Connection error updating flag.");
    }
}

function compressImage(file, maxDimension, quality = 0.8) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (event) => {
            const img = new Image();
            img.src = event.target.result;
            img.onload = () => {
                let width = img.width;
                let height = img.height;

                if (width > maxDimension || height > maxDimension) {
                    if (width > height) {
                        height = Math.round((height * maxDimension) / width);
                        width = maxDimension;
                    } else {
                        width = Math.round((width * maxDimension) / height);
                        height = maxDimension;
                    }
                }

                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                canvas.toBlob((blob) => {
                    if (blob) {
                        const name = file.name.substring(0, file.name.lastIndexOf('.')) + '.jpg';
                        const compressedFile = new File([blob], name, { type: 'image/jpeg', lastModified: Date.now() });
                        resolve(compressedFile);
                    } else {
                        reject(new Error("Canvas to Blob failed"));
                    }
                }, 'image/jpeg', quality);
            };
            img.onerror = (err) => reject(err);
        };
        reader.onerror = (err) => reject(err);
    });
}

async function uploadAvatar(file) {
    const avatarEl = document.querySelector('.profile-avatar');
    const originalContent = avatarEl.innerHTML;

    // Optimistic UI
    avatarEl.innerHTML = '<span style="font-size:12px">...</span>';

    let processedFile = file;
    if (file && file.type !== 'image/gif') {
        try {
            processedFile = await compressImage(file, 300, 0.8);
        } catch (e) {
            console.error("Avatar compression failed, uploading original:", e);
        }
    } else if (file && file.size > 2 * 1024 * 1024) {
        alert("GIF files must be under 2MB.");
        avatarEl.innerHTML = originalContent;
        return;
    }

    const formData = new FormData();
    formData.append('avatar', processedFile);

    try {
        const response = await fetch('/api/profile/upload_avatar', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            avatarEl.innerHTML = originalContent;
            return;
        }

        if (data.avatar_url) {
            // Update UI with new image
            avatarEl.style.backgroundImage = `url('${data.avatar_url}?t=${Date.now()}')`;
            avatarEl.style.backgroundSize = 'contain';
            avatarEl.style.backgroundRepeat = 'no-repeat';
            avatarEl.style.backgroundPosition = 'center';
            avatarEl.style.backgroundColor = 'rgba(0,0,0,0.3)';
            avatarEl.innerText = '';
            avatarEl.style.backgroundBlendMode = 'normal';
        }

    } catch (err) {
        console.error("Upload failed", err);
        alert("Upload failed");
        avatarEl.innerHTML = originalContent;
    }
}

let _currentProfileSearchSeq = 0;

async function performProfileSearch(username, activeTab = null, period = 'all') {
    const errorEl = document.getElementById('profile-search-error');
    if (errorEl) {
        errorEl.style.display = 'none';
        errorEl.innerText = '';
    }

    if (!username || !username.trim()) return;

    username = username.trim();
    const searchSeq = ++_currentProfileSearchSeq;
    const container = document.getElementById('profile-display-container');
    const input = document.getElementById('profile-search-input');
    if (input) input.value = username;

    // Guests do not have profiles
    if (username.startsWith('Guest_')) {
        if (container) container.classList.add('hidden');
        if (errorEl) {
            errorEl.innerText = "The username you entered does not exist.";
            errorEl.style.display = 'block';
        }
        return;
    }

    // Check if the current displayed user is different from the requested user
    const currentDisplayed = document.getElementById('profile-username')?.innerText || '';
    const isSameUser = currentDisplayed && currentDisplayed.toLowerCase() === username.toLowerCase();

    // Fast client-side profile cache for instant tab transitions and re-renders
    if (!window._profileMemoryCache) {
        window._profileMemoryCache = new Map();
    }
    const cacheKey = `${username.toLowerCase()}|${period}`;
    const cachedEntry = window._profileMemoryCache.get(cacheKey);
    const now = Date.now();

    if (cachedEntry && (now - cachedEntry.time < 30000)) {
        // Render instantly from cache (0ms latency)
        await renderProfile(cachedEntry.data);
        if (container) container.classList.remove('hidden');
    } else {
        // Immediately hide container if switching to a different user to prevent flashing previous user
        if (container && !isSameUser) {
            container.classList.add('hidden');
        }
    }

    try {
        const response = await fetch(`/api/profile/${encodeURIComponent(username)}?period=${period}&t=${Date.now()}`);
        const data = await response.json();

        // If another search was started while this fetch was in-flight, discard this result
        if (searchSeq !== _currentProfileSearchSeq) {
            return;
        }

        if (data.error) {
            // User not found, just don't show the profile
            if (container) container.classList.add('hidden');
            if (errorEl) {
                errorEl.innerText = "The username you entered does not exist.";
                errorEl.style.display = 'block';
            }
            return;
        }

        // Cache the fresh profile
        window._profileMemoryCache.set(cacheKey, { data, time: Date.now() });

        await renderProfile(data);
        if (container) container.classList.remove('hidden');

        // Activate specific tab if requested
        if (activeTab) {
            const tabToggle = document.querySelector(`.profile-tab-toggle[data-tab="${activeTab}"]`);
            if (tabToggle) {
                tabToggle.click();
                if (activeTab === 'history') {
                    setTimeout(() => {
                        const targetEl = document.querySelector('.profile-tabs-header') || document.getElementById('profile-tab-history');
                        if (targetEl) {
                            try {
                                targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
                            } catch (e) {
                                targetEl.scrollIntoView();
                            }
                            const profilePage = document.getElementById('page-profile');
                            if (profilePage && profilePage.scrollHeight > profilePage.clientHeight) {
                                const topPos = targetEl.offsetTop - 15;
                                profilePage.scrollTo({ top: Math.max(0, topPos), behavior: 'smooth' });
                            }
                            const mainContent = document.querySelector('.main-content');
                            if (mainContent && mainContent.scrollHeight > mainContent.clientHeight) {
                                const topPos = targetEl.offsetTop - 15;
                                mainContent.scrollTo({ top: Math.max(0, topPos), behavior: 'smooth' });
                            }
                        }
                    }, 120);
                }
            }
        }

    } catch (err) {
        console.error("Profile search error:", err);
    }
}
window.performProfileSearch = performProfileSearch;

async function renderProfile(user) {
    // Check Ownership for Editing
    const globalUser = window.currentUser || (typeof currentUser !== 'undefined' ? currentUser : null);
    const currentName = (typeof globalUser === 'object') ? globalUser.username : globalUser;
    const isOwner = currentName && currentName.toLowerCase() === user.username.toLowerCase();

    const usernameEl = document.getElementById('profile-username');
    if (usernameEl) usernameEl.innerText = user.username;

    // Full Name
    const fullNameEl = document.getElementById('profile-full-name');
    if (fullNameEl) fullNameEl.innerText = user.full_name || '-';

    // PT SUM
    const ptSumEl = document.getElementById('profile-pt-sum');
    if (ptSumEl) ptSumEl.innerText = (user.pt_sum || 0).toLocaleString();

    // Rating & Color (Global display removed, but we still need color for avatar theme)
    const avatar = document.querySelector('.profile-avatar.large');

    // Determine Color based on global rating
    const r = user.rating || 0;
    const color = window.getRatingColor ? window.getRatingColor(r) : '#b3b3b3';

    // Avatar Handling
    if (avatar) {
        if (user.avatar_url) {
            avatar.style.backgroundColor = 'rgba(0,0,0,0.3)';
            avatar.style.backgroundImage = `url('${user.avatar_url}')`;
            avatar.style.backgroundSize = 'contain';
            avatar.style.backgroundRepeat = 'no-repeat';
            avatar.style.backgroundPosition = 'center';
            avatar.innerText = '';
            if (isOwner) {
                avatar.style.cursor = 'pointer';
                avatar.onclick = null;
            } else {
                avatar.style.cursor = 'pointer';
                avatar.onclick = () => showImageLightbox(user.avatar_url, `${user.username}'s Profile Image`);
            }
        } else {
            avatar.style.cursor = isOwner ? 'pointer' : 'default';
            avatar.onclick = null;
            avatar.style.backgroundImage = 'none';
            avatar.style.background = `linear-gradient(135deg, ${color}, #444)`;
            avatar.innerText = user.username ? user.username.charAt(0).toUpperCase() : '?';
        }
        avatar.style.boxShadow = `0 10px 30px ${color}33`;
    }

    // Flag Handling
    const flagEl = document.getElementById('profile-flag');
    const flagNameEl = document.getElementById('profile-flag-name');
    if (flagEl) {
        const flagEmoji = user.country_flag || '🏳️';
        flagEl.innerHTML = window.getFlagHtml ? window.getFlagHtml(flagEmoji) : flagEmoji;

        // Find country name
        const country = typeof ALL_FLAGS !== 'undefined' ? ALL_FLAGS.find(f => f.flag === flagEmoji) : null;
        if (country) {
            flagEl.title = country.name;
            if (flagNameEl) flagNameEl.innerText = country.name;
        } else {
            flagEl.title = "Unknown Location";
            if (flagNameEl) flagNameEl.innerText = "";
        }
    }

    // Stats
    const gamesEl = document.getElementById('profile-games');
    if (gamesEl) gamesEl.innerText = user.games_played || 0;

    const winRateEl = document.getElementById('profile-win-rate');
    if (winRateEl) {
        if (user.games_played > 0) {
            const wins = user.wins || 0;
            const rate = ((wins / user.games_played) * 100).toFixed(1);
            winRateEl.innerText = `${rate}%`;
        } else {
            winRateEl.innerText = '0%';
        }
    }

    const wpmEl = document.getElementById('profile-avg-wpm');
    if (wpmEl) {
        wpmEl.innerText = user.avg_wpm_300 || 0;
        wpmEl.title = "Average Words Per Minute in boards with 100+ potential words (requires finding 20+ words in a round)";
    }

    const bestScoreEl = document.getElementById('profile-best-score');
    if (bestScoreEl) {
        bestScoreEl.innerText = user.best_score || '-';
    }

    // Profile Details
    const ageEl = document.getElementById('profile-age-val');
    const genderEl = document.getElementById('profile-gender-val');
    const quoteEl = document.getElementById('profile-quote-val');
    const descriptionEl = document.getElementById('profile-description-val');
    const locationEl = document.getElementById('profile-location-val');

    if (ageEl) ageEl.innerText = user.age || '-';
    if (genderEl) genderEl.innerText = user.gender || '-';
    if (locationEl) locationEl.innerText = user.location || '-';
    if (quoteEl) quoteEl.innerText = user.quote || 'Enter a personal quote';
    if (descriptionEl) {
        descriptionEl.innerText = user.description || 'Add a detailed description about yourself...';
        setTimeout(() => {
            if (typeof initCustomScrollbarForElement === 'function') {
                initCustomScrollbarForElement('profile-description-val', 'profile-desc-scrollbar-track', 'profile-desc-scrollbar-thumb');
            }
        }, 50);
    }

    // Registration Date
    const joinedValEl = document.getElementById('profile-joined-val');
    if (joinedValEl && user.created_at) {
        const joinedDate = new Date(user.created_at);
        const now = new Date();
        const years = now.getFullYear() - joinedDate.getFullYear();
        const months = (now.getMonth() + 1) - (joinedDate.getMonth() + 1) + (years * 12);

        let durationStr = '';
        if (months >= 12) {
            const y = Math.floor(months / 12);
            const m = months % 12;
            durationStr = `${y}y${m > 0 ? ` ${m}m` : ''}`;
        } else {
            durationStr = months > 0 ? `${months}m` : '< 1m';
        }

        const formattedJoined = typeof window.formatAppDate === 'function' ? window.formatAppDate(joinedDate) : joinedDate.toLocaleDateString();
        joinedValEl.innerText = `Registered: ${formattedJoined} (${durationStr})`;
    } else if (joinedValEl) {
        joinedValEl.innerText = 'Registered: -';
    }

    // Last Visited Date
    const lastVisitedValEl = document.getElementById('profile-last-visited-val');
    if (lastVisitedValEl) {
        const isOnline = user.status && user.status.is_online;
        const lvStr = formatLastVisited(user.last_visited, isOnline);
        lastVisitedValEl.innerText = `Last Visited: ${lvStr}`;
    }

    // Proof of Legitimacy Rendering
    const proofLink = document.getElementById('profile-proof-link');
    const proofPlaceholder = document.getElementById('profile-proof-placeholder');
    const proofInput = document.getElementById('profile-proof-input');

    if (user.proof_url) {
        if (proofLink) {
            proofLink.href = user.proof_url;
            proofLink.classList.remove('hidden');
        }
        if (proofPlaceholder) proofPlaceholder.classList.add('hidden');
    } else {
        if (proofLink) proofLink.classList.add('hidden');
        if (proofPlaceholder) proofPlaceholder.classList.remove('hidden');
    }

    // Online Status & Follow Button
    const statusDot = document.getElementById('profile-status-indicator');
    const followBtn = document.getElementById('profile-follow-btn');
    const roomInput = document.getElementById('profile-current-room');

    // Check Ownership for Editing (already checked at the top of renderProfile)

    // Dynamically toggle disabled state, display, & cursor on avatar trigger based on ownership
    const avatarInput = document.getElementById('profile-avatar-input');
    const avatarTrigger = document.getElementById('profile-avatar-trigger');
    if (avatarInput) {
        if (isOwner) {
            avatarInput.disabled = false;
            avatarInput.style.pointerEvents = 'auto';
            avatarInput.style.cursor = 'pointer';
            if (avatarTrigger) {
                avatarTrigger.style.setProperty('cursor', 'pointer', 'important');
                avatarTrigger.title = "Click to upload photo";
            }
        } else {
            avatarInput.disabled = true;
            avatarInput.style.pointerEvents = 'none';
            avatarInput.style.cursor = 'default';
            if (avatarTrigger) {
                avatarTrigger.style.setProperty('cursor', 'default', 'important');
                avatarTrigger.title = "";
            }
        }
    }

    // Proof Editing
    if (isOwner && proofInput) {
        proofInput.classList.remove('hidden');
        proofInput.value = user.proof_url || '';
        // Add save listener
        const newProofInput = proofInput.cloneNode(true);
        proofInput.parentNode.replaceChild(newProofInput, proofInput);
        newProofInput.addEventListener('blur', () => saveProfileField('proof_url', newProofInput.value.trim()));
        newProofInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                newProofInput.blur();
            }
        });
    } else if (proofInput) {
        proofInput.classList.add('hidden');
    }

    if (statusDot) {
        if (user.status && user.status.is_online) {
            statusDot.className = 'status-indicator online';
            statusDot.title = 'Online';
        } else {
            statusDot.className = 'status-indicator offline';
            statusDot.title = 'Offline';
        }
    }

    if (followBtn && roomInput) {
        const currentRoom = (user.status && user.status.current_room) ? user.status.current_room : '';
        roomInput.value = currentRoom;

        // Only show follow button if online, in a room, and not viewing self
        if (currentRoom && !isOwner) {
            followBtn.classList.remove('hidden');
        } else {
            followBtn.classList.add('hidden');
        }
    }

    const messageBtn = document.getElementById('profile-message-btn');
    const friendBtn = document.getElementById('profile-friend-btn');

    if (messageBtn) {
        const allowPm = user.allow_pm !== false;
        if (currentName && !isOwner && allowPm) {
            messageBtn.classList.remove('hidden');
            const newMsgBtn = messageBtn.cloneNode(true);
            messageBtn.parentNode.replaceChild(newMsgBtn, messageBtn);
            newMsgBtn.addEventListener('click', () => {
                openPrivateChat(user.username, false);
            });
        } else {
            messageBtn.classList.add('hidden');
        }
    }

    if (friendBtn) {
        if (currentName && !isOwner) {
            friendBtn.classList.remove('hidden');

            // WE MUST CLONE FIRST to clear old listeners, THEN update status on the NEW element
            const newFriendBtn = friendBtn.cloneNode(true);
            friendBtn.parentNode.replaceChild(newFriendBtn, friendBtn);

            await updateFriendButtonStatus(user.username, newFriendBtn);

            newFriendBtn.addEventListener('click', () => {
                handleFriendAction(user.username, newFriendBtn);
            });
        } else {
            friendBtn.classList.add('hidden');
        }
    }

    // --- Friends Tab (Only for Owner) ---
    const friendsTabToggle = document.getElementById('profile-tab-toggle-friends');
    if (friendsTabToggle) {
        if (isOwner) {
            friendsTabToggle.classList.remove('hidden');
            fetchAndRenderFriends();
        } else {
            friendsTabToggle.classList.add('hidden');
            // If friends tab was active, switch back to rankings
            if (friendsTabToggle.classList.contains('active')) {
                document.querySelector('[data-tab="rankings"]').click();
            }
        }
    }

    // Helper for rendering a dense data row for a round
    window.renderRoundGridItem = (round) => {
        const gameTypeLabel = round.game_type === 'split' ? 'Split' :
            round.game_type === 'fcfs' ? 'FCFS' :
            round.game_type === '3d' ? 'Cube' : 'Acc';
        const typeClass = `history-type-${round.game_type}`;

        // Board dimension + time display (replaces mini board icon)
        const dims = round.dimensions || '?';
        const dur = round.round_duration || 0;
        const mins = Math.floor(dur / 60);
        const secs = dur % 60;
        const timeStr = mins > 0
            ? `${mins}:${String(secs).padStart(2, '0')}`
            : `0:${String(secs).padStart(2, '0')}`;

        // Date Formatting
        let dateStr = '-';
        if (round.timestamp) {
            dateStr = typeof window.formatAppDate === 'function' ? window.formatAppDate(round.timestamp) : String(round.timestamp);
        }

        return `
        <div class="history-grid-item" onclick="watchRoundHistory('${round.room_id}', ${round.round_number}, true, ${round.game_id || 'null'})" style="display: grid; grid-template-columns: repeat(7, 1fr); gap:8px; padding: 10px 15px; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.03); border-radius: 10px; margin-bottom: 8px; align-items: center; transition: all 0.2s; cursor: pointer; position: relative; overflow: hidden; min-width: 620px;">
            <div class="history-mode-tag ${typeClass}" style="font-size: 0.65rem; padding: 3px 6px; border-radius: 6px; text-align: center; width: fit-content; font-weight: 800; text-transform: uppercase;">${gameTypeLabel}</div>

            <!-- Board: dimension + time -->
            <div style="display: flex; flex-direction: column; gap: 2px; align-items: flex-start; justify-content: center; text-align: left;">
                <span style="font-size: 0.78rem; font-weight: 800; color: rgba(255,255,255,0.85);">${dims}</span>
                <span style="font-size: 0.62rem; color: rgba(255,255,255,0.35); font-weight: 600;">${timeStr}</span>
            </div>

            <div style="font-weight: 900; color: #fff; font-size: 0.95rem;">${round.total_score} <small style="font-size: 0.6rem; opacity: 0.5;">PTS</small></div>

            <div style="font-weight: 900; color: ${round.performance_value >= 140 ? '#60a5fa' : 'rgba(255,255,255,0.2)'}; font-size: 0.85rem;">${round.performance_value ? (round.performance_value / 100).toFixed(2) + 'x' : '-'}</div>
            <div style="display: flex; flex-direction: column; gap: 1px;">
                <span style="color: #fff; font-size: 0.7rem; font-weight: 700;">${round.num_words} words</span>
                <span style="color: rgba(255,255,255,0.3); font-size: 0.6rem;">Avg: ${round.avg_len}</span>
            </div>
            <div style="color: #ffd700; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;" title="${round.top_word}">${round.top_word}</div>

            <!-- Date Column -->
            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6); font-weight: 600; text-align: right;">${dateStr}</div>
        </div>
        `;
    };

    window.roundGridHeader = `
        <div class="history-grid-header" style="display: grid; grid-template-columns: repeat(7, 1fr); gap:8px; padding: 12px 15px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 12px; font-size: 0.7rem; color: rgba(255,255,255,0.4); font-weight: 800; text-transform: uppercase; letter-spacing: 1px; min-width: 620px;">
            <div>Mode</div>
            <div style="text-align: left;">Board</div>
            <div>Score</div>
            <div>PE</div>
            <div>Stats</div>
            <div>Top Word</div>
            <div style="text-align: right;">Date</div>
        </div>
    `;

    // --- Render Round History & Exceptional Rounds ---
    const historyList = document.getElementById('profile-history-list');
    const exceptionalList = document.getElementById('profile-exceptional-list');

    if (historyList) {
        if (!user.recent_rounds || user.recent_rounds.length === 0) {
            historyList.innerHTML = '<p class="placeholder">No recently tracked rounds.</p>';
        } else {
            const displayRounds = user.recent_rounds.slice(0, 50);
            historyList.innerHTML = window.roundGridHeader + displayRounds.map(r => window.renderRoundGridItem(r)).join('');
        }
    }

    if (exceptionalList) {
        if (!user.exceptional_rounds || user.exceptional_rounds.length === 0) {
            exceptionalList.innerHTML = '<p class="placeholder">No exceptional achievements recorded yet.</p>';
        } else {
            // Limits to 50 rows as requested
            const displayRounds = user.exceptional_rounds.slice(0, 50);
            const greatestPE = user.max_pe || 0;
            const peFormatted = greatestPE ? (greatestPE / 100).toFixed(2) + 'x' : '0.00x';
            const peHeader = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 12px 20px; border-radius: 8px;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 0.5px;">Exceptional Performances</span>
                    <span style="font-size: 0.9rem; font-weight: 800; color: #60a5fa;">Greatest PE: <span style="color: #fff;">${peFormatted}</span></span>
                </div>
            `;
            exceptionalList.innerHTML = peHeader + window.roundGridHeader + displayRounds.map(r => window.renderRoundGridItem(r)).join('');
        }
    }

    // Cache all rounds for review (both recent and exceptional)
    window.lastRenderedRounds = [...(user.recent_rounds || []), ...(user.exceptional_rounds || [])];

    // Render Ratings Grid (32 setups)
    renderRatingsGrid(user.config_ratings || {}, user);

    setupProfileEditing(isOwner);
}

// Helper: Find a valid Boggle path for a word on the current board
function findWordPath(board, word) {
    if (!board || !word) return null;
    const rows = board.length;
    const cols = (board[0] && Array.isArray(board[0])) ? board[0].length : 0;
    const is3D = rows === 6 && Array.isArray(board[0]) && Array.isArray(board[0][0]);

    if (is3D) {
        return findWordPathOnCube(word, board);
    }

    const targetWord = word.toUpperCase();
    function dfs(r, c, index, visited) {
        if (index >= targetWord.length) return [];
        const cellVal = board[r][c].toUpperCase();
        let matchLen = 0;
        if (cellVal.includes('/')) {
            const options = cellVal.split('/');
            for (const opt of options) {
                if (opt === 'Q') {
                    if (targetWord.substring(index, index + 2) === 'QU') {
                        matchLen = 2;
                        break;
                    } else if (targetWord[index] === 'Q') {
                        matchLen = 1;
                        break;
                    }
                } else if (targetWord.substring(index).startsWith(opt)) {
                    matchLen = opt.length;
                    break;
                }
            }
        } else {
            const letter = cellVal;
            if (targetWord[index] === letter) {
                matchLen = 1;
            } else if (letter === 'Q' && targetWord.substring(index, index + 2) === 'QU') {
                matchLen = 2;
            }
        }
        if (matchLen === 0) return null;
        if (index + matchLen === targetWord.length) {
            return [{ row: r, col: c }];
        }
        visited.add(`${r},${c}`);
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const nr = r + dr;
                const nc = c + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !visited.has(`${nr},${nc}`)) {
                    const result = dfs(nr, nc, index + matchLen, visited);
                    if (result) return [{ row: r, col: c }, ...result];
                }
            }
        }
        visited.delete(`${r},${c}`);
        return null;
    }
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const path = dfs(r, c, 0, new Set());
            if (path) return path;
        }
    }
    return null;
}

function findWordPathOnCube(word, board) {
    if (!word || !board || board.length !== 6) return null;
    const upperWord = word.toUpperCase();

    function getCubeNeighbors(f, r, c) {
        const res = [];
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const nr = r + dr, nc = c + dc;
                if (nr >= 0 && nr < 3 && nc >= 0 && nc < 3) res.push({ f, r: nr, c: nc });
            }
        }
        if (f === 0) {
            if (r === 0) res.push({ f: 4, r: 2, c }, { f: 4, r: 2, c: c - 1 }, { f: 4, r: 2, c: c + 1 });
            if (r === 2) res.push({ f: 5, r: 0, c }, { f: 5, r: 0, c: c - 1 }, { f: 5, r: 0, c: c + 1 });
            if (c === 0) res.push({ f: 2, r, c: 2 }, { f: 2, r: r - 1, c: 2 }, { f: 2, r: r + 1, c: 2 });
            if (c === 2) res.push({ f: 3, r, c: 0 }, { f: 3, r: r - 1, c: 0 }, { f: 3, r: r + 1, c: 0 });
        } else if (f === 1) {
            if (r === 0) res.push({ f: 4, r: 0, c: 2 - c }, { f: 4, r: 0, c: 2 - (c - 1) }, { f: 4, r: 0, c: 2 - (c + 1) });
            if (r === 2) res.push({ f: 5, r: 2, c: 2 - c }, { f: 5, r: 2, c: 2 - (c - 1) }, { f: 5, r: 2, c: 2 - (c + 1) });
            if (c === 0) res.push({ f: 3, r, c: 2 }, { f: 3, r: r - 1, c: 2 }, { f: 3, r: r + 1, c: 2 });
            if (c === 2) res.push({ f: 2, r, c: 0 }, { f: 2, r: r - 1, c: 0 }, { f: 2, r: r + 1, c: 0 });
        } else if (f === 2) {
            if (r === 0) res.push({ f: 4, r: c, c: 0 }, { f: 4, r: c - 1, c: 0 }, { f: 4, r: c + 1, c: 0 });
            if (r === 2) res.push({ f: 5, r: 2 - c, c: 0 }, { f: 5, r: 2 - (c - 1), c: 0 }, { f: 5, r: 2 - (c + 1), c: 0 });
            if (c === 0) res.push({ f: 1, r, c: 2 }, { f: 1, r: r - 1, c: 2 }, { f: 1, r: r + 1, c: 2 });
            if (c === 2) res.push({ f: 0, r, c: 0 }, { f: 0, r: r - 1, c: 0 }, { f: 0, r: r + 1, c: 0 });
        } else if (f === 3) {
            if (r === 0) res.push({ f: 4, r: 2 - c, c: 2 }, { f: 4, r: 2 - (c - 1), c: 2 }, { f: 4, r: 2 - (c + 1), c: 2 });
            if (r === 2) res.push({ f: 5, r: c, c: 2 }, { f: 5, r: c - 1, c: 2 }, { f: 5, r: c + 1, c: 2 });
            if (c === 0) res.push({ f: 0, r, c: 2 }, { f: 0, r: r - 1, c: 2 }, { f: 0, r: r + 1, c: 2 });
            if (c === 2) res.push({ f: 1, r, c: 0 }, { f: 1, r: r - 1, c: 0 }, { f: 1, r: r + 1, c: 0 });
        } else if (f === 4) {
            if (r === 0) res.push({ f: 1, r: 0, c: 2 - c }, { f: 1, r: 0, c: 2 - (c - 1) }, { f: 1, r: 0, c: 2 - (c + 1) });
            if (r === 2) res.push({ f: 0, r: 0, c }, { f: 0, r: 0, c: c - 1 }, { f: 0, r: 0, c: c + 1 });
            if (c === 0) res.push({ f: 2, r: 0, c: r }, { f: 2, r: 0, c: r - 1 }, { f: 2, r: 0, c: r + 1 });
            if (c === 2) res.push({ f: 3, r: 0, c: 2 - r }, { f: 3, r: 0, c: 2 - (r - 1) }, { f: 3, r: 0, c: 2 - (r + 1) });
        } else if (f === 5) {
            if (r === 0) res.push({ f: 0, r: 2, c }, { f: 0, r: 2, c: c - 1 }, { f: 0, r: 2, c: c + 1 });
            if (r === 2) res.push({ f: 1, r: 2, c: 2 - c }, { f: 1, r: 2, c: 2 - (c - 1) }, { f: 1, r: 2, c: 2 - (c + 1) });
            if (c === 0) res.push({ f: 2, r: 2, c: 2 - r }, { f: 2, r: 2, c: 2 - (r - 1) }, { f: 2, r: 2, c: 2 - (r + 1) });
            if (c === 2) res.push({ f: 3, r: 2, c: r }, { f: 3, r: 2, c: r - 1 }, { f: 3, r: 2, c: r + 1 });
        }
        return res.filter(n => n.f >= 0 && n.f < 6 && n.r >= 0 && n.r < 3 && n.c >= 0 && n.c < 3);
    }

    function dfs(f, r, c, index, currentPath, visited) {
        if (index >= upperWord.length) return currentPath;
        if (visited.has(`${f},${r},${c}`)) return null;
        const cellValue = board[f][r][c].toUpperCase();
        let matchLength = 0;
        if (cellValue.includes('/')) {
            const options = cellValue.split('/');
            for (const opt of options) {
                if (opt === 'Q') {
                    if (upperWord.substring(index, index + 2) === 'QU') {
                        matchLength = 2;
                        break;
                    } else if (upperWord[index] === 'Q') {
                        matchLength = 1;
                        break;
                    }
                } else if (upperWord.substring(index).startsWith(opt)) {
                    matchLength = opt.length;
                    break;
                }
            }
        } else if (cellValue === 'Q') {
            if (upperWord.substring(index, index + 2) === 'QU') matchLength = 2;
            else if (upperWord[index] === 'Q') matchLength = 1;
        } else if (upperWord[index] === cellValue) matchLength = 1;
        if (matchLength === 0) return null;
        const newVisited = new Set(visited);
        newVisited.add(`${f},${r},${c}`);
        const newPath = [...currentPath, { f, r, c }];
        const nextIndex = index + matchLength;
        if (nextIndex >= upperWord.length) return newPath;
        for (const n of getCubeNeighbors(f, r, c)) {
            const result = dfs(n.f, n.r, n.c, nextIndex, newPath, newVisited);
            if (result) return result;
        }
        return null;
    }
    for (let f = 0; f < 6; f++) {
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                const path = dfs(f, r, c, 0, [], new Set());
                if (path) return path;
            }
        }
    }
    return null;
}

// Global function to review a round (Legitimacy Walkthrough)
window.watchRoundHistory = function (roomId, roundNum, isSnapshot = false, gameId = null, timestamp = null) {
    console.log(`Reviewing Round ${roundNum} from Room ${roomId} (GameID: ${gameId}, Timestamp: ${timestamp})`);

    // 1. Find Round Data
    let rounds = window.lastRenderedRounds || [];
    let round = null;

    // PRIORITY: Tournament explicitly set replay
    if (roomId && String(roomId).startsWith('tournament_') && window.lastTournamentReplay && window.lastTournamentReplay.room_id === roomId && window.lastTournamentReplay.round_number == roundNum) {
        round = window.lastTournamentReplay;
    }

    // A) If an exact gameId is provided, use it to find the precise profile round.
    if (!round && gameId) {
        round = rounds.find(r => r.game_id == gameId);
    }

    // B) Prefer Lobby History when no gameId — it is definitively from the CURRENT session.
    //    Profile rounds are matched only by room_id+round_number and can span multiple sessions
    //    of the same room, returning stale data from a previous game.
    if (!round && window.lastGameState && window.lastGameState.winners_history && window.lastGameState.room_id === roomId) {
        let foundInLobby = null;
        if (timestamp) {
            foundInLobby = window.lastGameState.winners_history.find(h => h.round == roundNum && Math.abs((h.timestamp || 0) - timestamp) < 5000);
        }
        if (!foundInLobby) {
            foundInLobby = window.lastGameState.winners_history.find(h => h.round == roundNum);
        }
        if (foundInLobby && foundInLobby.board) {
            console.log(`[Review] Using Round ${roundNum} from Lobby winners_history (current session)`);

            // Use the current player's words only if reviewing the currently concluding round,
            // otherwise use the snapshot's recorded words from that round to prevent using current round's words for older rounds!
            let wordsForReplay = foundInLobby.words || [];
            if (window.lastGameState && window.lastGameState.state === 'intermission' && window.lastGameState.current_round === roundNum) {
                if (window.lastGameState.players && window.lastGameState.your_username) {
                    const myUsername = window.lastGameState.your_username;
                    const myPlayer = window.lastGameState.players.find(p => p.username === myUsername);
                    if (myPlayer && myPlayer.submitted_words && myPlayer.submitted_words.length > 0) {
                        const visible = myPlayer.submitted_words.filter(w => w.word && !w.obfuscated && w.word.indexOf('?') === -1);
                        if (visible.length > 0) {
                            wordsForReplay = visible;
                            console.log(`[Review] Using ${wordsForReplay.length} of YOUR own words for current round replay`);
                        }
                    }
                }
            }

            round = {
                ...foundInLobby,
                room_id: roomId,
                round_number: foundInLobby.round,
                total_score: foundInLobby.score,
                game_type: foundInLobby.game_type || 'accumulative',
                words: wordsForReplay
            };
        }
    }

    // C) Fallback to Profile/Recent Rounds (cross-session, matched by room_id+round_number).
    //    Only used when lobby data isn't available (e.g., viewing history from profile page).
    if (!round) {
        round = rounds.find(r => r.room_id == roomId && r.round_number == roundNum);
    }




    if (!round) {
        alert("Round details not available. This round may have happened before the snapshot system was enabled.");
        return;
    }

    // Normalize board if it's a dictionary
    if (round.board && typeof round.board === 'object' && !Array.isArray(round.board)) {
        round.board = round.board.board;
    }

    window.currentActiveReplayRound = round;

    // Helper to format game type name
    function getReplayGameTypeName(r) {
        if (!r) return 'Accumulative';
        const gt = String(r.game_type || r.mode || '').toLowerCase().trim();
        const dims = String(r.dimensions || r.board_dimensions || '').toUpperCase();
        const isCube = (gt === 'cube' || gt === '3d' || gt === 'solo_3d' || gt === '3d_cube' || gt === '3d cube' ||
                        dims.includes('CUBE') || dims.includes('3X3X3') || dims.includes('3D') ||
                        (Array.isArray(r.board) && r.board.length === 6 && Array.isArray(r.board[0]) && Array.isArray(r.board[0][0])));
        if (isCube) return 'Cube';
        if (gt === 'fcfs' || gt === 'solo_fcfs' || gt.includes('first_come') || gt.includes('first come')) return 'First Come First Serve';
        if (gt === 'split' || gt === 'solo_split' || gt.includes('split')) return 'Split Points';
        if (gt === 'tournament') return 'Tournament';
        if (gt === 'accumulative' || gt === 'acc' || gt === 'solo_accumulative') return 'Accumulative';
        if (gt) return gt.charAt(0).toUpperCase() + gt.slice(1);
        return 'Accumulative';
    }

    const gameTypeName = getReplayGameTypeName(round);

    // Update Game Type Display
    const gameTypeEl = document.getElementById('history-review-game-type');
    const intGameTypeEl = document.getElementById('integrated-history-review-game-type');
    if (gameTypeEl) gameTypeEl.innerText = gameTypeName;
    if (intGameTypeEl) intGameTypeEl.innerText = gameTypeName;

    // Update Date Display
    const dateEl = document.getElementById('history-review-date');
    const intDateEl = document.getElementById('integrated-history-review-date');
    let dateStr = '';
    if (round.timestamp) {
        dateStr = typeof window.formatAppDate === 'function' ? window.formatAppDate(round.timestamp, true) : String(round.timestamp);
    }
    if (dateEl) dateEl.innerText = dateStr;
    if (intDateEl) intDateEl.innerText = dateStr;

    // Try to find the Overlay first (preferred for a "window that appears")
    const overlay = document.getElementById('history-review-overlay');
    const integratedPanel = document.getElementById('integrated-replay-panel');

    // Choose IDs based on which UI is available
    const useOverlay = !!overlay;
    const prefix = useOverlay ? 'review' : 'integrated';

    if (useOverlay) {
        overlay.classList.add('forced-show');
        overlay.classList.remove('hidden');
        // Setup Close Handler
        const closeBtn = document.getElementById('close-history-review');
        if (closeBtn) {
            closeBtn.onclick = () => {
                overlay.classList.add('hidden');
                overlay.classList.remove('forced-show');
                if (window.replayInterval) {
                    clearInterval(window.replayInterval);
                    window.replayInterval = null;
                }
            };
        }
    } else if (integratedPanel) {
        integratedPanel.classList.remove('hidden');
        integratedPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // --- Cleanup any existing playback ---
    if (window.replayInterval) {
        clearInterval(window.replayInterval);
        window.replayInterval = null;
    }


    // 2. Reset Replay UI
    const startBtn = document.getElementById(useOverlay ? 'start-replay-btn' : 'integrated-start-btn');
    const skipBtn = document.getElementById(useOverlay ? 'skip-replay-btn' : 'integrated-skip-btn');
    const progressUI = document.getElementById(useOverlay ? 'replay-progress-container' : 'integrated-progress-ui');
    const walkthroughList = document.getElementById(`${prefix}-walkthrough-list`);
    const currentTimeEl = document.getElementById(useOverlay ? 'replay-current-time' : 'integrated-current-time');
    const progressBar = document.getElementById(useOverlay ? 'replay-progress-bar' : 'integrated-progress-bar');

    if (startBtn) {
        startBtn.classList.remove('hidden');
        startBtn.innerText = "▶ Watch Replay";
    }
    if (skipBtn) skipBtn.classList.add('hidden');
    if (progressUI) progressUI.classList.add('hidden');
    if (progressBar) {
        progressBar.style.transition = 'none';
        progressBar.style.width = '0%';
        progressBar.offsetHeight; // Force reflow
        progressBar.style.transition = '';
    }
    if (walkthroughList) walkthroughList.innerHTML = '<p class="placeholder" style="color:rgba(255,255,255,0.3); text-align:center; padding:40px; font-weight:700;">Ready to watch the walkthrough...</p>';

    // 3. Render Board with Dynamic Scaling
    const boardContainer = document.getElementById(`${prefix}-board-container`);
    if (boardContainer && round.board && round.board.length > 0) {
        // Mobile Board Transposition for Replay: orient so the longest side runs HORIZONTALLY
        // (saves vertical space on mobile screens; Replay ≠ gameplay orientation)
        try {
            if (window.innerWidth <= 900 && Array.isArray(round.board[0])) {
                const isReplay3D = round.board.length === 6 && Array.isArray(round.board[0]) && Array.isArray(round.board[0][0]);
                if (!isReplay3D) {
                    const rows = round.board.length;
                    const cols = round.board[0].length;
                    // Transpose only when board is taller than wide (so longest side ends up horizontal)
                    if (rows > cols) {
                        const transposed = [];
                        for (let c = 0; c < cols; c++) {
                            transposed[c] = [];
                            for (let r = 0; r < rows; r++) {
                                transposed[c][r] = (round.board[r] && round.board[r][c] !== undefined) ? round.board[r][c] : '';
                            }
                        }
                        round.board = transposed;
                    }
                }
            }
        } catch (transpositionError) {
            console.error("[Replay] Transposition failed safely:", transpositionError);
        }

        const rows = round.board.length;
        const cols = round.board[0].length;

        // Use a small delay to ensure modal layout is stable
        setTimeout(() => {
            const layoutMain = document.querySelector('.history-review-layout') || document.getElementById('integrated-replay-panel');
            if (!layoutMain) return;

            let availWidth = boardContainer.parentElement.clientWidth * 0.6; // Board area usually gets ~60%
            let availHeight = layoutMain.clientHeight - 80; // Minus header/padding padding
            let gap = 12;

            if (window.innerWidth <= 900) {
                availWidth = boardContainer.parentElement.clientWidth - 40; // Full width of stacked area
                availHeight = 350; // Balanced vertical height on mobile
                gap = 6; // Compact gap on mobile
            }

            // Calculate max cell size to fit width and height constraints
            const maxCellW = (availWidth - (cols - 1) * gap - 20) / cols;
            const maxCellH = (availHeight - (rows - 1) * gap - 20) / rows;

            // Optimal cell size (capped for aesthetics on 4x4)
            const cellSize = Math.floor(Math.min(maxCellW, maxCellH, 120));
            const fontSize = Math.floor(cellSize * 0.6) + 'px';


            const is3D = rows === 6 && (Array.isArray(round.board[0]) && Array.isArray(round.board[0][0]) || Array.isArray(round.board[0]) && round.board[0].length === 3);
            if (is3D) {
                // On mobile: 2 columns × 3 rows of faces. On desktop: 3 columns × 2 rows.
                const isMobile3D = window.innerWidth <= 900;
                const faceCols = isMobile3D ? 2 : 3;
                const faceGap = isMobile3D ? 10 : 20;
                const cellGap = 4;
                const pad = isMobile3D ? 12 : 20;
                const faceRows = isMobile3D ? 3 : 2;

                // Calculate non-cell overhead for accurate cell size
                // Horizontal: faceCols faces × 3 cells; faceGap between faces; cellGap within faces; pad on sides
                const nonCellW = (faceCols - 1) * faceGap + faceCols * 2 * cellGap + 2 * pad;
                const maxCellW3D = (availWidth - nonCellW) / (faceCols * 3);

                // Vertical: faceRows faces × 3 cells
                const nonCellH = (faceRows - 1) * faceGap + faceRows * 2 * cellGap + 2 * pad;
                const maxCellH3D = (availHeight - nonCellH) / (faceRows * 3);

                const cellSize3D = Math.max(22, Math.floor(Math.min(maxCellW3D, maxCellH3D, 50)));
                const fontSize3D = Math.floor(cellSize3D * 0.55) + 'px';

                // boardContainer = full-width block centering host (no background here)
                boardContainer.style.cssText = '';          // clear any stale inline styles
                boardContainer.style.display = 'block';
                boardContainer.style.textAlign = 'center';

                // Inner wrapper: inline-grid so it sizes to content → background hugs faces
                const innerGrid = document.createElement('div');
                innerGrid.style.display = 'inline-grid';
                innerGrid.style.gridTemplateColumns = `repeat(${faceCols}, max-content)`;
                innerGrid.style.gap = `${faceGap}px`;
                innerGrid.style.padding = `${pad}px`;
                innerGrid.style.background = 'rgba(0,0,0,0.2)';
                innerGrid.style.borderRadius = '15px';
                innerGrid.style.verticalAlign = 'top';

                innerGrid.innerHTML = round.board.map((face, fIdx) => {
                    let faceHTML = '';
                    for (let r = 0; r < 3; r++) {
                        for (let c = 0; c < 3; c++) {
                            const val = (face[r] && face[r][c] !== undefined) ? face[r][c] : '?';
                            const displayVal = val === 'Q' ? 'QU' : val;
                            faceHTML += `<div class="review-cell" style="width:${cellSize3D}px;height:${cellSize3D}px;font-size:${fontSize3D};border-radius:4px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">${displayVal}</div>`;
                        }
                    }
                    return `
                        <div style="display:flex;flex-direction:column;align-items:center;gap:${isMobile3D ? 4 : 8}px;flex-shrink:0;">
                            <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);font-weight:900;text-transform:uppercase;white-space:nowrap;">Face ${fIdx}</div>
                            <div style="display:grid;grid-template-columns:repeat(3,${cellSize3D}px);gap:${cellGap}px;flex-shrink:0;">
                                ${faceHTML}
                            </div>
                        </div>
                    `;
                }).join('');

                boardContainer.innerHTML = '';
                boardContainer.appendChild(innerGrid);
                return;
            }

            const flatBoard = round.board.flat();

            // Clear any stale inline styles from a previous 3D board render
            boardContainer.style.cssText = '';
            boardContainer.style.display = 'grid';
            boardContainer.style.gridTemplateColumns = `repeat(${cols}, ${cellSize}px)`;
            boardContainer.style.gridTemplateRows = `repeat(${rows}, ${cellSize}px)`;
            boardContainer.style.gap = `${gap}px`;

            boardContainer.innerHTML = flatBoard.map((letter, i) => {
                return `
                    <div class="review-cell" style="width: ${cellSize}px; height: ${cellSize}px; font-size: ${fontSize}">${letter}</div>
                `;
            }).join('');

            console.log(`[Replay] Scaled ${cols}x${rows} board to ${cellSize}px cells`);
        }, 50);
    }

    // 4. Playback Logic
    const rawWords = round.words || [];
    // Prefer stored round_duration; fall back to the live game state's time_limit for the
    // current room (covers older winners_history entries that predate the round_duration field);
    // last resort is 60s.
    const liveTimeLimitForRoom = (window.lastGameState && window.lastGameState.room_id === round.room_id)
        ? window.lastGameState.time_limit
        : null;
    const roundDuration = round.round_duration || liveTimeLimitForRoom || 60;

    // START TIME LOGIC:
    // Preferred: round_start_time (absolute s)
    // Fallback 1: First word timestamp - 2s
    // Fallback 2: Entry timestamp (converted to s)
    let startTime = 0;
    if (round.round_start_time) {
        startTime = parseFloat(round.round_start_time);
    } else if (round.timestamp) {
        const parsedDate = window.parseUTCTimestamp ? window.parseUTCTimestamp(round.timestamp) : new Date(round.timestamp);
        const tVal = parsedDate.getTime() / 1000.0;
        startTime = isNaN(tVal) ? (Date.now() / 1000) - roundDuration : tVal - roundDuration;
    } else {
        startTime = (Date.now() / 1000) - roundDuration;
    }

    // Normalize and convert all timestamps to SECONDS relative to epoch
    let processedWords = rawWords.map(w => {
        let ts = 0;
        // 1. Support multiple possible keys: timestamp, time, time_offset
        if (w.timestamp !== undefined && w.timestamp !== null) {
            ts = parseFloat(w.timestamp);
        } else if (w.time !== undefined && w.time !== null) {
            ts = parseFloat(w.time);
        } else if (w.time_offset !== undefined && w.time_offset !== null) {
            ts = startTime + parseFloat(w.time_offset);
        } else {
            ts = startTime; // Fallback to start
        }

        // 2. Detect millisecond vs second timestamps
        if (ts > 1000000000000) {
            ts = ts / 1000.0;
        }

        return {
            ...w,
            timestamp: ts
        };
    });

    // Sort words chronologically
    processedWords.sort((a, b) => {
        const tA = Number(a.timestamp) || 0;
        const tB = Number(b.timestamp) || 0;
        return tA - tB;
    });

    // 3. Fallback: If all words have nearly identical timestamps, distribute them evenly
    // (e.g., if they were batch-submitted at the end of a round)
    const allSameTime = processedWords.length > 1 && processedWords.every((w, idx, arr) => 
        idx === 0 || Math.abs(w.timestamp - arr[0].timestamp) < 0.1
    );

    if (allSameTime || (processedWords.length === 1 && Math.abs(processedWords[0].timestamp - startTime) < 0.1)) {
        console.log(`[Replay-Fallback] Batch/identical timestamps detected. Spacing ${processedWords.length} words evenly.`);
        const N = processedWords.length;
        processedWords = processedWords.map((w, idx) => {
            // Distribute them evenly over the first 85% of the round duration so they don't hit the absolute end
            const offset = (idx + 1) * ((roundDuration * 0.85) / (N + 1));
            return {
                ...w,
                timestamp: startTime + offset
            };
        });
    }

    const sortedWords = processedWords;

    console.log(`[Review] Playback Setup: ${sortedWords.length} words, duration ${roundDuration}s, startTime ${startTime}`);
    if (sortedWords.length > 0) {
        console.log(`[Review] First Word: ${sortedWords[0].word} @ ${sortedWords[0].timestamp}, Rel: ${sortedWords[0].timestamp - startTime}`);
    }

    const renderWord = (word) => {
        const wTimestamp = parseFloat(word.timestamp) || 0;
        let relTimeSec = Math.max(0, wTimestamp - startTime);

        const min = Math.floor(relTimeSec / 60);
        const sec = (relTimeSec % 60).toFixed(1);
        const timeStr = `${min}:${sec.padStart(4, '0')}`;

        // Styling based on points
        let ptsClass = 'walkthrough-pts';
        if (word.points < 0) ptsClass += ' penalty';
        if (word.is_bonus) ptsClass += ' bonus';

        return `
        <div class="walkthrough-item reveal" style="cursor: pointer;" onclick="highlightWordPathOnReplay('${word.word.replace(/'/g, "\\'")}')">
            <span class="walkthrough-time">${timeStr}</span>
            <span class="walkthrough-word">${word.word}</span>
            <span class="${ptsClass}">${word.points} pts</span>
        </div>
        `;
    };

    const showAllWords = () => {
        // Always show in chronological order (the order words were found)
        const displayWords = [...sortedWords];

        if (walkthroughList) {
            // Always first-found first (chronological) on all screen sizes
            const htmlContent = displayWords.map(w => renderWord(w)).join('');
            walkthroughList.innerHTML = htmlContent;
            if (sortedWords.length === 0) {
                walkthroughList.innerHTML = '<p class="placeholder" style="color:rgba(255,255,255,0.2); text-align:center; padding:40px;">No words found in this round.</p>';
            }
            // Auto-scroll to bottom to show latest? Or top? 
            // Usually start at top.
            walkthroughList.scrollTop = 0;
        }

        // Ensure "Show All" is hidden, and "Watch" is visible (reset state)
        if (skipBtn) skipBtn.classList.add('hidden');
        if (startBtn) {
            startBtn.classList.remove('hidden');
            startBtn.innerText = "▶ Watch Replay"; // Reset text
        }
        if (progressUI) progressUI.classList.add('hidden');
        if (progressBar) {
            progressBar.style.transition = 'none';
            progressBar.style.width = '0%';
            progressBar.offsetHeight; // Force reflow
            progressBar.style.transition = '';
        }

        const currentScoreEl = document.getElementById(useOverlay ? 'replay-current-score' : 'integrated-current-score');
        if (currentScoreEl) currentScoreEl.innerText = `${round.total_score} pts`;
    };

    // ALWAYS Show All Words initially
    showAllWords();

    // Snapshot Mode logic merged with above (always show initially)

    // Interactive Replay
    if (startBtn) {
        startBtn.onclick = () => {
            console.log(`[Review] Starting Replay...`);
            
            // Bulletproof cleanup: Stop any currently running interval to prevent overlap
            if (window.replayInterval) {
                clearInterval(window.replayInterval);
                window.replayInterval = null;
            }

            if (progressBar) {
                progressBar.style.transition = 'none';
                progressBar.style.width = '0%';
                progressBar.offsetHeight; // Force reflow
                progressBar.style.transition = '';
            }

            startBtn.classList.add('hidden');
            if (skipBtn) skipBtn.classList.remove('hidden');
            if (progressUI) progressUI.classList.remove('hidden');

            // Score tracking (internal + UI updates)
            let currentScore = 0;
            const currentScoreEl = document.getElementById(useOverlay ? 'replay-current-score' : 'integrated-current-score');
            if (currentScoreEl) currentScoreEl.innerText = "0 pts";

            if (walkthroughList) walkthroughList.innerHTML = ''; // CLEAR LIST FOR ANIMATION

            let elapsed = 0;
            let wordIndex = 0;
            const tick = 100;

            // Clear Highlights
            if (boardContainer) {
                boardContainer.querySelectorAll('.review-cell').forEach(c => c.classList.remove('highlight'));
            }

            window.replayInterval = setInterval(() => {
                elapsed += tick / 1000;

                // Update Progress — cap display at roundDuration so timer never shows past end of round
                const displayElapsed = Math.min(elapsed, roundDuration);
                if (progressBar) progressBar.style.width = `${(displayElapsed / roundDuration) * 100}%`;
                if (currentTimeEl) {
                    const m = Math.floor(displayElapsed / 60);
                    const s = (displayElapsed % 60).toFixed(1);
                    currentTimeEl.innerText = `${m}:${s.padStart(4, '0')}`;
                }

                // Append new words in order
                while (wordIndex < sortedWords.length) {
                    const word = sortedWords[wordIndex];
                    const wTimestamp = parseFloat(word.timestamp) || 0;
                    const relWordTime = wTimestamp - startTime;

                    if (elapsed >= relWordTime || isNaN(relWordTime)) {
                        console.log(`[Review] Displaying word: ${word.word} (relative: ${relWordTime ? relWordTime.toFixed(1) : 'NaN'}s)`);

                        try {
                            // Insert at TOP on mobile, or BOTTOM on desktop
                            if (walkthroughList) {
                                // Always display the (new) word just displayed at the top of the list
                                walkthroughList.insertAdjacentHTML('afterbegin', renderWord(word));
                                walkthroughList.scrollTop = 0; // Keep scrolled to top so the newest is visible
                            }

                            currentScore += word.points;
                            if (currentScoreEl) currentScoreEl.innerText = `${currentScore} pts`;

                            // Highlight Path
                            const rows = round.board.length;
                            const firstRow = round.board[0];
                            const is3D = rows === 6 && Array.isArray(firstRow) && Array.isArray(firstRow[0]);

                            if (is3D) {
                                // 3D Cube Highlighting
                                const path = findWordPathOnCube(word.word, round.board);
                                if (path && boardContainer) {
                                    const cells = boardContainer.querySelectorAll('.review-cell');
                                    // Clear and apply new highlight
                                    cells.forEach(c => c.classList.remove('highlight'));
                                    path.forEach((p, i) => {
                                        // Index is f*9 + r*3 + c
                                        const idx = p.f * 9 + p.r * 3 + p.c;
                                        setTimeout(() => {
                                            if (cells[idx]) cells[idx].classList.add('highlight');
                                        }, i * 40);
                                    });
                                }
                            } else {
                                // 2D Board Highlighting
                                const path = findWordPath(round.board, word.word);
                                if (path && boardContainer) {
                                    const cells = boardContainer.querySelectorAll('.review-cell');
                                    const gridCols = round.board[0].length;

                                    // Clear and apply new highlight
                                    cells.forEach(c => c.classList.remove('highlight'));
                                    path.forEach((p, i) => {
                                        const idx = p.row * gridCols + p.col;
                                        setTimeout(() => {
                                            if (cells[idx]) cells[idx].classList.add('highlight');
                                        }, i * 40);
                                    });
                                }
                            }
                        } catch (err) {
                            console.error("[Review] Error processing word in replay:", err);
                        } finally {
                            // IMPORTANT: Increment wordIndex even if rendering fails to prevent infinite loops!
                            wordIndex++;
                        }
                    } else {
                        break;
                    }
                }

                if (elapsed >= roundDuration) {
                    // Flush any remaining words before stopping (catches words near round's end)
                    while (wordIndex < sortedWords.length) {
                        try {
                            const word = sortedWords[wordIndex];
                            currentScore += word.points;
                            if (currentScoreEl) currentScoreEl.innerText = `${currentScore} pts`;
                        } catch (err) {
                            console.error('[Review] Error flushing word at round end:', err);
                        } finally {
                            wordIndex++;
                        }
                    }
                    if (window.replayInterval) clearInterval(window.replayInterval);
                    window.replayInterval = null;
                    if (skipBtn) skipBtn.classList.add('hidden');
                    if (startBtn) {
                        startBtn.classList.remove('hidden');
                        startBtn.innerText = "↺ Replay";
                    }
                    // Reset list to show all words in chronological order (first found at the top) when complete
                    showAllWords();
                }
            }, tick);
        };
    }

    if (skipBtn) {
        skipBtn.onclick = () => {
            if (window.replayInterval) clearInterval(window.replayInterval);
            showAllWords();
        };
    }
};



// Global initialization for follow button
document.addEventListener('DOMContentLoaded', () => {
    const followBtn = document.getElementById('profile-follow-btn');
    if (followBtn) {
        followBtn.addEventListener('click', async () => {
            const roomId = document.getElementById('profile-current-room').value;
            if (!roomId) return;

            console.log(`Following user to room: ${roomId}`);

            try {
                // 1. Fetch room state to check player count and rating limits
                const resp = await fetch(`/api/room/${roomId}/state`);
                const data = await resp.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // 2. Decide if join as player or spectator
                // Rule: Max 8 players; or rating outside set limits -> Spectator
                const playerCount = (data.players && data.players.length) || 0;
                const isFull = playerCount >= (Number(data.max_players) || 8);

                const roomMin = Number(data.min_rating) || 0;
                const roomMax = Number(data.max_rating) || 9999;
                const hasLimits = (roomMin > 0 || roomMax < 9999);

                let userRating = (data.your_rating !== undefined && data.your_rating !== null) ? Number(data.your_rating) : 1200;
                if (data.your_rating === undefined || data.your_rating === null) {
                    if (typeof getUserConfigRating === 'function') {
                        userRating = getUserConfigRating(data.game_type, data.board_dimensions, data.time_limit);
                    } else if (window.currentUserConfigRatings) {
                        const cfgKey = `${(data.game_type || '').replace('solo_', '')}|${data.board_dimensions}|${data.time_limit}`;
                        const rObj = window.currentUserConfigRatings[cfgKey];
                        if (rObj && rObj.rating !== undefined) userRating = rObj.rating;
                    }
                }

                const currentUser = window.currentUser || '';
                const isGuest = !currentUser || currentUser.startsWith('Guest_') || Boolean(window.currentUserIsGuest);
                const isRatingOutOfRange = hasLimits && (userRating < roomMin || userRating > roomMax || isGuest);

                const shouldSpectate = isFull || isRatingOutOfRange;
                console.log(`Following user: Room ${roomId}, Population: ${playerCount} (Full: ${isFull}), User Rating: ${userRating} (Req: ${roomMin}-${roomMax}), Out of Range: ${isRatingOutOfRange} -> Spectator: ${shouldSpectate}`);

                // 3. Join the room
                const joinResp = await fetch(`/api/room/${roomId}/join`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ as_spectator: shouldSpectate })
                });
                const joinData = await joinResp.json();

                if (joinData.success) {
                    const isSpectator = Boolean(joinData.role === 'spectator' || shouldSpectate);
                    window.currentRoomId = roomId;
                    window.isSpectatorMode = isSpectator;
                    localStorage.setItem('last_joined_room', roomId);

                    const playBtn = document.getElementById('play-btn');
                    if (playBtn) {
                        playBtn.disabled = false;
                        playBtn.title = "";
                    }
                    if (window.updateManualToolState) window.updateManualToolState();

                    if (typeof showPage === 'function') {
                        showPage('page-play');
                    }

                    if (window.startGamePolling) window.startGamePolling();

                    // Force focus for Word Input if not spectator
                    setTimeout(() => {
                        const input = document.getElementById('word-input');
                        if (input && !window.isSpectatorMode) {
                            input.disabled = false;
                            input.focus();
                        }
                    }, 100);
                } else {
                    alert('Failed to follow: ' + (joinData.error || 'Unknown error'));
                }

            } catch (err) {
                console.error('Follow error:', err);
                alert('Follow failed due to network error.');
            }
        });
    }

    // Modal Achievement Tab Listeners
    document.querySelectorAll('.modal-tabs .ach-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            if (!currentAchConfig) return;
            const period = tab.dataset.period;
            showRoomAchievements(
                currentAchConfig.username,
                currentAchConfig.mode,
                currentAchConfig.board,
                currentAchConfig.time,
                period
            );
        });
    });

    // Profile Page Period Tab Listeners (Rankings & Exceptional)
    document.querySelectorAll('.ach-tab-profile').forEach(tab => {
        tab.addEventListener('click', () => {
            const period = tab.dataset.period;

            // Sync all profile timeframe tabs
            document.querySelectorAll(`.ach-tab-profile[data-period="${period}"]`).forEach(t => t.classList.add('active'));
            document.querySelectorAll(`.ach-tab-profile:not([data-period="${period}"])`).forEach(t => t.classList.remove('active'));

            // Refresh profile with new period
            const displayedName = document.getElementById('profile-username')?.innerText;
            if (displayedName && displayedName !== 'Player') {
                const activeTabEl = document.querySelector('.profile-tab-toggle.active');
                const activeTab = activeTabEl ? activeTabEl.dataset.tab : 'rankings';
                performProfileSearch(displayedName, activeTab, period);
            }
        });
    });
});

function updateRankingsFilterDropdowns(selectedMode) {
    const dimsSelect = document.getElementById('rankings-filter-dims');
    const timeSelect = document.getElementById('rankings-filter-time');
    if (!dimsSelect || !timeSelect) return;

    // Define valid dimensions and times per game mode matching the Lobby exactly
    const modeConfig = {
        'all': {
            dims: [
                { value: '4x4', text: '4x4' },
                { value: '4x6', text: '4x6' },
                { value: '5x7', text: '5x7' },
                { value: '6x8', text: '6x8' },
                { value: '3x3x3', text: '3x3x3 Cube' }
            ],
            times: [
                { value: '45', text: '45 Seconds' },
                { value: '180', text: '3 Minutes' },
                { value: '300', text: '5 Minutes' },
                { value: '600', text: '10 Minutes' },
                { value: '86400', text: '24 Hours' }
            ]
        },
        'accumulative': {
            dims: [
                { value: '4x4', text: '4x4' },
                { value: '4x6', text: '4x6' },
                { value: '5x7', text: '5x7' },
                { value: '6x8', text: '6x8' }
            ],
            times: [
                { value: '45', text: '45 Seconds' },
                { value: '180', text: '3 Minutes' },
                { value: '600', text: '10 Minutes' },
                { value: '86400', text: '24 Hours' }
            ]
        },
        'fcfs': {
            dims: [
                { value: '4x4', text: '4x4' },
                { value: '4x6', text: '4x6' },
                { value: '5x7', text: '5x7' },
                { value: '6x8', text: '6x8' }
            ],
            times: [
                { value: '45', text: '45 Seconds' },
                { value: '180', text: '3 Minutes' }
            ]
        },
        'split': {
            dims: [
                { value: '4x4', text: '4x4' },
                { value: '4x6', text: '4x6' },
                { value: '5x7', text: '5x7' },
                { value: '6x8', text: '6x8' }
            ],
            times: [
                { value: '45', text: '45 Seconds' },
                { value: '180', text: '3 Minutes' }
            ]
        },
        '3d': {
            dims: [
                { value: '3x3x3', text: '3x3x3 Cube' }
            ],
            times: [
                { value: '180', text: '3 Minutes' },
                { value: '300', text: '5 Minutes' },
                { value: '600', text: '10 Minutes' }
            ]
        }
    };

    const cfg = modeConfig[selectedMode] || modeConfig['all'];

    // Update Dimensions dropdown
    const currentDim = dimsSelect.value;
    dimsSelect.innerHTML = '<option value="all">All Sizes</option>' + 
        cfg.dims.map(d => `<option value="${d.value}">${d.text}</option>`).join('');
    if (cfg.dims.some(d => d.value === currentDim) || currentDim === 'all') {
        dimsSelect.value = currentDim;
    } else {
        dimsSelect.value = 'all';
    }

    // Update Time Limit dropdown
    const currentTime = timeSelect.value;
    timeSelect.innerHTML = '<option value="all">All Times</option>' + 
        cfg.times.map(t => `<option value="${t.value}">${t.text}</option>`).join('');
    if (cfg.times.some(t => t.value === currentTime) || currentTime === 'all') {
        timeSelect.value = currentTime;
    } else {
        timeSelect.value = 'all';
    }
}
window.updateRankingsFilterDropdowns = updateRankingsFilterDropdowns;

function renderRatingsGrid(configRatings, user = null) {
    const grid = document.getElementById('profile-ratings-grid');
    if (!grid) {
        console.warn('[Tools] #profile-ratings-grid not found in DOM.');
        return;
    }

    // Cache the original data and user on the element if not already there
    if (configRatings) grid._configRatings = configRatings;
    if (user) grid._user = user;

    const ratings = grid._configRatings || {};
    const u = grid._user || null;

    grid.innerHTML = '';

    const modeSelect = document.getElementById('rankings-filter-mode');
    const filterMode = modeSelect?.value || 'all';
    const filterDims = document.getElementById('rankings-filter-dims')?.value || 'all';
    const filterTime = document.getElementById('rankings-filter-time')?.value || 'all';

    const modes = ['accumulative', 'fcfs', 'split', '3d'];
    const boards = ['4x4', '4x6', '5x7', '6x8', '3x3x3'];
    const allTimes = [45, 180, 300, 600, 86400];

    const formatTimeShort = (s) => {
        if (s === 45) return '45s';
        if (s === 180) return '3m';
        if (s === 300) return '5m';
        if (s === 600) return '10m';
        if (s === 86400) return '24h';
        return s + 's';
    };

    let visibleCount = 0;

    modes.forEach(mode => {
        if (filterMode !== 'all' && mode !== filterMode) return;

        boards.forEach(board => {
            if (filterDims !== 'all' && board !== filterDims) return;

            // COMPATIBILITY FILTER: 3x3x3 is for Cube ONLY; traditional boards for others
            if (mode === '3d' && board !== '3x3x3') return;
            if (mode !== '3d' && board === '3x3x3') return;

            allTimes.forEach(time => {
                if (filterTime !== 'all' && String(time) !== filterTime) return;

                // COMPATIBILITY FILTER: Cube supports 3m (180), 5m (300), 10m (600)
                if (mode === '3d' && time !== 180 && time !== 300 && time !== 600) return;

                // COMPATIBILITY FILTER: Accumulative supports 45s (45), 3m (180), 10m (600), 24h (86400)
                if (mode === 'accumulative' && time !== 45 && time !== 180 && time !== 600 && time !== 86400) return;

                // COMPATIBILITY FILTER: FCFS and Split support 45s (45) and 3m (180)
                if ((mode === 'fcfs' || mode === 'split') && time !== 45 && time !== 180) return;

                const configKey = `${mode}|${board}|${time}`;
                const configData = ratings[configKey] || { rating: 1200, games_played: 0, wins: 0, point_sum: 0, avg_score: 0, avg_words: 0, avg_pct_found: 0 };
                const rating = configData.rating;

                const rColor = window.getRatingColor ? window.getRatingColor(rating) : '#b3b3b3';

                const box = document.createElement('div');
                box.className = 'rating-box clickable';
                box.title = "Click to view achievements for this room type";
                box.style.cssText = 'cursor: pointer; display: flex; align-items: center; gap: 12px; padding: 12px 14px; border-radius: 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); transition: background 0.2s, transform 0.2s; touch-action: manipulation; -webkit-tap-highlight-color: transparent; user-select: none; -webkit-user-select: none; outline: none;';
                box.onmouseenter = () => { box.style.background = 'rgba(255,255,255,0.08)'; };
                box.onmouseleave = () => { box.style.background = 'rgba(255,255,255,0.03)'; };
                box.innerHTML = `
                    <div class="rating-box-swatch" style="background: ${rColor};"></div>
                    <div class="rating-box-info" style="flex: 1;">
                        <div class="rating-box-mode" style="font-size: 0.65rem; color: rgba(255,255,255,0.4); text-transform: uppercase; font-weight: 800;">${mode === '3d' ? 'CUBE' : mode}</div>
                        <div class="rating-box-config" style="font-weight: 700;">${board} | ${formatTimeShort(time)}</div>
                        <div style="display: flex; flex-direction: column; gap: 2px; margin-top: 4px; font-size: 0.65rem; color: rgba(255,255,255,0.3); font-weight: 700;">
                           <div>Played: <span style="color: #fff;">${configData.games_played || 0}</span> | Wins: <span style="color: #fff;">${configData.wins || 0}</span></div>
                           <div>Avg Score: <span style="color: #fff;">${configData.avg_score || 0}</span> | Total Points: <span style="color: #fff;">${configData.point_sum || 0}</span></div>
                           <div>Avg Words: <span style="color: #fff;">${configData.avg_words || 0}</span> | Avg Found: <span style="color: #fff;">${configData.avg_pct_found || 0}%</span></div>
                        </div>
                    </div>
                    <div class="rating-box-value" style="color: ${rColor}; font-size: 1.25rem; font-weight: 900; margin: 0 15px;">${rating}</div>
                `;

                const handleBoxClick = (e) => {
                    if (e) {
                        try { e.preventDefault(); e.stopPropagation(); } catch (err) {}
                    }
                    const targetUsername = (u && u.username)
                        ? u.username
                        : (window.currentProfileUsername || document.getElementById('profile-username')?.innerText?.trim() || (typeof window.currentUser === 'object' ? window.currentUser?.username : window.currentUser) || (typeof currentUser !== 'undefined' ? (currentUser.username || currentUser) : null) || localStorage.getItem('morpheme_username') || null);
                    if (targetUsername) {
                        showRoomAchievements(targetUsername, mode, board, time);
                    } else {
                        console.warn('[Achievements] Could not determine username for achievement lookup.');
                    }
                };

                box.onclick = handleBoxClick;

                grid.appendChild(box);
                visibleCount++;
            });
        });
    });

    if (visibleCount === 0) {
        grid.innerHTML = '<p class="placeholder" style="grid-column: 1 / -1; text-align: center; padding: 40px; color: rgba(255,255,255,0.2);">No room types match the selected filters.</p>';
    }

    // Setup filter listeners once
    if (!grid._filtersInitialized) {
        const modeSelectEl = document.getElementById('rankings-filter-mode');
        if (modeSelectEl) {
            modeSelectEl.onchange = () => {
                updateRankingsFilterDropdowns(modeSelectEl.value);
                renderRatingsGrid();
            };
        }
        const selects = ['rankings-filter-mode', 'rankings-filter-dims', 'rankings-filter-time'];
        selects.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                if (id !== 'rankings-filter-mode') {
                    el.onchange = () => renderRatingsGrid();
                }
                // Hover effect for select
                el.onmouseenter = () => el.style.borderColor = 'rgba(255,255,255,0.3)';
                el.onmouseleave = () => el.style.borderColor = 'rgba(255,255,255,0.1)';
            }
        });
        grid._filtersInitialized = true;
    }
}

function setupImageLightbox() {
    const modal = document.getElementById('image-lightbox-modal');
    const closeBtn = document.getElementById('image-lightbox-close');

    if (modal && closeBtn) {
        const closeModal = () => {
            modal.classList.add('hidden');
            modal.classList.remove('forced-show');
        };
        closeBtn.onclick = closeModal;
        modal.onclick = (e) => {
            if (e.target === modal) closeModal();
        };

        // ESC key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });
    }

    // Also setup achievement modal
    const achModal = document.getElementById('room-achievements-modal');
    const achClose = document.getElementById('achievement-modal-close');
    if (achModal && achClose) {
        const close = () => {
            achModal.classList.add('hidden');
            achModal.classList.remove('forced-show');
            achModal.style.display = 'none';
            achModal.style.opacity = '0';
            achModal.style.pointerEvents = 'none';
        };
        achClose.onclick = close;
        achModal.onclick = (e) => {
            if (e.target === achModal) close();
        };
    }
}

// Achievement state tracking for period switching
let currentAchConfig = null;

async function showRoomAchievements(username, mode, board, time, period = 'all') {
    const modal = document.getElementById('room-achievements-modal');
    if (!modal) { console.error('[Achievements] Modal element not found'); return; }

    // Track state for period switching
    currentAchConfig = { username, mode, board, time };

    // Capture Scroll Position to prevent jumping to top on filter change
    const card = modal.querySelector('.achievement-card');
    let previousScroll = 0;
    if (!modal.classList.contains('hidden') && card) {
        previousScroll = card.scrollTop;
    }

    // Show modal first so any crash in tab/title setup is visible
    modal.classList.remove('hidden');
    modal.classList.add('forced-show');
    modal.style.display = 'flex';
    modal.style.opacity = '1';
    modal.style.pointerEvents = 'auto';

    // Update tab UI
    const tabs = document.querySelectorAll('.modal-tabs .ach-tab');
    tabs.forEach(tab => {
        if (tab.dataset.period === period) tab.classList.add('active');
        else tab.classList.remove('active');
    });

    // Set titles (null-guarded)
    const titleEl = document.getElementById('achievement-title');
    const subtitleEl = document.getElementById('achievement-subtitle');
    if (titleEl) titleEl.textContent = `${username}'s Achievements`;
    if (subtitleEl) subtitleEl.textContent =
        `${mode.charAt(0).toUpperCase() + mode.slice(1)} ${board} | ${time < 300 ? time + 's' : (time / 60) + 'm'}`;

    try {
        const response = await fetch(`/api/profile/${username}/achievements/${mode}/${board}/${time}?period=${period}`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        // Update Rating
        document.getElementById('achievement-rating-val').textContent = data.rating || 1200;

        // Reset fields helpers
        const setFields = (obj, mapping) => {
            for (const [key, id] of Object.entries(mapping)) {
                const el = document.getElementById(id);
                if (el) {
                    if (key.includes('word') && typeof obj[key] === 'object') {
                        el.textContent = obj[key].word ? `${obj[key].word} (${obj[key].points} pts)` : 'None';
                    } else {
                        el.textContent = obj[key] || (typeof obj[key] === 'number' ? '0' : '-');
                    }
                }
            }
        };

        // 1. Populate Global Stats (Top Sections) - Always All-Time
        if (data.global_stats) {
            setFields(data.global_stats, {
                'high_score': 'ach-high-score',
                'max_words': 'ach-max-words',
                'longest_word': 'ach-longest-word',
                'best_word': 'ach-best-word',
                'games_played': 'ach-games-played',
                'wins': 'ach-wins'
            });
            // Ensure labels are static
            document.getElementById('ach-label-bests').textContent = 'Personal Bests';
            document.getElementById('ach-label-stats').textContent = 'Lifetime Stats';

            // Win rate for global
            const gwr = (data.global_stats.wins / data.global_stats.games_played * 100).toFixed(1);
            document.getElementById('ach-win-rate').textContent = (isNaN(gwr) ? '0' : gwr) + '%';
            document.getElementById('ach-total-words').textContent = data.global_stats.total_words;

        }

        // 2. Populate Period Stats (Bottom Lists)
        if (!data.stats) {
            // Period specific tables - Clear ALL to prevent historical bleed-over
            if (document.getElementById('ach-table-perf')) document.getElementById('ach-table-perf').innerHTML = '';
            if (document.getElementById('ach-table-wins')) document.getElementById('ach-table-wins').innerHTML = '';
            if (document.getElementById('ach-table-recent')) document.getElementById('ach-table-recent').innerHTML = '';
            if (document.getElementById('ach-table-words')) document.getElementById('ach-table-words').innerHTML = '';
            if (document.getElementById('ach-table-scores')) document.getElementById('ach-table-scores').innerHTML = '';
            if (document.getElementById('ach-table-wordcounts')) document.getElementById('ach-table-wordcounts').innerHTML = '';
            if (document.getElementById('ach-table-pcts')) document.getElementById('ach-table-pcts').innerHTML = '';
            if (document.getElementById('ach-table-obscure')) document.getElementById('ach-table-obscure').innerHTML = '';
            return;
        }

        const stats = data.stats;

        // Cache all retrieved rounds in window.lastRenderedRounds by their unique game_id
        if (!window.lastRenderedRounds) window.lastRenderedRounds = [];
        const retrievedRounds = [
            ...(stats.exceptional_rounds || []),
            ...(stats.winning_rounds || []),
            ...(stats.recent_rounds || []),
            ...(stats.best_scores || []),
            ...(stats.best_word_counts || []),
            ...(stats.best_pcts || []),
            ...(stats.best_obscure || []),
            ...(stats.best_words_rounds || [])
        ];
        retrievedRounds.forEach(r => {
            if (r && r.game_id) {
                const exists = window.lastRenderedRounds.some(cr => cr.game_id === r.game_id);
                if (!exists) {
                    window.lastRenderedRounds.push(r);
                }
            }
        });

        // Average and Period specific labels
        document.getElementById('ach-avg-perf').textContent = stats.avg_perf || '-';
        const achGreatestPeEl = document.getElementById('ach-greatest-pe');
        if (achGreatestPeEl) {
            achGreatestPeEl.textContent = stats.max_pe || '0';
        }
        document.getElementById('ach-avg-winrate').textContent = (stats.win_rate || 0) + '%';
        document.getElementById('ach-total-games').textContent = stats.games_played || '0';
        document.getElementById('ach-avg-score').textContent = (stats.avg_score || 0).toLocaleString();
        document.getElementById('ach-avg-words').textContent = stats.avg_words || '0';
        document.getElementById('ach-avg-word-pts').textContent = stats.avg_word_pts || '0';
        
        const achAvgPctEl = document.getElementById('ach-avg-pct');
        if (achAvgPctEl) {
            achAvgPctEl.textContent = (stats.avg_pct_found || 0) + '%';
            if (stats.max_pct_found > 50) {
                achAvgPctEl.textContent += ` (Max Words Found: ${stats.max_pct_found}%)`;
                achAvgPctEl.style.color = '#ff4a4a';
                achAvgPctEl.style.fontWeight = '800';
            } else {
                achAvgPctEl.style.color = '#60a5fa'; // Reset color
                achAvgPctEl.style.fontWeight = '800';
            }
        }

        const achAvgPctHeaderEl = document.getElementById('ach-avg-pct-header');
        if (achAvgPctHeaderEl) {
            achAvgPctHeaderEl.textContent = (stats.avg_pct_found || 0) + '%';
        }

        const parseDate = (isoStr) => {
            if (!isoStr) return new Date();
            const dateStr = isoStr.includes('Z') || isoStr.includes('+') ? isoStr.replace(' ', 'T') : isoStr.replace(' ', 'T') + 'Z';
            return new Date(dateStr);
        };

        const renderAchRow = (r, cols) => {
            // Cache if not present or upgrade if better data available
            if (!window.lastRenderedRounds) window.lastRenderedRounds = [];
            const existingIdx = window.lastRenderedRounds.findIndex(cr => cr.room_id === r.room_id && cr.round_number === r.round_number);

            if (existingIdx === -1) {
                window.lastRenderedRounds.push(r);
            } else if (r.game_id && !window.lastRenderedRounds[existingIdx].game_id) {
                // Overwrite with better data (contains game_id)
                window.lastRenderedRounds[existingIdx] = r;
            } else if (r.game_id && window.lastRenderedRounds[existingIdx].game_id !== r.game_id) {
                // If distinct game IDs for what looks like same room/round (collision?), push as new
                window.lastRenderedRounds.push(r);
            }

            return `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03); cursor: pointer; transition: background 0.2s;" 
                onmouseenter="this.style.background='rgba(255,255,255,0.02)'" 
                onmouseleave="this.style.background='transparent'" 
                onclick="watchRoundHistory('${r.room_id}', ${r.round_number}, true, ${r.game_id || 'null'});">
                ${cols.map(c => `<td style="padding: 10px 15px; ${c.style || ''}">${c.val}</td>`).join('')}
                <td style="padding: 10px 15px; text-align: right;"><div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 4px 8px; display: inline-block;">📷</div></td>
            </tr>`;
        };

        // 1. Exceptional Performances
        const tablePerf = document.getElementById('ach-table-perf');
        if (tablePerf && stats.exceptional_rounds) {
            // Sort chronologically (Conveyor Belt): Timestamp DESC, then Ratio DESC
            const sortedByTimestamp = [...stats.exceptional_rounds].sort((a, b) => {
                const dateDiff = window.parseUTCTimestamp(b.timestamp) - window.parseUTCTimestamp(a.timestamp);
                if (dateDiff !== 0) return dateDiff;
                return b.ratio - a.ratio;
            });
            tablePerf.innerHTML = sortedByTimestamp.map(r => renderAchRow(r, [
                { val: r.performance_value, style: 'font-weight: 800; color: #60a5fa;' },
                { val: r.ratio + 'x', style: 'color: rgba(255,255,255,0.6);' },
                { val: r.total_score, style: 'font-weight: 700;' },
                { val: `<div style="font-size: 0.75rem;">${r.num_words} words</div><div style="font-size: 0.6rem; color: rgba(255,255,255,0.3);">${r.top_word}</div>` },
                { val: dateToShort(parseDate(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 2. Winning Rounds
        const tableWins = document.getElementById('ach-table-wins');
        if (tableWins && stats.winning_rounds) {
            // Sort by Score DESC (Impressiveness), then Timestamp DESC
            const sortedWins = [...stats.winning_rounds].sort((a, b) => {
                if (b.total_score !== a.total_score) return b.total_score - a.total_score;
                return window.parseUTCTimestamp(b.timestamp) - window.parseUTCTimestamp(a.timestamp);
            });
            tableWins.innerHTML = sortedWins.map(r => renderAchRow(r, [
                { val: r.total_score, style: 'font-weight: 800; color: #4ade80;' },
                { val: r.performance_value, style: 'font-weight: 700;' },
                { val: r.all_players.length, style: 'color: rgba(255,255,255,0.5);' },
                { val: dateToShort(parseDate(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 3. Games Played (Recent list - keep as Recency)
        const tableRecent = document.getElementById('ach-table-recent');
        if (tableRecent && stats.recent_rounds) {
            // Sort by Timestamp (True Recency for "Recent" list)
            const sortedRecent = [...stats.recent_rounds].sort((a, b) => window.parseUTCTimestamp(b.timestamp) - window.parseUTCTimestamp(a.timestamp));
            tableRecent.innerHTML = sortedRecent.map(r => renderAchRow(r, [
                { val: r.total_score, style: 'font-weight: 700;' },
                { val: r.ratio + 'x', style: 'color: rgba(255,255,255,0.4); font-size: 0.75rem;' },
                { val: r.is_win ? '<span style="color:#4ade80">WIN</span>' : '<span style="color:rgba(255,255,255,0.3)">-</span>', style: 'font-weight: 800;' },
                { val: dateToShort(parseDate(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 4. Best Scores
        const tableScores = document.getElementById('ach-table-scores');
        if (tableScores && stats.best_scores) {
            const sortedByScore = [...stats.best_scores].sort((a, b) => b.total_score - a.total_score);
            tableScores.innerHTML = sortedByScore.map(r => renderAchRow(r, [
                { val: r.total_score, style: 'font-weight: 800; color: #ffd700;' },
                { val: r.performance_value, style: 'font-weight: 700;' },
                { val: dateToShort(parseDate(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 5. Best Word Counts
        const tableWordCounts = document.getElementById('ach-table-wordcounts');
        if (tableWordCounts && stats.best_word_counts) {
            const sortedByCount = [...stats.best_word_counts].sort((a, b) => b.num_words - a.num_words);
            tableWordCounts.innerHTML = sortedByCount.map(r => renderAchRow(r, [
                { val: r.num_words, style: 'font-weight: 800; color: #a5b4fc;' },
                { val: `${r.avg_len} len | ${r.pct_found || 0}%`, style: 'color: rgba(255,255,255,0.6);' },
                { val: dateToShort(parseDate(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 5b. Best Percentages
        const tablePcts = document.getElementById('ach-table-pcts');
        if (tablePcts && stats.best_pcts) {
            const sortedByPct = [...stats.best_pcts].sort((a, b) => b.pct_found - a.pct_found);
            tablePcts.innerHTML = sortedByPct.map(r => renderAchRow(r, [
                { val: `${r.pct_found}%`, style: 'font-weight: 800; color: #ff4a4a;' },
                { val: `Pts: ${r.total_score}`, style: 'color: rgba(255,255,255,0.6);' },
                { val: dateToShort(parseDate(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 5c. Best Obscure Counts
        const tableObscure = document.getElementById('ach-table-obscure');
        if (tableObscure && stats.best_obscure) {
            const sortedByObscure = [...stats.best_obscure].sort((a, b) => b.obscure_count - a.obscure_count);
            tableObscure.innerHTML = sortedByObscure.map(r => renderAchRow(r, [
                { val: r.obscure_count, style: 'font-weight: 800; color: #60a5fa;' },
                { val: `Pts: ${r.total_score}`, style: 'color: rgba(255,255,255,0.6);' },
                { val: dateToShort(parseDate(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 6. Best Words (Individual)
        const tableWords = document.getElementById('ach-table-words');
        if (tableWords && stats.best_words) {
            const sortedByPoints = [...stats.best_words].sort((a, b) => b.points - a.points);
            tableWords.innerHTML = sortedByPoints.map(w => {
                const date = new Date(w.timestamp);
                return `
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03); cursor: pointer; transition: background 0.2s;" 
                    onmouseenter="this.style.background='rgba(255,255,255,0.02)'" 
                    onmouseleave="this.style.background='transparent'" 
                    onclick="watchRoundHistory('${w.room_id}', ${w.round_number}, true, ${w.game_id || 'null'});">
                    <td style="padding: 10px 15px; font-weight: 800; color: #fff; text-transform: uppercase;">${w.word}</td>
                    <td style="padding: 10px 15px; font-weight: 700; color: #ffd700;">${w.points}</td>
                    <td style="padding: 10px 15px; color: rgba(255,255,255,0.5);">${w.word.length}</td>
                    <td style="padding: 10px 15px; font-size: 0.75rem; color: rgba(255,255,255,0.4);">${dateToShort(date)}</td>
                    <td style="padding: 10px 15px; text-align: right;"><div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 4px 8px; display: inline-block;">📷</div></td>
                </tr>`;
            }).join('');
        }

        function dateToShort(d) {
            return typeof window.formatAppDate === 'function' ? window.formatAppDate(d) : d.toLocaleDateString();
        }

        // Restore Scroll Position
        if (card && previousScroll > 0) {
            setTimeout(() => {
                card.scrollTop = previousScroll;
            }, 0);
        }

    } catch (err) {
        console.error("Failed to fetch achievements:", err);
        // showToast?
    }
}
window.showRoomAchievements = showRoomAchievements;

function showImageLightbox(url, caption = "") {
    const modal = document.getElementById('image-lightbox-modal');
    const img = document.getElementById('image-lightbox-img');
    const captionEl = document.getElementById('image-lightbox-caption');

    if (!modal || !img) return;

    img.src = url;
    if (captionEl) captionEl.innerText = caption;

    modal.classList.remove('hidden');
}
window.showImageLightbox = showImageLightbox;

function setupProfileEditing(isOwner) {
    const editableFields = [
        { id: 'profile-full-name', key: 'full_name', placeholder: 'Full Name' },
        { id: 'profile-age-val', key: 'age', placeholder: 'Age' },
        { id: 'profile-gender-val', key: 'gender', placeholder: 'Gender' },
        { id: 'profile-location-val', key: 'location', placeholder: 'Location' },
        { id: 'profile-quote-val', key: 'quote', placeholder: 'Enter a personal quote' },
        { id: 'profile-description-val', key: 'description', placeholder: 'Add a detailed description about yourself...' }
    ];

    editableFields.forEach(field => {
        const el = document.getElementById(field.id);
        if (!el) return;

        if (isOwner) {
            el.contentEditable = "true";
            el.title = "Click to edit";
            el.dataset.placeholder = field.placeholder;

            // Remove existing to avoid double listeners if re-rendered
            const newEl = el.cloneNode(true);
            el.parentNode.replaceChild(newEl, el);

            newEl.addEventListener('blur', () => {
                saveProfileField(field.key, newEl.innerText.trim());
                if (field.key === 'description' && typeof initCustomScrollbarForElement === 'function') {
                    initCustomScrollbarForElement('profile-description-val', 'profile-desc-scrollbar-track', 'profile-desc-scrollbar-thumb');
                }
            });
            newEl.addEventListener('input', () => {
                if (field.key === 'description' && typeof initCustomScrollbarForElement === 'function') {
                    initCustomScrollbarForElement('profile-description-val', 'profile-desc-scrollbar-track', 'profile-desc-scrollbar-thumb');
                }
            });
            newEl.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && field.key !== 'description') {
                    e.preventDefault();
                    newEl.blur();
                }
            });
        } else {
            el.contentEditable = "false";
            el.title = "";
        }
    });
}

async function saveProfileField(key, value) {
    try {
        const response = await fetch('/api/profile/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [key]: value })
        });
        const data = await response.json();
        if (data.error) {
            console.error("Profile update failed:", data.error);
        }
    } catch (err) {
        console.error("Profile update error:", err);
    }
}

// --- Lists Tool Logic ---

let listsDataLoaded = false;
let currentWordsList = [];
let currentWordsRenderedCount = 0;
let currentWordsType = '';
const WORDS_PAGE_SIZE = 200; // Smaller chunks for buttery-smooth mobile rendering
let currentProgressiveLoadId = 0;
let listsFetchAbortController = null;
let listsFetchTimeoutId = null; // Module-level so it can be cancelled on re-fetch
let listsShowAll = false;

function startProgressiveRendering() {
    const loadId = ++currentProgressiveLoadId;
    
    function renderChunk() {
        if (loadId !== currentProgressiveLoadId) return;
        const maxAllowed = listsShowAll ? Infinity : 10000;
        if (currentWordsRenderedCount >= Math.min(currentWordsList.length, maxAllowed)) {
            if ((currentWordsList.length > maxAllowed || window.listsServerTruncated) && !document.getElementById('list-truncation-notice')) {
                const scrollArea = document.getElementById('main-list-results');
                if (scrollArea) {
                    let noticeHtml = '';
                    if (window.listsServerTruncated) {
                        noticeHtml = `
                            <div id="list-truncation-notice" style="padding: 15px; text-align: center; color: #ffb703; font-weight: 500; border-top: 1px dashed rgba(255, 255, 255, 0.1); margin-top: 10px;">
                                ⚠️ Showing first ${currentWordsList.length.toLocaleString()} words.<br>
                                <span style="font-size: 0.82rem; opacity: 0.8; font-weight: normal;">
                                    Please select a specific <strong>word length</strong> or <strong>starting letter</strong> to filter and see more.
                                </span>
                            </div>
                        `;
                    } else {
                        noticeHtml = `
                            <div id="list-truncation-notice" style="padding: 15px; text-align: center; color: #ffb703; font-weight: 500; border-top: 1px dashed rgba(255, 255, 255, 0.1); margin-top: 10px;">
                                ⚠️ Showing first 10,000 words.<br>
                                <span style="font-size: 0.82rem; opacity: 0.8; font-weight: normal;">
                                    Please select a specific <strong>word length</strong> or <strong>starting letter</strong> to narrow down the list, or 
                                    <button id="show-all-words-btn" style="background: rgba(255, 183, 3, 0.15); border: 1px solid #ffb703; color: #ffb703; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; font-weight: 600; margin-left: 5px; transition: all 0.2s;" onmouseover="this.style.background='rgba(255, 183, 3, 0.3)'" onmouseout="this.style.background='rgba(255, 183, 3, 0.15)'">Load All ${currentWordsList.length.toLocaleString()} Words</button>
                                </span>
                            </div>
                        `;
                    }
                    scrollArea.insertAdjacentHTML('beforeend', noticeHtml);
                    const btn = document.getElementById('show-all-words-btn');
                    if (btn) {
                        btn.onclick = () => {
                            listsShowAll = true;
                            const notice = document.getElementById('list-truncation-notice');
                            if (notice) notice.remove();
                            startProgressiveRendering();
                        };
                    }
                }
            }
            return;
        }
        
        renderNextWordsPage();
        setTimeout(renderChunk, 100);
    }
    
    renderChunk();
}

function renderNextWordsPage() {
    const scrollArea = document.getElementById('main-list-results');
    if (!scrollArea || currentWordsRenderedCount >= currentWordsList.length) return;

    const maxAllowed = listsShowAll ? Infinity : 10000;
    if (currentWordsRenderedCount >= maxAllowed) {
        if (!document.getElementById('list-truncation-notice')) {
            let noticeHtml = '';
            if (window.listsServerTruncated) {
                noticeHtml = `
                    <div id="list-truncation-notice" style="padding: 15px; text-align: center; color: #ffb703; font-weight: 500; border-top: 1px dashed rgba(255, 255, 255, 0.1); margin-top: 10px;">
                        ⚠️ Showing first ${currentWordsList.length.toLocaleString()} words.<br>
                        <span style="font-size: 0.82rem; opacity: 0.8; font-weight: normal;">
                            Please select a specific <strong>word length</strong> or <strong>starting letter</strong> to filter and see more.
                        </span>
                    </div>
                `;
            } else {
                noticeHtml = `
                    <div id="list-truncation-notice" style="padding: 15px; text-align: center; color: #ffb703; font-weight: 500; border-top: 1px dashed rgba(255, 255, 255, 0.1); margin-top: 10px;">
                        ⚠️ Showing first 10,000 words.<br>
                        <span style="font-size: 0.82rem; opacity: 0.8; font-weight: normal;">
                            Please select a specific <strong>word length</strong> or <strong>starting letter</strong> to narrow down the list, or 
                            <button id="show-all-words-btn" style="background: rgba(255, 183, 3, 0.15); border: 1px solid #ffb703; color: #ffb703; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; font-weight: 600; margin-left: 5px; transition: all 0.2s;" onmouseover="this.style.background='rgba(255, 183, 3, 0.3)'" onmouseout="this.style.background='rgba(255, 183, 3, 0.15)'">Load All ${currentWordsList.length.toLocaleString()} Words</button>
                        </span>
                    </div>
                `;
            }
            scrollArea.insertAdjacentHTML('beforeend', noticeHtml);
            const btn = document.getElementById('show-all-words-btn');
            if (btn) {
                btn.onclick = () => {
                    listsShowAll = true;
                    const notice = document.getElementById('list-truncation-notice');
                    if (notice) notice.remove();
                    startProgressiveRendering();
                };
            }
        }
        return;
    }

    const nextPageWords = currentWordsList.slice(
        currentWordsRenderedCount,
        Math.min(currentWordsRenderedCount + WORDS_PAGE_SIZE, maxAllowed)
    );

    let html = '';
    if (currentWordsType === 'likelihood') {
        html = nextPageWords.map(item => `
            <div class="list-item">
                <span class="likelihood-score">${item.score}</span> <span class="clickable-word-link" onclick="window.lookupWord('${item.word}', event)">${item.word}</span>
            </div>
        `).join('');
    } else if (currentWordsType === 'added') {
        const isMod = window.currentUserIsMod;
        html = nextPageWords.map(w => `
            <div class="list-item added-word" style="display: flex; justify-content: space-between; align-items: center;">
                <span class="clickable-word-link" onclick="window.lookupWord('${w}', event)">${w}</span>
                ${isMod ? `<button onclick="removeAddedWordFromTools('${w}')" style="background:none; border:none; color:#f43f5e; cursor:pointer; font-weight:bold; padding:0 5px;" title="Remove">&times;</button>` : ''}
            </div>
        `).join('');
    } else {
        html = nextPageWords.map(w => `<div class="list-item"><span class="clickable-word-link" onclick="window.lookupWord('${w}', event)">${w}</span></div>`).join('');
    }

    if (currentWordsRenderedCount === 0) {
        scrollArea.innerHTML = html;
        scrollArea.scrollTop = 0;
    } else {
        scrollArea.insertAdjacentHTML('beforeend', html);
    }

    currentWordsRenderedCount += nextPageWords.length;
}

function setupListsTool() {
    const updateBtn = document.getElementById('list-update-btn');
    if (updateBtn) {
        updateBtn.addEventListener('click', () => {
            listsDataLoaded = false; // Force refresh
            fetchListsData();
        });
    }

    const typeFilter = document.getElementById('list-type-filter');
    if (typeFilter) {
        typeFilter.addEventListener('change', () => {
            fetchListsData();
        });
    }

    const mainListEl = document.getElementById('main-list-results');
    if (mainListEl) {
        mainListEl.addEventListener('selectstart', (e) => e.preventDefault());
        mainListEl.addEventListener('dragstart', (e) => e.preventDefault());
    }

    const listsContEl = document.getElementById('lists-container');
    if (listsContEl) {
        listsContEl.addEventListener('selectstart', (e) => e.preventDefault());
        listsContEl.addEventListener('dragstart', (e) => e.preventDefault());
    }

    function scrollListsToTopOptions() {
        const scrollArea = document.getElementById('main-list-results');
        if (scrollArea) {
            scrollArea.scrollTo({ top: 0, behavior: 'smooth' });
        }
        const listsPane = document.getElementById('tool-lists');
        if (listsPane) {
            listsPane.scrollTo({ top: 0, behavior: 'smooth' });
        }
        const toolsContent = document.querySelector('.tools-content');
        if (toolsContent) {
            toolsContent.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }

// --- View Full List Modal & High-Performance Virtual Windowing ---
window.isFullListLoading = false;
let _fullListAllWords = [];
let _fullListWindowStart = 0;
const FULL_LIST_WINDOW_SIZE = 800; // Optimal rendering chunk: <1ms render time, zero memory bloat

function showFullListToast(msg, isError = false) {
    const toast = document.getElementById('full-list-jump-toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.style.borderColor = isError ? 'rgba(244, 63, 94, 0.6)' : 'rgba(167, 139, 250, 0.6)';
    toast.style.color = isError ? '#fca5a5' : '#e9d5ff';
    toast.style.display = 'block';
    toast.style.opacity = '1';
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => { toast.style.display = 'none'; }, 200);
    }, 2800);
}
window.showFullListToast = showFullListToast;

function generateFullListItemsHtml(slice) {
    if (!slice || slice.length === 0) return '';
    const wordType = (typeof currentWordsType !== 'undefined' ? currentWordsType : 'nwl');
    if (wordType === 'likelihood') {
        return slice.map(item => `<span class="full-list-item" data-word="${item.word}"><span class="likelihood-score">${item.score}</span> <span class="clickable-word-link" onclick="window.lookupWord('${item.word}', event)">${item.word}</span></span>`).join('');
    } else {
        return slice.map(w => `<span class="full-list-item" data-word="${w}"><span class="clickable-word-link" onclick="window.lookupWord('${w}', event)">${w}</span></span>`).join('');
    }
}

function renderFullListWindow(startIndex, targetWordToHighlight = null) {
    const resultsEl = document.getElementById('full-list-modal-results');
    const countEl = document.getElementById('full-list-modal-count');
    if (!resultsEl || !_fullListAllWords || _fullListAllWords.length === 0) return;

    const total = _fullListAllWords.length;
    const maxStart = Math.max(0, total - FULL_LIST_WINDOW_SIZE);
    const clampedStart = Math.max(0, Math.min(maxStart, startIndex));
    _fullListWindowStart = clampedStart;

    const slice = _fullListAllWords.slice(clampedStart, clampedStart + FULL_LIST_WINDOW_SIZE);
    resultsEl.innerHTML = generateFullListItemsHtml(slice);

    if (countEl) {
        countEl.textContent = `${total.toLocaleString()} words`;
    }

    updateFullListVirtualScrollbar();

    if (targetWordToHighlight) {
        requestAnimationFrame(() => {
            let targetEl = null;
            try {
                targetEl = resultsEl.querySelector(`.full-list-item[data-word="${CSS.escape(targetWordToHighlight)}"]`);
            } catch (e) {}

            if (targetEl) {
                targetEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
                targetEl.classList.add('jump-target-pulse');
                setTimeout(() => { targetEl.classList.remove('jump-target-pulse'); }, 2500);
            }
        });
    }
}

function updateFullListVirtualScrollbar() {
    const track = document.getElementById('full-list-scrollbar-track');
    const thumb = document.getElementById('full-list-scrollbar-thumb');
    const resultsEl = document.getElementById('full-list-modal-results');
    if (!track || !thumb || !resultsEl || !_fullListAllWords || _fullListAllWords.length === 0) return;

    const total = _fullListAllWords.length;
    if (total <= 50) {
        track.style.display = 'none';
        return;
    }
    track.style.display = 'block';

    const trackHeight = track.clientHeight || resultsEl.clientHeight;
    const ratio = Math.max(0.08, Math.min(1, FULL_LIST_WINDOW_SIZE / total));
    const thumbHeight = Math.max(28, Math.min(trackHeight, trackHeight * ratio));
    thumb.style.height = `${thumbHeight}px`;

    const maxThumbTop = Math.max(0, trackHeight - thumbHeight);
    const maxStartIndex = Math.max(1, total - FULL_LIST_WINDOW_SIZE);
    const progress = Math.min(1, Math.max(0, _fullListWindowStart / maxStartIndex));
    const thumbTop = progress * maxThumbTop;

    thumb.style.top = `${thumbTop}px`;
    thumb.style.setProperty('top', `${thumbTop}px`, 'important');
}

function initFullListVirtualScrollbar() {
    const results = document.getElementById('full-list-modal-results');
    const track = document.getElementById('full-list-scrollbar-track');
    const thumb = document.getElementById('full-list-scrollbar-thumb');
    if (!results || !track || !thumb) return;

    let isDragging = false;
    let startY = 0;
    let startThumbTop = 0;
    let _dragRafId = null;

    function applyDragPosition(clientY) {
        if (!_fullListAllWords || _fullListAllWords.length === 0) return;
        const total = _fullListAllWords.length;
        const trackHeight = track.clientHeight || results.clientHeight;
        const thumbHeight = thumb.offsetHeight;
        const maxThumbTop = Math.max(0, trackHeight - thumbHeight);
        if (maxThumbTop <= 0) return;

        const deltaY = clientY - startY;
        let newThumbTop = Math.max(0, Math.min(maxThumbTop, startThumbTop + deltaY));

        thumb.style.top = `${newThumbTop}px`;
        thumb.style.setProperty('top', `${newThumbTop}px`, 'important');

        const progress = newThumbTop / maxThumbTop;
        const maxStartIndex = Math.max(0, total - FULL_LIST_WINDOW_SIZE);
        const targetIndex = Math.floor(progress * maxStartIndex);

        if (_dragRafId) cancelAnimationFrame(_dragRafId);
        _dragRafId = requestAnimationFrame(() => {
            _dragRafId = null;
            renderFullListWindow(targetIndex);
        });
    }

    function onDragMove(e) {
        if (!isDragging) return;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        applyDragPosition(clientY);
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
        }
    }

    function onDragStart(e) {
        isDragging = true;
        thumb.classList.add('dragging');
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        startY = clientY;
        startThumbTop = parseFloat(thumb.style.top) || 0;
        document.body.style.userSelect = 'none';

        document.addEventListener('mousemove', onDragMove);
        document.addEventListener('mouseup', onDragEnd);
        document.addEventListener('touchmove', onDragMove, { passive: false });
        document.addEventListener('touchend', onDragEnd);
        document.addEventListener('touchcancel', onDragEnd);

        if (e.cancelable !== false) e.preventDefault();
    }

    thumb.onmousedown = onDragStart;
    thumb.ontouchstart = onDragStart;

    track.onmousedown = (e) => {
        if (e.target === thumb) return;
        const rect = track.getBoundingClientRect();
        const clickY = e.clientY - rect.top;
        const thumbHeight = thumb.offsetHeight;
        startThumbTop = Math.max(0, clickY - thumbHeight / 2);
        startY = e.clientY;
        onDragStart(e);
        applyDragPosition(e.clientY);
    };

    track.ontouchstart = (e) => {
        if (e.target === thumb) return;
        const rect = track.getBoundingClientRect();
        const touchY = e.touches[0].clientY - rect.top;
        const thumbHeight = thumb.offsetHeight;
        startThumbTop = Math.max(0, touchY - thumbHeight / 2);
        startY = e.touches[0].clientY;
        onDragStart(e);
        applyDragPosition(e.touches[0].clientY);
    };

    // Smooth continuous window pagination when scrolling inside word grid
    results.onscroll = () => {
        if (isDragging) return;
        if (!_fullListAllWords || _fullListAllWords.length === 0) return;
        const total = _fullListAllWords.length;
        const maxStartIndex = Math.max(0, total - FULL_LIST_WINDOW_SIZE);

        const scrollHeight = results.scrollHeight;
        const scrollTop = results.scrollTop;
        const clientHeight = results.clientHeight;

        if (scrollTop + clientHeight >= scrollHeight - 40 && _fullListWindowStart < maxStartIndex) {
            const nextStart = Math.min(maxStartIndex, _fullListWindowStart + 200);
            renderFullListWindow(nextStart);
            results.scrollTop = Math.max(0, scrollTop - 100);
        } else if (scrollTop <= 30 && _fullListWindowStart > 0) {
            const prevStart = Math.max(0, _fullListWindowStart - 200);
            renderFullListWindow(prevStart);
            results.scrollTop = 100;
        }
    };
}

function handleFullListWordJump() {
    const input = document.getElementById('full-list-jump-input');
    if (!input) return;
    const query = input.value.trim().toUpperCase();
    input.blur();
    if (!query) return;

    if (!_fullListAllWords || _fullListAllWords.length === 0) {
        _fullListAllWords = (typeof currentWordsList !== 'undefined' && currentWordsList) ? currentWordsList : [];
    }

    const total = _fullListAllWords.length;
    let targetIdx = -1;
    const wordType = (typeof currentWordsType !== 'undefined' ? currentWordsType : 'nwl');
    if (wordType === 'likelihood') {
        targetIdx = _fullListAllWords.findIndex(item => (typeof item === 'object' ? item.word : item).toUpperCase() === query);
    } else {
        targetIdx = _fullListAllWords.findIndex(w => (typeof w === 'object' ? w.word : w).toUpperCase() === query);
    }

    if (targetIdx === -1) {
        showFullListToast(`"${query}" was not found in this list.`, true);
        return;
    }

    // Center window around target word
    const targetStart = Math.max(0, targetIdx - 40);
    renderFullListWindow(targetStart, query);
}
window.handleFullListWordJump = handleFullListWordJump;

window.openFullListModal = function() {
    console.log('[Full List] Opening full list modal');
    let modal = document.getElementById('full-list-modal');
    let results = document.getElementById('full-list-modal-results');
    if (!modal || !results) {
        console.error('[Full List] Modal or results element missing in DOM');
        return;
    }

    if (modal.parentElement !== document.body) {
        document.body.appendChild(modal);
    }

    const lengthSelect = document.getElementById('list-length-filter');
    const startSelect = document.getElementById('list-start-filter');
    const typeSelect = document.getElementById('list-type-filter');
    const selectedType = (typeof currentWordsType !== 'undefined' && currentWordsType) ? currentWordsType : (typeSelect ? typeSelect.value : 'nwl');
    const selectedLength = lengthSelect ? lengthSelect.value : 'all';
    const selectedStart = startSelect ? startSelect.value : 'all';
    const currentFilterKey = `${selectedType}_${selectedLength}_${selectedStart}`;

    const titleEl = document.getElementById('list-display-title');
    const fullListTitle = document.getElementById('full-list-modal-title');
    if (fullListTitle) fullListTitle.textContent = (titleEl ? titleEl.textContent : 'Full List');

    const fullListJumpInput = document.getElementById('full-list-jump-input');
    if (fullListJumpInput) fullListJumpInput.value = '';

    const fullListJumpToast = document.getElementById('full-list-jump-toast');
    if (fullListJumpToast) fullListJumpToast.style.display = 'none';

    const fullListCount = document.getElementById('full-list-modal-count');

    window._lastFullListFilterKey = currentFilterKey;
    _fullListWindowStart = 0;

    results.innerHTML = '<div style="padding: 40px; text-align: center; color: #a78bfa; font-size: 1.1rem; font-weight: 700; width: 100%;">Loading words…</div>';
    results.scrollTop = 0;

    modal.classList.add('active');
    modal.style.display = 'flex';
    modal.style.setProperty('display', 'flex', 'important');
    modal.style.setProperty('visibility', 'visible', 'important');
    modal.style.setProperty('opacity', '1', 'important');
    modal.style.setProperty('z-index', '9999999', 'important');
    document.body.style.overflow = 'hidden';
    window.isFullListLoading = true;

    if (fullListCount) {
        fullListCount.textContent = `Loading…`;
    }

    const sortAlphabetical = (list, type) => {
        if (!list || list.length === 0) return [];
        if (type === 'likelihood') return list;
        return list.slice().sort((a, b) => {
            const wa = (typeof a === 'object' ? a.word : a) || '';
            const wb = (typeof b === 'object' ? b.word : b) || '';
            return wa.localeCompare(wb);
        });
    };

    if (window._cachedFullWordLists[currentFilterKey]) {
        const fullWords = sortAlphabetical(window._cachedFullWordLists[currentFilterKey], selectedType);
        _fullListAllWords = fullWords;
        window.isFullListLoading = false;
        initFullListVirtualScrollbar();
        renderFullListWindow(0);
        return;
    }

    let url = `/api/tools/lists?list_type=${selectedType}&no_limit=true`;
    if (selectedLength && selectedLength !== 'all') url += `&length=${selectedLength}`;
    if (selectedStart && selectedStart !== 'all') url += `&starts_with=${selectedStart}`;
    url += `&t=${Date.now()}`;

    fetch(url)
        .then(r => r.json())
        .then(data => {
            if (window._lastFullListFilterKey !== currentFilterKey) return;
            const rawWords = data[selectedType] || data['nwl'] || data['added'] || data['csw'] || [];
            const fullWords = sortAlphabetical(rawWords, selectedType);
            _fullListAllWords = fullWords;
            window._cachedFullWordLists[currentFilterKey] = fullWords;
            window.isFullListLoading = false;
            initFullListVirtualScrollbar();
            renderFullListWindow(0);
        })
        .catch(err => {
            console.error('[Full List] Failed to fetch full word list:', err);
            if (fullListCount) fullListCount.textContent = `Fetch failed`;
            window.isFullListLoading = false;
        });
};

window.closeFullListModal = function() {
    const modal = document.getElementById('full-list-modal');
    if (modal) {
        modal.classList.remove('active');
        modal.style.display = 'none';
        modal.style.setProperty('display', 'none', 'important');
    }
    document.body.style.overflow = '';
    window.isFullListLoading = false;
};

const fullListJumpBtnEl = document.getElementById('full-list-jump-btn');
if (fullListJumpBtnEl) {
    fullListJumpBtnEl.addEventListener('click', handleFullListWordJump);
}

const fullListJumpInputEl = document.getElementById('full-list-jump-input');
if (fullListJumpInputEl) {
    fullListJumpInputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleFullListWordJump();
        }
    });
}

const viewFullBtnEl = document.getElementById('list-view-full-btn');
if (viewFullBtnEl) {
    viewFullBtnEl.addEventListener('click', window.openFullListModal);
}

const fullListCloseEl = document.getElementById('full-list-modal-close');
if (fullListCloseEl) {
    fullListCloseEl.addEventListener('click', window.closeFullListModal);
}

    const colHeaderTitle = document.getElementById('list-column-header-title');
    if (colHeaderTitle) {
        colHeaderTitle.addEventListener('click', (e) => {
            if (e.target.closest('#list-view-full-btn')) {
                return;
            }
            const filterRow = document.querySelector('.list-filter-row');
            if (filterRow) {
                filterRow.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    }

    // Initialize custom draggable scrollbar
    initCustomScrollbar();
}

function initCustomScrollbarForElement(scrollAreaId, trackId, thumbId) {
    const scrollArea = typeof scrollAreaId === 'string' ? document.getElementById(scrollAreaId) : scrollAreaId;
    const track = typeof trackId === 'string' ? document.getElementById(trackId) : trackId;
    const thumb = typeof thumbId === 'string' ? document.getElementById(thumbId) : thumbId;
    if (!scrollArea || !track || !thumb) return;

    let isDragging = false;
    let startY = 0;
    let startThumbTop = 0;
    let _rafId = null;

    function updateThumb() {
        if (isDragging) return; // Never override position while user is actively dragging

        const scrollHeight = scrollArea.scrollHeight;
        const clientHeight = scrollArea.clientHeight;
        const scrollTop = scrollArea.scrollTop;

        // Show scrollbar only if list content overflows
        if (scrollHeight <= clientHeight + 5 || clientHeight <= 0) {
            track.style.display = 'none';
            return;
        }
        track.style.display = 'block';

        const trackHeight = track.clientHeight || clientHeight;
        const ratio = Math.min(1, clientHeight / scrollHeight);
        const thumbHeight = Math.max(28, Math.min(trackHeight, trackHeight * ratio));
        thumb.style.height = `${thumbHeight}px`;

        const maxScrollTop = scrollHeight - clientHeight;
        const maxThumbTop = Math.max(0, trackHeight - thumbHeight);
        const thumbTop = maxScrollTop > 0 ? (scrollTop / maxScrollTop) * maxThumbTop : 0;
        thumb.style.top = `${thumbTop}px`;
        thumb.style.setProperty('top', `${thumbTop}px`, 'important');
    }

    function scheduleUpdate() {
        if (_rafId) return;
        _rafId = requestAnimationFrame(() => {
            _rafId = null;
            updateThumb();
        });
    }

    // Expose update function directly on the scrollArea DOM element
    scrollArea._updateCustomScrollbar = scheduleUpdate;

    // Cleanup previous observers on this scrollArea if re-initialized
    if (scrollArea._customScrollbarRO) {
        try { scrollArea._customScrollbarRO.disconnect(); } catch (_) {}
    }
    if (scrollArea._customScrollbarMO) {
        try { scrollArea._customScrollbarMO.disconnect(); } catch (_) {}
    }

    // Bind event listeners for scroll
    scrollArea.addEventListener('scroll', scheduleUpdate, { passive: true });

    // Watch with ResizeObserver throttled via requestAnimationFrame
    if (window.ResizeObserver) {
        const ro = new ResizeObserver(scheduleUpdate);
        ro.observe(scrollArea);
        scrollArea._customScrollbarRO = ro;
    } else {
        window.addEventListener('resize', scheduleUpdate, { passive: true });
    }

    // Watch with MutationObserver so newly appended/prepended children dynamically resize & reposition thumb
    if (window.MutationObserver) {
        const mo = new MutationObserver(scheduleUpdate);
        mo.observe(scrollArea, { childList: true });
        scrollArea._customScrollbarMO = mo;
    }

    function onDragMove(e) {
        if (!isDragging) return;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const deltaY = clientY - startY;

        const trackHeight = track.clientHeight || scrollArea.clientHeight;
        const thumbHeight = thumb.offsetHeight;
        const maxThumbTop = Math.max(0, trackHeight - thumbHeight);

        let newThumbTop = startThumbTop + deltaY;
        newThumbTop = Math.max(0, Math.min(maxThumbTop, newThumbTop));

        thumb.style.top = `${newThumbTop}px`;
        thumb.style.setProperty('top', `${newThumbTop}px`, 'important');

        const scrollHeight = scrollArea.scrollHeight;
        const clientHeight = scrollArea.clientHeight;
        const maxScrollTop = scrollHeight - clientHeight;
        if (maxThumbTop > 0) {
            scrollArea.scrollTop = (newThumbTop / maxThumbTop) * maxScrollTop;
        }

        if (e.cancelable !== false) {
            e.preventDefault();
        }
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

            updateThumb();
        }
    }

    function onDragStart(e) {
        isDragging = true;
        thumb.classList.add('dragging');
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        startY = clientY;
        startThumbTop = parseFloat(thumb.style.top) || 0;
        document.body.style.userSelect = 'none';

        // Attach listeners ONLY while dragging
        document.addEventListener('mousemove', onDragMove);
        document.addEventListener('mouseup', onDragEnd);
        document.addEventListener('touchmove', onDragMove, { passive: false });
        document.addEventListener('touchend', onDragEnd);
        document.addEventListener('touchcancel', onDragEnd);
        
        if (e.cancelable !== false) {
            e.preventDefault();
        }
    }

    // Mouse and touch events for thumb
    thumb.addEventListener('mousedown', onDragStart);
    thumb.addEventListener('touchstart', onDragStart, { passive: false });

    // Click/tap on track to jump and drag
    track.addEventListener('mousedown', (e) => {
        if (e.target === thumb) return;
        const rect = track.getBoundingClientRect();
        const clickY = e.clientY - rect.top;
        const clientHeight = scrollArea.clientHeight;
        const thumbHeight = thumb.offsetHeight;

        let newThumbTop = clickY - thumbHeight / 2;
        const maxThumbTop = clientHeight - thumbHeight;
        newThumbTop = Math.max(0, Math.min(maxThumbTop, newThumbTop));

        const scrollHeight = scrollArea.scrollHeight;
        const maxScrollTop = scrollHeight - clientHeight;
        if (maxThumbTop > 0) {
            scrollArea.scrollTop = (newThumbTop / maxThumbTop) * maxScrollTop;
        }

        onDragStart(e);
    });

    track.addEventListener('touchstart', (e) => {
        if (e.target === thumb) return;
        const rect = track.getBoundingClientRect();
        const touchY = e.touches[0].clientY - rect.top;
        const clientHeight = scrollArea.clientHeight;
        const thumbHeight = thumb.offsetHeight;

        let newThumbTop = touchY - thumbHeight / 2;
        const maxThumbTop = clientHeight - thumbHeight;
        newThumbTop = Math.max(0, Math.min(maxThumbTop, newThumbTop));

        const scrollHeight = scrollArea.scrollHeight;
        const maxScrollTop = scrollHeight - clientHeight;
        if (maxThumbTop > 0) {
            scrollArea.scrollTop = (newThumbTop / maxThumbTop) * maxScrollTop;
        }

        onDragStart(e);
    }, { passive: false });

    // Initial position triggers with RAF and timeouts to guarantee execution post-layout
    requestAnimationFrame(updateThumb);
    setTimeout(updateThumb, 50);
    setTimeout(updateThumb, 200);
}

function initCustomScrollbar() {
    initCustomScrollbarForElement('main-list-results', 'list-scrollbar-track', 'list-scrollbar-thumb');
    initCustomScrollbarForElement('full-list-modal-results', 'full-list-scrollbar-track', 'full-list-scrollbar-thumb');
}



window.removeAddedWordFromTools = async function(word) {
    try {
        const response = await fetch('/api/mods/added_words/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word.toUpperCase() })
        });
        const data = await response.json();
        if (data.success) {
            fetchListsData(); // Refresh
        } else {
            alert(data.error || "Failed to remove word.");
        }
    } catch (err) {
        console.error("Error removing word:", err);
    }
};

window.loadAddedWords = fetchListsData;
async function fetchListsData(typeOverride) {
    // Cancel previous progressive render chunk loop
    currentProgressiveLoadId++;
    listsShowAll = false;
    window._lastFullListFilterKey = null;
    window._savedFullListScrollTop = 0;

    // Cancel previous fetch if it's still running
    if (listsFetchAbortController) {
        listsFetchAbortController.abort();
    }
    // Cancel previous warning timer (prevents ghost warnings after fast loads)
    if (listsFetchTimeoutId) {
        clearTimeout(listsFetchTimeoutId);
        listsFetchTimeoutId = null;
    }
    listsFetchAbortController = new AbortController();
    const currentController = listsFetchAbortController;

    // Get Filter Values
    const lengthSelect = document.getElementById('list-length-filter');
    const startSelect = document.getElementById('list-start-filter');
    const typeSelect = document.getElementById('list-type-filter');
    
    // If we have a type override, update the select's value to keep UI in sync
    if (typeOverride && typeSelect) {
        typeSelect.value = typeOverride;
    }
    
    const selectedType = typeSelect ? typeSelect.value : 'nwl';
    const selectedLength = lengthSelect ? lengthSelect.value : 'all';
    const selectedStart = startSelect ? startSelect.value : 'all';

    const titleEl = document.getElementById('list-display-title');
    const countEl = document.getElementById('main-list-count');
    const scrollArea = document.getElementById('main-list-results');

    // Update UI title mapping
    const typeMap = {
        'nwl': 'NWL (North American)',
        'csw': 'CSW (International)',
        'csw_only': 'CSW Only',
        'likelihood': 'Likelihood (Scrabble)',
        'uniques': 'NWL Uniques',
        'added': 'Added Words',
        'new_nwl': 'New NWL Words',
        'new_csw': 'New CSW Words'
    };

    if (titleEl) {
        titleEl.innerText = typeMap[selectedType] || 'Word List';
    }

    if (scrollArea) {
        scrollArea.innerHTML = '<div style="padding:20px; opacity:0.6; text-align:center;">Loading list data...</div>';
    }
    if (countEl) countEl.innerText = '';

    const controller = currentController;
    const timeoutId = setTimeout(() => {
        if (listsFetchTimeoutId === timeoutId) {
            listsFetchTimeoutId = null;
        }
        controller.abort();
        if (scrollArea) {
            scrollArea.innerHTML = `
                <div style="padding:30px; text-align:center; color: #ffb703; font-weight: 500;">
                    <div style="font-size: 1.5rem; margin-bottom: 10px;">⚠️ Heavy Computation Warning</div>
                    <div>This list is taking longer than 3 minutes to load, especially if you are using data, and not wi-fi.</div>
                    <div style="margin-top: 15px; font-size: 0.95rem; opacity: 0.9; line-height: 1.6;">
                        Loading massive list configurations without filters can overload the browser or server.<br>
                        <strong>Please select a specific word length</strong> or a <strong>starting letter</strong> to reduce the size of the request.
                    </div>
                </div>
            `;
        }
    }, 180000);
    listsFetchTimeoutId = timeoutId;

    try {
        // Build Query URL
        let url = `/api/tools/lists?list_type=${selectedType}&`;

        if (lengthSelect && lengthSelect.value !== 'all') {
            url += `length=${lengthSelect.value}&`;
        }
        if (startSelect && startSelect.value !== 'all') {
            url += `starts_with=${startSelect.value}`;
        }

        const response = await fetch(url + `&t=${Date.now()}`, { signal: controller.signal });
        // Clear the warning timer — fetch completed in time
        if (listsFetchTimeoutId === timeoutId) {
            clearTimeout(timeoutId);
            listsFetchTimeoutId = null;
        }
        const data = await response.json();

        // Clear active fetch controller tracking if it is still this controller
        if (listsFetchAbortController === currentController) {
            listsFetchAbortController = null;
        }

        if (data.error) {
            console.error(data.error);
            if (scrollArea) scrollArea.innerHTML = `<div style="color:red; padding:20px; text-align:center;">Error: ${data.error}</div>`;
            return;
        }

        // The API returns an object where the key is the list type
        const words = data[selectedType] || [];
        console.log(`[Lists] Received ${words.length} words for type: ${selectedType}. First 5:`, words.slice(0, 5));
        
        if (countEl) {
            countEl.textContent = words && words.length ? `(${words.length.toLocaleString()})` : '(0)';
        }

        currentWordsList = words;
        currentWordsRenderedCount = 0;
        currentWordsType = selectedType;
        window.listsServerTruncated = data.is_truncated || false;

        if (!words || words.length === 0) {
            if (scrollArea) scrollArea.innerHTML = '<div style="padding:20px; opacity:0.6; text-align:center;">No words found matching these filters.</div>';
            return;
        }

        startProgressiveRendering();

        listsDataLoaded = true;

    } catch (err) {
        if (listsFetchTimeoutId === timeoutId) {
            clearTimeout(timeoutId);
            listsFetchTimeoutId = null;
        }
        if (listsFetchAbortController === currentController) {
            listsFetchAbortController = null;
        }
        if (err.name === 'AbortError') {
            console.log('[Lists] Fetch aborted (newer selection or timeout).');
            return;
        }
        console.error('Failed to fetch lists:', err);
        if (scrollArea) {
            scrollArea.innerHTML = '<div style="color:red; padding:20px; text-align:center;">Network error. Check console for details.</div>';
        }
    }
}

// --- Sequence Tool Logic ---

function setupSequenceTool() {
    const searchBtn = document.getElementById('seq-search-btn');
    const input = document.getElementById('seq-input');

    if (searchBtn) {
        searchBtn.addEventListener('click', runSequenceSearch);
    }

    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runSequenceSearch();
        });
    }
}

async function runSequenceSearch() {
    const inputEl = document.getElementById('seq-input');
    const modeEl = document.getElementById('seq-mode');
    const lengthEl = document.getElementById('seq-length');
    const dictEl = document.getElementById('seq-dict');
    const resultsContainer = document.getElementById('seq-results-container');

    const seq = inputEl.value.trim();
    const mode = modeEl.value;
    const length = lengthEl.value;
    const dictionary = dictEl ? dictEl.value : 'NWL';

    if (!seq) {
        resultsContainer.innerHTML = '<div class="seq-results-placeholder">Please enter a sequence.</div>';
        return;
    }

    resultsContainer.innerHTML = '<div style="padding:20px; text-align:center; color:#rgba(255,255,255,0.7);">Searching...</div>';

    try {
        const response = await fetch('/api/tools/sequence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sequence: seq,
                mode: mode,
                length: length,
                dictionary: dictionary
            })
        });

        const data = await response.json();

        if (data.error) {
            resultsContainer.innerHTML = `<div style="padding:20px; color:#f43f5e;">Error: ${data.error}</div>`;
            return;
        }

        const words = data.results;
        const count = data.count;

        if (words.length === 0) {
            resultsContainer.innerHTML = '<div class="seq-results-placeholder">No words found.</div>';
            return;
        }

        // Render Results Table
        let html = `
            <div style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.2);">
                Found ${count} words
            </div>
            <div class="list-scroll-area-wrapper" style="position: relative; flex: 1; min-height: 0; display: flex; flex-direction: column;">
                <div class="seq-scroll-area list-scroll-area" id="seq-list-results" style="height: 100%; overflow-y: auto; padding: 10px;">
                    <table class="group-table" style="width: 100%;">
                        <tbody>
        `;

        // Clickable words for Sequence search
        html += words.map(w => `
            <tr><td style="padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span class="clickable-word-link" onclick="window.lookupWord('${w}', event)" style="font-family: monospace;">${w}</span>
            </td></tr>
        `).join('');

        html += `
                        </tbody>
                    </table>
                </div>
                <div class="custom-scrollbar-track" id="seq-scrollbar-track">
                    <div class="custom-scrollbar-thumb" id="seq-scrollbar-thumb"></div>
                </div>
            </div>
        `;

        resultsContainer.innerHTML = html;
        initCustomScrollbarForElement('seq-list-results', 'seq-scrollbar-track', 'seq-scrollbar-thumb');

    } catch (err) {
        console.error("Sequence search failed:", err);
        resultsContainer.innerHTML = '<div style="padding:20px; color:#f43f5e;">Search failed.</div>';
    }
}

// --- Manual Tool Logic ---

let manualSolvedWords = [];

function setupManualTool() {
    try {
        const solveBtn = document.getElementById('direct-solve-btn');
        if (solveBtn) {
            solveBtn.onclick = (e) => {
                console.log("Button clicked from onclick");
                runManualSolve();
            };
        }

        const clearBtn = document.getElementById('manual-clear-btn');
        if (clearBtn) {
            const handleClear = (e) => {
                if (e) e.preventDefault();
                console.log("[ManualSolver] Clear button clicked.");
                clearManualGrid();
            };
            clearBtn.onclick = handleClear;
            clearBtn.addEventListener('click', handleClear);
        }

        const dimSelect = document.getElementById('manual-dim');
        if (dimSelect) {
            dimSelect.onchange = (e) => renderManualGrid(e.target.value);
            renderManualGrid(dimSelect.value);
        }
    } catch (e) {
        console.error("Manual tool setup failed:", e);
    }
}

function clearManualGrid() {
    console.log("[ManualSolver] clearManualGrid initiated.");
    const gridEl = document.getElementById('manual-grid');
    if (gridEl) {
        // Query both class and input tags for bulletproof lookup
        const cells = gridEl.querySelectorAll('input, .manual-cell');
        console.log(`[ManualSolver] Clearing ${cells.length} grid inputs.`);
        cells.forEach(cell => {
            cell.value = '';
            cell.setAttribute('value', '');
        });
        if (cells[0]) {
            cells[0].focus();
        }
    } else {
        console.warn("[ManualSolver] #manual-grid element not found!");
    }
    
    manualSolvedWords = [];
    const res = document.getElementById('manual-results-container');
    if (res) {
        res.innerHTML = '<div class="seq-results-placeholder" style="padding: 20px; color: var(--muted-text);">Ready to solve...</div>';
    }
}

function renderManualGrid(dims) {
    const gridEl = document.getElementById('manual-grid');
    if (!gridEl) return;

    try {
        const [rows, cols] = dims.split('x').map(Number);
        if (window.innerWidth <= 900) {
            gridEl.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
            gridEl.style.gridTemplateRows = `repeat(${rows}, auto)`;
            gridEl.style.width = '100%';
            gridEl.style.maxWidth = `${cols * 45}px`;
        } else {
            gridEl.style.gridTemplateColumns = `repeat(${cols}, 55px)`;
            gridEl.style.gridTemplateRows = `repeat(${rows}, 55px)`;
            gridEl.style.width = 'auto';
            gridEl.style.maxWidth = 'none';
        }
        gridEl.innerHTML = '';

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const input = document.createElement('input');
                input.className = 'manual-cell';
                input.maxLength = 2;
                input.placeholder = '?';
                input.oninput = (e) => {
                    const val = e.target.value.toUpperCase();
                    e.target.value = val;
                    if (val.length === 2 || (val.length === 1 && val !== 'Q')) {
                        if (input.nextElementSibling) input.nextElementSibling.focus();
                    }
                };
                input.onkeydown = (e) => {
                    if (e.key === 'Backspace' && !input.value) {
                        if (input.previousElementSibling) input.previousElementSibling.focus();
                    }
                };
                gridEl.appendChild(input);
            }
        }

        const res = document.getElementById('manual-results-container');
        if (res) {
            res.style.display = 'flex';
            res.innerHTML = '<div class="seq-results-placeholder" style="padding: 20px; color: var(--muted-text);">Ready to solve...</div>';
        }
    } catch (err) {
        console.error("Grid render failed:", err);
    }
}

async function runManualSolve() {
    const solveBtn = document.getElementById('direct-solve-btn');
    const gridEl = document.getElementById('manual-grid');
    const dictEl = document.getElementById('manual-dict');
    const dimSelect = document.getElementById('manual-dim');
    const resultsContainer = document.getElementById('manual-results-container');

    if (!gridEl || !dimSelect || !solveBtn || !resultsContainer) {
        alert("Tool elements not found. Please refresh.");
        return;
    }

    try {
        const [rows, cols] = dimSelect.value.split('x').map(Number);
        const cells = gridEl.querySelectorAll('.manual-cell');
        const board = [];
        let idx = 0;
        let missing = false;

        for (let r = 0; r < rows; r++) {
            const row = [];
            for (let c = 0; c < cols; c++) {
                const v = cells[idx++].value.trim().toUpperCase();
                if (!v) missing = true;
                row.push(v);
            }
            board.push(row);
        }

        if (missing) {
            alert("Please fill all cells first.");
            return;
        }

        solveBtn.innerText = "Solving...";
        solveBtn.disabled = true;

        const min_word_length = window._displayedParams ? (window._displayedParams.min || 3) : 3;
        const resp = await fetch('/api/tools/manual_solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board, dictionary: dictEl.value, min_word_length })
        });
        const data = await resp.json();

        if (data.error) {
            alert("Solve error: " + data.error);
            return;
        }

        if (data.board_matches_active_room) {
            resultsContainer.innerHTML = '<div style="padding: 30px; text-align: center; color: #f87171;">⚠️ Active Room Match - Solve Blocked</div>';
            return;
        }

        manualSolvedWords = data.results;
        revealManualWords(true);

    } catch (err) {
        alert("Solve failed: " + err.message);
    } finally {
        solveBtn.innerText = "Solve";
        solveBtn.disabled = false;
    }
}

function revealManualWords(forceShow = false) {
    const resultsContainer = document.getElementById('manual-results-container');
    const revealBtn = document.getElementById('manual-reveal-btn');

    if (!forceShow && resultsContainer.style.display === 'flex') {
        resultsContainer.style.display = 'none';
        if (revealBtn) revealBtn.innerText = "Reveal Words";
        return;
    }

    if (manualSolvedWords.length === 0) {
        resultsContainer.innerHTML = '<div class="seq-results-placeholder" style="padding: 20px; color: var(--muted-text);">No words found on this board.</div>';
    } else {
        let html = `
            <div style="padding: 12px 20px; border-bottom: 1px solid rgba(var(--text-primary-rgb), 0.1); background: rgba(var(--text-primary-rgb), 0.05); font-weight: 700; color: #4facfe; text-transform: uppercase; letter-spacing: 1px; font-size: 0.85rem;">
                Found ${manualSolvedWords.length} words
            </div>
            <div style="flex: 1; overflow-y: auto; padding: 20px;">
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px;">
        `;

        html += manualSolvedWords.map(w => {
            let fontSize = "1.1rem";
            if (w.length >= 12) {
                fontSize = "0.75rem";
            } else if (w.length >= 10) {
                fontSize = "0.85rem";
            } else if (w.length >= 8) {
                fontSize = "0.95rem";
            }
            return `
                <div title="${w}" style="padding: 10px 6px; background: rgba(var(--text-primary-rgb), 0.05); border: 1px solid rgba(var(--text-primary-rgb), 0.1); border-radius: 10px; color: var(--text-primary); font-family: 'JetBrains Mono', monospace; text-align: center; font-size: ${fontSize}; transition: all 0.2s; cursor: default; box-shadow: 0 2px 4px rgba(0,0,0,0.05); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    ${w}
                </div>
            `;
        }).join('');

        html += `
                </div>
            </div>
        `;
        resultsContainer.innerHTML = html;
    }

    resultsContainer.style.display = 'flex';
    if (revealBtn) revealBtn.innerText = "Hide Words";
}

// --- Random Word Tool Logic ---

function setupRandomWordTool() {
    const genBtn = document.getElementById('random-gen-btn');
    if (genBtn) {
        genBtn.addEventListener('click', generateRandomWord);
    }
}

async function generateRandomWord() {
    const lengthEl = document.getElementById('random-length');
    const dictEl = document.getElementById('random-dict');
    const displayEl = document.getElementById('random-word-display');
    const genBtn = document.getElementById('random-gen-btn');

    const length = lengthEl.value;
    const dictionary = dictEl.value;

    genBtn.innerText = "Generating...";
    genBtn.disabled = true;
    displayEl.innerHTML = ''; // Clear while loading
    const defEl = document.getElementById('random-word-definition');
    if (defEl) defEl.innerHTML = '';

    try {
        const url = `/api/tools/random_word?length=${length}&dictionary=${dictionary}`;
        const response = await fetch(url);
        const data = await response.json();

        if (data.error) {
            displayEl.innerHTML = `<span style="font-size: 1.5rem; color: #f43f5e;">${data.error}</span>`;
            return;
        }

        const word = data.word;
        const definition = data.definition || "No definition available.";

        // Add a class to re-trigger animation
        displayEl.classList.remove('random-word-large');
        void displayEl.offsetWidth; // Trigger reflow
        displayEl.classList.add('random-word-large');

        displayEl.innerText = word;
        if (defEl) {
            defEl.style.opacity = '0';
            let html = '';
            if (data.pronunciation) {
                html += `<div class="pronunciation" style="margin-bottom: 8px; font-size: 1.5rem; letter-spacing: 2px;">${data.pronunciation}</div>`;
            }
            html += `<div class="definition-text" style="font-size: 1.2rem; line-height: 1.5;">${definition}</div>`;
            if (data.image_url) {
                html += `<div class="definition-image-container" style="margin-top: 15px; text-align: center;"><img src="${data.image_url}" class="definition-image" style="max-width: 100%; max-height: 180px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.4);" /></div>`;
            }
            defEl.innerHTML = html;
            setTimeout(() => {
                defEl.style.transition = 'opacity 0.5s ease';
                defEl.style.opacity = '1';
                // Scroll to the bottom on mobile viewports so all info is visible
                const isMobile = window.innerWidth <= 900 || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
                if (isMobile) {
                    const contentEl = document.querySelector('#page-tools .tools-content');
                    if (contentEl) {
                        setTimeout(() => {
                            contentEl.scrollTo({
                                top: contentEl.scrollHeight,
                                behavior: 'smooth'
                            });
                        }, 100);
                    }
                }
            }, 100);
        }

    } catch (err) {
        console.error("Random word fetch failed:", err);
        displayEl.innerHTML = `<span style="font-size: 1.5rem; color: #f43f5e;">Error loading random word.</span>`;
    } finally {
        genBtn.innerText = "Generate Random Word";
        genBtn.disabled = false;
    }
}

// --- Word of the Day Tool Logic ---

function setupWotdTool() {
    // This is mainly for manual navigation/initialization
}

let lastWotdDate = null;

async function updateWotd() {
    const displayEl = document.getElementById('wotd-display');
    if (!displayEl) return;

    // Only skip if we already have the word for TODAY
    const todayStr = new Date().toISOString().split('T')[0];
    if (displayEl.innerText.trim() !== '' && lastWotdDate === todayStr) return;

    displayEl.innerHTML = '<span style="font-size: 1.5rem; opacity: 0.5;">Loading...</span>';

    try {
        const response = await fetch(`/api/tools/wotd?_t=${Date.now()}`);
        const data = await response.json();

        if (data.error) {
            displayEl.innerText = 'Error loading word';
            return;
        }

        displayEl.innerText = data.word;
        lastWotdDate = data.date; // Use the date confirmed by the server
        const defEl = document.getElementById('wotd-definition');
        if (defEl) {
            let html = '';
            if (data.pronunciation) {
                html += `<div class="pronunciation" style="margin-bottom: 5px;">${data.pronunciation}</div>`;
            }
            html += `<div class="definition-text">${data.definition || "No definition available."}</div>`;
            if (data.image_url) {
                html += `<div class="definition-image-container" style="margin-top: 15px; text-align: center;"><img src="${data.image_url}" class="definition-image" style="max-width: 100%; max-height: 180px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.4);" /></div>`;
            }
            defEl.innerHTML = html;
        }
    } catch (err) {
        console.error("WOTD fetch failed:", err);
        displayEl.innerText = 'Offline';
    }
}

// --- Subanagrams Tool Logic ---

function setupSubanagramsTool() {
    const searchBtn = document.getElementById('sub-search-btn');
    const input = document.getElementById('sub-input');

    if (searchBtn) {
        searchBtn.addEventListener('click', runSubanagramSearch);
    }

    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runSubanagramSearch();
        });
    }
}

async function runSubanagramSearch() {
    const inputEl = document.getElementById('sub-input');
    const dictEl = document.getElementById('sub-dict');
    const resultsContainer = document.getElementById('sub-results-container');

    const input = inputEl.value.trim();
    const dictionary = dictEl.value;

    if (!input) {
        resultsContainer.innerHTML = '<div class="seq-results-placeholder">Please enter letters to search.</div>';
        return;
    }

    resultsContainer.innerHTML = '<div style="padding:20px; text-align:center; color:rgba(255,255,255,0.7);">Finding subanagrams...</div>';

    try {
        const response = await fetch('/api/tools/subanagrams', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input: input,
                dictionary: dictionary
            })
        });

        const data = await response.json();

        if (data.error) {
            resultsContainer.innerHTML = `<div style="padding:20px; color:#f43f5e;">Error: ${data.error}</div>`;
            return;
        }

        const words = data.results;
        const count = data.count;

        if (words.length === 0) {
            resultsContainer.innerHTML = '<div class="seq-results-placeholder">No subanagrams found.</div>';
            return;
        }

        // Render Results Table
        let html = `
            <div style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.2); text-align: left;">
                Found ${count} subanagrams
            </div>
            <div class="list-scroll-area-wrapper" style="position: relative; flex: 1; min-height: 0; display: flex; flex-direction: column;">
                <div class="sub-scroll-area list-scroll-area" id="sub-list-results" style="height: 100%; overflow-y: auto; padding: 10px;">
                    <table class="group-table" style="width: 100%;">
                        <tbody>
        `;

        // Clickable words for Subanagram search
        html += words.map(w => `
            <tr><td style="padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span class="clickable-word-link" onclick="window.lookupWord('${w}', event)" style="font-family: monospace;">${w}</span>
            </td></tr>
        `).join('');

        html += `
                        </tbody>
                    </table>
                </div>
                <div class="custom-scrollbar-track" id="sub-scrollbar-track">
                    <div class="custom-scrollbar-thumb" id="sub-scrollbar-thumb"></div>
                </div>
            </div>
        `;

        resultsContainer.innerHTML = html;
        initCustomScrollbarForElement('sub-list-results', 'sub-scrollbar-track', 'sub-scrollbar-thumb');

    } catch (err) {
        console.error("Subanagram search failed:", err);
        resultsContainer.innerHTML = '<div style="padding:20px; color:#f43f5e;">Search failed.</div>';
    }
}

// --- Is Valid Tool Logic ---

function setupIsValidTool() {
    const checkBtn = document.getElementById('valid-check-btn');
    const input = document.getElementById('valid-input');

    if (checkBtn) {
        checkBtn.addEventListener('click', runValidationCheck);
    }

    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runValidationCheck();
        });
    }
}

async function runValidationCheck() {
    const inputEl = document.getElementById('valid-input');
    const dictEl = document.getElementById('valid-dict');
    const displayEl = document.getElementById('valid-result-display');
    const checkBtn = document.getElementById('valid-check-btn');

    const word = inputEl.value.trim();
    const dictionary = dictEl.value;

    if (!word) return;

    displayEl.innerText = '';
    const defEl = document.getElementById('valid-definition-display');
    if (defEl) defEl.style.opacity = '0';

    checkBtn.innerText = "Checking...";
    checkBtn.disabled = true;

    try {
        const response = await fetch('/api/tools/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word, dictionary: dictionary })
        });

        const data = await response.json();
        const defEl = document.getElementById('valid-definition-display');

        if (data.error) {
            displayEl.innerHTML = `<span style="font-size: 1.5rem; color: #f43f5e;">${data.error}</span>`;
            if (defEl) defEl.style.opacity = '0';
            return;
        }

        const color = data.is_valid ? '#4ade80' : '#f43f5e';
        const statusText = data.is_valid ? 'IS VALID' : 'IS NOT VALID';

        displayEl.style.color = color;
        displayEl.innerText = `${data.word} ${statusText}`;

        // Re-trigger animation
        displayEl.classList.remove('random-word-large');
        void displayEl.offsetWidth;
        displayEl.classList.add('random-word-large');

        // Handle definition and pronunciation
        if (defEl) {
            if (data.is_valid && (data.definition || data.pronunciation || data.image_url)) {
                let html = '';
                if (data.pronunciation) {
                    html += `<div class="pronunciation" style="margin-bottom: 10px; font-size: 1.8rem; letter-spacing: 2px;">${data.pronunciation}</div>`;
                }
                if (data.definition) {
                    html += `<div class="definition-text" style="font-size: 1.3rem; line-height: 1.6; color: #fff; font-style: normal;">${data.definition}</div>`;
                }
                if (data.image_url) {
                    html += `<div class="definition-image-container" style="margin-top: 15px; text-align: center;"><img src="${data.image_url}" class="definition-image" style="max-width: 100%; max-height: 180px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.4);" /></div>`;
                }
                defEl.innerHTML = html;
                setTimeout(() => {
                     defEl.style.transition = 'opacity 0.5s ease';
                     defEl.style.opacity = '1';
                     // Scroll to the bottom on mobile viewports so all info is visible
                     const isMobile = window.innerWidth <= 900 || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
                     if (isMobile) {
                         const contentEl = document.querySelector('#page-tools .tools-content');
                         if (contentEl) {
                             setTimeout(() => {
                                 contentEl.scrollTo({
                                     top: contentEl.scrollHeight,
                                     behavior: 'smooth'
                                 });
                             }, 100);
                         }
                     }
                 }, 100);
            } else {
                defEl.style.opacity = '0';
                setTimeout(() => defEl.innerHTML = '', 500);
            }
        }

    } catch (err) {
        console.error("Validation check failed:", err);
        displayEl.innerHTML = `<span style="font-size: 1.5rem; color: #f43f5e;">Error checking word.</span>`;
    } finally {
        checkBtn.innerText = "Validate";
        checkBtn.disabled = false;
        
        // Clear input box after submission
        if (inputEl) {
            inputEl.value = '';
            const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            if (isMobile) {
                inputEl.blur(); // Remove focus to hide mobile keyboard
            } else {
                inputEl.focus(); // Keep focus for next check
            }
        }
    }
}

// --- Private Messaging Logic ---

let pmPollingInterval = null;
let currentChatTarget = null;
// Use localStorage to sync notification state across multiple tabs
const getPMState = () => {
    try {
        const defaults = { lastNotifiedContext: null, lastUnreadCount: 0, activeChat: null, lastTimestamp: null };
        const saved = JSON.parse(localStorage.getItem('morpheme_pm_state') || '{}');
        return { ...defaults, ...saved };
    } catch (e) {
        return { lastNotifiedContext: null, lastUnreadCount: 0, activeChat: null, lastTimestamp: null };
    }
};
const setPMState = (state) => {
    localStorage.setItem('morpheme_pm_state', JSON.stringify(state));
};

function setupPrivateMessaging() {
    const closeBtn = document.getElementById('pm-close-btn');
    const sendBtn = document.getElementById('pm-send-btn');
    const input = document.getElementById('pm-input');

    if (closeBtn) {
        closeBtn.addEventListener('click', closePrivateChat);
    }

    if (sendBtn) sendBtn.addEventListener('click', sendPM);
    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendPM();
        });
    }

    // ON INIT: Clear stale activeChat from localStorage to prevent stuck notifications
    const initialState = getPMState();
    initialState.activeChat = null;
    setPMState(initialState);

    // Escape key listener for PM box
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const modal = document.getElementById('private-chat-modal');
            if (modal && !modal.classList.contains('hidden')) {
                closePrivateChat();
            }
        }
    });

    // Polling for new messages globally
    setInterval(checkForUnreadPMs, 5000);
}

function closePrivateChat() {
    const modal = document.getElementById('private-chat-modal');
    if (!modal) return;

    // Aggressively delete history from server
    if (currentChatTarget) {
        fetch(`/api/pm/clear/${encodeURIComponent(currentChatTarget)}`, { method: 'POST' })
            .catch(err => console.error("Failed to clear PMs on server:", err));
    }

    modal.classList.add('hidden');

    // Clear ALL UI elements immediately
    const history = document.getElementById('pm-history');
    if (history) {
        history.innerHTML = '';
        history.dataset.chatTarget = '';
    }

    const input = document.getElementById('pm-input');
    if (input) input.value = '';

    const nameEl = document.getElementById('pm-target-name');
    if (nameEl) nameEl.innerText = 'Chat';

    stopPMPolling();
    currentChatTarget = null;

    // Update shared state
    const pmState = getPMState();
    pmState.activeChat = null;
    setPMState(pmState);
}

async function openPrivateChat(username, clearHistory = false) {
    // Check if current user is timed out before opening private chat
    try {
        const toResp = await fetch('/api/user/my_timeout_status?_t=' + Date.now(), { cache: 'no-store' });
        const toData = await toResp.json();
        if (toData && toData.timed_out) {
            if (typeof window.showTimeoutBanModal === 'function') {
                window.showTimeoutBanModal(toData);
            } else {
                const rText = toData.timeout_reason || 'Moderator timeout';
                alert(`Action Restricted: Your account is currently timed out (${toData.remaining} remaining).\nReason: ${rText}\n\nPrivate messaging is temporarily disabled.`);
            }
            return;
        }
    } catch (e) {
        console.warn("[PM] Could not check timeout status:", e);
    }

    currentChatTarget = username;

    if (clearHistory) {
        try {
            await fetch(`/api/pm/clear/${encodeURIComponent(username)}`, { method: 'POST' });
        } catch (err) {
            console.error("Failed to clear PMs on open:", err);
        }
    }

    const history = document.getElementById('pm-history');
    if (history) {
        history.dataset.chatTarget = username;
        history.innerHTML = '<div style="text-align:center; opacity:0.5; padding:20px;">Loading conversation...</div>';
    }

    const targetNameEl = document.getElementById('pm-target-name');
    if (targetNameEl) targetNameEl.innerText = username;

    const chatModal = document.getElementById('private-chat-modal');
    if (chatModal) {
        chatModal.classList.remove('hidden');
        chatModal.style.display = 'block';
    }

    // Update synchronized state to reflect we've interacted with this
    const pmState = getPMState();
    pmState.lastNotifiedContext = `OPEN:${username}`;
    pmState.activeChat = username;
    setPMState(pmState);

    await refreshConversation();
    startPMPolling();

    // Auto-focus input
    const pmInput = document.getElementById('pm-input');
    if (pmInput) pmInput.focus();
}

async function refreshConversation() {
    const targetAtStart = currentChatTarget;
    if (!targetAtStart) return;

    try {
        const response = await fetch(`/api/pm/conversation/${encodeURIComponent(targetAtStart)}?t=${Date.now()}`);
        const data = await response.json();

        // CHECK RACING CONDITION: 
        // 1. If global target changed
        // 2. If the UI element itself was repurposed or cleared
        const history = document.getElementById('pm-history');
        if (currentChatTarget !== targetAtStart || !history || history.dataset.chatTarget !== targetAtStart) {
            return;
        }

        if (data && Array.isArray(data.messages)) {
            renderPMHistory(data.messages);

            // Update high-water mark for notifications
            if (data.messages.length > 0) {
                const latest = data.messages[data.messages.length - 1];
                if (latest && latest.timestamp) {
                    const pmState = getPMState();
                    if (!pmState.lastTimestamp || latest.timestamp > pmState.lastTimestamp) {
                        pmState.lastTimestamp = latest.timestamp;
                        setPMState(pmState);
                    }
                }
            }
        } else if (data && data.error) {
            history.innerHTML = `<div style="text-align:center; color:#ef4444; opacity:0.8; padding:20px;">${data.error}</div>`;
        } else {
            renderPMHistory([]);
        }
    } catch (err) {
        console.error("Failed to fetch conversation:", err);
        const history = document.getElementById('pm-history');
        if (history && history.dataset.chatTarget === targetAtStart) {
            history.innerHTML = '<div style="text-align:center; opacity:0.4; padding:20px;">No messages yet. Say hello!</div>';
        }
    }
}

function renderPMHistory(messages) {
    const historyEl = document.getElementById('pm-history');
    if (!historyEl || !currentChatTarget || historyEl.dataset.chatTarget !== currentChatTarget) return;

    if (messages.length === 0) {
        historyEl.innerHTML = '<div style="text-align:center; opacity:0.3; padding:20px;">No messages yet. Say hello!</div>';
        return;
    }

    const html = messages.map(m => {
        const typeClass = m.is_me ? 'me' : 'them';
        const timeStr = new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        return `
            <div class="pm-entry ${typeClass}">
                <div class="pm-bubble">${m.message}</div>
                <div class="pm-time">${timeStr}</div>
            </div>
        `;
    }).join('');

    historyEl.innerHTML = html;
    historyEl.scrollTop = historyEl.scrollHeight;
}

async function sendPM() {
    const input = document.getElementById('pm-input');
    const msg = input.value.trim();
    if (!msg || !currentChatTarget) return;

    try {
        const response = await fetch('/api/pm/send', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                recipient: currentChatTarget,
                message: msg
            })
        });

        const data = await response.json();
        if (data.success) {
            input.value = '';
            await refreshConversation();
        } else {
            alert(data.error || "Failed to send message");
        }
    } catch (err) {
        console.error("PM send error:", err);
    }
}

function startPMPolling() {
    stopPMPolling();
    pmPollingInterval = setInterval(refreshConversation, 3000);
}

function stopPMPolling() {
    if (pmPollingInterval) {
        clearInterval(pmPollingInterval);
        pmPollingInterval = null;
    }
}

// --- Friends Management ---

async function updateFriendButtonStatus(username, btn) {
    try {
        const response = await fetch(`/api/friends/status/${encodeURIComponent(username)}`);
        const data = await response.json();

        if (data.is_friend) {
            btn.innerText = 'Friends';
            btn.classList.add('is-friend');
        } else {
            btn.innerText = 'Add Friend';
            btn.classList.remove('is-friend');
        }
    } catch (err) {
        console.error("Error updating friend button:", err);
    }
}

async function handleFriendAction(username, btn) {
    const isFriend = btn.classList.contains('is-friend');
    const endpoint = isFriend ? '/api/friends/remove' : '/api/friends/add';

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username })
        });
        const data = await response.json();

        if (data.success) {
            updateFriendButtonStatus(username, btn);
        } else if (data.error) {
            alert(data.error);
        }
    } catch (err) {
        console.error("Friend action error:", err);
    }
}

window.fetchAndRenderFriends = fetchAndRenderFriends;

async function fetchAndRenderFriends() {
    const friendsList = document.getElementById('profile-friends-list');
    if (!friendsList) return;

    try {
        const response = await fetch('/api/friends/list');
        const data = await response.json();

        if (data.error) return;

        if (data.friends.length === 0) {
            friendsList.innerHTML = '<p class="placeholder">You haven\'t added any friends yet.</p>';
            return;
        }

        friendsList.innerHTML = data.friends.map(friend => {
            const ratingColor = getRatingColor(friend.rating);
            const isOnline = friend.is_online;
            const avatarHtml = friend.avatar_url
                ? `<div class="friend-avatar-mini" style="background-image: url('${friend.avatar_url}')"></div>`
                : `<div class="friend-avatar-mini" style="background-color: ${ratingColor}">${friend.username[0].toUpperCase()}</div>`;

            return `
                <div class="friend-card ${isOnline ? 'online' : ''}" onclick="window.performProfileSearch('${friend.username}')">
                    ${avatarHtml}
                    <div class="friend-name-mini">
                        ${friend.username}
                        ${isOnline ? '<span class="status-indicator-mini online" style="display:inline-block; margin-left:5px; width:8px; height:8px;"></span>' : ''}
                    </div>
                    <div class="friend-flag-mini">${window.getFlagHtml ? window.getFlagHtml(friend.country_flag) : (friend.country_flag || '🏳️')}</div>
                </div>
            `;
        }).join('');
    } catch (err) {
        console.error("Error fetching friends:", err);
    }
}

// Redundant local getRatingColor removed to use central definition in app.js

async function checkForUnreadPMs() {
    try {
        const response = await fetch('/api/pm/unread_count', { cache: 'no-store' });
        const data = await response.json();

        const count = data.count || 0;
        const senders = data.senders || [];
        const latestTimestamp = data.latest_timestamp;
        const pmState = getPMState();

        if (count > 0 && senders.length > 0 && latestTimestamp) {
            const latestSender = senders[senders.length - 1];

            // A message is "new" if its timestamp is strictly greater than what we last notified about.
            const lastSeen = pmState.lastTimestamp || "";
            const isNewer = String(latestTimestamp) > String(lastSeen);

            if (isNewer) {
                const chatModal = document.getElementById('private-chat-modal');
                const isChatHidden = !chatModal || chatModal.classList.contains('hidden');

                // We are "already chatting" ONLY if the chat is actually open in THIS tab
                // OR if another tab is actively heart-beating? (For now, let's stick to local visibility + activeChat)
                const isAlreadyChatting = (currentChatTarget === latestSender && !isChatHidden);

                if (isAlreadyChatting) {
                    pmState.lastTimestamp = latestTimestamp;
                } else {
                    const delay = Math.random() * 500;
                    setTimeout(() => {
                        const finalCheck = getPMState();
                        // Double check against shared lastTimestamp to prevent multi-tab noise
                        if (String(latestTimestamp) > String(finalCheck.lastTimestamp || "")) {
                            showPMNotification(latestSender, count);
                            finalCheck.lastTimestamp = latestTimestamp;
                            finalCheck.lastUnreadCount = count;
                            setPMState(finalCheck);
                        }
                    }, delay);
                    return;
                }
            }
        }

        // Always sync the unread count
        pmState.lastUnreadCount = count;
        setPMState(pmState);
    } catch (err) {
        // Silent
    }
}

function showPMNotification(sender, count) {
    const existing = document.getElementById('pm-toast');
    if (existing) {
        // Update existing toast content instead of ignoring
        const title = existing.querySelector('.pm-toast-title');
        const text = existing.querySelector('.pm-toast-text');
        if (title) title.innerText = `New Messages (${count})`;
        if (text) text.innerHTML = `<strong>${sender}</strong> and others sent messages`;
        return;
    }

    const toast = document.createElement('div');
    toast.id = 'pm-toast';
    toast.className = 'pm-toast-notification';
    toast.innerHTML = `
        <div class="pm-toast-content">
            <div class="pm-toast-icon">✉️</div>
            <div class="pm-toast-details">
                <div class="pm-toast-title">New Private Message</div>
                <div class="pm-toast-text"><strong>${sender}</strong> sent you a message</div>
            </div>
            <div class="pm-toast-actions">
                <button class="pm-toast-btn respond" onclick="handleToastRespond('${sender}')">Respond</button>
                <button class="pm-toast-btn close" onclick="this.closest('.pm-toast-notification').remove()">Dismiss</button>
            </div>
        </div>
    `;
    document.body.appendChild(toast);
}

window.handleToastRespond = (sender) => {
    document.getElementById('pm-toast')?.remove();
    openPrivateChat(sender);
};

// Make openPrivateChat global for potential use elsewhere
window.openPrivateChat = openPrivateChat;

let unscrambleState = {
    solution: new Set(),
    found: [],
    incorrect: [],
    jumbled: "",
    isWaiting: false,
    isLoading: false,
    history: [],
    nextData: null
};

// Unscramble history starts completely fresh with 0 entries on each session
unscrambleState.history = [];
try {
    localStorage.removeItem('morpheme_unscramble_history');
} catch (e) {}

function saveUnscrambleHistory() {
    // In-memory session tracking only
}

window.clearUnscrambleHistory = function() {
    unscrambleState.history = [];
    try {
        localStorage.removeItem('morpheme_unscramble_history');
    } catch (e) {}
    renderUnscrambleFound();
};

function recordCurrentRoundToHistory() {
    if (!unscrambleState.jumbled || !unscrambleState.solution || unscrambleState.solution.size === 0) return;
    
    // Check if the latest history entry is already for this exact jumbled puzzle
    if (unscrambleState.history.length > 0 && unscrambleState.history[0].jumbled === unscrambleState.jumbled) {
        unscrambleState.history[0].found = [...unscrambleState.found];
        saveUnscrambleHistory();
        return;
    }
    
    unscrambleState.history.unshift({
        jumbled: unscrambleState.jumbled,
        found: [...unscrambleState.found],
        solutions: Array.from(unscrambleState.solution),
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    });
    if (unscrambleState.history.length > 50) unscrambleState.history.pop();
    saveUnscrambleHistory();
}

function setupUnscrambleTool() {
    const genBtn = document.getElementById('unscramble-gen-btn');
    const checkBtn = document.getElementById('unscramble-check-btn');
    const revealBtn = document.getElementById('unscramble-reveal-btn');
    const input = document.getElementById('unscramble-input');

    if (genBtn) genBtn.onclick = () => startNewUnscramble();
    if (checkBtn) checkBtn.onclick = () => checkUnscrambleGuess();
    if (revealBtn) revealBtn.onclick = () => revealUnscrambleSolutions();

    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                checkUnscrambleGuess();
            }
        });
        input.addEventListener('focus', () => {
            const isMobile = window.innerWidth <= 900 || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
            if (isMobile) {
                const contentEl = document.querySelector('#page-tools .tools-content');
                const gameAreaEl = document.querySelector('.unscramble-game-area');
                if (contentEl && gameAreaEl) {
                    setTimeout(() => {
                        contentEl.scrollTo({
                            top: gameAreaEl.offsetTop - 15,
                            behavior: 'smooth'
                        });
                    }, 200);
                }
            }
        });
    }

    // PURGE PREFETCH ON SETTING CHANGE
    const lengthSel = document.getElementById('unscramble-length');
    const dictSel = document.getElementById('unscramble-dict');
    const mustSel = document.getElementById('unscramble-must-have');

    [lengthSel, dictSel, mustSel].forEach(sel => {
        if (sel) {
            sel.addEventListener('change', () => {
                unscrambleState.nextData = null;
                prefetchUnscramble(); 
            });
        }
    });
}

async function startNewUnscramble(keepFound = false) {
    if (unscrambleState.isLoading) {
        console.log('[Unscramble] Generation already in progress, skipping duplicate call.');
        return;
    }

    if (unscrambleNextTimeout) {
        clearTimeout(unscrambleNextTimeout);
        unscrambleNextTimeout = null;
    }

    if (!keepFound && unscrambleState.jumbled) {
        recordCurrentRoundToHistory();
    }

    unscrambleState.isLoading = true;
    unscrambleState.isWaiting = false;

    const lenInput = document.getElementById('unscramble-length');
    const dictInput = document.getElementById('unscramble-dict');
    const mustInput = document.getElementById('unscramble-must-have');
    const len = lenInput ? lenInput.value : 7;
    const dict = dictInput ? dictInput.value : 'NWL';
    const must = mustInput ? mustInput.value.trim().toUpperCase() : '';

    const display = document.getElementById('unscramble-jumbled');
    const info = document.getElementById('unscramble-count-info');
    const foundList = document.getElementById('unscramble-found-list');
    const input = document.getElementById('unscramble-input');
    const revealBtn = document.getElementById('unscramble-reveal-btn');
    const genBtn = document.getElementById('unscramble-gen-btn');
    const checkBtn = document.getElementById('unscramble-check-btn');

    if (display) display.innerText = "Generating...";

    const resContainer = document.getElementById('unscramble-found-container');
    if (resContainer) {
        resContainer.classList.remove('hidden');
        resContainer.style.display = 'flex';
    }

    if (revealBtn) {
        revealBtn.innerText = "Reveal";
        revealBtn.disabled = false;
        revealBtn.style.background = 'linear-gradient(135deg, #4facfe, #2980b9)';
    }
    if (genBtn) genBtn.disabled = false;
    if (checkBtn) checkBtn.disabled = false;

    if (input) {
        input.value = '';
        input.placeholder = "Loading...";
        input.disabled = true;
    }

    if (!keepFound) {
        unscrambleState.found = [];
        unscrambleState.incorrect = [];
        if (foundList) foundList.innerHTML = '';
    }

    try {
        let data;
        // USE PREFETCHED DATA IF AVAILABLE AND MATCHES SETTINGS
        if (unscrambleState.nextData && unscrambleState.nextData.len == len && unscrambleState.nextData.dict == dict && unscrambleState.nextData.must == must) {
            data = unscrambleState.nextData.data;
            unscrambleState.nextData = null;
            console.log("[Unscramble] Using prefetched unscramble data");
        } else {
            const resp = await fetch(`/api/tools/unscramble/random?length=${len}&dictionary=${dict}&must_have=${encodeURIComponent(must)}`);
            data = await resp.json();
        }

        if (data.error) {
            alert(data.error);
            if (display) display.innerText = "Error";
            return;
        }

        unscrambleState.jumbled = data.jumbled;
        unscrambleState.solution = new Set(data.words.map(w => w.toUpperCase()));

        if (display) display.innerText = data.jumbled.toUpperCase();
        if (info) info.innerText = `${data.count} word${data.count !== 1 ? 's' : ''} possible`;

        // FINAL SAFETY CHECK: If we requested a letter and it's missing, FORCE RE-FETCH
        if (must && !data.jumbled.toUpperCase().includes(must)) {
            console.error("CRITICAL: Scrambled word missing required letter. Auto-correcting...");
            unscrambleState.nextData = null;
            unscrambleState.isLoading = false;
            return startNewUnscramble(keepFound);
        }

        renderUnscrambleFound();

        if (input) {
            input.placeholder = "Type here...";
            input.disabled = false;
            input.focus();
        }

        // TRIGGER PREFETCH FOR THE NEXT ONE
        prefetchUnscramble();

    } catch (err) {
        console.error("Unscramble Fetch Error:", err);
        if (display) display.innerText = "Network Error";
        if (input) {
            input.placeholder = "Error - Retry";
            input.disabled = false;
        }
    } finally {
        unscrambleState.isLoading = false;
    }
}

async function prefetchUnscramble() {
    const lenInput = document.getElementById('unscramble-length');
    const dictInput = document.getElementById('unscramble-dict');
    const mustInput = document.getElementById('unscramble-must-have');
    const len = lenInput ? lenInput.value : 7;
    const dict = dictInput ? dictInput.value : 'NWL';
    const must = mustInput ? mustInput.value.trim().toUpperCase() : '';

    try {
        const resp = await fetch(`/api/tools/unscramble/random?length=${len}&dictionary=${dict}&must_have=${encodeURIComponent(must)}`);
        const data = await resp.json();
        if (!data.error) {
            unscrambleState.nextData = { data, len, dict, must };
            console.log("[Unscramble] Next unscramble prefetched");
        }
    } catch (e) { }
}

function renderUnscrambleHistory() {
    renderUnscrambleFound();
}

function checkUnscrambleGuess() {
    const input = document.getElementById('unscramble-input');
    if (!input || input.disabled) return;

    const guess = input.value.trim().toUpperCase();
    if (!guess) return;

    if (unscrambleState.solution && unscrambleState.solution.has(guess)) {
        if (!unscrambleState.found.includes(guess)) {
            unscrambleState.found.push(guess);
            renderUnscrambleFound();

            input.style.borderColor = '#4caf50';
            setTimeout(() => { if (input) input.style.borderColor = ''; }, 300);

            // Only auto-advance if ALL possible words are found
            if (unscrambleState.found.length === unscrambleState.solution.size) {
                recordCurrentRoundToHistory();
                input.disabled = true;
                setTimeout(() => startNewUnscramble(), 800);
            }
        }
    } else {
        // Track incorrect guess to show in table
        if (!unscrambleState.incorrect.includes(guess)) {
            unscrambleState.incorrect.push(guess);
            renderUnscrambleFound();
        }
        input.style.borderColor = '#ff5252';
        setTimeout(() => {
            if (input) input.style.borderColor = '';
        }, 500);
    }

    input.value = '';
    const isMobile = window.innerWidth <= 900 || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    if (!isMobile) {
        input.focus();
    }
}

let unscrambleNextTimeout = null;

async function revealUnscrambleSolutions() {
    // If already waiting/countdown, clicking again SKIPS immediately
    if (unscrambleState.isWaiting) {
        startNewUnscramble();
        return;
    }

    unscrambleState.isWaiting = true;

    if (unscrambleNextTimeout) clearTimeout(unscrambleNextTimeout);

    if (!unscrambleState.solution || unscrambleState.solution.size === 0) {
        startNewUnscramble();
        return;
    }

    // 1. Record current round to history & Show all solutions (Found = Green, Missed = Red)
    recordCurrentRoundToHistory();
    renderUnscrambleFound(true);

    // 2. Lock tools except the reveal button (which now becomes "Next")
    const input = document.getElementById('unscramble-input');
    const revealBtn = document.getElementById('unscramble-reveal-btn');
    const genBtn = document.getElementById('unscramble-gen-btn');
    const checkBtn = document.getElementById('unscramble-check-btn');

    if (input) {
        input.value = '';
        input.disabled = true;
    }
    if (genBtn) genBtn.disabled = true;
    if (checkBtn) checkBtn.disabled = true;

    // 3. Start visible countdown
    let timeLeft = 2;
    const updateCountdown = () => {
        if (revealBtn) {
            revealBtn.innerText = `Next in ${timeLeft}s (Click to Skip)`;
            revealBtn.disabled = false; // ALLOW CLICK TO SKIP
            revealBtn.style.background = 'linear-gradient(135deg, #f39c12, #e67e22)';
        }
        if (timeLeft <= 0) {
            startNewUnscramble();
        } else {
            timeLeft--;
            unscrambleNextTimeout = setTimeout(updateCountdown, 1000);
        }
    };

    updateCountdown();
}

function renderUnscrambleFound(revealMissed = false) {
    const list = document.getElementById('unscramble-found-list');
    const resContainer = document.getElementById('unscramble-found-container');
    if (resContainer) {
        resContainer.classList.remove('hidden');
        resContainer.style.display = 'flex';
    }
    if (!list) return;

    let html = '';

    // 1. CURRENT ROUND SECTION
    if (unscrambleState.jumbled) {
        const solutions = Array.from(unscrambleState.solution).sort();

        html += `<div style="width: 100%; border-bottom: 1px solid rgba(255,255,255,0.12); padding-bottom: 14px; margin-bottom: 16px; display: flex; flex-direction: column; gap: 10px;">
                    <div style="font-size: 0.85rem; text-transform: uppercase; color: #ffd700; letter-spacing: 2px; font-weight: 800; text-shadow: 0 1px 3px rgba(0,0,0,0.5);">Active: ${unscrambleState.jumbled.toUpperCase()} (${unscrambleState.found.length}/${solutions.length} Found)</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; width: 100%;">`;

        solutions.forEach(w => {
            const isFound = unscrambleState.found.includes(w);
            let style = "background: rgba(255, 255, 255, 0.08); color: #94a3b8; border: 1.5px dashed rgba(255, 255, 255, 0.22);";
            let displayWord = w.replace(/./g, '_');
            let isClickable = false;

            if (isFound) {
                style = "background: rgba(46, 204, 113, 0.25); border: 1.5px solid #2ecc71; color: #4ade80;";
                displayWord = w;
                isClickable = true;
            } else if (revealMissed) {
                style = "background: rgba(239, 68, 68, 0.25); border: 1.5px solid #ef4444; color: #fca5a5;";
                displayWord = w;
                isClickable = true;
            }

            if (isClickable) {
                html += `<div class="clickable-word-link" onclick="window.lookupWord('${w}', event)" style="${style} padding: 8px 16px; border-radius: 8px; font-weight: 700; font-size: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.3); cursor: pointer; transition: all 0.2s ease; text-shadow: 0 1px 2px rgba(0,0,0,0.4); white-space: nowrap; display: inline-flex; align-items: center; justify-content: center; min-height: 36px; box-sizing: border-box;">${displayWord}</div>`;
            } else {
                html += `<div style="${style} padding: 8px 16px; border-radius: 8px; font-weight: 700; font-size: 1rem; letter-spacing: 3px; box-shadow: 0 2px 6px rgba(0,0,0,0.2); user-select: none; white-space: nowrap; display: inline-flex; align-items: center; justify-content: center; min-height: 36px; box-sizing: border-box;">${displayWord}</div>`;
            }
        });

        // Incorrect Guesses for current round
        unscrambleState.incorrect.forEach(w => {
            html += `<div style="background: rgba(239, 68, 68, 0.18); color: #f87171; padding: 7px 14px; border-radius: 8px; font-weight: 700; border: 1.5px dotted rgba(239, 68, 68, 0.5); font-size: 0.95rem; text-decoration: line-through; white-space: nowrap; display: inline-flex; align-items: center; justify-content: center; min-height: 36px; box-sizing: border-box;">${w}</div>`;
        });

        html += `   </div>
                </div>`;
    }

    // 2. HISTORY SECTION
    html += `<div style="width: 100%; margin-top: 6px; padding-top: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.12); padding-bottom: 8px;">
                    <div style="font-size: 0.85rem; text-transform: uppercase; color: #f1f5f9; letter-spacing: 2px; font-weight: 800;">Session History (${unscrambleState.history.length})</div>
                    ${unscrambleState.history.length > 0 ? `<button onclick="window.clearUnscrambleHistory()" style="background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25); color: #f8fafc; font-size: 0.78rem; font-weight: 700; padding: 4px 12px; border-radius: 6px; cursor: pointer; transition: all 0.2s;">Clear History</button>` : ''}
                </div>`;

    if (unscrambleState.history.length === 0) {
        html += `<div style="text-align: center; color: #94a3b8; font-size: 0.92rem; font-style: italic; padding: 18px 10px;">No rounds completed yet this session. Solve words or click Reveal to build your history!</div>`;
    } else {
        html += `
            <div class="unscramble-history-scroll-wrapper" style="position: relative; width: 100%;">
                <div id="unscramble-history-scroll" class="unscramble-history-list" style="display: flex; flex-direction: column; gap: 12px; width: 100%; max-height: 380px; overflow-y: auto; box-sizing: border-box; padding-right: 22px;">`;
        unscrambleState.history.forEach((h) => {
            const foundCount = h.found.length;
            const totalCount = h.solutions.length;
            const isPerfect = (foundCount === totalCount && totalCount > 0);

            html += `
                <div class="unscramble-history-item" style="background: rgba(15, 20, 38, 0.9); border-radius: 12px; padding: 12px 14px 14px 14px; border: 1.5px solid rgba(255, 255, 255, 0.14); display: flex; flex-direction: column; gap: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.35); width: 100%; box-sizing: border-box; min-height: fit-content; overflow: visible;">
                    <div class="unscramble-history-header" style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; flex-wrap: nowrap; gap: 6px; width: 100%;">
                        <span class="unscramble-history-jumbled" style="font-weight: 800; color: #ffd700; font-size: 1.05rem; letter-spacing: 1.5px; white-space: nowrap; flex-shrink: 0;">${h.jumbled.toUpperCase()}</span>
                        <div class="unscramble-history-meta" style="display: flex; align-items: center; gap: 6px; flex-shrink: 0;">
                            <span class="unscramble-history-pill" style="font-size: 0.8rem; color: ${isPerfect ? '#4ade80' : '#e2e8f0'}; font-weight: 700; background: ${isPerfect ? 'rgba(46, 204, 113, 0.18)' : 'rgba(255, 255, 255, 0.08)'}; padding: 2px 7px; border-radius: 5px; border: 1px solid ${isPerfect ? 'rgba(46, 204, 113, 0.4)' : 'rgba(255, 255, 255, 0.12)'}; white-space: nowrap;">
                                ${foundCount}/${totalCount} Found
                            </span>
                            <span class="unscramble-history-time" style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; font-family: monospace; white-space: nowrap;">${h.timestamp}</span>
                        </div>
                    </div>
                    <!-- Full-width, generous space for every single word -->
                    <div class="unscramble-history-words" style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-start; width: 100%; align-items: center; min-height: 36px; box-sizing: border-box; overflow: visible;">
                        ${h.solutions.map(s => {
                            const wereFound = h.found.includes(s);
                            const color = wereFound ? '#4ade80' : '#fca5a5';
                            const bg = wereFound ? 'rgba(46, 204, 113, 0.22)' : 'rgba(239, 68, 68, 0.18)';
                            const bdr = wereFound ? 'rgba(46, 204, 113, 0.5)' : 'rgba(239, 68, 68, 0.4)';
                            return `<span class="clickable-word-link" onclick="window.lookupWord('${s}', event)" style="font-size: 0.9rem; font-weight: 700; background: ${bg}; padding: 6px 12px; border-radius: 8px; color: ${color}; border: 1.5px solid ${bdr}; cursor: pointer; text-shadow: 0 1px 2px rgba(0,0,0,0.4); white-space: nowrap; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 2px 5px rgba(0,0,0,0.25); min-height: 34px; height: auto; line-height: 1.2; box-sizing: border-box; text-decoration: none; overflow: visible;">${s}</span>`;
                        }).join('')}
                    </div>
                </div>
            `;
        });
        html += `
                </div>
                <div class="custom-scrollbar-track" id="unscramble-history-scrollbar-track" style="right: 2px; top: 2px; bottom: 2px;">
                    <div class="custom-scrollbar-thumb" id="unscramble-history-scrollbar-thumb"></div>
                </div>
            </div>`;
    }
    html += `</div>`;

    list.innerHTML = html;

    if (unscrambleState.history.length > 0) {
        requestAnimationFrame(() => {
            initCustomScrollbarForElement('unscramble-history-scroll', 'unscramble-history-scrollbar-track', 'unscramble-history-scrollbar-thumb');
        });
    }
}

window.startNewUnscramble = startNewUnscramble;
window.unscrambleState = unscrambleState;

// ==========================================
// FIND COUNT
// ==========================================
function setupFindCountTool() {
    const input = document.getElementById('find-count-input');
    const btn = document.getElementById('find-count-btn');
    
    if (btn) {
        btn.addEventListener('click', runFindCountSearch);
    }
    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runFindCountSearch();
        });
    }

    const moreBtn = document.getElementById('more-random-words-btn');
    if (moreBtn) {
        moreBtn.addEventListener('click', (e) => {
            if (e && e.isTrusted) {
                loadRandomSuggestedWords(true);
            }
        });
        moreBtn.addEventListener('mouseenter', () => {
            moreBtn.style.background = 'rgba(165, 180, 252, 0.15)';
            moreBtn.style.borderColor = 'rgba(165, 180, 252, 0.5)';
        });
        moreBtn.addEventListener('mouseleave', () => {
            moreBtn.style.background = 'rgba(165, 180, 252, 0.08)';
            moreBtn.style.borderColor = 'rgba(165, 180, 252, 0.3)';
        });
    }

    const dictSelect = document.getElementById('random-words-dict-select');
    if (dictSelect) {
        dictSelect.addEventListener('change', (e) => {
            if (e && e.isTrusted) {
                loadRandomSuggestedWords(true);
            }
        });
    }

    // Pre-fetch and lock initial 6 random words immediately on setup so they are fixed on page load
    loadRandomSuggestedWords(false);
}

let _isFetchingRandomWords = false;
let _randomWordsLoadedOnce = false;
let _cachedRandomWords = null;
let _initialDesktopRandomWords = null;
let _randomWordsFetchPromise = null;

function renderSuggestedWordsTable(tableBody, words) {
    if (!tableBody || !words || words.length === 0) return;
    tableBody.innerHTML = words.map(word => `
        <tr class="suggested-word-row" data-word="${word}" style="cursor: pointer; border-bottom: 1px solid rgba(var(--text-primary-rgb), 0.05); transition: background 0.2s;">
            <td style="padding: 7px 10px; color: var(--accent-color); font-weight: 500;">${word}</td>
        </tr>
    `).join('');

    // Add click events to suggested rows
    tableBody.querySelectorAll('.suggested-word-row').forEach(row => {
        row.addEventListener('click', () => {
            const word = row.dataset.word;
            const input = document.getElementById('find-count-input');
            if (input && word) {
                input.value = word;
                runFindCountSearch();
            }
        });
        row.addEventListener('mouseenter', () => {
            row.style.background = 'rgba(var(--text-primary-rgb), 0.05)';
        });
        row.addEventListener('mouseleave', () => {
            row.style.background = 'transparent';
        });
    });
}

async function loadRandomSuggestedWords(force = false) {
    const tableBody = document.getElementById('random-words-table-body');
    if (!tableBody) return;

    const targetWords = _initialDesktopRandomWords || _cachedRandomWords;

    // 1. If words are already cached and not forcing a refresh, keep original display permanently
    if (!force && targetWords && targetWords.length > 0) {
        renderSuggestedWordsTable(tableBody, targetWords);
        return;
    }

    // 2. If a fetch request is already in progress, await existing promise to prevent duplicate concurrent fetches
    if (_randomWordsFetchPromise) {
        try {
            await _randomWordsFetchPromise;
        } catch (e) {}
        const currentWords = _initialDesktopRandomWords || _cachedRandomWords;
        if (!force && currentWords && currentWords.length > 0) {
            renderSuggestedWordsTable(tableBody, currentWords);
        }
        return;
    }

    // 3. If not forced and already loaded once, preserve display
    if (!force && _randomWordsLoadedOnce && _cachedRandomWords && _cachedRandomWords.length > 0) {
        renderSuggestedWordsTable(tableBody, _cachedRandomWords);
        return;
    }

    _isFetchingRandomWords = true;

    const dictSelect = document.getElementById('random-words-dict-select');
    const selectedDict = dictSelect ? dictSelect.value : 'ALL';

    if (tableBody.querySelectorAll('.suggested-word-row').length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td style="padding: 12px; opacity: 0.6;">
                    <div class="loading-spinner" style="margin: 0 auto; width: 20px; height: 20px; border-width: 2px;"></div>
                </td>
            </tr>
        `;
    }

    _randomWordsFetchPromise = (async () => {
        try {
            const response = await fetch(`/api/tools/random-words?dictionary=${encodeURIComponent(selectedDict)}`);
            const data = await response.json();
            
            if (data.error) {
                if (tableBody.querySelectorAll('.suggested-word-row').length === 0) {
                    tableBody.innerHTML = `
                        <tr>
                            <td style="padding: 12px; color: #ff6b6b;">Error: ${data.error}</td>
                        </tr>
                    `;
                }
                return;
            }

            if (data.words && data.words.length > 0) {
                if (!_initialDesktopRandomWords) {
                    _initialDesktopRandomWords = data.words;
                }
                _cachedRandomWords = data.words;
                _randomWordsLoadedOnce = true;
                renderSuggestedWordsTable(tableBody, data.words);
            } else {
                tableBody.innerHTML = `
                    <tr>
                        <td style="padding: 15px; opacity: 0.6;">No random words available.</td>
                    </tr>
                `;
            }
        } catch (err) {
            console.error('Failed to load random words:', err);
            if (tableBody.querySelectorAll('.suggested-word-row').length === 0) {
                tableBody.innerHTML = `
                    <tr>
                        <td style="padding: 15px; color: #ff6b6b;">Failed to load words.</td>
                    </tr>
                `;
            }
        } finally {
            _isFetchingRandomWords = false;
            _randomWordsFetchPromise = null;
        }
    })();

    await _randomWordsFetchPromise;
}
window.loadRandomSuggestedWords = loadRandomSuggestedWords;

async function runFindCountSearch() {
    const input = document.getElementById('find-count-input');
    if (!input) return;
    
    const word = input.value.trim().toUpperCase();
    if (!word) return;
    
    const resultsContainer = document.getElementById('find-count-results-container');
    const summaryEl = document.getElementById('find-count-summary');
    const tableBody = document.getElementById('find-count-table-body');
    
    if (summaryEl) summaryEl.innerHTML = '<div class="loading-spinner">Searching...</div>';
    if (resultsContainer) resultsContainer.classList.remove('hidden');
    if (tableBody) tableBody.innerHTML = '';
    
    try {
        const response = await fetch(`/api/tools/find-count?word=${encodeURIComponent(word)}`);
        const data = await response.json();
        
        if (data.error) {
            if (summaryEl) summaryEl.innerText = `Error: ${data.error}`;
            return;
        }
        
        if (summaryEl) {
            if (data.is_valid) {
                summaryEl.innerText = `The word "${data.word}" has been found ${data.count} ${data.count === 1 ? 'time' : 'times'} since Morpheme began in 2026.`;
            } else {
                const escapedWord = (data.word || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
                let html = `<span style="color: #ff6b6b; font-weight: 700;">⚠️ "${escapedWord}" is not a valid word in Morpheme's dictionaries.</span>`;
                if (data.count > 0) {
                    html += `<br><span style="font-size: 0.95rem; opacity: 0.95;">However, it has been found ${data.count} ${data.count === 1 ? 'time' : 'times'} in custom rounds or historical matches.</span>`;
                }
                summaryEl.innerHTML = html;
            }
        }
        
        if (data.recent && data.recent.length > 0) {
            tableBody.innerHTML = data.recent.map(item => {
                const formattedDate = typeof window.formatAppDate === 'function' ? window.formatAppDate(item.timestamp, true) : item.timestamp;
                
                const flagHtml = window.getFlagHtml ? window.getFlagHtml(item.country_flag) : (item.country_flag || '');
                return `
                    <tr class="finder-row" data-username="${item.username}" style="cursor: pointer; border-bottom: 1px solid rgba(var(--text-primary-rgb), 0.05); transition: background 0.2s;">
                        <td style="padding: 12px; color: var(--accent-color); font-weight: 500; white-space: nowrap; word-break: normal;">
                            <div style="display: inline-flex; align-items: center; gap: 8px; white-space: nowrap;">
                                ${flagHtml} <span style="white-space: nowrap; font-weight: 600;">${item.username}</span>
                            </div>
                        </td>
                        <td style="padding: 12px; color: var(--muted-text); white-space: nowrap;">${formattedDate}</td>
                    </tr>
                `;
            }).join('');
            
            // Add click events to rows
            tableBody.querySelectorAll('.finder-row').forEach(row => {
                row.addEventListener('click', () => {
                    const username = row.dataset.username;
                    if (username && username !== 'System') {
                        window.showMiniProfile(username);
                    }
                });
                row.addEventListener('mouseenter', () => {
                    row.style.background = 'rgba(var(--text-primary-rgb), 0.05)';
                });
                row.addEventListener('mouseleave', () => {
                    row.style.background = 'transparent';
                });
            });
        } else {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="2" style="padding: 20px; text-align: center; color: var(--muted-text);">No one has found this word yet.</td>
                </tr>
            `;
        }
    } catch (err) {
        console.error('Find Count error:', err);
        if (summaryEl) summaryEl.innerText = 'Search failed. Please try again.';
    }
}

// ==========================================
// PERSONAL TIMER
// ==========================================
let personalTimerInterval = null;
let personalTimerSeconds = 0;

function setupPersonalTimer() {
    const startBtn = document.getElementById('timer-start-btn');
    const stopBtn = document.getElementById('timer-stop-btn');
    const displayContainer = document.getElementById('timer-display-container');
    const displayLabel = document.getElementById('timer-countdown-display');
    const hoursInput = document.getElementById('timer-hours');
    const minutesInput = document.getElementById('timer-minutes');
    const defPanel = document.querySelector('.definitions-panel');
    const defContent = document.getElementById('definition-content');
    const defHeader = document.getElementById('definition-header');

    if (!startBtn) return;

    function formatTime(totalSeconds) {
        const h = Math.floor(totalSeconds / 3600);
        const m = Math.floor((totalSeconds % 3600) / 60);
        const s = totalSeconds % 60;
        return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }

    startBtn.addEventListener('click', () => {
        const h = parseInt(hoursInput.value) || 0;
        const m = parseInt(minutesInput.value) || 0;
        const totalSecs = (h * 3600) + (m * 60);

        if (totalSecs <= 0) {
            alert('Please set a valid duration.');
            return;
        }

        personalTimerSeconds = totalSecs;

        // UI Update
        startBtn.style.display = 'none';
        stopBtn.style.display = 'block';
        displayContainer.style.display = 'block';
        displayLabel.textContent = formatTime(personalTimerSeconds);
        hoursInput.disabled = true;
        minutesInput.disabled = true;

        // Reset any previous flashing
        if (defPanel) defPanel.classList.remove('timer-flash');

        clearInterval(personalTimerInterval);
        personalTimerInterval = setInterval(() => {
            personalTimerSeconds--;
            if (personalTimerSeconds <= 0) {
                // Time is up
                clearInterval(personalTimerInterval);
                displayLabel.textContent = "00:00:00";

                // Flash definitions panel
                if (defPanel) {
                    defPanel.classList.add('timer-flash');
                    if (defHeader) defHeader.style.display = 'none';
                    if (defContent) defContent.innerHTML = `
                        <div style="display: flex; justify-content: center; align-items: center; height: 100%; text-align: center;">
                            <h2 style="font-size: 2.5rem; color: #fff; text-shadow: 0 0 20px rgba(0,0,0,0.9); font-weight: 900; text-transform: uppercase; letter-spacing: 5px; animation: textPulse 1s infinite; margin: 0;">Time is up!</h2>
                        </div>`;
                }
            } else {
                displayLabel.textContent = formatTime(personalTimerSeconds);
            }
        }, 1000);
    });

    stopBtn.addEventListener('click', () => {
        clearInterval(personalTimerInterval);
        startBtn.style.display = 'block';
        stopBtn.style.display = 'none';
        displayContainer.style.display = 'none';
        hoursInput.disabled = false;
        minutesInput.disabled = false;

        if (defPanel) defPanel.classList.remove('timer-flash');
    });
}

// --- Global In-Place Word Definition Popover for Tools ---
window._wordDefCache = window._wordDefCache || new Map();

window.openWordInIsValid = function (word) {
    if (!word) return;
    window.hideWordDefinitionPopup();

    // Check if full list modal is currently open
    const fullListModal = document.getElementById('full-list-modal');
    const fullListResults = document.getElementById('full-list-modal-results');
    if (fullListModal && fullListModal.style.display !== 'none' && fullListResults) {
        // Save exact scroll position
        window._savedFullListScrollTop = fullListResults.scrollTop;
        if (typeof window.closeFullListModal === 'function') {
            window.closeFullListModal();
        } else {
            fullListModal.style.display = 'none';
            document.body.style.overflow = '';
            window.isFullListLoading = false;
        }
    }

    // 1. Save scroll position if we are currently in Lists tool
    const currentActivePane = document.querySelector('.tool-pane.active');
    if (currentActivePane && currentActivePane.id === 'tool-lists') {
        const listScrollArea = document.getElementById('main-list-results');
        if (listScrollArea) {
            window._savedListsScrollTop = listScrollArea.scrollTop;
        }
    }

    // 2. Switch tool navigation to "Is Valid"
    if (typeof window.showTool === 'function') {
        window.showTool('is-valid');
    } else {
        const navBtns = document.querySelectorAll('.tool-nav-btn');
        const isValidBtn = Array.from(navBtns).find(b => b.dataset.tool === 'is-valid');
        if (isValidBtn) isValidBtn.click();
    }

    // 3. Set dictionary select to "ALL"
    const dictEl = document.getElementById('valid-dict');
    if (dictEl) dictEl.value = 'ALL';

    // 4. Set input and run check
    const input = document.getElementById('valid-input');
    if (input) {
        input.value = word;
        if (typeof runValidationCheck === 'function') {
            runValidationCheck();
        }
    }

    // 5. Scroll to results if needed
    const container = document.getElementById('valid-results-container');
    if (container) container.scrollIntoView({ behavior: 'smooth', block: 'center' });
};

window.hideWordDefinitionPopup = function () {
    const popover = document.getElementById('tool-word-def-popover');
    if (popover) {
        popover.classList.remove('active');
        setTimeout(() => {
            if (!popover.classList.contains('active')) {
                popover.style.display = 'none';
            }
        }, 150);
    }
};

window.showWordDefinitionPopup = async function (word, event) {
    if (!word) return;
    const cleanWord = String(word).trim().toUpperCase();

    let popover = document.getElementById('tool-word-def-popover');
    if (!popover) {
        popover = document.createElement('div');
        popover.id = 'tool-word-def-popover';
        popover.className = 'tool-def-popover';
        popover.innerHTML = `
            <div class="tool-def-popover-header">
                <div class="tool-def-word-title">
                    <span id="tool-def-word-text"></span>
                    <span id="tool-def-len-badge" class="tool-def-len-badge"></span>
                </div>
                <button type="button" class="tool-def-close-btn" onclick="window.hideWordDefinitionPopup()" title="Close">✕</button>
            </div>
            <div id="tool-def-pronunciation" class="tool-def-pronunciation" style="display: none;"></div>
            <div id="tool-def-content" class="tool-def-content">Loading definition...</div>
            <div class="tool-def-actions">
                <a href="javascript:void(0)" id="tool-def-isvalid-btn" class="tool-def-isvalid-link">Open in Is Valid ↗</a>
            </div>
        `;
        document.body.appendChild(popover);

        // Close on document click outside
        document.addEventListener('click', (e) => {
            if (popover && popover.classList.contains('active')) {
                if (!popover.contains(e.target) && !e.target.closest('.clickable-word-link') && !e.target.closest('.full-list-item')) {
                    window.hideWordDefinitionPopup();
                }
            }
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                window.hideWordDefinitionPopup();
            }
        });
    }

    const wordTextEl = document.getElementById('tool-def-word-text');
    const lenBadgeEl = document.getElementById('tool-def-len-badge');
    const pronEl = document.getElementById('tool-def-pronunciation');
    const contentEl = document.getElementById('tool-def-content');
    const isValidBtn = document.getElementById('tool-def-isvalid-btn');

    if (wordTextEl) wordTextEl.textContent = cleanWord;
    if (lenBadgeEl) lenBadgeEl.textContent = `${cleanWord.length}L`;
    if (pronEl) pronEl.style.display = 'none';
    if (isValidBtn) {
        isValidBtn.onclick = () => window.openWordInIsValid(cleanWord);
    }

    // Smart Positioning
    const evt = event || window.event;
    const target = evt ? (evt.currentTarget || evt.target) : null;
    popover.style.display = 'block';

    if (target && typeof target.getBoundingClientRect === 'function') {
        const rect = target.getBoundingClientRect();
        const popWidth = Math.min(320, window.innerWidth - 24);
        let left = rect.left;
        if (left + popWidth > window.innerWidth - 12) {
            left = window.innerWidth - popWidth - 12;
        }
        if (left < 12) left = 12;

        let top = rect.bottom + 8;
        if (top + 200 > window.innerHeight && rect.top > 200) {
            top = rect.top - 190;
        }
        if (top < 12) top = 12;

        popover.style.left = `${left}px`;
        popover.style.top = `${top}px`;
        popover.style.transform = 'scale(1) translateY(0)';
    } else {
        popover.style.left = '50%';
        popover.style.top = '50%';
        popover.style.transform = 'translate(-50%, -50%)';
    }

    // Activate transition
    void popover.offsetWidth;
    popover.classList.add('active');

    // Check in-memory cache for 0ms instant display
    if (window._wordDefCache.has(cleanWord)) {
        renderDefData(window._wordDefCache.get(cleanWord));
        return;
    }

    if (contentEl) contentEl.innerHTML = '<span style="opacity: 0.6;">Loading definition...</span>';

    try {
        const res = await fetch('/api/tools/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: cleanWord, dictionary: 'ALL' })
        });
        const data = await res.json();
        window._wordDefCache.set(cleanWord, data);
        renderDefData(data);
    } catch (err) {
        if (contentEl) contentEl.innerHTML = '<span style="color: #f87171;">Failed to load definition.</span>';
    }

    function renderDefData(data) {
        if (data && data.pronunciation && pronEl) {
            pronEl.textContent = data.pronunciation;
            pronEl.style.display = 'block';
        } else if (pronEl) {
            pronEl.style.display = 'none';
        }

        if (contentEl) {
            if (data && data.definition && data.definition.trim()) {
                contentEl.textContent = data.definition;
            } else {
                contentEl.innerHTML = '<span style="opacity: 0.6; font-style: italic;">No definition available for this word.</span>';
            }
        }
    }
};

window.lookupWord = function (word, event) {
    window.showWordDefinitionPopup(word, event);
};

window.findWordPath = findWordPath;

window.highlightWordPathOnReplay = function(wordText) {
    const round = window.currentActiveReplayRound;
    if (!round || !round.board) return;
    
    let board = round.board;
    if (board && typeof board === 'object' && !Array.isArray(board)) {
        board = board.board;
    }
    if (!board) return;
    
    const rows = board.length;
    const firstRow = board[0];
    const is3D = rows === 6 && Array.isArray(firstRow) && Array.isArray(firstRow[0]);
    const overlay = document.getElementById('history-review-overlay');
    const prefix = overlay ? 'review' : 'integrated';
    const boardContainer = document.getElementById(`${prefix}-board-container`);
    if (!boardContainer) return;
    
    const cells = boardContainer.querySelectorAll('.review-cell');
    cells.forEach(c => c.classList.remove('highlight'));
    
    if (is3D) {
        if (typeof findWordPathOnCube === 'function') {
            const path = findWordPathOnCube(wordText, board);
            if (path) {
                path.forEach((p, i) => {
                    const idx = p.f * 9 + p.r * 3 + p.c;
                    setTimeout(() => {
                        if (cells[idx]) cells[idx].classList.add('highlight');
                    }, i * 40);
                });
            }
        }
    } else {
        if (typeof findWordPath === 'function') {
            const path = findWordPath(board, wordText);
            if (path) {
                const gridCols = board[0].length;
                path.forEach((p, i) => {
                    const idx = p.row * gridCols + p.col;
                    setTimeout(() => {
                        if (cells[idx]) cells[idx].classList.add('highlight');
                    }, i * 40);
                });
            }
        }
    }
};

// Change Password / Email Account Settings Logic
async function loadAccountCredentialsInfo() {
    const emailDisplay = document.getElementById('email-current-display');
    const guestBanner = document.getElementById('account-guest-banner');
    const pwdForm = document.getElementById('form-change-password');
    const emailForm = document.getElementById('form-change-email');

    try {
        const resp = await fetch('/api/user/account-info');
        const data = await resp.json();
        
        if (data.is_guest || !data.success) {
            if (guestBanner) guestBanner.classList.remove('hidden');
            if (emailDisplay) emailDisplay.textContent = 'Guest (No email registered)';
            if (pwdForm) {
                pwdForm.querySelectorAll('input, button').forEach(el => el.disabled = true);
            }
            if (emailForm) {
                emailForm.querySelectorAll('input, button').forEach(el => el.disabled = true);
            }
        } else {
            if (guestBanner) guestBanner.classList.add('hidden');
            if (emailDisplay) {
                emailDisplay.textContent = data.email || 'None set';
                emailDisplay.style.color = data.email ? '#38bdf8' : '#94a3b8';
            }
            if (pwdForm) {
                pwdForm.querySelectorAll('input, button').forEach(el => el.disabled = false);
            }
            if (emailForm) {
                emailForm.querySelectorAll('input, button').forEach(el => el.disabled = false);
            }
        }
    } catch (e) {
        console.error('[AccountSettings] Error fetching account info:', e);
        if (emailDisplay) emailDisplay.textContent = 'Error loading email';
    }
}
window.loadAccountCredentialsInfo = loadAccountCredentialsInfo;

function setupAccountSettings() {
    const pwdForm = document.getElementById('form-change-password');
    const emailForm = document.getElementById('form-change-email');
    
    if (pwdForm) {
        pwdForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const currentPwd = document.getElementById('pwd-current')?.value.trim();
            const newPwd = document.getElementById('pwd-new')?.value.trim();
            const confirmPwd = document.getElementById('pwd-confirm')?.value.trim();
            const alertEl = document.getElementById('password-alert');
            const submitBtn = document.getElementById('btn-change-password');

            const showAlert = (msg, isError) => {
                if (!alertEl) return;
                alertEl.style.display = 'block';
                alertEl.style.background = isError ? 'rgba(239, 68, 68, 0.15)' : 'rgba(34, 197, 94, 0.15)';
                alertEl.style.border = isError ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid rgba(34, 197, 94, 0.4)';
                alertEl.style.color = isError ? '#fca5a5' : '#86efac';
                alertEl.textContent = msg;
            };

            if (!currentPwd) {
                showAlert('Please enter your current password.', true);
                return;
            }
            if (!newPwd) {
                showAlert('Please enter a new password.', true);
                return;
            }
            if (newPwd.length < 4) {
                showAlert('New password must be at least 4 characters long.', true);
                return;
            }
            if (newPwd !== confirmPwd) {
                showAlert('New password entries do not match. Please re-enter.', true);
                return;
            }

            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.textContent = 'Updating Password...';
            }

            try {
                const resp = await fetch('/api/user/change-password', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        current_password: currentPwd,
                        new_password: newPwd,
                        confirm_password: confirmPwd
                    })
                });
                const data = await resp.json();
                if (data.success) {
                    showAlert('✓ ' + (data.message || 'Password successfully changed!'), false);
                    pwdForm.reset();
                } else {
                    showAlert('✗ ' + (data.error || 'Failed to change password.'), true);
                }
            } catch (err) {
                showAlert('✗ Connection error: ' + err.message, true);
            } finally {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Change Password';
                }
            }
        });
    }

    if (emailForm) {
        emailForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const newEmail = document.getElementById('email-new')?.value.trim();
            const emailDisplay = document.getElementById('email-current-display');
            const alertEl = document.getElementById('email-alert');
            const submitBtn = document.getElementById('btn-change-email');

            const showAlert = (msg, isError) => {
                if (!alertEl) return;
                alertEl.style.display = 'block';
                alertEl.style.background = isError ? 'rgba(239, 68, 68, 0.15)' : 'rgba(34, 197, 94, 0.15)';
                alertEl.style.border = isError ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid rgba(34, 197, 94, 0.4)';
                alertEl.style.color = isError ? '#fca5a5' : '#86efac';
                alertEl.textContent = msg;
            };

            if (!newEmail) {
                showAlert('Please enter a new email address.', true);
                return;
            }

            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(newEmail)) {
                showAlert('Please enter a valid email address.', true);
                return;
            }

            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.textContent = 'Updating Email...';
            }

            try {
                const resp = await fetch('/api/user/change-email', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ new_email: newEmail })
                });
                const data = await resp.json();
                if (data.success) {
                    showAlert('✓ ' + (data.message || 'Email successfully changed!'), false);
                    if (emailDisplay) {
                        emailDisplay.textContent = data.email || newEmail;
                        emailDisplay.style.color = '#38bdf8';
                    }
                    emailForm.reset();
                } else {
                    showAlert('✗ ' + (data.error || 'Failed to change email.'), true);
                }
            } catch (err) {
                showAlert('✗ Connection error: ' + err.message, true);
            } finally {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Change Email';
                }
            }
        });
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupAccountSettings);
} else {
    setupAccountSettings();
}
