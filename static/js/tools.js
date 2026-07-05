document.addEventListener('DOMContentLoaded', () => {
    setupToolsNavigation();
    setupProfileTool();
    setupComboChecker();
    setupListsTool();
    setupSequenceTool();
    setupManualTool();
    setupRandomWordTool();
    setupWotdTool();
    setupSubanagramsTool();
    setupIsValidTool();
    setupPrivateMessaging();
    setupMiniProfileModal();
    setupImageLightbox();
    setupUnscrambleTool();
    setupFindCountTool();
    setupPersonalTimer();
});

// NEW: Global UTC Timestamp Parser to prevent local timezone offsets
window.parseUTCTimestamp = function(isoStr) {
    if (!isoStr) return new Date();
    if (typeof isoStr === 'number') return new Date(isoStr);
    const dateStr = isoStr.includes('Z') || isoStr.includes('+') ? isoStr.replace(' ', 'T') : isoStr.replace(' ', 'T') + 'Z';
    return new Date(dateStr);
};

// NEW: Global Tool Switcher Helper
window.showTool = function(toolId) {
    const navBtns = document.querySelectorAll('.tool-nav-btn');
    const panes = document.querySelectorAll('.tool-pane');
    
    // Update buttons
    navBtns.forEach(b => {
        if (b.dataset.tool === toolId) b.classList.add('active');
        else b.classList.remove('active');
    });

    // Show pane
    panes.forEach(p => {
        if (p.id === `tool-${toolId}`) p.classList.add('active');
        else p.classList.remove('active');
    });

    // Trigger lazy loads
    if (toolId === 'profile') {
        if (typeof refreshProfileTool === 'function') refreshProfileTool();
    }
    if (toolId === 'lists') {
        if (typeof fetchListsData === 'function') fetchListsData();
    }
    if (toolId === 'wotd') {
        if (typeof updateWotd === 'function') updateWotd();
    }
    if (toolId === 'manual') {
        fetch('/api/tools/flag_manual', { method: 'POST' }).catch(e => console.error(e));
    }
    if (toolId === 'find-count') {
        if (typeof loadRandomSuggestedWords === 'function') loadRandomSuggestedWords();
    }

    // Scroll tools content into view on mobile
    const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    if (isMobile) {
        const layout = document.querySelector('#page-tools .tools-split-layout');
        if (layout) {
            layout.scrollTo({ left: layout.clientWidth, behavior: 'smooth' });
        }
    }
};

function setupToolsNavigation() {
    const sidebar = document.querySelector('.tools-sidebar');
    if (!sidebar) return;

    sidebar.addEventListener('click', (e) => {
        const btn = e.target.closest('.tool-nav-btn');
        if (!btn) return;

        const toolId = btn.dataset.tool;
        if (toolId) {
            // Check if this tool is already active to prevent double-firing with inline onclick
            const currentActive = sidebar.querySelector('.tool-nav-btn.active');
            if (currentActive && currentActive.dataset.tool === toolId) {
                return;
            }
            window.showTool(toolId);
        }
    });

    // Mobile Layout snapping on navigation
    const toolsPage = document.getElementById('page-tools');
    if (toolsPage) {
        const observer = new MutationObserver(() => {
            if (toolsPage.classList.contains('active')) {
                const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
                if (isMobile) {
                    setTimeout(() => {
                        const layoutEl = document.querySelector('#page-tools .tools-split-layout');
                        if (layoutEl) {
                            layoutEl.scrollLeft = 0;
                        }
                    }, 50);
                }
            }
        });
        observer.observe(toolsPage, {
            attributes: true,
            attributeFilter: ['class']
        });
    }

    // Mobile touch swipe handling for sliding back to tools list
    const toolsContent = document.querySelector('.tools-content');
    const toolsSidebar = document.querySelector('.tools-sidebar');
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

    // Mobile back button inside tools content
    const mobileBackBtn = document.getElementById('tools-mobile-back-btn');
    if (mobileBackBtn) {
        mobileBackBtn.addEventListener('click', () => {
            const layoutEl = document.querySelector('#page-tools .tools-split-layout');
            if (layoutEl) {
                layoutEl.scrollTo({ left: 0, behavior: 'smooth' });
            }
        });
    }
}

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

    // Prevent horizontal scroll/swipe chaining to the parent .tools-split-layout on mobile and desktop
    const containers = [document.getElementById('mp-container'), document.getElementById('lic-container')];
    containers.forEach(container => {
        if (!container) return;

        let startX = 0;
        let startY = 0;
        let isHorizontalSwipe = false;
        let touchIdentified = false;

        container.addEventListener('touchstart', (e) => {
            if (e.touches.length > 0) {
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                isHorizontalSwipe = false;
                touchIdentified = false;
            }
        }, { passive: true });

        container.addEventListener('touchmove', (e) => {
            if (e.touches.length > 0) {
                if (!touchIdentified) {
                    const diffX = Math.abs(e.touches[0].clientX - startX);
                    const diffY = Math.abs(e.touches[0].clientY - startY);
                    // Determine if horizontal or vertical swipe once at the start of movement
                    if (diffX > 5 || diffY > 5) {
                        isHorizontalSwipe = diffX > diffY;
                        touchIdentified = true;
                    }
                }

                if (isHorizontalSwipe) {
                    // Stop propagation to prevent the parent .tools-split-layout from swiping back to categories
                    e.stopPropagation();
                }
            }
        }, { passive: false });

        container.addEventListener('wheel', (e) => {
            if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
                // Stop propagation to prevent horizontal trackpad scroll from shifting the split-layout
                e.stopPropagation();
            }
        }, { passive: true });
    });
}

async function runComboSearch() {
    const inputEl = document.getElementById('combo-input');
    const dictEl = document.getElementById('combo-dict');
    const resultsContainer = document.getElementById('combo-results');

    const searchTerm = inputEl.value.trim().toUpperCase();
    if (!searchTerm || searchTerm.length < 3) return;

    const mpContainer = document.getElementById('mp-container');
    const licContainer = document.getElementById('lic-container');
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
        
        const colDiv = document.createElement('div');
        colDiv.className = 'group-column';
        colDiv.innerHTML = `
            <div class="group-header">${label}</div>
            <div class="group-table-container">
                <div class="group-word-list">
                    ${words.map(w => `<div class="group-row">${w}</div>`).join('')}
                </div>
            </div>
        `;
        container.appendChild(colDiv);
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

            joinedEl.innerText = `${joinedDate.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })} (${durationStr})`;
        } else if (joinedEl) {
            joinedEl.innerText = "Joined: -";
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
        if (descEl) descEl.innerText = data.description || 'No description provided.';

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
                const toolsBtn = document.querySelector('.nav-btn[data-page="tools"]');
                if (toolsBtn) toolsBtn.click();
                const profileToolBtn = document.querySelector('.tool-nav-btn[data-tool="profile"]');
                if (profileToolBtn) profileToolBtn.click();
                window.performProfileSearch(data.username);
            };
        }

        const roundReviewsBtn = document.getElementById('mini-profile-round-reviews');
        if (roundReviewsBtn) {
            roundReviewsBtn.onclick = () => {
                modal.classList.add('hidden');
                modal.classList.remove('forced-show');
                const toolsBtn = document.querySelector('.nav-btn[data-page="tools"]');
                if (toolsBtn) toolsBtn.click();
                const profileToolBtn = document.querySelector('.tool-nav-btn[data-tool="profile"]');
                if (profileToolBtn) profileToolBtn.click();
                window.performProfileSearch(data.username, 'history');
            };
        }

        const msgBtn = document.getElementById('mini-profile-message');
        const globalUser = window.currentUser || (typeof currentUser !== 'undefined' ? currentUser : null);
        const currentName = (typeof globalUser === 'object') ? globalUser.username : globalUser;

        if (msgBtn) {
            if (currentName && currentName.toLowerCase() !== data.username.toLowerCase()) {
                msgBtn.classList.remove('hidden');
                msgBtn.onclick = () => {
                    modal.classList.add('hidden');
                    modal.classList.remove('forced-show');
                    window.openPrivateChat(data.username, true);
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

        // Finally Show
        if (modal) {
            modal.classList.add('forced-show');
            modal.classList.remove('hidden');
        }

    } catch (err) {
        console.error("Mini profile fetch error:", err);
    }
};

// --- Profile Tool Logic ---

function setupProfileTool() {
    const searchBtn = document.getElementById('profile-search-btn');
    const input = document.getElementById('profile-search-input');

    if (searchBtn) {
        searchBtn.addEventListener('click', () => {
            performProfileSearch(input.value);
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
        refreshBtn.onclick = () => {
            console.log("[Profile] Manual refresh triggered");
            window.refreshProfileTool(true);
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

async function performProfileSearch(username, activeTab = null, period = 'all') {
    const errorEl = document.getElementById('profile-search-error');
    if (errorEl) {
        errorEl.style.display = 'none';
        errorEl.innerText = '';
    }

    if (!username || !username.trim()) return;

    username = username.trim();
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

    // Check if we are refreshing the current user (e.g. changing tabs/periods)
    const currentDisplayed = document.getElementById('profile-username')?.innerText || '';
    const isRefresh = currentDisplayed === username;

    if (container && !isRefresh) {
        container.classList.add('hidden');
    }

    try {
        const response = await fetch(`/api/profile/${encodeURIComponent(username)}?period=${period}&t=${Date.now()}`);
        const data = await response.json();

        if (data.error) {
            // User not found, just don't show the profile
            if (container) container.classList.add('hidden');
            if (errorEl) {
                errorEl.innerText = "The username you entered does not exist.";
                errorEl.style.display = 'block';
            }
            return;
        }

        await renderProfile(data);
        if (container) container.classList.remove('hidden');

        // Activate specific tab if requested
        if (activeTab) {
            const tabToggle = document.querySelector(`.profile-tab-toggle[data-tab="${activeTab}"]`);
            if (tabToggle) tabToggle.click();
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
    if (descriptionEl) descriptionEl.innerText = user.description || 'Add a detailed description about yourself...';

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
            durationStr = `${y} year${y > 1 ? 's' : ''}${m > 0 ? ` ${m} month${m > 1 ? 's' : ''}` : ''}`;
        } else {
            durationStr = months > 0 ? `${months} month${months > 1 ? 's' : ''}` : 'Less than 1 month';
        }

        joinedValEl.innerText = `${joinedDate.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })} (${durationStr})`;
    } else if (joinedValEl) {
        joinedValEl.innerText = '-';
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
        if (currentName && !isOwner) {
            messageBtn.classList.remove('hidden');
            const newMsgBtn = messageBtn.cloneNode(true);
            messageBtn.parentNode.replaceChild(newMsgBtn, messageBtn);
            newMsgBtn.addEventListener('click', () => {
                openPrivateChat(user.username, true);
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

        let miniBoardHTML = '';
        if (round.board && Array.isArray(round.board)) {
            const rows = round.board.length;
            const firstRow = round.board[0];
            const is3D = rows === 6 && Array.isArray(firstRow) && Array.isArray(firstRow[0]);

            if (is3D) {
                // Render the front face (Face 0)
                const frontFace = round.board[0];
                let cellsHTML = '';
                for (let r = 0; r < 3; r++) {
                    for (let c = 0; c < 3; c++) {
                        const letter = frontFace[r][c] || '?';
                        cellsHTML += `<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: rgba(var(--accent-color-rgb), 0.2); border-radius: 1px; font-size: 8px; color: #fff; font-weight: 800;">${letter}</div>`;
                    }
                }
                miniBoardHTML = `
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2px; width: 40px; height: 40px; border: 1px solid var(--accent-color); border-radius: 4px; overflow: hidden;">
                        ${cellsHTML}
                    </div>
                `;
            } else if (firstRow && Array.isArray(firstRow)) {
                const cols = firstRow.length;
                let gridCells = '';
                for (let r = 0; r < rows; r++) {
                    for (let c = 0; c < cols; c++) {
                        const letter = round.board[r] ? round.board[r][c] : '?';
                        gridCells += `<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.1); border-radius: 1px; font-size: 5px; color: rgba(255,255,255,0.5);">${letter}</div>`;
                    }
                }

                miniBoardHTML = `
                    <div style="display: grid; grid-template-columns: repeat(${cols}, 1fr); gap: 1px; width: 40px; height: 40px; pointer-events: none;">
                        ${gridCells}
                    </div>
                `;
            } else {
                miniBoardHTML = '<span style="opacity:0.3; font-size: 0.7rem;">No Preview</span>';
            }
        } else {
            miniBoardHTML = '<span style="opacity:0.3; font-size: 0.7rem;">No Preview</span>';
        }

        // Date Formatting
        let dateStr = '-';
        if (round.timestamp) {
            const d = window.parseUTCTimestamp(round.timestamp);
            try {
                dateStr = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
            } catch (e) {
                dateStr = '-';
            }
        }

        return `
        <div class="history-grid-item" onclick="watchRoundHistory('${round.room_id}', ${round.round_number}, true, ${round.game_id || 'null'})" style="display: grid; grid-template-columns: repeat(8, 1fr); gap:8px; padding: 10px 15px; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.03); border-radius: 10px; margin-bottom: 8px; align-items: center; transition: all 0.2s; cursor: pointer; position: relative; overflow: hidden; min-width: 700px;">
            <div class="history-mode-tag ${typeClass}" style="font-size: 0.65rem; padding: 3px 6px; border-radius: 6px; text-align: center; width: fit-content; font-weight: 800; text-transform: uppercase;">${gameTypeLabel}</div>
            
            <!-- Mini Board Preview Column -->
            <div style="display: flex; justify-content: center;">
                ${miniBoardHTML}
            </div>

            <div style="font-weight: 900; color: #fff; font-size: 0.95rem;">${round.total_score} <small style="font-size: 0.6rem; opacity: 0.5;">PTS</small></div>

            <div style="font-weight: 900; color: ${round.performance_value >= 140 ? '#60a5fa' : 'rgba(255,255,255,0.2)'}; font-size: 0.85rem;">${round.performance_value ? (round.performance_value / 100).toFixed(2) + 'x' : '-'}</div>
            <div style="display: flex; flex-direction: column; gap: 1px;">
                <span style="color: #fff; font-size: 0.7rem; font-weight: 700;">${round.num_words} words</span>
                <span style="color: rgba(255,255,255,0.3); font-size: 0.6rem;">Avg: ${round.avg_len}</span>
            </div>
            <div style="color: #ffd700; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;" title="${round.top_word}">${round.top_word}</div>
            <div style="display: flex; flex-direction: column; gap: 1px;">
                <span style="font-size: 0.7rem; color: #60a5fa; font-weight: 700; opacity: 0.8;">${round.room_id}</span>
                <span style="font-size: 0.6rem; color: rgba(255,255,255,0.3); font-weight: 600;">Str: ${round.room_strength || '-'}</span>
            </div>
            
            <!-- Date Column -->
            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6); font-weight: 600; text-align: right;">${dateStr}</div>
        </div>
        `;
    };

    window.roundGridHeader = `
        <div class="history-grid-header" style="display: grid; grid-template-columns: repeat(8, 1fr); gap:8px; padding: 12px 15px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 12px; font-size: 0.7rem; color: rgba(255,255,255,0.4); font-weight: 800; text-transform: uppercase; letter-spacing: 1px; min-width: 700px;">
            <div>Mode</div>
            <div style="text-align: center;">Board</div>
            <div>Score</div>
            <div>PE</div>
            <div>Stats</div>
            <div>Top Word</div>
            <div>Room</div>
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
            const peHeader = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 12px 20px; border-radius: 8px;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 0.5px;">Exceptional Performances</span>
                    <span style="font-size: 0.9rem; font-weight: 800; color: #60a5fa;">Greatest PE: <span style="color: #fff;">${greatestPE}%</span></span>
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
window.watchRoundHistory = function (roomId, roundNum, isSnapshot = false, gameId = null) {
    console.log(`Reviewing Round ${roundNum} from Room ${roomId} (GameID: ${gameId})`);

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
        const foundInLobby = window.lastGameState.winners_history.find(h => h.round == roundNum);
        if (foundInLobby && foundInLobby.board) {
            console.log(`[Review] Using Round ${roundNum} from Lobby winners_history (current session)`);

            // CRITICAL: winners_history only stores the WINNER's words.
            // Use the current player's own submitted_words from the live game state instead,
            // so the replay shows YOUR words in the order YOU found them — not the winner's words.
            let myWords = null;
            if (window.lastGameState.players && window.lastGameState.your_username) {
                const myUsername = window.lastGameState.your_username;
                const myPlayer = window.lastGameState.players.find(p => p.username === myUsername);
                if (myPlayer && myPlayer.submitted_words && myPlayer.submitted_words.length > 0) {
                    // Keep only words with actual text (filter obfuscated '???' words, though during intermission all should be visible)
                    const visible = myPlayer.submitted_words.filter(w => w.word && !w.obfuscated && w.word.indexOf('?') === -1);
                    if (visible.length > 0) {
                        myWords = visible;
                        console.log(`[Review] Using ${myWords.length} of YOUR own words for replay (not winner's words)`);
                    }
                }
            }

            round = {
                ...foundInLobby,
                room_id: roomId,
                round_number: foundInLobby.round,
                total_score: foundInLobby.score,
                game_type: foundInLobby.game_type || 'accumulative',
                // Override words with current player's own words if available
                words: myWords !== null ? myWords : (foundInLobby.words || [])
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

    // Update Date Display
    const dateEl = document.getElementById('history-review-date');
    if (dateEl) {
        if (round.timestamp) {
            const d = window.parseUTCTimestamp(round.timestamp);
            try {
                dateEl.innerText = d.toLocaleDateString(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' });
            } catch (e) {
                dateEl.innerText = '';
            }
        } else {
            dateEl.innerText = '';
        }
    }

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
        // Mobile Board Transposition: Turn landscape flat boards (rows < cols) into portrait (longest side runs vertically)
        try {
            if (window.innerWidth <= 900 && Array.isArray(round.board[0])) {
                const isReplay3D = round.board.length === 6 && Array.isArray(round.board[0]) && Array.isArray(round.board[0][0]);
                if (!isReplay3D) {
                    const rows = round.board.length;
                    const cols = round.board[0].length;
                    if (rows < cols) {
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
                // Calculate cell size dynamically to fit availWidth and availHeight
                // Horizontal: 3 faces. Gaps: 2 * 20px = 40px (between faces).
                // Gaps between cells: 3 faces * 2 gaps * 4px = 24px.
                // Padding: 40px (20px left/right).
                // Total non-cell width = 104px.
                const nonCellW = 104;
                const maxCellW3D = (availWidth - nonCellW) / 9;

                // Vertical: 2 faces. Gaps: 1 * 20px = 20px (between faces).
                // Gaps between cells: 2 faces * 2 gaps * 4px = 16px.
                // Padding: 40px (20px top/bottom).
                // Total non-cell height = 76px.
                const nonCellH = 76;
                const maxCellH3D = (availHeight - nonCellH) / 6;

                const cellSize3D = Math.max(24, Math.floor(Math.min(maxCellW3D, maxCellH3D, 50)));
                const fontSize3D = Math.floor(cellSize3D * 0.55) + 'px';

                boardContainer.style.display = 'grid';
                boardContainer.style.gridTemplateColumns = `repeat(3, max-content)`; 
                boardContainer.style.justifyContent = 'center';
                boardContainer.style.gap = `20px`; 
                boardContainer.style.padding = `20px`;
                boardContainer.style.background = `rgba(0,0,0,0.2)`;
                boardContainer.style.borderRadius = `15px`;
                boardContainer.style.overflow = 'auto';

                boardContainer.innerHTML = round.board.map((face, fIdx) => {
                    let faceHTML = '';
                    for (let r = 0; r < 3; r++) {
                        for (let c = 0; c < 3; c++) {
                            const val = (face[r] && face[r][c] !== undefined) ? face[r][c] : '?';
                            const displayVal = val === 'Q' ? 'QU' : val;
                            faceHTML += `<div class="review-cell" style="width: ${cellSize3D}px; height: ${cellSize3D}px; font-size: ${fontSize3D}; border-radius: 4px; display: flex; align-items: center; justify-content: center; aspect-ratio: 1; flex-shrink: 0;">${displayVal}</div>`;
                        }
                    }
                    return `
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; flex-shrink: 0;">
                            <div style="font-size: 0.6rem; color: rgba(255,255,255,0.3); font-weight: 900; text-transform: uppercase; white-space: nowrap;">Face ${fIdx}</div>
                            <div style="display: grid; grid-template-columns: repeat(3, ${cellSize3D}px); gap: 4px; flex-shrink: 0;">
                                ${faceHTML}
                            </div>
                        </div>
                    `;
                }).join('');
                return;
            }

            const flatBoard = round.board.flat();
            
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
                // 1. Fetch room state to check player count
                const resp = await fetch(`/api/room/${roomId}/state`);
                const data = await resp.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // 2. Decide if join as player or spectator
                // Rule: Max 8 players; more than that -> Spectator
                // Note: accumulative and fcfs modes have higher limits, but request said "8 players" specifically
                const playerCount = (data.players && data.players.length) || 0;
                const isFull = playerCount >= 8;
                console.log(`Room population: ${playerCount} (Full: ${isFull})`);

                // 3. Join the room
                const joinResp = await fetch(`/api/room/${roomId}/join`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ as_spectator: isFull })
                });
                const joinData = await joinResp.json();

                if (joinData.success) {
                    // Navigate to Play Page
                    window.currentRoomId = roomId;
                    window.isSpectatorMode = isFull;

                    if (typeof showPage === 'function') {
                        showPage('page-play');
                    }

                    if (window.startGamePolling) window.startGamePolling();

                    // Force focus for Word Input if not spectator
                    setTimeout(() => {
                        const input = document.getElementById('word-input');
                        if (input && !isFull) {
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

    const filterMode = document.getElementById('rankings-filter-mode')?.value || 'all';
    const filterDims = document.getElementById('rankings-filter-dims')?.value || 'all';
    const filterTime = document.getElementById('rankings-filter-time')?.value || 'all';

    const modes = ['accumulative', 'fcfs', 'split', '3d'];
    const boards = ['4x4', '4x6', '5x7', '6x8', '3x3x3'];
    const accTimes = [45, 180, 300, 600];
    const otherTimes = [45, 180, 300, 600];

    const formatTimeShort = (s) => {
        if (s === 45) return '45s';
        if (s === 180) return '3m';
        if (s === 300) return '5m';
        if (s === 600) return '10m';
        return s + 's';
    };

    let visibleCount = 0;

    modes.forEach(mode => {
        if (filterMode !== 'all' && mode !== filterMode) return;

        const times = (mode === 'accumulative' || mode === '3d') ? accTimes : otherTimes;
        boards.forEach(board => {
            if (filterDims !== 'all' && board !== filterDims) return;

            // COMPATIBILITY FILTER: 3x3x3 is for Cube ONLY; traditional boards for others
            if (mode === '3d' && board !== '3x3x3') return;
            if (mode !== '3d' && board === '3x3x3') return;

            times.forEach(time => {
                if (filterTime !== 'all' && String(time) !== filterTime) return;

                // COMPATIBILITY FILTER: No 45s for Cube
                if (mode === '3d' && time === 45) return;

                // COMPATIBILITY FILTER: FCFS and Split do not support 5m (300) and 10m (600)
                if ((mode === 'fcfs' || mode === 'split') && (time === 300 || time === 600)) return;

                // COMPATIBILITY FILTER: 2D rooms do not support 5m (300)
                if (mode !== '3d' && time === 300) return;

                const configKey = `${mode}|${board}|${time}`;
                const configData = ratings[configKey] || { rating: 1200, games_played: 0, wins: 0, point_sum: 0, avg_pct_found: 0 };
                const rating = configData.rating;

                const rColor = window.getRatingColor ? window.getRatingColor(rating) : '#b3b3b3';

                const box = document.createElement('div');
                box.className = 'rating-box clickable';
                box.title = "Click to view achievements for this room type";
                box.innerHTML = `
                    <div class="rating-box-swatch" style="background: ${rColor};"></div>
                    <div class="rating-box-info" style="flex: 1;">
                        <div class="rating-box-mode" style="font-size: 0.65rem; color: rgba(255,255,255,0.4); text-transform: uppercase; font-weight: 800;">${mode === '3d' ? 'CUBE' : mode}</div>
                        <div class="rating-box-config" style="font-weight: 700;">${board} | ${formatTimeShort(time)}</div>
                        <div style="display: flex; flex-direction: column; gap: 2px; margin-top: 4px; font-size: 0.65rem; color: rgba(255,255,255,0.3); font-weight: 700;">
                           <div>Played: <span style="color: #fff;">${configData.games_played || 0}</span> | Wins: <span style="color: #fff;">${configData.wins || 0}</span></div>
                           <div>Points: <span style="color: #fff;">${configData.point_sum || 0}</span> | Avg Found: <span style="color: #fff;">${configData.avg_pct_found || 0}%</span></div>
                        </div>
                    </div>
                    <div class="rating-box-value" style="color: ${rColor}; font-size: 1.25rem; font-weight: 900; margin: 0 15px;">${rating}</div>
                `;

                box.onclick = () => {
                    if (u && u.username) {
                        showRoomAchievements(u.username, mode, board, time);
                    }
                };

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
        const selects = ['rankings-filter-mode', 'rankings-filter-dims', 'rankings-filter-time'];
        selects.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.onchange = () => renderRatingsGrid();
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
            achModal.style.display = 'none';
            achModal.style.opacity = '0';
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
    if (!modal) return;

    // Track state for period switching
    currentAchConfig = { username, mode, board, time };

    // Capture Scroll Position to prevent jumping to top on filter change
    const card = modal.querySelector('.achievement-card');
    let previousScroll = 0;
    if (!modal.classList.contains('hidden') && card) {
        previousScroll = card.scrollTop;
    }

    // Update tab UI
    const tabs = document.querySelectorAll('.modal-tabs .ach-tab');
    tabs.forEach(tab => {
        if (tab.dataset.period === period) tab.classList.add('active');
        else tab.classList.remove('active');
    });

    // Set titles
    document.getElementById('achievement-title').textContent = `${username}'s Achievements`;
    document.getElementById('achievement-subtitle').textContent =
        `${mode.charAt(0).toUpperCase() + mode.slice(1)} ${board} | ${time < 300 ? time + 's' : (time / 60) + 'm'}`;


    // Show loading state
    modal.classList.remove('hidden');
    modal.style.display = 'flex';
    modal.style.opacity = '1';

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
                onclick="watchRoundHistory('${r.room_id}', ${r.round_number}, true, ${r.game_id || 'null'}); document.getElementById('room-achievements-modal').classList.add('hidden');">
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
                    onclick="watchRoundHistory('${w.room_id}', ${w.round_number}, true, ${w.game_id || 'null'}); document.getElementById('room-achievements-modal').classList.add('hidden');">
                    <td style="padding: 10px 15px; font-weight: 800; color: #fff; text-transform: uppercase;">${w.word}</td>
                    <td style="padding: 10px 15px; font-weight: 700; color: #ffd700;">${w.points}</td>
                    <td style="padding: 10px 15px; color: rgba(255,255,255,0.5);">${w.word.length}</td>
                    <td style="padding: 10px 15px; font-size: 0.75rem; color: rgba(255,255,255,0.4);">${dateToShort(date)}</td>
                    <td style="padding: 10px 15px; text-align: right;"><div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 4px 8px; display: inline-block;">📷</div></td>
                </tr>`;
            }).join('');
        }

        function dateToShort(d) {
            return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: '2-digit' });
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

            newEl.addEventListener('blur', () => saveProfileField(field.key, newEl.innerText.trim()));
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
const WORDS_PAGE_SIZE = 2000; // Render in slightly larger chunks for speed
let currentProgressiveLoadId = 0;
let listsFetchAbortController = null;
let listsFetchTimeoutId = null; // Module-level so it can be cancelled on re-fetch

function startProgressiveRendering() {
    const loadId = ++currentProgressiveLoadId;
    
    function renderChunk() {
        if (loadId !== currentProgressiveLoadId) return;
        if (currentWordsRenderedCount >= currentWordsList.length) {
            console.log(`[Lists] Finished progressive rendering of ${currentWordsList.length} words.`);
            return;
        }
        
        renderNextWordsPage();
        setTimeout(renderChunk, 5);
    }
    
    renderChunk();
}

function renderNextWordsPage() {
    const scrollArea = document.getElementById('main-list-results');
    if (!scrollArea || currentWordsRenderedCount >= currentWordsList.length) return;

    const nextPageWords = currentWordsList.slice(
        currentWordsRenderedCount,
        currentWordsRenderedCount + WORDS_PAGE_SIZE
    );

    let html = '';
    if (currentWordsType === 'likelihood') {
        html = nextPageWords.map(item => `
            <div class="list-item">
                <span class="likelihood-score">${item.score}</span> ${item.word}
            </div>
        `).join('');
    } else if (currentWordsType === 'added') {
        const isMod = window.currentUserIsMod;
        html = nextPageWords.map(w => `
            <div class="list-item added-word" oncopy="return false;" oncut="return false;" oncontextmenu="return false;" ondragstart="return false;" style="display: flex; justify-content: space-between; align-items: center; -webkit-user-select: none; -moz-user-select: none; -ms-user-select: none; user-select: none;">
                <span style="-webkit-user-select: none; -moz-user-select: none; -ms-user-select: none; user-select: none;">${w}</span>
                ${isMod ? `<button onclick="removeAddedWordFromTools('${w}')" style="background:none; border:none; color:#f43f5e; cursor:pointer; font-weight:bold; padding:0 5px;" title="Remove">&times;</button>` : ''}
            </div>
        `).join('');
    } else {
        html = nextPageWords.map(w => `<div class="list-item">${w}</div>`).join('');
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

    const scrollArea = document.getElementById('main-list-results');
    if (scrollArea) {
        scrollArea.addEventListener('scroll', () => {
            if (scrollArea.scrollTop + scrollArea.clientHeight >= scrollArea.scrollHeight - 200) {
                renderNextWordsPage();
            }
        });

        // Prevent copying and context menu actions on the Added Words list
        scrollArea.addEventListener('copy', (e) => {
            if (currentWordsType === 'added') {
                e.preventDefault();
            }
        });
        scrollArea.addEventListener('cut', (e) => {
            if (currentWordsType === 'added') {
                e.preventDefault();
            }
        });
        scrollArea.addEventListener('contextmenu', (e) => {
            if (currentWordsType === 'added') {
                e.preventDefault();
            }
        });
        scrollArea.addEventListener('selectstart', (e) => {
            if (currentWordsType === 'added') {
                e.preventDefault();
            }
        });
    }

    // Initialize custom draggable scrollbar
    initCustomScrollbar();
}

function initCustomScrollbar() {
    const scrollArea = document.getElementById('main-list-results');
    const track = document.getElementById('list-scrollbar-track');
    const thumb = document.getElementById('list-scrollbar-thumb');
    if (!scrollArea || !track || !thumb) return;

    function updateThumb() {
        const scrollHeight = scrollArea.scrollHeight;
        const clientHeight = scrollArea.clientHeight;
        const scrollTop = scrollArea.scrollTop;

        // Show scrollbar only if list content overflows
        if (scrollHeight <= clientHeight + 5) {
            track.style.display = 'none';
            return;
        }
        track.style.display = 'block';

        const ratio = clientHeight / scrollHeight;
        const thumbHeight = Math.max(40, clientHeight * ratio);
        thumb.style.height = `${thumbHeight}px`;

        const maxScrollTop = scrollHeight - clientHeight;
        const maxThumbTop = clientHeight - thumbHeight;
        const thumbTop = (scrollTop / maxScrollTop) * maxThumbTop;
        thumb.style.top = `${thumbTop}px`;
    }

    // Bind event listeners for scroll and resize
    scrollArea.addEventListener('scroll', updateThumb);
    window.addEventListener('resize', updateThumb);

    // Watch for dynamic content changes inside the scroll area to auto-update
    const observer = new MutationObserver(updateThumb);
    observer.observe(scrollArea, { childList: true, subtree: true });

    // Dragging state tracking
    let isDragging = false;
    let startY = 0;
    let startThumbTop = 0;

    function onDragStart(e) {
        isDragging = true;
        thumb.classList.add('dragging');
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        startY = clientY;
        startThumbTop = parseFloat(thumb.style.top) || 0;
        document.body.style.userSelect = 'none';
        
        if (e.cancelable !== false) {
            e.preventDefault();
        }
    }

    function onDragMove(e) {
        if (!isDragging) return;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const deltaY = clientY - startY;

        const clientHeight = scrollArea.clientHeight;
        const thumbHeight = thumb.offsetHeight;
        const maxThumbTop = clientHeight - thumbHeight;

        let newThumbTop = startThumbTop + deltaY;
        newThumbTop = Math.max(0, Math.min(maxThumbTop, newThumbTop));

        thumb.style.top = `${newThumbTop}px`;

        const scrollHeight = scrollArea.scrollHeight;
        const maxScrollTop = scrollHeight - clientHeight;
        scrollArea.scrollTop = (newThumbTop / maxThumbTop) * maxScrollTop;

        if (e.cancelable !== false) {
            e.preventDefault();
        }
    }

    function onDragEnd() {
        if (isDragging) {
            isDragging = false;
            thumb.classList.remove('dragging');
            document.body.style.userSelect = '';
        }
    }

    // Mouse events for thumb
    thumb.addEventListener('mousedown', onDragStart);
    document.addEventListener('mousemove', onDragMove);
    document.addEventListener('mouseup', onDragEnd);

    // Touch events for thumb
    thumb.addEventListener('touchstart', onDragStart, { passive: false });
    document.addEventListener('touchmove', onDragMove, { passive: false });
    document.addEventListener('touchend', onDragEnd);

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
        scrollArea.scrollTop = (newThumbTop / maxThumbTop) * maxScrollTop;

        // Directly start drag session
        isDragging = true;
        thumb.classList.add('dragging');
        startY = e.clientY;
        startThumbTop = newThumbTop;
        e.preventDefault();
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
        scrollArea.scrollTop = (newThumbTop / maxThumbTop) * maxScrollTop;

        isDragging = true;
        thumb.classList.add('dragging');
        startY = e.touches[0].clientY;
        startThumbTop = newThumbTop;
        if (e.cancelable !== false) {
            e.preventDefault();
        }
    }, { passive: false });

    // Initial position trigger
    updateThumb();
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
            <div style="flex: 1; overflow-y: auto; padding: 10px;">
                <table class="group-table" style="width: 100%;">
                    <tbody>
        `;

        // Clickable words for Sequence search
        html += words.map(w => `
            <tr><td style="padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span class="clickable-word-link" onclick="window.lookupWord('${w}')" style="font-family: monospace;">${w}</span>
            </td></tr>
        `).join('');

        html += `
                    </tbody>
                </table>
            </div>
        `;

        resultsContainer.innerHTML = html;

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
        const response = await fetch('/api/tools/wotd');
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
            <div style="flex: 1; overflow-y: auto; padding: 10px;">
                <table class="group-table" style="width: 100%;">
                    <tbody>
        `;

        // Clickable words for Subanagram search
        html += words.map(w => `
            <tr><td style="padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span class="clickable-word-link" onclick="window.lookupWord('${w}')" style="font-family: monospace;">${w}</span>
            </td></tr>
        `).join('');

        html += `
                    </tbody>
                </table>
            </div>
        `;

        resultsContainer.innerHTML = html;

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
            if (data.is_valid && (data.definition || data.pronunciation)) {
                let html = '';
                if (data.pronunciation) {
                    html += `<div class="pronunciation" style="margin-bottom: 10px; font-size: 1.8rem; letter-spacing: 2px;">${data.pronunciation}</div>`;
                }
                if (data.definition) {
                    html += `<div class="definition-text" style="font-size: 1.3rem; line-height: 1.6; color: #fff; font-style: normal;">${data.definition}</div>`;
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
    currentChatTarget = username;

    if (clearHistory) {
        // Aggressively clear messages before loading to ensure a fresh session
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

    document.getElementById('pm-target-name').innerText = username;
    document.getElementById('private-chat-modal').classList.remove('hidden');

    // Update synchronized state to reflect we've interacted with this
    const pmState = getPMState();
    // We don't know the exact count yet, but we've seen the "latest" notification for this person
    // Incrementing or just setting context to something that won't trigger a re-notify
    pmState.lastNotifiedContext = `OPEN:${username}`;
    pmState.activeChat = username;
    setPMState(pmState);

    await refreshConversation();
    startPMPolling();

    // Auto-focus input
    document.getElementById('pm-input').focus();
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

        if (data.messages && data.messages.length > 0) {
            renderPMHistory(data.messages);

            // Update high-water mark for notifications
            const latest = data.messages[data.messages.length - 1];
            if (latest && latest.timestamp) {
                const pmState = getPMState();
                if (!pmState.lastTimestamp || latest.timestamp > pmState.lastTimestamp) {
                    pmState.lastTimestamp = latest.timestamp;
                    setPMState(pmState);
                }
            }
        }
    } catch (err) {
        console.error("Failed to fetch conversation:", err);
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
    history: [],
    nextData: null
};

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
            if (!unscrambleState.jumbled) startNewUnscramble();
            
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

    // Trigger initial load if empty
    setTimeout(() => {
        const display = document.getElementById('unscramble-jumbled');
        if (display && (!display.innerText || display.innerText === "Loading...")) {
            startNewUnscramble();
        }
    }, 500);
}

async function startNewUnscramble(keepFound = false) {
    if (unscrambleNextTimeout) {
        clearTimeout(unscrambleNextTimeout);
        unscrambleNextTimeout = null;
    }

    if (!keepFound && unscrambleState.jumbled) {
        // Save CURRENT round to history
        unscrambleState.history.unshift({
            jumbled: unscrambleState.jumbled,
            found: [...unscrambleState.found],
            solutions: Array.from(unscrambleState.solution),
            timestamp: new Date().toLocaleTimeString()
        });
        if (unscrambleState.history.length > 50) unscrambleState.history.pop();
    }

    unscrambleState.isWaiting = false;

    const lenInput = document.getElementById('unscramble-length');
    const dictInput = document.getElementById('unscramble-dict');
    const mustInput = document.getElementById('unscramble-must-have');
    const len = lenInput ? lenInput.value : 5;
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
        revealBtn.innerText = "Unscramble";
        revealBtn.disabled = false;
        revealBtn.style.background = '';
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
            console.log("Using prefetched unscramble data");
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
    }
}

async function prefetchUnscramble() {
    const lenInput = document.getElementById('unscramble-length');
    const dictInput = document.getElementById('unscramble-dict');
    const mustInput = document.getElementById('unscramble-must-have');
    const len = lenInput ? lenInput.value : 5;
    const dict = dictInput ? dictInput.value : 'NWL';
    const must = mustInput ? mustInput.value.trim().toUpperCase() : '';

    try {
        const resp = await fetch(`/api/tools/unscramble/random?length=${len}&dictionary=${dict}&must_have=${encodeURIComponent(must)}`);
        const data = await resp.json();
        if (!data.error) {
            unscrambleState.nextData = { data, len, dict, must };
            console.log("Next unscramble prefetched");
        }
    } catch (e) { }
}

function renderUnscrambleHistory() {
    // This is now integrated into renderUnscrambleFound
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
                input.disabled = true;
                // Faster auto-advance (0.8s instead of 1.5s)
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
    input.focus();
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

    // 1. Show all solutions (Found = Green, Missed = Red)
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

    // 3. Start visible but shorter countdown (2s instead of 4s)
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
    if (!list) return;

    let html = '';

    // 1. CURRENT ROUND SECTION
    if (unscrambleState.jumbled) {
        const solutions = Array.from(unscrambleState.solution).sort();

        html += `<div style="width: 100%; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 10px; margin-bottom: 15px; display: flex; flex-direction: column; gap: 10px;">
                    <div style="font-size: 0.7rem; text-transform: uppercase; color: #ffd700; letter-spacing: 2px; font-weight: 800;">Active: ${unscrambleState.jumbled.toUpperCase()}</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">`;

        solutions.forEach(w => {
            const isFound = unscrambleState.found.includes(w);
            let style = "background: rgba(255, 255, 255, 0.05); color: rgba(255, 255, 255, 0.15); border: 1px solid rgba(255, 255, 255, 0.05);";
            let displayWord = w.replace(/./g, '_');

            if (isFound) {
                style = "background: rgba(76, 175, 80, 0.2); border: 1px solid #4caf50; color: #81c784;";
                displayWord = w;
            } else if (revealMissed) {
                style = "background: rgba(244, 67, 54, 0.2); border: 1px solid #f44336; color: #e57373;";
                displayWord = w;
            }

            html += `<div style="${style} padding: 6px 14px; border-radius: 6px; font-weight: 700; font-size: 0.95rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.2s ease;">${displayWord}</div>`;
        });

        // Incorrect Guesses for current round
        unscrambleState.incorrect.forEach(w => {
            html += `<div style="background: rgba(0, 0, 0, 0.2); color: #ff5252; padding: 6px 14px; border-radius: 6px; font-weight: 600; border: 1px dotted rgba(255, 82, 82, 0.3); font-size: 0.9rem; text-decoration: line-through; opacity: 0.7;">${w}</div>`;
        });

        html += `   </div>
                </div>`;
    }

    // 2. HISTORY SECTION
    if (unscrambleState.history.length > 0) {
        html += `<div style="width: 100%; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 20px;">
                    <div style="font-size: 0.7rem; text-transform: uppercase; color: rgba(255,255,255,0.3); letter-spacing: 2px; font-weight: 800; text-align: center; margin-bottom: 15px;">Session History</div>
                    <div style="display: flex; flex-direction: column; gap: 12px; max-height: 260px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.2) transparent; padding-right: 4px;">`;

        unscrambleState.history.forEach((h, idx) => {
            const foundCount = h.found.length;
            const totalCount = h.solutions.length;
            const isPerfect = foundCount === totalCount;

            html += `
                <div style="background: rgba(255,255,255,0.03); border-radius: 10px; padding: 12px 18px; border: 1px solid rgba(255,255,255,0.05); display: flex; flex-direction: column; gap: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 800; color: #ffd700; font-size: 1rem; letter-spacing: 1px;">${h.jumbled.toUpperCase()}</span>
                        <span style="font-size: 0.7rem; color: rgba(255,255,255,0.3); font-family: monospace;">${h.timestamp}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; gap: 15px;">
                        <span style="font-size: 0.75rem; color: ${isPerfect ? '#81c784' : 'rgba(255,255,255,0.5)'}; font-weight: 700; white-space: nowrap;">
                            ${foundCount}/${totalCount} Words
                        </span>
                        <div style="display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end;">
                            ${h.solutions.map(s => {
                const wereFound = h.found.includes(s);
                const color = wereFound ? '#81c784' : 'rgba(255,255,255,0.2)';
                const bg = wereFound ? 'rgba(76, 175, 80, 0.1)' : 'rgba(255,255,255,0.02)';
                return `<span style="font-size: 0.7rem; background: ${bg}; padding: 3px 8px; border-radius: 4px; color: ${color}; border: 1px solid ${wereFound ? 'rgba(76, 175, 80, 0.2)' : 'transparent'}">${s}</span>`;
            }).join('')}
                        </div>
                    </div>
                </div>
            `;
        });

        html += `   </div>
                </div>`;
    }

    list.innerHTML = html;
}

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
        moreBtn.addEventListener('click', () => {
            loadRandomSuggestedWords();
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
}

async function loadRandomSuggestedWords() {
    const tableBody = document.getElementById('random-words-table-body');
    if (!tableBody) return;

    tableBody.innerHTML = `
        <tr>
            <td style="padding: 15px; opacity: 0.6;">
                <div class="loading-spinner" style="margin: 0 auto; width: 20px; height: 20px; border-width: 2px;"></div>
            </td>
        </tr>
    `;

    try {
        const response = await fetch('/api/tools/random-words');
        const data = await response.json();
        
        if (data.error) {
            tableBody.innerHTML = `
                <tr>
                    <td style="padding: 15px; color: #ff6b6b;">Error: ${data.error}</td>
                </tr>
            `;
            return;
        }

        if (data.words && data.words.length > 0) {
            tableBody.innerHTML = data.words.map(word => `
                <tr class="suggested-word-row" data-word="${word}" style="cursor: pointer; border-bottom: 1px solid rgba(var(--text-primary-rgb), 0.05); transition: background 0.2s;">
                    <td style="padding: 10px; color: var(--accent-color); font-weight: 500;">${word}</td>
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
        } else {
            tableBody.innerHTML = `
                <tr>
                    <td style="padding: 15px; opacity: 0.6;">No random words available.</td>
                </tr>
            `;
        }
    } catch (err) {
        console.error('Failed to load random words:', err);
        tableBody.innerHTML = `
            <tr>
                <td style="padding: 15px; color: #ff6b6b;">Failed to load words.</td>
            </tr>
        `;
    }
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
                summaryEl.innerText = `The word "${data.word}" has been found ${data.count} ${data.count === 1 ? 'time' : 'times'} since Morpheme began.`;
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
                const date = window.parseUTCTimestamp(item.timestamp);
                const formattedDate = date.toLocaleDateString(undefined, {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                });
                
                const flagHtml = window.getFlagHtml ? window.getFlagHtml(item.country_flag) : (item.country_flag || '');
                return `
                    <tr class="finder-row" data-username="${item.username}" style="cursor: pointer; border-bottom: 1px solid rgba(var(--text-primary-rgb), 0.05); transition: background 0.2s;">
                        <td style="padding: 12px; color: var(--accent-color); font-weight: 500; display: flex; align-items: center; gap: 8px;">
                            ${flagHtml} <span>${item.username}</span>
                        </td>
                        <td style="padding: 12px; color: var(--muted-text);">${formattedDate}</td>
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
// Global function to lookup word in Tool Validator
window.lookupWord = function (word) {
    if (!word) return;

    // 1. Switch tool navigation to "Is Valid"
    const navBtns = document.querySelectorAll('.tool-nav-btn');
    const isValidBtn = Array.from(navBtns).find(b => b.dataset.tool === 'is-valid');
    if (isValidBtn) isValidBtn.click();

    // 2. Set input and run check
    const input = document.getElementById('valid-input');
    if (input) {
        input.value = word;
        runValidationCheck();
    }

    // 3. Scroll to results if needed
    const container = document.getElementById('valid-results-container');
    if (container) container.scrollIntoView({ behavior: 'smooth', block: 'center' });
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
