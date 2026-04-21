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
    setupPersonalTimer();
});

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
};

function setupToolsNavigation() {
    const navBtns = document.querySelectorAll('.tool-nav-btn');
    const panes = document.querySelectorAll('.tool-pane');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update buttons
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Show pane
            const toolId = btn.dataset.tool; // e.g. "combo"
            panes.forEach(p => p.classList.remove('active'));
            const targetPane = document.getElementById(`tool-${toolId}`);
            if (targetPane) targetPane.classList.add('active');

            // Trigger fetch for Lists if selected (lazy load)
            if (toolId === 'profile') {
                refreshProfileTool();
            }
            if (toolId === 'lists') {
                fetchListsData();
            }
            if (toolId === 'manual') {
                fetch('/api/tools/flag_manual', { method: 'POST' });
            }

            if (toolId === 'wotd') {
                updateWotd();
            }
        });
    });
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
                <table class="group-table">
                    <tbody>
                        ${words.map(w => `<tr><td>${w}</td></tr>`).join('')}
                    </tbody>
                </table>
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
            const rate = games > 0 ? ((wins / games) * 100).toFixed(1) : '0.0';
            winRateEl.innerText = `${rate}%`;
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
        if (flagEl) flagEl.innerText = data.country_flag || '🏳️';

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
    const avatarTrigger = document.getElementById('profile-avatar-trigger');
    const avatarInput = document.getElementById('profile-avatar-input');

    if (avatarTrigger && avatarInput) {
        avatarTrigger.addEventListener('click', () => {
            const displayedName = document.getElementById('profile-username').innerText;
            const globalUser = window.currentUser || currentUser;
            const currentName = (typeof globalUser === 'object') ? globalUser.username : globalUser;

            // Only allow if it matches current user (case-insensitive)
            if (currentName && currentName.toLowerCase() === displayedName.toLowerCase()) {
                avatarInput.click();
            }
        });

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
                    setTimeout(() => flagDropdownSearch.focus(), 50);
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
}

// Full Country Flag List (ISO 3166-1)
const ALL_FLAGS = [
    { code: 'AF', flag: '🇦🇫', name: 'Afghanistan' },
    { code: 'AL', flag: '🇦🇱', name: 'Albania' },
    { code: 'DZ', flag: '🇩🇿', name: 'Algeria' },
    { code: 'AS', flag: '🇦🇸', name: 'American Samoa' },
    { code: 'AD', flag: '🇦🇩', name: 'Andorra' },
    { code: 'AO', flag: '🇦🇴', name: 'Angola' },
    { code: 'AI', flag: '🇦🇮', name: 'Anguilla' },
    { code: 'AQ', flag: '🇦🇶', name: 'Antarctica' },
    { code: 'AG', flag: '🇦🇬', name: 'Antigua and Barbuda' },
    { code: 'AR', flag: '🇦🇷', name: 'Argentina' },
    { code: 'AM', flag: '🇦🇲', name: 'Armenia' },
    { code: 'AW', flag: '🇦🇼', name: 'Aruba' },
    { code: 'AU', flag: '🇦🇺', name: 'Australia' },
    { code: 'AT', flag: '🇦🇹', name: 'Austria' },
    { code: 'AZ', flag: '🇦🇿', name: 'Azerbaijan' },
    { code: 'BS', flag: '🇧🇸', name: 'Bahamas' },
    { code: 'BH', flag: '🇧🇭', name: 'Bahrain' },
    { code: 'BD', flag: '🇧🇩', name: 'Bangladesh' },
    { code: 'BB', flag: '🇧🇧', name: 'Barbados' },
    { code: 'BY', flag: '🇧🇾', name: 'Belarus' },
    { code: 'BE', flag: '🇧🇪', name: 'Belgium' },
    { code: 'BZ', flag: '🇧🇿', name: 'Belize' },
    { code: 'BJ', flag: '🇧🇯', name: 'Benin' },
    { code: 'BM', flag: '🇧🇲', name: 'Bermuda' },
    { code: 'BT', flag: '🇧🇹', name: 'Bhutan' },
    { code: 'BO', flag: '🇧🇴', name: 'Bolivia' },
    { code: 'BQ', flag: '🇧🇶', name: 'Bonaire, Sint Eustatius and Saba' },
    { code: 'BA', flag: '🇧🇦', name: 'Bosnia and Herzegovina' },
    { code: 'BW', flag: '🇧🇼', name: 'Botswana' },
    { code: 'BV', flag: '🇧🇻', name: 'Bouvet Island' },
    { code: 'BR', flag: '🇧🇷', name: 'Brazil' },
    { code: 'IO', flag: '🇮🇴', name: 'British Indian Ocean Territory' },
    { code: 'BN', flag: '🇧🇳', name: 'Brunei Darussalam' },
    { code: 'BG', flag: '🇧🇬', name: 'Bulgaria' },
    { code: 'BF', flag: '🇧🇫', name: 'Burkina Faso' },
    { code: 'BI', flag: '🇧🇮', name: 'Burundi' },
    { code: 'CV', flag: '🇨🇻', name: 'Cabo Verde' },
    { code: 'KH', flag: '🇰🇭', name: 'Cambodia' },
    { code: 'CM', flag: '🇨🇲', name: 'Cameroon' },
    { code: 'CA', flag: '🇨🇦', name: 'Canada' },
    { code: 'KY', flag: '🇰🇾', name: 'Cayman Islands' },
    { code: 'CF', flag: '🇨🇫', name: 'Central African Republic' },
    { code: 'TD', flag: '🇹🇩', name: 'Chad' },
    { code: 'CL', flag: '🇨🇱', name: 'Chile' },
    { code: 'CN', flag: '🇨🇳', name: 'China' },
    { code: 'CX', flag: '🇨🇽', name: 'Christmas Island' },
    { code: 'CC', flag: '🇨🇨', name: 'Cocos (Keeling) Islands' },
    { code: 'CO', flag: '🇨🇴', name: 'Colombia' },
    { code: 'KM', flag: '🇰🇲', name: 'Comoros' },
    { code: 'CD', flag: '🇨🇩', name: 'Congo (DRC)' },
    { code: 'CG', flag: '🇨🇬', name: 'Congo (Republic)' },
    { code: 'CK', flag: '🇨🇰', name: 'Cook Islands' },
    { code: 'CR', flag: '🇨🇷', name: 'Costa Rica' },
    { code: 'HR', flag: '🇭🇷', name: 'Croatia' },
    { code: 'CU', flag: '🇨🇺', name: 'Cuba' },
    { code: 'CW', flag: '🇨🇼', name: 'Curaçao' },
    { code: 'CY', flag: '🇨🇾', name: 'Cyprus' },
    { code: 'CZ', flag: '🇨🇿', name: 'Czech Republic' },
    { code: 'CI', flag: '🇨🇮', name: 'Côte d\'Ivoire' },
    { code: 'DK', flag: '🇩🇰', name: 'Denmark' },
    { code: 'DJ', flag: '🇩🇯', name: 'Djibouti' },
    { code: 'DM', flag: '🇩🇲', name: 'Dominica' },
    { code: 'DO', flag: '🇩🇴', name: 'Dominican Republic' },
    { code: 'EC', flag: '🇪🇨', name: 'Ecuador' },
    { code: 'EG', flag: '🇪🇬', name: 'Egypt' },
    { code: 'SV', flag: '🇸🇻', name: 'El Salvador' },
    { code: 'GQ', flag: '🇬🇶', name: 'Equatorial Guinea' },
    { code: 'ER', flag: '🇪🇷', name: 'Eritrea' },
    { code: 'EE', flag: '🇪🇪', name: 'Estonia' },
    { code: 'SZ', flag: '🇸🇿', name: 'Eswatini' },
    { code: 'ET', flag: '🇪🇹', name: 'Ethiopia' },
    { code: 'FK', flag: '🇫🇰', name: 'Falkland Islands' },
    { code: 'FO', flag: '🇫🇴', name: 'Faroe Islands' },
    { code: 'FJ', flag: '🇫🇯', name: 'Fiji' },
    { code: 'FI', flag: '🇫🇮', name: 'Finland' },
    { code: 'FR', flag: '🇫🇷', name: 'France' },
    { code: 'GF', flag: '🇬🇫', name: 'French Guiana' },
    { code: 'PF', flag: '🇵🇫', name: 'French Polynesia' },
    { code: 'TF', flag: '🇹🇫', name: 'French Southern Territories' },
    { code: 'GA', flag: '🇬🇦', name: 'Gabon' },
    { code: 'GM', flag: '🇬🇲', name: 'Gambia' },
    { code: 'GE', flag: '🇬🇪', name: 'Georgia' },
    { code: 'DE', flag: '🇩🇪', name: 'Germany' },
    { code: 'GH', flag: '🇬🇭', name: 'Ghana' },
    { code: 'GI', flag: '🇬🇮', name: 'Gibraltar' },
    { code: 'GR', flag: '🇬🇷', name: 'Greece' },
    { code: 'GL', flag: '🇬🇱', name: 'Greenland' },
    { code: 'GD', flag: '🇬🇩', name: 'Grenada' },
    { code: 'GP', flag: '🇬🇵', name: 'Guadeloupe' },
    { code: 'GU', flag: '🇬🇺', name: 'Guam' },
    { code: 'GT', flag: '🇬🇹', name: 'Guatemala' },
    { code: 'GG', flag: '🇬🇬', name: 'Guernsey' },
    { code: 'GN', flag: '🇬🇳', name: 'Guinea' },
    { code: 'GW', flag: '🇬🇼', name: 'Guinea-Bissau' },
    { code: 'GY', flag: '🇬🇾', name: 'Guyana' },
    { code: 'HT', flag: '🇭🇹', name: 'Haiti' },
    { code: 'HM', flag: '🇭🇲', name: 'Heard Island and McDonald Islands' },
    { code: 'VA', flag: '🇻🇦', name: 'Holy See' },
    { code: 'HN', flag: '🇭🇳', name: 'Honduras' },
    { code: 'HK', flag: '🇭🇰', name: 'Hong Kong' },
    { code: 'HU', flag: '🇭🇺', name: 'Hungary' },
    { code: 'IS', flag: '🇮🇸', name: 'Iceland' },
    { code: 'IN', flag: '🇮🇳', name: 'India' },
    { code: 'ID', flag: '🇮🇩', name: 'Indonesia' },
    { code: 'IR', flag: '🇮🇷', name: 'Iran' },
    { code: 'IQ', flag: '🇮🇶', name: 'Iraq' },
    { code: 'IE', flag: '🇮🇪', name: 'Ireland' },
    { code: 'IM', flag: '🇮🇲', name: 'Isle of Man' },
    { code: 'IL', flag: '🇮🇱', name: 'Israel' },
    { code: 'IT', flag: '🇮🇹', name: 'Italy' },
    { code: 'JM', flag: '🇯🇲', name: 'Jamaica' },
    { code: 'JP', flag: '🇯🇵', name: 'Japan' },
    { code: 'JE', flag: '🇯🇪', name: 'Jersey' },
    { code: 'JO', flag: '🇯🇴', name: 'Jordan' },
    { code: 'KZ', flag: '🇰🇿', name: 'Kazakhstan' },
    { code: 'KE', flag: '🇰🇪', name: 'Kenya' },
    { code: 'KI', flag: '🇰🇮', name: 'Kiribati' },
    { code: 'KP', flag: '🇰🇵', name: 'North Korea' },
    { code: 'KR', flag: '🇰🇷', name: 'South Korea' },
    { code: 'KW', flag: '🇰🇼', name: 'Kuwait' },
    { code: 'KG', flag: '🇰🇬', name: 'Kyrgyzstan' },
    { code: 'LA', flag: '🇱🇦', name: 'Lao People\'s Democratic Republic' },
    { code: 'LV', flag: '🇱🇻', name: 'Latvia' },
    { code: 'LB', flag: '🇱🇧', name: 'Lebanon' },
    { code: 'LS', flag: '🇱🇸', name: 'Lesotho' },
    { code: 'LR', flag: '🇱🇷', name: 'Liberia' },
    { code: 'LY', flag: '🇱🇾', name: 'Libya' },
    { code: 'LI', flag: '🇱🇮', name: 'Liechtenstein' },
    { code: 'LT', flag: '🇱🇹', name: 'Lithuania' },
    { code: 'LU', flag: '🇱🇺', name: 'Luxembourg' },
    { code: 'MO', flag: '🇲🇴', name: 'Macao' },
    { code: 'MG', flag: '🇲🇬', name: 'Madagascar' },
    { code: 'MW', flag: '🇲🇼', name: 'Malawi' },
    { code: 'MY', flag: '🇲🇾', name: 'Malaysia' },
    { code: 'MV', flag: '🇲🇻', name: 'Maldives' },
    { code: 'ML', flag: '🇲🇱', name: 'Mali' },
    { code: 'MT', flag: '🇲🇹', name: 'Malta' },
    { code: 'MH', flag: '🇲🇭', name: 'Marshall Islands' },
    { code: 'MQ', flag: '🇲🇶', name: 'Martinique' },
    { code: 'MR', flag: '🇲🇷', name: 'Mauritania' },
    { code: 'MU', flag: '🇲🇺', name: 'Mauritius' },
    { code: 'YT', flag: '🇾🇹', name: 'Mayotte' },
    { code: 'MX', flag: '🇲🇽', name: 'Mexico' },
    { code: 'FM', flag: '🇫🇲', name: 'Micronesia' },
    { code: 'MD', flag: '🇲🇩', name: 'Moldova' },
    { code: 'MC', flag: '🇲🇨', name: 'Monaco' },
    { code: 'MN', flag: '🇲🇳', name: 'Mongolia' },
    { code: 'ME', flag: '🇲🇪', name: 'Montenegro' },
    { code: 'MS', flag: '🇲🇸', name: 'Montserrat' },
    { code: 'MA', flag: '🇲🇦', name: 'Morocco' },
    { code: 'MZ', flag: '🇲🇿', name: 'Mozambique' },
    { code: 'MM', flag: '🇲🇲', name: 'Myanmar' },
    { code: 'NA', flag: '🇳🇦', name: 'Namibia' },
    { code: 'NR', flag: '🇳🇷', name: 'Nauru' },
    { code: 'NP', flag: '🇳🇵', name: 'Nepal' },
    { code: 'NL', flag: '🇳🇱', name: 'Netherlands' },
    { code: 'NC', flag: '🇳🇨', name: 'New Caledonia' },
    { code: 'NZ', flag: '🇳🇿', name: 'New Zealand' },
    { code: 'NI', flag: '🇳🇮', name: 'Nicaragua' },
    { code: 'NE', flag: '🇳🇪', name: 'Niger' },
    { code: 'NG', flag: '🇳🇬', name: 'Nigeria' },
    { code: 'NU', flag: '🇳🇺', name: 'Niue' },
    { code: 'NF', flag: '🇳🇫', name: 'Norfolk Island' },
    { code: 'MK', flag: '🇲🇰', name: 'North Macedonia' },
    { code: 'MP', flag: '🇲🇵', name: 'Northern Mariana Islands' },
    { code: 'NO', flag: '🇳🇴', name: 'Norway' },
    { code: 'OM', flag: '🇴🇲', name: 'Oman' },
    { code: 'PK', flag: '🇵🇰', name: 'Pakistan' },
    { code: 'PW', flag: '🇵🇼', name: 'Palau' },
    { code: 'PS', flag: '🇵🇸', name: 'Palestine, State of' },
    { code: 'PA', flag: '🇵🇦', name: 'Panama' },
    { code: 'PG', flag: '🇵🇬', name: 'Papua New Guinea' },
    { code: 'PY', flag: '🇵🇾', name: 'Paraguay' },
    { code: 'PE', flag: '🇵🇪', name: 'Peru' },
    { code: 'PH', flag: '🇵🇭', name: 'Philippines' },
    { code: 'PN', flag: '🇵🇳', name: 'Pitcairn' },
    { code: 'PL', flag: '🇵🇱', name: 'Poland' },
    { code: 'PT', flag: '🇵🇹', name: 'Portugal' },
    { code: 'PR', flag: '🇵🇷', name: 'Puerto Rico' },
    { code: 'QA', flag: '🇶🇦', name: 'Qatar' },
    { code: 'RO', flag: '🇷🇴', name: 'Romania' },
    { code: 'RU', flag: '🇷🇺', name: 'Russia' },
    { code: 'RW', flag: '🇷🇼', name: 'Rwanda' },
    { code: 'RE', flag: '🇷🇪', name: 'Réunion' },
    { code: 'BL', flag: '🇧🇱', name: 'Saint Barthélemy' },
    { code: 'SH', flag: '🇸🇭', name: 'Saint Helena, Ascension and Tristan da Cunha' },
    { code: 'KN', flag: '🇰🇳', name: 'Saint Kitts and Nevis' },
    { code: 'LC', flag: '🇱🇨', name: 'Saint Lucia' },
    { code: 'MF', flag: '🇲🇫', name: 'Saint Martin (French part)' },
    { code: 'PM', flag: '🇵🇲', name: 'Saint Pierre and Miquelon' },
    { code: 'VC', flag: '🇻🇨', name: 'Saint Vincent and the Grenadines' },
    { code: 'WS', flag: '🇼🇸', name: 'Samoa' },
    { code: 'SM', flag: '🇸🇲', name: 'San Marino' },
    { code: 'ST', flag: '🇸🇹', name: 'Sao Tome and Principe' },
    { code: 'SA', flag: '🇸🇦', name: 'Saudi Arabia' },
    { code: 'SN', flag: '🇸🇳', name: 'Senegal' },
    { code: 'RS', flag: '🇷🇸', name: 'Serbia' },
    { code: 'SC', flag: '🇸🇨', name: 'Seychelles' },
    { code: 'SL', flag: '🇸🇱', name: 'Sierra Leone' },
    { code: 'SG', flag: '🇸🇬', name: 'Singapore' },
    { code: 'SX', flag: '🇸🇽', name: 'Sint Maarten (Dutch part)' },
    { code: 'SK', flag: '🇸🇰', name: 'Slovakia' },
    { code: 'SI', flag: '🇸🇮', name: 'Slovenia' },
    { code: 'SB', flag: '🇸🇧', name: 'Solomon Islands' },
    { code: 'SO', flag: '🇸🇴', name: 'Somalia' },
    { code: 'ZA', flag: '🇿🇦', name: 'South Africa' },
    { code: 'GS', flag: '🇬🇸', name: 'South Georgia and the South Sandwich Islands' },
    { code: 'SS', flag: '🇸🇸', name: 'South Sudan' },
    { code: 'ES', flag: '🇪🇸', name: 'Spain' },
    { code: 'LK', flag: '🇱🇰', name: 'Sri Lanka' },
    { code: 'SD', flag: '🇸🇩', name: 'Sudan' },
    { code: 'SR', flag: '🇸🇷', name: 'Suriname' },
    { code: 'SJ', flag: '🇸🇯', name: 'Svalbard and Jan Mayen' },
    { code: 'SE', flag: '🇸🇪', name: 'Sweden' },
    { code: 'CH', flag: '🇨🇭', name: 'Switzerland' },
    { code: 'SY', flag: '🇸🇾', name: 'Syrian Arab Republic' },
    { code: 'TW', flag: '🇹🇼', name: 'Taiwan' },
    { code: 'TJ', flag: '🇹🇯', name: 'Tajikistan' },
    { code: 'TZ', flag: '🇹🇿', name: 'Tanzania' },
    { code: 'TH', flag: '🇹🇭', name: 'Thailand' },
    { code: 'TL', flag: '🇹🇱', name: 'Timor-Leste' },
    { code: 'TG', flag: '🇹🇬', name: 'Togo' },
    { code: 'TK', flag: '🇹🇰', name: 'Tokelau' },
    { code: 'TO', flag: '🇹🇴', name: 'Tonga' },
    { code: 'TT', flag: '🇹🇹', name: 'Trinidad and Tobago' },
    { code: 'TN', flag: '🇹🇳', name: 'Tunisia' },
    { code: 'TR', flag: '🇹🇷', name: 'Turkey' },
    { code: 'TM', flag: '🇹🇲', name: 'Turkmenistan' },
    { code: 'TC', flag: '🇹🇨', name: 'Turks and Caicos Islands' },
    { code: 'TV', flag: '🇹🇻', name: 'Tuvalu' },
    { code: 'UG', flag: '🇺🇬', name: 'Uganda' },
    { code: 'UA', flag: '🇺🇦', name: 'Ukraine' },
    { code: 'AE', flag: '🇦🇪', name: 'United Arab Emirates' },
    { code: 'GB', flag: '🇬🇧', name: 'United Kingdom' },
    { code: 'US', flag: '🇺🇸', name: 'United States' },
    { code: 'UY', flag: '🇺🇾', name: 'Uruguay' },
    { code: 'UZ', flag: '🇺🇿', name: 'Uzbekistan' },
    { code: 'VU', flag: '🇻🇺', name: 'Vanuatu' },
    { code: 'VE', flag: '🇻🇪', name: 'Venezuela' },
    { code: 'VN', flag: '🇻🇳', name: 'Vietnam' },
    { code: 'VG', flag: '🇻🇬', name: 'Virgin Islands (British)' },
    { code: 'VI', flag: '🇻🇮', name: 'Virgin Islands (U.S.)' },
    { code: 'WF', flag: '🇼🇫', name: 'Wallis and Futuna' },
    { code: 'EH', flag: '🇪🇭', name: 'Western Sahara' },
    { code: 'YE', flag: '🇾🇪', name: 'Yemen' },
    { code: 'ZM', flag: '🇿🇲', name: 'Zambia' },
    { code: 'ZW', flag: '🇿🇼', name: 'Zimbabwe' },
    { code: 'ZZ', flag: '🏳️', name: 'None / International' }
];

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
            <span class="dropdown-item-flag">${item.flag}</span>
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
    flagEl.innerText = flag;

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

async function uploadAvatar(file) {
    const formData = new FormData();
    formData.append('avatar', file);

    const avatarEl = document.querySelector('.profile-avatar');
    const originalContent = avatarEl.innerHTML;

    // Optimistic UI
    avatarEl.innerHTML = '<span style="font-size:12px">...</span>';

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
    if (!username || !username.trim()) return;

    username = username.trim();
    const container = document.getElementById('profile-display-container');
    const input = document.getElementById('profile-search-input');
    if (input) input.value = username;

    // Guests do not have profiles
    if (username.startsWith('Guest_')) {
        if (container) container.classList.add('hidden');
        return;
    }

    // Check if we are refreshing the current user (e.g. changing tabs/periods)
    const currentDisplayed = document.getElementById('profile-username')?.innerText || '';
    const isRefresh = currentDisplayed === username;

    if (container && !isRefresh) {
        container.classList.add('hidden');
    }

    try {
        const response = await fetch(`/api/profile/${encodeURIComponent(username)}?period=${period}`);
        const data = await response.json();

        if (data.error) {
            // User not found, just don't show the profile
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
            avatar.style.cursor = 'pointer';
            avatar.onclick = () => showImageLightbox(user.avatar_url, `${user.username}'s Profile Image`);
        } else {
            avatar.style.cursor = 'default';
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
        flagEl.innerText = flagEmoji;

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

    // Check Ownership for Editing
    const globalUser = window.currentUser || currentUser;
    const currentName = (typeof globalUser === 'object') ? globalUser.username : globalUser;
    const isOwner = currentName && currentName.toLowerCase() === user.username.toLowerCase();

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
            const d = new Date(round.timestamp);
            try {
                dateStr = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
            } catch (e) {
                dateStr = '-';
            }
        }

        return `
        <div class="history-grid-item" onclick="watchRoundHistory('${round.room_id}', ${round.round_number}, true, ${round.game_id || 'null'})" style="display: grid; grid-template-columns: 80px 50px 80px 60px 80px 80px 100px 1fr 100px 50px; gap:8px; padding: 10px 15px; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.03); border-radius: 10px; margin-bottom: 8px; align-items: center; transition: all 0.2s; cursor: pointer; position: relative; overflow: hidden;">
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
            <div style="color: #ffd700; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; letter-spacing: 0.5px;" title="${round.top_word}">${round.top_word}</div>
            <div style="display: flex; flex-direction: column; gap: 1px;">
                <span style="font-size: 0.7rem; color: #60a5fa; font-weight: 700; opacity: 0.8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 140px;">${round.room_id}</span>
                <span style="font-size: 0.6rem; color: rgba(255,255,255,0.3); font-weight: 600;">Str: ${round.room_strength || '-'}</span>
            </div>
            
            <!-- Date Column -->
            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6); font-weight: 600; text-align: right;">${dateStr}</div>
        </div>
        `;
    };

    window.roundGridHeader = `
        <div class="history-grid-header" style="display: grid; grid-template-columns: 80px 50px 80px 60px 80px 80px 100px 1fr 100px 50px; gap:8px; padding: 12px 15px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 12px; font-size: 0.7rem; color: rgba(255,255,255,0.4); font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">
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
            const displayRounds = user.recent_rounds.slice(0, 10);
            historyList.innerHTML = window.roundGridHeader + displayRounds.map(r => window.renderRoundGridItem(r)).join('');
        }
    }

    if (exceptionalList) {
        if (!user.exceptional_rounds || user.exceptional_rounds.length === 0) {
            exceptionalList.innerHTML = '<p class="placeholder">No exceptional achievements recorded yet.</p>';
        } else {
            // Limits to 50 rows as requested
            const displayRounds = user.exceptional_rounds.slice(0, 50);
            exceptionalList.innerHTML = window.roundGridHeader + displayRounds.map(r => window.renderRoundGridItem(r)).join('');
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
        const letter = board[r][c].toUpperCase();
        let matchLen = 0;
        if (targetWord[index] === letter) {
            matchLen = 1;
        } else if (letter === 'Q' && targetWord.substring(index, index + 2) === 'QU') {
            matchLen = 2;
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
        if (cellValue === 'Q') {
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

    // A) Try Profile/Recent Rounds First (Preferred source for own history)
    if (gameId) {
        round = rounds.find(r => r.game_id == gameId);
    }
    if (!round) {
        // Fallback to oldheuristic
        round = rounds.find(r => r.room_id == roomId && r.round_number == roundNum);
    }

    // B) Fallback to Lobby History (If not found in profile)
    // Only use if we didn't find it in detailed history
    if (!round && window.lastGameState && window.lastGameState.winners_history && window.lastGameState.room_id === roomId) {
        const foundInLobby = window.lastGameState.winners_history.find(h => h.round == roundNum);
        if (foundInLobby && foundInLobby.board) {
            console.log(`[Review] Using Round ${roundNum} from Lobby winners_history`);
            round = {
                ...foundInLobby,
                room_id: roomId,
                round_number: foundInLobby.round,
                total_score: foundInLobby.score,
                game_type: foundInLobby.game_type || 'accumulative'
            };
        }
    }



    if (!round) {
        alert("Round details not available. This round may have happened before the snapshot system was enabled.");
        return;
    }

    // Update Date Display
    const dateEl = document.getElementById('history-review-date');
    if (dateEl) {
        if (round.timestamp) {
            const d = new Date(round.timestamp);
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
    if (walkthroughList) walkthroughList.innerHTML = '<p class="placeholder" style="color:rgba(255,255,255,0.3); text-align:center; padding:40px; font-weight:700;">Ready to watch the walkthrough...</p>';

    // 3. Render Board with Dynamic Scaling
    const boardContainer = document.getElementById(`${prefix}-board-container`);
    if (boardContainer && round.board && round.board.length > 0) {
        const rows = round.board.length;
        const cols = round.board[0].length;

        // Use a small delay to ensure modal layout is stable
        setTimeout(() => {
            const layoutMain = document.querySelector('.history-review-layout') || document.getElementById('integrated-replay-panel');
            if (!layoutMain) return;

            const availWidth = boardContainer.parentElement.clientWidth * 0.6; // Board area usually gets ~60%
            const availHeight = layoutMain.clientHeight - 80; // Minus header/padding padding

            // Calculate max cell size to fit width and height constraints
            const gap = 12;
            const maxCellW = (availWidth - (cols - 1) * gap - 50) / cols;
            const maxCellH = (availHeight - (rows - 1) * gap - 50) / rows;

            // Optimal cell size (capped for aesthetics on 4x4)
            const cellSize = Math.floor(Math.min(maxCellW, maxCellH, 120));
            const fontSize = Math.floor(cellSize * 0.6) + 'px';


            const is3D = rows === 6 && (Array.isArray(round.board[0]) && Array.isArray(round.board[0][0]) || Array.isArray(round.board[0]) && round.board[0].length === 3);
            if (is3D) {
                boardContainer.style.display = 'grid';
                boardContainer.style.gridTemplateColumns = `repeat(3, 1fr)`; 
                boardContainer.style.gap = `20px`; 
                boardContainer.style.padding = `20px`;
                boardContainer.style.background = `rgba(0,0,0,0.2)`;
                boardContainer.style.borderRadius = `15px`;

                boardContainer.innerHTML = round.board.map((face, fIdx) => {
                    let faceHTML = '';
                    for (let r = 0; r < 3; r++) {
                        for (let c = 0; c < 3; c++) {
                            const val = face[r][c] || '?';
                            faceHTML += `<div class="review-cell" style="width: 30px; height: 30px; font-size: 16px; border-radius: 4px;">${val}</div>`;
                        }
                    }
                    return `
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                            <div style="font-size: 0.6rem; color: rgba(255,255,255,0.3); font-weight: 900; text-transform: uppercase;">Face ${fIdx}</div>
                            <div style="display: grid; grid-template-columns: repeat(3, 30px); gap: 4px;">
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
    // Ensure numeric timestamps (handle legacy string dates)
    const processedWords = rawWords.map(w => ({
        ...w,
        timestamp: w.timestamp ? parseFloat(w.timestamp) : 0
    }));

    const sortedWords = processedWords.sort((a, b) => a.timestamp - b.timestamp);
    const roundDuration = round.round_duration || 60;

    // START TIME LOGIC: 
    // Preferred: round_start_time (absolute s)
    // Fallback 1: First word timestamp - 5s
    // Fallback 2: Entry timestamp (converted to s)
    let startTime = 0;
    if (round.round_start_time) {
        startTime = parseFloat(round.round_start_time);
    } else if (sortedWords.length > 0) {
        startTime = sortedWords[0].timestamp - 2.0; // Start shortly before first word
        if (startTime < 0) startTime = 0;
    } else {
        startTime = parseFloat(round.timestamp) / 1000 || (Date.now() / 1000);
    }

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
        <div class="walkthrough-item reveal">
            <span class="walkthrough-time">${timeStr}</span>
            <span class="walkthrough-word">${word.word}</span>
            <span class="${ptsClass}">${word.points} pts</span>
        </div>
        `;
    };

    const showAllWords = () => {
        // Default: Show in chronological order (Order Found)
        let displayWords = [...sortedWords];

        // USER REQUEST: For "With Friends" (Private) history, sort by length (Biggest first)
        if (roomId && String(roomId).startsWith('private_')) {
            displayWords.sort((a, b) => b.word.length - a.word.length || a.word.localeCompare(b.word));
        }

        if (walkthroughList) {
            walkthroughList.innerHTML = displayWords.map(w => renderWord(w)).join('');
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

                // Update Progress
                if (progressBar) progressBar.style.width = `${Math.min(100, (elapsed / roundDuration) * 100)}%`;
                if (currentTimeEl) {
                    const m = Math.floor(elapsed / 60);
                    const s = (elapsed % 60).toFixed(1);
                    currentTimeEl.innerText = `${m}:${s.padStart(4, '0')}`;
                }

                // Append new words in order
                while (wordIndex < sortedWords.length) {
                    const word = sortedWords[wordIndex];
                    const wTimestamp = floatTimestamp(word.timestamp);
                    const relWordTime = wTimestamp - startTime;

                    if (elapsed >= relWordTime || isNaN(relWordTime)) {
                        console.log(`[Review] Displaying word: ${word.word} (relative: ${relWordTime ? relWordTime.toFixed(1) : 'NaN'}s)`);

                        try {
                            // Insert at BOTTOM (Chronological: Order Found)
                            if (walkthroughList) {
                                walkthroughList.insertAdjacentHTML('beforeend', renderWord(word));
                                walkthroughList.scrollTop = walkthroughList.scrollHeight;
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

                if (elapsed >= roundDuration + 2) { // Add buffer
                    if (window.replayInterval) clearInterval(window.replayInterval);
                    if (skipBtn) skipBtn.classList.add('hidden');
                    if (startBtn) {
                        startBtn.classList.remove('hidden');
                        startBtn.innerText = "↺ Replay";
                    }
                    // Ensure everything is shown at end just in case
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

    // Helper for timestamp
    function floatTimestamp(ts) {
        return ts ? parseFloat(ts) : 0;
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
    const accTimes = [45, 180, 300, 600, 86400];
    const otherTimes = [45, 180, 300, 600];

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

        const times = (mode === 'accumulative' || mode === '3d') ? accTimes : otherTimes;
        boards.forEach(board => {
            if (filterDims !== 'all' && board !== filterDims) return;

            // COMPATIBILITY FILTER: 3x3x3 is for Cube ONLY; traditional boards for others
            if (mode === '3d' && board !== '3x3x3') return;
            if (mode !== '3d' && board === '3x3x3') return;

            times.forEach(time => {
                if (filterTime !== 'all' && String(time) !== filterTime) return;

                // COMPATIBILITY FILTER: No 45s or 24h for Cube
                if (mode === '3d' && (time === 45 || time === 86400)) return;

                const configKey = `${mode}|${board}|${time}`;
                const configData = ratings[configKey] || { rating: 1200, avg_score: 0, avg_perf: 0 };
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
                        <div style="display: flex; gap: 10px; margin-top: 4px; font-size: 0.65rem; color: rgba(255,255,255,0.3); font-weight: 700;">
                           <span>AVG S: <span style="color: #fff;">${configData.avg_score}</span></span>
                           <span>AVG P: <span style="color: #fff;">${configData.avg_perf}</span></span>
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
        `${mode.charAt(0).toUpperCase() + mode.slice(1)} | ${board} | ${time < 300 ? time + 's' : (time / 60) + 'm'}`;

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
            return;
        }

        const stats = data.stats;
        // Average and Period specific labels
        document.getElementById('ach-avg-perf').textContent = stats.avg_perf || '-';
        document.getElementById('ach-avg-winrate').textContent = (stats.win_rate || 0) + '%';
        document.getElementById('ach-total-games').textContent = stats.games_played || '0';
        document.getElementById('ach-avg-score').textContent = (stats.avg_score || 0).toLocaleString();
        document.getElementById('ach-avg-words').textContent = stats.avg_words || '0';
        document.getElementById('ach-avg-word-pts').textContent = stats.avg_word_pts || '0';

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
            // Sort by Ratio (Impressiveness) DESC, then Timestamp DESC
            const sortedByRatio = [...stats.exceptional_rounds].sort((a, b) => {
                if (b.ratio !== a.ratio) return b.ratio - a.ratio;
                return new Date(b.timestamp) - new Date(a.timestamp);
            });
            tablePerf.innerHTML = sortedByRatio.map(r => renderAchRow(r, [
                { val: r.performance_value, style: 'font-weight: 800; color: #60a5fa;' },
                { val: r.ratio + 'x', style: 'color: rgba(255,255,255,0.6);' },
                { val: r.total_score, style: 'font-weight: 700;' },
                { val: `<div style="font-size: 0.75rem;">${r.num_words} words</div><div style="font-size: 0.6rem; color: rgba(255,255,255,0.3);">${r.top_word}</div>` },
                { val: dateToShort(new Date(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 2. Winning Rounds
        const tableWins = document.getElementById('ach-table-wins');
        if (tableWins && stats.winning_rounds) {
            // Sort by Score DESC (Impressiveness), then Timestamp DESC
            const sortedWins = [...stats.winning_rounds].sort((a, b) => {
                if (b.total_score !== a.total_score) return b.total_score - a.total_score;
                return new Date(b.timestamp) - new Date(a.timestamp);
            });
            tableWins.innerHTML = sortedWins.map(r => renderAchRow(r, [
                { val: r.total_score, style: 'font-weight: 800; color: #4ade80;' },
                { val: r.performance_value, style: 'font-weight: 700;' },
                { val: r.all_players.length, style: 'color: rgba(255,255,255,0.5);' },
                { val: dateToShort(new Date(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 3. Games Played (Recent list - keep as Recency)
        const tableRecent = document.getElementById('ach-table-recent');
        if (tableRecent && stats.recent_rounds) {
            // Sort by Timestamp (True Recency for "Recent" list)
            const sortedRecent = [...stats.recent_rounds].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            tableRecent.innerHTML = sortedRecent.map(r => renderAchRow(r, [
                { val: r.total_score, style: 'font-weight: 700;' },
                { val: r.ratio + 'x', style: 'color: rgba(255,255,255,0.4); font-size: 0.75rem;' },
                { val: r.is_win ? '<span style="color:#4ade80">WIN</span>' : '<span style="color:rgba(255,255,255,0.3)">-</span>', style: 'font-weight: 800;' },
                { val: dateToShort(new Date(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 4. Best Scores
        const tableScores = document.getElementById('ach-table-scores');
        if (tableScores && stats.best_scores) {
            const sortedByScore = [...stats.best_scores].sort((a, b) => b.total_score - a.total_score);
            tableScores.innerHTML = sortedByScore.map(r => renderAchRow(r, [
                { val: r.total_score, style: 'font-weight: 800; color: #ffd700;' },
                { val: r.performance_value, style: 'font-weight: 700;' },
                { val: dateToShort(new Date(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 5. Best Word Counts
        const tableWordCounts = document.getElementById('ach-table-wordcounts');
        if (tableWordCounts && stats.best_word_counts) {
            const sortedByCount = [...stats.best_word_counts].sort((a, b) => b.num_words - a.num_words);
            tableWordCounts.innerHTML = sortedByCount.map(r => renderAchRow(r, [
                { val: r.num_words, style: 'font-weight: 800; color: #a5b4fc;' },
                { val: r.avg_len + ' len', style: 'color: rgba(255,255,255,0.6);' },
                { val: dateToShort(new Date(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
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
    // Get Filter Values
    const lengthSelect = document.getElementById('list-length-filter');
    const startSelect = document.getElementById('list-start-filter');
    const typeSelect = document.getElementById('list-type-filter');
    
    // If we have a type override, update the select's value to keep UI in sync
    if (typeOverride && typeSelect) {
        typeSelect.value = typeOverride;
    }
    
    const selectedType = typeSelect ? typeSelect.value : 'nwl';

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

    try {
        // Build Query URL
        let url = `/api/tools/lists?list_type=${selectedType}&`;

        if (lengthSelect && lengthSelect.value !== 'all') {
            url += `length=${lengthSelect.value}&`;
        }
        if (startSelect && startSelect.value !== 'all') {
            url += `starts_with=${startSelect.value}`;
        }

        const response = await fetch(url + `&t=${Date.now()}`);
        const data = await response.json();

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

        if (!words || words.length === 0) {
            if (scrollArea) scrollArea.innerHTML = '<div style="padding:20px; opacity:0.6; text-align:center;">No words found matching these filters.</div>';
            return;
        }

        // Render based on type
        let html = '';
        if (selectedType === 'likelihood') {
            // Each item is { score, word }
            html = words.map(item => `
                <div class="list-item">
                    <span class="likelihood-score">${item.score}</span> ${item.word}
                </div>
            `).join('');
        } else if (selectedType === 'added') {
            // Added words: show removal for mods
            const isMod = window.currentUserIsMod;
            html = words.map(w => `
                <div class="list-item added-word" style="display: flex; justify-content: space-between; align-items: center;">
                    <span>${w}</span>
                    ${isMod ? `<button onclick="removeAddedWordFromTools('${w}')" style="background:none; border:none; color:#f43f5e; cursor:pointer; font-weight:bold; padding:0 5px;" title="Remove">&times;</button>` : ''}
                </div>
            `).join('');
        } else {
            // Each item is just a string
            html = words.map(w => `<div class="list-item">${w}</div>`).join('');
        }

        if (scrollArea) {
            scrollArea.innerHTML = html;
            scrollArea.scrollTop = 0;
        }

        listsDataLoaded = true;

    } catch (err) {
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
    const resultsContainer = document.getElementById('seq-results-container');

    const seq = inputEl.value.trim();
    const mode = modeEl.value;
    const length = lengthEl.value;

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
                dictionary: 'NWL' // Defaulting to NWL for now, could add selector later
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
    const dimSelect = document.getElementById('manual-dim');
    const solveBtn = document.getElementById('manual-solve-btn');
    const revealBtn = document.getElementById('manual-reveal-btn');

    if (dimSelect) {
        dimSelect.addEventListener('change', (e) => renderManualGrid(e.target.value));
        // Initial render
        renderManualGrid(dimSelect.value);
    }

    if (solveBtn) {
        solveBtn.addEventListener('click', runManualSolve);
    }

    if (revealBtn) {
        revealBtn.addEventListener('click', revealManualWords);
    }
}

function renderManualGrid(dims) {
    const gridEl = document.getElementById('manual-grid');
    if (!gridEl) return;

    const [rows, cols] = dims.split('x').map(Number);

    gridEl.style.gridTemplateColumns = `repeat(${cols}, 45px)`;
    gridEl.style.gridTemplateRows = `repeat(${rows}, 45px)`;

    gridEl.innerHTML = '';

    // Create inputs
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'manual-cell';
            input.maxLength = 1;
            input.dataset.r = r;
            input.dataset.c = c;

            // Auto-advance logic
            input.addEventListener('input', (e) => {
                const val = e.target.value;
                if (val && val.length === 1) {
                    const next = input.nextElementSibling;
                    if (next) next.focus();
                }
            });

            // Backspace logic
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Backspace' && !input.value) {
                    const prev = input.previousElementSibling;
                    if (prev) prev.focus();
                }
            });

            gridEl.appendChild(input);
        }
    }

    // Reset state
    manualSolvedWords = [];
    const resultsContainer = document.getElementById('manual-results-container');
    const revealBtn = document.getElementById('manual-reveal-btn');
    if (resultsContainer) resultsContainer.style.display = 'none';
    if (revealBtn) revealBtn.style.display = 'none';
}

async function runManualSolve() {
    const gridEl = document.getElementById('manual-grid');
    const dictEl = document.getElementById('manual-dict');
    const solveBtn = document.getElementById('manual-solve-btn');
    const revealBtn = document.getElementById('manual-reveal-btn');
    const resultsContainer = document.getElementById('manual-results-container');
    const dimSelect = document.getElementById('manual-dim');

    if (!gridEl || !dimSelect) return;

    const [rows, cols] = dimSelect.value.split('x').map(Number);
    const cells = gridEl.querySelectorAll('.manual-cell');

    // Build 2D board
    const board = [];
    let cellIdx = 0;
    let missing = false;

    for (let r = 0; r < rows; r++) {
        const row = [];
        for (let c = 0; c < cols; c++) {
            const val = cells[cellIdx++].value.trim().toUpperCase();
            if (!val) {
                missing = true;
            }
            row.push(val);
        }
        board.push(row);
    }

    if (missing) {
        alert("Please fill in all letters first.");
        return;
    }

    solveBtn.innerText = "Solving...";
    solveBtn.disabled = true;

    try {
        const response = await fetch('/api/tools/manual_solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board: board,
                dictionary: dictEl.value
            })
        });

        const data = await response.json();

        if (data.error) {
            alert("Solve failed: " + data.error);
            return;
        }

        // If this board matches a currently live room, block results
        if (data.board_matches_active_room) {
            manualSolvedWords = [];
            resultsContainer.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #f87171;">
                    <div style="font-size: 1.5rem; margin-bottom: 8px;">⚠️</div>
                    <div style="font-weight: 700; font-size: 1rem; margin-bottom: 6px;">Board In Use</div>
                    <div style="font-size: 0.85rem; opacity: 0.7;">This board is currently being played in a live room.<br>Results are not available while the round is active.</div>
                </div>`;
            resultsContainer.style.display = 'flex';
            revealBtn.style.display = 'none';
            return;
        }

        manualSolvedWords = data.results;

        // Show reveal button, hide results space initially
        revealBtn.style.display = 'inline-block';
        revealBtn.innerText = "Reveal Words";
        resultsContainer.style.display = 'none';

    } catch (err) {
        console.error("Manual solve failed:", err);
        alert("Server error during solve.");
    } finally {
        solveBtn.innerText = "Solve";
        solveBtn.disabled = false;
    }
}

function revealManualWords() {
    const resultsContainer = document.getElementById('manual-results-container');
    const revealBtn = document.getElementById('manual-reveal-btn');

    if (resultsContainer.style.display === 'flex') {
        resultsContainer.style.display = 'none';
        revealBtn.innerText = "Reveal Words";
        return;
    }

    if (manualSolvedWords.length === 0) {
        resultsContainer.innerHTML = '<div class="seq-results-placeholder">No words found on this board.</div>';
    } else {
        let html = `
            <div style="padding: 12px 20px; border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.3); font-weight: 700; color: #4facfe; text-transform: uppercase; letter-spacing: 1px; font-size: 0.85rem;">
                Found ${manualSolvedWords.length} words
            </div>
            <div style="flex: 1; overflow-y: auto; padding: 20px;">
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px;">
        `;

        html += manualSolvedWords.map(w => `
            <div style="padding: 8px 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: rgba(255,255,255,0.9); font-family: 'JetBrains Mono', monospace; text-align: center; font-size: 1rem; transition: background 0.2s; cursor: default;">
                ${w}
            </div>
        `).join('');

        html += `
                </div>
            </div>
        `;
        resultsContainer.innerHTML = html;
    }

    resultsContainer.style.display = 'flex';
    revealBtn.innerText = "Hide Words";
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
            inputEl.focus(); // Keep focus for next check
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
                    <div class="friend-flag-mini">${friend.country_flag || '🏳️'}</div>
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
        // Auto-focus the input when the pane might be clicked
        input.addEventListener('focus', () => {
            if (!unscrambleState.jumbled) startNewUnscramble();
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
                    <div style="display: flex; flex-direction: column; gap: 12px;">`;

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
