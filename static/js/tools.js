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
});

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

    const searchTerm = inputEl.value.trim();
    const dictionary = dictEl.value;

    if (!searchTerm) return;

    // Clear previous results
    document.getElementById('mp-container').innerHTML = '';
    document.getElementById('lic-container').innerHTML = '';

    resultsContainer.classList.remove('hidden');

    try {
        const response = await fetch('/api/tools/combo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ search_term: searchTerm, dictionary: dictionary })
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Render MP Groups (0MP to 5MP)
        renderGroups(data.mp_groups, 'mp-container', 'MP');

        // Render LIC Groups (Shared Count)
        renderGroups(data.lic_groups, 'lic-container', 'LIC');

    } catch (error) {
        console.error('Combo check failed:', error);
        alert('An error occurred while checking combo.');
    }
}

function renderGroups(groupsData, containerId, type) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Sort keys logically
    // MP keys are 0, 1, 2... (Integers)
    // LIC keys are Lengths (Integers)
    const keys = Object.keys(groupsData).map(Number).sort((a, b) => a - b);

    keys.forEach(key => {
        const words = groupsData[key];
        if (words.length === 0) return;

        let label = '';
        if (type === 'MP') {
            label = `${key}MP`; // e.g. 0MP (0 Ops)
        } else {
            label = `${key}LIC`; // e.g. 5LIC
        }

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
}

// --- Mini Profile Logic ---

function setupMiniProfileModal() {
    const modal = document.getElementById('mini-profile-modal');
    const closeBtn = document.getElementById('mini-profile-close');

    if (modal && closeBtn) {
        closeBtn.onclick = () => modal.classList.add('hidden');
        modal.onclick = (e) => {
            if (e.target === modal) modal.classList.add('hidden');
        };
    }
}

async function showMiniProfile(username) {
    if (!username) return;

    const modal = document.getElementById('mini-profile-modal');
    if (!modal) return;

    try {
        const response = await fetch(`/api/profile/${encodeURIComponent(username)}`);
        const data = await response.json();

        if (data.error) return;

        // Populate Modal
        document.getElementById('mini-profile-username').innerText = data.username;
        document.getElementById('mini-profile-fullname').innerText = data.full_name || '-';
        document.getElementById('mini-profile-games').innerText = data.games_played || 0;
        document.getElementById('mini-profile-age').innerText = data.age || '-';
        document.getElementById('mini-profile-gender').innerText = data.gender || '-';
        document.getElementById('mini-profile-flag').innerText = data.country_flag || '🏳️';
        document.getElementById('mini-profile-quote').innerText = data.quote ? `"${data.quote}"` : '"No quote provided."';
        document.getElementById('mini-profile-description').innerText = data.description || 'No description provided.';

        // Country Name Lookup
        const flagEmoji = data.country_flag || '🏳️';
        const country = typeof ALL_FLAGS !== 'undefined' ? ALL_FLAGS.find(f => f.flag === flagEmoji) : null;
        document.getElementById('mini-profile-country-name').innerText = country ? country.name : 'International';

        const statusEl = document.getElementById('mini-profile-status');
        const statusIcon = document.getElementById('mini-profile-status-icon');
        const isOnline = data.status && data.status.is_online;

        statusEl.innerText = isOnline ? 'Online' : 'Offline';
        statusEl.style.color = isOnline ? '#4ade80' : 'rgba(255,255,255,0.5)';

        if (statusIcon) {
            statusIcon.innerText = isOnline ? '🟢' : '⚪';
            statusIcon.style.filter = isOnline ? 'drop-shadow(0 0 5px #4ade80)' : 'none';
        }
        const rating = data.rating || 0;
        const ratingBadge = document.getElementById('mini-profile-rating-badge');
        ratingBadge.innerText = rating;
        const ratingColor = window.getRatingColor ? window.getRatingColor(rating) : '#fff';
        ratingBadge.style.color = ratingColor;
        ratingBadge.style.borderColor = `${ratingColor}44`;

        if (data.avatar_url) {
            ratingBadge.style.cursor = 'pointer';
            ratingBadge.title = "View user image";
            ratingBadge.onclick = () => showImageLightbox(data.avatar_url, `${data.username}'s Profile Image`);
        } else {
            ratingBadge.style.cursor = 'default';
            ratingBadge.title = "";
            ratingBadge.onclick = null;
        }

        const avatar = document.getElementById('mini-profile-avatar');
        if (data.avatar_url) {
            avatar.style.background = 'none'; // Clear any previous gradient
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
            avatar.style.background = `linear-gradient(135deg, ${ratingColor}, #444)`;
            avatar.innerText = data.username.charAt(0).toUpperCase();
        }

        // Setup Buttons
        const viewFullBtn = document.getElementById('mini-profile-view-full');
        viewFullBtn.onclick = () => {
            modal.classList.add('hidden');
            // Navigate to tools page and search
            const toolsBtn = document.querySelector('.nav-btn[data-page="tools"]');
            if (toolsBtn) toolsBtn.click();

            const profileToolBtn = document.querySelector('.tool-nav-btn[data-tool="profile"]');
            if (profileToolBtn) profileToolBtn.click();

            window.performProfileSearch(data.username);
        };

        const msgBtn = document.getElementById('mini-profile-message');
        const globalUser = window.currentUser || (typeof currentUser !== 'undefined' ? currentUser : null);
        const currentName = (typeof globalUser === 'object') ? globalUser.username : globalUser;

        if (currentName && currentName.toLowerCase() !== data.username.toLowerCase()) {
            msgBtn.classList.remove('hidden');
            msgBtn.onclick = () => {
                modal.classList.add('hidden');
                window.openPrivateChat(data.username, true);
            };

            const friendBtn = document.getElementById('mini-profile-friend');
            if (friendBtn) {
                friendBtn.classList.remove('hidden');
                await updateFriendButtonStatus(data.username, friendBtn);
                friendBtn.onclick = () => handleFriendAction(data.username, friendBtn);
            }
        } else {
            msgBtn.classList.add('hidden');
            const friendBtn = document.getElementById('mini-profile-friend');
            if (friendBtn) friendBtn.classList.add('hidden');
        }

        modal.classList.remove('hidden');

    } catch (err) {
        console.error("Mini profile fetch error:", err);
    }
}
window.showMiniProfile = showMiniProfile;

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

            // If switching AWAY from history, maybe hide the replay panel?
            // Actually, keep it if they want to review later.
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

async function performProfileSearch(username) {
    if (!username || !username.trim()) return;

    username = username.trim();
    const container = document.getElementById('profile-display-container');
    const input = document.getElementById('profile-search-input');
    if (input) input.value = username;

    // Guests do not have profiles
    if (username.startsWith('Guest_')) {
        container.classList.add('hidden');
        return;
    }

    container.classList.add('hidden');

    try {
        const response = await fetch(`/api/profile/${encodeURIComponent(username)}`);
        const data = await response.json();

        if (data.error) {
            // User not found, just don't show the profile
            return;
        }

        await renderProfile(data);
        container.classList.remove('hidden');

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

    const bestScoreEl = document.getElementById('profile-best-score');
    if (bestScoreEl && user.recent_rounds && user.recent_rounds.length > 0) {
        const best = Math.max(...user.recent_rounds.map(r => r.total_score || 0));
        bestScoreEl.innerText = best;
    } else if (bestScoreEl) {
        bestScoreEl.innerText = '-';
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
            round.game_type === 'fcfs' ? 'FCFS' : 'Acc';
        const typeClass = `history-type-${round.game_type}`;
        const dims = round.dimensions || (round.board ? `${round.board.length}x${round.board[0].length}` : '4x4');

        return `
        <div class="history-grid-item" onclick="watchRoundHistory('${round.room_id}', ${round.round_number}, true)" style="display: grid; grid-template-columns: 100px 70px 90px 70px 110px 100px 1fr 100px; gap:10px; padding: 14px 20px; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.03); border-radius: 10px; margin-bottom: 8px; align-items: center; transition: all 0.2s; cursor: pointer; position: relative; overflow: hidden;">
            <div class="history-mode-tag ${typeClass}" style="font-size: 0.65rem; padding: 3px 8px; border-radius: 6px; text-align: center; width: fit-content; font-weight: 800; text-transform: uppercase;">${gameTypeLabel}</div>
            <div style="font-family: monospace; font-size: 0.8rem; color: rgba(255,255,255,0.7); font-weight: 700;">${dims}</div>
            <div style="font-weight: 900; color: #fff; font-size: 1rem;">${round.total_score} <small style="font-size: 0.6rem; opacity: 0.5;">PTS</small></div>
            <div style="font-weight: 900; color: ${round.performance_value >= 140 ? '#60a5fa' : 'rgba(255,255,255,0.2)'}; font-size: 0.9rem;">${round.performance_value || '-'}</div>
            <div style="display: flex; flex-direction: column; gap: 2px;">
                <span style="color: #fff; font-size: 0.75rem; font-weight: 700;">${round.num_words} words</span>
                <span style="color: rgba(255,255,255,0.3); font-size: 0.6rem;">Avg: ${round.avg_len}</span>
            </div>
            <div style="color: #ffd700; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; letter-spacing: 0.5px;" title="${round.top_word}">${round.top_word}</div>
            <div style="display: flex; flex-direction: column; gap: 2px;">
                <span style="font-size: 0.75rem; color: #60a5fa; font-weight: 700; opacity: 0.8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 150px;">${round.room_id}</span>
                <span style="font-size: 0.65rem; color: rgba(255,255,255,0.3); font-weight: 600;">Str: ${round.room_strength || '-'}</span>
            </div>
            <div style="text-align: right;">
                <button class="history-snap-btn" title="View Snapshot"
                         onclick="event.stopPropagation(); watchRoundHistory('${round.room_id}', ${round.round_number}, true)"
                         style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 6px 10px; cursor: pointer;">
                    <span style="font-size: 1.1rem;">📷</span>
                </button>
            </div>
        </div>
        `;
    };

    window.roundGridHeader = `
        <div class="history-grid-header" style="display: grid; grid-template-columns: 100px 70px 90px 70px 110px 100px 1fr 100px; gap:10px; padding: 12px 20px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 12px; font-size: 0.7rem; color: rgba(255,255,255,0.4); font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">
            <div>Mode</div>
            <div>Board</div>
            <div>Score</div>
            <div>Perf</div>
            <div>Stats</div>
            <div>Top Word</div>
            <div>Room / Str</div>
            <div style="text-align: right;">Snapshot</div>
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

    // Cache rounds for review
    window.lastRenderedRounds = user.recent_rounds || [];

    setupProfileEditing(isOwner);
}

// Helper: Find a valid Boggle path for a word on the current board
function findWordPath(board, word) {
    if (!board || !word) return null;
    const rows = board.length;
    const cols = board[0].length;
    const targetWord = word.toUpperCase();

    function dfs(r, c, index, visited) {
        if (index >= targetWord.length) return [];

        const letter = board[r][c].toUpperCase();
        let matchLen = 0;

        // Boggle Logic: Q tile usually represents 'QU'
        if (targetWord[index] === letter) {
            matchLen = 1;
        } else if (letter === 'Q' && targetWord.substring(index, index + 2) === 'QU') {
            matchLen = 2;
        }

        if (matchLen === 0) return null;

        // Final letter check
        if (index + matchLen === targetWord.length) {
            return [{ row: r, col: c }];
        }

        visited.add(`${r},${c}`);

        // 8 directions
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const nr = r + dr;
                const nc = c + dc;

                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !visited.has(`${nr},${nc}`)) {
                    const result = dfs(nr, nc, index + matchLen, visited);
                    if (result) {
                        return [{ row: r, col: c }, ...result];
                    }
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

// Global function to review a round (Legitimacy Walkthrough)
window.watchRoundHistory = function (roomId, roundNum, isSnapshot = false) {
    console.log(`Reviewing Round ${roundNum} from Room ${roomId}`);

    let rounds = window.lastRenderedRounds || [];
    let round = rounds.find(r => r.room_id == roomId && r.round_number == roundNum);

    // FALLBACK: If not in profile rounds, check the Lobby's winners_history
    if (!round && window.lastGameState && window.lastGameState.winners_history) {
        const foundInLobby = window.lastGameState.winners_history.find(h => h.round == roundNum);
        if (foundInLobby && foundInLobby.board) {
            console.log(`[Review] Found round ${roundNum} in Lobby winners_history`);
            round = {
                ...foundInLobby,
                room_id: roomId,
                round_number: foundInLobby.round,
                total_score: foundInLobby.score
            };
        }
    }

    if (!round) {
        alert("Round details not available. This round may have happened before the snapshot system was enabled or you need to refresh.");
        return;
    }

    const panel = document.getElementById('integrated-replay-panel');
    if (!panel) return;

    // Show the panel
    panel.classList.remove('hidden');
    panel.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // --- Cleanup any existing playback ---
    if (window.replayInterval) {
        clearInterval(window.replayInterval);
        window.replayInterval = null;
    }

    // 1. Reset & Populate Summary
    document.getElementById('integrated-total-score').innerText = `${round.total_score} PTS`;

    // Reset Replay UI
    const startBtn = document.getElementById('integrated-start-btn');
    const skipBtn = document.getElementById('integrated-skip-btn');
    const progressUI = document.getElementById('integrated-progress-ui');
    const walkthroughList = document.getElementById('integrated-walkthrough-list');

    startBtn.classList.remove('hidden');
    skipBtn.classList.add('hidden');
    progressUI.classList.add('hidden');
    walkthroughList.innerHTML = '<p class="placeholder" style="color:var(--muted-text); text-align:center; padding:20px;">Ready to watch the walkthrough...</p>';

    // 2. Render Board
    const boardContainer = document.getElementById('integrated-board-container');
    if (boardContainer && round.board && round.board.length > 0) {
        boardContainer.innerHTML = ''; // Clear prior content
        boardContainer.style.gridTemplateColumns = ''; // Reset CSS
        const rows = round.board.length;
        const cols = round.board[0].length;
        boardContainer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        boardContainer.innerHTML = round.board.flat().map(letter => `
            <div class="review-cell">${letter}</div>
        `).join('');
    }

    // 2b. Render Players List (Leaderboard Snapshot)
    const playersList = document.getElementById('integrated-players-list');
    const playersBody = document.getElementById('integrated-players-body');
    if (playersList && playersBody) {
        if (round.all_players && round.all_players.length > 0) {
            playersList.classList.remove('hidden');
            playersBody.innerHTML = round.all_players.map((p, idx) => `
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 0; ${idx === 0 ? 'border-bottom: 2px solid rgba(255,215,0,0.3); padding-bottom: 8px;' : ''}">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 0.7rem; color: ${idx === 0 ? '#ffd700' : 'rgba(255,255,255,0.4)'}; font-weight: 800;">#${idx + 1}</span>
                        <span style="font-weight: 800; color: ${idx === 0 ? '#fff' : 'rgba(255,255,255,0.7)'}; font-size: 0.85rem;">${p.username}</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: 900; color: ${idx === 0 ? '#ffd700' : '#fff'}; font-size: 0.9rem;">${p.score}</div>
                        <div style="font-size: 0.6rem; color: rgba(255,255,255,0.3); font-weight: 700;">${p.rating || 1200}</div>
                    </div>
                </div>
            `).join('');
        } else {
            playersList.classList.add('hidden');
        }
    }

    // 3. Playback Logic
    // Fix: the data structure might have 'total_score' or 'score'
    const words = round.words || [];
    const sortedWords = [...words].sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
    const roundDuration = round.round_duration || 60;
    const startTime = round.round_start_time || (sortedWords[0] ? sortedWords[0].timestamp - 5 : 0);

    const renderWord = (word) => {
        const relTimeSec = Math.max(0, (word.timestamp || 0) - startTime);
        const min = Math.floor(relTimeSec / 60);
        const sec = (relTimeSec % 60).toFixed(1);
        const timeStr = `${min}:${sec.padStart(4, '0')}`;

        return `
        <div class="walkthrough-item reveal" style="padding:10px; border-bottom:1px solid rgba(255,255,255,0.05); display:flex; justify-content:space-between; align-items:center;">
            <div style="font-family:monospace; color:var(--accent-color); font-weight:700;">${timeStr}</div>
            <div style="font-weight:800; text-transform:uppercase; letter-spacing:1px; flex:1; margin-left:15px;">${word.word}</div>
            <div style="color:#ffd700; font-weight:700;">${word.points} pts</div>
        </div>
        `;
    };

    const showAllWords = () => {
        walkthroughList.innerHTML = sortedWords.map(w => renderWord(w)).join('');
        if (sortedWords.length === 0) walkthroughList.innerHTML = '<p class="placeholder" style="color:var(--muted-text); text-align:center; padding:20px;">No words discovered in this round.</p>';
        skipBtn.classList.add('hidden');
        progressUI.classList.add('hidden');
    };

    // IF SNAPSHOT MODE: Jump straight to end
    if (isSnapshot) {
        showAllWords();
        startBtn.classList.add('hidden');
        return;
    }

    startBtn.onclick = () => {
        startBtn.classList.add('hidden');
        skipBtn.classList.remove('hidden');
        progressUI.classList.remove('hidden');
        walkthroughList.innerHTML = '';

        let elapsed = 0;
        let wordIndex = 0;
        let localScore = 0;
        const tick = 100; // 0.1s increments

        // Clear any existing highlights
        document.querySelectorAll('.review-cell').forEach(c => c.className = 'review-cell');

        window.replayInterval = setInterval(() => {
            elapsed += tick / 1000;

            // Update Progress Bar
            const progress = (elapsed / roundDuration) * 100;
            document.getElementById('integrated-progress-bar').style.width = `${Math.min(100, progress)}%`;

            // Update Timer
            const m = Math.floor(elapsed / 60);
            const s = (elapsed % 60).toFixed(1);
            document.getElementById('integrated-current-time').innerText = `${m}:${s.padStart(4, '0')}`;

            // Check for newly discovered words
            while (wordIndex < sortedWords.length) {
                const word = sortedWords[wordIndex];
                const relWordTime = (word.timestamp || 0) - startTime;

                if (elapsed >= relWordTime) {
                    walkthroughList.insertAdjacentHTML('afterbegin', renderWord(word));
                    localScore += word.points;
                    document.getElementById('integrated-total-score').innerText = `${localScore} PTS`;

                    // --- SYNCHRONIZED BOARD HIGHLIGHT ---
                    const path = findWordPath(round.board, word.word);
                    if (path) {
                        const cells = boardContainer.querySelectorAll('.review-cell');
                        const cols = round.board[0].length;

                        // Clear previous highlight
                        cells.forEach(c => c.classList.remove('highlight', 'highlight-bonus'));

                        // Apply new highlight
                        path.forEach((p, i) => {
                            const cellIdx = p.row * cols + p.col;
                            setTimeout(() => {
                                if (cells[cellIdx]) {
                                    cells[cellIdx].classList.add('highlight');
                                }
                            }, i * 50);
                        });
                    }

                    wordIndex++;
                } else {
                    break;
                }
            }

            if (elapsed >= roundDuration) {
                clearInterval(window.replayInterval);
                skipBtn.classList.add('hidden');
                startBtn.classList.remove('hidden');
                startBtn.innerText = "Replay";
            }
        }, tick);
    };

    skipBtn.onclick = () => {
        if (window.replayInterval) clearInterval(window.replayInterval);
        showAllWords();
    };
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
});

function renderRatingsGrid(configRatings, user = null) {
    const grid = document.getElementById('profile-ratings-grid');
    if (!grid) return;

    // Cache the original data and user on the element if not already there
    if (configRatings) grid._configRatings = configRatings;
    if (user) grid._user = user;

    const ratings = grid._configRatings || {};
    const u = grid._user || null;

    grid.innerHTML = '';

    const filterMode = document.getElementById('rankings-filter-mode')?.value || 'all';
    const filterDims = document.getElementById('rankings-filter-dims')?.value || 'all';
    const filterTime = document.getElementById('rankings-filter-time')?.value || 'all';

    const modes = ['accumulative', 'fcfs', 'split'];
    const boards = ['4x4', '4x6', '5x7', '6x8'];
    const accTimes = [45, 180, 600]; // Removed 86400 (24h)
    const otherTimes = [45, 180];

    const formatTimeShort = (s) => {
        if (s === 45) return '45s';
        if (s === 180) return '3m';
        if (s === 600) return '10m';
        return s + 's';
    };

    let visibleCount = 0;

    modes.forEach(mode => {
        if (filterMode !== 'all' && mode !== filterMode) return;

        const times = (mode === 'accumulative') ? accTimes : otherTimes;
        boards.forEach(board => {
            if (filterDims !== 'all' && board !== filterDims) return;

            times.forEach(time => {
                if (filterTime !== 'all' && String(time) !== filterTime) return;

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
                        <div class="rating-box-mode" style="font-size: 0.65rem; color: rgba(255,255,255,0.4); text-transform: uppercase; font-weight: 800;">${mode}</div>
                        <div class="rating-box-config" style="font-weight: 700;">${board} | ${formatTimeShort(time)}</div>
                        <div style="display: flex; gap: 10px; margin-top: 4px; font-size: 0.65rem; color: rgba(255,255,255,0.3); font-weight: 700;">
                           <span>AVG S: <span style="color: #fff;">${configData.avg_score}</span></span>
                           <span>AVG P: <span style="color: #fff;">${configData.avg_perf}</span></span>
                        </div>
                    </div>
                    <div class="rating-box-value" style="color: ${rColor}; font-size: 1.25rem; font-weight: 900; margin: 0 15px;">${rating}</div>
                    <div class="rating-box-snapshot" title="View Best Round Snapshot" 
                         style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 6px 10px; display: flex; align-items: center; justify-content: center; gap: 6px; cursor: pointer;"
                         onclick="event.stopPropagation(); if('${u?.username}') { fetch('/api/room_achievements?username=${u.username}&mode=${mode}&board=${board}&time=${time}').then(r => r.json()).then(d => { if(d.stats && d.stats.exceptional_round) { if(!window.lastRenderedRounds) window.lastRenderedRounds=[]; window.lastRenderedRounds.push(d.stats.exceptional_round); window.watchRoundHistory(d.stats.exceptional_round.room_id, d.stats.exceptional_round.round_number, true); } }); }">
                        <span style="font-size: 1rem;">📷</span>
                    </div>
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
        closeBtn.onclick = () => modal.classList.add('hidden');
        modal.onclick = (e) => {
            if (e.target === modal) modal.classList.add('hidden');
        };

        // ESC key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') modal.classList.add('hidden');
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

async function showRoomAchievements(username, mode, board, time) {
    const modal = document.getElementById('room-achievements-modal');
    if (!modal) return;

    // Set titles
    document.getElementById('achievement-title').textContent = `${username}'s Achievements`;
    document.getElementById('achievement-subtitle').textContent =
        `${mode.charAt(0).toUpperCase() + mode.slice(1)} | ${board} | ${time < 300 ? time + 's' : (time / 60) + 'm'}`;

    // Show loading state
    modal.classList.remove('hidden');
    modal.style.display = 'flex';
    modal.style.opacity = '1';

    try {
        const response = await fetch(`/api/profile/${username}/achievements/${mode}/${board}/${time}`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        // Update Rating
        document.getElementById('achievement-rating-val').textContent = data.rating || 1200;

        if (!data.stats) {
            // No history for this config
            const fields = ['ach-high-score', 'ach-max-words', 'ach-longest-word', 'ach-best-word',
                'ach-games-played', 'ach-wins', 'ach-win-rate', 'ach-total-words', 'ach-total-points',
                'ach-exc-ratio', 'ach-exc-score', 'ach-exc-date'];
            fields.forEach(f => {
                const el = document.getElementById(f);
                if (el) el.textContent = '-';
            });
            document.getElementById('ach-total-points').textContent = '0';

            document.getElementById('exceptional-round-info')?.classList.add('hidden');
            document.getElementById('no-exceptional-msg')?.classList.remove('hidden');
            return;
        }

        const stats = data.stats;
        document.getElementById('ach-high-score').textContent = stats.high_score;
        document.getElementById('ach-max-words').textContent = stats.max_words;
        document.getElementById('ach-longest-word').textContent = stats.longest_word || 'None';
        document.getElementById('ach-best-word').textContent = stats.best_word.word ?
            `${stats.best_word.word} (${stats.best_word.points} pts)` : 'None';

        document.getElementById('ach-games-played').textContent = stats.games_played;
        document.getElementById('ach-wins').textContent = stats.wins;
        document.getElementById('ach-win-rate').textContent = stats.win_rate + '%';
        document.getElementById('ach-total-words').textContent = stats.total_words;
        document.getElementById('ach-total-points').textContent = stats.total_score.toLocaleString();

        // Update Averages and Totals
        document.getElementById('ach-avg-perf').textContent = stats.avg_perf || '-';
        document.getElementById('ach-avg-winrate').textContent = (stats.win_rate || 0) + '%';
        document.getElementById('ach-total-games').textContent = stats.games_played || '0';
        document.getElementById('ach-avg-score').textContent = (stats.avg_score || 0).toLocaleString();
        document.getElementById('ach-avg-words').textContent = stats.avg_words || '0';
        document.getElementById('ach-avg-word-pts').textContent = stats.avg_word_pts || '0';

        const renderAchRow = (r, cols) => {
            // Cache if not present
            if (!window.lastRenderedRounds) window.lastRenderedRounds = [];
            if (!window.lastRenderedRounds.find(cr => cr.room_id === r.room_id && cr.round_number === r.round_number)) {
                window.lastRenderedRounds.push(r);
            }

            return `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03); cursor: pointer; transition: background 0.2s;" 
                onmouseenter="this.style.background='rgba(255,255,255,0.02)'" 
                onmouseleave="this.style.background='transparent'" 
                onclick="watchRoundHistory('${r.room_id}', ${r.round_number}, true); document.getElementById('room-achievements-modal').classList.add('hidden');">
                ${cols.map(c => `<td style="padding: 10px 15px; ${c.style || ''}">${c.val}</td>`).join('')}
                <td style="padding: 10px 15px; text-align: right;"><div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 4px 8px; display: inline-block;">📷</div></td>
            </tr>`;
        };

        // 1. Exceptional Performances
        const tablePerf = document.getElementById('ach-table-perf');
        if (tablePerf && stats.exceptional_rounds) {
            tablePerf.innerHTML = stats.exceptional_rounds.map(r => renderAchRow(r, [
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
            tableWins.innerHTML = stats.winning_rounds.map(r => renderAchRow(r, [
                { val: r.total_score, style: 'font-weight: 800; color: #4ade80;' },
                { val: r.performance_value, style: 'font-weight: 700;' },
                { val: r.all_players.length, style: 'color: rgba(255,255,255,0.5);' },
                { val: dateToShort(new Date(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 3. Recent Rounds
        const tableRecent = document.getElementById('ach-table-recent');
        if (tableRecent && stats.recent_rounds) {
            tableRecent.innerHTML = stats.recent_rounds.map(r => renderAchRow(r, [
                { val: r.total_score, style: 'font-weight: 700;' },
                { val: r.is_win ? '<span style="color:#4ade80">WIN</span>' : '<span style="color:rgba(255,255,255,0.3)">-</span>', style: 'font-weight: 800;' },
                { val: dateToShort(new Date(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 4. Best Scores
        const tableScores = document.getElementById('ach-table-scores');
        if (tableScores && stats.best_scores) {
            tableScores.innerHTML = stats.best_scores.map(r => renderAchRow(r, [
                { val: r.total_score, style: 'font-weight: 800; color: #ffd700;' },
                { val: r.performance_value, style: 'font-weight: 700;' },
                { val: dateToShort(new Date(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 5. Best Word Counts
        const tableWordCounts = document.getElementById('ach-table-wordcounts');
        if (tableWordCounts && stats.best_word_counts) {
            tableWordCounts.innerHTML = stats.best_word_counts.map(r => renderAchRow(r, [
                { val: r.num_words, style: 'font-weight: 800; color: #a5b4fc;' },
                { val: r.avg_len + ' len', style: 'color: rgba(255,255,255,0.6);' },
                { val: dateToShort(new Date(r.timestamp)), style: 'font-size: 0.75rem; color: rgba(255,255,255,0.4);' }
            ])).join('');
        }

        // 6. Best Words (Individual)
        const tableWords = document.getElementById('ach-table-words');
        if (tableWords && stats.best_words) {
            tableWords.innerHTML = stats.best_words.map(w => {
                const date = new Date(w.timestamp);
                return `
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03); cursor: pointer; transition: background 0.2s;" 
                    onmouseenter="this.style.background='rgba(255,255,255,0.02)'" 
                    onmouseleave="this.style.background='transparent'" 
                    onclick="watchRoundHistory('${w.room_id}', ${w.round_number}, true); document.getElementById('room-achievements-modal').classList.add('hidden');">
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
}

async function fetchListsData() {
    // Get Filter Values
    const lengthSelect = document.getElementById('list-length-filter');
    const startSelect = document.getElementById('list-start-filter');

    // UI Feedback
    const colIds = ['col-nwl', 'col-csw', 'col-csw-only', 'col-likelihood', 'col-uniques', 'col-added'];
    colIds.forEach(id => {
        const el = document.querySelector(`#${id} .list-scroll-area`);
        if (el) el.innerHTML = '<div style="padding:10px; opacity:0.6;">Loading...</div>';
    });

    try {
        // Build Query URL
        let url = '/api/tools/lists?';

        if (lengthSelect && lengthSelect.value !== 'all') {
            url += `length=${lengthSelect.value}&`;
        }
        if (startSelect && startSelect.value !== 'all') {
            url += `starts_with=${startSelect.value}`;
        }

        const response = await fetch(url);
        const data = await response.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        renderListColumn('col-nwl', data.nwl);
        renderListColumn('col-csw', data.csw);
        renderListColumn('col-csw-only', data.csw_only);
        renderListColumn('col-likelihood', data.likelihood);
        renderListColumn('col-added', data.added);
        renderListColumn('col-uniques', data.uniques);

        listsDataLoaded = true;

    } catch (err) {
        console.error('Failed to fetch lists:', err);
        colIds.forEach(id => {
            const el = document.querySelector(`#${id} .list-scroll-area`);
            if (el) el.innerHTML = '<div style="color:red; padding:10px;">Error loading.</div>';
        });
    }
}

function renderListColumn(colId, words) {
    const container = document.querySelector(`#${colId} .list-scroll-area`);
    if (!container) return;

    if (!words || words.length === 0) {
        container.innerHTML = '<div style="padding:10px; opacity:0.6;">(Empty)</div>';
        return;
    }

    // Creating a huge string is faster than creating elements one by one.
    const html = words.map(w => `<div class="list-item">${w}</div>`).join('');
    container.innerHTML = html;
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
            <div style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.2);">
                Found ${count} words
            </div>
            <div style="flex: 1; overflow-y: auto; padding: 10px;">
                <table class="group-table" style="width: 100%;">
                    <tbody>
        `;

        // Use chunks to avoid blocking if list is huge? For now direct map.
        html += words.map(w => `
            <tr><td style="padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); color: rgba(255,255,255,0.9); font-family: monospace;">${w}</td></tr>
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
    if (window.currentRoomId) {
        alert("The manual solver is disabled while you are in a room.");
        return;
    }
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
            <div style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.2);">
                Found ${manualSolvedWords.length} words
            </div>
            <div style="flex: 1; overflow-y: auto; padding: 10px;">
                <table class="group-table" style="width: 100%;">
                    <tbody>
        `;

        html += manualSolvedWords.map(w => `
            <tr><td style="padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); color: rgba(255,255,255,0.9); font-family: monospace;">${w}</td></tr>
        `).join('');

        html += `
                    </tbody>
                </table>
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

    try {
        const url = `/api/tools/random_word?length=${length}&dictionary=${dictionary}`;
        const response = await fetch(url);
        const data = await response.json();

        if (data.error) {
            displayEl.innerHTML = `<span style="font-size: 1.5rem; color: #f43f5e;">${data.error}</span>`;
            return;
        }

        const word = data.word;

        // Add a class to re-trigger animation
        displayEl.classList.remove('random-word-large');
        void displayEl.offsetWidth; // Trigger reflow
        displayEl.classList.add('random-word-large');

        displayEl.innerText = word;

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

async function updateWotd() {
    const displayEl = document.getElementById('wotd-display');
    if (!displayEl) return;

    // Only fetch if empty to avoid redundant calls on every toggle
    if (displayEl.innerText.trim() !== '') return;

    displayEl.innerHTML = '<span style="font-size: 1.5rem; opacity: 0.5;">Loading...</span>';

    try {
        const response = await fetch('/api/tools/wotd');
        const data = await response.json();

        if (data.error) {
            displayEl.innerText = 'Error loading word';
            return;
        }

        displayEl.innerText = data.word;
        const defEl = document.getElementById('wotd-definition');
        if (defEl) {
            defEl.innerText = data.definition || "No definition available.";
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
            <div style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.2); text-align: center;">
                Found ${count} subanagrams
            </div>
            <div style="flex: 1; overflow-y: auto; padding: 10px;">
                <table class="group-table" style="width: 100%;">
                    <tbody>
        `;

        html += words.map(w => `
            <tr><td style="padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); color: rgba(255,255,255,0.9); font-family: monospace;">${w}</td></tr>
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

    checkBtn.innerText = "Checking...";
    checkBtn.disabled = true;

    try {
        const response = await fetch('/api/tools/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word, dictionary: dictionary })
        });

        const data = await response.json();

        if (data.error) {
            displayEl.innerHTML = `<span style="font-size: 1.5rem; color: #f43f5e;">${data.error}</span>`;
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

    } catch (err) {
        console.error("Validation check failed:", err);
        displayEl.innerHTML = `<span style="font-size: 1.5rem; color: #f43f5e;">Error checking word.</span>`;
    } finally {
        checkBtn.innerText = "Validate";
        checkBtn.disabled = false;
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

