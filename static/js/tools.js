document.addEventListener('DOMContentLoaded', () => {
    setupToolsNavigation();
    setupProfileTool();
    setupComboChecker();
    setupListsTool();
    setupSequenceTool();
    setupManualTool();
    setupRandomWordTool();
    setupWotdTool();
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

        renderProfile(data);
        container.classList.remove('hidden');

    } catch (err) {
        console.error("Profile search error:", err);
    }
}

function renderProfile(user) {
    const usernameEl = document.getElementById('profile-username');
    if (usernameEl) usernameEl.innerText = user.username;

    // Full Name
    const fullNameEl = document.getElementById('profile-full-name');
    if (fullNameEl) fullNameEl.innerText = user.full_name || '-';

    // Rating & Color (Global display removed, but we still need color for avatar theme)
    const avatar = document.querySelector('.profile-avatar.large');

    // Determine Color based on global rating
    let color = '#b3b3b3';
    const r = user.rating || 0;

    if (r < 700) color = '#66ff66';
    else if (r < 1400) color = '#0088ff';
    else if (r < 2000) color = '#ffd700';
    else color = '#e60000';

    // Avatar Handling
    if (avatar) {
        if (user.avatar_url) {
            avatar.style.backgroundColor = 'rgba(0,0,0,0.3)';
            avatar.style.backgroundImage = `url('${user.avatar_url}')`;
            avatar.style.backgroundSize = 'contain';
            avatar.style.backgroundRepeat = 'no-repeat';
            avatar.style.backgroundPosition = 'center';
            avatar.innerText = '';
        } else {
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

    // Profile Details
    const ageEl = document.getElementById('profile-age-val');
    const genderEl = document.getElementById('profile-gender-val');
    const quoteEl = document.getElementById('profile-quote-val');
    const locationEl = document.getElementById('profile-location-val');

    if (ageEl) ageEl.innerText = user.age || '-';
    if (genderEl) genderEl.innerText = user.gender || '-';
    if (locationEl) locationEl.innerText = user.location || '-';
    if (quoteEl) quoteEl.innerText = user.quote || 'Welcome to Morpheme.';

    // Online Status & Follow Button
    const statusDot = document.getElementById('profile-status-indicator');
    const followBtn = document.getElementById('profile-follow-btn');
    const roomInput = document.getElementById('profile-current-room');

    // Check Ownership for Editing
    const globalUser = window.currentUser || currentUser;
    const currentName = (typeof globalUser === 'object') ? globalUser.username : globalUser;
    const isOwner = currentName && currentName.toLowerCase() === user.username.toLowerCase();

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

    // Render Ratings Grid (32 setups)
    renderRatingsGrid(user.config_ratings || {});

    setupProfileEditing(isOwner);
}

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

function renderRatingsGrid(configRatings) {
    const grid = document.getElementById('profile-ratings-grid');
    if (!grid) return;

    grid.innerHTML = '';

    const modes = ['accumulative', 'fcfs', 'split'];
    const boards = ['4x4', '4x6', '5x7', '6x8'];
    const accTimes = [45, 180, 600, 86400];
    const otherTimes = [45, 180];

    const formatTimeShort = (s) => {
        if (s === 45) return '45s';
        if (s === 180) return '3m';
        if (s === 600) return '10m';
        if (s === 86400) return '24h';
        return s + 's';
    };

    modes.forEach(mode => {
        const times = (mode === 'accumulative') ? accTimes : otherTimes;
        boards.forEach(board => {
            times.forEach(time => {
                const configKey = `${mode}|${board}|${time}`;
                const rating = configRatings[configKey] || 1200;

                // Color for this specific rating
                let rColor = '#b3b3b3';
                if (rating < 700) rColor = '#66ff66';
                else if (rating < 1400) rColor = '#0088ff';
                else if (rating < 2000) rColor = '#ffd700';
                else rColor = '#e60000';

                const box = document.createElement('div');
                box.className = 'rating-box';
                box.innerHTML = `
                    <div class="rating-box-swatch" style="background: ${rColor}; box-shadow: 0 0 15px ${rColor}55"></div>
                    <div class="rating-box-mode">${mode}</div>
                    <div class="rating-box-config">${board} | ${formatTimeShort(time)}</div>
                    <div class="rating-box-value" style="color: ${rColor}">${rating}</div>
                `;
                grid.appendChild(box);
            });
        });
    });
}

function setupProfileEditing(isOwner) {
    const editableFields = [
        { id: 'profile-full-name', key: 'full_name', placeholder: 'Full Name' },
        { id: 'profile-age-val', key: 'age', placeholder: 'Age' },
        { id: 'profile-gender-val', key: 'gender', placeholder: 'Gender' },
        { id: 'profile-location-val', key: 'location', placeholder: 'Location' },
        { id: 'profile-quote-val', key: 'quote', placeholder: 'Enter a personal quote' }
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
                if (e.key === 'Enter') {
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
    } catch (err) {
        console.error("WOTD fetch failed:", err);
        displayEl.innerText = 'Offline';
    }
}
