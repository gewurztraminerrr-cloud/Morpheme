/* Forum Module for Morpheme */

window.emojiToCountryCode = function(emoji) {
    if (!emoji) return '';
    if (emoji.length === 2 && /^[A-Z]{2}$/i.test(emoji)) {
        return emoji.toUpperCase();
    }
    const letters = [];
    for (const char of emoji) {
        const codePoint = char.codePointAt(0);
        if (codePoint >= 0x1F1E6 && codePoint <= 0x1F1FF) {
            letters.push(String.fromCharCode(codePoint - 0x1F1E6 + 65));
        }
    }
    if (letters.length === 2) {
        return letters.join('');
    }
    return '';
};

window.getFlagHtml = function(flag, extraStyles = '') {
    if (!flag || flag === '🏳️') {
        return flag || '';
    }
    const code = window.emojiToCountryCode(flag);
    if (code && code.length === 2) {
        return `<img src="https://flagcdn.com/w40/${code.toLowerCase()}.png" class="flag-icon-img" alt="${code}" title="${code}" style="width: 1.25em; height: auto; display: inline-block; vertical-align: middle; margin-right: 4px; margin-left: 0; border-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.2); ${extraStyles}">`;
    }
    return flag;
};

// Full Country Flag List (ISO 3166-1)
window.ALL_FLAGS = [
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
    { code: 'ZW', flag: '🇿🇼', name: 'Zimbabwe' }
];

const parseUTCTimestamp = (isoStr) => {
    if (!isoStr) return new Date();
    if (typeof isoStr === 'number') return new Date(isoStr);
    const dateStr = isoStr.includes('Z') || isoStr.includes('+') ? isoStr.replace(' ', 'T') : isoStr.replace(' ', 'T') + 'Z';
    return new Date(dateStr);
};

const Forum = {
    categories: [],
    currentCategoryId: null,
    currentPostId: null,
    selectedPostFiles: [],
    selectedCommentFiles: [],
    initialized: false,
    friends: [],
    friendsLoaded: false,
    activeMentionState: null,

    init: async function () {
        if (this.initialized) return;
        console.log("[Forum] Initializing forum module...");
        this.setupEventListeners();
        this.loadFriends();
        await this.loadCategories();
        this.resetToEmptyState();
        this.initialized = true;

        // Auto-refresh categories every 30s while the forum is open to show new posts from others
        setInterval(() => {
            if (document.getElementById('page-forums').classList.contains('active')) {
                this.loadCategories();
            }
        }, 30000);

        // Periodic heartbeat every 20s while viewing forums
        setInterval(() => {
            this.sendHeartbeat();
        }, 20000);

        // Immediate heartbeat when returning from background tab
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                const forumPage = document.getElementById('page-forums');
                if (forumPage && forumPage.classList.contains('active')) {
                    this.sendHeartbeat();
                }
            }
        });

        // Leave beacon on page unload
        window.addEventListener('pagehide', () => {
            const forumPage = document.getElementById('page-forums');
            if (forumPage && forumPage.classList.contains('active')) {
                this.onLeave();
            }
        });
        window.addEventListener('beforeunload', () => {
            const forumPage = document.getElementById('page-forums');
            if (forumPage && forumPage.classList.contains('active')) {
                this.onLeave();
            }
        });

        // Mobile Layout snapping on navigation
        const forumPage = document.getElementById('page-forums');
        if (forumPage) {
            const observer = new MutationObserver(() => {
                if (forumPage.classList.contains('active')) {
                    const isMobile = (window.innerWidth <= 820) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
                    if (isMobile) {
                        setTimeout(() => {
                            const sidebar = document.querySelector('.forum-sidebar');
                            if (sidebar) sidebar.scrollIntoView({ behavior: 'auto', inline: 'start' });
                        }, 100);
                    }
                }
            });
            observer.observe(forumPage, {
                attributes: true,
                attributeFilter: ['class']
            });
        }

        // Mobile touch swipe handling for sliding back to categories
        const forumMain = document.querySelector('.forum-main');
        const forumSidebar = document.querySelector('.forum-sidebar');
        if (forumMain && forumSidebar) {
            let touchStartX = 0;
            let touchStartY = 0;
            forumMain.addEventListener('touchstart', (e) => {
                touchStartX = e.changedTouches[0].screenX;
                touchStartY = e.changedTouches[0].screenY;
            }, { passive: true });
            
            forumMain.addEventListener('touchend', (e) => {
                const touchEndX = e.changedTouches[0].screenX;
                const touchEndY = e.changedTouches[0].screenY;
                const diffX = touchEndX - touchStartX;
                const diffY = touchEndY - touchStartY;
                
                // If swiped right (diffX > 80) and horizontal movement was dominant
                if (diffX > 80 && Math.abs(diffX) > Math.abs(diffY)) {
                    forumSidebar.scrollIntoView({ behavior: 'smooth', inline: 'start' });
                }
            }, { passive: true });
        }
    },

    setupEventListeners: function () {
        // Mobile Categories back button
        const mobileBackBtn = document.getElementById('forum-mobile-back-btn');
        if (mobileBackBtn) {
            mobileBackBtn.addEventListener('click', () => {
                const sidebar = document.querySelector('.forum-sidebar');
                if (sidebar) sidebar.scrollIntoView({ behavior: 'smooth', inline: 'start' });
            });
        }

        // Fixed bottom Back button on mobile thread list
        const categoryBackBtn = document.getElementById('forum-category-back-btn');
        if (categoryBackBtn) {
            categoryBackBtn.addEventListener('click', () => {
                const container = document.querySelector('#page-forums .forum-container');
                if (container) {
                    container.scrollTo({ left: 0, behavior: 'smooth' });
                }
                const sidebar = document.querySelector('.forum-sidebar');
                if (sidebar) sidebar.scrollIntoView({ behavior: 'smooth', inline: 'start' });
            });
        }

        // New post button
        const newPostBtn = document.getElementById('forum-new-post-btn');
        if (newPostBtn) {
            newPostBtn.addEventListener('click', () => this.showCreateView());
        }

        // Refresh posts button
        const refreshBtn = document.getElementById('forum-refresh-posts-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', async () => {
                const icon = refreshBtn.querySelector('.refresh-icon');
                if (icon) {
                    icon.style.transition = 'transform 0.5s ease-in-out';
                    icon.style.transform = 'rotate(360deg)';
                }
                refreshBtn.style.opacity = '0.7';
                
                if (this.currentCategoryId === 'responders') {
                    await this.loadRespondersFeed();
                } else if (this.currentCategoryId) {
                    await this.loadPosts(this.currentCategoryId);
                } else {
                    const username = document.getElementById('forum-user-search-input').value.trim();
                    if (username) {
                        await this.handleUserSearch();
                    }
                }
                
                setTimeout(() => {
                    if (icon) {
                        icon.style.transition = 'none';
                        icon.style.transform = '';
                    }
                    refreshBtn.style.opacity = '1';
                }, 500);
            });
        }

        // Refresh thread comments button
        const commentsRefreshBtn = document.getElementById('forum-refresh-comments-btn');
        if (commentsRefreshBtn) {
            commentsRefreshBtn.addEventListener('click', () => this.refreshCurrentThread(commentsRefreshBtn));
        }

        // Jump to Bookmark button
        const jumpBookmarkBtn = document.getElementById('forum-jump-bookmark-btn');
        if (jumpBookmarkBtn) {
            jumpBookmarkBtn.addEventListener('click', () => this.jumpToBookmark());
        }

        // Scroll to Bottom button
        const scrollBottomBtn = document.getElementById('forum-scroll-bottom-btn');
        if (scrollBottomBtn) {
            scrollBottomBtn.addEventListener('click', () => this.scrollToBottom());
        }

        // Back to list button
        const backToListBtn = document.getElementById('forum-back-to-list');
        if (backToListBtn) {
            backToListBtn.addEventListener('click', () => this.showListView());
        }

        // Cancel create button
        const cancelCreateBtn = document.getElementById('forum-cancel-create');
        if (cancelCreateBtn) {
            cancelCreateBtn.addEventListener('click', () => this.showListView());
        }

        // Post create form
        const postForm = document.getElementById('forum-post-form');
        if (postForm) {
            postForm.addEventListener('submit', (e) => this.handlePostSubmit(e));
        }

        // Dynamic character counters for Forum inputs
        const postTitleInput = document.getElementById('forum-post-title');
        const postTitleCounter = document.getElementById('forum-post-title-counter');
        const postContentInput = document.getElementById('forum-post-content');
        const postContentCounter = document.getElementById('forum-post-content-counter');
        const commentInput = document.getElementById('forum-comment-input');
        const commentCounter = document.getElementById('forum-comment-counter');

        const updateCounter = (input, counter, max) => {
            if (!input || !counter) return;
            const remaining = Math.max(0, max - (input.value || '').length);
            counter.textContent = `${remaining} remaining`;
            counter.style.color = (remaining === 0) ? '#f43f5e' : (remaining <= (max * 0.1) ? '#fbbf24' : '');
        };
        this.updateCounter = updateCounter;

        if (postTitleInput && postTitleCounter) {
            postTitleInput.addEventListener('input', () => updateCounter(postTitleInput, postTitleCounter, 100));
        }
        if (postContentInput && postContentCounter) {
            postContentInput.addEventListener('input', () => updateCounter(postContentInput, postContentCounter, 2000));
        }
        if (commentInput && commentCounter) {
            commentInput.addEventListener('input', () => updateCounter(commentInput, commentCounter, 2000));
        }

        // Comment form controls
        const submitCommentBtn = document.getElementById('forum-submit-comment');
        if (submitCommentBtn) {
            submitCommentBtn.addEventListener('click', () => this.handleCommentSubmit());
        }

        const closeCommentBtn = document.getElementById('forum-comment-form-close');
        if (closeCommentBtn) {
            closeCommentBtn.addEventListener('click', () => {
                const formContainer = document.getElementById('forum-comment-form-container');
                if (formContainer) {
                    formContainer.classList.add('hidden');
                    formContainer.classList.remove('minimized');
                }
            });
        }

        const minCommentBtn = document.getElementById('forum-comment-form-minimize');
        if (minCommentBtn) {
            minCommentBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const formContainer = document.getElementById('forum-comment-form-container');
                if (formContainer) {
                    const isMin = formContainer.classList.toggle('minimized');
                    minCommentBtn.textContent = isMin ? '▢' : '—';
                }
            });
        }

        const cancelCommentBtn = document.getElementById('forum-cancel-comment');
        if (cancelCommentBtn) {
            cancelCommentBtn.addEventListener('click', () => {
                const formContainer = document.getElementById('forum-comment-form-container');
                if (formContainer) {
                    formContainer.classList.add('hidden');
                    formContainer.classList.remove('minimized');
                }
            });
        }

        const commentFormHeader = document.getElementById('forum-comment-form-header');
        if (commentFormHeader) {
            commentFormHeader.addEventListener('click', (e) => {
                if (e.target.closest('.forum-window-btn')) return;
                const formContainer = document.getElementById('forum-comment-form-container');
                if (formContainer && formContainer.classList.contains('minimized')) {
                    formContainer.classList.remove('minimized');
                    if (minCommentBtn) minCommentBtn.textContent = '—';
                }
            });
        }

        // Image wrappers and inputs
        const postImageInput = document.getElementById('forum-post-image');
        const postImageWrapper = document.getElementById('forum-post-image-wrapper');
        if (postImageInput) {
            postImageInput.addEventListener('click', () => {
                try { postImageInput.value = ''; } catch (e) { }
            });
            postImageInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files.length > 0) {
                    this.addFiles('post', Array.from(e.target.files));
                }
            });
        }
        if (postImageWrapper && postImageInput) {
            postImageWrapper.addEventListener('click', (e) => {
                if (e.target !== postImageInput && !e.target.closest('label[for="forum-post-image"]')) {
                    postImageInput.click();
                }
            });
            postImageWrapper.addEventListener('dragover', (e) => {
                e.preventDefault();
                const box = postImageWrapper.querySelector('.file-upload-box') || postImageWrapper.querySelector('label') || postImageWrapper.firstElementChild;
                if (box) { box.style.borderColor = 'var(--accent-color)'; box.style.background = 'rgba(0,0,0,0.4)'; }
            });
            postImageWrapper.addEventListener('dragleave', () => {
                const box = postImageWrapper.querySelector('.file-upload-box') || postImageWrapper.querySelector('label') || postImageWrapper.firstElementChild;
                if (box) { box.style.borderColor = 'var(--input-border)'; box.style.background = 'rgba(0,0,0,0.2)'; }
            });
            postImageWrapper.addEventListener('drop', (e) => {
                e.preventDefault();
                const box = postImageWrapper.querySelector('.file-upload-box') || postImageWrapper.querySelector('label') || postImageWrapper.firstElementChild;
                if (box) { box.style.borderColor = 'var(--input-border)'; box.style.background = 'rgba(0,0,0,0.2)'; }
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    this.addFiles('post', Array.from(e.dataTransfer.files));
                }
            });
        }

        const commentImageInput = document.getElementById('forum-comment-image');
        const commentImageWrapper = document.getElementById('forum-comment-image-wrapper');
        if (commentImageInput) {
            commentImageInput.addEventListener('click', () => {
                try { commentImageInput.value = ''; } catch (e) { }
            });
            commentImageInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files.length > 0) {
                    this.addFiles('comment', Array.from(e.target.files));
                }
            });
        }
        if (commentImageWrapper && commentImageInput) {
            commentImageWrapper.addEventListener('click', (e) => {
                if (e.target !== commentImageInput && !e.target.closest('label[for="forum-comment-image"]')) {
                    commentImageInput.click();
                }
            });
            commentImageWrapper.addEventListener('dragover', (e) => {
                e.preventDefault();
                const box = commentImageWrapper.querySelector('.file-upload-box') || commentImageWrapper.querySelector('label') || commentImageWrapper.firstElementChild;
                if (box) { box.style.borderColor = 'var(--accent-color)'; box.style.background = 'rgba(0,0,0,0.4)'; }
            });
            commentImageWrapper.addEventListener('dragleave', () => {
                const box = commentImageWrapper.querySelector('.file-upload-box') || commentImageWrapper.querySelector('label') || commentImageWrapper.firstElementChild;
                if (box) { box.style.borderColor = 'var(--input-border)'; box.style.background = 'rgba(0,0,0,0.2)'; }
            });
            commentImageWrapper.addEventListener('drop', (e) => {
                e.preventDefault();
                const box = commentImageWrapper.querySelector('.file-upload-box') || commentImageWrapper.querySelector('label') || commentImageWrapper.firstElementChild;
                if (box) { box.style.borderColor = 'var(--input-border)'; box.style.background = 'rgba(0,0,0,0.2)'; }
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    this.addFiles('comment', Array.from(e.dataTransfer.files));
                }
            });
        }

        // User search
        const searchBtn = document.getElementById('forum-user-search-btn');
        const searchInput = document.getElementById('forum-user-search-input');
        if (searchBtn && searchInput) {
            searchBtn.addEventListener('click', () => this.handleUserSearch());
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleUserSearch();
            });
        }

        // Setup mention autocomplete
        this.initMentionAutocomplete();
    },

    addFiles: function (type, files) {
        const targetArray = type === 'post' ? this.selectedPostFiles : this.selectedCommentFiles;
        let addedAny = false;
        const fileList = Array.from(files || []);
        for (let i = 0; i < fileList.length; i++) {
            const file = fileList[i];
            const isImage = (file.type && file.type.startsWith('image/')) ||
                /\.(jpe?g|png|gif|webp|bmp|svg|heic|heif|avif|ico|tiff?)$/i.test(file.name || '');
            if (!isImage) continue;
            if (targetArray.length >= 4) {
                alert("You can attach a maximum of 4 images per post.");
                break;
            }
            if (!targetArray.some(f => f.name === file.name && f.size === file.size)) {
                targetArray.push(file);
                addedAny = true;
            }
        }
        if (addedAny || targetArray.length === 0) {
            this.renderImagePreviews(type);
        }
    },

    removeFile: function (type, index) {
        const targetArray = type === 'post' ? this.selectedPostFiles : this.selectedCommentFiles;
        if (index >= 0 && index < targetArray.length) {
            targetArray.splice(index, 1);
        }
        this.renderImagePreviews(type);
    },

    renderImagePreviews: function (type) {
        const targetArray = type === 'post' ? this.selectedPostFiles : this.selectedCommentFiles;
        const previewEl = document.getElementById(type === 'post' ? 'forum-image-preview' : 'forum-comment-image-preview');
        const wrapperEl = document.getElementById(type === 'post' ? 'forum-post-image-wrapper' : 'forum-comment-image-wrapper');
        
        if (wrapperEl) {
            const textSpan = wrapperEl.querySelector('.file-text');
            if (textSpan) {
                if (targetArray.length === 0) {
                    textSpan.textContent = type === 'post' 
                        ? 'Click to choose images (up to 4) or drag and drop' 
                        : '📎 Attach images (up to 4)';
                } else if (targetArray.length < 4) {
                    textSpan.textContent = `📎 Attached ${targetArray.length}/4 images (Click to add more)`;
                } else {
                    textSpan.textContent = `📎 Maximum 4 images selected`;
                }
            }
        }

        if (!previewEl) return;

        if (targetArray.length === 0) {
            previewEl.innerHTML = '';
            previewEl.classList.add('hidden');
            previewEl.style.display = 'none';
            return;
        }

        previewEl.classList.remove('hidden');
        previewEl.style.display = 'block';
        previewEl.innerHTML = `
            <div class="forum-image-preview-grid">
                ${targetArray.map((file, idx) => `
                    <div class="preview-item-wrapper" title="Click thumbnail to enlarge">
                        <img class="preview-thumb" id="preview-img-${type}-${idx}" alt="Preview ${idx+1}">
                        <button type="button" class="remove-preview-btn" data-type="${type}" data-index="${idx}" title="Remove image">✕</button>
                    </div>
                `).join('')}
            </div>
        `;

        targetArray.forEach((file, idx) => {
            const imgEl = document.getElementById(`preview-img-${type}-${idx}`);
            if (!imgEl) return;

            let objectUrl = null;
            try {
                if (typeof URL !== 'undefined' && URL.createObjectURL) {
                    objectUrl = URL.createObjectURL(file);
                    imgEl.src = objectUrl;
                }
            } catch (err) {
                console.warn("[Forum] URL.createObjectURL failed:", err);
            }

            const openLightbox = (srcUrl) => {
                if (typeof window.showImageLightbox === 'function') {
                    window.showImageLightbox(srcUrl, `Attachment Preview (${idx+1}/${targetArray.length}): ${file.name}`);
                }
            };

            if (objectUrl) {
                imgEl.addEventListener('click', (ev) => {
                    ev.stopPropagation();
                    openLightbox(objectUrl);
                });
            }

            // Fallback or secondary verification via FileReader
            if (!imgEl.src || imgEl.src === window.location.href) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    if (imgEl && e.target.result) {
                        imgEl.src = e.target.result;
                        imgEl.onclick = (ev) => {
                            ev.stopPropagation();
                            openLightbox(e.target.result);
                        };
                    }
                };
                reader.onerror = (e) => {
                    console.error("[Forum] FileReader error on preview:", e);
                };
                try {
                    reader.readAsDataURL(file);
                } catch (e) {
                    console.error("[Forum] readAsDataURL failed:", e);
                }
            }
        });

        previewEl.querySelectorAll('.remove-preview-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const t = btn.getAttribute('data-type');
                const idx = parseInt(btn.getAttribute('data-index'), 10);
                this.removeFile(t, idx);
            });
        });
    },

    getViewerId: function () {
        if (!this._viewerId) {
            let vid = null;
            try { vid = sessionStorage.getItem('morpheme_fvid'); } catch (e) {}
            if (!vid) {
                vid = 'fv_' + Math.random().toString(36).substring(2, 11) + Date.now().toString(36);
                try { sessionStorage.setItem('morpheme_fvid', vid); } catch (e) {}
            }
            this._viewerId = vid;
        }
        return this._viewerId;
    },

    updateViewersBadge: function (count) {
        const badge = document.getElementById('forum-viewers-badge');
        if (!badge) return;
        const c = (count !== undefined && count !== null) ? parseInt(count, 10) : 1;
        const displayCount = (isNaN(c) || c < 1) ? 1 : c;
        badge.textContent = `${displayCount} viewing`;
        badge.title = `${displayCount} ${displayCount === 1 ? 'person' : 'people'} viewing the Forums`;
        if (displayCount > 0) {
            badge.classList.add('has-posts');
        } else {
            badge.classList.remove('has-posts');
        }
    },

    sendHeartbeat: async function () {
        const forumPage = document.getElementById('page-forums');
        if (!forumPage || !forumPage.classList.contains('active') || document.hidden) return;
        try {
            const vid = this.getViewerId();
            const res = await fetch(`/api/forum/heartbeat?viewer_id=${encodeURIComponent(vid)}`);
            if (res.ok) {
                const data = await res.json();
                if (data && data.viewers_count !== undefined) {
                    this.updateViewersBadge(data.viewers_count);
                }
            }
        } catch (err) {
            // Heartbeat failure is non-fatal
        }
    },

    onLeave: function () {
        try {
            const vid = this.getViewerId();
            if (navigator.sendBeacon) {
                navigator.sendBeacon(`/api/forum/leave?viewer_id=${encodeURIComponent(vid)}`);
            } else {
                fetch(`/api/forum/leave?viewer_id=${encodeURIComponent(vid)}`, { method: 'POST', keepalive: true });
            }
        } catch (e) {}
    },

    loadFriends: async function (force = false) {
        if (this.friendsLoaded && !force && this.friends && this.friends.length > 0) {
            return this.friends;
        }
        if (this._loadingFriendsPromise) {
            return this._loadingFriendsPromise;
        }

        this._loadingFriendsPromise = (async () => {
            try {
                const res = await fetch('/api/friends/list');
                if (res.ok) {
                    const data = await res.json();
                    this.friends = data.friends || [];
                    this.friendsLoaded = true;
                    return this.friends;
                } else if (res.status === 401) {
                    // Not authenticated yet
                    this.friends = [];
                    this.friendsLoaded = false;
                    return [];
                }
            } catch (e) {
                console.warn('[Forum] Failed to load friends for autocomplete:', e);
            } finally {
                this._loadingFriendsPromise = null;
            }
            return this.friends || [];
        })();

        return this._loadingFriendsPromise;
    },

    initMentionAutocomplete: function () {
        const targetIds = ['forum-post-content', 'forum-comment-input', 'forum-post-title'];
        targetIds.forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;

            el.addEventListener('input', () => this.handleMentionInput(el));
            el.addEventListener('keyup', (e) => {
                if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp' && e.key !== 'Enter' && e.key !== 'Escape') {
                    this.handleMentionInput(el);
                }
            });
            el.addEventListener('keydown', (e) => this.handleMentionKeydown(e, el));
            el.addEventListener('click', () => this.handleMentionInput(el));
            el.addEventListener('touchend', () => {
                setTimeout(() => this.handleMentionInput(el), 60);
            });
        });

        // Track cursor/selection changes on mobile virtual keyboards
        document.addEventListener('selectionchange', () => {
            const active = document.activeElement;
            if (active && targetIds.includes(active.id)) {
                this.handleMentionInput(active);
            }
        });

        // Close mention popup when clicking or tapping outside
        const handleOutside = (e) => {
            const popup = document.getElementById('forum-mention-popup');
            if (!popup || popup.classList.contains('hidden')) return;
            if (e.target && e.target.closest && e.target.closest('#forum-mention-popup')) return;
            if (targetIds.some(id => e.target && e.target.closest && e.target.closest(`#${id}`))) return;
            this.hideMentionPopup();
        };
        document.addEventListener('click', handleOutside);
        document.addEventListener('touchstart', handleOutside, { passive: true });

        // Reposition on window resize, scroll, or visualViewport change
        const reposition = () => {
            if (this.activeMentionState && this.activeMentionState.inputEl) {
                this.positionMentionPopup(this.activeMentionState.inputEl);
            }
        };
        window.addEventListener('resize', reposition);
        window.addEventListener('scroll', reposition, true);
        if (window.visualViewport) {
            window.visualViewport.addEventListener('resize', reposition);
            window.visualViewport.addEventListener('scroll', reposition);
        }
    },

    handleMentionInput: function (inputEl) {
        if (!inputEl) return;
        const val = inputEl.value;
        const cursor = inputEl.selectionStart;
        if (cursor === undefined || cursor === null) {
            this.hideMentionPopup();
            return;
        }

        const textBefore = val.slice(0, cursor);
        // Match @ followed by username characters immediately before the cursor
        // Supports typing @ at start of input, after whitespace, or punctuation
        const match = textBefore.match(/(?:^|[\s(<\'"([{])@([a-zA-Z0-9_]*)$/);
        if (!match) {
            this.hideMentionPopup();
            return;
        }

        const query = match[1] || '';
        const matchFull = match[0];
        const atRelativePos = matchFull.indexOf('@');
        const atIndex = (textBefore.length - matchFull.length) + atRelativePos;

        // If friends not loaded yet, or empty and we haven't checked recently, fetch from server
        if (!this.friendsLoaded || !this.friends || this.friends.length === 0) {
            this.loadFriends(true).then((friends) => {
                if (document.activeElement === inputEl) {
                    this.handleMentionInput(inputEl);
                }
            });
            if (!this.friends || this.friends.length === 0) {
                return;
            }
        }

        const queryLower = query.toLowerCase();
        const matches = this.friends.filter(f => {
            const u = (f.username || '').toLowerCase();
            return u.startsWith(queryLower);
        });

        if (matches.length === 0) {
            // When query is non-empty and no friends match, hide popup so user can tag someone not on their friend list
            if (query.length > 0) {
                this.hideMentionPopup();
                return;
            }
            // If user just typed "@" and has 0 friends on their list, show friendly empty notice
            this.renderMentionEmpty(inputEl, atIndex, query);
            return;
        }

        this.renderMentionPopup(matches, inputEl, atIndex, query);
    },

    renderMentionEmpty: function (inputEl, atIndex, query) {
        const popup = document.getElementById('forum-mention-popup');
        if (!popup) return;

        this.activeMentionState = {
            inputEl: inputEl,
            atIndex: atIndex,
            query: query,
            matches: [],
            selectedIndex: -1
        };

        popup.innerHTML = `
            <div class="forum-mention-header">
                <span>Friends (0)</span>
            </div>
            <div class="forum-mention-empty">
                No friends found on your list.<br>Type username to mention anyone.
            </div>
        `;

        this.positionMentionPopup(inputEl);
        popup.classList.remove('hidden');
    },

    handleMentionKeydown: function (e, inputEl) {
        const popup = document.getElementById('forum-mention-popup');
        if (!popup || popup.classList.contains('hidden') || !this.activeMentionState) {
            return;
        }

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            e.stopPropagation();
            this.updateMentionSelection(this.activeMentionState.selectedIndex + 1);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            e.stopPropagation();
            this.updateMentionSelection(this.activeMentionState.selectedIndex - 1);
        } else if (e.key === 'Enter' || e.key === 'Tab') {
            if (this.activeMentionState.matches && this.activeMentionState.matches.length > 0) {
                e.preventDefault();
                e.stopPropagation();
                const selected = this.activeMentionState.matches[this.activeMentionState.selectedIndex];
                if (selected) {
                    this.selectMentionFriend(selected);
                }
            }
        } else if (e.key === 'Escape') {
            e.preventDefault();
            e.stopPropagation();
            this.hideMentionPopup();
        }
    },

    renderMentionPopup: function (matches, inputEl, atIndex, query) {
        const popup = document.getElementById('forum-mention-popup');
        if (!popup) return;

        this.activeMentionState = {
            inputEl: inputEl,
            atIndex: atIndex,
            query: query,
            matches: matches,
            selectedIndex: 0
        };

        const itemsHtml = matches.map((friend, idx) => {
            const uname = friend.username || '';
            const queryLen = query.length;
            const matchedPart = uname.slice(0, queryLen);
            const remainingPart = uname.slice(queryLen);
            const nameHtml = `<strong>${this.escapeHtml(matchedPart)}</strong>${this.escapeHtml(remainingPart)}`;

            const initial = uname ? uname[0].toUpperCase() : '?';
            const avatarStyle = friend.avatar_url ? `style="background-image: url('${friend.avatar_url}');"` : '';
            const flagHtml = window.getFlagHtml ? window.getFlagHtml(friend.country_flag) : (friend.country_flag || '');
            const onlineDot = friend.is_online ? '<span class="forum-mention-status-dot" title="Online"></span>' : '';

            return `
                <div class="forum-mention-item ${idx === 0 ? 'active' : ''}" data-index="${idx}" role="option">
                    <div class="forum-mention-avatar" ${avatarStyle}>${friend.avatar_url ? '' : initial}</div>
                    ${flagHtml ? `<span class="forum-mention-flag">${flagHtml}</span>` : ''}
                    <span class="forum-mention-name">${nameHtml}</span>
                    ${onlineDot}
                </div>
            `;
        }).join('');

        popup.innerHTML = `
            <div class="forum-mention-header">
                <span>Friends (${matches.length})</span>
            </div>
            <div class="forum-mention-list">
                ${itemsHtml}
            </div>
        `;

        let touchStartY = 0;
        let isTouchScroll = false;

        const itemEls = popup.querySelectorAll('.forum-mention-item');
        itemEls.forEach((el, idx) => {
            el.addEventListener('touchstart', (e) => {
                if (e.touches && e.touches[0]) {
                    touchStartY = e.touches[0].clientY;
                    isTouchScroll = false;
                }
            }, { passive: true });

            el.addEventListener('touchmove', (e) => {
                if (e.touches && e.touches[0]) {
                    if (Math.abs(e.touches[0].clientY - touchStartY) > 10) {
                        isTouchScroll = true;
                    }
                }
            }, { passive: true });

            el.addEventListener('touchend', (e) => {
                if (isTouchScroll) return;
                e.preventDefault();
                e.stopPropagation();
                this.selectMentionFriend(matches[idx]);
            });

            el.addEventListener('mousedown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.selectMentionFriend(matches[idx]);
            });

            el.addEventListener('mouseenter', () => {
                this.activeMentionState.selectedIndex = idx;
                itemEls.forEach((it, i) => it.classList.toggle('active', i === idx));
            });
        });

        this.positionMentionPopup(inputEl);
        popup.classList.remove('hidden');
    },

    updateMentionSelection: function (newIndex) {
        if (!this.activeMentionState || !this.activeMentionState.matches || this.activeMentionState.matches.length === 0) return;
        const count = this.activeMentionState.matches.length;
        const validIndex = ((newIndex % count) + count) % count;
        this.activeMentionState.selectedIndex = validIndex;

        const popup = document.getElementById('forum-mention-popup');
        if (!popup) return;
        const items = popup.querySelectorAll('.forum-mention-item');
        items.forEach((item, idx) => {
            const isActive = (idx === validIndex);
            item.classList.toggle('active', isActive);
            if (isActive) {
                item.scrollIntoView({ block: 'nearest' });
            }
        });
    },

    selectMentionFriend: function (friend) {
        if (!this.activeMentionState || !friend) return;
        const { inputEl, atIndex, query } = this.activeMentionState;
        const val = inputEl.value || '';
        const cur = (inputEl.selectionStart !== undefined && inputEl.selectionStart !== null) 
            ? inputEl.selectionStart 
            : (atIndex + 1 + query.length);

        const beforeAt = val.slice(0, atIndex);
        const afterToken = val.slice(cur);
        const insertion = `@${friend.username} `;

        inputEl.value = beforeAt + insertion + afterToken;
        const newCursorPos = beforeAt.length + insertion.length;
        if (typeof inputEl.setSelectionRange === 'function') {
            inputEl.setSelectionRange(newCursorPos, newCursorPos);
        }
        inputEl.focus();

        // Trigger input event to update character counters
        inputEl.dispatchEvent(new Event('input', { bubbles: true }));

        this.hideMentionPopup();
    },

    positionMentionPopup: function (inputEl) {
        const popup = document.getElementById('forum-mention-popup');
        if (!popup || !inputEl) return;

        // Guarantee popup is a direct child of document.body so it isn't clipped by any ancestor overflow
        if (popup.parentElement !== document.body) {
            document.body.appendChild(popup);
        }

        const rect = inputEl.getBoundingClientRect();
        const vv = window.visualViewport;
        const vHeight = vv ? vv.height : window.innerHeight;
        const vWidth = vv ? vv.width : window.innerWidth;
        const vTop = vv ? vv.offsetTop : 0;
        const vLeft = vv ? vv.offsetLeft : 0;

        const margin = 6;
        const desiredHeight = 220;

        // Calculate visible space inside visual viewport (above and below virtual keyboard)
        const spaceBelow = (vTop + vHeight) - rect.bottom - margin;
        const spaceAbove = rect.top - vTop - margin;

        let top;
        let maxHeight;

        if (spaceBelow >= 150) {
            top = rect.bottom + margin;
            maxHeight = Math.min(desiredHeight, spaceBelow);
        } else if (spaceAbove >= 110) {
            maxHeight = Math.min(desiredHeight, spaceAbove);
            top = rect.top - margin - maxHeight;
        } else {
            // Very tight screen height
            if (spaceBelow >= spaceAbove) {
                top = rect.bottom + margin;
                maxHeight = Math.max(90, spaceBelow);
            } else {
                maxHeight = Math.max(90, spaceAbove);
                top = Math.max(vTop + 6, rect.top - margin - maxHeight);
            }
        }

        const popupWidth = Math.min(280, vWidth - 20);
        let left = rect.left;
        if (left + popupWidth > vLeft + vWidth - 10) {
            left = vLeft + vWidth - popupWidth - 10;
        }
        if (left < vLeft + 10) {
            left = vLeft + 10;
        }

        popup.style.position = 'fixed';
        popup.style.top = `${Math.round(top)}px`;
        popup.style.left = `${Math.round(left)}px`;
        popup.style.width = `${popupWidth}px`;
        popup.style.maxHeight = `${Math.round(maxHeight)}px`;
        popup.style.zIndex = '2147483640';
    },

    hideMentionPopup: function () {
        const popup = document.getElementById('forum-mention-popup');
        if (popup) {
            popup.classList.add('hidden');
        }
        this.activeMentionState = null;
    },

    loadCategories: async function () {
        try {
            // Calculate 12:00 AM (midnight) of today in the user's local/profile time
            const now = new Date();
            const midnight = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0);
            const sinceIso = midnight.toISOString();
            const vid = this.getViewerId();

            const promises = [
                fetch(`/api/forum/categories?since=${encodeURIComponent(sinceIso)}&viewer_id=${encodeURIComponent(vid)}`).then(r => r.json())
            ];
            if (window.currentUser && !window.currentUserIsGuest) {
                promises.push(
                    fetch('/api/forum/responders/status')
                        .then(r => r.json())
                        .catch(() => ({ has_new: false }))
                );
            } else {
                promises.push(Promise.resolve({ has_new: false }));
            }

            const [catData, respData] = await Promise.all(promises);
            this.categories = catData.categories;
            this.hasNewResponders = respData && respData.has_new;
            if (catData && catData.viewers_count !== undefined) {
                this.updateViewersBadge(catData.viewers_count);
            }
            this.renderCategories();
        } catch (err) {
            console.error("[Forum] Failed to load categories:", err);
        }
    },

    renderCategories: function () {
        const listEl = document.getElementById('forum-categories-list');
        if (!listEl) return;

        let respondersHtml = '';
        if (window.currentUser && !window.currentUserIsGuest) {
            const isRespActive = (this.currentCategoryId === 'responders');
            const hasNewResp = this.hasNewResponders;
            respondersHtml = `
                <div class="forum-cat-item forum-responders-tab ${isRespActive ? 'active' : ''} ${hasNewResp ? 'has-new' : ''}" data-id="responders">
                    <span class="forum-cat-name">
                        <span class="forum-cat-name-text">View Responses</span>
                    </span>
                    <span class="forum-cat-desc">Recent replies to you across the forum</span>
                </div>
            `;
        }

        const categoriesHtml = this.categories.map(cat => {
            const isActive = (this.currentCategoryId === cat.id);
            const count = parseInt(cat.posts_today_count || 0, 10);
            const countLabel = count === 1 ? '1 post today' : `${count} posts today`;

            return `
                <div class="forum-cat-item ${isActive ? 'active' : ''}" data-id="${cat.id}">
                    <span class="forum-cat-name">
                        <span class="forum-cat-name-text">${cat.name}</span>
                        <span class="forum-cat-today-badge ${count > 0 ? 'has-posts' : ''}" title="${countLabel} since 12AM">${countLabel}</span>
                    </span>
                    <span class="forum-cat-desc">${cat.description}</span>
                </div>
            `;
        }).join('');

        const respContainer = document.getElementById('forum-responders-container');
        if (respContainer) {
            respContainer.innerHTML = respondersHtml;
            const respItem = respContainer.querySelector('.forum-cat-item');
            if (respItem) {
                respItem.addEventListener('click', () => {
                    this.selectRespondersCategory();
                });
            }
            listEl.innerHTML = categoriesHtml;
        } else {
            listEl.innerHTML = respondersHtml + categoriesHtml;
        }

        // Attach listeners
        listEl.querySelectorAll('.forum-cat-item').forEach(item => {
            item.addEventListener('click', () => {
                const idAttr = item.getAttribute('data-id');
                if (idAttr === 'responders') {
                    this.selectRespondersCategory();
                } else {
                    const catId = parseInt(idAttr, 10);
                    this.selectCategory(catId);
                }
            });
        });
    },

    selectRespondersCategory: async function () {
        this.currentCategoryId = 'responders';

        // Update UI
        document.querySelectorAll('.forum-cat-item').forEach(item => {
            const isThis = item.getAttribute('data-id') === 'responders';
            item.classList.toggle('active', isThis);
        });

        // Notify server that user viewed responders
        try {
            await fetch('/api/forum/responders/view', { method: 'POST' });
        } catch (e) {
            console.error('[Forum] View responders error:', e);
        }

        const titleEl = document.getElementById('forum-category-title');
        if (titleEl) titleEl.textContent = 'View Responses';

        const descEl = document.getElementById('forum-category-desc');
        if (descEl) {
            descEl.textContent = 'Recent replies to your posts across the Forum';
            descEl.classList.remove('forum-desc-scrolling-box');
        }

        const newPostBtn = document.getElementById('forum-new-post-btn');
        if (newPostBtn) newPostBtn.classList.add('hidden');

        await this.loadRespondersFeed();
        this.showListView();

        const isMobile = (window.innerWidth <= 820) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
        if (isMobile) {
            const forumMain = document.querySelector('.forum-main');
            if (forumMain) {
                forumMain.scrollIntoView({ behavior: 'smooth', inline: 'start' });
            }
        }
    },

    formatDateGroupHeader: function (dateObj) {
        const today = new Date();
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);

        const isSameDay = (d1, d2) => (
            d1.getFullYear() === d2.getFullYear() &&
            d1.getMonth() === d2.getMonth() &&
            d1.getDate() === d2.getDate()
        );

        if (isSameDay(dateObj, today)) {
            return 'Today';
        } else if (isSameDay(dateObj, yesterday)) {
            return 'Yesterday';
        } else {
            return dateObj.toLocaleDateString(undefined, {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
                year: 'numeric'
            });
        }
    },

    loadRespondersFeed: async function () {
        const postsList = document.getElementById('forum-posts-list');
        if (!postsList) return;

        postsList.innerHTML = '<div class="forum-cat-loading" style="padding: 24px; text-align: center;">Loading responses...</div>';

        try {
            const res = await fetch('/api/forum/responders');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            const responders = data.responders || [];

            const hasUnclicked = responders.some(r => (r.is_clicked === 0 || r.is_clicked === '0' || !r.is_clicked));
            this.hasNewResponders = hasUnclicked;
            const respTab = document.querySelector('.forum-responders-tab');
            if (respTab) {
                if (hasUnclicked) respTab.classList.add('has-new');
                else respTab.classList.remove('has-new');
            }
            const btnForums = document.getElementById('btn-forums');
            if (btnForums) {
                if (hasUnclicked) btnForums.classList.add('has-new');
                else btnForums.classList.remove('has-new');
            }

            if (responders.length === 0) {
                postsList.innerHTML = `
                    <div class="forum-placeholder">
                        <div class="placeholder-icon">📬</div>
                        <h3>No replies yet</h3>
                        <p>When another user responds to your posts across the Forum, their replies will appear here.</p>
                    </div>
                `;
                return;
            }

            // Group responses by calendar date
            const groups = [];
            let currentGroupKey = null;
            let currentGroup = null;

            responders.forEach(r => {
                const parsedDate = parseUTCTimestamp(r.timestamp);
                const groupKey = `${parsedDate.getFullYear()}-${parsedDate.getMonth()}-${parsedDate.getDate()}`;
                if (groupKey !== currentGroupKey) {
                    currentGroupKey = groupKey;
                    currentGroup = {
                        title: this.formatDateGroupHeader(parsedDate),
                        items: []
                    };
                    groups.push(currentGroup);
                }
                currentGroup.items.push(r);
            });

            postsList.innerHTML = groups.map(group => {
                const cardsHtml = group.items.map(r => {
                    const dateStr = typeof window.formatAppDate === 'function' ? window.formatAppDate(r.timestamp, true) : r.timestamp;
                    const flagHtml = window.getFlagHtml ? window.getFlagHtml(r.responder_flag) : (r.responder_flag || '');
                    const isUnclicked = (r.is_clicked === 0 || r.is_clicked === '0' || !r.is_clicked);

                    return `
                        <div class="forum-post-card responder-card ${isUnclicked ? 'responder-card-gold' : ''}" data-responder-id="${r.id}" data-post-id="${r.post_id}" data-comment-id="${r.comment_id || ''}" data-post-number="${r.post_number}">
                            <div class="post-card-header">
                                <span class="post-card-title responder-title">
                                    ${this.escapeHtml(r.category_name)} 
                                    <span class="forum-user-clickable" data-username="${this.escapeHtml(r.recipient_username)}" title="View ${this.escapeHtml(r.recipient_username)}'s Profile">@${this.escapeHtml(r.recipient_username)}</span>
                                    #${r.post_number} by 
                                    <span class="forum-user-clickable" data-username="${this.escapeHtml(r.responder_username)}" title="View ${this.escapeHtml(r.responder_username)}'s Profile">${flagHtml}<strong>${this.escapeHtml(r.responder_username)}</strong></span>
                                </span>
                                <span class="post-card-meta">
                                    <span>${dateStr}</span>
                                </span>
                            </div>
                            <div class="post-card-excerpt">${this.escapeHtml(r.content)}</div>
                        </div>
                    `;
                }).join('');

                return `
                    <div class="forum-responses-date-group">
                        <div class="forum-responses-date-header">
                            <span>${this.escapeHtml(group.title)}</span>
                        </div>
                        <div class="forum-responses-date-cards">
                            ${cardsHtml}
                        </div>
                    </div>
                `;
            }).join('');

            // Attach user-clickable listeners inside responder cards
            postsList.querySelectorAll('.responder-card .forum-user-clickable').forEach(el => {
                el.addEventListener('click', (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    const uname = el.getAttribute('data-username');
                    Forum.openMiniProfile(uname, e);
                });
            });

            // Attach listeners to responder cards
            postsList.querySelectorAll('.responder-card').forEach(card => {
                card.addEventListener('click', async (e) => {
                    // Ignore clicks on user links
                    if (e.target.closest('.forum-user-clickable')) return;

                    const responderId = card.getAttribute('data-responder-id');
                    const postId = parseInt(card.getAttribute('data-post-id'), 10);
                    const commentId = card.getAttribute('data-comment-id') ? parseInt(card.getAttribute('data-comment-id'), 10) : null;
                    const postNumber = parseInt(card.getAttribute('data-post-number'), 10);

                    if (card.classList.contains('responder-card-gold')) {
                        card.classList.remove('responder-card-gold');
                        try {
                            fetch(`/api/forum/responders/click/${responderId}`, { method: 'POST' });
                        } catch (e) {
                            console.error('[Forum] Click responder error:', e);
                        }
                    }

                    const remainingGoldCards = postsList.querySelectorAll('.responder-card-gold').length;
                    if (remainingGoldCards === 0) {
                        this.hasNewResponders = false;
                        const respTab = document.querySelector('.forum-responders-tab');
                        if (respTab) respTab.classList.remove('has-new');
                        const btnForums = document.getElementById('btn-forums');
                        if (btnForums) btnForums.classList.remove('has-new');
                    }

                    if (postId) {
                        await this.loadPostDetail(postId, {
                            highlightCommentId: commentId,
                            highlightPostNumber: postNumber
                        });
                    }
                });
            });

        } catch (err) {
            console.error("[Forum] Failed to load responses feed:", err);
            postsList.innerHTML = `
                <div class="forum-placeholder">
                    <div class="placeholder-icon">⚠️</div>
                    <h3>Failed to load responses</h3>
                    <p>Please try refreshing the page.</p>
                </div>
            `;
        }
    },

    selectCategory: async function (catId) {
        this.currentCategoryId = catId;
        const category = this.categories.find(c => c.id === catId);

        // Update UI
        document.querySelectorAll('.forum-cat-item').forEach(item => {
            const isThisCat = parseInt(item.getAttribute('data-id')) === catId;
            item.classList.toggle('active', isThisCat);
        });

        // Immediately update global nav button status
        if (typeof window.checkForumActivity === 'function') {
            window.checkForumActivity();
        }

        document.getElementById('forum-category-title').textContent = category.name;
        const isSuggestions = Boolean(category.name && (
            category.name.toLowerCase().includes('suggestion') ||
            category.id === 6
        ));
        const descEl = document.getElementById('forum-category-desc');
        const longSuggestionsDesc = "Share your ideas for improving Morpheme! You can like 👍 or dislike 👎 suggestions and replies to show how favorable an idea is. A decision of the mods will be made based on the level of its popularity, and favorable features may be added in the future.";
        if (descEl) {
            descEl.textContent = isSuggestions ? longSuggestionsDesc : (category.header_description || category.description);
            descEl.classList.toggle('forum-desc-scrolling-box', isSuggestions);
            if (isSuggestions) {
                descEl.scrollTop = 0;
            }
        }

        // Show/hide New Post button based on guest status
        // restriction: guests cannot post
        const isGuest = window.currentUserIsGuest || (window.currentUser === null);
        let hideNewPost = isGuest;
        
        // Restriction: Only moderators can post in the News category
        if (category.name === "News" && !window.currentUserIsMod) {
            hideNewPost = true;
        }
        
        document.getElementById('forum-new-post-btn').classList.toggle('hidden', hideNewPost);

        await this.loadPosts(catId);
        this.showListView();

        const isMobile = (window.innerWidth <= 820) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
        if (isMobile) {
            const forumMain = document.querySelector('.forum-main');
            if (forumMain) {
                forumMain.scrollIntoView({ behavior: 'smooth', inline: 'start' });
            }
        }
    },

    loadPosts: async function (catId) {
        const postsList = document.getElementById('forum-posts-list');
        postsList.innerHTML = '<div class="forum-placeholder"><h3>Loading posts...</h3></div>';

        try {
            const response = await fetch(`/api/forum/posts/${catId}`);
            const data = await response.json();
            this.renderPosts(data.posts);
        } catch (err) {
            console.error("[Forum] Failed to load posts:", err);
            postsList.innerHTML = '<div class="forum-placeholder"><h3>Error loading posts.</h3></div>';
        }
    },

    handleUserSearch: async function () {
        const username = document.getElementById('forum-user-search-input').value.trim();
        if (!username) return;

        console.log(`[Forum] Searching posts for user: ${username}`);

        // Clear active category
        document.querySelectorAll('.forum-cat-item').forEach(item => item.classList.remove('active'));
        this.currentCategoryId = null;

        // Update UI Header
        document.getElementById('forum-category-title').textContent = `Posts by ${username}`;
        const descEl = document.getElementById('forum-category-desc');
        descEl.textContent = `Viewing all forum contributions from ${username}.`;
        descEl.classList.remove('forum-desc-scrolling-box');
        document.getElementById('forum-new-post-btn').classList.add('hidden');

        const postsList = document.getElementById('forum-posts-list');
        postsList.innerHTML = '<div class="forum-placeholder"><h3>Searching...</h3></div>';

        try {
            const response = await fetch(`/api/forum/posts/user/${encodeURIComponent(username)}`);
            const data = await response.json();

            if (data.posts && data.posts.length > 0) {
                this.renderPosts(data.posts);
            } else {
                postsList.innerHTML = `
                    <div class="forum-placeholder">
                        <div class="placeholder-icon">🔍</div>
                        <h3>No posts found</h3>
                        <p>User "${username}" has not posted anything yet.</p>
                    </div>
                `;
            }
            this.showListView();

            const isMobile = (window.innerWidth <= 820) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
            if (isMobile) {
                const forumMain = document.querySelector('.forum-main');
                if (forumMain) {
                    forumMain.scrollIntoView({ behavior: 'smooth', inline: 'start' });
                }
            }
        } catch (err) {
            console.error("[Forum] User search error:", err);
            postsList.innerHTML = '<div class="forum-placeholder"><h3>Error performing search.</h3></div>';
        }
    },

    renderPosts: function (posts) {
        const postsList = document.getElementById('forum-posts-list');

        if (posts.length === 0) {
            postsList.innerHTML = `
                <div class="forum-placeholder">
                    <div class="placeholder-icon">📭</div>
                    <h3>No threads yet</h3>
                    <p>Be the first to start a conversation in this category!</p>
                </div>
            `;
            return;
        }

        postsList.innerHTML = posts.map(post => {
            const dateStr = typeof window.formatAppDate === 'function' ? window.formatAppDate(post.timestamp, true) : post.timestamp;
            const isComment = post.type === 'comment';
            const postId = post.post_id || post.id;
            const hasImages = (post.image_url || (post.image_urls && post.image_urls.length > 0));
            const numBadge = post.post_number ? `<span class="forum-post-number-badge">#${post.post_number}</span>` : '';
            const pinnedBadge = post.is_pinned ? `<span class="forum-pinned-badge">📌 PINNED</span>` : '';
            const isClosed = Boolean(post.is_closed || (!isComment && (1 + (post.comment_count || 0)) >= 500));
            const closedBadge = isClosed ? `<span class="forum-closed-badge">🔒 CLOSED</span>` : '';
            
            return `
                <div class="forum-post-card ${post.is_pinned ? 'is-pinned' : ''}" data-id="${postId}">
                    <div class="post-card-header">
                        <span class="post-card-title">${pinnedBadge}${closedBadge}${numBadge}${isComment ? 'Re: ' : ''}${this.escapeHtml(post.title)}</span>
                        <span class="post-card-meta">
                            <span>${isComment ? 'Replied' : 'Posted'} by <strong class="forum-user-clickable" data-username="${this.escapeHtml(post.username)}" title="View ${this.escapeHtml(post.username)}'s Profile">${window.getFlagHtml ? window.getFlagHtml(post.country_flag) : (post.country_flag || '')}${this.escapeHtml(post.username)}</strong></span>
                            <span>${dateStr}</span>
                        </span>
                    </div>
                    <div class="post-card-excerpt">${this.escapeHtml(post.content)}</div>
                    <div class="post-stats">
                        ${isComment ? '' : `<div class="stat-item">💬 ${post.comment_count} comments</div>`}
                        <div class="stat-item stat-reaction-item" title="${post.likes || 0} likes">👍 ${post.likes || 0}</div>
                        <div class="stat-item stat-reaction-item" title="${post.dislikes || 0} dislikes">👎 ${post.dislikes || 0}</div>
                        ${hasImages ? '<div class="stat-item">🖼️ Includes images</div>' : ''}
                    </div>
                </div>
            `;
        }).join('');

        // Attach user-clickable listeners inside post cards
        postsList.querySelectorAll('.forum-post-card .forum-user-clickable').forEach(el => {
            el.addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();
                const uname = el.getAttribute('data-username');
                Forum.openMiniProfile(uname, e);
            });
        });

        // Attach listeners
        postsList.querySelectorAll('.forum-post-card').forEach(card => {
            card.addEventListener('click', (e) => {
                // Ignore clicks on user links
                if (e.target.closest('.forum-user-clickable')) return;
                const postId = parseInt(card.getAttribute('data-id'));
                this.loadPostDetail(postId);
            });
        });
    },

    loadPostDetail: async function (postId, options = null) {
        this.currentPostId = postId;
        try {
            const response = await fetch(`/api/forum/post/${postId}`, { cache: 'no-store' });
            const data = await response.json();
            this.currentPostLookup = data.post_lookup || {};
            this.renderPostDetail(data.post, data.comments, false, data.user_bookmark);
            this.showPostView();

            if (options && (options.highlightCommentId || options.highlightPostNumber)) {
                setTimeout(() => {
                    this.highlightTargetPost(options);
                }, 150);
            }
        } catch (err) {
            console.error("[Forum] Failed to load post detail:", err);
        }
    },

    highlightTargetPost: function (options) {
        let targetEl = null;

        if (options.highlightCommentId) {
            targetEl = document.getElementById(`forum-comment-${options.highlightCommentId}`);
        }
        if (!targetEl && options.highlightPostNumber) {
            targetEl = document.querySelector(`[data-post-number="${options.highlightPostNumber}"]`);
        }

        if (targetEl) {
            targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            targetEl.classList.remove('forum-item-highlight-gold');
            void targetEl.offsetWidth; // Force reflow
            targetEl.classList.add('forum-item-highlight-gold');
            setTimeout(() => {
                targetEl.classList.remove('forum-item-highlight-gold');
            }, 3600);
        }
    },

    closeResponsesPopup: function () {
        const existing = document.getElementById('forum-responses-popup');
        if (existing) existing.remove();
        if (this._responsesEscHandler) {
            document.removeEventListener('keydown', this._responsesEscHandler);
            this._responsesEscHandler = null;
        }
    },

    showResponsesPopup: function (postNumber) {
        this.closeResponsesPopup();
        const resps = [...((this.currentResponsesMap && this.currentResponsesMap[postNumber]) || [])]
            .sort((a, b) => parseUTCTimestamp(b.timestamp) - parseUTCTimestamp(a.timestamp));
        if (resps.length === 0) return;

        const overlay = document.createElement('div');
        overlay.id = 'forum-responses-popup';
        overlay.className = 'forum-responses-popup-overlay';

        const items = resps.map(c => {
            const cDate = typeof window.formatAppDate === 'function' ? window.formatAppDate(c.timestamp, true) : c.timestamp;
            const snippetRaw = (c.content || '').replace(/\s+/g, ' ').trim();
            const snippet = snippetRaw.length > 110 ? snippetRaw.slice(0, 110) + '…' : snippetRaw;
            return `
                <div class="forum-responses-popup-item" data-comment-id="${c.id}" data-post-number="${c.post_number || ''}" role="button" tabindex="0">
                    <div class="forum-responses-popup-item-head">
                        <strong>${this.escapeHtml(c.username)}</strong>
                        <span class="forum-post-number-badge">#${c.post_number || '?'}</span>
                        <span class="forum-responses-popup-date">${cDate}</span>
                    </div>
                    <div class="forum-responses-popup-snippet">${this.escapeHtml(snippet).replace(/((?:@|&amp;)[A-Za-z0-9_]+\s*#\d+(?:\s*\(Thread\))?)/g, '<span class="forum-mention-tag">$1</span>')}</div>
                </div>`;
        }).join('');

        overlay.innerHTML = `
            <div class="forum-responses-popup" role="dialog" aria-label="Responses to #${postNumber}">
                <div class="forum-responses-popup-header">
                    <span>Responses to #${postNumber} (${resps.length})</span>
                    <button type="button" class="forum-responses-popup-close" aria-label="Close">&times;</button>
                </div>
                <div class="forum-responses-popup-list">${items}</div>
            </div>`;
        document.body.appendChild(overlay);

        const close = () => this.closeResponsesPopup();
        overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
        overlay.querySelector('.forum-responses-popup-close').addEventListener('click', close);
        this._responsesEscHandler = (e) => { if (e.key === 'Escape') close(); };
        document.addEventListener('keydown', this._responsesEscHandler);

        overlay.querySelectorAll('.forum-responses-popup-item').forEach(item => {
            const go = () => {
                const commentId = parseInt(item.getAttribute('data-comment-id'), 10);
                const pnum = parseInt(item.getAttribute('data-post-number'), 10);
                close();
                this.highlightTargetPost({
                    highlightCommentId: commentId,
                    highlightPostNumber: isNaN(pnum) ? null : pnum
                });
            };
            item.addEventListener('click', go);
            item.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); }
            });
        });
    },

    refreshCurrentThread: async function (btn) {
        if (!this.currentPostId) return;
        const icon = btn ? btn.querySelector('.refresh-icon') : null;
        if (icon) {
            icon.style.transition = 'transform 0.5s ease-in-out';
            icon.style.transform = 'rotate(360deg)';
        }
        if (btn) btn.style.opacity = '0.7';

        try {
            const response = await fetch(`/api/forum/post/${this.currentPostId}`, { cache: 'no-store' });
            if (response.ok) {
                const data = await response.json();
                this.currentPostLookup = data.post_lookup || {};
                this.renderPostDetail(data.post, data.comments, true, data.user_bookmark);
            }
        } catch (err) {
            console.error("[Forum] Failed to refresh thread:", err);
        }

        setTimeout(() => {
            if (icon) {
                icon.style.transition = 'none';
                icon.style.transform = '';
            }
            if (btn) btn.style.opacity = '1';
        }, 500);
    },

    handlePostDelete: async function (postId) {
        if (!confirm("Are you sure you want to PERMANENTLY delete this thread and ALL of its comments? This cannot be undone.")) {
            return;
        }

        try {
            const response = await fetch(`/api/forum/post/delete/${postId}`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success) {
                await this.loadCategories();
                await this.selectCategory(this.currentCategoryId);
                this.showListView();
            } else {
                alert(data.error || "Failed to delete post.");
            }
        } catch (err) {
            console.error("[Forum] Post delete error:", err);
            alert("Failed to delete post.");
        }
    },

    handlePostPin: async function (postId) {
        try {
            const response = await fetch(`/api/forum/post/pin/${postId}`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success) {
                // Refresh post detail view to reflect new pinned status
                await this.loadPostDetail(postId);
            } else {
                alert(data.error || "Failed to update thread pin status.");
            }
        } catch (err) {
            console.error("[Forum] Post pin error:", err);
            alert("Failed to update thread pin status.");
        }
    },

    handleCommentDelete: async function (commentId) {
        if (!confirm("Delete this comment permanently?")) return;

        try {
            const response = await fetch(`/api/forum/comment/delete/${commentId}`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success) {
                await this.loadPostDetail(this.currentPostId);
            } else {
                alert(data.error || "Failed to delete comment.");
            }
        } catch (err) {
            console.error("[Forum] Comment delete error:", err);
            alert("Failed to delete comment.");
        }
    },

    renderPostDetail: function (post, comments, preserveDraft = false, userBookmark = null) {
        if (!post) return;
        comments = comments || [];

        if (!preserveDraft) {
            const commentInput = document.getElementById('forum-comment-input');
            if (commentInput) commentInput.value = '';
            this.selectedCommentFiles = [];
            this.renderImagePreviews('comment');
        }

        const detailEl = document.getElementById('forum-post-detail');
        const commentsListEl = document.getElementById('forum-comments-list');
        const dateStr = typeof window.formatAppDate === 'function' ? window.formatAppDate(post.timestamp, true) : post.timestamp;

        const postUrls = post.image_urls || (post.image_url ? [post.image_url] : []);
        let postImagesHtml = '';
        if (postUrls.length > 0) {
            postImagesHtml = `
                <div class="post-images-grid grid-count-${postUrls.length}">
                    ${postUrls.map((url, idx) => `
                        <div class="post-image-item">
                            <img src="${url}" class="post-image forum-lightbox-trigger" data-url="${url}" data-caption="${this.escapeHtml(post.title)} by ${this.escapeHtml(post.username)} (${idx+1}/${postUrls.length})" alt="Post attachment ${idx+1}" style="cursor: pointer;">
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Map each post number -> comments that respond to it (via "@user #N" references)
        this.currentResponsesMap = {};
        const respRegex = /[@&][A-Za-z0-9_]+\s*#(\d+)/g;
        comments.forEach(c => {
            const seen = new Set();
            let m;
            respRegex.lastIndex = 0;
            while ((m = respRegex.exec(c.content || '')) !== null) {
                const n = parseInt(m[1], 10);
                if (seen.has(n) || n === c.post_number) continue;
                seen.add(n);
                (this.currentResponsesMap[n] = this.currentResponsesMap[n] || []).push(c);
            }
        });
        const buildNumBadge = (num) => {
            if (!num) return '';
            const resps = this.currentResponsesMap[num] || [];
            if (resps.length === 0) return `<span class="forum-post-number-badge">#${num}</span>`;
            return `<span class="forum-post-number-badge has-responses" data-resp-num="${num}" role="button" tabindex="0" title="View ${resps.length} ${resps.length === 1 ? 'response' : 'responses'} to #${num}">#${num} <span class="forum-resp-count">💬 ${resps.length}</span></span>`;
        };

        const isClosed = Boolean((post && post.is_closed) || (1 + comments.length) >= 500);
        const closedBadge = isClosed ? `<span class="forum-closed-badge">🔒 CLOSED</span>` : '';
        const postNumBadge = buildNumBadge(post.post_number);
        const pinnedBadge = post.is_pinned ? `<span class="forum-pinned-badge">📌 PINNED</span>` : '';

        const isPostBookmarked = Boolean(
            this.currentThreadBookmark &&
            (!this.currentThreadBookmark.target_comment_id) &&
            (this.currentThreadBookmark.target_post_number === (post.post_number || 1))
        );

        this.currentThreadBookmark = (userBookmark !== undefined && userBookmark !== null)
            ? userBookmark
            : (this.getGuestBookmark(post.id) || null);
        detailEl.innerHTML = `
            <div id="forum-post-root" data-post-number="${post.post_number || 1}">
                <div class="post-detail-header">
                    <div class="post-author-box">
                        <div class="post-author-meta" style="display: flex; align-items: center; gap: 10px; min-width: 0;">
                            <div class="author-avatar forum-user-clickable" data-username="${this.escapeHtml(post.username)}" title="View ${this.escapeHtml(post.username)}'s Profile">${post.username[0].toUpperCase()}</div>
                            <div class="author-info">
                                <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                                    <span class="author-name forum-user-clickable" data-username="${this.escapeHtml(post.username)}" title="View ${this.escapeHtml(post.username)}'s Profile">${window.getFlagHtml ? window.getFlagHtml(post.country_flag) : (post.country_flag || '')}${this.escapeHtml(post.username)}</span>
                                    ${postNumBadge}
                                </div>
                                <span class="post-date">${dateStr}</span>
                            </div>
                        </div>
                        <div class="post-header-actions" style="margin-left: auto; display: flex; align-items: center; gap: 8px;">
                            <div class="forum-reaction-group" data-post-id="${post.id}" data-comment-id="0">
                                <button type="button" class="forum-reaction-btn forum-like-btn ${post.user_reaction === 'like' ? 'active' : ''}" data-reaction="like" title="Like">
                                    <span class="reaction-icon">👍</span>
                                    <span class="reaction-count">${post.likes || 0}</span>
                                </button>
                                <button type="button" class="forum-reaction-btn forum-dislike-btn ${post.user_reaction === 'dislike' ? 'active' : ''}" data-reaction="dislike" title="Dislike">
                                    <span class="reaction-icon">👎</span>
                                    <span class="reaction-count">${post.dislikes || 0}</span>
                                </button>
                            </div>
                            <button type="button" class="forum-bookmark-post-btn ${isPostBookmarked ? 'active' : ''}" data-post-id="${post.id}" data-comment-id="" data-post-number="${post.post_number || 1}" title="${isPostBookmarked ? 'Remove bookmark' : 'Bookmark this post'}">🔖 <span class="bm-label">${isPostBookmarked ? 'Bookmarked' : 'Bookmark'}</span></button>
                            ${!isClosed ? `<button class="forum-reply-btn forum-post-reply-btn" data-username="${this.escapeHtml(post.username)}" data-post-number="${post.post_number || 1}">↩ Reply</button>` : ''}
                        </div>
                    </div>
                    <h1 class="post-detail-title">${pinnedBadge}${closedBadge}${this.escapeHtml(post.title)}</h1>
                </div>
                <div class="post-content">${this.renderContentWithLinks(post.content)}</div>
                ${postImagesHtml}
            </div>
        `;

        // Static Moderator Action buttons in HTML — show for mods, hide for others
        const modContainer = document.getElementById('forum-delete-post-container');
        const deleteBtn = document.getElementById('forum-delete-post-btn');
        const pinBtn = document.getElementById('forum-pin-post-btn');

        if (modContainer) {
            modContainer.style.display = window.currentUserIsMod ? 'flex' : 'none';
        }

        if (pinBtn) {
            const isPinned = Boolean(post.is_pinned);
            pinBtn.textContent = isPinned ? '📍 Unpin Thread' : '📌 Pin Thread';
            const newPinBtn = pinBtn.cloneNode(true);
            pinBtn.parentNode.replaceChild(newPinBtn, pinBtn);
            newPinBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.handlePostPin(post.id);
            });
        }

        if (deleteBtn) {
            // Remove old listeners by cloning
            const newBtn = deleteBtn.cloneNode(true);
            deleteBtn.parentNode.replaceChild(newBtn, deleteBtn);
            newBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.handlePostDelete(post.id);
            });
        }

        const countEl = document.getElementById('forum-comment-count');
        if (countEl) {
            countEl.textContent = `${comments.length} ${comments.length === 1 ? 'comment' : 'comments'}`;
        }

        const sortedComments = [...comments].sort((a, b) => parseUTCTimestamp(b.timestamp) - parseUTCTimestamp(a.timestamp));

        if (commentsListEl) {
            if (sortedComments.length === 0) {
                commentsListEl.innerHTML = '<p class="forum-placeholder">No comments yet. Start the discussion!</p>';
            } else {
                commentsListEl.innerHTML = sortedComments.map(c => {
                    const cDate = typeof window.formatAppDate === 'function' ? window.formatAppDate(c.timestamp, true) : c.timestamp;
                    const cUrls = c.image_urls || (c.image_url ? [c.image_url] : []);
                    let cImagesHtml = '';
                    if (cUrls.length > 0) {
                        cImagesHtml = `
                            <div class="comment-images-grid grid-count-${cUrls.length}">
                                ${cUrls.map((url, idx) => `
                                    <div class="comment-image-item">
                                        <img src="${url}" class="forum-lightbox-trigger" data-url="${url}" data-caption="Reply by ${this.escapeHtml(c.username)} (${idx+1}/${cUrls.length})" alt="Comment attachment ${idx+1}" style="cursor: pointer;">
                                    </div>
                                `).join('')}
                            </div>
                        `;
                    }

                    const commentNumBadge = buildNumBadge(c.post_number);

                    const isCommentBookmarked = Boolean(
                        this.currentThreadBookmark &&
                        (this.currentThreadBookmark.target_comment_id === c.id ||
                         (!this.currentThreadBookmark.target_comment_id && this.currentThreadBookmark.target_post_number === c.post_number))
                    );

                    return `
                        <div class="forum-comment" id="forum-comment-${c.id}" data-comment-id="${c.id}" data-post-number="${c.post_number}">
                            <div class="comment-header">
                                <div class="comment-author-box">
                                    <div class="comment-avatar forum-user-clickable" data-username="${this.escapeHtml(c.username)}" title="View ${this.escapeHtml(c.username)}'s Profile">${c.username[0].toUpperCase()}</div>
                                    <div class="comment-author-info">
                                        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                                            <span class="comment-author forum-user-clickable" data-username="${this.escapeHtml(c.username)}" title="View ${this.escapeHtml(c.username)}'s Profile">${window.getFlagHtml ? window.getFlagHtml(c.country_flag) : (c.country_flag || '')}${this.escapeHtml(c.username)}</span>
                                            ${commentNumBadge}
                                        </div>
                                        <span class="comment-date">${cDate}</span>
                                    </div>
                                </div>
                                <div class="comment-header-actions" style="margin-left: auto; display: flex; align-items: center; gap: 8px;">
                                    <div class="forum-reaction-group" data-post-id="${post.id}" data-comment-id="${c.id}">
                                        <button type="button" class="forum-reaction-btn forum-like-btn ${c.user_reaction === 'like' ? 'active' : ''}" data-reaction="like" title="Like">
                                            <span class="reaction-icon">👍</span>
                                            <span class="reaction-count">${c.likes || 0}</span>
                                        </button>
                                        <button type="button" class="forum-reaction-btn forum-dislike-btn ${c.user_reaction === 'dislike' ? 'active' : ''}" data-reaction="dislike" title="Dislike">
                                            <span class="reaction-icon">👎</span>
                                            <span class="reaction-count">${c.dislikes || 0}</span>
                                        </button>
                                    </div>
                                    <button type="button" class="forum-bookmark-post-btn ${isCommentBookmarked ? 'active' : ''}" data-post-id="${post.id}" data-comment-id="${c.id}" data-post-number="${c.post_number}" title="${isCommentBookmarked ? 'Remove bookmark' : 'Bookmark this reply'}">🔖 <span class="bm-label">${isCommentBookmarked ? 'Bookmarked' : 'Bookmark'}</span></button>
                                    ${!isClosed ? `<button class="forum-reply-btn forum-comment-reply-btn" data-username="${this.escapeHtml(c.username)}" data-post-number="${c.post_number || post.post_number || 1}">↩ Reply</button>` : ''}
                                    ${window.currentUserIsMod ? `
                                        <button class="forum-comment-delete-btn" data-id="${c.id}" style="margin-left: 4px; background: none; border: none; color: #f43f5e; cursor: pointer; font-size: 0.75rem; opacity: 0.6;">Delete</button>
                                    ` : ''}
                                </div>
                            </div>
                            <div class="comment-content">${this.renderContentWithLinks(c.content)}</div>
                            ${cImagesHtml}
                        </div>
                    `;
                }).join('');

                commentsListEl.querySelectorAll('.forum-comment-delete-btn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const commentId = parseInt(btn.getAttribute('data-id'));
                        this.handleCommentDelete(commentId);
                    });
                });
            }
        }

        // Attach listeners for user-clickable links in post detail & comments
        const detailUserClickables = [
            ...detailEl.querySelectorAll('.post-author-box .forum-user-clickable'),
            ...detailEl.querySelectorAll('.post-content .forum-user-clickable'),
            ...commentsListEl.querySelectorAll('.forum-user-clickable')
        ];
        detailUserClickables.forEach(el => {
            el.addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();
                const uname = el.getAttribute('data-username');
                Forum.openMiniProfile(uname, e);
            });
        });

        // Attach listeners for reply buttons
        const allReplyBtns = [
            ...detailEl.querySelectorAll('.forum-reply-btn'),
            ...commentsListEl.querySelectorAll('.forum-reply-btn')
        ];
        allReplyBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const uname = btn.getAttribute('data-username');
                const pnum = btn.getAttribute('data-post-number');
                const isThread = btn.classList.contains('forum-post-reply-btn');
                this.handleReplyToUser(uname, pnum, isThread);
            });
        });

        // Clicking a post number that has responses opens a popup listing them (newest first)
        [...detailEl.querySelectorAll('.forum-post-number-badge.has-responses'),
         ...commentsListEl.querySelectorAll('.forum-post-number-badge.has-responses')].forEach(badge => {
            const open = (e) => {
                e.stopPropagation();
                e.preventDefault();
                this.showResponsesPopup(parseInt(badge.getAttribute('data-resp-num'), 10));
            };
            badge.addEventListener('click', open);
            badge.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') open(e);
            });
        });

        const triggers = document.querySelectorAll('.forum-lightbox-trigger');
        triggers.forEach(img => {
            img.addEventListener('click', () => {
                const url = img.getAttribute('data-url');
                const caption = img.getAttribute('data-caption');
                if (typeof window.showImageLightbox === 'function') {
                    window.showImageLightbox(url, caption);
                }
            });
        });

        // Closed thread banner & comment form visibility
        const commentFormContainer = document.getElementById('forum-comment-form-container');
        let closedBanner = document.getElementById('forum-thread-closed-banner');
        if (isClosed) {
            if (commentFormContainer) commentFormContainer.classList.add('hidden');
            if (!closedBanner) {
                closedBanner = document.createElement('div');
                closedBanner.id = 'forum-thread-closed-banner';
                closedBanner.className = 'forum-thread-closed-banner';
                closedBanner.innerHTML = '🔒 This thread has reached the maximum of 500 posts and is closed to further replies.';
                const commentsSec = document.querySelector('.forum-comments-section');
                if (commentsSec) {
                    commentsSec.insertBefore(closedBanner, document.querySelector('.comments-header'));
                }
            } else {
                closedBanner.style.display = 'flex';
            }
        } else {
            if (closedBanner) closedBanner.style.display = 'none';
            const isGuest = window.currentUserIsGuest || (window.currentUser === null);
            if (commentFormContainer) {
                commentFormContainer.classList.toggle('hidden', isGuest);
            }
        }

        // Attach listeners for bookmark buttons
        const allBmBtns = [
            ...detailEl.querySelectorAll('.forum-bookmark-post-btn'),
            ...commentsListEl.querySelectorAll('.forum-bookmark-post-btn')
        ];
        allBmBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const pId = parseInt(btn.getAttribute('data-post-id'), 10);
                const pNum = parseInt(btn.getAttribute('data-post-number'), 10);
                const cIdRaw = btn.getAttribute('data-comment-id');
                const cId = cIdRaw ? parseInt(cIdRaw, 10) : null;
                this.handleBookmarkClick(pId, pNum, cId);
            });
        });

        // Attach listeners for reaction buttons (likes / dislikes)
        const allReactionBtns = [
            ...detailEl.querySelectorAll('.forum-reaction-btn'),
            ...commentsListEl.querySelectorAll('.forum-reaction-btn')
        ];
        allReactionBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const group = btn.closest('.forum-reaction-group');
                if (!group) return;
                const pId = parseInt(group.getAttribute('data-post-id'), 10);
                const cId = parseInt(group.getAttribute('data-comment-id') || '0', 10);
                const reactionType = btn.getAttribute('data-reaction');
                this.handleReactionClick(pId, cId, reactionType, group);
            });
        });
    },

    handleReactionClick: async function (postId, commentId, reactionType, groupEl) {
        const isGuest = window.currentUserIsGuest || (window.currentUser === null);
        if (isGuest) {
            this.showToast('Please log in or register to like or dislike posts.');
            return;
        }

        if (groupEl && groupEl._reacting) return;
        if (groupEl) groupEl._reacting = true;

        try {
            const res = await fetch(`/api/forum/post/${postId}/react`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    comment_id: commentId || 0,
                    reaction: reactionType
                })
            });
            const data = await res.json();
            if (res.ok && data.success) {
                if (groupEl) {
                    const likeBtn = groupEl.querySelector('.forum-like-btn');
                    const dislikeBtn = groupEl.querySelector('.forum-dislike-btn');
                    if (likeBtn) {
                        likeBtn.classList.toggle('active', data.user_reaction === 'like');
                        const countEl = likeBtn.querySelector('.reaction-count');
                        if (countEl) countEl.textContent = data.likes || 0;
                    }
                    if (dislikeBtn) {
                        dislikeBtn.classList.toggle('active', data.user_reaction === 'dislike');
                        const countEl = dislikeBtn.querySelector('.reaction-count');
                        if (countEl) countEl.textContent = data.dislikes || 0;
                    }
                }
            } else {
                this.showToast(data.error || 'Failed to update reaction');
            }
        } catch (e) {
            console.error('Error updating reaction:', e);
            this.showToast('Failed to update reaction');
        } finally {
            if (groupEl) groupEl._reacting = false;
        }
    },

    getGuestBookmark: function (postId) {
        try {
            const raw = localStorage.getItem('morpheme_bm_' + postId);
            return raw ? JSON.parse(raw) : null;
        } catch (e) {
            return null;
        }
    },

    setGuestBookmark: function (postId, bm) {
        try {
            if (bm) {
                localStorage.setItem('morpheme_bm_' + postId, JSON.stringify(bm));
            } else {
                localStorage.removeItem('morpheme_bm_' + postId);
            }
        } catch (e) {}
    },

    handleBookmarkClick: async function (postId, postNumber, commentId) {
        const isCurrent = Boolean(
            this.currentThreadBookmark &&
            this.currentThreadBookmark.target_post_number === postNumber &&
            ((this.currentThreadBookmark.target_comment_id || null) === (commentId || null))
        );

        if (!window.currentUserId || window.isGuestUser) {
            // Guest mode via localStorage
            if (isCurrent) {
                this.currentThreadBookmark = null;
                this.setGuestBookmark(postId, null);
                this.updateBookmarkButtonStates();
                this.showToast('Bookmark removed');
            } else {
                this.currentThreadBookmark = { target_post_number: postNumber, target_comment_id: commentId || null };
                this.setGuestBookmark(postId, this.currentThreadBookmark);
                this.updateBookmarkButtonStates();
                this.showToast('Bookmark saved');
            }
            return;
        }

        try {
            const res = await fetch(`/api/forum/post/${postId}/bookmark`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target_post_number: postNumber,
                    target_comment_id: commentId || null
                })
            });
            const data = await res.json();
            if (res.ok && data.success) {
                if (data.bookmarked) {
                    this.currentThreadBookmark = {
                        target_post_number: data.target_post_number,
                        target_comment_id: data.target_comment_id || null
                    };
                    this.showToast('Bookmark saved');
                } else {
                    this.currentThreadBookmark = null;
                    this.showToast('Bookmark removed');
                }
                this.updateBookmarkButtonStates();
            } else {
                this.showToast(data.error || 'Failed to update bookmark');
            }
        } catch (e) {
            console.error('Error updating bookmark:', e);
            this.showToast('Failed to update bookmark');
        }
    },

    updateBookmarkButtonStates: function () {
        const bm = this.currentThreadBookmark;
        document.querySelectorAll('.forum-bookmark-post-btn').forEach(btn => {
            const pnum = parseInt(btn.getAttribute('data-post-number'), 10);
            const cidRaw = btn.getAttribute('data-comment-id');
            const cid = cidRaw ? parseInt(cidRaw, 10) : null;

            let isActive = false;
            if (bm) {
                if (bm.target_comment_id && cid) {
                    isActive = (bm.target_comment_id === cid);
                } else if (!bm.target_comment_id && !cid) {
                    isActive = (bm.target_post_number === pnum);
                } else if (!bm.target_comment_id && bm.target_post_number === pnum) {
                    isActive = true;
                }
            }

            if (isActive) {
                btn.classList.add('active');
                const label = btn.querySelector('.bm-label');
                if (label) label.textContent = 'Bookmarked';
                btn.title = 'Remove bookmark';
            } else {
                btn.classList.remove('active');
                const label = btn.querySelector('.bm-label');
                if (label) label.textContent = 'Bookmark';
                btn.title = cid ? 'Bookmark this reply' : 'Bookmark this post';
            }
        });
    },

    jumpToBookmark: function () {
        const bm = this.currentThreadBookmark;
        if (!bm) {
            this.showToast('No bookmark saved for this thread yet. Click 🔖 Bookmark on any post to save your spot.');
            return;
        }

        let targetEl = null;
        if (bm.target_comment_id) {
            targetEl = document.getElementById('forum-comment-' + bm.target_comment_id);
        }
        if (!targetEl && bm.target_post_number) {
            targetEl = document.querySelector(`.forum-comment[data-post-number="${bm.target_post_number}"]`);
            if (!targetEl) {
                const root = document.getElementById('forum-post-root');
                if (root && parseInt(root.getAttribute('data-post-number'), 10) === bm.target_post_number) {
                    targetEl = root;
                }
            }
        }
        if (!targetEl && !bm.target_comment_id) {
            targetEl = document.getElementById('forum-post-root');
        }

        if (targetEl) {
            targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            targetEl.classList.remove('forum-highlight-pulse');
            void targetEl.offsetWidth; // Force reflow
            targetEl.classList.add('forum-highlight-pulse');
            setTimeout(() => {
                targetEl.classList.remove('forum-highlight-pulse');
            }, 2500);
        } else {
            this.showToast('Bookmarked post not found in thread.');
        }
    },

    scrollToBottom: function () {
        const scrollBody = document.querySelector('#forum-view-post .forum-view-scroll-body') || document.querySelector('.forum-view-scroll-body');
        const commentsList = document.getElementById('forum-comments-list');
        if (commentsList && commentsList.lastElementChild) {
            commentsList.lastElementChild.scrollIntoView({ behavior: 'smooth', block: 'end' });
        } else if (scrollBody) {
            scrollBody.scrollTo({ top: scrollBody.scrollHeight, behavior: 'smooth' });
        }
    },

    showToast: function (msg) {
        let toast = document.getElementById('forum-toast-notification');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'forum-toast-notification';
            toast.style.cssText = 'position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%); background: rgba(15, 23, 42, 0.95); color: #fff; padding: 10px 20px; border-radius: 24px; font-size: 0.86rem; font-weight: 600; box-shadow: 0 6px 20px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.2); z-index: 10000; pointer-events: none; transition: opacity 0.3s ease, transform 0.3s ease; opacity: 0;';
            document.body.appendChild(toast);
        }
        toast.textContent = msg;
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(-50%) translateY(0)';
        clearTimeout(this._toastTimeout);
        this._toastTimeout = setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(10px)';
        }, 2400);
    },

    handleReplyToUser: function (username, postNumber, isThread) {
        const isGuest = window.currentUserIsGuest || (window.currentUser === null);
        if (isGuest) {
            if (window.showAuthModal) {
                window.showAuthModal('login');
            } else {
                alert("Forum replies are restricted to registered members only. Please log in or register!");
            }
            return;
        }

        const formContainer = document.getElementById('forum-comment-form-container');
        if (formContainer) {
            formContainer.classList.remove('hidden');
            formContainer.classList.remove('minimized');
            const minBtn = document.getElementById('forum-comment-form-minimize');
            if (minBtn) minBtn.textContent = '—';
        }

        const input = document.getElementById('forum-comment-input');
        if (!input) return;

        const tag = `@${username} #${postNumber}\n`;
        const currentVal = input.value;
        const cleanTag = tag.trim();

        if (!currentVal.trim()) {
            input.value = tag;
        } else {
            // Check if input already ends with this exact mention tag to avoid duplicate insertion on rapid clicks
            if (!currentVal.trim().endsWith(cleanTag)) {
                input.value = currentVal.trim() + '\n\n' + tag;
            }
        }

        const isMobile = (window.innerWidth <= 820) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
        if (isMobile) {
            input.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
        input.scrollTop = input.scrollHeight;
        input.dispatchEvent(new Event('input', { bubbles: true }));
    },

    handlePostSubmit: async function (e) {
        e.preventDefault();
        const rawTitle = document.getElementById('forum-post-title').value;
        const rawContent = document.getElementById('forum-post-content').value;
        const catId = document.getElementById('forum-post-category-id').value;

        const title = (rawTitle || '').trim().slice(0, 100);
        const content = (rawContent || '').trim().slice(0, 2000);

        if (!title || !content) return;

        const submitPostBtn = document.querySelector('#forum-post-form button[type="submit"]');
        const originalBtnText = submitPostBtn ? submitPostBtn.textContent : 'Create Post';
        if (submitPostBtn) {
            submitPostBtn.disabled = true;
            submitPostBtn.textContent = 'Posting...';
        }

        const formData = new FormData();
        formData.append('category_id', catId);
        formData.append('title', title);
        formData.append('content', content);

        for (let imageFile of this.selectedPostFiles) {
            if (imageFile.type === 'image/gif') {
                if (imageFile.size > 2 * 1024 * 1024) {
                    alert(`GIF file "${imageFile.name}" must be under 2MB.`);
                    if (submitPostBtn) {
                        submitPostBtn.disabled = false;
                        submitPostBtn.textContent = originalBtnText;
                    }
                    return;
                }
                formData.append('images', imageFile);
            } else {
                try {
                    const compressed = await this.compressImage(imageFile, 1200, 0.8);
                    formData.append('images', compressed);
                } catch (err) {
                    console.error("[Forum] Compression failed, uploading original:", err);
                    formData.append('images', imageFile);
                }
            }
        }

        try {
            const response = await fetch('/api/forum/posts', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.success) {
                document.getElementById('forum-post-form').reset();
                if (this.updateCounter) {
                    this.updateCounter(document.getElementById('forum-post-title'), document.getElementById('forum-post-title-counter'), 100);
                    this.updateCounter(document.getElementById('forum-post-content'), document.getElementById('forum-post-content-counter'), 2000);
                }
                this.selectedPostFiles = [];
                this.renderImagePreviews('post');
                
                await this.loadCategories();
                await this.selectCategory(this.currentCategoryId);
            } else {
                alert(data.error || "Failed to create post.");
            }
        } catch (err) {
            console.error("[Forum] Post submit error:", err);
            alert("Failed to create post.");
        } finally {
            if (submitPostBtn) {
                submitPostBtn.disabled = false;
                submitPostBtn.textContent = originalBtnText;
            }
        }
    },

    handleCommentSubmit: async function () {
        const rawContent = document.getElementById('forum-comment-input').value;
        const content = (rawContent || '').trim().slice(0, 2000);

        if (!content) return;

        const submitCommentBtn = document.getElementById('forum-submit-comment');
        const originalBtnText = submitCommentBtn ? submitCommentBtn.textContent : 'Post Comment';
        if (submitCommentBtn) {
            submitCommentBtn.disabled = true;
            submitCommentBtn.textContent = 'Posting...';
        }

        const formData = new FormData();
        formData.append('post_id', this.currentPostId);
        formData.append('content', content);

        for (let imageFile of this.selectedCommentFiles) {
            if (imageFile.type === 'image/gif') {
                if (imageFile.size > 2 * 1024 * 1024) {
                    alert(`GIF file "${imageFile.name}" must be under 2MB.`);
                    if (submitCommentBtn) {
                        submitCommentBtn.disabled = false;
                        submitCommentBtn.textContent = originalBtnText;
                    }
                    return;
                }
                formData.append('images', imageFile);
            } else {
                try {
                    const compressed = await this.compressImage(imageFile, 1200, 0.8);
                    formData.append('images', compressed);
                } catch (err) {
                    console.error("[Forum] Compression failed, uploading original:", err);
                    formData.append('images', imageFile);
                }
            }
        }

        try {
            const response = await fetch('/api/forum/comments', {
                method: 'POST',
                body: formData // No Content-Type header needed for FormData
            });
            const data = await response.json();
            if (data.success) {
                document.getElementById('forum-comment-input').value = '';
                if (this.updateCounter) {
                    this.updateCounter(document.getElementById('forum-comment-input'), document.getElementById('forum-comment-counter'), 2000);
                }
                this.selectedCommentFiles = [];
                this.renderImagePreviews('comment');

                const formContainer = document.getElementById('forum-comment-form-container');
                if (formContainer) {
                    formContainer.classList.add('hidden');
                    formContainer.classList.remove('minimized');
                }

                await this.loadCategories(); // Refresh side buttons (to clear/update gold)
                await this.loadPostDetail(this.currentPostId);
            } else {
                alert(data.error || "Failed to post comment.");
            }
        } catch (err) {
            console.error("[Forum] Comment submit error:", err);
            alert("Failed to post comment.");
        } finally {
            if (submitCommentBtn) {
                submitCommentBtn.disabled = false;
                submitCommentBtn.textContent = originalBtnText;
            }
        }
    },

    compressImage: function (file, maxDimension, quality = 0.8) {
        return new Promise((resolve, reject) => {
            const processImage = (imgSrc, shouldRevoke) => {
                const img = new Image();
                img.onload = () => {
                    if (shouldRevoke && typeof URL !== 'undefined' && URL.revokeObjectURL) {
                        try { URL.revokeObjectURL(imgSrc); } catch (e) { }
                    }
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
                            const baseName = (file.name && file.name.lastIndexOf('.') > 0)
                                ? file.name.substring(0, file.name.lastIndexOf('.'))
                                : (file.name || 'image');
                            const compressedFile = new File([blob], `${baseName}.jpg`, { type: 'image/jpeg', lastModified: Date.now() });
                            resolve(compressedFile);
                        } else {
                            reject(new Error("Canvas to Blob failed"));
                        }
                    }, 'image/jpeg', quality);
                };
                img.onerror = (err) => {
                    if (shouldRevoke && typeof URL !== 'undefined' && URL.revokeObjectURL) {
                        try { URL.revokeObjectURL(imgSrc); } catch (e) { }
                    }
                    reject(err);
                };
                img.src = imgSrc;
            };

            if (typeof URL !== 'undefined' && URL.createObjectURL) {
                try {
                    const blobUrl = URL.createObjectURL(file);
                    processImage(blobUrl, true);
                    return;
                } catch (e) {
                    console.warn("[Forum] compressImage createObjectURL failed, falling back to FileReader:", e);
                }
            }

            const reader = new FileReader();
            reader.onload = (event) => processImage(event.target.result, false);
            reader.onerror = (err) => reject(err);
            try {
                reader.readAsDataURL(file);
            } catch (err) {
                reject(err);
            }
        });
    },

    handleImagePreview: function (e, previewId) {
        // Delegate to renderImagePreviews — never render a full-size inline image
        const type = (previewId && previewId.includes('comment')) ? 'comment' : 'post';
        this.renderImagePreviews(type);
    },

    showListView: function (noMobileScroll = false) {
        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-list').classList.add('active');

        const activeScroll = document.querySelector('#forum-view-list .forum-view-scroll-body');
        if (activeScroll) activeScroll.scrollTop = 0;

        // Hide comment form when leaving post view
        const formContainer = document.getElementById('forum-comment-form-container');
        if (formContainer) {
            formContainer.classList.add('hidden');
            formContainer.classList.remove('minimized');
        }

        // Hide static Delete Post button when leaving post view
        const deleteContainer = document.getElementById('forum-delete-post-container');
        if (deleteContainer) deleteContainer.style.display = 'none';

        // On mobile devices, scroll down so they see the category title and threads
        if (!noMobileScroll && window.innerWidth <= 820) {
            const titleEl = document.getElementById('forum-category-title');
            if (titleEl) {
                titleEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
    },

    showPostView: function () {
        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-post').classList.add('active');

        const activeScroll = document.querySelector('#forum-view-post .forum-view-scroll-body');
        if (activeScroll) activeScroll.scrollTop = 0;

        // On mobile devices, scroll down so they see the post details
        if (window.innerWidth <= 820) {
            const postViewEl = document.getElementById('forum-view-post');
            if (postViewEl) {
                postViewEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
    },

    showCreateView: function () {
        if (!this.currentCategoryId) return;

        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-create').classList.add('active');
        document.getElementById('forum-post-category-id').value = this.currentCategoryId;

        const activeScroll = document.querySelector('#forum-view-create .forum-view-scroll-body');
        if (activeScroll) activeScroll.scrollTop = 0;

        const formContainer = document.getElementById('forum-comment-form-container');
        if (formContainer) {
            formContainer.classList.add('hidden');
            formContainer.classList.remove('minimized');
        }

        // User Request Update: Allow all posts in every topic to attach an image
        document.getElementById('forum-image-upload-section').classList.remove('hidden');

        // On mobile devices, scroll down so they see the create post form
        if (window.innerWidth <= 820) {
            const createEl = document.getElementById('forum-view-create');
            if (createEl) {
                createEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
    },

    showRestrictedView: function () {
        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-restricted').classList.add('active');
    },

    escapeHtml: function (text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    renderContentWithLinks: function (text) {
        if (!text) return '';
        const escaped = this.escapeHtml(text);
        // Regex to auto-link URLs (YouTube, HTTP, HTTPS)
        const urlRegex = /(https?:\/\/[^\s<]+[^<.,:;"')\]\s])/gi;
        let processed = escaped.replace(urlRegex, function (match) {
            return `<a href="${match}" target="_blank" rel="noopener noreferrer" class="forum-clickable-link" onclick="event.stopPropagation();">${match}</a>`;
        });

        // Regex for @username (#NUM)? or &username (#NUM)?
        // Match @username or &username with optional #NUM
        const mentionRegex = /(?<![\w.])([@&][A-Za-z0-9_]+)(?:\s*#(\d+)(\s*\(Thread\))?)?/g;
        processed = processed.replace(mentionRegex, (match, tag, numStr, threadSuffix) => {
            if (numStr) {
                const lookupKey = `#${numStr}`;
                const target = this.currentPostLookup ? this.currentPostLookup[lookupKey] : null;
                if (target) {
                    // References to the thread's opening message are labelled "(Thread)"
                    const label = target.type === 'post' ? ' (Thread)' : '';
                    return `<span class="forum-mention-tag">${tag} <a href="#" class="forum-post-link" data-post-number="${numStr}" onclick="event.preventDefault(); event.stopPropagation(); Forum.navigateToPostNumber(${numStr});">#${numStr}</a>${label}</span>`;
                }
                return `<span class="forum-mention-tag">${tag} #${numStr}${threadSuffix || ''}</span>`;
            }
            // Standalone @username or &username
            const uname = tag.slice(1);
            return `<span class="forum-mention-tag forum-user-clickable" data-username="${this.escapeHtml(uname)}" title="View ${this.escapeHtml(uname)}'s Profile">${tag}</span>`;
        });

        return processed;
    },

    navigateToPostNumber: function (postNumber) {
        const lookupKey = `#${postNumber}`;
        const target = this.currentPostLookup ? this.currentPostLookup[lookupKey] : null;

        if (!target) {
            alert(`Post #${postNumber} no longer exists or could not be found.`);
            return;
        }

        // Check if the target is within the same thread currently loaded
        if (target.thread_id === this.currentPostId) {
            this.highlightTargetPost({
                highlightCommentId: target.type === 'comment' ? target.item_id : null,
                highlightPostNumber: target.post_number
            });
            return;
        }

        // The target is in a different thread within the same category
        const threadTitle = target.title || 'Another thread';
        const modalMsg = `This link leads to an external post in "${threadTitle}" within the same category.\n\nContinue?`;

        const proceed = () => {
            this.loadPostDetail(target.thread_id, {
                highlightCommentId: target.type === 'comment' ? target.item_id : null,
                highlightPostNumber: target.post_number
            });
        };

        if (typeof window.showConfirmModal === 'function') {
            window.showConfirmModal(
                "External Post",
                modalMsg,
                proceed,
                "Continue",
                "Cancel"
            );
        } else {
            if (confirm(modalMsg)) {
                proceed();
            }
        }
    },

    openMiniProfile: function (username, evt) {
        if (evt) {
            evt.stopPropagation();
            evt.preventDefault();
        }
        if (window.getSelection) {
            try { window.getSelection().removeAllRanges(); } catch (e) {}
        }
        if (!username) return;
        if (typeof window.showMiniProfile === 'function') {
            window.showMiniProfile(username);
        }
    },

    resetToEmptyState: function () {
        this.currentCategoryId = null;
        this.currentPostId = null;

        // Ensure list view is active, skipping mobile auto-scroll to main
        this.showListView(true);

        // Unpress all categories and responders tab
        document.querySelectorAll('.forum-cat-item').forEach(item => {
            item.classList.remove('active');
        });

        // Hide New Post button
        const newPostBtn = document.getElementById('forum-new-post-btn');
        if (newPostBtn) newPostBtn.classList.add('hidden');

        // Reset category header titles
        const titleEl = document.getElementById('forum-category-title');
        if (titleEl) titleEl.textContent = 'Community Board';

        const descEl = document.getElementById('forum-category-desc');
        if (descEl) {
            descEl.textContent = 'Connect with other players.';
            descEl.classList.remove('forum-desc-scrolling-box');
        }

        // Clear user search input
        const searchInput = document.getElementById('forum-user-search-input');
        if (searchInput) searchInput.value = '';

        // Default Forum content panel to empty with welcome placeholder
        const postsList = document.getElementById('forum-posts-list');
        if (postsList) {
            postsList.innerHTML = `
                <div class="forum-placeholder">
                    <div class="placeholder-icon">💬</div>
                    <h3>Welcome to the Forums</h3>
                    <p>Select a category from the sidebar to begin exploring the community.</p>
                </div>
            `;
        }

        // Reset mobile layout scrolls and ensure sidebar is in view
        const container = document.querySelector('#page-forums .forum-container');
        if (container) {
            container.scrollTo({ left: 0, top: 0, behavior: 'auto' });
        }
        const sidebar = document.querySelector('.forum-sidebar');
        if (sidebar) {
            sidebar.classList.remove('hidden-mobile');
            sidebar.scrollIntoView({ behavior: 'auto', inline: 'start' });
        }
        const main = document.querySelector('.forum-main');
        if (main) {
            main.classList.remove('hidden-mobile');
        }
    }
};

window.initForum = function () {
    Forum.init();
    if (typeof Forum.loadFriends === 'function') {
        Forum.loadFriends();
    }
};

window.resetForumTab = function (immediate = false) {
    if (typeof Forum !== 'undefined' && typeof Forum.resetToEmptyState === 'function') {
        Forum.resetToEmptyState();
    }
};

window.navigateToForumUserPosts = async function (username) {
    if (!username) return;

    // 1. Close mini profile modal if open
    const miniModal = document.getElementById('mini-profile-modal');
    if (miniModal) {
        miniModal.classList.add('hidden');
        miniModal.classList.remove('forced-show');
        miniModal.style.display = 'none';
    }

    // 2. Ensure user is logged in
    if (!window.currentUser) {
        if (typeof window.navigateToPage === 'function') {
            window.navigateToPage('login');
        } else {
            const btnLogin = document.getElementById('nav-login-btn');
            if (btnLogin) btnLogin.click();
        }
        return;
    }

    // 3. Switch to Forum page
    if (typeof window.navigateToPage === 'function') {
        window.navigateToPage('forums');
    } else {
        const btnForums = document.getElementById('btn-forums');
        if (btnForums) btnForums.click();
    }

    // 4. Ensure Forum module is initialized
    if (typeof Forum !== 'undefined') {
        if (!Forum.initialized && typeof Forum.init === 'function') {
            await Forum.init();
        }
    }

    // 5. Populate search input and trigger user search
    const executeSearch = () => {
        const searchInput = document.getElementById('forum-user-search-input');
        if (searchInput) {
            searchInput.value = username;
        }
        if (typeof Forum !== 'undefined' && typeof Forum.handleUserSearch === 'function') {
            Forum.handleUserSearch();
        }
    };

    setTimeout(executeSearch, 150);
};


