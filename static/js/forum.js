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
        return `<img src="https://flagcdn.com/w40/${code.toLowerCase()}.png" class="flag-icon-img" alt="${code}" title="${code}" style="width: 1.25em; height: auto; display: inline-block; vertical-align: middle; margin-left: 4px; border-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.2); ${extraStyles}">`;
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
    { code: 'ZW', flag: '🇿🇼', name: 'Zimbabwe' },
    { code: 'ZZ', flag: '🏳️', name: 'None / International' }
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

    init: async function () {
        if (this.initialized) return;
        console.log("[Forum] Initializing forum module...");
        this.setupEventListeners();
        await this.loadCategories();
        this.initialized = true;

        // Auto-refresh categories every 30s while the forum is open to show new posts from others
        setInterval(() => {
            if (document.getElementById('page-forums').classList.contains('active')) {
                this.loadCategories();
            }
        }, 30000);

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
                
                if (this.currentCategoryId) {
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

        // Comment form
        const submitCommentBtn = document.getElementById('forum-submit-comment');
        if (submitCommentBtn) {
            submitCommentBtn.addEventListener('click', () => this.handleCommentSubmit());
        }

        // Image wrappers and inputs
        const postImageInput = document.getElementById('forum-post-image');
        const postImageWrapper = document.getElementById('forum-post-image-wrapper');
        if (postImageInput) {
            postImageInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files.length > 0) {
                    this.addFiles('post', e.target.files);
                    e.target.value = '';
                }
            });
        }
        if (postImageWrapper && postImageInput) {
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
                    this.addFiles('post', e.dataTransfer.files);
                }
            });
        }

        const commentImageInput = document.getElementById('forum-comment-image');
        const commentImageWrapper = document.getElementById('forum-comment-image-wrapper');
        if (commentImageInput) {
            commentImageInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files.length > 0) {
                    this.addFiles('comment', e.target.files);
                    e.target.value = '';
                }
            });
        }
        if (commentImageWrapper && commentImageInput) {
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
                    this.addFiles('comment', e.dataTransfer.files);
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
    },

    addFiles: function (type, files) {
        const targetArray = type === 'post' ? this.selectedPostFiles : this.selectedCommentFiles;
        let addedAny = false;
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (!file.type.startsWith('image/')) continue;
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
            return;
        }

        previewEl.classList.remove('hidden');
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
            const reader = new FileReader();
            reader.onload = (e) => {
                const imgEl = document.getElementById(`preview-img-${type}-${idx}`);
                if (imgEl) {
                    imgEl.src = e.target.result;
                    imgEl.addEventListener('click', (ev) => {
                        ev.stopPropagation();
                        if (typeof window.showImageLightbox === 'function') {
                            window.showImageLightbox(e.target.result, `Attachment Preview (${idx+1}/${targetArray.length}): ${file.name}`);
                        }
                    });
                }
            };
            reader.readAsDataURL(file);
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

    loadCategories: async function () {
        try {
            const response = await fetch('/api/forum/categories');
            const data = await response.json();
            this.categories = data.categories;
            this.renderCategories();
        } catch (err) {
            console.error("[Forum] Failed to load categories:", err);
        }
    },

    renderCategories: function () {
        const listEl = document.getElementById('forum-categories-list');
        if (!listEl) return;

        const lastViewed = JSON.parse(localStorage.getItem('forum_last_viewed') || '{}');
        
        listEl.innerHTML = this.categories.map(cat => {
            const lastContent = cat.last_content_at ? parseUTCTimestamp(cat.last_content_at).getTime() : 0;
            // Use sessionStartTime as default so that ancient posts do not highlight for new sessions
            const lastView = Number(lastViewed[cat.id]) || window.sessionStartTime || Date.now();
            const hasNew = lastContent > lastView;
            
            return `
                <div class="forum-cat-item ${hasNew ? 'has-new' : ''}" data-id="${cat.id}">
                    <span class="forum-cat-name">${cat.name}</span>
                    <span class="forum-cat-desc">${cat.description}</span>
                </div>
            `;
        }).join('');

        // Attach listeners
        listEl.querySelectorAll('.forum-cat-item').forEach(item => {
            item.addEventListener('click', () => {
                const catId = parseInt(item.getAttribute('data-id'));
                this.selectCategory(catId);
            });
        });
    },

    selectCategory: async function (catId) {
        this.currentCategoryId = catId;
        const category = this.categories.find(c => c.id === catId);

        // Update UI
        document.querySelectorAll('.forum-cat-item').forEach(item => {
            const isThisCat = parseInt(item.getAttribute('data-id')) === catId;
            item.classList.toggle('active', isThisCat);
            if (isThisCat) {
                item.classList.remove('has-new');
            }
        });

        // Update last viewed timestamp in localStorage
        const lastViewed = JSON.parse(localStorage.getItem('forum_last_viewed') || '{}');
        lastViewed[catId] = Date.now();
        localStorage.setItem('forum_last_viewed', JSON.stringify(lastViewed));

        // Immediately update global nav button status
        if (typeof window.checkForumActivity === 'function') {
            window.checkForumActivity();
        }

        document.getElementById('forum-category-title').textContent = category.name;
        document.getElementById('forum-category-desc').textContent = category.description;

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
        document.getElementById('forum-category-desc').textContent = `Viewing all forum contributions from ${username}.`;
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
            
            return `
                <div class="forum-post-card" data-id="${postId}">
                    <div class="post-card-header">
                        <span class="post-card-title">${isComment ? 'Re: ' : ''}${this.escapeHtml(post.title)}</span>
                        <span class="post-card-meta">
                            <span>${isComment ? 'Replied' : 'Posted'} by <strong>${post.username}${window.getFlagHtml ? window.getFlagHtml(post.country_flag) : (post.country_flag || '')}</strong></span>
                            <span>${dateStr}</span>
                        </span>
                    </div>
                    <div class="post-card-excerpt">${this.escapeHtml(post.content)}</div>
                    <div class="post-stats">
                        ${isComment ? '' : `<div class="stat-item">💬 ${post.comment_count} comments</div>`}
                        ${hasImages ? '<div class="stat-item">🖼️ Includes images</div>' : ''}
                    </div>
                </div>
            `;
        }).join('');


        // Attach listeners
        postsList.querySelectorAll('.forum-post-card').forEach(card => {
            card.addEventListener('click', () => {
                const postId = parseInt(card.getAttribute('data-id'));
                this.loadPostDetail(postId);
            });
        });
    },

    loadPostDetail: async function (postId) {
        this.currentPostId = postId;
        try {
            const response = await fetch(`/api/forum/post/${postId}`);
            const data = await response.json();
            this.renderPostDetail(data.post, data.comments);
            this.showPostView();
        } catch (err) {
            console.error("[Forum] Failed to load post detail:", err);
        }
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

    renderPostDetail: function (post, comments) {
        const commentInput = document.getElementById('forum-comment-input');
        if (commentInput) commentInput.value = '';
        this.selectedCommentFiles = [];
        this.renderImagePreviews('comment');

        const detailEl = document.getElementById('forum-post-detail');
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

        detailEl.innerHTML = `
            <div class="post-detail-header">
                <h1 class="post-detail-title">${this.escapeHtml(post.title)}</h1>
                <div class="post-author-box">
                    <div class="author-avatar">${post.username[0].toUpperCase()}</div>
                    <div class="author-info">
                        <span class="author-name">${post.username}${window.getFlagHtml ? window.getFlagHtml(post.country_flag) : (post.country_flag || '')}</span>
                        <span class="post-date">${dateStr}</span>
                    </div>
                </div>
            </div>
            <div class="post-content">${this.renderContentWithLinks(post.content)}</div>
            ${postImagesHtml}
        `;

        // Static Delete Post button in HTML — show for mods, hide for others
        const deleteContainer = document.getElementById('forum-delete-post-container');
        const deleteBtn = document.getElementById('forum-delete-post-btn');
        if (deleteContainer) {
            deleteContainer.style.display = window.currentUserIsMod ? 'block' : 'none';
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

        const commentsListEl = document.getElementById('forum-comments-list');
        document.getElementById('forum-comment-count').textContent = `${comments.length} comments`;

        const sortedComments = [...comments].sort((a, b) => parseUTCTimestamp(b.timestamp) - parseUTCTimestamp(a.timestamp));

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

                return `
                    <div class="forum-comment">
                        <div class="comment-avatar">${c.username[0].toUpperCase()}</div>
                        <div class="comment-body">
                            <div class="comment-header">
                                <span class="comment-author">${c.username}${window.getFlagHtml ? window.getFlagHtml(c.country_flag) : (c.country_flag || '')}</span>
                                <span class="comment-date">${cDate}</span>
                                ${window.currentUserIsMod ? `
                                    <button class="forum-comment-delete-btn" data-id="${c.id}" style="margin-left: auto; background: none; border: none; color: #f43f5e; cursor: pointer; font-size: 0.75rem; opacity: 0.6;">Delete</button>
                                ` : ''}
                            </div>
                            <div class="comment-content">${this.renderContentWithLinks(c.content)}</div>
                            ${cImagesHtml}
                        </div>
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

        const isGuest = window.currentUserIsGuest || (window.currentUser === null);
        document.getElementById('forum-comment-form-container').classList.toggle('hidden', isGuest);
    },

    handlePostSubmit: async function (e) {
        e.preventDefault();
        const title = document.getElementById('forum-post-title').value;
        const content = document.getElementById('forum-post-content').value;
        const catId = document.getElementById('forum-post-category-id').value;

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
        const content = document.getElementById('forum-comment-input').value;

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
                this.selectedCommentFiles = [];
                this.renderImagePreviews('comment');

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
    },

    handleImagePreview: function (e, previewId) {
        // Delegate to renderImagePreviews — never render a full-size inline image
        const type = (previewId && previewId.includes('comment')) ? 'comment' : 'post';
        this.renderImagePreviews(type);
    },

    showListView: function () {
        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-list').classList.add('active');

        // Hide static Delete Post button when leaving post view
        const deleteContainer = document.getElementById('forum-delete-post-container');
        if (deleteContainer) deleteContainer.style.display = 'none';

        // On mobile devices, scroll down so they see the category title and threads
        if (window.innerWidth <= 820) {
            const titleEl = document.getElementById('forum-category-title');
            if (titleEl) {
                titleEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
    },

    showPostView: function () {
        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-post').classList.add('active');

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
        return escaped.replace(urlRegex, function (match) {
            return `<a href="${match}" target="_blank" rel="noopener noreferrer" class="forum-clickable-link" onclick="event.stopPropagation();">${match}</a>`;
        });
    }
};

window.initForum = function () {
    Forum.init();
};
