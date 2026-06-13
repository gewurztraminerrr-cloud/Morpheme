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
    },

    setupEventListeners: function () {
        // New post button
        const newPostBtn = document.getElementById('forum-new-post-btn');
        if (newPostBtn) {
            newPostBtn.addEventListener('click', () => this.showCreateView());
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
            postImageInput.addEventListener('change', (e) => this.handleImagePreview(e, 'forum-image-preview'));
        }
        if (postImageWrapper && postImageInput) {
            postImageWrapper.addEventListener('dragover', (e) => {
                e.preventDefault();
                postImageWrapper.style.background = 'rgba(0, 0, 0, 0.4)';
                postImageWrapper.style.borderColor = 'var(--accent-color)';
            });
            postImageWrapper.addEventListener('dragleave', () => {
                postImageWrapper.style.background = 'rgba(0, 0, 0, 0.2)';
                postImageWrapper.style.borderColor = 'var(--input-border)';
            });
            postImageWrapper.addEventListener('drop', (e) => {
                e.preventDefault();
                postImageWrapper.style.background = 'rgba(0, 0, 0, 0.2)';
                postImageWrapper.style.borderColor = 'var(--input-border)';
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    postImageInput.files = e.dataTransfer.files;
                    postImageInput.dispatchEvent(new Event('change'));
                }
            });
        }

        const commentImageInput = document.getElementById('forum-comment-image');
        const commentImageWrapper = document.getElementById('forum-comment-image-wrapper');
        if (commentImageInput) {
            commentImageInput.addEventListener('change', (e) => this.handleImagePreview(e, 'forum-comment-image-preview'));
        }
        if (commentImageWrapper && commentImageInput) {
            commentImageWrapper.addEventListener('dragover', (e) => {
                e.preventDefault();
                commentImageWrapper.style.background = 'rgba(0, 0, 0, 0.4)';
                commentImageWrapper.style.borderColor = 'var(--accent-color)';
            });
            commentImageWrapper.addEventListener('dragleave', () => {
                commentImageWrapper.style.background = 'rgba(0, 0, 0, 0.2)';
                commentImageWrapper.style.borderColor = 'var(--input-border)';
            });
            commentImageWrapper.addEventListener('drop', (e) => {
                e.preventDefault();
                commentImageWrapper.style.background = 'rgba(0, 0, 0, 0.2)';
                commentImageWrapper.style.borderColor = 'var(--input-border)';
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    commentImageInput.files = e.dataTransfer.files;
                    commentImageInput.dispatchEvent(new Event('change'));
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
            
            if (hasNew) {
                console.debug(`[Forum Rendering] Category ${cat.name} (ID: ${cat.id}) IS GOLD: content=${lastContent}, view=${lastView}`);
            } else {
                console.debug(`[Forum Rendering] Category ${cat.name} (ID: ${cat.id}) IS GREY: content=${lastContent}, view=${lastView}`);
            }

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
        } catch (err) {
            console.error("[Forum] User search error:", err);
            postsList.innerHTML = '<div class="forum-placeholder"><h3>Error performing search.</h3></div>';
        }
    },

    renderPosts: function (posts) {
        console.log("[Forum] renderPosts received:", posts);
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
            const date = parseUTCTimestamp(post.timestamp);
            const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const isComment = post.type === 'comment';
            const postId = post.post_id || post.id;
            
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
                        ${post.image_url ? '<div class="stat-item">🖼️ Includes image</div>' : ''}
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
            console.log(`[Forum] Rendering post ${postId} with ${data.comments.length} comments (Newest First)`);
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
                // Return to list and reload everything
                await this.loadCategories(); // Refresh side counts (though we don't show counts yet)
                await this.selectCategory(this.currentCategoryId); // Refresh posts list for current category
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
        // Clear/reset comment inputs for the new post
        const commentInput = document.getElementById('forum-comment-input');
        if (commentInput) commentInput.value = '';
        const commentImageInput = document.getElementById('forum-comment-image');
        if (commentImageInput) commentImageInput.value = '';
        this.handleImagePreview({ target: { files: [] } }, 'forum-comment-image-preview');

        const detailEl = document.getElementById('forum-post-detail');
        const date = parseUTCTimestamp(post.timestamp);
        const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        detailEl.innerHTML = `
            <div class="post-detail-header">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <h1 class="post-detail-title">${this.escapeHtml(post.title)}</h1>
                    ${window.currentUserIsMod ? `
                        <button id="forum-delete-post-btn" class="forum-action-btn remove" style="background: #f43f5e; font-size: 0.8rem; padding: 5px 12px;">Delete Post</button>
                    ` : ''}
                </div>
                <div class="post-author-box">
                    <div class="author-avatar">${post.username[0].toUpperCase()}</div>
                    <div class="author-info">
                        <span class="author-name">${post.username}${window.getFlagHtml ? window.getFlagHtml(post.country_flag) : (post.country_flag || '')}</span>
                        <span class="post-date">${dateStr}</span>
                    </div>
                </div>
            </div>
            <div class="post-content">${this.escapeHtml(post.content)}</div>
            ${post.image_url ? `
                <div class="post-image-container">
                    <img src="${post.image_url}" class="post-image" alt="Post attachment">
                </div>
            ` : ''}
        `;

        // Attach delete listener if button exists
        const deleteBtn = document.getElementById('forum-delete-post-btn');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.handlePostDelete(post.id);
            });
        }

        // Render comments
        const commentsListEl = document.getElementById('forum-comments-list');
        document.getElementById('forum-comment-count').textContent = `${comments.length} comments (Newest First)`;

        // Sort comments by timestamp newest first to be absolutely sure
        const sortedComments = [...comments].sort((a, b) => {
            const dateA = parseUTCTimestamp(a.timestamp);
            const dateB = parseUTCTimestamp(b.timestamp);
            return dateB - dateA;
        });

        if (sortedComments.length === 0) {
            commentsListEl.innerHTML = '<p class="forum-placeholder">No comments yet. Start the discussion!</p>';
        } else {
            commentsListEl.innerHTML = sortedComments.map(c => {
                const cDate = parseUTCTimestamp(c.timestamp).toLocaleString([], { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' });
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
                            <div class="comment-content">${this.escapeHtml(c.content)}</div>
                            ${c.image_url ? `
                                <div class="comment-image-container" style="margin-top: 10px; border-radius: 8px; overflow: hidden; border: 1px solid var(--input-border);">
                                    <img src="${c.image_url}" style="max-width: 100%; display: block;" alt="Comment attachment">
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }).join('');

            // Attach comment delete listeners
            commentsListEl.querySelectorAll('.forum-comment-delete-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const commentId = parseInt(btn.getAttribute('data-id'));
                    this.handleCommentDelete(commentId);
                });
            });
        }

        // Show/hide comment form
        const isGuest = window.currentUserIsGuest || (window.currentUser === null);
        document.getElementById('forum-comment-form-container').classList.toggle('hidden', isGuest);
    },

    handlePostSubmit: async function (e) {
        e.preventDefault();
        const title = document.getElementById('forum-post-title').value;
        const content = document.getElementById('forum-post-content').value;
        const catId = document.getElementById('forum-post-category-id').value;
        const imageFile = document.getElementById('forum-post-image') ? document.getElementById('forum-post-image').files[0] : null;

        if (!title || !content) return;

        // Visual loading feedback
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

        if (imageFile) {
            if (imageFile.type === 'image/gif') {
                if (imageFile.size > 2 * 1024 * 1024) {
                    alert("GIF files must be under 2MB.");
                    if (submitPostBtn) {
                        submitPostBtn.disabled = false;
                        submitPostBtn.textContent = originalBtnText;
                    }
                    return;
                }
                formData.append('image', imageFile);
            } else {
                try {
                    const compressed = await this.compressImage(imageFile, 1200, 0.8);
                    formData.append('image', compressed);
                } catch (err) {
                    console.error("[Forum] Compression failed, uploading original:", err);
                    formData.append('image', imageFile);
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
                // Return to list and reload everything
                document.getElementById('forum-post-form').reset();
                this.handleImagePreview({ target: { files: [] } }, 'forum-image-preview'); 
                
                await this.loadCategories(); // Refresh ALL side buttons first
                await this.selectCategory(this.currentCategoryId); // Then load posts for the current one
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
        const imageFile = document.getElementById('forum-comment-image') ? document.getElementById('forum-comment-image').files[0] : null;

        if (!content) return;

        // Visual loading feedback
        const submitCommentBtn = document.getElementById('forum-submit-comment');
        const originalBtnText = submitCommentBtn ? submitCommentBtn.textContent : 'Post Comment';
        if (submitCommentBtn) {
            submitCommentBtn.disabled = true;
            submitCommentBtn.textContent = 'Posting...';
        }

        const formData = new FormData();
        formData.append('post_id', this.currentPostId);
        formData.append('content', content);

        if (imageFile) {
            if (imageFile.type === 'image/gif') {
                if (imageFile.size > 2 * 1024 * 1024) {
                    alert("GIF files must be under 2MB.");
                    if (submitCommentBtn) {
                        submitCommentBtn.disabled = false;
                        submitCommentBtn.textContent = originalBtnText;
                    }
                    return;
                }
                formData.append('image', imageFile);
            } else {
                try {
                    const compressed = await this.compressImage(imageFile, 1200, 0.8);
                    formData.append('image', compressed);
                } catch (err) {
                    console.error("[Forum] Compression failed, uploading original:", err);
                    formData.append('image', imageFile);
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
                const commentImageInput = document.getElementById('forum-comment-image');
                if (commentImageInput) commentImageInput.value = '';
                this.handleImagePreview({ target: { files: [] } }, 'forum-comment-image-preview');

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
        const file = e.target.files[0];
        const previewEl = document.getElementById(previewId);
        if (!previewEl) return;

        if (file) {
            const reader = new FileReader();
            reader.onload = function (event) {
                previewEl.innerHTML = `<img src="${event.target.result}" style="max-width: 100%; border-radius: 8px;">`;
                previewEl.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        } else {
            previewEl.innerHTML = '';
            previewEl.classList.add('hidden');
        }
    },

    showListView: function () {
        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-list').classList.add('active');
    },

    showPostView: function () {
        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-post').classList.add('active');
    },

    showCreateView: function () {
        if (!this.currentCategoryId) return;

        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-create').classList.add('active');
        document.getElementById('forum-post-category-id').value = this.currentCategoryId;

        // User Request Update: Allow all posts in every topic to attach an image
        document.getElementById('forum-image-upload-section').classList.remove('hidden');
    },

    showRestrictedView: function () {
        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-restricted').classList.add('active');
    },

    escapeHtml: function (text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

window.initForum = function () {
    Forum.init();
};
