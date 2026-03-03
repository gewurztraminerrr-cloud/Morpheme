// Navigation system
const pages = {
    'nav-login-btn': 'page-login',
    'btn-how-to-play': 'page-how-to-play',
    'btn-lobby': 'page-lobby',
    'btn-play': 'page-play',
    'btn-leaderboards': 'page-leaderboards',
    'nav-tournaments-btn': 'page-tournaments',
    'btn-store': 'page-store',
    'btn-forums': 'page-forums',
    'btn-tools': 'page-tools',
    'btn-settings': 'page-settings',
    'btn-contact': 'page-contact'
};

let currentUser = null;
let currentUserEmail = null;
let selectedRoom = null;

// Define standardized rating ranges globaly for reuse
const RATING_RANGES = [
    // --- THE CLIMB (1 - 1399) ---
    // Greens (1 - 699): The Foundation
    { min: 1, max: 99, color: '#e6ffe6', label: '1 - 99', name: 'Initiate' },
    { min: 100, max: 199, color: '#ccffcc', label: '100 - 199', name: 'Apprentice' },
    { min: 200, max: 299, color: '#99ff99', label: '200 - 299', name: 'Practitioner' },
    { min: 300, max: 399, color: '#66ff66', label: '300 - 399', name: 'Scholar' },
    { min: 400, max: 499, color: '#33ff33', label: '400 - 499', name: 'Adept' },
    { min: 500, max: 599, color: '#00ff00', label: '500 - 599', name: 'Specialist' },
    { min: 600, max: 699, color: '#00cc00', label: '600 - 699', name: 'Expert' },

    // Blues (700 - 1399): The Sky & Ocean
    { min: 700, max: 799, color: '#66ccff', label: '700 - 799', name: 'Vanguard' },
    { min: 800, max: 899, color: '#33bbff', label: '800 - 899', name: 'Sentinel' },
    { min: 900, max: 999, color: '#00aaff', label: '900 - 999', name: 'Strategist' },
    { min: 1000, max: 1099, color: '#0088ff', label: '1000 - 1099', name: 'Tactician' },
    { min: 1100, max: 1199, color: '#0066ff', label: '1100 - 1199', name: 'Virtuoso' },
    { min: 1200, max: 1299, color: '#0044ff', label: '1200 - 1299', name: 'Master' },
    { min: 1300, max: 1399, color: '#0000ff', label: '1300 - 1399', name: 'Grandmaster' },

    // --- THE HEAT (1400 - 2499) ---
    // Yellows
    { min: 1400, max: 1499, color: '#ffff66', label: '1400 - 1499', name: 'Elite' },
    { min: 1500, max: 1599, color: '#ffff00', label: '1500 - 1599', name: 'Champion' },
    { min: 1600, max: 1699, color: '#ffcc00', label: '1600 - 1699', name: 'Titan' },
    { min: 1700, max: 1799, color: '#ffaa00', label: '1700 - 1799', name: 'Paragon' },

    // Oranges
    { min: 1800, max: 1899, color: '#ff8800', label: '1800 - 1899', name: 'Sovereign' },
    { min: 1900, max: 1999, color: '#ff6600', label: '1900 - 1999', name: 'Exalted' },
    { min: 2000, max: 2099, color: '#ff4400', label: '2000 - 2099', name: 'Overlord' },
    { min: 2100, max: 2199, color: '#ff2200', label: '2100 - 2199', name: 'Conqueror' },

    // Reds
    { min: 2200, max: 2299, color: '#ff0000', label: '2200 - 2299', name: 'Warlord' },
    { min: 2300, max: 2399, color: '#e60000', label: '2300 - 2399', name: 'Juggernaut' },
    { min: 2400, max: 2499, color: '#cc0000', label: '2400 - 2499', name: 'Apex' },

    // --- THE VOID (2500 - 6000+) ---
    { min: 2500, max: 2599, color: '#b30000', label: '2500 - 2599', name: 'Harbinger' },
    { min: 2600, max: 2699, color: '#990000', label: '2600 - 2699', name: 'Oracle' },
    { min: 2700, max: 2799, color: '#800000', label: '2700 - 2799', name: 'Revenant' },
    { min: 2800, max: 2899, color: '#660000', label: '2800 - 2899', name: 'Specter' },
    { min: 2900, max: 2999, color: '#4d0000', label: '2900 - 2999', name: 'Phantom' },

    { min: 3000, max: 3999, color: '#330000', label: '3000 - 3999', name: 'Ascendant' },
    { min: 4000, max: 4999, color: '#220000', label: '4000 - 4999', name: 'Transcendent' },
    { min: 5000, max: 5999, color: '#110000', label: '5000 - 5999', name: 'Ethereal' },
    { min: 6000, max: 99999, color: '#000000', label: '6000+', name: 'ALIEN BEING' }
];

// Single Instance Logic
(function () {
    const channel = new BroadcastChannel('morpheme_instance');
    let isOriginal = true;

    // Check if other tabs exist
    channel.postMessage('check_if_exists');

    channel.onmessage = (event) => {
        if (event.data === 'check_if_exists') {
            // Another tab is asking if we exist. We do.
            channel.postMessage('i_exist');
        } else if (event.data === 'i_exist') {
            // Another tab exists! We are the duplicate.
            isOriginal = false;
            document.body.innerHTML = `
                <div style="
                    display: flex; 
                    justify-content: center; 
                    align-items: center; 
                    height: 100vh; 
                    background: #0d1117; 
                    color: #fff; 
                    font-family: sans-serif; 
                    text-align: center;
                    padding: 20px;
                ">
                    <div>
                        <h1 style="color: #e74c3c;">Morpheme is already open</h1>
                        <p style="font-size: 1.2rem; color: #aaa;">
                            Please close this tab and use the existing window.
                        </p>
                    </div>
                </div>
            `;
            // Stop further execution
            throw new Error("Morpheme already running in another tab.");
        }
    };
})();

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    setupNavigation();
    setupModalListeners();
    setupAuth(); // Initialize auth listeners
    setupContactForm(); // Initialize contact form listeners
    initSettings(); // Initialize settings logic
    await checkSession();

    // Handle initial navigation
    const hash = window.location.hash;
    const tournamentActive = localStorage.getItem('tournament_play_active');
    const privateActive = localStorage.getItem('private_match_active');

    if (currentUser) {
        // AUTHENTICATED: Always stay out of login page
        if (tournamentActive || privateActive || (hash === '#page-play' && window.currentRoomId)) {
            showPage('page-play');
            const playBtn = document.querySelector('.nav-btn[data-page="play"]');
            if (playBtn) updateActiveNav(playBtn);
        } else if (hash && hash.startsWith('#page-') && hash !== '#page-play' && hash !== '#page-login') {
            const pageId = hash.substring(1);
            showPage(pageId);
            const pageName = pageId.replace('page-', '');
            const navBtn = document.querySelector(`.nav-btn[data-page="${pageName}"]`);
            if (navBtn) updateActiveNav(navBtn);
        } else {
            // Default to lobby for authenticated users (even if hash is empty or #page-login)
            showPage('page-lobby');
            const lobbyBtn = document.querySelector('.nav-btn[data-page="lobby"]');
            if (lobbyBtn) updateActiveNav(lobbyBtn);
            // Clean up URL if it was stuck on #page-login
            if (hash === '#page-login') {
                history.replaceState(null, null, '#page-lobby');
            }
        }
    } else {
        // UNAUTHENTICATED: Always force login unless it's a known public page
        if (hash === '#page-leaderboards') {
            showPage('page-leaderboards');
            const lbBtn = document.querySelector('.nav-btn[data-page="leaderboards"]');
            if (lbBtn) updateActiveNav(lbBtn);
        } else {
            showPage('page-login');
        }

        // Only clear if we aren't in a special match and reached this fallback
        if (!tournamentActive && !privateActive) {
            localStorage.removeItem('tournament_play_active');
            localStorage.removeItem('private_match_active');
        }
    }

    fetchUserCount(); // Fetch user count for login page
});

async function fetchUserCount() {
    try {
        const res = await fetch('/api/stats/user_count');
        const data = await res.json();
        const el = document.getElementById('total-user-count');
        if (el && data.count !== undefined) {
            const online = data.online_count || 0;
            el.textContent = `Join ${data.count} registered and ${online} online players!`;
        }
    } catch (e) {
        console.warn('Failed to fetch user count', e);
    }
}

// Global Settings State
window.userSettings = {
    lobby_music: true, // Default ON
    chat_font_size: 13,
    def_font_size: 15
};

// Initialize Defaults immediately
document.documentElement.style.setProperty('--chat-font-size', '13px');
document.documentElement.style.setProperty('--def-font-size', '15px');

function initSettings() {
    const musicToggle = document.getElementById('setting-lobby-music');

    // 1. Lobby Music
    if (musicToggle) {
        musicToggle.addEventListener('change', async (e) => {
            const isEnabled = e.target.checked;
            window.userSettings.lobby_music = isEnabled;
            handleLobbyMusicState();
            saveSetting('lobby_music', isEnabled);
        });
    }

    // 2. Font Size Controls
    setupFontSizeControl('setting-chat-size', 'setting-chat-size-val', 'preview-chat-text', '--chat-font-size', 'chat_font_size');
    setupFontSizeControl('setting-def-size', 'setting-def-size-val', 'preview-def-text', '--def-font-size', 'def_font_size');
}

function setupFontSizeControl(sliderId, labelId, previewId, cssVar, dbKey) {
    const slider = document.getElementById(sliderId);
    const label = document.getElementById(labelId);
    const preview = document.getElementById(previewId);

    if (!slider) return;

    // Helper to apply visual changes
    const applyVisuals = (val) => {
        if (label) label.textContent = val + 'px';
        if (preview) preview.style.fontSize = val + 'px';
        document.documentElement.style.setProperty(cssVar, val + 'px');
    };

    // Live Preview (Input event)
    slider.addEventListener('input', (e) => {
        applyVisuals(e.target.value);
    });

    // Save on release (Change event)
    slider.addEventListener('change', (e) => {
        const val = e.target.value;
        window.userSettings[dbKey] = val;
        applyVisuals(val); // Ensure consistency
        saveSetting(dbKey, val);
    });
}

async function saveSetting(key, value) {
    if (currentUser) {
        try {
            await fetch('/api/settings/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key, value })
            });
        } catch (err) {
            console.error('Failed to save setting:', err);
        }
    }
}

// Helper to start/stop music based on Page AND Setting
function handleLobbyMusicState() {
    const lobbyMusic = document.getElementById('lobby-music');
    if (!lobbyMusic) return;

    // Only play if: 1) On Lobby Page AND 2) Setting is TRUE
    const onLobby = document.getElementById('page-lobby').classList.contains('active');
    const shouldPlay = onLobby && window.userSettings.lobby_music;

    if (shouldPlay) {
        // If already playing, do nothing. If paused, play.
        if (lobbyMusic.paused) {
            // Set to start of loop section (3:25 = 205 seconds) if at 0
            if (lobbyMusic.currentTime < 1) lobbyMusic.currentTime = 205;

            lobbyMusic.play().catch(e => console.log('Autoplay blocked:', e));

            // Ensure loop logic is attached
            lobbyMusic.ontimeupdate = function () {
                if (lobbyMusic.currentTime >= 295) { // 4:55 = 295 seconds
                    lobbyMusic.currentTime = 205; // Loop back to 3:25
                }
            };
        }
    } else {
        // Stop
        if (!lobbyMusic.paused) {
            lobbyMusic.pause();
        }
    }
}

// Setup contact form submission
function setupContactForm() {
    const contactForm = document.getElementById('contact-form');
    if (!contactForm) return;

    contactForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const email = document.getElementById('contact-user-email').value;
        const message = document.getElementById('contact-message').value;
        const statusEl = document.getElementById('contact-status');
        const submitBtn = contactForm.querySelector('.submit-contact-btn');

        // Reset status
        statusEl.textContent = 'Sending...';
        statusEl.className = 'contact-status';
        submitBtn.disabled = true;

        try {
            const response = await fetch('/api/contact', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email, message })
            });

            const data = await response.json();

            if (data.success) {
                statusEl.textContent = 'Message sent! We\'ll get back to you soon.';
                statusEl.className = 'contact-status success';
                contactForm.reset();
            } else {
                statusEl.textContent = data.error || 'Failed to send message.';
                statusEl.className = 'contact-status error';
            }
        } catch (error) {
            console.error('Contact error:', error);
            statusEl.textContent = 'An error occurred. Please try again later.';
            statusEl.className = 'contact-status error';
        } finally {
            submitBtn.disabled = false;
        }
    });
}

// Check if user is already logged in
async function checkSession() {
    try {
        const response = await fetch('/api/session');
        const data = await response.json();

        if (data.authenticated) {
            currentUser = data.username;
            window.currentUser = currentUser;  // Expose globally
            currentUserEmail = data.email;
            window.currentUserEmail = currentUserEmail;
            window.currentUserIsGuest = data.is_guest; // Store guest status
            localStorage.setItem('morpheme_username', currentUser);

            updateAuthUI(); // Update UI for logged in state

            // FETCH SETTINGS
            try {
                const sRes = await fetch('/api/settings');
                const sData = await sRes.json();
                if (sData.settings) {
                    // Apply Lobby Music
                    if (sData.settings.lobby_music !== undefined) {
                        let val = sData.settings.lobby_music;
                        // Handle potential string types from DB
                        if (val === 'true' || val === 'True' || val === true) val = true;
                        else if (val === 'false' || val === 'False' || val === false) val = false;

                        window.userSettings.lobby_music = val;

                        // Update Checkbox
                        const cb = document.getElementById('setting-lobby-music');
                        if (cb) cb.checked = val;

                        // Apply state
                        handleLobbyMusicState();
                    }

                    // Apply Font Sizes
                    applySavedFontSize(sData.settings.chat_font_size, 'setting-chat-size', 'setting-chat-size-val', 'preview-chat-text', '--chat-font-size', 'chat_font_size');
                    applySavedFontSize(sData.settings.def_font_size, 'setting-def-size', 'setting-def-size-val', 'preview-def-text', '--def-font-size', 'def_font_size');
                }
            } catch (e) { console.warn('Error fetching settings', e); }

            // NEW: Check if user is already in a room
            try {
                const roomRes = await fetch('/api/user/current-room');
                const roomData = await roomRes.json();
                if (roomData && roomData.room_id) {
                    console.log('Session Check: User is currently in room:', roomData.room_id);
                    window.currentRoomId = roomData.room_id;
                    const playBtn = document.getElementById('play-btn');
                    if (playBtn) {
                        playBtn.disabled = false;
                        playBtn.title = "";
                    }
                }
                // Update tool states based on room status
                if (typeof window.updateManualToolState === 'function') {
                    window.updateManualToolState();
                }

                // NEW: Load private matches instantly
                if (typeof window.loadPrivateMatches === 'function') {
                    window.loadPrivateMatches();
                }
            } catch (e) { console.warn('Error checking current room', e); }

            function applySavedFontSize(val, sliderId, labelId, previewId, cssVar, settingsKey) {
                console.log(`Applying saved font size: ${settingsKey} = ${val}`);
                if (val !== undefined && val !== null) {
                    const numVal = parseInt(val);
                    if (!isNaN(numVal)) {
                        window.userSettings[settingsKey] = numVal;

                        // DEBUG: confirm we are setting the property
                        console.log(`Setting ${cssVar} to ${numVal}px`);
                        document.documentElement.style.setProperty(cssVar, numVal + 'px');

                        const slider = document.getElementById(sliderId);
                        if (slider) slider.value = numVal;

                        const label = document.getElementById(labelId);
                        if (label) label.textContent = numVal + 'px';

                        const preview = document.getElementById(previewId);
                        if (preview) preview.style.fontSize = numVal + 'px';
                    }
                }
            }
        } else {
            updateAuthUI();
        }
    } catch (error) {
        console.error('Session check failed:', error);
        updateAuthUI();
    }
}

// Setup navigation
async function checkTournamentTurn() {
    if (!currentUser || window.currentUserIsGuest) return;
    try {
        const res = await fetch('/api/tournament/status');
        const data = await res.json();
        const btn = document.getElementById('nav-tournaments-btn');

        if (btn && data.user_status && data.user_status.has_turn) {
            btn.classList.add('has-turn');
        } else if (btn) {
            btn.classList.remove('has-turn');
        }
    } catch (e) { }
}

// Check initially and periodically (every 60s)
setTimeout(checkTournamentTurn, 2000);
setInterval(checkTournamentTurn, 60000);

// Note: Private Match polling is handled via loadPrivateMatches in private_matches.js

function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    // ... existing loop ...
    navButtons.forEach(btn => {
        btn.addEventListener('click', async () => {
            if (!btn.disabled) {
                const pageTarget = btn.getAttribute('data-page');

                // Special Case: How to Play is a Modal
                if (pageTarget === 'howtoplay') {
                    const modal = document.getElementById('modal-howtoplay');
                    if (modal) {
                        modal.classList.remove('hidden');
                    }
                    return; // Do not navigation
                }

                const pageId = 'page-' + pageTarget;

                // Store Tab Logic
                if (pageTarget === 'store') {
                    setupStoreTabs();
                }

                showPage(pageId);
                updateActiveNav(btn);
            }
        });
    });
}

function setupStoreTabs() {
    const tabs = document.querySelectorAll('.store-tab');
    if (tabs.length === 0) return;

    tabs.forEach(tab => {
        // Remove old listeners to avoid duplicates
        const newTab = tab.cloneNode(true);
        tab.parentNode.replaceChild(newTab, tab);

        newTab.addEventListener('click', () => {
            document.querySelectorAll('.store-tab').forEach(t => t.classList.remove('active'));
            newTab.classList.add('active');
            const category = newTab.getAttribute('data-category');
            console.log('Switching store category to:', category);
            // In a real app we'd filter items here
        });
    });
}


function setupModalListeners() {
    const modal = document.getElementById('modal-howtoplay');
    const closeBtn = document.getElementById('close-howtoplay');

    if (modal && closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.classList.add('hidden');
        });

        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
                modal.classList.add('hidden');
            }
        });

        // Close on click outside (optional, but nice)
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.add('hidden');
            }
        });
    }

    const genericModal = document.getElementById('generic-info-modal');
    const closeGenericBtn = document.getElementById('close-generic-modal');
    if (genericModal && closeGenericBtn) {
        closeGenericBtn.onclick = () => genericModal.classList.add('hidden');
        genericModal.onclick = (e) => { if (e.target === genericModal) genericModal.classList.add('hidden'); };
    }
}

function showPage(pageId) {
    // 0. Synchronize URL Hash (for Reload/Navigation consistency)
    if (window.location.hash !== "#" + pageId) {
        history.replaceState(null, null, "#" + pageId);
    }

    // 1. Update Page Visibility
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
    }

    // NEW: Load Private Matches instantly when entering Lobby
    if (pageId === 'page-lobby') {
        if (typeof window.loadPrivateMatches === 'function') {
            window.loadPrivateMatches();
        }
    }

    // 2. Handle Game Polling
    if (pageId === 'page-play') {
        renderGameColorBar(); // Render the rating color bar
        if (window.startGamePolling) {
            console.log('Entering Play page - starting polling');
            window.startGamePolling();
        }

        // Auto-focus the input field
        setTimeout(() => {
            const input = document.getElementById('word-input');
            if (input && !input.disabled) {
                input.focus();
            }
        }, 100);
    } else if (pageId === 'page-tools') {
        if (typeof window.refreshProfileTool === 'function') {
            window.refreshProfileTool(true); // Force refresh to current user
        }
        // Ensure manual tool state is correct
        if (typeof window.updateManualToolState === 'function') {
            window.updateManualToolState();
        }
    } else if (pageId === 'page-forums') {
        if (!currentUser) {
            navigateToPage('login');
            return;
        }
        if (typeof window.initForum === 'function') {
            window.initForum();
        }
    } else if (pageId === 'page-tournaments') {
        if (typeof window.initTournamentsPage === 'function') {
            window.initTournamentsPage();
        }
    } else if (pageId === 'page-contact') {
        // Default "FROM USER EMAIL" to theirs if they are signed up
        const contactEmailInput = document.getElementById('contact-user-email');
        if (contactEmailInput && currentUserEmail) {
            contactEmailInput.value = currentUserEmail;
        }
    } else {
        if (window.stopGamePolling) {
            console.log('Leaving Play page - stopping polling');
            window.stopGamePolling();
        }
    }

    // 3. Handle Lobby Music via Helper
    handleLobbyMusicState();
}

function navigateToPage(pageName) {
    const btn = document.querySelector(`.nav-btn[data-page="${pageName}"]`);
    showPage('page-' + pageName);
    if (btn) updateActiveNav(btn);
}
window.navigateToPage = navigateToPage;

function updateActiveNav(activeBtn) {
    // Whenever navigation changes, update states
    if (typeof window.updateManualToolState === 'function') {
        window.updateManualToolState();
    }

    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    if (activeBtn && activeBtn.classList) {
        activeBtn.classList.add('active');
    }
}

// Setup authentication listeners (Handled by second definition below)

function switchAuthTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelectorAll('.auth-form').forEach(form => {
        form.classList.remove('active');
    });

    if (tab === 'signin') {
        document.querySelector('[data-tab="signin"]').classList.add('active');
        document.getElementById('signin-form').classList.add('active');
    } else {
        document.querySelector('[data-tab="signup"]').classList.add('active');
        document.getElementById('signup-form').classList.add('active');
    }

    // Clear errors
    document.getElementById('signin-error').textContent = '';
    document.getElementById('signup-error').textContent = '';
}

async function handleSignIn() {
    const username = document.getElementById('signin-username').value;
    const password = document.getElementById('signin-password').value;
    const errorEl = document.getElementById('signin-error');

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        if (data.success) {
            currentUser = data.username;
            window.currentUser = currentUser;
            currentUserEmail = data.email;
            window.currentUserEmail = currentUserEmail;
            window.currentUserIsGuest = data.is_guest || false;
            navigateToLobby();
        } else {
            errorEl.textContent = data.error || data.message;
        }
    } catch (error) {
        errorEl.textContent = 'An error occurred. Please try again.';
        console.error('Login error:', error);
    }
}

async function handleSignUp() {
    const username = document.getElementById('signup-username').value;
    const email = document.getElementById('signup-email').value;
    const password = document.getElementById('signup-password').value;
    const confirmPassword = document.getElementById('signup-password-confirm').value;
    const errorEl = document.getElementById('signup-error');

    // Validation
    if (password !== confirmPassword) {
        errorEl.textContent = 'Passwords do not match';
        return;
    }

    if (password.length < 6) {
        errorEl.textContent = 'Password must be at least 6 characters';
        return;
    }

    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password, email })
        });

        const data = await response.json();

        if (data.success) {
            currentUser = data.username;
            window.currentUser = currentUser;
            currentUserEmail = email; // From the signup form
            window.currentUserEmail = currentUserEmail;
            window.currentUserIsGuest = false;
            navigateToLobby();
        } else {
            errorEl.textContent = data.error || data.message;
        }
    } catch (error) {
        errorEl.textContent = 'An error occurred. Please try again.';
        console.error('Registration error:', error);
    }
}

async function handleGuestLogin() {
    try {
        const response = await fetch('/api/guest-login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();

        if (data.success) {
            currentUser = data.username;
            window.currentUser = currentUser;
            window.currentUserIsGuest = true;
            navigateToLobby();
        } else {
            alert('Failed to login as guest. Please try again.');
        }
    } catch (error) {
        alert('An error occurred. Please try again.');
        console.error('Guest login error:', error);
    }
}

function navigateToLobby() {
    // Update UI for logged in state
    updateAuthUI();

    // Show lobby page
    showPage('page-lobby');
    const lobbyBtn = document.querySelector('.nav-btn[data-page="lobby"]');
    if (lobbyBtn) {
        updateActiveNav(lobbyBtn);
    }

    // Ensure private matches load instantly
    if (typeof window.loadPrivateMatches === 'function') {
        window.loadPrivateMatches();
    }
}

function updateAuthUI() {
    const loginNavBtn = document.getElementById('nav-login-btn');
    const userDisplay = document.getElementById('user-display');
    const usernameEl = document.getElementById('username-display');

    if (currentUser) {
        if (loginNavBtn) loginNavBtn.classList.add('hidden');
        if (userDisplay) userDisplay.classList.remove('hidden');
        if (usernameEl) {
            usernameEl.textContent = currentUser;
            const color = window.getRatingColor ? window.getRatingColor(0) : '#fff'; // Default or fetch actual rating
            usernameEl.style.color = 'var(--accent-color)';
        }
    } else {
        if (loginNavBtn) loginNavBtn.classList.remove('hidden');
        if (userDisplay) userDisplay.classList.add('hidden');
    }
}

async function handleLogout() {
    try {
        const response = await fetch('/api/logout', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            currentUser = null;
            window.currentUser = null;
            currentUserEmail = null;
            window.currentUserEmail = null;
            window.currentUserIsGuest = false;
            localStorage.removeItem('morpheme_username');
            localStorage.removeItem('morpheme_pm_state');
            localStorage.removeItem('private_match_active');
            localStorage.removeItem('tournament_play_active');

            updateAuthUI();
            showPage('page-login');

            // Reset Play button
            const playBtn = document.getElementById('play-btn');
            if (playBtn) {
                playBtn.disabled = true;
                playBtn.classList.remove('active');
                playBtn.title = "Authentication required";
            }
        }
    } catch (error) {
        console.error('Logout error:', error);
    }
}

// Setup authentication
function setupAuth() {
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.getAttribute('data-tab');
            switchAuthTab(tab);
        });
    });

    // Logout button
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }

    // Sign in form
    const signinForm = document.getElementById('signin-form');
    if (signinForm) {
        signinForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await handleSignIn();
        });
    }

    // Sign up form
    const signupForm = document.getElementById('signup-form');
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await handleSignUp();
        });
    }

    // Guest login button
    const guestBtn = document.getElementById('guest-login-btn');
    if (guestBtn) {
        guestBtn.addEventListener('click', async () => {
            await handleGuestLogin();
        });
    }
}

function updateActiveRoomsPanel(gameType, time, dimensions) {
    const infoDiv = document.getElementById('selected-game-info');
    const roomsList = document.getElementById('rooms-list');

    if (infoDiv) {
        infoDiv.innerHTML = `
            <p><strong>Game Type:</strong> ${gameType}</p>
            <p><strong>Board Dimensions:</strong> ${dimensions}</p>
            <p><strong>Time:</strong> ${time}</p>
        `;
    }

    if (roomsList) {
        roomsList.innerHTML = `
            <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
                No active rooms found. Create a new room to get started!
            </p>
        `;
    }
}

function selectRoom(roomName) {
    selectedRoom = roomName;
    const playBtn = document.getElementById('play-btn');
    if (playBtn) playBtn.disabled = false;
}

// Define standardized rating ranges globaly for reuse


// Helper to get color for any rating
window.getRatingColor = function (rating) {
    if (rating === undefined || rating === null || rating === 0) return '#ffffff';
    // Find matching range
    const match = RATING_RANGES.find(r => rating >= r.min && rating <= r.max);
    if (match) return match.color;

    // Fallback logic if out of bounds
    if (rating < 1) return RATING_RANGES[0].color;
    return '#000000'; // Super high
};

window.showAlertModal = function (title, message) {
    const modal = document.getElementById('generic-info-modal');
    const titleEl = document.getElementById('generic-modal-title');
    const bodyEl = document.getElementById('generic-modal-body');
    if (modal && titleEl && bodyEl) {
        titleEl.textContent = title;
        bodyEl.innerHTML = `<p style="padding: 30px; text-align: center; font-size: 1.2rem; line-height: 1.6; color: var(--text-primary);">${message}</p>`;
        modal.classList.remove('hidden');
        modal.style.display = 'flex';
    } else {
        alert(message);
    }
};

window.showConfirmModal = function (title, message, onConfirm) {
    const modal = document.getElementById('generic-confirm-modal');
    const titleEl = document.getElementById('generic-confirm-title');
    const bodyEl = document.getElementById('generic-confirm-body');
    const cancelBtn = document.getElementById('generic-confirm-cancel');
    const okBtn = document.getElementById('generic-confirm-ok');

    if (modal && titleEl && bodyEl && cancelBtn && okBtn) {
        titleEl.textContent = title;
        bodyEl.innerHTML = `<p style="white-space: pre-wrap; margin: 0;">${message}</p>`;

        const cleanup = () => {
            modal.style.display = 'none';
            modal.classList.add('hidden');
            cancelBtn.onclick = null;
            okBtn.onclick = null;
        };

        cancelBtn.onclick = () => { cleanup(); };
        document.getElementById('close-generic-confirm').onclick = () => { cleanup(); };

        okBtn.onclick = () => {
            cleanup();
            if (onConfirm) onConfirm();
        };

        modal.classList.remove('hidden');
        modal.style.display = 'flex';
    } else {
        // Fallback to native confirm if modal isn't injected yet
        if (confirm(message)) {
            if (onConfirm) onConfirm();
        }
    }
};

function renderGameColorBar() {
    const bar = document.getElementById('game-color-bar');
    if (!bar) return;

    // Use global rating ranges
    const ranges = RATING_RANGES;

    bar.innerHTML = '';

    ranges.forEach(range => {
        const segment = document.createElement('div');
        segment.className = 'color-bar-segment';
        segment.style.backgroundColor = range.color;
        // Tooltip text - Uppercase names for maximum legibility
        segment.setAttribute('data-name', `${range.name.toUpperCase()}`);
        segment.setAttribute('data-label', `${range.label}`);
        bar.appendChild(segment);
    });
}

// Presence Beacon: Notify server when browser tab is closed
// pagehide is more reliable than beforeunload in Safari and mobile browsers
window.addEventListener('pagehide', (e) => {
    if (window.currentUser) {
        // Note: pagehide fires on both navigation and closing
        // We use sendBeacon for reliable delivery
        navigator.sendBeacon('/api/presence/leave');
    }
});

// Optional: Also notify on visibility hidden (but keep short timeout on server to be safe)
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden' && window.currentUser) {
        // We don't necessarily want to mark offline just by switching tabs, 
        // but it's a good time to ensure the last_active is updated or beaconed if needed.
    }
});

// Export utility for other files
window.updateManualToolState = function () {
    const manualBtn = document.querySelector('.tool-nav-btn[data-tool="manual"]');
    if (manualBtn) {
        const inRoom = !!window.currentRoomId;
        // Also check if Play button is active (implies potential room session, though primarily currentRoomId matters)
        // Restoring strict disable if in room:
        if (inRoom) {
            manualBtn.disabled = true;
            manualBtn.title = "Manual tool is disabled while you are in a room.";
            manualBtn.classList.add('disabled');
        } else {
            manualBtn.disabled = false;
            manualBtn.title = "";
            manualBtn.classList.remove('disabled');
        }
    }
};

window.setCurrentUser = function (user) {
    currentUser = user;
    window.currentUser = user;
};
