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
    'nav-donate-btn': 'page-donate'
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
    { min: 6000, max: 6999, color: '#000000', label: '6000 - 6999', name: 'ALIEN BEING' },
    { min: 7000, max: 99999, color: '#a020f0', label: '7000+', name: 'SINGULARITY' }
];

// Single Instance Logic (Harden to prevent self-collision)
async function validateSingleInstance() {
    if (window.location.search.includes('force=true')) return true;
    
    return new Promise((resolve) => {
        const instanceId = Math.random().toString(36).substring(7);
        const channel = new BroadcastChannel('morpheme_instance');
        let duplicateFound = false;

        channel.onmessage = (event) => {
            if (event.data.type === 'PING' && event.data.senderId !== instanceId) {
                channel.postMessage({ type: 'PONG', senderId: instanceId });
            } else if (event.data.type === 'PONG' && event.data.senderId !== instanceId) {
                console.warn('[app.js] Duplicate Morpheme instance detected via BroadcastChannel.');
                duplicateFound = true;
                showAlreadyOpenScreen();
                resolve(false);
            }
        };

        // Ping and wait a short moment for a response
        channel.postMessage({ type: 'PING', senderId: instanceId });
        
        // Ensure we always resolve even if BroadcastChannel is blocked or fails
        setTimeout(() => {
            if (!duplicateFound) {
                console.info('[app.js] Instance validation passed (no duplicate found).');
                resolve(true); 
            }
        }, 800); 
    });
}

function showAlreadyOpenScreen() {
    document.body.innerHTML = `
        <div id="already-open-blocker" style="
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100vh; 
            background: #0d1117; 
            color: #fff; 
            font-family: 'Inter', sans-serif; 
            text-align: center;
            padding: 20px;
            position: fixed;
            top: 0; left: 0; width: 100%;
            z-index: 999999;
        ">
            <div style="background: rgba(255,255,255,0.05); padding: 40px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); max-width: 500px;">
                <img src="/static/images/morpheme.png" style="width: 80px; margin-bottom: 20px;">
                <h1 style="color: var(--accent-color, #e74c3c); margin-bottom: 10px;">Morpheme is already open</h1>
                <p style="color: #8b949e; line-height: 1.6; margin-bottom: 25px;">
                    To prevent rating desync and session issues, Morpheme's restricted to one tab at a time.
                </p>
                <div style="display: flex; gap: 10px; justify-content: center;">
                    <button onclick="window.close()" style="padding: 10px 20px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; cursor: pointer; font-weight: 600;">Close Tab</button>
                    <button onclick="window.location.href = window.location.pathname + '?force=true'" style="padding: 10px 20px; background: var(--accent-color, #e74c3c); border: none; color: #fff; border-radius: 6px; cursor: pointer; font-weight: 600;">Open Anyway</button>
                </div>
            </div>
        </div>
    `;
}

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    // 1. Core UI Setup (Always Run First for Responsiveness)
    setupNavigation();
    setupModalListeners();
    setupAuth(); // Initialize auth listeners
    setupContactForm(); // Initialize contact form listeners
    setupFirstInteractionMusic(); // Active immediately for early loading clicks
    
    // Mobile restriction: Hide/filter out Cube/3D options
    if (typeof filterCubeOnMobile === 'function') {
        filterCubeOnMobile();
    }

    // Dynamic layout panel restructuring for mobile devices
    if (typeof adjustLobbyLayoutForDevice === 'function') {
        adjustLobbyLayoutForDevice();
    }

    // Mobile swipe listener for the top header navigation
    if (typeof setupMobileHeaderSwipe === 'function') {
        setupMobileHeaderSwipe();
    }

    // 2. Single Instance Validation (Non-blocking for UI)
    const isSingle = await validateSingleInstance();
    if (!isSingle) return;

    // 3. Application Domain Logic
    // Stub or existing system initializers
    if (typeof handleLobbyMusicState === 'function') handleLobbyMusicState();
    if (typeof checkLobbyNotice === 'function') checkLobbyNotice();
    
    // NEW: Proper setup for Global listeners
    setupGlobalProfileLogic();
    
    // Initial State Check
    requestAnimationFrame(() => {
        if (typeof checkNavVisibility === 'function') checkNavVisibility();
        if (typeof checkForumActivity === 'function') checkForumActivity();
    });

    await checkSession();

    // NEW: Handle initial navigation based on hash OR landing page
    const hash = window.location.hash || '';
    const tournamentActive = localStorage.getItem('tournament_play_active');
    const privateActive = localStorage.getItem('private_match_active');

    if (currentUser) {
        // AUTHENTICATED: Always stay out of login page
        if (tournamentActive || privateActive || (hash === '#page-play' && window.currentRoomId)) {
            showPage('page-play');
            const playBtn = document.querySelector('.nav-btn[data-page="play"]');
            if (playBtn) updateActiveNav(playBtn);
        } else if (hash && hash.startsWith('#page-') && hash !== '#page-play' && hash !== '#page-login' && hash !== '#page-lobby') {
            const pageId = hash.substring(1);
            showPage(pageId);
            const pageName = pageId.replace('page-', '');
            const navBtn = document.querySelector(`.nav-btn[data-page="${pageName}"]`);
            if (navBtn) updateActiveNav(navBtn);
        } else {
            // Default to lobby for authenticated users (even if hash is empty, #page-login or #page-lobby)
            showPage('page-lobby');
            const lobbyBtn = document.querySelector('.nav-btn[data-page="lobby"]');
            if (lobbyBtn) updateActiveNav(lobbyBtn);
            handleLobbyMusicState();
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


// // Helper to start/stop music based on Page AND Setting
function handleLobbyMusicState() {
    console.log('[LobbyMusic] handleLobbyMusicState() triggered.');
    const lobbyMusic = document.getElementById('lobby-music');
    if (!lobbyMusic) {
        console.warn('[LobbyMusic] #lobby-music element not found in DOM.');
        return;
    }

    const onLobby = document.getElementById('page-lobby')?.classList.contains('active');
    const hasSettings = !!window.userSettings;
    const lobbyMusicSetting = hasSettings ? window.userSettings.lobby_music : undefined;
    const shouldPlay = onLobby && hasSettings && lobbyMusicSetting;

    console.log('[LobbyMusic] State assessment:', {
        onLobby,
        hasSettings,
        lobbyMusicSetting,
        shouldPlay,
        paused: lobbyMusic.paused,
        currentTime: lobbyMusic.currentTime,
        readyState: lobbyMusic.readyState
    });

    if (shouldPlay) {
        if (lobbyMusic.paused) {
            if (lobbyMusic.currentTime < 1) {
                try {
                    if (lobbyMusic.readyState >= 1) {
                        lobbyMusic.currentTime = 205;
                        console.log('[LobbyMusic] Direct seek to 205 succeeded (readyState >= 1).');
                    } else {
                        console.log('[LobbyMusic] readyState < 1, binding loadedmetadata seek hook.');
                        lobbyMusic.addEventListener('loadedmetadata', () => {
                            try { 
                                lobbyMusic.currentTime = 205; 
                                console.log('[LobbyMusic] Deferred seek to 205 succeeded on loadedmetadata.');
                            } catch(err) { console.warn('[LobbyMusic] Deferred seek failed:', err); }
                        }, { once: true });
                    }
                } catch (e) {
                    console.warn('[LobbyMusic] Seeking failed, postponing seek until ready:', e);
                }
            }

            console.log('[LobbyMusic] Attempting programatic .play()...');
            lobbyMusic.play()
                .then(() => {
                    console.log('[LobbyMusic] Programatic play() resolved successfully!');
                })
                .catch(e => {
                    console.log('[LobbyMusic] Programatic play() blocked by browser:', e.message);
                    setupFirstInteractionMusic();
                });

            // Ensure loop logic is attached
            lobbyMusic.ontimeupdate = function () {
                if (lobbyMusic.currentTime >= 295) { // 4:55 = 295 seconds
                    try { 
                        lobbyMusic.currentTime = 205; 
                        console.log('[LobbyMusic] Loop section boundary reached, rewound to 205.');
                    } catch(err) { console.warn('[LobbyMusic] Loop rewinding failed:', err); }
                }
            };
        } else {
            console.log('[LobbyMusic] Music is already playing.');
        }
    } else {
        console.log('[LobbyMusic] shouldPlay is false, ensuring audio is paused.');
        if (!lobbyMusic.paused) {
            lobbyMusic.pause();
            console.log('[LobbyMusic] Paused active playback.');
        }
    }
}

// Modern Browser Autoplay bypass helpers
function playMusicOnFirstInteraction() {
    console.log('[LobbyMusic] playMusicOnFirstInteraction() triggered by gesture.');
    const onLobby = document.getElementById('page-lobby')?.classList.contains('active');
    const onLoading = document.getElementById('page-loading')?.classList.contains('active');
    const onLogin = document.getElementById('page-login')?.classList.contains('active');
    const hasSettings = !!window.userSettings;
    const lobbyMusicSetting = hasSettings ? window.userSettings.lobby_music : undefined;

    console.log('[LobbyMusic] Gesture state evaluation:', {
        onLobby,
        onLoading,
        onLogin,
        hasSettings,
        lobbyMusicSetting
    });

    // Only attempt to play if we are in the lobby or loading, but NOT on the login page!
    if ((onLobby || onLoading) && !onLogin && hasSettings && lobbyMusicSetting) {
        const lobbyMusic = document.getElementById('lobby-music');
        if (lobbyMusic) {
            console.log('[LobbyMusic] Found audio element on gesture:', {
                paused: lobbyMusic.paused,
                currentTime: lobbyMusic.currentTime,
                readyState: lobbyMusic.readyState
            });
            if (lobbyMusic.currentTime < 1) {
                try {
                    if (lobbyMusic.readyState >= 1) {
                        lobbyMusic.currentTime = 205;
                        console.log('[LobbyMusic] Gesture seek to 205 succeeded (readyState >= 1).');
                    } else {
                        console.log('[LobbyMusic] Gesture readyState < 1, binding loadedmetadata seek hook.');
                        lobbyMusic.addEventListener('loadedmetadata', () => {
                            try { 
                                lobbyMusic.currentTime = 205; 
                                console.log('[LobbyMusic] Gesture deferred seek to 205 succeeded.');
                            } catch(err) { console.warn(err); }
                        }, { once: true });
                    }
                } catch (e) {
                    console.warn('[LobbyMusic] Gesture seeking failed:', e);
                }
            }
            console.log('[LobbyMusic] Attempting play() on gesture to unlock/unmute stream...');
            lobbyMusic.play()
                .then(() => {
                    console.log('[LobbyMusic] Lobby music playback started/unlocked successfully on user gesture!');
                    removeInteractionListeners();
                })
                .catch(e => console.log('[LobbyMusic] Gesture play() still blocked:', e.message));
        } else {
            console.warn('[LobbyMusic] #lobby-music element not found on gesture.');
        }
    }
}

function removeInteractionListeners() {
    const events = ['click', 'keydown', 'mousedown', 'touchstart'];
    events.forEach(evt => {
        document.removeEventListener(evt, playMusicOnFirstInteraction, { capture: true });
    });
}

function setupFirstInteractionMusic() {
    // Add event listeners without once: true so we don't prematurely delete them on early loading clicks!
    const events = ['click', 'keydown', 'mousedown', 'touchstart'];
    events.forEach(evt => {
        document.addEventListener(evt, playMusicOnFirstInteraction, { capture: true });
    });
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
            localStorage.setItem('morpheme_logged_in', 'true');
            currentUser = data.username;
            window.currentUser = currentUser;  // Expose globally
            currentUserEmail = data.email;
            window.currentUserEmail = currentUserEmail;
            window.currentUserIsGuest = data.is_guest; // Store guest status
            window.currentUserIsMod = data.is_mod; // Store mod status
            localStorage.setItem('morpheme_username', currentUser);

            window.lastPlayerRating = data.rating;
            updateAuthUI(data.rating); // Update UI for logged in state


            // LOAD ALL SETTINGS
            if (window.loadSettings) {
                window.loadSettings();
            } else {
                console.warn('[app.js] window.loadSettings not available yet');
            }

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

        } else {
            localStorage.removeItem('morpheme_logged_in');
            updateAuthUI();
        }
    } catch (error) {
        console.error('Session check failed:', error);
        localStorage.removeItem('morpheme_logged_in');
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
setTimeout(() => {
    checkTournamentTurn();
    checkForumActivity();
}, 2000);
setInterval(() => {
    checkTournamentTurn();
    checkForumActivity();
}, 60000);

async function checkForumActivity() {
    if (!currentUser) return;
    try {
        const res = await fetch('/api/forum/categories');
        const data = await res.json();
        const btn = document.getElementById('btn-forums');
        if (!btn) return;

        const lastViewed = JSON.parse(localStorage.getItem('forum_last_viewed') || '{}');
        let hasNewGlobally = false;

        data.categories.forEach(cat => {
            const lastContent = cat.last_content_at ? new Date(cat.last_content_at).getTime() : 0;
            // Coerce to number and check for 0
            const lastView = Number(lastViewed[cat.id]) || 0;
            const hasNew = lastContent > lastView;
            if (hasNew) {
                hasNewGlobally = true;
            }
            console.debug(`[Forum Global] Cat ${cat.id}: content=${lastContent}, view=${lastView}, new=${hasNew}`);
        });

        if (hasNewGlobally) {
            btn.classList.add('has-new');
        } else {
            btn.classList.remove('has-new');
        }
    } catch (e) {
        console.warn('[Forum] Activity check failed', e);
    }
}
window.checkForumActivity = checkForumActivity;

function filterCubeOnMobile() {
    const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    if (!isMobile) return;

    console.log('[Mobile] Mobile device detected. Filtering out Cube/3D game options.');

    // 1. Hide the Lobby 3D game matrix completely
    const matrix3D = document.querySelector('.matrix-3d');
    if (matrix3D) {
        matrix3D.style.display = 'none';
    }

    // 2. Remove the 3x3x3 option from Solo & Friends dropdown in Lobby
    const sfConfigDims = document.getElementById('sf-config-dims');
    if (sfConfigDims) {
        const cubeOption = sfConfigDims.querySelector('option[value="3x3x3"]');
        if (cubeOption) {
            cubeOption.remove();
        }
    }

    // 3. Remove 3x3x3 option from rankings/leaderboard filter dropdown
    const rankingsFilterDims = document.getElementById('rankings-filter-dims');
    if (rankingsFilterDims) {
        const cubeOption = rankingsFilterDims.querySelector('option[value="3x3x3"]');
        if (cubeOption) {
            cubeOption.remove();
        }
    }
}
window.filterCubeOnMobile = filterCubeOnMobile;

function adjustLobbyLayoutForDevice() {
    const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const soloFriends = document.querySelector('.solo-friends-section');
    const lobbyGrid = document.querySelector('.lobby-grid');
    const gameTypesPanel = document.querySelector('.game-types-panel');
    
    if (!soloFriends || !lobbyGrid || !gameTypesPanel) return;
    
    if (isMobile) {
        // Move Solo & Friends section to be the last direct child of lobbyGrid (after active-rooms-panel)
        if (soloFriends.parentNode !== lobbyGrid) {
            console.log('[Layout] Moving Solo & Friends section to the end of lobby-grid for mobile.');
            lobbyGrid.appendChild(soloFriends);
        }
    } else {
        // Restore Solo & Friends section to be inside gameTypesPanel for desktop
        if (soloFriends.parentNode !== gameTypesPanel) {
            console.log('[Layout] Restoring Solo & Friends section inside game-types-panel for desktop.');
            gameTypesPanel.appendChild(soloFriends);
        }
    }
}
window.adjustLobbyLayoutForDevice = adjustLobbyLayoutForDevice;

// Setup responsive window resize listener to dynamically reposition panels
window.addEventListener('resize', () => {
    if (typeof adjustLobbyLayoutForDevice === 'function') {
        adjustLobbyLayoutForDevice();
    }
});

function setupMobileHeaderSwipe() {
    const logo = document.querySelector('.logo');
    const header = document.querySelector('.header');
    
    if (!logo || !header) return;
    
    let startX = 0;
    let startY = 0;
    
    logo.addEventListener('touchstart', (e) => {
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
    }, { passive: true });
    
    logo.addEventListener('touchend', (e) => {
        const diffX = e.changedTouches[0].clientX - startX;
        const diffY = e.changedTouches[0].clientY - startY;
        
        // Detect horizontal swipe left (threshold of 30px)
        if (diffX < -30 && Math.abs(diffX) > Math.abs(diffY)) {
            console.log('[Swipe] Swiped left on logo. Revealing navigation menu.');
            header.scrollTo({
                left: header.clientWidth,
                behavior: 'smooth'
            });
        }
    }, { passive: true });
    
    // Also, if they swipe right on the nav bar, allow them to swipe back to the logo!
    const nav = document.querySelector('.nav');
    if (nav) {
        nav.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        }, { passive: true });
        
        nav.addEventListener('touchend', (e) => {
            const diffX = e.changedTouches[0].clientX - startX;
            const diffY = e.changedTouches[0].clientY - startY;
            
            // Detect horizontal swipe right (threshold of 30px)
            if (diffX > 30 && Math.abs(diffX) > Math.abs(diffY)) {
                // Only swipe back if they are scrolled to the nav menu start
                if (header.scrollLeft > 50) {
                    console.log('[Swipe] Swiped right on nav. Showing logo.');
                    header.scrollTo({
                        left: 0,
                        behavior: 'smooth'
                    });
                }
            }
        }, { passive: true });
    }
}
window.setupMobileHeaderSwipe = setupMobileHeaderSwipe;

// Note: Private Match polling is handled via loadPrivateMatches in private_matches.js

function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    console.log(`[setupNavigation] Initializing ${navButtons.length} buttons.`);
    
    navButtons.forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const pageTarget = btn.dataset.page;
            console.log(`[setupNavigation] Clicked Target: ${pageTarget}`);
            
            if (btn.disabled) return;

            // Handle Modals (Not Pages)
            if (pageTarget === 'howtoplay') {
                const modal = document.getElementById('modal-howtoplay');
                if (modal) {
                    modal.classList.add('forced-show');
                    modal.classList.remove('hidden');
                }
                return;
            }

            // Default Page Navigation
            const pageId = 'page-' + pageTarget;
            showPage(pageId);
            updateActiveNav(btn);
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
            
            // Filter items
            document.querySelectorAll('.store-item').forEach(item => {
                if (item.getAttribute('data-category') === category) {
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            });

            console.log('Switching store category to:', category);
        });
    });

    // Default to first active tab's category
    const activeTab = document.querySelector('.store-tab.active');
    if (activeTab) {
        const category = activeTab.getAttribute('data-category');
        document.querySelectorAll('.store-item').forEach(item => {
            if (item.getAttribute('data-category') === category) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
    }
}


function setupModalListeners() {
    const modal = document.getElementById('modal-howtoplay');
    const closeBtn = document.getElementById('close-howtoplay');

    if (modal && closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.classList.remove('forced-show');
            modal.classList.add('hidden');
        });

        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
                modal.classList.remove('forced-show');
                modal.classList.add('hidden');
            }
        });

        // Close on click outside (optional, but nice)
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('forced-show');
                modal.classList.add('hidden');
            }
        });
    }

    const genericModal = document.getElementById('generic-info-modal');
    const closeGenericBtn = document.getElementById('close-generic-modal');
    if (genericModal && closeGenericBtn) {
        const closeModal = () => {
            genericModal.classList.remove('forced-show');
            genericModal.classList.add('hidden');
            window._hasPriorityModal = false; // Reset priority flag on close
        };
        closeGenericBtn.onclick = closeModal;
        genericModal.onclick = (e) => { 
            if (e.target === genericModal) closeModal(); 
        };
    }
}

function showPage(pageId) {
    if (window.hideLoadingOverlay) window.hideLoadingOverlay();
    // 0. Synchronize URL Hash (for Reload/Navigation consistency)
    if (window.location.hash !== "#" + pageId) {
        history.replaceState(null, null, "#" + pageId);
    }

    // Auto-hide modals/overlays when navigating pages
    const overlays = document.querySelectorAll('.modal-window, .mini-profile-overlay, .review-overlay, .overlay');
    overlays.forEach(o => {
        o.classList.remove('forced-show');
        o.classList.add('hidden');
    });

    // 1. Update Page Visibility
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
        if (page.id === pageId) {
            page.classList.add('active');
            page.style.opacity = '1'; // Explicitly force visibility
        }
    });

    // Standardize: Rating color bar ONLY appears on the Play page
    const colorBar = document.getElementById('game-color-bar');
    if (colorBar) {
        if (pageId === 'page-play') {
            colorBar.style.display = 'flex';
        } else {
            colorBar.style.display = 'none';
        }
    }

    // NEW: Load Private Matches instantly when entering Lobby
    if (pageId === 'page-lobby') {
        if (typeof window.loadPrivateMatches === 'function') {
            window.loadPrivateMatches();
        }
        if (typeof window.checkLobbyNotice === 'function' && !window._lobbyNoticeShownThisSession) {
            window.checkLobbyNotice();
        }
    }

    // NEW: Automatically update navigation button active state
    const pageName = pageId.replace('page-', '');
    const navBtn = document.querySelector(`.nav-btn[data-page="${pageName}"]`);
    if (navBtn) {
        updateActiveNav(navBtn);
    }

    // 2. Handle Game Polling
    if (pageId === 'page-play') {
        renderGameColorBar(); // Render the rating color bar
        if (window.startGamePolling) {
            console.log('app.js fully loaded - version with UI optimizations');
            window.startGamePolling();
        }

        // Trigger dynamic layout adjustment to fit board snugly on entry
        if (typeof window.checkBoardOverflow === 'function') {
            setTimeout(window.checkBoardOverflow, 50);
        }

        // Auto-focus the input field
        setTimeout(() => {
            const input = document.getElementById('word-input');
            if (input && !input.disabled) {
                input.focus();
            }
        }, 100);
    } else if (pageId === 'page-mods') {
        if (typeof window.loadAddedWordsConfig === 'function') {
            window.loadAddedWordsConfig();
        }
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
    } else if (pageId === 'page-donate') {
        if (typeof window.initDonatePage === 'function') {
            window.initDonatePage();
        }
    } else {
        if (window.stopGamePolling) {
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
    const captcha = document.getElementById('signin-captcha').value;
    const errorEl = document.getElementById('signin-error');

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password, captcha })
        });

        const data = await response.json();

        if (data.success) {
            localStorage.setItem('morpheme_logged_in', 'true');
            currentUser = data.username;
            window.currentUser = currentUser;
            currentUserEmail = data.email;
            window.currentUserEmail = currentUserEmail;
            window.currentUserIsGuest = data.is_guest || false;
            window.currentUserIsMod = data.is_mod || false; // Set here too
            
            // Critical: Re-check mod status immediately after successful login
            if (typeof checkModStatus === 'function') {
                checkModStatus();
            }
            if (data.settings) {
                console.log('[settings.js] Settings loaded:', data.settings);
                applySettings(data.settings);
            } else {
                console.log('[settings.js] No settings found, applying defaults');
                applySettings(window.userSettings || {});
            }
            
            window.lastPlayerRating = data.rating;
            navigateToLobby(data.rating);
        } else {
            errorEl.textContent = data.error || data.message;
            if (typeof window.refreshCaptchas === 'function') window.refreshCaptchas();
        }
    } catch (error) {
        errorEl.textContent = 'An error occurred. Please try again.';
        console.error('Login error:', error);
        if (typeof window.refreshCaptchas === 'function') window.refreshCaptchas();
    }
}

async function handleSignUp() {
    const username = document.getElementById('signup-username').value;
    const email = document.getElementById('signup-email').value;
    const password = document.getElementById('signup-password').value;
    const confirmPassword = document.getElementById('signup-password-confirm').value;
    const captcha = document.getElementById('signup-captcha').value;
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
            body: JSON.stringify({ username, password, email, captcha })
        });

        const data = await response.json();

        if (data.success) {
            localStorage.setItem('morpheme_logged_in', 'true');
            currentUser = data.username;
            window.currentUser = currentUser;
            currentUserEmail = email; // From the signup form
            window.currentUserEmail = currentUserEmail;
            window.currentUserIsGuest = false;
            window.lastPlayerRating = data.rating;
            navigateToLobby(data.rating);
        } else {
            errorEl.textContent = data.error || data.message;
            if (typeof window.refreshCaptchas === 'function') window.refreshCaptchas();
        }
    } catch (error) {
        errorEl.textContent = 'An error occurred. Please try again.';
        console.error('Registration error:', error);
        if (typeof window.refreshCaptchas === 'function') window.refreshCaptchas();
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
            localStorage.setItem('morpheme_logged_in', 'true');
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

function navigateToLobby(rating = null) {
    if (rating) window.lastPlayerRating = rating;

    // Update UI for logged in state
    updateAuthUI(rating);

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

    // Load Settings
    if (window.loadSettings) {
        window.loadSettings();
    }

    // New: Re-check moderator status on lobby navigation
    if (typeof checkModStatus === 'function') {
        checkModStatus();
    }
}


function updateAuthUI(rating = null) {
    const loginNavBtn = document.getElementById('nav-login-btn');
    const userDisplay = document.getElementById('user-display');
    const usernameEl = document.getElementById('username-display');

    if (currentUser) {
        if (loginNavBtn) loginNavBtn.classList.add('hidden');
        if (userDisplay) userDisplay.classList.remove('hidden');
        if (usernameEl) {
            usernameEl.textContent = currentUser;
            usernameEl.style.color = 'var(--accent-color)';
            // RESTORED: Profile navigation on click
            usernameEl.onclick = () => {
                if (typeof window.performProfileSearch === 'function') {
                    showPage('page-tools');
                    window.showTool('profile'); // Uses the new window.showTool helper
                    window.performProfileSearch(currentUser);
                }
            };
        }

        // Handle Rating Bar
        renderGameColorBar();
        if (rating || window.lastPlayerRating) {
            updateUserRatingHighlight(rating || window.lastPlayerRating);
        }

        // Handle Mods Button
        const modsBtn = document.getElementById('nav-mods-btn');
        if (modsBtn) {
            const isAuthorized = window.currentUserIsMod;
            modsBtn.style.display = isAuthorized ? 'block' : 'none';
        }

        // Auto-scroll header to reveal the menu items when logged in on mobile devices
        const header = document.querySelector('.header');
        if (header && window.innerWidth <= 900) {
            setTimeout(() => {
                header.scrollTo({
                    left: header.clientWidth,
                    behavior: 'smooth'
                });
            }, 350); // Fluid delay to align with page routing rendering
        }

    } else {
        if (loginNavBtn) loginNavBtn.classList.remove('hidden');
        if (userDisplay) userDisplay.classList.add('hidden');
        const modsBtn = document.getElementById('nav-mods-btn');
        if (modsBtn) modsBtn.style.display = 'none';

        // Hide Rating Bar when logged out
        const bar = document.getElementById('game-color-bar');
        if (bar) bar.innerHTML = '';

        // Scroll back to logo on logout on mobile devices
        const header = document.querySelector('.header');
        if (header && window.innerWidth <= 900) {
            header.scrollTo({
                left: 0,
                behavior: 'smooth'
            });
        }
    }
}




async function handleLogout() {
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.textContent = 'Logging out...';
        logoutBtn.style.opacity = '0.7';
        logoutBtn.disabled = true;
    }

    try {
        console.info('[Auth] Logout initiated. Preserving global markers...');
        await fetch('/api/logout', { method: 'POST' });
        
        // Preserve global "read" states (Notices, Forum markers) across login sessions
        const noticeId = localStorage.getItem('morpheme_read_notice_id');
        const forumViewed = localStorage.getItem('forum_last_viewed');
        const userSettings = localStorage.getItem('morpheme_user_settings');
        
        console.info(`[Auth] Preservation: noticeId=${noticeId}, forumViewed=${forumViewed}`);

        // Clear only session-specific or sensitive data
        localStorage.clear();
        sessionStorage.clear();
        
        // Restore non-sensitive global markers
        if (noticeId) localStorage.setItem('morpheme_read_notice_id', noticeId);
        if (forumViewed) localStorage.setItem('forum_last_viewed', forumViewed);
        if (userSettings) localStorage.setItem('morpheme_user_settings', userSettings);
        
        console.info('[Auth] Markers restored. Redirecting...');
        // Use a short delay to ensure UI shows the 'Logging out' state briefly for feedback
        setTimeout(() => {
            window.location.href = '/';
        }, 100);
    } catch (error) {
        console.error('Logout error:', error);
        if (logoutBtn) {
            logoutBtn.textContent = 'Logout';
            logoutBtn.style.opacity = '1';
            logoutBtn.disabled = false;
        }
        alert('Logout failed. Please check your connection.');
    }
}



// setupGlobalProfileLogic - Handle global profile triggers (like the rating bar)
function setupGlobalProfileLogic() {
    const bar = document.getElementById('game-color-bar');
    if (bar) {
        bar.addEventListener('click', (e) => {
            const segment = e.target.closest('.color-bar-segment');
            if (segment && window.showMiniProfile) {
                // If it's the user-rating-segment, we might show current user, 
                // but usually the user wants to see their stats.
                // For now, let's look for any user whose rating might be clicked? 
                // Actually, the request says "colored square representing rating does not open a brief profile".
                // We'll show the current user's mini profile as a default if none selected.
                if (window.currentUser) {
                    window.showMiniProfile(window.currentUser);
                }
            }
        });
    }
}

function setupMobileLogic() {
    // Add any mobile-specific listeners or adjustments
    window.addEventListener('resize', () => {
        const isMobile = window.innerWidth <= 768;
        document.body.classList.toggle('is-mobile', isMobile);
    });
}

// Setup authentication
function setupAuth() {
    // CAPTCHA helper logic
    window.refreshCaptchas = function() {
        document.querySelectorAll('.captcha-img').forEach(img => {
            img.src = '/api/captcha?t=' + Date.now();
        });
        const signinCaptcha = document.getElementById('signin-captcha');
        const signupCaptcha = document.getElementById('signup-captcha');
        if (signinCaptcha) signinCaptcha.value = '';
        if (signupCaptcha) signupCaptcha.value = '';
    };

    // Attach click handlers to refresh CAPTCHA on image container clicks
    document.querySelectorAll('.captcha-image-box').forEach(box => {
        box.addEventListener('click', () => {
            window.refreshCaptchas();
        });
    });

    // Initial CAPTCHA population
    window.refreshCaptchas();

    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.getAttribute('data-tab');
            switchAuthTab(tab);
            window.refreshCaptchas(); // Automatically refresh when switching views
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
    return '#a020f0'; // Super high (Purple)
};

window.showAlertModal = function (title, message, priority = false) {
    const modal = document.getElementById('generic-info-modal');
    const titleEl = document.getElementById('generic-modal-title');
    const bodyEl = document.getElementById('generic-modal-body');
    
    // If a priority modal is already showing (e.g. Inactivity Kick), 
    // don't let a normal notice (e.g. Lobby Notice) overwrite it immediately.
    if (window._hasPriorityModal && !priority) {
        console.log('[Modal] Normal modal suppressed by priority modal.');
        return;
    }

    if (modal && titleEl && bodyEl) {
        if (priority) window._hasPriorityModal = true;
        titleEl.textContent = title;
        bodyEl.innerHTML = `<p style="padding: 30px; text-align: center; font-size: 1.2rem; line-height: 1.6; color: var(--text-primary);">${message}</p>`;
        modal.classList.remove('hidden');
        modal.style.display = 'flex';
        
        // Reset priority flag when modal is manually closed or after a long timeout
        // (Close listener is in setupModalListeners)
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
    
    // Optimization: Build entire HTML string first to avoid DOM thrashing
    let html = '';
    RATING_RANGES.forEach(range => {
        html += `<div class="color-bar-segment" 
                      style="background-color: ${range.color};" 
                      data-name="${range.name.toUpperCase()}" 
                      data-label="${range.min}-${range.max === 9999 ? '∞' : range.max}">
                 </div>`;
    });
    
    if (bar.innerHTML !== html) {
        bar.innerHTML = html;
    }
}

/**
 * Highlights the segment in the color bar that matches current uniqueness/difficulty
 */
/**
 * Highlights the segment in the color bar that matches current board difficulty
 */
function updateColorBarHighlight(difficulty, uniqueness) {
    const bar = document.getElementById('game-color-bar');
    if (!bar) return;

    const segments = bar.querySelectorAll('.color-bar-segment');
    segments.forEach(s => s.classList.remove('active'));

    let targetIndex = 0;
    const pct = Math.round(uniqueness * 100);
    
    // TIER MAPPING FOR SPINNER HIGHLIGHT (13 segments total)
    if (pct < 40) {
        targetIndex = Math.floor((pct / 40) * 5);
    } else if (pct < 70) {
        targetIndex = 5 + Math.floor(((pct - 40) / 30) * 7);
    } else {
        targetIndex = 12;
    }
    
    targetIndex = Math.min(Math.max(targetIndex, 0), 12);
    
    if (segments[targetIndex]) {
        segments[targetIndex].classList.add('active');
    }
}

/**
 * Specifically highlights the segment for the USER'S CURRENT RATING with the pulsing effect.
 */
window.updateUserRatingHighlight = function(rating) {
    const bar = document.getElementById('game-color-bar');
    if (!bar) return;

    const segments = bar.querySelectorAll('.color-bar-segment');
    segments.forEach(s => s.classList.remove('user-rating-segment'));

    if (rating === undefined || rating === null || rating <= 0) return;

    // Find the range index that matches the user's rating
    const rangeIndex = RATING_RANGES.findIndex(r => rating >= r.min && rating <= r.max);
    
    if (rangeIndex !== -1 && segments[rangeIndex]) {
        segments[rangeIndex].classList.add('user-rating-segment');
    }
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

// Global Idle Logout (24 Hours)
(function () {
    let idleSeconds = 0;
    const idleLimit = 86400; // 86400s = 24 Hours

    function resetIdle() {
        idleSeconds = 0;
    }

    // Interaction resets the timer
    window.addEventListener('mousemove', resetIdle, { passive: true });
    window.addEventListener('mousedown', resetIdle, { passive: true });
    window.addEventListener('keydown', resetIdle, { passive: true });
    window.addEventListener('touchstart', resetIdle, { passive: true });
    window.addEventListener('scroll', resetIdle, { passive: true });

    setInterval(() => {
        // If the tab is hidden, we increment faster or just treat as idle
        // But the user's rule says "if idle for longer than an hour"
        idleSeconds++;

        if (idleSeconds >= idleLimit) {
            // Check if we are actually logged in (currentUser is set in app.js context)
            if (window.currentUser || localStorage.getItem('morpheme_username')) {
                console.log("[IdleLogout] User inactive for 24 hours. Logging out.");
                handleLogout();
            }
        }
    }, 1000);
})();

window.checkLobbyNotice = async function() {
    if (window._lobbyNoticeShownThisSession) {
        console.debug('[LobbyNotice] Already checked/shown this session.');
        return;
    }
    
    try {
        const res = await fetch('/api/mods/lobby-notice');
        const data = await res.json();
        
        if (data && data.notice && data.notice.trim() !== '') {
             const viewedId = localStorage.getItem('morpheme_read_notice_id');
             const currentId = String(data.notice_id);
             
             console.info(`[LobbyNotice] Current: ${currentId}, Local: ${viewedId}`);
             
             if (viewedId !== currentId) {
                  console.info(`[LobbyNotice] Version mismatch - showing alert.`);
                  window.showAlertModal("Morpheme News & Notices", data.notice);
                  
                  localStorage.setItem('morpheme_read_notice_id', currentId);
                  window._lobbyNoticeShownThisSession = true;
             } else {
                  console.info(`[LobbyNotice] Versions match (${currentId}). Skipping.`);
                  window._lobbyNoticeShownThisSession = true;
             }
        } else {
             window._lobbyNoticeShownThisSession = true;
        }
    } catch (e) {
        console.warn('[LobbyNotice] Failed to check notice:', e);
    }
};


// UI Feedback Helpers
window.showLoadingOverlay = function(message = 'Loading...') {
    let overlay = document.getElementById('global-loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'global-loading-overlay';
        overlay.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(5px); display: flex; flex-direction: column; justify-content: center; align-items: center; z-index: 99999; color: #fff; font-family: sans-serif;';
        overlay.innerHTML = '<div class="loading-spinner" style="width: 50px; height: 50px; border: 5px solid rgba(255,255,255,0.1); border-top-color: #e94560; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 20px;"></div><div id="loading-message" style="font-size: 1.2rem; font-weight: 600; letter-spacing: 1px;">' + message + '</div><style>@keyframes spin { to { transform: rotate(360deg); } }</style>';
        document.body.appendChild(overlay);
    } else {
        const msgEl = overlay.querySelector('#loading-message');
        if (msgEl) msgEl.textContent = message;
        overlay.style.display = 'flex';
    }
};

window.hideLoadingOverlay = function() {
    const overlay = document.getElementById('global-loading-overlay');
    if (overlay) overlay.style.display = 'none';
};

console.log('app.js fully loaded - version with UI optimizations');
