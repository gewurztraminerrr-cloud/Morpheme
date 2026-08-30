// Disable browser's automatic scroll restoration on navigation/reload
if ('scrollRestoration' in history) {
    history.scrollRestoration = 'manual';
}

// Client Auto-Sync Version Check
const CURRENT_APP_BUILD = '33127';
(function() {
    try {
        const lastBuild = localStorage.getItem('morpheme_build_version');
        if (lastBuild && lastBuild !== CURRENT_APP_BUILD) {
            localStorage.setItem('morpheme_build_version', CURRENT_APP_BUILD);
            if (!sessionStorage.getItem('morpheme_reload_done')) {
                sessionStorage.setItem('morpheme_reload_done', '1');
                window.location.reload(true);
                return;
            }
        } else {
            localStorage.setItem('morpheme_build_version', CURRENT_APP_BUILD);
        }
        sessionStorage.removeItem('morpheme_reload_done');
    } catch(e) {}
})();

// Navigation system
const pages = {
    'nav-login-btn': 'page-login',
    'btn-how-to-play': 'page-how-to-play',
    'btn-lobby': 'page-lobby',
    'btn-play': 'page-play',
    'btn-leaderboards': 'page-leaderboards',
    'nav-tournaments-btn': 'page-tournaments',
    'nav-profile-btn': 'page-profile',
    'btn-forums': 'page-forums',
    'btn-tools': 'page-tools',
    'btn-settings': 'page-settings',
    'nav-donate-btn': 'page-donate'
};

let currentUser = null;
let currentUserEmail = null;
let selectedRoom = null;
let sessionStartTime = Date.now();
window.sessionStartTime = sessionStartTime;
window.currentUserIsMod = false;
window.currentUserIsRootMod = false;

window.currentUserConfigRatings = {};

async function loadCurrentUserConfigRatings() {
    if (!window.currentUser || window.currentUserIsGuest) {
        window.currentUserConfigRatings = {};
        return;
    }
    try {
        const resp = await fetch(`/api/profile/${encodeURIComponent(window.currentUser)}?t=${Date.now()}`);
        const data = await resp.json();
        if (data && data.config_ratings) {
            window.currentUserConfigRatings = data.config_ratings;
            console.log('Loaded config ratings for current user:', window.currentUserConfigRatings);
            if (window.currentLobbyConfig) {
                const activeCfg = window.currentLobbyConfig;
                if (window.updateMyRatingButton) {
                    window.updateMyRatingButton(
                        activeCfg.gameType,
                        activeCfg.boardDimensions,
                        activeCfg.timeLimit
                    );
                } else {
                    const ratingBtn = document.getElementById('my-rating-btn');
                    if (ratingBtn && typeof window.getUserConfigRating === 'function') {
                        const exact = window.getUserConfigRating(activeCfg.gameType, activeCfg.boardDimensions, activeCfg.timeLimit);
                        ratingBtn.textContent = `My Rating (${exact})`;
                        ratingBtn.dataset.rating = exact;
                        ratingBtn.style.display = 'inline-block';
                    }
                }
            } else {
                const ratingBtn = document.getElementById('my-rating-btn');
                if (ratingBtn) {
                    ratingBtn.textContent = 'My Rating';
                    ratingBtn.removeAttribute('data-rating');
                }
            }
        } else {
            window.currentUserConfigRatings = {};
        }
    } catch (e) {
        console.error('Failed to load user config ratings:', e);
        window.currentUserConfigRatings = {};
    }
}
window.loadCurrentUserConfigRatings = loadCurrentUserConfigRatings;

// Define standardized rating ranges globaly for reuse
const RATING_RANGES = [
    // --- THE CLIMB (1 - 1399) ---
    // Greens (1 - 699): The Foundation
    { min: 1, max: 99, color: '#e6ffe6', label: '1 - 99', name: 'Initiate', desc: "You've taken your first step into Morpheme! Every expert started right here. Focus on learning the grid layouts, and watch your score climb!" },
    { min: 100, max: 199, color: '#ccffcc', label: '100 - 199', name: 'Apprentice', desc: "You're building a solid foundation. You've learned the basics and are beginning to spot longer words. Keep practicing to sharpen your vocabulary!" },
    { min: 200, max: 299, color: '#99ff99', label: '200 - 299', name: 'Practitioner', desc: "Your skills are growing! You're consistently finding good words and adjusting to different board parameters. Keep pushing your limits!" },
    { min: 300, max: 399, color: '#66ff66', label: '300 - 399', name: 'Scholar', desc: "A dedicated student of the game! Your knowledge of prefixes, suffixes, and word structures is starting to show. You're well on your way to mastery." },
    { min: 400, max: 499, color: '#33ff33', label: '400 - 499', name: 'Adept', desc: "You've achieved impressive proficiency! You spot high-value words quickly and navigate complex grid patterns with ease. Excellent progress!" },
    { min: 500, max: 599, color: '#00ff00', label: '500 - 599', name: 'Specialist', desc: "Your gameplay is highly focused and efficient. You excel under pressure and can turn any difficult board into a high-scoring round. Keep it up!" },
    { min: 600, max: 699, color: '#00cc00', label: '600 - 699', name: 'Expert', desc: "An expert word finder! Your vocabulary is extensive, and you consistently dominate standard rooms. You have a deep understanding of board dynamics." },

    // Blues (700 - 1399): The Sky & Ocean
    { min: 700, max: 799, color: '#66ccff', label: '700 - 799', name: 'Vanguard', desc: "You're at the forefront of the competition! You lead the pack with quick reflexes and strategic word selections. You're ready for the highest tiers." },
    { min: 800, max: 899, color: '#33bbff', label: '800 - 899', name: 'Sentinel', desc: "A watchful guardian of the leaderboard! Your defensive and offensive strategies are perfectly balanced. You rarely miss a high-value bonus word." },
    { min: 900, max: 999, color: '#00aaff', label: '900 - 999', name: 'Strategist', desc: "A master of tactics! You plan your moves ahead, using Either/Or and Valued Letters formats to maximize every single point. Brilliant mind!" },
    { min: 1000, max: 1099, color: '#0088ff', label: '1000 - 1099', name: 'Tactician', desc: "Precision is your weapon! You execute perfect paths and find obscure combinations that others overlook. A truly formidable competitor." },
    { min: 1100, max: 1199, color: '#0066ff', label: '1100 - 1199', name: 'Virtuoso', desc: "Your play style is an art form! You glide through the grid with incredible speed and fluidity, putting on a clinic in every round." },
    { min: 1200, max: 1299, color: '#0044ff', label: '1200 - 1299', name: 'Master', desc: "You have achieved true mastery! Your name commands respect on the server. You play with absolute confidence and standard-setting execution." },
    { min: 1300, max: 1399, color: '#0000ff', label: '1300 - 1399', name: 'Grandmaster', desc: "One of the absolute elite! Your deep understanding of the game's mechanics is matched only by your vast vocabulary. A legendary competitor." },

    // --- THE HEAT (1400 - 2499) ---
    // Yellows → light-to-dark progression into oranges
    { min: 1400, max: 1499, color: '#ffff99', label: '1400 - 1499', name: 'Elite', desc: "A rare tier of excellence! You've broken into the highest brackets through sheer dedication and skill. You are an inspiration to other players." },      // Pale lemon yellow
    { min: 1500, max: 1599, color: '#ffff00', label: '1500 - 1599', name: 'Champion', desc: "A true champion of Morpheme! You thrive in the most intense rooms and tournaments, consistently rising to meet every challenge." },   // True pure yellow
    { min: 1600, max: 1699, color: '#ffd700', label: '1600 - 1699', name: 'Titan', desc: "A colossal force on the board! Your presence dominates any room you enter, and your high scores are a testament to your outstanding ability." },      // Rich golden yellow
    { min: 1700, max: 1799, color: '#ffaa00', label: '1700 - 1799', name: 'Paragon', desc: "The model of perfect play! You demonstrate flawless word-finding ability and strategic positioning in every single round." },    // Deep amber

    // Oranges
    { min: 1800, max: 1899, color: '#ff8800', label: '1800 - 1899', name: 'Sovereign', desc: "You rule the grids! Your deep vocabulary and tactical dominance make you almost unbeatable. You set the standard for high-level play." },
    { min: 1900, max: 1999, color: '#ff6600', label: '1900 - 1999', name: 'Exalted', desc: "Held in the highest regard by the community! Your play is exceptionally creative, finding words in positions that seem impossible to others." },
    { min: 2000, max: 2099, color: '#ff4400', label: '2000 - 2099', name: 'Overlord', desc: "Unrivaled control and dominance! You dictate the pace of the game, leaving opponents in awe of your rapid word discovery and execution." },
    { min: 2100, max: 2199, color: '#ff2200', label: '2100 - 2199', name: 'Conqueror', desc: "You have conquered all standard challenges! Your strategic mastery is absolute, and you represent the pinnacle of mortal word-finding ability." },

    // Reds
    { min: 2200, max: 2299, color: '#ff0000', label: '2200 - 2299', name: 'Warlord', desc: "A fierce and relentless competitor! You battle through the toughest boards with unmatched intensity and drive, claiming victory after victory." },
    { min: 2300, max: 2399, color: '#e60000', label: '2300 - 2399', name: 'Juggernaut', desc: "An unstoppable force! No board layout or complex format can slow you down. You crush high-score records with ease." },
    { min: 2400, max: 2499, color: '#cc0000', label: '2400 - 2499', name: 'Apex', desc: "At the very top of the food chain! Your play is exceptionally sharp and deadly accurate. You stand as one of the ultimate players." },

    // --- THE VOID (2500 - 6000+) ---
    { min: 2500, max: 2599, color: '#b30000', label: '2500 - 2599', name: 'Harbinger', desc: "A sign of what's to come! Your play style is futuristic and incredibly advanced, hinting at a level of skill that transcends normal limits." },
    { min: 2600, max: 2699, color: '#990000', label: '2600 - 2699', name: 'Oracle', desc: "You see paths before they even register to others! Your foresight and pattern recognition are supernatural. A true visionary of the grid." },
    { min: 2700, max: 2799, color: '#800000', label: '2700 - 2799', name: 'Revenant', desc: "A relentless spirit that never yields! You make legendary comebacks and find spectacular high-value words under absolute pressure." },
    { min: 2800, max: 2899, color: '#660000', label: '2800 - 2899', name: 'Specter', desc: "A hauntingly fast presence! You sweep through the board invisibly and discover complex sequences before anyone else can react." },
    { min: 2900, max: 2999, color: '#4d0000', label: '2900 - 2999', name: 'Phantom', desc: "Elusive, fast, and incredibly precise! You maneuver through the grid like a shadow, leaving opponents wondering how you found those words." },

    { min: 3000, max: 3999, color: '#330000', label: '3000 - 3999', name: 'Ascendant', desc: "You have ascended past human limitations! Your play is transcendent, demonstrating a level of speed and vocabulary that is truly awe-inspiring." },
    { min: 4000, max: 4999, color: '#220000', label: '4000 - 4999', name: 'Transcendent', desc: "You exist in a state of pure grid enlightenment! Every move is optimal, and your high scores are legendary achievements in Morpheme history." },
    { min: 5000, max: 5999, color: '#110000', label: '5000 - 5999', name: 'Ethereal', desc: "A legend whispered in the lobby! Your skill is so rarefied and perfect that it seems almost mythical. You represent the peak of absolute dedication." },
    { min: 6000, max: 6999, color: '#000000', label: '6000 - 6999', name: 'ALIEN BEING', desc: "An otherworldly intelligence! Your word-finding speed and pattern execution defy all logic and human capability. Absolute perfection." },
    { min: 7000, max: 99999, color: '#a020f0', label: '7000+', name: 'SINGULARITY', desc: "You are the board, and the board is you! You have achieved infinite complexity, collapsing all word-finding possibilities into instantaneous mastery." }
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
    // Check if the user's last visit was more than 1 hour ago (3600s)
    try {
        const lastActiveTime = parseInt(localStorage.getItem('morpheme_last_active_time') || localStorage.getItem('morpheme_last_active_timestamp') || '0', 10);
        const nowTime = Date.now();
        const exceededOneHour = (lastActiveTime > 0) && ((nowTime - lastActiveTime) >= 60 * 60 * 1000);
        if (exceededOneHour) {
            console.log(`[app.js] Last visit was ${Math.round((nowTime - lastActiveTime) / 60000)} minutes ago (>= 1 hour). Silently clearing room session without Session Expired notice.`);
            window._suppressInactivityNotice = true;
            sessionStorage.setItem('morpheme_suppress_inactivity_notice', 'true');
            localStorage.removeItem('last_joined_room');
            if (window.currentRoomId) window.currentRoomId = null;
        }
    } catch(e) {}

    // 1. Core UI Setup (Always Run First for Responsiveness)
    setupNavigation();
    setupModalListeners();
    setupAuth(); // Initialize auth listeners
    setupContactForm(); // Initialize contact form listeners
    setupFirstInteractionMusic(); // Active immediately for early loading clicks
    if (window.loadFAQUserCounts) window.loadFAQUserCounts();
    if (window.loadFAQDictionaryStats) window.loadFAQDictionaryStats();
    
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

    // 2. Parallel Session & Single-Instance Validation (Non-blocking for instant Gateway render)
    const [isSingle] = await Promise.all([
        validateSingleInstance(),
        checkSession()
    ]);
    if (!isSingle) return;

    // 3. Application Domain Logic
    if (typeof handleLobbyMusicState === 'function') handleLobbyMusicState();
    if (typeof checkLobbyNotice === 'function') checkLobbyNotice();
    
    // Setup Global Profile listeners
    setupGlobalProfileLogic();
    
    // Initial State Check
    requestAnimationFrame(() => {
        if (typeof checkNavVisibility === 'function') checkNavVisibility();
        if (typeof checkForumActivity === 'function') checkForumActivity();
    });

    // Handle initial navigation based on hash OR landing page
    const hash = window.location.hash || '';
    const privateActive = localStorage.getItem('private_match_active');

    // Clean up any stale tournament_play_active left from a previous session
    const tournamentActive = localStorage.getItem('tournament_play_active');
    if (tournamentActive) {
        try {
            const tCheck = await fetch('/api/tournament/game-state', { cache: 'no-store' });
            const tData = await tCheck.json();
            if (tData.error) {
                console.log('[app.js] Clearing stale tournament_play_active:', tData.error);
                localStorage.removeItem('tournament_play_active');
            }
        } catch (e) {
            console.warn('[app.js] Could not verify tournament turn on startup:', e);
        }
    }

    if (currentUser) {
        // If the user already clicked ENTER LOBBY or entered a room while app.js was initializing, DO NOT reset them back to page-loading!
        const activePageEl = document.querySelector('.page.active');
        const activePageId = activePageEl ? activePageEl.id : '';
        const alreadyInGameOrLobby = window._gatewayPassed || window.currentRoomId || (activePageId && activePageId !== 'page-loading');

        if (privateActive || (hash === '#page-play' && window.currentRoomId) || activePageId === 'page-play') {
            showPage('page-play');
            const playBtn = document.querySelector('.nav-btn[data-page="play"]');
            if (playBtn) updateActiveNav(playBtn);
        } else if (alreadyInGameOrLobby) {
            showPage(activePageId || 'page-lobby');
            const navBtn = document.querySelector(`.nav-btn[data-page="${(activePageId || 'page-lobby').replace('page-', '')}"]`);
            if (navBtn) updateActiveNav(navBtn);
            handleLobbyMusicState();
        } else {
            const gatewayBtn = document.getElementById('btn-enter-lobby-gateway');
            const spinnerCont = document.getElementById('loading-spinner-container');
            const gatewayCont = document.getElementById('loading-gateway-container');

            if (gatewayBtn && gatewayCont) {
                // Ensure gateway container is active and visible
                showPage('page-loading');
                if (spinnerCont) spinnerCont.style.display = 'none';
                gatewayCont.style.display = 'flex';
                document.body.classList.remove('loading-active');
                handleLobbyMusicState();

                // Trigger playback on any pointer/touch/hover event on the gateway button or container for Safari/Firefox
                const audioTriggers = ['pointerenter', 'pointerdown', 'touchstart', 'mousedown', 'mouseover', 'focus', 'click'];
                audioTriggers.forEach(evt => {
                    gatewayBtn.addEventListener(evt, () => {
                        if (typeof window.playLobbyAudioImmediate === 'function') {
                            window.playLobbyAudioImmediate();
                        } else {
                            handleLobbyMusicState();
                        }
                    }, { passive: true });
                    gatewayCont.addEventListener(evt, () => {
                        if (typeof window.playLobbyAudioImmediate === 'function') {
                            window.playLobbyAudioImmediate();
                        } else {
                            handleLobbyMusicState();
                        }
                    }, { passive: true });
                });

                // Customize button text based on destination
                let targetPageId = 'page-lobby';
                let targetNavName = 'lobby';
                if (hash && hash.startsWith('#page-') && hash !== '#page-play' && hash !== '#page-login' && hash !== '#page-loading') {
                    targetPageId = hash.substring(1);
                    targetNavName = targetPageId.replace('page-', '');
                    gatewayBtn.textContent = 'ENTER ' + targetNavName.toUpperCase();
                } else {
                    gatewayBtn.textContent = 'ENTER LOBBY';
                }

                let gatewayClicked = false;
                const handleGatewayTransition = async (e) => {
                    if (gatewayClicked) return;
                    gatewayClicked = true;

                    console.log(`[Gateway] Transitioning via event: ${e ? e.type : 'manual'}`);

                    // 1. Play audio synchronously first to preserve user gesture context on Safari
                    try {
                        const lobbyMusic = document.getElementById('lobby-music');
                        if (lobbyMusic) {
                            playLobbyMusicHelper(lobbyMusic, removeInteractionListeners);
                        }
                    } catch (audioErr) {
                        console.error('[LobbyMusic] Exception during gateway play initialization:', audioErr);
                    }

                    // 2. Leave the room if we are not going to the play page
                    if (targetNavName !== 'play') {
                        if (window.leaveCurrentRoom && (window.currentRoomId || localStorage.getItem('last_joined_room'))) {
                            console.log('[Gateway] Leaving current room on gateway transition.');
                            try {
                                await window.leaveCurrentRoom();
                            } catch (err) {
                                console.error('[Gateway] Failed to leave room during gateway transition:', err);
                            }
                        }
                    }

                    // 3. Perform the page transition
                    try {
                        showPage(targetPageId);
                        const navBtn = document.querySelector(`.nav-btn[data-page="${targetNavName}"]`);
                        if (navBtn) updateActiveNav(navBtn);
                        handleLobbyMusicState();
                        if (hash === '#page-login') {
                            history.replaceState(null, null, '#page-lobby');
                        }
                    } catch (transitionErr) {
                        console.error('[Gateway] Exception performing page transition:', transitionErr);
                    }
                };

                // Robust interaction handlers: Outer socket click + Drag-out cancellation back to 3D
                const housingEl = document.getElementById('gateway-housing') || gatewayBtn.parentElement;
                let isPointerDown = false;
                let transitionTimeout = null;

                function getCoords(e) {
                    if (e.touches && e.touches.length > 0) {
                        return { x: e.touches[0].clientX, y: e.touches[0].clientY };
                    }
                    if (e.changedTouches && e.changedTouches.length > 0) {
                        return { x: e.changedTouches[0].clientX, y: e.changedTouches[0].clientY };
                    }
                    return { x: e.clientX, y: e.clientY };
                }

                function isInsideButton(e) {
                    const targetEl = housingEl || gatewayBtn;
                    const rect = targetEl.getBoundingClientRect();
                    const coords = getCoords(e);
                    return (
                        coords.x >= (rect.left - 4) &&
                        coords.x <= (rect.right + 4) &&
                        coords.y >= (rect.top - 4) &&
                        coords.y <= (rect.bottom + 4)
                    );
                }

                const handlePressStart = (e) => {
                    if (gatewayClicked) return;
                    isPointerDown = true;
                    gatewayBtn.classList.add('pressed', 'flattened');
                    try {
                        const lobbyMusic = document.getElementById('lobby-music');
                        if (lobbyMusic) {
                            playLobbyMusicHelper(lobbyMusic, removeInteractionListeners);
                        }
                    } catch (audioErr) {
                        console.error('[LobbyMusic] Synchronous play error:', audioErr);
                    }
                };

                const handlePressMove = (e) => {
                    if (!isPointerDown || gatewayClicked) return;
                    if (isInsideButton(e)) {
                        gatewayBtn.classList.add('pressed', 'flattened');
                    } else {
                        // User dragged across and out of the button: bring back to 3D standing position
                        gatewayBtn.classList.remove('pressed', 'flattened');
                        if (transitionTimeout) {
                            clearTimeout(transitionTimeout);
                            transitionTimeout = null;
                        }
                    }
                };

                const handlePressEnd = (e) => {
                    if (!isPointerDown || gatewayClicked) return;
                    isPointerDown = false;

                    if (isInsideButton(e)) {
                        // Released inside button or outer housing: keep flattened and trigger transition
                        gatewayBtn.classList.add('pressed', 'flattened');
                        if (transitionTimeout) return;
                        transitionTimeout = setTimeout(() => {
                            handleGatewayTransition(e);
                        }, 200);
                    } else {
                        // Released OUTSIDE: bring back to 3D standing position and do not enter Lobby
                        gatewayBtn.classList.remove('pressed', 'flattened');
                        if (transitionTimeout) {
                            clearTimeout(transitionTimeout);
                            transitionTimeout = null;
                        }
                    }
                };

                const handlePressCancel = () => {
                    if (gatewayClicked) return;
                    isPointerDown = false;
                    gatewayBtn.classList.remove('pressed', 'flattened');
                    if (transitionTimeout) {
                        clearTimeout(transitionTimeout);
                        transitionTimeout = null;
                    }
                };

                // Attach to button AND outer housing socket
                const interactiveElements = [gatewayBtn, housingEl].filter(Boolean);
                interactiveElements.forEach(el => {
                    el.addEventListener('pointerdown', handlePressStart);
                    el.addEventListener('mousedown', handlePressStart);
                    el.addEventListener('touchstart', handlePressStart, { passive: true });
                });

                // Global window drag and release tracking
                window.addEventListener('pointermove', handlePressMove, { passive: true });
                window.addEventListener('touchmove', handlePressMove, { passive: true });
                window.addEventListener('mousemove', handlePressMove, { passive: true });

                window.addEventListener('pointerup', handlePressEnd);
                window.addEventListener('touchend', handlePressEnd);
                window.addEventListener('mouseup', handlePressEnd);

                window.addEventListener('pointercancel', handlePressCancel);
                window.addEventListener('touchcancel', handlePressCancel);

                window.handleEnterLobbyClick = (btn, evt) => {
                    if (!gatewayClicked) {
                        gatewayBtn.classList.add('pressed', 'flattened');
                        if (!transitionTimeout) {
                            transitionTimeout = setTimeout(() => {
                                handleGatewayTransition(evt);
                            }, 200);
                        }
                    }
                };
            } else {
                // Fallback if elements not in DOM
                showPage('page-lobby');
                const lobbyBtn = document.querySelector('.nav-btn[data-page="lobby"]');
                if (lobbyBtn) updateActiveNav(lobbyBtn);
                handleLobbyMusicState();
                if (hash === '#page-login') {
                    history.replaceState(null, null, '#page-lobby');
                }
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
    setupFirstInteractionMusic(); // Set up gesture listeners immediately on page load!
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


// Helper to play lobby music safely across all platforms (desktops, laptops, tablets, mobile).
function playLobbyMusicHelper(lobbyMusic, onSuccess) {
    if (!lobbyMusic) return;
    lobbyMusic.loop = true;
    if (typeof lobbyMusic.volume === 'number' && lobbyMusic.volume === 1) {
        lobbyMusic.volume = 0.5;
    }

    // Unlock Web Audio Context for Safari/Firefox
    try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) {
            if (!window._morphemeAudioCtx) {
                window._morphemeAudioCtx = new AudioCtx();
            }
            if (window._morphemeAudioCtx.state === 'suspended') {
                window._morphemeAudioCtx.resume();
            }
        }
    } catch(e) {}

    try {
        lobbyMusic.muted = false;
    } catch(e) {}

    // If already playing smoothly and unmuted, continue playback without restarting!
    if (!lobbyMusic.paused && !lobbyMusic.muted) {
        console.log('[LobbyMusic] Already playing continuously at:', lobbyMusic.currentTime);
        if (onSuccess) onSuccess();
        return;
    }

    console.log('[LobbyMusic] Playing lobby music.');
    const playPromise = lobbyMusic.play();
    if (playPromise !== undefined) {
        playPromise
            .then(() => {
                console.log('[LobbyMusic] Play succeeded.');
                if (onSuccess) onSuccess();
            })
            .catch(err => {
                console.warn('[LobbyMusic] Play failed / trying muted buffer fallback:', err ? err.name : '');
                try {
                    lobbyMusic.muted = true;
                    lobbyMusic.play().catch(() => {});
                } catch(mErr) {}
                setupFirstInteractionMusic();
            });
    }
}

// Helper to start/stop music based on Page AND Setting
function handleLobbyMusicState() {
    console.log('[LobbyMusic] handleLobbyMusicState() triggered.');
    const lobbyMusic = document.getElementById('lobby-music');
    if (!lobbyMusic) {
        console.warn('[LobbyMusic] #lobby-music element not found in DOM.');
        return;
    }

    // Use window.currentPageId if available to avoid DOM ID race conditions during transition
    const activePage = window.currentPageId || (document.querySelector('.page.active')?.id);
    const onLobby = (activePage === 'page-lobby');
    const onLoading = (activePage === 'page-loading');
    const onPlay = (activePage === 'page-play');
    const inGameRoom = onPlay || (window.currentRoomId && activePage !== 'page-loading' && activePage !== 'page-lobby');
    const lobbyMusicSetting = (!window.userSettings || window.userSettings.lobby_music !== false);
    
    // STRICT REQUIREMENT: Only play the Lobby music if the user is on the ENTER LOBBY screen or in the Main Lobby!
    const shouldPlay = (onLobby || onLoading || !activePage) && !inGameRoom && lobbyMusicSetting;

    console.log('[LobbyMusic] State assessment:', {
        activePage,
        onLobby,
        onLoading,
        inGameRoom,
        lobbyMusicSetting,
        shouldPlay,
        paused: lobbyMusic.paused,
        currentTime: lobbyMusic.currentTime
    });

    if (shouldPlay) {
        if (lobbyMusic.paused) {
            console.log('[LobbyMusic] Attempting programmatic .play()...');
            playLobbyMusicHelper(lobbyMusic, null);
        }
    } else {
        console.log('[LobbyMusic] shouldPlay is false (not on ENTER LOBBY or Main Lobby), ensuring audio is paused.');
        if (!lobbyMusic.paused) {
            lobbyMusic.pause();
            console.log('[LobbyMusic] Paused active playback.');
        }
    }
}

// Modern Browser Autoplay bypass helpers
function playMusicOnFirstInteraction() {
    console.log('[LobbyMusic] playMusicOnFirstInteraction() triggered by gesture.');
    const activePage = window.currentPageId || (document.querySelector('.page.active')?.id);
    const onLobby = (activePage === 'page-lobby');
    const onLoading = (activePage === 'page-loading');
    const onPlay = (activePage === 'page-play');
    const inGameRoom = onPlay || (window.currentRoomId && activePage !== 'page-loading' && activePage !== 'page-lobby');
    const lobbyMusicSetting = (!window.userSettings || window.userSettings.lobby_music !== false);
    const shouldPlay = (onLobby || onLoading || !activePage) && !inGameRoom && lobbyMusicSetting;

    console.log('[LobbyMusic] Gesture state evaluation:', {
        onLobby,
        onLoading,
        inGameRoom,
        lobbyMusicSetting,
        shouldPlay
    });

    if (shouldPlay) {
        const lobbyMusic = document.getElementById('lobby-music');
        if (lobbyMusic) {
            console.log('[LobbyMusic] Attempting play() on gesture to unlock/unmute stream...');
            playLobbyMusicHelper(lobbyMusic, removeInteractionListeners);
        } else {
            console.warn('[LobbyMusic] #lobby-music element not found on gesture.');
        }
    } else {
        const lobbyMusic = document.getElementById('lobby-music');
        if (lobbyMusic && !lobbyMusic.paused) {
            lobbyMusic.pause();
        }
    }
}

function removeInteractionListeners() {
    const events = ['pointerdown', 'pointerup', 'touchstart', 'touchend', 'mousedown', 'mouseup', 'click', 'keydown', 'keyup', 'mousemove', 'pointermove'];
    events.forEach(evt => {
        window.removeEventListener(evt, playMusicOnFirstInteraction, { capture: true });
        document.removeEventListener(evt, playMusicOnFirstInteraction, { capture: true });
    });
}

function setupFirstInteractionMusic() {
    // Add event listeners without once: true so we don't prematurely delete them on early loading clicks!
    const events = ['pointerdown', 'pointerup', 'touchstart', 'touchend', 'mousedown', 'mouseup', 'click', 'keydown', 'keyup', 'mousemove', 'pointermove'];
    events.forEach(evt => {
        window.addEventListener(evt, playMusicOnFirstInteraction, { capture: true, passive: true });
        document.addEventListener(evt, playMusicOnFirstInteraction, { capture: true, passive: true });
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
        let response = await fetch('/api/session');
        let data = await response.json();

        // If server says authenticated, always trust it and clear any stale logged-out flag
        if (data.authenticated) {
            sessionStorage.removeItem('morpheme_logged_out');
            localStorage.removeItem('morpheme_logged_out');
        } else if (sessionStorage.getItem('morpheme_logged_out') === 'true' || localStorage.getItem('morpheme_logged_out') === 'true') {
            // Server says not authenticated AND user explicitly logged out — respect logout intent.
            // Still try auto-login via stored token as a last resort.
            const token = localStorage.getItem('morpheme_auth_token');
            if (token) {
                console.info('[Auth] Logged-out flag set, but attempting auto-login via stored token...');
                try {
                    const autoLoginRes = await fetch('/api/auth/auto-login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ auth_token: token })
                    });
                    const autoLoginData = await autoLoginRes.json();
                    if (autoLoginData.success) {
                        console.info('[Auth] Auto-login succeeded (overriding logged-out flag).');
                        sessionStorage.removeItem('morpheme_logged_out');
                        localStorage.removeItem('morpheme_logged_out');
                        data = {
                            authenticated: true,
                            username: autoLoginData.username,
                            email: autoLoginData.email,
                            rating: autoLoginData.rating,
                            is_guest: false,
                            is_mod: autoLoginData.is_mod
                        };
                    } else {
                        console.info('[Auth] User explicitly logged out, no valid token. Staying on login page.');
                        updateAuthUI();
                        return;
                    }
                } catch (e) {
                    console.error('[Auth] Auto-login error:', e);
                    updateAuthUI();
                    return;
                }
            } else {
                console.info('[Auth] User explicitly logged out, no token. Staying on login page.');
                updateAuthUI();
                return;
            }

        } else if (!data.authenticated) {
            // Not logged out intentionally, but no server session — try auto-login
            const token = localStorage.getItem('morpheme_auth_token');
            if (token) {
                console.info('[Auth] Session empty. Attempting auto-login via stored token...');
                try {
                    const autoLoginRes = await fetch('/api/auth/auto-login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ auth_token: token })
                    });
                    const autoLoginData = await autoLoginRes.json();
                    if (autoLoginData.success) {
                        console.info('[Auth] Auto-login succeeded.');
                        data = {
                            authenticated: true,
                            username: autoLoginData.username,
                            email: autoLoginData.email,
                            rating: autoLoginData.rating,
                            is_guest: false,
                            is_mod: autoLoginData.is_mod
                        };
                    } else {
                        console.warn('[Auth] Auto-login failed:', autoLoginData.error);
                        localStorage.removeItem('morpheme_auth_token');
                    }
                } catch (e) {
                    console.error('[Auth] Auto-login error:', e);
                }
            }
        }

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
            window.currentUserRating = data.rating;
            updateAuthUI(data.rating); // Update UI for logged in state

            // Check and sync timeout state immediately
            if (typeof window.syncLobbyTimeoutState === 'function') {
                window.syncLobbyTimeoutState().catch(function() {});
            }

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
            // Server says not authenticated and auto-login did not succeed.
            // Check if localStorage still marks the user as logged in — this can happen
            // transiently during a server restart when the new Flask process has no session
            // and the auto-login endpoint is still unavailable.
            const prevLoggedIn = localStorage.getItem('morpheme_logged_in') === 'true';
            const prevUsername = localStorage.getItem('morpheme_username');
            if (prevLoggedIn && prevUsername) {
                // Optimistically keep the user in their current page.
                // The next checkSession (or any API call) will catch a genuine logout.
                console.warn('[Auth] Server returned !authenticated but localStorage shows prior session. Keeping current UI (server may be restarting).');
                currentUser = prevUsername;
                window.currentUser = currentUser;
                updateAuthUI();
            } else {
                localStorage.removeItem('morpheme_logged_in');
                updateAuthUI();
            }
        }
    } catch (error) {
        // Network error or invalid JSON (e.g. server restarting and Nginx is serving the
        // splash page instead of JSON). This is NOT a logout — it is a transient server
        // unavailability. Do NOT clear the session or redirect to login; just keep the
        // current UI state so the user stays where they are until the server comes back.
        console.warn('[Auth] Session check failed (server may be restarting):', error.message || error);
        // Leave currentUser, localStorage.morpheme_logged_in, and the current page untouched.
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
window.checkTournamentTurn = checkTournamentTurn;

// Check initially (500ms) and periodically every 4s for instant turn green button flashing
setTimeout(() => {
    checkTournamentTurn();
    checkForumActivity();
}, 500);
setInterval(() => {
    checkTournamentTurn();
}, 4000);
setInterval(() => {
    checkForumActivity();
}, 60000);

window.addEventListener('focus', () => {
    checkTournamentTurn();
});
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        checkTournamentTurn();
    }
});

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
            // Use sessionStartTime as default so that ancient posts do not highlight for new sessions
            const lastView = Number(lastViewed[cat.id]) || window.sessionStartTime || Date.now();
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
    const soloFriends = document.getElementById('mobile-panel-solo');
    const lobbyGrid = document.querySelector('.lobby-grid');
    const gameTypesPanel = document.getElementById('mobile-panel-main');
    
    if (!soloFriends || !lobbyGrid || !gameTypesPanel) return;
    
    if (isMobile) {
        // MOBILE CAROUSEL: Solo must be a direct sibling of lobbyGrid, positioned to the LEFT of Game Types (Main)
        if (soloFriends.parentNode !== lobbyGrid) {
            console.log('[Layout] Moving Solo panel to the start of lobby-grid for mobile carousel.');
            lobbyGrid.insertBefore(soloFriends, gameTypesPanel);
        }
    } else {
        // DESKTOP GRID: Solo must be nested INSIDE Game Types so they share the same box/scroll region
        if (soloFriends.parentNode !== gameTypesPanel) {
            console.log('[Layout] Restoring Solo panel inside game-types-panel for desktop view.');
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
    // Single-panel stacked layout on mobile: swipe-and-snap gesture is not needed
    if (window.innerWidth <= 900) return;

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
                    if (window.loadFAQUserCounts) window.loadFAQUserCounts();
                    if (window.loadFAQDictionaryStats) window.loadFAQDictionaryStats();
                }
                return;
            }

            // NAVIGATION GUARD FOR ACTIVE MATCHES
            if (window.isTournamentPlay) {
                const confirmLeave = confirm("Leaving mid-round will end your tournament turn and record a score of 0. Are you sure?");
                if (!confirmLeave) return;
                try { await fetch('/api/tournament/forfeit', { method: 'POST' }); } catch (e) { }
                if (window.exitTournamentPlay) window.exitTournamentPlay(pageTarget);
                return;
            }
            if (window.isPrivateMatchPlay) {
                const confirmLeave = confirm("Leaving mid-round will end your turn and submit your current words. Are you sure?");
                if (!confirmLeave) return;
                if (window.finishPrivateMatchTurn) {
                    await window.finishPrivateMatchTurn(pageTarget);
                }
                return;
            }

            // USER NAVIGATION LEAVE HARNESS: If navigating to lobby from another page, leave the current room
            if (pageTarget === 'lobby') {
                if (window.leaveCurrentRoom && (window.currentRoomId || localStorage.getItem('last_joined_room'))) {
                    console.log('[setupNavigation] Leaving current room on Lobby navigation click.');
                    await window.leaveCurrentRoom();
                }
            }

            if (pageTarget === 'play') {
                if (window._userIsTimedOut || (window._userTimeoutInfo && window._userTimeoutInfo.timed_out)) {
                    if (typeof window.showTimeoutBanModal === 'function') {
                        window.showTimeoutBanModal(window._userTimeoutInfo);
                    } else if (typeof window.checkAccountTimeoutAndAlert === 'function') {
                        window.checkAccountTimeoutAndAlert();
                    }
                    return;
                }
                if (typeof window.checkAccountTimeoutAndAlert === 'function' && await window.checkAccountTimeoutAndAlert()) {
                    return;
                }
            }

            // Default Page Navigation
            const pageId = 'page-' + pageTarget;
            showPage(pageId);
            updateActiveNav(btn);

            if (pageTarget === 'profile' && window.currentUser) {
                if (typeof window.performProfileSearch === 'function') {
                    window.performProfileSearch(window.currentUser);
                }
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

    // FAQ Selector List Navigation
    const faqLinks = document.querySelectorAll('.faq-nav-link');
    faqLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const targetId = link.getAttribute('data-target');
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                // Clear any previous highlights
                document.querySelectorAll('.faq-item.highlight-pulse').forEach(item => {
                    item.classList.remove('highlight-pulse');
                });
                
                // Scroll the target element into view within the modal content
                targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
                
                // Trigger reflow to restart animation, then add the pulsing glow class
                void targetEl.offsetWidth;
                targetEl.classList.add('highlight-pulse');
            }
        });
    });

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

    // Spinner Set modal trigger (Desktop header title, Mobile spinner label, and Parameter text)
    const gameParams = document.querySelector('.game-params');
    const headerTitleGroup = document.querySelector('.header-title-group');
    const spinnerLabel = document.querySelector('.spinner-set-label');
    const spinnerModal = document.getElementById('spinner-set-modal');

    if (spinnerModal) {
        const openSpinnerModal = () => {
            spinnerModal.classList.remove('hidden');
        };

        if (gameParams) {
            gameParams.addEventListener('click', openSpinnerModal);
        }
        if (headerTitleGroup) {
            headerTitleGroup.addEventListener('click', openSpinnerModal);
        }
        if (spinnerLabel) {
            spinnerLabel.addEventListener('click', openSpinnerModal);
        }

        // Escape key close support
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !spinnerModal.classList.contains('hidden')) {
                spinnerModal.classList.add('hidden');
            }
        });
    }
}

function showPage(pageId) {
    if (pageId === 'page-mods' && !window.currentUserIsMod) {
        console.warn('[Navigation] Unauthorized access to mods page. Redirecting to lobby.');
        pageId = 'page-lobby';
    }
    if (pageId === 'page-play') {
        if (window._userIsTimedOut || (window._userTimeoutInfo && window._userTimeoutInfo.timed_out)) {
            console.warn('[Navigation] User is timed out. Preventing navigation to page-play.');
            pageId = 'page-lobby';
            if (typeof window.showTimeoutBanModal === 'function') {
                window.showTimeoutBanModal(window._userTimeoutInfo);
            } else if (typeof window.checkAccountTimeoutAndAlert === 'function') {
                window.checkAccountTimeoutAndAlert();
            }
        } else if (!window.currentRoomId && !localStorage.getItem('last_joined_room') && !window._isEnteringRoom
                   && !localStorage.getItem('tournament_play_active') && !localStorage.getItem('private_match_active')) {
            console.warn('[Navigation] No active room found. Redirecting to lobby.');
            pageId = 'page-lobby';
        }
    }
    window.currentPageId = pageId;
    // Intercept leaving tournament play mid-round
    if (pageId !== 'page-play' && window.isTournamentPlay && localStorage.getItem('tournament_play_active')) {
        if (typeof window.finishTournamentTurn === 'function') {
            const targetPageName = pageId.replace('page-', '');
            console.log('[Navigation] Finalizing tournament turn because user navigated away mid-round to:', targetPageName);
            window.isTournamentPlay = false; // Prevent recursion
            window.finishTournamentTurn(targetPageName);
            return;
        }
    }

    if (pageId !== 'page-play' && window.hideLoadingOverlay) {
        window.hideLoadingOverlay();
    }
    // 0. Synchronize URL Hash (for Reload/Navigation consistency)
    if (window.location.hash !== "#" + pageId) {
        history.replaceState(null, null, "#" + pageId);
    }

    // Auto-hide modals/overlays when navigating pages (except active priority alert modals)
    const overlays = document.querySelectorAll('.modal-window, .mini-profile-overlay, .review-overlay, .overlay');
    overlays.forEach(o => {
        if (o.id === 'generic-info-modal' && window._hasPriorityModal) return;
        o.classList.remove('forced-show');
        o.classList.add('hidden');
    });

    // 1. Update Page Visibility
    document.querySelectorAll('.page').forEach(page => {
        const isMatch = (page.id === pageId) || (page.dataset && page.dataset.pageId === pageId.replace('page-', ''));
        if (isMatch) {
            page.classList.add('active');
            page.style.display = 'block';
            page.style.opacity = '1';
            page.style.visibility = 'visible';
            page.scrollTop = 0;
            const layout = page.querySelector('.tools-split-layout');
            if (layout) {
                layout.scrollLeft = 0;
            }
        } else {
            page.classList.remove('active');
            page.style.display = 'none';
        }
    });
    if (pageId !== 'page-loading') {
        document.body.classList.remove('loading-active');
    }
    window.scrollTo(0, 0);
    if (typeof handleLobbyMusicState === 'function') {
        handleLobbyMusicState();
    }

    // Standardize: Rating color bar ONLY appears on the Play page
    const colorBar = document.getElementById('game-color-bar');
    if (colorBar) {
        if (pageId === 'page-play') {
            colorBar.style.display = 'flex';
            setTimeout(() => {
                if (typeof adjustPlayHeaderForDevice === 'function') {
                    adjustPlayHeaderForDevice();
                }
            }, 50);
        } else {
            colorBar.style.display = 'none';
        }
    }

    // NEW: Load Private Matches & snap to main Lobby window on mobile when entering Lobby
    if (pageId === 'page-lobby') {
        const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
        if (isMobile) {
            const scrollToMain = () => {
                const mainPanel = document.getElementById('mobile-panel-main');
                if (mainPanel) {
                    const lobbyGrid = mainPanel.closest('.lobby-grid') || mainPanel.parentElement;
                    if (lobbyGrid) {
                        lobbyGrid.scrollLeft = mainPanel.offsetLeft;
                    }
                }
            };
            scrollToMain();
            requestAnimationFrame(scrollToMain);
            setTimeout(scrollToMain, 50);
            setTimeout(scrollToMain, 150);
        }
        if (typeof window.resetLobbyButtons === 'function') {
            window.resetLobbyButtons();
        }
        if (typeof window.loadPrivateMatches === 'function') {
            window.loadPrivateMatches();
        }
        if (typeof window.checkLobbyNotice === 'function' && !window._lobbyNoticeShownThisSession) {
            window.checkLobbyNotice();
        }
        if (typeof window.loadCurrentUserConfigRatings === 'function') {
            window.loadCurrentUserConfigRatings();
        }
        if (typeof window.fetchLobbyStats === 'function') {
            window.fetchLobbyStats('all');
        }
        if (typeof window.startStatsPolling === 'function') {
            window.startStatsPolling();
        }
    }

    if (pageId === 'page-mods') {
        if (typeof window.loadAddedWordsConfig === 'function') {
            window.loadAddedWordsConfig();
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

        // Trigger dynamic Spinner Set layout adjustment once the page is fully active/visible
        if (typeof window.adjustSpinnerSetFontSize === 'function') {
            setTimeout(window.adjustSpinnerSetFontSize, 100);
        }

        // Auto-focus the input field (Desktop only to prevent mobile carousel snap-back)
        setTimeout(() => {
            const input = document.getElementById('word-input');
            const isMobile = window.innerWidth <= 992;
            if (input && !input.disabled && !isMobile) {
                input.focus();
    if (window.innerWidth <= 768 && !window.hasCenteredBoard) {
        window.hasCenteredBoard = true;
        setTimeout(() => {
            const board = document.getElementById('play-panel-board');
            if (board) board.scrollIntoView({ behavior: 'auto', block: 'nearest', inline: 'start' });
        }, 500);
    }
    
            }

            // Mobile carousel: Ensure board is centered by default
            if (window.innerWidth <= 992) {
                const boardPanel = document.querySelector('.board-panel');
                if (boardPanel) {
                    console.log('[app.js] Centering board panel in mobile carousel.');
                    boardPanel.scrollIntoView({ behavior: 'auto', block: 'nearest', inline: 'center' });
                }
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
    } else if (pageId === 'page-lobby') {
        // Handled in top branch
    } else {
        if (window.stopGamePolling) {
            window.stopGamePolling();
        }
    }

    if (pageId !== 'page-lobby') {
        if (typeof window.stopStatsPolling === 'function') {
            window.stopStatsPolling();
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
    const verifyForm = document.getElementById('verify-form');
    if (verifyForm) verifyForm.style.display = 'none';
    const tabsContainer = document.querySelector('.auth-tabs');
    if (tabsContainer) tabsContainer.style.display = 'flex';

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelectorAll('.auth-form').forEach(form => {
        form.classList.remove('active');
        form.style.display = ''; // restore default display
    });

    if (tab === 'signin') {
        document.querySelector('[data-tab="signin"]').classList.add('active');
        document.getElementById('signin-form').classList.add('active');
    } else {
        document.querySelector('[data-tab="signup"]').classList.add('active');
        document.getElementById('signup-form').classList.add('active');
        populateSignupFlagDropdown();
    }

    // Clear errors & states
    document.getElementById('signin-error').textContent = '';
    document.getElementById('signup-error').textContent = '';
    
    const signupStatus = document.getElementById('signup-email-status');
    if (signupStatus) signupStatus.textContent = '';
    
    const signupVerifyBox = document.getElementById('signup-verification-box');
    if (signupVerifyBox) signupVerifyBox.style.display = 'none';
    
    const signupSubmitBtn = document.getElementById('signup-submit-btn');
    if (signupSubmitBtn) {
        signupSubmitBtn.disabled = true;
        signupSubmitBtn.style.opacity = '0.6';
    }
    
    const sendEmailBtn = document.getElementById('signup-send-email-btn');
    if (sendEmailBtn) {
        sendEmailBtn.textContent = 'Send Email';
    }
}

async function handleSignIn() {
    const username = document.getElementById('signin-username').value;
    const password = document.getElementById('signin-password').value;
    const captcha = document.getElementById('signin-captcha').value;
    const captcha_id = window._signinCaptchaId || '';
    const errorEl = document.getElementById('signin-error');

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password, captcha, captcha_id })
        });

        const responseText = await response.text();
        let data;
        try {
            data = JSON.parse(responseText);
        } catch (e) {
            console.error('Failed to parse JSON response:', responseText);
            errorEl.textContent = 'Server returned an invalid response. Please try again.';
            if (typeof window.refreshCaptchas === 'function') window.refreshCaptchas();
            return;
        }

        if (data.success) {
            sessionStorage.removeItem('morpheme_logged_out');
            localStorage.removeItem('morpheme_logged_out');

            localStorage.setItem('morpheme_logged_in', 'true');
            if (data.auth_token) {
                localStorage.setItem('morpheme_auth_token', data.auth_token);
            }
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
            errorEl.textContent = data.error || data.message || 'Failed to login.';
            if (typeof window.refreshCaptchas === 'function') window.refreshCaptchas();
        }
    } catch (error) {
        errorEl.textContent = 'An error occurred. Please try again.';
        console.error('Login error:', error);
        if (typeof window.refreshCaptchas === 'function') window.refreshCaptchas();
    }
}

async function handleSignUp() {
    const username = document.getElementById('signup-username').value.trim();
    const email = document.getElementById('signup-email').value.trim();
    const password = document.getElementById('signup-password').value;
    const confirmPassword = document.getElementById('signup-password-confirm').value;
    const code = document.getElementById('signup-verification-code').value.trim();
    const captcha = document.getElementById('signup-captcha').value.trim();
    const captcha_id = window._signupCaptchaId || '';
    const flag = document.getElementById('signup-flag').value;
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

    if (!flag) {
        errorEl.textContent = 'Please select a flag representing where you live';
        return;
    }

    if (!code || code.length !== 6) {
        errorEl.textContent = 'Please request and enter your 6-digit verification code';
        return;
    }

    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password, email, captcha, captcha_id, code, flag })
        });

        const responseText = await response.text();
        let data;
        try {
            data = JSON.parse(responseText);
        } catch (e) {
            console.error('Failed to parse JSON response:', responseText);
            errorEl.textContent = 'Server returned an invalid response. Please try again.';
            return;
        }

        if (data.success) {
            localStorage.setItem('morpheme_logged_in', 'true');
            if (data.auth_token) {
                localStorage.setItem('morpheme_auth_token', data.auth_token);
            }
            currentUser = data.username;
            window.currentUser = currentUser;
            currentUserEmail = email; // From the signup form
            window.currentUserEmail = currentUserEmail;
            window.currentUserIsGuest = false;
            window.lastPlayerRating = data.rating;
            navigateToLobby(data.rating);
        } else {
            errorEl.textContent = data.error || data.message || 'Failed to register.';
            if (typeof window.refreshCaptchas === 'function') window.refreshCaptchas();
        }
    } catch (error) {
        errorEl.textContent = 'An error occurred. Please try again.';
        console.error('Registration error:', error);
        if (typeof window.refreshCaptchas === 'function') window.refreshCaptchas();
    }
}

async function handleGuestLogin() {
    const captchaInput = document.getElementById('signin-captcha');
    const errorEl = document.getElementById('signin-error');
    const captcha = captchaInput ? captchaInput.value.trim() : '';
    const captcha_id = window._signinCaptchaId || '';
    
    if (!captcha) {
        if (errorEl) errorEl.textContent = 'Please complete the CAPTCHA first to play as a guest.';
        if (captchaInput) captchaInput.focus();
        return;
    }

    try {
        const response = await fetch('/api/guest-login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ captcha, captcha_id })
        });

        const data = await response.json();

        if (data.success) {
            sessionStorage.removeItem('morpheme_logged_out');
            localStorage.removeItem('morpheme_logged_out');

            localStorage.setItem('morpheme_logged_in', 'true');
            currentUser = data.username;
            window.currentUser = currentUser;
            window.currentUserIsGuest = true;
            navigateToLobby();
        } else {
            if (errorEl) errorEl.textContent = data.error || data.message || 'Failed to login as guest.';
            if (typeof window.refreshCaptchas === 'function') window.refreshCaptchas();
        }
    } catch (error) {
        if (errorEl) errorEl.textContent = 'An error occurred. Please try again.';
        console.error('Guest login error:', error);
        if (typeof window.refreshCaptchas === 'function') window.refreshCaptchas();
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
                    showPage('page-profile');
                    window.performProfileSearch(currentUser);
                }
            };
        }

        // Handle Rating Bar
        renderGameColorBar();
        if (rating || window.lastPlayerRating) {
            updateUserRatingHighlight(rating || window.lastPlayerRating);
        }

        // Load config-specific ratings
        if (window.loadCurrentUserConfigRatings) {
            window.loadCurrentUserConfigRatings();
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
        
        // Use fetch and await to ensure session is cleared before we reload the page!
        await fetch('/api/logout', { method: 'POST' });
        
        // Preserve global "read" states (Notices, Forum markers) across login sessions
        const noticeId = localStorage.getItem('morpheme_read_notice_id');
        const forumViewed = localStorage.getItem('forum_last_viewed');
        const userSettings = localStorage.getItem('morpheme_user_settings');
        
        console.info(`[Auth] Preservation: noticeId=${noticeId}, forumViewed=${forumViewed}`);

        // Clear only session-specific or sensitive data
        localStorage.clear();
        sessionStorage.clear();
        window.currentUserConfigRatings = {};
        window.currentUserIsMod = false;
        window.currentUserIsRootMod = false;
        window.currentUser = null;
        currentUser = null;
        const modsBtn = document.getElementById('nav-mods-btn');
        if (modsBtn) modsBtn.style.display = 'none';
        document.querySelectorAll('.mod-only-btn').forEach(btn => btn.style.display = 'none');
        
        // Set logged out flag to prevent auto-login on mobile — use sessionStorage so it only
        // applies to this tab/session and never bleeds into a future visit to morpheme.games
        sessionStorage.setItem('morpheme_logged_out', 'true');
        localStorage.removeItem('morpheme_logged_out'); // Clear any legacy localStorage copy
        
        // Restore non-sensitive global markers
        if (noticeId) localStorage.setItem('morpheme_read_notice_id', noticeId);
        if (forumViewed) localStorage.setItem('forum_last_viewed', forumViewed);
        if (userSettings) localStorage.setItem('morpheme_user_settings', userSettings);
        
        console.info('[Auth] Markers restored. Redirecting in 500ms...');
        setTimeout(() => {
            window.location.href = '/';
        }, 500);
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
            if (segment) {
                const tierName = segment.getAttribute('data-name') || 'Rating Tier';
                const tierLabel = segment.getAttribute('data-label') || '';
                const tierDesc = segment.getAttribute('data-desc') || '';
                const bgColor = segment.style.backgroundColor || '#fff';
                
                const modal = document.getElementById('color-tier-modal');
                if (modal) {
                    document.getElementById('color-tier-title').textContent = tierName;
                    document.getElementById('color-tier-title').style.color = bgColor;
                    document.getElementById('color-tier-icon').style.color = bgColor;
                    document.getElementById('color-tier-range').textContent = `Rating Range: ${tierLabel}`;
                    document.getElementById('color-tier-desc').textContent = tierDesc;
                    document.getElementById('color-tier-swatch').style.backgroundColor = bgColor;
                    modal.classList.remove('hidden');
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

function populateSignupFlagDropdown() {
    const select = document.getElementById('signup-flag');
    if (!select) return;
    if (select.children.length > 1) return; // already populated

    const list = window.ALL_FLAGS;
    if (!list) {
        console.warn("ALL_FLAGS is not defined yet.");
        return;
    }

    list.forEach(item => {
        const option = document.createElement('option');
        option.value = item.flag;
        option.textContent = `${item.name} ${item.flag}`;
        select.appendChild(option);
    });
}

// Setup authentication
function setupAuth() {
    // CAPTCHA helper logic
    window.refreshCaptchas = function() {
        const generateCaptchaId = () => 'cap_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
        
        const signinForm = document.getElementById('signin-form');
        if (signinForm) {
            const signinImg = signinForm.querySelector('.captcha-img');
            const signinId = generateCaptchaId();
            window._signinCaptchaId = signinId;
            if (signinImg) {
                signinImg.src = '/api/captcha?id=' + encodeURIComponent(signinId) + '&t=' + Date.now();
            }
            const signinCaptcha = document.getElementById('signin-captcha');
            if (signinCaptcha) signinCaptcha.value = '';
        }

        const signupForm = document.getElementById('signup-form');
        if (signupForm) {
            const signupImg = signupForm.querySelector('.captcha-img');
            const signupId = generateCaptchaId();
            window._signupCaptchaId = signupId;
            if (signupImg) {
                signupImg.src = '/api/captcha?id=' + encodeURIComponent(signupId) + '&t=' + Date.now();
            }
            const signupInput = document.getElementById('signup-captcha');
            if (signupInput) signupInput.value = '';
        }
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

    // Populate registration flag dropdown dynamically if active
    const isSignupActive = document.getElementById('signup-form') && document.getElementById('signup-form').classList.contains('active');
    if (isSignupActive) {
        populateSignupFlagDropdown();
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

    // Signup Send Email button
    const sendEmailBtn = document.getElementById('signup-send-email-btn');
    if (sendEmailBtn) {
        sendEmailBtn.addEventListener('click', async () => {
            const username = document.getElementById('signup-username').value.trim();
            const email = document.getElementById('signup-email').value.trim();
            const errorEl = document.getElementById('signup-error');
            const statusEl = document.getElementById('signup-email-status');
            const verificationBox = document.getElementById('signup-verification-box');
            const signupSubmitBtn = document.getElementById('signup-submit-btn');

            errorEl.textContent = '';
            statusEl.textContent = '';
            statusEl.style.color = '#c5c6c7';

            if (!username) {
                errorEl.textContent = 'Please enter a username first';
                return;
            }
            if (!email) {
                errorEl.textContent = 'Please enter an email first';
                return;
            }

            sendEmailBtn.disabled = true;
            sendEmailBtn.textContent = 'Sending...';

            try {
                const response = await fetch('/api/send-signup-verification', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, email })
                });

                const data = await response.json();

                if (data.success) {
                    statusEl.textContent = 'Verification code sent to your email! Please check your Junk email in 1 or 2 minutes if you do not see it.';
                    statusEl.style.color = '#00ff66';
                    
                    // Reveal the 6-digit box
                    if (verificationBox) {
                        verificationBox.style.display = 'flex';
                    }
                    
                    // Enable the main signup submit button
                    if (signupSubmitBtn) {
                        signupSubmitBtn.disabled = false;
                        signupSubmitBtn.style.opacity = '1';
                    }

                    // Change button to Resend Email
                    sendEmailBtn.textContent = 'Resend Email';
                    
                    // Focus on the code input
                    const codeInput = document.getElementById('signup-verification-code');
                    if (codeInput) {
                        codeInput.value = '';
                        codeInput.focus();
                    }
                } else {
                    errorEl.textContent = data.error || 'Failed to send verification code.';
                    sendEmailBtn.textContent = 'Send Email';
                }
            } catch (err) {
                errorEl.textContent = 'An error occurred. Please try again.';
                console.error(err);
                sendEmailBtn.textContent = 'Send Email';
            } finally {
                sendEmailBtn.disabled = false;
            }
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
    const okBtn = document.getElementById('generic-modal-ok-btn');
    const closeBtn = document.getElementById('close-generic-modal');
    
    // If a priority modal is already showing (e.g. Inactivity Kick), 
    // don't let a normal notice (e.g. Lobby Notice) overwrite it immediately.
    if (window._hasPriorityModal && !priority) {
        console.log('[Modal] Normal modal suppressed by priority modal.');
        return;
    }

    if (modal && titleEl && bodyEl) {
        if (priority) window._hasPriorityModal = true;
        titleEl.textContent = title || 'Notice';
        bodyEl.innerHTML = `<div style="text-align: center; font-size: 1.05rem; line-height: 1.6; color: var(--text-primary, #ffffff);">${message}</div>`;
        modal.classList.remove('hidden');
        modal.style.display = 'flex';
        modal.style.zIndex = '100001';
        
        const closeModal = (e) => {
            if (e) {
                try { e.preventDefault(); e.stopPropagation(); } catch(err) {}
            }
            modal.classList.add('hidden');
            modal.style.display = 'none';
            window._hasPriorityModal = false;
        };

        if (okBtn) okBtn.onclick = closeModal;
        if (closeBtn) closeBtn.onclick = closeModal;
        modal.onclick = (e) => {
            if (e.target === modal) closeModal(e);
        };
    } else if (window._nativeAlert) {
        window._nativeAlert(message);
    } else {
        alert(message);
    }
};

let _timeoutCountdownInterval = null;

// Global showTimeoutBanModal definition with live real-time dynamic countdown
window.showTimeoutBanModal = function(toData) {
    const data = toData || window._userTimeoutInfo || {};
    const rText = data.reason || data.timeout_reason || 'Moderator timeout';
    const serverRemSec = typeof data.remaining_seconds === 'number' ? data.remaining_seconds : 0;
    const fetchTime = Date.now();

    function getLiveRemainingText() {
        if (!serverRemSec || serverRemSec <= 0) {
            return data.remaining || 'a temporary timeout';
        }
        const elapsedSec = Math.floor((Date.now() - fetchTime) / 1000);
        const currentRemSec = Math.max(0, serverRemSec - elapsedSec);
        if (currentRemSec <= 0) {
            return 'Expired (Lifting...)';
        }
        const totalMins = Math.ceil(currentRemSec / 60);
        if (totalMins >= 60) {
            const h = Math.floor(totalMins / 60);
            const m = totalMins % 60;
            const hStr = h === 1 ? '1 hour' : `${h} hours`;
            if (m === 0) return hStr;
            const mStr = m === 1 ? '1 minute' : `${m} minutes`;
            return `${hStr} ${mStr}`;
        }
        return totalMins === 1 ? '1 minute' : `${totalMins} minutes`;
    }

    const dText = getLiveRemainingText();
    const msg = `You are currently placed on a temporary timeout from all game rooms.<br><br><strong>Reason:</strong> <span style="color: var(--text-primary); font-weight: 600;">${rText}</span><br><br><strong>Time Remaining:</strong> <span id="active-timeout-countdown-display" style="color: #f59e0b; font-size: 1.15rem; font-weight: 700;">${dText}</span><br><br>To keep matches fair and respectful for all players, room access is temporarily restricted during a timeout period.<br><br>Please wait until your timeout expires before joining another match!`;
    
    if (typeof window.showAlertModal === 'function') {
        window.showAlertModal('Account Timed Out', msg, true);
    } else {
        alert(`Account Timed Out\n\nReason: ${rText}\nTime Remaining: ${dText}`);
    }

    if (_timeoutCountdownInterval) {
        clearInterval(_timeoutCountdownInterval);
        _timeoutCountdownInterval = null;
    }

    _timeoutCountdownInterval = setInterval(async () => {
        const countDisplay = document.getElementById('active-timeout-countdown-display');
        const liveText = getLiveRemainingText();
        if (countDisplay) {
            countDisplay.textContent = liveText;
        }
        if (liveText.startsWith('Expired')) {
            clearInterval(_timeoutCountdownInterval);
            _timeoutCountdownInterval = null;
            if (typeof window.syncLobbyTimeoutState === 'function') {
                await window.syncLobbyTimeoutState();
            }
        }
    }, 1000);
};

window.checkAccountTimeoutAndAlert = async function() {
    try {
        const toResp = await fetch('/api/user/my_timeout_status?_t=' + Date.now(), { cache: 'no-store' });
        const toData = await toResp.json();
        const isTimedOut = !!(toData && toData.timed_out);
        window._userTimeoutInfo = isTimedOut ? toData : null;
        window._userIsTimedOut = isTimedOut;
        
        const lobbyButtons = document.querySelectorAll('.game-btn, .confirm-create-room-btn, .join-room-btn, .nav-btn[data-page="play"]');
        lobbyButtons.forEach(btn => {
            if (isTimedOut) {
                btn.classList.add('timeout-locked');
                btn.title = "Account Timed Out: Click for details";
            } else {
                btn.classList.remove('timeout-locked');
                if (btn.title === "Account Timed Out: Click for details") btn.title = "";
            }
        });

        if (isTimedOut) {
            if (typeof window.showTimeoutBanModal === 'function') {
                window.showTimeoutBanModal(toData);
            }
            return true;
        }
    } catch(e) {}
    return false;
};

// Capture-phase global click interceptor for all game entry buttons while on active timeout
document.addEventListener('click', async (e) => {
    const rawTarget = e.target;
    if (!rawTarget) return;
    const target = rawTarget.nodeType === 3 ? rawTarget.parentElement : rawTarget;
    if (!target || typeof target.closest !== 'function') return;

    const lockedBtn = target.closest('.game-btn, .confirm-create-room-btn, .join-room-btn, .nav-btn[data-page="play"]');
    if (!lockedBtn) return;

    if (window._userIsTimedOut || (window._userTimeoutInfo && window._userTimeoutInfo.timed_out)) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        
        // Re-verify immediately with server on click so expiration is recognized instantaneously without app reload
        const isStillTimedOut = await window.checkAccountTimeoutAndAlert();
        if (!isStillTimedOut) {
            // Timeout expired! Simulate click to trigger original button action smoothly
            setTimeout(() => {
                lockedBtn.click();
            }, 50);
        }
        return false;
    }
}, true);

// Global intercept for native alerts so all popup messages use the styled modal layout
if (!window._nativeAlert) {
    window._nativeAlert = window.alert;
}
window.alert = function (message) {
    if (typeof window.showAlertModal === 'function') {
        window.showAlertModal('Notice', message);
    } else if (window._nativeAlert) {
        window._nativeAlert(message);
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
        const rangeText = `${range.min}-${range.max === 99999 ? '∞' : range.max}`;
        const escapedDesc = (range.desc || '').replace(/"/g, '&quot;');
        html += `<div class="color-bar-segment" 
                      style="background-color: ${range.color};" 
                      data-name="${range.name.toUpperCase()}" 
                      data-label="${rangeText}"
                      data-desc="${escapedDesc}"
                      title="${range.name.toUpperCase()} (${rangeText})">
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

// === Mobile Viewport & Fullscreen Banner Recovery Engine ===
// Fixes the "half window / split screen frozen midway" bug on iOS/Android PWA and mobile browsers.
// When minimizing/returning to Morpheme or when the Android/iOS system banner
// ("To exit full screen, drag from the top...") appears/disappears, the viewport dimensions
// shift and can leave horizontal panels misaligned.
// This engine safely aligns panels when the app resumes, without interfering with active swipe gestures.

let _isUserTouching = false;
window.addEventListener('touchstart', () => { _isUserTouching = true; }, { passive: true, capture: true });
window.addEventListener('touchend', () => { setTimeout(() => { _isUserTouching = false; }, 300); }, { passive: true, capture: true });
window.addEventListener('touchcancel', () => { setTimeout(() => { _isUserTouching = false; }, 300); }, { passive: true, capture: true });

function _isKeyboardOpen() {
    const el = document.activeElement;
    return !!(el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT'));
}

function _updateVhVariable() {
    // Skip updating --vh if the soft keyboard is open, preventing DOM-wide style invalidation & blackouts
    if (_isKeyboardOpen()) return;
    document.documentElement.style.setProperty('--vh', (window.innerHeight * 0.01) + 'px');
}

function _restoreAllMobilePanels() {
    // If the user is currently touching, swiping, or typing in an input, never override scroll or styles!
    if (_isUserTouching || _isKeyboardOpen()) return;

    const isMobile = window.innerWidth <= 900 || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    if (!isMobile) return;
    
    _updateVhVariable();

    // 1. Play Page (.play-grid)
    const playGrid = document.querySelector('.play-grid');
    if (playGrid && typeof window._restorePlayPanel === 'function') {
        window._restorePlayPanel();
    }

    // 2. Tools / Settings / Mods (.tools-split-layout)
    document.querySelectorAll('.tools-split-layout').forEach(layoutEl => {
        const activePane = layoutEl.querySelector('.tools-content .tool-pane.active, .mod-details.active');
        const targetLeft = activePane ? (layoutEl.clientWidth || layoutEl.scrollWidth) : 0;
        layoutEl.scrollLeft = targetLeft;
    });

    // 3. Forum Page (.forum-container)
    const forumContainer = document.querySelector('.forum-container');
    if (forumContainer) {
        const activeThread = document.querySelector('.thread-view:not(.hidden), .create-thread-view:not(.hidden)');
        if (activeThread) {
            const mainContent = forumContainer.querySelector('.forum-main');
            if (mainContent) forumContainer.scrollLeft = mainContent.offsetLeft;
        }
    }
}

let _viewportRecoveryTimers = [];
window.scheduleMobileViewportRecovery = function() {
    if (_isKeyboardOpen()) return;
    _viewportRecoveryTimers.forEach(clearTimeout);
    _viewportRecoveryTimers = [];

    _restoreAllMobilePanels();
    _viewportRecoveryTimers.push(setTimeout(_restoreAllMobilePanels, 100));
    _viewportRecoveryTimers.push(setTimeout(_restoreAllMobilePanels, 350));
};

// Initialize immediately and bind across system lifecycle events
_updateVhVariable();
window.addEventListener('resize', () => {
    if (_isKeyboardOpen()) return;
    _updateVhVariable();
    if (!_isUserTouching) {
        scheduleMobileViewportRecovery();
    }
}, { passive: true });
window.addEventListener('orientationchange', () => {
    _updateVhVariable();
    scheduleMobileViewportRecovery();
}, { passive: true });
window.addEventListener('focus', (e) => {
    if (_isKeyboardOpen() || (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT'))) {
        return;
    }
    scheduleMobileViewportRecovery();
}, { passive: true });
window.addEventListener('pageshow', scheduleMobileViewportRecovery, { passive: true });
document.addEventListener('fullscreenchange', scheduleMobileViewportRecovery, { passive: true });
document.addEventListener('webkitfullscreenchange', scheduleMobileViewportRecovery, { passive: true });

document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        scheduleMobileViewportRecovery();
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
            manualBtn.style.display = 'none';
        } else {
            manualBtn.style.display = '';
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
window.showLoadingOverlay = function(message = 'Loading...', autoHideMs = null) {
    let overlay = document.getElementById('global-loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'global-loading-overlay';
        overlay.innerHTML = '<div class="loading-spinner"></div><div id="loading-message">' + message + '</div>';
        document.body.appendChild(overlay);
    } else {
        const msgEl = overlay.querySelector('#loading-message');
        if (msgEl) msgEl.textContent = message;
    }
    overlay.classList.remove('hidden');
    overlay.style.setProperty('display', 'flex', 'important');
    if (window._loadingOverlayTimeout) {
        clearTimeout(window._loadingOverlayTimeout);
        window._loadingOverlayTimeout = null;
    }
    if (typeof autoHideMs === 'number' && autoHideMs > 0) {
        window._loadingOverlayTimeout = setTimeout(() => {
            if (window.hideLoadingOverlay) window.hideLoadingOverlay();
        }, autoHideMs);
    }
};

window.hideLoadingOverlay = function() {
    if (window._loadingOverlayTimeout) {
        clearTimeout(window._loadingOverlayTimeout);
        window._loadingOverlayTimeout = null;
    }
    const overlay = document.getElementById('global-loading-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
        overlay.style.setProperty('display', 'none', 'important');
    }
};

window.loadFAQUserCounts = async function() {
    try {
        const response = await fetch('/api/stats/user_count');
        if (response.ok) {
            const data = await response.json();
            const regCountEl = document.getElementById('faq-reg-count');
            const onlineCountEl = document.getElementById('faq-online-count');
            if (regCountEl) regCountEl.textContent = data.count || 0;
            if (onlineCountEl) onlineCountEl.textContent = data.online_count || 0;
        }
    } catch (e) {
        console.error('[loadFAQUserCounts] Error fetching user counts:', e);
    }
};

window.loadFAQDictionaryStats = async function() {
    try {
        const response = await fetch('/api/stats/dictionary');
        if (response.ok) {
            const data = await response.json();
            
            // 1. Total counts
            const nwlTotalEl = document.getElementById('faq-nwl-total');
            const cswTotalEl = document.getElementById('faq-csw-total');
            const awTotalEl = document.getElementById('faq-aw-total');
            const longTotalEl = document.getElementById('faq-long-total');
            
            if (nwlTotalEl) nwlTotalEl.textContent = (data.nwl_total || 0).toLocaleString();
            if (cswTotalEl) cswTotalEl.textContent = (data.csw_total || 0).toLocaleString();
            if (awTotalEl) awTotalEl.textContent = (data.aw_total || 0).toLocaleString();
            if (longTotalEl) longTotalEl.textContent = (data.long_total || 0).toLocaleString();
            
            // 2. Length breakdown table body
            const tbody = document.getElementById('faq-dict-stats-tbody');
            if (tbody) {
                let html = '';
                
                // Length ranges: 2 to 15, then 16+
                const lengths = [];
                for (let i = 2; i <= 15; i++) {
                    lengths.push(String(i));
                }
                lengths.push('16+');
                
                lengths.forEach(len => {
                    const nwlCount = data.nwl_dist[len] || 0;
                    const cswCount = data.csw_dist[len] || 0;
                    const awCount = data.aw_dist[len] || 0;
                    const longCount = data.long_dist[len] || 0;
                    
                    const lenLabel = len === '16+' ? '16+ Letters' : `${len} Letters`;
                    
                    html += `
                        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                            <td style="padding: 8px 10px; text-align: left; font-weight: 600; opacity: 0.8;">${lenLabel}</td>
                            <td style="padding: 8px 10px; font-weight: ${nwlCount > 0 ? '700' : 'normal'}; color: ${nwlCount > 0 ? '#60a5fa' : 'inherit'};">${nwlCount.toLocaleString()}</td>
                            <td style="padding: 8px 10px; font-weight: ${cswCount > 0 ? '700' : 'normal'}; color: ${cswCount > 0 ? '#fbbf24' : 'inherit'};">${cswCount.toLocaleString()}</td>
                            <td style="padding: 8px 10px; font-weight: ${awCount > 0 ? '700' : 'normal'}; color: ${awCount > 0 ? '#c084fc' : 'inherit'};">${awCount.toLocaleString()}</td>
                            <td style="padding: 8px 10px; font-weight: ${longCount > 0 ? '700' : 'normal'}; color: ${longCount > 0 ? '#f87171' : 'inherit'};">${longCount.toLocaleString()}</td>
                        </tr>
                    `;
                });
                
                tbody.innerHTML = html;
            }
        }
    } catch (e) {
        console.error('[loadFAQDictionaryStats] Error fetching dictionary stats:', e);
    }
};

// Activity timestamp tracker to support silence on > 1 hour return
function touchMorphemeActivity() {
    try {
        const now = Date.now().toString();
        localStorage.setItem('morpheme_last_active_time', now);
        localStorage.setItem('morpheme_last_active_timestamp', now);
    } catch(e) {}
}
['mousedown', 'keydown', 'touchstart', 'pointerdown', 'scroll'].forEach(evt => {
    window.addEventListener(evt, touchMorphemeActivity, { passive: true });
});
window.addEventListener('beforeunload', touchMorphemeActivity);
window.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        try {
            const last = parseInt(localStorage.getItem('morpheme_last_active_time') || localStorage.getItem('morpheme_last_active_timestamp') || '0', 10);
            if (last > 0 && (Date.now() - last >= 60 * 60 * 1000)) {
                window._suppressInactivityNotice = true;
                sessionStorage.setItem('morpheme_suppress_inactivity_notice', 'true');
                localStorage.removeItem('last_joined_room');
                if (window.currentRoomId) window.currentRoomId = null;
            }
        } catch(e) {}
    }
});

console.log('app.js fully loaded - version with UI optimizations');
