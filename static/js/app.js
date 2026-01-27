// Navigation system
const pages = {
    'btn-login': 'page-login',
    'btn-how-to-play': 'page-how-to-play',
    'btn-lobby': 'page-lobby',
    'btn-play': 'page-play',
    'btn-leaderboards': 'page-leaderboards',
    'btn-tools': 'page-tools',
    'btn-settings': 'page-settings',
    'btn-contact': 'page-contact'
};

let currentUser = null;
let selectedRoom = null;

// Define standardized rating ranges globaly for reuse
const RATING_RANGES = [
    // --- THE CLIMB (1 - 1399) ---
    // Greens (1 - 699): The Foundation
    { min: 1, max: 99, color: '#e6ffe6', label: '1 - 99' },
    { min: 100, max: 199, color: '#ccffcc', label: '100 - 199' },
    { min: 200, max: 299, color: '#99ff99', label: '200 - 299' },
    { min: 300, max: 399, color: '#66ff66', label: '300 - 399' },
    { min: 400, max: 499, color: '#33ff33', label: '400 - 499' },
    { min: 500, max: 599, color: '#00ff00', label: '500 - 599' },
    { min: 600, max: 699, color: '#00cc00', label: '600 - 699' },

    // Blues (700 - 1399): The Sky & Ocean
    { min: 700, max: 799, color: '#66ccff', label: '700 - 799' }, // distinctly blue
    { min: 800, max: 899, color: '#33bbff', label: '800 - 899' },
    { min: 900, max: 999, color: '#00aaff', label: '900 - 999' },
    { min: 1000, max: 1099, color: '#0088ff', label: '1000 - 1099' },
    { min: 1100, max: 1199, color: '#0066ff', label: '1100 - 1199' },
    { min: 1200, max: 1299, color: '#0044ff', label: '1200 - 1299' },
    { min: 1300, max: 1399, color: '#0000ff', label: '1300 - 1399' },

    // --- THE HEAT (1400 - 2499) ---
    // Yellows
    { min: 1400, max: 1499, color: '#ffff66', label: '1400 - 1499' },
    { min: 1500, max: 1599, color: '#ffff00', label: '1500 - 1599' },
    { min: 1600, max: 1699, color: '#ffcc00', label: '1600 - 1699' },
    { min: 1700, max: 1799, color: '#ffaa00', label: '1700 - 1799' },

    // Oranges
    { min: 1800, max: 1899, color: '#ff8800', label: '1800 - 1899' },
    { min: 1900, max: 1999, color: '#ff6600', label: '1900 - 1999' },
    { min: 2000, max: 2099, color: '#ff4400', label: '2000 - 2099' },
    { min: 2100, max: 2199, color: '#ff2200', label: '2100 - 2199' },

    // Reds
    { min: 2200, max: 2299, color: '#ff0000', label: '2200 - 2299' },
    { min: 2300, max: 2399, color: '#e60000', label: '2300 - 2399' },
    { min: 2400, max: 2499, color: '#cc0000', label: '2400 - 2499' },

    // --- THE VOID (2500 - 6000+) ---
    { min: 2500, max: 2599, color: '#b30000', label: '2500 - 2599' },
    { min: 2600, max: 2699, color: '#990000', label: '2600 - 2699' },
    { min: 2700, max: 2799, color: '#800000', label: '2700 - 2799' },
    { min: 2800, max: 2899, color: '#660000', label: '2800 - 2899' },
    { min: 2900, max: 2999, color: '#4d0000', label: '2900 - 2999' },

    { min: 3000, max: 3999, color: '#330000', label: '3000 - 3999' },
    { min: 4000, max: 4999, color: '#220000', label: '4000 - 4999' },
    { min: 5000, max: 5999, color: '#110000', label: '5000 - 5999' },
    { min: 6000, max: 99999, color: '#000000', label: '6000+' }
];

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    setupNavigation();
    setupModalListeners();
    setupAuth(); // Initialize auth listeners
    // handleGuestLogin(); // Don't auto-login guest, wait for button click
    checkSession();
});

// Check if user is already logged in
async function checkSession() {
    try {
        const response = await fetch('/api/session');
        const data = await response.json();

        if (data.logged_in) {
            currentUser = data.username;
            window.currentUser = currentUser;  // Expose globally
            navigateToLobby();
        }
    } catch (error) {
        console.error('Session check failed:', error);
    }
}

// Setup navigation
function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
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

                // If navigating to lobby, clean up room state
                /* 
                   DISABLED: Keep user directly in room to persist stats count [1].
                   User only leaves if they join another room or explicitly quit.
                if (pageId === 'page-lobby') {
                    await leaveCurrentRoom();
                }
                */

                showPage(pageId);
                updateActiveNav(btn);
            }
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
}

function showPage(pageId) {
    // Handle Game Polling
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
                // console.log('Focused word input on page switch');
            }
        }, 100);
    } else {
        if (window.stopGamePolling) {
            console.log('Leaving Play page - stopping polling');
            window.stopGamePolling();
        }
    }

    // Handle lobby music
    const lobbyMusic = document.getElementById('lobby-music');
    if (lobbyMusic) {
        if (pageId === 'page-lobby') {
            // Set to start of loop section (3:25 = 205 seconds)
            lobbyMusic.currentTime = 205;
            lobbyMusic.play().catch(e => {
                console.log('Audio autoplay blocked, waiting for interaction:', e);
                const resumeAudio = () => {
                    lobbyMusic.play().catch(err => console.log('Retry play failed', err));
                    document.removeEventListener('click', resumeAudio);
                };
                document.addEventListener('click', resumeAudio);
            });

            // Add event listener to loop between 3:25 and 4:55
            lobbyMusic.ontimeupdate = function () {
                if (lobbyMusic.currentTime >= 295) { // 4:55 = 295 seconds
                    lobbyMusic.currentTime = 205; // Loop back to 3:25
                }
            };
        } else {
            lobbyMusic.pause();
            lobbyMusic.currentTime = 0;
            lobbyMusic.ontimeupdate = null; // Remove event listener
        }
    }

    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.getElementById(pageId).classList.add('active');
}

function updateActiveNav(activeBtn) {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    if (activeBtn && activeBtn.classList) {
        activeBtn.classList.add('active');
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

    // Sign in form
    document.getElementById('signin-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleSignIn();
    });

    // Sign up form
    document.getElementById('signup-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleSignUp();
    });

    // Guest login button
    const guestBtn = document.getElementById('guest-login-btn');
    if (guestBtn) {
        guestBtn.addEventListener('click', async () => {
            await handleGuestLogin();
        });
    }
}

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
            navigateToLobby();
        } else {
            errorEl.textContent = data.message;
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
            navigateToLobby();
        } else {
            errorEl.textContent = data.message;
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
    // Update username display if it exists
    const usernameDisplay = document.getElementById('username-display');
    if (usernameDisplay) {
        usernameDisplay.textContent = currentUser;
    }

    // Show lobby page
    showPage('page-lobby');
    const lobbyBtn = document.querySelector('.nav-btn[data-page="lobby"]');
    if (lobbyBtn) {
        updateActiveNav(lobbyBtn);
    }
}

// Lobby game grid button handlers
// These are handled by lobby.js to create/join rooms via API
// Removing the duplicate listeners here to prevent conflicts/double-actions
/*
document.addEventListener('DOMContentLoaded', () => {
   // Handlers removed to avoid conflict with lobby.js
});
*/

function updateActiveRoomsPanel(gameType, time, dimensions) {
    const infoDiv = document.getElementById('selected-game-info');
    const roomsList = document.getElementById('rooms-list');

    infoDiv.innerHTML = `
        <p><strong>Game Type:</strong> ${gameType}</p>
        <p><strong>Board Dimensions:</strong> ${dimensions}</p>
        <p><strong>Time:</strong> ${time}</p>
    `;

    // Placeholder for actual rooms (would come from backend)
    roomsList.innerHTML = `
        <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
            No active rooms found. Create a new room to get started!
        </p>
    `;
}

function selectRoom(roomName) {
    selectedRoom = roomName;

    // Enable Play button
    document.getElementById('play-btn').disabled = false;

    // Visual feedback
    document.querySelectorAll('.room-card').forEach(card => {
        card.style.borderColor = 'var(--border)';
    });

    event.target.closest('.room-card').style.borderColor = 'var(--primary)';
}

// Define standardized rating ranges globaly for reuse


// Helper to get color for any rating
window.getRatingColor = function (rating) {
    if (rating === undefined || rating === null) return '#ffffff';
    // Find matching range
    const match = RATING_RANGES.find(r => rating >= r.min && rating <= r.max);
    if (match) return match.color;

    // Fallback logic if out of bounds
    if (rating < 1) return RATING_RANGES[0].color;
    return '#000000'; // Super high
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
        // Tooltip text
        segment.setAttribute('data-label', `${range.label}`);
        bar.appendChild(segment);
    });
}
