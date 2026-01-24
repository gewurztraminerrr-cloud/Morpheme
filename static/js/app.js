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

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    setupNavigation();
    setupAuth();
    await checkSession();
});

// Check if user is already logged in
async function checkSession() {
    try {
        const response = await fetch('/api/check-session');
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
                const pageId = 'page-' + btn.getAttribute('data-page');

                // If navigating to lobby, clean up room state
                if (pageId === 'page-lobby') {
                    await leaveCurrentRoom();
                }

                showPage(pageId);
                updateActiveNav(btn);
            }
        });
    });
}

function showPage(pageId) {
    // Handle lobby music
    const lobbyMusic = document.getElementById('lobby-music');
    if (lobbyMusic) {
        if (pageId === 'page-lobby') {
            // Set to start of loop section (3:25 = 205 seconds)
            lobbyMusic.currentTime = 205;
            lobbyMusic.play().catch(e => console.log('Audio play failed:', e));

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
document.addEventListener('DOMContentLoaded', () => {
    // Accumulative game type - goes directly to Play tab
    document.querySelectorAll('.grid-btn.accumulative').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const time = btn.getAttribute('data-time');
            const dimensions = btn.getAttribute('data-dimensions');

            // Store game parameters
            selectedRoom = {
                gameType: 'Accumulative',
                time: time,
                dimensions: dimensions
            };

            // Enable and navigate to Play tab
            document.getElementById('btn-play').disabled = false;
            showPage('page-play');
            updateActiveNav('btn-play');
        });
    });

    // First Come First Serve - shows rooms in right panel
    document.querySelectorAll('.grid-btn.fcfs').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const time = btn.getAttribute('data-time');
            const dimensions = btn.getAttribute('data-dimensions');

            updateActiveRoomsPanel('First Come First Serve', time, dimensions);
        });
    });

    // Split Points - shows rooms in right panel
    document.querySelectorAll('.grid-btn.split').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const time = btn.getAttribute('data-time');
            const dimensions = btn.getAttribute('data-dimensions');

            updateActiveRoomsPanel('Split Points', time, dimensions);
        });
    });
});

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
    document.getElementById('btn-play').disabled = false;

    // Visual feedback
    document.querySelectorAll('.room-card').forEach(card => {
        card.style.borderColor = 'var(--border)';
    });

    event.target.closest('.room-card').style.borderColor = 'var(--primary)';
}
