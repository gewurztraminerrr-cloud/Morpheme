// settings.js - Handle user preferences

// Debounce helper
function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

(function () {
    console.log('[settings.js] Loading settings module...');

    // DOM Elements
    const boardSizeSlider = document.getElementById('setting-board-size');
    const boardSizeVal = document.getElementById('setting-board-size-val');

    // 1. Load Settings on Startup
    async function loadSettings() {
        try {
            const response = await fetch('/api/settings');
            const data = await response.json();

            if (data.settings) {
                console.log('[settings.js] Settings loaded:', data.settings);
                applySettings(data.settings);
            }
        } catch (error) {
            console.error('[settings.js] Failed to load settings:', error);
        }
    }

    // 2. Apply Settings to UI and State
    function applySettings(settings) {
        // Board Size
        if (settings.board_size) {
            const size = parseInt(settings.board_size);
            if (!isNaN(size) && boardSizeSlider) {
                document.documentElement.style.setProperty('--cell-size', `${size}px`);
                boardSizeSlider.value = size;
                if (boardSizeVal) boardSizeVal.textContent = `${size}px`;

                const playPage = document.getElementById('page-play');
                if (playPage) {
                    if (size > 65) playPage.classList.add('layout-huge-board');
                    else playPage.classList.remove('layout-huge-board');
                }
            }
        }

        // Chat Font Size
        if (settings.chat_font_size) {
            const size = parseInt(settings.chat_font_size);
            if (!isNaN(size)) {
                document.documentElement.style.setProperty('--chat-font-size', `${size}px`);
                const slider = document.getElementById('setting-chat-size');
                if (slider) slider.value = size;
                const label = document.getElementById('setting-chat-size-val');
                if (label) label.textContent = `${size}px`;
                const preview = document.getElementById('preview-chat-text');
                if (preview) preview.style.fontSize = `${size}px`;
            }
        }

        // Definition Font Size
        if (settings.def_font_size) {
            const size = parseInt(settings.def_font_size);
            if (!isNaN(size)) {
                document.documentElement.style.setProperty('--def-font-size', `${size}px`);
                const slider = document.getElementById('setting-def-size');
                if (slider) slider.value = size;
                const label = document.getElementById('setting-def-size-val');
                if (label) label.textContent = `${size}px`;
                const preview = document.getElementById('preview-def-text');
                if (preview) preview.style.fontSize = `${size}px`;
            }
        }

        // Lobby Music
        if (settings.lobby_music !== undefined) {
            let val = settings.lobby_music;
            if (val === 'true' || val === 'True' || val === true) val = true;
            else if (val === 'false' || val === 'False' || val === false) val = false;

            const musicToggle = document.getElementById('setting-lobby-music');
            if (musicToggle) musicToggle.checked = val;

            if (window.userSettings) window.userSettings.lobby_music = val;

            // Trigger update if function exists (in app.js)
            if (typeof handleLobbyMusicState === 'function') {
                handleLobbyMusicState();
            }
        }

        // App Theme
        if (settings.app_theme) {
            applyTheme(settings.app_theme);
        }
    }

    // 3. Handle Updates
    const saveSettingDebounced = debounce(async (key, value) => {
        console.log(`[settings.js] Saving ${key}: ${value}`);
        try {
            await fetch('/api/settings/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key, value })
            });
        } catch (error) {
            console.error('[settings.js] Failed to save setting:', error);
        }
    }, 500); // 500ms debounce

    // 4. Input Listeners
    if (boardSizeSlider) {
        boardSizeSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            document.documentElement.style.setProperty('--cell-size', `${val}px`);
            if (boardSizeVal) boardSizeVal.textContent = `${val}px`;
            window.dispatchEvent(new Event('resize'));
            saveSettingDebounced('board_size', val);
        });
    }

    // Chat Font Size Listener
    const chatSizeSlider = document.getElementById('setting-chat-size');
    if (chatSizeSlider) {
        chatSizeSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            document.documentElement.style.setProperty('--chat-font-size', `${val}px`);
            const label = document.getElementById('setting-chat-size-val');
            if (label) label.textContent = `${val}px`;
            const preview = document.getElementById('preview-chat-text');
            if (preview) preview.style.fontSize = `${val}px`;
            saveSettingDebounced('chat_font_size', val);
        });
    }

    // Definition Font Size Listener
    const defSizeSlider = document.getElementById('setting-def-size');
    if (defSizeSlider) {
        defSizeSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            document.documentElement.style.setProperty('--def-font-size', `${val}px`);
            const label = document.getElementById('setting-def-size-val');
            if (label) label.textContent = `${val}px`;
            const preview = document.getElementById('preview-def-text');
            if (preview) preview.style.fontSize = `${val}px`;
            saveSettingDebounced('def_font_size', val);
        });
    }

    // Lobby Music Listener
    const musicToggle = document.getElementById('setting-lobby-music');
    if (musicToggle) {
        musicToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            if (window.userSettings) window.userSettings.lobby_music = val;
            if (typeof handleLobbyMusicState === 'function') handleLobbyMusicState();
            saveSettingDebounced('lobby_music', val);
        });
    }

    // Theme Selection Listeners
    const themeBtns = document.querySelectorAll('.theme-btn');
    themeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const theme = btn.getAttribute('data-theme');
            applyTheme(theme);
            saveSettingDebounced('app_theme', theme);
        });
    });

    function applyTheme(theme) {
        // Remove existing theme classes
        document.body.className = document.body.className.replace(/\btheme-\S+/g, '');

        // Add new theme class (if not default)
        if (theme && theme !== 'default') {
            document.body.classList.add(`theme-${theme}`);
        }

        // Update Active Button State
        themeBtns.forEach(b => {
            if (b.getAttribute('data-theme') === theme) b.classList.add('active');
            else b.classList.remove('active');
        });
    }

    // Initialize
    loadSettings();
    initPreviewInteraction();

    // --- Preview Board Interaction ---
    function initPreviewInteraction() {
        const previewBoard = document.getElementById('preview-board');
        if (!previewBoard) return;

        let isDown = false;

        // Start
        const start = (e) => {
            isDown = true;
            highlightCell(e);
            e.preventDefault(); // Prevent scroll on touch
        };

        // Move
        const move = (e) => {
            if (!isDown) return;
            highlightCell(e);
            e.preventDefault();
        };

        // End
        const end = () => {
            isDown = false;
            // Clear selection after a delay for effect, or keep it? 
            // Better to keep it briefly or clear on next interaction.
            // Let's clear immediately on end for "test" feel.
            setTimeout(() => {
                previewBoard.querySelectorAll('.board-cell').forEach(c => c.classList.remove('selected'));
            }, 300);
        };

        previewBoard.addEventListener('mousedown', start);
        previewBoard.addEventListener('touchstart', start, { passive: false });

        document.addEventListener('mousemove', move);
        document.addEventListener('touchmove', move, { passive: false });

        document.addEventListener('mouseup', end);
        document.addEventListener('touchend', end);
    }

    function highlightCell(e) {
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const point = { x: clientX, y: clientY };

        // Scope to preview board cells only
        const cells = document.querySelectorAll('#preview-board .board-cell');

        for (const cell of cells) {
            if (isPointInOctagon(point, cell)) {
                cell.classList.add('selected');
                break; // Found the cell
            }
        }
    }

    // Octagonal hit test (Copied from mouse_selection.js for independence)
    function isPointInOctagon(point, cell) {
        const rect = cell.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        // Dynamic Radius Logic to match visual sizing
        // cell width is variable now, so we calculate radius relative to it.
        // Original: 60px cell -> 35px radius (~1.16x half-width) or just slightly larger than half-width (30px).
        // Let's use 58% of the rendered width to allow bridging gaps.
        const r = rect.width * 0.58;

        // Distance from center
        const dx = point.x - centerX;
        const dy = point.y - centerY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Circular boundary check
        if (distance > r) return false;

        // Octagonal bounds
        const absDx = Math.abs(dx);
        const absDy = Math.abs(dy);
        const maxDist = Math.max(absDx, absDy);
        const minDist = Math.min(absDx, absDy);

        return maxDist + 0.414 * minDist <= r;
    }

})();
