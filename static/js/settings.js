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

    // Global Settings State - Load from localStorage for instant availability (prevents race conditions)
    const savedSettings = localStorage.getItem('morpheme_settings');
    window.userSettings = savedSettings ? JSON.parse(savedSettings) : {
        lobby_music: true,
        chat_font_size: 13,
        def_font_size: 15,
        board_size: 60,
        cube_size: 220,
        highlight_typing: true,
        highlight_mouse: true,
        next_round_bell_enabled: true,
        letter_colors: {}
    };

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
            } else {
                console.log('[settings.js] No server settings, providing default view');
                applySettings(window.userSettings || {});
            }
        } catch (error) {
            console.error('[settings.js] Failed to load settings:', error);
        }
    }

    // 2. Apply Settings to UI and State
    function applySettings(settings) {
        // ... (existing code for board size, chat size, def size, music, theme, highlight typing/mouse, etc.)
        // Board Size
        if (settings.board_size) {
            const size = parseInt(settings.board_size);
            if (!isNaN(size) && boardSizeSlider) {
                document.documentElement.style.setProperty('--cell-size', `${size}px`);
                const previewBoard = document.getElementById('preview-board');
                if (previewBoard) {
                    previewBoard.style.setProperty('--cell-size', `${size}px`);
                }
                boardSizeSlider.value = size;
                if (boardSizeVal) boardSizeVal.textContent = `${size}px`;

                const playPage = document.getElementById('page-play');
                if (playPage) {
                    if (size > 65) playPage.classList.add('layout-huge-board');
                    else playPage.classList.remove('layout-huge-board');
                }
            }
        }

        // Cube Size (3D)
        if (settings.cube_size) {
            const size = parseInt(settings.cube_size);
            if (!isNaN(size)) {
                document.documentElement.style.setProperty('--cube-face-size', `${size}px`);
                document.documentElement.style.setProperty('--cube-half-size', `${size / 2}px`);
                document.documentElement.style.setProperty('--cube-container-size', `${size * 1.45}px`);
                
                const slider = document.getElementById('setting-cube-size');
                if (slider) slider.value = size;
                const label = document.getElementById('setting-cube-size-val');
                if (label) label.textContent = `${size}px`;
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
                if (preview) {
                    const container = preview.closest('.settings-preview-box');
                    if (container) container.style.fontSize = `${size}px`;
                    else preview.style.fontSize = `${size}px`;
                }
                const chatInput = document.getElementById('chat-input');
                if (chatInput) chatInput.style.fontSize = `${size}px`;
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
                if (preview) {
                    const container = preview.closest('.settings-preview-box');
                    if (container) container.style.fontSize = `${size}px`;
                    else preview.style.fontSize = `${size}px`;
                }
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

            if (typeof handleLobbyMusicState === 'function') {
                handleLobbyMusicState();
            }
        }

        // App Theme
        if (settings.app_theme) {
            applyTheme(settings.app_theme);
        }

        // Highlight as you type
        if (settings.highlight_typing !== undefined) {
            let val = settings.highlight_typing;
            if (val === 'true' || val === 'True' || val === true) val = true;
            else if (val === 'false' || val === 'False' || val === false) val = false;

            const highlightToggle = document.getElementById('setting-highlight-typing');
            if (highlightToggle) highlightToggle.checked = val;

            if (!window.userSettings) window.userSettings = {};
            window.userSettings.highlight_typing = val;
        }

        // Highlight typing color
        if (settings.highlight_typing_color) {
            const color = settings.highlight_typing_color;
            document.documentElement.style.setProperty('--highlight-typing-color', color);

            const dots = document.querySelectorAll('#highlight-color-picker .color-dot');
            dots.forEach(dot => {
                if (dot.getAttribute('data-color') === color) dot.classList.add('active');
                else dot.classList.remove('active');
            });
        }

        // Highlight as you mouse
        if (settings.highlight_mouse !== undefined) {
            let val = settings.highlight_mouse;
            if (val === 'true' || val === 'True' || val === true) val = true;
            else if (val === 'false' || val === 'False' || val === false) val = false;

            const highlightToggle = document.getElementById('setting-highlight-mouse');
            if (highlightToggle) highlightToggle.checked = val;

            if (!window.userSettings) window.userSettings = {};
            window.userSettings.highlight_mouse = val;
        }

        // Highlight mouse color
        if (settings.highlight_mouse_color) {
            const color = settings.highlight_mouse_color;
            document.documentElement.style.setProperty('--highlight-mouse-color', color);

            const dots = document.querySelectorAll('#highlight-mouse-color-picker .color-dot');
            dots.forEach(dot => {
                if (dot.getAttribute('data-color') === color) dot.classList.add('active');
                else dot.classList.remove('active');
            });
        }

        // Next Round Bell Enabled
        if (settings.next_round_bell_enabled !== undefined) {
            let val = settings.next_round_bell_enabled;
            if (val === 'true' || val === 'True' || val === true) val = true;
            else if (val === 'false' || val === 'False' || val === false) val = false;

            const bellToggle = document.getElementById('setting-next-round-bell');
            if (bellToggle) bellToggle.checked = val;

            const container = document.getElementById('bell-selection-container');
            if (container) container.style.display = val ? 'flex' : 'none';

            if (!window.userSettings) window.userSettings = {};
            window.userSettings.next_round_bell_enabled = val;
            localStorage.setItem('morpheme_settings', JSON.stringify(window.userSettings));
        }

        // Next Round Bell Type
        if (settings.next_round_bell_type) {
            const type = settings.next_round_bell_type;
            const bellBtns = document.querySelectorAll('.bell-btn');
            bellBtns.forEach(btn => {
                if (btn.getAttribute('data-bell') === type) btn.classList.add('active');
                else btn.classList.remove('active');
            });

            if (!window.userSettings) window.userSettings = {};
            window.userSettings.next_round_bell_type = type;
            localStorage.setItem('morpheme_settings', JSON.stringify(window.userSettings));
        }

        // Synesthesia: Letter Colors
        if (settings.letter_colors) {
            let colors = settings.letter_colors;
            if (typeof colors === 'string') {
                try {
                    colors = JSON.parse(colors);
                } catch (e) {
                    console.warn('[settings.js] Failed to parse letter colors:', e);
                    colors = {};
                }
            }
            if (!window.userSettings) window.userSettings = {};
            window.userSettings.letter_colors = colors;

            // Apply all variables
            Object.keys(colors).forEach(letter => {
                document.documentElement.style.setProperty(`--letter-${letter}-color`, colors[letter]);
            });

            // Update UI if grid is built
            updateSynesthesiaUI();
        }
    }

    // 3. Update Helpers
    const saveSettingDebounced = debounce(async (key, value) => {
        let saveVal = value;
        if (key === 'letter_colors') {
            saveVal = JSON.stringify(value);
        }
        console.log(`[settings.js] Saving ${key}: ${saveVal}`);
        try {
            // Cache locally for instant next load
            localStorage.setItem('morpheme_settings', JSON.stringify(window.userSettings));

            await fetch('/api/settings/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    key,
                    value: saveVal
                })
            });
        } catch (error) {
            console.error('[settings.js] Failed to save setting:', error);
        }
    }, 500);

    // Initializer for UI
    function initSynesthesia() {
        const grid = document.getElementById('synesthesia-letters-grid');
        if (!grid) return;

        grid.innerHTML = '';
        const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

        alphabet.forEach(letter => {
            const unit = document.createElement('div');
            unit.className = 'synesthesia-unit';
            unit.innerHTML = `
                <label>${letter}</label>
                <input type="color" data-letter="${letter}" value="${window.userSettings.letter_colors[letter] || '#111111'}">
            `;
            grid.appendChild(unit);

            const picker = unit.querySelector('input');
            picker.addEventListener('input', (e) => {
                const color = e.target.value;
                document.documentElement.style.setProperty(`--letter-${letter}-color`, color);
                window.userSettings.letter_colors[letter] = color;
                saveSettingDebounced('letter_colors', window.userSettings.letter_colors);
            });
        });

        const resetBtn = document.getElementById('setting-synesthesia-reset');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
                alphabet.forEach(letter => {
                    document.documentElement.style.removeProperty(`--letter-${letter}-color`);
                    delete window.userSettings.letter_colors[letter];
                });
                updateSynesthesiaUI();
                saveSettingDebounced('letter_colors', window.userSettings.letter_colors);
            });
        }
    }

    function updateSynesthesiaUI() {
        const grid = document.getElementById('synesthesia-letters-grid');
        if (!grid) return;

        const pickers = grid.querySelectorAll('input[type="color"]');
        pickers.forEach(p => {
            const letter = p.getAttribute('data-letter');
            p.value = window.userSettings.letter_colors[letter] || '#111111';
        });
    }

    // 4. Existing Listeners (boardSize, chatSize, defSize, music, highlight typing/mouse, theme, bell)
    if (boardSizeSlider) {
        boardSizeSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            window.userManuallyOverrodeBoardSize = true;
            window.cachedCellSize = val;
            document.documentElement.style.setProperty('--cell-size', `${val}px`);
            const previewBoard = document.getElementById('preview-board');
            if (previewBoard) {
                previewBoard.style.setProperty('--cell-size', `${val}px`);
            }
            if (boardSizeVal) boardSizeVal.textContent = `${val}px`;
            window.dispatchEvent(new Event('resize'));
            saveSettingDebounced('board_size', val);
        });
    }

    const chatSizeSlider = document.getElementById('setting-chat-size');
    if (chatSizeSlider) {
        chatSizeSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            document.documentElement.style.setProperty('--chat-font-size', `${val}px`);
            const label = document.getElementById('setting-chat-size-val');
            if (label) label.textContent = `${val}px`;
            const preview = document.getElementById('preview-chat-text');
            if (preview) {
                const container = preview.closest('.settings-preview-box');
                if (container) container.style.fontSize = `${val}px`;
                else preview.style.fontSize = `${val}px`;
            }
            saveSettingDebounced('chat_font_size', val);
        });
    }

    const defSizeSlider = document.getElementById('setting-def-size');
    if (defSizeSlider) {
        defSizeSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            document.documentElement.style.setProperty('--def-font-size', `${val}px`);
            const label = document.getElementById('setting-def-size-val');
            if (label) label.textContent = `${val}px`;
            const preview = document.getElementById('preview-def-text');
            if (preview) {
                const container = preview.closest('.settings-preview-box');
                if (container) container.style.fontSize = `${val}px`;
                else preview.style.fontSize = `${val}px`;
            }
            saveSettingDebounced('def_font_size', val);
        });
    }

    const cubeSizeSlider = document.getElementById('setting-cube-size');
    if (cubeSizeSlider) {
        cubeSizeSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            document.documentElement.style.setProperty('--cube-face-size', `${val}px`);
            document.documentElement.style.setProperty('--cube-half-size', `${val / 2}px`);
            document.documentElement.style.setProperty('--cube-container-size', `${val * 1.45}px`);
            const label = document.getElementById('setting-cube-size-val');
            if (label) label.textContent = `${val}px`;
            saveSettingDebounced('cube_size', val);
        });
    }

    const musicToggle = document.getElementById('setting-lobby-music');
    if (musicToggle) {
        musicToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            window.userSettings.lobby_music = val;
            if (typeof handleLobbyMusicState === 'function') handleLobbyMusicState();
            saveSettingDebounced('lobby_music', val);
        });
    }

    const highlightToggle = document.getElementById('setting-highlight-typing');
    if (highlightToggle) {
        highlightToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            window.userSettings.highlight_typing = val;
            saveSettingDebounced('highlight_typing', val);
        });
    }

    const typingColorDots = document.querySelectorAll('#highlight-color-picker .color-dot');
    typingColorDots.forEach(dot => {
        dot.addEventListener('click', () => {
            const color = dot.getAttribute('data-color');
            document.documentElement.style.setProperty('--highlight-typing-color', color);
            window.userSettings.highlight_typing_color = color;
            typingColorDots.forEach(d => d.classList.remove('active'));
            dot.classList.add('active');
            saveSettingDebounced('highlight_typing_color', color);
        });
    });

    const mouseHighlightToggle = document.getElementById('setting-highlight-mouse');
    if (mouseHighlightToggle) {
        mouseHighlightToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            window.userSettings.highlight_mouse = val;
            saveSettingDebounced('highlight_mouse', val);
        });
    }

    const mouseColorDots = document.querySelectorAll('#highlight-mouse-color-picker .color-dot');
    mouseColorDots.forEach(dot => {
        dot.addEventListener('click', () => {
            const color = dot.getAttribute('data-color');
            document.documentElement.style.setProperty('--highlight-mouse-color', color);
            window.userSettings.highlight_mouse_color = color;
            mouseColorDots.forEach(d => d.classList.remove('active'));
            dot.classList.add('active');
            saveSettingDebounced('highlight_mouse_color', color);
        });
    });

    const themeBtns = document.querySelectorAll('.theme-btn');
    themeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const theme = btn.getAttribute('data-theme');
            applyTheme(theme);
            saveSettingDebounced('app_theme', theme);
        });
    });

    const bellToggle = document.getElementById('setting-next-round-bell');
    if (bellToggle) {
        bellToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            const container = document.getElementById('bell-selection-container');
            if (container) container.style.display = val ? 'flex' : 'none';
            window.userSettings.next_round_bell_enabled = val;
            saveSettingDebounced('next_round_bell_enabled', val);
        });
    }

    const bellBtns = document.querySelectorAll('.bell-btn');
    bellBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const bell = btn.getAttribute('data-bell');
            window.userSettings.next_round_bell_type = bell;
            bellBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            playPreviewBell(bell);
            saveSettingDebounced('next_round_bell_type', bell);
        });
    });

    function playPreviewBell(type) {
        const audio = new Audio(`/static/audio/${type}.wav`);
        audio.play().catch(e => console.log('Audio play failed:', e));
    }

    function applyTheme(theme) {
        document.body.className = document.body.className.replace(/\btheme-\S+/g, '');
        if (theme && theme !== 'default') document.body.classList.add(`theme-${theme}`);
        themeBtns.forEach(b => {
            if (b.getAttribute('data-theme') === theme) b.classList.add('active');
            else b.classList.remove('active');
        });
    }

    // Initialize
    initSynesthesia();
    loadSettings();
    initPreviewInteraction();

    // Board Mouse interaction logic for preview
    function initPreviewInteraction() {
        const previewBoard = document.getElementById('preview-board');
        if (!previewBoard) return;
        let isDown = false;
        const start = (e) => {
            isDown = true;
            highlightCell(e);
            e.preventDefault();
        };
        const move = (e) => {
            if (!isDown) return;
            highlightCell(e);
            e.preventDefault();
        };
        const end = () => {
            isDown = false;
            setTimeout(() => {
                previewBoard.querySelectorAll('.board-cell').forEach(c => c.classList.remove('selected'));
            }, 300);
        };
        previewBoard.addEventListener('mousedown', start);
        previewBoard.addEventListener('touchstart', start, {
            passive: false
        });
        document.addEventListener('mousemove', move);
        document.addEventListener('touchmove', move, {
            passive: false
        });
        document.addEventListener('mouseup', end);
        document.addEventListener('touchend', end);
    }

    function highlightCell(e) {
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const point = {
            x: clientX,
            y: clientY
        };
        const cells = document.querySelectorAll('#preview-board .board-cell');
        for (const cell of cells) {
            if (isPointInOctagon(point, cell)) {
                cell.classList.add('selected');
                break;
            }
        }
    }

    function isPointInOctagon(point, cell) {
        const rect = cell.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const r = rect.width * 0.58;
        const dx = point.x - centerX;
        const dy = point.y - centerY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance > r) return false;
        const absDx = Math.abs(dx);
        const absDy = Math.abs(dy);
        const maxDist = Math.max(absDx, absDy);
        const minDist = Math.min(absDx, absDy);
        return maxDist + 0.414 * minDist <= r;
    }

    window.loadSettings = loadSettings;
    window.applySettings = applySettings;
})();
