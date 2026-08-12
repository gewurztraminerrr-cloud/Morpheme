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
        triple_music: true,
        chat_font_size: 13,
        def_font_size: 15,
        board_size: 54,
        corner_cutoff: 39,
        board_sizes: { '4x4': 82, '4x6': 82, '5x7': 65, '6x8': 54 },
        cube_size: 220,
        highlight_typing: true,
        highlight_mouse: true,
        next_round_bell_enabled: true,
        vibration_alert: true,
        letter_colors: {},
        word_flash: true,
        board_sounds: true
    };

    if (window.userSettings) {
        if (typeof window.userSettings.board_sounds === 'undefined') {
            window.userSettings.board_sounds = true;
        }
        if (typeof window.userSettings.triple_music === 'undefined') {
            window.userSettings.triple_music = true;
        }
        if (typeof window.userSettings.vibration_alert === 'undefined') {
            window.userSettings.vibration_alert = true;
        }
    }

    // DOM Elements
    // (Global boardSizeSlider removed as per user request)

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
        // Dimension Specific Board Sizes
        if (settings.board_sizes) {
            let sizes = settings.board_sizes;
            if (typeof sizes === 'string') {
                try { sizes = JSON.parse(sizes); } catch (e) { sizes = { '4x4': 82, '4x6': 82, '5x7': 65, '6x8': 54 }; }
            }
            if (!window.userSettings) window.userSettings = {};
            window.userSettings.board_sizes = sizes;

            Object.keys(sizes).forEach(dim => {
                const slider = document.querySelector(`.dim-size-slider[data-dim="${dim}"]`);
                const valEl = document.getElementById(`val-dim-${dim}`);
                if (slider) slider.value = sizes[dim];
                if (valEl) valEl.textContent = `${sizes[dim]}px`;
            });

            // Initial preview board size set to the 4x4 grid size
            const previewBoard = document.getElementById('preview-board');
            if (previewBoard) {
                previewBoard.style.setProperty('--cell-size', `${sizes['4x4'] || 82}px`);
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

        // Corner Cutoff (Octagon vs Diamond Selectable Space)
        if (settings.corner_cutoff !== undefined) {
            const val = parseInt(settings.corner_cutoff);
            if (!isNaN(val)) {
                document.documentElement.style.setProperty('--corner-cutoff', `${val}%`);
                const slider = document.getElementById('setting-corner-cutoff');
                if (slider) slider.value = val;
                const label = document.getElementById('setting-corner-cutoff-val');
                if (label) label.textContent = `${val}%`;
                const shape = document.getElementById('preview-hitbox-shape');
                if (shape) {
                    shape.style.clipPath = `polygon(${val}% 0%, calc(100% - ${val}%) 0%, 100% ${val}%, 100% calc(100% - ${val}%), calc(100% - ${val}%) 100%, ${val}% 100%, 0% calc(100% - ${val}%), 0% ${val}%)`;
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

        // Triple Format Music
        if (settings.triple_music !== undefined) {
            let val = settings.triple_music;
            if (val === 'true' || val === 'True' || val === true) val = true;
            else if (val === 'false' || val === 'False' || val === false) val = false;

            const tripleMusicToggle = document.getElementById('setting-triple-music');
            if (tripleMusicToggle) tripleMusicToggle.checked = val;

            if (window.userSettings) window.userSettings.triple_music = val;

            if (typeof updateTripleMusicState === 'function') {
                const remaining = (window.lastGameState && window.lastGameState.time_remaining) || 0;
                updateTripleMusicState(remaining);
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

        // Word Flash Effect
        if (settings.word_flash !== undefined) {
            let val = settings.word_flash;
            if (val === 'true' || val === 'True' || val === true) val = true;
            else if (val === 'false' || val === 'False' || val === false) val = false;

            const flashToggle = document.getElementById('setting-word-flash');
            if (flashToggle) flashToggle.checked = val;

            if (!window.userSettings) window.userSettings = {};
            window.userSettings.word_flash = val;
        }

        // Board Sound Effects
        if (settings.board_sounds !== undefined) {
            let val = settings.board_sounds;
            if (val === 'true' || val === 'True' || val === true) val = true;
            else if (val === 'false' || val === 'False' || val === false) val = false;

            const soundsToggle = document.getElementById('setting-board-sounds');
            if (soundsToggle) soundsToggle.checked = val;

            if (!window.userSettings) window.userSettings = {};
            window.userSettings.board_sounds = val;
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

        // Intermission Vibration Alert
        if (settings.vibration_alert !== undefined) {
            let val = settings.vibration_alert;
            if (val === 'true' || val === 'True' || val === true) val = true;
            else if (val === 'false' || val === 'False' || val === false) val = false;

            const vibrationToggle = document.getElementById('setting-vibration-alert');
            if (vibrationToggle) vibrationToggle.checked = val;

            if (!window.userSettings) window.userSettings = {};
            window.userSettings.vibration_alert = val;
            localStorage.setItem('morpheme_settings', JSON.stringify(window.userSettings));
        }

        // Allow Private Messages
        if (settings.allow_pm !== undefined) {
            let val = settings.allow_pm;
            if (val === 'true' || val === 'True' || val === true) val = true;
            else if (val === 'false' || val === 'False' || val === false) val = false;

            const pmToggle = document.getElementById('setting-allow-pm');
            if (pmToggle) pmToggle.checked = val;

            if (!window.userSettings) window.userSettings = {};
            window.userSettings.allow_pm = val;
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

            if (typeof window.updateIntermissionBellSource === 'function') {
                window.updateIntermissionBellSource();
            }
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

        // Cache the fully merged settings locally so the warm cache matches server configurations immediately
        if (window.userSettings) {
            localStorage.setItem('morpheme_settings', JSON.stringify(window.userSettings));
        }
    }

    // 3. Update Helpers
    const saveSettingDebounced = debounce(async (key, value) => {
        let saveVal = value;
        if (key === 'letter_colors' || key === 'board_sizes') {
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

    const dimSliders = document.querySelectorAll('.dim-size-slider');
    dimSliders.forEach(slider => {
        slider.addEventListener('input', (e) => {
            const dim = e.target.getAttribute('data-dim');
            const val = parseInt(e.target.value);
            console.log('[settings.js] dimSlider input:', dim, val);
            const valEl = document.getElementById(`val-dim-${dim}`);
            if (valEl) valEl.textContent = `${val}px`;

            const previewBoard = document.getElementById('preview-board');
            if (previewBoard) {
                previewBoard.style.setProperty('--cell-size', `${val}px`);
            }

            if (!window.userSettings) window.userSettings = {};
            if (!window.userSettings.board_sizes) window.userSettings.board_sizes = {};
            window.userSettings.board_sizes[dim] = val;
            localStorage.setItem('morpheme_settings', JSON.stringify(window.userSettings));
            saveSettingDebounced('board_sizes', window.userSettings.board_sizes);

            // Check if the active room matches this dimension; if so apply immediately
            const gs = window.lastGameState;
            if (gs && gs.board && gs.board[0]) {
                const activeCols = gs.board[0].length;
                const activeRows = gs.board.length;
                const minD = Math.min(activeCols, activeRows);
                const maxD = Math.max(activeCols, activeRows);
                if (`${minD}x${maxD}` === dim) {
                    window.userManuallyOverrodeBoardSize = true;
                    if (!window.cachedCellSizes) window.cachedCellSizes = {};
                    window.cachedCellSizes[dim] = val;
                    document.documentElement.style.setProperty('--cell-size', `${val}px`);
                    const playPage = document.getElementById('page-play');
                    if (playPage) playPage.style.setProperty('--cell-size', `${val}px`);
                    const boardEl = document.getElementById('game-board');
                    if (boardEl) boardEl.style.setProperty('--cell-size', `${val}px`);
                    // Recalculate panel widths directly
                    if (typeof window.checkBoardOverflow === 'function') {
                        window.checkBoardOverflow();
                    } else if (typeof window.applyPanelLayout === 'function') {
                        window.applyPanelLayout(val, activeCols);
                    }
                }
            }
        });
    });
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

    const cutoffSlider = document.getElementById('setting-corner-cutoff');
    if (cutoffSlider) {
        cutoffSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            document.documentElement.style.setProperty('--corner-cutoff', `${val}%`);
            const label = document.getElementById('setting-corner-cutoff-val');
            if (label) label.textContent = `${val}%`;
            const shape = document.getElementById('preview-hitbox-shape');
            if (shape) {
                shape.style.clipPath = `polygon(${val}% 0%, calc(100% - ${val}%) 0%, 100% ${val}%, 100% calc(100% - ${val}%), calc(100% - ${val}%) 100%, ${val}% 100%, 0% calc(100% - ${val}%), 0% ${val}%)`;
            }
            if (!window.userSettings) window.userSettings = {};
            window.userSettings.corner_cutoff = val;
            localStorage.setItem('morpheme_settings', JSON.stringify(window.userSettings));
            saveSettingDebounced('corner_cutoff', val);
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

    const tripleMusicToggle = document.getElementById('setting-triple-music');
    if (tripleMusicToggle) {
        tripleMusicToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            window.userSettings.triple_music = val;
            if (typeof updateTripleMusicState === 'function') {
                const remaining = (window.lastGameState && window.lastGameState.time_remaining) || 0;
                updateTripleMusicState(remaining);
            }
            saveSettingDebounced('triple_music', val);
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
 
    const flashToggle = document.getElementById('setting-word-flash');
    if (flashToggle) {
        flashToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            window.userSettings.word_flash = val;
            saveSettingDebounced('word_flash', val);
        });
    }

    const soundsToggle = document.getElementById('setting-board-sounds');
    if (soundsToggle) {
        soundsToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            window.userSettings.board_sounds = val;
            saveSettingDebounced('board_sounds', val);
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

    const vibrationToggle = document.getElementById('setting-vibration-alert');
    if (vibrationToggle) {
        vibrationToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            window.userSettings.vibration_alert = val;
            saveSettingDebounced('vibration_alert', val);
        });
    }

    const pmToggle = document.getElementById('setting-allow-pm');
    if (pmToggle) {
        pmToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            window.userSettings.allow_pm = val;
            saveSettingDebounced('allow_pm', val);
        });
    }

    const bellBtns = document.querySelectorAll('.bell-btn');
    bellBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const bell = btn.getAttribute('data-bell');
            window.userSettings.next_round_bell_type = bell;
            bellBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            if (typeof window.updateIntermissionBellSource === 'function') {
                window.updateIntermissionBellSource();
            }
            
            playPreviewBell(bell);
            saveSettingDebounced('next_round_bell_type', bell);
        });
    });
 
    function playPreviewBell(type) {
        if (typeof MorphemeAudioBridge !== 'undefined') {
            try {
                // Send bell chime natively for mobile app wrappers
                MorphemeAudioBridge.postMessage(JSON.stringify({ sound: 'bell', type: type }));
            } catch (e) {
                console.error("MorphemeAudioBridge error:", e);
            }
            return;
        }
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

    // --- Settings Tab Navigation ---
    window.showSettingTab = function(tabId) {
        const sidebar = document.querySelector('#page-settings .tools-sidebar');
        const content = document.querySelector('#page-settings .tools-content');
        if (!sidebar || !content) return;

        // Update active class on buttons
        sidebar.querySelectorAll('.tool-nav-btn').forEach(btn => {
            if (btn.dataset.settingTab === tabId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Update active class on panes
        content.querySelectorAll('.tool-pane').forEach(pane => {
            if (pane.id === `setting-tab-${tabId}`) {
                pane.classList.add('active');
            } else {
                pane.classList.remove('active');
            }
        });

        // Trigger scroll to content area on mobile
        const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
        if (isMobile) {
            const layout = document.querySelector('#page-settings .tools-split-layout');
            if (layout) {
                layout.scrollLeft = layout.clientWidth || layout.scrollWidth;
            }
        }
    };

    function setupSettingsNavigation() {
        const sidebar = document.querySelector('#page-settings .tools-sidebar');
        if (!sidebar) return;

        sidebar.addEventListener('click', (e) => {
            const btn = e.target.closest('.tool-nav-btn');
            if (!btn) return;

            const tabId = btn.dataset.settingTab;
            if (tabId) {
                window.showSettingTab(tabId);
            }
        });

        // Mobile Layout snapping on navigation
        const settingsPage = document.getElementById('page-settings');
        if (settingsPage) {
            const observer = new MutationObserver(() => {
                if (settingsPage.classList.contains('active')) {
                    const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
                    if (isMobile) {
                        setTimeout(() => {
                            const layoutEl = document.querySelector('#page-settings .tools-split-layout');
                            if (layoutEl) layoutEl.scrollLeft = 0;
                        }, 100);
                    }
                }
            });
            observer.observe(settingsPage, {
                attributes: true,
                attributeFilter: ['class']
            });
        }


        // Mobile touch swipe handling for sliding back to settings list
        const settingsContent = document.querySelector('#page-settings .tools-content');
        const settingsSidebar = document.querySelector('#page-settings .tools-sidebar');
        if (settingsContent && settingsSidebar) {
            let touchStartX = 0;
            let touchStartY = 0;
            settingsContent.addEventListener('touchstart', (e) => {
                touchStartX = e.changedTouches[0].screenX;
                touchStartY = e.changedTouches[0].screenY;
            }, { passive: true });
            
            settingsContent.addEventListener('touchend', (e) => {
                const touchEndX = e.changedTouches[0].screenX;
                const touchEndY = e.changedTouches[0].screenY;
                const diffX = touchEndX - touchStartX;
                const diffY = touchEndY - touchStartY;
                
                // If swiped right (diffX > 80) and horizontal movement was dominant
                if (diffX > 80 && Math.abs(diffX) > Math.abs(diffY)) {
                    const layoutEl = document.querySelector('#page-settings .tools-split-layout');
                    if (layoutEl) layoutEl.scrollTo({ left: 0, behavior: 'smooth' });
                }
            }, { passive: true });
        }

        // Mobile back button inside settings content
        const mobileBackBtn = document.getElementById('settings-mobile-back-btn');
        if (mobileBackBtn) {
            mobileBackBtn.addEventListener('click', () => {
            const layoutEl = document.querySelector('#page-settings .tools-split-layout');
            if (layoutEl) layoutEl.scrollTo({ left: 0, behavior: 'smooth' });
            });
        }
    }

    // Initialize navigation
    setupSettingsNavigation();

    window.loadSettings = loadSettings;
    window.applySettings = applySettings;
})();
