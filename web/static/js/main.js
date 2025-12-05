document.addEventListener('DOMContentLoaded', () => {
    // STATE
    const state = {
        algorithms: [],
        currentAlgo: null,
        generatedData: null,
        playMode: {
            active: false,
            difficulty: null,
            grid: null,
            foundWords: new Set(),
            allWords: new Set(),
            difficultWords: new Set(),
            score: 0,
            startTime: null,
            timerInterval: null
        }
    };

    // DOM ELEMENTS
    const tabs = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const algoSelect = document.getElementById('algo-select');
    const algoDesc = document.getElementById('algo-desc');
    const generateBtn = document.getElementById('generate-btn');
    const sandboxGrid = document.getElementById('sandbox-grid');
    const sandboxWords = document.getElementById('sandbox-words');

    // Play Mode Elements
    const difficultyCards = document.querySelectorAll('.card');
    const startGameBtn = document.getElementById('start-game-btn');
    const playSetup = document.getElementById('play-setup');
    const playArea = document.getElementById('play-area');
    const playSummary = document.getElementById('play-summary');
    const playGrid = document.getElementById('play-grid');
    const wordInput = document.getElementById('word-input');
    const messageArea = document.getElementById('message-area');
    const currentScoreEl = document.getElementById('current-score');
    const wordsFoundCountEl = document.getElementById('words-found-count');
    const playerWordsList = document.getElementById('player-words');
    const endGameBtn = document.getElementById('end-game-btn');
    const playAgainBtn = document.getElementById('play-again-btn');
    const timerEl = document.querySelector('.timer');

    // INITIALIZATION
    fetchAlgorithms();

    // EVENT LISTENERS
    tabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    algoSelect.addEventListener('change', updateAlgoInfo);
    generateBtn.addEventListener('click', generateSandboxPuzzle);

    difficultyCards.forEach(card => {
        card.addEventListener('click', () => selectDifficulty(card));
    });

    startGameBtn.addEventListener('click', startPlayGame);
    endGameBtn.addEventListener('click', endPlayGame);
    playAgainBtn.addEventListener('click', resetPlayMode);

    wordInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            submitWord();
        }
    });

    // FUNCTIONS

    function switchTab(tabId) {
        tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tabId));
        tabContents.forEach(c => c.classList.toggle('active', c.id === tabId));
    }

    async function fetchAlgorithms() {
        try {
            const response = await fetch('/api/algorithms');
            const data = await response.json();
            state.algorithms = data.algorithms;

            algoSelect.innerHTML = state.algorithms.map(algo =>
                `<option value="${algo.id}">${algo.name}</option>`
            ).join('');

            updateAlgoInfo();
        } catch (error) {
            console.error('Failed to fetch algorithms:', error);
        }
    }

    function updateAlgoInfo() {
        const algoId = algoSelect.value;
        const algo = state.algorithms.find(a => a.id === algoId);
        if (algo) {
            algoDesc.textContent = algo.description;
            state.currentAlgo = algo;
            renderAlgoParams(algo.id);
        }
    }

    function renderAlgoParams(algoId) {
        const container = document.getElementById('algo-params');
        container.innerHTML = '';

        if (algoId === 'hybrid') {
            container.innerHTML = `
                <label>Beam Width</label>
                <input type="number" id="param-beam-width" value="5" min="1" max="20">
            `;
        } else if (algoId === 'adaptive') {
            container.innerHTML = `
                <label>Density Type</label>
                <select id="param-density-type">
                    <option value="radial">Radial</option>
                    <option value="uniform">Uniform</option>
                </select>
            `;
        } else if (algoId === 'optimized') {
            container.innerHTML = `
                <label>Min Difficult Words</label>
                <input type="number" id="param-min-diff" value="5" min="0">
                <label>Max Attempts per iteration</label>
                <input type="number" id="param-max-attempts" value="100" min="10" max="10000">
                <label>Difficult % Required</label>
                <input type="number" id="param-diff-pct" value="0" min="0" max="100" step="5">
            `;
        } else if (algoId === 'checkerboard') {
            container.innerHTML = `
                <label>Iterations</label>
                <input type="number" id="param-iterations" value="2" min="1" max="10">
            `;
        } else if (algoId === 'checkerboard_v2') {
            container.innerHTML = `
                <label>Strategy</label>
                <select id="param-strategy">
                    <option value="distinct_difficult">Distinct Difficult</option>
                    <option value="total_difficult">Total Difficult</option>
                    <option value="difficult_ratio_6plus">6+ Letter Ratio</option>
                </select>
                <label>Iterations</label>
                <input type="number" id="param-iterations" value="2" min="1" max="10">
                <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem">
                    <input type="checkbox" id="param-adaptive"> Adaptive (auto-iterate)
                </label>
                <label>Target Ratio (if adaptive)</label>
                <input type="number" id="param-target-ratio" value="0.8" min="0" max="1" step="0.05">
            `;
        }
    }

    async function generateSandboxPuzzle() {
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';

        try {
            const width = parseInt(document.getElementById('grid-width').value);
            const height = parseInt(document.getElementById('grid-height').value);
            const seed = document.getElementById('seed-input').value;
            const algoId = algoSelect.value;

            const params = {};
            if (algoId === 'hybrid') {
                params.beam_width = parseInt(document.getElementById('param-beam-width').value);
            } else if (algoId === 'adaptive') {
                params.density_type = document.getElementById('param-density-type').value;
            } else if (algoId === 'optimized') {
                params.min_difficult_words = parseInt(document.getElementById('param-min-diff').value);
                params.max_attempts = parseInt(document.getElementById('param-max-attempts').value);
                params.difficult_percentage = parseFloat(document.getElementById('param-diff-pct').value);
            } else if (algoId === 'checkerboard') {
                params.iterations = parseInt(document.getElementById('param-iterations').value);
            } else if (algoId === 'checkerboard_v2') {
                params.strategy = document.getElementById('param-strategy').value;
                params.iterations = parseInt(document.getElementById('param-iterations').value);
                if (document.getElementById('param-adaptive').checked) {
                    params.adaptive = true;
                    params.target_ratio = parseFloat(document.getElementById('param-target-ratio').value);
                }
            }

            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    algorithm: algoId,
                    width,
                    height,
                    seed: seed ? parseInt(seed) : null,
                    params
                })
            });

            const data = await response.json();
            if (data.error) throw new Error(data.error);

            renderSandboxResult(data);
        } catch (error) {
            alert('Generation failed: ' + error.message);
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Puzzle';
        }
    }

    // Store current puzzle data for filtering
    let currentPuzzleData = null;

    function renderSandboxResult(data) {
        currentPuzzleData = data;

        // Basic counts
        document.getElementById('metric-total').textContent = data.found_words.length;
        document.getElementById('metric-difficult').textContent = data.difficult_words.length;
        document.getElementById('metric-seed').textContent = data.seed;

        // Compute 6+ letter stats
        const words6plus = data.found_words.filter(w => w.length >= 6);
        const diff6plus = data.difficult_words.filter(w => w.length >= 6);
        const longWords = data.found_words.filter(w => w.length >= 7);
        const ratio = words6plus.length > 0 ? (diff6plus.length / words6plus.length * 100).toFixed(1) : 0;

        document.getElementById('metric-6plus').textContent = words6plus.length;
        document.getElementById('metric-diff-6plus').textContent = diff6plus.length;
        document.getElementById('metric-ratio').textContent = ratio + '%';
        document.getElementById('metric-long').textContent = longWords.length;

        renderGrid(sandboxGrid, data.grid);
        updateWordList();
    }

    function updateWordList() {
        if (!currentPuzzleData) return;

        const filter = document.getElementById('word-filter')?.value || 'all';
        let words = currentPuzzleData.found_words;

        // Apply filter
        switch (filter) {
            case 'difficult':
                words = currentPuzzleData.difficult_words;
                break;
            case '6plus':
                words = currentPuzzleData.found_words.filter(w => w.length >= 6);
                break;
            case 'diff6plus':
                words = currentPuzzleData.difficult_words.filter(w => w.length >= 6);
                break;
        }

        sandboxWords.innerHTML = words.map(word => {
            const isDiff = currentPuzzleData.difficult_words.includes(word);
            return `<div class="word-item ${isDiff ? 'difficult' : ''}">${word}</div>`;
        }).join('');
    }

    // Word filter change listener
    document.getElementById('word-filter')?.addEventListener('change', updateWordList);

    function renderGrid(container, grid) {
        container.innerHTML = '';
        const height = grid.length;
        const width = grid[0].length;

        container.style.gridTemplateColumns = `repeat(${width}, 1fr)`;

        grid.forEach((row, r) => {
            row.forEach((char, c) => {
                const tile = document.createElement('div');
                tile.className = 'tile';
                tile.textContent = char;
                tile.dataset.r = r;
                tile.dataset.c = c;
                container.appendChild(tile);
            });
        });
    }

    // PLAY MODE LOGIC

    function selectDifficulty(card) {
        difficultyCards.forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        state.playMode.difficulty = card.dataset.difficulty;
        startGameBtn.disabled = false;
    }

    async function startPlayGame() {
        if (!state.playMode.difficulty) return;

        startGameBtn.disabled = true;
        startGameBtn.textContent = 'Loading...';

        let algoId = 'checkerboard';
        let params = {};

        switch (state.playMode.difficulty) {
            case 'easy':
                algoId = 'checkerboard';
                break;
            case 'medium':
                algoId = 'adaptive';
                params = { density_type: 'radial' };
                break;
            case 'hard':
                algoId = 'checkerboard_v2';
                params = { adaptive: true, target_ratio: 0.8 };
                break;
        }

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    algorithm: algoId,
                    width: 6,
                    height: 6,
                    params
                })
            });

            const data = await response.json();

            state.playMode.active = true;
            state.playMode.grid = data.grid;
            state.playMode.allWords = new Set(data.found_words);
            state.playMode.difficultWords = new Set(data.difficult_words);
            state.playMode.foundWords = new Set();
            state.playMode.score = 0;
            state.playMode.possibleScore = data.score;
            state.playMode.startTime = Date.now();

            playSetup.classList.add('hidden');
            playArea.classList.remove('hidden');

            renderGrid(playGrid, data.grid);
            updateScoreBoard();
            startTimer();

            wordInput.value = '';
            wordInput.focus();
            playerWordsList.innerHTML = '';
            messageArea.textContent = '';

        } catch (error) {
            alert('Failed to start game: ' + error.message);
            startGameBtn.disabled = false;
            startGameBtn.textContent = 'Start Game';
        }
    }

    function submitWord() {
        if (!state.playMode.active) return;

        const word = wordInput.value.trim().toUpperCase();
        wordInput.value = '';

        if (!word) return;

        if (state.playMode.foundWords.has(word)) {
            showMessage('Already found!', 'info');
            return;
        }

        if (state.playMode.allWords.has(word)) {
            state.playMode.foundWords.add(word);
            const isDiff = state.playMode.difficultWords.has(word);
            const points = isDiff ? 11 : 1;
            state.playMode.score += points;

            showMessage(isDiff ? 'Excellent! Difficult word found!' : 'Good!', 'success');

            const div = document.createElement('div');
            div.className = 'found-word';
            div.innerHTML = `<span>${word}</span><span>${points}</span>`;
            if (isDiff) div.style.color = 'var(--accent-color)';
            playerWordsList.prepend(div);

            updateScoreBoard();
        } else {
            showMessage('Not in puzzle', 'error');
        }
    }

    function showMessage(text, type) {
        messageArea.textContent = text;
        messageArea.className = `message-${type}`;
        setTimeout(() => {
            messageArea.textContent = '';
        }, 2000);
    }

    function updateScoreBoard() {
        currentScoreEl.textContent = state.playMode.score;
        wordsFoundCountEl.textContent = state.playMode.foundWords.size;
    }

    function startTimer() {
        clearInterval(state.playMode.timerInterval);
        state.playMode.timerInterval = setInterval(() => {
            const delta = Math.floor((Date.now() - state.playMode.startTime) / 1000);
            const mins = Math.floor(delta / 60).toString().padStart(2, '0');
            const secs = (delta % 60).toString().padStart(2, '0');
            timerEl.textContent = `${mins}:${secs}`;
        }, 1000);
    }

    function endPlayGame() {
        state.playMode.active = false;
        clearInterval(state.playMode.timerInterval);

        playArea.classList.add('hidden');
        playSummary.classList.remove('hidden');

        document.getElementById('final-score').textContent = state.playMode.score;
        document.getElementById('possible-score').textContent = state.playMode.possibleScore;

        const percent = Math.round((state.playMode.score / state.playMode.possibleScore) * 100) || 0;
        document.getElementById('score-percentage').textContent = `${percent}%`;

        const missedContainer = document.getElementById('missed-words-list');
        missedContainer.innerHTML = '';

        state.playMode.difficultWords.forEach(word => {
            if (!state.playMode.foundWords.has(word)) {
                const div = document.createElement('div');
                div.className = 'word-item difficult';
                div.textContent = word;
                missedContainer.appendChild(div);
            }
        });
    }

    function resetPlayMode() {
        playSummary.classList.add('hidden');
        playSetup.classList.remove('hidden');
        startGameBtn.disabled = false;
        startGameBtn.textContent = 'Start Game';
    }
});
