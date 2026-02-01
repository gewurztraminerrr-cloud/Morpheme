document.addEventListener('DOMContentLoaded', () => {
    setupToolsNavigation();
    setupComboChecker();
    setupListsTool();
    setupSequenceTool();
    setupManualTool();
    setupRandomWordTool();
    setupWotdTool();
});

function setupToolsNavigation() {
    const navBtns = document.querySelectorAll('.tool-nav-btn');
    const panes = document.querySelectorAll('.tool-pane');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update buttons
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Show pane
            const toolId = btn.dataset.tool; // e.g. "combo"
            panes.forEach(p => p.classList.remove('active'));
            const targetPane = document.getElementById(`tool-${toolId}`);
            if (targetPane) targetPane.classList.add('active');

            // Trigger fetch for Lists if selected (lazy load)
            if (toolId === 'lists') {
                fetchListsData();
            }
            if (toolId === 'wotd') {
                updateWotd();
            }
        });
    });
}

function setupComboChecker() {
    const searchBtn = document.getElementById('combo-search-btn');
    const input = document.getElementById('combo-input');

    if (searchBtn) {
        searchBtn.addEventListener('click', runComboSearch);
    }

    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runComboSearch();
        });
    }
}

async function runComboSearch() {
    const inputEl = document.getElementById('combo-input');
    const dictEl = document.getElementById('combo-dict');
    const resultsContainer = document.getElementById('combo-results');

    const searchTerm = inputEl.value.trim();
    const dictionary = dictEl.value;

    if (!searchTerm) return;

    // Clear previous results
    document.getElementById('mp-container').innerHTML = '';
    document.getElementById('lic-container').innerHTML = '';

    resultsContainer.classList.remove('hidden');

    try {
        const response = await fetch('/api/tools/combo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ search_term: searchTerm, dictionary: dictionary })
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Render MP Groups (0MP to 5MP)
        renderGroups(data.mp_groups, 'mp-container', 'MP');

        // Render LIC Groups (Shared Count)
        renderGroups(data.lic_groups, 'lic-container', 'LIC');

    } catch (error) {
        console.error('Combo check failed:', error);
        alert('An error occurred while checking combo.');
    }
}

function renderGroups(groupsData, containerId, type) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Sort keys logically
    // MP keys are 0, 1, 2... (Integers)
    // LIC keys are Lengths (Integers)
    const keys = Object.keys(groupsData).map(Number).sort((a, b) => a - b);

    keys.forEach(key => {
        const words = groupsData[key];
        if (words.length === 0) return;

        let label = '';
        if (type === 'MP') {
            label = `${key}MP`; // e.g. 0MP (0 Ops)
        } else {
            label = `${key}LIC`; // e.g. 5LIC
        }

        const colDiv = document.createElement('div');
        colDiv.className = 'group-column';

        colDiv.innerHTML = `
            <div class="group-header">${label}</div>
            <div class="group-table-container">
                <table class="group-table">
                    <tbody>
                        ${words.map(w => `<tr><td>${w}</td></tr>`).join('')}
                    </tbody>
                </table>
            </div>
        `;

        container.appendChild(colDiv);
    });
}

// --- Lists Tool Logic ---

let listsDataLoaded = false;

function setupListsTool() {
    const updateBtn = document.getElementById('list-update-btn');
    if (updateBtn) {
        updateBtn.addEventListener('click', () => {
            listsDataLoaded = false; // Force refresh
            fetchListsData();
        });
    }
}

async function fetchListsData() {
    // Get Filter Values
    const lengthSelect = document.getElementById('list-length-filter');
    const startSelect = document.getElementById('list-start-filter');

    // UI Feedback
    const colIds = ['col-nwl', 'col-csw', 'col-csw-only', 'col-likelihood', 'col-uniques', 'col-added'];
    colIds.forEach(id => {
        const el = document.querySelector(`#${id} .list-scroll-area`);
        if (el) el.innerHTML = '<div style="padding:10px; opacity:0.6;">Loading...</div>';
    });

    try {
        // Build Query URL
        let url = '/api/tools/lists?';

        if (lengthSelect && lengthSelect.value !== 'all') {
            url += `length=${lengthSelect.value}&`;
        }
        if (startSelect && startSelect.value !== 'all') {
            url += `starts_with=${startSelect.value}`;
        }

        const response = await fetch(url);
        const data = await response.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        renderListColumn('col-nwl', data.nwl);
        renderListColumn('col-csw', data.csw);
        renderListColumn('col-csw-only', data.csw_only);
        renderListColumn('col-likelihood', data.likelihood);
        renderListColumn('col-added', data.added);
        renderListColumn('col-uniques', data.uniques);

        listsDataLoaded = true;

    } catch (err) {
        console.error('Failed to fetch lists:', err);
        colIds.forEach(id => {
            const el = document.querySelector(`#${id} .list-scroll-area`);
            if (el) el.innerHTML = '<div style="color:red; padding:10px;">Error loading.</div>';
        });
    }
}

function renderListColumn(colId, words) {
    const container = document.querySelector(`#${colId} .list-scroll-area`);
    if (!container) return;

    if (!words || words.length === 0) {
        container.innerHTML = '<div style="padding:10px; opacity:0.6;">(Empty)</div>';
        return;
    }

    // Creating a huge string is faster than creating elements one by one.
    const html = words.map(w => `<div class="list-item">${w}</div>`).join('');
    container.innerHTML = html;
}

// --- Sequence Tool Logic ---

function setupSequenceTool() {
    const searchBtn = document.getElementById('seq-search-btn');
    const input = document.getElementById('seq-input');

    if (searchBtn) {
        searchBtn.addEventListener('click', runSequenceSearch);
    }

    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runSequenceSearch();
        });
    }
}

async function runSequenceSearch() {
    const inputEl = document.getElementById('seq-input');
    const modeEl = document.getElementById('seq-mode');
    const lengthEl = document.getElementById('seq-length');
    const resultsContainer = document.getElementById('seq-results-container');

    const seq = inputEl.value.trim();
    const mode = modeEl.value;
    const length = lengthEl.value;

    if (!seq) {
        resultsContainer.innerHTML = '<div class="seq-results-placeholder">Please enter a sequence.</div>';
        return;
    }

    resultsContainer.innerHTML = '<div style="padding:20px; text-align:center; color:#rgba(255,255,255,0.7);">Searching...</div>';

    try {
        const response = await fetch('/api/tools/sequence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sequence: seq,
                mode: mode,
                length: length,
                dictionary: 'NWL' // Defaulting to NWL for now, could add selector later
            })
        });

        const data = await response.json();

        if (data.error) {
            resultsContainer.innerHTML = `<div style="padding:20px; color:#f43f5e;">Error: ${data.error}</div>`;
            return;
        }

        const words = data.results;
        const count = data.count;

        if (words.length === 0) {
            resultsContainer.innerHTML = '<div class="seq-results-placeholder">No words found.</div>';
            return;
        }

        // Render Results Table
        let html = `
            <div style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.2);">
                Found ${count} words
            </div>
            <div style="flex: 1; overflow-y: auto; padding: 10px;">
                <table class="group-table" style="width: 100%;">
                    <tbody>
        `;

        // Use chunks to avoid blocking if list is huge? For now direct map.
        html += words.map(w => `
            <tr><td style="padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); color: rgba(255,255,255,0.9); font-family: monospace;">${w}</td></tr>
        `).join('');

        html += `
                    </tbody>
                </table>
            </div>
        `;

        resultsContainer.innerHTML = html;

    } catch (err) {
        console.error("Sequence search failed:", err);
        resultsContainer.innerHTML = '<div style="padding:20px; color:#f43f5e;">Search failed.</div>';
    }
}

// --- Manual Tool Logic ---

let manualSolvedWords = [];

function setupManualTool() {
    const dimSelect = document.getElementById('manual-dim');
    const solveBtn = document.getElementById('manual-solve-btn');
    const revealBtn = document.getElementById('manual-reveal-btn');

    if (dimSelect) {
        dimSelect.addEventListener('change', (e) => renderManualGrid(e.target.value));
        // Initial render
        renderManualGrid(dimSelect.value);
    }

    if (solveBtn) {
        solveBtn.addEventListener('click', runManualSolve);
    }

    if (revealBtn) {
        revealBtn.addEventListener('click', revealManualWords);
    }
}

function renderManualGrid(dims) {
    const gridEl = document.getElementById('manual-grid');
    if (!gridEl) return;

    const [rows, cols] = dims.split('x').map(Number);

    gridEl.style.gridTemplateColumns = `repeat(${cols}, 45px)`;
    gridEl.style.gridTemplateRows = `repeat(${rows}, 45px)`;

    gridEl.innerHTML = '';

    // Create inputs
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'manual-cell';
            input.maxLength = 1;
            input.dataset.r = r;
            input.dataset.c = c;

            // Auto-advance logic
            input.addEventListener('input', (e) => {
                const val = e.target.value;
                if (val && val.length === 1) {
                    const next = input.nextElementSibling;
                    if (next) next.focus();
                }
            });

            // Backspace logic
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Backspace' && !input.value) {
                    const prev = input.previousElementSibling;
                    if (prev) prev.focus();
                }
            });

            gridEl.appendChild(input);
        }
    }

    // Reset state
    manualSolvedWords = [];
    const resultsContainer = document.getElementById('manual-results-container');
    const revealBtn = document.getElementById('manual-reveal-btn');
    if (resultsContainer) resultsContainer.style.display = 'none';
    if (revealBtn) revealBtn.style.display = 'none';
}

async function runManualSolve() {
    const gridEl = document.getElementById('manual-grid');
    const dictEl = document.getElementById('manual-dict');
    const solveBtn = document.getElementById('manual-solve-btn');
    const revealBtn = document.getElementById('manual-reveal-btn');
    const resultsContainer = document.getElementById('manual-results-container');
    const dimSelect = document.getElementById('manual-dim');

    if (!gridEl || !dimSelect) return;

    const [rows, cols] = dimSelect.value.split('x').map(Number);
    const cells = gridEl.querySelectorAll('.manual-cell');

    // Build 2D board
    const board = [];
    let cellIdx = 0;
    let missing = false;

    for (let r = 0; r < rows; r++) {
        const row = [];
        for (let c = 0; c < cols; c++) {
            const val = cells[cellIdx++].value.trim().toUpperCase();
            if (!val) {
                missing = true;
            }
            row.push(val);
        }
        board.push(row);
    }

    if (missing) {
        alert("Please fill in all letters first.");
        return;
    }

    solveBtn.innerText = "Solving...";
    solveBtn.disabled = true;

    try {
        const response = await fetch('/api/tools/manual_solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board: board,
                dictionary: dictEl.value
            })
        });

        const data = await response.json();

        if (data.error) {
            alert("Solve failed: " + data.error);
            return;
        }

        manualSolvedWords = data.results;

        // Show reveal button, hide results space initially
        revealBtn.style.display = 'inline-block';
        revealBtn.innerText = "Reveal Words";
        resultsContainer.style.display = 'none';

    } catch (err) {
        console.error("Manual solve failed:", err);
        alert("Server error during solve.");
    } finally {
        solveBtn.innerText = "Solve";
        solveBtn.disabled = false;
    }
}

function revealManualWords() {
    const resultsContainer = document.getElementById('manual-results-container');
    const revealBtn = document.getElementById('manual-reveal-btn');

    if (resultsContainer.style.display === 'flex') {
        resultsContainer.style.display = 'none';
        revealBtn.innerText = "Reveal Words";
        return;
    }

    if (manualSolvedWords.length === 0) {
        resultsContainer.innerHTML = '<div class="seq-results-placeholder">No words found on this board.</div>';
    } else {
        let html = `
            <div style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.2);">
                Found ${manualSolvedWords.length} words
            </div>
            <div style="flex: 1; overflow-y: auto; padding: 10px;">
                <table class="group-table" style="width: 100%;">
                    <tbody>
        `;

        html += manualSolvedWords.map(w => `
            <tr><td style="padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); color: rgba(255,255,255,0.9); font-family: monospace;">${w}</td></tr>
        `).join('');

        html += `
                    </tbody>
                </table>
            </div>
        `;
        resultsContainer.innerHTML = html;
    }

    resultsContainer.style.display = 'flex';
    revealBtn.innerText = "Hide Words";
}

// --- Random Word Tool Logic ---

function setupRandomWordTool() {
    const genBtn = document.getElementById('random-gen-btn');
    if (genBtn) {
        genBtn.addEventListener('click', generateRandomWord);
    }
}

async function generateRandomWord() {
    const lengthEl = document.getElementById('random-length');
    const dictEl = document.getElementById('random-dict');
    const displayEl = document.getElementById('random-word-display');
    const genBtn = document.getElementById('random-gen-btn');

    const length = lengthEl.value;
    const dictionary = dictEl.value;

    genBtn.innerText = "Generating...";
    genBtn.disabled = true;
    displayEl.innerHTML = ''; // Clear while loading

    try {
        const url = `/api/tools/random_word?length=${length}&dictionary=${dictionary}`;
        const response = await fetch(url);
        const data = await response.json();

        if (data.error) {
            displayEl.innerHTML = `<span style="font-size: 1.5rem; color: #f43f5e;">${data.error}</span>`;
            return;
        }

        const word = data.word;

        // Add a class to re-trigger animation
        displayEl.classList.remove('random-word-large');
        void displayEl.offsetWidth; // Trigger reflow
        displayEl.classList.add('random-word-large');

        displayEl.innerText = word;

    } catch (err) {
        console.error("Random word fetch failed:", err);
        displayEl.innerHTML = `<span style="font-size: 1.5rem; color: #f43f5e;">Error loading random word.</span>`;
    } finally {
        genBtn.innerText = "Generate Random Word";
        genBtn.disabled = false;
    }
}

// --- Word of the Day Tool Logic ---

function setupWotdTool() {
    // This is mainly for manual navigation/initialization
}

async function updateWotd() {
    const displayEl = document.getElementById('wotd-display');
    if (!displayEl) return;

    // Only fetch if empty to avoid redundant calls on every toggle
    if (displayEl.innerText.trim() !== '') return;

    displayEl.innerHTML = '<span style="font-size: 1.5rem; opacity: 0.5;">Loading...</span>';

    try {
        const response = await fetch('/api/tools/wotd');
        const data = await response.json();

        if (data.error) {
            displayEl.innerText = 'Error loading word';
            return;
        }

        displayEl.innerText = data.word;
    } catch (err) {
        console.error("WOTD fetch failed:", err);
        displayEl.innerText = 'Offline';
    }
}
