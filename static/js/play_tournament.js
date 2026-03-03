
// Globals expected by mouse_selection.js
let mouseState = {
    isDown: false,
    selectedPath: [],
    visitedCells: new Set()
};

// Ensure settings exist for mouse_selection.js
window.userSettings = {
    highlight_mouse: true
};

let gameState = {
    board: [],
    foundWords: [], // List of {word, points}
    score: 0,
    timeLimit: 0,
    startTime: 0,
    timerInterval: null
};

document.addEventListener('DOMContentLoaded', () => {
    initGame();

    // Bind Enter key
    document.getElementById('word-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            submitWord(e.target.value.trim());
            e.target.value = '';
        }
    });

    document.getElementById('submit-btn').addEventListener('click', () => {
        const input = document.getElementById('word-input');
        submitWord(input.value.trim());
        input.value = '';
    });

    document.getElementById('return-btn').addEventListener('click', () => {
        window.location.href = '/tournaments';
    });
});

async function initGame() {
    try {
        const response = await fetch('/api/tournament/play');
        if (!response.ok) {
            alert("Could not load game. Redirecting.");
            window.location.href = '/tournaments';
            return;
        }

        const data = await response.json();
        gameState.board = data.board;
        gameState.timeLimit = data.parameters.time_limit || 180;

        renderBoard(gameState.board);
        startTimer();

        // Initialize mouse selection? Handled by mouse_selection.js automatically on DOMContentLoaded?
        // mouse_selection.js handles initialization automatically on DOMContentLoaded
        // No manual call needed here.

    } catch (e) {
        console.error("Init error:", e);
    }
}

function renderBoard(board) {
    const boardEl = document.getElementById('game-board');
    boardEl.innerHTML = '';

    const rows = board.length;
    const cols = board[0].length;

    boardEl.style.gridTemplateColumns = `repeat(${cols}, 60px)`;
    boardEl.style.gridTemplateRows = `repeat(${rows}, 60px)`;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const letter = board[r][c];
            const cell = document.createElement('div');
            cell.className = 'board-cell';
            cell.textContent = letter === 'Q' ? 'QU' : letter;
            cell.dataset.row = r;
            cell.dataset.col = c;
            cell.dataset.letter = letter;

            // Replicate structure expected by mouse_selection info?
            // "board-cell" class is key.
            boardEl.appendChild(cell);
        }
    }
}

function startTimer() {
    gameState.startTime = Date.now();
    const end = gameState.startTime + (gameState.timeLimit * 1000);

    gameState.timerInterval = setInterval(() => {
        const now = Date.now();
        const left = end - now;

        if (left <= 0) {
            endGame();
            return;
        }

        const m = Math.floor(left / 60000);
        const s = Math.floor((left % 60000) / 1000);
        const str = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;

        const timerEl = document.getElementById('game-timer');
        timerEl.textContent = str;

        if (left < 10000) timerEl.classList.add('timer-warning');

    }, 100);
}

// Global submitWord (called by mouse_selection.js)
async function submitWord(word) {
    if (!word) return;
    word = word.toUpperCase();

    // 1. Check if already found
    if (gameState.foundWords.some(w => w.word === word)) {
        flashMessage("Already Found");
        return;
    }

    // 2. Client-side Board Validation
    if (!findWordPathOnBoard(word, gameState.board)) {
        flashMessage("Not on Board");
        return;
    }

    // 3. Server Validation (Dictionary)
    try {
        const res = await fetch('/api/tournament/validate_word', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word })
        });
        const data = await res.json();

        if (data.valid) {
            gameState.foundWords.push({ word: word, points: data.points });
            gameState.score += data.points;
            updateScoreUI();
            addFoundWordUI(word, data.points);
            flashMessage(`+${data.points}`, 'success');
        } else {
            flashMessage("Invalid Word");
        }

    } catch (e) {
        console.error("Validation error:", e);
    }

    // Clear input
    document.getElementById('word-input').value = '';

    // Clear board highlights (mouse selection clears natively via mouseup)
    // But typing highlights might linger if we implemented them.
}

function updateScoreUI() {
    document.getElementById('game-score').textContent = gameState.score + " pts";
}

function addFoundWordUI(word, points) {
    const panel = document.getElementById('found-words-list');
    // Remove initial placeholder
    if (panel.children.length === 1 && panel.children[0].innerText.includes('Finding')) {
        panel.innerHTML = '';
    }

    const div = document.createElement('div');
    div.className = 'found-word-item';
    div.innerHTML = `<span>${word}</span><span>${points}</span>`;
    panel.prepend(div); // Newest top
}

function flashMessage(msg, type = 'error') {
    // Simple toast or overlay?
    // Let's reuse input placeholder or create a toast
    const input = document.getElementById('word-input');
    const old = input.placeholder;
    input.value = '';
    input.placeholder = msg;
    input.style.borderColor = type === 'error' ? 'red' : 'lime';
    setTimeout(() => {
        input.placeholder = "TYPE OR DRAG";
        input.style.borderColor = 'rgba(var(--text-primary-rgb),0.2)';
    }, 1500);
}

async function endGame() {
    clearInterval(gameState.timerInterval);

    document.getElementById('game-board').style.opacity = '0.5';
    document.getElementById('game-over-overlay').classList.add('game-over-visible');
    document.getElementById('final-score').textContent = gameState.score;

    // Submit Final Score
    try {
        await fetch('/api/tournament/submit_score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                score: gameState.score,
                words: gameState.foundWords // Send full objects
            })
        });
        console.log("Score submitted!");
    } catch (e) {
        console.error("Submission error:", e);
        alert("Error submitting score! Please check connection.");
    }
}


// --- Helper: DFS Path Finding (Reuse from play.js) ---
function findWordPathOnBoard(word, board) {
    if (!word || !board) return null;
    const upperWord = word.toUpperCase();
    const rows = board.length;
    const cols = board[0].length;

    function dfs(r, c, index, currentPath, visited) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) return null;
        if (visited.has(`${r},${c}`)) return null;

        const cellChar = board[r][c].toUpperCase();
        let matchLength = 0;

        if (cellChar === 'Q') {
            if (upperWord.substring(index, index + 2) === 'QU') {
                matchLength = 2;
            } else if (upperWord[index] === 'Q') {
                matchLength = 1;
            } else {
                return null;
            }
        } else {
            if (upperWord[index] === cellChar) {
                matchLength = 1;
            } else {
                return null;
            }
        }

        const newVisited = new Set(visited);
        newVisited.add(`${r},${c}`);
        const newPath = [...currentPath, { r, c }];

        const nextIndex = index + matchLength;
        if (nextIndex >= upperWord.length) return newPath;

        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const result = dfs(r + dr, c + dc, nextIndex, newPath, newVisited);
                if (result) return result;
            }
        }
        return null;
    }

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const path = dfs(r, c, 0, [], new Set());
            if (path) return path;
        }
    }
    return null;
}
