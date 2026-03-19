// ========================================
// MOUSE SELECTION SYSTEM (OPTIMIZED)
// ========================================

let cachedCellRects = [];

// Initialize mouse selection handlers
function initializeMouseSelection() {
    const board = document.getElementById('game-board');
    if (!board) return;

    board.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    console.log('[Mouse] Optimized selection system initialized');
}

// Start selection on mousedown
function handleMouseDown(e) {
    const cell = e.target.closest('.board-cell');
    if (!cell || cell.classList.contains('grayed')) return;

    // Cache cell boundaries to avoid getBoundingClientRect during mousemove (expensive)
    cacheCellRects();

    // Start new selection
    mouseState.isDown = true;
    mouseState.selectedPath = [];
    mouseState.visitedCells = new Set();

    const row = parseInt(cell.dataset.row);
    const col = parseInt(cell.dataset.col);
    let letter = cell.dataset.letter || cell.textContent.trim();

    // Handle Either/Or dual letters (Top/Bottom half)
    if (letter.includes('/')) {
        const rect = cell.getBoundingClientRect();
        const centerY = rect.top + rect.height / 2;
        const [top, bottom] = letter.split('/');
        letter = (e.clientY < centerY) ? top : bottom;
    }

    addToPath(row, col, letter, cell);
}

// Cache all cell positions once at the start of a drag
function cacheCellRects() {
    const cells = document.querySelectorAll('.board-cell');
    cachedCellRects = Array.from(cells).map(cell => {
        const rect = cell.getBoundingClientRect();
        return {
            element: cell,
            centerX: rect.left + rect.width / 2,
            centerY: rect.top + rect.height / 2,
            row: parseInt(cell.dataset.row),
            col: parseInt(cell.dataset.col),
            letter: cell.dataset.letter || cell.textContent.trim(),
            isGrayed: cell.classList.contains('grayed')
        };
    });
}

// Track drag movement
function handleMouseMove(e) {
    if (!mouseState.isDown) return;

    const point = { x: e.clientX, y: e.clientY };
    const cellData = findCellAtPointOptimized(point);

    if (!cellData || cellData.isGrayed) return;

    let { row, col, letter, element } = cellData;
    const cellKey = `${row},${col}`;

    // Handle Either/Or dual letters (Top/Bottom half)
    if (letter.includes('/')) {
        const rect = element.getBoundingClientRect();
        const centerY = rect.top + rect.height / 2;
        const [top, bottom] = letter.split('/');
        letter = (point.y < centerY) ? top : bottom;
    }

    // BACKTRACKING: Check if cell is already in path - if so, truncate
    const existingIndex = mouseState.selectedPath.findIndex(p => p.row === row && p.col === col);
    if (existingIndex !== -1) {
        if (existingIndex === mouseState.selectedPath.length - 1) return; // Already on last cell

        // Remove all cells after this position
        mouseState.selectedPath = mouseState.selectedPath.slice(0, existingIndex + 1);
        mouseState.visitedCells = new Set(mouseState.selectedPath.map(p => `${p.row},${p.col}`));
        refreshPathDisplay();
        return;
    }

    // Skip if already visited
    if (mouseState.visitedCells.has(cellKey)) return;

    // Validate adjacency to last cell
    if (mouseState.selectedPath.length > 0) {
        const last = mouseState.selectedPath[mouseState.selectedPath.length - 1];
        if (!isAdjacent(last.row, last.col, row, col)) return;
    }

    addToPath(row, col, letter, element);
}

// Submit word on mouseup
function handleMouseUp(e) {
    if (!mouseState.isDown) return;

    mouseState.isDown = false;
    cachedCellRects = []; // Clear cache

    // Build word from path
    const word = mouseState.selectedPath.map(cell => cell.letter).join('');

    // Clear internal path tracking so it doesn't linger
    mouseState.selectedPath = [];
    if (mouseState.visitedCells) mouseState.visitedCells.clear();

    // Clear visual feedback
    // Small delay to let user see final path
    setTimeout(clearSelection, 50);

    // Submit word if long enough
    if (word.length >= 3) {
        // Pass the path for specialized scoring (e.g. Bonus Letter)
        if (typeof submitWord === 'function') {
            submitWord(word, mouseState.selectedPath.map(p => [p.row, p.col]));
        } else {
            console.warn('submitWord not found in global scope');
        }
    }
}

// Helper: Add cell to selection path
function addToPath(row, col, letter, cellElement) {
    const cellKey = `${row},${col}`;

    mouseState.selectedPath.push({ row, col, letter });
    mouseState.visitedCells.add(cellKey);

    const isHighlightEnabled = window.userSettings && window.userSettings.highlight_mouse !== false;
    if (isHighlightEnabled) {
        // Optimally highlight just the new cell
        cellElement.classList.add('selected');

        // Update current marker
        document.querySelectorAll('.board-cell.current').forEach(c => c.classList.remove('current'));
        cellElement.classList.add('current');
    }

    // Live update the word input box
    const wordInputEl = document.getElementById('word-input');
    if (wordInputEl) {
        wordInputEl.value = mouseState.selectedPath.map(p => {
            const L = p.letter.includes('/') ? p.letter.split('/')[0] : p.letter;
            return L === 'Q' ? 'QU' : L;
        }).join('');
    }
}

// Helper: Refresh visual display of entire path (used for backtracking)
function refreshPathDisplay() {
    const isHighlightEnabled = window.userSettings && window.userSettings.highlight_mouse !== false;
    if (!isHighlightEnabled) {
        clearSelection();
        return;
    }

    const pathSet = mouseState.visitedCells;
    const lastCell = mouseState.selectedPath[mouseState.selectedPath.length - 1];

    document.querySelectorAll('.board-cell').forEach(cell => {
        const row = cell.dataset.row;
        const col = cell.dataset.col;
        const key = `${row},${col}`;

        if (pathSet.has(key)) {
            cell.classList.add('selected');
            if (lastCell && lastCell.row == row && lastCell.col == col) {
                cell.classList.add('current');
            } else {
                cell.classList.remove('current');
            }
        } else {
            cell.classList.remove('selected', 'current');
        }
    });

    // Live update the word input box during backtracking
    const wordInputEl = document.getElementById('word-input');
    if (wordInputEl) {
        wordInputEl.value = mouseState.selectedPath.map(p => {
            const L = p.letter.includes('/') ? p.letter.split('/')[0] : p.letter;
            return L === 'Q' ? 'QU' : L;
        }).join('');
    }
}

// Helper: Optimized find cell at point using cached rects
function findCellAtPointOptimized(point) {
    const radiusSq = 35 * 35; // Use squared distance for faster comparison

    for (const cell of cachedCellRects) {
        const dx = point.x - cell.centerX;
        const dy = point.y - cell.centerY;

        // Fast circular coarse check
        const distSq = dx * dx + dy * dy;
        if (distSq > radiusSq) continue;

        // Precise octagonal check
        const absDx = Math.abs(dx);
        const absDy = Math.abs(dy);
        const maxDist = Math.max(absDx, absDy);
        const minDist = Math.min(absDx, absDy);

        if (maxDist + 0.414 * minDist <= 35) {
            return cell;
        }
    }

    return null;
}

// Helper: Check if two cells are adjacent (Boggle rules - 8 directions)
function isAdjacent(row1, col1, row2, col2) {
    const dRow = Math.abs(row1 - row2);
    const dCol = Math.abs(col1 - col2);
    return dRow <= 1 && dCol <= 1 && (dRow !== 0 || dCol !== 0);
}

// Helper: Clear visual selection
function clearSelection() {
    document.querySelectorAll('.board-cell.selected, .board-cell.current')
        .forEach(cell => {
            cell.classList.remove('selected', 'current');
        });

    // Clear live input display
    const wordInputEl = document.getElementById('word-input');
    if (wordInputEl) wordInputEl.value = '';
}

// Initialize on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    initializeMouseSelection();
});
