// ========================================
// MOUSE SELECTION SYSTEM
// ========================================

// Initialize mouse selection handlers
function initializeMouseSelection() {
    const board = document.getElementById('game-board');
    if (!board) return;

    board.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    console.log('[Mouse] Selection system initialized');
}

// Start selection on mousedown
function handleMouseDown(e) {
    const cell = e.target.closest('.board-cell');
    if (!cell || cell.classList.contains('grayed')) return;

    // Start new selection
    mouseState.isDown = true;
    mouseState.selectedPath = [];
    mouseState.visitedCells = new Set();

    const row = parseInt(cell.dataset.row);
    const col = parseInt(cell.dataset.col);
    const letter = cell.textContent.trim();

    addToPath(row, col, letter, cell);
    console.log('[Mouse] Started selection:', letter);
}

// Track drag movement
function handleMouseMove(e) {
    if (!mouseState.isDown) return;

    const point = { x: e.clientX, y: e.clientY };
    const cell = findCellAtPoint(point);

    if (!cell || cell.classList.contains('grayed')) return;

    const row = parseInt(cell.dataset.row);
    const col = parseInt(cell.dataset.col);
    const cellKey = `${row},${col}`;

    // BACKTRACKING: Check if cell is already in path - if so, truncate
    const existingIndex = mouseState.selectedPath.findIndex(p => p.row === row && p.col === col);
    if (existingIndex !== -1) {
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

    const letter = cell.textContent.trim();
    addToPath(row, col, letter, cell);
}

// Submit word on mouseup
function handleMouseUp(e) {
    if (!mouseState.isDown) return;

    mouseState.isDown = false;

    // Build word from path
    const word = mouseState.selectedPath.map(cell => cell.letter).join('');

    console.log('[Mouse] Selection ended:', word);

    // Clear visual feedback
    clearSelection();

    // Submit word if long enough
    if (word.length >= 3) {
        submitWord(word);
    }
}

// Helper: Add cell to selection path
function addToPath(row, col, letter, cellElement) {
    const cellKey = `${row},${col}`;

    mouseState.selectedPath.push({ row, col, letter });
    mouseState.visitedCells.add(cellKey);

    // Ensure ALL cells in the path have the 'selected' class
    const allCells = document.querySelectorAll('.board-cell');
    mouseState.selectedPath.forEach(pathCell => {
        allCells.forEach(cell => {
            if (cell.dataset.row == pathCell.row && cell.dataset.col == pathCell.col) {
                cell.classList.add('selected');
            }
        });
    });

    // Remove 'current' from all cells, then mark only the newest one as current
    document.querySelectorAll('.board-cell.current').forEach(c => {
        c.classList.remove('current');
    });
    cellElement.classList.add('current');

    console.log('[Mouse] Path:', mouseState.selectedPath.map(c => c.letter).join(''));
}

// Helper: Refresh visual display of entire path (used for backtracking)
function refreshPathDisplay() {
    const allCells = document.querySelectorAll('.board-cell');

    // Clear ALL selected and current markers from all cells
    allCells.forEach(cell => cell.classList.remove('selected', 'current'));

    // Reapply selected class to all cells in path
    mouseState.selectedPath.forEach((pathCell, index) => {
        allCells.forEach(cell => {
            if (cell.dataset.row == pathCell.row && cell.dataset.col == pathCell.col) {
                cell.classList.add('selected');
                // Mark the last one as current
                if (index === mouseState.selectedPath.length - 1) {
                    cell.classList.add('current');
                }
            }
        });
    });

    console.log('[Mouse] Path refreshed:', mouseState.selectedPath.map(c => c.letter).join(''));
}

// Helper: Find cell at mouse position using octagonal hit detection
function findCellAtPoint(point) {
    const cells = document.querySelectorAll('.board-cell');

    for (const cell of cells) {
        if (isPointInOctagon(point, cell)) {
            return cell;
        }
    }

    return null;
}

// Helper: Octagonal hit test
function isPointInOctagon(point, cell) {
    const rect = cell.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    // Octagon radius - sized to bridge 4px gaps without overlapping adjacent cells
    // Cells are 60x60px with 4px gaps (64px center-to-center)
    // Using 35px radius = 70px diameter (extends 5px beyond cell edge, bridges gap)
    const radius = 35;

    // Distance from center
    const dx = point.x - centerX;
    const dy = point.y - centerY;
    const distance = Math.sqrt(dx * dx + dy * dy);

    // Circular boundary check
    if (distance > radius) return false;

    // Octagonal bounds - proper 8-sided shape
    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    // Octagon constraint: diamond shape with clipped corners
    // For a true octagon: max(|dx|, |dy|) + 0.414*(min(|dx|, |dy|)) <= radius
    const maxDist = Math.max(absDx, absDy);
    const minDist = Math.min(absDx, absDy);

    return maxDist + 0.414 * minDist <= radius;
}

// Helper: Check if two cells are adjacent (Boggle rules - 8 directions)
function isAdjacent(row1, col1, row2, col2) {
    const dRow = Math.abs(row1 - row2);
    const dCol = Math.abs(col1 - col2);

    // Adjacent if within 1 step in any direction (but not same cell)
    return dRow <= 1 && dCol <= 1 && (dRow !== 0 || dCol !== 0);
}

// Helper: Clear visual selection
function clearSelection() {
    document.querySelectorAll('.board-cell.selected, .board-cell.current')
        .forEach(cell => {
            cell.classList.remove('selected', 'current');
        });
}

// Initialize on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    initializeMouseSelection();
});
