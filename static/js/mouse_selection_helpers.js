// Helper: Refresh visual display of entire path
function refreshPathDisplay() {
    const allCells = document.querySelectorAll('.board-cell');

    // Clear all current markers
    allCells.forEach(cell => cell.classList.remove('current'));

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

    console.log('[Mouse] Path after backtrack:', mouseState.selectedPath.map(c => c.letter).join(''));
}
