
import json
import logging

# Setup a dedicated scoring logger
score_logger = logging.getLogger('scoring_debug')
score_logger.setLevel(logging.DEBUG)
if not score_logger.handlers:
    fh = logging.FileHandler('/tmp/scoring.log')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    score_logger.addHandler(fh)

LETTER_VALUES = {
    'A': 2, 'B': 4, 'C': 4, 'D': 3, 'E': 1, 'F': 5, 'G': 3, 'H': 5, 'I': 2, 'J': 10,
    'K': 6, 'L': 3, 'M': 4, 'N': 2, 'O': 2, 'P': 4, 'Q': 10, 'R': 2, 'S': 2, 'T': 2,
    'U': 4, 'V': 5, 'W': 5, 'X': 9, 'Y': 5, 'Z': 9
}

def calculate_word_score(word, bonus_word=None, board_format='Normal', path=None, bonus_cell=None, board=None, return_details=False, **kwargs):
    """
    Calculate points for a word.
    
    Args:
        word: The word string.
        bonus_word: The round's hidden bonus word.
        board_format: 'Normal', 'Valued Letters', 'Bonus Letter', 'Either/Or', etc.
        path: List of (r,c) coordinates for the word on the board.
        bonus_cell: (r,c) coordinate of the highlighted bonus letter.
        board: The current game board (optional, used for pathfinding if path is missing).
        return_details: If True, returns a dict with breakdown.
    """
    if not word:
        if return_details:
            return {'total': 0, 'base': 0, 'bonus_word_points': 0, 'bonus_letter_points': 0}
        return 0
        
    length = len(word)
    score = 0
    
    # 1. Base Word Scoring (Boggle Standard vs Valued Letters)
    fmt_lower = str(board_format).lower() if board_format else ''
    is_valued_format = ('valued' in fmt_lower or 'value' in fmt_lower)
    
    if is_valued_format:
        # Valued Letters Format: Scoring is ONLY the sum of original letter values (No length points)
        for char in word.upper():
            score += LETTER_VALUES.get(char, 1)
        score_logger.debug(f"[Scorer] {word} - Valued Format Detected. Letter Sum: {score}")
    else:
        # Standard Boggle Base Scoring (Based on Length)
        if length <= 2: score = 0
        elif length <= 4:
            score = 1
        elif length == 5:
            score = 2
        elif length == 6: score = 3
        elif length == 7: score = 5
        elif length >= 8: score = 11

    score_logger.debug(f"[Scorer] {word} - Base Score: {score} (is_private={kwargs.get('is_private')}, fmt={board_format})")

    base_score = score # Capture score before bonuses

    # Hidden Bonus Word (+Length points)
    bonus_word_score = 0
    if bonus_word and word.upper() == bonus_word.upper():
        bonus_word_score = length
        score += bonus_word_score
        
    # 2. Bonus Point Logic (+3 points if path hits historical or active bonus_cell)
    # USER REQUEST: "Simply give bonus points for using the Either/Or tile or the Bonus Letter tile"
    used_bonus = False
    
    # 3. Board Pathfinding (Check if word uses the bonus tile)
    if board and len(board) > 0:
        is_3d = (len(board) == 6 and isinstance(board[0], list) and isinstance(board[0][0], list))
        
        # Coordinate extraction
        bx, by, bf = -1, -1, -1
        if bonus_cell:
            if isinstance(bonus_cell, dict):
                bx, by, bf = int(bonus_cell.get('r', 0)), int(bonus_cell.get('c', 0)), int(bonus_cell.get('f', -1))
            elif isinstance(bonus_cell, (list, tuple)):
                if len(bonus_cell) == 3:
                    bf, bx, by = int(bonus_cell[0]), int(bonus_cell[1]), int(bonus_cell[2])
                elif len(bonus_cell) == 2:
                    bx, by = int(bonus_cell[0]), int(bonus_cell[1])
                    bf = -1

        # A. Use explicit path if provided
        if path:
            for node in path:
                nx, ny, nf = -1, -1, -1
                if isinstance(node, dict):
                    nx, ny, nf = int(node.get('r', node.get('row', -1))), int(node.get('c', node.get('col', -1))), int(node.get('f', node.get('face', -1)))
                elif isinstance(node, (list, tuple)):
                    if len(node) == 3:
                        nf, nx, ny = int(node[0]), int(node[1]), int(node[2])
                    elif len(node) == 2:
                        nf, nx, ny = -1, int(node[0]), int(node[1])
                
                if nf == bf and nx == bx and ny == by:
                    used_bonus = True
                    break
        
        # B. Fallback: Recalculate path manually via DFS
        if not used_bonus:
            word_target = word.upper()
            
            def get_neighbors(f, r, c):
                if not is_3d:
                    res = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < len(board) and 0 <= nc < len(board[0]):
                                res.append((-1, nr, nc))
                    return res
                else:
                    # Comprehensive 3D Surface Neighbors (Ported from BoardGenerator)
                    res = []
                    # Intra-face
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < 3 and 0 <= nc < 3: res.append((f, nr, nc))
                    # Inter-face Wrap Logic
                    if f == 0:
                        if r == 0: res.extend([(4, 2, c), (4, 2, c-1), (4, 2, c+1)])
                        if r == 2: res.extend([(5, 0, c), (5, 0, c-1), (5, 0, c+1)])
                        if c == 0: res.extend([(2, r, 2), (2, r-1, 2), (2, r+1, 2)])
                        if c == 2: res.extend([(3, r, 0), (3, r-1, 0), (3, r+1, 0)])
                    elif f == 1:
                        if r == 0: res.extend([(4, 0, 2-c), (4, 0, 2-(c-1)), (4, 0, 2-(c+1))])
                        if r == 2: res.extend([(5, 2, 2-c), (5, 2, 2-(c-1)), (5, 2, 2-(c+1))])
                        if c == 0: res.extend([(3, r, 2), (3, r-1, 2), (3, r+1, 2)])
                        if c == 2: res.extend([(2, r, 0), (2, r-1, 0), (2, r+1, 0)])
                    elif f == 2:
                        if r == 0: res.extend([(4, c, 0), (4, c-1, 0), (4, c+1, 0)])
                        if r == 2: res.extend([(5, 2-c, 0), (5, 2-(c-1), 0), (5, 2-(c+1), 0)])
                        if c == 0: res.extend([(1, r, 2), (1, r-1, 2), (1, r+1, 2)])
                        if c == 2: res.extend([(0, r, 0), (0, r-1, 0), (0, r+1, 0)])
                    elif f == 3:
                        if r == 0: res.extend([(4, 2-c, 2), (4, 2-(c-1), 2), (4, 2-(c+1), 2)])
                        if r == 2: res.extend([(5, c, 2), (5, c-1, 2), (5, c+1, 2)])
                        if c == 0: res.extend([(0, r, 2), (0, r-1, 2), (0, r+1, 2)])
                        if c == 2: res.extend([(1, r, 0), (1, r-1, 0), (1, r+1, 0)])
                    elif f == 4:
                        if r == 0: res.extend([(1, 0, 2-c), (1, 0, 2-(c-1)), (1, 0, 2-(c+1))])
                        if r == 2: res.extend([(0, 0, c), (0, 0, c-1), (0, 0, c+1)])
                        if c == 0: res.extend([(2, 0, r), (2, 0, r-1), (2, 0, r+1)])
                        if c == 2: res.extend([(3, 0, 2-r), (3, 0, 2-(r-1)), (3, 0, 2-(r+1))])
                    elif f == 5:
                        if r == 0: res.extend([(0, 2, c), (0, 2, c-1), (0, 2, c+1)])
                        if r == 2: res.extend([(1, 2, 2-c), (1, 2, 2-(c-1)), (1, 2, 2-(c+1))])
                        if c == 0: res.extend([(2, 2, 2-r), (2, 2, 2-(r-1)), (2, 2, 2-(r+1))])
                        if c == 2: res.extend([(3, 2, r), (3, 2, r-1), (3, 2, r+1)])
                    return [(nf, nr, nc) for nf, nr, nc in res if 0 <= nf < 6 and 0 <= nr < 3 and 0 <= nc < 3]

            def find_through(f, r, c, index, has_hit_bonus, visited):
                now_hit = has_hit_bonus or (f == bf and r == bx and c == by)
                cell_val = str(board[f][r][c] if is_3d else board[r][c]).upper()
                letters = cell_val.split('/') if '/' in cell_val else [cell_val]
                
                for char in letters:
                    match_len = 0
                    if char == 'Q' and word_target.startswith('QU', index): match_len = 2
                    elif word_target.startswith(char, index): match_len = len(char)
                    
                    if match_len > 0:
                        if index + match_len >= len(word_target):
                            if now_hit: return True
                            continue
                        for nf, nr, nc in get_neighbors(f, r, c):
                            if (nf, nr, nc) not in visited:
                                if find_through(nf, nr, nc, index + match_len, now_hit, visited | {(nf, nr, nc)}):
                                    return True
                return False

            if is_3d:
                for f in range(6):
                    if used_bonus: break
                    for r in range(3):
                        if used_bonus: break
                        for c in range(3):
                            if find_through(f, r, c, 0, False, {(f, r, c)}):
                                used_bonus = True; break
            else:
                for r in range(len(board)):
                    if used_bonus: break
                    for c in range(len(board[0])):
                        if find_through(-1, r, c, 0, False, {(-1, r, c)}):
                            used_bonus = True; break

    # Final tally
    bonus_letter_score = 0
    # USER REQUEST: No bonuses for Checkerboard matches
    if used_bonus and 'checkerboard' not in fmt_lower:
        bonus_letter_score = 3
        score += bonus_letter_score
    
    score_logger.debug(f"[Scorer] {word} - Final Total: {score} (Bonus: {bonus_letter_score}, used_bonus={used_bonus}, fmt_lower={fmt_lower})")
            
    if return_details:
        res = {
            'total': score,
            'base': base_score,
            'bonus_word_points': bonus_word_score,
            'bonus_letter_points': bonus_letter_score
        }
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/scoring_debug.log', 'a') as f:
            f.write(f"[Scoring] Word: {word}, Format: {board_format}, BonusCell: {bonus_cell}, UsedBonus: {'used_bonus' in locals() and used_bonus}, Total: {score}\n")
        return res
        
    return score
