
print("[Scoring] SCORING.PY LOADED - VERSION 8:44 PM")
import json
import logging

# Initialize logger for scoring-related diagnostic messages
score_logger = logging.getLogger("scoring")
score_logger.setLevel(logging.INFO)
# Ensure at least a NullHandler exists to prevent "No handlers found" warnings
if not score_logger.handlers:
    score_logger.addHandler(logging.NullHandler())

__all__ = ['calculate_word_score', 'score_logger', 'LETTER_VALUES']


LETTER_VALUES = {
    'A': 2, 'B': 4, 'C': 4, 'D': 3, 'E': 1, 'F': 5, 'G': 3, 'H': 5, 'I': 2, 'J': 10,
    'K': 6, 'L': 3, 'M': 4, 'N': 2, 'O': 2, 'P': 4, 'Q': 10, 'R': 2, 'S': 2, 'T': 2,
    'U': 4, 'V': 5, 'W': 5, 'X': 10, 'Y': 5, 'Z': 10
}

def calculate_word_score(word, bonus_word=None, board_format='Normal', path=None, bonus_cell=None, board=None, return_details=False, **kwargs):
    """
    Calculate points for a word. (OPTIMIZED for high-speed batch processing)
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
        for char in word.upper():
            score += LETTER_VALUES.get(char, 1)
    else:
        # Standard Boggle Base Scoring
        if length <= 2: score = 0
        elif length <= 4: score = 1
        elif length == 5: score = 2
        elif length == 6: score = 3
        elif length == 7: score = 5
        elif length >= 8: score = 11

    base_score = score 

    # Hidden Bonus Word (+Length points)
    bonus_word_score = 0
    if bonus_word and word.upper() == bonus_word.upper():
        bonus_word_score = length
        score += bonus_word_score
        
    used_bonus = False
    
    # 3. Board Pathfinding (Check if word uses the bonus tile)
    is_spec_bonus_fmt = ('bonus letter' in fmt_lower or 'either' in fmt_lower)
    should_skip_pathfinding = (not path and not is_spec_bonus_fmt and not bonus_cell)

    if board and len(board) > 0 and not should_skip_pathfinding:
        is_3d = (len(board) == 6 and isinstance(board[0], list) and isinstance(board[0][0], list))
        
        # A. Use explicit path if provided (Check it's a list for optimized solver)
        if path and isinstance(path, (list, tuple)):
            # PRE-CALCULATE SPECIAL CELLS SET:
            # We check for bonus_cell AND any Either/Or tiles (identified by '/')
            special_coords = set()
            if bonus_cell:
                 if isinstance(bonus_cell, dict):
                     special_coords.add((int(bonus_cell.get('f', -1)), int(bonus_cell.get('r', 0)), int(bonus_cell.get('c', 0))))
                 elif isinstance(bonus_cell, (list, tuple)):
                     if len(bonus_cell) == 3: special_coords.add((int(bonus_cell[0]), int(bonus_cell[1]), int(bonus_cell[2])))
                     else: special_coords.add((-1, int(bonus_cell[0]), int(bonus_cell[1])))
            
            # FAST PATH ITERATION
            for node in path:
                if isinstance(node, dict):
                    nf, nx, ny = int(node.get('f', -1)), int(node.get('r', -1)), int(node.get('c', -1))
                elif isinstance(node, (list, tuple)):
                    if len(node) == 3: nf, nx, ny = int(node[0]), int(node[1]), int(node[2])
                    else: nf, nx, ny = -1, int(node[0]), int(node[1])
                
                # O(1) Membership Check for Bonus Letter
                if (nf, nx, ny) in special_coords:
                    used_bonus = True; break
                
                # Check for Either/Or tile Hit (Identified by '/' in cell value)
                cell_val = str(board[nf][nx][ny] if is_3d else board[nx][ny])
                if '/' in cell_val:
                    used_bonus = True; break
        
        elif is_spec_bonus_fmt:
            # Fallback: Recalculate path manually via DFS (Only if no path was provided for special formats)
            # This is essentially unchanged but cleaned up to avoid repeated calls
            word_target = word.upper()
            if not (is_3d and len(word_target) > 12):
                bx, by, bf = -1, -1, -1
                if bonus_cell:
                    if isinstance(bonus_cell, dict):
                        bx, by, bf = int(bonus_cell.get('r', 0)), int(bonus_cell.get('c', 0)), int(bonus_cell.get('f', -1))
                    elif isinstance(bonus_cell, (list, tuple)):
                        if len(bonus_cell) == 3: bf, bx, by = int(bonus_cell[0]), int(bonus_cell[1]), int(bonus_cell[2])
                        else: bx, by = int(bonus_cell[0]), int(bonus_cell[1])
                
                def get_neighbors(f, r, c):
                    res = []
                    if not is_3d:
                        rows, cols = len(board), len(board[0])
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < rows and 0 <= nc < cols: res.append((-1, nr, nc))
                    else:
                        # Full 3D surface neighbors... (unchanged logic)
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = r+dr, c+dc
                                if 0 <= nr < 3 and 0 <= nc < 3: res.append((f, nr, nc))
                        # Inter-face Wrap Logic (Shortened for brevity but keeping logic)
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
                        res = [(nf, nr, nc) for nf, nr, nc in res if 0 <= nf < 6 and 0 <= nr < 3 and 0 <= nc < 3]
                    return res

                def find_through(f, r, c, index, has_hit_bonus, visited):
                    cell_val = str(board[f][r][c] if is_3d else board[r][c]).upper()
                    now_hit = has_hit_bonus or (f == bf and r == bx and c == by) or ('/' in cell_val)
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
                        for r in range(3):
                            for c in range(3):
                                if find_through(f, r, c, 0, False, {(f, r, c)}): used_bonus = True; break
                            if used_bonus: break
                        if used_bonus: break
                else:
                    for r in range(len(board)):
                        for c in range(len(board[0])):
                            if find_through(-1, r, c, 0, False, {(-1, r, c)}): used_bonus = True; break
                        if used_bonus: break

    bonus_letter_points = 0
    either_or_points = 0
    if used_bonus and is_spec_bonus_fmt and 'checkerboard' not in fmt_lower:
        if 'either' in fmt_lower: either_or_points = 3
        else: bonus_letter_points = 3
        score += 3
            
    if return_details:
        return {
            'total': score, 'base': base_score,
            'bonus_word_points': bonus_word_score,
            'bonus_letter_points': bonus_letter_points,
            'either_or_points': either_or_points
        }
    return score
