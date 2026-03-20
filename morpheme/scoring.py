
# Letter values for "Valued Letters" format
LETTER_VALUES = {
    'A': 2, 'B': 4, 'C': 4, 'D': 3, 'E': 1, 'F': 5, 'G': 3, 'H': 5, 'I': 2, 'J': 10,
    'K': 6, 'L': 3, 'M': 4, 'N': 2, 'O': 2, 'P': 4, 'Q': 10, 'R': 2, 'S': 2, 'T': 2,
    'U': 4, 'V': 5, 'W': 5, 'X': 9, 'Y': 5, 'Z': 9
}

def calculate_word_score(word, bonus_word=None, board_format='Normal', path=None, bonus_cell=None, return_details=False, **kwargs):
    """
    Calculate points for a word.
    
    Args:
        word: The word string.
        bonus_word: The round's hidden bonus word.
        board_format: 'Normal', 'Valued Letters', 'Bonus Letter', etc.
        path: List of (r,c) coordinates for the word on the board.
        bonus_cell: (r,c) coordinate of the highlighted bonus letter.
        return_details: If True, returns a dict with breakdown.
    """
    if not word:
        if return_details:
            return {'total': 0, 'base': 0, 'bonus_word_points': 0, 'bonus_letter_points': 0}
        return 0
        
    length = len(word)
    score = 0
    
    if board_format == 'Valued Letters':
        # Sum of individual letter values
        for char in word.upper():
            score += LETTER_VALUES.get(char, 1)
    else:
        # Standard Boggle scoring (with Private Match exception for 5-letter words)
        if length <= 2:
            score = 0
        elif length <= 4:
            score = 1
        elif length == 5:
            # USER REQUEST: 5-letter words in Private Matches award 5 points
            score = 5 if kwargs.get('is_private') else 2
        elif length == 6:
            score = 3
        elif length == 7:
            score = 5
        else:  # 8+ letters
            score = 11
            
    base_score = score
    bonus_word_score = 0
    bonus_letter_score = 0

    # Hidden Bonus Word (Bonus length points)
    if bonus_word and word.upper() == bonus_word.upper():
        bonus_word_score = length
        score += bonus_word_score
        
    # Bonus Letter OR Either/Or (+3 points if path contains bonus_cell)
    # Reusing bonus_letter_points logic if Either/Or is active
    is_eo = board_format == 'Either/Or'
    if (board_format == 'Bonus Letter' or is_eo) and bonus_cell and path:
        # Robust coordinate comparison: check if any cell in path matches bonus_cell
        bx, by = int(bonus_cell[0]), int(bonus_cell[1])
        for cx, cy in path:
            if int(cx) == bx and int(cy) == by:
                bonus_letter_score = 3
                score += bonus_letter_score
                break
            
    if return_details:
        return {
            'total': score,
            'base': base_score,
            'bonus_word_points': bonus_word_score,
            'bonus_letter_points': bonus_letter_score
        }
    return score
