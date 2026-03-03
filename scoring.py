def calculate_word_score(word, bonus_word=None):
    """Calculate points for a word using standard Boggle scoring"""
    if not word:
        return 0
        
    length = len(word)
    
    # Base score by word length
    if length <= 2:
        base_score = 0
    elif length <= 4:
        base_score = 1
    elif length == 5:
        base_score = 2
    elif length == 6:
        base_score = 3
    elif length == 7:
        base_score = 5
    else:  # 8+ letters
        base_score = 11
    
    # Bonus word gets extra points
    if bonus_word and word.upper() == bonus_word.upper():
        base_score += length  # Extra points equal to word length
        
    return base_score
