import sys
import os
import time

# Add current path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator

def test_max_uniqueness():
    gen = BoardGenerator()
    
    # 4x4 - Hard (60%+ uniqueness) - 3LM - NWL
    dimensions = '4x4'
    bonus_word = 'MORPHEME'
    word_count_range = (50, 100)
    dictionary = 'NWL'
    min_word_length = 3
    difficulty = 'Hard'
    
    max_ratio = 0
    total_attempts = 1000
    
    # Bypass generate_board logic to just roll and solve
    from board_generator import LETTER_FREQ_USER
    
    results = []
    for i in range(total_attempts):
        board = gen._create_normal_board(4, 4, LETTER_FREQ_USER)
        # Embed bonus word to be realistic
        gen._embed_bonus_word(board, bonus_word)
        
        words_dict = gen._solve_board(board, dictionary, (0, 99999), 3)
        if words_dict:
            found = list(words_dict.keys())
            unique_set = gen._get_difficulty_set(dictionary)
            count_total = len(found)
            count_unique = sum(1 for w in found if w.upper() in unique_set)
            ratio = count_unique / count_total if count_total > 0 else 0
            
            if ratio > max_ratio:
                max_ratio = ratio
                print(f"Attempt {i}: New Max {ratio:.2%} ({count_total} words)")
            
            if ratio >= 0.60:
                print(f"Goal met at attempt {i}!")
                break
                
    print(f"Finished {total_attempts} attempts. Max Uniqueness: {max_ratio:.2%}")

if __name__ == "__main__":
    test_max_uniqueness()
