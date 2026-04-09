import sys
import os
import time

# Add current path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator

def debug():
    gen = BoardGenerator()
    
    # Check if sets are loaded
    unique_set = gen._get_difficulty_set('NWL')
    print(f"Unique Set (NWL) size: {len(unique_set)}")
    if unique_set:
        print(f"Sample words: {list(unique_set)[:10]}")
        
    # Check word validator
    from word_validator import word_validator
    print(f"WordValidator UniqueNWL size: {len(word_validator.unique_nwl_words)}")
    
    # Generate a small board
    board, words, bonus_c, fmt, dict_full, ratio = gen.generate_board(
        '4x4', 'MORPHEME', (50, 100), 'NWL', 'Normal', 3, 'Medium'
    )
    
    print(f"Final Board Ratio: {ratio:.1%}")
    print(f"Total Words: {len(words)}")
    unique_found = [w for w in words if w.upper() in unique_set]
    print(f"Unique count (Manual): {len(unique_found)}")
    print(f"Sample Unique: {unique_found[:10]}")

if __name__ == "__main__":
    debug()
