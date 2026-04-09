import sys
import os
import time

# Add current path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator

def debug_6x8():
    gen = BoardGenerator()
    
    unique_set = gen._get_difficulty_set('NWL')
    print(f"Unique Set (NWL) size: {len(unique_set)}")
    
    # Force 6x8 IO
    # dimensions, bonus, word_range, dictionary, format, min_l, diff
    print("Generating 6x8 Hard board...")
    board, words, bonus_c, fmt, dict_full, ratio, bonus_word = gen.generate_board(
        '6x8', 'MORPHEME', (150, 400), 'NWL', 'Normal', 3, 'Hard'
    )
    
    print(f"Final Board Ratio: {ratio:.1%}")
    print(f"Total Words: {len(words)}")
    unique_found = [w for w in words if w.upper() in unique_set]
    print(f"Unique count: {len(unique_found)}")
    if unique_found:
        print(f"Sample Unique: {unique_found[:10]}")
    else:
        print("No unique words found!")

if __name__ == "__main__":
    debug_6x8()
