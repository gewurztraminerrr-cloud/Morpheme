import sys
import os
import time

# Add current path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator

def test_ultra_rare():
    gen = BoardGenerator()
    
    # dimensions, bonus, range_w, dict_name, min_l, diff
    print("Testing 4x4 with ULTRA RARE letters for Hard difficulty...")
    
    # We'll artificially set the weights in IO to be very rare
    # Actually, I'll modify BoardGenerator to use a rarer set for Hard
    from board_generator import RARE_SET, VOWELS, CONSONANTS
    
    results = []
    # Force 40 attempts
    for i in range(5):
        board, words, bonus_c, fmt, dict_full, ratio = gen.generate_board(
            '4x4', 'MORPHEME', (50, 100), 'NWL', 'Normal', 3, 'Hard'
        )
        results.append(ratio)
        print(f"Attempt {i}: {ratio:.1%}")
        
    print(f"Max ratio found: {max(results):.1%}")

if __name__ == "__main__":
    test_ultra_rare()
