import sys
import os
import time

sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator

def test_high_density_6x8():
    gen = BoardGenerator()
    
    print("Testing 6x8 Medium Density (Target: 200+ words, no upper limit)...")
    start = time.time()
    # dimensions, bonus, range_w, dict_name, min_l, diff
    board, words, bonus_c, fmt, dict_full, ratio = gen.generate_board(
        '6x8', 'MORPHEME', '200+', 'NWL', 'Normal', 3, 'Medium'
    )
    elapsed = time.time() - start
    
    print(f"\nFinal Board Stats:")
    print(f"Dimensions: 6x8")
    print(f"Total Words: {len(words)}")
    print(f"Uniqueness: {ratio:.1%}")
    print(f"Time Taken: {elapsed:.2f}s")
    
    if len(words) >= 500:
        print("\n✓ SUCCESS: Hit 500+ Word Count requirement.")
    else:
        print(f"\n✗ FAILED: Only hit {len(words)} words.")

if __name__ == "__main__":
    test_high_density_6x8()
