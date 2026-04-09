import sys
import os
import time

# Add current path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator

def calibrate():
    gen = BoardGenerator()
    
    # Test 4x4 Hard
    print("Calibrating 4x4 Hard boards (Targeting max uniqueness via IO)...")
    
    params = [
        ('4x4', 'MORPHEME', (50, 100), 'NWL', 3, 'Hard'),
        ('4x6', 'MORPHEME', (80, 150), 'NWL', 3, 'Hard'),
        ('5x7', 'MORPHEME', (100, 200), 'NWL', 3, 'Hard'),
        ('6x8', 'MORPHEME', (150, 400), 'NWL', 3, 'Hard')
    ]
    
    results = []
    
    for dim, bonus, range_w, dict_name, min_l, diff in params:
        print(f"\nTesting {dim} {diff}...")
        start = time.time()
        # The generate_board will now use IO with Unique dictionary
        board, words, bonus_c, fmt, dict_full, ratio = gen.generate_board(
            dim, bonus, range_w, dict_name, 'Normal', min_l, diff
        )
        elapsed = time.time() - start
        results.append({
            'dim': dim,
            'time': elapsed,
            'words': len(words),
            'ratio': ratio
        })
        print(f"Result {dim}: {elapsed:.2f}s, {len(words)} words, {ratio:.1%} uniqueness.")

    print("\nCalibration Results:")
    for r in results:
        print(f"{r['dim']}: {r['ratio']:.1%} uniqueness found in {r['time']:.2f}s ({r['words']} words)")

if __name__ == "__main__":
    calibrate()
