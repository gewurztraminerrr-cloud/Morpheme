import sys
import time
from board_generator import BoardGenerator

def test_cube_density():
    gen = BoardGenerator()
    print("Testing 3x3x3 Cube Board with 6L+ Minimum Word Length...")
    
    # User Request Context: 3x3x3 | NWL | Words: 100-200 | Min: 6L
    start = time.time()
    res = gen.generate_board('3x3x3', 'CUBE', '100-200', 'NWL', 'Normal', 6, 'Medium')
    duration = time.time() - start
    
    if res:
        board, words, bonus_cell, fmt, word_dict, ratio = res
        print(f"✓ Cube Board Generated in {duration:.2f}s")
        print(f"  Format: {fmt}")
        print(f"  Word Count: {len(words)} (Goal: 100-200)")
        print(f"  Uniqueness Ratio: {ratio:.2%}")
        
        # Check for 1000 word explosion
        if len(words) > 300:
             print("! ALERT: Word explosion still present!")
        else:
             print("✓ Word count within reasonable range.")
             
        # Check min word length
        lengths = [len(w) for w in words]
        if min(lengths) < 6:
             print(f"! ALERT: Minimum word length violation: {min(lengths)}L found")
        else:
             print(f"✓ Min word length respected: {min(lengths)}L")
    else:
        print("✗ Failed to generate cube board.")

if __name__ == '__main__':
    test_cube_density()
