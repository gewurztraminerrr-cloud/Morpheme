import sys
import os
import time

sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator

def test_no_ing():
    gen = BoardGenerator()
    
    print("Generating 5 Hard 4x4 boards and checking for ING sequences...")
    for i in range(5):
        board, words, bonus_c, fmt, dict_full, ratio = gen.generate_board(
            '4x4', 'MORPHEME', (50, 100), 'NWL', 'Normal', 3, 'Hard'
        )
        
        has_ing = gen._has_forbidden_sequence(board, "ING")
        print(f"Board #{i+1}: Words={len(words)}, Unique={ratio:.1%}, Has ING={has_ing}")
        
    print("\nGenerating 5 Easy 4x4 boards (should allow ING)...")
    for i in range(5):
        board, words, bonus_c, fmt, dict_full, ratio = gen.generate_board(
            '4x4', 'MORPHEME', (50, 100), 'NWL', 'Normal', 3, 'Easy'
        )
        has_ing = gen._has_forbidden_sequence(board, "ING")
        print(f"Easy Board #{i+1}: Words={len(words)}, Has ING={has_ing}")

if __name__ == "__main__":
    test_no_ing()
