import sys
import os
import time

# Add project path to sys.path
sys.path.insert(0, '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator

def test_100_200_logic():
    gen = BoardGenerator()
    
    print("Testing 100-200 word board generation with Stage 2 IO & B Checkerboard (Base=50)...")
    start_time = time.time()
    
    board, words, bonus_cell, fmt, words_dict, ratio, bonus_upper = gen.generate_board(
        dimensions="4x4",
        bonus_word="GOLD",
        word_count_range="100-200",
        dictionary="NWL",
        board_format="Normal",
        min_word_length=3,
        difficulty="Hard"
    )
    
    duration = time.time() - start_time
    
    if board:
        print("\nSUCCESS!")
        print(f"Format: {fmt}")
        print(f"Words: {len(words)}")
        print(f"Uniqueness Ratio: {ratio:.2%}")
        print(f"Duration: {duration:.2f}s")
        
        print("\nFinal Board:")
        for r in range(4):
            print(" ".join(board[r]))
    else:
        print("\nFAILED to generate board.")

if __name__ == "__main__":
    test_100_200_logic()
