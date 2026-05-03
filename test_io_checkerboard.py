import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from board_generator import BoardGenerator

def test_io_checkerboard():
    bg = BoardGenerator()
    
    print("Testing Checkerboard IO and Base Procedure...")
    start_time = time.time()
    
    # Generate 4x4 board using the new procedure
    board, words, bonus_cell, fmt, words_dict, ratio, bonus_word = bg.generate_board(
        dimensions="4x4",
        bonus_word="MORPHEME",
        word_count_range=(50, 200),
        dictionary="NWL",
        board_format="IO-Checkerboard",
        min_word_length=3,
        difficulty="Medium"
    )
    
    duration = time.time() - start_time
    
    print(f"\nGeneration Complete in {duration:.2f}s")
    print(f"Format: {fmt}")
    print(f"Word Count: {len(words)}")
    print(f"Bonus Word: {bonus_word}")
    print(f"Bonus Cell: {bonus_cell}")
    
    print("\nBoard:")
    for row in board:
        print(" ".join(row))
        
    print("\nFirst 20 words:")
    print(", ".join(words[:20]))

if __name__ == "__main__":
    test_io_checkerboard()
