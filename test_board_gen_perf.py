
import time
import sys
import os

# Set up path to import from current directory
sys.path.append(os.getcwd())

from board_generator import BoardGenerator

def test_6x8_perf():
    print("Testing 6x8 Board Generation Performance...")
    bg = BoardGenerator()
    
    start_time = time.time()
    # 6x8, Medium/Hard (Masters mapping), NWL, Normal format, min 3 length, Hard difficulty
    # Hard difficulty on 6x8 uses StepwiseOptimization
    board, all_words, bonus_cell, fmt, words_dict, ratio = bg.generate_board(
        dimensions='6x8',
        bonus_word='ADVENTURE',
        word_count_range='100-200',
        dictionary='NWL',
        board_format='Normal',
        min_word_length=3,
        difficulty='Hard'
    )
    end_time = time.time()
    
    print(f"\nFinal Result:")
    print(f"Time Taken: {end_time - start_time:.2f}s")
    print(f"Word Count: {len(all_words)}")
    print(f"Uniqueness Ratio: {ratio:.2%}")
    print(f"Board Format: {fmt}")

if __name__ == "__main__":
    test_6x8_perf()
