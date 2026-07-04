import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board_generator import BoardGenerator

def main():
    bg = BoardGenerator()
    dims = "6x8"
    
    print("=== Testing 6x8 board generation word counts ===")
    for min_len in [6, 7, 8]:
        for diff in ["Easy", "Medium", "Hard"]:
            print(f"\nDimensions: {dims}, Min Len: {min_len}, Difficulty: {diff}")
            start_time = time.time()
            res = bg.generate_board(
                dimensions=dims,
                bonus_word="TESTING",
                word_count_range="300-400",
                dictionary="NWL",
                board_format="Normal",
                min_word_length=min_len,
                difficulty=diff
            )
            elapsed = time.time() - start_time
            board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio, final_bonus_word = res
            print(f"  Time taken: {elapsed:.2f} seconds")
            print(f"  Words found: {len(all_words)}")
            if board:
                print(f"  Board generated successfully.")
            else:
                print(f"  Board generation FAILED/TIMEOUT.")

if __name__ == "__main__":
    main()
