import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board_generator import BoardGenerator

def main():
    bg = BoardGenerator()
    
    # Test 5x7 with 6L+
    print("Testing 5x7 with min_len=6, target='300-400'...")
    start = time.time()
    res = bg.generate_board(
        dimensions="5x7",
        bonus_word="HELLO",
        word_count_range="300-400",
        dictionary="NWL",
        board_format="Normal",
        min_word_length=6,
        difficulty="Medium"
    )
    print(f"  Time: {time.time() - start:.2f}s, Words: {len(res[1]) if res[0] else 0}")
    
    # Test 6x8 with 7L+
    print("Testing 6x8 with min_len=7, target='300-400'...")
    start = time.time()
    res = bg.generate_board(
        dimensions="6x8",
        bonus_word="TESTING",
        word_count_range="300-400",
        dictionary="NWL",
        board_format="Normal",
        min_word_length=7,
        difficulty="Medium"
    )
    print(f"  Time: {time.time() - start:.2f}s, Words: {len(res[1]) if res[0] else 0}")

if __name__ == "__main__":
    main()
