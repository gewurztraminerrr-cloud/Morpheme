import time
import sys
from board_generator import BoardGenerator
from word_validator import word_validator

def test_aw_generation():
    print("Initializing BoardGenerator...")
    gen = BoardGenerator()
    
    # 1. Test NWL + AW board generation (large target word count, e.g., 300-400 words)
    print("\n--- Testing NWL + AW board generation (4x6, range 300-400) ---")
    start = time.time()
    res = gen.generate_board(
        dimensions='4x6',
        bonus_word='',
        word_count_range='300-400',
        dictionary='NWL + AW',
        board_format='Normal',
        min_word_length=4
    )
    elapsed = time.time() - start
    board, all_words, bonus_cell, board_format_ret, all_words_dict, ratio, final_bonus_word = res
    print(f"Generated board in {elapsed:.2f} seconds.")
    print(f"Actual word count: {len(all_words_dict)}")
    print(f"Meets target (300-400): {300 <= len(all_words_dict) <= 400}")
    
    # 2. Test standard NWL board generation (small target word count, e.g., 50-100 words)
    print("\n--- Testing NWL board generation (4x6, range 50-100, length 6) ---")
    start = time.time()
    res = gen.generate_board(
        dimensions='4x6',
        bonus_word='',
        word_count_range='50-100',
        dictionary='NWL',
        board_format='Normal',
        min_word_length=6
    )
    elapsed = time.time() - start
    board, all_words, bonus_cell, board_format_ret, all_words_dict, ratio, final_bonus_word = res
    print(f"Generated board in {elapsed:.2f} seconds.")
    print(f"Actual word count: {len(all_words_dict)}")
    print(f"Meets target (50-100): {50 <= len(all_words_dict) <= 100}")
    
    # Check minimum word length on 4x6 grid with target range 50-100
    lengths = [len(w) for w in all_words_dict.keys()]
    print(f"Min length found: {min(lengths) if lengths else 'None'} (Expected: >= 6)")
    
    # 3. Test CSW + AW board generation on Equality Freq format
    print("\n--- Testing CSW + AW board generation on Equality Freq format (4x4, range 300-400) ---")
    start = time.time()
    res = gen.generate_board(
        dimensions='4x4',
        bonus_word='',
        word_count_range='300-400',
        dictionary='CSW + AW',
        board_format='Equality Freq',
        min_word_length=3
    )
    elapsed = time.time() - start
    board, all_words, bonus_cell, board_format_ret, all_words_dict, ratio, final_bonus_word = res
    print(f"Generated board in {elapsed:.2f} seconds.")
    print(f"Actual word count: {len(all_words_dict)}")
    print(f"Meets target (300-400): {300 <= len(all_words_dict) <= 400}")

if __name__ == '__main__':
    test_aw_generation()
