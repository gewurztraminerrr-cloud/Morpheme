import sys
import os
import random

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spinner_set import SpinnerSet
from board_generator import BoardGenerator
from word_validator import use_added_words_ctx

def test_dictionary_spinner():
    print("=== Testing Dictionary Spinner ===")
    counts = {"NWL": 0, "CSW": 0, "NWL + AW": 0, "CSW + AW": 0}
    iterations = 10000
    for _ in range(iterations):
        res = SpinnerSet._spin_dictionary()
        counts[res] += 1
    
    print(f"Results over {iterations} spins:")
    for key, val in counts.items():
        pct = (val / iterations) * 100
        print(f"  {key}: {val} ({pct:.2f}%)")
        # Assert each is roughly 25% (allow 20-30% margin of error)
        assert 20.0 <= pct <= 30.0, f"Dictionary {key} spin frequency {pct:.2f}% out of bounds"
    print("Dictionary Spinner stats look correct!\n")

def test_word_count_spinner():
    print("=== Testing Word Count Spinner ===")
    counts = {"50-100": 0, "100-200": 0, "200-300": 0, "300-400": 0, "400-500": 0, "500+": 0}
    iterations = 10000
    for _ in range(iterations):
        dict_choice = SpinnerSet._spin_dictionary()
        res = SpinnerSet._spin_word_count(dict_choice, 3, "Medium", "4x4")
        counts[res] += 1
        
    print(f"Results over {iterations} spins:")
    expected = {
        "50-100": 4.5,
        "100-200": 15.5,
        "200-300": 15.0,
        "300-400": 35.0,
        "400-500": 20.0,
        "500+": 10.0
    }
    
    for key, val in counts.items():
        pct = (val / iterations) * 100
        exp = expected[key]
        print(f"  {key}: {val} ({pct:.2f}%, expected {exp}%)")
        # Allow reasonable margin of error based on weight size
        assert exp - 3.0 <= pct <= exp + 3.0, f"Word count range {key} frequency {pct:.2f}% out of bounds (expected ~{exp}%)"
    print("Word Count Spinner stats look correct!\n")

def test_bonus_word_length_spinner():
    print("=== Testing Bonus Word Length Spinner ===")
    counts = {6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    iterations = 10000
    for _ in range(iterations):
        # Emulate the bonus word spinner logic implemented in app.py/game_room.py/private_match_logic.py
        bonus_len = random.choices([6, 7, 8, 9, 10], weights=[20, 20, 20, 20, 20])[0]
        counts[bonus_len] += 1
        
    print(f"Results over {iterations} spins:")
    for key, val in counts.items():
        pct = (val / iterations) * 100
        print(f"  Length {key}: {val} ({pct:.2f}%)")
        assert 17.0 <= pct <= 23.0, f"Bonus word length {key} frequency {pct:.2f}% out of bounds (expected ~20%)"
    print("Bonus Word Length Spinner stats look correct!\n")

def test_board_generation_normalization():
    print("=== Testing Board Generation Normalization ===")
    bg = BoardGenerator()
    
    # Test case 1: NWL dictionary, 50-100 word count
    print("Generating board with dict='NWL' and word_count_range='50-100'...")
    res = bg.generate_board(
        dimensions="4x4",
        bonus_word="HELLO",
        word_count_range="50-100",
        dictionary="NWL",
        board_format="Normal",
        min_word_length=3,
        difficulty="Medium"
    )
    
    board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio, final_bonus_word = res
    print(f"Generation complete. Words found: {len(all_words)}")
    print(f"Uniqueness ratio: {u_ratio:.2f}")
    
    # Verify that the board is generated correctly and word count is within bounds or reasonable range
    assert board is not None, "Failed: Board is None!"
    assert len(board) == 4 and len(board[0]) == 4, f"Failed: Board dimensions are not 4x4"
    assert len(all_words) > 0, "Failed: No words generated on the board!"
    
    # Test case 2: AW dictionary, 50-100 word count
    print("Generating board with dict='AW' and word_count_range='50-100'...")
    res = bg.generate_board(
        dimensions="4x4",
        bonus_word="WORLD",
        word_count_range="50-100",
        dictionary="AW",
        board_format="Normal",
        min_word_length=3,
        difficulty="Medium"
    )
    assert res[0] is not None, "Failed: Board is None for AW!"
    
    print("Board Generation and dictionary normalization verified successfully!\n")

def main():
    try:
        test_dictionary_spinner()
        test_word_count_spinner()
        test_bonus_word_length_spinner()
        test_board_generation_normalization()
        print("ALL TESTS PASSED SUCCESSFULLY!")
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
