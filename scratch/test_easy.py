import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board_generator import BoardGenerator

bg = BoardGenerator()

test_cases = [
    ("4x4", "NWL"),
    ("4x4", "CSW"),
    ("5x7", "NWL"),
    ("5x7", "CSW"),
    ("6x8", "NWL"),
    ("6x8", "CSW"),
    ("3x3x3", "NWL"),
    ("3x3x3", "CSW")
]

print("Running Easy board generation tests across different dimensions and dictionaries...")
for dims, dict_name in test_cases:
    print(f"\n--- Testing {dims} with {dict_name} ---")
    res = bg.generate_board(
        dimensions=dims,
        bonus_word="",
        word_count_range="100-200" if dims != "3x3x3" else "300-400",
        dictionary=dict_name,
        board_format="Normal",
        min_word_length=3 if dims == "4x4" else (4 if dims == "5x7" else 5),
        difficulty="Easy"
    )
    board, words, bonus_cell, fmt, words_dict, ratio, bonus = res
    depth, rows, cols = (6, 3, 3) if dims == "3x3x3" else (1, int(dims.split("x")[0]), int(dims.split("x")[1]))
    achieved = bg.get_difficulty_label(ratio, rows, cols, dict_name, depth)
    print(f"Result for {dims} ({dict_name}): Ratio = {ratio:.2%}, Achieved = {achieved}, Words = {len(words)}")
