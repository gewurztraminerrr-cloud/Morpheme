import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board_generator import BoardGenerator
from word_validator import word_validator

# Let's add dummy 5-letter and 6-letter alphabetic words to the added_words list
word_validator.added_words.clear()
word_validator.added_words.add("MYAWF")
word_validator.added_words.add("MYAWSX")
print(f"Added Words in validator: {word_validator.added_words}")

bg = BoardGenerator()

# We will intercept the board right after _create_normal_board
orig_create_normal_board = bg._create_normal_board

def debug_create_normal_board(*args, **kwargs):
    board = orig_create_normal_board(*args, **kwargs)
    # Solve this initial board
    words = bg._solve_board(board, "CSW", (0, 99999), 3, store_paths=False)
    found = [w for w in words if w in ["MYAWF", "MYAWSX"]]
    print(f"\n[DEBUG] Immediately after embedding: Custom words found = {found}")
    return board

bg._create_normal_board = debug_create_normal_board

print("\nGenerating board with CSW + AW...")
res = bg.generate_board(
    dimensions="4x4",
    bonus_word="",
    word_count_range="100-200",
    dictionary="CSW + AW",
    board_format="Normal",
    min_word_length=3,
    difficulty="Easy"
)
board, words, bonus_cell, fmt, words_dict, ratio, bonus = res

print("\nFinal Generated Board:")
for row in board:
    print(" ".join(row))

# Check if our custom words were found in the final board
found_aw = [w for w in words if w in ["MYAWF", "MYAWSX"]]
print(f"\nCustom Added Words found on the final board: {found_aw}")
