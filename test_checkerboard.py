
import sys
import os
sys.path.append(os.getcwd())
from board_generator import BoardGenerator

bg = BoardGenerator()
board, words, bonus_cell, fmt, words_dict, ratio, bonus_word = bg.generate_board(
    dimensions='4x4',
    bonus_word='SMILE',
    word_count_range='100-200',
    dictionary='NWL',
    board_format='Checkerboard',
    min_word_length=3,
    difficulty='Medium'
)

print(f"Format: {fmt}")
for r in range(len(board)):
    row_str = " ".join(board[r])
    types = " ".join(["V" if bg._is_vowel(c) else "C" for c in board[r]])
    print(f"{row_str}  |  {types}")

def check_checkerboard(board):
    for r in range(len(board)):
        for c in range(len(board[0])):
            expected_vowel = (r + c) % 2 != 0
            is_vowel = bg._is_vowel(board[r][c])
            if is_vowel != expected_vowel:
                return False, (r, c)
    return True, None

is_cb, fail_pos = check_checkerboard(board)
print(f"Is Checkerboard: {is_cb}")
if not is_cb:
    print(f"Failed at {fail_pos}")
