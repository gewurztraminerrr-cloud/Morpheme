
import sys
import os
sys.path.append(os.getcwd())
from board_generator import BoardGenerator

bg = BoardGenerator()
# dimensions='3x3x3'
# In generate_board, dimensions='3x3x3' sets depth=6, rows=3, cols=3
board, words, bonus_cell, fmt, words_dict, ratio, bonus_word = bg.generate_board(
    dimensions='3x3x3',
    bonus_word='SMILE',
    word_count_range='100-200',
    dictionary='NWL',
    board_format='Checkerboard',
    min_word_length=3,
    difficulty='Medium'
)

print(f"Format: {fmt}")
depth = len(board)
rows = len(board[0])
cols = len(board[0][0])
print(f"Dims: {depth}x{rows}x{cols}")

def check_checkerboard_3d(board):
    depth = len(board)
    rows = len(board[0])
    cols = len(board[0][0])
    for f in range(depth):
        for r in range(rows):
            for c in range(cols):
                # _create_checkerboard uses (f+r+c)%2 == 0 for Consonant, == 1 for Vowel
                expected_vowel = (f + r + c) % 2 != 0
                is_vowel = bg._is_vowel(board[f][r][c])
                if is_vowel != expected_vowel:
                    return False, (f, r, c)
    return True, None

is_cb, fail_pos = check_checkerboard_3d(board)
print(f"Is Checkerboard 3D: {is_cb}")
if not is_cb:
    print(f"Failed at {fail_pos}")
    f, r, c = fail_pos
    print(f"Value: {board[f][r][c]}, Expected Vowel: {(f+r+c)%2 != 0}")
