
import sys
import os

# Add morpheme directory to path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator
import word_validator

bg = BoardGenerator()
board = [
    ['A', 'S', 'T', 'A'],
    ['T', 'R', 'N', 'O'],
    ['C', 'A', 'I', 'O'],
    ['E', 'E', 'T', 'T']
]

# Solve with 3L
words_3l = bg._solve_board(board, dictionary='CSW', min_word_length=3)
print(f"Total 3L+ words: {len(words_3l)}")

# Group by length
counts = {}
for w in words_3l:
    l = len(w)
    counts[l] = counts.get(l, 0) + 1

print("Counts By Length:")
for l in sorted(counts.keys()):
    print(f"  {l}LW: {counts[l]}")

# Solve with 5L
words_5l = bg._solve_board(board, dictionary='CSW', min_word_length=5)
print(f"\nTotal 5L+ words: {len(words_5l)}")
