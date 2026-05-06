
from board_generator import BoardGenerator
import time

gen = BoardGenerator()
print("Starting Cube Density Test: 3x3x3, 100-200 target, NWL")
start = time.time()
# 3x3x3, Bonus 'TESTING', 50-100, NWL, Normal, 3, Medium
# Note: dimensions for 3x3x3 are '3x3x3' in some cases, or f, r, c
# BoardGenerator.generate_board handles its own parsing
board, words, bonus_cell, fmt, words_dict, u_ratio, *extra = gen.generate_board('3x3x3', 'TESTING', '100-200', 'NWL', 'Normal', 3, 'Medium')
duration = time.time() - start

print(f"Results in {duration:.2f}s:")
print(f"Total Words: {len(words)}")
print(f"Target: 100-200")
if 100 <= len(words) <= 200:
    print("SUCCESS: 100-200 reached!")
else:
    print(f"FAILURE: {len(words)} is outside the range.")
