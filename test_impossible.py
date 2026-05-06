
from board_generator import BoardGenerator
import time

gen = BoardGenerator()
print("Starting Impossible Test: 6x8, 200+ target, NWL, 6L min")
start = time.time()
# 6x8, Bonus 'TESTINGIT', 200+, NWL, Normal, 6, Medium
board, words, bonus_cell, fmt, words_dict, u_ratio, *extra = gen.generate_board('6x8', 'TESTINGIT', '200+', 'NWL', 'Normal', 6, 'Medium')
duration = time.time() - start

print(f"Results in {duration:.2f}s:")
print(f"Total Words: {len(words)}")
print(f"Target: 200+")
if len(words) >= 200:
    print("SUCCESS: 200+ reached!")
else:
    print(f"FAILURE: {len(words)} is < 200.")
