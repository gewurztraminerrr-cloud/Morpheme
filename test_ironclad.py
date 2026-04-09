
from board_generator import BoardGenerator
import time

gen = BoardGenerator()
print("Starting Ironclad Test: 4x4, 100-200 target, CSW")
start = time.time()
# 4x4, Bonus 'TEST', 100-200, CSW, Normal, 3, Medium
board, words, bonus_cell, fmt, words_dict, u_ratio = gen.generate_board('4x4', 'TEST', '100-200', 'CSW', 'Normal', 3, 'Medium')
duration = time.time() - start

print(f"Results in {duration:.2f}s:")
print(f"Total Words: {len(words)}")
print(f"Target: 100-200")
if 100 <= len(words) <= 200:
    print("SUCCESS: Within range!")
else:
    print(f"FAILURE: {len(words)} is NOT in 100-200.")
