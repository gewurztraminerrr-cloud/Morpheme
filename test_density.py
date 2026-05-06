import sys
import os
import time

# Add backend to path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')
from board_generator import BoardGenerator

bg = BoardGenerator()
print("Starting 4x4 Ultra-Density Test...")
best = 0
for i in range(50):
    board, words, bonus, fmt, wd, ratio, fb = bg.generate_board(
        "4x4",
        "EATING",
        "100-200",
        "NWL",
        "Normal",
        5,
        "Easy",
        is_emergency=True
    )
    c = sum(1 for w in words if len(w) >= 5)
    if c > best:
        best = c
        print(f"New Best: {best}")

print(f"Final Best 5L+ words: {best}")
