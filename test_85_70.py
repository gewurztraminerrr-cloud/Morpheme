import random
import os
import sys
import time

# Add project path to sys.path to import local modules
sys.path.insert(0, '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator
from word_validator import word_validator

# Pre-load dictionaries
UNIQUE_NWL = set()
path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/uniqueNWL.txt'
if os.path.exists(path):
    with open(path, 'r') as f:
        UNIQUE_NWL = set(line.strip().upper() for line in f if line.strip())

BONUS_POOL = [w for w in UNIQUE_NWL if 8 <= len(w) <= 10]

def get_stats(board, gen, min_len=6):
    all_words = gen._solve_board(board, 'NWL', (0, 99999), min_len)
    words = set(all_words.keys())
    total = len(words)
    uniques = len(words.intersection(UNIQUE_NWL))
    ratio = uniques / total if total > 0 else 0
    return total, uniques, ratio

if __name__ == '__main__':
    print("Searching for 6x6 (6-Letter Min) board with 85+ total words and 70%+ uniqueness...")
    gen = BoardGenerator()
    start = time.time()
    attempts = 0
    
    # Try for 2 minutes or 2000 attempts
    while time.time() - start < 120 and attempts < 2000:
        attempts += 1
        bonus = random.choice(BONUS_POOL)
        # 6-letter minimum, target high word count (85-115)
        board, _, _, _, _ = gen.generate_board('6x6', bonus, (85, 115), 'NWL', 'Normal', 6)
        t, u, r = get_stats(board, gen, 6)
        
        # Check against target
        if t >= 85 and r >= 0.7:
             elapsed = time.time() - start
             print(f"\nSUCCESS! Found 70%+ board with {t} words in {elapsed:.2f}s after {attempts} attempts.")
             print("Board Layout:")
             for row in board:
                 print("  " + " ".join(row))
             sys.exit(0)
             
        if attempts % 50 == 0:
             print(f"Attempt {attempts}... Current best ratio found at 85+ words: {r:.2%} (Today's best: {t} words)")

    print(f"\nFAILED to find exact match in 2 minutes/2000 attempts.")
