import random
import os
import sys

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

def get_stats(board, gen, min_len=6):
    all_words = gen._solve_board(board, 'NWL', (0, 99999), min_len)
    words = set(all_words.keys())
    total = len(words)
    uniques = len(words.intersection(UNIQUE_NWL))
    ratio = uniques / total if total > 0 else 0
    return total, uniques, ratio

if __name__ == '__main__':
    gen = BoardGenerator()
    min_len = 6
    
    # 1. Generate a stable base board
    bonus = "ASTONISH"
    board, _, _, _, _ = gen.generate_board('6x6', bonus, (50, 100), 'NWL', 'Normal', min_len)
    current = [row[:] for row in board]
    
    # 2. Pick a good central position for maximum impact
    r, c = 2, 2
    
    print(f"Testing all letters at position (2,2) on a 6x6 (6L-min) board...")
    results = []
    
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        current[r][c] = char
        t, u, r_val = get_stats(current, gen, min_len)
        results.append((char, t, u, r_val))
    
    # Sort by UNIQUE count
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("\nRESULTS (Sorted by Unique Word Count):")
    for char, t, u, r in results:
        is_uncommon = char in "ZXQJKVWYPFB"
        tag = "[LESS COMMON]" if is_uncommon else ""
        print(f"Letter {char} {tag:15}: Total={t:3}, Unique={u:3}, Ratio={r:.2%}")
    
    # Sort by RATIO
    results.sort(key=lambda x: x[3], reverse=True)
    print("\nRESULTS (Sorted by Difficulty Ratio):")
    for char, t, u, r in results:
        is_uncommon = char in "ZXQJKVWYPFB"
        tag = "[LESS COMMON]" if is_uncommon else ""
        print(f"Letter {char} {tag:15}: Total={t:3}, Unique={u:3}, Ratio={r:.2%}")
