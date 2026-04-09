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
DIFFICULT_LETTERS = "ZXQJKVWYPFB"

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
    
    print("Testing IO Optimization using ONLY UNCOMMON LETTERS (ZXQJKVWYPFB)...")
    bonus = random.choice(BONUS_POOL)
    board, _, _, _, _ = gen.generate_board('6x6', bonus, (30, 80), 'NWL', 'Normal', min_len)
    t, u, r = get_stats(board, gen, min_len)
    print(f"Initial: {r:.2%} ({u}/{t}) w/ Bonus '{bonus}'")
    
    current = [row[:] for row in board]
    excluded = set() # Standard protection
    
    pos = [(r, c) for r in range(6) for c in range(6)]
    random.shuffle(pos)
    
    # 10 IO steps
    for idx, (r, c) in enumerate(pos[:10]):
        _, start_u, start_r = get_stats(current, gen, min_len)
        best_char = current[r][c]
        max_ratio = start_r
        best_u = start_u
        
        # ONLY test difficult letters
        for char in DIFFICULT_LETTERS:
            current[r][c] = char
            t_c, u_c, r_c = get_stats(current, gen, min_len)
            
            # Weighing Ratio + Unique count
            # We want high ratio, but some uniques are needed
            if r_c > max_ratio:
                 max_ratio = r_c
                 best_char = char
                 best_u = u_c
            elif r_c == max_ratio and u_c > best_u:
                 best_char = char
                 best_u = u_c
                 
        current[r][c] = best_char
        t, u, r = get_stats(current, gen, min_len)
        print(f"Step {idx+1}: {r:.2%} ({u}/{t}) [Letter {best_char}]")
        if r >= 0.7 and 50 <= t <= 100:
             print("\nCONGRATULATIONS! Found the target.")
             break
             
    print(f"\nFinal: {r:.2%} ({u}/{t})")
