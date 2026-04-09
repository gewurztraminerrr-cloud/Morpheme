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

# Expanded pool based on user input
DIFFICULT_LETTERS = "ZXQJKVWYPFBHFPCMA" # User included 'A' in the list - though frequent, I'll include it for the test
# Wait, user list was "H F P C M A". A is definitely not rare, but maybe they mean M/A combinations?
# I'll use HFPCM as the "Medium-Rare" set.

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
    print(f"Final 6x6 IO Test (Targeting 85+ words, 70%+ uniqueness)...")
    
    # 1. Start with a VERY dense board (Target 100+ words to have room to 'spend')
    seeds = ["ASTONISHMENT", "INTERMAT", "MANIKINS", "HYDRATOR"]
    bonus = random.choice(seeds)
    board, _, _, _, _ = gen.generate_board('6x6', bonus, (100, 150), 'NWL', 'Normal', min_len)
    t, u, r = get_stats(board, gen, min_len)
    print(f"Initial Dense Board: {r:.2%} ({u}/{t})")
    
    current = [row[:] for row in board]
    pos = [(r, c) for r in range(6) for c in range(6)]
    random.shuffle(pos)
    
    # 2. IO with Expanded difficult letters
    diff_set = "ZXQJKVWYPFBHFPCM" 
    
    for idx, (r, c) in enumerate(pos[:15]): # 15 steps
        _, start_u, start_r = get_stats(current, gen, min_len)
        best_char = current[r][c]
        max_ratio = start_r
        
        for char in diff_set:
            current[r][c] = char
            t_c, u_c, r_c = get_stats(current, gen, min_len)
            
            # Condition: Higher ratio, AND total words must stay above 80
            if r_c > max_ratio and t_c >= 80:
                 max_ratio = r_c
                 best_char = char
                 
        current[r][c] = best_char
        t, u, r = get_stats(current, gen, min_len)
        if (idx+1) % 3 == 0:
            print(f"Step {idx+1}: {r:.2%} ({u}/{t}) [Best Ratio so far]")
            
        if r >= 0.7 and 85 <= t <= 115:
             print(f"\nGOLDEN BOARD FOUND in {idx+1} steps!")
             break
             
    print(f"\nFINAL TEST RESULT: {r:.2%} ({u}/{t})")
    if r >= 0.7:
        for row in current: print("  " + " ".join(row))
