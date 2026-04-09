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

BONUS_POOL = [w for w in UNIQUE_NWL if 8 <= len(w) <= 12]
# Rare/Difficult Set (Expanded by User)
RARE_SET = "ZXQJKVWYPFBHFPCM" + "BKWY" # Concatenated final set

def get_stats(board, gen, min_len=6):
    all_words = gen._solve_board(board, 'NWL', (0, 99999), min_len)
    words = set(all_words.keys())
    total = len(words)
    uniques = len(words.intersection(UNIQUE_NWL))
    ratio = uniques / total if total > 0 else 0
    return total, uniques, ratio

def count_vowels(board):
    vowels = "AEIOU"
    count = 0
    total = 0
    for r in range(len(board)):
        for c in range(len(board[0])):
            total += 1
            if str(board[r][c]).upper() in vowels:
                count += 1
    return count, total

def enforce_vowel_ratio(board):
    v_count, total = count_vowels(board)
    vowels = "AEIOU"
    consonants = "BCDFGHJKLMNPQRSTVWXYZ"
    
    # Target 34% (Mid of 30-38%)
    target_v = int(total * 0.34)
    min_v = int(total * 0.30) + 1
    max_v = int(total * 0.38)
    
    # Pick random positions that aren't part of the bonus word? 
    # (Actually simpler to just pick any position not already been IO'd or similar)
    all_pos = [(r, c) for r in range(len(board)) for c in range(len(board[0]))]
    random.shuffle(all_pos)
    
    if v_count < min_v:
        needed = min_v - v_count
        for i in range(needed):
            r, c = all_pos[i]
            if board[r][c] not in vowels:
                board[r][c] = random.choice(vowels)
    elif v_count > max_v:
        over = v_count - max_v
        for i in range(over):
            r, c = all_pos[i]
            if board[r][c] in vowels:
                board[r][c] = random.choice(consonants)

if __name__ == '__main__':
    gen = BoardGenerator()
    min_len = 6
    print(f"ULTIMATE MASTER GENERATION (6x6, 6L-min, 30-38% Vowels, 85+ Words, 70%+ Uniques)")
    
    start_time = time.time()
    attempts = 0
    
    while time.time() - start_time < 300: # 5 Minutes Max
        attempts += 1
        bonus = random.choice(BONUS_POOL)
        
        # 1. Start with a dense board
        board, _, _, _, _ = gen.generate_board('6x6', bonus, (100, 150), 'NWL', 'Normal', min_len)
        enforce_vowel_ratio(board)
        
        # 2. IO Step (Difficult Letters only)
        current = [row[:] for row in board]
        pos = [(r, c) for r in range(6) for c in range(6)]
        random.shuffle(pos)
        
        for idx, (r, c) in enumerate(pos[:12]):
            _, start_u, start_r = get_stats(current, gen, min_len)
            best_char = current[r][c]
            max_ratio = start_r
            
            for char in RARE_SET:
                current[r][c] = char
                # Quick Vowel Check (Skip if it breaks 30-38%)
                v_c, v_total = count_vowels(current)
                if not (0.30 <= v_c/v_total <= 0.38):
                    continue
                    
                t_c, u_c, r_c = get_stats(current, gen, min_len)
                if r_c > max_ratio and t_c >= 80:
                    max_ratio = r_c
                    best_char = char
            
            current[r][c] = best_char
        
        t, u, r = get_stats(current, gen, min_len)
        v_c, v_total = count_vowels(current)
        
        if r >= 0.7 and 85 <= t <= 118:
            print(f"\n✅ PERFECT MASTER BOARD FOUND in {time.time() - start_time:.2f}s!")
            print(f"Attempts: {attempts}")
            print(f"Bonus: {bonus}")
            print(f"Stats: Ratio={r:.2%}, Total={t}, Unique={u}, Vowels={v_c/v_total:.1%}")
            
            for row in current: print("  " + " ".join(row))
            sys.exit(0)
            
        if attempts % 10 == 0:
            print(f"Attempt {attempts}... Best this loop: Ratio={r:.2%} Total={t}")

    print(f"\nFAILED to find PERFECT board in 5 minutes.")
