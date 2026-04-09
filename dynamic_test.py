import time
import random
import os
import sys

# Add project path to sys.path to import local modules
sys.path.insert(0, '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator
from word_validator import word_validator

# Pre-load dictionaries
print("Loading dictionaries...")
UNIQUE_NWL = set()
path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/uniqueNWL.txt'
if os.path.exists(path):
    with open(path, 'r') as f:
        UNIQUE_NWL = set(line.strip().upper() for line in f if line.strip())

# Extract 8-letter words from UNIQUE_NWL for bonus word pool
BONUS_POOL = [w for w in UNIQUE_NWL if len(w) == 8]
if not BONUS_POOL:
    BONUS_POOL = ["HYDRATOR", "GREGATIM", "TEMPESTS", "UNIFYING", "MANIKINS"] # Fallback

def get_stats(board, gen):
    all_words = gen._solve_board(board, 'NWL', (0, 99999), 3)
    words = set(all_words.keys())
    total = len(words)
    uniques = len(words.intersection(UNIQUE_NWL))
    ratio = uniques / total if total > 0 else 0
    return total, uniques, ratio

def run_s1(gen):
    print("\n--- Scenario 1: Re-roll with Random Bonus ---")
    start = time.time()
    best_r = 0
    attempts = 0
    while time.time() - start < 30:
        attempts += 1
        bonus = random.choice(BONUS_POOL)
        board, _, _, _, _ = gen.generate_board('4x4', bonus, (50, 100), 'NWL', 'Normal', 3)
        t, u, r = get_stats(board, gen)
        if 50 <= t <= 100:
            if r > best_r:
                best_r = r
                print(f"New Best: {best_r:.2%} ({u}/{t}) w/ Bonus '{bonus}' at att {attempts}")
                if r >= 0.7: return True, time.time() - start, r, bonus
    return False, 30, best_r, None

def run_s2(gen):
    print("\n--- Scenario 2: IO with Random Bonus ---")
    start = time.time()
    bonus = random.choice(BONUS_POOL)
    board, _, _, _, _ = gen.generate_board('4x4', bonus, (50, 100), 'NWL', 'Normal', 3)
    t, u, initial_r = get_stats(board, gen)
    print(f"Initial: {initial_r:.2%} ({u}/{t}) w/ Bonus '{bonus}'")
    
    current_board = [row[:] for row in board]
    # Protect bonus cells
    excluded = set()
    # Find bonus path to protect
    all_w = gen._solve_board(board, 'NWL', (0, 99999), 3)
    if bonus.upper() in all_w:
        excluded = set(all_w[bonus.upper()])
    
    pos = [(r, c) for r in range(4) for c in range(4) if (r, c) not in excluded]
    random.shuffle(pos)
    
    # Do 5 IO steps
    best_r = initial_r
    for idx, (r, c) in enumerate(pos[:5]):
        cur_t, cur_u, cur_r = get_stats(current_board, gen)
        best_char = current_board[r][c]
        
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            current_board[r][c] = char
            _, u_c, _ = get_stats(current_board, gen)
            if u_c > cur_u:
                cur_u = u_c
                best_char = char
        
        current_board[r][c] = best_char
        t, u, r = get_stats(current_board, gen)
        print(f"Step {idx+1}: {r:.2%} ({u}/{t})")
        if r > best_r: best_r = r
        if r >= 0.7 and 50 <= t <= 100:
             return True, time.time() - start, r, bonus
             
    return False, time.time() - start, best_r, bonus

if __name__ == '__main__':
    gen = BoardGenerator()
    s1_res = run_s1(gen)
    s2_res = run_s2(gen)
    
    print("\n" + "="*50)
    print("FINAL RESULTS (WITH DYNAMIC BONUS WORDS)")
    print("="*50)
    print("Scenario 1: Success={}, Best Ratio={:.2%}, Time={:.2f}s".format(s1_res[0], s1_res[2], s1_res[1]))
    print("Scenario 2: Success={}, Best Ratio={:.2%}, Time={:.2f}s".format(s2_res[0], s2_res[2], s2_res[1]))
    print("="*50)
