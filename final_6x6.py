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
if not BONUS_POOL: BONUS_POOL = ["HYDRATOR", "GREGATIM", "TEMPESTS", "UNIFYING", "MANIKINS"]

def get_stats(board, gen, min_len=6):
    all_words = gen._solve_board(board, 'NWL', (0, 99999), min_len)
    words = set(all_words.keys())
    total = len(words)
    uniques = len(words.intersection(UNIQUE_NWL))
    ratio = uniques / total if total > 0 else 0
    return total, uniques, ratio, sorted(list(words.intersection(UNIQUE_NWL)))[:10]

def run_scenario_1_6x6():
    print("\n[6x6 Scenario 1] Re-rolling with random bonus words...")
    gen = BoardGenerator()
    start = time.time()
    attempts = 0
    best_ratio = 0
    last_board = None
    last_stats = None
    last_bonus = None
    
    while time.time() - start < 15:
        attempts += 1
        bonus = random.choice(BONUS_POOL)
        # 6-letter minimum
        board, _, _, _, _ = gen.generate_board('6x6', bonus, (10, 100), 'NWL', 'Normal', 6)
        t, u, r, samples = get_stats(board, gen, 6)
        if r > best_ratio:
            best_ratio = r
            last_board = board
            last_stats = (t, u, r, samples)
            last_bonus = bonus
            if r >= 0.7: break
            
    return last_board, last_stats, last_bonus, attempts

def run_scenario_2_6x6():
    print("\n[6x6 Scenario 2] IO Optimization starting...")
    gen = BoardGenerator()
    bonus = random.choice(BONUS_POOL)
    board, _, _, _, _ = gen.generate_board('6x6', bonus, (10, 100), 'NWL', 'Normal', 6)
    
    current = [row[:] for row in board]
    # Protect bonus cell if found
    all_w = gen._solve_board(board, 'NWL', (0, 99999), 6)
    excluded = set()
    if bonus.upper() in all_w:
        excluded = set(all_w[bonus.upper()])
    
    # Do 5 IO steps on 6x6
    pos = [(r, c) for r in range(6) for c in range(6) if (r, c) not in excluded]
    random.shuffle(pos)
    
    for r, c in pos[:5]:
        _, start_u, _, _ = get_stats(current, gen, 6)
        best_char = current[r][c]
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            current[r][c] = char
            _, uni, _, _ = get_stats(current, gen, 6)
            if uni > start_u:
                start_u = uni
                best_char = char
        current[r][c] = best_char
        
    t, u, r, samples = get_stats(current, gen, 6)
    return current, (t, u, r, samples), bonus

if __name__ == '__main__':
    b1, s1, bon1, att1 = run_scenario_1_6x6()
    b2, s2, bon2 = run_scenario_2_6x6()
    
    print("\n" + "="*50)
    print("FINAL 6x6 COMPARISON (6-LETTER MINIMUM)")
    print("="*50)
    
    print(f"\n[Scenario 1] Best After {att1} Attempts")
    print(f"Bonus Word: {bon1}")
    print(f"Stats: Total={s1[0]}, Unique={s1[1]}, Ratio={s1[2]:.2%}")
    print("Board:")
    for row in b1: print("  " + " ".join(row))
    
    print(f"\n[Scenario 2] Best After 5 IO Steps")
    print(f"Bonus Word: {bon2}")
    print(f"Stats: Total={s2[0]}, Unique={s2[1]}, Ratio={s2[2]:.2%}")
    print("Board:")
    for row in b2: print("  " + " ".join(row))
    print("="*50)
