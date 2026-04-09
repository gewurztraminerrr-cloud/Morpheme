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

BONUS_POOL = [w for w in UNIQUE_NWL if len(w) == 8]
if not BONUS_POOL: BONUS_POOL = ["HYDRATOR", "GREGATIM", "TEMPESTS", "UNIFYING", "MANIKINS"]

def get_stats(board, gen):
    all_words = gen._solve_board(board, 'NWL', (0, 99999), 3)
    words = set(all_words.keys())
    total = len(words)
    uniques = len(words.intersection(UNIQUE_NWL))
    ratio = uniques / total if total > 0 else 0
    return total, uniques, ratio, sorted(list(words.intersection(UNIQUE_NWL)))[:10]

def print_board(name, board, total, uniques, ratio, samples, bonus):
    print(f"\n--- {name} ---")
    print(f"Bonus Word: {bonus}")
    print(f"Stats: Total={total}, Unique={uniques}, Difficulty={ratio:.2%}")
    print("Board Layout:")
    for row in board:
        print("  " + " ".join(row))
    print(f"Sample Unique Words: {', '.join(samples)}...")

if __name__ == '__main__':
    gen = BoardGenerator()
    
    # --- Example 1: Scenario 1 ---
    bonus1 = random.choice(BONUS_POOL)
    board1, _, _, _, _ = gen.generate_board('4x4', bonus1, (50, 100), 'NWL', 'Normal', 3)
    t1, u1, r1, s1 = get_stats(board1, gen)
    print_board("Scenario 1 (Random + Bonus)", board1, t1, u1, r1, s1, bonus1)
    
    # --- Example 2: Scenario 2 ---
    bonus2 = random.choice(BONUS_POOL)
    board2, _, _, _, _ = gen.generate_board('4x4', bonus2, (50, 100), 'NWL', 'Normal', 3)
    # Optimized once
    current = [row[:] for row in board2]
    r, c = random.randint(0, 3), random.randint(0, 3)
    _, start_u, _, _ = get_stats(current, gen)
    best_char = current[r][c]
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        current[r][c] = char
        _, uni, _, _ = get_stats(current, gen)
        if uni > start_u:
            start_u = uni
            best_char = char
    current[r][c] = best_char
    t2, u2, r2, s2 = get_stats(current, gen)
    print_board("Scenario 2 (IO Optimized + Bonus)", current, t2, u2, r2, s2, bonus2)
