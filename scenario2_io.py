import time
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

def solve(board):
    gen = BoardGenerator()
    all_words = gen._solve_board(board, 'NWL', (0, 99999), 3)
    return set(all_words.keys())

def get_stats(board):
    words = solve(board)
    total = len(words)
    uniques = len(words.intersection(UNIQUE_NWL))
    ratio = uniques / total if total > 0 else 0
    return total, uniques, ratio

if __name__ == '__main__':
    print("Starting IO Optimization (Scenario 2)...")
    start_time = time.time()
    
    # 1. Start with a "crappy" board
    while True:
        board = [[random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)] for _ in range(4)]
        total, u, r = get_stats(board)
        if 50 <= total <= 100:
            print(f"Initial: {r:.2%} ({u}/{total})")
            break
            
    # 2. Iterate positions
    positions = [(r, c) for r in range(4) for c in range(4)]
    random.shuffle(positions)
    
    current_board = [row[:] for row in board]
    for idx, (r, c) in enumerate(positions):
        best_char = current_board[r][c]
        current_total, current_uni, current_ratio = get_stats(current_board)
        
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            current_board[r][c] = char
            t, u, r_ratio = get_stats(current_board)
            if u > current_uni:
                current_uni = u
                best_char = char
        
        current_board[r][c] = best_char
        stat_t, stat_u, stat_r = get_stats(current_board)
        print(f"Step {idx+1}: {stat_r:.2%} ({stat_u}/{stat_t})")
        if stat_r >= 0.7:
             print(f"\nSUCCESS! Found 70%+ board in {time.time() - start_time:.2f}s")
             break
    
    elapsed = time.time() - start_time
    print(f"\nOptimization Finished in {elapsed:.2f}s")
    print(f"Final Ratio: {stat_r:.2%} (Words: {stat_t})")
