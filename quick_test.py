import time
import random
import os
import sys

# Add project path to sys.path to import local modules
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator
from word_validator import word_validator

# Pre-load dictionaries
UNIQUE_NWL = set()
path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/uniqueNWL.txt'
if os.path.exists(path):
    with open(path, 'r') as f:
        UNIQUE_NWL = set(line.strip().upper() for line in f if line.strip())

def solve(board, min_len=3):
    gen = BoardGenerator()
    all_words = gen._solve_board(board, 'NWL', (0, 99999), min_len)
    return set(all_words.keys())

def get_stats(board):
    words = solve(board)
    total = len(words)
    uniques = len(words.intersection(UNIQUE_NWL))
    ratio = uniques / total if total > 0 else 0
    return total, uniques, ratio

if __name__ == '__main__':
    # 1. Random Sample (to simulate Scenario 1)
    results_s1 = []
    print("Testing Scenario 1 (10 Random Attempts)...")
    for _ in range(10):
        board = [[random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)] for _ in range(4)]
        results_s1.append(get_stats(board))
    
    # 2. IO Sample (Scenario 2 - 5 positions)
    print("Testing Scenario 2 (IO Process)...")
    board = [[random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)] for _ in range(4)]
    initial_stats = get_stats(board)
    current_board = [row[:] for row in board]
    
    positions = [(r, c) for r in range(4) for c in range(4)]
    random.shuffle(positions)
    
    io_steps = []
    # Test just 5 positions to be fast
    for i in range(min(5, len(positions))):
        r, c = positions[i]
        best_char = current_board[r][c]
        _, max_uni, _ = get_stats(current_board)
        
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            current_board[r][c] = char
            _, uni, _ = get_stats(current_board)
            if uni > max_uni:
                max_uni = uni
                best_char = char
        
        current_board[r][c] = best_char
        io_steps.append(get_stats(current_board))

    print("\nREPORT:")
    print("Scenario 1 (Random): Average Ratio={:.2%}, Best Ratio={:.2%}".format(
        sum(r[2] for r in results_s1)/len(results_s1),
        max(r[2] for r in results_s1)
    ))
    print("Scenario 2 (IO): Initial Ratio={:.2%}, Final Ratio={:.2%}".format(
        initial_stats[2],
        io_steps[-1][2]
    ))
    print("Best Unique Count found in IO: ", max(s[1] for s in io_steps))
