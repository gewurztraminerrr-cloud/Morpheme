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

if __name__ == '__main__':
    print(f"Starting Stress Test for Scenario 1 (Target: 70% unique words)...")
    start_time = time.time()
    best_ratio = 0
    attempts = 0
    
    # Run for 1 minute or 10,000 attempts
    while time.time() - start_time < 60 and attempts < 5000:
        attempts += 1
        board = [[random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)] for _ in range(4)]
        words = solve(board)
        total = len(words)
        if total == 0: continue
        uniques = len(words.intersection(UNIQUE_NWL))
        ratio = uniques / total
        
        if ratio > best_ratio:
            best_ratio = ratio
            print(f"New best: {best_ratio:.2%} ({uniques}/{total}) at attempt {attempts}")
            
        if ratio >= 0.7:
             elapsed = time.time() - start_time
             print(f"\nSUCCESS! Found 70%+ board in {elapsed:.2f}s at attempt {attempts}")
             sys.exit(0)

    print(f"\nFAILED to find 70%+ board after {time.time() - start_time:.2f}s and {attempts} attempts.")
    print(f"Best ratio found: {best_ratio:.2%}")
