import time
import random
import os
import sys

# Add project path to sys.path to import local modules
sys.path.insert(0, '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator

def benchmark(dimensions, bonus, word_range, dictionary, difficulty, min_len):
    gen = BoardGenerator()
    print(f"\nBenchmarking: {dimensions} | {word_range} | {difficulty} | {min_len} | {dictionary}")
    
    # Test Scenario 1: Re-roll (Limit 15s)
    start1 = time.time()
    s1_success = False
    attempts1 = 0
    while time.time() - start1 < 15:
        attempts1 += 1
        try:
            # We use a fresh generator call to simulate the app's behavior
            board, words, bonus_c, fmt, dict_found = gen.generate_board(dimensions, bonus, word_range, dictionary, 'Normal', min_len, difficulty)
            s1_success = True
            break
        except Exception:
            continue
    elapsed1 = time.time() - start1
    
    # Scenario 2: IO (if word count is high or difficulty is Hard)
    # We'll just report based on the generate_board's internal decision making and speed
    
    print(f"  Result: Success={s1_success}, Time={elapsed1:.2f}s, Attempts={attempts1}")
    return s1_success, elapsed1

if __name__ == '__main__':
    # Mapping representative targets
    targets = [
        ("4x4", "ASTONISH", "50-100", "NWL", "Easy", 3),     # Easy Baseline
        ("4x4", "IDONEITY", "50-100", "NWL", "Hard", 3),     # Hard 4x4 (The hardest)
        ("6x8", "HYDRATOR", "200+", "NWL", "Medium", 6),     # Dense Large
        ("4x6", "TEMPESTS", "100-200", "CSW", "Hard", 4)     # Mid-range Hard
    ]
    
    for t in targets:
        benchmark(*t)
