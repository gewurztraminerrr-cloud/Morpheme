import time
import random
import os
import sys

# Add project path to sys.path to import local modules
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator
from word_validator import word_validator

def load_difficulty_dict(dict_name):
    path = f'/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/{dict_name}.txt'
    if not os.path.exists(path):
        return set()
    with open(path, 'r') as f:
        return set(line.strip().upper() for line in f if line.strip())

def solve_for_uniques(board, difficulty_set, min_len=3):
    # Use a simplified solver for speed in simulation
    gen = BoardGenerator()
    all_words = gen._solve_board(board, 'NWL', (0, 99999), min_len)
    all_found = set(all_words.keys())
    unique_found = all_found.intersection(difficulty_set)
    return all_found, unique_found

def scenario_1(difficulty_set, bonus_words, target_range=(50, 100), target_ratio=0.7):
    print("\n--- Running Scenario 1 (Re-rolling) ---")
    gen = BoardGenerator()
    start_time = time.time()
    attempts = 0
    
    while time.time() - start_time < 60:
        attempts += 1
        bonus = random.choice(bonus_words)
        # Generate a board (Normal 4x4)
        board, words, _, _, _ = gen.generate_board('4x4', bonus, target_range, 'NWL', 'Normal', 3)
        
        all_found, unique_found = solve_for_uniques(board, difficulty_set)
        total = len(all_found)
        if total == 0: continue
        
        ratio = len(unique_found) / total
        if ratio >= target_ratio and target_range[0] <= total <= target_range[1]:
            elapsed = time.time() - start_time
            return {
                'success': True,
                'time': elapsed,
                'attempts': attempts,
                'total_words': total,
                'unique_words': len(unique_found),
                'ratio': ratio
            }
            
    return {'success': False, 'time': 60, 'attempts': attempts}

def scenario_2(difficulty_set, target_range=(50, 100)):
    print("\n--- Running Scenario 2 (IO Optimization) ---")
    gen = BoardGenerator()
    start_time = time.time()
    
    # 1. Start with a "crappy" board
    board = gen._create_normal_board(4, 4, [1]*26)
    all_found, unique_found = solve_for_uniques(board, difficulty_set)
    
    # 2. Perform IO on positions
    positions = [(r, c) for r in range(4) for c in range(4)]
    random.shuffle(positions)
    
    best_ratio = 0
    final_total = 0
    final_unique = 0
    
    for r, c in positions:
        best_char = board[r][c]
        max_unique_count = len(unique_found)
        
        for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            board[r][c] = char
            _, current_uniques = solve_for_uniques(board, difficulty_set)
            if len(current_uniques) > max_unique_count:
                max_unique_count = len(current_uniques)
                best_char = char
        
        board[r][c] = best_char
        # Check current state
        all_found, unique_found = solve_for_uniques(board, difficulty_set)
        final_total = len(all_found)
        final_unique = len(unique_found)
        if final_total > 0:
            best_ratio = final_unique / final_total
            if best_ratio >= 0.7 and target_range[0] <= final_total <= target_range[1]:
                break

    elapsed = time.time() - start_time
    return {
        'success': best_ratio >= 0.7,
        'time': elapsed,
        'total_words': final_total,
        'unique_words': final_unique,
        'ratio': best_ratio
    }

if __name__ == '__main__':
    print("Loading dictionaries...")
    unique_nwl = load_difficulty_dict('uniqueNWL')
    print(f"Loaded {len(unique_nwl)} unique NWL words.")
    
    # Sample bonus words (8 letters)
    bonus_samples = ["HYDRATOR", "GREGATIM", "TEMPESTS", "UNIFYING", "MANIKINS"]
    
    res1 = scenario_1(unique_nwl, bonus_samples)
    res2 = scenario_2(unique_nwl)
    
    print("\nRESULTS:")
    print(f"Scenario 1: Success={res1['success']}, Time={res1['time']:.2f}s, Ratio={res1.get('ratio', 0):.2%}, Uniques={res1.get('unique_words', 0)}")
    print(f"Scenario 2: Success={res2['success']}, Time={res2['time']:.2f}s, Ratio={res2.get('ratio', 0):.2%}, Uniques={res2.get('unique_words', 0)}")
