import time
import random
import os
import sys

# Add project path to sys.path to import local modules
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator
from word_validator import word_validator

# Pre-load dictionaries
print("Loading dictionaries...")
UNIQUE_NWL = set()
path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/uniqueNWL.txt'
if os.path.exists(path):
    with open(path, 'r') as f:
        UNIQUE_NWL = set(line.strip().upper() for line in f if line.strip())

def solve(board, min_len=3):
    # Fast solver for simulation
    rows, cols = 4, 4
    found = set()
    
    def dfs(r, c, visited, word):
        if len(word) >= min_len and word_validator.is_valid_word(word, 'NWL'):
            found.add(word)
        if len(word) >= 10: return
        
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0: continue
                nr, nc = r+dr, c+dc
                if 0<=nr<4 and 0<=nc<4 and (nr,nc) not in visited:
                    char = board[nr][nc]
                    dfs(nr, nc, visited | {(nr,nc)}, word + char)
                    if char == 'Q':
                        dfs(nr, nc, visited | {(nr,nc)}, word + 'QU')

    for r in range(4):
        for c in range(4):
            dfs(r, c, {(r,c)}, board[r][c])
            if board[r][c] == 'Q':
                dfs(r, c, {(r,c)}, 'QU')
    return found

def get_stats(board):
    words = solve(board)
    total = len(words)
    uniques = len(words.intersection(UNIQUE_NWL))
    ratio = uniques / total if total > 0 else 0
    return total, uniques, ratio

def run_scenario_1():
    print("\n[Scenario 1] Starting Bruteforce/Re-roll...")
    start_time = time.time()
    attempts = 0
    while time.time() - start_time < 60:
        attempts += 1
        # Create random board
        board = [[random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)] for _ in range(4)]
        total, uniques, ratio = get_stats(board)
        
        if 50 <= total <= 100 and ratio >= 0.7:
            return True, time.time() - start_time, attempts, total, uniques, ratio
            
    return False, 60, attempts, 0, 0, 0

def run_scenario_2():
    print("\n[Scenario 2] Starting IO Optimization...")
    start_time = time.time()
    # 1. Start with a "crappy" board (50-100 words)
    board = None
    while True:
        temp_board = [[random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)] for _ in range(4)]
        total, _, _ = get_stats(temp_board)
        if 50 <= total <= 100:
            board = temp_board
            break
            
    # 2. Iterate positions to optimize for UNIQUES
    positions = [(r, c) for r in range(4) for c in range(4)]
    random.shuffle(positions)
    
    for r, c in positions:
        best_char = board[r][c]
        _, max_uniques, _ = get_stats(board)
        
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            board[r][c] = char
            _, u_count, _ = get_stats(board)
            if u_count > max_uniques:
                max_uniques = u_count
                best_char = char
        
        board[r][c] = best_char
        total, uniques, ratio = get_stats(board)
        if 50 <= total <= 100 and ratio >= 0.7:
            return True, time.time() - start_time, total, uniques, ratio

    total, uniques, ratio = get_stats(board)
    return False, time.time() - start_time, total, uniques, ratio

if __name__ == '__main__':
    s1_success, s1_time, s1_attempts, s1_tot, s1_uni, s1_rat = run_scenario_1()
    s2_success, s2_time, s2_tot, s2_uni, s2_rat = run_scenario_2()
    
    print("\n" + "="*40)
    print("FINAL COMPARISON")
    print("="*40)
    print(f"Scenario 1 (Re-roll):")
    print(f"  Success: {s1_success}")
    print(f"  Time:    {s1_time:.2f}s")
    print(f"  Attempts: {s1_attempts}")
    print(f"  Ratio:   {s1_rat:.2%}")
    print(f"  Uniques: {s1_uni} of {s1_tot}")
    
    print(f"\nScenario 2 (IO):")
    print(f"  Success: {s2_success}")
    print(f"  Time:    {s2_time:.2f}s")
    print(f"  Ratio:   {s2_rat:.2%}")
    print(f"  Uniques: {s2_uni} of {s2_tot}")
    print("="*40)
