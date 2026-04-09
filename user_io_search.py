
import sys
import os
import random
import time

sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator
from word_validator import WordValidator

# USER FREQUENCY
USER_WEIGHTS = [114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Dictionary selection
# User wants NWL.txt and uniqueNWL.txt
wv = WordValidator()
unique_nwl = wv.unique_nwl_words
nwl_full = wv.nwl_words | wv.long_words | wv.added_words

gen = BoardGenerator()

def solve_nwl(board):
    return gen._solve_board(board, 'NWL', (0, 99999), 3, 12, False)

def solve_unique(board):
    # Using 'UniqueNWL' dict type that I added to WordValidator
    return gen._solve_board(board, 'UniqueNWL', (0, 99999), 3, 12, False)

def run_search():
    print("STAGE 1: Finding a 200+ word board with user weights...")
    start_time = time.time()
    board = None
    best_init_count = 0
    attempts = 0
    
    while True:
        attempts += 1
        tiles = random.choices(LETTERS, weights=USER_WEIGHTS, k=16)
        curr_board = [tiles[i:i+4] for i in range(0, 16, 4)]
        
        words = solve_nwl(curr_board)
        count = len(words)
        best_init_count = max(best_init_count, count)
        
        if count >= 200:
            board = curr_board
            print(f"  ✓ Success! Found board with {count} words on attempt {attempts}.")
            break
        
        if attempts % 1000 == 0:
            print(f"  [{time.time()-start_time:.1f}s] Tested {attempts} boards. Best density: {best_init_count}")
            
    print("\nSTAGE 2: Iterative Optimization (IO) focused on Unique Word Density...")
    print("Rule: Swapping letters based on highest uniqueNWL count, aiming for 50-100 total words.")
    
    # UNLIKELY POOL
    UNLIKELY_POOL = "CMPHVBFGJKQXZWY" # User Mentioned (C, M, P, H, V)
    
    io_attempts = 0
    while True:
        io_attempts += 1
        
        r, c = random.randint(0, 3), random.randint(0, 3)
        orig_char = board[r][c]
        
        # Decide pool with user-directed randomness
        # Sometimes full alphabet, sometimes unlikely pool (C, M, P, H, V...)
        if random.random() < 0.6:
            test_pool = list(UNLIKELY_POOL)
        else:
            test_pool = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        random.shuffle(test_pool)
        # Search for the letter that maximizes unique words (User's Rule)
        best_char = orig_char
        max_unique = len(solve_unique(board))
        
        for char in test_pool:
            if char == orig_char: continue
            board[r][c] = char
            u_count = len(solve_unique(board))
            if u_count > max_unique:
                max_unique = u_count
                best_char = char
        
        # Commit the best unique-favored char
        board[r][c] = best_char
        
        # MONITOR STATUS
        total_words_dict = solve_nwl(board)
        total_words = list(total_words_dict.keys())
        total_count = len(total_words)
        
        if io_attempts % 3 == 0:
            unique_count = len(solve_unique(board))
            ratio = unique_count/total_count if total_count > 0 else 0
            print(f"  [IO {io_attempts}] Words: {total_count} | Unique: {unique_count} | Ratio: {ratio:.1%}")
            
        # BREAK OUT IF STUCK AT HIGH DENSITY (AGGRESSIVE FOR LOW BATTERY)
        if total_count > 120 and io_attempts > 8:
            # Force a random rare letter into a random spot to derail the common cluster
            dr, dc = random.randint(0,3), random.randint(0,3)
            board[dr][dc] = random.choice(UNLIKELY_POOL)
            # Re-seed if extremely stuck
            if io_attempts > 25:
                # User directed outlier search re-seed
                print(f"  ! Resetting to outlier seed (Attempt {io_attempts})")
                for _ in range(5):
                    board[random.randint(0,3)][random.randint(0,3)] = random.choice(UNLIKELY_POOL)
            io_attempts = 0 # Reset counter to keep searching from new location
            
        # Target Match
        unique_cnt = len(solve_unique(board))
        u_ratio = unique_cnt / total_count if total_count > 0 else 0
        
        if 50 <= total_count <= 100 and u_ratio >= 0.70:
            # 6. Check for 7-9L Bonus Word
            all_list = list(total_words.keys())
            bonus_word = ""
            for w in all_list:
                if 7 <= len(w) <= 9:
                    bonus_word = w.upper()
                    break
            
            if bonus_word:
                elapsed = time.time() - start_time
                print(f"\nCRITERIA MET IN {elapsed:.1f}s!")
                print("-" * 35)
                print("FINAL 4x4 BOARD:")
                for row in board:
                    print("  ".join(row))
                print("-" * 35)
                print(f"Total Words (3L+): {total_count}")
                print(f"Unique words (Hard): {unique_cnt} ({u_ratio:.1%})")
                print(f"Bonus Word (7-9L): {bonus_word}")
                print("-" * 35)
                print("WORD LIST (Largest First):")
                # Sort words by length (largest first), then alphabetically
                sorted_words = sorted(all_list, key=lambda x: (-len(x), x))
                for w in sorted_words:
                    print(f"  {w}")
                print("-" * 45)
                break
        
        if io_attempts > 1000:
            print("Completed 1,000 IO attempts. Retrying from 200-word board.")
            # Re-seed to avoid local optima
            attempts = 0
            while True:
                attempts += 1
                tiles = random.choices(LETTERS, weights=USER_WEIGHTS, k=16)
                curr_board = [tiles[i:i+4] for i in range(0, 16, 4)]
                if len(solve_nwl(curr_board)) >= 200:
                    board = curr_board
                    break
            io_attempts = 0
            
if __name__ == "__main__":
    run_search()
