
import sys
import random
import time
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')
from board_generator import BoardGenerator
from word_validator import word_validator

# User's weights
# 114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8
# A,   B,  C,  D,  E,   F,  G,  H,  I,   J, K,  L,  M,  N,  O,  P,  Q, R,  S,  T,  U,  V,  W,  X, Y,  Z
WEIGHTS = [114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

gen = BoardGenerator()

def find_hard_outlier():
    # To hit 70% uniqueness, we need rare letters. 
    # But the user provided weights are common-heavy.
    # We will search aggressively for 60 seconds.
    print("CRITICAL POWER SEARCH: Searching for 70%+ uniqueness outliers...")
    start = time.time()
    
    unique_set = gen._get_difficulty_set('NWL')
    
    attempts = 0
    while time.time() - start < 60:
        attempts += 1
        # Use user weights but sometimes skew them to rare ones to find the outlier faster
        current_weights = list(WEIGHTS)
        if random.random() < 0.8:
            # SKEW: Lower common vowel weights by 80% to find rare segments
            for idx in [0, 4, 8, 14, 20]: # A, E, I, O, U
                current_weights[idx] = max(1, int(current_weights[idx] * 0.2))
        
        tiles = random.choices(LETTERS, weights=current_weights, k=16)
        board = [tiles[i:i+4] for i in range(0, 16, 4)]
        
        words = gen._solve_board(board, 'NWL', (0, 99999), 3, max_depth=15)
        count = len(words)
        if 50 <= count <= 120:
            unique_count = sum(1 for w in words if w.upper() in unique_set)
            ratio = unique_count / count
            if ratio >= 0.70:
                print(f"SUCCESS in {time.time()-start:.1f}s!")
                # Print results
                print("-" * 35)
                for row in board:
                    print(" ".join(row))
                print("-" * 35)
                print(f"Total Words: {count}")
                print(f"Uniqueness: {ratio:.1%} (Hard)")
                
                # Bonus word find
                bw_list = sorted(list(words.keys()), key=len, reverse=True)
                bonus_word = ""
                for w in bw_list:
                    if 7 <= len(w) <= 9:
                        bonus_word = w
                        break
                print(f"Bonus Word: {bonus_word.upper()}")
                print("-" * 35)
                print("WORD LIST (Largest First):")
                for w in sorted(bw_list, key=lambda x: (-len(x), x)):
                    print(f"  {w}")
                return True
    
    print("FAILED TO FIND 70% OUTLIER IN 60S. TERMINATING SEARCH TO SAVE POWER.")
    return False

if __name__ == "__main__":
    find_hard_outlier()
