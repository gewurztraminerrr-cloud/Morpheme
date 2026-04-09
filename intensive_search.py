
import sys
import os
import random
import time

sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator
from word_validator import word_validator

# USER FREQUENCY
CUSTOM_WEIGHTS = [
    114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8
]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def run_search():
    gen = BoardGenerator()
    dictionary = 'NWL'
    unique_set = gen._get_difficulty_set(dictionary)
    
    attempts = 0
    start_time = time.time()
    
    max_density = 0
    max_unique_at_100 = 0
    max_overall_unique = 0
    
    print(f"SEARCHING FOR STATISTICAL OUTLIER")
    print(f"Criteria: 4x4, 4L MIN, 100+ WORDS, 60%+ UNIQUE")
    print(f"Using exactly the provided frequency weights.")
    
    while True:
        attempts += 1
        
        # 1. GENERATE BOARD FROM WEIGHTS
        board_tiles = random.choices(LETTERS, weights=CUSTOM_WEIGHTS, k=16)
        board = [board_tiles[i:i+4] for i in range(0, 16, 4)]
        
        # 2. SOLVE IT (FAST) 
        all_words_dict = gen._solve_board(board, dictionary, (0, 99999), 4, 12, False)
        if not all_words_dict: continue
        
        all_words = list(all_words_dict.keys())
        total = len(all_words)
        
        # 3. GET UNIQUENESS RATIO
        count_unique = sum(1 for w in all_words if w.upper() in unique_set)
        u_ratio = count_unique / total if total > 0 else 0
        uniqueness_pct = int(u_ratio * 100)
        
        # Update metrics
        max_density = max(max_density, total)
        max_overall_unique = max(max_overall_unique, uniqueness_pct)
        if total >= 100:
            max_unique_at_100 = max(max_unique_at_100, uniqueness_pct)
        
        # 4. Check word count and uniqueness
        if total >= 100 and u_ratio >= 0.60:
            # 5. RE-SOLVE WITH PATHS 
            all_words_dict_full = gen._solve_board(board, dictionary, (0, 99999), 4, 12, True)
            all_words_full = list(all_words_dict_full.keys())
            
            # FINAL CHECK: DO WE HIT 100 AND 60?
            if len(all_words_full) >= 100 and (sum(1 for w in all_words_full if w.upper() in unique_set)/len(all_words_full)) >= 0.60:
                # 6. CHECK FOR 7-9L BONUS PATH
                bonus_word = ""
                for w in all_words_full:
                    if 7 <= len(w) <= 9:
                        bonus_word = w.upper()
                        break
                
                if bonus_word:
                    elapsed = time.time() - start_time
                    print(f"\nCRITERIA MET IN {attempts} ATTEMPTS ({elapsed:.1f}s)!")
                    print("-" * 35)
                    print("FINAL 4x4 BOARD:")
                    for row in board:
                        print("  ".join(row))
                    print("-" * 35)
                    print(f"Bonus Word (Contiguous Path): {bonus_word}")
                    print(f"Bonus Path (r,c): {all_words_dict_full.get(bonus_word)}")
                    print(f"Total Words (4L+): {len(all_words_full)}")
                    print(f"Uniqueness Ratio: {int(u_ratio*100)}%")
                    print(f"Unique Words: {', '.join([w for w in all_words_full if w.upper() in unique_set][:15])}")
                    print("-" * 35)
                    break
        
        # LOGGING EVERY 500 TRIALS
        if attempts % 500 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Tested {attempts} boards. No 60%/100+ hit yet.")
            print(f"   --> Max Density: {max_density} words (4L+)")
            print(f"   --> Max Uniqueness on a 100+ board: {max_unique_at_100}%")
            print(f"   --> Max Overall Uniqueness (Any Count): {max_overall_unique}%")
            
        if attempts > 1000000:
            print("Completed 1,000,000 attempts.")
            break

if __name__ == "__main__":
    run_search()
