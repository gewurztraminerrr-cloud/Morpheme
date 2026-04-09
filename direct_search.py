
import sys
import time
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')
from board_generator import BoardGenerator

gen = BoardGenerator()

def find_board():
    print("Searching for 4x4 Hard board (50-100 words, 70%+ uniqueness, NWL)...")
    start = time.time()
    
    # We use the gen.generate_board method which I just repaired.
    # It now uses LETTER_FREQ_USER and HardOptimization for 4x4 Hard boards.
    
    attempts = 0
    while True:
        attempts += 1
        # Use NWL, Hard, 3LM, 50-100 range
        res = gen.generate_board('4x4', 'NWL', '50-100', 'NWL', 'Standard', 3, 'Hard')
        if res:
            board, word_list, bonus_cell, fmt, words_dict, ratio = res
            
            # Double check criteria
            if 50 <= len(word_list) <= 100 and ratio >= 0.70:
                elapsed = time.time() - start
                print(f"\nCRITERIA MET IN {elapsed:.1f}s (Attempts: {attempts})!")
                print("-" * 35)
                for row in board:
                    print("  ".join(row))
                print("-" * 35)
                
                # Bonus word finding logic (7-9 letters)
                bonus_word = ""
                for w in sorted(word_list, key=len, reverse=True):
                    if 7 <= len(w) <= 9:
                        bonus_word = w
                        break
                
                print(f"Total Words: {len(word_list)}")
                print(f"Uniqueness: {ratio:.1%} (Hard)")
                print(f"Bonus Word: {bonus_word}")
                print("-" * 35)
                print("WORD LIST (Largest First):")
                for w in sorted(word_list, key=lambda x: (-len(x), x)):
                    print(f"  {w}")
                return
        
        if attempts % 10 == 0:
            print(f"  [{time.time()-start:.1f}s] Tested {attempts} optimized generation pulses...")

if __name__ == "__main__":
    find_board()
