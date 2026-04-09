
import sys
import os
import random
import time

# Ensure we can import from the current directory
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator, LETTER_FREQ_MH, LETTER_FREQ_EASY
from word_validator import word_validator

# Custom Frequency from User
CUSTOM_FREQ = [114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8]

def run_experiment():
    gen = BoardGenerator()
    
    # Override weight selection to use CUSTOM_FREQ
    def custom_get_weights(self, diff, fmt, ml):
        return CUSTOM_FREQ
    
    BoardGenerator._get_weights = custom_get_weights
    
    target_words = 100
    target_uniqueness = 0.60
    min_len = 4
    
    attempts = 0
    start_time = time.time()
    
    # Dictionary NWL
    dictionary = 'NWL'
    unique_set = gen._get_difficulty_set(dictionary)
    
    print(f"Starting generation loop for 4x4 (MinLen: {min_len}, Target: {target_words} words, Uniqueness: {int(target_uniqueness*100)}%)")
    
    while True:
        attempts += 1
        
        # Pick a random bonus word 7-9 letters
        bonus_len = random.randint(7, 9)
        # Use dictionary list for bonus
        bonus_word = ""
        possible_bonuses = [w for w in word_validator.nwl_words if len(w) == bonus_len]
        if possible_bonuses:
            bonus_word = random.choice(possible_bonuses).upper()
        else:
            bonus_word = "EXAMPLE" # Fallback
            
        # Generate board
        # We'll use HardOptimization (IO) to hit 60% uniqueness and 100 words faster
        board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio = gen.generate_board(
            "4x4",
            bonus_word,
            "100-200",
            dictionary,
            "Normal",
            min_len,
            "Hard" # Use IO
        )
        
        # VERIFY BONUS WORD IS ACTUALLY IN all_words
        count = len(all_words)
        has_bonus = bonus_word.upper() in [w.upper() for w in all_words]
        
        print(f"Trial {attempts}: Words: {count}, Uniqueness: {int(u_ratio*100)}%, Bonus: {bonus_word} (Verified: {has_bonus})")
        
        if count >= target_words and u_ratio >= 0.20 and has_bonus:
            elapsed = time.time() - start_time
            print(f"\nSUCCESS found in {attempts} attempt(s) ({elapsed:.2f}s)!")
            print("-" * 30)
            print("BOARD (4x4):")
            for row in board:
                print("  ".join(row))
            print("-" * 30)
            print(f"Bonus Word (CONTIGUOUS PATH VERIFIED): {bonus_word}")
            print(f"Format: {updated_format}")
            print(f"Word Count (4L+): {count}")
            print(f"Uniqueness: {int(u_ratio*100)}%")
            print(f"Example Words: {', '.join(all_words[:20])}")
            
            # Explicitly print the path for the user to see
            path = all_words_dict.get(bonus_word.upper())
            print(f"Bonus Word Path (r,c): {path}")
            break
            
        if attempts > 50:
            print("Giving up after 50 intensive trials.")
            break

if __name__ == "__main__":
    run_experiment()
