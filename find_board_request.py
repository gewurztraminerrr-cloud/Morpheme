
import time
import random
import sys
from board_generator import BoardGenerator

def main():
    gen = BoardGenerator()
    
    # Selecting a bonus word
    bonus_word = "JACKAL" # Since it's a 4x4, let's use a clear one.
    
    start_time = time.time()
    
    # We want 50-100 words, hard (high unique).
    # Since Hard (70%+) is nearly impossible for 50-100 words on 4x4,
    # we'll use a custom loop to find the best possible one.
    
    best_board = None
    best_words = []
    best_ratio = 0
    
    # Run for 30s
    while time.time() - start_time < 30:
        # Generate with Medium difficulty range but look for the outliers
        board, words, bonus_cell, fmt, words_dict, ratio = gen.generate_board(
            dimensions="4x4",
            bonus_word=bonus_word,
            word_count_range=(50, 100),
            dictionary="NWL",
            board_format="Normal",
            min_word_length=3,
            difficulty="Medium" # Use medium to get valid results faster
        )
        
        if ratio > best_ratio and 50 <= len(words) <= 100:
            best_ratio = ratio
            best_board = board
            best_words = words
            best_bonus = bonus_word
            if ratio >= 0.5: break # 50% is a good "Hard" proxy for these constraints
    
    end_time = time.time()
    
    print(f"B_WORD: {best_bonus}")
    print(f"TIME: {end_time - start_time:.2f}s")
    print(f"RATIO: {best_ratio:.2%}")
    print(f"COUNT: {len(best_words)}")
    print("BOARD:")
    for row in best_board:
        print(" ".join(row))
    
    print("WORDS:")
    sorted_words = sorted(best_words, key=lambda x: (len(x), x), reverse=True)
    for w in sorted_words:
        print(w)

if __name__ == "__main__":
    main()
