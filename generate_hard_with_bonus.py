
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

from board_generator import BoardGenerator

def main():
    gen = BoardGenerator()
    
    # Parameters provided: 4x4 - 100-200 words - hard - 4LM - NWL
    dimensions = "4x4"
    dictionary = "NWL"
    word_count_range = (100, 200)
    board_format = "Normal"
    min_word_length = 4
    difficulty = "Hard"
    
    # Generate a Hard-appropriate bonus word (6-8 letters)
    # The BoardGenerator has a _get_bonus_word logic but we can just use the internal one
    # For now let's use the gen method if available or roll our own
    import random
    bonus_pool = []
    unique_set = gen._get_difficulty_set(dictionary)
    bonus_pool = [w for w in unique_set if 6 <= len(w) <= 9]
    bonus_word = random.choice(bonus_pool) if bonus_pool else "OUTLIER"

    print(f"Generating HARD board (4x4, 4LM, NWL) with bonus word '{bonus_word}'...")
    
    # The BoardGenerator.generate_board handles embedding and optimizing
    board, all_words, bonus_cell, fmt, all_words_dict, ratio = gen.generate_board(
        dimensions, 
        bonus_word, 
        word_count_range, 
        dictionary, 
        board_format, 
        min_word_length, 
        difficulty
    )
    
    if board:
        print("\n" + "="*40)
        print("GENERATED HARD BOARD")
        print("="*40)
        for row in board:
            print(" ".join(row))
        print("="*40)
        print(f"BONUS WORD: {bonus_word}")
        print(f"WORD COUNT: {len(all_words)}")
        print(f"UNIQUENESS RATIO: {ratio:.1%}")
        print(f"MIN WORD LENGTH: {min_word_length}")
        print("="*40)
        
        # Show top words including the bonus word if found
        sorted_words = sorted(all_words, key=lambda x: (-len(x), x))
        print("Top 15 Words:")
        for w in sorted_words[:15]:
            print(f"  {w}")
    else:
        print("Failed to generate a board meeting criteria.")

if __name__ == "__main__":
    main()
