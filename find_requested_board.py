
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

from board_generator import BoardGenerator

def main():
    gen = BoardGenerator()
    
    # Parameters: 4x4 - 100-200 words - hard - 4LM - NWL
    dimensions = "4x4"
    bonus_word = "" # No bonus word specified in the prompt
    word_count_range = (100, 200)
    dictionary = "NWL"
    board_format = "Normal"
    min_word_length = 4
    difficulty = "Hard"
    
    print(f"Generating board: {dimensions}, {word_count_range} words, difficulty={difficulty}, min_len={min_word_length}, dict={dictionary}")
    
    board, all_words, bonus_cell, fmt, all_words_dict, ratio = gen.generate_board(
        dimensions, bonus_word, word_count_range, dictionary, board_format, min_word_length, difficulty
    )
    
    print("\n" + "="*40)
    print("GENERATED BOARD")
    print("="*40)
    for row in board:
        print(" ".join(row))
    print("="*40)
    print(f"Word Count: {len(all_words)}")
    print(f"Uniqueness Ratio: {ratio:.1%}")
    print(f"Dictionary: {dictionary}")
    print(f"Min Word Length: {min_word_length}")
    print("="*40)
    print("Top 20 Words:")
    sorted_words = sorted(all_words, key=lambda x: (-len(x), x))
    for w in sorted_words[:20]:
        print(f"  {w}")
    print("="*40)

if __name__ == "__main__":
    main()
