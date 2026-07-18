from board_generator import BoardGenerator
import time

def test_round(dictionary, range_str, dims, bonus):
    bg = BoardGenerator()
    start = time.time()
    res = bg.generate_board(
        dimensions=dims,
        bonus_word=bonus,
        word_count_range=range_str,
        dictionary=dictionary,
        board_format="Normal",
        min_word_length=4,
        difficulty="Medium",
        is_emergency=False,
        use_added_words=False
    )
    elapsed = time.time() - start
    board, all_words, bonus_cell, board_format_ret, all_words_dict, ratio, final_bonus_word = res
    print(f"[{dictionary} | {range_str}] Generated {len(all_words)} words in {elapsed:.2f}s. Uniqueness: {ratio:.2f}. Bonus: {final_bonus_word}")
    
def main():
    print("Testing IO-Base procedure...")
    test_round("NWL", "300-400", "4x6", "TEST")
    test_round("CSW", "500+", "4x6", "TESTING")

if __name__ == "__main__":
    main()
