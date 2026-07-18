import sys
import os
import json
import time
import sqlite3

# Insert project path
sys.path.insert(0, '/home/morpheme/morpheme')

from board_generator import BoardGenerator, serialize_param_key

print("Initializing BoardGenerator...")
bg = BoardGenerator()

db_path = '/home/morpheme/morpheme/morpheme.db'
conn = sqlite3.connect(db_path, timeout=30)

# We want to generate 32 boards for 6x8
configs = [
    # (dictionary, board_format, min_word_length, word_count_range)
    ('NWL', 'Normal', 6, '100-200'),
    ('CSW', 'Normal', 6, '100-200'),
    ('NWL', 'Normal', 6, '200-300'),
    ('CSW', 'Normal', 6, '200-300'),
    ('NWL', 'Valued Letters', 6, '200-300'),
    ('CSW', 'Valued Letters', 6, '200-300'),
    ('NWL', 'Normal', 7, '100-200'),
    ('CSW', 'Normal', 7, '100-200'),
]

generated_count = 0
for dictionary, fmt, min_len, wc_range in configs:
    for i in range(4): # Generate 4 boards for each configuration
        try:
            print(f"Generating 6x8 board #{i+1} for: dict={dictionary}, format={fmt}, min_len={min_len}, range={wc_range}")
            # Determine bonus word length based on min_len
            bonus_word_len = max(6, min_len)
            
            # Select a random bonus word of this length from dictionary
            from word_validator import word_validator
            if dictionary == 'CSW':
                word_validator.ensure_csw_loaded()
                words = [w for w in word_validator.csw_words if len(w) == bonus_word_len]
            else:
                words = [w for w in word_validator.nwl_words if len(w) == bonus_word_len]
            
            import random
            bonus_word = random.choice(words) if words else "MATRIX"
            
            res = bg.generate_board(
                dimensions="6x8",
                bonus_word=bonus_word,
                word_count_range=wc_range,
                dictionary=dictionary,
                board_format=fmt,
                min_word_length=min_len,
                difficulty="Medium",
                is_emergency=False,
                timeout=15.0
            )
            
            if res and len(res) >= 7:
                board, all_words, bonus_cell, board_format_ret, all_words_dict, ratio, final_bonus_word = res[:7]
                board_data = {
                    "board": board,
                    "all_words": all_words,
                    "bonus_cell": list(bonus_cell) if bonus_cell else None,
                    "board_format_ret": board_format_ret,
                    "all_words_dict": all_words_dict,
                    "ratio": ratio,
                    "final_bonus_word": final_bonus_word
                }
                
                param_key_str = serialize_param_key(
                    "6x8", final_bonus_word or bonus_word, wc_range, dictionary, fmt, min_len, "Medium", use_added_words=False
                )
                
                conn.execute(
                    "INSERT INTO pregenerated_boards (param_key, board_json, created_at) VALUES (?, ?, ?);",
                    (param_key_str, json.dumps(board_data), time.time())
                )
                conn.commit()
                generated_count += 1
                print(f"--> Successfully generated and saved to DB (Total: {generated_count})")
        except Exception as e:
            print(f"--> Failed to generate: {e}")

conn.close()
print(f"Populated {generated_count} pregenerated 6x8 boards in DB.")
