import sqlite3
import json

conn = sqlite3.connect('morpheme.db')
cursor = conn.cursor()

formats = ["Normal", "Checkerboard", "Equality Freq", "Mania", "Either/Or"]
print("Querying pregenerated board counts from production...")

for fmt in formats:
    # Construct the JSON parameter key
    param_key_dict = {
        "board_format": fmt,
        "bonus_word_len": 8,
        "dictionary": "AW",
        "difficulty": "Easy",
        "dimensions": "4x6",
        "min_word_length": 4,
        "word_count_range": "200-300"
    }
    param_key_str = json.dumps(param_key_dict, sort_keys=True)
    
    cursor.execute("SELECT COUNT(*) FROM pregenerated_boards WHERE param_key = ?;", (param_key_str,))
    count = cursor.fetchone()[0]
    print(f"Format '{fmt}': {count} boards cached.")

conn.close()
