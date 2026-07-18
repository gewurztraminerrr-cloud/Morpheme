#!/usr/bin/env python3
"""Refill pregenerated board caches for large and 3D dimensions (6x8, 5x7, 3x3x3) in the background with low CPU and memory footprint."""
import sys
sys.stdout.reconfigure(line_buffering=True)
import os
import json
import time
import sqlite3
import random
import gc

# Add parent directory to path so we can import project modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from board_generator import BoardGenerator, serialize_param_key
from spinner_set import SpinnerSet

print("Initializing BoardGenerator...")
bg = BoardGenerator()

db_path = os.path.join(parent_dir, 'morpheme.db')
print(f"Database path: {db_path}")

# Target counts for each dimension in cache
TARGETS = {
    '6x8': 200,
    '5x7': 150,
    '3x3x3': 80
}

def get_random_bonus_word(dictionary, length):
    filename = 'CSW.txt' if str(dictionary).upper() == 'CSW' else 'NWL.txt'
    filepath = os.path.join(parent_dir, 'dictionaries', filename)
    if not os.path.exists(filepath):
        return "MATRIX"
    
    candidates = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                word = line.strip()
                if len(word) == length:
                    candidates.append(word)
    except Exception as e:
        print(f"Error reading dictionary file {filename}: {e}")
        
    return random.choice(candidates) if candidates else "MATRIX"

def get_cache_counts():
    counts = {d: 0 for d in TARGETS}
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()
        cursor.execute("SELECT param_key FROM pregenerated_boards")
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            try:
                p = json.loads(row[0])
                d = p.get('dimensions')
                if d in counts:
                    counts[d] += 1
            except:
                pass
    except Exception as e:
        print(f"Error querying cache counts: {e}")
    return counts

def generate_one_board(dimensions):
    # Use SpinnerSet to generate standard, sanitized parameters for this dimension
    # Default is_24h = False, but randomly make 10% of them Valued Letters / 24h style format for diversity
    is_24h = (random.random() < 0.1)
    params = SpinnerSet.generate_params(dimensions, is_24h=is_24h)
    
    # 6x8 and 3x3x3 clamps min_word_length to 6
    min_len = int(params.get('min_word_length', 3))
    if dimensions in ['6x8', '3x3x3'] and min_len < 6:
        min_len = 6
        params['min_word_length'] = 6
    
    dictionary = params.get('dictionary', 'NWL')
    wc_range = params.get('word_count_range', '100-200')
    fmt = params.get('board_format', 'Normal')
    difficulty = params.get('difficulty', 'Medium')
    
    # Select a random bonus word from the chosen dictionary using low-memory file scan
    bonus_word_len = max(6, min_len)
    bonus_word = get_random_bonus_word(dictionary, bonus_word_len)
    
    print(f"Generating {dimensions} board: dict={dictionary}, format={fmt}, min_len={min_len}, range={wc_range}, bonus={bonus_word}")
    
    res = bg.generate_board(
        dimensions=dimensions,
        bonus_word=bonus_word,
        word_count_range=wc_range,
        dictionary=dictionary,
        board_format=fmt,
        min_word_length=min_len,
        difficulty=difficulty,
        is_emergency=False,
        timeout=30.0
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
            dimensions, final_bonus_word or bonus_word, wc_range, dictionary, fmt, min_len, difficulty, use_added_words=False
        )
        
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            conn.execute(
                "INSERT INTO pregenerated_boards (param_key, board_json, created_at) VALUES (?, ?, ?);",
                (param_key_str, json.dumps(board_data), time.time())
            )
            conn.commit()
            conn.close()
            print(f"--> Successfully saved to cache DB.")
            return True
        except Exception as e:
            print(f"--> Failed to insert to DB: {e}")
    else:
        print(f"--> Generation returned no results or failed.")
    return False

def main():
    print("Morpheme Cache Refiller Daemon started.")
    while True:
        try:
            counts = get_cache_counts()
            print(f"Current Cache Counts: {counts} (Targets: {TARGETS})")
            
            # Find dimensions that are below target
            under_targets = [d for d, target in TARGETS.items() if counts[d] < target]
            
            if not under_targets:
                print("All caches are fully populated. Sleeping for 60 seconds...")
                time.sleep(60.0)
                continue
                
            # Pick the dimension that is furthest below its target (percentage-wise)
            # This ensures we prioritize the most starved cache
            starved_dim = min(under_targets, key=lambda d: counts[d] / TARGETS[d])
            
            success = generate_one_board(starved_dim)
            
            # Force GC collect to release memory
            gc.collect()
            
            # Throttling sleep to prevent CPU warnings on 1-core VPS
            sleep_time = 10.0 if success else 3.0
            print(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            print("Refiller exiting.")
            break
        except Exception as e:
            print(f"Unexpected loop error: {e}")
            time.sleep(10.0)

if __name__ == "__main__":
    main()
