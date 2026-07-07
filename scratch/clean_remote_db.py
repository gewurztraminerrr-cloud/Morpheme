import sqlite3
import json
import os

# 1. Clean DB
db_path = '/home/morpheme/morpheme/morpheme.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, param_key FROM pregenerated_boards")
    rows = c.fetchall()
    to_delete = []
    for row_id, param_key in rows:
        try:
            params = json.loads(param_key)
            if params.get('dictionary') == 'AW':
                to_delete.append(row_id)
        except:
            pass
    if to_delete:
        print(f"Deleting {len(to_delete)} rows matching 'AW' dictionary...")
        c.executemany("DELETE FROM pregenerated_boards WHERE id = ?", [(rid,) for rid in to_delete])
        conn.commit()
    conn.close()

# 2. Filter dictionaries/added_words.txt
added_path = '/home/morpheme/morpheme/dictionaries/added_words.txt'
csw_path = '/home/morpheme/morpheme/dictionaries/CSW.txt'

if os.path.exists(added_path) and os.path.exists(csw_path):
    print("Filtering CSW words out of added_words.txt on remote server...")
    with open(csw_path, 'r') as f:
        csw_words = {line.strip().upper() for line in f if line.strip()}
    with open(added_path, 'r') as f:
        added_words = [line.strip().upper() for line in f if line.strip()]
        
    filtered = []
    seen = set()
    for w in added_words:
        if w not in csw_words and w not in seen:
            seen.add(w)
            filtered.append(w)
            
    print(f"Original: {len(added_words)} -> Filtered: {len(filtered)}")
    with open(added_path, 'w') as f:
        for w in filtered:
            f.write(w + '\n')
    print("Filtering complete on remote server!")
