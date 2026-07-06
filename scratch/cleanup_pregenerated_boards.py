import sqlite3
import os
import json

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'morpheme.db')
    
    if not os.path.exists(db_path):
        print(f"No database found at {db_path}")
        return
        
    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    
    # 1. Purge legacy "+ AW" / "+AW" pregenerated boards
    cursor.execute("SELECT param_key FROM pregenerated_boards;")
    rows = cursor.fetchall()
    
    deleted_count = 0
    total_count = len(rows)
    for (param_key,) in rows:
        try:
            params = json.loads(param_key)
            dictionary = str(params.get("dictionary", ""))
            if "+ AW" in dictionary or "+AW" in dictionary:
                cursor.execute("DELETE FROM pregenerated_boards WHERE param_key = ?;", (param_key,))
                deleted_count += 1
        except Exception as e:
            # If not valid JSON or other error, delete it to be safe
            cursor.execute("DELETE FROM pregenerated_boards WHERE param_key = ?;", (param_key,))
            deleted_count += 1
            
    conn.commit()
    print(f"Purged {deleted_count} out of {total_count} total pregenerated boards with legacy +AW configurations.")
    
    # 2. Let's make sure the wiktionary_definitions table exists and contains records
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='wiktionary_definitions';")
    exists = cursor.fetchone()
    if exists:
        cursor.execute("SELECT COUNT(*) FROM wiktionary_definitions;")
        count = cursor.fetchone()[0]
        print(f"wiktionary_definitions table exists and has {count} entries.")
    else:
        print("WARNING: wiktionary_definitions table does NOT exist!")
        
    conn.close()

if __name__ == '__main__':
    main()
