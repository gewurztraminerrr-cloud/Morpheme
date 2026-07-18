import sqlite3
import json
import os

def main():
    db_path = 'morpheme.db'
    if not os.path.exists(db_path):
        print("Database not found local, connecting to remote...")
        # Since we're running locally, we can just ssh or run locally if it's there
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT param_key, COUNT(*) FROM pregenerated_boards GROUP BY param_key;")
    rows = cursor.fetchall()
    
    print("Pregenerated boards count by param_key:")
    for param_key_str, count in rows:
        params = json.loads(param_key_str)
        if params.get('dimensions') == '4x6':
            print(f"Params: {params} | Count: {count}")
            
    conn.close()

if __name__ == '__main__':
    main()
