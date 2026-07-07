import sqlite3
import json

db_path = '/home/morpheme/morpheme/morpheme.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Find how many rows contain "dictionary": "AW"
c.execute("SELECT id, param_key FROM pregenerated_boards")
rows = c.fetchall()

to_delete = []
for row_id, param_key in rows:
    try:
        params = json.loads(param_key)
        if params.get('dictionary') == 'AW':
            to_delete.append(row_id)
    except Exception as e:
        print(f"Error parsing param_key for row {row_id}: {e}")

if to_delete:
    print(f"Deleting {len(to_delete)} rows matching 'AW' dictionary...")
    c.executemany("DELETE FROM pregenerated_boards WHERE id = ?", [(rid,) for rid in to_delete])
    conn.commit()
    print("Deleted successfully!")
else:
    print("No rows matched 'AW' dictionary.")

conn.close()
