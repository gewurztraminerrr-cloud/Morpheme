import sqlite3
import json

conn = sqlite3.connect('morpheme.db')
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT param_key FROM pregenerated_boards;")
keys = cursor.fetchall()
dicts = set()
for key in keys:
    try:
        data = json.loads(key[0])
        dicts.add(data.get('dictionary'))
    except Exception as e:
        print(f"Error parsing {key[0]}: {e}")
print("Unique dictionaries in pregenerated_boards:")
for d in dicts:
    print(d)
conn.close()
