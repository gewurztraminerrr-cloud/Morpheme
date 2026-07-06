import sqlite3

conn = sqlite3.connect('morpheme.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

print("Searching SQLite database for '+ AW'...")
for table in tables:
    if table.startswith('sqlite_'):
        continue
    try:
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [c[1] for c in cursor.fetchall() if 'text' in c[2].lower() or 'char' in c[2].lower() or 'clob' in c[2].lower() or c[2] == '']
        for col in cols:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} LIKE '%+ AW%' OR {col} LIKE '%+AW%';")
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"Found {count} matches in table '{table}', column '{col}'!")
                # Print a sample
                cursor.execute(f"SELECT {col} FROM {table} WHERE {col} LIKE '%+ AW%' OR {col} LIKE '%+AW%' LIMIT 3;")
                samples = cursor.fetchall()
                for s in samples:
                    print(f"  Sample: {s[0][:150]}")
    except Exception as e:
        print(f"Error searching table {table}: {e}")

conn.close()
