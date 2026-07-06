import sqlite3
import os
import time

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'morpheme.db')
    wikdefs_path = os.path.join(base_dir, 'dictionaries', 'wikdefs.txt')
    
    print(f"Opening DB at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Creating wiktionary_definitions table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wiktionary_definitions (
            word TEXT PRIMARY KEY,
            definition TEXT NOT NULL
        );
    """)
    conn.commit()
    
    print("Reading and importing wikdefs.txt...")
    start_time = time.time()
    batch = []
    count = 0
    
    with open(wikdefs_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                word = parts[0].strip().upper()
                definition = parts[1].strip()
                if word and definition:
                    batch.append((word, definition))
                    count += 1
                    
                    if len(batch) >= 10000:
                        cursor.executemany(
                            "INSERT OR REPLACE INTO wiktionary_definitions (word, definition) VALUES (?, ?);",
                            batch
                        )
                        conn.commit()
                        batch = []
                        print(f"Imported {count} definitions...")
                        
    if batch:
        cursor.executemany(
            "INSERT OR REPLACE INTO wiktionary_definitions (word, definition) VALUES (?, ?);",
            batch
        )
        conn.commit()
        
    print(f"\nSuccessfully imported {count} definitions in {time.time() - start_time:.2f} seconds!")
    conn.close()

if __name__ == '__main__':
    main()
