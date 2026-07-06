import sqlite3
import os

def main():
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'morpheme.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM pregenerated_boards;")
    count = cursor.fetchone()[0]
    print(f"Total pregenerated boards: {count}")
    
    cursor.execute("SELECT DISTINCT param_key FROM pregenerated_boards LIMIT 10;")
    print("\nSample param keys:")
    for row in cursor.fetchall():
        print(row[0])
        
    conn.close()

if __name__ == '__main__':
    main()
