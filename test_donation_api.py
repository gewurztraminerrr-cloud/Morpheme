import sqlite3
import json

def test_api():
    try:
        conn = sqlite3.connect('morpheme.db')
        conn.row_factory = sqlite3.Row
        
        # Check if tables exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('site_config', 'donations')")
        tables = [row['name'] for row in cursor.fetchall()]
        print(f"Tables found: {tables}")
        
        if 'site_config' not in tables:
            print("Error: site_config table is missing!")
            return

        # Try to query config
        cursor = conn.execute("SELECT * FROM site_config WHERE config_key IN ('yearly_budget', 'paypal_url', 'paypal_client_id')")
        rows = cursor.fetchall()
        config = {row['config_key']: row['config_value'] for row in rows}
        print(f"Config: {config}")
        
        # Try to query donations sum
        cursor = conn.execute("SELECT SUM(amount) as total FROM donations WHERE status = 'confirmed'")
        total = cursor.fetchone()['total'] or 0
        print(f"Total donated: {total}")
        
        print("API Logic check: SUCCESS")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
