import sqlite3
import json

def calculate_pe():
    conn = sqlite3.connect('morpheme.db')
    
    # Get the specific round (timestamp 17:11:43)
    cursor = conn.execute("SELECT room_id, round_number, timestamp FROM round_history WHERE total_score=88 ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    if not row:
        print("Round not found")
        return

    r_id, r_num, ts = row
    print(f"Analyzing Round: Room={r_id} Round={r_num} Timestamp={ts}")

    # Fetch room entries (as app.py does)
    cursor_room = conn.execute('''
        SELECT rh.total_score, rh.user_rating, u.username
        FROM round_history rh
        JOIN users u ON rh.user_id = u.id
        WHERE rh.room_id = ? AND rh.round_number = ? AND rh.timestamp = ?
    ''', (r_id, r_num, ts))
    room_entries = cursor_room.fetchall()

    print(f"Room Entries Found: {len(room_entries)}")
    for e in room_entries:
        print(f"  User: {e[2]}, Score: {e[0]}, Rating: {e[1]}")

    my_score = 88
    
    # 1. Arithmetic Mean
    avg_room_score = sum(e[0] for e in room_entries) / len(room_entries) if room_entries else 0
    ratio_mean = round(my_score / avg_room_score, 2) if avg_room_score > 0 else 1.0
    print(f"Method 1 (Mean): Avg={avg_room_score:.2f}, Ratio={ratio_mean}")

    # 2. Linear Rating Share
    total_score = sum(e[0] for e in room_entries)
    total_rating = sum(e[1] for e in room_entries)
    
    my_entry = next((e for e in room_entries if e[0] == 88), None)
    if my_entry:
        my_rating = my_entry[1]
        expected_share = my_rating / total_rating
        expected_score = expected_share * total_score
        ratio_linear = round(my_score / expected_score, 2) if expected_score > 0 else 1.0
        print(f"Method 2 (Linear Rating): TotalRating={total_rating}, MyRating={my_rating}, ExpShare={expected_share:.3f}, ExpScore={expected_score:.2f}, Ratio={ratio_linear}")

    # 3. Inverse Mean?
    # 4. Excluding Self?

    conn.close()

if __name__ == "__main__":
    calculate_pe()
