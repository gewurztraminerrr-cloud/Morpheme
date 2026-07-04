import time
import json
import sqlite3
from tournament_logic import TournamentManager

tm = TournamentManager()

# Let's inspect the current tournament in morpheme.db
conn = sqlite3.connect('morpheme.db')
conn.row_factory = sqlite3.Row

print("=== CURRENT STATE OF TOURNAMENTS ===")
t = conn.execute('SELECT * FROM tournaments ORDER BY id DESC LIMIT 1').fetchone()
if t:
    t_dict = dict(t)
    print(f"Tournament ID: {t_dict['id']}")
    print(f"Status: {t_dict['status']}")
    print(f"Start Date: {t_dict['start_date']} (Time remaining: {t_dict['start_date'] - time.time():.1f}s)")
    
    # Check participants
    parts = conn.execute('SELECT * FROM tournament_participants WHERE tournament_id = ?', (t_dict['id'],)).fetchall()
    print(f"Number of participants: {len(parts)}")
else:
    print("No tournaments found.")

print("\n=== FORCE SIGNUP PERIOD TO BE OVER ===")
if t and t_dict['status'] == 'signup':
    params = json.loads(t_dict['parameters'])
    params['difficulty'] = 'Easy'
    conn.execute('UPDATE tournaments SET parameters = ?, start_date = ? WHERE id = ?', (json.dumps(params), time.time() - 10, t_dict['id']))
    conn.commit()
    print("Forced start_date to the past and difficulty to Easy.")
else:
    # Create a dummy signup tournament and force it
    t_dict = tm.create_new_tournament()
    params = json.loads(t_dict['parameters'])
    params['difficulty'] = 'Easy'
    conn.execute('UPDATE tournaments SET parameters = ?, start_date = ? WHERE id = ?', (json.dumps(params), time.time() - 10, t_dict['id']))
    conn.commit()
    print(f"Created tournament {t_dict['id']} (Easy) and forced start_date to the past.")

# Trigger update
print("\n=== TRIGGERING LIFE-CYCLE UPDATE ===")
tm.update_tournament_status()

# Check state after update
t_after = conn.execute('SELECT * FROM tournaments ORDER BY id DESC LIMIT 1').fetchone()
t_after_dict = dict(t_after)
print(f"Tournament ID: {t_after_dict['id']}")
print(f"Status: {t_after_dict['status']}")

# Check participants
parts = conn.execute('SELECT tp.*, u.username FROM tournament_participants tp JOIN users u ON tp.user_id = u.id WHERE tp.tournament_id = ?', (t_after_dict['id'],)).fetchall()
print(f"Participants count: {len(parts)}")
for p in parts:
    print(f"  - User {p['user_id']}: {p['username']} ({p['status']})")

# Clean up / revert test tournament if needed (rollback test tournament 8+)
# We will just let it be, but let's make sure it worked perfectly.
print("\n=== TEST COMPLETED ===")
conn.close()
