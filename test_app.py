import json
import sqlite3
import game_room

rm = game_room.RoomManager()
r = rm.create_room('test_user', 'public')
rm.join_room(r, 'test_user')
r_obj = rm.get_room(r)

r_obj.current_uniqueness = 0.49
r_obj.current_difficulty = 'Hard'

# Fake app.py logic
state = {
    'current_uniqueness': getattr(r_obj, 'current_uniqueness', 0.0),
    'current_difficulty': getattr(r_obj, 'current_difficulty', 'Medium'),
}
print(json.dumps(state, indent=2))
