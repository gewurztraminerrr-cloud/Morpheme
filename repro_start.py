
import sys
import os
import time

# Add current dir to sys.path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from game_room import RoomManager, GameRoom

rm = RoomManager()
# Create a dummy room
room_id = "test_start_round_repro"
room = rm.create_room(room_id, "accumulative", 45, "4x4")

print(f"Room state: {room.state}, board: {room.board}")

# Try to start round 1
try:
    print("Attempting start_round...")
    res = rm.start_round(room_id)
    print(f"Result: {res}")
    print(f"New state: {room.state}")
    print(f"Board found: {len(room.board) > 0}")
except Exception as e:
    import traceback
    print(f"CRASH: {e}")
    traceback.print_exc()

if room.state != "active":
    print("STALL REPRODUCED!")
else:
    print("START SUCCESSFUL!")
