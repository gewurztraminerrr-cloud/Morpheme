
import sys
import os
import time

# Add current dir to sys.path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from game_room import RoomManager, GameRoom

rm = RoomManager()
# Create a dummy room
room_id = "test_stall_fix"
room = rm.create_room(room_id, "accumulative", 45, "4x4")

# Set it to intermission at 0s
room.state = "intermission"
room.intermission_start_time = time.time() - 60
room.intermission_duration = 15

print(f"Room state: {room.state}, TR: {room.time_remaining}")
print(f"Milestone: {room.get_next_round_milestone()}")

# Try to transition
try:
    print("Attempting start_next_round...")
    res = rm.start_next_round(room_id)
    print(f"Result: {res}")
    print(f"New state: {room.state}")
    print(f"Board: {room.board}")
    print(f"Solved words count: {len(room.solved_words_with_scores)}")
except Exception as e:
    import traceback
    print(f"CRASH: {e}")
    traceback.print_exc()

if room.state != "active":
    print("STALL REPRODUCED!")
else:
    print("TRANSITION SUCCESSFUL!")
