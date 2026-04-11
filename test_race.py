
import sys
import os
import time
import threading

# Add current dir to sys.path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from game_room import RoomManager, GameRoom

rm = RoomManager()
# Create a dummy room
room_id = "test_stall_final"
room = rm.create_room(room_id, "accumulative", 45, "4x4")

# Set it to intermission at 0s
room.state = "intermission"
room.intermission_start_time = time.time() - 60
room.intermission_duration = 15

def run_transition():
    tid = threading.get_ident()
    print(f"[Thread {tid}] Attempting start_next_round...")
    res = rm.start_next_round(room_id)
    print(f"[Thread {tid}] Result: {res}")

# Launch TWO threads at once to see race condition
t1 = threading.Thread(target=run_transition)
t2 = threading.Thread(target=run_transition)

t1.start()
t2.start()

t1.join()
t2.join()

print(f"Final state: {room.state}")
print(f"Final round: {room.current_round}")
