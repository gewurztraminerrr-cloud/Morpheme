
import sys
import os
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')
from game_room import room_manager
from spinner_set import SpinnerSet

room_id = "test_room_123"
room = room_manager.create_room(room_id, "accumulative", 45, "4x4")
print(f"Room {room_id} created. State: {room.state}")
print(f"Spinner Params: {room.spinner_params}")

print("Starting board search...")
room_manager.start_board_search(room_id)

import time
time.sleep(5) # Wait for thread to start

print(f"Board Search Started: {getattr(room, 'board_search_started', False)}")
print(f"Board Search Loading: {getattr(room, 'board_search_loading', False)}")
