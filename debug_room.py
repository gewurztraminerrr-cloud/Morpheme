import os
import sys
import time

# Mocking the environment to load the RoomManager
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from app import room_manager

room_id = "pub_accumulative_4x4_45"
room = room_manager.get_room(room_id)

if room:
    print(f"Room: {room_id}")
    print(f"State: {room.state}")
    print(f"Next Board: {'Ready' if getattr(room, 'next_round_board', None) else 'None'}")
    print(f"Search Loading: {getattr(room, 'board_search_loading', False)}")
    print(f"Search Started: {getattr(room, 'board_search_started', False)}")
    print(f"Starting Round: {getattr(room, 'starting_round', False)}")
else:
    print("Room not found")
