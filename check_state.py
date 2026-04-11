
import sys
import os
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from game_room import room_manager
import time

print(f"Room Count: {len(room_manager.rooms)}")
for room_id, room in room_manager.rooms.items():
    print(f"Room: {room_id}")
    print(f"  State: {room.state}")
    print(f"  Total Words: {len(room.all_words)}")
    print(f"  Total Points: {getattr(room, 'total_points_count', 'N/A')}")
    print(f"  Solved Words Count: {len(getattr(room, 'solved_words_with_scores', {}))}")
