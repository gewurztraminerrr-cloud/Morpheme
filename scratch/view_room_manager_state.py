import sys
sys.path.append('/home/morpheme/morpheme')
from game_room import room_manager
import time

print("="*60)
print(f"Room Manager Status at {time.ctime()}")
print(f"Active rooms: {len(room_manager.rooms)}")
for rid, r in room_manager.rooms.items():
    print(f"Room ID: {rid}")
    print(f"  State: {r.state}")
    print(f"  Players: {len(r.players)}")
    print(f"  Loading: {getattr(r, 'board_search_loading', False)}")
    print(f"  Started: {getattr(r, 'board_search_started', False)}")
    print(f"  Starting round: {getattr(r, 'starting_round', False)}")
    print(f"  Spinner params generated: {getattr(r, 'spinner_params_generated', False)}")
    print(f"  Next round board: {'Yes' if getattr(r, 'next_round_board', None) is not None else 'No'}")
    print(f"  Is Private: {r.is_private}")
    print(f"  Is Solo: {r.is_solo}")
print("="*60)
