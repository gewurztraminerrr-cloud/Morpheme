import sys
import os
import time

sys.path.insert(0, '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from game_room import room_manager

def test_flow():
    # 1. Create a room for game_type='fcfs', board='4x4', time=45
    room_id = "room_fcfs_4x4_45_test1234"
    room = room_manager.create_room(room_id, 'fcfs', 45, '4x4', 0, 9999, is_private=False)
    
    # 2. Add user 'jeffb' (user_id=42)
    room.add_player(42, 'jeffb', 1200)
    
    print("Room ID created:", room.room_id)
    print("Players in room:", [p.username for p in room.players])
    print("Human players count:", len([p for p in room.players if not getattr(p, 'is_ai', False)]))
    
    # 3. Simulate list_rooms query for game_type='fcfs', board='4x4', time=45
    active_rooms = []
    game_type = 'fcfs'
    board_dimensions = '4x4'
    time_limit = 45
    
    for r_id, r in list(room_manager.rooms.items()):
        if r.is_solo or getattr(r, 'is_private', False):
            continue
        matches_game = not game_type or str(r.game_type).lower() == str(game_type).lower()
        matches_board = not board_dimensions or str(r.board_dimensions).lower() == str(board_dimensions).lower()
        try:
            r_t = int(r.time_limit)
        except:
            r_t = 45
        matches_time = time_limit is None or r_t == time_limit
        
        if matches_game and matches_board and matches_time:
            humans = [p for p in r.players if not getattr(p, 'is_ai', False)]
            is_daily = (r_t >= 7200)
            if len(humans) == 0 and len(r.spectators) == 0 and not is_daily:
                continue
            active_rooms.append({
                'room_id': r.room_id,
                'players': [{'username': p.username, 'rating': p.rating} for p in r.players]
            })
            
    print("Active rooms listed:", active_rooms)

if __name__ == '__main__':
    test_flow()
