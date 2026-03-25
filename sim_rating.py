from game_room import GameRoom, Player
import time

room = GameRoom(room_id='test', game_type='accumulative', time_limit=300, board_dimensions='4x4')
room.state = 'intermission'
room.intermission_start_time = time.time() - 30 
room.spinner_params['board_format'] = 'Normal'

# Add 2 players from intermission
room.add_player(1, 'Alice', 1200)
room.add_player(2, 'Bob', 1200)

for p in room.players:
    p.joined_mid_round = False
    p.score = 0
room.state = 'active'
room.round_start_time = time.time()

# Simulate play
room.get_player(1).score = 100
room.get_player(1).submitted_words = [{'word':'FOO'}]

room.get_player(2).score = 50
room.get_player(2).submitted_words = [{'word':'BAR'}]

# Simulate end_round rating block exact lines
from rating_logic import calculate_proportional_rating_change, is_player_guest
competitive_human_starters = [
    p for p in room.players + room.round_quitters 
    if not getattr(p, 'is_ai', False) and not is_player_guest(p) and not getattr(p, 'joined_mid_round', False)
]
board_format = room.current_board_format
is_ranked_format = (str(board_format).strip() == 'Normal')
is_500plus = False

print(f"Len competitive starters: {len(competitive_human_starters)}")

if not is_ranked_format or is_500plus or (not room.is_private and len(competitive_human_starters) <= 1):
    print("DISABLED")
else:
    print("CALCULATING")
    rating_changes = calculate_proportional_rating_change(room.players, is_private=room.is_private)
    print("Rating changes:", rating_changes)
