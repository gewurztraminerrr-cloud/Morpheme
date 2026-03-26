import time
from game_room import RoomManager

rm = RoomManager()
room = rm.create_room('test_room', 'fcfs', 180, '4x4')
room.add_player(1, 'Player1', 1200)

room.current_board_format = 'Normal'
room.state = 'active'
room.round_start_time = time.time() - 30
room.spinner_params = {'board_format': 'Normal'}

room.add_player(2, 'Player2', 1200)

p1 = room.get_player(1)
p2 = room.get_player(2)
p1.score = 10
p2.score = 5

room.custom_end_time = time.time() - 1
room.check_and_update_state()

print(f"P1 rating change: {p1.rating_change}")
print(f"P2 rating change: {p2.rating_change}")
