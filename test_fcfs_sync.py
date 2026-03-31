import sys
import os
import time

# Add current directory to path
sys.path.append(os.getcwd())

from game_room import GameRoom, Player
from dataclasses import asdict

def test_fcfs_sync():
    # 1. Create a room
    room = GameRoom(room_id="test_room", game_type="fcfs", time_limit=180, board_dimensions="4x4")
    room.all_words = ["APPLE", "BANANA", "CHERRY"]
    room.state = 'active'
    room.round_start_time = time.time()
    
    # 2. Add players
    player_a = Player(user_id=1, username="PlayerA", rating=1200)
    player_b = Player(user_id=2, username="PlayerB", rating=1200)
    room.players = [player_a, player_b]
    
    # 3. Player A submits a word
    success, msg, pts, word = room.submit_word(1, "APPLE")
    print(f"Player A submits APPLE: {success}, {msg}, {pts}, {word}")
    
    # 4. Player B submits a word
    success, msg, pts, word = room.submit_word(2, "BANANA")
    print(f"Player B submits BANANA: {success}, {msg}, {pts}, {word}")
    
    # 5. Check fcfs_found_words
    print(f"FCFS Found Words: {room.fcfs_found_words}")
    
    # 6. Check if both are there
    assert len(room.fcfs_found_words) == 2
    assert room.fcfs_found_words[0]['word'] == "APPLE"
    assert room.fcfs_found_words[1]['word'] == "BANANA"
    
    # 7. Check if they have the 'finder' field
    assert room.fcfs_found_words[0]['finder'] == "PlayerA"
    assert room.fcfs_found_words[1]['finder'] == "PlayerB"
    
    print("Test Passed!")

if __name__ == "__main__":
    try:
        test_fcfs_sync()
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
