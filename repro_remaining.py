
import sys
import os
import time
import json
from dataclasses import asdict

# Add current dir to path
sys.path.append(os.getcwd())

from game_room import room_manager, GameRoom, Player
from spinner_set import SpinnerSet

def test_transition_and_remaining():
    # 1. Create a room
    room_id = "test_rem"
    room = room_manager.create_room(room_id, "accumulative", 45, "4x4")
    
    # 2. Add a player
    p = Player(123, "testuser", 1200)
    room.players.append(p)
    
    # 3. Start a round
    print("Starting Round 1...")
    room_manager.start_next_round(room_id)
    time.sleep(2) # Wait for generation
    
    print(f"State: {room.state}")
    print(f"Words in round: {len(room.all_words)}")
    
    # 4. Simulate round end
    print("Simulating Round End...")
    room.state = 'intermission'
    room.intermission_start_time = time.time()
    
    # Run the authoritative update check
    room_manager.process_advancements()
    
    # 5. Check if all_words exists
    print(f"Intermission State: {room.state}")
    print(f"Room all_words count: {len(room.all_words)}")
    
    # 6. Check Spinner Reveal
    room.intermission_start_time = time.time() - 20 # 20s elapsed
    room_manager.process_advancements()
    print(f"Revealed: {room.spinner_params_revealed}")
    
    # 7. Check API response structure (What the client sees)
    # Mocking app.py logic
    is_intermission = room.state == 'intermission'
    is_revealed = room.spinner_params_revealed
    
    words_to_return = []
    if is_intermission:
        words_to_return = list(room.all_words)
        
    print(f"API all_words count: {len(words_to_return)}")
    
    if len(words_to_return) == 0:
        print("FAILURE: all_words is empty during intermission!")
    else:
        print("SUCCESS: all_words is populated.")

if __name__ == "__main__":
    test_transition_and_remaining()
