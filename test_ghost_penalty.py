
import sys
import os
import time

# Set up path to import from current directory
sys.path.append(os.getcwd())

from app import apply_leave_penalty
from game_room import GameRoom

class MockLog:
    def __init__(self):
        self.entries = []
    def write(self, msg):
        self.entries.append(msg)
        print(f"LOG: {msg.strip()}")

def test_ghost_penalty():
    print("--- Starting Ghost Penalty Test ---")
    
    # 1. Create a mock room and player
    room = GameRoom('test-room', 'accumulative', 180, '4x4')
    room.state = 'active'
    room.time_limit = 180
    
    # Add a player
    room.add_player('test-user-1', 'TestUser', 1200)
    player = room.get_player('test-user-1')
    
    # 2. Test Case: Leaving with NO activity in ACTIVE round
    print("\nCase 1: Leaving with NO activity in ACTIVE round")
    # Expected: SKIP penalty
    apply_leave_penalty('test-user-1', room)
    # Check log later? Actually, the log is hardcoded to a path in app.py.
    # I should check if the rating in the DB was changed.
    # Wait, apply_leave_penalty modifies the DB.
    
    # 3. Test Case: Leaving WITH activity in ACTIVE round
    print("\nCase 2: Leaving WITH activity in ACTIVE round")
    player.score = 50
    # Expected: APPLY penalty (-16)
    apply_leave_penalty('test-user-1', room)
    
    # 4. Test Case: Leaving during INTERMISSION (even with score)
    print("\nCase 3: Leaving during INTERMISSION (with score from previous round)")
    room.state = 'intermission'
    player.score = 100
    # Expected: SKIP penalty
    apply_leave_penalty('test-user-1', room)

    # 5. Test Case: Leaving a 24h Room
    print("\nCase 4: Leaving a 24h room (even with activity)")
    room.time_limit = 86400
    room.state = 'active'
    player.score = 200
    # Expected: SKIP penalty
    apply_leave_penalty('test-user-1', room)

    print("\n--- Ghost Penalty Test Complete ---")

if __name__ == "__main__":
    test_ghost_penalty()
