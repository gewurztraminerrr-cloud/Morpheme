
from board_generator import BoardGenerator
from game_room import GameRoom
import sys

# Mock word_validator so we don't need full dict load
class MockValidator:
    def __init__(self):
        self.words = {"QUATE", "QINDAR"}
    
    def is_valid_word(self, word, dictionary):
        return word in self.words
    
    def has_valid_prefix(self, prefix, dictionary):
        for w in self.words:
            if w.startswith(prefix):
                return True
        return False

import word_validator
word_validator.word_validator = MockValidator()

def test_q_logic():
    print("Beginning Q/QU Logic Verification...")
    
    # --- Test 1: Board Generation ---
    gen = BoardGenerator()
    # Mock board: Q A T E (4x1 for simplicity)
    # We will invoke _solve_board directly with a custom board
    board = [['Q', 'A', 'T', 'E']]
    
    print("\n[Test 1] Solving board: Q-A-T-E")
    found_words = gen._solve_board(board, 'MOCK', (0, 100))
    
    if "QUATE" in found_words:
        print("✓ SUCCESS: Found 'QUATE' from 'Q' tile (using Q->QU branch)")
    else:
        print("✗ FAILURE: Did NOT find 'QUATE'")
        print(f"  Found: {found_words}")
        return False

    # --- Test 2: Submission Logic ---
    print("\n[Test 2] Submitting 'QATE' (input) -> 'QUATE' (matching)")
    room = GameRoom("test_room", "acc", 60, "4x4")
    room.all_words = ["QUATE", "QINDAR"] # Simulate board finding these
    room.spinner_params = {'min_word_length': 3}
    room.add_player(123, "test_user", 1200)
    
    # Try submitting "QATE" (which is what frontend sends for Q-A-T-E tiles)
    success, msg = room.submit_word(123, "QATE")
    player = room.get_player(123)
    
    if success and "QUATE" in player.submitted_words:
        print("✓ SUCCESS: Submission of 'QATE' accepted as 'QUATE'")
    else:
        print(f"✗ FAILURE: Submission failed or not mapped correctly. Msg: {msg}")
        print(f"  Player words: {player.submitted_words}")
        return False
        
    return True

if __name__ == "__main__":
    if test_q_logic():
        print("\nAll Q/QU tests passed!")
        sys.exit(0)
    else:
        sys.exit(1)
