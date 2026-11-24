import sys
import os

# Mock Player class
class Player:
    def __init__(self, user_id, username, rating, score, is_ai=False, is_guest=False):
        self.user_id = user_id
        self.username = username
        self.rating = rating
        self.score = score
        self.is_ai = is_ai
        self.is_guest = is_guest

from rating_logic import calculate_proportional_rating_change

def test_rating():
    print("=== SCENARIO 1: Solo user ===")
    p1 = Player(1, "UserA", 1200, 100)
    changes = calculate_proportional_rating_change([p1], board_format='Normal')
    assert changes[1] == 0, f"Expected 0 change for solo player, got {changes[1]}"
    print("Pass!")

    print("=== SCENARIO 2: Equal score UserA vs UserB ===")
    p1 = Player(1, "UserA", 1200, 100)
    p2 = Player(2, "UserB", 1200, 100)
    changes = calculate_proportional_rating_change([p1, p2], board_format='Normal')
    assert changes[1] == 0, f"Expected 0 change, got {changes[1]}"
    assert changes[2] == 0, f"Expected 0 change, got {changes[2]}"
    print("Pass!")

    print("=== SCENARIO 3: Normal Format Clamping (-16/+16) ===")
    p1 = Player(1, "UserA", 1200, 100)
    p2 = Player(2, "UserB", 1200, 900)
    changes = calculate_proportional_rating_change([p1, p2], board_format='Normal')
    assert changes[1] == -16, f"Expected Normal format rating change to be clamped to -16, got {changes[1]}"
    assert changes[2] == 16, f"Expected Normal format rating change to be clamped to 16, got {changes[2]}"
    print("Pass!")

    print("=== SCENARIO 4: Double Format Scaling (2x clamped) ===")
    p1 = Player(1, "UserA", 1200, 100)
    p2 = Player(2, "UserB", 1200, 900)
    changes = calculate_proportional_rating_change([p1, p2], board_format='Double')
    assert changes[1] == -32, f"Expected Double format rating change to be -32 (2x -16), got {changes[1]}"
    assert changes[2] == 32, f"Expected Double format rating change to be 32 (2x 16), got {changes[2]}"
    print("Pass!")

    print("=== SCENARIO 5: Triple Format Scaling (3x clamped) ===")
    p1 = Player(1, "UserA", 1200, 100)
    p2 = Player(2, "UserB", 1200, 900)
    changes = calculate_proportional_rating_change([p1, p2], board_format='Triple')
    assert changes[1] == -48, f"Expected Triple format rating change to be -48 (3x -16), got {changes[1]}"
    assert changes[2] == 48, f"Expected Triple format rating change to be 48 (3x 16), got {changes[2]}"
    print("Pass!")

    print("=== SCENARIO 6: Score 0 player must have change 0 ===")
    p1 = Player(1, "UserA", 1200, 100)
    p2 = Player(2, "UserB", 1200, 0)
    changes = calculate_proportional_rating_change([p1, p2], board_format='Normal')
    assert changes[1] == 0, f"Expected 0 change for UserA (only 1 active player with score >= 1), got {changes[1]}"
    assert changes[2] == 0, f"Expected 0 change for UserB (score < 1), got {changes[2]}"
    print("Pass!")

    print("=== SCENARIO 7: Mid-round player must have change 0 ===")
    p1 = Player(1, "UserA", 1200, 100)
    p2 = Player(2, "UserB", 1200, 200)
    p2.joined_mid_round = True
    changes = calculate_proportional_rating_change([p1, p2], board_format='Normal')
    assert changes[1] == 0, f"Expected 0 change for UserA (only 1 starter), got {changes[1]}"
    assert changes[2] == 0, f"Expected 0 change for UserB (joined mid-round), got {changes[2]}"
    print("Pass!")

    print("=== SCENARIO 8: 4 players with mid-round and 0-score players ===")
    p1 = Player(1, "StarterA", 1200, 100)
    p2 = Player(2, "StarterB", 1200, 300)
    p3 = Player(3, "MidRoundC", 1200, 500)
    p3.joined_mid_round = True
    p4 = Player(4, "StarterD", 1200, 0)
    changes = calculate_proportional_rating_change([p1, p2, p3, p4], board_format='Normal')
    assert changes[1] < 0, f"Expected StarterA to lose rating against StarterB, got {changes[1]}"
    assert changes[2] > 0, f"Expected StarterB to gain rating against StarterA, got {changes[2]}"
    assert changes[3] == 0, f"Expected MidRoundC to have 0 change, got {changes[3]}"
    assert changes[4] == 0, f"Expected StarterD to have 0 change, got {changes[4]}"
    print("Pass!")

    print("\n🎉 ALL RATING LOGIC TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_rating()
