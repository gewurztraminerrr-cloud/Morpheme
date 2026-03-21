
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
    # Scenario 1: One human player alone
    p1 = Player(1, "UserA", 1200, 100)
    players = [p1]
    changes = calculate_proportional_rating_change(players)
    print(f"Solo UserA (Score 100): {changes}")

    # Scenario 2: Two human players with same rating
    p1 = Player(1, "UserA", 1200, 100)
    p2 = Player(2, "UserB", 1200, 100)
    players = [p1, p2]
    changes = calculate_proportional_rating_change(players)
    print(f"UserA (100) vs UserB (100): {changes}")

    # Scenario 3: UserA (100) vs UserB (900)
    p1 = Player(1, "UserA", 1200, 100)
    p2 = Player(2, "UserB", 1200, 900)
    players = [p1, p2]
    changes = calculate_proportional_rating_change(players)
    print(f"UserA (100) vs UserB (900): {changes}")

    # Scenario 4: UserA (100) vs Bot (1000)
    p1 = Player(1, "UserA", 1200, 100)
    p2 = Player(2, "Bot", 1200, 1000, is_ai=True)
    players = [p1, p2]
    changes = calculate_proportional_rating_change(players)
    print(f"UserA (100) vs Bot (1000, is_ai=True): {changes}")

    # Scenario 5: UserA (100) vs Guest (1000)
    p1 = Player(1, "UserA", 1200, 100)
    p2 = Player(2, "Guest_123", 1000, 1000)
    players = [p1, p2]
    changes = calculate_proportional_rating_change(players)
    print(f"UserA (100) vs Guest (1000): {changes}")

if __name__ == "__main__":
    test_rating()
