
import sys
import os

# Mock Player class
class Player:
    def __init__(self, user_id, username, rating, score, is_ai=False):
        self.user_id = user_id
        self.username = username
        self.rating = rating
        self.score = score
        self.is_ai = is_ai

# Setup paths
sys.path.append('/Users/jeffbabiak/')
from rating_logic import calculate_proportional_rating_change

def test_bot_isolation():
    print("Testing Bot Isolation...")
    # 1. Human (100) vs Bot (1000)
    p1 = Player(1, "UserA", 1200, 100, is_ai=False)
    p2 = Player(-100, "Bot_Pro", 1200, 1000, is_ai=True)
    players = [p1, p2]
    
    changes = calculate_proportional_rating_change(players)
    print(f"Human (100) vs Bot (1000): {changes}")
    
    # 2. Human A (100) vs Human B (100) vs Bot (1000)
    p1 = Player(1, "UserA", 1200, 100, is_ai=False)
    p2 = Player(2, "UserB", 1200, 100, is_ai=False)
    p3 = Player(-100, "Bot_Pro", 1200, 1000, is_ai=True)
    players = [p1, p2, p3]
    
    changes = calculate_proportional_rating_change(players)
    print(f"HumanA (100), HumanB (100) vs Bot (1000): {changes}")

if __name__ == "__main__":
    test_bot_isolation()
