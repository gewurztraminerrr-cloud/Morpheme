
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from rating_logic import calculate_proportional_rating_change

class MockPlayer:
    def __init__(self, user_id, username, rating, score=0):
        self.user_id = user_id
        self.username = username
        self.rating = rating
        self.score = score
        self.is_ai = False
        self.joined_mid_round = False
        self.submitted_words = []
        self.invalid_words = []

def test():
    p1 = MockPlayer(1, "jeffy", 1072)
    p2 = MockPlayer(2, "jeffles", 1086)
    players = [p1, p2]
    
    print("--- Testing 0 vs 0 DNP ---")
    res = calculate_proportional_rating_change(players)
    print(f"Result: {res}")
    
    print("\n--- Testing 10 vs 0 (P2 DNP) ---")
    p1.score = 10
    res = calculate_proportional_rating_change(players)
    print(f"Result: {res}")

    print("\n--- Testing 10 vs 0 (P2 Typed Invalid) ---")
    p1.score = 10
    p2.score = 0
    p2.invalid_words = ["WRONG"]
    res = calculate_proportional_rating_change(players)
    print(f"Result: {res}")

if __name__ == '__main__':
    test()
