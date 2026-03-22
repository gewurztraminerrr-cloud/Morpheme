import sys
import os
os.environ['PYTHONPATH'] = '/Users/jeffbabiak/'
sys.path.append('/Users/jeffbabiak/')

class Player:
    def __init__(self, user_id, username, rating, score):
        self.user_id = user_id
        self.username = username
        self.rating = rating
        self.score = score

from rating_logic import calculate_proportional_rating_change

def test():
    # UserA wins massively
    p1 = Player(1, "UserA", 1200, 10000)
    p2 = Player(2, "UserB", 1200, 1)
    players = [p1, p2]
    changes = calculate_proportional_rating_change(players)
    print(f"UserA (10000, 1200) vs UserB (1, 1200): {changes}")

if __name__ == "__main__":
    test()
