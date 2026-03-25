from rating_logic import calculate_proportional_rating_change

class MockPlayer:
    def __init__(self, id, name, rating, score, mid_round=False):
        self.user_id = id
        self.username = name
        self.rating = rating
        self.score = score
        self.joined_mid_round = mid_round
        self.is_ai = False
        self.is_guest = False
        self.invalid_words = []

players = [
    MockPlayer(1, "Alice", 1200, 100),
    MockPlayer(2, "Bob", 1200, 50)
]
changes = calculate_proportional_rating_change(players, is_private=False)
print("Changes:", changes)
