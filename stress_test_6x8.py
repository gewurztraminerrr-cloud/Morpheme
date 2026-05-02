import sys
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')
from game_room import RoomManager
import time

rm = RoomManager()
undershoots = 0
total = 20
for i in range(total):
    bw = rm._get_bonus_word(9, "CSW", difficulty="Medium")
    start = time.time()
    res = rm.board_generator.generate_board(
        "6x8",
        bw,
        "50-100",
        "CSW",
        "Normal",
        7,
        "Medium",
        is_emergency=True
    )
    count = sum(1 for w in res[1] if len(w) >= 7)
    print(f"Board {i+1}: {count} words (Time: {time.time()-start:.2f}s)")
    if count < 50 or count > 110:
        undershoots += 1

print(f"Total Undershoots/Overshoots: {undershoots}/{total}")
