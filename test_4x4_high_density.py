import sys
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')
from game_room import RoomManager
import time

rm = RoomManager()
bw = rm._get_bonus_word(9, "CSW", difficulty="Hard")
print(f"Random 9L Bonus Word: {bw}")
start = time.time()
try:
    res = rm.board_generator.generate_board(
        "4x4",
        bw,
        "100-200",
        "NWL",
        "Normal",
        5,
        "Hard",
        is_emergency=True
    )
    count = sum(1 for w in res[1] if len(w) >= 5)
    print(f"Words: {count} (Time: {time.time()-start:.2f}s)")
except Exception as e:
    import traceback
    traceback.print_exc()
