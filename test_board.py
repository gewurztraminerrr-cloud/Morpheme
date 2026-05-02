import sys
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')
from game_room import RoomManager
import time

rm = RoomManager()

for i in range(5):
    bw = rm._get_bonus_word(9, "NWL", difficulty="Medium")
    print(f"--- TEST {i+1} : {bw} ---")
    start = time.time()
    try:
        res = rm.board_generator.generate_board(
            "4x4",
            bw,
            "100-200",
            "NWL",
            "Normal",
            4,
            "Medium",
            is_emergency=False
        )
        print(f"Words: {len(res[1])} (Time: {time.time()-start:.2f}s)")
    except Exception as e:
        print(f"Error: {e}")
