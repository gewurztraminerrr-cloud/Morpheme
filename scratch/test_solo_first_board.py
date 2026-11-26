import sys, time
sys.path.insert(0, '/Users/jeffbabiak')
from game_room import RoomManager
from spinner_set import SpinnerSet

diff_req = sys.argv[1]
fmts = sys.argv[2].split(',')
dims = sys.argv[3]
n = int(sys.argv[4])
rm = RoomManager()
bad = 0
for i in range(n):
    fmt = fmts[i % len(fmts)]
    params = {'difficulty': diff_req, 'board_format': fmt, 'dictionary': 'random',
              'min_word_length': 'random', 'word_count_range': 'random', 'bonus_word_length': 'random'}
    rid = f"test_first_{int(time.time()*1000)}"
    t0 = time.time()
    room = rm.create_room(rid, game_type='solo_accumulative', time_limit=60, board_dimensions=dims, is_solo=True, initial_solo_params=params)
    bg = rm.board_generator
    p = [int(x) for x in dims.split('x')]
    min_l = room.current_min_length
    ratio = room.current_uniqueness
    real = bg.get_difficulty_label(ratio, p[0], p[1], str(room.current_dictionary).split()[0], depth=1, min_word_length=min_l)
    ok = real == diff_req
    print(f"RUN{i+1} req={fmt}/{diff_req} got fmt={room.current_board_format!r} label={room.current_difficulty} real={real} ratio={ratio:.3f} {time.time()-t0:.1f}s {'OK' if ok else 'MISMATCH'}")
    if not ok: bad += 1
print("MISMATCHES:", bad)
