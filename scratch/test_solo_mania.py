import sys, time
sys.path.insert(0, '/Users/jeffbabiak')
from game_room import RoomManager
from spinner_set import SpinnerSet

fmt_req = sys.argv[1] if len(sys.argv) > 1 else 'Mania'
diff_req = sys.argv[2] if len(sys.argv) > 2 else 'Hard'
dims = sys.argv[3] if len(sys.argv) > 3 else '4x4'
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 5

rm = RoomManager()
params = {'difficulty': diff_req, 'board_format': fmt_req, 'dictionary': 'NWL',
          'min_word_length': 'random', 'word_count_range': 'random', 'bonus_word_length': 'random'}
rid = f"test_solo_mania_{int(time.time())}"
room = rm.create_room(rid, game_type='solo_accumulative', time_limit=60, board_dimensions=dims,
                      is_solo=True, initial_solo_params=params)
bad = 0

def report(tag):
    global bad
    flat = ''.join(c for r in room.board for c in r) if room.board and isinstance(room.board[0], list) and not isinstance(room.board[0][0], list) else ''
    fmt = room.current_board_format
    sp_fmt = room.spinner_params.get('board_format')
    parts = str(fmt).split()
    letter = parts[0].upper() if len(parts) >= 2 and len(parts[0]) == 1 else None
    cnt = flat.count(letter) if letter else None
    diff = SpinnerSet.strip_difficulty_pct(room.current_difficulty)
    print(f"[{tag}] fmt={fmt!r} spinner_fmt={sp_fmt!r} letter={letter} count={cnt}/{len(flat)} diff={room.current_difficulty} uniq={room.current_uniqueness}")
    print("   board:", flat)
    ok = True
    if 'mania' in fmt_req.lower():
        if not letter: ok = False; print("   FAIL: no mania letter in format")
        elif cnt < max(2, int(len(flat) * 0.2)): ok = False; print("   FAIL: abundance missing")
        if sp_fmt != fmt: ok = False; print("   FAIL: spinner/current format mismatch")
    if diff_req.lower() != 'random' and diff != diff_req: ok = False; print("   FAIL: difficulty mismatch")
    if not ok: bad += 1

report('R1')
for r in range(2, rounds + 1):
    room = rm.get_room(rid)
    room.state = 'intermission'
    rm.pre_generate_next_round(rid)
    time.sleep(1)
    rm.start_next_round(rid)
    room = rm.get_room(rid)
    report(f'R{r}')
print("FAILURES:", bad)
