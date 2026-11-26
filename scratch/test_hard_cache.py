import sys, time, json
sys.path.insert(0, '/Users/jeffbabiak')
from board_generator import BoardGenerator, serialize_param_key, get_db, get_board_hash
from spinner_set import SpinnerSet
from game_room import RoomManager

bg = BoardGenerator()
print("Seeding Hard 4x4 NWL Normal min3 boards...")
for i in range(3):
    t0 = time.time()
    res = bg._generate_board_internal('4x4', 'PLANETS', (100, 200), 'NWL', 'Normal', 3, 'Hard', is_emergency=False, timeout=60, use_added_words=False, mark_used=False)
    board, words, bc, fr, wd, ratio, fbw = res[:7]
    diff = bg.get_difficulty_label(ratio, 4, 4, 'NWL', 1, min_word_length=3)
    wc = SpinnerSet.get_factchecked_wc_range(len(words), use_added_words=False, dictionary='NWL')
    key = serialize_param_key('4x4', '', wc, 'NWL', 'Normal', 3, diff, use_added_words=False)
    data = {"board": board, "all_words": words, "bonus_cell": list(bc) if bc else None, "board_format_ret": fr,
            "all_words_dict": wd, "ratio": ratio, "final_bonus_word": fbw, "word_count_range": wc}
    with get_db() as conn:
        conn.execute("INSERT OR IGNORE INTO pregenerated_boards (param_key, board_json, created_at, board_hash) VALUES (?, ?, ?, ?);",
                     (key, json.dumps(data), time.time(), get_board_hash(board)))
    print(f"seeded {diff} ratio={ratio:.3f} words={len(words)} {time.time()-t0:.1f}s")

rm = RoomManager()
for i in range(3):
    params = {'difficulty': 'Hard', 'board_format': 'Normal', 'dictionary': 'NWL', 'min_word_length': 3,
              'word_count_range': 'random', 'bonus_word_length': 'random'}
    t0 = time.time()
    room = rm.create_room(f"test_cache_{int(time.time()*1000)}", game_type='solo_accumulative', time_limit=60,
                          board_dimensions='4x4', is_solo=True, initial_solo_params=params)
    real = bg.get_difficulty_label(room.current_uniqueness, 4, 4, 'NWL', 1, min_word_length=room.current_min_length)
    print(f"ROOM{i+1} {time.time()-t0:.1f}s label={room.current_difficulty} real={real}")
