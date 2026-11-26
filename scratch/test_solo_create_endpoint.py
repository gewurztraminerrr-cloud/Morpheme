import sys, time
sys.path.insert(0, '/Users/jeffbabiak')
import app as appmod

c = appmod.app.test_client()
with c.session_transaction() as s:
    s['user_id'] = 999001
    s['username'] = 'solo_endpoint_test'
    s['is_guest'] = True

cfgs = [
    {'board_dimensions': '4x4', 'time_limit': 60, 'dictionary': 'random', 'min_word_length': 'random', 'bonus_word_length': 'random', 'difficulty': 'Hard', 'board_format': 'Mania', 'word_count_range': 'random'},
    {'board_dimensions': '4x4', 'time_limit': 60, 'dictionary': 'random', 'min_word_length': 'random', 'bonus_word_length': 'random', 'difficulty': 'random', 'board_format': 'random', 'word_count_range': 'random'},
    {'board_dimensions': '5x7', 'time_limit': 86400, 'dictionary': 'NWL', 'min_word_length': 'random', 'bonus_word_length': 'random', 'difficulty': 'random', 'board_format': 'random', 'word_count_range': 'random'},
]
bad = 0
for p in cfgs:
    t0 = time.time()
    r = c.post('/api/solo-match/create', json={'parameters': p, 'participants': []})
    d = r.get_json(silent=True)
    ok = r.status_code == 200 and d and d.get('success')
    print(p['board_format'], p['difficulty'], p['board_dimensions'], '->', r.status_code, 'success' if ok else d, f'{time.time()-t0:.2f}s')
    if not ok: bad += 1
print('FAILURES:', bad)
