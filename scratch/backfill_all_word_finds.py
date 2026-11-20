import os
import sys
import glob
import json
import re
import sqlite3
import datetime

def format_ts(ts):
    if not ts:
        return '2026-01-01T00:00:00Z'
    if isinstance(ts, (int, float)):
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    s = str(ts).strip()
    if 'T' in s:
        if s.endswith('Z'):
            return s
        if '+' in s:
            return s.split('+')[0] + 'Z'
        if '.' in s:
            s = s.split('.')[0]
        return s + 'Z'
    if ' ' in s:
        d, t = s.split(' ', 1)
        if '.' in t:
            t = t.split('.')[0]
        return f'{d}T{t}Z'
    return s

def run_backfill(base_dir=None):
    if not base_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    main_db_path = os.path.join(base_dir, 'morpheme.db')
    if not os.path.exists(main_db_path):
        # Check subfolder or parent
        alt_db = os.path.join(base_dir, 'morpheme', 'morpheme.db')
        if os.path.exists(alt_db):
            main_db_path = alt_db

    print(f"=== Backfilling word_finds for database: {main_db_path} ===")
    conn = sqlite3.connect(main_db_path, timeout=60)
    c = conn.cursor()

    # 1. Ensure word_finds table and indices exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS word_finds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            room_id TEXT,
            round_number INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_word_finds_word ON word_finds(word)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_word_finds_user ON word_finds(user_id)')
    conn.commit()

    # User mappings
    user_rows = c.execute('SELECT id, username FROM users').fetchall()
    user_map = {r[1].lower(): r[0] for r in user_rows}
    uid_to_name = {r[0]: r[1] for r in user_rows}
    print(f"Loaded {len(user_map)} registered users from main DB.")

    existing_keys = set()
    for row in c.execute('SELECT word, user_id, timestamp FROM word_finds').fetchall():
        existing_keys.add((row[0], row[1], str(row[2])[:19]))
    print(f"Existing entries in word_finds: {len(existing_keys)}")

    new_inserts = []

    from collections import defaultdict
    recent_room_finds = defaultdict(list)
    for row in c.execute('SELECT word, user_id, room_id, timestamp FROM word_finds').fetchall():
        try:
            dt = datetime.datetime.fromisoformat(str(row[3]).replace('Z', '+00:00'))
            recent_room_finds[(row[0], row[1], row[2] or '')].append(dt)
        except Exception:
            pass

    def add_find(word, uid, ts_val, room_id=None, round_num=None):
        w_up = str(word).strip().upper()
        if len(w_up) < 2 or not uid or uid == -1 or uid not in uid_to_name:
            return
        if str(uid_to_name[uid]).startswith('Guest_'):
            return
        if room_id and ('24h' in str(room_id) or '86400' in str(room_id)):
            return
        iso = format_ts(ts_val)
        key = (w_up, uid, iso[:19])
        if key in existing_keys:
            return

        room_key = (w_up, uid, room_id or '')
        try:
            curr_dt = datetime.datetime.fromisoformat(iso.replace('Z', '+00:00'))
            if room_key in recent_room_finds:
                for prev_dt in recent_room_finds[room_key]:
                    if abs((curr_dt - prev_dt).total_seconds()) <= 120:
                        return
            recent_room_finds[room_key].append(curr_dt)
        except Exception:
            pass

        existing_keys.add(key)
        new_inserts.append((w_up, uid, iso, room_id, round_num))

    # --- SOURCE 1: SQLite Databases & Snapshots ---
    candidate_dbs = []
    # Production paths
    for p in glob.glob('/home/morpheme/**/*.db*', recursive=True) + glob.glob('/home/morpheme/*.db*'):
        candidate_dbs.append(p)
    # Local paths
    local_patterns = [
        f'{base_dir}/*.db*',
        f'{base_dir}/instance/*.db*',
        f'{base_dir}/scratch/*.db*',
        f'{base_dir}/dictionaries/*.db*',
        f'{base_dir}/boggle-gen/*.db*',
        f'{base_dir}/morpheme/*.db*',
    ]
    for pat in local_patterns:
        candidate_dbs.extend(glob.glob(pat))
    candidate_dbs = sorted(set(candidate_dbs))

    seen_dbs = set()
    db_count = 0
    for p in candidate_dbs:
        if p in seen_dbs or 'wal' in p or 'shm' in p or 'developer' in p or '.gemini' in p:
            continue
        seen_dbs.add(p)
        try:
            dbc = sqlite3.connect(p, timeout=10)
            cur = dbc.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            
            # Map foreign DB users by username if needed
            db_uid_to_main_uid = {}
            if 'users' in tables:
                for f_uid, f_uname in cur.execute("SELECT id, username FROM users").fetchall():
                    if f_uname and f_uname.lower() in user_map:
                        db_uid_to_main_uid[f_uid] = user_map[f_uname.lower()]

            if 'round_history' in tables:
                q = '''
                    SELECT user_id, words_json, timestamp, room_id, round_number, round_duration, game_type 
                    FROM round_history 
                    WHERE words_json IS NOT NULL AND words_json != ""
                '''
                for uid, wj, ts, rid, rnum, rdur, gtype in cur.execute(q).fetchall():
                    if not uid or uid == -1:
                        continue
                    if rdur and rdur >= 7200:
                        continue
                    if gtype in ('subanagrams', '24h'):
                        continue
                    if rid and ('24h' in rid or '86400' in rid):
                        continue
                    target_uid = db_uid_to_main_uid.get(uid, uid)
                    if target_uid not in uid_to_name:
                        continue
                    try:
                        words = json.loads(wj)
                        if isinstance(words, list):
                            for item in words:
                                if isinstance(item, dict):
                                    if item.get('is_penalty') or item.get('points', 0) <= 0:
                                        continue
                                    w = item.get('word', '')
                                    item_ts = item.get('timestamp') or ts
                                else:
                                    w = str(item)
                                    item_ts = ts
                                add_find(w, target_uid, item_ts, rid, rnum)
                    except Exception:
                        pass

            if 'tournament_scores' in tables:
                q = '''
                    SELECT user_id, submitted_words, submitted_at, tournament_id, round_number 
                    FROM tournament_scores 
                    WHERE submitted_words IS NOT NULL AND submitted_words != ""
                '''
                for uid, wj, ts, tid, rnum in cur.execute(q).fetchall():
                    target_uid = db_uid_to_main_uid.get(uid, uid)
                    if target_uid not in uid_to_name:
                        continue
                    try:
                        words = json.loads(wj)
                        if isinstance(words, list):
                            for item in words:
                                w = item.get('word', '') if isinstance(item, dict) else str(item)
                                item_ts = (item.get('timestamp') if isinstance(item, dict) else None) or ts
                                add_find(w, target_uid, item_ts, f'tourn_{tid}', rnum)
                    except Exception:
                        pass

            if 'private_match_turns' in tables:
                q = '''
                    SELECT user_id, submitted_words, submitted_at, match_id, round_number 
                    FROM private_match_turns 
                    WHERE submitted_words IS NOT NULL AND submitted_words != ""
                '''
                for uid, wj, ts, mid, rnum in cur.execute(q).fetchall():
                    target_uid = db_uid_to_main_uid.get(uid, uid)
                    if target_uid not in uid_to_name:
                        continue
                    try:
                        words = json.loads(wj)
                        if isinstance(words, list):
                            for item in words:
                                w = item.get('word', '') if isinstance(item, dict) else str(item)
                                item_ts = (item.get('timestamp') if isinstance(item, dict) else None) or ts
                                add_find(w, target_uid, item_ts, f'priv_{mid}', rnum)
                    except Exception:
                        pass

            dbc.close()
            db_count += 1
        except Exception:
            pass

    print(f"Processed {db_count} database snapshot files. New finds queued: {len(new_inserts)}")

    # --- SOURCE 2: launcher.log ---
    log_candidates = [
        os.path.join(base_dir, 'launcher.log'),
        os.path.join(base_dir, 'morpheme', 'launcher.log'),
        '/home/morpheme/launcher.log',
        '/home/morpheme/morpheme/launcher.log',
    ]
    seen_log = set()
    for log_path in log_candidates:
        if os.path.exists(log_path) and log_path not in seen_log:
            seen_log.add(log_path)
            print(f"Scanning launcher.log at {log_path}...")
            last_ts = '2026-05-15T16:30:26Z'
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if 'Time: ' in line:
                            m = re.search(r'Time:\s*(\d+\.\d+)', line)
                            if m:
                                last_ts = format_ts(float(m.group(1)))
                        elif '--- Morpheme Startup:' in line:
                            m = re.search(r'--- Morpheme Startup:\s*([A-Za-z0-9: ]+)\s*---', line)
                            if m:
                                try:
                                    dt = datetime.datetime.strptime(m.group(1).strip(), '%a %b %d %H:%M:%S UTC %Y')
                                    last_ts = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                                except Exception:
                                    pass
                        elif '[Submit-Diag]' in line:
                            m = re.search(r'\[Submit-Diag\] User (\d+) submitting word \'([^\']+)\' for room ([\w\-]+)', line)
                            if m:
                                uid = int(m.group(1))
                                word = m.group(2)
                                rid = m.group(3)
                                add_find(word, uid, last_ts, rid, None)
            except Exception as e:
                print(f"Error parsing {log_path}: {e}")

    # --- SOURCE 3: word_tally.log ---
    tally_candidates = [
        os.path.join(base_dir, 'word_tally.log'),
        os.path.join(base_dir, 'morpheme', 'word_tally.log'),
        '/home/morpheme/word_tally.log',
        '/home/morpheme/morpheme/word_tally.log',
    ]
    seen_tally = set()
    for tally_path in tally_candidates:
        if os.path.exists(tally_path) and tally_path not in seen_tally:
            seen_tally.add(tally_path)
            print(f"Scanning word_tally.log at {tally_path}...")
            try:
                with open(tally_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            rid = data.get('room_id', '')
                            ts = data.get('timestamp', '')
                            rnum = data.get('round')
                            tally = data.get('tally', {})
                            if rid.startswith('practice_'):
                                m = re.match(r'practice_(.+?)_\d+', rid)
                                if m:
                                    uname = m.group(1).lower()
                                    if uname in user_map:
                                        p_uid = user_map[uname]
                                        for w, cnt in tally.items():
                                            for _ in range(cnt):
                                                add_find(w, p_uid, ts, rid, rnum)
                        except Exception:
                            pass
            except Exception as e:
                print(f"Error parsing {tally_path}: {e}")

    # --- SOURCE 4: Confirmed Historical Matches Pruned from round_history ---
    confirmed_historical = [
        # May 17, 2026: pub_v2_split_4x4_45 round 3
        ("pub_v2_split_4x4_45", 3, "2026-05-17T15:59:02Z", [
            ("ANIMALS", "jeffb"), ("CENTARE", "jeffb"), ("ANIMA", "jeffb"), ("LAMINAE", "jeffb"),
            ("LAMINAR", "jeffb"), ("ANIMAL", "jeffb"), ("MALINE", "jeffb"), ("CAME", "jeffb"),
            ("CENTARES", "jeffb"), ("MALINES", "jeffb"), ("LAMINA", "jeffb"), ("ANTAS", "jeffb"),
            ("LIANAS", "jeffb"), ("ANTAE", "jeffb"), ("LINES", "jeffb"), ("ANTA", "jeffb"),
            ("LIANES", "jeffb"), ("LIANE", "jeffb"), ("LIANA", "jeffb"), ("CENTAS", "jeffb"), ("CENTS", "jeffb"),
            ("ANIMALS", "jeffjeff"), ("CENTARE", "jeffjeff"), ("ANIMA", "jeffjeff"), ("ANIMAL", "jeffjeff"),
            ("MALINE", "jeffjeff"), ("CENTARES", "jeffjeff"), ("MALINES", "jeffjeff"), ("LAMINA", "jeffjeff")
        ]),
        # May 20, 2026: pub_v2_accumulative_4x6_45 round 3
        ("pub_v2_accumulative_4x6_45", 3, "2026-05-20T04:21:09Z", [
            ("ANIMAL", "jeffy")
        ]),
        # May 21, 2026: pub_v2_accumulative_5x7_45 round 31
        ("pub_v2_accumulative_5x7_45", 31, "2026-05-21T01:41:15Z", [
            (w, "jeffy") for w in [
                "BENNET", "LAMINAR", "BENNETS", "PARENTS", "ANIMAL", "PANIER", "PANIERS",
                "PARLEY", "PANNIER", "RENNET", "PARRELS", "RENNETS", "FENNIER", "PARENT",
                "PARRAL", "TENNER", "PARREL", "PARRALS", "LAMINAE", "PARLEYS", "PANNIERS",
                "PANSIES", "LAMINA", "PARLES"
            ]
        ]),
        # June 2, 2026: pub_v2_accumulative_4x4_45 round 5
        ("pub_v2_accumulative_4x4_45", 5, "2026-06-02T01:03:21Z", [
            (w, "jeffy") for w in [
                "LAMINA", "ANIMAL", "SPUME", "LAMINAE", "ANIMAS", "ANIMUS", "ANIMA", "ANIME", "LAMINAS"
            ]
        ]),
        # June 14, 2026: pub_v2_accumulative_4x4_45 round 10
        ("pub_v2_accumulative_4x4_45", 10, "2026-06-14T16:54:32Z", [
            ("ANIMAL", "jeffb"), ("ANIMALS", "jeffb")
        ]),
        # June 17, 2026: pub_v2_accumulative_4x4_45 round 2695
        ("pub_v2_accumulative_4x4_45", 2695, "2026-06-17T22:47:59Z", [
            ("ANIMAL", "jeffy")
        ]),
        # October 26, 2026: pub_v2_accumulative_6x8_45 round 7
        ("pub_v2_accumulative_6x8_45", 7, "2026-10-26T23:16:38Z", [
            ("SHARKLIKE", "jeffb")
        ]),
    ]

    for rid, rnum, rts, words_list in confirmed_historical:
        for w, uname in words_list:
            if uname.lower() in user_map:
                add_find(w, user_map[uname.lower()], rts, rid, rnum)

    # --- Batch Insert ---
    print(f"Total new word finds to insert: {len(new_inserts)}")
    if new_inserts:
        batch_size = 2000
        for i in range(0, len(new_inserts), batch_size):
            chunk = new_inserts[i:i+batch_size]
            c.executemany('''
                INSERT INTO word_finds (word, user_id, timestamp, room_id, round_number)
                VALUES (?, ?, ?, ?, ?)
            ''', chunk)
        conn.commit()

    total_count = c.execute('SELECT COUNT(*) FROM word_finds').fetchone()[0]
    unique_words = c.execute('SELECT COUNT(DISTINCT word) FROM word_finds').fetchone()[0]
    print(f"=== Backfill Complete! word_finds total rows: {total_count}, unique words: {unique_words} ===")

    # Verify LAMINA and ANIMAL
    for test_w in ['LAMINA', 'ANIMAL', 'LAMINAR', 'LAMINAE', 'MALINE', 'CENTARE']:
        cur = c.execute('''
            SELECT wf.timestamp, u.username, u.country_flag, wf.room_id, wf.round_number
            FROM word_finds wf
            JOIN users u ON wf.user_id = u.id
            WHERE wf.word = ?
            ORDER BY wf.timestamp DESC
        ''', (test_w,))
        rows = cur.fetchall()
        print(f"Verification for {test_w} ({len(rows)} finds):")
        for r in rows[:5]:
            print(f"  {r[0]} | {r[1]} {r[2]} | {r[3]} R{r[4]}")

    conn.close()

if __name__ == '__main__':
    run_backfill()
