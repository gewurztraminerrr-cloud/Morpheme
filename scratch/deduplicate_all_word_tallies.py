import os
import sys
import json
import datetime
import sqlite3
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def deduplicate_word_tallies(base_dir=None):
    if not base_dir:
        base_dir = BASE_DIR
    
    tally_path = os.path.join(base_dir, 'word_tally.log')
    if not os.path.exists(tally_path):
        alt = os.path.join(base_dir, 'morpheme', 'word_tally.log')
        if os.path.exists(alt):
            tally_path = alt
        else:
            print(f"word_tally.log not found in {base_dir}")
            return

    db_path = os.path.join(base_dir, 'morpheme.db')
    if not os.path.exists(db_path):
        alt = os.path.join(base_dir, 'morpheme', 'morpheme.db')
        if os.path.exists(alt):
            db_path = alt

    print(f"=== Deduplicating word tallies and finds in: {base_dir} ===")
    print(f"Tally file: {tally_path}")
    print(f"Database: {db_path}")

    # --- 1. Deduplicate word_tally.log ---
    with open(tally_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Original word_tally.log lines: {len(lines)}")
    backup_path = tally_path + '.bak'
    shutil.copyfile(tally_path, backup_path)
    print(f"Created backup at {backup_path}")

    clean_entries = []
    dupes_count = 0
    dupe_words_affected = set()

    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        
        is_dupe = False
        if clean_entries:
            curr_tally = entry.get('tally', {})
            # Check against recent clean entries in the same room
            for prev in reversed(clean_entries[-20:]):
                if prev.get('room_id') == entry.get('room_id'):
                    try:
                        t_curr = datetime.datetime.fromisoformat(entry['timestamp'])
                        t_prev = datetime.datetime.fromisoformat(prev['timestamp'])
                        diff = abs((t_curr - t_prev).total_seconds())
                        if diff <= 120:
                            prev_tally = prev.get('tally', {})
                            # 1. Exact match duplicate
                            if prev_tally == curr_tally and curr_tally:
                                is_dupe = True
                                break
                            # 2. Subset duplicate (un-cleared words re-tallied in next round)
                            if curr_tally and prev_tally and all(w in prev_tally and curr_tally[w] <= prev_tally[w] for w in curr_tally):
                                is_dupe = True
                                break
                    except Exception:
                        pass
                    break
        
        if is_dupe:
            dupes_count += 1
            for w in entry.get('tally', {}).keys():
                dupe_words_affected.add(w)
        else:
            clean_entries.append(entry)

    print(f"Removed {dupes_count} phantom duplicate round entries from word_tally.log.")
    print(f"Unique words corrected from duplicate logging: {len(dupe_words_affected)}")
    print(f"Clean word_tally.log lines: {len(clean_entries)}")

    # Write cleaned word_tally.log
    with open(tally_path, 'w', encoding='utf-8') as f:
        for entry in clean_entries:
            f.write(json.dumps(entry) + '\n')
    print("Cleaned word_tally.log written successfully.")

    # --- 2. Synchronize word_stats.json via tally_divisions ---
    import tally_divisions as td
    updated = td.sync_word_stats_from_db(db_path=db_path, tally_log_path=tally_path)
    print(f"Synchronized word_stats.json: {updated} words updated.")

    # --- 3. Clean duplicate finds in word_finds table ---
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path, timeout=60)
        c = conn.cursor()
        
        # Check if word_finds table exists
        has_wf = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='word_finds'").fetchone()
        if has_wf:
            # Find duplicate records for the same (word, user_id, room_id) within 120s
            c.execute("SELECT id, word, user_id, timestamp, room_id FROM word_finds ORDER BY word, user_id, room_id, timestamp")
            all_wf = c.fetchall()
            
            ids_to_delete = []
            for i in range(1, len(all_wf)):
                prev = all_wf[i-1]
                curr = all_wf[i]
                if prev[1] == curr[1] and prev[2] == curr[2] and prev[4] == curr[4] and prev[4]:
                    try:
                        ts_p = datetime.datetime.fromisoformat(str(prev[3]).replace('Z', '+00:00'))
                        ts_c = datetime.datetime.fromisoformat(str(curr[3]).replace('Z', '+00:00'))
                        if abs((ts_c - ts_p).total_seconds()) <= 120:
                            ids_to_delete.append(curr[0])
                    except Exception:
                        pass
            
            if ids_to_delete:
                print(f"Pruning {len(ids_to_delete)} duplicate find records from word_finds...")
                batch_size = 500
                for i in range(0, len(ids_to_delete), batch_size):
                    chunk = ids_to_delete[i:i+batch_size]
                    q = f"DELETE FROM word_finds WHERE id IN ({','.join(['?']*len(chunk))})"
                    c.execute(q, chunk)
                conn.commit()
                print("Pruned duplicate finds from word_finds.")

            total_wf = c.execute("SELECT COUNT(*) FROM word_finds").fetchone()[0]
            print(f"Total rows in word_finds after cleanup: {total_wf}")

        # --- 4. Verify target words ---
        print("\n=== Verification ===")
        for test_w in ['FLARE', 'SHARKLIKE', 'ANIMAL', 'LAMINA', 'REMISSION', 'POLITESSE', 'CRATERED']:
            info = td.get_word_division_info(test_w)
            wf_rows = c.execute("""
                SELECT wf.timestamp, u.username, u.country_flag, wf.room_id, wf.round_number
                FROM word_finds wf
                JOIN users u ON wf.user_id = u.id
                WHERE wf.word = ?
                ORDER BY wf.timestamp DESC
            """, (test_w,)).fetchall()
            print(f"{test_w}: Tally count={info.get('count', 0)}, Division={info.get('division_str')}, word_finds rows={len(wf_rows)}")
            for r in wf_rows:
                print(f"  {r[0]} | {r[1]} {r[2]} | {r[3]} R{r[4]}")

        conn.close()

if __name__ == '__main__':
    deduplicate_word_tallies()
