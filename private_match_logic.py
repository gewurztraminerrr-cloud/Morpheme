import sqlite3
import json
import time
import random
from typing import List, Dict
from scoring import calculate_word_score
from rating_logic import calculate_proportional_rating_change

class PrivateMatchManager:
    def __init__(self, db_path='morpheme.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS private_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creator_id INTEGER NOT NULL,
                match_type TEXT NOT NULL, -- 'solo', 'with_friends'
                parameters TEXT NOT NULL, -- JSON
                status TEXT DEFAULT 'active', -- 'active', 'completed', 'expired'
                created_at REAL,
                last_activity REAL,
                current_round INTEGER DEFAULT 0,
                FOREIGN KEY(creator_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS private_match_players (
                match_id INTEGER,
                user_id INTEGER,
                username TEXT, -- Cache for AI bots or invited users
                is_ai INTEGER DEFAULT 0,
                ai_rating INTEGER,
                status TEXT DEFAULT 'accepted', -- 'invited', 'accepted', 'declined'
                PRIMARY KEY(match_id, user_id),
                FOREIGN KEY(match_id) REFERENCES private_matches(id)
            );

            CREATE TABLE IF NOT EXISTS private_match_rounds (
                match_id INTEGER,
                round_number INTEGER,
                board_data TEXT, -- JSON
                bonus_word TEXT,
                bonus_cell TEXT, -- NEW: (r, c) or (f, r, c)
                word_count_range TEXT, -- NEW: Specifically selected range
                all_words TEXT, -- JSON of all valid words on board
                start_time REAL,
                end_time REAL,
                PRIMARY KEY(match_id, round_number)
            );

            CREATE TABLE IF NOT EXISTS private_match_turns (
                match_id INTEGER,
                round_number INTEGER,
                user_id INTEGER,
                score INTEGER DEFAULT 0,
                submitted_words TEXT, -- JSON list of objects {word, points, timestamp}
                submitted_at REAL,
                PRIMARY KEY(match_id, round_number, user_id)
            );
            
            CREATE TABLE IF NOT EXISTS match_invites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                sender_id INTEGER,
                recipient_username TEXT,
                status TEXT DEFAULT 'pending',
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS private_match_starts (
                match_id INTEGER,
                round_number INTEGER,
                user_id INTEGER,
                start_time REAL,
                PRIMARY KEY(match_id, round_number, user_id)
            );
        ''')
        conn.commit()
        
        # MIGRATION: Add word_count_range column if it doesn't exist
        try:
            conn.execute('ALTER TABLE private_match_rounds ADD COLUMN word_count_range TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass # Column likely already exists
            
        try:
            conn.execute('ALTER TABLE private_match_rounds ADD COLUMN bonus_cell TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass # Column likely already exists

        try:
            conn.execute('ALTER TABLE private_match_rounds ADD COLUMN board_format TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass # Column likely already exists
            
        conn.close()

    def get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def cleanup_old_data(self):
        """Delete matches and invites older than 7 days"""
        try:
            conn = sqlite3.connect(self.db_path)
            now = time.time()
            seven_days_ago = now - (7 * 24 * 60 * 60)
            
            # 1. Get IDs of matches older than 7 days (based on created_at and last_activity)
            old_matches = conn.execute("SELECT id FROM private_matches WHERE created_at < ? OR (last_activity IS NOT NULL AND last_activity < ?)", 
                                       (seven_days_ago, seven_days_ago)).fetchall()
            match_ids = [m[0] for m in old_matches]
            
            if match_ids:
                placeholders = ','.join(['?'] * len(match_ids))
                conn.execute(f"DELETE FROM private_match_players WHERE match_id IN ({placeholders})", match_ids)
                conn.execute(f"DELETE FROM private_match_rounds WHERE match_id IN ({placeholders})", match_ids)
                conn.execute(f"DELETE FROM private_match_turns WHERE match_id IN ({placeholders})", match_ids)
                conn.execute(f"DELETE FROM match_invites WHERE match_id IN ({placeholders})", match_ids)
                conn.execute(f"DELETE FROM private_match_starts WHERE match_id IN ({placeholders})", match_ids) # Added for new table
                conn.execute(f"DELETE FROM private_matches WHERE id IN ({placeholders})", match_ids)
                
            # 2. Cleanup stale standalone invites
            conn.execute("DELETE FROM match_invites WHERE created_at < ?", (seven_days_ago,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Cleanup Error: {e}")

    def create_match(self, creator_id, match_type, parameters, participants=None):
        """
        participants: list of {'user_id': id, 'username': name, 'is_ai': bool, 'ai_rating': optional}
        """
        conn = self.get_db()
        now = time.time()
        
        # 1. Verify all non-AI participants exist
        if participants:
            for p in participants:
                if not p.get('is_ai'):
                    user = conn.execute('SELECT id FROM users WHERE username = ?', (p['username'],)).fetchone()
                    if not user:
                        conn.close()
                        raise ValueError(f"User '{p['username']}' does not exist.")
        
        # 2. Create Match Entry (Start at round 0 so it's not active until board is ready)
        cur = conn.execute('''
            INSERT INTO private_matches (creator_id, match_type, parameters, created_at, last_activity, current_round)
            VALUES (?, ?, ?, ?, ?, 0)
        ''', (creator_id, match_type, json.dumps(parameters), now, now))
        match_id = cur.lastrowid
        
        # 3. Add Participants
        # Creator is always in
        conn.execute('''
            INSERT INTO private_match_players (match_id, user_id, username, status)
            VALUES (?, ?, (SELECT username FROM users WHERE id=?), 'accepted')
        ''', (match_id, creator_id, creator_id))
        
        if participants:
            for p in participants:
                # If they are AI, add them directly
                if p.get('is_ai'):
                    conn.execute('''
                        INSERT INTO private_match_players (match_id, user_id, username, is_ai, ai_rating, status)
                        VALUES (?, ?, ?, 1, ?, 'accepted')
                    ''', (match_id, -random.randint(1000, 999999), p['username'], p.get('ai_rating', 1200)))
                else:
                    # If they are invited users
                    # We create an invitation. They only become 'accepted' when they click it.
                    conn.execute('''
                        INSERT INTO match_invites (match_id, sender_id, recipient_username, created_at)
                        VALUES (?, ?, ?, ?)
                    ''', (match_id, creator_id, p['username'], now))

        conn.commit()
        
        # 4. Generate first round board (Pass existing conn to avoid locks)
        self.generate_round(match_id, 1, parameters, conn=conn)

        # 5. NOW set current_round to 1 (making it active)
        conn.execute('UPDATE private_matches SET current_round = 1 WHERE id = ?', (match_id,))
        conn.commit()
        
        conn.close()
        return match_id

    def generate_round(self, match_id, round_number, parameters, conn=None):
        from board_generator import BoardGenerator
        from word_validator import word_validator
        
        # USE UNIQUE SEEDING TO PREVENT BOARD REUSE ACROSS PROCESSES
        import random
        random.seed()

        bg = BoardGenerator()
        bonus_cell = None # Initialize to avoid NameError
        dims = parameters.get('board_dimensions', '4x4')
        dict_name = parameters.get('dictionary', 'NWL')
        min_len = parameters.get('min_word_length', 3)

        # Bonus word selection BEFORE board generation to allow embedding
        bonus_len = parameters.get('bonus_word_length', 0)
        if bonus_len == 0:
             # Force a bonus word for private rooms if none was specified (standard UI behavior)
             bonus_len = random.randint(6, 10)
        bonus_word = ""
        # Check if the format allows a bonus word
        target_format = parameters.get('board_format', 'Normal')
        
        # 0. Handle "Mania" without a prefix (ensure 33% vowels, 67% consonants)
        if target_format.strip() == 'Mania':
            import random
            if random.random() < 0.33:
                mania_letter = random.choice('AEIOU')
            else:
                mania_letter = random.choice('BCDFGHJKLMNPQRSTVWXYZ')
            target_format = f"{mania_letter} Mania"
            
        from spinner_set import SpinnerSet
        
        target_difficulty = parameters.get('difficulty', 'random')
        if target_difficulty == 'random':
            target_difficulty = SpinnerSet._spin_difficulty()
            
        # Randomize dictionary for each round if it's not fixed
        if dict_name == 'random':
            dict_name = SpinnerSet._spin_dictionary()

        target_range = parameters.get('word_count_range', 'random')
        if target_range == 'random':
            target_range = SpinnerSet._spin_word_count(dict_name, min_len, target_difficulty, dims)
        
        fmt_check = target_format.lower()
        if bonus_len > 0:
            from word_validator import word_validator
            dictionary_set = word_validator.csw_words if dict_name == 'CSW' else word_validator.nwl_words
            potential_dict_words = [w for w in dictionary_set if len(w) == bonus_len]
            if potential_dict_words:
                bonus_word = random.choice(potential_dict_words)
        
        # Generate board
        res = bg.generate_board(
            dimensions=dims,
            bonus_word=bonus_word,
            word_count_range=target_range,
            dictionary=dict_name,
            board_format=target_format,
            min_word_length=min_len,
            difficulty=target_difficulty
        )
        board, all_words_on_board, bonus_cell, updated_format, all_words_dict = res[0], res[1], res[2], res[3], res[4]
        # Use the updated format (e.g. "X Mania" instead of just "Mania")
        target_format = updated_format
        
        # User Request Fix: Ensure private matches also respect the lockdown for Normal format
        f_low = str(target_format).lower()
        if 'bonus letter' not in f_low and 'either' not in f_low:
             bonus_cell = None
        
        now = time.time()

        now = time.time()
        # Round-start time is when board is generated, but turn-timers start on client when they click Play.
        # We set end_time to 0 to signal it hasn't started its timed phase yet.
        end_time = 0
        
        external_conn = conn is not None
        if not external_conn:
            conn = self.get_db()

        try:
            conn.execute('''
                INSERT INTO private_match_rounds (match_id, round_number, board_data, bonus_word, bonus_cell, word_count_range, all_words, start_time, end_time, board_format)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (match_id, round_number, json.dumps(board), bonus_word, json.dumps(bonus_cell) if bonus_cell else None, json.dumps(target_range), json.dumps(all_words_dict), now, end_time, target_format))
        except Exception as e:
            print(f"FAILED TO INSERT ROUND: {e}")
            print(f"DEBUG LOCALS: board={ 'board' in locals() }, bonus_cell={ 'bonus_cell' in locals() }, target_format={ 'target_format' in locals() }")
            raise e
        
        if not external_conn:
            conn.commit()
            conn.close()

    def get_matches_for_user(self, user_id, username):
        """
        Returns { 'your_turn': [], 'their_turn': [], 'history': [] }
        """
        self.cleanup_old_data()
        
        conn = self.get_db()
        conn.row_factory = sqlite3.Row
        
        # Matches where user is a participant
        all_p_matches = conn.execute('''
            SELECT m.*, mp.status as my_status
            FROM private_matches m
            JOIN private_match_players mp ON m.id = mp.match_id
            WHERE mp.user_id = ? AND m.status != 'expired' AND m.match_type != 'solo'
        ''', (user_id,)).fetchall()
        
        results = {'your_turn': [], 'their_turn': [], 'history': []}
        
        now = time.time()
        
        for m in all_p_matches:
            match_id = m['id']
            curr_round = m['current_round']
            
            # Skip matches that are still initializing (round 0)
            if curr_round == 0:
                continue
            
            # Check if user has submitted for this round
            turn = conn.execute('''
                SELECT 1 FROM private_match_turns 
                WHERE match_id = ? AND round_number = ? AND user_id = ?
            ''', (match_id, curr_round, user_id)).fetchone()
            
            # Check round timing and get round-specific parameters
            round_info = conn.execute('''
                SELECT end_time, word_count_range FROM private_match_rounds 
                WHERE match_id = ? AND round_number = ?
            ''', (match_id, curr_round)).fetchone()
            
            if round_info and round_info['end_time'] > 0 and round_info['end_time'] < now:
                # Expired match
                conn.execute("UPDATE private_matches SET status = 'expired' WHERE id = ?", (match_id,))
                continue

            # Participants (Accepted)
            players = conn.execute('''
                SELECT user_id, username, is_ai, 'accepted' as status FROM private_match_players 
                WHERE match_id = ?
            ''', (match_id,)).fetchall()
            players_list = [dict(p) for p in players]
            
            # Invited (Pending)
            invites = conn.execute('''
                SELECT -1 as user_id, recipient_username as username, 0 as is_ai, 'pending' as status 
                FROM match_invites WHERE match_id = ?
            ''', (match_id,)).fetchall()
            players_list.extend([dict(i) for i in invites])
            
            # Submissions for this round
            submissions = conn.execute('''
                SELECT user_id FROM private_match_turns 
                WHERE match_id = ? AND round_number = ?
            ''', (match_id, curr_round)).fetchall()
            submitted_ids = [s['user_id'] for s in submissions]
            
            # --- FILTER ABANDONED MATCHES ---
            # (Fixing 'With Friends' matches in history with only one player)
            # If no one joined or everyone declined, hide it.
            if len(players_list) <= 1:
                continue

            match_data = dict(m)
            match_data['parameters'] = json.loads(m['parameters'])
            
            # Override with round-specific range if it exists
            if round_info and round_info['word_count_range']:
                try:
                    match_data['parameters']['word_count_range'] = json.loads(round_info['word_count_range'])
                except:
                    pass
                    
            match_data['players'] = players_list
            match_data['round_info'] = {'end_time': round_info['end_time']} if round_info else {}
            
            if not turn:
                results['your_turn'].append(match_data)
            else:
                # Check if anyone else (accepted or pending) still has to go
                others_pending = False
                for p in players_list:
                    # Status 'pending' means they haven't joined yet, but we're still waiting on them
                    if p['status'] in ('accepted', 'pending') and not p['is_ai'] and p['user_id'] != user_id and p['user_id'] not in submitted_ids:
                        others_pending = True
                        break
                
                if others_pending:
                    results['their_turn'].append(match_data)
                else:
                    results['history'].append(match_data)
                    
        # Sort history by last activity and limit to 25
        results['history'].sort(key=lambda x: x.get('last_activity', 0), reverse=True)
        results['history'] = results['history'][:25]

        conn.close()
        return results

    def record_start_time(self, match_id, round_number, user_id):
        """Records when a user starts their turn, or returns the existing start time"""
        conn = self.get_db()
        try:
            # Check if turn already exists. If it exists, we strictly CANNOT reset it.
            turn = conn.execute('''
                SELECT 1 FROM private_match_turns 
                WHERE match_id = ? AND round_number = ? AND user_id = ?
            ''', (match_id, round_number, user_id)).fetchone()
            
            row = conn.execute('''
                SELECT start_time FROM private_match_starts 
                WHERE match_id = ? AND round_number = ? AND user_id = ?
            ''', (match_id, round_number, user_id)).fetchone()
            
            now = time.time()
            if row:
                st = row['start_time']
                # STALE CHECK: If it was recorded more than 30 mins ago AND they haven't submitted yet, 
                # we'll allow a reset (this handles cases where the browser crashed or they disconnected).
                if (now - st > 1800) and not turn:
                    conn.execute('''
                        DELETE FROM private_match_starts 
                        WHERE match_id = ? AND round_number = ? AND user_id = ?
                    ''', (match_id, round_number, user_id))
                    # Fall through to record a fresh one below
                else:
                    return st
            
            # Record it now
            conn.execute('''
                INSERT INTO private_match_starts (match_id, round_number, user_id, start_time)
                VALUES (?, ?, ?, ?)
            ''', (match_id, round_number, user_id, now))
            conn.commit()
            return now
        except Exception as e:
            print(f"Error in record_start_time: {e}")
            return time.time()
        finally:
            conn.close()

    def submit_turn(self, match_id, round_number, user_id, words_data, score):
        conn = self.get_db()
        try:
            match_id = int(match_id)
            round_number = int(round_number)
            user_id = int(user_id)
            now = time.time()
            
            # Check if already submitted
            existing = conn.execute('SELECT 1 FROM private_match_turns WHERE match_id=? AND round_number=? AND user_id=?',
                                  (match_id, round_number, user_id)).fetchone()
            if existing:
                # Idempotent success (avoid double-submit error)
                conn.close()
                return

            # Record Turn (Enforce floor of 0)
            score = max(0, int(score))

            conn.execute('''
                INSERT INTO private_match_turns (match_id, round_number, user_id, score, submitted_words, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (match_id, round_number, user_id, score, json.dumps(words_data), now))
            
            # If all humans have submitted (including invited ones who haven't accepted yet), generate AI turns
            players = conn.execute('SELECT user_id, is_ai, ai_rating FROM private_match_players WHERE match_id = ?', (match_id,)).fetchall()
            invites = conn.execute('SELECT 1 FROM match_invites WHERE match_id = ? AND status = ?', (match_id, 'pending')).fetchall()
            
            submissions = conn.execute('SELECT user_id FROM private_match_turns WHERE match_id = ? AND round_number = ?', (match_id, round_number)).fetchall()
            submitted_ids = [s['user_id'] for s in submissions]
            
            print(f"DEBUG: Match {match_id} Round {round_number}. Submitted: {submitted_ids}. Players: {[p['user_id'] for p in players]}. Pending Invites: {len(invites)}")

            humans = [p for p in players if not p['is_ai']]
            ais = [p for p in players if p['is_ai']]
            
            # Only consider humans done if all accepted humans have submitted AND no pending invites remain
            all_humans_done = all(h['user_id'] in submitted_ids for h in humans) and len(invites) == 0
            
            if all_humans_done and ais:
                match_data = conn.execute('SELECT parameters FROM private_matches WHERE id = ?', (match_id,)).fetchone()
                duration = 60
                if match_data:
                    params = json.loads(match_data['parameters'])
                    duration = params.get('time_limit', 60)

                round_data = conn.execute('SELECT * FROM private_match_rounds WHERE match_id = ? AND round_number = ?', (match_id, round_number)).fetchone()
                if round_data:
                    rd = dict(round_data)
                    all_possible_words = json.loads(rd['all_words'])
                    bonus_word = rd['bonus_word']
                    board_format = rd.get('board_format', 'Normal')
                    bonus_cell_str = rd.get('bonus_cell', None)
                    bonus_cell = json.loads(bonus_cell_str) if bonus_cell_str else None
                    
                    for ai in ais:
                        if ai['user_id'] not in submitted_ids:
                            ai_words, ai_score = self.generate_ai_submission(
                                ai['ai_rating'], 
                                all_possible_words, 
                                bonus_word, 
                                board_format=board_format,
                                bonus_cell=bonus_cell,
                                duration=duration
                            )
                            conn.execute('''
                                INSERT INTO private_match_turns (match_id, round_number, user_id, score, submitted_words, submitted_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                            ''', (match_id, round_number, ai['user_id'], ai_score, json.dumps(ai_words), now))

            # --- ROUND ADVANCEMENT ---
            # Re-fetch submissions to include newly generated AI turns
            submissions = conn.execute('SELECT user_id FROM private_match_turns WHERE match_id = ? AND round_number = ?', (match_id, round_number)).fetchall()
            submitted_ids = [s['user_id'] for s in submissions]
            
            all_done = all(p['user_id'] in submitted_ids for p in players)
            
            if all_done:
                # --- SPLIT POINTS RECALCULATION (Private Match) ---
                match_row = conn.execute('SELECT parameters FROM private_matches WHERE id = ?', (match_id,)).fetchone()
                if match_row:
                    params = json.loads(match_row['parameters'])
                    if params.get('game_type') == 'split':
                        print(f"DEBUG: All players done for Split Points match {match_id}. Recalculating scores...")
                        # 1. Gather ALL submissions for this round
                        turns = conn.execute('SELECT user_id, submitted_words FROM private_match_turns WHERE match_id = ? AND round_number = ?', (match_id, round_number)).fetchall()
                        user_submissions = {t['user_id']: json.loads(t['submitted_words']) for t in turns}
                        
                        # 2. Count finders for each word
                        word_finders = {} # {WORD: [user_id, ...]}
                        for uid, words in user_submissions.items():
                            for w in words:
                                word_text = w['word'].upper()
                                if word_text not in word_finders:
                                    word_finders[word_text] = []
                                word_finders[word_text].append(uid)
                        
                        # 3. Recalculate each user's score and points
                        for uid, words in user_submissions.items():
                            new_total_score = 0
                            for w in words:
                                word_text = w['word'].upper()
                                finders = word_finders.get(word_text, [])
                                count = len(finders)
                                
                                # Original score was for "only me"
                                original_points = w.get('points', 0)
                                
                                # If it's a negative penalty, everyone gets -3 (NOT split)
                                if original_points < 0:
                                    w['shared_count'] = 1
                                    # No change to pts
                                else:
                                    w['shared_count'] = count
                                    # SPLIT LOGIC: (points + count -1) // count
                                    new_pts = (original_points + count - 1) // count
                                    w['points'] = new_pts
                                    # Update score_details if present
                                    if 'score_details' in w and w['score_details']:
                                        sd = w['score_details']
                                        sd['total'] = new_pts
                                        sd['base'] = (sd.get('base', 0) + count - 1) // count
                                        sd['bonus_word_points'] = (sd.get('bonus_word_points', 0) + count - 1) // count
                                        sd['bonus_letter_points'] = (sd.get('bonus_letter_points', 0) + count - 1) // count

                                new_total_score += w['points']
                                
                            # Enforce floor of 0
                            new_total_score = max(0, new_total_score)
                            
                            # 4. Update the turn in the DB
                            conn.execute('UPDATE private_match_turns SET score = ?, submitted_words = ? WHERE match_id = ? AND round_number = ? AND user_id = ?',
                                         (new_total_score, json.dumps(words), match_id, round_number, uid))
                        
                # Apply Rating Changes
                try:
                    self._apply_match_ratings(match_id, round_number, conn)
                except Exception as re:
                    print(f"Rating Error in Private Match {match_id}: {re}")

                # Mark match as completed instead of advancing round
                # This ensures it moves to History correctly.
                # Rematch can be used to start a fresh match.
                conn.execute("UPDATE private_matches SET status = 'completed', last_activity = ? WHERE id = ?", (now, match_id))
                conn.commit()
            else:
                conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"ERROR in submit_turn: {e}")
            raise e
        finally:
            conn.close()

    def generate_ai_submission(self, rating, possible_words, bonus_word, board_format='Normal', bonus_cell=None, duration=60):
        # AI Logic (WPM Model):
        # Rating 800: ~4.0 WPM 
        # Rating 1200: ~10.0 WPM
        # Rating 3000: ~45 WPM (Elite)
        
        if rating is None: rating = 1200
        r = max(400, min(3000, rating))
        
        # Calculate WPM based on rating (linear scale, floor at 2 WPM for better gameplay)
        # 800 -> 4, 1200 -> 10, 3000 -> 45
        # wpm = (r / 60) - 10? No.
        # Adjusted formula:
        wpm = max(2.0, (r / 65.0))
        
        # Total words count based on duration
        count = int((duration / 60.0) * wpm)
        
        if not possible_words:
             return [], 0
            
        # Avoid 0
        count = max(1, min(count, len(possible_words)))

        # Sort words to identify high-scorers
        # We'll score all possible words first to pick the best ones
        word_scores = []
        is_dict = isinstance(possible_words, dict)
        
        for w in possible_words:
            # OPTIMIZATION: Use path from dictionary if available to avoid slow DFS
            w_path = possible_words[w] if is_dict else None
            
            # Note: We don't have the full board object here, but calculate_word_score 
            # handles basic format scoring without it if needed (path omitted).
            # For AI, we assume they hit the bonus cell if it exists (simplified).
            word_scores.append((w, calculate_word_score(
                w, 
                bonus_word, 
                path=w_path,
                board_format=board_format, 
                bonus_cell=bonus_cell, 
                is_private=True
            )))
        
        # Sort by points descending
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_words = []
        
        if r >= 2400:
            # Elite bots: Pick from the absolute top words (best 10% or top 30)
            pool_size = max(30, int(len(word_scores) * 0.15))
            pool = word_scores[:pool_size]
            # Take mostly from the top, some random from the pool
            selected_words = random.sample(pool, min(count, len(pool)))
        elif r >= 1800:
            # Advanced bots: Pick from top 40%
            pool_size = max(50, int(len(word_scores) * 0.4))
            pool = word_scores[:pool_size]
            selected_words = random.sample(pool, min(count, len(pool)))
        elif r >= 1200:
            # Average bots: Pick from entire list but skewed toward the top half
            pool_size = max(20, int(len(word_scores) * 0.7))
            pool = word_scores[:pool_size]
            selected_words = random.sample(pool, min(count, len(pool)))
        else:
            # Low rating: Pick from bottom 70% (skipping the absolute best words)
            start_idx = max(5, int(len(word_scores) * 0.3))
            pool = word_scores[start_idx:]
            if not pool: pool = word_scores # Fallback
            selected_words = random.sample(pool, min(count, len(pool)))
        
        # Bonus word chance (scales with rating)
        bonus_chance = max(0, min(0.95, (r - 600) / 1800))
        if bonus_word and random.random() < bonus_chance:
            # Ensure index 0 doesn't just get it if we can find it
            if not any(w[0] == bonus_word for w in selected_words):
                # Replace a word or just add it
                bonus_pts = calculate_word_score(
                    bonus_word, 
                    bonus_word, 
                    board_format=board_format, 
                    bonus_cell=bonus_cell, 
                    is_private=True
                )
                bonus_tuple = (bonus_word, bonus_pts)
                if len(selected_words) > 0:
                    idx = random.randint(0, len(selected_words) - 1)
                    selected_words[idx] = bonus_tuple
                else:
                    selected_words.append(bonus_tuple)
        
        # Construct word objects with randomized points and timestamps
        submission = []
        total_score = 0
        
        # Spread words across the entire round
        # bots start slightly late and finish before round end
        start_offset = 2.0
        effective_duration = max(5.0, duration - 6.0)
        
        # Randomize the order of submission times
        times = [start_offset + (random.random() * effective_duration) for _ in range(len(selected_words))]
        times.sort()
        
        for i, (w, pts) in enumerate(selected_words):
            # Recalculate with details for the submission record
            res = calculate_word_score(
                w, 
                bonus_word, 
                board_format=board_format, 
                bonus_cell=bonus_cell, 
                is_private=True,
                return_details=True
            )
            submission.append({
                'word': w,
                'points': res['total'],
                'is_bonus': (bonus_word and w == bonus_word.upper()),
                'score_details': res,
                'time_offset': times[i]
            })
            total_score += res['total']
            
        total_score = max(0, total_score)
        return submission, total_score

    def _apply_match_ratings(self, match_id, round_number, conn):
        """Calculates and persists rating changes for a completed private match round."""
        m = conn.execute('SELECT creator_id, match_type, parameters FROM private_matches WHERE id = ?', (match_id,)).fetchone()
        if not m: return
        
        params = json.loads(m['parameters'])
        # Config key for user_ratings table
        board_dims = params.get('board_dimensions', '4x4')
        time_limit = params.get('time_limit', 60)
        match_type_raw = m['match_type']
        
        # We use a similar config key to GameRoom for cross-compatibility
        # Format: game_type|dims|time
        config_key = f"{match_type_raw}|{board_dims}|{time_limit}"
        
        is_24h = (int(time_limit) >= 7200)
        if is_24h:
            print(f"[PrivateMatch] 24-hour match: skipping rating updates.")
            return
            
        # Participants and turns
        players_rows = conn.execute('SELECT user_id, username, is_ai, ai_rating FROM private_match_players WHERE match_id = ?', (match_id,)).fetchall()
        turns_rows = conn.execute('SELECT user_id, score, submitted_words FROM private_match_turns WHERE match_id = ? AND round_number = ?', (match_id, round_number)).fetchall()
        
        turns_map = {t['user_id']: {'score': t['score'], 'words': json.loads(t['submitted_words'] or '[]')} for t in turns_rows}
        
        # Build Player objects for the rating logic
        class MockPlayer:
            def __init__(self, user_id, username, rating, score, words, is_ai):
                self.user_id = user_id
                self.username = username
                self.rating = rating
                self.score = score
                self.submitted_words = words
                self.invalid_words = [] # Private turns only store successful submissions usually
                self.is_ai = is_ai
                self.is_guest = False # Private match participants are never guests
                self.joined_mid_round = False
        
        players = []
        for pr in players_rows:
            uid = pr['user_id']
            turn_data = turns_map.get(uid, {'score': 0, 'words': []})
            score = turn_data['score']
            words = turn_data['words']
            
            rating = 1200
            if pr['is_ai']:
                rating = pr['ai_rating']
            else:
                # First check config-specific, then default to 1200
                r_row = conn.execute('SELECT rating FROM user_ratings WHERE user_id = ? AND config_key = ?', (uid, config_key)).fetchone()
                if r_row:
                    rating = r_row[0]
                else:
                    rating = 1200
            
            players.append(MockPlayer(uid, pr['username'], rating, score, words, bool(pr['is_ai'])))
            
        # Calculate changes (Private matches always follow the DNP score >= 1 rule)
        changes = calculate_proportional_rating_change(players, is_private=True)
        
        # Persist changes
        for p in players:
            change = changes.get(p.user_id, 0)
            if change == 0: continue
            
            new_rating = p.rating + change
            
            if not p.is_ai:
                # 1. Update Config-Specific Rating
                conn.execute('''
                    INSERT INTO user_ratings (user_id, config_key, rating) VALUES (?, ?, ?)
                    ON CONFLICT(user_id, config_key) DO UPDATE SET rating = rating + ?
                ''', (p.user_id, config_key, new_rating, change))
                
                # 2. Update Global Rating (If competitive)
                # Competitive if more than one human OR one human + bot?
                # The user requested Bots to count as competition.
                conn.execute('UPDATE users SET rating = ?, games_played = games_played + 1 WHERE id = ?', (new_rating, p.user_id))
                
                # Update Wins
                max_score = max(p.score for p in players)
                if p.score == max_score and p.score > 0:
                    conn.execute('UPDATE users SET wins = wins + 1 WHERE id = ?', (p.user_id,))
            else:
                # Update AI bot rating in this match (so rematch is harder/easier)
                conn.execute('UPDATE private_match_players SET ai_rating = ? WHERE match_id = ? AND user_id = ?', (new_rating, match_id, p.user_id))

    def get_invites_for_user(self, username):
        conn = self.get_db()
        conn.row_factory = sqlite3.Row
        invites = conn.execute('''
            SELECT i.*, u.username as sender_name, m.parameters
            FROM match_invites i
            JOIN users u ON i.sender_id = u.id
            JOIN private_matches m ON i.match_id = m.id
            WHERE i.recipient_username = ? AND i.status = 'pending'
        ''', (username,)).fetchall()
        conn.close()
        
        results = []
        for inv in invites:
            d = dict(inv)
            d['parameters'] = json.loads(inv['parameters'])
            results.append(d)
        return results

private_match_manager = PrivateMatchManager()
