import sqlite3
import json
import time
import random
from typing import List, Dict

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
                current_round INTEGER DEFAULT 1,
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
        ''')
        conn.commit()
        conn.close()

    def get_db(self):
        return sqlite3.connect(self.db_path)

    def create_match(self, creator_id, match_type, parameters, participants=None):
        """
        participants: list of {'user_id': id, 'username': name, 'is_ai': bool, 'ai_rating': optional}
        """
        conn = self.get_db()
        now = time.time()
        
        # 1. Create Match Entry
        cur = conn.execute('''
            INSERT INTO private_matches (creator_id, match_type, parameters, created_at, last_activity)
            VALUES (?, ?, ?, ?, ?)
        ''', (creator_id, match_type, json.dumps(parameters), now, now))
        match_id = cur.lastrowid
        
        # 2. Add Participants
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
        
        # 3. Generate first round board
        self.generate_round(match_id, 1, parameters)
        
        conn.close()
        return match_id

    def generate_round(self, match_id, round_number, parameters):
        from board_generator import BoardGenerator
        from word_validator import word_validator
        
        bg = BoardGenerator()
        dims = parameters.get('board_dimensions', '4x4')
        try:
            w, h = map(int, dims.lower().split('x'))
        except:
            w, h = 4, 4
            
        board = bg.generate_board(w, h)
        
        dict_name = parameters.get('dictionary', 'CSW')
        min_len = parameters.get('min_word_length', 3)
        
        all_words_on_board = bg.find_all_words(board, dictionary_name=dict_name, min_length=min_len)
        
        # Bonus word
        bonus_len = parameters.get('bonus_word_length', 0)
        bonus_word = ""
        if bonus_len > 0:
            potential_bonuses = [w for w in all_words_on_board if len(w) == bonus_len]
            if potential_bonuses:
                bonus_word = random.choice(potential_bonuses)

        conn = self.get_db()
        now = time.time()
        # 1 week expiry as requested
        end_time = now + (7 * 24 * 3600) 
        
        conn.execute('''
            INSERT INTO private_match_rounds (match_id, round_number, board_data, bonus_word, all_words, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (match_id, round_number, json.dumps(board), bonus_word, json.dumps(all_words_on_board), now, end_time))
        conn.commit()
        conn.close()

    def get_matches_for_user(self, user_id, username):
        """
        Returns { 'your_turn': [], 'their_turn': [], 'history': [] }
        """
        conn = self.get_db()
        conn.row_factory = sqlite3.Row
        
        # Matches where user is a participant
        # and has not yet submitted for current_round -> Your Turn
        # and has submitted but others haven't -> Their Turn
        # and all have submitted -> Next Round Transition (or History if finished)
        
        all_p_matches = conn.execute('''
            SELECT m.*, mp.status as my_status
            FROM private_matches m
            JOIN private_match_players mp ON m.id = mp.match_id
            WHERE mp.user_id = ? AND m.status != 'expired'
        ''', (user_id,)).fetchall()
        
        results = {'your_turn': [], 'their_turn': [], 'history': []}
        
        now = time.time()
        
        for m in all_p_matches:
            match_id = m['id']
            curr_round = m['current_round']
            
            # Check if user has submitted for this round
            turn = conn.execute('''
                SELECT 1 FROM private_match_turns 
                WHERE match_id = ? AND round_number = ? AND user_id = ?
            ''', (match_id, curr_round, user_id)).fetchone()
            
            # Check round timing
            round_info = conn.execute('''
                SELECT end_time FROM private_match_rounds 
                WHERE match_id = ? AND round_number = ?
            ''', (match_id, curr_round)).fetchone()
            
            if round_info and round_info['end_time'] < now:
                # Expired match
                conn.execute("UPDATE private_matches SET status = 'expired' WHERE id = ?", (match_id,))
                continue

            # Participants
            players = conn.execute('''
                SELECT user_id, username, is_ai FROM private_match_players 
                WHERE match_id = ?
            ''', (match_id,)).fetchall()
            players_list = [dict(p) for p in players]
            
            # Submissions for this round
            submissions = conn.execute('''
                SELECT user_id FROM private_match_turns 
                WHERE match_id = ? AND round_number = ?
            ''', (match_id, curr_round)).fetchall()
            submitted_ids = [s['user_id'] for s in submissions]
            
            match_data = dict(m)
            match_data['parameters'] = json.loads(m['parameters'])
            match_data['players'] = players_list
            match_data['round_info'] = dict(round_info) if round_info else {}
            
            if not turn:
                results['your_turn'].append(match_data)
            else:
                # Check if anyone else (non-AI) still has to go
                others_pending = False
                for p in players_list:
                    if not p['is_ai'] and p['user_id'] != user_id and p['user_id'] not in submitted_ids:
                        others_pending = True
                        break
                
                if others_pending:
                    results['their_turn'].append(match_data)
                else:
                    # All have submitted for current round (AIs submit instantly on first human submission usually, or we can lazy trigger)
                    # For now, if all humans are done, it's essentially ready for next round or in history
                    results['history'].append(match_data)
                    
        conn.close()
        return results

    def submit_turn(self, match_id, round_number, user_id, words_data, score):
        conn = self.get_db()
        now = time.time()
        
        # Record Turn
        conn.execute('''
            INSERT INTO private_match_turns (match_id, round_number, user_id, score, submitted_words, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (match_id, round_number, user_id, score, json.dumps(words_data), now))
        
        # If all humans have submitted, generate AI turns
        players = conn.execute('SELECT user_id, is_ai, ai_rating FROM private_match_players WHERE match_id = ?', (match_id,)).fetchall()
        submissions = conn.execute('SELECT user_id FROM private_match_turns WHERE match_id = ? AND round_number = ?', (match_id, round_number)).fetchall()
        submitted_ids = [s['user_id'] for s in submissions]
        
        humans = [p for p in players if not p['is_ai']]
        ais = [p for p in players if p['is_ai']]
        
        all_humans_done = all(h['user_id'] in submitted_ids for h in humans)
        
        if all_humans_done:
            # Generate AI turns
            round_data = conn.execute('SELECT * FROM private_match_rounds WHERE match_id = ? AND round_number = ?', (match_id, round_number)).fetchone()
            all_possible_words = json.loads(round_data['all_words'])
            bonus_word = round_data['bonus_word']
            
            for ai in ais:
                if ai['user_id'] not in submitted_ids:
                    ai_words, ai_score = self.generate_ai_submission(ai['ai_rating'], all_possible_words, bonus_word)
                    conn.execute('''
                        INSERT INTO private_match_turns (match_id, round_number, user_id, score, submitted_words, submitted_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (match_id, round_number, ai['user_id'], ai_score, json.dumps(ai_words), now))

            # Advance Round? 
            # User wants "History" to show results. If we advance current_round, the previous one becomes History.
            # Usually we advance ONLY when someone clicks "Rematch" or starts next. 
            # Actually, "With Friends" usually has matches that progress.
            # Let's advance automatically if all are done.
            # conn.execute('UPDATE private_matches SET current_round = current_round + 1, last_activity = ? WHERE id = ?', (now, match_id))
            # new_round = round_number + 1
            # self.generate_round(match_id, new_round, json.loads(conn.execute('SELECT parameters FROM private_matches WHERE id=?', (match_id,)).fetchone()[0]))
            
        conn.commit()
        conn.close()

    def generate_ai_submission(self, rating, possible_words, bonus_word):
        # AI Logic:
        # Rating 800: finds 10-20% of words, mostly short.
        # Rating 1200: finds 20-40% of words, mixed.
        # Rating 2000+: finds 50-80% of words, includes bonus often.
        
        # Clamp rating for logic
        r = max(400, min(3000, rating))
        
        # Percentage of words to find
        # 400 -> 5%, 3000 -> 90%
        # Linear interp: percentage = 5 + (r-400) * (85/2600)
        percentage = 5 + (r - 400) * (85 / 2600)
        percentage = percentage / 100.0
        
        count = int(len(possible_words) * percentage)
        # Avoid 0 if board has words
        if len(possible_words) > 0 and count == 0: count = 1
        
        # Higher rating bot picks longer words more often
        # Weighted selection: weight = len(word)^2 * factor(rating)
        # For simplicity, let's just shuffle and pick top N after sorting by length?
        # No, better to randomly sample with bias.
        
        # Sort words so we can bias towards shorter/longer
        sorted_words = sorted(possible_words, key=len)
        
        selected = []
        if r < 1000:
            # Bias toward short
            # Pick from first 40% of sorted list mostly
            pool = sorted_words[:int(len(sorted_words)*0.5)]
            if pool:
                selected = random.sample(pool, min(count, len(pool)))
        elif r < 1800:
            # Mixed
            selected = random.sample(possible_words, min(count, len(possible_words)))
        else:
            # Bias toward long
            pool = sorted_words[int(len(sorted_words)*0.3):]
            if pool:
                selected = random.sample(pool, min(count, len(pool)))
        
        # Bonus word chance
        # 400 -> 0%, 3000 -> 100%
        bonus_chance = (r - 400) / 2600
        if bonus_word and random.random() < bonus_chance:
            if bonus_word not in selected:
                selected.append(bonus_word)
        
        # Construct word objects
        submission = []
        total_score = 0
        now = time.time()
        # Randomize typing times within a 60s window (or time_limit)
        # Bot starts submission after 5s
        
        for w in selected:
            # Basic scoring
            pts = len(w)
            if len(w) == 6: pts = 10
            elif len(w) == 7: pts = 15
            elif len(w) >= 8: pts = 25
            if w == bonus_word: pts += 10
            
            total_score += pts
            submission.append({
                'word': w,
                'points': pts,
                'timestamp': now - random.randint(5, 55)
            })
            
        return submission, total_score

private_match_manager = PrivateMatchManager()
