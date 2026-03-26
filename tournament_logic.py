
import sqlite3
import time
import json
import random
from spinner_set import SpinnerSet
from word_validator import word_validator

class TournamentManager:
    def __init__(self, db_path='morpheme.db'):
        self.db_path = db_path
        self.signup_duration = 7 * 24 * 60 * 60  # 1 week
        self.turn_duration = 2 * 24 * 60 * 60    # 2 days

    def get_db(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def get_current_tournament(self):
        conn = self.get_db()
        # Get the most recent tournament that is not fully completed or the latest completed one for history
        row = conn.execute('SELECT * FROM tournaments ORDER BY id DESC LIMIT 1').fetchone()
        conn.close()
        
        if not row:
            # Initialize the first tournament if none exist
            return self.create_new_tournament()
            
        return dict(row)

    def create_new_tournament(self):
        conn = self.get_db()
        params = SpinnerSet.generate_tournament_params()
        
        # Ensure 'difficulty' and word count range are included in the params as requested
        # SpinnerSet.generate_tournament_params() already includes them based on my check of spinner_set.py
        
        now = time.time()
        start_date = now + self.signup_duration
        
        cursor = conn.execute('''
            INSERT INTO tournaments (status, start_date, parameters, current_round, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', ('signup', start_date, json.dumps(params), 0, now))
        
        tournament_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return self.get_tournament_by_id(tournament_id)

    def get_tournament_by_id(self, tid):
        conn = self.get_db()
        row = conn.execute('SELECT * FROM tournaments WHERE id = ?', (tid,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def update_tournament_status(self):
        """Main lifecycle update loop"""
        current = self.get_current_tournament()
        if not current: return
        
        now = time.time()
        
        if current['status'] == 'signup':
            if now >= current['start_date']:
                # Start the tournament!
                self.start_tournament(current['id'])
                
        elif current['status'] == 'active':
            # Check if the current round has ended
            self.check_round_advancement(current['id'])
            
        # If completed, the frontend logic will handle the week-long signup window for the NEXT one
        # but we need to ensure a new tournament is created if the current is COMPLETED and the cooldown is over.
        elif current['status'] == 'completed':
            # After a tournament is completed, we wait 1 week before starting a NEW signup window?
            # User: "When the tournament is over, keep the opening panel with the Spinner Set and Sign-in button open for about a week again"
            # This means as soon as one ends, the NEXT signup period begins.
            if now >= current['completed_at']:
                 self.create_new_tournament()

    def start_tournament(self, tid):
        conn = self.get_db()
        try:
            # 1. Filter participants
            participants = conn.execute('SELECT user_id FROM tournament_participants WHERE tournament_id = ?', (tid,)).fetchall()
            if not participants:
                # Handle empty tournament? Just complete it.
                conn.execute("UPDATE tournaments SET status = 'completed', completed_at = ? WHERE id = ?", (time.time(), tid))
                conn.commit()
                return

            # 2. Generate Round 1 Board FIRST
            # This ensures when we set current_round=1, the data exists
            self.start_new_round(tid, 1, conn=conn)

            # 3. NOW activate the tournament and set round pointer
            conn.execute('UPDATE tournaments SET status = ?, current_round = 1 WHERE id = ?', ('active', tid))
            
            conn.commit()
        finally:
            conn.close()

    def start_new_round(self, tid, round_number, conn=None):
        should_close = False
        if conn is None:
            conn = self.get_db()
            should_close = True
            
        tournament = self.get_tournament_by_id(tid)
        params = json.loads(tournament['parameters'])
        
        # Generate a board for this round
        dims = params.get('board_dimensions', '4x4')
        try:
            d_parts = dims.split('x')
            rows, cols = int(d_parts[0]), int(d_parts[1])
        except:
            rows, cols = 4, 4
            
        from board_generator import BoardGenerator
        bg = BoardGenerator()
        
        # USE UNIQUE SEEDING TO PREVENT BOARD REUSE ACROSS PROCESSES
        import random
        random.seed() 

        # Use tournament parameters
        board, all_words_on_board, _bonus_cell = bg.generate_board(
            dimensions=dims,
            bonus_word="", # Picked later if needed, or non-active in tournament
            word_count_range=params.get('word_count_range', '50-100'),
            dictionary=params.get('dictionary', 'CSW'),
            board_format=params.get('board_format', 'Normal'),
            min_word_length=params.get('min_word_length', 3),
            difficulty=params.get('difficulty', 'Normal')
        )
        
        now = time.time()
        end_time = now + self.turn_duration
        
        conn.execute('''
            INSERT INTO tournament_rounds (tournament_id, round_number, start_time, end_time, board_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (tid, round_number, now, end_time, json.dumps(board)))
        
        # 4. Generate Matchups for this round
        self.create_matchups(tid, round_number, conn)

        if should_close:
            conn.commit()
            conn.close()

    def create_matchups(self, tid, round_number, conn):
        # Get all active participants
        participants = conn.execute('''
            SELECT user_id FROM tournament_participants 
            WHERE tournament_id = ? AND status = 'active'
        ''', (tid,)).fetchall()
        
        user_ids = [p['user_id'] for p in participants]
        random.shuffle(user_ids)
        
        now = time.time()
        
        matchups = []
        for i in range(0, len(user_ids), 2):
            u1 = user_ids[i]
            u2 = user_ids[i+1] if i+1 < len(user_ids) else -1 # -1 denotes a bye
            matchups.append((tid, round_number, u1, u2, now))
            
        conn.executemany('''
            INSERT INTO tournament_matchups (tournament_id, round_number, user1_id, user2_id, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', matchups)

    def check_round_advancement(self, tid):
        conn = self.get_db()
        tournament = self.get_tournament_by_id(tid)
        round_num = tournament['current_round']
        
        round_info = conn.execute('''
            SELECT * FROM tournament_rounds 
            WHERE tournament_id = ? AND round_number = ?
        ''', (tid, round_num)).fetchone()
        
        if not round_info:
            conn.close()
            return
            
        now = time.time()
        if now >= round_info['end_time']:
            # Round over! Advance winners.
            self.advance_tournament(tid, round_num)
            
        conn.close()

    def advance_tournament(self, tid, round_num):
        conn = self.get_db()
        try:
            # 1. Get all matchups for this round
            matchups = conn.execute('''
                SELECT * FROM tournament_matchups 
                WHERE tournament_id = ? AND round_number = ?
            ''', (tid, round_num)).fetchall()
            
            # 2. Get scores for this round
            scores = conn.execute('''
                SELECT user_id, score FROM tournament_scores
                WHERE tournament_id = ? AND round_number = ?
            ''', (tid, round_num)).fetchall()
            
            score_dict = {row['user_id']: row['score'] for row in scores}
            
            # 3. Process Matchups
            winners = []
            eliminated = []
            
            # Count total participants (active + previously eliminated) for accurate ranking
            total_participants = conn.execute('SELECT COUNT(*) FROM tournament_participants WHERE tournament_id = ?', (tid,)).fetchone()[0]

            for m in matchups:
                u1 = m['user1_id']
                u2 = m['user2_id']
                
                if u2 == -1:
                    # Bye! u1 advances automatically
                    winners.append(u1)
                    continue
                    
                s1 = score_dict.get(u1, 0)
                s2 = score_dict.get(u2, 0)
                
                # Determine winner
                if s1 > s2:
                    winners.append(u1)
                    eliminated.append(u2)
                    conn.execute('UPDATE tournament_matchups SET winner_id = ? WHERE id = ?', (u1, m['id']))
                elif s2 > s1:
                    winners.append(u2)
                    eliminated.append(u1)
                    conn.execute('UPDATE tournament_matchups SET winner_id = ? WHERE id = ?', (u2, m['id']))
                else:
                    # TIE (or both 0)! Random winner
                    # Check if they both left (forfeited)
                    w = random.choice([u1, u2])
                    winners.append(w)
                    eliminated.append(u2 if w == u1 else u1)
                    conn.execute('UPDATE tournament_matchups SET winner_id = ? WHERE id = ?', (w, m['id']))

            # Perform eliminations
            for uid in eliminated:
                conn.execute('''
                    UPDATE tournament_participants 
                    SET status = 'eliminated', final_rank = ?
                    WHERE tournament_id = ? AND user_id = ?
                ''', (total_participants, tid, uid))
                
            # Check if tournament is over
            if len(winners) <= 1:
                final_results = []
                for w in winners:
                    final_results.append({'user_id': w, 'score': score_dict.get(w, 0)})
                self.complete_tournament(tid, final_results, conn=conn)
            else:
                # Next round
                next_round = round_num + 1
                self.start_new_round(tid, next_round, conn=conn)
                conn.execute('UPDATE tournaments SET current_round = ? WHERE id = ?', (next_round, tid))
                
            conn.commit()
        finally:
            conn.close()

    def get_user_matchup(self, tid, round_number, user_id):
        conn = self.get_db()
        row = conn.execute('''
            SELECT m.*, 
                   u1.username as u1_name, u2.username as u2_name,
                   (SELECT score FROM tournament_scores WHERE tournament_id = m.tournament_id AND round_number = m.round_number AND user_id = m.user1_id) as u1_score,
                   (SELECT score FROM tournament_scores WHERE tournament_id = m.tournament_id AND round_number = m.round_number AND user_id = m.user2_id) as u2_score
            FROM tournament_matchups m
            LEFT JOIN users u1 ON m.user1_id = u1.id
            LEFT JOIN users u2 ON m.user2_id = u2.id
            WHERE m.tournament_id = ? AND m.round_number = ? AND (m.user1_id = ? OR m.user2_id = ?)
        ''', (tid, round_number, user_id, user_id)).fetchone()
        conn.close()
        
        if not row: return None
        
        res = dict(row)
        # Determine who the opponent is
        if res['user1_id'] == user_id:
            res['opponent_id'] = res['user2_id']
            res['opponent_name'] = res['u2_name'] if res['user2_id'] != -1 else "BYE"
            res['opponent_score'] = res['u2_score'] if res['user2_id'] != -1 else 0
            res['my_score'] = res['u1_score']
        else:
            res['opponent_id'] = res['user1_id']
            res['opponent_name'] = res['u1_name']
            res['opponent_score'] = res['u1_score']
            res['my_score'] = res['u2_score']
            
        return res

    def forfeit_turn(self, tid, round_number, user_id):
        """Mark user turn as done with 0 score (forfeit)"""
        # We don't use has_user_turn here because we want to allow forfeiting even if they already played?
        # Actually no, if they've played, they've played.
        # But if they haven't, we insert a 0 score.
        if not self.has_user_turn(tid, user_id):
            return False
            
        conn = self.get_db()
        try:
            conn.execute('''
                INSERT INTO tournament_scores (tournament_id, round_number, user_id, score, submitted_words, submitted_at, round_start_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (tid, round_number, user_id, 0, json.dumps([]), time.time(), time.time()))
            conn.commit()
            return True
        except Exception as e:
            print(f"Forfeit Error: {e}")
            return False
        finally:
            conn.close()

    def complete_tournament(self, tid, final_results, conn=None):
        should_close = False
        if conn is None:
            conn = self.get_db()
            should_close = True
            
        now = time.time()
        
        # Mark final ranks for survivors
        for idx, res in enumerate(final_results):
            conn.execute('''
                UPDATE tournament_participants 
                SET status = 'completed', final_rank = ?
                WHERE tournament_id = ? AND user_id = ?
            ''', (idx + 1, tid, res['user_id']))
            
        conn.execute('UPDATE tournaments SET status = ?, completed_at = ? WHERE id = ?', 
                    ('completed', now, tid))
        
        if should_close:
            conn.commit()
            conn.close()

    def get_all_matchups(self, tid, round_number):
        conn = self.get_db()
        rows = conn.execute('''
            SELECT m.*, 
                   u1.username as u1_name, u2.username as u2_name,
                   (SELECT score FROM tournament_scores WHERE tournament_id = m.tournament_id AND round_number = m.round_number AND user_id = m.user1_id) as u1_score,
                   (SELECT score FROM tournament_scores WHERE tournament_id = m.tournament_id AND round_number = m.round_number AND user_id = m.user2_id) as u2_score
            FROM tournament_matchups m
            LEFT JOIN users u1 ON m.user1_id = u1.id
            LEFT JOIN users u2 ON m.user2_id = u2.id
            WHERE m.tournament_id = ? AND m.round_number = ?
            ORDER BY m.id ASC
        ''', (tid, round_number)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_history(self):
        conn = self.get_db()
        # Get winners of past tournaments (Final rank 1)
        # We also need their total score or final round score to display
        rows = conn.execute('''
            SELECT t.id, t.completed_at, t.current_round, u.username, tp.final_rank,
                   (SELECT score FROM tournament_scores ts 
                    WHERE ts.tournament_id = t.id AND ts.user_id = u.id AND ts.round_number = t.current_round) as winning_score
            FROM tournaments t
            JOIN tournament_participants tp ON t.id = tp.tournament_id
            JOIN users u ON tp.user_id = u.id
            WHERE t.status = 'completed' AND tp.final_rank = 1
            ORDER BY t.completed_at DESC
        ''').fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_winner_turn_data(self, tid, username):
        """Fetches the winner's finalized turn data for replay"""
        conn = self.get_db()
        conn.row_factory = sqlite3.Row
        
        # Get tournament info
        t = conn.execute('SELECT current_round, parameters FROM tournaments WHERE id = ?', (tid,)).fetchone()
        if not t:
            conn.close()
            return None
        
        # Get user turn for the final round
        row = conn.execute('''
            SELECT ts.user_id, u.username, ts.score, ts.submitted_words, ts.submitted_at, ts.round_start_time,
                   r.board_data
            FROM tournament_scores ts
            JOIN users u ON ts.user_id = u.id
            JOIN tournament_rounds r ON ts.tournament_id = r.tournament_id AND ts.round_number = r.round_number
            WHERE ts.tournament_id = ? AND u.username = ? AND ts.round_number = ?
        ''', (tid, username, t['current_round'])).fetchone()
        
        conn.close()
        if not row: return None
        
        data = dict(row)
        data['parameters'] = json.loads(t['parameters'])
        if data.get('submitted_words'):
            data['submitted_words'] = json.loads(data['submitted_words'])
        if data.get('board_data'):
            data['board_data'] = json.loads(data['board_data'])
            
        return data

    def has_user_turn(self, tid, user_id):
        """Checks if it's currently the user's turn and they haven't played yet"""
        conn = self.get_db()
        tournament = self.get_tournament_by_id(tid)
        if not tournament or tournament['status'] != 'active':
            conn.close()
            return False
            
        round_num = tournament['current_round']
        
        # Check if user is active participant
        participant = conn.execute('''
            SELECT status FROM tournament_participants 
            WHERE tournament_id = ? AND user_id = ?
        ''', (tid, user_id)).fetchone()
        
        if not participant or participant['status'] != 'active':
            conn.close()
            return False
            
        # Check if already submitted score for this round
        score = conn.execute('''
            SELECT 1 FROM tournament_scores 
            WHERE tournament_id = ? AND round_number = ? AND user_id = ?
        ''', (tid, round_num, user_id)).fetchone()
        
        conn.close()
        return score is None

    def get_round_scores(self, tid, round_num):
        """Returns leaderboard for a specific round"""
        conn = self.get_db()
        conn.row_factory = sqlite3.Row
        rows = conn.execute('''
            SELECT ts.user_id, u.username, ts.score, ts.submitted_words, ts.submitted_at, ts.round_start_time,
                   (SELECT board_data FROM tournament_rounds WHERE tournament_id = ? AND round_number = ?) as board_data
            FROM tournament_scores ts
            JOIN users u ON ts.user_id = u.id
            WHERE ts.tournament_id = ? AND ts.round_number = ?
            ORDER BY ts.score DESC
        ''', (tid, round_num, tid, round_num)).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_tournament_standings(self, tid):
        """Returns list of all participants and their status"""
        conn = self.get_db()
        conn.row_factory = sqlite3.Row
        
        rows = conn.execute('''
            SELECT tp.user_id, u.username, tp.status, tp.final_rank
            FROM tournament_participants tp
            JOIN users u ON tp.user_id = u.id
            WHERE tp.tournament_id = ?
            ORDER BY CASE WHEN tp.status = 'active' THEN 0 ELSE 1 END, tp.final_rank ASC, u.username ASC
        ''', (tid,)).fetchall()
        
        conn.close()
        return [dict(row) for row in rows]

tournament_manager = TournamentManager()
