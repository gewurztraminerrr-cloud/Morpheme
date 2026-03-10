
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

            # 2. Shuffle and Create Initial Matchups
            user_ids = [p['user_id'] for p in participants]
            random.shuffle(user_ids)
            
            self.create_matchups(tid, 1, user_ids, conn)

            # 3. Generate Round 1 Board
            self.start_new_round(tid, 1, conn=conn)

            # 4. Activate the tournament
            conn.execute('UPDATE tournaments SET status = ?, current_round = 1 WHERE id = ?', ('active', tid))
            
            conn.commit()
        finally:
            conn.close()

    def create_matchups(self, tid, round_num, user_ids, conn):
        """Pairs users for a specific round. Handles byes if odd number."""
        # Pair them up
        for i in range(0, len(user_ids), 2):
            u1 = user_ids[i]
            u2 = user_ids[i+1] if i+1 < len(user_ids) else None
            match_idx = i // 2
            
            conn.execute('''
                INSERT INTO tournament_matchups (tournament_id, round_number, user_1_id, user_2_id, match_index)
                VALUES (?, ?, ?, ?, ?)
            ''', (tid, round_num, u1, u2, match_idx))
            
            # If it's a bye, u1 automatically wins? 
            # Or we wait for the round to end? 
            # Better to wait or handle it in advancement.

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
        board, all_words_on_board = bg.generate_board(
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
        
        if should_close:
            conn.commit()
            conn.close()

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
            
            winners = []
            losers = []
            
            # 3. Process each matchup
            for m in matchups:
                u1 = m['user_1_id']
                u2 = m['user_2_id']
                
                s1 = score_dict.get(u1, -1) # -1 means didn't play (forfeit)
                s2 = score_dict.get(u2, -1) if u2 else -2 # -2 means bye
                
                winner = None
                if u2 is None:
                    # Bye
                    winner = u1
                else:
                    if s1 > s2:
                        winner = u1
                        losers.append(u2)
                    elif s2 > s1:
                        winner = u2
                        losers.append(u1)
                    else:
                        # Tie or both didn't play
                        if s1 == -1 and s2 == -1:
                            # Both forfeit! Select one at random
                            winner = random.choice([u1, u2])
                            losers.append(u2 if winner == u1 else u1)
                        else:
                            # Actual tie in score, pick random
                            winner = random.choice([u1, u2])
                            losers.append(u2 if winner == u1 else u1)
                            
                winners.append(winner)
                conn.execute('UPDATE tournament_matchups SET winner_id = ? WHERE tournament_id = ? AND round_number = ? AND match_index = ?',
                            (winner, tid, round_num, m['match_index']))
            
            # 4. Update participant statuses
            # Total participants (starting)
            total_ppl = conn.execute('SELECT COUNT(*) FROM tournament_participants WHERE tournament_id = ?', (tid,)).fetchone()[0]
            # Current participants (active)
            current_ppl = len(matchups) * 2 - (1 if any(m['user_2_id'] is None for m in matchups) else 0)
            
            for l_id in losers:
                # Rank should be roughly the level they reached
                # If 64 players, and you lose in R1, you are in 33-64 range.
                # Let's just use current_ppl as a basis.
                conn.execute('''
                    UPDATE tournament_participants 
                    SET status = 'eliminated', final_rank = ?
                    WHERE tournament_id = ? AND user_id = ?
                ''', (len(winners) + 1, tid, l_id))
                
            # 5. Check if tournament is over
            if len(winners) <= 1:
                self.complete_tournament(tid, [{'user_id': w} for w in winners], conn=conn)
            else:
                # Next round
                next_round = round_num + 1
                self.create_matchups(tid, next_round, winners, conn)
                self.start_new_round(tid, next_round, conn=conn)
                conn.execute('UPDATE tournaments SET current_round = ? WHERE id = ?', (next_round, tid))
                
            conn.commit()
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

    def get_user_opponent(self, tid, round_num, user_id):
        """Returns the username and id of the opponent for a user in a given round"""
        conn = self.get_db()
        row = conn.execute('''
            SELECT u.id, u.username FROM tournament_matchups m
            JOIN users u ON (u.id = m.user_1_id OR u.id = m.user_2_id)
            WHERE m.tournament_id = ? AND m.round_number = ?
              AND (m.user_1_id = ? OR m.user_2_id = ?)
              AND u.id != ?
        ''', (tid, round_num, user_id, user_id, user_id)).fetchone()
        conn.close()
        return dict(row) if row else None

    def forfeit_user(self, tid, round_num, user_id):
        """Records a 0 score for the user to mark their turn as completed/forfeited"""
        conn = self.get_db()
        try:
            # Check if already submitted
            existing = conn.execute('''
                SELECT 1 FROM tournament_scores 
                WHERE tournament_id = ? AND round_number = ? AND user_id = ?
            ''', (tid, round_num, user_id)).fetchone()
            
            if not existing:
                conn.execute('''
                    INSERT INTO tournament_scores (tournament_id, round_number, user_id, score, submitted_words, submitted_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (tid, round_num, user_id, 0, '[]', time.time()))
                conn.commit()
                return True
            return False
        finally:
            conn.close()

    def get_total_participants(self, tid):
        conn = self.get_db()
        count = conn.execute('SELECT COUNT(*) FROM tournament_participants WHERE tournament_id = ?', (tid,)).fetchone()[0]
        conn.close()
        return count

tournament_manager = TournamentManager()
