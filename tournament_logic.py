
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
        conn = sqlite3.connect(self.db_path)
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
        # Initialize Round 1
        conn.execute('UPDATE tournaments SET status = ?, current_round = 1 WHERE id = ?', ('active', tid))
        
        # Filter participants
        participants = conn.execute('SELECT user_id FROM tournament_participants WHERE tournament_id = ?', (tid,)).fetchall()
        if not participants:
            # Handle empty tournament? Just complete it.
            conn.execute("UPDATE tournaments SET status = 'completed', completed_at = ? WHERE id = ?", (time.time(), tid))
            conn.commit()
            conn.close()
            return

        self.start_new_round(tid, 1)
        conn.commit()
        conn.close()

    def start_new_round(self, tid, round_number):
        conn = self.get_db()
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
        board = bg.generate_board(rows, cols)
        
        now = time.time()
        end_time = now + self.turn_duration
        
        conn.execute('''
            INSERT INTO tournament_rounds (tournament_id, round_number, start_time, end_time, board_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (tid, round_number, now, end_time, json.dumps(board)))
        
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
        
        # 1. Get all active participants
        active_users = conn.execute('''
            SELECT user_id FROM tournament_participants 
            WHERE tournament_id = ? AND status = 'active'
        ''', (tid,)).fetchall()
        
        # 2. Get scores for this round
        scores = conn.execute('''
            SELECT user_id, score FROM tournament_scores
            WHERE tournament_id = ? AND round_number = ?
        ''', (tid, round_num)).fetchall()
        
        score_dict = {row['user_id']: row['score'] for row in scores}
        
        # 3. Advancement Logic: Fair comparison
        # We sort all active users by their score in this round.
        # Top 50% advance. If score is 0 and they didn't play, they are eliminated.
        
        results = []
        for u in active_users:
            score = score_dict.get(u['user_id'], 0)
            results.append({'user_id': u['user_id'], 'score': score})
            
        results.sort(key=lambda x: x['score'], reverse=True)
        
        num_participants = len(results)
        num_to_advance = max(1, num_participants // 2)
        
        # If only 1 or 2 people left, the tournament might end.
        if num_participants <= 1:
            self.complete_tournament(tid, results)
            conn.close()
            return

        winners = results[:num_to_advance]
        losers = results[num_to_advance:]
        
        # Eliminate losers
        for l in losers:
            conn.execute('''
                UPDATE tournament_participants 
                SET status = 'eliminated', final_rank = ?
                WHERE tournament_id = ? AND user_id = ?
            ''', (num_participants, tid, l['user_id']))
            
        # Check if we have a definitive winner
        if len(winners) == 1 and num_participants > 1:
            # We have a winner!
            self.complete_tournament(tid, winners)
        else:
            # Next round
            next_round = round_num + 1
            conn.execute('UPDATE tournaments SET current_round = ? WHERE id = ?', (next_round, tid))
            self.start_new_round(tid, next_round)
            
        conn.commit()
        conn.close()

    def complete_tournament(self, tid, final_results):
        conn = self.get_db()
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
        conn.commit()
        conn.close()

    def get_history(self):
        conn = self.get_db()
        # Get winners of past tournaments
        # Final rank 1 in completed tournaments
        rows = conn.execute('''
            SELECT t.id, t.completed_at, u.username, tp.final_rank
            FROM tournaments t
            JOIN tournament_participants tp ON t.id = tp.tournament_id
            JOIN users u ON tp.user_id = u.id
            WHERE t.status = 'completed' AND tp.final_rank = 1
            ORDER BY t.completed_at DESC
        ''').fetchall()
        conn.close()
        return [dict(row) for row in rows]

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

tournament_manager = TournamentManager()
