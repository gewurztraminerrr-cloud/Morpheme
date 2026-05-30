import os
import sys
import unittest
import sqlite3
import json
import time

# Add parent directory to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game_room import RoomManager, GameRoom, Player

class Test24hBoardPersistence(unittest.TestCase):
    def setUp(self):
        # We will use the live SQLite db but use a unique test room ID
        self.room_id = "test_persistence_pub_accumulative_4x4_86400"
        self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'morpheme.db'))
        
        # Clean any existing active board for our test room
        conn = sqlite3.connect(self.db_path)
        conn.execute('DELETE FROM active_boards WHERE room_id = ?', (self.room_id,))
        conn.commit()
        conn.close()
        
        self.room_manager = RoomManager()

    def tearDown(self):
        # Clean up test room database record
        conn = sqlite3.connect(self.db_path)
        conn.execute('DELETE FROM active_boards WHERE room_id = ?', (self.room_id,))
        conn.commit()
        conn.close()

    def test_24h_board_restoration(self):
        print("\n=== STARTING 24H BOARD PERSISTENCE VERIFICATION ===")
        
        # 1. Manually insert a mock active board into the active_boards table
        mock_board = [['T','E','S','T'],['P','L','A','Y'],['W','O','R','D'],['G','A','M','E']]
        mock_words = ['TEST', 'PLAY', 'WORD', 'GAME']
        mock_dictionary = 'NWL'
        mock_min_length = 4
        mock_updated_at = time.time() # same-day!
        mock_bonus_word = 'PLAY'
        mock_bonus_cell = [1, 0]
        mock_format = 'Normal'
        mock_uniqueness = 0.55
        mock_range = '100-200'
        
        # Setup mock active players list
        mock_players = [
            {
                'user_id': 12345,
                'username': 'jeffy',
                'rating': 1350,
                'submitted_words': [
                    {'word': 'TEST', 'time': time.time(), 'points': 1, 'path': [[0,0],[0,1],[0,2],[0,3]]}
                ],
                'invalid_words': [],
                'score': 1,
                'previous_round_score': 0,
                'games_played': 10,
                'previous_submitted_words': [],
                'found_bonus_word': False,
                'last_active': time.time(),
                'input_method': 'mouse',
                'country_flag': '🇺🇸',
                'joined_mid_round': False,
                'has_exceptional_round': False,
                'is_guest': False,
                'is_ai': False,
                'ai_rating': 1200,
                'has_abandoned': False
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        # Ensure migration columns exist
        try: conn.execute('ALTER TABLE active_boards ADD COLUMN bonus_word TEXT')
        except sqlite3.OperationalError: pass
        try: conn.execute('ALTER TABLE active_boards ADD COLUMN bonus_cell_json TEXT')
        except sqlite3.OperationalError: pass
        try: conn.execute('ALTER TABLE active_boards ADD COLUMN board_format TEXT')
        except sqlite3.OperationalError: pass
        try: conn.execute('ALTER TABLE active_boards ADD COLUMN uniqueness REAL')
        except sqlite3.OperationalError: pass
        try: conn.execute('ALTER TABLE active_boards ADD COLUMN word_count_range TEXT')
        except sqlite3.OperationalError: pass
        try: conn.execute('ALTER TABLE active_boards ADD COLUMN active_players_json TEXT')
        except sqlite3.OperationalError: pass
        conn.commit()
        
        conn.execute('''
            INSERT OR REPLACE INTO active_boards (
                room_id, board_data, all_words, dictionary, min_length, updated_at,
                bonus_word, bonus_cell_json, board_format, uniqueness, word_count_range,
                active_players_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.room_id,
            json.dumps(mock_board),
            json.dumps(mock_words),
            mock_dictionary,
            mock_min_length,
            mock_updated_at,
            mock_bonus_word,
            json.dumps(mock_bonus_cell),
            mock_format,
            mock_uniqueness,
            mock_range,
            json.dumps(mock_players)
        ))
        conn.commit()
        conn.close()
        
        print(f"1. Mock active board and players inserted into active_boards for {self.room_id}")
        
        # 2. Re-create/load the 24h room using create_room
        # This simulates players leaving completely (wiping room from memory) and a new one entering.
        print("2. Re-creating/loading the 24h room through create_room()...")
        room = self.room_manager.create_room(
            self.room_id, 
            game_type="accumulative", 
            time_limit=86400, 
            board_dimensions="4x4"
        )
        
        # 3. Verify that the board and parameters were perfectly restored
        print("3. Verifying restored active board values...")
        self.assertEqual(room.state, 'active')
        self.assertEqual(room.board, mock_board)
        self.assertEqual(room.all_words, set(mock_words))
        self.assertEqual(room.current_dictionary, mock_dictionary)
        self.assertEqual(room.current_min_length, mock_min_length)
        self.assertEqual(room.bonus_word, mock_bonus_word)
        self.assertEqual(room.bonus_cell, mock_bonus_cell)
        self.assertEqual(room.current_board_format, 'Valued Letters')
        self.assertEqual(room.current_uniqueness, mock_uniqueness)
        self.assertEqual(room.current_word_count_range, mock_range)
        
        # Verify active players restoration
        print("4. Verifying active players and words restoration...")
        self.assertEqual(len(room.players), 1)
        player = room.players[0]
        self.assertEqual(player.username, 'jeffy')
        self.assertEqual(player.rating, 1350)
        self.assertEqual(player.score, 1)
        self.assertEqual(len(player.submitted_words), 1)
        self.assertEqual(player.submitted_words[0]['word'], 'TEST')
        
        print("   Restored board, players, and submitted words match the DB record exactly!")
        print("🎉 24H ROOM BOARD & WORDS PERSISTENCE TEST PASSED SUCCESSFULLY!")

    def test_24h_midnight_reset_db_clearing(self):
        print("\n=== STARTING 24H MIDNIGHT RESET DB CLEARING VERIFICATION ===")
        
        # 1. Create a 24h room
        room = self.room_manager.create_room(
            self.room_id, 
            game_type="accumulative", 
            time_limit=86400, 
            board_dimensions="4x4"
        )
        
        # Mock initial row in active_boards so save_active_players UPDATE succeeds
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT OR REPLACE INTO active_boards (room_id, board_data, all_words, updated_at) VALUES (?, ?, ?, ?)',
                     (self.room_id, '[]', '[]', time.time()))
        conn.commit()
        conn.close()
        
        # 2. Add an active player with a word
        player_id = 99999
        room.add_player(
            user_id=player_id,
            username="test_user",
            rating=1200
        )
        p = room.get_player(player_id)
        p.submitted_words = [{'word': 'TEST', 'time': time.time(), 'points': 1, 'path': []}]
        p.score = 1
        
        # Save to DB to simulate active play
        room.save_active_players()
        
        # Verify it saved to DB first
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT active_players_json FROM active_boards WHERE room_id = ?', (self.room_id,))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        players_data = json.loads(row[0])
        self.assertEqual(len(players_data), 1)
        self.assertEqual(players_data[0]['username'], 'test_user')
        self.assertEqual(len(players_data[0]['submitted_words']), 1)
        conn.close()
        
        # 3. Trigger start_next_round (which simulates the midnight transition)
        print("Triggering start_next_round (midnight transition)...")
        # Setup next_round attributes so generator doesn't stall
        room.next_round_board = [['T','E','S','T'],['P','L','A','Y'],['W','O','R','D'],['G','A','M','E']]
        room.next_round_words = ['TEST', 'PLAY', 'WORD', 'GAME']
        room.next_round_word_paths = {'TEST': []}
        room.next_round_word_scores = {'TEST': {'total': 1, 'base': 1}}
        room.next_round_bonus = 'PLAY'
        room.next_round_total_words_count = 4
        room.state = 'intermission'
        
        self.room_manager.start_next_round(self.room_id)
        
        # 4. Verify that the database active_players_json is now '[]'
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT active_players_json FROM active_boards WHERE room_id = ?', (self.room_id,))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        players_data = json.loads(row[0])
        print(f"Database active_players_json length after reset: {len(players_data)}")
        self.assertEqual(len(players_data), 0, "Database active_players_json should be empty after midnight reset!")
        conn.close()
        
        print("🎉 24H MIDNIGHT RESET DB CLEARING TEST PASSED SUCCESSFULLY!")
        
    def test_24h_valued_letters_scoring(self):
        print("\n=== STARTING 24H VALUED LETTERS SCORING VERIFICATION ===")
        
        # 1. Create a 24h room
        room = self.room_manager.create_room(
            self.room_id, 
            game_type="accumulative", 
            time_limit=86400, 
            board_dimensions="4x4"
        )
        
        # 2. Add an active player
        player_id = 88888
        room.add_player(
            user_id=player_id,
            username="jeffy_scoring_test",
            rating=1200
        )
        
        # Force a board and valid words so we can submit
        room.board = [['A','B','L','E'],['X','X','X','X'],['X','X','X','X'],['X','X','X','X']]
        room.all_words = {'ABLE'}
        room.all_words_paths = {'ABLE': [[0,0],[0,1],[0,2],[0,3]]}
        room.current_board_format = 'Valued Letters'
        
        # 3. Submit word "ABLE" and assert it gets 10 points
        # A=2, B=4, L=3, E=1 -> Total = 10 points
        success, msg, points, details = room.submit_word(player_id, "ABLE", path=[[0,0],[0,1],[0,2],[0,3]])
        self.assertTrue(success)
        self.assertEqual(points, 10, f"Expected 10 points for 'ABLE' under Valued Letters, got {points} instead.")
        
        # Check that player score is correctly recalculated
        player = room.get_player(player_id)
        self.assertEqual(player.score, 10)
        
        print("🎉 24H VALUED LETTERS SCORING TEST PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    unittest.main()
