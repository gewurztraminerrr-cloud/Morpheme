
import sys
import os

sys.path.append(os.getcwd())

from game_room import GameRoom
from board_generator import BoardGenerator

def test_penalty_multiple():
    room = GameRoom('test-penalty-123', 'accumulative', 180, '4x4')
    room.dictionary = 'CSW'
    room.spinner_params = {
        'board_format': 'Penalty',
        'dictionary': 'CSW',
        'word_count_range': (50, 100),
        'min_word_length': 3,
        'difficulty': 'Normal',
        'bonus_word_length': 0
    }
    room.current_board_format = 'Penalty'
    room.current_min_length = 3
    room.all_words = ['DOG', 'CAT'] # FAKE WORDS
    bg = BoardGenerator()
    room.board = bg._create_normal_board(4, 4, [1]*26)
    room.board[0][0] = 'A'
    room.board[0][1] = 'B'
    room.board[0][2] = 'C'
    room.board[0][3] = 'D'
    room.board[1][0] = 'W'
    room.board[1][1] = 'X'
    room.board[1][2] = 'Y'
    room.board[1][3] = 'Z'
    room.add_player('test_user', 'Tester', 1200)
    
    print("Testing word 1 (ABCD)...")
    s, m, pts, fw = room.submit_word('test_user', 'ABCD', path=[[0,0],[0,1],[0,2],[0,3]])
    p = room.get_player('test_user')
    print(f"Success: {s}, Pts: {pts}, Score: {p.score}")
    
    print("Testing word 2 (WXYZ)...")
    s2, m2, pts2, fw2 = room.submit_word('test_user', 'ABCD')
    print(f"Success: {s2}, Pts: {pts2}, Score: {p.score}")

if __name__ == "__main__":
    test_penalty_multiple()
