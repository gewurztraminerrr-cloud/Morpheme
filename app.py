from flask import Flask, request, jsonify, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import time

app = Flask(__name__, static_folder='static')
app.secret_key = 'morpheme-secret-key-2024'

# Initialize database
def init_db():
    conn = sqlite3.connect('morpheme.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            rating INTEGER DEFAULT 1000
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Serve static files
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

# Authentication endpoints
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if len(username) < 3 or len(password) < 6:
        return jsonify({'error': 'Username 3+ chars, password 6+ chars'}), 400
    
    conn = sqlite3.connect('morpheme.db')
    try:
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                    (username, password_hash))
        conn.commit()
        
        cursor = conn.execute('SELECT id, rating FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        session['user_id'] = user[0]
        session['username'] = username
        
        return jsonify({'success': True, 'username': username})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 400
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    conn = sqlite3.connect('morpheme.db')
    cursor = conn.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if not user or not check_password_hash(user[1], password):
        return jsonify({'error': 'Invalid username or password'}), 401
    
    session['user_id'] = user[0]
    session['username'] = username
    
    return jsonify({'success': True, 'username': username})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/guest-login', methods=['POST'])
def guest_login():
    import random
    # Generate unique guest username
    guest_id = random.randint(10000, 99999)
    guest_username = f'Guest_{guest_id}'
    
    # Create guest session (no database entry needed)
    session['user_id'] = -guest_id  # Negative ID to distinguish from real users
    session['username'] = guest_username
    session['is_guest'] = True
    
    return jsonify({'success': True, 'username': guest_username})

@app.route('/api/session', methods=['GET'])
def get_session():
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'username': session['username']
        })
    return jsonify({'authenticated': False})

# Game Room APIs
from game_room import room_manager
import uuid

@app.route('/api/room/create', methods=['POST'])
def create_room():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    game_type = data.get('game_type')
    time_limit = data.get('time_limit')
    board_dimensions = data.get('board_dimensions')
    
    # Create room
    room_id = str(uuid.uuid4())
    room = room_manager.create_room(room_id, game_type, time_limit, board_dimensions)
    
    # Get player rating - guests default to 1000, registered users query database
    rating = 1000  # Default for guests
    if not session.get('is_guest', False):
        # Only query database for registered users
        conn = sqlite3.connect('morpheme.db')
        cursor = conn.execute('SELECT rating FROM users WHERE id = ?', (session['user_id'],))
        user = cursor.fetchone()
        conn.close()
        rating = user[0] if user else 1000
    
    room.add_player(session['user_id'], session['username'], rating)
    
    # Start first round immediately in background for faster loading
    import threading
    thread = threading.Thread(target=room_manager.start_round, args=(room_id,), daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'room_id': room_id})

@app.route('/api/room/<room_id>/join', methods=['POST'])
def join_room(room_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    room = room_manager.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
    
    # Get user rating
    conn = sqlite3.connect('morpheme.db')
    cursor = conn.execute('SELECT rating FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    conn.close()
    
    rating = user[0] if user else 1000
    room.add_player(session['user_id'], session['username'], rating)
    
    return jsonify({'success': True})

@app.route('/api/room/<room_id>/leave', methods=['POST'])
def leave_room(room_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    room = room_manager.get_room(room_id)
    if room:
        room.remove_player(session['user_id'])
        
        # Delete room if empty
        if len(room.players) == 0:
            room_manager.delete_room(room_id)
    
    return jsonify({'success': True})

@app.route('/api/room/<room_id>/state', methods=['GET'])
def get_room_state(room_id):
    print(f"\n=== GET STATE REQUEST for room {room_id} ===")
    room = room_manager.get_room(room_id)
    if not room:
        print(f"ERROR: Room {room_id} not found")
        return jsonify({'error': 'Room not found'}), 404
    
    print(f"Room found - game_type: {room.game_type}, current_round: {room.current_round}, state: {room.state}")
    
    # Check and update state based on timers
    prev_state = room.state
    state_changed = room.check_and_update_state()
    
    # If just transitioned to intermission, start complete solving in background
    if state_changed and room.state == 'intermission' and prev_state == 'active':
        print(f"Transitioned to intermission, starting complete solving...")
        room.solving_complete = False  # Reset flag
        room.complete_words = []  # Clear previous complete words
        room_manager.start_complete_solving(room_id)
    
    # If intermission just ended for Accumulative, check for timing milestones
    if room.state == 'intermission' and room.game_type == 'accumulative':
        milestone = room.get_intermission_milestone()
        
        if milestone == 'spinner':
            # At 45s remaining: Generate Spinner Set parameters
            print(f"[Milestone] 45s remaining - Generating Spinner Set parameters")
            room_manager.generate_spinner_params(room_id)
        
        elif milestone == 'search':
            # At 15s remaining: Start board search
            print(f"[Milestone] 15s remaining - Starting board search")
            room_manager.start_board_search(room_id)
        
        elif milestone == 'start':
            # At 0s: Start next round with pre-generated board
            print(f"[Milestone] 0s remaining - Starting next round")
            room_manager.start_next_round(room_id)

    
    # Determine which word list to return
    # During intermission: use complete_words if available and solving is done, otherwise all_words
    # During active: use all_words (fast initial list for validation)
    words_to_return = room.all_words
    if room.state == 'intermission':
        # Use complete words if solving is done, otherwise show initial words while solving
        if room.solving_complete and room.complete_words:
            words_to_return = room.complete_words
        else:
            words_to_return = room.all_words
    
    return jsonify({
        'state': room.state,
        'current_round': room.current_round,
        'time_remaining': room.time_remaining,
        'server_time': time.time(),  # Current server timestamp
        'round_end_time': room.round_end_time,  # When active round ends
        'intermission_end_time': room.intermission_end_time,  # When intermission ends
        'board': room.board,
        'board_dimensions': room.board_dimensions,
        'time_limit': room.time_limit,
        'all_words': words_to_return,
        'bonus_word': room.bonus_word,
        'spinner_params': room.spinner_params,
        'solving_complete': room.solving_complete,  # Let frontend know if still solving
        'players': [
            {
                'username': p.username,
                'rating': p.rating,
                'words_count': len(p.submitted_words),
                'score': p.score,
                'rating_change': p.rating_change,
                'found_bonus_word': p.found_bonus_word,
                'submitted_words': p.submitted_words
            } for p in room.players
        ]
    })

@app.route('/api/room/<room_id>/submit', methods=['POST'])
def submit_word(room_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    room = room_manager.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
    
    data = request.get_json()
    word = data.get('word', '').strip()
    
    success, message = room.submit_word(session['user_id'], word)
    
    return jsonify({'success': success, 'message': message})

if __name__ == '__main__':
    print('Morpheme server running on http://localhost:3000')
    app.run(host='0.0.0.0', port=3000, debug=True)
