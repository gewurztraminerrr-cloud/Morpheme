from flask import Flask, request, jsonify, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import time
import os

app = Flask(__name__, static_folder='static')
app.secret_key = 'morpheme-secret-key-2024'

DEFINITIONS_CACHE = None

# Initialize database
def init_db():
    conn = sqlite3.connect('morpheme.db')
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            rating INTEGER DEFAULT 1200
        );
        CREATE TABLE IF NOT EXISTS user_ratings (
            user_id INTEGER,
            config_key TEXT,
            rating INTEGER DEFAULT 1200,
            PRIMARY KEY (user_id, config_key),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
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
from word_validator import word_validator
import uuid

def cleanup_user_rooms(user_id, exclude_room_id=None):
    """Remove user from all rooms except exclude_room_id and 24h persistent rooms"""
    for rid in list(room_manager.rooms.keys()):
        if str(rid) == str(exclude_room_id):
            continue
        room = room_manager.rooms[rid]
        
        # PERSISTENCE RULE: Keep users in 24h rooms even if they join another
        if room.time_limit >= 86400:
            continue
            
        # Remove from players
        room.remove_player(user_id)
        
        if len(room.players) == 0:
            room_manager.delete_room(rid)

@app.route('/api/room/create', methods=['POST'])
def create_room():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    game_type = data.get('game_type')
    time_limit = data.get('time_limit')
    board_dimensions = data.get('board_dimensions')
    min_rating = data.get('min_rating', 0)
    max_rating = data.get('max_rating', 9999)
    
    # Guest Restriction: Guests cannot create rooms with rating limits
    if session.get('is_guest', False):
        if int(min_rating) > 0 or int(max_rating) < 9999:
            return jsonify({'error': 'Guests can only create/join rooms with no rating limits (0-∞).'}), 403
    
    # Create room
    generated_id = str(uuid.uuid4())
    room = room_manager.create_room(generated_id, game_type, int(time_limit), board_dimensions)
    room.min_rating = int(min_rating)
    room.max_rating = int(max_rating)
    
    # Ensure user is not in any other room
    cleanup_user_rooms(session['user_id'], exclude_room_id=room.room_id)
    
    # Use the actual ID (could be existing one if singleton)
    room_id = room.room_id
    
    # Get configuration-specific rating
    config_key = f"{game_type}|{board_dimensions}|{time_limit}"
    rating = 1200  # Default
    
    if session.get('is_guest', False):
        rating = 0
    else:
        conn = sqlite3.connect('morpheme.db')
        cursor = conn.execute('SELECT rating FROM user_ratings WHERE user_id = ? AND config_key = ?', 
                            (session['user_id'], config_key))
        row = cursor.fetchone()
        if row:
            rating = row[0]
        else:
            # If no specific rating exists, check if legacy global rating exists (optional, or just start at 1200)
            # For now, start fresh at 1200 for each new mode
            rating = 1200
        conn.close()
    
    room.add_player(session['user_id'], session['username'], rating)
    
    # Start first round immediately in background for faster loading
    # Only if NOT already running/active (check logic inside start_round already handles this)
    import threading
    thread = threading.Thread(target=room_manager.start_round, args=(room_id,), daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'room_id': room_id})

@app.route('/api/room/<room_id>/join', methods=['POST'])
def join_room(room_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    room = room_manager.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
    
    # Get configuration-specific rating
    config_key = f"{room.game_type}|{room.board_dimensions}|{room.time_limit}"
    rating = 1200
    
    if session.get('is_guest', False):
        rating = 0
    else:
        conn = sqlite3.connect('morpheme.db')
        cursor = conn.execute('SELECT rating FROM user_ratings WHERE user_id = ? AND config_key = ?', 
                            (session['user_id'], config_key))
        row = cursor.fetchone()
        if row:
            rating = row[0]
        conn.close()
        
    # Ensure user is not in any other room
    cleanup_user_rooms(session['user_id'], exclude_room_id=room_id)

    # Check for spectator request
    data = request.get_json() or {}
    as_spectator = data.get('as_spectator', False)

    # Force player mode for Accumulative
    if room.game_type == 'accumulative':
        as_spectator = False

    if as_spectator:
        room.add_spectator(user_id, session['username'], rating)
        room.update_player_activity(user_id)
        return jsonify({'success': True, 'role': 'spectator'})

    # Guest Restriction: Guests can only join rooms with NO rating limits
    if session.get('is_guest', False):
        if room.min_rating > 0 or room.max_rating < 9999:
            return jsonify({'error': 'Guests can only join rooms with no rating limits (0-∞).'}), 403

    # Validate Rating Range
    if rating < room.min_rating:
         return jsonify({'error': f'Rating {rating} too low (Min: {room.min_rating})'}), 403
    
    if rating > room.max_rating:
         return jsonify({'error': f'Rating {rating} too high (Max: {room.max_rating})'}), 403

    # Try to join as player
    # Guests are welcome unless rating check failed (Guest rating is usually 1200)
    success = room.add_player(user_id, session['username'], rating)
    if not success:
        # Room full
        msg = f"Room is full (Max {room.max_players} players). You can watch instead."
        if room.game_type == 'accumulative':
             msg = "Could not join Accumulative room. Please try again."
        return jsonify({'error': msg}), 409
    
    room.update_player_activity(user_id)
    return jsonify({'success': True, 'role': 'player', 'max_players': room.max_players})

@app.route('/api/room/<room_id>/leave', methods=['POST'])
def leave_room(room_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    room = room_manager.get_room(room_id)
    if room:
        room.remove_player(session['user_id'])
        
        # Delete room if empty (except for 24h rooms which persist)
        if len(room.players) == 0 and room.time_limit < 86400:
            room_manager.delete_room(room_id)
    
    return jsonify({'success': True})

@app.route('/api/rooms', methods=['GET'])
def list_rooms():
    game_type = request.args.get('game_type')
    board_dimensions = request.args.get('board_dimensions')
    time_limit = request.args.get('time_limit', type=int)
    
    # Clean up rooms before listing (ensures zombie rooms are removed)
    room_manager.cleanup_rooms(timeout=420)
    
    active_rooms = []
    
    for room_id, room in room_manager.rooms.items():
        if (room.game_type == game_type and 
            room.board_dimensions == board_dimensions and 
            room.time_limit == time_limit):
            
            # Calculate combined rating (avg or sum?)
            # Prompt says "Filter by combined rating", assume Sum for now
            combined_rating = sum(p.rating for p in room.players)
            
            active_rooms.append({
                'room_id': room.room_id,
                'player_count': len(room.players),
                'max_players': room.max_players,
                'combined_rating': combined_rating,
                'state': room.state,
                'current_round': room.current_round,
                'players': [{'username': p.username, 'rating': p.rating, 'user_id': p.user_id} for p in room.players]
            })
            
    return jsonify({'rooms': active_rooms})

@app.route('/api/lobby-stats', methods=['GET'])
def get_lobby_stats():
    """Get aggregated player counts for all game configurations"""
    # Clean up first
    room_manager.cleanup_rooms(timeout=420)
    
    stats = {}
    
    for room in room_manager.rooms.values():
        # Create a unique key for this configuration
        # Format: game_type|board|time
        key = f"{room.game_type}|{room.board_dimensions}|{room.time_limit}"
        
        if key not in stats:
            stats[key] = 0
            
        stats[key] += len(room.players)
    
    return jsonify({'stats': stats})

@app.route('/api/room/<room_id>/state', methods=['GET'])
def get_room_state(room_id):
    print(f"\n=== GET STATE REQUEST for room {room_id} ===")
    room = room_manager.get_room(room_id)
    if not room:
        print(f"ERROR: Room {room_id} not found")
        return jsonify({'error': 'Room not found'}), 404
    

    try:
        print(f"Room found - game_type: {room.game_type}, current_round: {room.current_round}, state: {room.state}")

        
        # Check for inactive players (zombies)
        # Use 7 minutes (420s) globally for all game modes
        timeout = 420
        
        players_removed = room.check_inactivity(timeout=timeout)
        
        # Strict Check: If room is empty of players (even if spectators exist), delete it
        # BUT allow a grace period (e.g. 15s) for new rooms where creator hasn't joined yet
        time_alive = time.time() - room.creation_time
        
        # Skip deletion for daily rooms (>= 24h) so they persist
        is_daily_room = room.time_limit >= 86400
        
        if not is_daily_room and len(room.players) == 0 and time_alive > 15:
            print(f"Room {room_id} empty (0 players) and old enough ({time_alive:.1f}s) - deleting")
            room_manager.delete_room(room_id)
            return jsonify({'error': 'Room deleted due to inactivity'}), 404

        # Check and update state based on timers
        prev_state = room.state
        state_changed = room.check_and_update_state()

        
        # If just transitioned to intermission, start complete solving in background
        if state_changed and room.state == 'intermission' and prev_state == 'active':
            print(f"Transitioned to intermission, starting complete solving...")
            room.solving_complete = False  # Reset flag
            room.complete_words = []  # Clear previous complete words
            room_manager.start_complete_solving(room_id)
        
        # If intermission just ended, check for timing milestones (Accumulative & FCFS)
        # If intermission just ended, check for timing milestones (Accumulative & FCFS)
        if room.state == 'intermission' and room.game_type in ['accumulative', 'fcfs', 'split']:
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
                if room_manager.start_next_round(room_id):
                    # PERSISTENCE: Update metrics in database
                    print(f"[Persistence] Updating player ratings to DB...")
                    try:
                        conn = sqlite3.connect('morpheme.db')
                        config_key = f"{room.game_type}|{room.board_dimensions}|{room.time_limit}"
                        
                        for p in room.players:
                            # Only update registered users (positive IDs)
                            if p.user_id > 0:
                                # Use INSERT OR REPLACE to handle upsert
                                conn.execute('''
                                    INSERT OR REPLACE INTO user_ratings (user_id, config_key, rating)
                                    VALUES (?, ?, ?)
                                ''', (p.user_id, config_key, p.rating))
                                
                        conn.commit()
                        conn.close()
                        print(f"[Persistence] Ratings updated successfully for key: {config_key}")
                    except Exception as e:
                        print(f"[Persistence] ERROR updating ratings: {e}")

        
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
        
        # Create response with Cache-Control headers
        resp = jsonify({
            'room_id': room.room_id,
            'game_type': room.game_type,
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
            'all_word_scores': room.solved_words_with_scores,
            'csw_only_words': [w for w in words_to_return if word_validator.is_csw_only(w)],
            'bonus_word': room.bonus_word,
            'spinner_params': room.spinner_params,
            'solving_complete': room.solving_complete,  # Let frontend know if still solving
            'max_players': room.max_players,
            'min_rating': room.min_rating,
            'max_rating': room.max_rating,
            'previous_all_words': room.previous_all_words,
            'your_username': session.get('username'),
            'players': [
                {
                    'user_id': p.user_id,
                    'username': p.username,
                    'rating': p.rating,
                    'words_count': len(p.submitted_words),
                    'score': p.score,
                    'rating_change': p.rating_change,
                    'found_bonus_word': p.found_bonus_word,
                    'submitted_words': p.submitted_words,
                    'previous_submitted_words': p.previous_submitted_words,
                    'invalid_words': p.invalid_words,
                    'input_method': p.input_method,
                    'last_active_age': time.time() - p.last_active
                } for p in sorted(room.players, key=lambda p: p.score, reverse=True)
            ],
            'spectators': [
                {'username': s.username, 'rating': s.rating, 'user_id': s.user_id} for s in room.spectators
            ] if hasattr(room, 'spectators') else [],
            'chat_messages': room.chat_messages
        })
        
        return resp
        
    except Exception as e:
        import traceback
        error_msg = f"ERROR in get_room_state: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/room/<room_id>/chat', methods=['POST'])
def submit_chat_message(room_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    room = room_manager.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
        
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'Message required'}), 400
        
    # Optional: Truncate long messages
    if len(message) > 200:
        message = message[:200]
        
    room.add_chat_message(session['username'], message)
    room.update_player_activity(session['user_id'])
    
    return jsonify({'success': True})

@app.route('/room/<room_id>/submit_word', methods=['POST'])
def submit_word(room_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    room = room_manager.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
        
    data = request.get_json()
    word = data.get('word', '').strip()
    input_method = data.get('input_method')

    # Update input method if provided
    if input_method:
        player = room.get_player(user_id)
        if player:
            player.input_method = input_method
            
    success, message, points, final_word = room.submit_word(user_id, word)
    
    # Refresh activity on any submission attempt (valid or not)
    room.update_player_activity(user_id)
    
    return jsonify({
        'success': success, 
        'message': message,
        'points': points,
        'word': final_word
    })

@app.route('/room/<room_id>/update_input_method', methods=['POST'])
def update_input_method(room_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
        
    data = request.get_json()
    input_method = data.get('input_method')
    
    room = room_manager.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
        
    player = room.get_player(user_id)
    if player:
        player.input_method = input_method
        room.update_player_activity(user_id)
        return jsonify({'success': True})
        
    return jsonify({'error': 'Player not found'}), 404

# Definitions Cache
DEFINITIONS_CACHE = None

def load_definitions():
    global DEFINITIONS_CACHE
    if DEFINITIONS_CACHE is not None:
        return

    DEFINITIONS_CACHE = {}
    try:
        definitions_path = os.path.expanduser('~/Desktop/Definitions.txt')
        if os.path.exists(definitions_path):
            print(f"Loading definitions from {definitions_path}...")
            with open(definitions_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.split(' - ', 1)
                    if len(parts) == 2:
                        word = parts[0].strip()
                        definition = parts[1].strip()
                        DEFINITIONS_CACHE[word] = definition
            print(f"Loaded {len(DEFINITIONS_CACHE)} definitions")
        else:
            print(f"Definitions file not found at {definitions_path}")
            DEFINITIONS_CACHE = {}
    except Exception as e:
        print(f"Error loading definitions: {e}")
        DEFINITIONS_CACHE = {}

@app.route('/api/definition', methods=['GET'])
def get_definition():
    word = request.args.get('word', '').upper()
    if not word:
        return jsonify({'error': 'Word parameter required'}), 400
    
    if DEFINITIONS_CACHE is None:
        load_definitions()
    
    definition = DEFINITIONS_CACHE.get(word)
    if definition:
        return jsonify({'word': word, 'definition': definition})
    else:
        return jsonify({'error': 'Definition not found'}), 404


if __name__ == '__main__':
    print('Morpheme server running on http://localhost:3000')
    app.run(host='0.0.0.0', port=3000, debug=True)
