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

def apply_leave_penalty(user_id, room):
    """Apply -16 rating penalty if user leaves a non-24h room with score > 0"""
    if room.time_limit >= 86400:
        return # No penalty for 24h rooms
    
    player = room.get_player(user_id)
    if player and player.score > 0 and player.user_id > 0:
        print(f"[Penalty] Player {player.username} left room {room.room_id} with score {player.score}. Applying -16 rating penalty.")
        player.rating = max(0, player.rating - 16)
        
        # Persist to database immediately
        try:
            conn = sqlite3.connect('morpheme.db')
            config_key = f"{room.game_type}|{room.board_dimensions}|{room.time_limit}"
            conn.execute('''
                INSERT OR REPLACE INTO user_ratings (user_id, config_key, rating)
                VALUES (?, ?, ?)
            ''', (player.user_id, config_key, player.rating))
            conn.commit()
            conn.close()
            print(f"[Penalty] Rating updated in DB for {player.username} ({config_key})")
        except Exception as e:
            print(f"[Penalty] ERROR updating rating in DB: {e}")

def cleanup_user_rooms(user_id, exclude_room_id=None):
    """Remove user from all rooms except exclude_room_id and 24h persistent rooms"""
    for rid in list(room_manager.rooms.keys()):
        if str(rid) == str(exclude_room_id):
            continue
        room = room_manager.rooms[rid]
        
        # PERSISTENCE RULE: Keep users in 24h rooms even if they join another
        if room.time_limit >= 86400:
            continue
            
        # Apply leave penalty if applicable
        apply_leave_penalty(user_id, room)

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
        # Apply leave penalty if applicable
        apply_leave_penalty(session['user_id'], room)
        
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
        # IMPORTANT: We filter by min_word_length here so that validation-only words (length 2)
        # don't appear as "Missed Words" in the UI.
        min_len = room.spinner_params.get('min_word_length', 3)
        
        words_to_return = [w for w in room.all_words if len(w) >= min_len]
        if room.state == 'intermission':
            # Use complete words if solving is done, otherwise show initial words while solving
            if room.solving_complete and room.complete_words:
                words_to_return = [w for w in room.complete_words if len(w) >= min_len]
            else:
                words_to_return = [w for w in room.all_words if len(w) >= min_len]
        
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



# Initialize developer messages database
def init_contact_db():
    conn = sqlite3.connect('developer_messages.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            email TEXT,
            message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_contact_db()

@app.route('/api/contact', methods=['POST'])
def submit_contact():
    data = request.get_json()
    email = data.get('email')
    message = data.get('message')
    
    if not email or not message:
        return jsonify({'error': 'Email and message are required'}), 400
        
    user_id = session.get('user_id', 0)
    username = session.get('username', 'Guest')
    
    try:
        conn = sqlite3.connect('developer_messages.db')
        conn.execute('INSERT INTO messages (user_id, username, email, message) VALUES (?, ?, ?, ?)',
                    (user_id, username, email, message))
        conn.commit()
        conn.close()
        
        print(f"[Contact] Message from {username} ({email}): {message[:50]}...")
        return jsonify({'success': True, 'message': 'Message sent successfully!'})
    except Exception as e:
        print(f"[Contact] Error saving message: {e}")
        return jsonify({'error': 'Failed to send message'}), 500

# --- TOOLS ENDPOINTS ---
TOOLS_DICT_CACHE = {}

def load_tools_dictionary(dict_name):
    """Load dictionary for tools into memory cache"""
    if dict_name in TOOLS_DICT_CACHE:
        return TOOLS_DICT_CACHE[dict_name]
    
    dict_path = os.path.join(os.path.dirname(__file__), 'dictionaries', f'{dict_name}.txt')
    try:
        print(f"[Tools] Loading dictionary: {dict_path}")
        with open(dict_path, 'r') as f:
            words = set(word.strip().upper() for word in f)
        TOOLS_DICT_CACHE[dict_name] = words
        print(f"[Tools] Loaded {len(words)} words from {dict_name}")
        return words
    except FileNotFoundError:
        print(f"[Tools] Dictionary file not found: {dict_path}")
        return set()

def get_lis(nums):
    """Calculates Longest Increasing Subsequence length."""
    if not nums:
        return 0
    # Standard O(n log n) or O(n^2) approach. Words are short, O(n^2) is negligible.
    # Using DP (O(n^2)) for simplicity and correctness with small N.
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if dp else 0

def calculate_mp_pass(source, target, source_len, target_len):
    """Calculates MP score for a specific alignment pass."""
    # 1. Map Positions
    position = [-1] * target_len
    for s_idx, s_char in enumerate(source):
        for t_idx, t_char in enumerate(target):
            if s_char == t_char and position[t_idx] == -1:
                position[t_idx] = s_idx
                break
    
    # 2. Stats
    matched_indices = [p for p in position if p != -1]
    count = len(matched_indices)
    
    # 3. LIS
    count2 = get_lis(matched_indices)
    
    # 4. Calculation
    # Moves = count - count2
    # Inserts = target_len - count (Asterisks)
    # Deletes = source_len - count
    micro_procedures = (count - count2) + (target_len - count) + (source_len - count)
    
    # 5. Hamming Optimization (Java 'count3' check)
    if source_len == target_len:
        count3 = sum(1 for a, b in zip(source, target) if a != b)
        if micro_procedures > count3:
            micro_procedures = count3
            
    return micro_procedures, count

def check_and_add_mp(mp_groups, source_len, target_len, mp, word):
    """Applies strict filtering logic from combos.java."""
    # Check if word is already in this specific MP group
    if word in mp_groups[mp]: 
        return

    added = False
    
    if source_len == 3:
        # User requested 3-letter support. Loose logic inferred.
        if target_len >= 3: added = True
        
    elif source_len == 4:
        # User requested 4-letter support. Loose logic inferred.
        if target_len >= 4: added = True

    elif source_len == 5:
        if target_len >= 5 and mp <= 3:
            if mp >= 3:
                if target_len >= 6: added = True
            else:
                added = True
                
    elif source_len == 6:
        if target_len >= 5 and mp <= 3:
            if mp >= 3:
                if target_len >= 6: added = True
            else:
                added = True

    elif source_len == 7:
        if target_len >= 5 and mp <= 4:
            if mp >= 4:
                if target_len >= 8: added = True
            else:
                added = True
                
    elif source_len == 8:
        if target_len >= 5 and mp <= 4:
            added = True
            
    elif source_len == 9:
        if target_len >= 6 and mp <= 5:
             if mp >= 5:
                 if target_len >= 8: added = True
             else:
                 added = True
                 
    elif source_len == 10:
        if target_len >= 6 and mp <= 5:
            if mp == 5:
                if target_len >= 8: added = True
            else:
                added = True
    
    if added:
        mp_groups[mp].append(word)

def check_and_add_lic(lic_groups, count, target_len, word):
    """Applies strict LIC filtering from combos.java."""
    # Logic: 
    # 3 Matches: target < 5 (Inferred)
    # 4 Matches: target < 6 (Inferred)
    # 5 Matches: target < 7
    # 6 Matches: target < 8
    # 7 Matches: target < 10
    # 8,9,10 Matches: target < 9
    
    if count not in lic_groups:
        lic_groups[count] = []
        
    # Validations from Java
    valid = False
    
    if count == 3 and target_len < 5: valid = True # New for 3-letter inputs
    elif count == 4 and target_len < 6: valid = True # New for 4-letter inputs
    elif count == 5 and target_len < 7: valid = True
    elif count == 6 and target_len < 8: valid = True
    elif count == 7 and target_len < 10: valid = True
    elif count == 8 and target_len < 9: valid = True # Java groups 8,9,10 together for <9 constraint
    elif count == 9 and target_len < 9: valid = True
    elif count == 10 and target_len < 9: valid = True
    
    if valid:
        if word not in lic_groups[count]:
            lic_groups[count].append(word)

@app.route('/api/tools/combo', methods=['POST'])
def tools_combo_check():
    data = request.json
    search_term = data.get('search_term', '').upper().strip()
    dict_name = data.get('dictionary', 'NWL')
    
    # Relaxed validation for 3/4 letter words
    if not search_term or len(search_term) < 3 or len(search_term) > 10:
        if not search_term:
             return jsonify({'error': 'No search term provided'}), 400
        pass 
        
    dictionary = load_tools_dictionary(dict_name)
    if not dictionary:
        return jsonify({'error': f'Dictionary {dict_name} not found'}), 404

    from collections import Counter
    source_len = len(search_term)
    
    # Initialize Groups
    mp_groups = {i: [] for i in range(6)}
    lic_groups = {}
    
    search_term_rev = search_term[::-1]
    
    for word in dictionary:
        target_len = len(word)
        
        if abs(source_len - target_len) > 5: continue
        
        # 4 Passes
        # Pass 1: Fwd / Fwd
        mp1, count1 = calculate_mp_pass(search_term, word, source_len, target_len)
        if mp1 <= 5: check_and_add_mp(mp_groups, source_len, target_len, mp1, word)
        check_and_add_lic(lic_groups, count1, target_len, word)
        
        # Pass 2: Fwd / Rev 
        mp2, count2 = calculate_mp_pass(search_term, word[::-1], source_len, target_len)
        if mp2 <= 5: check_and_add_mp(mp_groups, source_len, target_len, mp2, word)
        check_and_add_lic(lic_groups, count2, target_len, word)

        # Pass 3: Rev / Fwd 
        mp3, count3 = calculate_mp_pass(search_term_rev, word, source_len, target_len)
        if mp3 <= 5: check_and_add_mp(mp_groups, source_len, target_len, mp3, word)
        check_and_add_lic(lic_groups, count3, target_len, word)

        # Pass 4: Rev / Rev 
        mp4, count4 = calculate_mp_pass(search_term_rev, word[::-1], source_len, target_len)
        if mp4 <= 5: check_and_add_mp(mp_groups, source_len, target_len, mp4, word)
        check_and_add_lic(lic_groups, count4, target_len, word)

    # Sort Groups
    for k in mp_groups:
        mp_groups[k].sort(key=lambda x: (-len(x), x))
    for k in lic_groups:
        lic_groups[k].sort(key=lambda x: (len(x), x))
    
    return jsonify({
        'mp_groups': mp_groups, 
        'lic_groups': lic_groups
    })

@app.route('/api/tools/lists', methods=['GET'])
def tools_get_lists():
    """Returns the 5 specific word lists for the Lists tool with optional filtering."""
    try:
        # Get Filter Params
        length_filter = request.args.get('length')
        start_filter = request.args.get('starts_with')
        
        # Convert length to int if provided and not "all"
        target_len = None
        if length_filter and length_filter.lower() != 'all':
            try:
                target_len = int(length_filter)
            except ValueError:
                pass # Ignore invalid format
        
        # Normalize start letter
        start_char = None
        if start_filter and start_filter.lower() != 'all':
            start_char = start_filter.upper().strip()
            if not start_char: start_char = None

        base_dir = os.path.dirname(__file__)
        dict_dir = os.path.join(base_dir, 'dictionaries')
        
        # Helper to load AND filter list
        def load_filtered_list(filename):
            path = os.path.join(dict_dir, filename)
            if not os.path.exists(path):
                return [] # Return empty list, not set, for consistency
            
            words = []
            with open(path, 'r') as f:
                for line in f:
                    w = line.strip().upper()
                    if not w: continue
                    
                    # Apply Filters
                    if target_len is not None and len(w) != target_len:
                        continue
                    if start_char is not None and not w.startswith(start_char):
                        continue
                        
                    words.append(w)
            return words # Already a list

        # 1. NWL
        nwl_list = load_filtered_list('NWL.txt')
        
        # 2. CSW
        csw_list = load_filtered_list('CSW.txt')
        
        # 3. CSW Only
        # We need the full sets to compute difference first if filtering logic is complex,
        # BUT since filtering is simple (length/start), we can filter the resulting set.
        # However, it's faster to verify validity against pre-loaded sets if we had them in memory.
        # Given no global memory cache, let's load full sets for diff logic then filter.
        # Actually, simpler: Load full CSW and NWL sets, diff them, then apply filters to result.
        
        def load_set(filename):
            path = os.path.join(dict_dir, filename)
            if not os.path.exists(path): return set()
            with open(path, 'r') as f:
                return {line.strip().upper() for line in f if line.strip()}
                
        nwl_set_full = load_set('NWL.txt')
        csw_set_full = load_set('CSW.txt')
        csw_only_full = csw_set_full - nwl_set_full
        
        def filter_iterable(iterable):
            filtered = []
            for w in iterable:
                if target_len is not None and len(w) != target_len: continue
                if start_char is not None and not w.startswith(start_char): continue
                filtered.append(w)
            return sorted(filtered)

        # Re-apply filtering to loaded lists (optimization: could merge logic but this is safe)
        # We already loaded filtered NWL/CSW above? 
        # Actually, load_filtered_list reads file line by line. 
        # Let's stick to the set logic for CSW Only to be correct.
        
        # Re-doing clean logic:
        
        # 1. NWL (Filtered)
        # Optimization: If no filters, load full. If filters, stream filter.
        # Since we need sets for CSW-Only, we must load full sets anyway unless we optimize diffing.
        # Let's use the sets we just loaded.
        
        # 4. Likelihood List (Frequency Based)
        # Custom freq: A=190, B=45, C=99, D=82, E=278, F=29, G=69, H=61, I=222, J=4, K=23, L=129, M=71,
        # N=165, O=163, P=74, Q=4, R=172, S=237, T=161, U=81, V=23, W=19, X=7, Y=40, Z=12
        freq = {
            'A': 190, 'B': 45, 'C': 99, 'D': 82, 'E': 278, 'F': 29, 'G': 69, 'H': 61, 'I': 222,
            'J': 4, 'K': 23, 'L': 129, 'M': 71, 'N': 165, 'O': 163, 'P': 74, 'Q': 4, 'R': 172,
            'S': 237, 'T': 161, 'U': 81, 'V': 23, 'W': 19, 'X': 7, 'Y': 40, 'Z': 12
        }
        
        def calculate_likelihood(word):
            return sum(freq.get(c, 0) for c in word)

        # We take NWL as the base for Likelihood
        likelihood_eligible = []
        for w in nwl_set_full:
            if target_len is not None and len(w) != target_len: continue
            if start_char is not None and not w.startswith(start_char): continue
            likelihood_eligible.append(w)
            
        # Sort by Length ASC, then score DESC, then Alpha ASC
        likelihood_eligible.sort(key=lambda x: (len(x), -calculate_likelihood(x), x))
        
        response_data = {
            'nwl': filter_iterable(nwl_set_full),
            'csw': filter_iterable(csw_set_full),
            'csw_only': filter_iterable(csw_only_full),
            'likelihood': likelihood_eligible[:1000], # Expanded to 1000 words
            'added': [],
            'uniques': load_filtered_list('randomTWLunique.txt')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error fetching lists: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tools/sequence', methods=['POST'])
def tools_sequence_search():
    """Handles Sequence Search: Starts/Ends With, Contains (Fwd/Rev)."""
    data = request.json
    sequence = data.get('sequence', '').upper().strip()
    mode = data.get('mode', 'contains') # starts, ends, contains
    length_filter = data.get('length', 'all')
    dict_name = data.get('dictionary', 'NWL')
    
    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400
        
    # Parse Length
    target_len = None
    if length_filter and str(length_filter).lower() != 'all':
        try:
            target_len = int(length_filter)
        except ValueError:
            pass
            
    dictionary = load_tools_dictionary(dict_name)
    if not dictionary:
        return jsonify({'error': f'Dictionary {dict_name} not found'}), 404
        
    results = []
    seq_rev = sequence[::-1]
    
    for word in dictionary:
        # 1. Length Filter
        if target_len is not None and len(word) != target_len:
            continue
            
        # 2. Mode Filter
        matched = False
        if mode == 'starts':
            if word.startswith(sequence): matched = True
        elif mode == 'ends':
            if word.endswith(sequence): matched = True
        elif mode == 'contains':
            # "Contains Sequence (Forwards or Backwards)"
            if sequence in word or seq_rev in word: matched = True
            
        if matched:
            results.append(word)
            
    # Sort results: Length ASC, then Alphabetical (User preference from LIC applied here too for consistency? 
    # Or just Alphabetical? Usually lists are Alpha. Let's do Length then Alpha as it's cleaner for lists)
    results.sort(key=lambda x: (len(x), x))
    
    return jsonify({
        'results': results,
        'count': len(results)
    })

@app.route('/api/tools/manual_solve', methods=['POST'])
def tools_manual_solve():
    """Solves a custom board provided by the user."""
    data = request.json
    board = data.get('board') # 2D list of letters
    dictionary = data.get('dictionary', 'NWL')
    
    if not board or not isinstance(board, list):
        return jsonify({'error': 'No board provided or invalid format'}), 400
        
    try:
        # We use the board_generator from the global room_manager instance
        # _solve_board(self, board, dictionary, word_count_range, min_word_length=3)
        # For manual solve, we don't care about word_count_range, so pass (0, float('inf'))
        all_words = room_manager.board_generator._solve_board(board, dictionary, (0, float('inf')), 3)
        
        # Sort by largest first (Length DESC, then Alpha ASC)
        all_words.sort(key=lambda x: (-len(x), x))
        
        return jsonify({
            'results': all_words,
            'count': len(all_words)
        })
    except Exception as e:
        print(f"Error solving manual board: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tools/random_word', methods=['GET'])
def tools_random_word():
    """Returns a random word based on length and dictionary."""
    dict_name = request.args.get('dictionary', 'NWL')
    length_filter = request.args.get('length', 'all')
    
    dictionary = load_tools_dictionary(dict_name)
    if not dictionary:
        return jsonify({'error': f'Dictionary {dict_name} not found'}), 404
        
    # Filter by length
    target_len = None
    if length_filter and length_filter.lower() != 'all':
        try:
            target_len = int(length_filter)
        except ValueError:
            pass
            
    filtered_words = list(dictionary)
    if target_len:
        filtered_words = [w for w in filtered_words if len(w) == target_len]
        
    if not filtered_words:
        return jsonify({'error': 'No words found for the specified criteria'}), 404
        
    import random
    random_word = random.choice(filtered_words)
    
    return jsonify({
        'word': random_word
    })

@app.route('/api/tools/wotd', methods=['GET'])
def tools_wotd():
    """Returns a deterministic Word of the Day based on the current date."""
    from datetime import datetime
    import hashlib
    
    # Use UTC date string as seed for consistency across timezones if needed, 
    # but local server date is standard for most apps.
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    # Load NWL dictionary (default for WOTD)
    dictionary = load_tools_dictionary('NWL')
    if not dictionary:
        return jsonify({'error': 'Dictionary not available'}), 500
        
    # Filter 6-10 letters
    eligible_words = sorted([w for w in dictionary if 6 <= len(w) <= 10])
    
    if not eligible_words:
        return jsonify({'error': 'No eligible words found'}), 500
        
    # Use hash of date string to get a stable random index
    # (Since we want to avoid changing random.seed() global state if possible)
    seed_hash = int(hashlib.md5(today_str.encode()).hexdigest(), 16)
    idx = seed_hash % len(eligible_words)
    wotd = eligible_words[idx]
    
    return jsonify({
        'word': wotd,
        'date': today_str
    })

if __name__ == '__main__':
    print('Morpheme server running on http://localhost:3000')
    app.run(host='0.0.0.0', port=3000, debug=True)
