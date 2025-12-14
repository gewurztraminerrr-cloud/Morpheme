import os
import sys
import json
import uuid
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

# Add parent directory to path to allow imports from generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import adaptive_puzzle, checkerboard_puzzle, checkerboard_v2, checkerboard_v3, generate_optimized, hybrid_puzzle
from web.models import db, User

app = Flask(__name__)
app.config['SECRET_KEY'] = 'morpheme-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///morpheme.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'index'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Configuration
WORD_LIST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generator", "TWL.txt")
DIFFICULT_LIST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generator", "randomTWLunique.txt")

ALGORITHMS = {
    'adaptive': adaptive_puzzle,
    'checkerboard': checkerboard_puzzle,
    'checkerboard_v2': checkerboard_v2,
    'checkerboard_v3': checkerboard_v3,
    'optimized': generate_optimized,
    'hybrid': hybrid_puzzle
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    return jsonify({
        'algorithms': [
            {'id': 'adaptive', 'name': 'Adaptive (Radial)', 'description': 'Balanced difficulty using radial density.'},
            {'id': 'checkerboard', 'name': 'Checkerboard', 'description': 'Distinct difficult word optimization.'},
            {'id': 'checkerboard_v2', 'name': 'Checkerboard v2', 'description': 'Enum strategies: total, distinct, or 6+ ratio.'},
            {'id': 'checkerboard_v3', 'name': 'Checkerboard v3 (Parallel)', 'description': 'Parallel Trie-based optimization with persistent pool.'},
            {'id': 'optimized', 'name': 'Optimized (Legacy)', 'description': 'Original optimization algorithm.'},
            {'id': 'hybrid', 'name': 'Hybrid Beam', 'description': 'High difficulty using beam search.'}
        ]
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        algo_id = data.get('algorithm', 'adaptive')
        width = int(data.get('width', 6))
        height = int(data.get('height', 6))
        seed = data.get('seed')
        if seed is not None:
            seed = int(seed)
        
        params = data.get('params', {})
        
        min_diff = int(data.get('min_difficult', 0))
        min_total = int(data.get('min_total', 0))
        min_ratio = int(data.get('min_ratio', 0))
        max_attempts = int(data.get('max_attempts', 10))

        if algo_id not in ALGORITHMS:
            return jsonify({'error': f'Unknown algorithm: {algo_id}'}), 400
            
        module = ALGORITHMS[algo_id]
        
        best_result = None
        best_score = -1.0
        
        current_seed = seed
        
        for attempt in range(max_attempts):
            # If retrying and using a fixed seed, increment it to explore
            if attempt > 0 and current_seed is not None:
                current_seed += 1
                
            result = module.generate_puzzle(
                width=width,
                height=height,
                word_list_path=WORD_LIST_PATH,
                difficult_list_path=DIFFICULT_LIST_PATH,
                seed=current_seed,
                params=params
            )
            
            # Check constraints
            found = result['found_words']
            difficult = result['difficult_words']
            
            c_total = len(found)
            c_diff = len(difficult)
            
            # Calculate 6+ ratio
            # Convert to list if it's a set to iterate, though iterating set is fine
            w6 = [w for w in found if len(w) >= 6]
            d6 = [w for w in difficult if len(w) >= 6]
            c_ratio = (len(d6) / len(w6) * 100) if w6 else 0
            
            # Check satisfied
            sat_diff = c_diff >= min_diff
            sat_total = c_total >= min_total
            sat_ratio = c_ratio >= min_ratio
            
            if sat_diff and sat_total and sat_ratio:
                # Success
                result['found_words'] = sorted(list(found))
                result['difficult_words'] = sorted(list(difficult))
                result['attempts_taken'] = attempt + 1
                return jsonify(result)
            
            # Score for best effort (sum of completion percentages)
            s_diff = min(1.0, c_diff / min_diff) if min_diff > 0 else 1.0
            s_total = min(1.0, c_total / min_total) if min_total > 0 else 1.0
            s_ratio = min(1.0, c_ratio / min_ratio) if min_ratio > 0 else 1.0
            
            score = s_diff + s_total + s_ratio
            
            if score > best_score:
                best_score = score
                best_result = result
        
        # If loop finishes without perfect match, return best result
        result = best_result
        if result:
            result['found_words'] = sorted(list(result['found_words']))
            result['difficult_words'] = sorted(list(result['difficult_words']))
            result['attempts_taken'] = max_attempts
            return jsonify(result)
            
        return jsonify({'error': 'Generation failed completely'}), 500
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Authentication Endpoints
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.json
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        # Validation
        if not username or len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        if not email or '@' not in email:
            return jsonify({'error': 'Invalid email address'}), 400
        if not password or len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create user
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        # Auto-login after registration
        login_user(new_user)
        
        return jsonify({
            'success': True,
            'user': new_user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        # Find user
        user = User.query.filter_by(username=username).first()
        
        if not user or not bcrypt.check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Login user
        login_user(user)
        
        return jsonify({
            'success': True,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'success': True}), 200

@app.route('/api/auth/user', methods=['GET'])
@login_required
def get_current_user():
    return jsonify({'user': current_user.to_dict()})

# Store active rooms in memory (use Redis/database in production)
active_rooms = {}
room_timers = {}  # Track active timers

INTERMISSION_DURATION = 60  # 1 minute in seconds

def generate_board_for_room(room_id):
    """Generate a new board for the room"""
    if room_id not in active_rooms:
        return
    
    room = active_rooms[room_id]
    
    try:
        result = ALGORITHMS['adaptive'].generate_puzzle(
            width=room['width'],
            height=room['height'],
            word_list_path=WORD_LIST_PATH,
            difficult_list_path=DIFFICULT_LIST_PATH,
            seed=None,
            params={'min_difficult_words': max(5, room['width'] * room['height'] // 4)}
        )
        
        new_board = {
            'grid': result['grid'],
            'words': list(result.get('found_words', result.get('words', []))),
            'difficult_words': list(result.get('difficult_words', []))
        }
        
        room['boards'].append(new_board)
        room['board_index'] = len(room['boards']) - 1
        print(f"Generated board {room['board_index']} for room {room_id}")
        
    except Exception as e:
        print(f"Error generating board for room {room_id}: {e}")

def start_board_phase(room_id):
    """Start a board playing phase"""
    if room_id not in active_rooms:
        return
    
    room = active_rooms[room_id]
    room['phase'] = 'PLAYING'
    room['phase_start_time'] = datetime.utcnow().isoformat()
    room['phase_end_time'] = (datetime.utcnow() + timedelta(seconds=room['time_limit'])).isoformat()
    
    # Schedule intermission after time_limit
    timer = threading.Timer(room['time_limit'], start_intermission_phase, args=[room_id])
    timer.daemon = True
    timer.start()
    room_timers[room_id] = timer
    print(f"Room {room_id}: Started PLAYING phase for {room['time_limit']}s")

def start_intermission_phase(room_id):
    """Start intermission phase and generate next board"""
    if room_id not in active_rooms:
        return
    
    room = active_rooms[room_id]
    room['phase'] = 'INTERMISSION'
    room['phase_start_time'] = datetime.utcnow().isoformat()
    room['phase_end_time'] = (datetime.utcnow() + timedelta(seconds=INTERMISSION_DURATION)).isoformat()
    
    # Generate next board during intermission
    generate_board_for_room(room_id)
    
    # Schedule next board phase after intermission
    timer = threading.Timer(INTERMISSION_DURATION, start_board_phase, args=[room_id])
    timer.daemon = True
    timer.start()
    room_timers[room_id] = timer
    print(f"Room {room_id}: Started INTERMISSION phase for {INTERMISSION_DURATION}s")

def get_time_remaining(room):
    """Calculate time remaining in current phase"""
    if 'phase_end_time' not in room:
        return 0
    
    try:
        end_time = datetime.fromisoformat(room['phase_end_time'])
        now = datetime.utcnow()
        remaining = (end_time - now).total_seconds()
        return max(0, int(remaining))
    except:
        return 0

def create_persistent_room(width, height, time_limit, game_type='accumulative'):
    """Create a persistent room that runs continuously"""
    # Use deterministic room ID based on parameters
    room_id = f"{game_type}_{width}x{height}_{time_limit}s"
    
    if room_id in active_rooms:
        return room_id  # Room already exists
    
    try:
        # Generate initial board
        result = ALGORITHMS['adaptive'].generate_puzzle(
            width=width,
            height=height,
            word_list_path=WORD_LIST_PATH,
            difficult_list_path=DIFFICULT_LIST_PATH,
            seed=None,
            params={'min_difficult_words': max(5, width * height // 4)}
        )
        
        board = result['grid']
        words = list(result.get('found_words', result.get('words', [])))
        difficult_words = list(result.get('difficult_words', []))
        
        # Create persistent room
        active_rooms[room_id] = {
            'room_id': room_id,
            'width': width,
            'height': height,
            'time_limit': time_limit,
            'game_type': game_type,
            'board_index': 0,
            'phase': 'PLAYING',
            'phase_start_time': datetime.utcnow().isoformat(),
            'phase_end_time': (datetime.utcnow() + timedelta(seconds=time_limit)).isoformat(),
            'boards': [{
                'grid': board,
                'words': words,
                'difficult_words': difficult_words
            }],
            'players': [],  # Each player will have {id, username, score}
            'created_at': datetime.utcnow().isoformat(),
            'persistent': True
        }
        
        # Start the timer for this room
        start_board_phase(room_id)
        print(f"Created persistent room: {room_id}")
        
        return room_id
    except Exception as e:
        print(f"Error creating persistent room: {e}")
        return None

def initialize_persistent_rooms():
    """Initialize all persistent rooms for Accumulative mode"""
    # Define all Accumulative room configurations from the lobby
    configs = [
        (4, 4, 45), (4, 4, 180), (4, 4, 300), (4, 4, 900),
        (5, 5, 45), (5, 5, 180), (5, 5, 300), (5, 5, 900),
        (6, 6, 45), (6, 6, 180), (6, 6, 300), (6, 6, 900),
        (7, 7, 45), (7, 7, 180), (7, 7, 300), (7, 7, 900),
    ]
    
    print("Initializing persistent rooms for Accumulative mode...")
    for width, height, time_limit in configs:
        create_persistent_room(width, height, time_limit, 'accumulative')
    print(f"Initialized {len(configs)} persistent rooms")

@app.route('/api/rooms/create', methods=['POST'])
def create_room():
    """Join or create a game room"""
    data = request.get_json()
    width = int(data.get('width', 4))
    height = int(data.get('height', 4))
    time_limit = int(data.get('time', 180))
    game_type = data.get('type', 'accumulative')
    
    # For Accumulative mode, join persistent room
    if game_type == 'accumulative':
        room_id = f"{game_type}_{width}x{height}_{time_limit}s"
        
        if room_id not in active_rooms:
            # Create persistent room if it doesn't exist
            room_id = create_persistent_room(width, height, time_limit, game_type)
            if not room_id:
                return jsonify({'error': 'Failed to create room'}), 500
        
        room = active_rooms[room_id]
        
        # Add user to room if authenticated
        if current_user.is_authenticated:
            user_info = {'id': current_user.id, 'username': current_user.username, 'score': 0}
            # Check if player already in room, update if exists
            existing_player = next((p for p in room['players'] if p['id'] == current_user.id), None)
            if not existing_player:
                room['players'].append(user_info)
        
        return jsonify({
            'success': True,
            'room_id': room_id,
            'room': room,
            'time_remaining': get_time_remaining(room)
        })
    
    # For other game types, create new room (keep existing logic)
    # Generate unique room ID
    room_id = str(uuid.uuid4())[:8]
    
    # Get user info (handle both logged in and not logged in)
    if current_user.is_authenticated:
        user_info = {'id': current_user.id, 'username': current_user.username}
    else:
        user_info = {'id': 'guest', 'username': 'Guest'}
    
    # Generate initial board
    try:
        result = ALGORITHMS['adaptive'].generate_puzzle(
            width=width,
            height=height,
            word_list_path=WORD_LIST_PATH,
            difficult_list_path=DIFFICULT_LIST_PATH,
            seed=None,
            params={'min_difficult_words': max(5, width * height // 4)}
        )
        
        board = result['grid']
        words = list(result.get('found_words', result.get('words', [])))
        difficult_words = list(result.get('difficult_words', []))
        
        # Store room state
        active_rooms[room_id] = {
            'room_id': room_id,
            'width': width,
            'height': height,
            'time_limit': time_limit,
            'game_type': game_type,
            'board_index': 0,
            'phase': 'PLAYING',
            'phase_start_time': datetime.utcnow().isoformat(),
            'phase_end_time': (datetime.utcnow() + timedelta(seconds=time_limit)).isoformat(),
            'boards': [{
                'grid': board,
                'words': words,
                'difficult_words': difficult_words
            }],
            'players': [user_info],
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Start the timer for this room
        start_board_phase(room_id)
        
        return jsonify({
            'success': True,
            'room_id': room_id,
            'room': active_rooms[room_id],
            'time_remaining': time_limit
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rooms/<room_id>', methods=['GET'])
def get_room(room_id):
    """Get room state with current time"""
    if room_id not in active_rooms:
        return jsonify({'error': 'Room not found'}), 404
    
    room = active_rooms[room_id]
    return jsonify({
        'room': room,
        'time_remaining': get_time_remaining(room),
        'phase': room.get('phase', 'PLAYING')
    })

@app.route('/api/rooms/<room_id>/status', methods=['GET'])
def get_room_status(room_id):
    """Get room status for polling"""
    if room_id not in active_rooms:
        return jsonify({'error': 'Room not found'}), 404
    
    room = active_rooms[room_id]
    # Sort players by score (highest first)
    sorted_players = sorted(room.get('players', []), key=lambda p: p.get('score', 0), reverse=True)
    
    return jsonify({
        'phase': room.get('phase', 'PLAYING'),
        'board_index': room.get('board_index', 0),
        'time_remaining': get_time_remaining(room),
        'total_boards': len(room.get('boards', [])),
        'players': sorted_players
    })

@app.route('/api/rooms/<room_id>/next-board', methods=['POST'])
@login_required
def next_board(room_id):
    """Generate and switch to next board"""
    if room_id not in active_rooms:
        return jsonify({'error': 'Room not found'}), 404
    
    room = active_rooms[room_id]
    
    # Generate new board
    try:
        result = ALGORITHMS['adaptive'].generate_puzzle(
            width=room['width'],
            height=room['height'],
            word_list_path=WORD_LIST_PATH,
            difficult_list_path=DIFFICULT_LIST_PATH,
            seed=None,
            params={'min_difficult_words': max(5, room['width'] * room['height'] // 4)}
        )
        
        new_board = {
            'grid': result['grid'],
            'words': list(result.get('found_words', result.get('words', []))),
            'difficult_words': list(result.get('difficult_words', []))
        }
        
        room['boards'].append(new_board)
        room['board_index'] = len(room['boards']) - 1
        
        return jsonify({
            'success': True,
            'board': new_board,
            'board_index': room['board_index']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    
    # Initialize persistent rooms for Accumulative mode
    initialize_persistent_rooms()
    
    app.run(debug=True, port=5000)
