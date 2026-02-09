from flask import Flask, request, jsonify, session, send_from_directory, g, redirect, url_for, render_template
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import time
import os
import json
from collections import Counter

app = Flask(__name__, static_folder='static')
app.secret_key = 'morpheme-secret-key-2024'

# Auth Helpers
class User:
    def __init__(self, id, username):
        self.id = id
        self.username = username
        self.is_authenticated = True

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required'}), 401
            return redirect('/')
        return f(*args, **kwargs)
    return decorated_function

@app.before_request
def load_user():
    if 'user_id' in session:
        g.user = User(session['user_id'], session['username'])
    else:
        g.user = None

# Auth Helpers



DEFINITIONS_CACHE = None

# Initialize database
def init_db():
    conn = sqlite3.connect('morpheme.db')
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            rating INTEGER DEFAULT 1200,
            games_played INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS user_ratings (
            user_id INTEGER,
            config_key TEXT,
            rating INTEGER DEFAULT 1200,
            PRIMARY KEY (user_id, config_key),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER,
            setting_key TEXT,
            setting_value TEXT,
            PRIMARY KEY (user_id, setting_key),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    ''')
    conn.commit()
    conn.commit()
    
    # MIGRATION: Ensure games_played column exists
    try:
        conn.execute('ALTER TABLE users ADD COLUMN games_played INTEGER DEFAULT 0')
        conn.commit()
        print("Migrated DB: Added games_played column to users")
    except sqlite3.OperationalError:
        pass # Column likely exists
        
    # MIGRATION: Ensure wins column exists
    try:
        conn.execute('ALTER TABLE users ADD COLUMN wins INTEGER DEFAULT 0')
        conn.commit()
        print("Migrated DB: Added wins column to users")
    except sqlite3.OperationalError:
        pass # Column likely exists

    # MIGRATION: Fix any existing 0 ratings for registered users
    # Users who previously joined might have 0 rating due to bug
    # We update them to 1200 (Default)
    try:
        # Update user_ratings where rating is 0 and user is registered (user_id > 0)
        conn.execute('UPDATE user_ratings SET rating = 1200 WHERE rating = 0 AND user_id > 0')
        changes = conn.total_changes
        if changes > 0:
            print(f"Migrated DB: Updated {changes} user ratings from 0 to 1200")
            conn.commit()
    except Exception as e:
        print(f"Migration Error (Rating Fix): {e}")

    # MIGRATION: Add avatar_url column
    try:
        conn.execute('ALTER TABLE users ADD COLUMN avatar_url TEXT')
        conn.commit()
        print("Migrated DB: Added avatar_url column to users")
    except sqlite3.OperationalError:
        pass # Column likely exists
    # MIGRATION: Add country_flag column
    try:
        conn.execute('ALTER TABLE users ADD COLUMN country_flag TEXT')
        conn.commit()
        print("Migrated DB: Added country_flag column to users")
    except sqlite3.OperationalError:
        pass # Column likely exists

    # MIGRATION: Add profile detail columns
    columns = [
        ('full_name', 'TEXT'),
        ('age', 'TEXT'),
        ('gender', 'TEXT'),
        ('location', 'TEXT'),
        ('quote', 'TEXT'),
        ('description', 'TEXT'),
        ('proof_url', 'TEXT')
    ]
    for col_name, col_type in columns:
        try:
            conn.execute(f'ALTER TABLE users ADD COLUMN {col_name} {col_type}')
            conn.commit()
        except sqlite3.OperationalError:
            pass

    # TOURNAMENTS TABLES
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS tournaments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT DEFAULT 'signup', -- signup, active, completed
            start_date REAL, -- timestamp
            parameters TEXT, -- JSON spinner params
            current_round INTEGER DEFAULT 0,
            created_at REAL,
            completed_at REAL
        );
        
        CREATE TABLE IF NOT EXISTS tournament_participants (
            tournament_id INTEGER,
            user_id INTEGER,
            status TEXT DEFAULT 'active', -- active, eliminated
            final_rank INTEGER,
            joined_at REAL,
            PRIMARY KEY (tournament_id, user_id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        
        CREATE TABLE IF NOT EXISTS tournament_rounds (
            tournament_id INTEGER,
            round_number INTEGER,
            start_time REAL,
            end_time REAL,
            board_data TEXT, -- JSON of board
            PRIMARY KEY (tournament_id, round_number)
        );

        CREATE TABLE IF NOT EXISTS tournament_scores (
            tournament_id INTEGER,
            round_number INTEGER,
            user_id INTEGER,
            score INTEGER DEFAULT 0,
            submitted_words TEXT, -- JSON
            submitted_at REAL,
            PRIMARY KEY (tournament_id, round_number, user_id)
        );
    ''')
    conn.commit()

    # MIGRATION: Add skill metrics to users table
    try:
        conn.execute('ALTER TABLE users ADD COLUMN max_pe REAL DEFAULT 0.0')
        conn.execute('ALTER TABLE users ADD COLUMN avg_pe REAL DEFAULT 0.0')
        conn.execute('ALTER TABLE users ADD COLUMN pe_count INTEGER DEFAULT 0')
        conn.commit()
        print("Migrated DB: Added PE columns to users")
    except sqlite3.OperationalError:
        pass

    # MIGRATION: Add user_rating to round_history
    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN user_rating INTEGER DEFAULT 1200')
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # MIGRATION: Add performance_ratio to round_history
    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN performance_ratio REAL DEFAULT 0.0')
        conn.commit()
        print("Migrated DB: Added performance_ratio column to round_history")
    except sqlite3.OperationalError:
        pass

    # MIGRATION: Add best_word, best_word_score, board_dimensions to round_history
    # MIGRATION: Add best_word
    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN best_word TEXT')
    except sqlite3.OperationalError:
        pass

    # MIGRATION: Add best_word_score
    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN best_word_score INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass

    # MIGRATION: Add board_dimensions
    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN board_dimensions TEXT')
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # MIGRATION: Add round_history table
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS round_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                room_id TEXT NOT NULL,
                game_type TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                board_json TEXT NOT NULL,
                words_json TEXT NOT NULL,
                total_score INTEGER NOT NULL,
                round_start_time REAL,
                round_duration INTEGER,
                user_rating INTEGER DEFAULT 1200,
                performance_ratio REAL DEFAULT 0.0,
                best_word TEXT,
                best_word_score INTEGER DEFAULT 0,
                board_dimensions TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()
        print("Migrated DB: Added round_history table")
    except Exception as e:
        print(f"Migration Error (round_history): {e}")

    # BACKFILL: Infer board_dimensions from board_json if missing (For Leaderboards All-Time)
    try:
        cursor = conn.execute("SELECT id, board_json FROM round_history WHERE board_dimensions IS NULL")
        rows = cursor.fetchall()
        if rows:
            print(f"Backfilling {len(rows)} round_history records with dimensions...")
            import json # Ensure json is available
            for row in rows:
                try:
                    rid, bjson = row
                    board = json.loads(bjson)
                    if board and len(board) > 0:
                        dims = f"{len(board)}x{len(board[0])}"
                        conn.execute("UPDATE round_history SET board_dimensions = ? WHERE id = ?", (dims, rid))
                except Exception as e:
                    print(f"Error backfilling round {rid}: {e}")
            conn.commit()
            print("Backfill complete.")
    except Exception as e:
        # Table might not exist yet or other error
        pass


    # MIGRATION: Add private_messages table
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS private_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_id INTEGER NOT NULL,
                receiver_id INTEGER NOT NULL,
                sender_username TEXT NOT NULL,
                message TEXT NOT NULL,
                is_read INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(sender_id) REFERENCES users(id),
                FOREIGN KEY(receiver_id) REFERENCES users(id)
            )
        ''')
        conn.commit()
        print("Migrated DB: Added private_messages table")
    except Exception as e:
        print(f"Migration Error (PM table): {e}")

    # MIGRATION: Add friends table
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS friends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                friend_id INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, friend_id),
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(friend_id) REFERENCES users(id)
            )
        ''')
        conn.commit()
        print("Migrated DB: Added friends table")
    except Exception as e:
        print(f"Migration Error (Friends table): {e}")

    # FORUM: Add tables
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS forum_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS forum_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                image_url TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(category_id) REFERENCES forum_categories(id),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS forum_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(post_id) REFERENCES forum_posts(id),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        
        # Initialize categories if they don't exist
        categories = [
            ("General", "General discussion about Morpheme."),
            ("Tips, Tricks, and Strategies", "Share your best gameplay advice."),
            ("Screenshots", "Show off your high scores and cool boards."),
            ("Introduce Yourself", "New here? Say hello!")
        ]
        for name, desc in categories:
            conn.execute('INSERT OR IGNORE INTO forum_categories (name, description) VALUES (?, ?)', (name, desc))
        
        conn.commit()
        print("Migrated DB: Added Forum tables and categories")
    except Exception as e:
        print(f"Migration Error (Forum tables): {e}")

    conn.close()

init_db()

# Configuration for Uploads
UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads/avatars')
FORUM_UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads/forum')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FORUM_UPLOAD_FOLDER'] = FORUM_UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB Limit

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FORUM_UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ... (rest of file) ...

# Settings Endpoints
@app.route('/api/settings/update', methods=['POST'])
def update_setting():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    key = data.get('key')
    value = data.get('value')
    
    # GUESTS: Do not save to DB (Settings are session-only)
    if session.get('is_guest'):
        return jsonify({'success': True})
    
    if not key or value is None:
        return jsonify({'error': 'Missing key or value'}), 400
        
    conn = sqlite3.connect('morpheme.db')
    try:
        conn.execute('''
            INSERT INTO user_settings (user_id, setting_key, setting_value)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, setting_key) 
            DO UPDATE SET setting_value=excluded.setting_value
        ''', (session['user_id'], key, str(value)))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/settings', methods=['GET'])
def get_settings():
    if 'user_id' not in session:
        return jsonify({'settings': {}}) # Return empty for guests/unauthed
    
    conn = sqlite3.connect('morpheme.db')
    cursor = conn.execute('SELECT setting_key, setting_value FROM user_settings WHERE user_id = ?', (session['user_id'],))
    rows = cursor.fetchall()
    conn.close()
    
    settings = {row[0]: row[1] for row in rows}
    return jsonify({'settings': settings})

@app.route('/api/stats/user_count', methods=['GET'])
def get_user_count():
    conn = sqlite3.connect('morpheme.db')
    try:
        # Count all users who are NOT guests (guests have usernames starting with Guest_)
        # AND check against guests created in guest_login if needed, but existing guest filter by name is good enough strictly speaking
        # Actually safer to check if password_hash is not the dummy one? 
        # But for 'signed up', we usually mean registered.
        # Guest users are in the users table but we can identify them by 'Guest_' prefix if we stick to that convention
        cursor = conn.execute("SELECT COUNT(*) FROM users WHERE username NOT LIKE 'Guest_%'") 
        count = cursor.fetchone()[0]
        online_count = room_manager.get_online_count() if 'room_manager' in globals() else 0
        return jsonify({
            'count': count,
            'online_count': online_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


# Serve static files
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

# Authentication endpoints
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    # Username validation: 1-16 chars, letters, numbers, underscores
    import re
    if not re.match(r'^[a-zA-Z0-9_]{1,16}$', username):
        return jsonify({'error': 'Username must be 1-16 characters (letters, numbers, underscores only)'}), 400

    if len(password) < 6:
        return jsonify({'error': 'Password must be 6+ characters'}), 400
    
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
        session.pop('is_guest', None) # Clear guest flag if present
        
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
    session.pop('is_guest', None) # Clear guest flag if present
    
    return jsonify({'success': True, 'username': username})

@app.route('/api/logout', methods=['POST'])
def logout():
    if 'user_id' in session:
        # USER REQUEST: When logging out, remove them from ANY room entirely (including 24h)
        cleanup_user_rooms_entirely(session['user_id'])
        room_manager.remove_presence(session['user_id'])
    session.clear()
    return jsonify({'success': True})

@app.route('/api/presence/leave', methods=['POST'])
def presence_leave():
    """Beacon endpoint for browser close"""
    if 'user_id' in session:
        room_manager.remove_presence(session['user_id'])
    return '', 204

@app.route('/api/guest-login', methods=['POST'])
def guest_login():
    import random
    import string
    # Generate unique guest username
    guest_id = random.randint(10000, 99999)
    guest_username = f'Guest_{guest_id}'
    
    # Create DB entry for guest so PMs work (they need a real ID in the users table)
    # Give them a random password hash that they'll never know/need
    dummy_password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    password_hash = generate_password_hash(dummy_password, method='pbkdf2:sha256')
    
    conn = sqlite3.connect('morpheme.db')
    try:
        cursor = conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                             (guest_username, password_hash))
        new_user_id = cursor.lastrowid
        conn.commit()
        
        session['user_id'] = new_user_id
        session['username'] = guest_username
        session['is_guest'] = True
        
        return jsonify({'success': True, 'username': guest_username})
    except Exception as e:
        return jsonify({'error': f'Guest login failed: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/api/session', methods=['GET'])
def get_session():
    if 'user_id' in session:
        room_manager.update_presence(session['user_id'])
        return jsonify({
            'authenticated': True,
            'username': session['username'],
            'is_guest': session.get('is_guest', False)
        })
    return jsonify({'authenticated': False})

@app.route('/api/profile/update_flag', methods=['POST'])
def update_flag():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    if session.get('is_guest'):
        return jsonify({'error': 'Guest accounts cannot update profile'}), 403
        
    data = request.get_json()
    flag = data.get('flag')
    
    if not flag:
        return jsonify({'error': 'No flag provided'}), 400
        
    try:
        conn = sqlite3.connect('morpheme.db')
        conn.execute('UPDATE users SET country_flag = ? WHERE id = ?', (flag, session['user_id']))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Flag update error: {e}")
        return jsonify({'error': 'Update failed'}), 500

@app.route('/api/profile/update', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    if session.get('is_guest'):
        return jsonify({'error': 'Guest accounts cannot update profile'}), 403
        
    data = request.get_json()
    fields = ['full_name', 'age', 'gender', 'location', 'quote', 'description', 'proof_url']
    updates = {k: v for k, v in data.items() if k in fields}
    
    if not updates:
        return jsonify({'error': 'No valid fields provided'}), 400
        
    try:
        conn = sqlite3.connect('morpheme.db')
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values())
        values.append(session['user_id'])
        
        conn.execute(f'UPDATE users SET {set_clause} WHERE id = ?', values)
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Profile update error: {e}")
        return jsonify({'error': 'Update failed'}), 500
        
@app.route('/api/profile/<username>', methods=['GET'])
def get_public_profile(username):
    if 'user_id' in session:
        room_manager.update_presence(session['user_id'])
    conn = sqlite3.connect('morpheme.db')
    cursor = conn.execute('''
        SELECT id, username, rating, games_played, avatar_url, country_flag, 
               full_name, age, gender, location, quote, description, proof_url, wins,
               max_pe, avg_pe, pe_count
        FROM users WHERE username = ? COLLATE NOCASE
    ''', (username,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    # Get config-specific ratings
    cursor = conn.execute('SELECT config_key, rating FROM user_ratings WHERE user_id = ?', (user[0],))
    config_ratings = {row[0]: row[1] for row in cursor.fetchall()}

    # Get recent rounds (last 50) with optional period filtering
    from flask import request
    period = request.args.get('period', 'all').lower()
    
    time_filter = ""
    if period == 'day':
        time_filter = "AND timestamp >= datetime('now', '-1 day')"
    elif period == 'week':
        time_filter = "AND timestamp >= datetime('now', '-7 days')"
    elif period == 'month':
        time_filter = "AND timestamp >= datetime('now', '-30 days')"
    elif period == 'year':
        time_filter = "AND timestamp >= datetime('now', '-365 days')"

    cursor_all = conn.execute(f'''
        SELECT room_id, game_type, round_number, board_json, words_json, total_score, 
               round_start_time, round_duration, timestamp, user_rating, performance_ratio, id
        FROM round_history
        WHERE user_id = ? {time_filter}
        ORDER BY timestamp DESC
    ''', (user[0],))
    all_rows = cursor_all.fetchall()
    
    # Filter and Deduplicate BEFORE processing (for performance and correctness)
    # 1. Deduplicate by (room_id, round_number) - Keep most recent (already sorted by DESC)
    # 2. Filter out rounds where user "did not play" (0 words found)
    seen_rounds = set()
    clean_rows = []
    
    for r in all_rows:
        room_id = r[0]
        round_val = r[2]
        wjson = r[4]
        
        # Dedup Check
        round_key = (room_id, round_val)
        if round_key in seen_rounds:
            continue
            
        # "Did not play" check (Empty words list)
        try:
            # If wjson is exactly '[]', skip
            if wjson == '[]':
                continue
            # Double check with load if unsure
            w_list = json.loads(wjson)
            if not w_list:
                continue
        except:
            continue
            
        seen_rounds.add(round_key)
        clean_rows.append(r)

    # helper to process a row into a rich dict
    def process_round_row(row, db_conn):
        room_id, gtype, rnum, bjson, wjson, score, rstart, rdur, ts, urat, pe_ratio, g_id = row
        
        # Get all players in this specific round
        c_room = db_conn.execute('''
            SELECT rh.total_score, rh.user_rating, u.username
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE rh.room_id = ? AND rh.round_number = ? AND rh.timestamp = ?
        ''', (room_id, rnum, ts))
        r_entries = c_room.fetchall()
        
        # Use stored pe_ratio (multiplied by 100 for percentage scale)
        perf_val = int(pe_ratio * 100) if pe_ratio else 100

        words = json.loads(wjson)
        num_words = len(words)
        top_word = max(words, key=lambda x: x.get('points', 0))['word'] if words else "-"
        avg_len = round(sum(len(w['word']) for w in words) / num_words, 1) if num_words > 0 else 0
        room_strength = sum(e[1] for e in r_entries) if r_entries else urat
        
        board = json.loads(bjson)
        dims = f"{len(board)}x{len(board[0])}" if board else "4x4"

        return {
            'game_id': g_id,
            'room_id': room_id,
            'game_type': gtype,
            'round_number': rnum,
            'board': board,
            'dimensions': dims,
            'words': words,
            'num_words': num_words,
            'top_word': top_word,
            'avg_len': avg_len,
            'total_score': score,
            'performance_value': perf_val,
            'room_strength': room_strength,
            'round_start_time': rstart,
            'round_duration': rdur,
            'timestamp': ts,
            'all_players': sorted([{'username': e[2], 'score': e[0], 'rating': e[1]} for e in r_entries], key=lambda x: x['score'], reverse=True)
        }

    # Process all rows once to get data for both averages and exceptional rounds
    processed_all = [process_round_row(r, conn) for r in clean_rows]
    
    # Calculate Averages and Config Stats
    config_stats = {}
    for cfg_key, rating in config_ratings.items():
        try:
            gtype, dims, dur = cfg_key.split('|')
            matching = [p for p in processed_all if p['game_type'] == gtype and p['dimensions'] == dims and p['round_duration'] == int(dur)]
            
            config_stats[cfg_key] = {
                'rating': rating,
                'avg_score': round(sum(p['total_score'] for p in matching) / len(matching), 1) if matching else 0,
                'avg_perf': round(sum(p['performance_value'] for p in matching) / len(matching), 1) if matching else 0
            }
        except Exception as e:
            print(f"Error processing config {cfg_key}: {e}")
            config_stats[cfg_key] = {'rating': rating, 'avg_score': 0, 'avg_perf': 0}

    # Recent rounds (last 10)
    recent_rounds = sorted(processed_all, key=lambda x: x['timestamp'], reverse=True)[:10]

    # Calculate exceptional rounds: any round where performance exceeds the personal average for that config
    exceptional_rounds = []
    for processed in processed_all:
        config_key = f"{processed['game_type']}|{processed['dimensions']}|{processed['round_duration']}"
        avg_p = config_stats.get(config_key, {}).get('avg_perf', 0)
        if processed['performance_value'] > avg_p and processed['performance_value'] > 0:
            exceptional_rounds.append(processed)
            
    # Sort by best performance value first
    exceptional_rounds = sorted(exceptional_rounds, key=lambda x: x['performance_value'], reverse=True)[:50]

    conn.close()
    
    # Get online status and current room info
    session_info = room_manager.find_user_session(user[0])
        
    return jsonify({
        'username': user[1],
        'rating': user[2],
        'games_played': user[3],
        'avatar_url': user[4] if user[4] else None,
        'country_flag': user[5] if user[5] else '🏳️',
        'full_name': user[6] if user[6] else '-',
        'age': user[7] if user[7] else '-',
        'gender': user[8] if user[8] else '-',
        'location': user[9] if user[9] else '-',
        'quote': user[10] if user[10] else 'Enter a personal quote',
        'description': user[11] if user[11] else 'Add a detailed description about yourself...',
        'proof_url': user[12] if user[12] else None,
        'wins': user[13] if user[13] else 0,
        'max_pe': round(user[14], 2) if user[14] else 0.0,
        'avg_pe': round(user[15], 2) if user[15] else 0.0,
        'recent_rounds': recent_rounds,
        'exceptional_rounds': exceptional_rounds,
        'config_ratings': config_stats,
        'status': {
            'is_online': session_info['is_online'] if session_info else False,
            'current_room': session_info['room_id'] if session_info else None,
            'session': session_info
        }
    })

@app.route('/api/profile/<username>/achievements/<game_type>/<board_dimensions>/<int:time_limit>', methods=['GET'])
def get_room_achievements(username, game_type, board_dimensions, time_limit):
    """Fetch personal achievements and stats for a specific user and room configuration"""
    import json
    from flask import request
    period = request.args.get('period', 'all').lower()
    
    conn = sqlite3.connect('morpheme.db')
    cursor = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (username,))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404
    
    user_id = user[0]
    config_key = f"{game_type}|{board_dimensions}|{time_limit}"
    
    # 1. Get All Matching Rounds (for Global Stats)
    all_query = '''
        SELECT words_json, total_score, timestamp, room_id, round_number, board_json, id, user_rating
        FROM round_history
        WHERE user_id = ? AND game_type = ? AND round_duration = ?
        ORDER BY timestamp DESC
    '''
    cursor_all = conn.execute(all_query, (user_id, game_type, time_limit))
    all_rows = cursor_all.fetchall()
    
    global_matching = []
    for row in all_rows:
        try:
            board = json.loads(row[5])
            if f"{len(board)}x{len(board[0]) if board else 0}" == board_dimensions:
                global_matching.append(row)
        except: continue
    
    # Calculate Global Best (All-Time)
    global_stats = {
        "high_score": 0, "max_words": 0, "longest_word": "", 
        "best_word": {"word": "", "points": 0},
        "games_played": len(global_matching), "total_score": 0, "total_words": 0, "wins": 0
    }
    
    for row in global_matching:
        words = json.loads(row[0])
        score = row[1]
        global_stats["total_score"] += score
        global_stats["total_words"] += len(words)
        if score > global_stats["high_score"]: global_stats["high_score"] = score
        if len(words) > global_stats["max_words"]: global_stats["max_words"] = len(words)
        for w in words:
            if len(w['word']) > len(global_stats["longest_word"]): global_stats["longest_word"] = w['word']
            if w.get('points', 0) > global_stats["best_word"]["points"]:
                global_stats["best_word"] = {"word": w['word'], "points": w.get('points',0)}

    # 2. Filter by Period for the lists
    time_filter = ""
    if period == 'day': time_filter = "AND timestamp >= datetime('now', '-1 day')"
    elif period == 'week': time_filter = "AND timestamp >= datetime('now', '-7 days')"
    elif period == 'month': time_filter = "AND timestamp >= datetime('now', '-30 days')"
    elif period == 'year': time_filter = "AND timestamp >= datetime('now', '-365 days')"
        
    query = f'''
        SELECT words_json, total_score, timestamp, room_id, round_number, board_json, id, user_rating
        FROM round_history
        WHERE user_id = ? AND game_type = ? AND round_duration = ? {time_filter}
        ORDER BY timestamp DESC
    '''
    cursor = conn.execute(query, (user_id, game_type, time_limit))
    period_rows = cursor.fetchall()
    
    period_matching = []
    for row in period_rows:
        try:
            board = json.loads(row[5])
            if f"{len(board)}x{len(board[0]) if board else 0}" == board_dimensions:
                period_matching.append(row)
        except: continue

    if not period_matching and period != 'all':
        conn.close()
        return jsonify({'username': username, 'rating': 1200, 'global_stats': global_stats, 'stats': None})

    # Calculations for Period
    performance_list = []
    all_period_words = []
    period_wins = 0
    total_period_score = 0
    total_period_words = 0
    
    for row in period_matching:
        words = json.loads(row[0])
        my_score = row[1]
        ts = row[2]
        r_id = row[3]
        r_num = row[4]
        g_id = row[6]
        
        total_period_score += my_score
        total_period_words += len(words)
        for w in words:
            w.update({'timestamp': ts, 'room_id': r_id, 'round_number': r_num, 'game_id': g_id})
            all_period_words.append(w)

        # Context fetch
        cursor_room = conn.execute('''
            SELECT rh.total_score, rh.user_rating, u.username
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE rh.room_id = ? AND rh.round_number = ? AND rh.timestamp = ?
        ''', (r_id, r_num, ts))
        room_entries = cursor_room.fetchall()
        
        max_s = max(e[0] for e in room_entries) if room_entries else 0
        is_win = (my_score == max_s and max_s > 0)
        if is_win: period_wins += 1

        avg_room_score = sum(e[0] for e in room_entries) / len(room_entries) if room_entries else 0
        ratio = round(my_score / avg_room_score, 2) if avg_room_score > 0 else 1.0
        
        perf_val = int(ratio * 100) # Simple metric for UI

        processed = {
            'game_id': g_id, 'room_id': r_id, 'round_number': r_num, 'timestamp': ts,
            'total_score': my_score, 'num_words': len(words), 'is_win': is_win,
            'ratio': ratio, 'performance_value': perf_val,
            'top_word': max(words, key=lambda x: x.get('points', 0))['word'] if words else "-",
            'all_players': room_entries
        }
        performance_list.append(processed)

    # SORTING BY IMPRESSIVENESS (Ratio DESC)
    # Exceptional: by Ratio DESC, then Score DESC
    exceptional = sorted([r for r in performance_list if r['ratio'] >= 1.5], key=lambda x: (float(x['ratio']), int(x['total_score'])), reverse=True)[:30]
    
    # Winning: by Ratio DESC (Best wins first)
    winning = sorted([r for r in performance_list if r['is_win']], key=lambda x: (float(x['ratio']), int(x['total_score'])), reverse=True)[:30]
    
    # Best Scores: Score DESC, then Ratio DESC
    best_scores = sorted(performance_list, key=lambda x: (int(x['total_score']), float(x['ratio'])), reverse=True)[:30]
    
    # Best Word Counts: Count DESC, then Ratio DESC
    best_counts = sorted(performance_list, key=lambda x: (int(x['num_words']), float(x['ratio'])), reverse=True)[:30]
    
    # Games Played: Best Ratio DESC
    recent = sorted(performance_list, key=lambda x: (float(x['ratio']), int(x['total_score'])), reverse=True)[:30] 
    
    # Best Words: Points DESC
    best_words = sorted(all_period_words, key=lambda x: int(x.get('points', 0)), reverse=True)[:30]

    # Get config rating
    cursor = conn.execute('SELECT rating FROM user_ratings WHERE user_id = ? AND config_key = ?', (user_id, config_key))
    rating_row = cursor.fetchone()
    rating = rating_row[0] if rating_row else 1200

    conn.close()
    
    return jsonify({
        'username': username,
        'rating': rating,
        'global_stats': global_stats,
        'stats': {
            'total_score': total_period_score,
            'total_words': total_period_words,
            'games_played': len(period_matching),
            'wins': period_wins,
            'win_rate': round((period_wins / len(period_matching))*100, 1) if period_matching else 0,
            'avg_score': round(total_period_score / len(period_matching)) if period_matching else 0,
            'avg_words': round(total_period_words / len(period_matching), 1) if period_matching else 0,
            'avg_perf': round(sum(r['ratio'] for r in performance_list)/len(performance_list), 2) if performance_list else 1.0,
            'avg_word_pts': round(total_period_score / total_period_words, 1) if total_period_words > 0 else 0,
            
            'exceptional_rounds': exceptional,
            'winning_rounds': winning,
            'best_scores': best_scores,
            'best_word_counts': best_counts,
            'recent_rounds': recent,
            'best_words': best_words
        }
    })


@app.route('/api/profile/upload_avatar', methods=['POST'])
def upload_avatar():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    if session.get('is_guest'):
        return jsonify({'error': 'Guest accounts cannot upload avatars'}), 403
        
    if 'avatar' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['avatar']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        try:
            from werkzeug.utils import secure_filename
            import time
            
            # Generate safe filename: user_id_timestamp.ext
            ext = file.filename.rsplit('.', 1)[1].lower()
            new_filename = f"user_{session['user_id']}_{int(time.time())}.{ext}"
            
            # Save file
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
            
            # Construct URL
            avatar_url = f"/static/uploads/avatars/{new_filename}"
            
            # Update DB
            conn = sqlite3.connect('morpheme.db')
            # First, get old avatar to delete if exists? (Optional cleanup)
            # For now just update
            conn.execute('UPDATE users SET avatar_url = ? WHERE id = ?', (avatar_url, session['user_id']))
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'avatar_url': avatar_url})
            
        except Exception as e:
            print(f"Upload error: {e}")
            return jsonify({'error': 'Upload failed'}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

# Game Room APIs
from game_room import room_manager
from word_validator import word_validator
import uuid

def apply_leave_penalty(user_id, room):
    """Apply -16 rating penalty if user leaves a non-24h room with score > 0"""
    if room.time_limit >= 7200:
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
        if room.time_limit >= 7200:
            continue
            
        # Apply leave penalty if applicable
        apply_leave_penalty(user_id, room)

        # Remove from players (Standard removal, respects 24h persistence)
        room.remove_player(user_id)

def cleanup_user_rooms_entirely(user_id):
    """FORCED removal from ALL rooms (including 24h) - used for Logout"""
    for rid in list(room_manager.rooms.keys()):
        room = room_manager.rooms[rid]
        
        # Apply leave penalty if applicable (non-24h only)
        apply_leave_penalty(user_id, room)

        # Force removal from players (Bypasses 24h persistence)
        room.remove_player(user_id, force=True)

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
    room = room_manager.create_room(generated_id, game_type, int(time_limit), board_dimensions, int(min_rating), int(max_rating))
    
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

    # Get extra stats (games_played, country_flag)
    games_played = 0
    country_flag = '🏳️'
    if not session.get('is_guest', False):
        conn = sqlite3.connect('morpheme.db')
        try:
             cur = conn.execute('SELECT games_played, country_flag FROM users WHERE id = ?', (session['user_id'],))
             row = cur.fetchone()
             if row:
                 games_played = row[0]
                 if row[1]: country_flag = row[1]
        except: pass
        conn.close()
    
    room.add_player(session['user_id'], session['username'], rating, games_played=games_played, country_flag=country_flag)
    
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

    # Get extra stats (games_played, country_flag)
    games_played = 0
    country_flag = '🏳️'
    has_exceptional = False # User request: Do not give trophy on join
    
    if not session.get('is_guest', False):
        conn = sqlite3.connect('morpheme.db')
        try:
             # Basic Stats
             cur = conn.execute('SELECT games_played, country_flag FROM users WHERE id = ?', (user_id,))
             row = cur.fetchone()
             if row:
                 games_played = row[0]
                 if row[1]: country_flag = row[1]
        except Exception as e:
            print(f"Error checking stats: {e}")
        finally:
            conn.close()
    
    # Try to join as player
    manual_accessed = session.pop('manual_accessed', False)
    # Pass has_exceptional_round
    success = room.add_player(user_id, session['username'], rating, games_played=games_played, country_flag=country_flag, manual_accessed=manual_accessed)
    if success:
        p = room.players[-1] # Valid since we just added or updated
        p.has_exceptional_round = has_exceptional 
    if not success:
        # Room full
        msg = f"Room is full (Max {room.max_players} players). You can watch instead."
        if room.game_type == 'accumulative':
             msg = "Could not join Accumulative room. Please try again."
        return jsonify({'error': msg}), 409
    
    room.update_player_activity(user_id)
    return jsonify({'success': True, 'role': 'player', 'max_players': room.max_players, 'joined_mid_round': manual_accessed})

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
        if len(room.players) == 0 and room.time_limit < 240:
            room_manager.delete_room(room_id)
    
    return jsonify({'success': True})

@app.route('/api/rooms', methods=['GET'])
def list_rooms():
    if 'user_id' in session:
        room_manager.update_presence(session['user_id'])
    game_type = request.args.get('game_type')
    board_dimensions = request.args.get('board_dimensions')
    time_limit = request.args.get('time_limit', type=int)
    
    # Clean up rooms before listing (ensures zombie rooms are removed)
    room_manager.cleanup_rooms(timeout=420, spec_timeout=1800)
    
    active_rooms = []
    
    for room_id, room in room_manager.rooms.items():
        if (room.game_type == game_type and 
            room.board_dimensions == board_dimensions and 
            room.time_limit == time_limit):
            
            # Calculate combined rating (avg or sum?)
            combined_rating = sum(p.rating for p in room.players)
            
            active_rooms.append({
                'room_id': room.room_id,
                'player_count': len(room.players),
                'max_players': room.max_players,
                'min_rating': room.min_rating,
                'max_rating': room.max_rating,
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
    room_manager.cleanup_rooms(timeout=420, spec_timeout=1800)
    
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
    if 'user_id' in session:
        room_manager.update_presence(session['user_id'])
    print(f"\n=== GET STATE REQUEST for room {room_id} ===")
    room = room_manager.get_room(room_id)
    if not room:
        print(f"ERROR: Room {room_id} not found")
        return jsonify({'error': 'Room not found'}), 404

    try:
        print(f"Room found - game_type: {room.game_type}, current_round: {room.current_round}, state: {room.state}")

        
        # Check for inactive players (zombies)
        # Use 7 minutes (420s) globally for players, 30m (1800s) for spectators
        timeout = 420
        spec_timeout = 1800
        
        players_removed = room.check_inactivity(timeout=timeout, spec_timeout=spec_timeout)
        
        # Immediate cleanup: If room is now empty and not a 24h room, delete it
        if len(room.players) == 0 and len(room.spectators) == 0 and room.time_limit < 7200:
            print(f"[app.py] Room {room_id} is empty after cleanup. Deleting immediately.")
            room_manager.delete_room(room_id)
            return jsonify({'error': 'Room closed due to inactivity'}), 404

        # Check and update state based on timers
        prev_state = room.state
        state_changed = room.check_and_update_state()

        
        # If just transitioned to intermission, start complete solving in background
        if state_changed and room.state == 'intermission' and prev_state == 'active':
            print(f"Transitioned to intermission, using fast solve words immediately.")
            
            # SAVE ROUND HISTORY
            room_manager.save_round_history(room)
            
            # PERSISTENCE: Save ratings, games_played, and wins immediately at round end
            if room.stats_recorded_round < room.current_round:
                playing_count = sum(1 for p in room.players if p.score > 0)
                if playing_count > 1:
                    print(f"[Persistence] Saving stats for Round {room.current_round} ({playing_count} players)...")
                    try:
                        registered_scores = [p.score for p in room.players if p.user_id > 0]
                        max_score = max(registered_scores) if registered_scores else 0
                        conn = sqlite3.connect('morpheme.db')
                        config_key = f"{room.game_type}|{room.board_dimensions}|{room.time_limit}"
                        
                        for p in room.players:
                            # Skip guests and mid-round joiners for persistence
                            if p.user_id > 0 and not getattr(p, 'joined_mid_round', False):
                                # Update Rating in DB
                                conn.execute('''
                                    INSERT OR REPLACE INTO user_ratings (user_id, config_key, rating)
                                    VALUES (?, ?, ?)
                                ''', (p.user_id, config_key, p.rating))
                                
                                # Update Games Played & Wins
                                if p.score > 0:
                                    if p.games_played is None: p.games_played = 0
                                    p.games_played += 1
                                    conn.execute('UPDATE users SET games_played = games_played + 1 WHERE id = ?', (p.user_id,))
                                    
                                    if p.score == max_score and max_score > 0:
                                        conn.execute('UPDATE users SET wins = wins + 1 WHERE id = ?', (p.user_id,))
                                    
                                    # Update Skill Rankings (PE Stats)
                                    if hasattr(p, 'performance_efficiency') and p.performance_efficiency > 0:
                                        conn.execute('''
                                            UPDATE users 
                                            SET max_pe = MAX(max_pe, ?),
                                                avg_pe = (avg_pe * pe_count + ?) / (pe_count + 1),
                                                pe_count = pe_count + 1
                                            WHERE id = ?
                                        ''', (p.performance_efficiency, p.performance_efficiency, p.user_id))
                        
                        conn.commit()
                        conn.close()
                        room.stats_recorded_round = room.current_round
                    except Exception as e:
                        print(f"[Persistence] ERROR saving stats: {e}")

            # Ensure solving_complete is True so frontend proceeds
            room.solving_complete = True
            if not room.complete_words and room.all_words:
                 room.complete_words = list(room.all_words)
        
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
                
                # Reset flags and state for new round
                room_manager.start_next_round(room_id)

        
        # Determine which word list to return
        # During intermission: use complete_words if available and solving is done, otherwise all_words
        # During active: use all_words (fast initial list for validation)
        # IMPORTANT: We filter by min_word_length here so that validation-only words (length 2)
        # don't appear as "Missed Words" in the UI.
        # FIX: Use room.current_min_length instead of spinner_params to avoid leak when next round params generated
        min_len = getattr(room, 'current_min_length', room.spinner_params.get('min_word_length', 3))
        
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
            'max_rating': room.max_rating,
            'previous_all_words': room.previous_all_words,
            'previous_day_history': room.previous_day_history,
            'fcfs_found_words': list(room.fcfs_found_words) if hasattr(room, 'fcfs_found_words') else [],
            'your_username': session.get('username'),
            'previous_day_history': room.previous_day_history,
            'players': [
                {
                    'user_id': p.user_id,
                    'username': p.username,
                    'rating': p.rating,
                    'words_count': len(p.submitted_words),
                    'debug_trace': print(f"STATE: {p.username} has {[w['word'] for w in p.submitted_words]}") if p.submitted_words else None,
                    'score': p.score,
                    'rating_change': p.rating_change,
                    'found_bonus_word': p.found_bonus_word,
                    'submitted_words': p.submitted_words,
                    'previous_submitted_words': p.previous_submitted_words,
                    'invalid_words': p.invalid_words,
                    'invalid_words': p.invalid_words,
                    'input_method': p.input_method,
                    'last_active_age': time.time() - p.last_active,
                    'games_played': p.games_played,
                    'country_flag': p.country_flag,
                    'joined_mid_round': getattr(p, 'joined_mid_round', False),
                    'has_exceptional_round': getattr(p, 'has_exceptional_round', False),
                    'performance_efficiency': getattr(p, 'performance_efficiency', 0.0)
                } for p in sorted(room.players, key=lambda p: p.score, reverse=True)
            ],
            'spectators': [
                {'username': s.username, 'rating': s.rating, 'user_id': s.user_id} for s in room.spectators
            ] if hasattr(room, 'spectators') else [],
            'chat_messages': room.chat_messages,
            'winners_history': room.winners_history
        })
        
        return resp

    except Exception as e:
        import traceback
        error_msg = f"ERROR in get_room_state: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/user/current-room')
def get_user_current_room():
    if 'user_id' not in session:
        return jsonify({'room_id': None})
    
    room_manager.update_presence(session['user_id'])
    session_info = room_manager.find_user_session(session['user_id'])
    
    if session_info and session_info.get('room_id'):
        return jsonify({
            'room_id': session_info['room_id'],
            'game_type': session_info.get('game_type'),
            'board_dimensions': session_info.get('board_dimensions'),
            'time_limit': session_info.get('time_limit')
        })
    
    return jsonify({'room_id': None})

@app.route('/api/room/<room_id>/chat', methods=['POST'])
def submit_chat_message(room_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    room = room_manager.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
        
    data = request.get_json()
    message = data.get('message', '').strip()
    image = data.get('image')
    
    if not message and not image:
        return jsonify({'error': 'Message or image required'}), 400
        
    # Optional: Truncate long messages
    if len(message) > 200:
        message = message[:200]
        
    # Optional: Basic validation on image size (length of base64 string) if needed
    # base64 factor is ~1.33. 1MB image is ~1.33MB string. Limit to ~2MB string.
    if image and len(image) > 2 * 1024 * 1024:
        return jsonify({'error': 'Image too large (max 1.5MB)'}), 400
        
    room.add_chat_message(session['username'], message, image=image)
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
    
    player = room.get_player(user_id)
    new_score = player.score if player else 0

    return jsonify({
        'success': success, 
        'message': message,
        'points': points,
        'word': final_word,
        'new_score': new_score
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
            # User requested Simple Summation (e.g. A+E = 190+278)
            return sum(freq.get(c, 0) for c in word)

        # We take NWL as the base for Likelihood
        likelihood_eligible = []
        for w in nwl_set_full:
            if target_len is not None and len(w) != target_len: continue
            if start_char is not None and not w.startswith(start_char): continue
            likelihood_eligible.append(w)
            
        # Sort by Likelihood Score (DESC), then Alpha (ASC)
        # We REMOVE the intermediate alphabetic re-sort to preserve Likelihood ranking
        likelihood_eligible.sort(key=lambda x: (-calculate_likelihood(x), x))
        
        response_data = {
            'nwl': filter_iterable(nwl_set_full),
            'csw': filter_iterable(csw_set_full),
            'csw_only': filter_iterable(csw_only_full),
            'likelihood': likelihood_eligible[:5000], # Top 5000 Most Likely
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

@app.route('/api/tools/subanagrams', methods=['POST'])
def tools_subanagrams():
    data = request.json
    input_text = data.get('input', '').upper().strip()
    dict_name = data.get('dictionary', 'NWL')
    
    if not input_text:
        return jsonify({'error': 'No input provided'}), 400
        
    dictionary = load_tools_dictionary(dict_name)
    if not dictionary:
        return jsonify({'error': f'Dictionary {dict_name} not found'}), 404
        
    from collections import Counter
    input_counter = Counter(input_text)
    
    results = []
    for word in dictionary:
        if len(word) > len(input_text):
            continue
            
        word_counter = Counter(word)
        is_subanagram = True
        for char, count in word_counter.items():
            if input_counter[char] < count:
                is_subanagram = False
                break
        
        if is_subanagram:
            results.append(word)
            
    # Sort by length (descending) then alphabetically
    results.sort(key=lambda x: (-len(x), x))
    
    return jsonify({
        'results': results,
        'count': len(results)
    })

@app.route('/api/tools/validate', methods=['POST'])
def tools_validate_word():
    data = request.json
    word = data.get('word', '').upper().strip()
    dict_name = data.get('dictionary', 'NWL')
    
    if not word:
        return jsonify({'error': 'No word provided'}), 400
        
    dictionary = load_tools_dictionary(dict_name)
    if not dictionary:
        return jsonify({'error': f'Dictionary {dict_name} not found'}), 404
        
    is_valid = word in dictionary
    
    return jsonify({
        'word': word,
        'is_valid': is_valid
    })

# --- Private Messaging Routes ---

@app.route('/api/pm/send', methods=['POST'])
def send_private_message():
    if 'username' not in session:
        return jsonify({'error': 'Login required'}), 401
    
    data = request.json
    target_username = data.get('recipient')
    message_text = data.get('message', '').strip()
    
    if not target_username or not message_text:
        return jsonify({'error': 'Recipient and message required'}), 400
    if len(message_text) > 500:
        message_text = message_text[:500]
        
    conn = sqlite3.connect('morpheme.db')
    try:
        # Get IDs (Case-insensitive)
        sender = conn.execute('SELECT id, username FROM users WHERE username = ? COLLATE NOCASE', (session['username'],)).fetchone()
        
        # If sender is a guest and not in DB, auto-create (Legacy session support)
        if not sender and session['username'].startswith('Guest_'):
            import random, string
            dummy_password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            password_hash = generate_password_hash(dummy_password, method='pbkdf2:sha256')
            cursor = conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (session['username'], password_hash))
            sender = (cursor.lastrowid, session['username'])
            conn.commit()

        receiver = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (target_username,)).fetchone()
        
        # If receiver is a guest and not in DB, we can't easily auto-create without more info,
        # but usually guests who receive PMs have already logged in and should have been created.
        # Fallback: if receiver is Guest_ and not found, they likely aren't online/existent anymore.
        
        if not sender or not receiver:
            return jsonify({'error': 'User not found'}), 404
            
        conn.execute('''
            INSERT INTO private_messages (sender_id, receiver_id, sender_username, message)
            VALUES (?, ?, ?, ?)
        ''', (sender[0], receiver[0], sender[1], message_text))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/pm/conversation/<target_username>', methods=['GET'])
def get_conversation(target_username):
    if 'username' not in session:
        return jsonify({'error': 'Login required'}), 401
        
    conn = sqlite3.connect('morpheme.db')
    try:
        # Get IDs (Case-insensitive)
        me = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (session['username'],)).fetchone()
        if not me and session['username'].startswith('Guest_'):
            import random, string
            dummy_password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            password_hash = generate_password_hash(dummy_password, method='pbkdf2:sha256')
            cursor = conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (session['username'], password_hash))
            me = (cursor.lastrowid,)
            conn.commit()

        them = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (target_username,)).fetchone()
        
        if not me or not them:
            return jsonify({'error': 'User not found'}), 404
            
        # Get messages in both directions
        messages = conn.execute('''
            SELECT sender_username, message, timestamp, is_read, sender_id
            FROM private_messages
            WHERE (sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?)
            ORDER BY timestamp ASC
        ''', (me[0], them[0], them[0], me[0])).fetchall()
        
        # Mark as read
        conn.execute('''
            UPDATE private_messages SET is_read = 1
            WHERE sender_id = ? AND receiver_id = ? AND is_read = 0
        ''', (them[0], me[0]))
        conn.commit()
        
        result = []
        for msg in messages:
            result.append({
                'sender': msg[0],
                'message': msg[1],
                'timestamp': msg[2],
                'is_read': msg[3],
                'is_me': msg[4] == me[0]
            })
        return jsonify({'messages': result})
    finally:
        conn.close()

@app.route('/api/pm/clear/<target_username>', methods=['POST'])
def clear_private_messages(target_username):
    if 'username' not in session:
        return jsonify({'error': 'Login required'}), 401
    
    conn = sqlite3.connect('morpheme.db')
    try:
        me = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (session['username'],)).fetchone()
        them = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (target_username,)).fetchone()
        
        if not me or not them:
            return jsonify({'error': 'User not found'}), 404
            
        conn.execute('''
            DELETE FROM private_messages 
            WHERE (sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?)
        ''', (me[0], them[0], them[0], me[0]))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/pm/unread_count', methods=['GET'])
def get_unread_count():
    if 'username' not in session:
        return jsonify({'count': 0})
        
    conn = sqlite3.connect('morpheme.db')
    try:
        user = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (session['username'],)).fetchone()
        if not user: return jsonify({'count': 0})
        
        count = conn.execute('SELECT COUNT(*) FROM private_messages WHERE receiver_id = ? AND is_read = 0', (user[0],)).fetchone()[0]
        
        # Also return who sent them, ordered by the latest message first
        senders_rows = conn.execute('''
            SELECT sender_username, MAX(timestamp) as latest
            FROM private_messages 
            WHERE receiver_id = ? AND is_read = 0 
            GROUP BY sender_username 
            ORDER BY latest ASC
        ''', (user[0],)).fetchall()
        
        latest_timestamp = conn.execute('SELECT MAX(timestamp) FROM private_messages WHERE receiver_id = ? AND is_read = 0', (user[0],)).fetchone()[0]
        
        return jsonify({
            'count': count, 
            'senders': [s[0] for s in senders_rows],
            'latest_timestamp': latest_timestamp
        })
    finally:
        conn.close()

@app.route('/api/tools/flag_manual', methods=['POST'])
def tools_flag_manual():
    """Flags the current session as having accessed the Manual tool."""
    session['manual_accessed'] = True
    return jsonify({'success': True})

@app.route('/api/tools/manual_solve', methods=['POST'])
def tools_manual_solve():
    """Solves a custom board provided by the user."""
    # Set flag just in case they skipped the frontend click trigger
    session['manual_accessed'] = True
    
    data = request.json
    board = data.get('board') # 2D list of letters
    dictionary = data.get('dictionary', 'NWL')
    
    if not board or not isinstance(board, list):
        return jsonify({'error': 'No board provided or invalid format'}), 400
        
    try:
        # We use the board_generator from the global room_manager instance
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
    
    # Get definition
    global DEFINITIONS_CACHE
    if DEFINITIONS_CACHE is None:
        load_definitions()
    definition = DEFINITIONS_CACHE.get(random_word, "No definition available for this word.")
    
    return jsonify({
        'word': random_word,
        'definition': definition
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
        
    seed_hash = int(hashlib.md5(today_str.encode()).hexdigest(), 16)
    idx = seed_hash % len(eligible_words)
    wotd = eligible_words[idx]
    
    # Get definition
    global DEFINITIONS_CACHE
    if DEFINITIONS_CACHE is None:
        load_definitions()
    definition = DEFINITIONS_CACHE.get(wotd, "No definition available for this word.")
    
    return jsonify({
        'word': wotd,
        'date': today_str,
        'definition': definition
    })


# --- Friends Management Routes ---

@app.route('/api/friends/add', methods=['POST'])
def add_friend():
    if 'username' not in session:
        return jsonify({'error': 'Login required'}), 401
    
    data = request.json
    friend_username = data.get('username')
    
    if not friend_username:
        return jsonify({'error': 'Username required'}), 400
    if friend_username == session['username']:
        return jsonify({'error': 'Cannot add yourself as a friend'}), 400
        
    conn = sqlite3.connect('morpheme.db')
    try:
        user = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (session['username'],)).fetchone()
        friend = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (friend_username,)).fetchone()
        
        if not friend:
            return jsonify({'error': 'User not found'}), 404
            
        # Check if already friends
        existing = conn.execute('SELECT 1 FROM friends WHERE user_id = ? AND friend_id = ?', 
                               (user[0], friend[0])).fetchone()
        if existing:
            return jsonify({'success': True, 'msg': 'Already friends'})
            
        conn.execute('INSERT INTO friends (user_id, friend_id) VALUES (?, ?)', (user[0], friend[0]))
        # Also add the reverse for mutual friendship? 
        # User said "populates the list of all the friends a user has"
        # Usually friendship is mutual. Let's make it mutual for simplicity in this app context.
        conn.execute('INSERT OR IGNORE INTO friends (user_id, friend_id) VALUES (?, ?)', (friend[0], user[0]))
        
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/friends/remove', methods=['POST'])
def remove_friend():
    if 'username' not in session:
        return jsonify({'error': 'Login required'}), 401
    
    data = request.json
    friend_username = data.get('username')
    
    conn = sqlite3.connect('morpheme.db')
    try:
        user = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (session['username'],)).fetchone()
        friend = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (friend_username,)).fetchone()
        
        if user and friend:
            conn.execute('DELETE FROM friends WHERE (user_id = ? AND friend_id = ?) OR (user_id = ? AND friend_id = ?)', 
                           (user[0], friend[0], friend[0], user[0]))
            conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/friends/status/<username>', methods=['GET'])
def get_friend_status(username):
    if 'username' not in session:
        return jsonify({'is_friend': False})
        
    conn = sqlite3.connect('morpheme.db')
    try:
        user = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (session['username'],)).fetchone()
        friend = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (username,)).fetchone()
        
        if not user or not friend:
            return jsonify({'is_friend': False})
            
        # Check both directions just in case of asymmetric legacy data
        existing = conn.execute('SELECT 1 FROM friends WHERE (user_id = ? AND friend_id = ?) OR (user_id = ? AND friend_id = ?)', 
                               (user[0], friend[0], friend[0], user[0])).fetchone()
        return jsonify({'is_friend': existing is not None})
    finally:
        conn.close()

@app.route('/api/friends/list', methods=['GET'])
def get_friends_list():
    if 'username' not in session:
        return jsonify({'error': 'Login required'}), 401
        
    conn = sqlite3.connect('morpheme.db')
    conn.row_factory = sqlite3.Row
    try:
        user = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (session['username'],)).fetchone()
        # Join with users to get usernames and avatars
        friends_rows = conn.execute('''
            SELECT u.id, u.username, u.avatar_url, u.rating, u.country_flag
            FROM friends f
            JOIN users u ON f.friend_id = u.id
            WHERE f.user_id = ?
        ''', (user['id'],)).fetchall()
        
        friends_data = []
        for row in friends_rows:
            f_dict = dict(row)
            # Get online status
            presence = room_manager.find_user_session(row['id'])
            f_dict['is_online'] = presence['is_online'] if presence else False
            friends_data.append(f_dict)
            
        # Sort by online (True first) then username
        friends_data.sort(key=lambda x: (not x['is_online'], x['username'].lower()))
        
        return jsonify({'friends': friends_data})
    finally:
        conn.close()

# --- FORUM ENDPOINTS ---

@app.route('/api/forum/categories', methods=['GET'])
def get_forum_categories():
    conn = sqlite3.connect('morpheme.db')
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute('SELECT * FROM forum_categories').fetchall()
        return jsonify({'categories': [dict(row) for row in rows]})
    finally:
        conn.close()

@app.route('/api/forum/posts/<int:category_id>', methods=['GET'])
def get_forum_posts(category_id):
    conn = sqlite3.connect('morpheme.db')
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute('''
            SELECT p.*, u.username, u.avatar_url, u.country_flag,
            (SELECT COUNT(*) FROM forum_comments WHERE post_id = p.id) as comment_count
            FROM forum_posts p
            JOIN users u ON p.user_id = u.id
            WHERE p.category_id = ?
            ORDER BY p.timestamp DESC
        ''', (category_id,)).fetchall()
        return jsonify({'posts': [dict(row) for row in rows]})
    finally:
        conn.close()

@app.route('/api/forum/post/<int:post_id>', methods=['GET'])
def get_forum_post_detail(post_id):
    conn = sqlite3.connect('morpheme.db')
    conn.row_factory = sqlite3.Row
    try:
        post = conn.execute('''
            SELECT p.*, u.username, u.avatar_url, u.country_flag
            FROM forum_posts p
            JOIN users u ON p.user_id = u.id
            WHERE p.id = ?
        ''', (post_id,)).fetchone()
        
        if not post:
            return jsonify({'error': 'Post not found'}), 404
            
        comments = conn.execute('''
            SELECT c.*, u.username, u.avatar_url, u.country_flag
            FROM forum_comments c
            JOIN users u ON c.user_id = u.id
            WHERE c.post_id = ?
            ORDER BY c.timestamp ASC
        ''', (post_id,)).fetchall()
        
        return jsonify({
            'post': dict(post),
            'comments': [dict(c) for c in comments]
        })
    finally:
        conn.close()

@app.route('/api/forum/posts', methods=['POST'])
def create_forum_post():
    if 'user_id' not in session or session.get('is_guest'):
        return jsonify({'error': 'Forum access is restricted to registered members only. Please sign up or login to participate!'}), 403
        
    data = request.form
    category_id = data.get('category_id')
    title = data.get('title')
    content = data.get('content')
    
    if not category_id or not title or not content:
        return jsonify({'error': 'Missing fields'}), 400
        
    image_url = None
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename != '' and allowed_file(file.filename):
            import uuid
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{uuid.uuid4()}.{ext}"
            file.save(os.path.join(app.config['FORUM_UPLOAD_FOLDER'], filename))
            image_url = f"/static/uploads/forum/{filename}"
            
    conn = sqlite3.connect('morpheme.db')
    try:
        cursor = conn.execute('''
            INSERT INTO forum_posts (category_id, user_id, title, content, image_url)
            VALUES (?, ?, ?, ?, ?)
        ''', (category_id, session['user_id'], title, content, image_url))
        conn.commit()
        return jsonify({'success': True, 'post_id': cursor.lastrowid})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/forum/comments', methods=['POST'])
def create_forum_comment():
    if 'user_id' not in session or session.get('is_guest'):
        return jsonify({'error': 'Registered users only'}), 403
        
    data = request.get_json()
    post_id = data.get('post_id')
    content = data.get('content')
    
    if not post_id or not content:
        return jsonify({'error': 'Missing fields'}), 400
        
    conn = sqlite3.connect('morpheme.db')
    try:
        conn.execute('''
            INSERT INTO forum_comments (post_id, user_id, content)
            VALUES (?, ?, ?)
        ''', (post_id, session['user_id'], content))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard_data():
    conn = sqlite3.connect('morpheme.db')
    conn.row_factory = sqlite3.Row
    try:
        # Params
        period = request.args.get('period', 'day')
        game_type = request.args.get('game_type', 'all')
        dims = request.args.get('board_dimensions', 'all') 
        time_limit = request.args.get('time_limit', 'all')

        # Base filters
        params = []
        # Exclude 24h rooms (duration is usually 86400, so < 43200 (12h) is safe)
        where_clauses = ["rh.round_duration < 43200"] 

        if game_type != 'all':
            where_clauses.append("rh.game_type = ?")
            params.append(game_type)
        if dims != 'all':
             where_clauses.append("rh.board_dimensions = ?")
             params.append(dims)
        if time_limit != 'all':
             where_clauses.append("rh.round_duration = ?")
             params.append(time_limit)

        # Time Filter
        if period == 'day':
             where_clauses.append("rh.timestamp >= datetime('now', '-1 day', 'localtime')")
        elif period == 'week':
             where_clauses.append("rh.timestamp >= datetime('now', '-7 days', 'localtime')")
        elif period == 'month':
             where_clauses.append("rh.timestamp >= datetime('now', '-30 days', 'localtime')")
        elif period == 'year':
             where_clauses.append("rh.timestamp >= datetime('now', '-365 days', 'localtime')")
             
        base_where = " AND ".join(where_clauses)
        
        # 1. Best Scores (Highest total score in a round)
        scores = conn.execute(f"""
            SELECT rh.total_score, rh.user_rating, u.username, u.country_flag, u.avatar_url, rh.room_id, rh.round_number, rh.timestamp, rh.board_json, rh.words_json, rh.round_duration, rh.id
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE {base_where}
            ORDER BY rh.total_score DESC
            LIMIT 50
        """, params).fetchall()
        
        # 2. Best Words (Highest point single word)
        words = conn.execute(f"""
            SELECT rh.best_word, rh.best_word_score, u.username, u.country_flag, u.avatar_url, rh.room_id, rh.round_number, rh.timestamp, rh.board_json, rh.words_json, rh.round_duration, rh.id
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE {base_where} AND rh.best_word IS NOT NULL
            ORDER BY rh.best_word_score DESC
            LIMIT 50
        """, params).fetchall()
        
        # 3. Best PE (Highest Performance Efficiency)
        pes = conn.execute(f"""
            SELECT rh.performance_ratio, rh.total_score, u.username, u.country_flag, u.avatar_url, rh.room_id, rh.round_number, rh.timestamp, rh.board_json, rh.words_json, rh.round_duration, rh.id
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE {base_where} AND rh.performance_ratio > 0
            ORDER BY rh.performance_ratio DESC
            LIMIT 50
        """, params).fetchall()
        
        # 4. Best Ratings Achieved (Max achieved in period - One per user)
        # Note: We group by user_id to get one entry per user
        ratings = conn.execute(f"""
            SELECT MAX(rh.user_rating) as max_rating, u.username, u.country_flag, u.avatar_url, rh.room_id, rh.timestamp
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE {base_where}
            GROUP BY u.id
            ORDER BY max_rating DESC
            LIMIT 50
        """, params).fetchall()
        
        # 5. Avg Score (Avg per user, Min 3 games)
        avgs = conn.execute(f"""
            SELECT AVG(rh.total_score) as avg_score, COUNT(*) as games, MAX(rh.timestamp) as last_active, u.username, u.country_flag, u.avatar_url
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE {base_where}
            GROUP BY u.id
            HAVING games >= 1
            ORDER BY avg_score DESC
            LIMIT 50
        """, params).fetchall()

        # 6. Most Games Played (Activity Leaderboard)
        most_games = conn.execute(f"""
            SELECT COUNT(*) as game_count, MAX(rh.timestamp) as last_active, u.username, u.country_flag, u.avatar_url, u.rating
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE {base_where}
            GROUP BY u.id
            ORDER BY game_count DESC
            LIMIT 50
        """, params).fetchall()
        
        # 7. Current Ratings (Users active in period, sorted by CURRENT rating)
        current_ratings = conn.execute(f"""
            SELECT u.username, u.rating, u.country_flag, u.avatar_url, MAX(rh.timestamp) as last_active
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE {base_where}
            GROUP BY u.id
            ORDER BY u.rating DESC
            LIMIT 1000
        """, params).fetchall()
        
        # Helper to dict
        def to_list(rows):
            return [dict(r) for r in rows]

        return jsonify({
            'best_scores': to_list(scores),
            'best_words': to_list(words),
            'best_pes': to_list(pes),
            'best_ratings': to_list(ratings),
            'avg_scores': to_list(avgs),
            'current_ratings': to_list(current_ratings),
            'most_games': to_list(most_games)
        })

    except Exception as e:
        import traceback
        print(f"Leaderboard Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()



if __name__ == '__main__':
    print('Morpheme server running on http://localhost:3000')
    app.run(host='0.0.0.0', port=3000, debug=False, use_reloader=False)
