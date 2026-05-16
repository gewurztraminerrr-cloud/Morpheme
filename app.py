print(f"[Main] SERVER STARTING - VERSION: 2026-05-05_04:00 (Tally Hardened)")

from flask import Flask, request, jsonify, session, send_from_directory, g, redirect, url_for, render_template
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import time
import os
import json
import random
import threading
import uuid
from collections import Counter
import datetime

app = Flask(__name__, static_folder='static')
app.secret_key = 'morpheme-secret-key-2024'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=30)

# --- DONATION ROUTES REMOVED ---

@app.route('/api/ping')
def ping_debug():
    return jsonify({'pong': True})

from tournament_logic import tournament_manager
from private_match_logic import private_match_manager
from word_validator import word_validator
from scoring import calculate_word_score
from game_room import room_manager, STATS_PATH
import fcntl

# MODERATOR SYSTEM
MODS_FILE = os.path.join(os.path.dirname(__file__), 'dictionaries', 'mods.txt')
ADDED_WORDS_FILE = os.path.join(os.path.dirname(__file__), 'dictionaries', 'added_words.txt')


# GLOBAL WORD TALLY CONTROLLER
# Absolute path ensures consistency across Gunicorn/Flask environments
STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dictionaries', 'word_stats.json')
TRACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dictionaries', 'stats_trace.log')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'morpheme.db')
RATING_AUDIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rating_audit.log')
DEBUG_FLOW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_flow.log')

def _update_word_stats(word, action="add"):
    """
    Centralized thread-safe and process-safe stats updater.
    action: 'add' (initialize at 0) or 'remove' (delete key)
    """
    try:
        with open(TRACE_PATH, 'a') as trace:
            trace.write(f"[{datetime.datetime.now()}] {action.upper()}_START: '{word}'\n")
            
            if not os.path.exists(STATS_PATH):
                trace.write(f"[{datetime.datetime.now()}] WARN: Creating missing stats file\n")
                with open(STATS_PATH, 'w') as f: json.dump({}, f)
            
            # Using 'r+' requires the file to exist.
            stats_file = open(STATS_PATH, 'r+')
            fcntl.flock(stats_file, fcntl.LOCK_EX)
            
            try:
                global_stats = json.load(stats_file)
            except:
                global_stats = {}
            
            changed = False
            if action == "add":
                if word not in global_stats:
                    global_stats[word] = 0
                    changed = True
                    trace.write(f"[{datetime.datetime.now()}] SUCCESS: Initialized '{word}'\n")
            elif action == "remove":
                if word in global_stats:
                    del global_stats[word]
                    changed = True
                    trace.write(f"[{datetime.datetime.now()}] SUCCESS: Removed '{word}'\n")
            
            if changed:
                stats_file.seek(0)
                stats_file.truncate()
                json.dump(global_stats, stats_file)
                stats_file.flush()
                os.fsync(stats_file.fileno())
            
            fcntl.flock(stats_file, fcntl.LOCK_UN)
            stats_file.close()
    except Exception as e:
        with open(TRACE_PATH, 'a') as trace:
            trace.write(f"[{datetime.datetime.now()}] FATAL_ERROR: {e}\n")
        print(f"[StatsSync] Fatal synchronization error: {e}")

_MODS_CACHE = None
_MODS_CACHE_TIME = 0

def get_moderators():
    global _MODS_CACHE, _MODS_CACHE_TIME
    if _MODS_CACHE is not None and time.time() - _MODS_CACHE_TIME < 60:
        return _MODS_CACHE

    mods = {'jeffbabiak', 'system'}
    if os.path.exists(MODS_FILE):
        try:
            with open(MODS_FILE, 'r') as f:
                lines = [line.strip().lower() for line in f if line.strip()]
                mods.update(lines)
        except Exception as e:
            print(f"[Mods] Error reading {MODS_FILE}: {e}")
    
    _MODS_CACHE = mods
    _MODS_CACHE_TIME = time.time()
    return mods


def save_moderator(username):
    mods = get_moderators()
    mods.add(username.strip().lower())
    try:
        with open(MODS_FILE, 'w') as f:
            for mod in sorted(mods):
                f.write(f"{mod}\n")
        return True
    except Exception as e:
        print(f"[Mods] Error saving to {MODS_FILE}: {e}")
        return False

def remove_moderator(username):
    username = username.strip().lower()
    if username == 'jeffbabiak':
        print("[Mods] Attempt to remove protected moderator jeffbabiak blocked.")
        return False
    mods = get_moderators()
    if username in mods:
        mods.remove(username)
        try:
            with open(MODS_FILE, 'w') as f:
                for mod in sorted(mods):
                    f.write(f"{mod}\n")
            return True
        except Exception as e:
            print(f"[Mods] Error removing from {MODS_FILE}: {e}")
            return False
    return False


def is_mod(username):
    if not username: return False
    res = username.lower() in get_moderators()
    print(f"[Mods] Checking if {username} is mod: {res}")
    return res


# Auth Helpers
class User:
    def __init__(self, id, username):
        self.id = id
        self.username = username
        self.is_authenticated = True
        self.is_mod = is_mod(username)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required'}), 401
            return redirect('/')
        return f(*args, **kwargs)
    return decorated_function

def mod_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or not is_mod(session.get('username')):
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Moderator access required'}), 403
            return redirect('/')
        return f(*args, **kwargs)
    return decorated_function

@app.before_request
def enforce_one_month_session():
    # Skip static files, login/register, presence beacon, and captcha
    if request.path.startswith('/static') or request.path in ['/api/login', '/api/register', '/api/presence/leave', '/api/captcha', '/api/guest-login']:
        return
        
    if 'user_id' in session:
        now = time.time()
        # Retrieve login time, defaulting to now for older sessions to migrate smoothly
        login_time = session.get('_morpheme_login_time')
        if login_time is None:
            session['_morpheme_login_time'] = now
            login_time = now
            
        # 30 days in seconds = 30 * 24 * 3600 = 2,592,000 seconds (1 month)
        if (now - login_time) > 2592000:
            print(f"[Auth] 1-month session expired for user {session.get('username')} (logged in {int(now - login_time)}s ago)")
            from game_room import room_manager
            room_manager.remove_presence(session['user_id'])
            session.clear()
            if request.path.startswith('/api/'):
                 return jsonify({'error': 'Session expired. Please log in again.'}), 401
            return redirect('/')

@app.before_request
def load_user():
    if 'user_id' in session:
        g.user = User(session['user_id'], session['username'])
    else:
        g.user = None

@app.before_request
def check_mod_status():
    if 'username' in session:
        m_status = is_mod(session['username'])
        if session.get('is_mod') != m_status:
            session['is_mod'] = m_status

# Auth Helpers

@app.route('/api/mods/status')
def get_mod_status():
    if 'username' not in session:
        return jsonify({'is_mod': False})
    return jsonify({'is_mod': is_mod(session['username'])})

@app.route('/api/mods/list', methods=['GET'])
@login_required
def list_mods():
    if not is_mod(session['username']):
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify({'mods': sorted(list(get_moderators()))})

@app.route('/api/mods/add', methods=['POST'])
@login_required
def add_mod():
    # USER REQUEST: "any user that jeffy allow to be a mod ... gets added"
    # This implies jeffy (and existing mods) can add others.
    if not is_mod(session['username']):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    new_mod = data.get('username')
    if not new_mod:
        return jsonify({'error': 'Username required'}), 400
    
    if save_moderator(new_mod):
        print(f"[Mods] User {session['username']} added {new_mod} as moderator")
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to save mod'}), 500

@app.route('/api/mods/remove', methods=['POST'])
@login_required
def delete_mod():
    if not is_mod(session['username']):
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Do not allow removing jeffy if you ARE jeffy?
    # Or maybe allow anything if you are jeffy.
    data = request.json
    target = data.get('username')
    if not target:
        return jsonify({'error': 'Username required'}), 400
    
    if remove_moderator(target):
        print(f"[Mods] User {session['username']} removed {target} from moderators")
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to remove mod'}), 500

@app.route('/api/pronunciations/add', methods=['POST'])
@login_required
def add_pronunciation():
    if not is_mod(session['username']):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    word = data.get('word', '').strip().upper()
    pronunciation = data.get('pronunciation', '').strip().upper()
    
    if not word or not pronunciation:
        return jsonify({'error': 'Word and pronunciation required'}), 400
    
    # Check if word is valid in NWL, CSW, or Added Words
    if not (word_validator.is_valid_word(word, 'CSW') or word_validator.is_valid_word(word, 'NWL')):
        return jsonify({'error': f'"{word}" is not a valid word in NWL, CSW, or Added Words.'}), 400
        
    pron_path = os.path.join(os.path.dirname(__file__), 'dictionaries', 'pronunciations.txt')
    
    try:
        # Update Cache
        global PRONUNCIATIONS_CACHE
        if PRONUNCIATIONS_CACHE is None:
            load_pronunciations()
        PRONUNCIATIONS_CACHE[word] = pronunciation
        
        # Check if already exists in mapping or file to avoid duplicates.
        # Simple append for now.
        with open(pron_path, 'a') as f:
            f.write(f"{word} - {pronunciation}\n")
        
        print(f"[Mods] {session['username']} added pronunciation for {word}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error adding pronunciation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pronunciations/remove', methods=['POST'])
@login_required
def remove_pronunciation():
    if not is_mod(session['username']):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    word = data.get('word', '').strip().upper()
    
    if not word:
        return jsonify({'error': 'Word required'}), 400
        
    pron_path = os.path.join(os.path.dirname(__file__), 'dictionaries', 'pronunciations.txt')
    
    try:
        # Update Cache
        global PRONUNCIATIONS_CACHE
        if PRONUNCIATIONS_CACHE is None:
            load_pronunciations()
            
        if PRONUNCIATIONS_CACHE and word in PRONUNCIATIONS_CACHE:
            del PRONUNCIATIONS_CACHE[word]
            
        # Rewrite file without that word
        if os.path.exists(pron_path):
            lines = []
            with open(pron_path, 'r') as f:
                for line in f:
                    if not line.strip().startswith(word + " - "):
                        lines.append(line)
            
            with open(pron_path, 'w') as f:
                f.writelines(lines)
                
        print(f"[Mods] {session['username']} removed pronunciation for {word}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error removing pronunciation: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/added_words/list', methods=['GET'])
def list_added_words_api():
    if not os.path.exists(ADDED_WORDS_FILE):
        return jsonify({'words': []})
    try:
        with open(ADDED_WORDS_FILE, 'r') as f:
            # User Request: Sort by date added, most recent first (last lines in file first)
            # We preserve unique words but maintain the latest appearance order.
            raw_lines = [line.strip().upper() for line in f if line.strip()]
            unique_words = []
            seen = set()
            # File is now newest-first, so we read directly
            for w in raw_lines:
                if w not in seen:
                    unique_words.append(w)
                    seen.add(w)
            return jsonify({'words': unique_words})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mods/added_words/config', methods=['GET'])
@mod_required
def get_added_words_config():
    return jsonify({
        'use_added_words': word_validator.get_use_added_words()
    })

@app.route('/api/mods/added_words/toggle', methods=['POST'])
@mod_required
def toggle_added_words():
    data = request.json
    enabled = data.get('enabled', True)
    new_state = word_validator.toggle_added_words(enabled)
    return jsonify({'success': True, 'use_added_words': new_state})

@app.route('/api/mods/added_words/add', methods=['POST'])
@mod_required
def add_added_word_api():
    word = request.json.get('word', '').strip().upper()
    if not word: return jsonify({'error': 'Word required'}), 400
    
    # Sync with Global Tally immediately (Self-healing)
    _update_word_stats(word, "add")

    # Check authoritative dictionaries
    if word_validator.is_valid_word_authoritative(word):
        dict_name = "NWL" if word in word_validator.nwl_words else "CSW" if word in word_validator.csw_words else "Long Words"
        return jsonify({'error': f"'{word}' already exists in {dict_name} dictionary."}), 400

        
    try:
        # Load existing lines to potentially remove duplicates and maintain order
        lines = []
        if os.path.exists(ADDED_WORDS_FILE):
            with open(ADDED_WORDS_FILE, 'r') as f:
                lines = [line.strip().upper() for line in f if line.strip()]
        
        # Reject if already present in Added Words list (User Request)
        # Use the authoritative WordValidator set to avoid file-system latency issues
        if word_validator.is_added_word(word):
            print(f"[Mods] REJECTED Duplicate: '{word}' is already in the Added Words set.")
            return jsonify({
                'error': f"'{word}' is already present on Added Words list.",
                'is_duplicate': True
            }), 400
        
        # Load existing lines to potentially remove duplicates and maintain order (Safety fallback)
        lines = []
        if os.path.exists(ADDED_WORDS_FILE):
            with open(ADDED_WORDS_FILE, 'r') as f:
                lines = [line.strip().upper() for line in f if line.strip()]
        
        # Final safety check against list (if cache somehow lagged)
        if word in lines:
             return jsonify({'error': f"'{word}' is already present on Added Words list."}), 400

        # Insert at the beginning (Top of the list) for NEW words
        lines.insert(0, word)
        
        # Write back full list to preserve order
        with open(ADDED_WORDS_FILE, 'w') as f:
            for l in lines:
                f.write(f"{l}\n")
                
        word_validator.reload_added_words()
        print(f"[Mods] Successfully added NEW word '{word}' to top of list.")

        return jsonify({
            'success': True, 
            'message': f'New word "{word}" added to Added Words list successfully.',
            'code_version': 'V5-STRICT-UNIQUE'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mods/added_words/remove', methods=['POST'])
def remove_added_word():
    # Only moderators can remove
    username = session.get('username')
    if not is_mod(username):
        return jsonify({'error': 'Unauthorized'}), 401
        
    data = request.json
    word = data.get('word', '').strip().upper()
    if not word:
        return jsonify({'error': 'Word is required'}), 400
        
    try:
        if not os.path.exists(ADDED_WORDS_FILE):
             return jsonify({'success': False, 'error': 'File not found'})
            
        with open(ADDED_WORDS_FILE, 'r') as f:
            lines = [line.strip().upper() for line in f if line.strip()]
        
        if word in lines:
            new_lines = [l for l in lines if l != word]
            with open(ADDED_WORDS_FILE, 'w') as f:
                for l in new_lines:
                    f.write(l + '\n')
            
            if word_validator:
                word_validator.reload_added_words()

            # Sync with Global Tally
            _update_word_stats(word, "remove")

            return jsonify({'success': True, 'message': f'Word "{word}" removed.'})
        
        return jsonify({'error': 'Word not found in the list.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mods/dictionary/submit', methods=['POST'])
def submit_dictionary_words():
    """
    User Request: If it is “newNWL.txt”, add these words to NWL.txt alphabetically 
    and remove all words in the file from the list in Added Words in Tools > Lists.
    """
    username = session.get('username')
    if not is_mod(username):
        return jsonify({'error': 'Unauthorized'}), 401

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    
    filename = file.filename
    if filename not in ['newNWL.txt', 'newCSW.txt']:
        return jsonify({'success': False, 'error': 'Invalid file name. Must be newNWL.txt or newCSW.txt'}), 400

    try:
        content = file.read().decode('utf-8')
        new_words = {line.strip().upper() for line in content.splitlines() if line.strip()}
        
        if not new_words:
            return jsonify({'success': False, 'error': 'File is empty'}), 400

        base_dir = os.path.dirname(__file__)
        dict_dir = os.path.join(base_dir, 'dictionaries')
        
        # Determine target dictionary
        target_dict_name = 'NWL.txt' if filename == 'newNWL.txt' else 'CSW.txt'
        target_path = os.path.join(dict_dir, target_dict_name)
        
        # 1. Load existing words
        existing_words = set()
        if os.path.exists(target_path):
            with open(target_path, 'r') as f:
                existing_words = {line.strip().upper() for line in f if line.strip()}
        
        # 2. Add new words alphabetically
        updated_words = sorted(list(existing_words | new_words))
        with open(target_path, 'w') as f:
            for w in updated_words:
                f.write(w + '\n')
                
        # 3. Tracking Table: "New XXX Words"
        tracking_filename = 'new_NWL.txt' if filename == 'newNWL.txt' else 'new_CSW.txt'
        tracking_path = os.path.join(dict_dir, tracking_filename)
        
        # Read existing tracked words to avoid duplicates in the "New" list
        tracked_words = []
        if os.path.exists(tracking_path):
            with open(tracking_path, 'r') as f:
                tracked_words = [line.strip().upper() for line in f if line.strip()]
        
        tracked_set = set(tracked_words)
        added_to_track = [w for w in sorted(list(new_words)) if w not in tracked_set]
        
        with open(tracking_path, 'a') as f:
            for w in added_to_track:
                f.write(w + '\n')

        # 4. Remove from Added Words (Staging Area)
        if os.path.exists(ADDED_WORDS_FILE):
            with open(ADDED_WORDS_FILE, 'r') as f:
                current_added = [line.strip().upper() for line in f if line.strip()]
            
            filtered_added = [w for w in current_added if w not in new_words]
            
            with open(ADDED_WORDS_FILE, 'w') as f:
                for w in filtered_added:
                    f.write(w + '\n')
        
        # 5. Initialize counts in word_stats.json for the new words
        for w in new_words:
            _update_word_stats(w, "add")

        # 6. Reload Word Validator
        if word_validator:
            word_validator._load_dictionaries()
            
        return jsonify({
            'success': True, 
            'added_count': len(new_words), 
            'target': target_dict_name
        })

    except Exception as e:
        print(f"[Admin] Dictionary upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/mods/definitions/add', methods=['POST'])
@login_required
def add_definition_api():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    word = data.get('word', '').strip().upper()
    definition = data.get('definition', '').strip()
    
    if not word or not definition:
        return jsonify({'error': 'Word and definition required'}), 400
        
    try:
        # Load all to memory to rewrite (needed for update/append logic)
        defs = {}
        if DEFINITIONS_PATH and os.path.exists(DEFINITIONS_PATH):
            with open(DEFINITIONS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.split(' - ', 1)
                    if len(parts) == 2:
                        defs[parts[0].strip().upper()] = parts[1].strip()
        
        # Add or Replace
        defs[word] = definition
        
        # Sort by key before writing (best practice for dictionaries)
        sorted_keys = sorted(defs.keys())
        
        # Use a temporary file to avoid corruption
        temp_path = DEFINITIONS_PATH + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            for k in sorted_keys:
                f.write(f"{k} - {defs[k]}\n")
        
        # Move back
        os.replace(temp_path, DEFINITIONS_PATH)
        
        # Flush and Reload
        global DEFINITIONS_CACHE
        DEFINITIONS_CACHE = {} # Force reload
        load_definitions()
        
        return jsonify({'success': True, 'message': f'Definition for {word} set.'})
    except Exception as e:
        print(f"Error updating definitions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mods/definitions/remove', methods=['POST'])
@login_required
def remove_definition_api():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    word = data.get('word', '').strip().upper()
    if not word:
        return jsonify({'error': 'Word required'}), 400
        
    try:
        defs = {}
        found = False
        if DEFINITIONS_PATH and os.path.exists(DEFINITIONS_PATH):
            with open(DEFINITIONS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.split(' - ', 1)
                    if len(parts) == 2:
                        k = parts[0].strip().upper()
                        if k == word:
                            found = True
                            continue
                        defs[k] = parts[1].strip()
        
        if not found:
            return jsonify({'error': 'Definition not found for this word.'}), 404
            
        sorted_keys = sorted(defs.keys())
        temp_path = DEFINITIONS_PATH + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            for k in sorted_keys:
                f.write(f"{k} - {defs[k]}\n")
        
        os.replace(temp_path, DEFINITIONS_PATH)
        
        global DEFINITIONS_CACHE
        DEFINITIONS_CACHE = {} # Force reload
        load_definitions()
        
        return jsonify({'success': True, 'message': f'Definition for {word} removed.'})
    except Exception as e:
        print(f"Error removing definition: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mods/ban_user', methods=['POST'])
@login_required
def ban_user_api():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    username = (data.get('username') or '').strip()
    
    if not username:
        return jsonify({'error': 'Username required'}), 400
        
    if username.lower() == 'jeffbabiak':
        return jsonify({'error': 'Cannot ban the ultimate authority jeffbabiak'}), 403

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # Get user ID
        cursor = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if not row:
            return jsonify({'error': 'User not found'}), 404
        user_id = row['id']
        
        # Start transaction
        conn.execute("BEGIN TRANSACTION")
        
        # Deletions (Erase all traces)
        # ID-based deletions
        conn.execute("DELETE FROM forum_comments WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM forum_posts WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM friends WHERE user_id = ? OR friend_id = ?", (user_id, user_id))
        conn.execute("DELETE FROM match_invites WHERE sender_id = ?", (user_id,))
        conn.execute("DELETE FROM private_match_players WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM private_match_starts WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM private_match_turns WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM private_matches WHERE creator_id = ?", (user_id,))
        conn.execute("DELETE FROM private_messages WHERE sender_id = ? OR receiver_id = ?", (user_id, user_id))
        conn.execute("DELETE FROM round_history WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM tournament_matchups WHERE user1_id = ? OR user2_id = ? OR winner_id = ?", (user_id, user_id, user_id))
        conn.execute("DELETE FROM tournament_participants WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM tournament_scores WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_ratings WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_settings WHERE user_id = ?", (user_id,))
        
        # Name-based string deletions
        conn.execute("DELETE FROM match_invites WHERE recipient_username = ?", (username,))
        conn.execute("DELETE FROM private_match_players WHERE username = ?", (username,))
        conn.execute("DELETE FROM private_messages WHERE sender_username = ?", (username,))

        # Finally, delete the user record
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error banning user: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()
        
    return jsonify({'success': True, 'message': f'User {username} successfully banned and all traces erased.'})


@app.route('/api/mods/lobby-notice', methods=['GET'])
def get_lobby_notice():
    notice_path = os.path.join(os.path.dirname(__file__), 'dictionaries', 'lobby_notice.txt')
    if not os.path.exists(notice_path):
        return jsonify({'notice': '', 'notice_id': 0})
    try:
        mtime = os.path.getmtime(notice_path)
        with open(notice_path, 'r', encoding='utf-8') as f:
            notice = f.read().strip()
        return jsonify({'notice': notice, 'notice_id': int(mtime)})
    except:
        return jsonify({'notice': '', 'notice_id': 0})

@app.route('/api/mods/lobby-notice/update', methods=['POST'])
@login_required
def update_lobby_notice():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    notice = data.get('notice', '').strip()
    
    notice_dir = os.path.join(os.path.dirname(__file__), 'dictionaries')
    if not os.path.exists(notice_dir):
        os.makedirs(notice_dir)
        
    notice_path = os.path.join(notice_dir, 'lobby_notice.txt')
    try:
        with open(notice_path, 'w', encoding='utf-8') as f:
            f.write(notice)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# --- AUTH ROUTES ---

DEFINITIONS_CACHE = None
PRONUNCIATIONS_CACHE = None
ADDED_WORDS_CACHE = None


# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
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

    # MIGRATION: Ensure email column exists
    try:
        conn.execute('ALTER TABLE users ADD COLUMN email TEXT')
        conn.commit()
        print("Migrated DB: Added email column to users")
    except sqlite3.OperationalError:
        pass # Column likely exists

    # MIGRATION: Fix any existing 0 ratings for registered users
    # Users who previously joined might have 0 rating due to bug
    # We update them to 1200 (Default)
    try:
        # Update user_ratings and users where rating is 0 and user is registered (user_id > 0)
        conn.execute('UPDATE user_ratings SET rating = 1200 WHERE rating = 0 AND user_id > 0')
        conn.execute('UPDATE users SET rating = 1200 WHERE rating = 0 AND id > 0')
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

    # MIGRATION: Add email verification columns
    try:
        conn.execute('ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 1')
        conn.commit()
        print("Migrated DB: Added is_verified column to users")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute('ALTER TABLE users ADD COLUMN verification_code TEXT')
        conn.commit()
        print("Migrated DB: Added verification_code column to users")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute('ALTER TABLE users ADD COLUMN verification_expires_at REAL')
        conn.commit()
        print("Migrated DB: Added verification_expires_at column to users")
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

        CREATE TABLE IF NOT EXISTS tournament_matchups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tournament_id INTEGER,
            round_number INTEGER,
            user1_id INTEGER,
            user2_id INTEGER,
            winner_id INTEGER,
            created_at REAL
        );

        CREATE TABLE IF NOT EXISTS site_config (
            config_key TEXT PRIMARY KEY,
            config_value TEXT
        );

        CREATE TABLE IF NOT EXISTS donations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            donor_name TEXT,
            amount REAL,
            is_anonymous INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending', -- pending, confirmed
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    ''')
    conn.commit()

    # MIGRATION: Add round_start_time to tournament_scores
    try:
        conn.execute('ALTER TABLE tournament_scores ADD COLUMN round_start_time REAL')
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # MIGRATION: Add skill metrics to users table
    try:
        conn.execute('ALTER TABLE users ADD COLUMN max_pe REAL DEFAULT 0.0')
        conn.execute('ALTER TABLE users ADD COLUMN avg_pe REAL DEFAULT 0.0')
        conn.execute('ALTER TABLE users ADD COLUMN pe_count INTEGER DEFAULT 0')
        conn.commit()
        print("Migrated DB: Added PE columns to users")
    except sqlite3.OperationalError:
        pass

    # MIGRATION: Add created_at to users
    try:
        conn.execute('ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP')
        conn.commit()
        print("Migrated DB: Added created_at column to users")
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
            ("Introduce Yourself", "New here? Say hello!"),
            ("News", "Official news and updates from the developers."),
            ("Suggestions", "Share your ideas for improving Morpheme."),
            ("Bugs/Errors", "Report bugs or technical issues encountered.")
        ]
        for name, desc in categories:
            conn.execute('INSERT OR IGNORE INTO forum_categories (name, description) VALUES (?, ?)', (name, desc))
        
        conn.commit()
        print("Migrated DB: Added Forum tables and categories")
    except Exception as e:
        print(f"Migration Error (Forum tables): {e}")

    # MIGRATION: Add wpm and total_words_avail to round_history
    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN wpm REAL DEFAULT 0.0')
        conn.commit()
    except Exception:
        pass
    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN total_words_avail INTEGER DEFAULT 0')
        conn.commit()
    except Exception:
        pass

    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN bonus_word TEXT')
        conn.commit()
    except Exception:
        pass

    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN bonus_cell TEXT')
        conn.commit()
    except Exception:
        pass

    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN board_format TEXT DEFAULT "Normal"')
        conn.commit()
    except Exception:
        pass

    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN all_solutions_json TEXT')
        conn.commit()
    except Exception:
        pass
    
    try:
        conn.execute('ALTER TABLE round_history ADD COLUMN all_words_paths TEXT')
        conn.commit()
    except Exception:
        pass

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
        
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.execute('SELECT setting_key, setting_value FROM user_settings WHERE user_id = ?', (session['user_id'],))
    rows = cursor.fetchall()
    conn.close()
    
    settings = {row[0]: row[1] for row in rows}
    return jsonify({'settings': settings})

@app.route('/api/stats/user_count', methods=['GET'])
def get_user_count():
    conn = sqlite3.connect(DB_PATH, timeout=30)
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


@app.route('/api/donations/recent', methods=['GET'])
def get_recent_donations():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # 1. Top Lifetime Supporters (highest summed amount DESC)
        cursor_top = conn.execute("""
            SELECT donor_name, SUM(amount) as total_amount, MAX(is_anonymous) as is_anonymous, MAX(timestamp) as timestamp 
            FROM donations 
            WHERE status = 'confirmed' 
            GROUP BY donor_name 
            ORDER BY total_amount DESC 
            LIMIT 10
        """)
        rows_top = cursor_top.fetchall()
        top_list = []
        for r in rows_top:
            top_list.append({
                'donor_name': "Anonymous" if r['is_anonymous'] else r['donor_name'],
                'amount': r['total_amount'],
                'timestamp': r['timestamp']
            })

        # 2. Recent Supporters (newest timestamp DESC)
        cursor_recent = conn.execute("""
            SELECT donor_name, amount, is_anonymous, timestamp 
            FROM donations 
            WHERE status = 'confirmed' 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        rows_recent = cursor_recent.fetchall()
        recent_list = []
        for r in rows_recent:
            recent_list.append({
                'donor_name': "Anonymous" if r['is_anonymous'] else r['donor_name'],
                'amount': r['amount'],
                'timestamp': r['timestamp']
            })

        return jsonify({
            'top': top_list,
            'recent': recent_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/')
def index():
    # USER REQUEST: once the user logs in, entering "morpheme.games" (or refreshing root)
    # should automatically take them to the lobby without having to login again.
    return render_template('index.html')


# Authentication endpoints
@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    import random
    chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    captcha_text = ''.join(random.choices(chars, k=5))
    
    session['captcha_text'] = captcha_text.upper()
    
    width = 150
    height = 50
    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background: #1a1a2e; border: 1px solid rgba(255,255,255,0.15); border-radius: 8px;">'
    ]
    
    # Noise lines
    for _ in range(6):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        color = random.choice(['#ff007f', '#00f0ff', '#ffaa00', '#00ff66', '#8800ff'])
        svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5" opacity="0.4"/>')
        
    # Noise dots
    for _ in range(40):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        r = random.uniform(1.0, 2.5)
        color = random.choice(['#ffffff', '#ff007f', '#00f0ff'])
        svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" opacity="0.3"/>')
        
    # Text
    font_families = ['sans-serif', 'monospace', 'serif']
    for i, char in enumerate(captcha_text):
        char_x = 15 + i * 26 + random.randint(-3, 3)
        char_y = 35 + random.randint(-5, 5)
        angle = random.randint(-25, 25)
        font_size = random.randint(22, 28)
        font_family = random.choice(font_families)
        font_weight = random.choice(['bold', '900'])
        color = random.choice(['#ffffff', '#00ffcc', '#ff007f', '#ffcc00', '#00f0ff'])
        
        svg_parts.append(
            f'<text x="{char_x}" y="{char_y}" '
            f'fill="{color}" font-size="{font_size}" font-family="{font_family}" font-weight="{font_weight}" '
            f'transform="rotate({angle} {char_x} {char_y})" '
            f'style="user-select: none;">{char}</text>'
        )
        
    svg_parts.append('</svg>')
    svg_data = ''.join(svg_parts)
    
    from flask import Response
    return Response(svg_data, mimetype='image/svg+xml')


def send_verification_email(user_email, username, code):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    subject = f"Your MORPHEME Verification Code: {code}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background-color: #0b0c10;
                color: #c5c6c7;
                padding: 40px 20px;
                margin: 0;
            }}
            .card {{
                max-width: 500px;
                margin: 0 auto;
                background: #1f2833;
                border: 2px solid #ff007f;
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(255, 0, 127, 0.15);
                text-align: center;
            }}
            .logo {{
                font-size: 24px;
                font-weight: 800;
                letter-spacing: 2px;
                color: #ffffff;
                margin-bottom: 20px;
            }}
            .logo span {{
                color: #ff007f;
            }}
            .title {{
                font-size: 20px;
                font-weight: 600;
                color: #ffffff;
                margin-bottom: 15px;
            }}
            .instructions {{
                font-size: 14px;
                line-height: 1.6;
                margin-bottom: 25px;
                color: #8f94a5;
            }}
            .code-box {{
                display: inline-block;
                background: rgba(0, 0, 0, 0.4);
                border: 1px dashed #ff007f;
                border-radius: 12px;
                padding: 15px 40px;
                font-size: 32px;
                font-weight: 800;
                letter-spacing: 6px;
                color: #ffffff;
                text-shadow: 0 0 10px rgba(255, 0, 127, 0.5);
                margin: 15px 0;
            }}
            .footer {{
                font-size: 11px;
                color: #555866;
                margin-top: 30px;
                line-height: 1.4;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="logo">M<span>ORPHEME</span></div>
            <div class="title">Verify Your Account</div>
            <p class="instructions">Hello <strong>{username}</strong>,<br>Welcome to Morpheme! Use the 6-digit verification code below to activate your account and start climbing the leaderboard:</p>
            <div class="code-box">{code}</div>
            <p class="instructions" style="margin-top: 25px; font-size: 12px;">This code is valid for 15 minutes. If you did not sign up for Morpheme, please ignore this email.</p>
            <div class="footer">
                &copy; 2026 Morpheme Games. All rights reserved.<br>
                This is an automated system email. Please do not reply.
            </div>
        </div>
    </body>
    </html>
    """

    # Print to logs first as fallback / reference
    print("\n" + "="*80)
    print(f" [EMAIL VERIFICATION] To: {user_email} | Username: {username}")
    print(f" [CODE]: {code}")
    print("="*80 + "\n")

    import subprocess
    import json
    import time

    data = {
        "from": "Morpheme <noreply@morpheme.games>",
        "to": [user_email],
        "subject": subject,
        "html": html_content
    }
    
    curl_command = [
        "curl", "-s", "-X", "POST", "https://api.resend.com/emails",
        "-H", "Authorization: Bearer re_JZxa2joE_5Gu6cYT9KiaDkK4YtJdnky2Q",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(data)
    ]
    
    try:
        print(f"[Email] Attempting to send via Curl subprocess...")
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        print(f"[Email] Successfully sent email via Curl: {result.stdout}")
        with open("email_error.log", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Success sending to {user_email}: {result.stdout}\n")
    except subprocess.CalledProcessError as e:
        print(f"[Email] Failed via Curl: {e.stderr}")
        with open("email_error.log", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Failed to send to {user_email} via Curl: {e.stderr}\n")
        print("[Email] Warning: Could not deliver verification email over Resend. Code printed to logs.")
    except Exception as e:
        print(f"[Email] Failed via Curl: {e}")
        with open("email_error.log", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Failed to send to {user_email} via Curl: {e}\n")
        print("[Email] Warning: Could not deliver verification email over Resend. Code printed to logs.")


@app.route('/api/send-signup-verification', methods=['POST'])
def send_signup_verification():
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    
    print(f"[Route] /api/send-signup-verification hit for '{username}' <{email}>")
    
    if not username or not email:
        return jsonify({'error': 'Username and email are required'}), 400
        
    import re
    if not re.match(r'^[a-zA-Z0-9_]{1,16}$', username):
        return jsonify({'error': 'Username must be 1-16 characters (letters, numbers, underscores only)'}), 400
        
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            return jsonify({'error': 'Username already exists'}), 400
            
        cursor = conn.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            return jsonify({'error': 'Email is already registered'}), 400
    except Exception as e:
        print(f"[SendVerificationCheckError] {e}")
        return jsonify({'error': 'Database error. Please try again.'}), 500
    finally:
        if 'conn' in locals():
            conn.close()
            
    # Generate 6-digit verification code
    import random
    code = str(random.randint(100000, 999999))
    session['signup_code'] = code
    session['signup_email'] = email
    session['signup_username'] = username
    session['signup_code_expires'] = time.time() + 900 # 15 minutes
    
    # Send verification email in a background thread
    import threading
    threading.Thread(target=send_verification_email, args=(email, username, code), daemon=True).start()
    
    return jsonify({'success': True, 'message': 'Verification code sent to your email. Please check your Junk email in 1 or 2 minutes if you do not see it.'})


@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    code = data.get('code', '').strip()
    captcha_val = data.get('captcha', '')
    
    session_captcha = session.get('captcha_text')
    session.pop('captcha_text', None) # Clear immediately to prevent replay attacks
    
    if not session_captcha or captcha_val.upper() != session_captcha:
        return jsonify({'error': 'Incorrect or expired CAPTCHA. Please click on the CAPTCHA image to refresh and try again.'}), 400
        
    # Username validation
    import re
    if not re.match(r'^[a-zA-Z0-9_]{1,16}$', username):
        return jsonify({'error': 'Username must be 1-16 characters (letters, numbers, underscores only)'}), 400

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    if len(password) < 6:
        return jsonify({'error': 'Password must be 6+ characters'}), 400
        
    # Verify the code from session
    saved_code = session.get('signup_code')
    saved_email = session.get('signup_email')
    saved_expires = session.get('signup_code_expires', 0)
    
    if not saved_code or saved_code != code:
        return jsonify({'error': 'Incorrect email verification code. Please request a new email or check the code.'}), 400
        
    if not saved_email or saved_email.lower() != email.lower():
        return jsonify({'error': 'The entered email does not match the verification code destination.'}), 400
        
    if time.time() > saved_expires:
        return jsonify({'error': 'Verification code has expired. Please send the email again.'}), 400
    
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        
        # Check if email already exists
        cursor = conn.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            return jsonify({'error': 'Email is already registered'}), 400
            
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Insert user cleanly
        conn.execute('INSERT INTO users (username, password_hash, email, is_verified) VALUES (?, ?, ?, 1)',
                    (username, password_hash, email))
        conn.commit()
        
        cursor = conn.execute('SELECT id, rating FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        # Clear the verification session data
        session.pop('signup_code', None)
        session.pop('signup_email', None)
        session.pop('signup_username', None)
        session.pop('signup_code_expires', None)
        
        # Automatically log the user in!
        session['user_id'] = user[0]
        session['username'] = username
        session['email'] = email
        session.pop('is_guest', None)
        session['_morpheme_login_time'] = time.time()
        session.permanent = True
        
        return jsonify({'success': True, 'username': username, 'email': email, 'rating': user[1]})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 400
    except Exception as e:
        print(f"[RegisterError] {e}")
        return jsonify({'error': 'Database error during registration. Please try again.'}), 500
    finally:
        if 'conn' in locals():
            conn.close()


@app.route('/api/admin/delete-user-by-email', methods=['GET'])
def delete_user_by_email():
    email = request.args.get('email')
    if not email:
        return "Email parameter is required", 400
        
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.execute('DELETE FROM users WHERE email = ?', (email,))
        conn.commit()
        count = cursor.rowcount
        conn.close()
        return f"Successfully deleted {count} accounts with email {email}"
    except Exception as e:
        return f"Error: {e}", 500


@app.route('/api/admin/logs', methods=['GET'])
def view_logs():
    import os
    try:
        lines = 100
        output = []
        
        if os.path.exists('server.log'):
            with open('server.log', 'r') as f:
                output.append("=== server.log ===\n")
                output.extend(f.readlines()[-lines:])
                
        if os.path.exists('server_debug_test.log'):
            with open('server_debug_test.log', 'r') as f:
                output.append("\n=== server_debug_test.log ===\n")
                output.extend(f.readlines()[-lines:])
                
        if os.path.exists('boggle_server_console.log'):
            with open('boggle_server_console.log', 'r') as f:
                output.append("\n=== boggle_server_console.log ===\n")
                output.extend(f.readlines()[-lines:])
                
        if os.path.exists('email_error.log'):
            with open('email_error.log', 'r') as f:
                output.append("\n=== email_error.log ===\n")
                output.extend(f.readlines()[-lines:])
                
        return "<pre>" + "".join(output) + "</pre>"
    except Exception as e:
        return f"Error: {e}", 500


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 200
            
        username = data.get('username')
        password = data.get('password')
        captcha_val = data.get('captcha', '')
        
        session_captcha = session.get('captcha_text')
        session.pop('captcha_text', None) # Clear immediately to prevent replay attacks
        
        if not session_captcha or captcha_val.upper() != session_captcha:
            return jsonify({'success': False, 'error': 'Incorrect or expired CAPTCHA. Please click on the CAPTCHA image to refresh and try again.'}), 200
            
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.execute('SELECT id, password_hash, email FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if not user or not check_password_hash(user[1], password):
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 200
        
        session['user_id'] = user[0]
        session['username'] = username
        session['email'] = user[2]
        session.pop('is_guest', None) # Clear guest flag if present
        session['_morpheme_login_time'] = time.time()
        session.permanent = True
        
        return jsonify({
            'success': True, 
            'username': username, 
            'email': user[2],
            'is_mod': is_mod(username)
        })
    except Exception as e:
        print(f"[LoginError] {e}")
        return jsonify({'success': False, 'error': f'Server error: {e}'}), 200


@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        user_id = session.get('user_id')
        if user_id:
            # USER REQUEST: When logging out, remove them from ANY room entirely (including 24h)
            try:
                cleanup_user_rooms_entirely(user_id)
            except Exception as e_rooms:
                print(f"[LogoutError] Error cleaning up rooms for user {user_id}: {e_rooms}")
                
            try:
                room_manager.remove_presence(user_id)
            except Exception as e_pres:
                print(f"[LogoutError] Error removing presence for user {user_id}: {e_pres}")
    except Exception as e:
        print(f"[LogoutError] Error during logout session retrieval: {e}")
    finally:
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
    data = request.get_json() or {}
    captcha_val = data.get('captcha', '')
    
    session_captcha = session.get('captcha_text')
    session.pop('captcha_text', None) # Clear immediately to prevent replay attacks
    
    if not session_captcha or captcha_val.upper() != session_captcha:
        return jsonify({'error': 'Incorrect or expired CAPTCHA. Please click on the CAPTCHA image to refresh and try again.'}), 400

    import random
    import string
    # Generate unique guest username
    guest_id = random.randint(10000, 99999)
    guest_username = f'Guest_{guest_id}'
    
    # Create DB entry for guest so PMs work (they need a real ID in the users table)
    # Give them a random password hash that they'll never know/need
    dummy_password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    password_hash = generate_password_hash(dummy_password, method='pbkdf2:sha256')
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        cursor = conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                             (guest_username, password_hash))
        new_user_id = cursor.lastrowid
        conn.commit()
        
        session['user_id'] = new_user_id
        session['username'] = guest_username
        session['is_guest'] = True
        session['_morpheme_login_time'] = time.time()
        session.permanent = True
        
        return jsonify({'success': True, 'username': guest_username})
    except Exception as e:
        return jsonify({'error': f'Guest login failed: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/api/session', methods=['GET'])
def get_session():
    if 'user_id' in session:
        room_manager.update_presence(session['user_id'])
        
        # Fetch fresh rating from DB
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.execute('SELECT rating FROM users WHERE id = ?', (session['user_id'],))
            rating_row = cursor.fetchone()
            rating = rating_row[0] if rating_row else 0
            conn.close()
        except Exception as e:
            print(f"[Session] Failed to fetch rating: {e}")
            rating = 0

        return jsonify({
            'authenticated': True,
            'username': session['username'],
            'email': session.get('email', ''),
            'is_guest': session.get('is_guest', False),
            'rating': rating,
            'is_mod': is_mod(session['username'])
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
        conn = sqlite3.connect(DB_PATH, timeout=30)
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
        conn = sqlite3.connect(DB_PATH, timeout=30)
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
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.execute('''
        SELECT id, username, rating, games_played, avatar_url, country_flag, 
               full_name, age, gender, location, quote, description, proof_url, wins,
               max_pe, avg_pe, pe_count, created_at
        FROM users WHERE username = ? COLLATE NOCASE
    ''', (username,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    user_id = user[0]
    period = request.args.get('period', 'all').lower()
    
    time_filter = ""
    if period == 'day':
        time_filter = "AND date(timestamp, 'localtime') = date('now', 'localtime')"
    elif period == 'week':
        time_filter = "AND date(timestamp, 'localtime') >= date('now', '-7 days', 'localtime')"
    elif period == 'month':
        time_filter = "AND date(timestamp, 'localtime') >= date('now', '-30 days', 'localtime')"
    elif period == 'year':
        time_filter = "AND date(timestamp, 'localtime') >= date('now', '-365 days', 'localtime')"

    # Calculate Period Stats (If 'all', we still calculate from round_history for consistency, 
    # but could use user table for performance if data volume is high)
    cursor_stats = conn.execute(f'''
        SELECT COUNT(DISTINCT room_id || '_' || round_number), SUM(total_score)
        FROM round_history
        WHERE user_id = ? {time_filter}
    ''', (user_id,))
    games_played_period, pt_sum_period = cursor_stats.fetchone()
    games_played_period = games_played_period or 0
    pt_sum_period = pt_sum_period or 0

    # Calculate Wins in Period (Optimized single query)
    cursor_wins = conn.execute(f'''
        SELECT COUNT(*) FROM (
            SELECT rh.room_id, rh.round_number, rh.timestamp, MAX(rh.total_score) as max_s
            FROM round_history rh
            WHERE rh.room_id IN (SELECT room_id FROM round_history WHERE user_id = ? {time_filter})
            GROUP BY rh.room_id, rh.round_number, rh.timestamp
        ) as room_winners
        JOIN round_history rh2 ON rh2.room_id = room_winners.room_id 
            AND rh2.round_number = room_winners.round_number 
            AND rh2.timestamp = room_winners.timestamp
        WHERE rh2.user_id = ? AND rh2.total_score >= room_winners.max_s AND room_winners.max_s > 0
    ''', (user_id, user_id))
    wins_period = cursor_wins.fetchone()[0] or 0

    # Get config-specific ratings (Current ratings are ALWAYS current/lifetime)
    cursor = conn.execute('SELECT config_key, rating FROM user_ratings WHERE user_id = ?', (user_id,))
    config_ratings = {row[0]: row[1] for row in cursor.fetchall()}

    # Get matching rounds for calculations
    cursor_all = conn.execute(f'''
        SELECT room_id, game_type, round_number, board_json, words_json, total_score, 
               round_start_time, round_duration, timestamp, user_rating, performance_ratio, id,
               wpm, total_words_avail, board_dimensions
        FROM round_history
        WHERE user_id = ? {time_filter}
        ORDER BY timestamp DESC, id DESC
    ''', (user_id,))
    all_rows = cursor_all.fetchall()
    
    # Filter and Deduplicate
    seen_rounds = set()
    clean_rows = []
    for r in all_rows:
        room_id, r_num, wjson = r[0], r[2], r[4]
        round_key = (room_id, r_num)
        if round_key in seen_rounds: continue
        try:
            if wjson == '[]' or not json.loads(wjson): continue
        except: continue
        seen_rounds.add(round_key)
        clean_rows.append(r)

    # helper to process a row
    def process_round_row(row, db_conn):
        room_id, gtype, rnum, bjson, wjson, score, rstart, rdur, ts, urat, pe_ratio, g_id, wpm, twa, saved_dims = row
        c_room = db_conn.execute('''
            SELECT rh.total_score, rh.user_rating, u.username
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE rh.room_id = ? AND rh.round_number = ? AND rh.timestamp = ?
        ''', (room_id, rnum, ts))
        r_entries = c_room.fetchall()
        perf_val = int(pe_ratio * 100) if pe_ratio else 100
        words = json.loads(wjson)
        num_words = len(words)
        top_word = "-"
        if num_words > 0:
            best_w_obj = max(words, key=lambda x: x.get('points', 0))
            top_word = best_w_obj.get('word', '-')
        avg_len = round(sum(len(str(w.get('word', ''))) for w in words)/num_words, 1) if num_words > 0 else 0
        room_strength = sum(e[1] for e in r_entries) if r_entries else urat
        board = json.loads(bjson)
        
        # Use saved dimensions if available, otherwise calculate fallback
        if saved_dims:
            dims = saved_dims
        elif gtype == '3d':
            dims = '3x3x3'
        else:
            dims = f"{len(board)}x{len(board[0])}" if board else "4x4"
            
        return {
            'game_id': g_id, 'room_id': room_id, 'game_type': gtype, 'round_number': rnum,
            'board': board, 'dimensions': dims, 'words': words, 'num_words': num_words,
            'top_word': top_word, 'avg_len': avg_len, 'total_score': score,
            'performance_value': perf_val, 'room_strength': room_strength,
            'round_start_time': rstart, 'round_duration': rdur, 'timestamp': ts,
            'wpm': wpm or 0, 'total_words_avail': twa or 0,
            'all_players': sorted([{'username': e[2], 'score': e[0], 'rating': e[1]} for e in r_entries], key=lambda x: x['score'], reverse=True)
        }

    processed_all = [process_round_row(r, conn) for r in clean_rows]
    
    # Config Stats (Averages for the period)
    config_stats = {}
    for cfg_key, rating in config_ratings.items():
        try:
            gtype, dims, dur = cfg_key.split('|')
            # 24-hour configurations exception: load global rating from users table
            if int(dur) >= 7200:
                rating = user[2]
            matching = [p for p in processed_all if p['game_type'] == gtype and p['dimensions'] == dims and p['round_duration'] == int(dur)]
            config_stats[cfg_key] = {
                'rating': rating,
                'avg_score': round(sum(p['total_score'] for p in matching) / len(matching), 1) if matching else 0,
                'avg_perf': round(sum(p['performance_value'] for p in matching) / len(matching), 1) if matching else 0
            }
        except:
            config_stats[cfg_key] = {'rating': rating, 'avg_score': 0, 'avg_perf': 0}

    # Period-specific AVG WPM
    wpm_games = [p['wpm'] for p in processed_all if p['total_words_avail'] >= 50 and p['wpm'] > 0]
    avg_wpm = round(sum(wpm_games) / len(wpm_games), 1) if wpm_games else 0

    # Best Score in Period
    best_score_period = max([p['total_score'] for p in processed_all]) if processed_all else 0

    # Sort recent and exceptional
    recent_rounds = sorted(processed_all, key=lambda x: x['timestamp'], reverse=True)[:50]
    exceptional_rounds = sorted([p for p in processed_all if 
                               (p['performance_value'] > config_stats.get(f"{p['game_type']}|{p['dimensions']}|{p['round_duration']}", {}).get('avg_perf', 0))
                               or (p['total_score'] >= 100)], 
                               key=lambda x: x['timestamp'], reverse=True)[:50]

    conn.close()
    session_info = room_manager.find_user_session(user[0])
        
    return jsonify({
        'username': user[1],
        'rating': user[2],
        'games_played': games_played_period, # PER-PERIOD
        'wins': wins_period,                 # PER-PERIOD
        'pt_sum': pt_sum_period,             # PER-PERIOD
        'best_score': best_score_period,     # PER-PERIOD
        'avg_wpm_300': avg_wpm,              # PER-PERIOD (already was)
        'avatar_url': user[4],
        'country_flag': user[5] or '🏳️',
        'full_name': user[6] or '-',
        'age': user[7] or '-',
        'gender': user[8] or '-',
        'location': user[9] or '-',
        'quote': user[10] or 'Enter a personal quote',
        'description': user[11] or 'Add a detailed description about yourself...',
        'proof_url': user[12],
        'max_pe': round(user[14], 2) if user[14] else 0.0,
        'avg_pe': round(user[15], 2) if user[15] else 0.0,
        'created_at': user[17],
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
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (username,))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404
    
    user_id = user[0]
    config_key = f"{game_type}|{board_dimensions}|{time_limit}"
    
    # 1. Get Matching Rounds (using the board_dimensions column)
    query_all = '''
        SELECT words_json, total_score, timestamp, room_id, round_number, board_json, id, user_rating, board_dimensions
        FROM round_history
        WHERE user_id = ? AND game_type = ? AND board_dimensions = ? AND round_duration = ?
        ORDER BY timestamp DESC
    '''
    cursor_all = conn.execute(query_all, (user_id, game_type, board_dimensions, time_limit))
    global_matching = cursor_all.fetchall()
    
    # Calculate Global Best (All-Time)
    global_stats = {
        "high_score": 0, "max_words": 0, "longest_word": "", 
        "best_word": {"word": "", "points": 0},
        "games_played": len(global_matching), "total_score": 0, "total_words": 0, "wins": 0
    }
    
    for row in global_matching:
        try:
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
        except: continue

    # 2. Filter by Period for the lists - Enforce Calendar Day logic
    time_filter = ""
    if period == 'day': time_filter = "AND date(timestamp, 'localtime') = date('now', 'localtime')"
    elif period == 'week': time_filter = "AND date(timestamp, 'localtime') >= date('now', '-7 days', 'localtime')"
    elif period == 'month': time_filter = "AND date(timestamp, 'localtime') >= date('now', '-30 days', 'localtime')"
    elif period == 'year': time_filter = "AND date(timestamp, 'localtime') >= date('now', '-365 days', 'localtime')"
        
    query = f'''
        SELECT words_json, total_score, timestamp, room_id, round_number, board_json, id, user_rating, board_dimensions
        FROM round_history
        WHERE user_id = ? AND game_type = ? AND board_dimensions = ? AND round_duration = ? {time_filter}
        ORDER BY timestamp DESC
    '''
    cursor = conn.execute(query, (user_id, game_type, board_dimensions, time_limit))
    period_rows = cursor.fetchall()
    
    period_matching = period_rows

    if not period_matching and period != 'all':
        conn.close()
        return jsonify({'username': username, 'rating': 1200, 'global_stats': global_stats, 'stats': None})

    # Calculations for Period
    performance_list = []
    all_period_words = []
    period_wins = 0
    total_period_score = 0
    total_period_words = 0
    
    seen_rounds = set()
    
    for row in period_matching:
        words = json.loads(row[0])
        my_score = row[1]
        ts = row[2]
        r_id = row[3]
        r_num = row[4]
        g_id = row[6]
        
        # User Request: Filter out 0 points or 0 words, AND Deduplicate
        if my_score <= 0 or len(words) == 0:
            continue
            
        round_key = f"{r_id}_{r_num}_{g_id}"
        if round_key in seen_rounds:
            continue
        seen_rounds.add(round_key)
        
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

        # Calculate Performance Efficiency (PE) using Rating-Based Expected Share
        # FAQ: "Expected Score... based on your current rating relative to your opponents"
        
        room_total_score = sum(e[0] for e in room_entries)
        room_total_rating = sum(e[1] for e in room_entries)
        
        # Find my rating from the room entries (to ensure we use the rating AT THAT TIME)
        my_avg_entry = next((e for e in room_entries if e[2] == username), None)
        # Fallback if username not found (shouldn't happen)
        my_rating_at_time = my_avg_entry[1] if my_avg_entry else 1200
        
        expected_score = 0
        if room_total_rating > 0:
            expected_share = my_rating_at_time / room_total_rating
            expected_score = expected_share * room_total_score
        else:
            # Fallback for unrated/guest rooms
            expected_score = room_total_score / len(room_entries) if room_entries else 0

        # Ratio = Actual / Expected
        ratio = round(my_score / expected_score, 2) if expected_score > 0 else 1.0
        
        avg_room_score = room_total_score / len(room_entries) if room_entries else 0 # Keep for data if needed
        
        perf_val = int(ratio * 100) # Simple metric for UI

        num_words = len(words) if isinstance(words, list) else 0
        top_word = "-"
        if num_words > 0:
            best_w_obj = max(words, key=lambda x: x.get('points', 0)) if words else {}
            top_word = best_w_obj.get('word', '-') if isinstance(best_w_obj, dict) else "-"
        
        avg_l = 0
        if num_words > 0:
            total_l = sum(len(str(w.get('word', ''))) for w in words if isinstance(w, dict))
            avg_l = round(total_l / num_words, 1)
        processed = {
            'game_id': g_id, 'room_id': r_id, 'round_number': r_num, 'timestamp': ts,
            'total_score': my_score, 'num_words': len(words), 'is_win': is_win,
            'avg_len': avg_l,
            'ratio': ratio, 'performance_value': perf_val,
            'top_word': top_word,
            'all_players': room_entries,
            'words': words,
            'board': json.loads(row[5])
        }
        performance_list.append(processed)

    # SORTING BY TIMESTAMP (Recency) as requested
    # Exceptional: by Timestamp DESC (Ratio >= 1.0 to include solo play/good rounds)
    exceptional = sorted([r for r in performance_list if r['ratio'] >= 1.0], key=lambda x: x['timestamp'], reverse=True)[:50]
    
    # Winning: by Timestamp DESC
    winning = sorted([r for r in performance_list if r['is_win']], key=lambda x: x['timestamp'], reverse=True)[:50]
    
    # Best Scores: Score DESC, then Timestamp DESC
    best_scores = sorted(performance_list, key=lambda x: (int(x['total_score']), x['timestamp']), reverse=True)[:50]
    
    # Best Word Counts: Count DESC, then Timestamp DESC
    best_counts = sorted(performance_list, key=lambda x: (int(x['num_words']), x['timestamp']), reverse=True)[:50]
    
    # Games Played: Timestamp DESC (True Recency)
    recent = sorted(performance_list, key=lambda x: x['timestamp'], reverse=True)[:50] 
    
    # Best Words: Points DESC (Unique words only)
    unique_words = {}
    for w in all_period_words:
        word_text = w.get('word')
        points = int(w.get('points', 0))
        if word_text not in unique_words or points > unique_words[word_text]['points']:
             unique_words[word_text] = {'word': word_text, 'points': points, 'timestamp': w.get('timestamp'), 'game_id': w.get('game_id')}
    
    unique_word_list = list(unique_words.values())
    best_words = sorted(unique_word_list, key=lambda x: x['points'], reverse=True)[:50]

    # Get config rating
    # 24-hour configurations exception: load global rating from users table
    parts = config_key.split('|')
    is_24h = (len(parts) >= 3 and int(parts[2]) >= 7200)
    if is_24h:
        cursor = conn.execute('SELECT rating FROM users WHERE id = ?', (user_id,))
        rating_row = cursor.fetchone()
        rating = rating_row[0] if rating_row else 1200
    else:
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
            conn = sqlite3.connect(DB_PATH, timeout=30)
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

def apply_leave_penalty(user_id, room):
    """Applies a -16 point penalty if a user leaves a room after participating.
    Exempts 24h persistent rooms and players with no score/words.
    """
    player = room.get_player(user_id)
    if not player or player.is_ai:
        return

    # 1. Broad Exemption: 24h Rooms (>= 2h time limit)
    if room.time_limit >= 7200:
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Penalty SKIPPED for {player.username} in {room.room_id}: 24h room\n")
        return

    # 2. Activity Check: Only penalize if they actually PARTICIPATED in this round
    # If they have 0 score and 0 words, they are a passive leaver (Ghost).
    # If the room is in INTERMISSION, the round is over, so leaving is NOT abandonment.
    if getattr(room, 'state', '') == 'intermission':
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Penalty SKIPPED for {player.username} in {room.room_id}: Intermission\n")
        return

    # 2b. Mid-Round Exemption: USER MANDATE - Do not penalize if they joined mid-round
    # We use a very strict check here to ensure NO ONE who joins late is penalized.
    is_late_joiner = getattr(player, 'joined_mid_round', False)
    if is_late_joiner:
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Penalty SKIPPED for {player.username} in {room.room_id}: Joined mid-round (Flag check: {is_late_joiner})\n")
        return

    has_score = (player.score > 0)
    has_words = (len(player.submitted_words) > 0)
    
    if not (has_score or has_words):
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Penalty SKIPPED for {player.username} in {room.room_id}: No activity (score={player.score}, words={len(player.submitted_words)})\n")
        return

    # 3. Check if others played with the user (Human participants only)
    # USER MANDATE: Only penalize if we are abandoning REGISTERED STARTER players.
    # We trigger the penalty if ANY other human starter is in the room, regardless of their score.
    other_participants = [
        p for p in room.players 
        if str(p.user_id) != str(user_id) 
        and not p.is_ai 
        and not getattr(p, 'is_guest', False)
        and not getattr(p, 'joined_mid_round', False)
    ]
    
    if not other_participants:
        num_humans = len([p for p in room.players if not p.is_ai])
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Penalty SKIPPED: No other registered human STARTERS in {room.room_id}.\n")
        return

    # Diagnostic: Log EXACTLY who is causing the penalty to trigger
    others_names = ", ".join([f"{p.username}(Starter={not getattr(p, 'joined_mid_round', False)})" for p in other_participants])
    with open(RATING_AUDIT_PATH, 'a') as log:
        log.write(f"[{time.time()}] Penalty TRIGGERED by presence of: {others_names}\n")

    # 4. Apply the -16 Penalty to the leaver
    with open(RATING_AUDIT_PATH, 'a') as log:
        log.write(f"[{time.time()}] Penalty APPLYING to {player.username} in {room.room_id}: -16 points\n")
    
    # Update DB
    display_game_type = room.game_type.replace('solo_', '')
    config_key = f"{display_game_type}|{room.board_dimensions}|{room.time_limit}"
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        # Subtract from leaver
        conn.execute('''
            INSERT INTO user_ratings (user_id, config_key, rating)
            VALUES (?, ?, MAX(400, 1200 - 16))
            ON CONFLICT(user_id, config_key) DO UPDATE SET rating = MAX(400, rating - 16)
        ''', (user_id, config_key))
        
        conn.execute('''
            UPDATE users 
            SET rating = MAX(400, rating - 16) 
            WHERE id = ?
        ''', (user_id,))
        
        # Update in-memory rating so subsequent round-end calcs use the penalized value
        player.rating = max(400, player.rating - 16)

        # 5. Add to the room's abandonment_bounty pool (To be distributed at round end results)
        # USER MANDATE: Distribute at the end when results are shown.
        room.abandonment_bounty += 16
        
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Bounty Collection: +16 added to {room.room_id} pool (Total: {room.abandonment_bounty})\n")
        
        conn.commit()
    except Exception as e:
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Penalty ERROR for {player.username}: {e}\n")
    finally:
        conn.close()

def cleanup_user_rooms(user_id, exclude_room_id=None):
    """Remove user from all rooms except exclude_room_id and 24h persistent rooms"""
    for rid in list(room_manager.rooms.keys()):
        if str(rid) == str(exclude_room_id):
            continue
        room = room_manager.rooms.get(rid)
        if not room:
            continue
        
        # PERSISTENCE RULE: Keep users in Hubs and 24h rooms even if they join another
        is_hub = str(rid).startswith('pub_')
        if room.time_limit >= 7200 or is_hub:
            continue
            
        # Apply leave penalty if applicable
        apply_leave_penalty(user_id, room)

        # Remove from players (Standard removal, respects 24h persistence)
        room.remove_player(user_id)

def cleanup_user_rooms_entirely(user_id):
    """FORCED removal from ALL rooms (skipping 24h skip for explicit Logout) - used for Logout"""
    for rid in list(room_manager.rooms.keys()):
        room = room_manager.rooms.get(rid)
        if not room:
            continue
        
        # Apply leave penalty if applicable (non-24h only, typically)
        if room.time_limit < 7200:
            apply_leave_penalty(user_id, room)

        # Force removal from players (clears past_players too so they don't auto-restore)
        room.remove_player(user_id, force=True)

@app.route('/api/room/create', methods=['POST'])
def create_room():
    with open(DEBUG_FLOW_PATH, 'a') as f:
        f.write(f"\n[app.py] create_room called at {time.time()}\n")
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    game_type = data.get('game_type')
    time_limit = data.get('time_limit')
    board_dimensions = data.get('board_dimensions')
    min_rating = data.get('min_rating', 0)
    max_rating = data.get('max_rating', 9999)
    
    # Guest Restriction: Guests cannot create custom/limited rooms
    # Guest Restriction: Guests cannot create CUSTOM rooms with limits
    if session.get('is_guest', False):
        try:
            m_rat = int(min_rating or 0)
            x_rat = int(max_rating or 9999)
            print(f"[app.py] Guest Create Attempt: game={game_type} min={m_rat} max={x_rat}")
            if m_rat > 0 or x_rat < 9999:
                 return jsonify({'error': 'RANK_REJECT: Guest users are not allowed to create rooms with rating limits. Please register to unlock this feature.'}), 403
        except (ValueError, TypeError) as e:
            print(f"[app.py] Guest Create Error in parsing: {e}")
            pass
    else:
        print(f"[app.py] Regular Create Attempt: user={session.get('username')} game={game_type} min={min_rating} max={max_rating}")
    
    # Create room
    # For all public rooms, generate STABLE IDs so history persists and users find each other
    is_public = (int(min_rating) == 0 and int(max_rating) == 9999)
    # is_long_running = (int(time_limit) >= 600) # User Request: even 45s rooms should be stable hubs
    
    if is_public:
        # Use deterministic ID: pub_v2_[game]_[dims]_[time]
        # v2: Forced reset for 6x8 compliance and rare letter lockdown.
        generated_id = f"pub_v2_{game_type}_{board_dimensions}_{time_limit}".replace(' ', '_').lower()
        print(f"[app.py] Using stable v2 ID for public room: {generated_id}")
    else:
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
        conn = sqlite3.connect(DB_PATH, timeout=30)
        # 24-hour rooms exception: load global rating from users table
        is_24h = (int(time_limit) >= 7200)
        if is_24h:
            cursor = conn.execute('SELECT rating FROM users WHERE id = ?', (session['user_id'],))
            row = cursor.fetchone()
            if row:
                rating = row[0]
            else:
                rating = 1200
        else:
            cursor = conn.execute('SELECT rating FROM user_ratings WHERE user_id = ? AND config_key = ?', 
                                (session['user_id'], config_key))
            row = cursor.fetchone()
            if row:
                rating = row[0]
            else:
                # Every room starts the user at 1200, completely unique to this room configuration
                rating = 1200
        conn.close()

    # Get extra stats (games_played, country_flag)
    games_played = 0
    country_flag = '🏳️'
    if not session.get('is_guest', False):
        conn = sqlite3.connect(DB_PATH, timeout=30)
        try:
             cur = conn.execute('SELECT games_played, country_flag FROM users WHERE id = ?', (session['user_id'],))
             row = cur.fetchone()
             if row:
                 games_played = row[0]
                 if row[1]: country_flag = row[1]
        except: pass
        conn.close()
    
    room.add_player(session['user_id'], session['username'], rating, 
                    games_played=games_played, country_flag=country_flag, 
                    is_guest=session.get('is_guest', False))
    
    # Start first round immediately in background for faster loading
    # Only if this is a BRAND NEW room (no board yet)
    if not room.board:
        print(f"[app.py] Kickstarting first round for NEW room {room_id}")
        import threading
        thread = threading.Thread(target=room_manager.start_round, args=(room_id,), daemon=True)
        thread.start()
    else:
        print(f"[app.py] Room {room_id} already has a board. Skipping redundant start_round.")
    
    return jsonify({'success': True, 'room_id': room_id})

@app.route('/api/room/<room_id>/join', methods=['POST'])
def join_room(room_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    room = room_manager.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
    
    # Get configuration-specific rating (Strip solo_ prefix to match game_room.py consolidation)
    game_type_base = room.game_type.replace('solo_', '')
    config_key = f"{game_type_base}|{room.board_dimensions}|{room.time_limit}"
    rating = 1200
    
    if session.get('is_guest', False):
        rating = 0
    else:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        # 24-hour rooms exception: load global rating from users table
        is_24h = (room.time_limit >= 7200)
        if is_24h:
            cursor = conn.execute('SELECT rating FROM users WHERE id = ?', (session['user_id'],))
            row = cursor.fetchone()
            if row:
                rating = row[0]
            else:
                rating = 1200
        else:
            cursor = conn.execute('SELECT rating FROM user_ratings WHERE user_id = ? AND config_key = ?', 
                                (session['user_id'], config_key))
            row = cursor.fetchone()
            if row:
                rating = row[0]
            else:
                # Every room starts the user at 1200, completely unique to this room configuration
                rating = 1200
        conn.close()
        
    # Ensure user is not in any other room
    cleanup_user_rooms(session['user_id'], exclude_room_id=room_id)

    # Check for spectator request
    data = request.get_json() or {}
    # Unlimited players for Accumulative, 8 for others
    if room.game_type in ['accumulative', 'solo_accumulative']:
        room.max_players = 9999
    as_spectator = data.get('as_spectator', False)

    if as_spectator:
        room.add_spectator(user_id, session['username'], rating)
        room.update_player_activity(user_id)
        return jsonify({'success': True, 'role': 'spectator'})

    # Force player mode for Accumulative
    if room.game_type in ['accumulative', 'solo_accumulative']:
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
        conn = sqlite3.connect(DB_PATH, timeout=30)
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
    success = room.add_player(user_id, session['username'], rating, 
                             games_played=games_played, country_flag=country_flag, 
                             manual_accessed=manual_accessed, is_guest=session.get('is_guest', False))
    if success:
        p = room.players[-1] # Valid since we just added or updated
        p.has_exceptional_round = has_exceptional 
    if not success:
        # Room full
        msg = f"Room is full (Max {room.max_players} players). You can watch instead."
        if room.game_type in ['accumulative', 'solo_accumulative']:
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
        
        # USER REQUEST: Immediate cleanup if no humans left
        humans = [p for p in room.players if not p.is_ai]
        is_daily = (room.time_limit >= 7200)
        is_public = room_id.startswith('pub_')
        if len(humans) == 0 and not is_daily and not is_public:
            # Kick remaining spectators if any (Double-down on game_room.py logic)
            room.is_closing = True
            room.spectators = []
                 
            print(f"[app.py] Room {room_id} has no humans left. Deleting immediately.")
            room_manager.delete_room(room_id)
    
    return jsonify({'success': True})

@app.route('/api/rooms', methods=['GET'])
def list_rooms():
    if 'user_id' in session:
        room_manager.update_presence(session['user_id'])
    game_type = request.args.get('game_type')
    board_dimensions = request.args.get('board_dimensions')
    time_limit = request.args.get('time_limit', type=int)
    
    active_rooms = []
    
    for room_id, room in room_manager.rooms.items():
        # Exclude solo and private rooms from public listing
        if room.is_solo or getattr(room, 'is_private', False):
            continue
            
        if (room.game_type == game_type and 
            room.board_dimensions == board_dimensions and 
            room.time_limit == time_limit):
            
            humans = [p for p in room.players if not getattr(p, 'is_ai', False)]
            # Never list empty rooms in the Active Rooms panel — UNLESS it's a persistent 24h room
            is_daily = (room.time_limit >= 7200)
            if len(humans) == 0 and len(room.spectators) == 0 and not is_daily:
                continue
            
            # Calculate average rating
            p_count = len(room.players)
            avg_rating = round(sum(p.rating for p in room.players) / p_count) if p_count > 0 else 0
            
            active_rooms.append({
                'room_id': room.room_id,
                'player_count': p_count,
                'max_players': room.max_players,
                'min_rating': room.min_rating,
                'max_rating': room.max_rating,
                'average_rating': avg_rating,
                'state': room.state,
                'current_round': room.current_round,
                'players': [{'username': p.username, 'rating': p.rating, 'user_id': p.user_id} for p in room.players]
            })
            
    return jsonify({'rooms': active_rooms})

@app.route('/api/lobby-stats', methods=['GET'])
def get_lobby_stats():
    """Get aggregated player counts for all game configurations"""
    stats = {}
    
    for room in room_manager.rooms.values():
        # Hide solo and private rooms from lobby stats
        if room.is_solo or getattr(room, 'is_private', False):
            continue
            
        humans = [p for p in room.players if not p.is_ai]
        # Skip empty rooms — UNLESS it's a persistent 24h room
        is_daily = (room.time_limit >= 7200)
        if len(humans) == 0 and not is_daily:
            continue
            
        # Create a unique key for this configuration
        key = f"{room.game_type}|{room.board_dimensions}|{room.time_limit}"
        
        if key not in stats:
            stats[key] = 0
        stats[key] += len(humans)
    
    return jsonify({'stats': stats})

@app.route('/api/room/<room_id>/state')
def get_room_state(room_id):
    if 'user_id' in session:
        uid = session['user_id']
        room_manager.update_presence(uid)
    
    room = room_manager.get_room(room_id)
    if room:
        print(f"[get_room_state] Room: {room_id} | State: {room.state} | PrevBonus: {getattr(room, 'previous_bonus_word', 'None')} | CurrBonus: {room.bonus_word}")
        if 'user_id' in session:
            uid = session['user_id']
            if str(uid) in getattr(room, 'evicted_users', {}):
                reason = room.evicted_users.pop(str(uid), 'inactivity')
                print(f"[get_room_state] User {uid} detected in room.evicted_users! Returning 403 eviction response.")
                return jsonify({'error': f'You have been evicted for inactivity: {reason}', 'evicted': True, 'reason': reason}), 403
    try:
        if not room:
            # User Request: STABILITY. Public hubs (pub_...) should be recreated on demand if they vanish (e.g. server restart)
            if room_id.startswith('pub_'):
                try:
                    parts = room_id.split('_')
                    if parts[1] == 'v2' and len(parts) >= 5:
                        g_type = parts[2]
                        dims = parts[3]
                        t_limit = int(parts[4])
                    elif len(parts) >= 4:
                        g_type = parts[1]
                        dims = parts[2]
                        t_limit = int(parts[3])
                    else:
                        raise ValueError("Invalid public room ID structure")
                        
                    print(f"[app.py] Reconstructing public singleton hub: {room_id} | Game: {g_type}, Dims: {dims}, Time: {t_limit}")
                    # Re-create room (create_room method handles singleton logic already)
                    room = room_manager.create_room(room_id, g_type, t_limit, dims, 0, 9999)
                    if room:
                        is_24h = (room.time_limit >= 7200)
                        if not is_24h:
                            # Start in intermission so the user has a graceful transition and doesn't jump midround!
                            with room._state_lock:
                                room.state = 'intermission'
                                room.intermission_start_time = time.time() - 45  # 15s remaining
                                room.spinner_params_generated = False
                                if hasattr(room, '_transition_spinner_launched'): delattr(room, '_transition_spinner_launched')
                                if hasattr(room, 'spinner_params_revealed'): delattr(room, 'spinner_params_revealed')
                                if hasattr(room, 'board_search_started'): delattr(room, 'board_search_started')
                                if hasattr(room, 'board_search_loading'): delattr(room, 'board_search_loading')
                                if hasattr(room, 'starting_round'): delattr(room, 'starting_round')
                        
                        # Re-add the active polling user as player right away
                        if 'user_id' in session:
                            user_id = session['user_id']
                            rating = 1200
                            games_played = 0
                            country_flag = '🏳️'
                            is_guest = session.get('is_guest', False)
                            
                            if not is_guest:
                                conn = sqlite3.connect(DB_PATH, timeout=30)
                                try:
                                    game_type_base = room.game_type.replace('solo_', '')
                                    config_key = f"{game_type_base}|{room.board_dimensions}|{room.time_limit}"
                                    # 24-hour rooms exception: load global rating from users table
                                    if is_24h:
                                        cursor = conn.execute('SELECT rating FROM users WHERE id = ?', (user_id,))
                                        row = cursor.fetchone()
                                        if row:
                                            rating = row[0]
                                        else:
                                            rating = 1200
                                    else:
                                        cursor = conn.execute('SELECT rating FROM user_ratings WHERE user_id = ? AND config_key = ?', 
                                                            (user_id, config_key))
                                        row = cursor.fetchone()
                                        if row:
                                            rating = row[0]
                                        else:
                                            # Every room starts the user at 1200, completely unique to this room configuration
                                            rating = 1200
                                    
                                    cur = conn.execute('SELECT games_played, country_flag FROM users WHERE id = ?', (user_id,))
                                    row2 = cur.fetchone()
                                    if row2:
                                        games_played = row2[0]
                                        if row2[1]:
                                            country_flag = row2[1]
                                except Exception as e:
                                    print(f"Error fetching stats on reconstruction: {e}")
                                finally:
                                    conn.close()
                            
                            room.add_player(user_id, session['username'], rating, 
                                            games_played=games_played, country_flag=country_flag, 
                                            manual_accessed=False, is_guest=is_guest)
                            room.update_player_activity(user_id)
                except Exception as re_err:
                    print(f"[app.py] Could not reconstruct public hub {room_id}: {re_err}")

        if not room:
             return jsonify({'error': 'Room not found (expired due to inactivity)'}), 404
            
        with room._state_lock:
            # LAZY LOAD YESTERDAY'S HISTORY FOR 24H ROOMS:
            if room.time_limit >= 7200 and (not getattr(room, 'previous_day_history', None) or len(room.previous_day_history) == 0):
                try:
                    room_manager.get_yesterdays_history(room, room.current_round)
                except Exception as e:
                    print(f"[app.py] Error lazy-loading yesterday's history for room {room_id}: {e}")

            # 1. Heartbeat Trigger (If TR=0/45/Search)
            milestone = room.get_next_round_milestone()
            if milestone == 'spinner':
                room_manager.generate_spinner_params(room_id, reveal=False)
            elif milestone == 'reveal':
                print(f"[API] Room {room_id}: TR={room.time_remaining} - REVEALING parameters (synchronous)")
                room_manager.generate_spinner_params(room_id, reveal=True)
            elif milestone == 'search':
                print(f"[API] Room {room_id}: TR={room.time_remaining} - STARTING board search (synchronous)")
                room_manager.start_board_search(room_id)
            elif milestone == 'start':
                # ATOMIC GUARD: Only launch ONE transition
                if not getattr(room, 'starting_round', False):
                    print(f"[Milestone] 0s remaining - Starting next round for {room_id} (Synchronous API Trigger)")
                    room_manager.start_next_round(room_id)

            # 2. Collect State Under Lock (Atomic Snapshot)
            is_revealed = room.spinner_params_revealed
            is_active = room.state == 'active'
            is_intermission = room.state == 'intermission'
            is_fcfs = (room.game_type == 'fcfs')
            
            # GLOBAL FOUND WORDS (Critical for FCFS accuracy)
            global_found = list(room.global_round_found_words) if hasattr(room, 'global_round_found_words') else []
            
            # FAIL-SAFE: If active but counts are missing or from a previous round, force a synchronous sync
            counts_obj = getattr(room, 'total_counts_by_len', {})
            needs_update = (not counts_obj or len(counts_obj) == 0 or counts_obj.get('_round') != room.current_round)
            
            if is_active and needs_update:
                print(f"[Remaining-Hardener] Found stale/missing counts (Round {counts_obj.get('_round') if counts_obj else 'None'} vs {room.current_round}) for {room_id}. Forcing sync.")
                room.update_counts_by_len()

            # TOTAL POINTS HARDENER: If active and total_points_count is 0, compute it now inline
            # This handles the race window between start_next_round setting state=active and
            # recalculate_total_points() being called, and also the first round of a new room.
            if is_active and room.total_points_count == 0:
                computed_pts = room.recalculate_total_points()
                if computed_pts == 0 and getattr(room, 'all_words', None):
                    # Ultra fallback: length-based estimate
                    fmt = str(getattr(room, 'current_board_format', '')).lower()
                    is_valued = 'valued' in fmt
                    pts = 0
                    for w in room.all_words:
                        l = len(w)
                        if is_valued: pts += l
                        elif l <= 4: pts += 1
                        elif l == 5: pts += 2
                        elif l == 6: pts += 3
                        elif l == 7: pts += 5
                        else: pts += 11
                    room.total_points_count = pts
                print(f"[PTS-HARDENER] Room {room_id} Round {room.current_round}: total_points_count={room.total_points_count} words={len(getattr(room,'all_words',set()))} scores={len(getattr(room,'solved_words_with_scores',{}))}")

            words_to_return = []
            word_scores_to_return = {}
            if is_intermission:
                # USER REQUEST: Absolute 'Last-Mile' Safety Net.
                # Never allow 3L/4L words to reach the client if the Spinner says 5L+.
                # Also enforce the global 4L floor for all solution lists.
                cur_min = getattr(room, 'current_min_length', 3)
                display_floor = cur_min
                words_to_return = [w for w in (room.all_words or []) if len(w) >= display_floor]
                
                # RE-SYNC: Ensure re-categorized lists also respect this floor
                if hasattr(word_validator, 'word_validator'):
                    room.csw_only_words = [w for w in words_to_return if word_validator.word_validator.is_csw_only(w)]
                    room.added_words = [w for w in words_to_return if word_validator.word_validator.is_added_word(w)]
                
                word_scores_to_return = getattr(room, 'solved_words_with_scores', {})
                # Purge scores as well
                word_scores_to_return = {w: word_scores_to_return[w] for w in words_to_return if w in word_scores_to_return}
                # Fallback to previous if current is somehow missing
                if not words_to_return:
                    prev_all = getattr(room, 'previous_all_word_scores', {}) or getattr(room, 'previous_all_words', {})
                    if isinstance(prev_all, dict):
                        words_to_return = list(prev_all.keys())
                        word_scores_to_return = prev_all
                    elif isinstance(prev_all, list):
                        words_to_return = prev_all
            elif is_active:
                # ACTIVE: Provide word scores for total-points calculation client-side
                # (Avoids showing '0 total pts' when total_points_count hasn't been computed yet)
                word_scores_to_return = getattr(room, 'solved_words_with_scores', {}) or {}
                if is_fcfs or room.time_limit >= 7200:
                    words_to_return = list(room.all_words)

            # Determine user visibility
            user_id = session.get('user_id')
            
            def get_incremental_data(p):
                """Helper to filter words and calculate score based on time for incremental bots"""
                now = time.time()
                is_cur_active = (room.state == 'active')
                v_words = []
                v_score = 0
                f_bonus = False
                
                is_me = (str(p.user_id) == str(user_id))
                is_shared_mode = (is_fcfs or not is_cur_active)
                
                for w in p.submitted_words:
                    # 1. TIME CHECK: Has the bot "found" this word yet?
                    if w.get('time', 0) > now:
                        continue # In the future, skip entirely
                        
                    # 2. Add to score and bonus status since it has been found
                    pts = w.get('points', 0)
                    if pts > 0:
                         v_score += pts
                    
                    is_b = (room.bonus_word and w['word'].upper() == room.bonus_word.upper())
                    if is_b: f_bonus = True
                    if is_fcfs and w.get('is_penalty'): continue
                    
                    # 3. VISIBILITY CHECK: Can the current user see the actual text of this word right now?
                    # - If it's me: YES
                    # - If the round is over or it's FCFS: YES
                    # - If it's a bot (or opponent) in an active Accumulative round: NO
                    can_see_text = is_me or is_shared_mode
                    
                    w_copy = dict(w)
                    w_copy['is_bonus'] = is_b
                    w_copy['finder'] = p.username # Guarantee finder name is attached
                    
                    if not can_see_text:
                        # Obfuscate the word so the client knows they scored, but can't see what they found
                        # The client uses '?' to render obfuscated words if we mask it here, or we can just omit 'word'
                        w_copy['word'] = '?' * len(w['word'])
                        w_copy['obfuscated'] = True
                        
                    v_words.append(w_copy)
                    
                return (v_words, v_score, f_bonus)

            # In FCFS, total_words_count should reflect what's left globally
            actual_total = room.total_words_count
            if is_active and is_fcfs:
                actual_total = max(0, room.total_words_count - len(global_found))

            raw_fmt = getattr(room, 'current_board_format', 'Normal')
            is_bonus_format = ('bonus letter' in str(raw_fmt).lower() or 'either' in str(raw_fmt).lower())
            
            return jsonify({
                'room_id': room.room_id,
                'game_type': room.game_type,
                'state': room.state,
                'current_round': room.current_round,
                'time_limit': room.time_limit,
                'time_remaining': room.time_remaining,
                'round_end_time': room.round_end_time if room.state == 'active' else 0,
                'intermission_end_time': room.intermission_end_time if room.state == 'intermission' else 0,
                'server_time': time.time(),
                'your_username': session.get('username'),
                'your_user_id': session.get('user_id'),
                'board': room.board,
                'board_dimensions': room.board_dimensions,
                'bonus_word': room.bonus_word,
                'bonus_cell': room.bonus_cell if 'bonus letter' in str(raw_fmt).lower() else None,
                'all_words': words_to_return,
                'total_words_count': (room.previous_total_words if is_intermission else actual_total),
                'next_round_total_words_count': getattr(room, 'next_round_total_words_count', 0),
                'initial_total_words': getattr(room, 'initial_total_words', actual_total),
                'total_points_count': (getattr(room, 'next_round_total_points', 0) if (is_intermission and is_revealed and getattr(room, 'next_round_total_points', 0) > 0) else (room.previous_total_points if is_intermission else room.total_points_count)),
                'total_counts_by_len': (room.previous_total_counts_by_len if is_intermission else getattr(room, 'total_counts_by_len', {})),
                'cell_density': (getattr(room, 'next_round_cell_density', []) if (is_intermission and is_revealed) else getattr(room, 'cell_density', [])),
                'max_cell_density': (getattr(room, 'next_round_max_cell_density', 0) if (is_intermission and is_revealed) else getattr(room, 'max_cell_density', 0)),
                'all_word_scores': word_scores_to_return,
                'global_found_words': global_found,
                'fcfs_found_words': list(getattr(room, 'fcfs_found_words', [])) if (is_active and is_fcfs) else [],
                'added_words': [w for w in words_to_return if word_validator.get_use_added_words() and word_validator.is_added_word(w)],
                'csw_only_words': [w for w in words_to_return if word_validator.is_csw_only(w)],
                'previous_all_words': [w for w in (getattr(room, 'previous_all_words', []) or []) if len(w) >= getattr(room, 'previous_min_length', 3)],
                'previous_all_word_scores': {w: v for w, v in (getattr(room, 'previous_all_word_scores', {}) or {}).items() if len(w) >= getattr(room, 'previous_min_length', 3)},
                'previous_board': getattr(room, 'previous_board', []),
                'previous_csw_only_words': getattr(room, 'previous_csw_only_words', []),
                'previous_added_words': getattr(room, 'previous_added_words', []),
                'previous_bonus_word': getattr(room, 'previous_bonus_word', ''),
                'spinner_params': {**room.spinner_params, 'uniqueness': getattr(room, 'next_round_uniqueness', None) or 0} if (is_intermission and is_revealed) else room.spinner_params,
                'current_min_length': getattr(room, 'current_min_length', 3),
                'current_board_format': getattr(room, 'current_board_format', 'Normal'),
                'current_word_count_range': getattr(room, 'current_word_count_range', 'Random'),
                'current_difficulty': getattr(room, 'current_difficulty', None) or 'Medium',
                'current_uniqueness': getattr(room, 'current_uniqueness', None) or 0.0,
                'spinner_params_revealed': is_revealed,
                'players': [
                    {
                        'user_id': p.user_id,
                        'username': p.username,
                        'is_ai': p.is_ai,
                        'rating': p.rating,
                        'words_count': len(data[0]),
                        'score': data[1],
                        'rating_change': p.rating_change,
                        'found_bonus_word': data[2],
                        'submitted_words': data[0],
                        'previous_submitted_words': p.previous_submitted_words,
                        'invalid_words': p.invalid_words,
                        'input_method': p.input_method,
                        'last_active_age': time.time() - p.last_active,
                        'games_played': p.games_played,
                        'country_flag': p.country_flag,
                        'joined_mid_round': getattr(p, 'joined_mid_round', False),
                        'has_exceptional_round': getattr(p, 'has_exceptional_round', False),
                        'performance_efficiency': getattr(p, 'performance_efficiency', 0.0)
                    } for p, data in sorted(
                        [(p, get_incremental_data(p)) for p in room.players], 
                        key=lambda x: x[1][1], 
                        reverse=True
                    )
                ],
                'spectators': [
                    {'username': s.username, 'rating': s.rating, 'user_id': s.user_id} for s in room.spectators
                ] if hasattr(room, 'spectators') else [],
                'chat_messages': getattr(room, 'chat_messages', []),
                'winners_history': getattr(room, 'winners_history', []),
                'previous_day_history': getattr(room, 'previous_day_history', {}),
                'your_username': session.get('username')
            })
            
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
            return resp

    except Exception as e:
        import traceback
        print(f"ERROR in get_room_state: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/room/<room_id>/propose-board', methods=['POST'])
def propose_board(room_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    proposed_board = data.get('board')
    username = session.get('username')
    
    if not proposed_board:
        return jsonify({'error': 'No board provided'}), 400
        
    result = room_manager.propose_board(room_id, proposed_board, username)
    return jsonify(result)

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
    path = data.get('path') # List of [r, c] pairs from Mouse/Touch

    print(f"[Submit-Diag] User {user_id} submitting word '{word}' for room {room_id}")
    
    # Update input method if provided
    if input_method:
        player = room.get_player(user_id)
        if player:
            player.input_method = input_method
            
    try:
        with room._state_lock:
            success, message, points, final_word = room.submit_word(user_id, word, path=path)
    except Exception as e:
        import traceback
        with open(DEBUG_FLOW_PATH, 'a') as f:
            f.write(f"[Submit-Error] Room: {room_id} | Error: {e}\n{traceback.format_exc()}\n")
        return jsonify({'success': False, 'message': f'Server Error: {str(e)}'}), 500
    
    # Refresh activity on any submission attempt (valid or not)
    room.update_player_activity(user_id)
    
    player = room.get_player(user_id)
    new_score = player.score if player else 0

    return jsonify({
        'success': success, 
        'message': message,
        'points': points,
        'word': final_word,
        'new_score': new_score,
        'cell_density': getattr(room, 'cell_density', None),
        'max_cell_density': getattr(room, 'max_cell_density', 0)
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

# Definitions and Pronunciations Cache
DEFINITIONS_CACHE = None
DEFINITIONS_PATH = None
PRONUNCIATIONS_CACHE = None

def load_definitions():
    global DEFINITIONS_CACHE, DEFINITIONS_PATH
    # Skip reload only if cache is already populated
    if DEFINITIONS_CACHE:
        return

    DEFINITIONS_CACHE = {}

    # Search multiple locations for the definitions file
    search_paths = [
        os.path.expanduser('~/Desktop/Definitions.txt'),
        os.path.join(os.path.dirname(__file__), 'dictionaries', 'Definitions.txt'),
        os.path.join(os.path.dirname(__file__), 'Definitions.txt'),
    ]

    definitions_path = None
    for path in search_paths:
        if os.path.exists(path):
            definitions_path = path
            DEFINITIONS_PATH = path
            break

    if not definitions_path:
        print(f"Definitions file not found. Searched: {search_paths}")
        return

    try:
        print(f"Loading definitions from {definitions_path}...")
        with open(definitions_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.split(' - ', 1)
                if len(parts) == 2:
                    word = parts[0].strip().upper()
                    definition = parts[1].strip()
                    DEFINITIONS_CACHE[word] = definition
        print(f"Loaded {len(DEFINITIONS_CACHE)} definitions")
    except Exception as e:
        print(f"Error loading definitions: {e}")
        DEFINITIONS_CACHE = {}

def load_pronunciations():
    global PRONUNCIATIONS_CACHE
    if PRONUNCIATIONS_CACHE is not None:
        return

    PRONUNCIATIONS_CACHE = {}
    pron_path = os.path.join(os.path.dirname(__file__), 'dictionaries', 'pronunciations.txt')
    
    if not os.path.exists(pron_path):
        print(f"Pronunciations file not found at {pron_path}")
        return

    try:
        print(f"Loading pronunciations from {pron_path}...")
        with open(pron_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.split(' - ', 1)
                if len(parts) == 2:
                    word = parts[0].strip().upper()
                    pron = parts[1].strip()
                    PRONUNCIATIONS_CACHE[word] = pron
        print(f"Loaded {len(PRONUNCIATIONS_CACHE)} pronunciations")
    except Exception as e:
        print(f"Error loading pronunciations: {e}")
        PRONUNCIATIONS_CACHE = {}

def lookup_word_definition_and_pronunciation(word):
    global DEFINITIONS_CACHE, PRONUNCIATIONS_CACHE
    if not DEFINITIONS_CACHE:
        load_definitions()
    if PRONUNCIATIONS_CACHE is None:
        load_pronunciations()

    word_upper = word.upper().strip()
    definition = DEFINITIONS_CACHE.get(word_upper)
    pronunciation = PRONUNCIATIONS_CACHE.get(word_upper)
    
    # ONLINE FALLBACK: If definition is not found locally, try Free Dictionary API
    if not definition:
        try:
            import urllib.request
            import json
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word_upper.lower()}"
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, timeout=3.0) as response:
                api_data = json.loads(response.read().decode('utf-8'))
                if isinstance(api_data, list) and len(api_data) > 0:
                    meanings = api_data[0].get('meanings', [])
                    def_parts = []
                    for m in meanings:
                        part_of_speech = m.get('partOfSpeech', '')
                        defs = m.get('definitions', [])
                        if defs:
                            first_def = defs[0].get('definition', '')
                            if first_def:
                                def_parts.append(f"({part_of_speech}) {first_def}")
                    if def_parts:
                        definition = "; ".join(def_parts)
                        DEFINITIONS_CACHE[word_upper] = definition
        except Exception as e:
            print(f"Online dictionary API fallback failed for '{word_upper}': {e}")

    # SECONDARY ONLINE FALLBACK: If still not found, try English Wiktionary REST API
    if not definition:
        try:
            import urllib.request
            import json
            import re
            import html
            url = f"https://en.wiktionary.org/api/rest_v1/page/definition/{word_upper.lower()}"
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'MorphemeApp/1.0 (jeff@morpheme.games) Python-urllib'}
            )
            with urllib.request.urlopen(req, timeout=3.0) as response:
                api_data = json.loads(response.read().decode('utf-8'))
                if isinstance(api_data, dict) and "en" in api_data:
                    def_parts = []
                    for item in api_data["en"]:
                        part_of_speech = item.get("partOfSpeech", "")
                        for d in item.get("definitions", []):
                            text = d.get("definition", "")
                            text = re.sub(r"<[^>]+>", "", text)
                            text = re.sub(r"\s+", " ", text).strip()
                            text = html.unescape(text)
                            if text:
                                def_parts.append(f"({part_of_speech}) {text}")
                    if def_parts:
                        definition = "; ".join(def_parts)
                        DEFINITIONS_CACHE[word_upper] = definition
        except Exception as e:
            print(f"Wiktionary API fallback failed for '{word_upper}': {e}")

    return definition, pronunciation

@app.route('/api/definition', methods=['GET'])
def get_definition():
    word = request.args.get('word', '').upper()
    if not word:
        return jsonify({'error': 'Word parameter required'}), 400

    definition, pronunciation = lookup_word_definition_and_pronunciation(word)

    if definition or pronunciation:
        return jsonify({
            'word': word, 
            'definition': definition or "No definition available for this word.",
            'pronunciation': pronunciation
        })
    else:
        return jsonify({'error': 'Word not found'}), 404



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
    """Load dictionary for tools into memory cache.
    Always merges the 16+ supplementary word list (16plus.txt) into the result
    so every tool/API route automatically includes long words."""
    cache_key = dict_name
    if cache_key in TOOLS_DICT_CACHE:
        return TOOLS_DICT_CACHE[cache_key]

    if dict_name == 'ALL':
        words = set()
        for d in ['NWL', 'CSW']:
            p = os.path.join(os.path.dirname(__file__), 'dictionaries', f'{d}.txt')
            if os.path.exists(p):
                with open(p, 'r') as f:
                    words.update(line.strip().upper() for line in f if line.strip())
        # Add manually added words
        words.update(word_validator.added_words)
        print(f"[Tools] Loaded ALL dictionary: {len(words)} unique words")
    elif dict_name == 'added_words':
        words = word_validator.added_words.copy()
        print(f"[Tools] Loaded Added Words dictionary: {len(words)} unique words")
    else:
        dict_path = os.path.join(os.path.dirname(__file__), 'dictionaries', f'{dict_name}.txt')
        try:
            print(f"[Tools] Loading dictionary: {dict_path}")
            with open(dict_path, 'r') as f:
                words = set(word.strip().upper() for word in f)
            print(f"[Tools] Loaded {len(words)} words from {dict_name}")
        except FileNotFoundError:
            print(f"[Tools] Dictionary file not found: {dict_path}")
            words = set()

    # Merge supplementary 16+ word list
    long_path = os.path.join(os.path.dirname(__file__), 'dictionaries', '16plus.txt')
    try:
        with open(long_path, 'r') as f:
            long_words = {line.strip().upper() for line in f if line.strip()}
        words = words | long_words
        print(f"[Tools] Merged {len(long_words)} supplementary 16+ words into {dict_name}")
    except FileNotFoundError:
        print(f"[Tools] 16plus.txt not found – skipping supplementary merge")

    # --- OPTIMIZATION: PRE-CALCULATE FREQUENCY MATRIX ---
    import numpy as np
    word_list = sorted(list(words))
    matrix = np.zeros((len(word_list), 26), dtype=np.uint8)
    masks = np.zeros(len(word_list), dtype=np.uint32)
    
    for i, word in enumerate(word_list):
        mask = 0
        for char in word:
            if 'A' <= char <= 'Z':
                c_idx = ord(char) - ord('A')
                matrix[i, c_idx] += 1
                mask |= (1 << c_idx)
        masks[i] = mask
    
    lens = np.array([len(w) for w in word_list], dtype=np.uint8)
    
    result = {
        'words': word_list,
        'set': words,
        'matrix': matrix,
        'lens': lens,
        'masks': masks
    }
    TOOLS_DICT_CACHE[cache_key] = result
    return result

def get_lis(nums):
    """Calculates Longest Increasing Subsequence length."""
    if not nums:
        return 0
    # Standard O(n log n) or O(n^2) approach. Words are short, O(n^2) is negligible.
    # Using DP (O(n^2)) for simplicity and correctness with small N.
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(len(nums[:i])):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if dp else 0

def calculate_morpheme_metric(source, target):
    s_len, t_len = len(source), len(target)
    if s_len == 0 or t_len == 0: return 99, 0
    
    # 1. Optimized LCS (Linearity) using sliding row
    prev = [0] * (t_len + 1)
    curr = [0] * (t_len + 1)
    for char_s in source:
        for j in range(1, t_len + 1):
            if char_s == target[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                p_v = prev[j]
                c_v = curr[j-1]
                curr[j] = p_v if p_v > c_v else c_v
        prev[:] = curr
    
    linearity = prev[t_len]
    if linearity == 0: return 99, 0
    
    # mp >= target_len - linearity. If target_len - linearity > 6, exit.
    if t_len - linearity > 6:
        return 99, linearity

    # 2. Backtrace (Only for high-linearity candidates)
    # We use a single 2D array but only if necessary
    dp = [[0] * (t_len + 1) for _ in range(s_len + 1)]
    for i in range(1, s_len + 1):
        s_i = source[i-1]
        dp_prev = dp[i-1]
        dp_curr = dp[i]
        for j in range(1, t_len + 1):
            if s_i == target[j-1]:
                dp_curr[j] = dp_prev[j-1] + 1
            else:
                v1 = dp_prev[j]
                v2 = dp_curr[j-1]
                dp_curr[j] = v1 if v1 >= v2 else v2
            
    matched_s_indices = []
    i, j = s_len, t_len
    while i > 0 and j > 0:
        if source[i-1] == target[j-1]:
            matched_s_indices.append(i-1)
            i -= 1; j -= 1
        elif dp[i-1][j] >= dp[i][j-1]: i -= 1
        else: j -= 1
    matched_s_indices.reverse()
    
    # 3. Dynamic Span Optimization
    # We find the sub-range of the LCS that minimizes (Insertions + Relocations + Paid Deletions)
    # This allows skipping expensive gaps in the source word if a partial match is cheaper.
    best_mp = t_len # Default: All characters as insertions
    
    for i in range(len(matched_s_indices)):
        for j in range(i, len(matched_s_indices)):
            sub = matched_s_indices[i:j+1]
            m_len = len(sub)
            f_idx = min(sub)
            l_idx = max(sub)
            
            # Metric components for this sub-range
            sub_lis = get_lis(sub)
            relocations = m_len - sub_lis
            paid_deletions = (l_idx - f_idx + 1) - m_len
            insertions = t_len - m_len
            
            total_mp = relocations + paid_deletions + insertions
            if total_mp < best_mp:
                best_mp = total_mp
    
    return best_mp, linearity


def check_and_add_mp(mp_groups, source_len, target_len, mp, word):
    """Applies strict filtering logic from combos.java."""
    # mp_groups is now a dict of sets
    added = False
    
    if source_len == 3:
        if target_len >= 3: added = True
    elif source_len == 4:
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
    elif source_len >= 9:
        if target_len >= 6 and mp <= 5:
             if mp >= 5:
                 if target_len >= 8: added = True
             else:
                 added = True
    
    if added:
        mp_groups[mp].add(word)

def check_and_add_lic(lic_groups, count, target_len, word):
    if count not in lic_groups: lic_groups[count] = set()
    valid = False
    if count == 1: valid = (target_len < 3)
    elif count == 2: valid = (target_len < 4)
    elif count == 3: valid = (target_len < 5)
    elif count == 4: valid = (target_len < 6)
    elif count == 5: valid = (target_len < 7)
    elif count == 6: valid = (target_len < 8)
    elif count == 7: valid = (target_len < 10)
    elif count >= 8: valid = (target_len < 11)
    if valid:
        lic_groups[count].add(word)

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
        
    dict_data = load_tools_dictionary(dict_name)
    if not dict_data:
        return jsonify({'error': f'Dictionary {dict_name} not found'}), 404

    import numpy as np
    word_list = dict_data['words']
    dict_matrix = dict_data['matrix']
    dict_lens = dict_data['lens']
    dict_masks = dict_data['masks']
    
    source_len = len(search_term)
    search_term_rev = search_term[::-1]
    
    # Vectorized Search Vector & Mask
    s_vec = np.zeros(26, dtype=np.uint8)
    s_mask = 0
    for char in search_term:
        if 'A' <= char <= 'Z':
            c_idx = ord(char) - ord('A')
            s_vec[c_idx] += 1
            s_mask |= (1 << c_idx)
    
    # 1. BITMASK FILTER (Extremely Fast Vectorized Bit Count)
    mask_intersection = dict_masks & s_mask
    
    # Vectorized Population Count (NumPy native)
    m = mask_intersection.astype(np.uint32)
    m = (m & 0x55555555) + ((m >> 1) & 0x55555555)
    m = (m & 0x33333333) + ((m >> 2) & 0x33333333)
    m = (m & 0x0F0F0F0F) + ((m >> 4) & 0x0F0F0F0F)
    m = (m & 0x00FF00FF) + ((m >> 8) & 0x00FF00FF)
    m = (m & 0x0000FFFF) + ((m >> 16) & 0x0000FFFF)
    passed_mask = (m >= 3)
    
    # 2. VECTORIZED PRUNING
    shared_counts = np.minimum(dict_matrix, s_vec).sum(axis=1)
    
    # HEURISTIC TIGHTENING: For 7+ letter words, we need a high shared count
    min_shared = 1
    if source_len == 5: min_shared = 3
    if source_len == 6: min_shared = 4 # TIGHTENED to 4
    if source_len >= 7: min_shared = 5 # TIGHTENED to 5 for speed
    
    len_diffs = np.abs(dict_lens.astype(np.int16) - source_len)
    candidates = np.where(
        passed_mask & 
        (len_diffs <= 6) & 
        (shared_counts >= min_shared) &
        (dict_lens.astype(np.int16) - shared_counts <= 6)
    )[0]
    
    # Initialize Groups (Using sets to prevent O(N^2) search bottleneck)
    mp_groups = {i: set() for i in range(7)} # 0MP to 6MP
    lic_groups = {}
    
    # --- OPTIMIZED SINGLE-THREADED LOOP ---
    for idx in candidates:
        word = word_list[idx]
        target_len = int(dict_lens[idx])
        shared_count = int(shared_counts[idx])
        
        # 1-pass primary check
        best_mp, linearity = calculate_morpheme_metric(search_term, word)
        
        # Subsequent passes only if promising
        # Early Exit: If linearity is already very low, m2/m3 might not help much
        # But we must be careful. For now, just optimize the calls.
        if best_mp > 1:
            # Mirror check (source vs target[::-1])
            m2, _ = calculate_morpheme_metric(search_term, word[::-1])
            best_mp = min(best_mp, m2)
        if best_mp > 1:
            # Reverse-Source check (source[::-1] vs target)
            # Note: These are NOT identical due to how paid_deletions/span are calculated.
            m3, _ = calculate_morpheme_metric(search_term[::-1], word)
            best_mp = min(best_mp, m3)
        
        if best_mp <= 6:
            check_and_add_mp(mp_groups, source_len, target_len, best_mp, word)
            
        # LIC logic: uses the pre-calculated shared_count (Vectorized)
        if shared_count >= 1:
            check_and_add_lic(lic_groups, shared_count, target_len, word)

    # Sort Groups
    for k in mp_groups:
        mp_groups[k] = sorted(list(mp_groups[k]), key=lambda x: (-len(x), x))
        
    for k in lic_groups:
        lic_groups[k] = sorted(list(lic_groups[k]), key=lambda x: (len(x), x))
    
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
                pass
        
        # Normalize start letter
        start_char = None
        if start_filter and start_filter.lower() != 'all':
            start_char = start_filter.upper().strip()
            if not start_char: start_char = None

        list_type = request.args.get('list_type', 'all').lower()

        base_dir = os.path.dirname(__file__)
        dict_dir = os.path.join(base_dir, 'dictionaries')
        
        # --- Logic: Unified 16+ Routing ---
        def load_source_set(filename):
            if target_len is not None and target_len >= 16:
                path = os.path.join(dict_dir, '16plus.txt')
            else:
                path = os.path.join(dict_dir, filename)
            
            if not os.path.exists(path):
                return set()
                
            words = set()
            with open(path, 'r') as f:
                for line in f:
                    w = line.strip().upper()
                    if not w: continue
                    if (target_len is None or target_len < 16) and len(w) >= 16:
                        continue
                    if target_len is not None and len(w) != target_len:
                        continue
                    if start_char is not None and not w.startswith(start_char):
                        continue
                    words.add(w)
            return words

        # Conditional fetching based on list_type
        response = {
            'nwl': [], 'csw': [], 'csw_only': [], 'likelihood': [], 'uniques': [], 'added': [],
            'new_nwl': [], 'new_csw': []
        }

        if list_type in ['all', 'nwl', 'csw_only', 'likelihood']:
            nwl_set = load_source_set('NWL.txt')
            if list_type in ['all', 'nwl']: response['nwl'] = sorted(list(nwl_set))

        if list_type in ['all', 'csw', 'csw_only']:
            csw_set = load_source_set('CSW.txt')
            if list_type in ['all', 'csw']: response['csw'] = sorted(list(csw_set))

        if list_type in ['all', 'csw_only']:
            # We need both sets for CSW only
            if 'nwl_set' not in locals(): nwl_set = load_source_set('NWL.txt')
            if 'csw_set' not in locals(): csw_set = load_source_set('CSW.txt')
            response['csw_only'] = sorted(list(csw_set - nwl_set))

        if list_type in ['all', 'likelihood']:
            if 'nwl_set' not in locals(): nwl_set = load_source_set('NWL.txt')
            scrabble_freq = {
                'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2, 'I': 9,
                'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2, 'Q': 1, 'R': 6,
                'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1, 'Y': 2, 'Z': 1
            }
            def calculate_scrabble_likelihood(word):
                letter_counts = {}
                total = 0
                for ch in word:
                    base = scrabble_freq.get(ch, 0)
                    seen = letter_counts.get(ch, 0)
                    total += max(0, base - seen)
                    letter_counts[ch] = seen + 1
                return total
            
            likelihood_list = []
            for w in nwl_set:
                score = calculate_scrabble_likelihood(w)
                likelihood_list.append({'score': score, 'word': w})
            likelihood_list.sort(key=lambda x: (-x['score'], x['word']))
            response['likelihood'] = likelihood_list

        if list_type in ['all', 'uniques']:
            response['uniques'] = sorted(list(load_source_set('uniqueNWL.txt')))
            
        if list_type in ['all', 'new_nwl']:
            response['new_nwl'] = list(load_source_set('new_NWL.txt'))
            response['new_nwl'].reverse() # Show most recent first
            
        if list_type in ['all', 'new_csw']:
            response['new_csw'] = list(load_source_set('new_CSW.txt'))
            response['new_csw'].reverse() # Show most recent first
            
        if list_type in ['all', 'added']:
            # Added Words: Preserve file order (which is chronological as they are appended)
            # and reverse to show most recent first per USER REQUEST.
            path = os.path.join(dict_dir, 'added_words.txt')
            unique_added = []
            seen_added = set()
            if os.path.exists(path):
                with open(path, 'r') as f:
                    # File is now newest-first, so we read directly
                    raw_lines = [line.strip().upper() for line in f if line.strip()]
                    for w in raw_lines:
                        if w in seen_added: continue
                        
                        # Filter by length and start char if provided
                        if target_len is not None and len(w) != target_len: continue
                        if start_char is not None and not w.startswith(start_char): continue
                        
                        unique_added.append(w)
                        seen_added.add(w)
            response['added'] = unique_added

        return jsonify(response)

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
    for word in dictionary['words']:
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
    for word in dictionary['words']:
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
        
    is_valid = word in dictionary['set']
    
    # Try to get definition and pronunciation if valid
    definition = None
    pronunciation = None
    if is_valid:
        definition, pronunciation = lookup_word_definition_and_pronunciation(word)
        if not definition:
            definition = "No definition available for this word."
        
    return jsonify({
        'word': word,
        'is_valid': is_valid,
        'definition': definition,
        'pronunciation': pronunciation
    })

@app.route('/api/tools/unscramble/random', methods=['GET'])
def tools_unscramble_random():
    import random
    length = request.args.get('length', type=int)
    dict_name = request.args.get('dictionary', 'NWL')
    must_have = request.args.get('must_have', '').upper().strip()
    print(f"[Unscramble-API-Debug] Length: {length}, MustHave: '{must_have}'")
    
    if not length:
        return jsonify({'error': 'Length required'}), 400
        
    dictionary = load_tools_dictionary(dict_name)
    if not dictionary:
        return jsonify({'error': f'Dictionary {dict_name} not found'}), 404
        
    # Filter dictionary for words of exactly this length and containing must_have
    eligible_words = [w for w in dictionary['words'] if len(w) == length]
    if must_have:
        eligible_words = [w for w in eligible_words if must_have in w]
    
    if not eligible_words:
        return jsonify({'error': 'No words found for this length'}), 404
        
    # Pick a random word
    target_word = random.choice(eligible_words)
    
    # Scramble it
    letters = list(target_word)
    random.shuffle(letters)
    jumbled = "".join(letters)
    
    # Find all anagrams
    target_counter = Counter(target_word)
    anagrams = [w for w in eligible_words if Counter(w) == target_counter]
    
    return jsonify({
        'jumbled': jumbled,
        'words': anagrams,
        'count': len(anagrams)
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
        
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
        
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
        
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
    """Solves a custom board provided by the user.
    Blocks results if the submitted board matches any currently active live room board.
    """
    data = request.json
    board = data.get('board') # 2D list of letters
    dictionary = data.get('dictionary', 'NWL')
    
    if not board or not isinstance(board, list):
        return jsonify({'error': 'No board provided or invalid format'}), 400
    
    # Flatten submitted board to a comparable string: "A|B|C|D\nE|F|G|H\n..."
    def flatten_board(b):
        return '\n'.join('|'.join(row) for row in b)
    
    submitted_flat = flatten_board(board)
    
    # Check against all active live rooms
    try:
        for room in room_manager.rooms.values():
            if room.state == 'active' and room.board:
                room_flat = flatten_board(room.board)
                if submitted_flat == room_flat:
                    print(f"[ManualSolve] Board matches active room {room.room_id} — blocking results")
                    return jsonify({
                        'board_matches_active_room': True,
                        'results': [],
                        'count': 0
                    })
    except Exception as check_err:
        print(f"[ManualSolve] Error during room board check (non-fatal): {check_err}")
        
    try:
        # SYNC: Ensure dictionary state is fresh for this process
        word_validator.get_use_added_words()

        # We use the board_generator from the global room_manager instance
        min_word_length = int(data.get('min_word_length', 3))
        all_words_dict = room_manager.board_generator._solve_board(board, dictionary, (0, float('inf')), min_word_length)
        
        # Sort by largest first (Length DESC, then Alpha ASC)
        all_words = sorted(list(all_words_dict.keys()), key=lambda x: (-len(x), x))
        
        return jsonify({
            'results': all_words,
            'count': len(all_words),
            'board_matches_active_room': False
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
            
    filtered_words = dictionary['words']
    if target_len:
        filtered_words = [w for w in filtered_words if len(w) == target_len]
        
    if not filtered_words:
        return jsonify({'error': 'No words found for the specified criteria'}), 404
        
    import random
    random_word = random.choice(filtered_words)
    
    # Get definition and pronunciation
    definition, pronunciation = lookup_word_definition_and_pronunciation(random_word)
    if not definition:
        definition = "No definition available for this word."
    
    return jsonify({
        'word': random_word,
        'definition': definition,
        'pronunciation': pronunciation
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
    eligible_words = sorted([w for w in dictionary['words'] if 6 <= len(w) <= 10])
    
    if not eligible_words:
        return jsonify({'error': 'No eligible words found'}), 500
        
    seed_hash = int(hashlib.md5(today_str.encode()).hexdigest(), 16)
    idx = seed_hash % len(eligible_words)
    wotd = eligible_words[idx]
    
    # Get definition and pronunciation
    definition, pronunciation = lookup_word_definition_and_pronunciation(wotd)
    if not definition:
        definition = "No definition available for this word."
    
    return jsonify({
        'word': wotd,
        'date': today_str,
        'definition': definition,
        'pronunciation': pronunciation
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
        
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
        
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
        
    # Mark user as online
    room_manager.update_presence(session.get('user_id'))
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
            
        # USER REQUEST: Online first, then alphabetical among online, then alphabetical among offline
        friends_data.sort(key=lambda x: (not x.get('is_online', False), x.get('username', '').lower()))
        
        return jsonify({'friends': friends_data})
    finally:
        conn.close()

# --- FORUM ENDPOINTS ---

@app.route('/api/forum/categories', methods=['GET'])
def get_forum_categories():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # Include last_content_at by checking latest post OR latest comment in each category
        query = '''
            SELECT c.*, 
                   (SELECT MAX(ts) FROM (
                       SELECT MAX(timestamp) as ts FROM forum_posts fp WHERE fp.category_id = c.id
                       UNION ALL
                       SELECT MAX(fc.timestamp) as ts FROM forum_comments fc 
                       JOIN forum_posts fp2 ON fc.post_id = fp2.id 
                       WHERE fp2.category_id = c.id
                   )) as last_content_at
            FROM forum_categories c
        '''
        rows = conn.execute(query).fetchall()
        categories = []
        for row in rows:
            d = dict(row)
            if d['last_content_at']:
                # Ensure it has Z for UTC parsing
                if ' ' in d['last_content_at'] and 'Z' not in d['last_content_at']:
                    d['last_content_at'] = d['last_content_at'].replace(' ', 'T') + 'Z'
            else:
                # Fallback to a very old date so new users don't see unread indicators for empty cats
                d['last_content_at'] = '2000-01-01T00:00:00Z'
            categories.append(d)
        return jsonify({'categories': categories})
    finally:
        conn.close()

@app.route('/api/forum/posts/<int:category_id>', methods=['GET'])
def get_forum_posts(category_id):
    conn = sqlite3.connect(DB_PATH, timeout=30)
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

@app.route('/api/forum/posts/user/<username>', methods=['GET'])
def get_forum_user_posts(username):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # Fetch posts created by user
        posts_rows = conn.execute('''
            SELECT 'post' as type, p.id as id, p.id as post_id, p.title, p.content, p.image_url, p.timestamp,
            (SELECT COUNT(*) FROM forum_comments WHERE post_id = p.id) as comment_count,
            u.username, u.avatar_url, u.country_flag
            FROM forum_posts p
            JOIN users u ON p.user_id = u.id
            WHERE LOWER(u.username) = LOWER(?)
        ''', (username,)).fetchall()

        # Fetch comments made by user
        comments_rows = conn.execute('''
            SELECT 'comment' as type, c.id as id, c.post_id as post_id, p.title, c.content, c.image_url, c.timestamp,
            0 as comment_count,
            u.username, u.avatar_url, u.country_flag
            FROM forum_comments c
            JOIN forum_posts p ON c.post_id = p.id
            JOIN users u ON c.user_id = u.id
            WHERE LOWER(u.username) = LOWER(?)
        ''', (username,)).fetchall()

        posts = [dict(row) for row in posts_rows]
        comments = [dict(row) for row in comments_rows]
        
        print(f"[Forum] User search for '{username}' found {len(posts)} threads and {len(comments)} replies.")
        
        # Combine and sort by timestamp DESC
        all_items = posts + comments
        all_items.sort(key=lambda x: x['timestamp'] or '', reverse=True)
        
        res = jsonify({'posts': all_items})
        res.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        return res

    finally:
        conn.close()




@app.route('/api/forum/post/<int:post_id>', methods=['GET'])
def get_forum_post_detail(post_id):
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
            ORDER BY c.timestamp DESC
        ''', (post_id,)).fetchall()
        
        response_data = {
            'post': dict(post),
            'comments': [dict(c) for c in comments],
            'sorting': 'newest_first'
        }
        res = jsonify(response_data)
        res.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        return res
    finally:
        conn.close()

@app.route('/api/forum/post/delete/<int:post_id>', methods=['POST'])
def delete_forum_post(post_id):
    if not session.get('username') or not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        # Delete associated comments first to maintain referential integrity
        conn.execute('DELETE FROM forum_comments WHERE post_id = ?', (post_id,))
        # Delete the main post
        conn.execute('DELETE FROM forum_posts WHERE id = ?', (post_id,))
        conn.commit()
        return jsonify({'success': True, 'message': 'Post and associated comments deleted.'})
    except Exception as e:
        print(f"Error deleting forum post: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/forum/comment/delete/<int:comment_id>', methods=['POST'])
def delete_forum_comment(comment_id):
    if not session.get('username') or not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute('DELETE FROM forum_comments WHERE id = ?', (comment_id,))
        conn.commit()
        return jsonify({'success': True, 'message': 'Comment deleted.'})
    except Exception as e:
        print(f"Error deleting forum comment: {e}")
        return jsonify({'error': str(e)}), 500
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
            
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        # [Restriction]: Only mods can post in 'News'
        cur = conn.execute('SELECT name FROM forum_categories WHERE id = ?', (category_id,))
        cat_row = cur.fetchone()
        if cat_row and cat_row[0] == "News" and not is_mod(session['username']):
            return jsonify({'error': 'Only moderators can post in the News category.'}), 403
            
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
        
    # Switch to request.form to support multipart/form-data for image uploads
    data = request.form
    post_id = data.get('post_id')
    content = data.get('content')
    
    if not post_id or not content:
        return jsonify({'error': 'Missing fields'}), 400
        
    image_url = None
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename != '' and allowed_file(file.filename):
            import uuid
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"reply_{uuid.uuid4()}.{ext}"
            file.save(os.path.join(app.config['FORUM_UPLOAD_FOLDER'], filename))
            image_url = f"/static/uploads/forum/{filename}"

    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute('''
            INSERT INTO forum_comments (post_id, user_id, content, image_url)
            VALUES (?, ?, ?, ?)
        ''', (post_id, session['user_id'], content, image_url))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard_data():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # Params
        period = request.args.get('period', 'day')
        game_type = request.args.get('game_type', 'all')
        dims = request.args.get('board_dimensions', 'all') 
        time_limit = request.args.get('time_limit', 'all')

        # Base filters
        params = []
        # Exclude Guests from leaderboards
        where_clauses = ["u.username NOT LIKE 'Guest_%'"] 

        if game_type != 'all':
            where_clauses.append("rh.game_type = ?")
            params.append(game_type)
        if dims != 'all':
             where_clauses.append("rh.board_dimensions = ?")
             params.append(dims)
             
        if time_limit != 'all':
             where_clauses.append("rh.round_duration = ?")
             params.append(time_limit)
        else:
             # Universally exclude 24h (86400) from all generic aggregated views
             where_clauses.append("rh.round_duration != 86400")
             
             # Exclude 10m (600) from generic aggregated views for Accumulative 
             if game_type == 'all' or game_type == 'accumulative':
                where_clauses.append("(rh.game_type != 'accumulative' OR rh.round_duration != 600)")

        # Time Filter - Calendar Day logic
        period_clause = "1=1"
        if period == 'day':
             period_clause = "date(rh.timestamp, 'localtime') = date('now', 'localtime')"
        elif period == 'week':
             period_clause = "rh.timestamp >= datetime('now', '-7 days', 'localtime')"
        elif period == 'month':
             period_clause = "rh.timestamp >= datetime('now', '-30 days', 'localtime')"
        elif period == 'year':
             period_clause = "rh.timestamp >= datetime('now', '-365 days', 'localtime')"
        
        where_clauses.append(period_clause)
        base_where = " AND ".join(where_clauses)
        
        # 1. Best Scores (Highest total score in a round - Max 1 per user)
        scores = conn.execute(f"""
            SELECT * FROM (
                SELECT rh.total_score, rh.user_rating, u.username, u.country_flag, u.avatar_url, rh.room_id, rh.round_number, rh.timestamp, rh.board_json, rh.words_json, rh.round_duration, rh.id, rh.game_type,
                ROW_NUMBER() OVER (PARTITION BY rh.user_id ORDER BY rh.total_score DESC) as rn
                FROM round_history rh
                JOIN users u ON rh.user_id = u.id
                WHERE {base_where}
            ) sub
            WHERE rn = 1
            ORDER BY total_score DESC
            LIMIT 50
        """, params).fetchall()
        
        # 2. Best Words (Highest point single word - Max 1 per user)
        words = conn.execute(f"""
            SELECT * FROM (
                SELECT rh.best_word, rh.best_word_score, u.username, u.country_flag, u.avatar_url, rh.room_id, rh.round_number, rh.timestamp, rh.board_json, rh.words_json, rh.round_duration, rh.id, rh.game_type,
                ROW_NUMBER() OVER (PARTITION BY rh.user_id ORDER BY rh.best_word_score DESC) as rn
                FROM round_history rh
                JOIN users u ON rh.user_id = u.id
                WHERE {base_where} AND rh.best_word IS NOT NULL
            ) sub
            WHERE rn = 1
            ORDER BY best_word_score DESC
            LIMIT 50
        """, params).fetchall()
        
        # 3. Best PE (Highest Performance Efficiency - Max 1 per user)
        pes = conn.execute(f"""
            SELECT * FROM (
                SELECT rh.performance_ratio, rh.total_score, u.username, u.country_flag, u.avatar_url, rh.room_id, rh.round_number, rh.timestamp, rh.board_json, rh.words_json, rh.round_duration, rh.id, rh.game_type,
                ROW_NUMBER() OVER (PARTITION BY rh.user_id ORDER BY rh.performance_ratio DESC) as rn
                FROM round_history rh
                JOIN users u ON rh.user_id = u.id
                WHERE {base_where} AND rh.performance_ratio > 0
            ) sub
            WHERE rn = 1
            ORDER BY performance_ratio DESC
            LIMIT 50
        """, params).fetchall()
        
        # 4. Best Ratings Achieved (Max achieved in period - One per user)
        # Note: We group by user_id to get one entry per user
        ratings = conn.execute(f"""
            SELECT MAX(rh.user_rating) as max_rating, u.username, u.country_flag, u.avatar_url, rh.room_id, rh.timestamp, rh.game_type
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
        # Showing the rating for the specific mode if filtered, else global
        if game_type != 'all':
            rating_pattern = f"{game_type}|%"
            if dims != 'all' and time_limit != 'all': rating_pattern = f"{game_type}|{dims}|{time_limit}"
            elif dims != 'all': rating_pattern = f"{game_type}|{dims}|%"
            elif time_limit != 'all': rating_pattern = f"{game_type}|%|{time_limit}"

            # 24-hour configurations exception: load global rating from users table
            is_24h_filter = (time_limit != 'all' and int(time_limit) >= 7200)
            rating_subquery = "u.rating" if is_24h_filter else "COALESCE((SELECT MAX(rating) FROM user_ratings WHERE user_id = u.id AND config_key LIKE ?), 1200)"

            m_sql = f"""SELECT u.username, u.country_flag, u.avatar_url, MAX(rh.timestamp) as last_active,
                               COUNT(rh.id) as game_count,
                               {rating_subquery} as rating,
                               rh.game_type
                        FROM round_history rh 
                        JOIN users u ON rh.user_id = u.id 
                        WHERE {base_where} 
                        GROUP BY u.id 
                        ORDER BY game_count DESC LIMIT 50"""
            m_params = [rating_pattern] + params if not is_24h_filter else params
        else:
            # For 'All Game Types', show the highest rating among modes actually played in this period
            m_sql = f"""SELECT u.username, u.country_flag, u.avatar_url, MAX(rh.timestamp) as last_active,
                               COUNT(rh.id) as game_count,
                               (SELECT MAX(rating) FROM user_ratings 
                                WHERE user_id = u.id 
                                AND config_key IN (
                                    SELECT DISTINCT (game_type || '|' || board_dimensions || '|' || round_duration)
                                    FROM round_history 
                                    WHERE user_id = u.id AND {period_clause}
                                )) as rating,
                               rh.game_type
                        FROM round_history rh 
                        JOIN users u ON rh.user_id = u.id 
                        WHERE {base_where} 
                        GROUP BY u.id 
                        ORDER BY game_count DESC LIMIT 50"""
            m_params = params

        most_games = conn.execute(m_sql, m_params).fetchall()
        
        # 7. Current Ratings (Users active in period, sorted by CURRENT rating)
        if game_type != 'all':
            rating_pattern = f"{game_type}|%"
            if dims != 'all' and time_limit != 'all': rating_pattern = f"{game_type}|{dims}|{time_limit}"
            elif dims != 'all': rating_pattern = f"{game_type}|{dims}|%"
            elif time_limit != 'all': rating_pattern = f"{game_type}|%|{time_limit}"

            current_ratings = conn.execute(f"""
                SELECT u.username, u.country_flag, u.avatar_url, MAX(rh.timestamp) as last_active,
                COALESCE((SELECT MAX(rating) FROM user_ratings WHERE user_id = u.id AND config_key LIKE ?), 1200) as rating,
                rh.game_type
                FROM round_history rh
                JOIN users u ON rh.user_id = u.id
                WHERE {base_where}
                GROUP BY u.id
                ORDER BY rating DESC
                LIMIT 1000
            """, [rating_pattern] + params).fetchall()
        else:
            # For 'All Game Types', show the highest rating among modes actually played in this period
            current_ratings = conn.execute(f"""
                SELECT u.username, u.country_flag, u.avatar_url, MAX(rh.timestamp) as last_active,
                COALESCE((SELECT MAX(rating) FROM user_ratings 
                 WHERE user_id = u.id 
                 AND config_key IN (
                     SELECT DISTINCT (game_type || '|' || board_dimensions || '|' || round_duration)
                     FROM round_history 
                     WHERE user_id = u.id AND {period_clause}
                 )), 1200) as rating,
                rh.game_type
                FROM round_history rh
                JOIN users u ON rh.user_id = u.id
                WHERE {base_where}
                GROUP BY u.id
                ORDER BY rating DESC
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

# --- TOURNAMENT ENDPOINTS ---

@app.route('/api/tournament/status', methods=['GET'])
def get_tournament_status():
    if 'user_id' in session:
        room_manager.update_presence(session['user_id'])
    
    # Update state before returning
    tournament_manager.update_tournament_status()
    
    t = tournament_manager.get_current_tournament()
    user_status = {'status': 'not_joined', 'has_turn': False}
    
    if t and 'user_id' in session and not session.get('is_guest'):
        user_id = session['user_id']
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        p = conn.execute('SELECT * FROM tournament_participants WHERE tournament_id = ? AND user_id = ?', 
                        (t['id'], user_id)).fetchone()
        
        if p:
            user_status['status'] = p['status']
            user_status['final_rank'] = p['final_rank']
            user_status['has_turn'] = tournament_manager.has_user_turn(t['id'], user_id)
            user_status['matchup'] = tournament_manager.get_user_matchup(t['id'], t['current_round'], user_id)
        conn.close()
        
    history = tournament_manager.get_history()
    
    # Get round end time if active
    round_end_time = 0
    if t and t['status'] == 'active':
        conn = sqlite3.connect(DB_PATH, timeout=30)
        r = conn.execute('SELECT end_time FROM tournament_rounds WHERE tournament_id = ? AND round_number = ?',
                        (t['id'], t['current_round'])).fetchone()
        conn.close()
        if r: round_end_time = r[0]

    round_scores = []
    if t and t['status'] == 'active':
        raw_scores = tournament_manager.get_round_scores(t['id'], t['current_round'])
        for rs in raw_scores:
            if rs.get('submitted_words'):
                rs['submitted_words'] = json.loads(rs['submitted_words'])
            round_scores.append(rs)

    conn = sqlite3.connect(DB_PATH, timeout=30)
    total_participants = conn.execute('SELECT COUNT(*) FROM tournament_participants WHERE tournament_id = ?', (t['id'],)).fetchone()[0]
    
    params = json.loads(t['parameters'])
    if t['status'] == 'active':
        # Fetch current round uniqueness for the spinner set display
        r_data = conn.execute('SELECT board_data FROM tournament_rounds WHERE tournament_id = ? AND round_number = ?',
                             (t['id'], t['current_round'])).fetchone()
        if r_data:
            try:
                board_meta = json.loads(r_data[0])
                params['uniqueness_ratio'] = board_meta.get('uniqueness_ratio', 0.0)
            except:
                params['uniqueness_ratio'] = 0.0
    conn.close()

    return jsonify({
        'status': t['status'],
        'id': t['id'],
        'start_date': t['start_date'],
        'parameters': params,
        'current_round': t['current_round'],
        'completed_at': t['completed_at'],
        'round_end_time': round_end_time,
        'user_status': user_status,
        'history': history,
        'round_scores': round_scores,
        'standings': tournament_manager.get_tournament_standings(t['id']),
        'all_matchups': tournament_manager.get_all_matchups(t['id'], t['current_round']) if t['status'] == 'active' else [],
        'total_participants': total_participants,
        'is_guest': session.get('is_guest', False)
    })

@app.route('/api/tournament/join', methods=['POST'])
@login_required
def join_tournament():
    if session.get('is_guest'):
        return jsonify({'error': 'Guests cannot participate in tournaments'}), 403
        
    t = tournament_manager.get_current_tournament()
    if t['status'] != 'signup':
        return jsonify({'error': 'Signup period is over'}), 400
        
    user_id = session['user_id']
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute('''
            INSERT OR IGNORE INTO tournament_participants (tournament_id, user_id, joined_at)
            VALUES (?, ?, ?)
        ''', (t['id'], user_id, time.time()))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/tournament/forfeit', methods=['POST'])
@login_required
def forfeit_tournament():
    if session.get('is_guest'):
        return jsonify({'error': 'Guest access denied'}), 403
        
    t = tournament_manager.get_current_tournament()
    user_id = session['user_id']
    
    success = tournament_manager.forfeit_turn(t['id'], t['current_round'], user_id)
    return jsonify({'success': success})

@app.route('/tournament/game')
@login_required
def tournament_game_page():
    if session.get('is_guest'):
        return redirect('/')
    return render_template('index.html') # The frontend will handle the specific layout via JS

@app.route('/api/tournament/game-state', methods=['GET'])
@login_required
def get_tournament_game_state():
    if session.get('is_guest'):
        return jsonify({'error': 'Guest access denied'}), 403
        
    t = tournament_manager.get_current_tournament()
    if t['status'] != 'active':
        return jsonify({'error': 'No active tournament'}), 400
        
    user_id = session['user_id']
    if not tournament_manager.has_user_turn(t['id'], user_id):
        return jsonify({'error': 'Not your turn or already played'}), 403
        
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    r = conn.execute('SELECT * FROM tournament_rounds WHERE tournament_id = ? AND round_number = ?',
                    (t['id'], t['current_round'])).fetchone()
    conn.close()
    
    if not r:
        return jsonify({'error': 'Round data not found'}), 404
        
    params = json.loads(t['parameters'])
    board_raw = json.loads(r['board_data'])
    
    # Support new dict format OR legacy list format
    if isinstance(board_raw, dict):
        board = board_raw.get('board')
        bonus_word = board_raw.get('bonus_word', '')
        all_words = board_raw.get('all_words', [])
    else:
        board = board_raw
        bonus_word = ''
        all_words = []
    
    return jsonify({
        'tournament_id': t['id'],
        'round_number': t['current_round'],
        'board': board,
        'bonus_word': bonus_word,
        'bonus_cell': board_raw.get('bonus_cell') if isinstance(board_raw, dict) else None,
        'board_format': board_raw.get('board_format', 'Normal') if isinstance(board_raw, dict) else 'Normal',
        'all_words': all_words,
        'params': params,
        'end_time': r['end_time'],
        'server_time': time.time()
    })

@app.route('/api/tournament/winner-turn/<int:tid>/<username>', methods=['GET'])
def get_tournament_winner_turn(tid, username):
    data = tournament_manager.get_winner_turn_data(tid, username)
    if not data:
        return jsonify({'error': 'Winner turn data not found'}), 404
    return jsonify(data)

@app.route('/api/tournament/submit', methods=['POST'])
@login_required
def submit_tournament_score():
    if session.get('is_guest'):
        return jsonify({'error': 'Guest access denied'}), 403
        
    data = request.json
    tid = data.get('tournament_id')
    round_num = data.get('round_number')
    words_data = data.get('words', []) # Now objects with 'word', 'points', 'timestamp'
    round_start_time = data.get('round_start_time', time.time())
    
    user_id = session['user_id']
    
    if not tournament_manager.has_user_turn(tid, user_id):
        return jsonify({'error': 'Invalid turn or already submitted'}), 403
        
    # FETCH ROUND DATA for validation
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    t = conn.execute('SELECT * FROM tournaments WHERE id = ?', (tid,)).fetchone()
    r = conn.execute('SELECT * FROM tournament_rounds WHERE tournament_id = ? AND round_number = ?',
                    (tid, round_num)).fetchone()
    conn.close()
    
    if not t or not r:
        return jsonify({'error': 'Tournament or round not found'}), 404
        
    params = json.loads(t['parameters'])
    board_raw = json.loads(r['board_data'])
    
    # NEW: Handle dict format with real bonus_word
    if isinstance(board_raw, dict):
        board = board_raw.get('board')
        target_bonus_word = board_raw.get('bonus_word', '').upper()
        # Fallback to length-only if word is missing but length is provided
        bonus_len_target = params.get('bonus_word_length', 0)
    else:
        board = board_raw
        target_bonus_word = None # Legacy tournaments don't have a specific word
        bonus_len_target = params.get('bonus_word_length', 0)

    dict_name = params.get('dictionary', 'NWL')
    min_len = params.get('min_word_length', 3)
    
    # LOAD DICTIONARY
    official_dict = word_validator.load_dictionary(dict_name)
    
    # VALIDATE WORDS & CALCULATE SCORE
    valid_words = []
    total_score = 0
    
    for item in words_data:
        word = item.get('word', '').strip().upper()
        if not word: continue
        if len(word) < min_len: continue
        if word not in official_dict: continue
        
        # Verify on board
        if not word_validator.find_word_on_board(board, word): continue
        
        # Check uniqueness in valid_words to avoid double scoring
        if not any(v['word'] == word for v in valid_words):
            # Scoring (Base)
            pts = len(word)
            if len(word) == 6: pts = 10
            elif len(word) == 7: pts = 15
            elif len(word) >= 8: pts = 25
            
            # --- Bonus Point Logic for Tournaments ---
            is_bonus = False
            fmt_low = str(board_raw.get('board_format', 'Normal')).lower() if isinstance(board_raw, dict) else 'normal'
            bonus_word_target = target_bonus_word.upper() if target_bonus_word else ""
            bonus_cell = board_raw.get('bonus_cell') if isinstance(board_raw, dict) else None
            
            # 1. Hidden Bonus Word (+Length)
            if bonus_word_target and word == bonus_word_target:
                pts += len(word)
                is_bonus = True
            
            # 2. Special Format Tiles (+3)
            # Find path and hit bonus cell if format uses them
            if bonus_cell and ('bonus letter' in fmt_low or 'either' in fmt_low):
                # We reuse word_validator.find_word_on_board which now returns (found, path)
                found, path = word_validator.find_word_on_board(board, word, return_path=True)
                if found and path:
                    # bx, by = bonus_cell['r'], bonus_cell['c'] ? or list [r, c]?
                    # Standardizing coordinate access
                    bx = bonus_cell[0] if isinstance(bonus_cell, (list, tuple)) else bonus_cell.get('r', -1)
                    by = bonus_cell[1] if isinstance(bonus_cell, (list, tuple)) else bonus_cell.get('c', -1)
                    
                    if any((p[0] == bx and p[1] == by) for p in path):
                        pts += 3
                    elif 'either' in fmt_low:
                        # On either/or, hit ANY either/or cell
                        if any('/' in str(board[p[0]][p[1]]) for p in path):
                            pts += 3
            elif 'either' in fmt_low:
                # Fallback if no specific bonus_cell but is Either/Or format
                found, path = word_validator.find_word_on_board(board, word, return_path=True)
                if found and path and any('/' in str(board[p[0]][p[1]]) for p in path):
                    pts += 3
                
            valid_words.append({
                'word': word,
                'points': pts,
                'timestamp': item.get('timestamp', time.time()),
                'is_bonus': is_bonus
            })
            total_score += pts
            
    total_score = max(0, total_score)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute('''
            INSERT INTO tournament_scores (tournament_id, round_number, user_id, score, submitted_words, submitted_at, round_start_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (tid, round_num, user_id, total_score, json.dumps(valid_words), time.time(), round_start_time))
        conn.commit()
        return jsonify({'success': True, 'score': total_score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()



@app.route('/api/solo-match/create', methods=['POST'])
def create_solo_match():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
        
    data = request.json
    parameters = data.get('parameters', {})
    participants = data.get('participants', [])
    
    print(f"[app.py] Creating Solo Match for {session.get('username')} (ID: {session.get('user_id')})")
    print(f"[app.py] Params: {parameters}")
    print(f"[app.py] Bots: {participants}")
    
    # 1. Create a GameRoom for Solo Practice
    # Force a unique ID and mark as private to skip singleton logic
    room_id = f"practice_{session['username']}_{int(time.time())}"
    game_type = 'solo_accumulative' # Isolated mode for solo play
    time_limit = int(parameters.get('time_limit', 60))
    board_dimensions = parameters.get('board_dimensions', '4x4')
    
    room = room_manager.create_room(room_id, game_type, time_limit, board_dimensions, is_private=True)
    room.is_solo = True # Disables history and statistics
    room.initial_solo_params = dict(parameters)
    
    # 2. Configure Parameters (Randomize by default if not strictly specified)
    dict_name = parameters.get('dictionary')
    if not dict_name or dict_name == 'random':
        from spinner_set import SpinnerSet
        dict_name = SpinnerSet._spin_dictionary()
    
    board_format = parameters.get('board_format', 'Normal')
    from spinner_set import SpinnerSet

    # First-round difficulty randomization
    target_difficulty = parameters.get('difficulty', 'random')
    if target_difficulty == 'random':
        from spinner_set import SpinnerSet
        target_difficulty = SpinnerSet._spin_difficulty()

    # Point range / word count: allow user to specify, else spin a default
    custom_word_count_range = parameters.get('word_count_range', 'random')
    min_word_len = int(parameters.get('min_word_length', 3))
    if custom_word_count_range == 'random':
        from spinner_set import SpinnerSet
        wc_range = SpinnerSet._spin_word_count(dict_name, min_word_len, target_difficulty, board_dimensions)
    else:
        # Use custom range provided by user
        wc_range = custom_word_count_range

    # Check if the user wants randomization per round
    room.randomize_spinner = (
        parameters.get('dictionary') == 'random' or
        parameters.get('difficulty', 'random') == 'random' or
        parameters.get('word_count_range', 'random') == 'random'
    )

    room.spinner_params = {
        'dictionary': dict_name,
        'min_word_length': int(parameters.get('min_word_length', 3)),
        'bonus_word_length': int(parameters.get('bonus_word_length', 8)),
        'board_format': board_format,
        'difficulty': target_difficulty,
        'word_count_range': wc_range
    }
    
    # Cleanup only if NOT in this room
    cleanup_user_rooms(session['user_id'], exclude_room_id=room_id)
    
    # Add Player
    rating = 1200
    if not session.get('is_guest'):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30)
            # Use mode-specific rating for initial entry to avoid global rating bleed
            display_game_type = game_type.replace('solo_', '')
            config_key = f"{display_game_type}|{board_dimensions}|{time_limit}"
            # 24-hour rooms exception: load global rating from users table
            is_24h = (time_limit >= 7200)
            if is_24h:
                cur = conn.execute('SELECT rating FROM users WHERE id = ?', (session['user_id'],))
                row = cur.fetchone()
                if row:
                    rating = row[0]
                else:
                    rating = 1200
            else:
                cur = conn.execute('SELECT rating FROM user_ratings WHERE user_id = ? AND config_key = ?', 
                                 (session['user_id'], config_key))
                row = cur.fetchone()
                if row: 
                    rating = row[0]
                else:
                     # Every room starts the user at 1200, completely unique to this room configuration
                     rating = 1200
            conn.close()
        except Exception as e:
            print(f"[app.py] DB Error fetching rating for solo: {e}")
            pass
        
    room.add_player(session['user_id'], session['username'], rating, is_guest=session.get('is_guest', False))
    print(f"[app.py] Added human player {session.get('username')} to room {room_id}")
    
    # Add Bots
    import random
    for bot in participants:
        if bot.get('is_ai'):
            ai_id = -random.randint(10000, 999999)
            room.add_player(ai_id, bot['username'], bot.get('ai_rating', 1200))
            p = room.get_player(ai_id)
            if p:
                p.is_ai = True
                p.ai_rating = bot.get('ai_rating', 1200)
                print(f"[app.py] Added AI bot {bot['username']} to room {room_id}")

    # Start loop
    # Important: Room is now fully configured, start_round will use these params
    print(f"[app.py] Starting first round for solo room: {room_id}")
    import threading
    thread = threading.Thread(target=room_manager.start_round, args=(room_id,), daemon=True)
    thread.start()
    
    print(f"[app.py] Solo match creation complete. Returning success to client.")
    return jsonify({'success': True, 'room_id': room_id})

# --- PRIVATE MATCHES ---

@app.route('/api/private-match/create', methods=['POST'])
@login_required
def create_private_match():
    if session.get('is_guest'):
        return jsonify({'error': 'Guests cannot use this feature'}), 403
        
    data = request.json
    match_type = data.get('match_type')
    parameters = data.get('parameters', {})
    
    participants = data.get('participants', [])
    
    try:
        match_id = private_match_manager.create_match(session['user_id'], match_type, parameters, participants)
        return jsonify({'success': True, 'match_id': match_id})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"Failed to create match: {str(e)}"
        print(f"DEBUG: Create Match Error Local State - match_type={match_type}, parameters={parameters}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/private-match/list', methods=['GET'])
@login_required
def list_private_matches():
    if session.get('is_guest'):
        return jsonify({'your_turn': [], 'their_turn': [], 'history': []})
        
    matches = private_match_manager.get_matches_for_user(session['user_id'], session['username'])
    return jsonify(matches)

@app.route('/api/private-match/invites', methods=['GET'])
@login_required
def list_private_match_invites():
    if session.get('is_guest'):
        return jsonify([])
    
    invites = private_match_manager.get_invites_for_user(session['username'])
    return jsonify(invites)

@app.route('/api/private-match/status/<int:match_id>', methods=['GET'])
@login_required
def get_private_match_status(match_id):
    # Get current round info, board, etc.
    conn = private_match_manager.get_db()
    conn.row_factory = sqlite3.Row
    m = conn.execute('SELECT * FROM private_matches WHERE id = ?', (match_id,)).fetchone()
    if not m:
        conn.close()
        return jsonify({'error': 'Match not found'}), 404
        
    round_num = m['current_round']
    if round_num == 0:
        conn.close()
        return jsonify({'error': 'Match is still being initialized. Please wait.'}), 202

    r = conn.execute('SELECT * FROM private_match_rounds WHERE match_id = ? AND round_number = ?', (match_id, round_num)).fetchone()
    
    # Check if user is participant
    p = conn.execute('SELECT 1 FROM private_match_players WHERE match_id = ? AND user_id = ?', (match_id, session['user_id'])).fetchone()
    if not p:
        conn.close()
        return jsonify({'error': 'Not a participant'}), 403
        
    if not r:
        conn.close()
        return jsonify({'error': f'Round {round_num} board data not found. It may still be generating.'}), 202

    conn.close()
    
    # Merge round-specific parameters if available
    params = json.loads(m['parameters'])
    if r['word_count_range']:
        try:
            params['word_count_range'] = json.loads(r['word_count_range'])
        except:
            pass
    if r['board_format']:
        params['board_format'] = r['board_format']
    try:
        if 'dictionary' in r.keys() and r['dictionary']:
            params['dictionary'] = r['dictionary']
    except:
        pass
    try:
        if 'difficulty' in r.keys() and r['difficulty']:
            params['difficulty'] = r['difficulty']
    except:
        pass

    # NEW: Record/Retrieve the persistent turn start time for this user
    # This prevents them from resetting the timer by leaving and re-entering the match.
    start_time = private_match_manager.record_start_time(match_id, round_num, session['user_id'])
    time_limit = params.get('time_limit', 60)
    calculated_end_time = start_time + time_limit
    now = time.time()
    time_remaining = max(0, calculated_end_time - now)

    return jsonify({
        'match_id': match_id,
        'current_round': round_num,
        'parameters': params,
        'board': json.loads(r['board_data']),
        'bonus_word': r['bonus_word'],
        'bonus_cell': json.loads(r['bonus_cell']) if r['bonus_cell'] else None,
        'end_time': calculated_end_time,
        'time_remaining': time_remaining
    })

@app.route('/api/private-match/submit', methods=['POST'])
@login_required
def submit_private_match_turn():
    try:
        data = request.json
        match_id = data.get('match_id')
        round_number = data.get('round_number')
        words_data = data.get('words', [])
        
        # RECALCULATE SCORE on the server for safety and consistency with scoring.py
        conn = private_match_manager.get_db()
        m = conn.execute('SELECT parameters FROM private_matches WHERE id = ?', (match_id,)).fetchone()
        r = conn.execute('SELECT board_data, bonus_word, bonus_cell, dictionary FROM private_match_rounds WHERE match_id = ? AND round_number = ?', (match_id, round_number)).fetchone()
        conn.close()
        
        if not m or not r:
            return jsonify({'error': 'Match or round data not found'}), 404
            
        params = json.loads(m['parameters'])
        board = json.loads(r['board_data'])
        bonus_word = r['bonus_word']
        bonus_cell_raw = r['bonus_cell']
        bonus_cell = json.loads(bonus_cell_raw) if bonus_cell_raw else None
        fmt = params.get('board_format', 'Normal')
        
        valid_words = []
        total_score = 0
        
        # Dictionary for validation
        dict_name = params.get('dictionary', 'NWL')
        try:
            if 'dictionary' in r.keys() and r['dictionary']:
                dict_name = r['dictionary']
        except:
            pass
        official_dict = word_validator.load_dictionary(dict_name)
        
        for item in words_data:
            word = item.get('word', '').strip().upper()
            if not word or word in [v['word'] for v in valid_words]:
                continue
            
            # Basic validation
            if len(word) < params.get('min_word_length', 3):
                continue
            is_on_board = word_validator.find_word_on_board(board, word)
                
            pts = 0
            is_bonus = False
            details = None
            if word in official_dict and is_on_board:
                res = calculate_word_score(word, bonus_word=bonus_word, board_format=fmt, board=board, bonus_cell=bonus_cell, is_private=True, return_details=True)
                pts = res['total']
                details = res
                is_bonus = (bonus_word and word == bonus_word.upper())
            elif 'penalty' in fmt.lower() and is_on_board:
                pts = -3 # Penalty for words on board but not in dict
                details = {'total': -3, 'base': -3, 'bonus_word_points': 0, 'bonus_letter_points': 0}
                
            if pts == 0 and not (is_on_board and 'penalty' in fmt.lower()):
                continue # Skip invalid words that aren't penalties

            valid_words.append({
                'word': word,
                'points': pts,
                'is_bonus': is_bonus,
                'score_details': details,
                'timestamp': item.get('timestamp', time.time())
            })
            total_score += pts
            if total_score < 0:
                total_score = 0
        
        private_match_manager.submit_turn(match_id, round_number, session['user_id'], valid_words, total_score)
        return jsonify({'success': True, 'score': total_score})
    except Exception as e:
        print(f"Submit Turn Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/private-match/invite/accept', methods=['POST'])
@login_required
def accept_match_invite():
    data = request.json
    invite_id = data.get('invite_id')
    action = data.get('action', 'accept')
    
    conn = private_match_manager.get_db()
    invite = conn.execute('SELECT * FROM match_invites WHERE id = ? AND recipient_username = ?', (invite_id, session['username'])).fetchone()
    if invite:
        if action == 'accept':
            match_id = invite[1]
            conn.execute('INSERT OR IGNORE INTO private_match_players (match_id, user_id, username) VALUES (?, ?, ?)',
                        (match_id, session['user_id'], session['username']))
        
        # Always delete invite after action
        conn.execute('DELETE FROM match_invites WHERE id = ?', (invite_id,))
        conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/private-match/rematch', methods=['POST'])
@login_required
def rematch_private_match():
    data = request.json
    old_match_id = data.get('match_id')
    
    conn = private_match_manager.get_db()
    conn.row_factory = sqlite3.Row
    
    # 1. Get Old Match Params
    old_match = conn.execute('SELECT * FROM private_matches WHERE id = ?', (old_match_id,)).fetchone()
    if not old_match:
        conn.close()
        return jsonify({'error': 'Match not found'}), 404
        
    parameters = json.loads(old_match['parameters'])
    
    # 2. Get Old Participants (excluding creator if they are the one requesting, to avoid dupe, but create_match handles creator separate)
    # Actually create_match expects a list of OTHER participants.
    # We need to find everyone who was in the old match EXCEPT the current user (who will be the new creator).
    
    old_players = conn.execute('SELECT * FROM private_match_players WHERE match_id = ?', (old_match_id,)).fetchall()
    
    participants = []
    for p in old_players:
        if p['user_id'] == session['user_id']:
            continue # Skip self, will be added as creator
            
        part = {
            'username': p['username'],
            'is_ai': bool(p['is_ai']),
            'ai_rating': p['ai_rating']
        }
        participants.append(part)
        
    conn.close()
    
    # 3. Create New Match
    try:
        new_match_id = private_match_manager.create_match(
            creator_id=session['user_id'],
            match_type='with_friends',
            parameters=parameters,
            participants=participants
        )
        return jsonify({'success': True, 'new_match_id': new_match_id})
    except Exception as e:
        print(f"Rematch Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/private-match/history/<int:match_id>', methods=['GET'])
@login_required
def get_private_match_history(match_id):
    conn = private_match_manager.get_db()
    conn.row_factory = sqlite3.Row
    # Fetch turns for all rounds for this match
    turns = conn.execute('''
        SELECT t.*, u.username, r.board_data, r.bonus_word, r.round_number
        FROM private_match_turns t
        JOIN private_match_rounds r ON t.match_id = r.match_id AND t.round_number = r.round_number
        LEFT JOIN users u ON t.user_id = u.id
        WHERE t.match_id = ?
        ORDER BY t.round_number DESC, t.score DESC
        LIMIT 25
    ''', (match_id,)).fetchall()
    
    # We might have AI bots, their IDs are negative and not in users table
    # Let's fix usernames for AI
    results = []
    players = conn.execute('SELECT user_id, username FROM private_match_players WHERE match_id = ?', (match_id,)).fetchall()
    p_map = {p['user_id']: p['username'] for p in players}
    
    for row in turns:
        d = dict(row)
        d['username'] = p_map.get(d['user_id'], d['username'] or "AI Bot")
        d['submitted_words'] = json.loads(d['submitted_words'])
        d['board'] = json.loads(d['board_data'])
        results.append(d)
        
    conn.close()
    return jsonify(results)

def room_tick_worker():
    """Background worker to advance room states without needing a client request."""
    from game_room import room_manager
    while True:
        try:
            # Create a copy of values to avoid concurrent modification errors
            rooms = list(room_manager.rooms.values())
            for room in rooms:
                # 1. Update state (active -> intermission transitions)
                room.check_and_update_state()
                
                # 2. Progress through intermission milestones
                if room.state == 'intermission':
                    milestone = room.get_next_round_milestone()
                    if milestone == 'spinner':
                        room_manager.generate_spinner_params(room.room_id, reveal=False)
                    elif milestone == 'reveal':
                        room_manager.generate_spinner_params(room.room_id, reveal=True)
                    elif milestone == 'search':
                        room_manager.start_board_search(room.room_id)
                    elif milestone == 'start' and not room.starting_round:
                        # Auto-start next round precisely
                        print(f"[RoomTickWorker] Auto-advancing room {room.room_id} to new round")
                        threading.Thread(target=room_manager.start_next_round, args=(room.room_id,), daemon=True).start()
            
        except Exception as e:
            import traceback
            print(f"[RoomTickWorker] CRITICAL ERROR: {e}")
            traceback.print_exc()
            time.sleep(10) # Wait longer on error
            
        time.sleep(2) # Polling every 2s for better response

@app.route('/<path:path>')
def static_files(path):
    if path.startswith('api/'):
        return jsonify({'error': 'Resource not found'}), 404
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Background room advancer is now handled by RoomManager's internal thread
    print("[Main] Background Room Advancer consolidated into RoomManager.")

    print('Morpheme server running on http://localhost:5001')
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Server startup error: {e}. Attempting fallback (No Reloader)...")
        # Retry without reloader if it failed due to termios/EINTR issues
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

