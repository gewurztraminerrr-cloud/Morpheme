print(f"[Main] SERVER STARTING - VERSION: 2026-05-05_04:00 (Tally Hardened)")

from flask import Flask, request, jsonify, session, send_from_directory, g, redirect, url_for, render_template, make_response
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import time
import os
import json
import random
import threading
import uuid
import gzip
import io
from collections import Counter
import datetime
import math
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import base64
import requests
from db import get_db, get_db_connection, DB_PATH, execute_with_retry

# Load environment variables from .env file
load_dotenv()

def parse_data_url(data_url):
    """
    Parses a data URL (e.g. "data:image/png;base64,...") and returns (raw_bytes, mime_type).
    """
    if not data_url or not data_url.startswith("data:"):
        return None, None
    try:
        header, encoded = data_url.split(",", 1)
        mime_type = header.split(";")[0].split(":")[1]
        raw_bytes = base64.b64decode(encoded)
        return raw_bytes, mime_type
    except Exception as e:
        print(f"[Moderation] Failed to parse data URL: {e}")
        return None, None

def moderate_content(text=None, image_bytes=None, mime_type=None):
    """
    Moderates content using Google Gemini 1.5 Flash API.
    Supports text and/or image bytes.
    Returns: {"inappropriate": True/False, "reason": "..."}
    """
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        # Fail-open if no key is configured (ideal for local/default setup)
        print("[Moderation] WARNING: GEMINI_API_KEY not set. Moderation bypassed (Fail-Open).")
        return {"inappropriate": False, "reason": "Service unconfigured"}

    # Determine url
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    parts = []
    
    # Base64-encode image if provided
    if image_bytes:
        # Ensure correct mime-type format
        if not mime_type:
            mime_type = "image/jpeg"
        elif not mime_type.startswith("image/"):
            mime_type = f"image/{mime_type}"
            
        b64_data = base64.b64encode(image_bytes).decode('utf-8')
        parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": b64_data
            }
        })
        
    # Append prompt
    prompt = (
        "You are a content moderation system. Analyze if the provided content (text and/or image) "
        "is inappropriate, X-rated, sexually explicit, highly offensive, or harmful.\n"
    )
    if text:
        prompt += f"Text to analyze: {text}\n"
        
    prompt += (
        "Respond with a JSON object containing exactly two fields: "
        "'inappropriate' (boolean) and 'reason' (string explaining why it was flagged, or empty if safe). "
        "Do not include any markdown formatting or extra text in your response, just the raw JSON."
    )
    
    parts.append({"text": prompt})
    
    payload = {
        "contents": [{
            "parts": parts
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"[Moderation] API Error: {response.status_code} - {response.text}")
            # Fail-open on API failure to prevent app disruption
            return {"inappropriate": False, "reason": f"API error: {response.status_code}"}
            
        res_data = response.json()
        
        # Parse output from candidate content
        candidates = res_data.get('candidates', [])
        if not candidates:
            return {"inappropriate": False, "reason": "No candidates returned"}
            
        text_content = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
        parsed_res = json.loads(text_content)
        
        # Normalize keys/values
        is_inappropriate = bool(parsed_res.get('inappropriate', False))
        reason = str(parsed_res.get('reason', ''))
        
        print(f"[Moderation] Check complete. Inappropriate: {is_inappropriate} | Reason: {reason}")
        return {"inappropriate": is_inappropriate, "reason": reason}
        
    except Exception as e:
        print(f"[Moderation] Error during API call: {e}")
        # Fail-open
        return {"inappropriate": False, "reason": f"Error: {e}"}

app = Flask(__name__, static_folder='static')
app.secret_key = 'morpheme-secret-key-2024'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=30)

@app.after_request
def compress(response):
    accept_encoding = request.headers.get('Accept-Encoding', '')
    if (response.status_code < 200 or 
        response.status_code >= 300 or 
        'gzip' not in accept_encoding.lower() or 
        'Content-Encoding' in response.headers):
        return response

    content_type = response.headers.get('Content-Type', '')
    if not any(t in content_type.lower() for t in ['json', 'text', 'javascript', 'css', 'xml', 'svg']):
        return response

    response.direct_passthrough = False
    data = response.get_data()
    if len(data) < 500: # Don't compress small responses
        return response

    gzip_buffer = io.BytesIO()
    gzip_file = gzip.GzipFile(mode='wb', fileobj=gzip_buffer)
    gzip_file.write(data)
    gzip_file.close()

    response.set_data(gzip_buffer.getvalue())
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Length'] = len(response.get_data())
    return response

# --- Leaderboard in-memory TTL cache ---
_lb_cache = {}
_lb_cache_expiry = {}
_LB_CACHE_TTL = 45  # seconds


def format_chicago_to_utc(chicago_ts_str):
    if not chicago_ts_str:
        return None
    try:
        ts_str = str(chicago_ts_str).replace('T', ' ')
        if '.' in ts_str:
            dt = datetime.datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
        else:
            dt = datetime.datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        
        # Historical Fix: Prior to May 29, 2026, the database timestamps were stored
        # in UTC (server system clock local time was UTC, before the Chicago transition on May 28).
        # We treat those as UTC directly to prevent double-conversion timezone offsets.
        if dt < datetime.datetime(2026, 5, 29):
            dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            
        dt = dt.replace(tzinfo=ZoneInfo("America/Chicago"))
        utc_dt = dt.astimezone(datetime.timezone.utc)
        return utc_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        return chicago_ts_str


@app.route('/api/ping')
def ping_debug():
    return jsonify({'pong': True})

@app.route('/api/debug-pwa', methods=['POST'])
def debug_pwa():
    import json
    try:
        data = request.json
        print(f"[PWA-Debug] {json.dumps(data)}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"[PWA-Debug] Error logging PWA debug: {e}")
        return jsonify({'error': str(e)}), 500

from tournament_logic import tournament_manager
from private_match_logic import private_match_manager
from word_validator import word_validator
from scoring import calculate_word_score, get_valued_word_score
from game_room import room_manager, lobby_manager, STATS_PATH
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
    if _MODS_CACHE is not None and time.time() - _MODS_CACHE_TIME < 30:
        return _MODS_CACHE

    mods = {'jeffb', 'system'}
    if os.path.exists(MODS_FILE):
        try:
            with open(MODS_FILE, 'r') as f:
                lines = [line.strip().lower() for line in f if line.strip()]
                mods.update(lines)
        except Exception as e:
            print(f"[Mods] Error reading {MODS_FILE}: {e}")

    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT username FROM moderators").fetchall()
        conn.close()
        for r in rows:
            if r['username'] and r['username'].strip().lower() != 'jeffbabiak':
                mods.add(r['username'].strip().lower())
    except Exception as e:
        pass

    if 'jeffbabiak' in mods:
        mods.remove('jeffbabiak')
    
    _MODS_CACHE = mods
    _MODS_CACHE_TIME = time.time()
    return mods


def save_moderator(username):
    username = username.strip().lower()
    if not username: return False
    mods = get_moderators()
    mods.add(username)
    try:
        with open(MODS_FILE, 'w') as f:
            for mod in sorted(mods):
                f.write(f"{mod}\n")
    except Exception as e:
        print(f"[Mods] Error saving to {MODS_FILE}: {e}")

    try:
        with get_db() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS moderators (username TEXT PRIMARY KEY, added_at REAL)")
            conn.execute("INSERT OR REPLACE INTO moderators (username, added_at) VALUES (?, ?)", (username, time.time()))
    except Exception as e:
        print(f"[Mods] Error saving to DB: {e}")

    global _MODS_CACHE
    _MODS_CACHE = mods
    return True

def remove_moderator(username):
    username = username.strip().lower()
    if username in ('jeffb', 'system'):
        print(f"[Mods] Attempt to remove protected moderator {username} blocked.")
        return False
    mods = get_moderators()
    if username in mods:
        mods.remove(username)
        try:
            with open(MODS_FILE, 'w') as f:
                for mod in sorted(mods):
                    f.write(f"{mod}\n")
        except Exception as e:
            print(f"[Mods] Error removing from {MODS_FILE}: {e}")

        try:
            with get_db() as conn:
                conn.execute("DELETE FROM moderators WHERE username = ?", (username,))
        except Exception as e:
            print(f"[Mods] Error removing from DB: {e}")

        global _MODS_CACHE
        _MODS_CACHE = mods
        return True
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

# ============================================================
# IP TRACKING & BANNING SYSTEM
# ============================================================
BANNED_IPS_CACHE = set()
BANNED_IPS_REASONS = {}
BANNED_USERNAMES_REASONS = {}

def reload_banned_ips():
    """Reloads active banned IPs and usernames from database into in-memory O(1) structures."""
    global BANNED_IPS_CACHE, BANNED_IPS_REASONS, BANNED_USERNAMES_REASONS
    try:
        conn = get_db_connection(DB_PATH, timeout=15.0)
        rows = conn.execute("SELECT ip_address, banned_username, reason FROM ip_bans ORDER BY id ASC").fetchall()
        new_cache = set()
        new_ip_reasons = {}
        new_u_reasons = {}
        for r in rows:
            ip = (r['ip_address'] or '').strip()
            u = (r['banned_username'] or '').strip().lower()
            reason = (r['reason'] or '').strip() or 'Violation of community rules'
            if ip:
                new_cache.add(ip)
                new_ip_reasons[ip] = reason
            if u:
                new_u_reasons[u] = reason
        BANNED_IPS_CACHE = new_cache
        BANNED_IPS_REASONS = new_ip_reasons
        BANNED_USERNAMES_REASONS = new_u_reasons
        conn.close()
        print(f"[IP_BANS] Loaded {len(BANNED_IPS_CACHE)} banned IP(s) and {len(BANNED_USERNAMES_REASONS)} banned username(s) into cache.")
    except Exception as e:
        print(f"[IP_BANS] Error reloading banned IPs: {e}")

def get_client_ip(req=None):
    """Extracts true remote client IP, respecting reverse-proxy headers (Nginx/Cloudflare)."""
    if req is None:
        req = request
    try:
        x_forwarded = req.headers.get('X-Forwarded-For')
        if x_forwarded:
            parts = [p.strip() for p in x_forwarded.split(',') if p.strip()]
            if parts:
                return parts[0]
        x_real = req.headers.get('X-Real-IP')
        if x_real and x_real.strip():
            return x_real.strip()
        return (req.remote_addr or '').strip()
    except Exception:
        return ''

@app.before_request
def check_ip_ban():
    """Blocks all incoming non-static traffic if client IP is present in BANNED_IPS_CACHE."""
    # Skip static files, service worker, audio, images, icons, manifest
    if (request.path.startswith('/static') or 
        request.path in ['/service-worker.js', '/favicon.ico', '/manifest.json'] or 
        any(request.path.endswith(ext) for ext in ('.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.mp3', '.wav', '.json', '.svg', '.woff2', '.woff', '.ttf'))):
        return

    client_ip = get_client_ip()
    if client_ip and client_ip in BANNED_IPS_CACHE:
        ban_reason = BANNED_IPS_REASONS.get(client_ip) or "Violation of community rules"
        if request.path.startswith('/api/'):
            return jsonify({
                'error': f'Your account / IP address has been permanently banned.\nReason: {ban_reason}',
                'banned': True,
                'is_banned': True,
                'ban_reason': ban_reason
            }), 403
        return (
            "<!DOCTYPE html><html><head><title>Access Restricted</title>"
            "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0f172a;color:#f8fafc;"
            "display:flex;align-items:center;justify-content:center;height:100vh;margin:0;padding:20px;box-sizing:border-box;}"
            "div{background:#1e293b;padding:36px;border-radius:16px;border:1px solid #334155;max-width:480px;text-align:center;box-shadow:0 10px 25px rgba(0,0,0,0.5);}"
            "h1{color:#f43f5e;margin-top:0;font-size:1.6rem;}p{line-height:1.6;color:#94a3b8;}strong{color:#f8fafc;}</style></head>"
            f"<body><div><h1>403 - Access Restricted</h1><p>Your account and IP address have been permanently banned from Morpheme.<br><br><strong>Reason:</strong> {ban_reason}</p></div></body></html>",
            403
        )

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

def ensure_guest_session():
    if 'user_id' not in session:
        import random, string
        for attempt in range(10):
            try:
                guest_id = random.randint(10000, 99999)
                guest_username = f'Guest_{guest_id}'
                dummy_password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
                password_hash = generate_password_hash(dummy_password, method='pbkdf2:sha256')
                client_ip = get_client_ip()
                with get_db() as conn:
                    cursor = conn.execute('INSERT INTO users (username, password_hash, registration_ip, last_ip) VALUES (?, ?, ?, ?)', 
                                          (guest_username, password_hash, client_ip, client_ip))
                    new_user_id = cursor.lastrowid
                session['user_id'] = new_user_id
                session['username'] = guest_username
                session['is_guest'] = True
                session['_morpheme_login_time'] = time.time()
                session.permanent = True
                print(f"[AutoGuest] Automatically initialized guest session for user: {guest_username}")
                break
            except Exception as e:
                if 'UNIQUE' in str(e) and attempt < 9:
                    continue
                print(f"[AutoGuest] Error creating guest session (attempt {attempt}): {e}")

@app.before_request
def load_user():
    if 'user_id' not in session and request.path and not request.path.startswith('/static') and not any(request.path.endswith(ext) for ext in ('.js', '.css', '.png', '.jpg', '.ico', '.mp3', '.wav', '.json')):
        ensure_guest_session()
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

@app.after_request
def add_cache_headers(response):
    # Allow aggressive caching for static files to improve performance and enable instant loading on Safari
    if request.endpoint in ('static', 'static_files') or (request.path and any(request.path.endswith(ext) for ext in ('.mp3', '.wav', '.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff2', '.woff', '.ttf'))):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        if "Pragma" in response.headers:
            del response.headers["Pragma"]
        if "Expires" in response.headers:
            del response.headers["Expires"]
        return response
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Auth Helpers

@app.route('/api/mods/status')
def get_mod_status():
    if 'username' not in session:
        return jsonify({'is_mod': False, 'is_root': False, 'username': ''})
    u = session['username']
    return jsonify({
        'is_mod': is_mod(u),
        'is_root': u.strip().lower() == 'jeffb',
        'username': u
    })

@app.route('/api/mods/list', methods=['GET'])
@login_required
def list_mods():
    if not is_mod(session['username']):
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify({'mods': sorted(list(get_moderators()))})

@app.route('/api/mods/add', methods=['POST'])
@login_required
def add_mod():
    current_user = (session.get('username') or '').strip().lower()
    if current_user != 'jeffb':
        return jsonify({'error': 'Unauthorized: Only jeffb can add moderators.'}), 403
    
    data = request.json or {}
    new_mod = data.get('username', '').strip()
    if not new_mod:
        return jsonify({'error': 'Username required'}), 400
    
    if save_moderator(new_mod):
        print(f"[Mods] Root user jeffb added {new_mod} as moderator")
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to save mod'}), 500

@app.route('/api/mods/remove', methods=['POST'])
@login_required
def delete_mod():
    current_user = (session.get('username') or '').strip().lower()
    if current_user != 'jeffb':
        return jsonify({'error': 'Unauthorized: Only jeffb can remove moderators.'}), 403
    
    data = request.json or {}
    target = data.get('username', '').strip()
    if not target:
        return jsonify({'error': 'Username required'}), 400
        
    if target.lower() in ('jeffb', 'system'):
        return jsonify({'error': 'Cannot remove root administrator.'}), 400
    
    if remove_moderator(target):
        print(f"[Mods] Root user jeffb removed {target} from moderators")
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
    if not word_validator.is_valid_word(word, 'ALL + AW', use_added_words=True):
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



ADDED_WORDS_LIST_CACHE = None
LAST_ADDED_WORDS_LIST_MTIME = None

@app.route('/api/added_words/list', methods=['GET'])
def list_added_words_api():
    global ADDED_WORDS_LIST_CACHE, LAST_ADDED_WORDS_LIST_MTIME
    if not os.path.exists(ADDED_WORDS_FILE):
        return jsonify({'words': []})
    try:
        curr_mtime = os.path.getmtime(ADDED_WORDS_FILE)
        if ADDED_WORDS_LIST_CACHE is not None and LAST_ADDED_WORDS_LIST_MTIME == curr_mtime:
            return jsonify({'words': ADDED_WORDS_LIST_CACHE})

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
            
            ADDED_WORDS_LIST_CACHE = unique_words
            LAST_ADDED_WORDS_LIST_MTIME = curr_mtime
            return jsonify({'words': unique_words})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mods/added_words/config', methods=['GET'])
@mod_required
def get_added_words_config():
    return jsonify({
        'use_added_words': word_validator.get_use_added_words(force=True)
    })

@app.route('/api/mods/added_words/toggle', methods=['POST'])
@mod_required
def toggle_added_words():
    data = request.json
    enabled = data.get('enabled', True)
    try:
        new_state = word_validator.toggle_added_words(enabled)
        return jsonify({'success': True, 'use_added_words': new_state})
    except Exception as e:
        import traceback
        print(f"[API] Error toggling added words: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': f"Failed to save configuration: {str(e)}"}), 500

@app.route('/api/mods/added_words/add', methods=['POST'])
@mod_required
def add_added_word_api():
    word = request.json.get('word', '').strip().upper()
    if not word: return jsonify({'error': 'Word required'}), 400
    
    # Reject if already present in standard dictionaries (CSW, NWL, or 16plus)
    if word_validator.is_valid_word_authoritative(word):
        return jsonify({
            'error': f"'{word}' is already a valid word in the official dictionaries (CSW/NWL/16plus).",
            'is_authoritative': True
        }), 400

    # Reject if already present in Added Words list (User Request)
    if word.upper() in word_validator.added_words:
        print(f"[Mods] REJECTED Duplicate: '{word}' is already in the Added Words set.")
        return jsonify({
            'error': f"'{word}' is already present on Added Words list.",
            'is_duplicate': True
        }), 400
        
    try:
        # Update in-memory sets instantly
        word_validator.add_word_in_memory(word)
        
        # Clear local/endpoint caches instantly
        TOOLS_DICT_CACHE.clear()
        LISTS_CACHE.clear()
        
        # Update ADDED_WORDS_LIST_CACHE in-memory so the list API is instantly updated
        global ADDED_WORDS_LIST_CACHE, LAST_ADDED_WORDS_LIST_MTIME
        ADDED_WORDS_LIST_CACHE = list(word_validator.added_words_list)
        LAST_ADDED_WORDS_LIST_MTIME = time.time() + 3600.0 # Prevent reload until thread finishes

        # Spawn asynchronous thread to update files on disk (prevents blocking)
        def save_added_word_async(w):
            try:
                # 1. Update Added Words file
                lines = []
                if os.path.exists(ADDED_WORDS_FILE):
                    with open(ADDED_WORDS_FILE, 'r') as f:
                        lines = [line.strip().upper() for line in f if line.strip()]
                if w not in lines:
                    lines.insert(0, w)
                    with open(ADDED_WORDS_FILE, 'w') as f:
                        for l in lines:
                            f.write(f"{l}\n")
                
                # 2. Sync with Global Tally stats file (heavy I/O)
                _update_word_stats(w, "add")
                
                global LAST_ADDED_WORDS_LIST_MTIME, LAST_ADDED_WORDS_MTIME
                if os.path.exists(ADDED_WORDS_FILE):
                    curr_mtime = os.path.getmtime(ADDED_WORDS_FILE)
                    LAST_ADDED_WORDS_LIST_MTIME = curr_mtime
                    LAST_ADDED_WORDS_MTIME = curr_mtime
                print(f"[AsyncMods] Finished saving new word '{w}' to disk and tally.")
            except Exception as e:
                print(f"[AsyncMods] Error saving '{w}' to disk: {e}")

        import threading
        threading.Thread(target=save_added_word_async, args=(word,), daemon=True).start()
        
        # Trigger dynamic definition mapping and auto-saving rules
        ensure_definitions_background([word])

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
        
    data = request.json or {}
    raw_word = data.get('word', '')
    if isinstance(raw_word, list):
        words = [w.strip().upper() for w in raw_word if w and w.strip()]
    else:
        # Split by comma or whitespace for rapid bulk removal
        words = [w.strip().upper() for w in str(raw_word).replace(',', ' ').split() if w.strip()]
        
    if not words:
        return jsonify({'error': 'Word is required'}), 400
        
    try:
        # Update in-memory sets instantly for all words
        for w in words:
            word_validator.remove_word_in_memory(w)
        
        # Clear local/endpoint caches instantly
        TOOLS_DICT_CACHE.clear()
        LISTS_CACHE.clear()
        
        # Update ADDED_WORDS_LIST_CACHE in-memory so the list API is instantly updated
        global ADDED_WORDS_LIST_CACHE, LAST_ADDED_WORDS_LIST_MTIME
        ADDED_WORDS_LIST_CACHE = list(word_validator.added_words_list)
        LAST_ADDED_WORDS_LIST_MTIME = time.time() + 3600.0 # Prevent reload until thread finishes

        # Spawn asynchronous thread to update files on disk (prevents blocking)
        def remove_added_words_async(word_list):
            try:
                # 1. Update Added Words file
                lines = []
                if os.path.exists(ADDED_WORDS_FILE):
                    with open(ADDED_WORDS_FILE, 'r') as f:
                        lines = [line.strip().upper() for line in f if line.strip()]
                
                remove_set = set(word_list)
                new_lines = [l for l in lines if l not in remove_set]
                if len(new_lines) != len(lines):
                    with open(ADDED_WORDS_FILE, 'w') as f:
                        for l in new_lines:
                            f.write(l + '\n')
                
                # 2. Sync with Global Tally
                for w in word_list:
                    _update_word_stats(w, "remove")
                
                # Update the mtime to the actual new file mtime
                global LAST_ADDED_WORDS_LIST_MTIME, LAST_ADDED_WORDS_MTIME
                if os.path.exists(ADDED_WORDS_FILE):
                    curr_mtime = os.path.getmtime(ADDED_WORDS_FILE)
                    LAST_ADDED_WORDS_LIST_MTIME = curr_mtime
                    LAST_ADDED_WORDS_MTIME = curr_mtime
                print(f"[AsyncMods] Finished removing {len(word_list)} word(s) from disk and tally.")
            except Exception as e:
                print(f"[AsyncMods] Error removing words from disk: {e}")

        import threading
        threading.Thread(target=remove_added_words_async, args=(words,), daemon=True).start()

        msg = f'Word "{words[0]}" removed.' if len(words) == 1 else f'{len(words)} words removed successfully.'
        return jsonify({'success': True, 'message': msg, 'removed_words': words})
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
        target_dict_name = 'custom_nwl.txt' if filename == 'newNWL.txt' else 'custom_csw.txt'
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
            
        global LISTS_CACHE
        LISTS_CACHE.clear()
        
        # Trigger dynamic definitions processing and auto-saving rules in background
        ensure_definitions_background(list(new_words))
            
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
    word_input = data.get('word', '').strip()
    definition = data.get('definition', '').strip()
    
    if not word_input or not definition:
        return jsonify({'error': 'Word and definition required'}), 400
        
    # Split by comma to support multiple words
    words = [w.strip().upper() for w in word_input.split(',')]
    words = [w for w in words if w]
    
    if not words:
        return jsonify({'error': 'Word and definition required'}), 400
        
    global DEFINITIONS_PATH, DEFINITIONS_CACHE
    if not DEFINITIONS_PATH:
        DEFINITIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dictionaries', 'Definitions.txt')
        
    try:
        # Ensure definitions file and directory exist if we are writing to it
        os.makedirs(os.path.dirname(DEFINITIONS_PATH), exist_ok=True)
        if not os.path.exists(DEFINITIONS_PATH):
            with open(DEFINITIONS_PATH, 'w', encoding='utf-8') as f:
                pass
                
        # Load all to memory to rewrite (needed for update/append logic)
        defs = {}
        if DEFINITIONS_PATH and os.path.exists(DEFINITIONS_PATH):
            with open(DEFINITIONS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.split(' - ', 1)
                    if len(parts) == 2:
                        defs[parts[0].strip().upper()] = parts[1].strip()
        
        # Add or Replace for each word (with dynamic resolution support)
        # Store in cache temporarily so format_resolved_definition can look up references
        for word in words:
            DEFINITIONS_CACHE[word] = definition
            
        for word in words:
            formatted_def = format_resolved_definition(word)
            defs[word] = formatted_def or definition
        
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
        DEFINITIONS_CACHE = {} # Force reload
        load_definitions()
        global _UNDEFINED_WORDS_CACHE
        _UNDEFINED_WORDS_CACHE = None
        
        if len(words) > 1:
            msg = f"Definitions for {', '.join(words)} set."
        else:
            msg = f"Definition for {words[0]} set."
            
        return jsonify({'success': True, 'message': msg, 'words': words})
    except Exception as e:
        print(f"Error updating definitions: {e}")
        return jsonify({'error': str(e)}), 500

_UNDEFINED_WORDS_CACHE = None
_UNDEFINED_WORDS_LOCK = threading.Lock()

def compute_undefined_words(force=False):
    global _UNDEFINED_WORDS_CACHE
    with _UNDEFINED_WORDS_LOCK:
        if not force and _UNDEFINED_WORDS_CACHE is not None:
            return _UNDEFINED_WORDS_CACHE
            
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dicts_dir = os.path.join(base_dir, 'dictionaries')

            # Definitions from DEFINITIONS_CACHE (or disk)
            global DEFINITIONS_CACHE
            if not DEFINITIONS_CACHE:
                load_definitions()
            defined_words = set(DEFINITIONS_CACHE.keys()) if DEFINITIONS_CACHE else set()

            # Also load from wiktionary_definitions DB table
            try:
                conn = sqlite3.connect(DB_PATH, timeout=10)
                cursor = conn.cursor()
                cursor.execute("SELECT word FROM wiktionary_definitions;")
                for row in cursor.fetchall():
                    defined_words.add(row[0].strip().upper())
                conn.close()
            except Exception as db_err:
                print(f"[Definition Management] Could not read wiktionary_definitions: {db_err}")

            def _read_wordlist(path):
                words = set()
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                        for ln in fh:
                            w = ln.strip().upper()
                            if w:
                                words.add(w)
                return words

            nwl_words   = _read_wordlist(os.path.join(dicts_dir, 'NWL.txt'))
            csw_words   = _read_wordlist(os.path.join(dicts_dir, 'CSW.txt'))
            long_words  = _read_wordlist(os.path.join(dicts_dir, '16plus.txt'))
            added_words = _read_wordlist(os.path.join(dicts_dir, 'added_words.txt'))
            new_nwl     = _read_wordlist(os.path.join(dicts_dir, 'new_NWL.txt'))
            new_csw     = _read_wordlist(os.path.join(dicts_dir, 'new_CSW.txt'))

            all_vocab = nwl_words | csw_words | long_words | added_words | new_nwl | new_csw

            # All words without a definition, sorted by length (shortest first) then alphabetically
            undefined_words = [w for w in all_vocab if w not in defined_words]
            undefined_words.sort(key=lambda x: (len(x), x))

            _UNDEFINED_WORDS_CACHE = {
                'success': True,
                'words': undefined_words,
                '_debug': {
                    'total_vocab_count': len(all_vocab),
                    'defined_count': len(defined_words),
                    'undefined_count': len(undefined_words),
                }
            }
            return _UNDEFINED_WORDS_CACHE
        except Exception as e:
            print(f"Error computing undefined words: {e}")
            return {'success': False, 'error': str(e), 'words': []}

@app.route('/api/mods/definitions/undefined', methods=['GET'])
@login_required
def get_undefined_words_api():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
        
    force = request.args.get('force', 'false').lower() == 'true'
    result = compute_undefined_words(force=force)
    status_code = 200 if result.get('success') else 500
    return jsonify(result), status_code

@app.route('/api/mods/definitions/remove', methods=['POST'])
@login_required
def remove_definition_api():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    word_input = data.get('word', '').strip()
    if not word_input:
        return jsonify({'error': 'Word required'}), 400
        
    # Split by comma to support multiple words
    words = [w.strip().upper() for w in word_input.split(',')]
    words = [w for w in words if w]
    
    if not words:
        return jsonify({'error': 'Word required'}), 400
        
    global DEFINITIONS_PATH
    if not DEFINITIONS_PATH:
        DEFINITIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dictionaries', 'Definitions.txt')
        
    try:
        # Ensure definitions file and directory exist if we are writing to it
        os.makedirs(os.path.dirname(DEFINITIONS_PATH), exist_ok=True)
        if not os.path.exists(DEFINITIONS_PATH):
            with open(DEFINITIONS_PATH, 'w', encoding='utf-8') as f:
                pass
                
        defs = {}
        removed_words = []
        if DEFINITIONS_PATH and os.path.exists(DEFINITIONS_PATH):
            with open(DEFINITIONS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.split(' - ', 1)
                    if len(parts) == 2:
                        k = parts[0].strip().upper()
                        if k in words:
                            removed_words.append(k)
                            continue
                        defs[k] = parts[1].strip()
        
        if not removed_words:
            return jsonify({'error': 'None of the specified words had definitions.'}), 404
            
        sorted_keys = sorted(defs.keys())
        temp_path = DEFINITIONS_PATH + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            for k in sorted_keys:
                f.write(f"{k} - {defs[k]}\n")
        
        os.replace(temp_path, DEFINITIONS_PATH)
        
        global DEFINITIONS_CACHE, _UNDEFINED_WORDS_CACHE
        DEFINITIONS_CACHE = {} # Force reload
        _UNDEFINED_WORDS_CACHE = None
        load_definitions()
        
        if len(removed_words) > 1:
            msg = f"Definitions for {', '.join(removed_words)} removed."
        else:
            msg = f"Definition for {removed_words[0]} removed."
            
        return jsonify({'success': True, 'message': msg, 'words': removed_words})
    except Exception as e:
        print(f"Error removing definition: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mods/ban_user', methods=['POST'])
@login_required
def ban_user_api():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json or {}
    username = (data.get('username') or '').strip()
    custom_reason = (data.get('reason') or '').strip()
    ban_reason = custom_reason if custom_reason else f"Permanent ban: Violation of community rules"
    
    if not username:
        return jsonify({'error': 'Username required'}), 400
        
    if username.lower() in ('jeffbabiak', 'jeffb'):
        return jsonify({'error': 'Cannot ban protected moderator'}), 403

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # Get user ID and recorded IP addresses before deletion
        cursor = conn.execute("SELECT id, registration_ip, last_ip FROM users WHERE username = ? COLLATE NOCASE", (username,))
        row = cursor.fetchone()
        if not row:
            return jsonify({'error': 'User not found'}), 404
        user_id = row['id']
        reg_ip = (row['registration_ip'] or '').strip()
        last_ip = (row['last_ip'] or '').strip()
        
        # Start transaction
        conn.execute("BEGIN TRANSACTION")
        
        # 1. IP Ban: Automatically ban all associated IP addresses
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        banned_by = session.get('username', 'Moderator')
        ips_to_ban = {ip for ip in (reg_ip, last_ip) if ip}
        if not ips_to_ban:
            conn.execute("""
                INSERT OR REPLACE INTO ip_bans (ip_address, banned_username, banned_by, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, ('', username, banned_by, ban_reason, now_str))
        else:
            for ip in ips_to_ban:
                conn.execute("""
                    INSERT OR REPLACE INTO ip_bans (ip_address, banned_username, banned_by, reason, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (ip, username, banned_by, ban_reason, now_str))
        
        # 2. Deletions (Erase all user traces)
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
        conn.execute("DELETE FROM match_invites WHERE recipient_username = ? COLLATE NOCASE", (username,))
        conn.execute("DELETE FROM private_match_players WHERE username = ? COLLATE NOCASE", (username,))
        conn.execute("DELETE FROM private_messages WHERE sender_username = ? COLLATE NOCASE", (username,))

        # Finally, delete the user record
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        conn.commit()
        
        # Reload cache
        reload_banned_ips()
    except Exception as e:
        conn.rollback()
        print(f"Error banning user: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()
        
    ip_msg = f" (and {len(ips_to_ban)} IP address{'es' if len(ips_to_ban) != 1 else ''})" if ips_to_ban else ""
    return jsonify({'success': True, 'message': f'User {username}{ip_msg} successfully banned and all traces erased.'})


@app.route('/api/mods/ip_bans', methods=['GET'])
@login_required
def get_ip_bans_api():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        with get_db() as conn:
            rows = conn.execute("SELECT id, ip_address, banned_username, banned_by, reason, created_at FROM ip_bans ORDER BY id DESC").fetchall()
            bans = [dict(r) for r in rows]
            return jsonify({'success': True, 'bans': bans})
    except Exception as e:
        print(f"Error fetching IP bans: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mods/ban_ip', methods=['POST'])
@login_required
def ban_ip_api():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.json or {}
    ip = (data.get('ip_address') or '').strip()
    reason = (data.get('reason') or 'Moderator manual IP ban').strip()
    username = (data.get('username') or '').strip()
    
    if not ip:
        return jsonify({'error': 'IP address is required'}), 400
        
    import ipaddress
    try:
        ipaddress.ip_address(ip)
    except ValueError:
        return jsonify({'error': 'Invalid IPv4 or IPv6 address format'}), 400
        
    banned_by = session.get('username', 'Moderator')
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ip_bans (ip_address, banned_username, banned_by, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (ip, username, banned_by, reason, now_str))
            conn.commit()
            
        reload_banned_ips()
        return jsonify({'success': True, 'message': f'IP address {ip} has been banned.'})
    except Exception as e:
        print(f"Error banning IP: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mods/lift_ip_ban', methods=['POST'])
@login_required
def lift_ip_ban_api():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.json or {}
    ip = (data.get('ip_address') or '').strip()
    ban_id = data.get('id')
    
    if not ip and not ban_id:
        return jsonify({'error': 'IP address or ban ID required'}), 400
        
    try:
        with get_db() as conn:
            if ban_id:
                conn.execute("DELETE FROM ip_bans WHERE id = ?", (ban_id,))
            else:
                conn.execute("DELETE FROM ip_bans WHERE ip_address = ?", (ip,))
            conn.commit()
            
        reload_banned_ips()
        return jsonify({'success': True, 'message': f'IP ban lifted for {ip or f"ID {ban_id}"}.'})
    except Exception as e:
        print(f"Error lifting IP ban: {e}")
        return jsonify({'error': str(e)}), 500


def format_duration_string(minutes):
    """Formats minute count into human-readable duration (e.g. '10 minutes', '1 hour 20 minutes')"""
    if minutes < 60:
        return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"
    hours = minutes // 60
    rem_mins = minutes % 60
    h_str = f"{hours} hour" if hours == 1 else f"{hours} hours"
    if rem_mins == 0:
        return h_str
    m_str = f"{rem_mins} minute" if rem_mins == 1 else f"{rem_mins} minutes"
    return f"{h_str} {m_str}"


def parse_timeout_datetime(dt_str):
    """Safely parses timeout timestamp strings or epoch floats into UTC datetime objects."""
    if not dt_str:
        return None
    try:
        ts = float(dt_str)
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    except (ValueError, TypeError):
        pass
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%f'):
        try:
            dt = datetime.datetime.strptime(str(dt_str).replace('Z', '').split('+')[0].strip(), fmt)
            return dt.replace(tzinfo=datetime.timezone.utc)
        except Exception:
            pass
    try:
        dt = datetime.datetime.fromisoformat(str(dt_str).replace(' ', 'T').replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except Exception:
        pass
    return None


def check_user_timeout(user_id_or_name):
    """
    Checks if a user is currently under timeout.
    Returns (is_timed_out, remaining_seconds, remaining_str, timeout_until_str, offense_count, timeout_reason)
    """
    if not user_id_or_name:
        return False, 0, "", None, 0, None
    if str(user_id_or_name).strip().lower() in ('jeffb', 'jeffbabiak'):
        return False, 0, "", None, 0, None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            if str(user_id_or_name).isdigit():
                row = conn.execute(
                    "SELECT id, timeout_until, timeout_offense_count, last_timeout_at, timeout_reason FROM users WHERE id = ? OR username = ? COLLATE NOCASE",
                    (int(user_id_or_name), str(user_id_or_name))
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT id, timeout_until, timeout_offense_count, last_timeout_at, timeout_reason FROM users WHERE username = ? COLLATE NOCASE",
                    (str(user_id_or_name),)
                ).fetchone()

            if not row or not row['timeout_until']:
                return False, 0, "", None, (row['timeout_offense_count'] if row else 0), None
            
            timeout_until_str = row['timeout_until']
            dt_until = parse_timeout_datetime(timeout_until_str)
            if dt_until:
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                diff_sec = (dt_until - now_utc).total_seconds()
                if diff_sec > 0:
                    mins = max(1, int(math.ceil(diff_sec / 60.0)))
                    rem_str = format_duration_string(mins)
                    reason_val = row['timeout_reason'] or 'Temporary restriction'
                    return True, diff_sec, rem_str, timeout_until_str, (row['timeout_offense_count'] or 0), reason_val
        finally:
            conn.close()
    except Exception as e:
        print(f"[check_user_timeout] DB error: {e}")
    return False, 0, "", None, 0, None


@app.route('/api/mods/timeout_user', methods=['POST'])
@login_required
def timeout_user_api():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json or {}
    username = (data.get('username') or '').strip()
    reason = (data.get('reason') or 'Moderator timeout').strip()
    custom_hours = data.get('hours')
    
    if not username:
        return jsonify({'error': 'Username required'}), 400
        
    if username.lower() in ('jeffbabiak', 'jeffb'):
        return jsonify({'error': 'Cannot timeout protected moderator'}), 403

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        user_row = conn.execute(
            "SELECT id, username, timeout_until, timeout_offense_count, last_timeout_at FROM users WHERE username = ? COLLATE NOCASE",
            (username,)
        ).fetchone()
        if not user_row:
            return jsonify({'error': 'User not found'}), 404
            
        user_id = user_row['id']
        actual_username = user_row['username']
        curr_offenses = user_row['timeout_offense_count'] or 0
        last_to_str = user_row['last_timeout_at']
        
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        # Calculate decay: 1 offense level reduction per 24 hours elapsed without an offense
        if last_to_str:
            last_dt = parse_timeout_datetime(last_to_str)
            if last_dt:
                elapsed_hours = (now_utc - last_dt).total_seconds() / 3600.0
                decay_levels = int(elapsed_hours // 24)
                curr_offenses = max(0, curr_offenses - decay_levels)
                
        new_offenses = curr_offenses + 1
        
        # If moderator specified explicit hours, calculate duration from hours
        if custom_hours is not None and str(custom_hours).strip() != '':
            try:
                parsed_hours = float(custom_hours)
                if parsed_hours <= 0:
                    return jsonify({'error': 'Timeout hours must be greater than 0'}), 400
                duration_minutes = int(round(parsed_hours * 60))
                # Cap duration at 30 days (43,200 minutes)
                duration_minutes = min(43200, duration_minutes)
            except ValueError:
                return jsonify({'error': 'Invalid hours format'}), 400
        else:
            duration_minutes = 10 * (2 ** (new_offenses - 1))
            # Cap default exponential duration at 7 days (10,080 minutes)
            duration_minutes = min(10080, duration_minutes)
        
        duration_str = format_duration_string(duration_minutes)
        timeout_until_dt = now_utc + datetime.timedelta(minutes=duration_minutes)
        timeout_until_str = timeout_until_dt.strftime('%Y-%m-%d %H:%M:%S')
        now_str = now_utc.strftime('%Y-%m-%d %H:%M:%S')
        
        conn.execute("""
            UPDATE users 
            SET timeout_until = ?, timeout_offense_count = ?, last_timeout_at = ?, timeout_reason = ?
            WHERE id = ?
        """, (timeout_until_str, new_offenses, now_str, reason, user_id))
        conn.commit()
        
        # In-memory kick from active rooms + broadcast notice in game room
        kicked_from_room = False
        for room in list(room_manager.rooms.values()):
            is_in_room = any(str(p.user_id) == str(user_id) for p in room.players)
            if is_in_room:
                kicked_from_room = True
                # Broadcast notice to everyone in the game room
                room.add_chat_message("System", f"{actual_username} has been kicked from all rooms for {duration_str}.", is_system=True)
                # Set eviction tag for user with duration and custom reason
                if not hasattr(room, 'evicted_users'):
                    room.evicted_users = {}
                room.evicted_users[str(user_id)] = f"timeout:{duration_str}|{reason}"
                room.remove_player(user_id, force=True)
                
        cleanup_user_rooms_entirely(user_id)
        
        return jsonify({
            'success': True,
            'message': f"User {actual_username} timed out for {duration_str} (Offense #{new_offenses}).",
            'duration': duration_str,
            'duration_minutes': duration_minutes,
            'offense_count': new_offenses,
            'timeout_until': timeout_until_str,
            'kicked_from_room': kicked_from_room
        })
    except Exception as e:
        print(f"[timeout_user_api] Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/user/my_timeout_status', methods=['GET'])
def get_my_timeout_status():
    user_id = session.get('user_id') or session.get('username')
    if not user_id:
        return jsonify({'timed_out': False, 'remaining': '', 'remaining_seconds': 0, 'reason': ''})
    is_to, diff_sec, rem_str, to_until, count, reason_val = check_user_timeout(user_id)
    return jsonify({
        'timed_out': is_to,
        'remaining': rem_str,
        'remaining_seconds': diff_sec,
        'timeout_until': to_until,
        'offense_count': count,
        'reason': reason_val or ''
    })


@app.route('/api/mods/lift_timeout', methods=['POST'])
@login_required
def lift_timeout_api():
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.json or {}
    username = (data.get('username') or '').strip()
    if not username:
        return jsonify({'error': 'Username required'}), 400
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        cursor = conn.execute("SELECT id, username FROM users WHERE username = ? COLLATE NOCASE", (username,))
        row = cursor.fetchone()
        if not row:
            return jsonify({'error': 'User not found'}), 404
        user_id, actual_name = row[0], row[1]
        conn.execute("""
            UPDATE users 
            SET timeout_until = NULL, timeout_offense_count = 0, last_timeout_at = NULL, timeout_reason = NULL 
            WHERE id = ?
        """, (user_id,))
        conn.commit()
        return jsonify({
            'success': True, 
            'message': f"Timeout lifted for {actual_name}. Timeout offense history has been reset back to 10 minutes."
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/mods/user_timeout_status/<username>', methods=['GET'])
@login_required
def get_user_timeout_status(username):
    if not is_mod(session.get('username')):
        return jsonify({'error': 'Unauthorized'}), 403
    username = username.strip()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id, username, timeout_until, timeout_offense_count, last_timeout_at, timeout_reason FROM users WHERE username = ? COLLATE NOCASE",
            (username,)
        ).fetchone()
        if not row:
            return jsonify({'error': 'User not found'}), 404
            
        user_id = row['id']
        actual_name = row['username']
        offenses = row['timeout_offense_count'] or 0
        last_to = row['last_timeout_at']
        to_until = row['timeout_until']
        
        # Calculate decayed current offenses
        decayed_offenses = offenses
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        if last_to:
            last_dt = parse_timeout_datetime(last_to)
            if last_dt:
                elapsed_hours = (now_utc - last_dt).total_seconds() / 3600.0
                decay_levels = int(elapsed_hours // 24)
                decayed_offenses = max(0, offenses - decay_levels)
                
        next_duration_mins = 10 * (2 ** max(0, decayed_offenses))
        next_duration_str = format_duration_string(next_duration_mins)
        
        to_res = check_user_timeout(user_id)
        is_to = to_res[0] if to_res else False
        rem_str = to_res[2] if to_res and len(to_res) > 2 else ''
        return jsonify({
            'username': actual_name,
            'is_timed_out': is_to,
            'remaining': rem_str,
            'timeout_until': to_until,
            'offense_count': offenses,
            'effective_offenses': decayed_offenses,
            'next_duration': next_duration_str,
            'last_timeout_at': last_to,
            'reason': row['timeout_reason']
        })
    finally:
        conn.close()


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

def normalize_24h_room_key(room_id, board_dimensions=None):
    if board_dimensions and board_dimensions != 'all':
        return f"24h_{board_dimensions}"
    s = str(room_id or '').lower()
    for dim in ['4x4', '4x6', '5x7', '6x8', '3x3x3', '2x2x2', '3x3']:
        if dim in s:
            return f"24h_{dim}"
    if s.startswith('24h_'):
        return s
    return f"24h_{s}"

# Initialize database
def init_db():
    conn = get_db_connection(DB_PATH, timeout=60.0)
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
        CREATE TABLE IF NOT EXISTS daily_score_sums (
            user_id INTEGER NOT NULL,
            room_id TEXT NOT NULL DEFAULT '24h_4x4',
            score_sum INTEGER NOT NULL,
            PRIMARY KEY (user_id, room_id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS wiktionary_definitions (
            word TEXT PRIMARY KEY,
            definition TEXT
        );
        CREATE TABLE IF NOT EXISTS ip_bans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT UNIQUE NOT NULL,
            banned_username TEXT,
            banned_by TEXT,
            reason TEXT,
            created_at TEXT
        );
    ''')
    conn.commit()
    
    # MIGRATION: Upgrade daily_score_sums table to include room_id composite key if needed
    try:
        table_info = conn.execute("PRAGMA table_info(daily_score_sums)").fetchall()
        column_names = [col[1] for col in table_info]
        if 'room_id' not in column_names:
            conn.execute('''
                CREATE TABLE daily_score_sums_new (
                    user_id INTEGER NOT NULL,
                    room_id TEXT NOT NULL DEFAULT '24h_4x4',
                    score_sum INTEGER NOT NULL,
                    PRIMARY KEY (user_id, room_id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            ''')
            conn.execute('''
                INSERT INTO daily_score_sums_new (user_id, room_id, score_sum)
                SELECT user_id, '24h_4x4', score_sum FROM daily_score_sums
            ''')
            conn.execute('DROP TABLE daily_score_sums')
            conn.execute('ALTER TABLE daily_score_sums_new RENAME TO daily_score_sums')
            conn.commit()
            print("Migrated DB: Upgraded daily_score_sums table to include room_id composite key")
    except Exception as e:
        print(f"Migration check for daily_score_sums info: {e}")

    # MIGRATION: Consolidate legacy daily_score_sums keys to canonical formats
    try:
        table_info = conn.execute("PRAGMA table_info(daily_score_sums)").fetchall()
        column_names = [col[1] for col in table_info]
        if 'room_id' in column_names:
            rows = conn.execute("SELECT user_id, room_id, score_sum FROM daily_score_sums").fetchall()
            consolidated = {}
            needs_migration = False
            for r in rows:
                uid, rid, ssum = r[0], r[1], r[2]
                can_id = normalize_24h_room_key(rid)
                if can_id != rid:
                    needs_migration = True
                key = (uid, can_id)
                consolidated[key] = consolidated.get(key, 0) + (ssum or 0)
            
            if needs_migration:
                conn.execute("DELETE FROM daily_score_sums")
                for (uid, can_id), total_s in consolidated.items():
                    if total_s > 0:
                        conn.execute("INSERT INTO daily_score_sums (user_id, room_id, score_sum) VALUES (?, ?, ?)", (uid, can_id, total_s))
                conn.commit()
                print(f"[init_db] Consolidated legacy daily_score_sums keys into {len(consolidated)} canonical records.")
    except Exception as e:
        print(f"[init_db] Migration error for daily_score_sums consolidation: {e}")

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

    # MIGRATION: Add auth_token column for persistent PWA logins
    try:
        conn.execute('ALTER TABLE users ADD COLUMN auth_token TEXT')
        conn.commit()
        print("Migrated DB: Added auth_token column to users")
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

    # MIGRATION: Add last_visited column for tracking user activity
    try:
        conn.execute('ALTER TABLE users ADD COLUMN last_visited DATETIME')
        conn.commit()
        print("Migrated DB: Added last_visited column to users")
    except sqlite3.OperationalError:
        pass

    # MIGRATION: Add registration_ip and last_ip columns for IP tracking and ban enforcement
    try:
        conn.execute('ALTER TABLE users ADD COLUMN registration_ip TEXT')
        conn.commit()
        print("Migrated DB: Added registration_ip column to users")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute('ALTER TABLE users ADD COLUMN last_ip TEXT')
        conn.commit()
        print("Migrated DB: Added last_ip column to users")
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

    # MIGRATION: Add timeout columns to users table
    for col_def in [
        ('timeout_until', 'DATETIME'),
        ('timeout_offense_count', 'INTEGER DEFAULT 0'),
        ('last_timeout_at', 'DATETIME'),
        ('timeout_reason', 'TEXT')
    ]:
        try:
            conn.execute(f'ALTER TABLE users ADD COLUMN {col_def[0]} {col_def[1]}')
            conn.commit()
            print(f"Migrated DB: Added {col_def[0]} column to users")
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
                image_url TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(post_id) REFERENCES forum_posts(id),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_forum_posts_category ON forum_posts(category_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_forum_comments_post ON forum_comments(post_id)')
        
        # Initialize categories if they don't exist
        categories = [
            ("General", "General discussion about Morpheme."),
            ("Tips, Tricks, and Strategies", "Share your best gameplay advice."),
            ("Screenshots", "Show off your high scores and cool boards."),
            ("Introduce Yourself", "New here? Say hello!"),
            ("News", "Official news and updates from the developers."),
            ("Suggestions/Ideas", "Share your ideas for improving Morpheme."),
            ("Complaints", "Voice your feedback, grievances, or criticisms."),
            ("Bugs/Errors", "Report bugs or technical issues encountered.")
        ]
        for name, desc in categories:
            conn.execute('INSERT OR IGNORE INTO forum_categories (name, description) VALUES (?, ?)', (name, desc))
        
        # MIGRATION: Ensure Suggestions category is renamed to Suggestions/Ideas and clean up duplicates
        conn.execute("UPDATE forum_categories SET name = 'Suggestions/Ideas' WHERE name = 'Suggestions' OR id = 6")
        conn.execute("DELETE FROM forum_categories WHERE name = 'Suggestions/Ideas' AND id != 6")
        conn.execute("INSERT OR IGNORE INTO forum_categories (name, description) VALUES ('Complaints', 'Voice your feedback, grievances, or criticisms.')")

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

    # MIGRATION: Ensure active_boards columns exist
    for col, col_type in [
        ('bonus_word', 'TEXT'),
        ('bonus_cell_json', 'TEXT'),
        ('board_format', 'TEXT'),
        ('uniqueness', 'REAL'),
        ('word_count_range', 'TEXT'),
        ('active_players_json', 'TEXT')
    ]:
        try:
            conn.execute(f'ALTER TABLE active_boards ADD COLUMN {col} {col_type}')
            conn.commit()
            print(f"Migrated DB: Added {col} column to active_boards")
        except sqlite3.OperationalError:
            pass

    # MIGRATION: Ensure forum_comments has image_url column
    try:
        conn.execute('ALTER TABLE forum_comments ADD COLUMN image_url TEXT')
        conn.commit()
        print("Migrated DB: Added image_url column to forum_comments")
    except sqlite3.OperationalError:
        pass

    # MIGRATION: Force all existing accounts to have the new default board sizes once
    try:
        cursor = conn.execute("SELECT config_value FROM site_config WHERE config_key = 'migration_forced_board_sizes_june_27'")
        row = cursor.fetchone()
        if not row:
            print("[Migration] Forcing new default board sizes for all existing accounts...")
            # Get all user IDs
            users_cursor = conn.execute("SELECT id FROM users")
            user_ids = [r[0] for r in users_cursor.fetchall()]
            
            # Insert or replace the board_sizes setting for every user
            for uid in user_ids:
                conn.execute(
                    "INSERT OR REPLACE INTO user_settings (user_id, setting_key, setting_value) VALUES (?, 'board_sizes', ?)",
                    (uid, '{"4x4":82,"4x6":82,"5x7":65,"6x8":54}')
                )
            
            # Record that this migration has run
            conn.execute("INSERT OR REPLACE INTO site_config (config_key, config_value) VALUES ('migration_forced_board_sizes_june_27', 'done')")
            conn.commit()
            print(f"[Migration] Successfully updated board sizes for {len(user_ids)} users.")
    except Exception as e:
        print(f"[Migration] Error forcing default board sizes: {e}")

    # STARTUP: Backfill used_boards from round_history to ensure all-time deduplication
    # survives PM2 restarts. This is idempotent — INSERT OR IGNORE skips already-tracked boards.
    try:
        rows = conn.execute('SELECT board_json, timestamp FROM round_history').fetchall()
        inserted = 0
        for board_json, ts in rows:
            try:
                import json as _json
                board = _json.loads(board_json)
                # Flatten board to a hash string
                h = ''
                for row2 in board:
                    for cell in row2:
                        if isinstance(cell, str):
                            h += cell
                        elif isinstance(cell, list):
                            for c in cell:
                                h += (c if isinstance(c, str) else '')
                h = h.upper()
                if h.strip():
                    conn.execute('INSERT OR IGNORE INTO used_boards (board_hash, used_at) VALUES (?, ?)', (h, ts or 0))
                    inserted += 1
            except Exception:
                pass
        conn.commit()
        if inserted:
            print(f"[init_db] Backfilled {inserted} board hashes from round_history into used_boards.")
    except Exception as e:
        print(f"[init_db] used_boards backfill error: {e}")

    # MIGRATION: Transfer email jeffbabiak@outlook.com to user jeffb and update moderators
    try:
        conn.execute("UPDATE users SET email = '' WHERE LOWER(email) = 'jeffbabiak@outlook.com' AND LOWER(username) != 'jeffb'")
        conn.execute("UPDATE users SET email = 'jeffbabiak@outlook.com' WHERE LOWER(username) = 'jeffb'")
        conn.execute("CREATE TABLE IF NOT EXISTS moderators (username TEXT PRIMARY KEY, added_at REAL)")
        conn.execute("DELETE FROM moderators WHERE LOWER(username) = 'jeffbabiak'")
        conn.execute("INSERT OR IGNORE INTO moderators (username, added_at) VALUES ('jeffb', 1700000000.0)")
        conn.commit()
    except Exception as e:
        print(f"[init_db] Error migrating email and moderators: {e}")

    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    except Exception:
        pass
    conn.close()

init_db()
reload_banned_ips()

# Configuration for Uploads
UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads/avatars')
FORUM_UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads/forum')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FORUM_UPLOAD_FOLDER'] = FORUM_UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB Limit

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
@app.route('/api/stats/dictionary', methods=['GET'])
def get_dictionary_stats():
    from word_validator import word_validator
    word_validator.ensure_csw_loaded()
    
    # Return pure counts for standard Scrabble lexicons (NWL/CSW)
    # without unioning the supplementary 16+ list, keeping them strictly separated.
    # Take a safe local list copy of sets to prevent RuntimeError due to multi-threaded modification
    try:
        nwl_words = list(word_validator.nwl_words)
    except RuntimeError:
        nwl_words = list(word_validator.nwl_words)
        
    try:
        csw_words = list(word_validator.csw_words)
    except RuntimeError:
        csw_words = list(word_validator.csw_words)
        
    try:
        aw_words = list(word_validator.added_words)
    except RuntimeError:
        aw_words = list(word_validator.added_words)
        
    try:
        long_words = list(word_validator.long_words)
    except RuntimeError:
        long_words = list(word_validator.long_words)
    
    def get_dist(w_set):
        dist = {str(i): 0 for i in range(2, 16)}
        dist["16+"] = 0
        for w in w_set:
            l = len(w)
            if l < 2: continue
            elif l >= 16: dist["16+"] += 1
            else: dist[str(l)] += 1
        return dist

    return jsonify({
        'nwl_total': len(nwl_words),
        'csw_total': len(csw_words),
        'aw_total': len(aw_words),
        'long_total': len(long_words),
        'nwl_dist': get_dist(nwl_words),
        'csw_dist': get_dist(csw_words),
        'aw_dist': get_dist(aw_words),
        'long_dist': get_dist(long_words)
    })

def _get_word_finds(word):
    word = word.strip().upper()
    if not word:
        return []
        
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    finds = []
    try:
        # 1. Query round_history
        cursor = conn.execute("""
            SELECT rh.timestamp, u.username, u.country_flag, rh.words_json
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE rh.words_json LIKE ?
        """, (f'%"{word}"%',))
        
        for row in cursor.fetchall():
            try:
                words = json.loads(row['words_json']) if row['words_json'] else []
                if isinstance(words, list):
                    for w_obj in words:
                        w_str = w_obj.get('word', '') if isinstance(w_obj, dict) else str(w_obj)
                        if w_str.upper() == word:
                            ts_val = (w_obj.get('timestamp') if isinstance(w_obj, dict) else None)
                            if ts_val and isinstance(ts_val, (int, float)):
                                dt = datetime.datetime.fromtimestamp(ts_val, tz=datetime.timezone.utc)
                                iso_ts = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                            elif ts_val and isinstance(ts_val, str) and ('T' in ts_val or '-' in ts_val):
                                iso_ts = ts_val
                            else:
                                iso_ts = format_chicago_to_utc(row['timestamp'])
                            
                            finds.append({
                                'username': row['username'],
                                'country_flag': row['country_flag'],
                                'timestamp': iso_ts or '2026-01-01T00:00:00Z'
                            })
            except Exception as ex:
                print(f"[FindCount] Error parsing round_history row: {ex}")

        # 2. Query tournament_scores
        cursor = conn.execute("""
            SELECT ts.submitted_at, u.username, u.country_flag, ts.submitted_words
            FROM tournament_scores ts
            JOIN users u ON ts.user_id = u.id
            WHERE ts.submitted_words LIKE ?
        """, (f'%"{word}"%',))
        
        for row in cursor.fetchall():
            try:
                words = json.loads(row['submitted_words']) if row['submitted_words'] else []
                if isinstance(words, list):
                    for w_obj in words:
                        w_str = w_obj.get('word', '') if isinstance(w_obj, dict) else str(w_obj)
                        if w_str.upper() == word:
                            ts_val = (w_obj.get('timestamp') if isinstance(w_obj, dict) else None) or row['submitted_at']
                            if ts_val and isinstance(ts_val, (int, float)):
                                dt = datetime.datetime.fromtimestamp(ts_val, tz=datetime.timezone.utc)
                                iso_ts = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                            elif ts_val and isinstance(ts_val, str) and ('T' in ts_val or '-' in ts_val):
                                iso_ts = ts_val
                            else:
                                iso_ts = format_chicago_to_utc(row['submitted_at'])
                            
                            finds.append({
                                'username': row['username'],
                                'country_flag': row['country_flag'],
                                'timestamp': iso_ts or '2026-01-01T00:00:00Z'
                            })
            except Exception as ex:
                print(f"[FindCount] Error parsing tournament_scores row: {ex}")

        # 3. Query private_match_turns
        cursor = conn.execute("""
            SELECT pmt.submitted_at, u.username, u.country_flag, pmt.submitted_words
            FROM private_match_turns pmt
            JOIN users u ON pmt.user_id = u.id
            WHERE pmt.submitted_words LIKE ?
        """, (f'%"{word}"%',))
        
        for row in cursor.fetchall():
            try:
                words = json.loads(row['submitted_words']) if row['submitted_words'] else []
                if isinstance(words, list):
                    for w_obj in words:
                        w_str = w_obj.get('word', '') if isinstance(w_obj, dict) else str(w_obj)
                        if w_str.upper() == word:
                            ts_val = (w_obj.get('timestamp') if isinstance(w_obj, dict) else None) or row['submitted_at']
                            if ts_val and isinstance(ts_val, (int, float)):
                                dt = datetime.datetime.fromtimestamp(ts_val, tz=datetime.timezone.utc)
                                iso_ts = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                            elif ts_val and isinstance(ts_val, str) and ('T' in ts_val or '-' in ts_val):
                                iso_ts = ts_val
                            else:
                                iso_ts = format_chicago_to_utc(row['submitted_at'])
                            
                            finds.append({
                                'username': row['username'],
                                'country_flag': row['country_flag'],
                                'timestamp': iso_ts or '2026-01-01T00:00:00Z'
                            })
            except Exception as ex:
                print(f"[FindCount] Error parsing private_match_turns row: {ex}")

    except Exception as e:
        print(f"[get_word_finds] DB Error: {e}")
    finally:
        conn.close()

    # Sort finds descending by timestamp string (newest first, oldest last)
    finds.sort(key=lambda x: str(x['timestamp']), reverse=True)
    return finds

@app.route('/api/word_tally/<word>', methods=['GET'])
def get_word_tally_api(word):
    word = word.upper()
    finds = _get_word_finds(word)
    return jsonify({'word': word, 'count': len(finds)})


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

        # 3. Monthly Total (sum of confirmed donations in the current calendar month, UTC)
        cursor_month = conn.execute("""
            SELECT SUM(amount) as total 
            FROM donations 
            WHERE status = 'confirmed' 
              AND strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')
        """)
        monthly_total = cursor_month.fetchone()['total'] or 0.0

        return jsonify({
            'top': top_list,
            'recent': recent_list,
            'monthly_total': monthly_total
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/')
def index():
    # USER REQUEST: once the user logs in, entering "morpheme.games" (or refreshing root)
    # should automatically take them to the lobby without having to login again.
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


# Thread-safe CAPTCHA store to eliminate session-cookie race conditions
_CAPTCHA_STORE = {}
_CAPTCHA_LOCK = threading.Lock()

def _store_captcha(captcha_id, captcha_text):
    if not captcha_id:
        return
    now = time.time()
    with _CAPTCHA_LOCK:
        # Prune expired tokens (> 10 mins)
        expired = [k for k, v in _CAPTCHA_STORE.items() if (now - v[1]) > 600]
        for k in expired:
            _CAPTCHA_STORE.pop(k, None)
        _CAPTCHA_STORE[str(captcha_id)] = (captcha_text.upper(), now)

def _validate_captcha(captcha_id, submitted_captcha):
    if not submitted_captcha:
        return False
    submitted_upper = str(submitted_captcha).strip().upper()
    expected = None
    if captcha_id:
        with _CAPTCHA_LOCK:
            val = _CAPTCHA_STORE.pop(str(captcha_id), None)
            if val:
                text, created_at = val
                if (time.time() - created_at) <= 600:
                    expected = text
    if not expected:
        expected = session.pop('captcha_text', None)
    return bool(expected and submitted_upper == expected)

# Authentication endpoints
@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    import random
    chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    captcha_text = ''.join(random.choices(chars, k=5))
    
    captcha_id = request.args.get('id')
    if captcha_id:
        _store_captcha(captcha_id, captcha_text)
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
        cursor = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (username,))
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
    flag = data.get('flag', '').strip()
    code = data.get('code', '').strip()
    captcha_val = data.get('captcha', '')
    captcha_id = data.get('captcha_id')
    
    if not _validate_captcha(captcha_id, captcha_val):
        return jsonify({'error': 'Incorrect or expired CAPTCHA. Please click on the CAPTCHA image to refresh and try again.'}), 400
        
    # Username validation
    import re
    if not re.match(r'^[a-zA-Z0-9_]{1,16}$', username):
        return jsonify({'error': 'Username must be 1-16 characters (letters, numbers, underscores only)'}), 400

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    if len(password) < 6:
        return jsonify({'error': 'Password must be 6+ characters'}), 400
        
    if not flag:
        return jsonify({'error': 'Flag selection is required'}), 400
        
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
        
        # Check if username already exists case-insensitively
        cursor = conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (username,))
        if cursor.fetchone():
            return jsonify({'error': 'Username already exists'}), 400
            
        # Check if email already exists
        cursor = conn.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            return jsonify({'error': 'Email is already registered'}), 400
            
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        client_ip = get_client_ip()
        
        # Insert user cleanly
        cursor = conn.execute('INSERT INTO users (username, password_hash, email, is_verified, country_flag, registration_ip, last_ip) VALUES (?, ?, ?, 1, ?, ?, ?)',
                    (username, password_hash, email, flag, client_ip, client_ip))
        user_id = cursor.lastrowid
        
        # Insert default settings for the new user
        default_settings = [
            ('board_sizes', '{"4x4":82,"4x6":82,"5x7":65,"6x8":54}'),
            ('corner_cutoff', '39')
        ]
        for key, val in default_settings:
            conn.execute('INSERT INTO user_settings (user_id, setting_key, setting_value) VALUES (?, ?, ?)', (user_id, key, val))
            
        conn.commit()
        
        cursor = conn.execute('SELECT id, rating FROM users WHERE username = ? COLLATE NOCASE', (username,))
        user = cursor.fetchone()
        
        # Clear the verification session data
        session.pop('signup_code', None)
        session.pop('signup_email', None)
        session.pop('signup_username', None)
        session.pop('signup_code_expires', None)
        
        # Automatically log the user in!
        auth_token = uuid.uuid4().hex
        conn.execute('UPDATE users SET auth_token = ? WHERE id = ?', (auth_token, user[0]))
        conn.commit()

        session['user_id'] = user[0]
        session['username'] = username
        session['email'] = email
        session.pop('is_guest', None)
        session['_morpheme_login_time'] = time.time()
        session.permanent = True
        
        return jsonify({'success': True, 'username': username, 'email': email, 'rating': user[1], 'auth_token': auth_token})
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
                
        if os.path.exists('login_debug.log'):
            with open('login_debug.log', 'r') as f:
                output.append("\n=== login_debug.log ===\n")
                output.extend(f.readlines()[-lines:])
                
        return "<pre>" + "".join(output) + "</pre>"
    except Exception as e:
        return f"Error: {e}", 500


@app.route('/api/login', methods=['POST'])
def login():
    try:
        # Debug log to see if mobile request reaches the server
        try:
            with open('login_debug.log', 'a') as f:
                f.write(f"[{time.time()}] Login call. Data: {request.data.decode('utf-8', errors='ignore')}\n")
        except Exception as e_log:
            print(f"Debug log error: {e_log}")
            
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 200
            
        username = (data.get('username') or '').strip()
        password = data.get('password')
        captcha_val = data.get('captcha', '')
        captcha_id = data.get('captcha_id')
        
        if not _validate_captcha(captcha_id, captcha_val):
            return jsonify({'success': False, 'error': 'Incorrect or expired CAPTCHA. Please click on the CAPTCHA image to refresh and try again.'}), 200

        client_ip = get_client_ip()
        u_lower = username.lower()
        
        # Check in-memory ban cache or database
        if (client_ip and client_ip in BANNED_IPS_CACHE) or (u_lower and u_lower in BANNED_USERNAMES_REASONS):
            ban_reason = BANNED_USERNAMES_REASONS.get(u_lower) or BANNED_IPS_REASONS.get(client_ip) or "Violation of community rules"
            return jsonify({
                'success': False,
                'banned': True,
                'is_banned': True,
                'ban_reason': ban_reason,
                'error': f'This account has been permanently banned from Morpheme.\nReason: {ban_reason}'
            }), 200

        conn = sqlite3.connect(DB_PATH, timeout=30)
        
        # Also check ip_bans directly in case it was just added
        ban_row = conn.execute("SELECT reason FROM ip_bans WHERE (banned_username = ? COLLATE NOCASE AND banned_username != '') OR (ip_address = ? AND ip_address != '') ORDER BY id DESC LIMIT 1", (username, client_ip)).fetchone()
        if ban_row:
            ban_reason = ban_row[0] or "Violation of community rules"
            conn.close()
            return jsonify({
                'success': False,
                'banned': True,
                'is_banned': True,
                'ban_reason': ban_reason,
                'error': f'This account has been permanently banned from Morpheme.\nReason: {ban_reason}'
            }), 200

        cursor = conn.execute('SELECT id, password_hash, email, username FROM users WHERE username = ? COLLATE NOCASE', (username,))
        user = cursor.fetchone()
        
        if not user or not check_password_hash(user[1], password):
            conn.close()
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 200
        
        canonical_username = user[3]
        auth_token = uuid.uuid4().hex
        client_ip = get_client_ip()
        conn.execute('UPDATE users SET auth_token = ?, last_ip = ? WHERE id = ?', (auth_token, client_ip, user[0]))
        conn.commit()
        conn.close()

        session['user_id'] = user[0]
        session['username'] = canonical_username
        session['email'] = user[2]
        session.pop('is_guest', None) # Clear guest flag if present
        session['_morpheme_login_time'] = time.time()
        session.permanent = True
        
        return jsonify({
            'success': True, 
            'username': canonical_username, 
            'email': user[2],
            'is_mod': is_mod(canonical_username),
            'auth_token': auth_token
        })
    except Exception as e:
        print(f"[LoginError] {e}")
        return jsonify({'success': False, 'error': f'Server error: {e}'}), 200


@app.route('/api/auth/auto-login', methods=['POST'])
def auto_login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request'}), 200
        token = data.get('auth_token')
        if not token:
            return jsonify({'success': False, 'error': 'Token missing'}), 200
            
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.execute('SELECT id, username, email, rating FROM users WHERE auth_token = ?', (token,))
        user = cursor.fetchone()
        
        if user:
            client_ip = get_client_ip()
            conn.execute('UPDATE users SET last_ip = ? WHERE id = ?', (client_ip, user[0]))
            conn.commit()
        conn.close()
        
        if not user:
            return jsonify({'success': False, 'error': 'Invalid or expired token'}), 200
            
        session['user_id'] = user[0]
        session['username'] = user[1]
        session['email'] = user[2]
        session.pop('is_guest', None)
        session['_morpheme_login_time'] = time.time()
        session.permanent = True
        
        return jsonify({
            'success': True,
            'username': user[1],
            'email': user[2],
            'rating': user[3],
            'is_mod': is_mod(user[1])
        })
    except Exception as e:
        print(f"[AutoLoginError] {e}")
        return jsonify({'success': False, 'error': str(e)}), 200


@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        user_id = session.get('user_id')
        if user_id:
            with get_db() as conn:
                conn.execute('UPDATE users SET auth_token = NULL WHERE id = ?', (user_id,))
    except Exception as e:
        print(f"[LogoutError] Error during logout: {e}")
    finally:
        session.clear()
        
    return jsonify({'success': True})

@app.route('/api/user/account-info', methods=['GET'])
def get_account_info():
    try:
        user_id = session.get('user_id')
        if not user_id or session.get('is_guest', False):
            return jsonify({'success': False, 'is_guest': True, 'error': 'Not logged in as a registered user'}), 200
        
        with get_db(row_factory=sqlite3.Row, auto_commit=False) as conn:
            row = conn.execute('SELECT username, email FROM users WHERE id = ?', (user_id,)).fetchone()
        
        if not row:
            return jsonify({'success': False, 'error': 'User not found'}), 404
            
        return jsonify({
            'success': True,
            'username': row['username'],
            'email': row['email'] or ''
        })
    except Exception as e:
        print(f"[get_account_info] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/change-password', methods=['POST'])
def change_password():
    try:
        user_id = session.get('user_id')
        if not user_id or session.get('is_guest', False):
            return jsonify({'success': False, 'error': 'Please log in to a registered account to change your password.'}), 403
        
        data = request.get_json() or {}
        current_password = data.get('current_password', '').strip()
        new_password = data.get('new_password', '').strip()
        confirm_password = data.get('confirm_password', '').strip()
        
        if not current_password:
            return jsonify({'success': False, 'error': 'Please enter your current password.'}), 400
        if not new_password:
            return jsonify({'success': False, 'error': 'Please enter a new password.'}), 400
        if len(new_password) < 4:
            return jsonify({'success': False, 'error': 'New password must be at least 4 characters long.'}), 400
        if new_password != confirm_password:
            return jsonify({'success': False, 'error': 'New password and confirmation do not match.'}), 400
            
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        
        if not row or not check_password_hash(row[0], current_password):
            conn.close()
            return jsonify({'success': False, 'error': 'Current password is incorrect.'}), 400
            
        new_hash = generate_password_hash(new_password, method='pbkdf2:sha256')
        conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user_id))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Password changed successfully!'})
    except Exception as e:
        print(f"[change_password] Error: {e}")
        return jsonify({'success': False, 'error': f'Failed to change password: {str(e)}'}), 500

@app.route('/api/user/change-email', methods=['POST'])
def change_email():
    try:
        user_id = session.get('user_id')
        if not user_id or session.get('is_guest', False):
            return jsonify({'success': False, 'error': 'Please log in to a registered account to change your email.'}), 403
        
        data = request.get_json() or {}
        new_email = data.get('new_email', '').strip()
        
        if not new_email:
            return jsonify({'success': False, 'error': 'Please enter a new email address.'}), 400
            
        import re
        email_regex = r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
        if not re.match(email_regex, new_email):
            return jsonify({'success': False, 'error': 'Please enter a valid email address.'}), 400
            
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.execute('SELECT id FROM users WHERE email = ? COLLATE NOCASE AND id != ?', (new_email, user_id))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return jsonify({'success': False, 'error': 'This email address is already registered to another account.'}), 400
            
        conn.execute('UPDATE users SET email = ? WHERE id = ?', (new_email, user_id))
        conn.commit()
        conn.close()
        
        session['email'] = new_email
        return jsonify({'success': True, 'message': 'Email address changed successfully!', 'email': new_email})
    except Exception as e:
        print(f"[change_email] Error: {e}")
        return jsonify({'success': False, 'error': f'Failed to change email: {str(e)}'}), 500


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
    captcha_id = data.get('captcha_id')
    
    if not _validate_captcha(captcha_id, captcha_val):
        return jsonify({'error': 'Incorrect or expired CAPTCHA. Please click on the CAPTCHA image to refresh and try again.'}), 400

    import random
    import string
    
    for attempt in range(10):
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
            if 'UNIQUE' in str(e) and attempt < 9:
                continue
            return jsonify({'error': f'Guest login failed: {str(e)}'}), 500
        finally:
            conn.close()

@app.route('/api/session', methods=['GET'])
def get_session():
    if 'user_id' in session:
        room_manager.update_presence(session['user_id'])
        touch_user_last_visited(session['user_id'])
        
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
        
_last_visited_touch_cache = {}

def touch_user_last_visited(user_id):
    if not user_id:
        return
    now = time.time()
    last_touch = _last_visited_touch_cache.get(str(user_id), 0)
    if now - last_touch > 30: # Throttled to at most once per 30s per user
        _last_visited_touch_cache[str(user_id)] = now
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            conn.execute('UPDATE users SET last_visited = CURRENT_TIMESTAMP WHERE id = ?', (user_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[LastVisited] Error touching last_visited for user {user_id}: {e}")

@app.route('/api/profile/<username>', methods=['GET'])
def get_public_profile(username):
    if 'user_id' in session:
        room_manager.update_presence(session['user_id'])
        touch_user_last_visited(session['user_id'])
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.execute('''
        SELECT id, username, rating, games_played, avatar_url, country_flag, 
               full_name, age, gender, location, quote, description, proof_url, wins,
               max_pe, avg_pe, pe_count, created_at, last_visited
        FROM users WHERE username = ? COLLATE NOCASE
    ''', (username,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    # Touch visited timestamp for target user if they match logged in user
    if 'user_id' in session and session['user_id'] == user[0]:
        touch_user_last_visited(user[0])

    user_id = user[0]

    # Query allow_pm setting for target user
    pm_setting_row = conn.execute("SELECT setting_value FROM user_settings WHERE user_id = ? AND setting_key = 'allow_pm'", (user_id,)).fetchone()
    allow_pm = True
    if pm_setting_row:
        val = str(pm_setting_row[0]).lower()
        if val in ('false', '0', 'off'):
            allow_pm = False

    period = request.args.get('period', 'all').lower()
    
    # Calculate Chicago local time boundaries
    chicago_tz = ZoneInfo("America/Chicago")
    chicago_now = datetime.datetime.now(chicago_tz)
    chicago_today_str = chicago_now.strftime('%Y-%m-%d')
    chicago_week_ago_str = (chicago_now - datetime.timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    chicago_month_ago_str = (chicago_now - datetime.timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    chicago_year_ago_str = (chicago_now - datetime.timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')

    time_filter = ""
    if period == 'day':
        time_filter = f"AND date(timestamp) = '{chicago_today_str}'"
    elif period == 'week':
        time_filter = f"AND timestamp >= '{chicago_week_ago_str}'"
    elif period == 'month':
        time_filter = f"AND timestamp >= '{chicago_month_ago_str}'"
    elif period == 'year':
        time_filter = f"AND timestamp >= '{chicago_year_ago_str}'"

    # Calculate Period Stats (If 'all', we still calculate from round_history for consistency, 
    # but could use user table for performance if data volume is high)
    # Only count rounds with a score > 0 as a played game, and exclude 24h rooms (duration >= 7200)
    cursor_stats = conn.execute(f'''
        SELECT COUNT(CASE WHEN total_score > 0 THEN 1 END), SUM(total_score)
        FROM round_history
        WHERE user_id = ? AND round_duration < 7200 {time_filter}
    ''', (user_id,))
    games_played_period, pt_sum_period = cursor_stats.fetchone()
    games_played_period = games_played_period or 0
    pt_sum_period = pt_sum_period or 0

    # Get config-specific ratings (Current ratings are ALWAYS current/lifetime)
    cursor = conn.execute('SELECT config_key, rating FROM user_ratings WHERE user_id = ?', (user_id,))
    config_ratings = {row[0]: row[1] for row in cursor.fetchall()}

    # Get matching rounds for calculations (Excluding 24h rounds with duration >= 7200)
    cursor_all = conn.execute(f'''
        SELECT room_id, game_type, round_number, board_json, words_json, total_score, 
               round_start_time, round_duration, timestamp, user_rating, performance_ratio, id,
               wpm, total_words_avail, board_dimensions, board_format
        FROM round_history
        WHERE user_id = ? AND round_duration < 7200 {time_filter}
        ORDER BY timestamp DESC, id DESC
    ''', (user_id,))
    all_rows = cursor_all.fetchall()
    
    # Filter and Deduplicate (Ensure all distinct rounds are included)
    seen_rounds = set()
    clean_rows = []
    for r in all_rows:
        g_id, wjson = r[11], r[4]
        if g_id in seen_rounds: continue
        try:
            if wjson == '[]' or not json.loads(wjson): continue
        except: continue
        seen_rounds.add(g_id)
        clean_rows.append(r)

    # Batch fetch all room entries in a single query to eliminate N+1 database queries
    room_entries_map = {}
    if clean_rows:
        # Extract unique room_id, round_number, timestamp tuples
        unique_round_keys = list(set((r[0], r[2], r[8]) for r in clean_rows))
        # Batch query matching room participants
        room_ids = list(set(k[0] for k in unique_round_keys))
        if room_ids:
            # Query in chunks of 500 room_ids if needed
            all_participants = []
            for i in range(0, len(room_ids), 500):
                chunk = room_ids[i:i+500]
                placeholders = ','.join(['?'] * len(chunk))
                cursor_part = conn.execute(f'''
                    SELECT rh.room_id, rh.round_number, rh.timestamp, rh.total_score, rh.user_rating, u.username
                    FROM round_history rh
                    JOIN users u ON rh.user_id = u.id
                    WHERE rh.room_id IN ({placeholders})
                ''', chunk)
                all_participants.extend(cursor_part.fetchall())
                
            for p_row in all_participants:
                r_key = (p_row[0], p_row[1], p_row[2])
                if r_key not in room_entries_map:
                    room_entries_map[r_key] = []
                room_entries_map[r_key].append((p_row[3], p_row[4], p_row[5]))

    # helper to process a row
    def process_round_row(row):
        room_id, gtype, rnum, bjson, wjson, score, rstart, rdur, ts, urat, pe_ratio, g_id, wpm, twa, saved_dims, b_fmt = row
        r_entries = room_entries_map.get((room_id, rnum, ts), [])
        if not pe_ratio or pe_ratio <= 0.0:
            if len(r_entries) > 1:
                r_score_sum = sum(e[0] for e in r_entries)
                r_rating_sum = sum(e[1] for e in r_entries)
                if r_rating_sum > 0:
                    expected = (urat / r_rating_sum) * r_score_sum
                    pe_ratio = round(score / expected, 2) if expected > 0 else 1.0
                else:
                    pe_ratio = round(score / (r_score_sum / len(r_entries)), 2) if r_score_sum > 0 else 1.0
            else:
                pe_ratio = 1.0
        perf_val = int(round(pe_ratio * 100)) if pe_ratio else 100
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
            'round_start_time': rstart, 'round_duration': rdur, 'timestamp': format_chicago_to_utc(ts),
            'wpm': wpm or 0, 'total_words_avail': twa or 0,
            'board_format': b_fmt or 'Normal',
            'all_players': sorted([{'username': e[2], 'score': e[0], 'rating': e[1]} for e in r_entries], key=lambda x: x['score'], reverse=True)
        }

    processed_all = [process_round_row(r) for r in clean_rows]
    wins_period = sum(1 for p in processed_all if p['all_players'] and p['total_score'] > 0 and p['total_score'] >= p['all_players'][0]['score'])
    
    # Config Stats (Averages for the period)
    config_stats = {}
    for cfg_key, rating in config_ratings.items():
        try:
            gtype, dims, dur = cfg_key.split('|')
            if int(dur) >= 7200:
                rating = user[2]
            matching = [p for p in processed_all if p['game_type'] == gtype and p['dimensions'] == dims and p['round_duration'] == int(dur)]
            matching_standard = [p for p in matching if 'valued' not in str(p.get('board_format', '')).lower()]
            matching_valid = [p for p in matching if p.get('total_words_avail', 0) > 0]
            avg_pct_found = round(sum(p['num_words'] / p['total_words_avail'] * 100 for p in matching_valid) / len(matching_valid), 1) if matching_valid else 0
            max_pct_found = round(max([p['num_words'] / p['total_words_avail'] * 100 for p in matching_valid]) if matching_valid else 0, 1)

            config_stats[cfg_key] = {
                'rating': rating,
                'games_played': len(matching),
                'wins': sum(1 for p in matching if p['all_players'] and p['total_score'] > 0 and p['total_score'] >= p['all_players'][0]['score']),
                'point_sum': sum(p['total_score'] for p in matching_standard),
                'avg_pct_found': avg_pct_found,
                'max_pct_found': max_pct_found,
                'avg_score': round(sum(p['total_score'] for p in matching_standard) / len(matching_standard), 1) if matching_standard else 0,
                'avg_words': round(sum(p['num_words'] for p in matching) / len(matching), 1) if matching else 0,
                'avg_perf': round(sum(p['performance_value'] for p in matching) / len(matching), 1) if matching else 0
            }
        except Exception as e:
            print(f"[Profile] Error calculating config stats for {cfg_key}: {e}")
            config_stats[cfg_key] = {
                'rating': rating,
                'games_played': 0,
                'wins': 0,
                'point_sum': 0,
                'avg_pct_found': 0,
                'max_pct_found': 0,
                'avg_score': 0,
                'avg_words': 0,
                'avg_perf': 0
            }

    # Period-specific AVG WPM
    wpm_games = [p['wpm'] for p in processed_all if p['total_words_avail'] >= 50 and p['wpm'] > 0]
    avg_wpm = round(sum(wpm_games) / len(wpm_games), 1) if wpm_games else 0

    # Best Score in Period (Strictly exclude Valued Letters)
    standard_rounds = [p for p in processed_all if 'valued' not in str(p.get('board_format', '')).lower()]
    best_score_period = max([p['total_score'] for p in standard_rounds]) if standard_rounds else 0

    # Sort recent and exceptional
    recent_rounds = sorted(processed_all, key=lambda x: x['timestamp'], reverse=True)[:50]
    exceptional_rounds = sorted([p for p in processed_all if p['performance_value'] >= 200],
                               key=lambda x: x['timestamp'], reverse=True)[:50]
    dynamic_max_pe = max([p['performance_value'] for p in processed_all]) if processed_all else 0

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
        'max_pe': dynamic_max_pe,
        'avg_pe': round(user[15], 2) if user[15] else 0.0,
        'created_at': user[17],
        'last_visited': user[18] or user[17],
        'allow_pm': allow_pm,
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
    
    canonical_game_types = [game_type]
    if game_type in ('classic', 'accumulative'):
        canonical_game_types = ['classic', 'accumulative']
    elif game_type in ('3d', 'cube'):
        canonical_game_types = ['3d', 'cube']

    placeholders = ','.join(['?'] * len(canonical_game_types))

    # 1. Get Matching Rounds (using canonical game types and board_dimensions)
    query_all = f'''
        SELECT words_json, total_score, timestamp, room_id, round_number, board_json, id, user_rating, board_dimensions, total_words_avail, board_format
        FROM round_history
        WHERE user_id = ? AND game_type IN ({placeholders}) AND board_dimensions = ? AND round_duration = ?
        ORDER BY timestamp DESC, id DESC
    '''
    cursor_all = conn.execute(query_all, (user_id, *canonical_game_types, board_dimensions, time_limit))
    global_matching = cursor_all.fetchall()
    
    # Calculate Global Best (All-Time) - Strictly exclude Valued Letters from high score and best word records
    global_stats = {
        "high_score": 0, "max_words": 0, "longest_word": "", 
        "best_word": {"word": "", "points": 0},
        "games_played": len(global_matching), "total_score": 0, "total_words": 0, "wins": 0
    }
    
    for row in global_matching:
        try:
            words = json.loads(row[0])
            score = row[1]
            b_fmt = row[10] if len(row) > 10 else 'Normal'
            is_valued = 'valued' in str(b_fmt).lower()

            if not is_valued:
                global_stats["total_score"] += score
                if score > global_stats["high_score"]: global_stats["high_score"] = score
                for w in words:
                    if w.get('points', 0) > global_stats["best_word"]["points"]:
                        global_stats["best_word"] = {"word": w['word'], "points": w.get('points',0)}

            global_stats["total_words"] += len(words)
            if len(words) > global_stats["max_words"]: global_stats["max_words"] = len(words)
            for w in words:
                if len(w['word']) > len(global_stats["longest_word"]): global_stats["longest_word"] = w['word']
        except: continue

    # 2. Filter by Period for the lists - Enforce Calendar Day logic
    chicago_tz = ZoneInfo("America/Chicago")
    chicago_now = datetime.datetime.now(chicago_tz)
    chicago_today_str = chicago_now.strftime('%Y-%m-%d')
    chicago_week_ago_str = (chicago_now - datetime.timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    chicago_month_ago_str = (chicago_now - datetime.timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    chicago_year_ago_str = (chicago_now - datetime.timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')

    time_filter = ""
    if period == 'day':
        time_filter = f"AND date(timestamp) = '{chicago_today_str}'"
    elif period == 'week':
        time_filter = f"AND timestamp >= '{chicago_week_ago_str}'"
    elif period == 'month':
        time_filter = f"AND timestamp >= '{chicago_month_ago_str}'"
    elif period == 'year':
        time_filter = f"AND timestamp >= '{chicago_year_ago_str}'"
        
    query = f'''
        SELECT words_json, total_score, timestamp, room_id, round_number, board_json, id, user_rating, board_dimensions, total_words_avail, board_format
        FROM round_history
        WHERE user_id = ? AND game_type IN ({placeholders}) AND board_dimensions = ? AND round_duration = ? {time_filter}
        ORDER BY timestamp DESC, id DESC
    '''
    cursor = conn.execute(query, (user_id, *canonical_game_types, board_dimensions, time_limit))
    period_rows = cursor.fetchall()
    
    # Only count rounds where the player actually scored points (> 0)
    period_matching = [r for r in period_rows if r[1] > 0]

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
        b_fmt = row[10] if len(row) > 10 else 'Normal'
        is_valued = 'valued' in str(b_fmt).lower()
        
        # User Request: Filter out 0 points or 0 words, AND Deduplicate
        if my_score <= 0 or len(words) == 0:
            continue
            
        round_key = g_id
        if round_key in seen_rounds:
            continue
        seen_rounds.add(round_key)
        
        if not is_valued:
            total_period_score += my_score
            for w in words:
                w.update({'timestamp': format_chicago_to_utc(ts), 'room_id': r_id, 'round_number': r_num, 'game_id': g_id})
                all_period_words.append(w)
        total_period_words += len(words)

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

        word_validator.ensure_csw_loaded()
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
        twa = row[9] if len(row) > 9 else 0
        pct_found = round(num_words / twa * 100, 1) if twa > 0 else 0
        obscure_count = sum(1 for w in words if isinstance(w, dict) and w.get('word', '').upper() in word_validator.unique_csw_words) if words else 0
        processed = {
            'game_id': g_id, 'room_id': r_id, 'round_number': r_num, 'timestamp': format_chicago_to_utc(ts),
            'total_score': my_score, 'num_words': len(words), 'is_win': is_win,
            'avg_len': avg_l,
            'ratio': ratio, 'performance_value': perf_val,
            'top_word': top_word,
            'all_players': room_entries,
            'words': words,
            'board': json.loads(row[5]),
            'board_format': b_fmt,
            'pct_found': pct_found,
            'obscure_count': obscure_count
        }
        performance_list.append(processed)

    # Exceptional: by Timestamp DESC (Ratio >= 2.0 — double the typical word count for that board)
    exceptional = sorted([r for r in performance_list if r['ratio'] >= 2.0], key=lambda x: x['timestamp'], reverse=True)[:50]
    
    # Winning: by Timestamp DESC
    winning = sorted([r for r in performance_list if r['is_win']], key=lambda x: x['timestamp'], reverse=True)[:50]
    
    # Best Scores: Score DESC, then Timestamp DESC (Strictly exclude Valued Letters)
    best_scores = sorted([r for r in performance_list if 'valued' not in str(r.get('board_format', '')).lower()], key=lambda x: (int(x['total_score']), x['timestamp']), reverse=True)[:50]
    
    # Best Word Counts: Count DESC, then Timestamp DESC
    best_counts = sorted(performance_list, key=lambda x: (int(x['num_words']), x['timestamp']), reverse=True)[:50]
    
    # Games Played: Timestamp DESC (True Recency)
    recent = sorted(performance_list, key=lambda x: x['timestamp'], reverse=True)[:50] 
    
    # Best Words: Points DESC (Unique words only, standard scoring)
    unique_words = {}
    for w in all_period_words:
        word_text = w.get('word')
        points = int(w.get('points', 0))
        if word_text not in unique_words or points >= unique_words[word_text]['points']:
            unique_words[word_text] = {
                'word': word_text, 
                'points': points, 
                'timestamp': w.get('timestamp'), 
                'game_id': w.get('game_id'),
                'room_id': w.get('room_id'),
                'round_number': w.get('round_number')
            }
    
    unique_word_list = list(unique_words.values())
    best_words = sorted(unique_word_list, key=lambda x: (x['points'], x['timestamp']), reverse=True)[:50]

    # Collect round objects for best_words
    best_word_game_ids = {w['game_id'] for w in best_words if w.get('game_id')}
    best_words_rounds = [r for r in performance_list if r['game_id'] in best_word_game_ids]

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
    
    matching_valid = [p for p in performance_list if p.get('pct_found', 0) > 0]
    avg_pct_found = round(sum(p['pct_found'] for p in matching_valid) / len(matching_valid), 1) if matching_valid else 0
    max_pct_found = round(max([p['pct_found'] for p in matching_valid]) if matching_valid else 0, 1)

    best_pcts = sorted(performance_list, key=lambda x: (x.get('pct_found', 0), x['timestamp']), reverse=True)[:50]
    best_obscure = sorted([x for x in performance_list if x.get('obscure_count', 0) > 0], key=lambda x: (x.get('obscure_count', 0), x['timestamp']), reverse=True)[:50]

    return jsonify({
        'username': username,
        'rating': rating,
        'global_stats': global_stats,
        'stats': {
            'total_score': total_period_score,
            'total_words': total_period_words,
            'avg_pct_found': avg_pct_found,
            'max_pct_found': max_pct_found,
            'games_played': len(period_matching),
            'wins': period_wins,
            'win_rate': round((period_wins / len(period_matching))*100, 1) if period_matching else 0,
            'avg_score': round(total_period_score / len(period_matching)) if period_matching else 0,
            'avg_words': round(total_period_words / len(period_matching), 1) if period_matching else 0,
            'avg_perf': round(sum(r['ratio'] for r in performance_list)/len(performance_list), 2) if performance_list else 1.0,
            'avg_word_pts': round(total_period_score / total_period_words, 1) if total_period_words > 0 else 0,
            'max_pe': max([r['performance_value'] for r in performance_list]) if performance_list else 0,
            
            'exceptional_rounds': exceptional,
            'winning_rounds': winning,
            'best_scores': best_scores,
            'best_word_counts': best_counts,
            'best_pcts': best_pcts,
            'best_obscure': best_obscure,
            'recent_rounds': recent,
            'best_words': best_words,
            'best_words_rounds': best_words_rounds
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
            
            # Read bytes for AI content moderation
            file_bytes = file.read()
            file.seek(0)
            ext = file.filename.rsplit('.', 1)[1].lower()
            
            # Run AI moderation check
            moderation_res = moderate_content(image_bytes=file_bytes, mime_type=ext)
            if moderation_res.get("inappropriate"):
                return jsonify({'error': f"Inappropriate content detected: {moderation_res.get('reason')}"}), 400
                
            # Generate safe filename: user_id_timestamp.ext
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
    """Marks a leaver for score-based proportional rating change at round end.
    The flat -16 penalty is replaced by the end-of-round proportional system,
    which uses the quitter's score at the time they left. This is fairer —
    a player who left with a high score gets a smaller negative change than
    one who left with no score. A small abandonment bounty is still added to
    the pool for remaining players as a signal that the quitter disrupted the round.
    """
    player = room.get_player(user_id)
    if not player or player.is_ai:
        return

    # 1. Broad Exemption: 24h Rooms (>= 2h time limit)
    if room.time_limit >= 7200:
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Leave-rating SKIPPED for {player.username} in {room.room_id}: 24h room\n")
        return

    # 2. Activity Check: Only apply if they actually PARTICIPATED in this round.
    # If the room is in INTERMISSION, the round is already over — no action needed.
    if getattr(room, 'state', '') == 'intermission':
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Leave-rating SKIPPED for {player.username} in {room.room_id}: Intermission\n")
        return

    # 2b. Mid-Round Exemption: Do not penalize if they joined mid-round.
    is_late_joiner = getattr(player, 'joined_mid_round', False)
    if is_late_joiner:
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Leave-rating SKIPPED for {player.username} in {room.room_id}: Joined mid-round\n")
        return

    has_score = (player.score > 0)
    has_words = (len(player.submitted_words) > 0)

    if not (has_score or has_words):
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Leave-rating SKIPPED for {player.username} in {room.room_id}: No activity (score={player.score}, words={len(player.submitted_words)})\n")
        return

    # 3. Check if others are still playing (Human starters only).
    other_participants = [
        p for p in room.players
        if str(p.user_id) != str(user_id)
        and not p.is_ai
        and not getattr(p, 'is_guest', False)
        and not getattr(p, 'joined_mid_round', False)
    ]

    if not other_participants:
        with open(RATING_AUDIT_PATH, 'a') as log:
            log.write(f"[{time.time()}] Leave-rating SKIPPED: No other registered human STARTERS in {room.room_id}.\n")
        return

    # 4. Log the leave event with the score at time of leaving.
    # The actual rating change will be computed proportionally at round end
    # via round_quitters (the player's score is preserved there).
    with open(RATING_AUDIT_PATH, 'a') as log:
        log.write(f"[{time.time()}] Leave-rating QUEUED for {player.username} in {room.room_id}: "
                  f"score={player.score}, words={len(player.submitted_words)}. "
                  f"Proportional change will apply at round end via round_quitters.\n")

    # 5. Add a small abandonment bounty to the pool for remaining players.
    # This is a disruption signal — the main rating impact comes from the proportional system.
    bounty = 8
    room.abandonment_bounty += bounty
    with open(RATING_AUDIT_PATH, 'a') as log:
        log.write(f"[{time.time()}] Bounty Collection: +{bounty} added to {room.room_id} pool (Total: {room.abandonment_bounty})\n")


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
    try:
        with open(DEBUG_FLOW_PATH, 'a') as f:
            f.write(f"\n[app.py] create_room called at {time.time()}\n")
        if 'user_id' not in session:
            ensure_guest_session()
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        # Timeout check
        is_to, _, rem_str, _, _, reason_val = check_user_timeout(session.get('user_id') or session.get('username'))
        if is_to:
            r_text = reason_val or 'Moderator timeout'
            return jsonify({
                'error': f'You are temporarily timed out for another {rem_str}.',
                'timed_out': True,
                'remaining': rem_str,
                'timeout_reason': r_text,
                'reason': f'timeout:{rem_str}|{r_text}'
            }), 403
        
        data = request.get_json() or {}
        game_type = data.get('game_type', 'standard')
        time_limit = int(data.get('time_limit') or 45)
        board_dimensions = str(data.get('board_dimensions') or '4x4')
        
        try:
            min_rating = int(data.get('min_rating') or 0)
        except (TypeError, ValueError):
            min_rating = 0
            
        try:
            max_rating = int(data.get('max_rating') or 9999)
        except (TypeError, ValueError):
            max_rating = 9999
        
        # Guest Restriction: Guests cannot create custom/limited rooms
        if session.get('is_guest', False):
            if min_rating > 0 or max_rating < 9999:
                return jsonify({'error': 'RANK_REJECT: Guest users are not allowed to create rooms with rating limits. Please register to unlock this feature.'}), 403
        
        # Accumulative and 24h rooms are permanent singletons per dimension; custom FCFS/Split rooms get unique IDs
        if str(game_type).lower() == 'accumulative' or int(time_limit) >= 7200:
            generated_id = f"pub_v2_{game_type}_{board_dimensions}_{time_limit}".replace(' ', '_').lower()
        else:
            generated_id = f"room_{game_type}_{board_dimensions}_{time_limit}_{uuid.uuid4().hex[:8]}".replace(' ', '_').lower()
        print(f"[app.py] Generated ID for room: {generated_id}")
            
        room = room_manager.create_room(generated_id, game_type, time_limit, board_dimensions, min_rating, max_rating, is_private=False)
        
        # Ensure user is not in any other room
        cleanup_user_rooms(session['user_id'], exclude_room_id=room.room_id)
        
        # Use the actual ID (could be existing one if singleton)
        room_id = room.room_id
        
        # Get configuration-specific rating & stats in 1 single connection
        config_key = f"{game_type}|{board_dimensions}|{time_limit}"
        rating = 1200
        games_played = 0
        country_flag = '🏳️'
        
        if session.get('is_guest', False):
            rating = 0
        else:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            try:
                cur = conn.execute('SELECT rating, games_played, country_flag FROM users WHERE id = ?', (session['user_id'],))
                u_row = cur.fetchone()
                if u_row:
                    rating = u_row[0] if u_row[0] is not None else 1200
                    games_played = u_row[1] if u_row[1] is not None else 0
                    if u_row[2]: country_flag = u_row[2]
                
                is_24h = (int(time_limit) >= 7200)
                if not is_24h:
                    r_cur = conn.execute('SELECT rating FROM user_ratings WHERE user_id = ? AND config_key = ?', 
                                         (session['user_id'], config_key))
                    r_row = r_cur.fetchone()
                    if r_row and r_row[0] is not None:
                        rating = r_row[0]
            except Exception as e:
                print(f"[create_room] user stats query warning: {e}")
            finally:
                conn.close()
        
        room.add_player(session['user_id'], session['username'], rating, 
                        games_played=games_played, country_flag=country_flag, 
                        is_guest=session.get('is_guest', False))
        
        # Start first round immediately if room does not have a board
        if not room.board:
            print(f"[app.py] Kickstarting first round for NEW room {room_id}")
            room.starting_round = True
            room._round_start_init_time = time.time()
            room_manager.start_round(room_id)
        else:
            print(f"[app.py] Room {room_id} already has a board. Skipping redundant start_round.")
        
        return jsonify({'success': True, 'room_id': room_id})
    except Exception as e:
        import traceback
        print(f"[create_room] Error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Room Creation Error: {str(e)}'}), 400

@app.route('/api/room/<room_id>/join', methods=['POST'])
def join_room(room_id):
    if 'user_id' not in session:
        ensure_guest_session()
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Timeout check
    is_to, _, rem_str, _, _, reason_val = check_user_timeout(session.get('user_id') or session.get('username'))
    if is_to:
        r_text = reason_val or 'Moderator timeout'
        return jsonify({
            'error': f'You are temporarily timed out for another {rem_str}.',
            'timed_out': True,
            'remaining': rem_str,
            'timeout_reason': r_text,
            'reason': f'timeout:{rem_str}|{r_text}'
        }), 403
    
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
    # Unlimited players for Accumulative/Solo, and force player mode
    if room.game_type in ['accumulative', 'solo_accumulative'] or getattr(room, 'is_solo', False):
        room.max_players = 9999
        as_spectator = False
    else:
        as_spectator = data.get('as_spectator', False)

    if as_spectator:
        room.add_spectator(user_id, session['username'], rating)
        room.update_player_activity(user_id)
        return jsonify({'success': True, 'role': 'spectator'})

    # Rating limit check for FCFS, SP, and rating-restricted rooms: force spectator mode if rating outside limits
    has_limits = (room.min_rating > 0 or room.max_rating < 9999)
    if has_limits and (rating < room.min_rating or rating > room.max_rating or session.get('is_guest', False)):
        room.add_spectator(user_id, session['username'], rating)
        room.update_player_activity(user_id)
        return jsonify({'success': True, 'role': 'spectator', 'notice': 'Rating outside room limit; spectator mode activated.'})

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
    # Pass has_exceptional_round
    success = room.add_player(user_id, session['username'], rating, 
                             games_played=games_played, country_flag=country_flag, 
                             manual_accessed=False, is_guest=session.get('is_guest', False))
    if success:
        p = room.players[-1] # Valid since we just added or updated
        p.has_exceptional_round = has_exceptional 
        room.update_player_activity(user_id)
        return jsonify({'success': True, 'role': 'player', 'max_players': room.max_players, 'joined_mid_round': False})
    else:
        # If room is full, automatically join as spectator instead of failing (except Accumulative)
        if room.game_type in ['accumulative', 'solo_accumulative']:
             return jsonify({'error': "Could not join Accumulative room. Please try again."}), 409
        
        print(f"[app.py] Room {room_id} is full. Automatically joining {session['username']} as spectator.")
        room.add_spectator(user_id, session['username'], rating)
        room.update_player_activity(user_id)
        return jsonify({'success': True, 'role': 'spectator'})

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
    time_limit_raw = request.args.get('time_limit')
    time_limit = None
    if time_limit_raw is not None:
        try:
            time_limit = int(time_limit_raw)
        except (ValueError, TypeError):
            pass
    
    active_rooms = []
    
    for room_id, room in list(room_manager.rooms.items()):
        try:
            # Exclude solo and private rooms from public listing
            if room.is_solo or getattr(room, 'is_private', False):
                continue
                
            matches_game = not game_type or str(room.game_type).lower() == str(game_type).lower()
            matches_board = not board_dimensions or str(room.board_dimensions).lower() == str(board_dimensions).lower()
            try:
                room_t = int(room.time_limit)
            except (ValueError, TypeError):
                room_t = 45
            matches_time = time_limit is None or room_t == time_limit
            
            if matches_game and matches_board and matches_time:
                humans = [p for p in room.players if not getattr(p, 'is_ai', False)]
                is_daily = (room_t >= 7200)
                # Only list rooms that have at least 1 active human player (or 24h daily rooms).
                # Empty rooms should never be visible in the lobby.
                if len(humans) == 0 and not is_daily:
                    continue
                
                # Calculate average rating safely
                p_count = len(room.players)
                ratings = [(getattr(p, 'rating', 1200) if getattr(p, 'rating', 1200) is not None else 1200) for p in room.players]
                avg_rating = round(sum(ratings) / p_count) if p_count > 0 else 0
                
                players_list = []
                for p in room.players:
                    p_uname = getattr(p, 'username', 'Player') or 'Player'
                    p_rat = getattr(p, 'rating', 1200)
                    if p_rat is None: p_rat = 1200
                    p_uid = getattr(p, 'user_id', 0) or 0
                    players_list.append({'username': p_uname, 'rating': p_rat, 'user_id': p_uid})

                active_rooms.append({
                    'room_id': room.room_id,
                    'game_type': str(room.game_type),
                    'board_dimensions': str(room.board_dimensions),
                    'time_limit': room_t,
                    'player_count': p_count,
                    'max_players': room.max_players,
                    'min_rating': room.min_rating,
                    'max_rating': room.max_rating,
                    'average_rating': avg_rating,
                    'display_average_rating': avg_rating,
                    'state': room.state,
                    'current_round': room.current_round,
                    'players': players_list
                })
        except Exception as err:
            print(f"[list_rooms] Error formatting room {room_id}: {err}")
            
    return jsonify({'rooms': active_rooms})

@app.route('/api/lobby-stats', methods=['GET'])
def get_lobby_stats():
    """Get aggregated player counts for all game configurations"""
    stats = {}
    config_humans = {}  # key -> set(user_id)
    now = time.time()
    
    for room in list(room_manager.rooms.values()):
        try:
            # Hide solo and private rooms from lobby stats
            if room.is_solo or getattr(room, 'is_private', False):
                continue
                
            try:
                t_lim = int(room.time_limit)
            except (ValueError, TypeError):
                t_lim = 45
            is_daily = (t_lim >= 7200)
            
            # Aggregate all active players (and daily archives for 24h rooms)
            all_candidate_players = list(room.players)
            if is_daily and hasattr(room, 'past_players') and isinstance(room.past_players, dict):
                seen_uids = {str(p.user_id) for p in room.players if getattr(p, 'user_id', None)}
                for pp in room.past_players.values():
                    if pp and str(getattr(pp, 'user_id', '')) not in seen_uids:
                        all_candidate_players.append(pp)
            
            humans = []
            for p in all_candidate_players:
                if getattr(p, 'is_ai', False):
                    continue
                # For real-time rooms (non-24h), count players active within 60s or freshly joined
                if not is_daily:
                    p_active = getattr(p, 'last_active', 0)
                    if p_active > 0 and (now - p_active) > 60:
                        continue
                humans.append(p)
                
            if len(humans) == 0 and not is_daily:
                continue
                
            # Create a unique key for this configuration (case-insensitive)
            key = f"{str(room.game_type).lower()}|{str(room.board_dimensions).lower()}|{t_lim}"
            
            if key not in config_humans:
                config_humans[key] = set()
            for h in humans:
                uid = str(getattr(h, 'user_id', None) or getattr(h, 'username', ''))
                if uid:
                    config_humans[key].add(uid)
        except Exception as err:
            print(f"[get_lobby_stats] Error processing room {getattr(room, 'room_id', 'unknown')}: {err}")
    
    for key, user_set in config_humans.items():
        stats[key] = len(user_set)
    
    return jsonify({'stats': stats})

@app.route('/api/lobby/chat', methods=['GET'])
def get_lobby_chat():
    """Fetch active lobby players and the 100-message chat history."""
    if 'user_id' in session:
        user_id = session['user_id']
        username = session.get('username') or 'Guest'
        is_guest = session.get('is_guest', False)
        rating = 1200
        avatar_url = None
        
        if not is_guest:
            try:
                conn = sqlite3.connect(DB_PATH, timeout=30)
                cur = conn.execute('SELECT rating, avatar_url FROM users WHERE id = ?', (user_id,))
                row = cur.fetchone()
                if row:
                    rating = row[0] if row[0] is not None else 1200
                    avatar_url = row[1]
                conn.close()
            except Exception as e:
                print(f"[get_lobby_chat] DB error: {e}")
        
        lobby_manager.update_presence(user_id, username, rating, avatar_url)
    
    state = lobby_manager.get_lobby_state()
    return jsonify(state)

@app.route('/api/lobby/chat', methods=['POST'])
def send_lobby_chat():
    """Send a message to Lobby Chat with 100-message FIFO buffer."""
    if 'user_id' not in session:
        ensure_guest_session()
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    username = session.get('username') or 'Guest'
    is_guest = session.get('is_guest', False)
    
    # Check user discipline / timeout
    is_to, _, rem_str, _, _, reason_val = check_user_timeout(user_id or username)
    if is_to:
        r_text = reason_val or 'Moderator timeout'
        return jsonify({
            'error': f'You are temporarily timed out from chatting for another {rem_str}.',
            'timed_out': True,
            'remaining': rem_str,
            'reason': f'timeout:{rem_str}|{r_text}'
        }), 403
    
    data = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    message = message[:300]
    
    rating = 1200
    avatar_url = None
    if not is_guest:
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30)
            cur = conn.execute('SELECT rating, avatar_url FROM users WHERE id = ?', (user_id,))
            row = cur.fetchone()
            if row:
                rating = row[0] if row[0] is not None else 1200
                avatar_url = row[1]
            conn.close()
        except Exception as e:
            print(f"[send_lobby_chat] DB error: {e}")
    
    lobby_manager.update_presence(user_id, username, rating, avatar_url)
    lobby_manager.add_message(user_id, username, rating, message)
    
    state = lobby_manager.get_lobby_state()
    return jsonify({
        'success': True,
        'count': state['count'],
        'players': state['players'],
        'messages': state['messages']
    })

@app.route('/api/room/<room_id>/state')
def get_room_state(room_id):
    if 'user_id' in session:
        uid = session['user_id']
        room_manager.update_presence(uid)
    
    room = room_manager.get_room(room_id)
    if room:
        # 1. On-demand state updates & next round transitions safeguard
        room.check_and_update_state()
        room_manager.check_6x8_rescue(room)
        if room.state == 'intermission' and room.time_remaining <= 0:
            if not getattr(room, 'starting_round', False):
                import threading
                print(f"[get_room_state] TR=0 for {room_id} — launching start_next_round async")
                threading.Thread(target=room_manager.start_next_round, args=(room.room_id,), daemon=True).start()

        # 2. On-demand database backfill for previous day's board/history
        if room.time_limit >= 7200:
            room_manager.load_previous_day_data(room)
            
        print(f"[get_room_state] Room: {room_id} | State: {room.state} | PrevBonus: {getattr(room, 'previous_bonus_word', 'None')} | CurrBonus: {room.bonus_word}")
        user_rating_for_snapshot = None
        if 'user_id' in session:
            uid = session['user_id']
            if str(uid) in getattr(room, 'evicted_users', {}):
                reason = room.evicted_users.pop(str(uid), 'inactivity')
                print(f"[get_room_state] User {uid} detected in room.evicted_users! Returning 403 eviction response (Reason: {reason}).")
                return jsonify({'error': f'You have been evicted: {reason}', 'evicted': True, 'reason': reason}), 403
            
            # Timeout check during room state polling
            is_to, _, rem_str, _, _, reason_val = check_user_timeout(uid)
            if is_to:
                room.remove_player(uid)
                r_text = reason_val or 'Moderator timeout'
                return jsonify({
                    'error': f'You are currently timed out for another {rem_str}.',
                    'evicted': True,
                    'timed_out': True,
                    'remaining': rem_str,
                    'timeout_reason': r_text,
                    'reason': f'timeout:{rem_str}|{r_text}'
                }), 403
            
            # Fetch user's rating for this room's configuration
            rating = 1200
            games_played = 0
            country_flag = '🏳️'
            is_guest = session.get('is_guest', False)
            if is_guest:
                rating = 0
            else:
                conn = sqlite3.connect(DB_PATH, timeout=30)
                try:
                    game_type_base = str(room.game_type).replace('solo_', '')
                    config_key = f"{game_type_base}|{room.board_dimensions}|{room.time_limit}"
                    is_24h = (room.time_limit >= 7200)
                    if is_24h:
                        cursor = conn.execute('SELECT rating FROM users WHERE id = ?', (uid,))
                        row = cursor.fetchone()
                        rating = row[0] if row else 1200
                    else:
                        cursor = conn.execute('SELECT rating FROM user_ratings WHERE user_id = ? AND config_key = ?', (uid, config_key))
                        row = cursor.fetchone()
                        rating = row[0] if row else 1200
                    cur = conn.execute('SELECT games_played, country_flag FROM users WHERE id = ?', (uid,))
                    row2 = cur.fetchone()
                    if row2:
                        games_played = row2[0]
                        if row2[1]: country_flag = row2[1]
                except Exception as e:
                    print(f"[get_room_state] Error fetching rating: {e}")
                finally:
                    conn.close()

            user_rating_for_snapshot = rating

            # Automatically sync active polling user into room.players OR room.spectators if not already present
            if not room.is_solo and not room.get_player(uid) and not room.get_spectator(uid):
                has_limits = (room.min_rating > 0 or room.max_rating < 9999)
                is_out_of_range = has_limits and (rating < room.min_rating or rating > room.max_rating or is_guest)
                p_count = len([p for p in room.players if not p.is_ai])
                is_full = (p_count >= getattr(room, 'max_players', 8)) and (room.game_type not in ['accumulative', 'solo_accumulative'])

                if is_out_of_range or is_full:
                    print(f"[get_room_state] Auto-syncing user {session.get('username')} as SPECTATOR (rating: {rating}, limits: {room.min_rating}-{room.max_rating}, full: {is_full})")
                    room.add_spectator(uid, session.get('username', 'Player'), rating)
                else:
                    room.add_player(uid, session.get('username', 'Player'), rating, games_played=games_played, country_flag=country_flag, is_guest=is_guest)

            room.update_player_activity(uid)
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
                                room.intermission_start_time = time.time()
                                room.spinner_params_generated = False
                                if hasattr(room, '_transition_spinner_launched'): delattr(room, '_transition_spinner_launched')
                                if hasattr(room, 'spinner_params_revealed'): delattr(room, 'spinner_params_revealed')
                                if hasattr(room, 'board_search_started'): delattr(room, 'board_search_started')
                                if hasattr(room, 'board_search_loading'): delattr(room, 'board_search_loading')
                                if hasattr(room, 'starting_round'): delattr(room, 'starting_round')
                                
                                # Clear all next_round/staging attributes to prevent bleed/outdated data promotion
                                room.next_round_board = None
                                room.next_round_words = None
                                room.next_round_word_paths = None
                                room.next_round_word_scores = None
                                room.next_round_bonus = None
                                room.next_round_format = None
                                room.next_round_total_words_count = 0
                                room.next_round_counts_by_len = {}
                                room.next_round_total_points = 0
                                room.next_round_cell_density = None
                                room.next_round_initial_cell_density = None
                                room.next_spinner_params = None
                                room.next_round_spinner_params = None
                                room.next_round_difficulty = None
                                room.next_round_uniqueness = None

                            # Kickstart the spinner parameters and board search immediately in a background thread
                            # so the server starts generating the board while the user's frontend is loading.
                            def kickstart_first_board():
                                try:
                                    print(f"[app.py] Kickstarting background board pre-generation for reconstructed room {room_id}")
                                    room_manager.generate_spinner_params(room_id, reveal=False)
                                    room_manager.generate_spinner_params(room_id, reveal=True)
                                    room_manager.start_board_search(room_id)
                                except Exception as ex:
                                    print(f"[app.py] Error kickstarting first board on reconstruction: {ex}")
                            
                            import threading
                            threading.Thread(target=kickstart_first_board, daemon=True).start()
                        
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
                                    is_24h = (room.time_limit >= 7200)
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
                            
                            # Kickstart 24h room round if it has no board (e.g. first join of the day / server restart)
                            if is_24h and not room.board:
                                print(f"[app.py] Kickstarting first round for reconstructed 24h room {room_id}")
                                room.starting_round = True
                                room._round_start_init_time = time.time()
                                import threading
                                thread = threading.Thread(target=room_manager.start_round, args=(room_id,), daemon=True)
                                thread.start()
                except Exception as re_err:
                    print(f"[app.py] Could not reconstruct public hub {room_id}: {re_err}")

        if not room:
             return jsonify({'error': 'Room not found (expired due to inactivity)'}), 404
            
        # USER REQUEST: Ensure the server transitions EXACTLY when the client hits 0:00.
        _hb_t0 = time.time()
        room.check_and_update_state()
        _hb_t1 = time.time()
        room_manager.check_6x8_rescue(room)
        _hb_t2 = time.time()

        # LAZY LOAD YESTERDAY'S HISTORY FOR 24H ROOMS (outside lock - may do DB I/O):
        if room.time_limit >= 7200 and (not getattr(room, 'previous_day_history', None) or len(room.previous_day_history) == 0):
            try:
                room_manager.get_yesterdays_history(room, room.current_round)
            except Exception as e:
                print(f"[app.py] Error lazy-loading yesterday's history for room {room_id}: {e}")

        # MILESTONE PROCESSING (synchronous, outside lock).
        milestone = room.get_next_round_milestone()
        _hb_t3 = time.time()
        if milestone == 'spinner':
            room_manager.generate_spinner_params(room_id, reveal=False)
        elif milestone == 'reveal':
            room_manager.generate_spinner_params(room_id, reveal=True)
        elif milestone == 'search':
            room_manager.start_board_search(room_id)
        elif milestone == 'start':
            if not getattr(room, 'starting_round', False):
                import threading
                threading.Thread(target=room_manager.start_next_round, args=(room_id,), daemon=True).start()
        _hb_t4 = time.time()

        ms_caus = (_hb_t1 - _hb_t0) * 1000
        ms_rescue = (_hb_t2 - _hb_t1) * 1000
        ms_milestone = (_hb_t4 - _hb_t3) * 1000
        if ms_caus > 50 or ms_rescue > 50 or ms_milestone > 50:
            print(f"[HB-TIMING] SLOW room={room_id} state={room.state} milestone={milestone} | caus={ms_caus:.0f}ms rescue={ms_rescue:.0f}ms milestone={ms_milestone:.0f}ms")

        # STATE SNAPSHOT (inside lock — fast attribute reads only, no I/O):
        with room._state_lock:
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
                        if is_valued: pts += get_valued_word_score(w)
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
                # USER REQUEST: Return completed round words, scores, and min length for Intermission
                cur_min = getattr(room, 'previous_min_length', getattr(room, 'current_min_length', 3))
                display_floor = cur_min

                prev_words = getattr(room, 'previous_all_words', None)
                prev_scores = getattr(room, 'previous_all_word_scores', None)
                if prev_words and len(prev_words) > 0:
                    if isinstance(prev_words, dict):
                        raw_w_list = list(prev_words.keys())
                        word_scores_to_return = prev_words
                    else:
                        raw_w_list = list(prev_words)
                        word_scores_to_return = prev_scores if isinstance(prev_scores, dict) else getattr(room, 'solved_words_with_scores', {})
                else:
                    raw_w_list = list(room.all_words or [])
                    word_scores_to_return = getattr(room, 'solved_words_with_scores', {}) or {}

                words_to_return = list(raw_w_list)

                # RE-SYNC: Ensure re-categorized lists also respect this floor using pre-cached, self-healing lists
                if hasattr(word_validator, 'word_validator'):
                    dict_name = str(getattr(room, 'current_dictionary', 'NWL')).upper()
                    if 'CSW' in dict_name or 'AW' in dict_name or 'ALL' in dict_name or 'ADDED' in dict_name:
                        word_validator.word_validator.ensure_csw_loaded()
                        prev_csw = getattr(room, 'previous_csw_only_words', None)
                        if prev_csw:
                            room.csw_only_words = list(prev_csw)
                        elif getattr(room, 'csw_only_words', None) is None:
                            room.csw_only_words = []
                    else:
                        room.csw_only_words = []
                    room.csw_only_words = [w for w in (room.csw_only_words or []) if len(w) >= display_floor]

                    if getattr(room, 'use_added_words', False) or 'AW' in dict_name or 'ALL' in dict_name or 'ADDED' in dict_name:
                        prev_aw = getattr(room, 'previous_added_words', None)
                        if prev_aw:
                            room.added_words = list(prev_aw)
                        elif getattr(room, 'added_words', None) is None:
                            room.added_words = []
                    else:
                        room.added_words = []
                    room.added_words = [w for w in (room.added_words or []) if len(w) >= display_floor]

                # Purge scores as well
                word_scores_to_return = {w: word_scores_to_return[w] for w in words_to_return if w in word_scores_to_return}
            elif is_active:
                # ACTIVE: Provide word scores for total-points calculation client-side
                # (Avoids showing '0 total pts' when total_points_count hasn't been computed yet)
                word_scores_to_return = getattr(room, 'solved_words_with_scores', {}) or {}
                # ALWAYS provide all_words for instant client-side validation
                words_to_return = list(room.all_words)

            # Determine user visibility
            user_id = session.get('user_id')
            requesting_player = room.get_player(user_id) if user_id else None
            
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
                    # 1. TIME CHECK: Only skip future-scheduled words for AI bots
                    if p.is_ai and w.get('time', 0) > now:
                        continue
                        
                    # 2. Add to score and bonus status since it has been found
                    pts = w.get('points', 0)
                    v_score += pts
                    if v_score < 0:
                        v_score = 0
                    
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

            # Build snapshot dict under lock (all room.xxx reads happen here)
            snapshot = {
                'room_id': room.room_id,
                'game_type': room.game_type,
                'state': room.state,
                'midnight_reset_occurred': getattr(room, 'midnight_reset_occurred', False),
                'current_round': room.current_round,
                'time_limit': room.time_limit,
                'time_remaining': room.time_remaining,
                'round_end_time': room.round_end_time if room.state == 'active' else 0,
                'intermission_end_time': room.intermission_end_time if room.state == 'intermission' else 0,
                'server_time': time.time(),
                'your_username': session.get('username'),
                'your_rating': user_rating_for_snapshot,
                'board': (getattr(room, 'previous_board', None) or room.board) if is_intermission else room.board,
                'board_dimensions': room.board_dimensions,
                'bonus_word': (getattr(room, 'previous_bonus_word', None) or room.bonus_word) if is_intermission else room.bonus_word,
                'bonus_cell': (getattr(room, 'previous_bonus_cell', None) if is_intermission else room.bonus_cell),
                'all_words': words_to_return,
                'total_words_count': (room.previous_total_words if is_intermission else actual_total),
                'next_round_total_words_count': getattr(room, 'next_round_total_words_count', 0),
                'initial_total_words': getattr(room, 'initial_total_words', actual_total),
                'total_points_count': (room.previous_total_points if is_intermission else room.total_points_count),
                'total_counts_by_len': (room.previous_total_counts_by_len if is_intermission else getattr(room, 'total_counts_by_len', {})),
                'cell_density': (
                    getattr(requesting_player, 'cell_density', []) if (requesting_player and getattr(requesting_player, 'cell_density', None)) else getattr(room, 'cell_density', [])
                ),
                'max_cell_density': getattr(room, 'max_cell_density', 0),
                'all_word_scores': word_scores_to_return,
                'all_words_paths': (getattr(room, 'previous_all_words_paths', None) or room.all_words_paths) if is_intermission else {},
                'global_found_words': global_found,
                'fcfs_found_words': list(getattr(room, 'fcfs_found_words', [])) if (is_active and is_fcfs) else [],
                'added_words': list(room.added_words) if (getattr(room, 'added_words', None) and getattr(room, 'use_added_words', False)) else [],
                'csw_only_words': list(room.csw_only_words) if getattr(room, 'csw_only_words', None) else [],
                'previous_all_words': list(getattr(room, 'previous_all_words', []) or []),
                'previous_all_word_scores': getattr(room, 'previous_all_word_scores', {}) or {},
                'previous_board': getattr(room, 'previous_board', []),
                'previous_csw_only_words': getattr(room, 'previous_csw_only_words', []),
                'previous_added_words': getattr(room, 'previous_added_words', []),
                'previous_bonus_word': getattr(room, 'previous_bonus_word', ''),
                'previous_dictionary': getattr(room, 'previous_dictionary', 'NWL'),
                'previous_use_added_words': getattr(room, 'previous_use_added_words', False),
                'spinner_params': (
                    getattr(room, 'frozen_revealed_params', None) or getattr(room, 'spinner_params', {}) or {}
                ),
                'use_added_words': getattr(room, 'use_added_words', False),
                'current_min_length': getattr(room, 'previous_min_length', room.current_min_length) if is_intermission else room.current_min_length,
                'min_rating': getattr(room, 'min_rating', 0),
                'max_rating': getattr(room, 'max_rating', 9999),
                'current_board_format': getattr(room, 'previous_board_format', room.current_board_format) if is_intermission else room.current_board_format,
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
                        'has_exceptional_round': (getattr(room, 'state', '') == 'intermission' and getattr(p, 'has_exceptional_round', False)),
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
            }
        # END LOCK — JSON serialization happens outside to avoid holding lock during encoding
        resp = jsonify(snapshot)
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
        
    # Run AI Content Moderation on text and image combined
    image_bytes = None
    mime_type = None
    if image:
        image_bytes, mime_type = parse_data_url(image)

    moderation_res = moderate_content(text=message, image_bytes=image_bytes, mime_type=mime_type)
    if moderation_res.get("inappropriate"):
        return jsonify({'error': f"Inappropriate content detected: {moderation_res.get('reason')}"}), 400
        
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
        
    # Timeout check
    is_to, _, rem_str, _, _, reason_val = check_user_timeout(user_id)
    if is_to:
        r_text = reason_val or 'Moderator timeout'
        return jsonify({'error': f'You are currently timed out for another {rem_str}.', 'timed_out': True, 'remaining': rem_str, 'timeout_reason': r_text, 'reason': f'timeout:{rem_str}|{r_text}'}), 403
        
    rating = None
    player = room.get_player(user_id)
    if player:
        rating = player.rating
    else:
        for s in getattr(room, 'spectators', []):
            if str(s.user_id) == str(user_id):
                rating = s.rating
                break
                
    if rating is None:
        try:
            import sqlite3
            conn = sqlite3.connect(DB_PATH, timeout=30)
            cursor = conn.execute('SELECT rating FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            if row:
                rating = row[0]
            conn.close()
        except Exception as e:
            print(f"[Chat] Failed to query rating: {e}")
            
    if rating is None:
        rating = 1200
        
    room.add_chat_message(session['username'], message, image=image, rating=rating)
    room.update_player_activity(session['user_id'])
    
    return jsonify({'success': True})

@app.route('/room/<room_id>/submit_word', methods=['POST'])
def submit_word(room_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Timeout check
    is_to, _, rem_str, _, _, reason_val = check_user_timeout(user_id)
    if is_to:
        r_text = reason_val or 'Moderator timeout'
        return jsonify({'error': f'You are currently timed out for another {rem_str}.', 'timed_out': True, 'remaining': rem_str, 'timeout_reason': r_text, 'reason': f'timeout:{rem_str}|{r_text}'}), 403
    
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
        _sw_t0 = time.time()
        acquired = room._state_lock.acquire(timeout=2.0)
        _sw_acq_ms = (time.time() - _sw_t0) * 1000
        if not acquired:
            print(f"[SW-TIMING] LOCK FAIL after {_sw_acq_ms:.0f}ms word='{word}' room={room_id}")
            return jsonify({'success': False, 'message': 'Server busy, please retry', 'retry': True}), 503
        print(f"[SW-TIMING] lock acquired after {_sw_acq_ms:.0f}ms word='{word}'")
        try:
            _sw_t1 = time.time()
            success, message, points, final_word = room.submit_word(user_id, word, path=path)
            print(f"[SW-TIMING] submit_word() done in {(time.time()-_sw_t1)*1000:.0f}ms result={success} msg='{message}'")
        finally:
            room._state_lock.release()
    except Exception as e:
        import traceback
        with open(DEBUG_FLOW_PATH, 'a') as f:
            f.write(f"[Submit-Error] Room: {room_id} | Error: {e}\n{traceback.format_exc()}\n")
        return jsonify({'success': False, 'message': f'Server Error: {str(e)}'}), 500
    
    # Refresh activity on any submission attempt (valid or not)
    room.update_player_activity(user_id)
    
    player = room.get_player(user_id)
    new_score = player.score if player else 0
    last_sub = (player.submitted_words[-1] if (player and player.submitted_words) else {})
    score_details = last_sub.get('score_details', {}) if isinstance(last_sub, dict) else {}
    returned_path = last_sub.get('path') if isinstance(last_sub, dict) else path

    return jsonify({
        'success': success, 
        'message': message,
        'points': points,
        'word': final_word,
        'new_score': new_score,
        'score_details': score_details,
        'path': returned_path,
        'cell_density': (getattr(player, 'cell_density', None) if (player and getattr(player, 'cell_density', None)) else getattr(room, 'cell_density', None)),
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
        # Safeguard: set DEFINITIONS_PATH to a sensible default so writing works
        DEFINITIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dictionaries', 'Definitions.txt')
        return

    try:
        print(f"Loading definitions from {definitions_path}...")
        defs = {}
        with open(definitions_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                sep = line.find(' - ')
                if sep != -1:
                    defs[line[:sep].strip().upper()] = line[sep+3:].rstrip('\r\n').strip()
        DEFINITIONS_CACHE = defs
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

import re

def extract_target_word(definition_text):
    if not definition_text:
        return None
    text = definition_text.strip()
    
    # 1. plural of
    m = re.search(r'(?i)\bplural of\s+([a-zA-Z\-]+)', text)
    if m:
        return m.group(1).upper()
        
    # 2. verb conjugation of (past, participle, third person)
    m = re.search(r'(?i)\b(?:present participle|past participle|past tense|past|third-person singular present|third-person singular|conjugation)\s+of\s+([a-zA-Z\-]+)', text)
    if m:
        return m.group(1).upper()
        
    # 3. alternative form of
    m = re.search(r'(?i)\balternative\s+(?:form|spelling)\s+of\s+([a-zA-Z\-]+)', text)
    if m:
        return m.group(1).upper()
        
    return None

def clean_def_text(def_text):
    pos = 'n'
    # Strip any leading (noun) tag
    def_text = re.sub(r'^\s*\(noun\)\s*', '', def_text.strip(), flags=re.IGNORECASE)
    
    # Default POS detection based on leading tag
    m = re.match(r'^\s*\((noun|verb|adjective|adverb|pronoun|preposition|conjunction|interjection)\)\s*(.*)', def_text, re.IGNORECASE)
    if m:
        pos_str = m.group(1).lower()
        if pos_str == 'noun': pos = 'n'
        elif pos_str == 'verb': pos = 'v'
        elif pos_str == 'adjective': pos = 'adj'
        elif pos_str == 'adverb': pos = 'adv'
        else: pos = pos_str[:3]
        def_text = m.group(2)
    
    # Clean up trailing tags like [n], [v], [adj]
    def_text = re.sub(r'\s*\[[a-z]+\]\s*$', '', def_text.strip())
    # Clean up trailing periods
    def_text = def_text.rstrip('.')
    return def_text.strip(), pos

def lookup_raw_definition_online(word_upper):
    # Free Dictionary API Fallback
    try:
        import urllib.request
        import json
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word_upper.lower()}"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=1.0) as response:
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
                            if part_of_speech.lower() == 'noun':
                                def_parts.append(first_def)
                            else:
                                def_parts.append(f"({part_of_speech}) {first_def}")
                if def_parts:
                    return "; ".join(def_parts)
    except Exception as e:
        pass

    # Wiktionary API Fallback
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
        with urllib.request.urlopen(req, timeout=1.0) as response:
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
                            if part_of_speech.lower() == 'noun':
                                def_parts.append(text)
                            else:
                                def_parts.append(f"({part_of_speech}) {text}")
                if def_parts:
                    return "; ".join(def_parts)
    except Exception as e:
        pass

    return None

def lookup_wiki_definition_from_db(word):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        cursor.execute("SELECT definition FROM wiktionary_definitions WHERE word = ?;", (word.upper(),))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception as e:
        print(f"[Definitions] Error querying local wiki definition: {e}")
    return None

def get_definition_cached_or_online(w):
    global DEFINITIONS_CACHE
    if not DEFINITIONS_CACHE:
        load_definitions()
    d = DEFINITIONS_CACHE.get(w)
    if not d:
        d = lookup_wiki_definition_from_db(w)
        if not d:
            d = lookup_raw_definition_online(w)
            if d:
                try:
                    conn = sqlite3.connect(DB_PATH, timeout=5)
                    with conn:
                        conn.execute("INSERT OR REPLACE INTO wiktionary_definitions (word, definition) VALUES (?, ?);", (w.upper(), d))
                    conn.close()
                except Exception as e:
                    print(f"[Definitions] Error caching wiki definition to DB: {e}")
        if d:
            DEFINITIONS_CACHE[w] = d
    return d

def get_definition_cached_or_online_with_guess(w):
    d = get_definition_cached_or_online(w)
    if d:
        return d
        
    def _local_lookup(root):
        global DEFINITIONS_CACHE
        if DEFINITIONS_CACHE and root in DEFINITIONS_CACHE:
            return DEFINITIONS_CACHE[root]
        return lookup_wiki_definition_from_db(root)

    # Guess root words (strip suffixes) - checked locally to guarantee instant sub-millisecond response
    if w.endswith('S') and not w.endswith('SS') and not w.endswith('US') and not w.endswith('IS') and not w.endswith('AS'):
        # Try stripping 'S'
        r = w[:-1]
        if _local_lookup(r):
            DEFINITIONS_CACHE[w] = f"plural of {r}"
            return DEFINITIONS_CACHE[w]
            
        # Try stripping 'ES'
        if w.endswith('ES'):
            r2 = w[:-2]
            if _local_lookup(r2):
                DEFINITIONS_CACHE[w] = f"plural of {r2}"
                return DEFINITIONS_CACHE[w]
                
    if w.endswith('ED'):
        # Try stripping 'ED'
        r = w[:-2]
        if _local_lookup(r):
            DEFINITIONS_CACHE[w] = f"(verb) conjugation of {r}"
            return DEFINITIONS_CACHE[w]
            
        # Try stripping 'D' (e.g. baked -> bake)
        r2 = w[:-1]
        if _local_lookup(r2):
            DEFINITIONS_CACHE[w] = f"(verb) conjugation of {r2}"
            return DEFINITIONS_CACHE[w]
            
    if w.endswith('ING'):
        # Try stripping 'ING'
        r = w[:-3]
        if _local_lookup(r):
            DEFINITIONS_CACHE[w] = f"(verb) conjugation of {r}"
            return DEFINITIONS_CACHE[w]
            
        # Try stripping 'ING' and adding 'E' (e.g. baking -> bake)
        r2 = w[:-3] + 'E'
        if _local_lookup(r2):
            DEFINITIONS_CACHE[w] = f"(verb) conjugation of {r2}"
            return DEFINITIONS_CACHE[w]
            
    return None

def format_resolved_definition(word_upper, visited=None):
    if visited is None:
        visited = set()
    if word_upper in visited:
        return None
    visited.add(word_upper)

    raw = get_definition_cached_or_online_with_guess(word_upper)
    if not raw:
        return None

    # Strip any leading (noun) from raw
    raw = re.sub(r'^\s*\(noun\)\s*', '', raw.strip(), flags=re.IGNORECASE)

    # Comprehensive pointer pattern: matches "plural of X", "diminutive of X", "synonym of X", etc.
    pointer_pattern = re.compile(
        r'(?i)\b((?:plural|present participle|past participle|simple past|past tense|past|third-person singular simple present indicative|third-person singular present|third-person singular|conjugation|gerund|alternative form|alternative spelling|variant form|variant spelling|variant|diminutive|diminutive form|synonym|synonym for|comparative form|comparative|superlative form|superlative|female equivalent|feminine form|masculine form|agent noun|frequentative|verbal noun|spelling)\s+(?:of|for)\s+)([a-zA-Z\-]+)',
        re.DOTALL
    )

    m = pointer_pattern.search(raw)
    if m:
        target = m.group(2).upper()
        if target != word_upper:
            target_resolved = format_resolved_definition(target, visited.copy())
            if target_resolved:
                target_clean = re.sub(r'^\s*\(noun\)\s*', '', target_resolved.strip(), flags=re.IGNORECASE)
                if target_clean and f"({target_clean})" not in raw:
                    end_idx = m.end(2)
                    raw = raw[:end_idx] + f" ({target_clean})" + raw[end_idx:]

    # Check if raw starts with leading parenthesis (e.g. (verb) meaning, (hawaiian) meaning)
    m = re.match(r'^\s*\(([^)]+)\)\s*(.*)', raw, re.IGNORECASE)
    if m:
        pos = m.group(1).lower()
        meaning = m.group(2).strip()
        if pos == 'noun':
            return meaning
        return f"({pos}) {meaning}"

    # Convert legacy format to clean format (no leading '(noun)')
    meaning, pos = clean_def_text(raw)
    pos_map = {
        'n': 'noun',
        'v': 'verb',
        'adj': 'adjective',
        'adv': 'adverb',
        'interj': 'interjection',
        'pron': 'pronoun',
        'prep': 'preposition',
        'conj': 'conjunction'
    }
    pos_full = pos_map.get(pos, pos)
    if pos_full == 'noun':
        return meaning
    return f"({pos_full}) {meaning}"

def lookup_word_definition_and_pronunciation(word):
    global DEFINITIONS_CACHE, PRONUNCIATIONS_CACHE
    if not DEFINITIONS_CACHE:
        load_definitions()
    if PRONUNCIATIONS_CACHE is None:
        load_pronunciations()

    word_upper = word.upper().strip()
    pronunciation = PRONUNCIATIONS_CACHE.get(word_upper)
    
    definition = format_resolved_definition(word_upper)
    
    return definition, pronunciation

def ensure_definitions_for_words(words_list):
    global DEFINITIONS_PATH, DEFINITIONS_CACHE
    if not DEFINITIONS_PATH:
        DEFINITIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dictionaries', 'Definitions.txt')
    
    if not DEFINITIONS_CACHE:
        load_definitions()
        
    # 1. Fetch raw definitions for any word in words_list not in cache
    needed_fetch = [w.upper().strip() for w in words_list if w.upper().strip() not in DEFINITIONS_CACHE]
    
    for w_upper in needed_fetch:
        get_definition_cached_or_online_with_guess(w_upper)
        
    # 2. Write resolved definitions to Definitions.txt
    try:
        defs = {}
        if os.path.exists(DEFINITIONS_PATH):
            with open(DEFINITIONS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.split(' - ', 1)
                    if len(parts) == 2:
                        defs[parts[0].strip().upper()] = parts[1].strip()
                        
        written = False
        for w in words_list:
            w_upper = w.upper().strip()
            if w_upper not in defs:
                formatted_def = format_resolved_definition(w_upper)
                if formatted_def:
                    defs[w_upper] = formatted_def
                    DEFINITIONS_CACHE[w_upper] = formatted_def
                    written = True
                    
        if written:
            sorted_keys = sorted(defs.keys())
            temp_path = DEFINITIONS_PATH + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                for k in sorted_keys:
                    f.write(f"{k} - {defs[k]}\n")
            os.replace(temp_path, DEFINITIONS_PATH)
            
            # Reload to sync memory
            DEFINITIONS_CACHE = {}
            load_definitions()
            print(f"[DefinitionsManager] Successfully auto-saved definitions for {len(words_list)} words.")
    except Exception as e:
        print(f"[DefinitionsManager] Error auto-saving definitions: {e}")

def ensure_definitions_background(words_list):
    import threading
    t = threading.Thread(target=ensure_definitions_for_words, args=(words_list,))
    t.daemon = True
    t.start()

def lookup_definition_image(word):
    image_url = None
    word_lower = word.lower()
    for ext in ['png', 'jpg', 'jpeg', 'webp']:
        img_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'definitions', f"{word_lower}.{ext}")
        if os.path.exists(img_path):
            image_url = f"/static/images/definitions/{word_lower}.{ext}"
            break
    return image_url

@app.route('/api/definition', methods=['GET'])
def get_definition():
    word = request.args.get('word', '').upper()
    if not word:
        return jsonify({'error': 'Word parameter required'}), 400

    definition, pronunciation = lookup_word_definition_and_pronunciation(word)
    image_url = lookup_definition_image(word)

    if definition or pronunciation or image_url:
        return jsonify({
            'word': word, 
            'definition': definition or "No definition available for this word.",
            'pronunciation': pronunciation,
            'image_url': image_url
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
STARTUP_WARMUP_COMPLETE = False

@app.route('/api/startup/status', methods=['GET'])
def get_startup_status():
    global STARTUP_WARMUP_COMPLETE
    return jsonify({
        'warmed_up': STARTUP_WARMUP_COMPLETE,
        'csw_loaded': word_validator.csw_words is not None and len(word_validator.csw_words) > 0,
        'definitions_loaded': DEFINITIONS_CACHE is not None and len(DEFINITIONS_CACHE) > 0
    })

# ---------------------------------------------------------------------------
# NIGHTLY CLEANUP: Null out all_words_paths for rounds older than 90 days.
# This column stores tile-path coordinates for every valid word on a board
# (~10 KB per row on average). After 90 days nobody reviews old replays, so
# we shred just that one column while leaving all scores, words, WPM, etc.
# intact. Runs once immediately at startup, then repeats every 24 hours.
# ---------------------------------------------------------------------------
def _prune_old_word_paths():
    import time as _time
    while True:
        try:
            with get_db() as conn:
                result = conn.execute(
                    """UPDATE round_history
                          SET all_words_paths = NULL
                        WHERE all_words_paths IS NOT NULL
                          AND timestamp < datetime('now', '-90 days')"""
                )
                pruned = result.rowcount
            if pruned > 0:
                print(f"[Nightly Cleanup] Nulled all_words_paths on {pruned} round_history row(s) older than 90 days.")
            else:
                print("[Nightly Cleanup] all_words_paths pruning: nothing to clear today.")
        except Exception as _e:
            print(f"[Nightly Cleanup] Error pruning all_words_paths: {_e}")
        _time.sleep(86400)  # 24 hours

threading.Thread(target=_prune_old_word_paths, daemon=True).start()

TOOLS_DICT_CACHE = {}
LAST_ADDED_WORDS_MTIME = None

# --- HIGH-PERFORMANCE C-ACCELERATED MORPHEME METRIC & FEATURE EXTRACTOR ---
import ctypes
import bisect
from functools import lru_cache

_c_morpheme_lib = None

def _init_c_morpheme_metric():
    global _c_morpheme_lib
    if _c_morpheme_lib is not None:
        return _c_morpheme_lib
    
    so_path = os.path.join(os.path.dirname(__file__), 'morpheme_metric.so')
    c_src_path = os.path.join(os.path.dirname(__file__), 'morpheme_metric.c')
    
    # Auto-compile if .so missing
    if not os.path.exists(so_path) and os.path.exists(c_src_path):
        import subprocess
        for comp in ['gcc', 'clang', 'cc']:
            try:
                cmd = [comp, '-O3', '-shared', '-fPIC', c_src_path, '-o', so_path]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    print(f"[MorphemeEngine] Compiled morpheme_metric.so using {comp}")
                    break
            except Exception:
                continue

    if os.path.exists(so_path):
        try:
            lib = ctypes.CDLL(so_path)
            lib.c_calculate_morpheme_metric.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            lib.c_calculate_morpheme_metric.restype = ctypes.c_int
            if hasattr(lib, 'c_build_word_features'):
                lib.c_build_word_features.argtypes = [
                    ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint8)
                ]
                lib.c_build_word_features.restype = None
            _c_morpheme_lib = lib
            print("[MorphemeEngine] Native C Morpheme Engine initialized successfully.")
            return _c_morpheme_lib
        except Exception as e:
            print(f"[MorphemeEngine] Failed to load morpheme_metric.so: {e}")
            _c_morpheme_lib = False
    else:
        _c_morpheme_lib = False
    return _c_morpheme_lib

def load_tools_dictionary(dict_name):
    """Load dictionary for tools into memory cache.
    Always merges the 16+ supplementary word list (16plus.txt) into the result
    so every tool/API route automatically includes long words."""
    global LAST_ADDED_WORDS_MTIME

    # Check for cache invalidation based on added_words.txt modification time
    added_path = os.path.join(os.path.dirname(__file__), 'dictionaries', 'added_words.txt')
    curr_mtime = 0
    if os.path.exists(added_path):
        curr_mtime = os.path.getmtime(added_path)

    if LAST_ADDED_WORDS_MTIME is not None and curr_mtime != LAST_ADDED_WORDS_MTIME:
        print("[Tools] added_words.txt changed. Clearing tools dictionary cache.")
        TOOLS_DICT_CACHE.clear()
        global LISTS_CACHE
        LISTS_CACHE.clear()
        if word_validator:
            word_validator.get_use_added_words(force=True)

    LAST_ADDED_WORDS_MTIME = curr_mtime

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

    # Merge custom Added Words if enabled in game/tools
    if word_validator and word_validator.get_use_added_words() and getattr(word_validator, 'added_words', None):
        words = words | word_validator.added_words

    # --- OPTIMIZATION: PRE-CALCULATE FREQUENCY MATRIX & BITMASKS (C-ACCELERATED) ---
    import numpy as np
    word_list = sorted(list(words))
    count = len(word_list)
    
    matrix = np.zeros((count, 26), dtype=np.uint8)
    masks = np.zeros(count, dtype=np.uint32)
    lens = np.zeros(count, dtype=np.uint8)
    
    c_engine = _init_c_morpheme_metric()
    if c_engine and hasattr(c_engine, 'c_build_word_features') and count > 0:
        packed_bytes = ("\0".join(word_list) + "\0").encode('ascii', errors='ignore')
        offsets = np.zeros(count, dtype=np.int32)
        curr = 0
        for i, w in enumerate(word_list):
            offsets[i] = curr
            curr += len(w) + 1
            
        c_engine.c_build_word_features(
            ctypes.c_char_p(packed_bytes),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(count),
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            lens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        )
    else:
        for i, word in enumerate(word_list):
            mask = 0
            for char in word:
                if 'A' <= char <= 'Z':
                    c_idx = ord(char) - ord('A')
                    matrix[i, c_idx] += 1
                    mask |= (1 << c_idx)
            masks[i] = mask
            lens[i] = len(word)
    
    result = {
        'words': word_list,
        'set': words,
        'matrix': matrix,
        'lens': lens,
        'masks': masks
    }
    TOOLS_DICT_CACHE[cache_key] = result
    return result

def warm_up_server_resources():
    global STARTUP_WARMUP_COMPLETE
    try:
        import time
        time.sleep(1.0)
        print("[Warmup] Starting comprehensive background pre-warming for Tools and Mods...")
        
        # 1. Authoritative pre-load of CSW dictionary into WordValidator
        word_validator.ensure_csw_loaded()
        
        # 2. Authoritative pre-load of Definitions and Pronunciations into memory
        print("[Warmup] Pre-loading Definitions and Pronunciations...")
        load_definitions()
        load_pronunciations()
        
        # 3. Pre-load and pre-compute NumPy matrices & bitmasks for all dictionaries used in Tools
        print("[Warmup] Pre-building Tools dictionary caches (NWL, CSW, ALL, added_words)...")
        for dict_name in ['NWL', 'CSW', 'ALL', 'added_words']:
            try:
                load_tools_dictionary(dict_name)
            except Exception as e:
                print(f"[Warmup] Error pre-building tools dict {dict_name}: {e}")
                
        # 4. Pre-initialize native C morpheme metric engine
        _init_c_morpheme_metric()
        
        # 5. Pre-compute Undefined Words for Mods Definition Management
        print("[Warmup] Pre-computing Undefined Words cache for Mods...")
        try:
            compute_undefined_words(force=True)
        except Exception as e:
            print(f"[Warmup] Error pre-computing undefined words: {e}")

        # 6. Warm up Tools Lists & Endpoint Routes via test_client
        with app.app_context():
            client = app.test_client()
            for lt in ['all', 'nwl', 'csw', 'added', 'likelihood', 'uniques']:
                url = f'/api/tools/lists?list_type={lt}&length=all&starts_with=all'
                print(f"[Warmup] Pre-caching lists for list_type={lt}...")
                client.get(url)
            
            print("[Warmup] Pre-caching added words list...")
            client.get('/api/added_words/list')
            
            print("[Warmup] Priming Tools endpoint routes (Is Valid, WOTD, Unscramble, Find & Count, Sequence, Subanagrams, Combo)...")
            for dict_name in ['NWL', 'CSW', 'ALL', 'added_words']:
                client.post('/api/tools/validate', json={'word': 'APPLE', 'dictionary': dict_name})
            client.get('/api/tools/wotd')
            client.get('/api/tools/unscramble/random')
            client.get('/api/tools/find-count')
            client.post('/api/tools/sequence', json={'sequence': 'WORD', 'dictionary': 'NWL'})
            client.post('/api/tools/subanagrams', json={'input': 'TESTING', 'dictionary': 'NWL'})
            client.post('/api/tools/combo', json={'search_term': 'TEST', 'dictionary': 'NWL'})
            
        STARTUP_WARMUP_COMPLETE = True
        print("[Warmup] Comprehensive Tools & Mods warming completed successfully!")
    except Exception as warmup_err:
        print(f"[Warmup] Error during warmup: {warmup_err}")
        STARTUP_WARMUP_COMPLETE = True

threading.Thread(target=warm_up_server_resources, daemon=True).start()

def calculate_morpheme_metric_py(source, target, limit=3):
    s_len, t_len = len(source), len(target)
    if s_len == 0 or t_len == 0: return 99, 0
    if target in source: return 0, t_len

    # LCS (Linearity)
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
    if linearity == 0 or t_len - linearity > limit:
        return 99, linearity

    best_mp = limit + 1

    char_to_s_indices = {}
    for idx, char in enumerate(source):
        if char not in char_to_s_indices:
            char_to_s_indices[char] = []
        char_to_s_indices[char].append(idx)

    def backtrack(t_idx, used_mask, m_len, min_s, max_s, tails):
        nonlocal best_mp

        if (t_idx - m_len) >= best_mp:
            return

        if m_len > 0:
            relocations = m_len - len(tails)
            paid_deletions = (max_s - min_s + 1) - m_len
            min_cost = relocations + paid_deletions + (t_idx - m_len)
            if min_cost >= best_mp:
                return

        if t_idx == t_len:
            if m_len > 0:
                relocations = m_len - len(tails)
                paid_deletions = (max_s - min_s + 1) - m_len
                actual_cost = relocations + paid_deletions + (t_len - m_len)
                if actual_cost < best_mp:
                    best_mp = actual_cost
            return

        char = target[t_idx]
        if char in char_to_s_indices:
            for s_idx in char_to_s_indices[char]:
                if not (used_mask & (1 << s_idx)):
                    idx_b = bisect.bisect_left(tails, s_idx)
                    if idx_b == len(tails):
                        new_tails = tails + (s_idx,)
                    else:
                        new_tails = tails[:idx_b] + (s_idx,) + tails[idx_b+1:]

                    new_min = s_idx if m_len == 0 else (min_s if min_s < s_idx else s_idx)
                    new_max = s_idx if m_len == 0 else (max_s if max_s > s_idx else s_idx)

                    backtrack(t_idx + 1, used_mask | (1 << s_idx), m_len + 1, new_min, new_max, new_tails)

        backtrack(t_idx + 1, used_mask, m_len, min_s, max_s, tails)

    backtrack(0, 0, 0, 99, -1, ())
    return best_mp, linearity

@lru_cache(maxsize=32768)
def calculate_morpheme_metric(source, target, limit=3):
    lib = _init_c_morpheme_metric()
    if lib:
        s_b = source.encode('ascii')
        t_b = target.encode('ascii')
        return lib.c_calculate_morpheme_metric(s_b, t_b, limit), 0
    return calculate_morpheme_metric_py(source, target, limit)


def check_and_add_mp(mp_groups, source_len, target_len, mp, word):
    """Applies filtering logic for Morpheme Procedure combinations."""
    # mp_groups is a dict of sets
    added = False
    
    if source_len == 3:
        if target_len >= 3: added = True
    elif source_len == 4:
        if target_len >= 4: added = True
    elif source_len == 5:
        if target_len >= 4 and mp <= 3:
            if mp >= 3:
                if target_len >= 5: added = True
            else:
                added = True
    elif source_len == 6:
        if target_len >= 4 and mp <= 3:
            if mp >= 3:
                if target_len >= 5: added = True
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
        if target_len >= 5 and mp <= 5:
             if mp >= 5:
                 if target_len >= 8: added = True
             else:
                 added = True
    
    if added and mp in mp_groups:
        mp_groups[mp].add(word)

def check_and_add_lic(lic_groups, count, target_len, word):
    if count not in lic_groups: lic_groups[count] = set()
    if target_len <= count + 4:
        lic_groups[count].add(word)

COMBO_QUERY_CACHE = {} # LRU cache of search results for instant return

@app.route('/api/tools/combo', methods=['POST'])
def tools_combo_check():
    data = request.json
    search_term = data.get('search_term', '').upper().strip()
    dict_name = data.get('dictionary', 'NWL')
    
    # Check server cache for instant response (< 1ms)
    cache_key = (search_term, dict_name)
    if cache_key in COMBO_QUERY_CACHE:
        return jsonify(COMBO_QUERY_CACHE[cache_key])
    
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
    dict_lens_int = dict_lens.astype(np.int16)
    
    # Capping max_mp at 3 globally to prevent large-MP candidate explosions and focus search on relevant game combinations.
    max_mp = 3
    
    # MP Candidates: absolute length diff <= 3, and shared >= T - max_mp, and unique shared >= 3
    # Check minimum candidate length rules based on specifications
    if source_len >= 8:
        mp_len_mask = (dict_lens_int >= 5)
    elif source_len >= 5:
        mp_len_mask = (dict_lens_int >= 4)
    else:
        mp_len_mask = (dict_lens_int >= 3)

    candidates_mp = np.where(
        passed_mask & 
        (np.abs(dict_lens_int - source_len) <= 3) & 
        (shared_counts >= dict_lens_int - max_mp) &
        mp_len_mask
    )[0]
    
    # LIC Candidates: target_len <= source_len + 4, and shared >= T - 4, and shared_counts >= min_lic_shared
    min_lic_shared = 3 if source_len <= 5 else (4 if source_len <= 6 else 5)
    candidates_lic = np.where(
        (dict_lens_int <= source_len + 4) & 
        (shared_counts >= dict_lens_int - 4) &
        (shared_counts >= min_lic_shared)
    )[0]
    
    candidates = np.union1d(candidates_mp, candidates_lic)
    
    # Sort candidates by lower bound MP cost (ascending), then shared count (descending), then length difference (ascending)
    candidate_shared = shared_counts[candidates]
    candidate_lens = dict_lens_int[candidates]
    candidate_len_diff = np.abs(candidate_lens - source_len)
    lower_bound = candidate_lens - candidate_shared
    sort_order = np.lexsort((candidate_len_diff, -candidate_shared, lower_bound))
    sorted_candidates = candidates[sort_order]
    
    # Initialize Groups (Using sets to prevent O(N^2) search bottleneck)
    mp_groups = {i: set() for i in range(max_mp + 1)} # 0MP to max_mp
    lic_groups = {}
    
    # 0. Guaranteed Substring Check for 0MP: Any valid dictionary word of length >= 4 (or >= 3)
    # contained within search_term or search_term_rev is 0MP by definition
    min_sub_len = 4 if source_len >= 5 else 3
    for l in range(min_sub_len, source_len + 1):
        for start in range(source_len - l + 1):
            sub1 = search_term[start:start+l]
            if sub1 in dict_data['set']:
                mp_groups[0].add(sub1)
            sub2 = search_term_rev[start:start+l]
            if sub2 in dict_data['set']:
                mp_groups[0].add(sub2)
    
    # Native C-accelerated evaluation or optimized Python fallback
    c_engine = _init_c_morpheme_metric()
    s_bytes = search_term.encode('ascii') if c_engine else None

    # --- OPTIMIZED SINGLE-THREADED LOOP ---
    for idx in sorted_candidates:
        word = word_list[idx]
        target_len = int(dict_lens[idx])
        shared_count = int(shared_counts[idx])
            
        # 1. MP Logic
        if np.abs(target_len - source_len) <= 3 and shared_count >= target_len - max_mp:
            if c_engine:
                w_bytes = word.encode('ascii')
                best_mp = c_engine.c_calculate_morpheme_metric(s_bytes, w_bytes, max_mp)
                if best_mp > 1:
                    w_rev_bytes = word[::-1].encode('ascii')
                    m2 = c_engine.c_calculate_morpheme_metric(s_bytes, w_rev_bytes, best_mp - 1)
                    if m2 < best_mp:
                        best_mp = m2
            else:
                best_mp, _ = calculate_morpheme_metric_py(search_term, word, limit=max_mp)
                if best_mp > 1:
                    m2, _ = calculate_morpheme_metric_py(search_term, word[::-1], limit=best_mp - 1)
                    if m2 < best_mp:
                        best_mp = m2

            if best_mp <= max_mp:
                check_and_add_mp(mp_groups, source_len, target_len, best_mp, word)
            
        # 2. LIC Logic
        if target_len <= source_len + 4 and shared_count >= target_len - 4 and shared_count >= min_lic_shared:
            check_and_add_lic(lic_groups, shared_count, target_len, word)

    # Sort groups by length and alphabetically without arbitrary truncation
    for k in mp_groups:
        mp_groups[k] = sorted(list(mp_groups[k]), key=lambda x: (-len(x), x))
        
    for k in lic_groups:
        lic_groups[k] = sorted(list(lic_groups[k]), key=lambda x: (len(x), x))
    
    result_payload = {
        'mp_groups': mp_groups, 
        'lic_groups': lic_groups
    }

    # Store in query cache (prune if exceeds 4096 entries)
    if len(COMBO_QUERY_CACHE) >= 4096:
        COMBO_QUERY_CACHE.clear()
    COMBO_QUERY_CACHE[cache_key] = result_payload

    return jsonify(result_payload)

LISTS_CACHE = {}

@app.route('/api/tools/lists', methods=['GET'])
def tools_get_lists():
    """Returns the 5 specific word lists for the Lists tool with optional filtering."""
    try:
        # Get Filter Params
        length_filter = request.args.get('length')
        start_filter = request.args.get('starts_with')
        list_type = request.args.get('list_type', 'all').lower()
        no_limit = request.args.get('no_limit', 'false').lower() == 'true'

        # Check cache first (only for capped/normal requests)
        cache_key = f"endpoint_{list_type}_{length_filter}_{start_filter}"
        if not no_limit and cache_key in LISTS_CACHE:
            return jsonify(LISTS_CACHE[cache_key])
        
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

        base_dir = os.path.dirname(__file__)
        dict_dir = os.path.join(base_dir, 'dictionaries')
        
        # Make sure CSW is loaded if we need it
        if list_type in ['all', 'csw', 'csw_only', 'new_csw']:
            word_validator.ensure_csw_loaded()
        
        # --- Logic: In-Memory Set Fetching and Filtering ---
        def get_source_set(dict_type):
            if target_len is not None and target_len >= 16:
                base_set = word_validator.long_words
            else:
                if dict_type == 'NWL':
                    base_set = word_validator.nwl_words
                elif dict_type == 'CSW':
                    base_set = word_validator.csw_words
                elif dict_type == 'uniqueNWL':
                    base_set = word_validator.unique_nwl_words
                elif dict_type == 'new_NWL':
                    if 'new_NWL' not in LISTS_CACHE:
                        path = os.path.join(dict_dir, 'new_NWL.txt')
                        if os.path.exists(path):
                            with open(path, 'r') as f:
                                LISTS_CACHE['new_NWL'] = {line.strip().upper() for line in f if line.strip()}
                        else:
                            LISTS_CACHE['new_NWL'] = set()
                    base_set = LISTS_CACHE['new_NWL']
                elif dict_type == 'new_CSW':
                    if 'new_CSW' not in LISTS_CACHE:
                        path = os.path.join(dict_dir, 'new_CSW.txt')
                        if os.path.exists(path):
                            with open(path, 'r') as f:
                                LISTS_CACHE['new_CSW'] = {line.strip().upper() for line in f if line.strip()}
                        else:
                            LISTS_CACHE['new_CSW'] = set()
                    base_set = LISTS_CACHE['new_CSW']
                else:
                    base_set = set()
            
            # Filter the base_set in-memory
            if target_len is None and start_char is None:
                return {w for w in base_set if len(w) < 16}

            filtered = set()
            for w in base_set:
                if (target_len is None or target_len < 16) and len(w) >= 16:
                    continue
                if target_len is not None and len(w) != target_len:
                    continue
                if start_char is not None and not w.startswith(start_char):
                    continue
                filtered.add(w)
            return filtered

        # Conditional fetching based on list_type
        response = {
            'nwl': [], 'csw': [], 'csw_only': [], 'likelihood': [], 'uniques': [], 'added': [],
            'new_nwl': [], 'new_csw': [], 'is_truncated': False
        }

        def cap_list(lst):
            if no_limit:
                return lst  # Return full list when explicitly requested
            if len(lst) > 10000:
                response['is_truncated'] = True
                return lst[:10000]
            return lst

        if list_type in ['all', 'nwl', 'csw_only', 'likelihood']:
            nwl_set = get_source_set('NWL')
            if list_type in ['all', 'nwl']: response['nwl'] = cap_list(sorted(list(nwl_set)))

        if list_type in ['all', 'csw', 'csw_only']:
            csw_set = get_source_set('CSW')
            if list_type in ['all', 'csw']: response['csw'] = cap_list(sorted(list(csw_set)))

        if list_type in ['all', 'csw_only']:
            if 'nwl_set' not in locals(): nwl_set = get_source_set('NWL')
            if 'csw_set' not in locals(): csw_set = get_source_set('CSW')
            response['csw_only'] = cap_list(sorted(list(csw_set - nwl_set)))

        if list_type in ['all', 'likelihood']:
            if 'nwl_set' not in locals(): nwl_set = get_source_set('NWL')
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
            response['likelihood'] = cap_list(likelihood_list)

        if list_type in ['all', 'uniques']:
            response['uniques'] = cap_list(sorted(list(get_source_set('uniqueNWL'))))
            
        if list_type in ['all', 'new_nwl']:
            raw_new_nwl = list(get_source_set('new_NWL'))
            if no_limit:
                response['new_nwl'] = sorted(raw_new_nwl)
            else:
                response['new_nwl'] = list(reversed(raw_new_nwl)) # Show most recent first
                response['new_nwl'] = cap_list(response['new_nwl'])
            
        if list_type in ['all', 'new_csw']:
            raw_new_csw = list(get_source_set('new_CSW'))
            if no_limit:
                response['new_csw'] = sorted(raw_new_csw)
            else:
                response['new_csw'] = list(reversed(raw_new_csw)) # Show most recent first
                response['new_csw'] = cap_list(response['new_csw'])
            
        if list_type in ['all', 'added']:
            # Added Words: Use preloaded in-memory list
            raw_lines = getattr(word_validator, 'added_words_list', [])
            unique_added = []
            for w in raw_lines:
                # Filter by length and start char if provided
                if target_len is not None and len(w) != target_len: continue
                if start_char is not None and not w.startswith(start_char): continue
                unique_added.append(w)

            if no_limit:
                # View Full Lists: always alphabetically sorted (A-to-Z)
                response['added'] = sorted(unique_added)
            else:
                # Default main tab (first 10,000): newest words first
                response['added'] = cap_list(unique_added)

        # Cache response (only for capped/normal requests to avoid polluting cache)
        if not no_limit:
            LISTS_CACHE[cache_key] = response

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
    word_list = dictionary['words']
    lens = dictionary['lens']
    
    for i in range(len(word_list)):
        if target_len is not None and lens[i] != target_len:
            continue
            
        word = word_list[i]
        matched = False
        if mode == 'starts':
            if word.startswith(sequence): matched = True
        elif mode == 'ends':
            if word.endswith(sequence): matched = True
        elif mode == 'contains':
            if sequence in word or seq_rev in word: matched = True
            
        if matched:
            results.append(word)
            
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
    input_len = len(input_text)
    
    input_mask = 0
    for char in input_text:
        if 'A' <= char <= 'Z':
            input_mask |= (1 << (ord(char) - ord('A')))
    input_inv_mask = (~input_mask) & 0xFFFFFFFF
            
    word_list = dictionary['words']
    masks = dictionary['masks']
    lens = dictionary['lens']
    
    results = []
    for i in range(len(word_list)):
        if lens[i] > input_len:
            continue
        if (masks[i] & input_inv_mask) != 0:
            continue
            
        word = word_list[i]
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
    image_url = None
    if is_valid:
        definition, pronunciation = lookup_word_definition_and_pronunciation(word)
        if not definition:
            definition = "No definition available for this word."
        image_url = lookup_definition_image(word)
        
    return jsonify({
        'word': word,
        'is_valid': is_valid,
        'definition': definition,
        'pronunciation': pronunciation,
        'image_url': image_url
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

    # Block timed out users from sending private messages
    user_id = session.get('user_id') or session.get('username')
    is_to, diff_sec, rem_str, to_until, count, reason_val = check_user_timeout(user_id)
    if is_to:
        r_msg = f" Reason: {reason_val}" if reason_val else ""
        return jsonify({'error': f"Your account is currently timed out ({rem_str} remaining).{r_msg} Private messaging is temporarily restricted."}), 403
    
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
            
        # Check if receiver has disabled private messages
        pm_setting = conn.execute("SELECT setting_value FROM user_settings WHERE user_id = ? AND setting_key = 'allow_pm'", (receiver[0],)).fetchone()
        if pm_setting:
            val = str(pm_setting[0]).lower()
            if val in ('false', '0', 'off'):
                return jsonify({'error': 'This user is not accepting private messages'}), 403

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
            
    load_definitions() # Ensure definitions are loaded to prevent slow API fallbacks
    filtered_words = dictionary['words']
    print(f"[RandomWord] target_len: {target_len}, total words in dict: {len(filtered_words)}")
    if target_len:
        filtered_words = [w for w in filtered_words if len(w) == target_len]
        print(f"[RandomWord] filtered words count: {len(filtered_words)}")
    else:
        if dict_name == 'added_words':
            filtered_words = list(filtered_words)
        else:
            filtered_words = [w for w in filtered_words if w in DEFINITIONS_CACHE]
        print(f"[RandomWord] filtered words count (with definitions check): {len(filtered_words)}")
        
    if not filtered_words:
        return jsonify({'error': 'No words found for the specified criteria'}), 404
        
    import random
    random_word = random.choice(filtered_words)
    
    # Get definition and pronunciation
    definition, pronunciation = lookup_word_definition_and_pronunciation(random_word)
    if not definition:
        definition = "No definition available for this word."
    image_url = lookup_definition_image(random_word)
    
    return jsonify({
        'word': random_word,
        'definition': definition,
        'pronunciation': pronunciation,
        'image_url': image_url
    })

@app.route('/api/tools/wotd', methods=['GET'])
def tools_wotd():
    """Returns a deterministic Word of the Day based on the current date in Chicago timezone."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import hashlib
    
    # Standardize to Chicago timezone matching the rest of the game platform
    chicago_tz = ZoneInfo("America/Chicago")
    today_str = datetime.now(chicago_tz).strftime('%Y-%m-%d')
    
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
    image_url = lookup_definition_image(wotd)
    
    response = jsonify({
        'word': wotd,
        'date': today_str,
        'definition': definition,
        'pronunciation': pronunciation,
        'image_url': image_url
    })
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/api/tools/find-count', methods=['GET'])
@login_required
def tools_find_count():
    word = request.args.get('word', '').strip().upper()
    if not word:
        return jsonify({'error': 'Word required'}), 400

    try:
        finds = _get_word_finds(word)
        total_count = len(finds)
        recent_finds = finds[:10]
        
        # Check if the word is valid in dictionaries (NWL, CSW, supplementary 16+, or added words)
        is_valid = word_validator.is_valid_word(word, 'ALL + AW', use_added_words=True)
        
        return jsonify({
            'word': word,
            'count': total_count,
            'recent': recent_finds,
            'is_valid': is_valid
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/tools/random-words', methods=['GET'])
@login_required
def tools_random_words():
    try:
        word_validator.ensure_csw_loaded()
        dict_type = request.args.get('dictionary', 'ALL').upper()
        sampled = []
        for length in [5, 6, 7, 8, 9, 10]:
            if dict_type == 'NWL':
                source_set = word_validator.nwl_by_len.get(length, [])
            elif dict_type == 'CSW':
                source_set = word_validator.csw_by_len.get(length, [])
            elif dict_type == 'AW':
                source_set = [w for w in word_validator.added_words if len(w) == length]
            else:  # ALL
                nwl_set = word_validator.nwl_by_len.get(length, [])
                csw_set = word_validator.csw_by_len.get(length, [])
                aw_set = [w for w in word_validator.added_words if len(w) == length]
                source_set = list(set(nwl_set) | set(csw_set) | set(aw_set))
            
            if source_set:
                sampled.append(random.choice(source_set))
            else:
                fallbacks = {5: 'KUDZU', 6: 'BURANS', 7: 'PLEROMA', 8: 'PRONOTUM', 9: 'MANGETOUT', 10: 'OVERSTATES'}
                sampled.append(fallbacks.get(length, 'MORPHEME'))
        return jsonify({'words': sampled})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


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

        # Canonical category ordering with Complaints placed under Suggestions
        ORDER_MAP = {
            'general': 1,
            'tips, tricks, and strategies': 2,
            'screenshots': 3,
            'introduce yourself': 4,
            'news': 5,
            'suggestions/ideas': 6,
            'suggestions': 6,
            'complaints': 7,
            'bugs/errors': 8,
        }
        categories.sort(key=lambda c: ORDER_MAP.get(str(c.get('name', '')).strip().lower(), 99))
        return jsonify({'categories': categories})
    finally:
        conn.close()

def parse_image_urls(val):
    if not val:
        return []
    val_str = str(val).strip()
    if val_str.startswith('['):
        try:
            urls = json.loads(val_str)
            if isinstance(urls, list):
                return [u for u in urls if isinstance(u, str) and u.strip()]
        except Exception:
            pass
    if ',' in val_str:
        return [u.strip() for u in val_str.split(',') if u.strip()]
    return [val_str] if val_str else []

@app.route('/api/forum/posts/<int:category_id>', methods=['GET'])
def get_forum_posts(category_id):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute('''
            SELECT p.*, u.username, u.avatar_url, u.country_flag,
            (SELECT COUNT(*) FROM forum_comments WHERE post_id = p.id) as comment_count,
            COALESCE((SELECT MAX(timestamp) FROM forum_comments WHERE post_id = p.id), p.timestamp) as last_activity
            FROM forum_posts p
            JOIN users u ON p.user_id = u.id
            WHERE p.category_id = ?
            ORDER BY last_activity DESC
            LIMIT 200
        ''', (category_id,)).fetchall()
        posts = []
        for r in rows:
            d = dict(r)
            urls = parse_image_urls(d.get('image_url'))
            d['image_urls'] = urls
            d['image_url'] = urls[0] if urls else None
            posts.append(d)
        return jsonify({'posts': posts})
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

        posts = []
        for row in posts_rows:
            d = dict(row)
            urls = parse_image_urls(d.get('image_url'))
            d['image_urls'] = urls
            d['image_url'] = urls[0] if urls else None
            posts.append(d)

        comments = []
        for row in comments_rows:
            d = dict(row)
            urls = parse_image_urls(d.get('image_url'))
            d['image_urls'] = urls
            d['image_url'] = urls[0] if urls else None
            comments.append(d)
        
        print(f"[Forum] User search for '{username}' found {len(posts)} threads and {len(comments)} replies.")
        
        # Combine and sort by timestamp DESC
        all_items = posts + comments
        all_items.sort(key=lambda x: x['timestamp'] or '', reverse=True)
        all_items = all_items[:100]
        
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

        post_dict = dict(post)
        p_urls = parse_image_urls(post_dict.get('image_url'))
        post_dict['image_urls'] = p_urls
        post_dict['image_url'] = p_urls[0] if p_urls else None

        comments_list = []
        for c in comments:
            cd = dict(c)
            c_urls = parse_image_urls(cd.get('image_url'))
            cd['image_urls'] = c_urls
            cd['image_url'] = c_urls[0] if c_urls else None
            comments_list.append(cd)
        
        response_data = {
            'post': post_dict,
            'comments': comments_list,
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
        
    upload_files = []
    for key in ['images', 'image', 'images[]']:
        for f in request.files.getlist(key):
            if f and f.filename != '' and allowed_file(f.filename):
                if f not in upload_files:
                    upload_files.append(f)
    upload_files = upload_files[:4]

    saved_urls = []
    text_to_moderate = f"Title: {title}\nContent: {content}"
    
    for idx, file in enumerate(upload_files):
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        image_bytes = file.read()
        file.seek(0)
        
        mod_text = text_to_moderate if idx == 0 else ""
        moderation_res = moderate_content(text=mod_text, image_bytes=image_bytes, mime_type=ext)
        if moderation_res.get("inappropriate"):
            return jsonify({'error': f"Inappropriate content detected in image #{idx+1}: {moderation_res.get('reason')}"}), 400

        import uuid
        filename = f"{uuid.uuid4()}.{ext}"
        file.save(os.path.join(app.config['FORUM_UPLOAD_FOLDER'], filename))
        saved_urls.append(f"/static/uploads/forum/{filename}")

    if not upload_files:
        moderation_res = moderate_content(text=text_to_moderate)
        if moderation_res.get("inappropriate"):
            return jsonify({'error': f"Inappropriate content detected: {moderation_res.get('reason')}"}), 400

    image_url_db = json.dumps(saved_urls) if saved_urls else None

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
        ''', (category_id, session['user_id'], title, content, image_url_db))
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
        
    data = request.form
    post_id = data.get('post_id')
    content = data.get('content')
    
    if not post_id or not content:
        return jsonify({'error': 'Missing fields'}), 400
        
    upload_files = []
    for key in ['images', 'image', 'images[]']:
        for f in request.files.getlist(key):
            if f and f.filename != '' and allowed_file(f.filename):
                if f not in upload_files:
                    upload_files.append(f)
    upload_files = upload_files[:4]

    saved_urls = []
    for idx, file in enumerate(upload_files):
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        image_bytes = file.read()
        file.seek(0)
        
        mod_text = content if idx == 0 else ""
        moderation_res = moderate_content(text=mod_text, image_bytes=image_bytes, mime_type=ext)
        if moderation_res.get("inappropriate"):
            return jsonify({'error': f"Inappropriate content detected in image #{idx+1}: {moderation_res.get('reason')}"}), 400

        import uuid
        filename = f"reply_{uuid.uuid4()}.{ext}"
        file.save(os.path.join(app.config['FORUM_UPLOAD_FOLDER'], filename))
        saved_urls.append(f"/static/uploads/forum/{filename}")

    if not upload_files:
        moderation_res = moderate_content(text=content)
        if moderation_res.get("inappropriate"):
            return jsonify({'error': f"Inappropriate content detected: {moderation_res.get('reason')}"}), 400

    image_url_db = json.dumps(saved_urls) if saved_urls else None

    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute('''
            INSERT INTO forum_comments (post_id, user_id, content, image_url)
            VALUES (?, ?, ?, ?)
        ''', (post_id, session['user_id'], content, image_url_db))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/daily-score-sums', methods=['GET'])
def get_daily_score_sums():
    raw_room_id = request.args.get('room_id', '24h_4x4')
    board_dimensions = request.args.get('board_dimensions', '')
    canonical_key = normalize_24h_room_key(raw_room_id, board_dimensions)
    dims = canonical_key.replace('24h_', '')

    try:
        with get_db(row_factory=sqlite3.Row) as conn:
            # 1. Automatic backfill / recovery from historical round_history if missing
            conn.execute('''
                INSERT INTO daily_score_sums (user_id, room_id, score_sum)
                SELECT rh.user_id, ?, SUM(rh.total_score)
                FROM round_history rh
                WHERE rh.user_id > 0 
                  AND (rh.board_dimensions = ? OR rh.room_id LIKE ? OR rh.room_id = ?)
                  AND (rh.round_duration >= 7200 OR rh.room_id LIKE '%86400%' OR rh.room_id LIKE '%24h%')
                GROUP BY rh.user_id
                HAVING SUM(rh.total_score) > 0
                ON CONFLICT(user_id, room_id) DO UPDATE SET score_sum = MAX(daily_score_sums.score_sum, excluded.score_sum)
            ''', (canonical_key, dims, f"%{dims}%", canonical_key))

            # 2. Fetch all scores for this 24h room from daily_score_sums (only score_sum >= 1)
            cursor = conn.execute('''
                SELECT u.username, d.user_id, SUM(d.score_sum) as score_sum
                FROM daily_score_sums d
                JOIN users u ON d.user_id = u.id
                WHERE (d.room_id = ? OR d.room_id = ? OR d.room_id LIKE ?)
                  AND d.score_sum > 0
                GROUP BY d.user_id, u.username
                HAVING SUM(d.score_sum) > 0
                ORDER BY score_sum DESC
            ''', (canonical_key, raw_room_id, f"%{dims}%"))
            rows = cursor.fetchall()
            
            # Score Sum reflects finalized round totals summed at 12 AM boundary and is never overwritten during ongoing rounds
            players = [{'username': row['username'], 'score_sum': row['score_sum']} for row in rows if row['score_sum'] and row['score_sum'] > 0]
            return jsonify({'players': players, 'room_id': canonical_key})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard_data():
    import time as _t
    # --- Params ---
    period     = request.args.get('period', 'day')
    game_type  = request.args.get('game_type', 'all')
    dims       = request.args.get('board_dimensions', 'all')
    time_limit = request.args.get('time_limit', 'all')

    # --- TTL Cache check ---
    cache_key = f"{period}|{game_type}|{dims}|{time_limit}"
    now = _t.time()
    if cache_key in _lb_cache and now < _lb_cache_expiry.get(cache_key, 0):
        return jsonify(_lb_cache[cache_key])

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # Base filters: exclude Guests, 24h rooms (>= 7200s), and Valued Letters format rounds
        params = []
        where_clauses = [
            "u.username NOT LIKE 'Guest_%'",
            "rh.round_duration < 7200",
            "LOWER(COALESCE(rh.board_format, '')) NOT LIKE '%valued%'"
        ]

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
            if game_type == 'all' or game_type == 'accumulative':
                where_clauses.append("(rh.game_type != 'accumulative' OR rh.round_duration != 600)")

        # Calculate Chicago local time boundaries
        chicago_tz = ZoneInfo("America/Chicago")
        chicago_now = datetime.datetime.now(chicago_tz)
        chicago_today_str = chicago_now.strftime('%Y-%m-%d')
        chicago_week_ago_str = (chicago_now - datetime.timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        chicago_month_ago_str = (chicago_now - datetime.timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        chicago_year_ago_str = (chicago_now - datetime.timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')

        period_clause = "1=1"
        if period == 'day':
            period_clause = f"date(timestamp) = '{chicago_today_str}'"
        elif period == 'week':
            period_clause = f"timestamp >= '{chicago_week_ago_str}'"
        elif period == 'month':
            period_clause = f"timestamp >= '{chicago_month_ago_str}'"
        elif period == 'year':
            period_clause = f"timestamp >= '{chicago_year_ago_str}'"

        outer_period_clause = period_clause.replace("timestamp", "rh.timestamp")
        where_clauses.append(outer_period_clause)
        base_where = " AND ".join(where_clauses)

        # Row cap: prevent unbounded full-table scans for Python-processed queries
        # Larger periods need a higher cap to ensure top-50 accuracy
        _row_cap = 500 if period == 'day' else (1000 if period == 'week' else 2000)

        # Helper to infer dimensions if missing
        def infer_lb_dims(b_dims, b_json, g_type):
            if b_dims:
                return b_dims
            if g_type == '3d':
                return '3x3x3'
            if b_json:
                try:
                    bj = json.loads(b_json) if isinstance(b_json, str) else b_json
                    if isinstance(bj, list) and len(bj) > 0:
                        if isinstance(bj[0], list):
                            if len(bj) == 3 and len(bj[0]) == 3 and isinstance(bj[0][0], list):
                                return '3x3x3'
                            return f"{len(bj)}x{len(bj[0])}"
                except:
                    pass
            return '4x4'

        # 1. Best Scores
        scores = conn.execute(f"""
            SELECT * FROM (
                SELECT rh.total_score, COALESCE(u.rating, 1200) as user_rating, u.username, u.country_flag, u.avatar_url,
                       rh.room_id, rh.round_number, rh.timestamp, rh.board_json, rh.words_json,
                       rh.round_duration, rh.id, rh.game_type, rh.board_dimensions, rh.round_start_time,
                       ROW_NUMBER() OVER (PARTITION BY rh.user_id ORDER BY rh.total_score DESC, rh.timestamp DESC) as rn
                FROM round_history rh
                JOIN users u ON rh.user_id = u.id
                WHERE {base_where}
            ) sub WHERE rn = 1
            ORDER BY total_score DESC, timestamp DESC LIMIT 50
        """, params).fetchall()

        # 2. Best Words
        words = conn.execute(f"""
            SELECT * FROM (
                SELECT rh.best_word, rh.best_word_score, COALESCE(u.rating, 1200) as user_rating, u.username, u.country_flag, u.avatar_url,
                       rh.room_id, rh.round_number, rh.timestamp, rh.board_json, rh.words_json,
                       rh.round_duration, rh.id, rh.game_type, rh.board_dimensions, rh.round_start_time,
                       ROW_NUMBER() OVER (PARTITION BY rh.user_id ORDER BY rh.best_word_score DESC, rh.timestamp DESC) as rn
                FROM round_history rh
                JOIN users u ON rh.user_id = u.id
                WHERE {base_where} AND rh.best_word IS NOT NULL
            ) sub WHERE rn = 1
            ORDER BY best_word_score DESC, timestamp DESC LIMIT 50
        """, params).fetchall()

        # 3. Best PE
        pes = conn.execute(f"""
            SELECT * FROM (
                SELECT rh.performance_ratio, rh.total_score, COALESCE(u.rating, 1200) as user_rating, u.username, u.country_flag, u.avatar_url,
                       rh.room_id, rh.round_number, rh.timestamp, rh.board_json, rh.words_json,
                       rh.round_duration, rh.id, rh.game_type, rh.board_dimensions, rh.total_words_avail, rh.round_start_time,
                       ROW_NUMBER() OVER (PARTITION BY rh.user_id ORDER BY rh.performance_ratio DESC, rh.timestamp DESC) as rn
                FROM round_history rh
                JOIN users u ON rh.user_id = u.id
                WHERE {base_where} AND rh.performance_ratio > 0
            ) sub WHERE rn = 1
            ORDER BY performance_ratio DESC, timestamp DESC LIMIT 50
        """, params).fetchall()

        # 4. Best Pct Found (capped to avoid full-table JSON scan)
        cursor_pcts = conn.execute(f"""
            SELECT rh.total_score, COALESCE(u.rating, 1200) as user_rating, u.username, u.country_flag, u.avatar_url,
                   rh.room_id, rh.round_number, rh.timestamp, rh.board_json, rh.words_json,
                   rh.round_duration, rh.id, rh.game_type, rh.board_dimensions, rh.total_words_avail, rh.round_start_time
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE {base_where} AND rh.total_words_avail > 0
            ORDER BY rh.timestamp DESC
            LIMIT {_row_cap}
        """, params).fetchall()

        pcts_processed_all = []
        for r in cursor_pcts:
            d = dict(r)
            try:
                words_list = json.loads(d.get('words_json', '[]'))
                twa = d.get('total_words_avail', 0)
                d['pct_found'] = round(len(words_list) / twa * 100, 1) if twa > 0 else 0
            except:
                d['pct_found'] = 0
            pcts_processed_all.append(d)

        user_pcts_max = {}
        user_pcts_list = {}
        for d in pcts_processed_all:
            user = d['username']
            if user not in user_pcts_max or d['pct_found'] > user_pcts_max[user]['pct_found'] or (
                    d['pct_found'] == user_pcts_max[user]['pct_found'] and d['timestamp'] > user_pcts_max[user]['timestamp']):
                user_pcts_max[user] = d
            user_pcts_list.setdefault(user, []).append(d['pct_found'])

        for user, d in user_pcts_max.items():
            pcts = user_pcts_list.get(user, [])
            d['avg_pct'] = round(sum(pcts) / len(pcts), 1) if pcts else 0

        best_pcts = sorted(user_pcts_max.values(), key=lambda x: (x['pct_found'], x['timestamp']), reverse=True)[:50]

        # 5. Best Ratings
        ratings = conn.execute(f"""
            SELECT * FROM (
                SELECT rh.user_rating as max_rating, COALESCE(u.rating, 1200) as user_rating, u.username, u.country_flag, u.avatar_url,
                       rh.room_id, rh.timestamp, rh.game_type, rh.board_dimensions, rh.round_duration,
                       ROW_NUMBER() OVER (PARTITION BY rh.user_id ORDER BY rh.user_rating DESC, rh.timestamp DESC) as rn
                FROM round_history rh
                JOIN users u ON rh.user_id = u.id
                WHERE {base_where}
            ) sub WHERE rn = 1
            ORDER BY max_rating DESC, timestamp DESC LIMIT 50
        """, params).fetchall()

        # 6. Avg Score
        avgs = conn.execute(f"""
            SELECT AVG(rh.total_score) as avg_score, COUNT(*) as games, MAX(rh.timestamp) as last_active,
                   COALESCE(u.rating, 1200) as user_rating, u.username, u.country_flag, u.avatar_url
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE {base_where}
            GROUP BY u.id
            HAVING games >= 1
            ORDER BY avg_score DESC LIMIT 50
        """, params).fetchall()

        # 7. Obscure words (capped — high-scoring rounds are most likely candidates)
        cursor_obscure = conn.execute(f"""
            SELECT rh.total_score, COALESCE(u.rating, 1200) as user_rating, u.username, u.country_flag, u.avatar_url,
                   rh.room_id, rh.round_number, rh.timestamp, rh.board_json, rh.words_json,
                   rh.round_duration, rh.id, rh.game_type, rh.board_dimensions, rh.round_start_time
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE {base_where}
            ORDER BY rh.total_score DESC
            LIMIT {_row_cap}
        """, params).fetchall()

        word_validator.ensure_csw_loaded()
        user_obscure = {}
        for r in cursor_obscure:
            d = dict(r)
            try:
                words_list = json.loads(d.get('words_json', '[]'))
                d['obscure_count'] = sum(1 for w in words_list if w.get('word', '').upper() in word_validator.unique_csw_words)
            except:
                d['obscure_count'] = 0
            user = d['username']
            if user not in user_obscure or d['obscure_count'] > user_obscure[user]['obscure_count'] or (
                    d['obscure_count'] == user_obscure[user]['obscure_count'] and d['timestamp'] > user_obscure[user]['timestamp']):
                user_obscure[user] = d

        best_obscure = sorted([x for x in user_obscure.values() if x['obscure_count'] > 0],
                              key=lambda x: (x['obscure_count'], x['timestamp']), reverse=True)[:50]

        # 8. Avg Pct Found (capped)
        cursor_avg_pct = conn.execute(f"""
            SELECT rh.total_score, COALESCE(u.rating, 1200) as user_rating, u.username, u.country_flag, u.avatar_url,
                   rh.total_words_avail, rh.words_json
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            WHERE {base_where} AND rh.total_words_avail > 0
            ORDER BY rh.timestamp DESC
            LIMIT {_row_cap}
        """, params).fetchall()

        user_pcts = {}
        for r in cursor_avg_pct:
            d = dict(r)
            user = d['username']
            try:
                words_list = json.loads(d.get('words_json', '[]'))
                twa = d.get('total_words_avail', 0)
                pct = len(words_list) / twa if twa > 0 else 0
                user_pcts.setdefault(user, {'pcts': [], 'last': d})
                user_pcts[user]['pcts'].append(pct)
                user_pcts[user]['last'] = d
            except:
                pass

        avg_pcts = []
        for user, info in user_pcts.items():
            pcts = info['pcts']
            d = info['last']
            avg_pcts.append({
                'username': user,
                'country_flag': d['country_flag'],
                'avatar_url': d['avatar_url'],
                'user_rating': d.get('user_rating', 1200),
                'avg_pct': round(sum(pcts) / len(pcts) * 100, 1),
                'games': len(pcts)
            })
        best_avg_pcts = sorted(avg_pcts, key=lambda x: x['avg_pct'], reverse=True)[:50]

        # 9. Most Games Played
        if game_type != 'all':
            rating_pattern = f"{game_type}|%"
            if dims != 'all' and time_limit != 'all': rating_pattern = f"{game_type}|{dims}|{time_limit}"
            elif dims != 'all': rating_pattern = f"{game_type}|{dims}|%"
            elif time_limit != 'all': rating_pattern = f"{game_type}|%|{time_limit}"
            is_24h_filter = (time_limit != 'all' and int(time_limit) >= 7200)
            rating_subquery = "u.rating" if is_24h_filter else "COALESCE((SELECT MAX(rating) FROM user_ratings WHERE user_id = u.id AND config_key LIKE ?), 1200)"
            m_sql = f"""SELECT u.username, u.country_flag, u.avatar_url, MAX(rh.timestamp) as last_active,
                               COUNT(rh.id) as game_count, {rating_subquery} as rating,
                               COALESCE(u.rating, 1200) as user_rating,
                               rh.game_type, rh.board_dimensions, rh.round_duration
                        FROM round_history rh JOIN users u ON rh.user_id = u.id
                        WHERE {base_where} GROUP BY u.id ORDER BY game_count DESC LIMIT 50"""
            m_params = [rating_pattern] + params if not is_24h_filter else params
        else:
            m_sql = f"""SELECT u.username, u.country_flag, u.avatar_url, MAX(rh.timestamp) as last_active,
                               COUNT(rh.id) as game_count,
                               (SELECT MAX(rating) FROM user_ratings
                                WHERE user_id = u.id
                                AND config_key IN (
                                    SELECT DISTINCT (game_type || '|' || board_dimensions || '|' || round_duration)
                                    FROM round_history WHERE user_id = u.id AND {period_clause}
                                )) as rating,
                               COALESCE(u.rating, 1200) as user_rating,
                               rh.game_type, rh.board_dimensions, rh.round_duration
                        FROM round_history rh JOIN users u ON rh.user_id = u.id
                        WHERE {base_where} GROUP BY u.id ORDER BY game_count DESC LIMIT 50"""
            m_params = params
        most_games = conn.execute(m_sql, m_params).fetchall()

        # 10. Current Ratings (with config metadata)
        gt_pat = game_type if game_type != 'all' else '%'
        dim_pat = dims if dims != 'all' else '%'
        time_pat = time_limit if time_limit != 'all' else '%'
        cur_rating_pattern = f"{gt_pat}|{dim_pat}|{time_pat}"

        current_ratings_rows = conn.execute(f"""
            SELECT u.username, u.country_flag, u.avatar_url, MAX(rh.timestamp) as last_active,
                   COALESCE(ur_best.rating, 1200) as rating,
                   COALESCE(u.games_played, 0) as user_total_games,
                   COUNT(rh.id) as period_games,
                   ur_best.config_key,
                   rh.game_type, rh.board_dimensions, rh.round_duration
            FROM round_history rh
            JOIN users u ON rh.user_id = u.id
            LEFT JOIN (
                SELECT user_id, rating, config_key,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY rating DESC) as rn
                FROM user_ratings
                WHERE config_key LIKE ?
            ) ur_best ON ur_best.user_id = u.id AND ur_best.rn = 1
            WHERE {base_where}
            GROUP BY u.id
            ORDER BY rating DESC, last_active DESC LIMIT 1000
        """, [cur_rating_pattern] + params).fetchall()

        current_ratings = []
        for r in current_ratings_rows:
            d = dict(r)
            rating_val = d.get('rating', 1200)
            total_games = max(d.get('user_total_games', 0), d.get('period_games', 0))
            # Rule: If rating is 1200, must have over 10 games played to be included
            if rating_val == 1200 and total_games <= 10:
                continue

            cfg_k = d.get('config_key') or ''
            parts = cfg_k.split('|')
            if len(parts) == 3:
                d['game_type'] = parts[0]
                d['board_dimensions'] = parts[1]
                try:
                    d['round_duration'] = int(parts[2])
                except:
                    d['round_duration'] = parts[2]
            else:
                d['game_type'] = d.get('game_type') or (game_type if game_type != 'all' else 'accumulative')
                d['board_dimensions'] = d.get('board_dimensions') or (dims if dims != 'all' else '4x4')
                d['round_duration'] = d.get('round_duration') or (int(time_limit) if time_limit != 'all' else 180)
            current_ratings.append(d)

        def format_result_timestamps(lst):
            formatted = []
            for item in lst:
                d = dict(item)
                if 'timestamp' in d:
                    d['timestamp'] = format_chicago_to_utc(d['timestamp'])
                if 'last_active' in d:
                    d['last_active'] = format_chicago_to_utc(d['last_active'])
                if 'board_dimensions' in d or 'board_json' in d or 'game_type' in d:
                    d['board_dimensions'] = infer_lb_dims(d.get('board_dimensions'), d.get('board_json'), d.get('game_type'))
                formatted.append(d)
            return formatted

        pes_processed = []
        for r in pes:
            d = dict(r)
            try:
                words_list = json.loads(d.get('words_json', '[]'))
                twa = d.get('total_words_avail', 0)
                d['pct_found'] = round(len(words_list) / twa * 100, 1) if twa > 0 else 0
            except:
                d['pct_found'] = 0
            pes_processed.append(d)

        result = {
            'best_scores':    format_result_timestamps(scores),
            'best_words':     format_result_timestamps(words),
            'best_pes':       format_result_timestamps(pes_processed),
            'best_pcts':      format_result_timestamps(best_pcts),
            'best_ratings':   format_result_timestamps(ratings),
            'avg_scores':     format_result_timestamps(avgs),
            'current_ratings': format_result_timestamps(current_ratings),
            'most_games':     format_result_timestamps(most_games),
            'best_avg_pcts':  format_result_timestamps(best_avg_pcts),
            'best_obscure':   format_result_timestamps(best_obscure)
        }

        # Store in TTL cache
        _lb_cache[cache_key] = result
        _lb_cache_expiry[cache_key] = now + _LB_CACHE_TTL

        return jsonify(result)

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
        with get_db(row_factory=sqlite3.Row) as conn:
            # Auto-finalize expired stale turn if user left mid-round
            if t['status'] == 'active':
                row = conn.execute('SELECT * FROM tournament_scores WHERE tournament_id = ? AND round_number = ? AND user_id = ?',
                                   (t['id'], t['current_round'], user_id)).fetchone()
                if row and row['submitted_at'] is None:
                    elapsed = time.time() - row['round_start_time']
                    params = json.loads(t['parameters'])
                    time_limit = int(params.get('time_limit', 60))
                    if elapsed > time_limit + 15: # 15-second grace buffer
                        conn.execute('UPDATE tournament_scores SET submitted_at = ? WHERE tournament_id = ? AND round_number = ? AND user_id = ?',
                                     (time.time(), t['id'], t['current_round'], user_id))
                        conn.commit()
                        print(f"[Tournament] Auto-finalized expired turn for user {user_id} (elapsed: {elapsed}s)")

            p = conn.execute('SELECT * FROM tournament_participants WHERE tournament_id = ? AND user_id = ?', 
                            (t['id'], user_id)).fetchone()
            
            if p:
                user_status['status'] = p['status']
                user_status['final_rank'] = p['final_rank']
                user_status['has_turn'] = tournament_manager.has_user_turn(t['id'], user_id)
                matchup = tournament_manager.get_user_matchup(t['id'], t['current_round'], user_id)
                if not matchup and p['status'] == 'eliminated':
                    for prev_r in range(t['current_round'] - 1, 0, -1):
                        matchup = tournament_manager.get_user_matchup(t['id'], prev_r, user_id)
                        if matchup:
                            break
                user_status['matchup'] = matchup
        
    history = tournament_manager.get_history()
    
    # Get round end time if active
    round_end_time = 0
    if t and t['status'] == 'active':
        with get_db(auto_commit=False) as conn:
            r = conn.execute('SELECT end_time FROM tournament_rounds WHERE tournament_id = ? AND round_number = ?',
                            (t['id'], t['current_round'])).fetchone()
            if r: round_end_time = r[0]

    round_scores = []
    if t and t['status'] == 'active':
        # Check if requesting user has completed/submitted their turn
        user_has_completed = False
        if 'user_id' in session and not session.get('is_guest'):
            user_id = session['user_id']
            with get_db(auto_commit=False) as conn:
                completed_row = conn.execute('SELECT 1 FROM tournament_scores WHERE tournament_id = ? AND round_number = ? AND user_id = ? AND submitted_at IS NOT NULL',
                                             (t['id'], t['current_round'], user_id)).fetchone()
                if completed_row:
                    user_has_completed = True

        raw_scores = tournament_manager.get_round_scores(t['id'], t['current_round'])
        for rs in raw_scores:
            rs_dict = dict(rs)
            is_own = ('user_id' in session and rs_dict['user_id'] == session['user_id'])
            
            # Censor if:
            # 1. The round is still active/ongoing
            # 2. AND the requesting user has NOT completed/submitted their own turn (and it's not their own score)
            is_round_active = (time.time() < round_end_time)
            should_censor = not is_own and is_round_active and not user_has_completed
            
            if should_censor:
                rs_dict['submitted_words'] = []
                rs_dict['board_data'] = None
            else:
                if rs_dict.get('submitted_words'):
                    rs_dict['submitted_words'] = json.loads(rs_dict['submitted_words'])
            round_scores.append(rs_dict)

    with get_db(auto_commit=False) as conn:
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

    from tournament_logic import GRACE_PERIOD as _GRACE_PERIOD
    grace_end_time = (t['completed_at'] + _GRACE_PERIOD) if t.get('completed_at') else 0

    return jsonify({
        'status': t['status'],
        'id': t['id'],
        'start_date': t['start_date'],
        'parameters': params,
        'current_round': t['current_round'],
        'completed_at': t['completed_at'],
        'grace_end_time': grace_end_time,
        'round_end_time': round_end_time,
        'user_status': user_status,
        'history': history,
        'round_scores': round_scores,
        'standings': tournament_manager.get_tournament_standings(t['id']),
        'all_matchups': tournament_manager.get_all_matchups(t['id'], t['current_round']) if t['status'] == 'active' else [],
        'all_tournament_matchups': tournament_manager.get_all_tournament_matchups(t['id']) if t['status'] in ('active', 'completed') else [],
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
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # Check active participant status
        p = conn.execute('SELECT status FROM tournament_participants WHERE tournament_id = ? AND user_id = ?', 
                         (t['id'], user_id)).fetchone()
        if not p or p['status'] != 'active':
            return jsonify({'error': 'Not an active participant'}), 403
            
        # Check if already started or finished this round
        existing = conn.execute('SELECT * FROM tournament_scores WHERE tournament_id = ? AND round_number = ? AND user_id = ?',
                                (t['id'], t['current_round'], user_id)).fetchone()
                                
        if existing:
            if existing['submitted_at'] is not None:
                return jsonify({'error': 'Not your turn or already played'}), 403
            else:
                # User started previously but left mid-round and is returning. End turn!
                conn.execute('UPDATE tournament_scores SET submitted_at = ? WHERE tournament_id = ? AND round_number = ? AND user_id = ?',
                             (time.time(), t['id'], t['current_round'], user_id))
                conn.commit()
                return jsonify({'error': 'Turn ended because you left mid-round'}), 403
                
        # First time starting. Initialize row.
        now = time.time()
        conn.execute('''
            INSERT INTO tournament_scores (tournament_id, round_number, user_id, score, submitted_words, submitted_at, round_start_time)
            VALUES (?, ?, ?, 0, '[]', NULL, ?)
        ''', (t['id'], t['current_round'], user_id, now))
        conn.commit()
        
        r = conn.execute('SELECT * FROM tournament_rounds WHERE tournament_id = ? AND round_number = ?',
                        (t['id'], t['current_round'])).fetchone()
    finally:
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
@login_required
def get_tournament_winner_turn(tid, username):
    if session.get('is_guest'):
        return jsonify({'error': 'Guest access denied'}), 403
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
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    p = conn.execute('SELECT status FROM tournament_participants WHERE tournament_id = ? AND user_id = ?', (tid, user_id)).fetchone()
    if not p or p['status'] != 'active':
        conn.close()
        return jsonify({'error': 'Not an active participant'}), 403
        
    existing = conn.execute('SELECT submitted_at FROM tournament_scores WHERE tournament_id = ? AND round_number = ? AND user_id = ?',
                            (tid, round_num, user_id)).fetchone()
    conn.close()
    
    if not existing:
        return jsonify({'error': 'Turn not started yet'}), 403
        
    if existing['submitted_at'] is not None:
        return jsonify({'error': 'Turn already completed/submitted'}), 403
        
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
    use_aw = params.get('use_added_words', False)
    
    # LOAD DICTIONARY
    official_dict = word_validator.load_dictionary(dict_name, use_added_words=use_aw)
    
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
            is_bonus = (target_bonus_word and word == target_bonus_word.upper())
            fmt = params.get('board_format', 'Normal')
            bonus_cell = board_raw.get('bonus_cell') if isinstance(board_raw, dict) else None
            
            # Authoritative scoring calculation from scoring.py
            pts = calculate_word_score(
                word=word,
                bonus_word=target_bonus_word,
                board_format=fmt,
                bonus_cell=bonus_cell,
                board=board
            )
            
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
            UPDATE tournament_scores 
            SET score = ?, submitted_words = ?, submitted_at = ?, round_start_time = ?
            WHERE tournament_id = ? AND round_number = ? AND user_id = ?
        ''', (total_score, json.dumps(valid_words), time.time(), round_start_time, tid, round_num, user_id))
        conn.commit()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

    try:
        tournament_manager.update_matchup_winners(tid, round_num)
        tournament_manager.update_tournament_status()
    except Exception as e:
        print(f"[Tournament] Exception evaluating matchup winner: {e}")

    return jsonify({'success': True, 'score': total_score})


@app.route('/api/tournament/save-draft', methods=['POST'])
@login_required
def save_tournament_draft():
    if session.get('is_guest'):
        return jsonify({'error': 'Guest access denied'}), 403
        
    data = request.json
    tid = data.get('tournament_id')
    round_num = data.get('round_number')
    words_data = data.get('words', []) # Now objects with 'word', 'points', 'timestamp'
    round_start_time = data.get('round_start_time', time.time())
    
    user_id = session['user_id']
    
    # FETCH ROUND DATA for validation
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    t = conn.execute('SELECT * FROM tournaments WHERE id = ?', (tid,)).fetchone()
    r = conn.execute('SELECT * FROM tournament_rounds WHERE tournament_id = ? AND round_number = ?',
                    (tid, round_num)).fetchone()
    if not t or not r:
        conn.close()
        return jsonify({'error': 'Tournament or round not found'}), 404
        
    # Check participant and draft validity
    p = conn.execute('SELECT status FROM tournament_participants WHERE tournament_id = ? AND user_id = ?', (tid, user_id)).fetchone()
    if not p or p['status'] != 'active':
        conn.close()
        return jsonify({'error': 'Not an active participant'}), 403
        
    existing = conn.execute('SELECT submitted_at FROM tournament_scores WHERE tournament_id = ? AND round_number = ? AND user_id = ?',
                            (tid, round_num, user_id)).fetchone()
    conn.close()
    
    if not existing:
        return jsonify({'error': 'Turn not started yet'}), 403
        
    if existing['submitted_at'] is not None:
        return jsonify({'error': 'Turn already completed/submitted'}), 403
        
    params = json.loads(t['parameters'])
    board_raw = json.loads(r['board_data'])
    
    if isinstance(board_raw, dict):
        board = board_raw.get('board')
        target_bonus_word = board_raw.get('bonus_word', '').upper()
        bonus_len_target = params.get('bonus_word_length', 0)
    else:
        board = board_raw
        target_bonus_word = None
        bonus_len_target = params.get('bonus_word_length', 0)

    dict_name = params.get('dictionary', 'NWL')
    min_len = params.get('min_word_length', 3)
    use_aw = params.get('use_added_words', False)
    
    official_dict = word_validator.load_dictionary(dict_name, use_added_words=use_aw)
    
    valid_words = []
    total_score = 0
    
    for item in words_data:
        word = item.get('word', '').strip().upper()
        if not word: continue
        if len(word) < min_len: continue
        if word not in official_dict: continue
        if not word_validator.find_word_on_board(board, word): continue
        
        if not any(v['word'] == word for v in valid_words):
            is_bonus = (target_bonus_word and word == target_bonus_word.upper())
            fmt = params.get('board_format', 'Normal')
            bonus_cell = board_raw.get('bonus_cell') if isinstance(board_raw, dict) else None
            
            # Authoritative scoring calculation from scoring.py
            pts = calculate_word_score(
                word=word,
                bonus_word=target_bonus_word,
                board_format=fmt,
                bonus_cell=bonus_cell,
                board=board
            )
            
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
            UPDATE tournament_scores 
            SET score = ?, submitted_words = ?, round_start_time = ?
            WHERE tournament_id = ? AND round_number = ? AND user_id = ?
        ''', (total_score, json.dumps(valid_words), round_start_time, tid, round_num, user_id))
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
    def _safe_int_param(val, default_val):
        if val is None or str(val).lower() in ['random', 'none', 'null', 'nan']:
            return default_val
        try:
            return int(val)
        except Exception:
            return default_val

    # 1. Create a GameRoom for Solo Practice
    # Force a unique ID and mark as private to skip singleton logic
    room_id = f"practice_{session['username']}_{int(time.time())}"
    game_type = 'solo_accumulative' # Isolated mode for solo play
    time_limit = _safe_int_param(parameters.get('time_limit'), 60)
    board_dimensions = parameters.get('board_dimensions', '4x4')
    
    room = room_manager.create_room(
        room_id, game_type, time_limit, board_dimensions, is_private=True,
        is_solo=True, initial_solo_params=dict(parameters)
    )
    room.is_solo = True # Disables history and statistics
    room.initial_solo_params = dict(parameters)
    
    # 2. Configure Parameters (Randomize by default if not strictly specified)
    dict_name = parameters.get('dictionary')
    if not dict_name or dict_name == 'random':
        from spinner_set import SpinnerSet
        dict_name = SpinnerSet._spin_dictionary()
        
    # Extract + AW from dictionary name
    is_24h_room = (time_limit >= 7200)
    use_aw_flag = False
    if dict_name and ('+ AW' in str(dict_name) or '+AW' in str(dict_name)):
        use_aw_flag = not is_24h_room
        dict_name = str(dict_name).replace('+ AW', '').replace('+AW', '').strip()
    elif dict_name == 'AW':
        use_aw_flag = not is_24h_room
        dict_name = 'NWL'
    
    if is_24h_room:
        use_aw_flag = False
        if not dict_name or dict_name not in ['NWL', 'CSW']:
            dict_name = 'NWL'

    board_format = 'Valued Letters' if is_24h_room else parameters.get('board_format', 'Normal')
    from spinner_set import SpinnerSet
    if not is_24h_room and board_format == 'random':
        board_format = SpinnerSet._spin_board_format(is_24h=False, dimensions=board_dimensions)

    # First-round difficulty randomization
    target_difficulty = parameters.get('difficulty', 'random')
    if target_difficulty == 'random':
        from spinner_set import SpinnerSet
        target_difficulty = SpinnerSet._spin_difficulty()

    # Point range / word count: allow user to specify, else spin a default
    custom_word_count_range = parameters.get('word_count_range', 'random')
    min_word_len = _safe_int_param(parameters.get('min_word_length'), 3)
    if is_24h_room:
        wc_range = '300-400'
    elif custom_word_count_range == 'random':
        from spinner_set import SpinnerSet
        wc_range = SpinnerSet._spin_word_count(dict_name, min_word_len, target_difficulty, board_dimensions, use_added_words=use_aw_flag)
    else:
        # Use custom range provided by user
        wc_range = custom_word_count_range

    # Bonus word length: spin if random (equal weights for 6-10)
    bonus_len_choice = parameters.get('bonus_word_length', 'random')
    if bonus_len_choice == 'random' or not bonus_len_choice or str(bonus_len_choice) in ['0', 'None', 'null', 'nan']:
        import random
        bonus_word_len = random.choices([6, 7, 8, 9, 10], weights=[20, 20, 20, 20, 20])[0]
    else:
        bonus_word_len = _safe_int_param(bonus_len_choice, 8)

    # Check if the user wants randomization per round
    room.randomize_spinner = (
        parameters.get('dictionary') == 'random' or
        parameters.get('difficulty', 'random') == 'random' or
        parameters.get('word_count_range', 'random') == 'random' or
        parameters.get('board_format', 'random') == 'random' or
        parameters.get('bonus_word_length', 'random') == 'random'
    )

    initial_bw_len = bonus_word_len
    if getattr(room, 'bonus_word', None):
        initial_bw_len = len(room.bonus_word)
    elif getattr(room, 'next_round_bonus', None):
        initial_bw_len = len(room.next_round_bonus)

    room.spinner_params = {
        'dictionary': dict_name,
        'min_word_length': min_word_len,
        'bonus_word_length': initial_bw_len,
        'board_format': board_format,
        'difficulty': target_difficulty,
        'word_count_range': wc_range,
        'use_added_words': use_aw_flag
    }
    room.use_added_words = use_aw_flag
    
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

    # USER DIRECTIVE: Dispatch board generation to background thread and return room_id INSTANTLY (0ms delay)
    # The user enters the room immediately, and the board loads smoothly inside the room view!
    print(f"[app.py] Dispatching round start to background thread for instant solo room entry: {room_id}")
    import threading
    threading.Thread(target=room_manager.start_round, args=(room_id,), daemon=True).start()
    
    print(f"[app.py] Solo match creation complete. Returning success to client instantly.")
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
        
    # Check if user has already played their turn for this round
    turn = conn.execute('SELECT 1 FROM private_match_turns WHERE match_id = ? AND round_number = ? AND user_id = ?',
                        (match_id, round_num, session['user_id'])).fetchone()
    if turn:
        conn.close()
        return jsonify({'error': 'You have already played your turn for this round.'}), 400
        
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
        if r['dictionary']:
            params['dictionary'] = r['dictionary']
    except:
        pass
    try:
        if r['difficulty']:
            params['difficulty'] = r['difficulty']
    except:
        pass
    try:
        if r['bonus_word']:
            params['bonus_word_length'] = len(r['bonus_word'])
        else:
            params['bonus_word_length'] = 'None'
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
        'all_words': json.loads(r['all_words']) if r['all_words'] else [],
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
        use_aw = params.get('use_added_words', True)
        official_dict = word_validator.load_dictionary(dict_name, use_added_words=use_aw)
        
        for item in words_data:
            word = item.get('word', '').strip().upper()
            path = item.get('path', None)
            if not word or word in [v['word'] for v in valid_words]:
                continue
            
            # Basic validation
            if len(word) < params.get('min_word_length', 3):
                continue
            
            # MANDATORY: Always verify the word is actually on the board using server-side logic
            is_on_board = word_validator.find_word_on_board(board, word)
                
            pts = 0
            is_bonus = False
            details = None
            if word in official_dict and is_on_board:
                res = calculate_word_score(word, bonus_word=bonus_word, board_format=fmt, board=board, path=path, bonus_cell=bonus_cell, is_private=True, return_details=True, strict_path=True)
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

def preload_dictionaries():
    print("[Preload] Starting dictionary preloading...")
    try:
        load_tools_dictionary('ALL')
        load_tools_dictionary('NWL')
        load_tools_dictionary('CSW')
        print("[Preload] Dictionary preloading complete.")
    except Exception as e:
        print(f"[Preload] Error preloading dictionaries: {e}")

# Disable memory-intensive preloading in container environment
# threading.Thread(target=preload_dictionaries, daemon=True).start()

try:
    from board_generator import seed_pregenerated_cache_bg
    seed_pregenerated_cache_bg()
except Exception as e:
    print(f"[AppInit] Error starting pregeneration bootstrapper: {e}")

if __name__ == '__main__':
    # Background room advancer is now handled by RoomManager's internal thread
    print("[Main] Background Room Advancer consolidated into RoomManager.")

    print('Morpheme server running on http://localhost:5001')
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=5001, threads=32)
    except Exception as e:
        print(f"Server startup error: {e}. Attempting fallback...")
        from waitress import serve
        serve(app, host='0.0.0.0', port=5001, threads=4)

