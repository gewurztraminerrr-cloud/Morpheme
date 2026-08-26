"""
Centralized Database Connection Manager for Morpheme
Ensures:
- WAL (Write-Ahead Logging) mode enabled everywhere
- 60-second busy timeout on every connection
- synchronous = NORMAL (crash-safe, eliminates fsync write stalls)
- In-memory temp store and 64MB cache
- Context managers with guaranteed connection closing and automatic rollback on error
- Automatic retry on database locked / busy operational errors
"""

import os
import sqlite3
import time
import random
import datetime
import math
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "morpheme.db")
_raw_sqlite3_connect = sqlite3.connect

def get_db_connection(db_path=None, timeout=60.0, row_factory=None):
    """
    Creates and configures an optimized SQLite connection.
    Guarantees WAL mode, 60s busy_timeout, and NORMAL synchronous.
    """
    if db_path is None:
        db_path = DB_PATH
        
    conn = _raw_sqlite3_connect(db_path, timeout=timeout, check_same_thread=False)
    if row_factory is not None:
        conn.row_factory = row_factory
        
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 60000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA cache_size = -64000;")
    except Exception:
        pass
        
    return conn

@contextmanager
def get_db(db_path=None, timeout=60.0, row_factory=None, auto_commit=True):
    """
    Context manager for database access.
    Automatically commits on normal exit, rolls back on exception,
    and ALWAYS closes the connection in a finally block to prevent descriptor leaks.
    """
    conn = get_db_connection(db_path=db_path, timeout=timeout, row_factory=row_factory)
    try:
        yield conn
        if auto_commit:
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

def execute_with_retry(query_func, max_retries=5, initial_delay=0.05):
    """
    Executes a function that interacts with SQLite, retrying with exponential
    backoff and jitter if database is locked or busy.
    """
    delay = initial_delay
    last_exception = None
    for attempt in range(max_retries):
        try:
            return query_func()
        except sqlite3.OperationalError as e:
            err_msg = str(e).lower()
            if "locked" in err_msg or "busy" in err_msg:
                last_exception = e
                jitter = random.uniform(0.5, 1.5)
                time.sleep(delay * jitter)
                delay = min(delay * 2.0, 2.0)
            else:
                raise
        except Exception as e:
            raise
            
    if last_exception:
        raise last_exception


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
    import datetime
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
    import datetime, math
    if not user_id_or_name:
        return False, 0, "", None, 0, None
    try:
        with get_db(timeout=10.0, row_factory=sqlite3.Row, auto_commit=False) as conn:
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
    except Exception as e:
        print(f"[check_user_timeout] Error: {e}")
    return False, 0, "", None, 0, None
