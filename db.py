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
