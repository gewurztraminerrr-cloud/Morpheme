import os
import json
import random
import fcntl
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_PATH = os.path.join(BASE_DIR, 'dictionaries', 'word_stats.json')
CONFIG_PATH = os.path.join(BASE_DIR, 'dictionaries', 'tally_difficulty_config.json')
NWL_PATH = os.path.join(BASE_DIR, 'dictionaries', 'NWL.txt')

# Standard Morpheme / Boggle base points by length
BASE_POINTS = {
    1: 0, 2: 0,
    3: 1, 4: 1,
    5: 2,
    6: 3,
    7: 5,
    8: 9,
    9: 13,
    10: 17,
    11: 21,
    12: 25,
    13: 29,
    14: 33,
    15: 37
}

DEFAULT_CONFIG = {
    "use_tally_divisions": False,
    "hard_thresholds": {
        "5": 2,  # Rare division of 2
        "6": 2,  # 2nd and 3rd divisions (Divs 2, 3 of 3)
        "7": 3,  # Rarest 3 divisions (Divs 3, 4, 5 of 5)
        "8": 3,  # Rarest 3 divisions (Divs 3, 4, 5 of 5)
        "9": 1,  # All 9L+ are considered hard
        "10": 1,
        "11": 1,
        "12": 1,
        "13": 1,
        "14": 1,
        "15": 1
    }
}

def get_base_points(length: int) -> int:
    """Return default base points for a given word length."""
    if length <= 2:
        return 0
    if length in BASE_POINTS:
        return BASE_POINTS[length]
    return 37

def get_num_divisions(length: int) -> int:
    """
    Return number of divisions for a given word length.
    Rule:
    - 3L and 4L: Divisions concept does not apply (returns 1).
    - 5L: (3 - 2) + 1 = 2 divisions
    - 6L: (5 - 3) + 1 = 3 divisions
    - 7L: (9 - 5) + 1 = 5 divisions
    - 8L: (13 - 9) + 1 = 5 divisions
    - 9L+: (17 - 13) + 1 = 5 divisions
    """
    if length <= 4:
        return 1
    next_base = get_base_points(length + 1)
    cur_base = get_base_points(length)
    diff = next_base - cur_base
    return max(1, diff + 1)

def get_max_points(length: int) -> int:
    """Return maximum points a word of given length can receive in the rarest division."""
    if length <= 4:
        return get_base_points(length)
    return get_base_points(length) + get_num_divisions(length) - 1

# In-memory cache for word_stats.json
_STATS_CACHE = {}
_STATS_MTIME = 0
_STATS_MAX_BY_LEN = {}

def get_word_stats() -> dict:
    """Thread-safe read of word_stats.json with mtime caching."""
    global _STATS_CACHE, _STATS_MTIME, _STATS_MAX_BY_LEN
    try:
        if not os.path.exists(STATS_PATH):
            return {}
        mtime = os.path.getmtime(STATS_PATH)
        if mtime != _STATS_MTIME or not _STATS_CACHE:
            with open(STATS_PATH, 'r') as f:
                _STATS_CACHE = json.load(f)
            _STATS_MTIME = mtime
            # Precompute max count by word length
            max_counts = {}
            for w, c in _STATS_CACHE.items():
                if isinstance(c, (int, float)) and c > 0:
                    l = len(w)
                    if c > max_counts.get(l, 0):
                        max_counts[l] = int(c)
            _STATS_MAX_BY_LEN = max_counts
    except Exception as e:
        print(f"[TallyDivisions] Error reading word stats: {e}")
    return _STATS_CACHE

def get_max_count_by_length(length: int) -> int:
    """Get the highest find count among all words of the specified length."""
    get_word_stats()
    return _STATS_MAX_BY_LEN.get(length, 0)

# Configuration storage & retrieval
_CONFIG_CACHE = None
_CONFIG_MTIME = 0

def get_tally_config() -> dict:
    """Get the current configuration for tally difficulty and thresholds."""
    global _CONFIG_CACHE, _CONFIG_MTIME
    try:
        if not os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            _CONFIG_CACHE = json.loads(json.dumps(DEFAULT_CONFIG))
            _CONFIG_MTIME = os.path.getmtime(CONFIG_PATH)
            return _CONFIG_CACHE

        mtime = os.path.getmtime(CONFIG_PATH)
        if mtime != _CONFIG_MTIME or _CONFIG_CACHE is None:
            with open(CONFIG_PATH, 'r') as f:
                _CONFIG_CACHE = json.load(f)
            _CONFIG_MTIME = mtime
            # Ensure all keys exist
            if 'use_tally_divisions' not in _CONFIG_CACHE:
                _CONFIG_CACHE['use_tally_divisions'] = False
            if 'hard_thresholds' not in _CONFIG_CACHE:
                _CONFIG_CACHE['hard_thresholds'] = dict(DEFAULT_CONFIG['hard_thresholds'])
            else:
                for k, v in DEFAULT_CONFIG['hard_thresholds'].items():
                    if k not in _CONFIG_CACHE['hard_thresholds']:
                        _CONFIG_CACHE['hard_thresholds'][k] = v
    except Exception as e:
        print(f"[TallyDivisions] Error reading config: {e}")
        return json.loads(json.dumps(DEFAULT_CONFIG))
    return _CONFIG_CACHE

def set_tally_config(new_config: dict, username: str) -> tuple[bool, str]:
    """Persist new configuration. Enforces that only 'jeffb' can toggle or modify settings."""
    if not username or str(username).strip().lower() != 'jeffb':
        return False, "Unauthorized: Only jeffb can modify Tally Divisions settings."

    global _CONFIG_CACHE, _CONFIG_MTIME
    try:
        cur_cfg = get_tally_config()
        if 'use_tally_divisions' in new_config:
            cur_cfg['use_tally_divisions'] = bool(new_config['use_tally_divisions'])
        if 'hard_thresholds' in new_config and isinstance(new_config['hard_thresholds'], dict):
            for k, v in new_config['hard_thresholds'].items():
                try:
                    cur_cfg['hard_thresholds'][str(k)] = int(v)
                except (ValueError, TypeError):
                    pass

        with open(CONFIG_PATH, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(cur_cfg, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

        _CONFIG_CACHE = cur_cfg
        _CONFIG_MTIME = os.path.getmtime(CONFIG_PATH)
        print(f"[TallyDivisions] Config updated by jeffb: use_tally={cur_cfg['use_tally_divisions']}")
        return True, "Settings saved successfully."
    except Exception as e:
        print(f"[TallyDivisions] Error saving config: {e}")
        return False, str(e)

def is_tally_divisions_enabled() -> bool:
    """Return whether the tally divisions toggle is currently switched ON."""
    return bool(get_tally_config().get('use_tally_divisions', False))

def get_hard_threshold(length: int) -> int:
    """Get the division threshold at or above which a word of given length is considered hard."""
    cfg = get_tally_config()
    thresholds = cfg.get('hard_thresholds', {})
    val = thresholds.get(str(length))
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            pass
    # Defaults:
    if length == 5:
        return 2  # Rare division of 2 (2nd division)
    if length == 6:
        return 2  # 2nd and 3rd divisions of 3
    return 3      # Rarest 3 divisions (3, 4, 5) for 7L+

def get_word_division_info(word: str) -> dict:
    """
    Calculate division and point value for a specific word.
    Returns:
    {
        'word': WORD,
        'length': length,
        'count': find_count,
        'max_count': max_count_for_length,
        'has_divisions': bool,
        'division': division (1 to K),
        'max_divisions': K,
        'base_points': base_pts,
        'points': calculated_pts,
        'max_points': max_pts,
        'division_str': "Division X of K (Y of Z pts)" or standard
    }
    """
    raw_word = str(word or '').strip().upper()
    length = len(raw_word)
    base_pts = get_base_points(length)
    max_pts = get_max_points(length)
    k_divisions = get_num_divisions(length)

    stats = get_word_stats()
    find_count = stats.get(raw_word, 0)
    max_count = get_max_count_by_length(length)

    if length <= 4:
        # User requirement: "Do not apply this concept of divisions to 3 letter words and 4 letter words."
        return {
            'word': raw_word,
            'length': length,
            'count': find_count,
            'max_count': max_count,
            'has_divisions': False,
            'division': 1,
            'max_divisions': 1,
            'base_points': base_pts,
            'points': base_pts,
            'max_points': base_pts,
            'division_str': f"{base_pts} pt{'s' if base_pts != 1 else ''}"
        }

    # If no words of this length have been found yet (max_count == 0):
    # "If no one has played a game, all the words in the tally file are at 0: there is simply no 'division'
    # to categorize the words based on difficulty... all words of a particular letter length are valued the same"
    if max_count <= 0:
        return {
            'word': raw_word,
            'length': length,
            'count': find_count,
            'max_count': 0,
            'has_divisions': False,
            'division': 1,
            'max_divisions': k_divisions,
            'base_points': base_pts,
            'points': base_pts,
            'max_points': max_pts,
            'division_str': f"Division 1 of {k_divisions} ({base_pts} of {max_pts} pts)"
        }

    # Calculate division based on find count range:
    # Division 1 = Most commonly found / Default for unplayed words (base + 0 pts)
    # Division K = Rarest words with low find counts (base + (K-1) pts)
    if find_count <= 0 or find_count >= max_count:
        # Default to Division 1 per user instruction:
        # "Default to 1, not 5. Let TESTING grow over time so that, probably being the
        # one of the most commonly found, it simply stays at the first division (5 pts)"
        division = 1
    else:
        # Distance from top divided by total span, mapped to 1..K
        frac = (max_count - find_count) / float(max_count)
        division = int(frac * k_divisions) + 1
        division = max(1, min(k_divisions, division))

    pts = base_pts + (division - 1)
    div_str = f"Division {division} of {k_divisions} ({pts} of {max_pts} pts)"

    return {
        'word': raw_word,
        'length': length,
        'count': find_count,
        'max_count': max_count,
        'has_divisions': True,
        'division': division,
        'max_divisions': k_divisions,
        'base_points': base_pts,
        'points': pts,
        'max_points': max_pts,
        'division_str': div_str
    }

def is_word_hard_by_tally(word: str) -> bool:
    """
    Determine whether a word is considered 'hard' based on its division in the tally file.
    Rule:
    - 3L and 4L: False (never counted as hard).
    - 9L+: All 9-letter words and greater are considered hard.
    - 5L to 8L: division >= hard_threshold (e.g. >= 2 for 5L, >= 2 for 6L, >= 3 for 7L/8L).
    """
    raw_word = str(word or '').strip().upper()
    length = len(raw_word)
    if length <= 4:
        return False

    if length >= 9:
        # User requirement: "Also, at the 9 letter words and greater, include all as part of the HIGHEST NUMBER OF HARD WORDS FOUND"
        return True

    div_info = get_word_division_info(raw_word)
    thresh = get_hard_threshold(length)
    return div_info['division'] >= thresh

# Cached dictionary words by length for fast sampling
_DICT_WORDS_BY_LEN = {}

def _load_nwl_words_by_len():
    global _DICT_WORDS_BY_LEN
    if _DICT_WORDS_BY_LEN:
        return _DICT_WORDS_BY_LEN
    by_len = {}
    if os.path.exists(NWL_PATH):
        try:
            with open(NWL_PATH, 'r') as f:
                for line in f:
                    w = line.strip().upper()
                    if w:
                        l = len(w)
                        if 5 <= l <= 10:
                            by_len.setdefault(l, []).append(w)
        except Exception as e:
            print(f"[TallyDivisions] Error loading NWL for examples: {e}")
    _DICT_WORDS_BY_LEN = by_len
    return _DICT_WORDS_BY_LEN

def get_division_examples() -> dict:
    """
    Fetch five examples of words from every division within 5, 6, 7, and 8 letter words.
    Returns:
    {
        "5": {
            "max_divisions": 2,
            "max_count": M5,
            "divisions": {
                "1": {"points": 2, "range_str": "...", "words": [{"word": "...", "count": X}, ...]},
                "2": {"points": 3, "range_str": "...", "words": [...]}
            }
        },
        "6": ...,
        "7": ...,
        "8": ...
    }
    """
    stats = get_word_stats()
    words_by_len = _load_nwl_words_by_len()
    result = {}

    for length in [5, 6, 7, 8]:
        k_divs = get_num_divisions(length)
        base_pts = get_base_points(length)
        max_c = get_max_count_by_length(length)

        div_buckets = {d: [] for d in range(1, k_divs + 1)}
        words_seen = set()

        # 1. First, check all non-zero words in stats for this length
        stats_words = [w for w, c in stats.items() if len(w) == length and c > 0]
        random.shuffle(stats_words)
        for w in stats_words:
            d_info = get_word_division_info(w)
            d = d_info['division']
            if len(div_buckets[d]) < 5 and w not in words_seen:
                div_buckets[d].append({
                    'word': w,
                    'count': d_info['count'],
                    'points': d_info['points']
                })
                words_seen.add(w)

        # 2. For any divisions not yet full (especially rare divisions with 0 finds), sample from NWL
        eligible_nwl = [w for w in words_by_len.get(length, []) if w not in words_seen]
        if eligible_nwl:
            sample_size = min(len(eligible_nwl), 3000)
            sample_pool = random.sample(eligible_nwl, sample_size)
            for w in sample_pool:
                d_info = get_word_division_info(w)
                d = d_info['division']
                if len(div_buckets[d]) < 5 and w not in words_seen:
                    div_buckets[d].append({
                        'word': w,
                        'count': d_info['count'],
                        'points': d_info['points']
                    })
                    words_seen.add(w)
                if all(len(div_buckets[d]) >= 5 for d in range(1, k_divs + 1)):
                    break

        divisions_data = {}
        for d in range(1, k_divs + 1):
            pts = base_pts + (d - 1)
            # Find count range description
            if max_c > 0:
                if d == 1:
                    low = int(max_c * (k_divs - 1) / float(k_divs))
                    high = max_c
                    rng_str = f"{low} – {high} finds"
                elif d == k_divs:
                    high = int(max_c / float(k_divs))
                    rng_str = f"0 – {high} finds"
                else:
                    low = int(max_c * (k_divs - d) / float(k_divs))
                    high = int(max_c * (k_divs - d + 1) / float(k_divs))
                    rng_str = f"{low} – {high} finds"
            else:
                rng_str = "0 finds (initial)"

            divisions_data[str(d)] = {
                'division': d,
                'points': pts,
                'range_str': rng_str,
                'words': div_buckets[d]
            }

        result[str(length)] = {
            'length': length,
            'max_divisions': k_divs,
            'base_points': base_pts,
            'max_count': max_c,
            'threshold': get_hard_threshold(length),
            'divisions': divisions_data
        }

    return result

def is_wotd_eligible(word: str) -> bool:
    """
    Check if a candidate word is eligible for Word of the Day.
    Rule:
    - 6-letter words: Only allow the rarest division (Division 3).
    - 7-letter words or greater: Only allow the 3 rarest divisions (Divisions 3, 4, 5).
    - If divisions for the letter length have not been made yet, default to anything (eligible).
    """
    raw_word = str(word or '').strip().upper()
    length = len(raw_word)
    if not (6 <= length <= 10):
        return False

    max_c = get_max_count_by_length(length)
    if max_c <= 0:
        # Divisions have not been made yet: default to anything
        return True

    div_info = get_word_division_info(raw_word)
    div = div_info['division']
    k_divs = div_info['max_divisions']

    if length == 6:
        # "as well as the rarest division for 6-letter words for candidates of the Word of the Day"
        return div == k_divs

    # "only allow the 3 rarest divisions (for 7-letter words or greater)"
    # For K=5 divisions, rarest 3 are 3, 4, 5 (or div >= k_divs - 2)
    return div >= max(1, k_divs - 2)
