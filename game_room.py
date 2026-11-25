"""
Game Room Management for Multiplayer Boggle
Handles room state, players, timers, and game logic

ARCHITECTURAL MANDATE (PREFACE) - READ BEFORE EDITING BOARDS / SPINNER SET:
  Any fix to game-room boards, Spinner Set configuration, word acceptance, the Remaining tab or the
  info above Words/All Words MUST be applied identically to Game Rooms, Solo, With Friends
  (private_match_logic.py) and Tournaments (tournament_logic.py). NOT to Multiplayer Subanagrams.
  Board generation must be one unified method everywhere. Parameters declared in the Spinner Set at
  0:45 of intermission are final: nothing may change them until the next intermission.
"""

import time
import random
import datetime
import threading
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Set
import os
import fcntl

try:
    from zoneinfo import ZoneInfo
    DAILY_TZ = ZoneInfo("America/Chicago")
except Exception:
    import datetime
    DAILY_TZ = datetime.timezone(datetime.timedelta(hours=-6), "CST")

# GLOBAL WORD TALLY CONTROLLER
# Absolute path ensures consistency across Gunicorn/Flask environments
STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dictionaries', 'word_stats.json')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'morpheme.db')
RATING_AUDIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rating_audit.log')
DEBUG_FLOW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_flow.log')
WORD_DEBUG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'word_debug.log')
WORD_TALLY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'word_tally.log')
TRACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dictionaries', 'stats_trace.log')
from db import get_db, get_db_connection, check_user_timeout
# STATS_LOCK (Memory-based) is insufficient for multi-worker environments. 
# We use file-based locking (fcntl) inside the I/O methods instead.
from spinner_set import SpinnerSet
from board_generator import BoardGenerator
from scoring import calculate_word_score, get_valued_word_score
from rating_logic import calculate_proportional_rating_change, is_player_guest
import word_validator
from word_validator import use_added_words_ctx

_room_manager_instance = None

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

@dataclass
class Player:
    user_id: int
    username: str
    rating: int
    submitted_words: List[Dict] = field(default_factory=list)
    invalid_words: List[str] = field(default_factory=list)
    score: int = 0
    previous_round_score: int = 0
    rating_change: int = 0
    games_played: int = 0
    previous_submitted_words: List[Dict] = field(default_factory=list)
    found_bonus_word: bool = False
    last_active: float = field(default_factory=time.time)
    user_last_interacted: float = field(default_factory=time.time)
    input_method: str = "mouse"  # 'keyboard', 'mouse', or 'touch'
    country_flag: str = '🏳️'
    joined_mid_round: bool = False
    has_exceptional_round: bool = False
    trophy_rounds_left: int = 0
    performance_efficiency: float = 0.0
    is_guest: bool = False
    is_ai: bool = False
    ai_rating: int = 1200
    has_abandoned: bool = False
    cell_density: List = field(default_factory=list)
    game_room_chat: bool = True
    is_unrated: bool = False
    provisional_games: int = 0
    provisional_ratings_sum: float = 0.0
    provisional_rating: float = 1200.0
    config_games_played: int = 0

    @property
    def is_registered(self) -> bool:
        if getattr(self, "is_guest", False) or getattr(self, "is_ai", False):
            return False
        try:
            return int(self.user_id) > 0
        except (ValueError, TypeError):
            return False

def is_board_count_valid(word_count, target_range):
    """Ensure actual word_count falls strictly within the spun target_range (e.g. 100-199 for 100-200)."""
    if not target_range:
        return True
    try:
        if isinstance(target_range, (list, tuple)) and len(target_range) >= 2:
            r_min = int(target_range[0])
            r_max = int(target_range[1]) - 1
            return (r_min <= word_count <= r_max)
            
        s_range = str(target_range).replace(',', '-').strip()
        if '-' in s_range:
            parts = s_range.split('-')
            if len(parts) == 2:
                r_min = int(parts[0])
                r_max = int(parts[1]) - 1
                return (r_min <= word_count <= r_max)
        elif '+' in s_range:
            r_min = int(s_range.replace('+', '').strip())
            return (word_count >= r_min)
    except Exception as e:
        print(f"[is_board_count_valid] Error parsing target_range '{target_range}': {e}")
        return False
    return False

def proportionally_sample_words(all_words_collection, max_limit, seed_val=None):
    """
    Proportionally sample words across all length categories (3L, 4L, 5L, 6L, 7L+)
    when truncating to fit a target word_count_range (e.g. max 399).
    Prevents short words (3LW/4LW) from being completely discarded or truncated to 0/1.
    """
    all_words_list = list(all_words_collection)
    total_count = len(all_words_list)
    if total_count <= max_limit:
        return set(all_words_list)
    
    words_by_len = {}
    for w in all_words_list:
        l = len(w)
        words_by_len.setdefault(l, []).append(w)
        
    # Using global random module
    rng = random.Random(seed_val) if seed_val else random.Random()
    
    allocated = {}
    total_alloc = 0
    min_category_len = min(words_by_len.keys()) if words_by_len else 3
    for l, group in sorted(words_by_len.items()):
        floor_quota = min(10, len(group)) if l == min_category_len else 1
        count = max(floor_quota, int(round(len(group) * max_limit / total_count)))
        count = min(count, len(group))
        allocated[l] = count
        total_alloc += count

    diff = max_limit - total_alloc
    sorted_lengths = sorted(words_by_len.keys(), key=lambda l: len(words_by_len[l]), reverse=True)
    idx = 0
    while diff != 0 and sorted_lengths:
        l = sorted_lengths[idx % len(sorted_lengths)]
        min_alloc = min(10, len(words_by_len[l])) if l == min_category_len else 1
        if diff > 0 and allocated[l] < len(words_by_len[l]):
            allocated[l] += 1
            diff -= 1
        elif diff < 0 and allocated[l] > min_alloc:
            allocated[l] -= 1
            diff += 1
        idx += 1

    sampled = []
    for l, group in words_by_len.items():
        rng.shuffle(group)
        sampled.extend(group[:allocated[l]])
        
    return set(sampled)

def verify_board_words_consistency(board, words, paths=None):
    """
    Ultra-fast consistency check verifying that 'words' actually belong to 'board'.
    Returns True if consistent, False if board and word list are desynchronized.
    """
    if not board or not words:
        return True  # Nothing to verify or empty

    # Subanagrams / 1D rack verification (< 0.1ms)
    is_sub = (len(board) == 1 and isinstance(board[0], list)) or (isinstance(board, list) and len(board) > 0 and isinstance(board[0], str))
    if is_sub:
        from collections import Counter
        raw_seq = "".join(board[0] if isinstance(board[0], list) else board).upper()
        expanded_seq = raw_seq.replace('Q', 'QU')
        seq_cnt = Counter(expanded_seq)
        sample = list(words)[:10]
        for w in sample:
            w_cnt = Counter(w.upper())
            if not all(seq_cnt[c] >= cnt for c, cnt in w_cnt.items()):
                return False
        return True

    is_3d = (len(board) == 6 and isinstance(board[0], list) and len(board[0]) > 0 and isinstance(board[0][0], list))
    
    # 1. Path-based instant verification (< 0.1ms)
    # If paths dictionary is provided, verify coordinate paths match board characters
    if paths and isinstance(paths, dict):
        sample_words = [w for w in words if w in paths and paths[w]][:15]
        if sample_words:
            for w in sample_words:
                p = paths[w]
                if not p:
                    return False
                is_word_match = True
                p_idx = 0
                for w_idx, char in enumerate(w):
                    if p_idx >= len(p):
                        is_word_match = False
                        break
                    pos = p[p_idx]
                    if is_3d:
                        if len(pos) != 3:
                            is_word_match = False
                            break
                        f, r, c = pos[0], pos[1], pos[2]
                        if not (0 <= f < 6 and 0 <= r < len(board[f]) and 0 <= c < len(board[f][0])):
                            is_word_match = False
                            break
                        cell_val = str(board[f][r][c]).upper()
                    else:
                        r, c = pos[-2], pos[-1]
                        if not (0 <= r < len(board) and 0 <= c < len(board[0])):
                            is_word_match = False
                            break
                        cell_val = str(board[r][c]).upper()
                    
                    if '/' in cell_val:
                        if char not in cell_val.split('/'):
                            is_word_match = False
                            break
                        p_idx += 1
                    elif cell_val == 'Q' and char == 'Q':
                        if w_idx + 1 < len(w) and w[w_idx + 1] == 'U':
                            pass  # wait for U char to advance p_idx
                        else:
                            p_idx += 1
                    elif cell_val == 'Q' and char == 'U' and w_idx > 0 and w[w_idx - 1] == 'Q':
                        p_idx += 1  # QU cell representation
                    elif cell_val == char:
                        p_idx += 1
                    else:
                        is_word_match = False
                        break
                if not is_word_match or p_idx != len(p):
                    return False
            return True

    # 2. Solver/validator verification on sample words
    from word_validator import word_validator
    sample_to_check = sorted(list(words), key=len, reverse=True)[:5]
    for w in sample_to_check:
        if not word_validator.find_word_on_board(board, w):
            return False

    return True


def sync_board_checkerboard_and_resolve(bg, board, fmt_val, min_len, dictionary, use_added_words, bonus_cell):
    """Ensure board strictly complies with checkerboard (or non-checkerboard) and re-solves words/paths if mutated."""
    if not board:
        return False, None, None
    repaired = False
    if 'checkerboard' in str(fmt_val).lower():
        if not bg._is_checkerboard_layout(board):
            repaired = bool(bg._verify_checkerboard_safeguard(board))
    else:
        if bg._is_checkerboard_layout(board):
            repaired = bool(bg._break_accidental_checkerboard(board))
    if repaired:
        resolved = bg._solve_board(
            board, dictionary, (0, 99999), min_len, max_depth=15, store_paths=True, timeout=10.0, bonus_cell=bonus_cell, use_added_words=use_added_words
        )
        words = sorted([w for w in resolved if len(w) >= min_len])
        paths = {w: p for w, p in resolved.items() if len(w) >= min_len}
        return True, words, paths
    return False, None, None

@dataclass
class GameRoom:
    room_id: str
    game_type: str  # 'accumulative', 'fcfs', 'split'
    time_limit: int  # seconds per round
    board_dimensions: str  # '4x4', '4x6', etc.
    
    # Rating limits
    min_rating: int = 0
    max_rating: int = 9999
    is_solo: bool = False # Solo practice mode: no history, auto-looping
    is_private: bool = False # Private match: hidden from lobby
    current_min_length: int = 3
    
    # Spectators
    spectators: List[Player] = field(default_factory=list)
    max_players: int = 8
    is_closing: bool = False
    round_starting_player_ids: Set[str] = field(default_factory=set)
    
    
    # Game state
    creation_time: float = field(default_factory=time.time)
    state: str = 'waiting'  # 'waiting', 'active', 'intermission', 'finished'
    current_round: int = 0
    starting_round: bool = False  # Prevents concurrent round starts
    midnight_reset_occurred: bool = False # Track if midnight reset has occurred for 24h rooms
    last_saved_round: int = -1    # tracks which round was last saved to DB
    stats_recorded_round: int = -1 # tracks if stats were updated for this round

    # Timer
    round_start_time: float = 0
    intermission_start_time: float = 0
    custom_end_time: float = 0 # For fixed-end-time rooms (e.g. daily at midnight)
    
    # Current board data
    board: List[List[str]] = field(default_factory=list)
    all_words: List[str] = field(default_factory=list)  # Fast initial word list
    previous_all_words: List[str] = field(default_factory=list) # Previous round words
    previous_board: List[List[str]] = field(default_factory=list) # Previous round board
    board_fingerprint_history: List[str] = field(default_factory=list) # Rolling dedup history (last 10 boards)
    previous_day_history: Dict = field(default_factory=dict) # Snapshot of yesterday's game (Found/Missed)
    complete_words: List[str] = field(default_factory=list)  # Complete word list from background solving
    solved_words_with_scores: Dict[str, int] = field(default_factory=dict)  # Pre-computed word scores
    bonus_word: str = ''
    bonus_cell: tuple = None # (r, c) for Bonus Letter format
    solving_complete: bool = False  # Track if background solving is done
    
    # FCFS Mode specific
    fcfs_found_words: List[Dict] = field(default_factory=list)
    _fcfs_found_words_set: set = field(default_factory=set)
    
    # Spinner parameters
    current_min_length: int = 3
    current_board_format: str = 'Normal'
    current_word_count_range: str = '100-200'
    current_dictionary: str = 'NWL'
    current_difficulty: str = 'Varying...'
    current_bonus_word_length: int = 0
    current_uniqueness: float = 0.0
    use_added_words: bool = False
    spinner_params: Dict = field(default_factory=dict)
    
    # Density Format Tracking
    global_round_found_words: set = field(default_factory=set) # Set of ALL words found by ANY player
    cell_density: List = field(default_factory=list) # Current density grid (remaining words)
    initial_cell_density: List = field(default_factory=list) # Snapshot of max density per cell
    max_cell_density: int = 0 # NEW: Global max density for normalization
    bonus_word_history: List[str] = field(default_factory=list)
    next_spinner_params: Dict = field(default_factory=dict) # NEW: Isolate pre-gen params from active params
    total_words_count: int = 0
    total_points_count: int = 0 
    # Next round pre-generation (for Accumulative timing)
    spinner_params_generated: bool = False  # Track if spinner set generated for next round
    spinner_params_revealed: bool = False   # NEW: Track if params are visible to players
    board_search_started: bool = False      # Track if board search started
    next_round_board: List[List[str]] = field(default_factory=list)  # Store pre-generated board
    next_round_words: List[str] = field(default_factory=list)  # Store pre-generated word list
    next_round_bonus: str = ''  # Store bonus word for next round
    next_round_bonus_cell: tuple = None # Store bonus cell for next round
    next_round_paths: Dict[str, List[tuple]] = field(default_factory=dict) # NEW: Store pre-generated paths
    all_words_paths: Dict[str, List[tuple]] = field(default_factory=dict) # NEW: Store active round paths
    csw_only_words: List[str] = field(default_factory=list) # Current round CSW words
    added_words: List[str] = field(default_factory=list) # Current round added words
    next_round_csw_only_words: List[str] = field(default_factory=list) # Pre-gen CSW words
    next_round_added_words: List[str] = field(default_factory=list) # Pre-gen added words
    previous_csw_only_words: List[str] = field(default_factory=list) # History
    previous_added_words: List[str] = field(default_factory=list) # History
    previous_bonus_word: str = '' # History
    previous_dictionary: str = 'NWL' # History
    previous_use_added_words: bool = False # History
    evicted_users: Dict = field(default_factory=dict) # user_id -> reason
    
    # Players
    players: List[Player] = field(default_factory=list)
    past_players: Dict[str, Player] = field(default_factory=dict) # Archive of players for persistence
    round_quitters: List[Player] = field(default_factory=list) # Players who left mid-round after playing
    abandonment_bounty: int = 0 # Points collected from quitters for distribution at round end
    
    # Chat
    chat_messages: List[Dict] = field(default_factory=list)
    
    # History of winners
    winners_history: List[Dict] = field(default_factory=list) # [{'round': N, 'winners': [names], 'score': S}]
    previous_total_points: int = 0
    creation_time: float = field(default_factory=time.time)
    previous_total_words: int = 0
    previous_total_counts_by_len: Dict = field(default_factory=dict)
    _did_6x8_fallback_rescue: bool = False
    last_round_placements: Dict[str, int] = field(default_factory=dict)
    @property
    def round_end_time(self):
        """Absolute timestamp when the current round expires"""
        if self.custom_end_time > 0:
            return self.custom_end_time
        if self.round_start_time <= 0: return 0
        return self.round_start_time + self.time_limit

    @property
    def intermission_end_time(self):
        """Absolute timestamp when the intermission expires"""
        if self.intermission_start_time <= 0: return 0
        limit = 5 if self.time_limit >= 7200 else 60
        return self.intermission_start_time + limit

    def __post_init__(self):
        if isinstance(self.board_dimensions, (tuple, list)):
            if len(self.board_dimensions) == 3:
                self.board_dimensions = f"{self.board_dimensions[0]}x{self.board_dimensions[1]}x{self.board_dimensions[2]}"
            else:
                self.board_dimensions = f"{self.board_dimensions[0]}x{self.board_dimensions[1]}"
        elif self.board_dimensions:
            self.board_dimensions = str(self.board_dimensions)

        # Force integer types for comparisons
        self.time_limit = int(self.time_limit)
        if self.min_rating is not None: self.min_rating = int(self.min_rating)
        if self.max_rating is not None: self.max_rating = int(self.max_rating)
        
        is_24h = (self.time_limit >= 7200) or ('24h' in str(self.room_id).lower()) or ('86400' in str(self.room_id)) or (str(self.game_type).lower() == '24h')
        if is_24h:
            grid_floor = 3
            if '4x6' in str(self.board_dimensions): grid_floor = 4
            elif '5x7' in str(self.board_dimensions): grid_floor = 5
            elif '6x8' in str(self.board_dimensions) or '3x3x3' in str(self.board_dimensions): grid_floor = 6
            self.current_min_length = grid_floor
            self.current_board_format = 'Valued Letters'
            self.current_word_count_range = '300-400'
            self.current_difficulty = 'Medium'
            self.spinner_params = {
                'board_format': 'Valued Letters',
                'word_count_range': '300-400',
                'dictionary': 'NWL',
                'difficulty': 'Medium',
                'min_word_length': grid_floor,
                'bonus_word_length': 0,
                'use_added_words': False,
            }
            self.state = 'active'
            self.current_round = max(1, self.current_round)
        
        # Configuration-specific max players
        if self.game_type in ['accumulative', 'solo_accumulative']:
            self.max_players = 9999 # Effectively unlimited
        elif self.game_type == 'fcfs':
            self.max_players = 16
        else:
            self.max_players = 8

        # INITIALIZE LOCKS
        self._state_lock = threading.RLock() # Reentrant to prevent deadlocks during transition
            
    def add_chat_message(self, username, message, is_system=False, image=None, color=None, is_winner=False, rating=None, placement=None):
        """Add chat message to room"""
        if rating is None and not is_system and username and username.upper() != 'SYSTEM':
            # Try to look up the user's rating from active players/spectators
            for p in self.players:
                if p.username == username:
                    rating = p.rating
                    break
            if rating is None:
                for s in self.spectators:
                    if s.username == username:
                        rating = s.rating
                        break

        if placement is None and not is_system and username and username.upper() != 'SYSTEM':
            uname_lower = username.lower()
            if hasattr(self, 'last_round_placements') and uname_lower in self.last_round_placements:
                placement = self.last_round_placements[uname_lower]
            elif getattr(self, 'winners_history', None) and len(self.winners_history) > 0:
                latest = self.winners_history[0]
                all_p = latest.get('all_players', [])
                for idx, p in enumerate(all_p):
                    p_name = p.get('username') if isinstance(p, dict) else getattr(p, 'username', '')
                    if p_name and p_name.lower() == uname_lower:
                        p_score = p.get('score', 0) if isinstance(p, dict) else getattr(p, 'score', 0)
                        if p_score > 0:
                            placement = idx + 1
                        break

        self.chat_messages.append({
            'username': username,
            'message': message,
            'image': image,
            'is_system': is_system,
            'is_winner': is_winner,
            'color': color,
            'time': time.time(),
            'rating': rating,
            'placement': placement
        })
        # Keep only last 30 messages
        if len(self.chat_messages) > 30:
            self.chat_messages.pop(0)

    def _trigger_wake_if_needed(self, is_ai, username):
        if not is_ai and self.time_limit < 7200:
            if self.state == 'waiting':
                print(f"[GameRoom] Human player {username} joined paused room {self.room_id}. Triggering instant wake-up.")
                self.check_and_update_state()
    
    def add_player(self, user_id, username, rating, games_played=0, country_flag='🏳️', manual_accessed=False, is_guest=False, is_ai=False, ai_rating=1200, game_room_chat=True, is_unrated=False, provisional_games=0, provisional_ratings_sum=0.0, provisional_rating=1200.0, config_games_played=0):
        """Add player to room"""
        if not is_ai and user_id:
            try:
                if 'lobby_manager' in globals():
                    lobby_manager.remove_user(user_id)
            except Exception:
                pass
        is_daily = self.time_limit >= 7200
        uid_str = str(user_id)
        uname_lower = str(username).lower()
        
        # Guard: Check user timeout
        if not is_ai and (user_id or username):
            try:
                to_res = check_user_timeout(user_id)
                if not (to_res and to_res[0]) and username:
                    to_res = check_user_timeout(username)
                if to_res and to_res[0]:
                    rem_str = to_res[2] if len(to_res) > 2 else ''
                    print(f"[GameRoom] BLOCKED add_player: User {username} (ID: {user_id}) is timed out for {rem_str}")
                    return False
            except Exception as ex:
                print(f"[GameRoom] Error checking timeout in add_player: {ex}")
        
        with self._state_lock:
            # Always remove from spectators if adding as player
            self.spectators = [s for s in self.spectators if str(s.user_id) != uid_str]
            
            # Clear eviction flag if they are re-joining
            if uid_str in self.evicted_users:
                del self.evicted_users[uid_str]
                print(f"[GameRoom] Cleared eviction flag for {username} on join.")
                
            
            # Check if player already exists (PERSISTENCE / REUSE)
            existing_player = self.get_player(user_id) or self.get_player(username)
            if existing_player:
                print(f"[GameRoom] Persistence: Reusing existing player {username} in room {self.room_id}")
                # SNAPSHOT last_active BEFORE overwriting — needed for mid-round check below
                prior_last_active = getattr(existing_player, 'last_active', 0)
                existing_player.last_active = time.time()
                existing_player.country_flag = country_flag # Update flag
                # CRITICAL: Always sync rating from DB even for persistent daily players
                if rating is not None and not is_guest:
                    existing_player.rating = rating
                existing_player.is_guest = is_guest
                if is_unrated is not None:
                    existing_player.is_unrated = is_unrated
                    existing_player.provisional_games = provisional_games
                    existing_player.provisional_ratings_sum = provisional_ratings_sum
                    existing_player.provisional_rating = provisional_rating
                    existing_player.config_games_played = config_games_played
                if game_room_chat is not None:
                    existing_player.game_room_chat = game_room_chat
                # MID-ROUND DETECTION: Check if player was in the room when round started
                if not is_daily and (self.state == 'active' or getattr(self, 'starting_round', False)):
                    round_starters = getattr(self, 'round_starting_player_ids', None)
                    if round_starters:
                        if uid_str not in round_starters or manual_accessed:
                            existing_player.joined_mid_round = True
                            print(f"[GameRoom] Mid-round flag set for {username} (existing player not in round_starting_player_ids or manual).")
                    else:
                        round_started_at = getattr(self, 'round_start_time', 0)
                        was_present_at_start = (
                            round_started_at > 0 and
                            prior_last_active >= round_started_at - 5
                        )
                        if not was_present_at_start or manual_accessed:
                            existing_player.joined_mid_round = True
                            print(f"[GameRoom] Mid-round flag set for {username} (existing player path fallback). round_start={round_started_at:.0f}, prior_last_active={prior_last_active:.0f}")

                # If they were in round_quitters, they left the active round and rejoined mid-round
                if any(str(q.user_id) == uid_str or str(q.username).lower() == uname_lower for q in self.round_quitters):
                    existing_player.joined_mid_round = True

                # Ensure they are removed from round_quitters if they were in there (REJOIN TRANSITION)
                self.round_quitters = [q for q in self.round_quitters if str(q.user_id) != uid_str and str(q.username).lower() != uname_lower]
                if not getattr(existing_player, 'cell_density', None):
                    self._initialize_player_density_grid(existing_player)
                
                # Deduplicate self.players list to ensure only one reference exists
                self.players = [p for p in self.players if str(p.user_id) != str(existing_player.user_id) and str(p.username).lower() != str(existing_player.username).lower()]
                self.players.append(existing_player)
                self.players.sort(key=lambda p: p.rating, reverse=True)
                
                if is_daily:
                    self.save_active_players()
                self._trigger_wake_if_needed(is_ai, username)
                return True
            
            # Check if player exists in round_quitters (RESTORE mid-round state)
            quitter = next((q for q in self.round_quitters if str(q.user_id) == uid_str or str(q.username).lower() == uname_lower), None)
            if quitter:
                print(f"[GameRoom] Restoring quitter {username} ({user_id}) to active players with {len(quitter.submitted_words)} words, score={quitter.score}.")
                quitter.last_active = time.time()
                quitter.country_flag = country_flag
                quitter.is_guest = is_guest
                quitter.joined_mid_round = True
                if is_unrated is not None:
                    quitter.is_unrated = is_unrated
                    quitter.provisional_games = provisional_games
                    quitter.provisional_ratings_sum = provisional_ratings_sum
                    quitter.provisional_rating = provisional_rating
                    quitter.config_games_played = config_games_played
                if game_room_chat is not None:
                    quitter.game_room_chat = game_room_chat
                if not getattr(quitter, 'cell_density', None):
                    self._initialize_player_density_grid(quitter)
                self.players = [p for p in self.players if str(p.user_id) != uid_str and str(p.username).lower() != uname_lower]
                self.players.append(quitter)
                self.players.sort(key=lambda p: p.rating, reverse=True)
                # CRITICAL: Remove from round_quitters so they aren't double-counted at round end
                self.round_quitters = [q for q in self.round_quitters if str(q.user_id) != uid_str and str(q.username).lower() != uname_lower]
                # Reverse the abandonment bounty that was charged when they left
                if self.abandonment_bounty >= 8:
                    self.abandonment_bounty -= 8
                    print(f"[GameRoom] Reversed abandonment bounty for returning player {username}. Pool now: {self.abandonment_bounty}")
                if is_daily:
                    self.save_active_players()
                self._trigger_wake_if_needed(is_ai, username)
                return True

            # Check if player exists in past_players
            existing_player = next((p for p in self.past_players.values() if str(p.user_id) == uid_str or str(p.username).lower() == uname_lower), None)
            
            if existing_player:
                last_p_round = getattr(existing_player, '_last_round_seen', -1)
                # SNAPSHOT last_active BEFORE overwriting — needed for mid-round check below
                prior_last_active = getattr(existing_player, 'last_active', 0)
                if last_p_round != self.current_round:
                    # NEW ROUND: Clear all round-specific activity
                    existing_player.found_bonus_word = False
                    existing_player.has_abandoned = False
                    existing_player.submitted_words = []
                    existing_player.invalid_words = []
                    existing_player.score = 0
                    existing_player.previous_round_score = 0
                    existing_player.rating_change = 0
                    existing_player.cell_density = []
                    # MID-ROUND DETECTION for past_player joining a new round that is already active
                    existing_player.joined_mid_round = (self.state == 'active')
                
                existing_player._last_round_seen = self.current_round
                existing_player.last_active = time.time()  # Update AFTER snapshot above
                existing_player.country_flag = country_flag
                existing_player.games_played = games_played
                # CRITICAL: Always sync rating from DB for rejoiners/refreshers
                if rating is not None and not is_guest:
                    existing_player.rating = rating
                existing_player.is_guest = is_guest
                if is_unrated is not None:
                    existing_player.is_unrated = is_unrated
                    existing_player.provisional_games = provisional_games
                    existing_player.provisional_ratings_sum = provisional_ratings_sum
                    existing_player.provisional_rating = provisional_rating
                    existing_player.config_games_played = config_games_played
                if game_room_chat is not None:
                    existing_player.game_room_chat = game_room_chat
                
                # MID-ROUND DETECTION: past_player joining an active round is joining mid-round
                if not is_daily and (self.state == 'active' or getattr(self, 'starting_round', False)):
                    round_starters = getattr(self, 'round_starting_player_ids', None)
                    if round_starters is not None and uid_str in round_starters and not manual_accessed:
                        pass
                    else:
                        existing_player.joined_mid_round = True
                        print(f"[GameRoom] Mid-round flag set for {username} (past_player path).")
                
                if not getattr(existing_player, 'cell_density', None):
                    self._initialize_player_density_grid(existing_player)
                self.players = [p for p in self.players if str(p.user_id) != uid_str and str(p.username).lower() != uname_lower]
                self.players.append(existing_player)
                self.players.sort(key=lambda p: p.rating, reverse=True)
                if is_daily:
                    self.save_active_players()
                self._trigger_wake_if_needed(is_ai, username)
                return True

            # Ensure player is not already in the room (prevent duplicates)
            self.players = [p for p in self.players if str(p.user_id) != uid_str and str(p.username).lower() != uname_lower]
            
            # Check max players specific to room
            if len(self.players) >= self.max_players:
                return False # Room full
                
            player = Player(
                user_id, 
                username, 
                rating, 
                games_played=games_played, 
                country_flag=country_flag, 
                is_guest=is_guest,
                is_ai=is_ai,
                ai_rating=ai_rating,
                game_room_chat=game_room_chat,
                is_unrated=is_unrated,
                provisional_games=provisional_games,
                provisional_ratings_sum=provisional_ratings_sum,
                provisional_rating=provisional_rating,
                config_games_played=config_games_played
            )
            if manual_accessed:
                player.joined_mid_round = True
            elif (self.state == 'active' or getattr(self, 'starting_round', False)) and not is_daily:
                player.joined_mid_round = True
                
            self._initialize_player_density_grid(player)
            self.players.append(player)
            self.past_players[uid_str] = player
            self.players.sort(key=lambda p: p.rating, reverse=True)
            
            # System Notice
            self.add_chat_message("System", f"{username} has entered the room.", is_system=True)
            
            if is_daily:
                self.save_active_players()
            self._trigger_wake_if_needed(is_ai, username)
            return True # Success

    def add_spectator(self, user_id, username, rating):
        """Add spectator to room"""
        if user_id or username:
            try:
                to_res = check_user_timeout(user_id)
                if not (to_res and to_res[0]) and username:
                    to_res = check_user_timeout(username)
                if to_res and to_res[0]:
                    rem_str = to_res[2] if len(to_res) > 2 else ''
                    print(f"[GameRoom] BLOCKED add_spectator: User {username} (ID: {user_id}) is timed out for {rem_str}")
                    return False
            except Exception as ex:
                print(f"[GameRoom] Error checking timeout in add_spectator: {ex}")
        
        # Always remove from active players if adding as spectator
        self.players = [p for p in self.players if str(p.user_id) != str(user_id)]
        
        # Clear eviction flag if they are re-joining
        if str(user_id) in self.evicted_users:
            del self.evicted_users[str(user_id)]
            print(f"[GameRoom] Cleared eviction flag for spectator {username} on join.")
            
        # Disable spectating for 24h rooms or closing rooms
        if self.time_limit >= 7200 or self.is_closing:
             return False
        
        # USER REQUEST: Prevent joining as spectator if no players exist
        # We check human count specifically to avoid 'spectator-only' rooms
        humans = [p for p in self.players if not p.is_ai]
        if not humans:
             return False

        # Ensure not already a spectator
        for s in self.spectators:
            if str(s.user_id) == str(user_id):
                return
        
        
        spec = Player(user_id, username, rating, games_played=0) # Spectators don't really use this, but Player needs it
        self.spectators.append(spec)
        
        # System Notice
        self.add_chat_message("System", f"{username} has entered the room.", is_system=True)
        return True
    
    def _get_wc_tuple(self, wc_range):
        """Helper to parse wc_range into (min, max) tuple"""
        if isinstance(wc_range, tuple):
            return wc_range
        # Map specific labels to numeric ranges for internal logic
        if wc_range == '50-100': return (50, 100)
        if wc_range == '100-200': return (100, 200)
        if wc_range == '200-300': return (200, 300)
        if wc_range == '300-400': return (300, 400)
        if wc_range == '200+': return (200, 500)
        if wc_range == '500+': return (500, 99999)
        if wc_range in ['1500+', '2000+']: return (500, 99999) # Backward compatibility
        
        # Generic dash parsing fallback if present in string format
        if '-' in str(wc_range):
            try:
                p = str(wc_range).split('-')
                return (int(p[0]), int(p[1]))
            except:
                pass
                
        return (0, 0)
    
    def remove_player(self, user_id, force=False):
        """Remove player or spectator from room. 
        Note: Penalties are handled in app.py via apply_leave_penalty()
        """
        uid_str = str(user_id)
        is_daily = (self.time_limit >= 7200)
        
        # PERSISTENCE: Never remove PLAYERS from 24h rooms unless forced (e.g. logout)
        # However, we ALWAYS allow removing spectators.
        if is_daily and not force:
            self.spectators = [p for p in self.spectators if str(p.user_id) != uid_str]
            return

        initial_players = len(self.players)
        initial_specs = len(self.spectators)
        
        leaving_player = next((p for p in self.players if str(p.user_id) == uid_str), None)
        username = leaving_player.username if leaving_player else "Someone"
        
        # Track removal in audit logs
        if leaving_player:
            try:
                with open(RATING_AUDIT_PATH, 'a') as log:
                    log.write(f"[{time.time()}] Removal: User {username}, Score: {leaving_player.score}, Words: {len(leaving_player.submitted_words)}, Room: {self.room_id}, Round: {self.current_round}, State: {self.state}\n")
            except: pass
            
            # If in active round, add to round_quitters to prevent re-entering for same round
            if self.state == 'active':
                if not any(str(q.user_id) == uid_str for q in self.round_quitters):
                    self.round_quitters.append(leaving_player)

        # Remove from both lists
        self.players = [p for p in self.players if str(p.user_id) != uid_str]
        self.spectators = [p for p in self.spectators if str(p.user_id) != uid_str]
        
        if is_daily:
            self.save_active_players()
        
        # When the last human player leaves a non-daily room, mark it as closing.
        # IMPORTANT: Do NOT call room_manager.delete_room() here — remove_player can be invoked
        # from check_inactivity() inside the cleanup_rooms background loop which may hold
        # self.lock. Calling delete_room() here would deadlock on self.lock.
        # Actual deletion is handled by:
        #   - leave_room() in app.py (for explicit /api/room/<id>/leave calls)
        #   - cleanup_rooms() loop (for inactivity evictions — checks is_closing flag)
        humans = [p for p in self.players if not p.is_ai]
        
        if leaving_player and not humans and not is_daily:
            print(f"[GameRoom] Last human player ({username}) has left room {self.room_id}. Marking as closing.")
            self.is_closing = True
            self.spectators = []

        # If forced (logout), clear from past_players archive (except for 24h rooms where persistence is mandatory)
        if force and not is_daily:
            if uid_str in self.past_players:
                del self.past_players[uid_str]
                print(f"[GameRoom] Cleared {username} from past_players (Logout/Force)")

        # Remove from spectators
        initial_specs = len(self.spectators)
        leaving_spec = next((s for s in self.spectators if str(s.user_id) == str(user_id)), None)
        spec_username = leaving_spec.username if leaving_spec else "Someone"
        
        self.spectators = [p for p in self.spectators if str(p.user_id) != str(user_id)]
        if len(self.spectators) < initial_specs:
            print(f"[GameRoom] Removed spectator {user_id} ({spec_username}) from room {self.room_id}")
            self.add_chat_message("System", f"{spec_username} has left the room.", is_system=True)

    def update_player_activity(self, user_id, is_user_interaction=False):
        """Update last_active timestamp for a player or spectator"""
        uid_str = str(user_id)
        player = self.get_player(user_id)
        if player:
            player.last_active = time.time()
            if is_user_interaction:
                player.user_last_interacted = time.time()
            return

        # Check spectators too
        for p in self.spectators:
            if str(p.user_id) == uid_str:
                p.last_active = time.time()
                if is_user_interaction:
                    p.user_last_interacted = time.time()
                break

    def check_inactivity(self, timeout=600, spec_timeout=600): 
        """Remove players and spectators who haven't been active for their respective timeout seconds"""
        now = time.time()
        is_daily = self.time_limit >= 7200
        
        # Collect IDs to remove to avoid modifying list during iteration (10-minute standard timeout)
        to_remove_ids = []
        for p in self.players:
            last_interacted = getattr(p, 'user_last_interacted', getattr(p, 'last_active', now))
            human_idle_age = now - last_interacted
            network_age = now - getattr(p, 'last_active', now)
            age = max(human_idle_age, network_age)
            if age >= timeout and not p.is_ai:
                # PERSISTENCE: Never remove players from 24h rooms for inactivity
                if is_daily:
                    continue 
                to_remove_ids.append(p.user_id)

        players_removed = False
        for uid in to_remove_ids:
            # First, find player for logging BEFORE removal
            leaver = next((p for p in self.players if str(p.user_id) == str(uid)), None)
            username = leaver.username if leaver else "Unknown"
            
            log_msg = f"[GameRoom] Removing inactive player {username} (ID={uid}) in room {self.room_id} (inactive for >{timeout}s)\n"
            print(log_msg.strip())
            with open('inactivity_debug.log', 'a') as f:
                f.write(f"{datetime.datetime.now()} {log_msg}")

            # remove_player handles the abandonment rating penalty logic!
            self.remove_player(uid, force=True)
            self.evicted_users[str(uid)] = 'inactivity'
            players_removed = True

        # Check spectators
        active_spectators = []
        specs_removed = False
        for p in self.spectators:
            last_interacted = getattr(p, 'user_last_interacted', getattr(p, 'last_active', now))
            human_idle_age = now - last_interacted
            network_age = now - getattr(p, 'last_active', now)
            age = max(human_idle_age, network_age)
            # Spectators get a longer timeout (default 10-30 mins depending on call)
            # PERSISTENCE: No idle limit for 24h rooms
            if age < spec_timeout or is_daily:
                active_spectators.append(p)
            else:
                log_msg = f"[GameRoom] Removing inactive spectator {p.username} (ID={p.user_id}) in room {self.room_id} (inactive for {age:.1f}s)\n"
                print(log_msg.strip())
                with open('inactivity_debug.log', 'a') as f:
                    f.write(f"{datetime.datetime.now()} {log_msg}")
                self.evicted_users[str(p.user_id)] = 'inactivity'
                specs_removed = True
                
        if specs_removed:
            self.spectators = active_spectators
            
        return players_removed or specs_removed
    
    def get_player(self, user_id):
        """Get player by ID or username"""
        uid_str = str(user_id).lower()
        for p in self.players:
            if str(p.user_id).lower() == uid_str or str(p.username).lower() == uid_str:
                return p
        return None

    def get_spectator(self, user_id):
        """Get spectator by ID"""
        uid_str = str(user_id)
        for s in self.spectators:
            if str(s.user_id) == uid_str:
                return s
        return None

    def get_state(self, user_id=None):
        """Get serialized room state snapshot dictionary for client consumption."""
        try:
            from app import get_room_state_dict
            st_dict, _ = get_room_state_dict(self.room_id, uid=user_id)
            return st_dict
        except Exception as e:
            return None
    
    @property
    def time_remaining(self):
        """Calculate time remaining in current state"""
        # PRIORITY: Intermission timer is always literal (60s)
        # 1. Intermission timer (Fixed 60s or 5s for Daily)
        if self.state == 'intermission':
            if not self.intermission_start_time or self.intermission_start_time <= 0:
                self.intermission_start_time = time.time()
            elapsed = time.time() - self.intermission_start_time
            intermission_limit = 5 if self.time_limit >= 7200 else 60
            return max(0.0, intermission_limit - elapsed)
            
        # 2. 24h Room ACTIVE: Align to real-world midnight boundary (America/Chicago)
        if self.state == 'active' and self.time_limit >= 7200:
            import datetime
            from zoneinfo import ZoneInfo
            tz = DAILY_TZ
            now_tz = datetime.datetime.now(tz)
            next_midnight = datetime.datetime.combine(now_tz.date() + datetime.timedelta(days=1), datetime.time.min, tzinfo=tz)
            delta = (next_midnight - now_tz).total_seconds()
            return max(0.0, delta)

        if self.state == 'active':
            if self.custom_end_time > 0:
                return max(0.0, self.custom_end_time - time.time())
            
            elapsed = time.time() - self.round_start_time
            return max(0.0, self.time_limit - elapsed)
        elif self.state == 'waiting':
             return self.time_limit # Use the limit as the waiting value
        return 0
    
    @property
    def round_end_time(self):
        """Get timestamp when current round ends (for client sync)"""
        # 24h Room (>= 2h limit): Always align dynamically to real-world midnight boundary (America/Chicago)
        if self.time_limit >= 7200:
            import datetime
            from zoneinfo import ZoneInfo
            tz = DAILY_TZ
            now_tz = datetime.datetime.now(tz)
            next_midnight = datetime.datetime.combine(now_tz.date() + datetime.timedelta(days=1), datetime.time.min, tzinfo=tz)
            return next_midnight.timestamp()

        if self.state == 'active':
            if self.custom_end_time > 0:
                return self.custom_end_time
            return self.round_start_time + self.time_limit
        return 0
    
    @property
    def intermission_end_time(self):
        """Get timestamp when intermission ends (for client sync)"""
        if self.state == 'intermission':
            if not self.intermission_start_time or self.intermission_start_time <= 0:
                self.intermission_start_time = time.time()
            limit = 5 if self.time_limit >= 7200 else 60
            return self.intermission_start_time + limit
        return 0
    
    def save_active_players(self):
        """Persist active players and their submissions for 24h rooms to DB"""
        if self.time_limit < 7200:
            return
        try:
            import sqlite3
            import json
            import os
            
            # Serialize active players
            players_data = []
            for p in self.past_players.values():
                players_data.append({
                    'user_id': p.user_id,
                    'username': p.username,
                    'rating': p.rating,
                    'submitted_words': p.submitted_words,
                    'invalid_words': p.invalid_words,
                    'score': p.score,
                    'previous_round_score': p.previous_round_score,
                    'games_played': p.games_played,
                    'previous_submitted_words': p.previous_submitted_words,
                    'found_bonus_word': p.found_bonus_word,
                    'last_active': p.last_active,
                    'input_method': p.input_method,
                    'country_flag': p.country_flag,
                    'joined_mid_round': p.joined_mid_round,
                    'has_exceptional_round': p.has_exceptional_round,
                    'is_guest': p.is_guest,
                    'is_ai': p.is_ai,
                    'ai_rating': p.ai_rating,
                    'has_abandoned': p.has_abandoned,
                    'game_room_chat': getattr(p, 'game_room_chat', True)
                })
            players_json = json.dumps(players_data)
            
            with get_db() as conn:
                conn.execute('''
                    UPDATE active_boards SET active_players_json = ? WHERE room_id = ?
                ''', (players_json, self.room_id))
        except Exception as e:
            print(f"[GameRoom] Error saving active players to DB for {self.room_id}: {e}")
    
    def submit_word(self, user_id, word, path=None):
        """Submit word for player"""
        now = time.time()
        # BUZZER-BEATER GRACE PERIOD: Allow in-flight submissions sent at the final second of the round
        # that arrive within 3.0s of intermission start to be validated and credited to the completed round.
        is_grace_period = (
            self.state == 'intermission' and 
            getattr(self, 'intermission_start_time', 0) > 0 and 
            (now - self.intermission_start_time) <= 3.0
        )
        if is_grace_period:
            print(f"[GameRoom] Buzzer-beater submission accepted during grace period for user {user_id} in {self.room_id}: '{word}' (offset: {now - self.intermission_start_time:.3f}s)")
        if self.state != 'active' and not is_grace_period:
            return False, "Round is not active", 0, None
            
        player = self.get_player(user_id)
        if not player:
            if self.get_spectator(user_id):
                return False, "Spectators cannot submit words", 0, None
            return False, "Player not in room", 0, None

        # Clean spectator list if they are an active player
        self.spectators = [s for s in self.spectators if str(s.user_id) != str(user_id)]
        
        # Safeguard: if player submits word while active and was not in round_starters, flag joined_mid_round
        if player and not getattr(self, 'is_daily', False) and (self.time_limit < 7200) and self.state == 'active':
            round_starters = getattr(self, 'round_starting_player_ids', None)
            if round_starters and str(user_id) not in round_starters:
                player.joined_mid_round = True

        # Update activity on submission
        self.update_player_activity(user_id)
        
        if not word or not word.strip():
             return False, "Empty word", 0, None
        
        word = word.strip().upper()

        # --- SUBANAGRAMS PRACTICE MODE WORD SUBMISSION ---
        if getattr(self, 'game_type', None) == 'subanagrams':
            min_len_req = self.current_min_length
            if len(word) < min_len_req:
                return False, f"{word} IS TOO SHORT (MIN: {min_len_req}L)", 0, None

            matched_word = None
            if word in self.all_words:
                matched_word = word
            else:
                from collections import Counter
                raw_seq = "".join(self.board[0]) if (self.board and len(self.board) > 0 and isinstance(self.board[0], list)) else ""
                # QU tile is stored as 'Q' but contributes both Q and U — expand before counting
                seq_letters = raw_seq.replace('Q', 'QU')
                seq_cnt = Counter(seq_letters)
                w_cnt = Counter(word)
                if all(seq_cnt[c] >= cnt for c, cnt in w_cnt.items()):
                    if word_validator.word_validator.is_valid_word(word, getattr(self, 'current_dictionary', 'NWL'), use_added_words=getattr(self, 'use_added_words', False)):
                        matched_word = word
                        self.all_words.add(word)
                        if not hasattr(self, 'solved_words_with_scores') or not self.solved_words_with_scores:
                            self.solved_words_with_scores = {}
                        from scoring import calculate_word_score
                        self.solved_words_with_scores[word] = calculate_word_score(
                            word, self.bonus_word, board_format='Normal', return_details=True
                        )
                        self.total_words_count = len(self.all_words)
                        self.recalculate_total_points()
                        self.update_counts_by_len()

            if not matched_word:
                player.invalid_words.append(word)
                return False, f"{word} INVALID", 0, None

            final_word = matched_word
            existing_words = {(w.get('word') if isinstance(w, dict) else str(w)).upper() for w in player.submitted_words}
            if final_word in existing_words:
                return False, f"{final_word} ALREADY FOUND", 0, None

            from scoring import calculate_word_score
            points_data = calculate_word_score(
                final_word,
                self.bonus_word,
                board_format='Normal',
                return_details=True
            )
            points = points_data['total']
            word_timestamp = time.time()
            is_bonus = bool(self.bonus_word and final_word.strip().upper() == self.bonus_word.strip().upper())
            word_metadata = {
                'word': final_word,
                'time': word_timestamp,
                'points': points,
                'score_details': points_data,
                'path': None,
                'is_bonus': is_bonus,
                'input_method': getattr(player, 'input_method', 'mouse') or 'mouse'
            }
            player.submitted_words.append(word_metadata)
            try:
                import app
                if hasattr(app, '_WORD_FINDS_CACHE') and final_word.upper() in app._WORD_FINDS_CACHE:
                    del app._WORD_FINDS_CACHE[final_word.upper()]
            except Exception:
                pass
            if is_bonus:
                player.found_bonus_word = True
                print(f"[GameRoom] Player {player.username} found the BONUS WORD: {final_word}!")
            self._recalculate_player_score(player)
            msg = f"BONUS WORD! ({points} PTS)" if is_bonus else f"{final_word} VALID (+{points})"
            return True, msg, points, final_word
        
        # Check if word is valid
        matched_word = None
        
        # For case-insensitive checks
        fmt_lower = self.current_board_format.lower()
        is_eo = 'either' in fmt_lower
        is_bonus_mode = ('bonus' in fmt_lower or is_eo)

        # In Either/Or format or with Q/QU tiles, mousing a path blindly sends the default tile letters.
        # We reconstruct the possible words from the provided path and find the one that is valid.
        has_special_tile = False
        if path:
            is_3d_check = isinstance(self.board[0][0], list)
            for node in path:
                f, r, c = -1, -1, -1
                if isinstance(node, dict):
                    f, r, c = int(node.get('f', -1)), int(node.get('r', -1)), int(node.get('c', -1))
                elif isinstance(node, (list, tuple)):
                    if len(node) == 3: f, r, c = int(node[0]), int(node[1]), int(node[2])
                    elif len(node) == 2: r, c = int(node[0]), int(node[1])
                cv = ''
                if is_3d_check and 0 <= f < len(self.board) and 0 <= r < len(self.board[f]) and 0 <= c < len(self.board[f][r]):
                    cv = str(self.board[f][r][c]).upper()
                elif 0 <= r < len(self.board) and 0 <= c < len(self.board[0]):
                    cv = str(self.board[r][c]).upper()
                if '/' in cv or cv in ('Q', 'QU'):
                    has_special_tile = True
                    break

        if path and (is_eo or has_special_tile):
            possible_words = ['']
            valid_path = True
            is_3d_board = isinstance(self.board[0][0], list)
            
            for node in path:
                f, r, c = -1, -1, -1
                if isinstance(node, dict):
                    f = int(node.get('f', -1))
                    r = int(node.get('r', -1))
                    c = int(node.get('c', -1))
                elif isinstance(node, (list, tuple)):
                    if len(node) == 3: f, r, c = int(node[0]), int(node[1]), int(node[2])
                    elif len(node) == 2: r, c = int(node[0]), int(node[1])

                cell_val = ''
                if is_3d_board and f >= 0 and f < len(self.board) and r >= 0 and r < len(self.board[f]) and c >= 0 and c < len(self.board[f][r]):
                    cell_val = str(self.board[f][r][c])
                elif r >= 0 and r < len(self.board) and c >= 0 and c < len(self.board[0]):
                    cell_val = str(self.board[r][c])
                else:
                    valid_path = False
                    break
                
                if '/' in cell_val:
                    options = cell_val.split('/')
                    new_words = []
                    for prefix in possible_words:
                        for opt in options:
                            opt_u = opt.upper()
                            if opt_u == 'Q' or opt_u == 'QU':
                                new_words.append(prefix + 'QU')
                                new_words.append(prefix + 'Q')
                            else:
                                new_words.append(prefix + opt)
                    possible_words = new_words
                else:
                    cell_u = cell_val.upper()
                    additions = [cell_val]
                    if cell_u == 'Q' or cell_u == 'QU':
                        additions = ['QU', 'Q']
                    new_words = []
                    for prefix in possible_words:
                        for add in additions:
                            new_words.append(prefix + add)
                    possible_words = new_words
            
            if valid_path:
                # Find which of the possible interpreted words from the path actually exists on the board
                # USER REQUEST: Use dictionary validation instead of just all_words to match Solo mode!
                
                submitted_word_upper = word.replace('[', '').replace(']', '').replace('/', '').strip().upper()
                
                valid_options = []
                for w in possible_words:
                    if w in self.all_words:
                        valid_options.append(w)
                    elif word_validator.word_validator.is_valid_word(w, getattr(self, 'current_dictionary', 'NWL'), use_added_words=getattr(self, 'use_added_words', False)):
                        valid_options.append(w)
                
                if submitted_word_upper in valid_options:
                    word = submitted_word_upper
                    matched_word = word
                elif len(valid_options) >= 1:
                    word = valid_options[0]  # Auto-correct the submission to the valid Either/Or letter
                    matched_word = word
                elif any(w.upper() == submitted_word_upper for w in possible_words):
                    word = submitted_word_upper
                elif len(possible_words) > 0:
                    # Fallback: Use the first possible word if none are valid (prevents outputting "F/U" in word)
                    word = possible_words[0]
        
        # 2. Logic Check
        is_in = word in self.all_words
        min_len_req = self.current_min_length

        # EARLY EXIT: Check minimum length FIRST (User Request: Clearer feedback)
        # Boggle usually treats 'Q' as 'QU', so check if length would be sufficient even with expansion.
        # Avoid overcounting if 'Q' is already followed by 'U' (e.g. "QUAKE" is length 5, not 6).
        import re
        effective_len = len(re.sub(r'Q(?!U)', 'QU', word))
        if effective_len < min_len_req:
            return False, f"{word.upper()} IS TOO SHORT (MIN: {min_len_req}L)", 0, None

        # Direct match check
        if is_in:
            matched_word = word
        elif 'Q' in word and word.replace('Q', 'QU') in self.all_words:
            matched_word = word.replace('Q', 'QU')
        elif 'QU' in word and word.replace('QU', 'Q') in self.all_words:
            matched_word = word.replace('QU', 'Q')
        else:
            # Self-healing fallback for truncated/missing valid words on the board
            # Test both original word and QU / Q variants
            candidates = [word]
            if 'Q' in word:
                candidates.append(word.replace('Q', 'QU'))
            if 'QU' in word:
                candidates.append(word.replace('QU', 'Q'))
                
            for cand in candidates:
                is_valid_dict = word_validator.word_validator.is_valid_word(cand, getattr(self, 'current_dictionary', 'NWL'), use_added_words=getattr(self, 'use_added_words', False))
                if is_valid_dict:
                    is_on_board, path = word_validator.word_validator.find_word_on_board(self.board, cand, return_path=True)
                    if is_on_board:
                        # Dynamically add to all_words and paths to accept it!
                        self.all_words.add(cand)
                        if not self.all_words_paths:
                            self.all_words_paths = {}
                        self.all_words_paths[cand] = path
                        
                        # Add to solved_words_with_scores so scoring works
                        if not hasattr(self, 'solved_words_with_scores') or not self.solved_words_with_scores:
                            self.solved_words_with_scores = {}
                        
                        from scoring import calculate_word_score
                        pts_data = calculate_word_score(
                            cand, 
                            self.bonus_word, 
                            board_format=self.current_board_format, 
                            path=path, 
                            bonus_cell=self.bonus_cell, 
                            board=self.board, 
                            return_details=True,
                            is_private=self.is_private,
                            strict_path=True
                        )
                        self.solved_words_with_scores[cand] = pts_data
                        
                        # Recalculate stats
                        self.recalculate_total_points()
                        self.total_words_count = len(self.all_words)
                        self.update_counts_by_len()
                        
                        matched_word = cand
                        print(f"[DynamicAccept] Dynamically accepted and added truncated/missing word: {cand}")
                        break
        
        if not matched_word:
            # PENALTY CHECK: Any sequence >= min length found on board but NOT in dictionary
            is_penalty = False
            # Penalty mode check: must be Penalty format AND NOT a 24h room (just in case)
            is_24h = self.time_limit >= 7200
            # Use snapshotted min_length to avoid contamination from next-round pre-generation
            min_len = self.current_min_length
            
            # PREFACE FIX (#7): In Penalty format, ANY invalid word of at least min length costs 3 points,
            # whether or not it can be traced on the board.
            if 'penalty' in fmt_lower and not is_24h:
                if len(word) >= int(min_len or 3):
                    is_penalty = True
            
            if is_penalty:
                # Apply penalty (-3 points)
                penalty_points = -3
                
                # Prevent spamming the same penalty word
                existing_words = {(w.get('word') if isinstance(w, dict) else str(w)).upper() for w in player.submitted_words}
                if word in existing_words:
                    return False, f"{word} ALREADY PENALIZED", 0, None
                
                player.submitted_words.append({
                    'word': word,
                    'time': time.time(),
                    'points': penalty_points,
                    'is_penalty': True,
                    'input_method': getattr(player, 'input_method', 'mouse') or 'mouse'
                })
                
                # Update score (Sequential floor at 0 to avoid negative debt)
                self._recalculate_player_score(player)
                return True, f"{word} INVALID (-3 PTS)", penalty_points, word
            else:
                # Standard invalid word
                player.invalid_words.append(word)
                return False, f"{word} INVALID", 0, None
        
        # Use the matched word (which might be the QU variant) for scoring/display
        final_word = matched_word
        
        # Check if already submitted (by this player)
        # Extract existing words from the list of dicts or strings safely
        existing_words = {(w.get('word') if isinstance(w, dict) else str(w)).upper() for w in player.submitted_words}
        if final_word in existing_words:
            return False, f"{final_word} ALREADY FOUND", 0, None
        
        # FCFS Mode: Check if word found by ANYONE
        if self.game_type == 'fcfs':
            if matched_word.upper() in self._fcfs_found_words_set:
                return False, f"{matched_word} FOUND BY ANOTHER", 0, None
            # Do NOT add to set yet; we do it after scoring so we have points and finder
        
        # Calculate score for this word (re-calculate with path for Either/Or and Bonus Letter support)
        from scoring import calculate_word_score
        points_data = calculate_word_score(
            final_word, 
            self.bonus_word, 
            board_format=self.current_board_format, 
            path=path, 
            bonus_cell=self.bonus_cell, 
            board=self.board, 
            return_details=True,
            is_private=self.is_private,
            strict_path=True
        )
        points = points_data['total']
        
        word_timestamp = time.time()
        is_bonus = bool(self.bonus_word and final_word.strip().upper() == self.bonus_word.strip().upper())
        word_metadata = {
            'word': final_word,
            'time': word_timestamp,
            'points': points,
            'score_details': points_data,
            'path': path,
            'is_bonus': is_bonus,
            'input_method': getattr(player, 'input_method', 'mouse') or 'mouse'
        }
        
        # USER: Update Density (Only after word is fully validated and scored)
        try:
            self.update_density_for_word(player, final_word, path)
        except Exception as e:
            print(f"[Density-Error] Failed to update density: {e}")
        
        player.submitted_words.append(word_metadata)
        try:
            import app
            if hasattr(app, '_WORD_FINDS_CACHE') and final_word.upper() in app._WORD_FINDS_CACHE:
                del app._WORD_FINDS_CACHE[final_word.upper()]
        except Exception:
            pass
        
        # FCFS: Update shared found words list for Live Feed synchronization
        if self.game_type == 'fcfs':
            shared_meta = dict(word_metadata)
            shared_meta['finder'] = player.username
            self.fcfs_found_words.append(shared_meta)
            self._fcfs_found_words_set.add(final_word.upper())
            print(f"[FCFS-Sync] Shared word '{final_word}' added for {player.username}. Total shared: {len(self.fcfs_found_words)}")
        
        # Check if this is the bonus word
        if is_bonus:
            player.found_bonus_word = True
            print(f"[GameRoom] Player {player.username} found the BONUS WORD: {final_word}!")
        
        # Update player score immediately (Sequential floor at 0 to avoid negative debt)
        self._recalculate_player_score(player)
        
        # Real-time Split Points Recalculation
        if self.game_type == 'split':
            self.calculate_split_scores()
            # After recalculation, re-fetch the points
            for w_obj in player.submitted_words:
                if isinstance(w_obj, dict) and w_obj.get('word') == final_word:
                    points = w_obj.get('points', points)
                    break
        else:
            # For non-split modes (Accumulative, FCFS, Penalty), update 'points' from the recalculated object
            # to ensure user receives the correct score in the notification
            for w_obj in player.submitted_words:
                if isinstance(w_obj, dict) and w_obj.get('word') == final_word:
                    points = w_obj.get('points', points)
                    break

        # Buzzer-beater grace period sync: Update winners_history if it was already created at round end
        if is_grace_period and hasattr(self, 'winners_history') and self.winners_history:
            for h in self.winners_history:
                if h.get('round') == self.current_round:
                    for p_entry in h.get('all_players', []):
                        if p_entry.get('username', '').lower() == player.username.lower():
                            p_entry['score'] = player.score
                    for w_user in h.get('winners', []):
                        if w_user.get('username', '').lower() == player.username.lower():
                            h.setdefault('words', []).append({'word': final_word, 'points': points, 'time': word_timestamp})

        # Persistence: Save active players for 24h rooms after successful submission in background thread
        if self.time_limit >= 7200:
            threading.Thread(target=self.save_active_players, daemon=True).start()

        return True, f"{final_word} VALID", points, final_word

    def update_live_pe(self):
        """Calculates performance efficiency in real-time for UI trophy and awards sensitive exceptional badges."""
        # Pool all participating players (from self.players and self.past_players if applicable)
        all_candidate_players = list(self.players)
        existing_uids = {p.user_id for p in self.players}
        if hasattr(self, 'past_players') and isinstance(self.past_players, dict):
            for p in self.past_players.values():
                if p.user_id not in existing_uids:
                    all_candidate_players.append(p)
        
        # Active participants in this round
        active_players = [p for p in all_candidate_players if not getattr(p, 'is_ai', False) and (p.score > 0 or len(p.submitted_words) > 0 or len(p.invalid_words) > 0)]
        multiple_players = len(active_players) > 1
        
        tot_score = sum(p.score for p in active_players)
        tot_rating = sum(getattr(p, 'rating', 1200) for p in active_players)
        sorted_scores = sorted([p.score for p in active_players], reverse=True)
        max_score = sorted_scores[0] if sorted_scores else 0
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
        num_players = len(active_players)

        # Dynamic PE threshold scaled by room size:
        # In a 2-player room, doubling opponent's score corresponds to PE = 1.33x
        # In a 3-player room, doubling opponents' scores corresponds to PE = 1.60x
        if num_players <= 2:
            pe_threshold = 1.30
        elif num_players == 3:
            pe_threshold = 1.45
        elif num_players <= 5:
            pe_threshold = 1.55
        else:
            pe_threshold = 1.60

        for p in self.players:
            if getattr(p, 'is_ai', False):
                p.has_exceptional_round = False
                p.trophy_rounds_left = 0
                continue

            earned_this_round = False
            if multiple_players and tot_score > 0 and tot_rating > 0 and p.score > 0:
                p_rating = getattr(p, 'rating', 1200)
                expected = (p_rating / tot_rating) * tot_score
                p.performance_efficiency = round(p.score / expected, 2) if expected > 0 else 1.0

                # Award trophy if:
                # 1. Player meets the dynamic PE threshold (e.g. 1.45 for 3 players, where 1.60 gets it)
                # 2. OR player doubled the score of opponents (p.score == max_score and p.score >= 1.75 * second_score and p.score >= 15)
                # 3. OR raw exceptional PE >= 2.0
                is_high_pe = (p.performance_efficiency >= pe_threshold and p.score >= 10)
                is_doubled_opponent = (p.score == max_score and p.score >= 1.75 * second_score and p.score >= 15)
                is_raw_domination = (p.performance_efficiency >= 2.0)

                if is_high_pe or is_doubled_opponent or is_raw_domination:
                    earned_this_round = True
            else:
                p.performance_efficiency = 1.0 if (p.score > 0 and not multiple_players) else 0.0

            if earned_this_round:
                p.has_exceptional_round = True
            else:
                p.has_exceptional_round = False
    
    def initialize_player_densities(self):
        """Initializes or resets player-specific cell densities from the room's initial density grid."""
        if not self.initial_cell_density:
            return
        is_3d = (self.board and isinstance(self.board[0], list) and len(self.board[0]) > 0 and isinstance(self.board[0][0], list))
        for p in self.players:
            p.cell_density = [row[:] for row in self.initial_cell_density] if not is_3d else [[row[:] for row in face] for face in self.initial_cell_density]

    def _initialize_player_density_grid(self, player):
        """Helper to safely initialize a player's density grid from the room's initial density."""
        if self.initial_cell_density:
            is_3d = (self.board and isinstance(self.board[0], list) and len(self.board[0]) > 0 and isinstance(self.board[0][0], list))
            player.cell_density = [row[:] for row in self.initial_cell_density] if not is_3d else [[row[:] for row in face] for face in self.initial_cell_density]
        else:
            player.cell_density = []

    def update_density_for_word(self, player, word, path=None):
        """Decrement cell density for found words in Density format for the given player"""
        cur_fmt = str(self.current_board_format).lower()
        if 'density' in cur_fmt and player:
            word_upper = word.upper()
            
            # Ensure player's density grid is initialized
            if not getattr(player, 'cell_density', None):
                self._initialize_player_density_grid(player)
                
            # Get path (User path or pre-calculated path)
            word_path = path or (self.all_words_paths.get(word_upper) if hasattr(self, 'all_words_paths') else None)
            if word_path and player.cell_density:
                # Handle 3D [f, r, c] vs 2D [r, c]
                is_3d = self.board and isinstance(self.board[0], list) and isinstance(self.board[0][0], list)
                for node in word_path:
                    try:
                        if is_3d:
                            coords = list(map(int, node))
                            if len(coords) == 3:
                                f, r, c = coords
                            elif len(coords) == 2:
                                f, r, c = 0, coords[0], coords[1]
                            else: continue
                            
                            if f < len(player.cell_density) and r < len(player.cell_density[f]) and c < len(player.cell_density[f][r]):
                                if player.cell_density[f][r][c] > 0:
                                    player.cell_density[f][r][c] -= 1
                        else:
                            coords = list(map(int, node))
                            r, c = coords[-2:]
                            if r < len(player.cell_density) and c < len(player.cell_density[r]):
                                if player.cell_density[r][c] > 0:
                                    player.cell_density[r][c] -= 1
                    except (IndexError, TypeError, ValueError): continue
                
                # Still track globally for FCFS/metrics
                self.global_round_found_words.add(word_upper)
                return True
        return False

    def initialize_density(self, board, all_words_paths, board_format, is_staging=False):
        """Pre-calculates word density per cell for the 'Density' format. Reset global tracking."""
        import math
        self.global_round_found_words = set()
        
        fmt_low = str(board_format).lower()
        if not board_format or fmt_low == 'none' or 'density' not in fmt_low:
            # Fallback for dynamic detection
            if is_staging and hasattr(self, 'next_spinner_params') and self.next_spinner_params:
                fmt_low = str(self.next_spinner_params.get('board_format', '')).lower()
            elif hasattr(self, 'current_board_format') and self.current_board_format:
                fmt_low = str(self.current_board_format).lower()
            elif hasattr(self, 'spinner_params') and self.spinner_params:
                fmt_low = str(self.spinner_params.get('board_format', '')).lower()
            
        # LOGGING START
        try:
            with open(DEBUG_FLOW_PATH, 'a') as f:
                f.write(f"[Density-Diag] Room: {self.room_id} | Format: {fmt_low} | Staging: {is_staging} | Board: {len(board) if board else 0} | Paths: {len(all_words_paths) if all_words_paths else 0} at {time.time()}\n")
        except Exception as e:
            print(f"[Density-Diag] Failed to write to log: {e}")

        if 'density' not in fmt_low or not board:
            if is_staging:
                self.next_round_cell_density = []
                self.next_round_initial_cell_density = []
                self.next_round_max_cell_density = 0
            else:
                self.cell_density = []
                self.initial_cell_density = []
                self.max_cell_density = 0
            return

        # Check if 3D board [face][row][col]
        is_3d = (board and isinstance(board[0], list) and len(board[0]) > 0 and isinstance(board[0][0], list))
        
        print(f"[Density] Initializing for format={board_format}. Board: is_3d={is_3d}. Words with paths: {len(all_words_paths) if all_words_paths else 0}")
        if is_3d:
            depth, rows, cols = len(board), len(board[0]), len(board[0][0])
            raw_counts = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(depth)]
            words_processed = 0
            for word, path in (all_words_paths or {}).items():
                if not path or not isinstance(path, (list, tuple)): continue
                words_processed += 1
                for node in path:
                    try:
                        coords = list(map(int, node))
                        if len(coords) == 3:
                            f, r, c = coords
                        elif len(coords) == 2:
                            f, r, c = 0, coords[0], coords[1]
                        else: continue
                        
                        if 0 <= f < depth and 0 <= r < rows and 0 <= c < cols:
                            raw_counts[f][r][c] += 1
                    except (IndexError, TypeError, ValueError): continue

            max_count = 0
            for f in range(depth):
                for r in range(rows):
                    for c in range(cols):
                        if raw_counts[f][r][c] > max_count:
                            max_count = raw_counts[f][r][c]

            density_grid = raw_counts
        else:
            rows, cols = len(board), len(board[0])
            raw_counts = [[0 for _ in range(cols)] for _ in range(rows)]
            for word, path in (all_words_paths or {}).items():
                if not path or not isinstance(path, (list, tuple)): continue
                for node in path:
                    try:
                        r, c = node[-2:]
                        if 0 <= r < rows and 0 <= c < cols:
                            raw_counts[r][c] += 1
                    except (IndexError, TypeError, ValueError): continue

            max_count = 0
            for r in range(rows):
                for c in range(cols):
                    if raw_counts[r][c] > max_count:
                        max_count = raw_counts[r][c]

            density_grid = raw_counts
        
        max_d = max(1, max_count)

        if is_staging:
            self.next_round_cell_density = [row[:] for row in density_grid] if not is_3d else [[row[:] for row in face] for face in density_grid]
            self.next_round_initial_cell_density = [row[:] for row in density_grid] if not is_3d else [[row[:] for row in face] for face in density_grid]
            self.next_round_max_cell_density = max_d
        else:
            self.cell_density = density_grid
            self.initial_cell_density = [row[:] for row in density_grid] if not is_3d else [[row[:] for row in face] for face in density_grid]
            self.max_cell_density = max_d
            self.initialize_player_densities()
        
        print(f"[Density] Initialization complete. Max density steps: {max_d}")
    
    def _recalculate_player_score(self, player):
        """
        Recalculate player score from submitted words sequentially.
        """
        # Normalize elements in submitted_words to ensure every entry is a dict
        normalized_words = []
        for w in (player.submitted_words or []):
            if isinstance(w, dict):
                normalized_words.append(w)
            elif isinstance(w, str):
                normalized_words.append({'word': w.upper(), 'time': 0, 'points': 1, 'score_details': {'total': 1}})
            else:
                normalized_words.append({'word': str(w).upper(), 'time': 0, 'points': 1, 'score_details': {'total': 1}})
        player.submitted_words = normalized_words

        # Sort by submission time
        sorted_words = sorted(player.submitted_words, key=lambda x: x.get('time', 0))
        current_score = 0
        fmt = self.current_board_format
        import logging
        logger = logging.getLogger("scoring")
        logger.debug(f"[Recalc] Re-evaluating score for {player.username}. Words: {len(player.submitted_words)} | Room FMT: {fmt}")
        
        for w_obj in sorted_words:
            p_val = w_obj.get('points')
            w_str = w_obj.get('word', '')
            
            if p_val is not None:
                # Use pre-calculated value
                points = p_val
                # Still need details for the frontend breakdown if possible
                points_details = w_obj.get('score_details', {'total': points})
            else:
                # Use word_path from solver to avoid slow DFS for typed words (essential for round-end fluid transitions)
                word_path = (self.all_words_paths or {}).get(w_str, w_obj.get('path'))
                
                points_details = calculate_word_score(
                    w_str, 
                    self.bonus_word, 
                    board_format=fmt,
                    path=word_path,
                    bonus_cell=self.bonus_cell,
                    board=self.board,
                    return_details=True,
                    is_private=self.is_private,
                    strict_path=True
                )
                points = points_details['total']
                # Record it for future iterations (prevents redundant calls/overwrites)
                w_obj['points'] = points
                w_obj['score_details'] = points_details
            
            # Apply points (Floor total score at 0 per user requirement)
            current_score += points
            if current_score < 0:
                current_score = 0
            
                
        player.score = current_score
        return current_score
    
    def get_next_round_milestone(self):
        """Returns which milestone we're at for the NEXT round.
        
        Returns:
            'spinner' - Time to generate Spinner Set for N+1 (Proactive)
            'search' - Time to start board search for N+1 (Proactive)
            'reveal' - At 15s elapsed in intermission, time to reveal params
            'start' - At 0s remaining in intermission, time to start round
            None - No milestone reached yet
        """
        now = time.time()
        
        # ISSUE 6 FIX: Never fire any milestone when state is 'waiting'.
        # round_start_time=0 in waiting state makes state_elapsed huge (now - 0),
        # which causes 'spinner' and 'search' milestones to trigger on every tick/poll,
        # generating board after board and triggering start_next_round repeatedly.
        if self.state in ['waiting', 'loading']:
            if self.state == 'waiting':
                all_p = [p for p in self.players + self.spectators if not getattr(p, 'is_ai', False)]
                humans = [p for p in all_p if (now - getattr(p, 'last_active', 0)) < 120.0]
                if len(humans) > 0:
                    print(f"[get_next_round_milestone] Unpausing waiting room {self.room_id} cleanly via check_and_update_state().")
                    self.check_and_update_state()
            return None
        
        # stuck watchdog: check if intermission is stuck for > 10s at 0:00:00 (timer at 0 and state == intermission)
        # Determine normal intermission duration
        intermission_limit = 5 if self.time_limit >= 7200 else 60
        is_at_or_past_zero = (self.state == 'intermission' and (now - self.intermission_start_time >= intermission_limit))

        
        # USER MANDATE: 0:50 Intermission Fallback Watchdog (50s remaining on clock = 10s elapsed in 60s intermission, BEFORE 0:45 reveal)
        # If no board has been secured by 0:50 time during intermission, pull a pregenerated board immediately,
        # test its Spinner Set parameters, stage it, and update next_spinner_params so 0:45 reveal locks the exact board parameters!
        if self.state == 'intermission' and self.time_limit < 7200:
            elapsed_intermission = now - self.intermission_start_time
            fallback_050_threshold = 10.0 if intermission_limit >= 20 else 0.2
            if elapsed_intermission >= fallback_050_threshold and not getattr(self, 'next_round_board', None) and not getattr(self, '_fallback_at_50s_done', False):
                print(f"[Watchdog] 0:50 mark in intermission ({elapsed_intermission:.1f}s elapsed) for {self.room_id} — no board ready. Pulling validated pregenerated board.")
                self._fallback_at_50s_done = True
                def _bg_ensure():
                    try:
                        self.ensure_next_board_ready()
                    except Exception as wd_err:
                        print(f"[Watchdog] Error in 0:50 ensure_next_board_ready for {self.room_id}: {wd_err}")
                import threading
                threading.Thread(target=_bg_ensure, daemon=True).start()

        if getattr(self, 'starting_round', False):
            curr_init = getattr(self, '_round_start_init_time', 0)
            timeout = 3.0
            if curr_init > 0 and (time.time() - curr_init > timeout):
                self.starting_round = False
                print(f"[RoomManager] STALE starting_round detected for {self.room_id} (>{timeout}s). Resetting.")
            
        # 1. Start Milestone: Trigger 0.5s early to ensure next round starts slightly early
        # and the new board is ready when the client timer hits 0:00.
        if self.state == 'intermission':
            if not self.intermission_start_time or self.intermission_start_time <= 0:
                self.intermission_start_time = now
            elapsed = now - self.intermission_start_time
            intermission_limit = 5 if self.time_limit >= 7200 else 60
            if elapsed >= intermission_limit - 0.6:
                return 'start'
            
        # 2. Parameter Reveal (15s into intermission = 0:45 on clock)
        if self.state == 'intermission':
            if not self.intermission_start_time or self.intermission_start_time <= 0:
                self.intermission_start_time = now
            elapsed = now - self.intermission_start_time
            intermission_duration = 5 if self.time_limit >= 7200 else 60
            reveal_threshold = 15.0 if intermission_duration >= 20 else 0.5
            if elapsed >= reveal_threshold and not getattr(self, 'spinner_params_revealed', False):
                return 'reveal'

        # 3. Proactive Parameter Generation (1s into ANY state)
        state_start = self.intermission_start_time if self.state == 'intermission' else self.round_start_time
        state_elapsed = now - (state_start if state_start > 0 else now)
        
        if state_elapsed >= 1.0 and not getattr(self, 'spinner_params_generated', False):
            if not getattr(self, 'spinner_params_loading', False):
                return 'spinner'
            
        # 4. Proactive Lead-Time Board Search (As soon as parameters are generated)
        if getattr(self, 'spinner_params_generated', False) and not getattr(self, 'board_search_started', False):
            if not getattr(self, 'board_search_loading', False):
                return 'search'
            
        return None

    def build_solo_spinner_params(self):
        """Resolve next round parameters for a Solo room, preserving all non-random initial_solo_params."""
        isp = getattr(self, 'initial_solo_params', None) or {}
        is_24h = (self.time_limit >= 7200)
        dict_choice = isp.get('dictionary', 'random')
        min_len_raw = isp.get('min_word_length', 'random')
        if min_len_raw == 'random' or not min_len_raw:
            min_word_len = SpinnerSet._spin_min_word_length(self.board_dimensions)
        else:
            try:
                min_word_len = int(min_len_raw)
            except Exception:
                min_word_len = SpinnerSet.get_allowed_min_lengths(self.board_dimensions)[0]

        bonus_len_choice = isp.get('bonus_word_length', 'random')
        if bonus_len_choice == 'random' or not bonus_len_choice or str(bonus_len_choice) in ['0', 'None', 'null', 'nan']:
            bonus_word_len = random.choices([6, 7, 8, 9, 10], weights=[20, 20, 20, 20, 20])[0]
        else:
            try:
                bonus_word_len = int(bonus_len_choice)
            except Exception:
                bonus_word_len = 8

        board_fmt = isp.get('board_format', 'random')
        difficulty_choice = isp.get('difficulty', 'random')
        wc_choice = isp.get('word_count_range', 'random')

        if not dict_choice or dict_choice == 'random':
            dictionary = SpinnerSet._spin_dictionary()
        else:
            dictionary = dict_choice

        use_aw = False
        if dictionary and ('+ AW' in str(dictionary) or '+AW' in str(dictionary)):
            use_aw = True
            dictionary = str(dictionary).replace('+ AW', '').replace('+AW', '').strip()
        elif dictionary == 'AW':
            use_aw = True
            dictionary = 'NWL'

        if not difficulty_choice or str(difficulty_choice).lower() == 'random':
            difficulty = SpinnerSet._spin_difficulty(self.board_dimensions, min_word_len)
        else:
            difficulty = SpinnerSet.strip_difficulty_pct(difficulty_choice)

        if not wc_choice or wc_choice == 'random':
            wc_range = SpinnerSet._spin_word_count(dictionary, min_word_len, difficulty, self.board_dimensions, use_added_words=use_aw)
        else:
            wc_range = wc_choice

        if use_aw and wc_range in ['50-100', '100-200', '200-300']:
            wc_range = '300-400'

        if not board_fmt or board_fmt == 'random':
            resolved_board_format = SpinnerSet._spin_board_format(is_24h=is_24h, dimensions=self.board_dimensions, previous_params=getattr(self, 'spinner_params', None))
        else:
            resolved_board_format = board_fmt

        dict_display = f"{dictionary} + AW" if use_aw else dictionary
        res = {
            'min_word_length': min_word_len,
            'difficulty': difficulty,
            'word_count_range': wc_range,
            'dictionary': dict_display,
            'board_format': resolved_board_format,
            'bonus_word_length': bonus_word_len,
            'use_added_words': use_aw,
            'board_dimensions': self.board_dimensions,
            'time_limit': self.time_limit,
            'generated_at': time.time()
        }
        res = SpinnerSet.sanitize_params(res, self.board_dimensions, is_24h)
        if difficulty_choice and str(difficulty_choice).lower() != 'random':
            res['difficulty'] = SpinnerSet.strip_difficulty_pct(difficulty_choice)
        return res

    def ensure_next_board_ready(self):
        """Ensure that self.next_round_board is populated immediately."""
        if getattr(self, 'next_round_board', None):
            return
            
        print(f"[GameRoom] ensure_next_board_ready: next_round_board is missing for {self.room_id}. Resolving instantly...")

        if getattr(self, 'game_type', None) == 'subanagrams':
            from subanagrams_generator import generate_subanagrams_board_and_words
            curr_p = getattr(self, 'spinner_params', None)
            p = getattr(self, 'next_spinner_params', None)
            if not p or not p.get('sequence_length') or (curr_p and p.get('sequence_length') == curr_p.get('sequence_length') and p.get('word_count_range') == curr_p.get('word_count_range') and p.get('dictionary') == curr_p.get('dictionary')):
                p = SpinnerSet.generate_subanagrams_params(previous_params=curr_p)
                self.next_spinner_params = p
            res = generate_subanagrams_board_and_words(p)
            if res:
                b_board, b_words, b_bonus, fparams = res
                from scoring import calculate_word_score
                self.next_round_board = b_board
                self.next_round_words = b_words
                self.next_round_word_paths = {w: [] for w in b_words}
                self.next_round_bonus = b_bonus
                self.next_round_format = fparams.get('board_format', 'Word Guaranteed')
                self.next_round_spinner_params = fparams
                self.next_spinner_params = fparams
                self.next_round_min_length = fparams.get('min_word_length', 3)
                self.next_round_word_scores = {
                    w: calculate_word_score(w, b_bonus, board_format='Normal', return_details=True)
                    for w in b_words
                }
                self.next_round_total_words_count = len(b_words)
                self.spinner_params_generated = True
                self.board_search_started = True
                self.board_search_loading = False
                self.board_search_started_actual = False
                print(f"[GameRoom] ensure_next_board_ready generated subanagrams board: {fparams.get('sequence_length')}L | {fparams.get('dictionary')} | {fparams.get('word_count_range')}")
            return
        from board_generator import pop_compatible_cached_board
        target_params = getattr(self, 'next_spinner_params', None)
        if not target_params:
            if getattr(self, 'is_solo', False) and getattr(self, 'initial_solo_params', None):
                target_params = self.build_solo_spinner_params()
            else:
                target_params = SpinnerSet.generate_params(
                    self.board_dimensions,
                    is_24h=(self.time_limit >= 7200),
                    previous_params=getattr(self, 'spinner_params', None)
                )
            self.next_spinner_params = dict(target_params)
        dict_val = target_params.get('dictionary', 'NWL')
        fmt_val = target_params.get('board_format', 'Normal')
        min_l_val = target_params.get('min_word_length', 3)
        use_aw_val = target_params.get('use_added_words', False) or '+ AW' in str(dict_val).upper() or '+AW' in str(dict_val).upper()
        bonus_word_len = target_params.get('bonus_word_length')
        diff_val = target_params.get('difficulty')
        target_range = target_params.get('word_count_range')
        fallback = None
        for pass_idx in range(2):
            req_range = target_range if pass_idx == 0 else None
            req_diff = diff_val if pass_idx == 0 else None
            if getattr(self, 'is_solo', False) and getattr(self, 'initial_solo_params', None):
                _solo_d_req = self.initial_solo_params.get('difficulty', 'random')
                if _solo_d_req and str(_solo_d_req).lower() != 'random':
                    req_diff = SpinnerSet.strip_difficulty_pct(_solo_d_req)
            for _ in range(10):
                candidate = pop_compatible_cached_board(
                    self.board_dimensions,
                    dict_val,
                    fmt_val,
                    min_l_val,
                    use_aw_val,
                    bonus_word_len=bonus_word_len if pass_idx == 0 else None,
                    word_count_range=req_range,
                    difficulty=req_diff
                )
                if not candidate:
                    break
                fb, fw, fc, ff, fp, fr, fbw, fparams = candidate
                
                grid_floor = 3
                if '4x6' in self.board_dimensions: grid_floor = 4
                elif '5x7' in self.board_dimensions: grid_floor = 5
                elif '6x8' in self.board_dimensions or '3x3x3' in self.board_dimensions: grid_floor = 6
                f_min_l = max(grid_floor, int(min_l_val) if min_l_val is not None else grid_floor)
                
                fw_filtered = [w for w in fw if len(w) >= f_min_l]
                if 'CSW' not in str(dict_val).upper() and hasattr(word_validator, 'word_validator'):
                    word_validator.word_validator.ensure_csw_loaded()
                    fw_filtered = [w for w in fw_filtered if not word_validator.word_validator.is_csw_only(w)]
                if fw_filtered:
                    actual_shortest = min(len(w) for w in fw_filtered)
                    if actual_shortest > f_min_l:
                        f_min_l = actual_shortest
                actual_cnt = len(fw_filtered)

                is_aw_effective = (use_aw_val or ((fparams.get('use_added_words', False) or '+ AW' in str(fparams.get('dictionary', '')).upper()) if fparams else False))
                if is_aw_effective and actual_cnt < 300:
                    print(f"[GameRoom] ensure_next_board_ready: Candidate for +AW had only {actual_cnt} words (< 300). Discarding...")
                    continue

                if (self.time_limit >= 7200) and not (300 <= actual_cnt <= 399):
                    print(f"[GameRoom] ensure_next_board_ready: Candidate for 24h had only {actual_cnt} words. Discarding...")
                    continue

                if target_range and not is_board_count_valid(actual_cnt, target_range):
                    print(f"[GameRoom] ensure_next_board_ready: Candidate count {actual_cnt} outside target range {target_range}. Discarding...")
                    continue

                if 'CSW' in str(dict_val).upper() and hasattr(word_validator, 'word_validator'):
                    word_validator.word_validator.ensure_csw_loaded()
                    if sum(1 for w in fw_filtered if word_validator.word_validator.is_csw_only(w)) == 0:
                        print(f"[GameRoom] ensure_next_board_ready: Candidate had 0 CSW-only words for CSW dictionary. Discarding...")
                        continue

                if is_aw_effective and hasattr(word_validator, 'word_validator'):
                    word_validator.word_validator.ensure_csw_loaded()
                    if sum(1 for w in fw_filtered if word_validator.word_validator.is_added_word(w)) == 0:
                        print(f"[GameRoom] ensure_next_board_ready: Candidate had 0 Added Words for +AW dictionary. Discarding...")
                        continue
                
                if req_diff:
                    _p_chk = self.board_dimensions.split('x')
                    _d_chk = int(_p_chk[0]) if len(_p_chk) == 3 else 1
                    _r_chk = int(_p_chk[1] if len(_p_chk) == 3 else _p_chk[0])
                    _c_chk = int(_p_chk[2] if len(_p_chk) == 3 else _p_chk[1])
                    _chk_d = _bg.get_difficulty_label(fr, rows=_r_chk, cols=_c_chk, dictionary=dict_val, depth=_d_chk, min_word_length=f_min_l)
                    if _chk_d != req_diff:
                        print(f"[GameRoom] ensure_next_board_ready: Candidate actual difficulty '{_chk_d}' != requested '{req_diff}'. Discarding...")
                        continue

                f_wc = req_range or (fparams.get('word_count_range', '100-200') if fparams else '100-200')
                min_accept = 50
                try:
                    min_accept = int(str(f_wc).split('-')[0])
                except:
                    if '50' in str(f_wc): min_accept = 50
                    elif '100' in str(f_wc): min_accept = 100
                    elif '200' in str(f_wc): min_accept = 200
                    elif '300' in str(f_wc): min_accept = 300
                    elif '400' in str(f_wc): min_accept = 400
                    elif '500' in str(f_wc): min_accept = 500
                
                if is_aw_effective:
                    min_accept = max(300, min_accept)
                elif f_min_l >= 6:
                    min_accept = min(min_accept, 30)
                else:
                    min_accept = max(30, min_accept)
                    
                if actual_cnt >= min_accept:
                    if str(fmt_val).lower() in ['equality freq', 'checkerboard'] and str(ff).lower() != str(fmt_val).lower():
                        print(f"[GameRoom] ensure_next_board_ready: Candidate format {ff} != requested {fmt_val}. Discarding...")
                        continue
                    matching_min = sum(1 for w in fw_filtered if len(w) == f_min_l)
                    if matching_min < 3:
                        print(f"[GameRoom] ensure_next_board_ready: Candidate had only {matching_min} words of min_len {f_min_l}. Discarding...")
                        continue
                    fallback = (fb, fw_filtered, fc, ff, {w: p for w, p in fp.items() if len(w) >= f_min_l and w in fw_filtered}, fr, fbw, fparams, f_min_l)
                    break
                else:
                    print(f"[GameRoom] ensure_next_board_ready Pop Candidate had only {actual_cnt} words of length >= {f_min_l} (needed {min_accept}). Discarding...")
            if fallback:
                break
                
        if fallback:
            fb, fw, fc, ff, fp, fr, fbw, fparams, f_min_l = fallback
            _bg = _room_manager_instance.board_generator if (_room_manager_instance and hasattr(_room_manager_instance, 'board_generator')) else BoardGenerator()
            _rep, _rw, _rp = sync_board_checkerboard_and_resolve(_bg, fb, fmt_val, f_min_l, dict_val, use_aw_val, fc)
            if _rep and _rw:
                fw, fp = _rw, _rp
            if 'CSW' not in str(dict_val).upper() and hasattr(word_validator, 'word_validator'):
                word_validator.word_validator.ensure_csw_loaded()
                fw = [w for w in fw if not word_validator.word_validator.is_csw_only(w)]
                fp = {w: p for w, p in fp.items() if w in fw}
            if fw:
                actual_shortest = min(len(w) for w in fw)
                if actual_shortest > f_min_l:
                    f_min_l = actual_shortest
            print(f"[GameRoom] ensure_next_board_ready: Popped fallback cached board for {self.room_id} ({len(fw)} words, min_len={f_min_l})")
            target_bw_len = int(bonus_word_len or (len(fbw) if fbw else (fparams.get('bonus_word_len', 8) if fparams else 8)) or 8)
            chosen_bw = pick_distinct_bonus_word(
                self, fbw, fw, target_bw_len,
                dictionary=dict_val, is_checkerboard=('checkerboard' in str(ff).lower()),
                room_manager=_room_manager_instance,
                extra_exclude=[getattr(self, 'bonus_word', None), getattr(self, 'previous_bonus_word', None)]
            )
            self.next_round_board = fb
            self.next_round_words = fw
            self.next_round_word_paths = fp
            self.next_round_bonus_cell = fc
            self.next_round_bonus = chosen_bw or fbw or ''
            self.next_round_format = ff
            self.next_round_uniqueness = fr
            self.next_round_total_words_count = len(fw)
            self.next_round_min_length = f_min_l
            if hasattr(word_validator, 'word_validator'):
                word_validator.word_validator.ensure_csw_loaded()
                self.next_round_csw_only_words = [w for w in fw if word_validator.word_validator.is_csw_only(w)]
                self.next_round_added_words = [w for w in fw if word_validator.word_validator.is_added_word(w)]
            else:
                self.next_round_csw_only_words = []
                self.next_round_added_words = []

            from scoring import calculate_word_score
            scored_dict = {}
            for w in fw:
                scored_dict[w] = calculate_word_score(
                    w, bonus_word=self.next_round_bonus, board_format=ff,
                    bonus_cell=fc, board=fb, path=_get_word_path(fp, w), return_details=True
                )
            self.next_round_word_scores = scored_dict
            self.next_round_total_points = sum((pts['total'] if isinstance(pts, dict) else pts) for pts in scored_dict.values())
            self.next_round_counts_by_len = {
                '_round': self.current_round + 1,
                '_total': len(fw),
                '_min_len': f_min_l,
                **{str(l): sum(1 for w in fw if len(w) == l) for l in range(1, 31)}
            }

            actual_wc = len(fw)
            fparams = dict(fparams) if isinstance(fparams, dict) else {}
            dict_from_p = target_params.get('dictionary') or fparams.get('dictionary', 'NWL')
            use_aw_from_p = bool(target_params.get('use_added_words', False) or fparams.get('use_added_words', False) or '+ AW' in str(dict_from_p).upper() or '+AW' in str(dict_from_p).upper())
            clean_dict = str(dict_from_p).replace('+ AW', '').replace('+AW', '').strip()
            dict_display = f"{clean_dict} + AW" if use_aw_from_p else clean_dict

            if self.time_limit >= 7200:
                wc_label = '300-400'
                _eff_fmt = 'Valued Letters'
                use_aw_from_p = False
                f_min_l = grid_floor
            elif target_range and is_board_count_valid(actual_wc, target_range):
                wc_label = target_range
            else:
                wc_label = SpinnerSet.get_factchecked_wc_range(actual_wc, use_added_words=use_aw_from_p, dictionary=dict_display)

            dims_parts = str(self.board_dimensions).lower().split('x')
            r_cnt = int(dims_parts[1] if len(dims_parts) == 3 else dims_parts[0]) if len(dims_parts) >= 2 else 4
            c_cnt = int(dims_parts[2] if len(dims_parts) == 3 else dims_parts[1]) if len(dims_parts) >= 2 else 4
            d_cnt = int(dims_parts[0]) if len(dims_parts) == 3 else 1
            raw_diff_label = _bg.get_difficulty_label(fr, rows=r_cnt, cols=c_cnt, dictionary=clean_dict, depth=d_cnt, board=fb, target_difficulty=target_params.get('difficulty') or fparams.get('difficulty'), min_word_length=f_min_l)
            if getattr(self, 'is_solo', False) and getattr(self, 'initial_solo_params', None):
                _s_d_req = self.initial_solo_params.get('difficulty')
                if _s_d_req and str(_s_d_req).lower() != 'random':
                    raw_diff_label = SpinnerSet.strip_difficulty_pct(_s_d_req)
                    if raw_diff_label == 'Easy' and fr > 0.15:
                        _bg._sanitize_uniqueness(fb, clean_dict, max_ratio=0.15, depth=d_cnt, bonus_cell=fc, min_word_length=f_min_l, use_added_words=use_aw_from_p)
                        _re_s = _bg._solve_board(fb, clean_dict, (0, 99999), f_min_l, max_depth=20, store_paths=True, timeout=5.0, bonus_cell=fc, use_added_words=use_aw_from_p)
                        self.next_round_word_paths = {w: p for w, p in _re_s.items() if len(w) >= f_min_l}
                        self.next_round_words = list(self.next_round_word_paths.keys())
                        fr = _bg._calculate_uniqueness_ratio(fb, clean_dict, d_cnt, fc, f_min_l, use_added_words=use_aw_from_p)
                        self.next_round_uniqueness = fr
            formatted_diff = SpinnerSet.format_difficulty_with_pct(raw_diff_label, fr)

            _eff_fmt = ff or target_params.get('board_format') or fparams.get('board_format', 'Normal')
            if getattr(self, 'is_solo', False) and getattr(self, 'initial_solo_params', None):
                _s_fmt_req = self.initial_solo_params.get('board_format')
                if _s_fmt_req and str(_s_fmt_req).lower() != 'random':
                    _eff_fmt = _s_fmt_req
                    self.next_round_format = _s_fmt_req

            new_sp = {
                'dictionary': dict_display,
                'difficulty': formatted_diff,
                'word_count_range': wc_label,
                'board_format': _eff_fmt,
                'min_word_length': f_min_l,
                'bonus_word_length': len(self.next_round_bonus) if self.next_round_bonus else target_bw_len,
                'use_added_words': use_aw_from_p,
                'board_dimensions': self.board_dimensions,
                'time_limit': self.time_limit,
                'uniqueness': fr,
                'generated_at': time.time()
            }
            self.next_spinner_params = dict(new_sp)
            self.next_round_spinner_params = dict(new_sp)
            self.next_round_difficulty = formatted_diff
            self.spinner_params_generated = True
            self.board_search_started = True
            self.board_search_loading = False
            self.board_search_started_actual = False
            self.solving_complete = True

            is_past_reveal = False
            if self.state == 'intermission':
                elapsed = time.time() - self.intermission_start_time
                intermission_duration = 2 if self.time_limit >= 7200 else 60
                reveal_threshold = 15.0 if intermission_duration >= 20 else 0.5
                if elapsed >= reveal_threshold:
                    is_past_reveal = True
            self.spinner_params_revealed = is_past_reveal
            return

        # If cache empty
        print(f"[GameRoom] ensure_next_board_ready: Cache empty. Using emergency fallback board.")
        source_sp = self.next_spinner_params or self.spinner_params or {}
        e_format = source_sp.get('board_format', 'Normal')
        e_dict = source_sp.get('dictionary', 'NWL')
        e_use_aw = source_sp.get('use_added_words', False) or '+ AW' in str(e_dict).upper() or '+AW' in str(e_dict).upper()
        e_wc = source_sp.get('word_count_range')
        if not e_wc:
            e_wc = SpinnerSet._spin_word_count(e_dict, 6 if '6x8' in str(self.board_dimensions) else 3, 'Medium', self.board_dimensions, use_added_words=e_use_aw)
        
        e_min_len = source_sp.get('min_word_length', 3)
        e_diff = source_sp.get('difficulty', 'Medium')
        e_results = get_emergency_fallback_board(
            self.board_dimensions, e_format, self.time_limit,
            dictionary=e_dict, use_added_words=e_use_aw, target_range=e_wc, min_word_length=e_min_len, difficulty=e_diff
        )
        
        if len(e_results) >= 9:
            e_board, e_words, e_bonus_c, e_fmt, e_paths, e_ratio, e_bonus_word, e_tr, e_params = e_results
        else:
            e_board, e_words, e_bonus_c, e_fmt, e_paths, e_ratio, e_bonus_word, e_tr = e_results
            e_params = {}

        e_params = dict(e_params) if isinstance(e_params, dict) else {}
        grid_floor = 3
        if '4x6' in self.board_dimensions: grid_floor = 4
        elif '5x7' in self.board_dimensions: grid_floor = 5
        elif '6x8' in self.board_dimensions or '3x3x3' in self.board_dimensions: grid_floor = 6
        final_e_min = max(grid_floor, int(e_params.get('min_word_length', e_min_len or grid_floor)))
        e_words = [w for w in (e_words or []) if len(w) >= final_e_min]
        if 'CSW' not in str(e_dict).upper() and hasattr(word_validator, 'word_validator'):
            word_validator.word_validator.ensure_csw_loaded()
            e_words = [w for w in e_words if not word_validator.word_validator.is_csw_only(w)]
        e_paths = {w: p for w, p in (e_paths or {}).items() if w in e_words}
        if e_words:
            actual_shortest = min(len(w) for w in e_words)
            if actual_shortest > final_e_min:
                final_e_min = actual_shortest

        target_e_bw_len = int(source_sp.get('bonus_word_length') or (len(e_bonus_word) if e_bonus_word else 8) or 8)
        e_bonus_word = pick_distinct_bonus_word(
            self, e_bonus_word, e_words, target_e_bw_len,
            dictionary=e_dict, is_checkerboard=('checkerboard' in str(e_fmt).lower()),
            room_manager=_room_manager_instance,
            extra_exclude=[getattr(self, 'bonus_word', None), getattr(self, 'previous_bonus_word', None)]
        )
            
        self.next_round_board = e_board
        self.next_round_words = e_words
        self.next_round_word_paths = e_paths
        self.next_round_total_words_count = len(e_words)
        self.next_round_bonus = e_bonus_word
        self.next_round_format = e_fmt
        self.next_round_bonus_cell = e_bonus_c
        self.next_round_uniqueness = e_ratio
        self.next_round_min_length = final_e_min
        if hasattr(word_validator, 'word_validator'):
            word_validator.word_validator.ensure_csw_loaded()
            self.next_round_csw_only_words = [w for w in e_words if word_validator.word_validator.is_csw_only(w)]
            self.next_round_added_words = [w for w in e_words if word_validator.word_validator.is_added_word(w)]
        else:
            self.next_round_csw_only_words = []
            self.next_round_added_words = []
        
        from scoring import calculate_word_score
        e_scores = {}
        for w in e_words:
            e_scores[w] = calculate_word_score(
                w, bonus_word=e_bonus_word, board_format=e_fmt,
                bonus_cell=e_bonus_c, board=e_board, path=_get_word_path(e_paths, w), return_details=True
            )
        self.next_round_word_scores = e_scores
        self.next_round_total_points = sum((pts['total'] if isinstance(pts, dict) else pts) for pts in e_scores.values())
        self.next_round_counts_by_len = {
            '_round': self.current_round + 1,
            '_total': len(e_words),
            '_min_len': final_e_min,
            **{str(l): sum(1 for w in e_words if len(w) == l) for l in range(1, 31)}
        }
        
        e_dict_val = source_sp.get('dictionary') or e_params.get('dictionary', e_dict)
        e_aw_val = bool(source_sp.get('use_added_words', False) or e_params.get('use_added_words', False) or '+ AW' in str(e_dict_val).upper() or '+AW' in str(e_dict_val).upper())
        clean_e_dict = str(e_dict_val).replace('+ AW', '').replace('+AW', '').strip()
        e_dict_display = f"{clean_e_dict} + AW" if e_aw_val else clean_e_dict
        actual_e_wc = len(e_words)
        if e_wc and is_board_count_valid(actual_e_wc, e_wc):
            e_wc_final = e_wc
        else:
            e_wc_final = SpinnerSet.get_factchecked_wc_range(actual_e_wc, use_added_words=e_aw_val, dictionary=e_dict_display)

        dims_p_e = str(self.board_dimensions).lower().split('x')
        d_cnt_e = int(dims_p_e[0]) if len(dims_p_e) == 3 else 1
        r_cnt_e = int(dims_p_e[1] if len(dims_p_e) == 3 else dims_p_e[0]) if len(dims_p_e) >= 2 else 4
        c_cnt_e = int(dims_p_e[2] if len(dims_p_e) == 3 else dims_p_e[1]) if len(dims_p_e) >= 2 else 4
        _bg_inst = _room_manager_instance.board_generator if (_room_manager_instance and hasattr(_room_manager_instance, 'board_generator')) else BoardGenerator()
        e_diff_raw = _bg_inst.get_difficulty_label(e_ratio, rows=r_cnt_e, cols=c_cnt_e, dictionary=clean_e_dict, depth=d_cnt_e, board=e_board, min_word_length=final_e_min)
        if getattr(self, 'is_solo', False) and getattr(self, 'initial_solo_params', None):
            _s_d_req = self.initial_solo_params.get('difficulty')
            if _s_d_req and str(_s_d_req).lower() != 'random':
                e_diff_raw = SpinnerSet.strip_difficulty_pct(_s_d_req)
                if e_diff_raw == 'Easy' and e_ratio > 0.15:
                    _bg_inst._sanitize_uniqueness(e_board, clean_e_dict, max_ratio=0.15, depth=d_cnt_e, bonus_cell=e_bonus_c, min_word_length=final_e_min, use_added_words=e_aw_val)
                    _re_s = _bg_inst._solve_board(e_board, clean_e_dict, (0, 99999), final_e_min, max_depth=20, store_paths=True, timeout=5.0, bonus_cell=e_bonus_c, use_added_words=e_aw_val)
                    e_paths = {w: p for w, p in _re_s.items() if len(w) >= final_e_min}
                    e_words = list(e_paths.keys())
                    e_ratio = _bg_inst._calculate_uniqueness_ratio(e_board, clean_e_dict, d_cnt_e, e_bonus_c, final_e_min, use_added_words=e_aw_val)
        e_diff_formatted = SpinnerSet.format_difficulty_with_pct(e_diff_raw, e_ratio)

        _eff_e_fmt = e_fmt or source_sp.get('board_format') or 'Normal'
        if getattr(self, 'is_solo', False) and getattr(self, 'initial_solo_params', None):
            _s_fmt_req = self.initial_solo_params.get('board_format')
            if _s_fmt_req and str(_s_fmt_req).lower() != 'random':
                _eff_e_fmt = _s_fmt_req
                self.next_round_format = _s_fmt_req

        esp = {
            'dictionary': e_dict_display,
            'difficulty': e_diff_formatted,
            'word_count_range': e_wc_final,
            'board_format': _eff_e_fmt,
            'min_word_length': final_e_min,
            'bonus_word_length': len(e_bonus_word) if e_bonus_word else target_e_bw_len,
            'use_added_words': e_aw_val,
            'board_dimensions': self.board_dimensions,
            'time_limit': self.time_limit,
            'uniqueness': e_ratio,
            'generated_at': time.time()
        }
        self.next_spinner_params = dict(esp)
        self.next_round_spinner_params = dict(esp)
        self.next_round_difficulty = e_diff_formatted
        self.spinner_params_generated = True
        self.board_search_started = True
        self.board_search_loading = False
        self.board_search_started_actual = False
        self.solving_complete = True
        
        is_past_reveal = False
        if self.state == 'intermission':
            elapsed = time.time() - self.intermission_start_time
            intermission_duration = 2 if self.time_limit >= 7200 else 60
            reveal_threshold = 15.0 if intermission_duration >= 20 else 0.5
            if elapsed >= reveal_threshold:
                is_past_reveal = True
        else:
            is_past_reveal = False
        self.spinner_params_revealed = is_past_reveal

    def check_and_update_state(self):
        """Authoritative state machine for game rooms.
        Handles transitions and timing for all game modes."""
        now = time.time()
        
        # 0. WAKE UP PAUSED ROOMS
        if self.state == 'waiting' and self.time_limit < 7200:
            all_p = [p for p in self.players + self.spectators if not getattr(p, 'is_ai', False)]
            humans = [p for p in all_p if (now - getattr(p, 'last_active', 0)) < 120.0]
            if len(humans) > 0:
                if getattr(self, '_wakeup_in_progress', False):
                    return False  # Already waking up
                self._wakeup_in_progress = True
                try:
                    global _room_manager_instance
                    if _room_manager_instance:
                        print(f"[GameRoom] Waking up paused room {self.room_id} cleanly via start_next_round.")
                        return _room_manager_instance.start_next_round(self.room_id)
                finally:
                    self._wakeup_in_progress = False

        should_end = False
        
        # 1. Round Timer Expiry Check
        if self.state == 'active':
            is_24h = (self.time_limit >= 7200)
            if not is_24h:
                if getattr(self, 'custom_end_time', 0) > 0:
                    if now >= self.custom_end_time:
                        should_end = True
                else:
                    elapsed = now - self.round_start_time
                    if elapsed >= self.time_limit - 0.4:
                        # Ensure we don't end round 0.5s after start due to uninitialized time
                        if self.round_start_time > 0:
                            should_end = True
            else:
                # 24H Reset Logic (Midnight Boundary)
                if self.round_start_time > 0:
                    import datetime
                    from zoneinfo import ZoneInfo
                    tz = DAILY_TZ
                    round_start_dt = datetime.datetime.fromtimestamp(self.round_start_time, tz)
                    now_dt = datetime.datetime.fromtimestamp(now, tz)
                    if now > self.round_start_time and now_dt.date() > round_start_dt.date():
                        print(f"[GameRoom] Midnight Reset DETECTED for {self.room_id}")
                        should_end = True
                        self.midnight_reset_occurred = True
                        self.previous_board = [list(row) for row in self.board] if self.board else None
                        self.previous_min_length = getattr(self, 'current_min_length', 3)
                        self.previous_all_words = list(self.all_words) if self.all_words else []
                        self.previous_all_word_scores = getattr(self, 'solved_words_with_scores', {}) if self.solved_words_with_scores else {w: {} for w in self.all_words} # Dict for scoring fallback
                        self.previous_csw_only_words = list(self.csw_only_words) if self.csw_only_words else []
                        self.previous_added_words = list(self.added_words) if self.added_words else []
                        self.previous_bonus_word = getattr(self, 'bonus_word', '')
                        
                        # PERSISTENCE: Snapshot current player findings for the "Previous Day" tab
                        self.previous_day_history = {
                            str(p.user_id): {
                                'username': p.username,
                                'found_words': [w['word'] for w in p.submitted_words]
                            } for p in (self.players or [])
                        }
                        
                        # EVICTION RECORD: Mark all users in 24h room to receive daily_reset notice
                        if not hasattr(self, 'evicted_users') or self.evicted_users is None:
                            self.evicted_users = {}
                        for p in (self.players or []):
                            self.evicted_users[str(p.user_id)] = 'daily_reset'
                        for s in (self.spectators or []):
                            self.evicted_users[str(s.user_id)] = 'daily_reset'
                        
                        # Capture intermission stats
                        self.previous_total_words = getattr(self, 'total_words_count', 0)
                        self.previous_total_points = getattr(self, 'total_points_count', 0)
                        
                        # Direct room resets instead of invalid method call
                        self.custom_end_time = 0
                        self.solving_complete = False
                        self.complete_words = []
                        self.midnight_reset_occurred = True

        # 2. Transition ACTIVE -> INTERMISSION
        if self.state == 'active' and should_end:
            with self._state_lock:
                if self.state != 'active':
                    return True
                
                self.state = 'intermission'
                self.intermission_start_time = now
                self.spinner_params_revealed = False
                self.was_revealed_this_intermission = False
                self._spinner_params_locked = False
                self.frozen_revealed_params = None
                self._did_050_fallback_rescue = False
                self._fallback_at_50s_done = False
                self.easy_fallback_applied = False
                print(f"[TRANSITION] Room {self.room_id}: ACTIVE -> INTERMISSION (Time: {self.intermission_start_time}, Elapsed: {now - self.round_start_time})")
                
                # [PROACTIVE] Do NOT clear generated/search flags here anymore.
                # They are now reset only at start_next_round.
                # Snapshot round_quitters BEFORE clearing so the async thread can use them
                quitters_snapshot = list(self.round_quitters)
                self.round_quitters = []
                self.custom_end_time = 0 # CLEAR ALWAYS AT TRANSITION
                
                # USER REQUEST: Absolute accuracy for 'All Words' panel scoring.
                # If background scoring isn't finished, perform a synchronous fallback score calculation.
                # OPTIMIZATION: Calculate fast length-based scores instantly to avoid transition delay, and refine in background thread!
                if not getattr(self, 'solved_words_with_scores', None) or not self.solved_words_with_scores:
                    print(f"[GameRoom] Transitioning {self.room_id}: computing missing solved_words_with_scores.")
                    from scoring import calculate_word_score
                    if getattr(self, 'game_type', None) == 'subanagrams':
                        bw = getattr(self, 'bonus_word', None)
                        self.solved_words_with_scores = {
                            w: calculate_word_score(w, bw, board_format='Normal', return_details=True)
                            for w in (self.all_words or [])
                        }
                    else:
                        fast_scores = {}
                        eval_board = self.board or getattr(self, 'previous_board', None)
                        eval_fmt = self.current_board_format or getattr(self, 'previous_board_format', 'Normal')
                        for word in (self.all_words or []):
                            path_v = _get_word_path(self.all_words_paths, word)
                            fast_scores[word] = calculate_word_score(
                                word, 
                                self.bonus_word, 
                                board_format=eval_fmt,
                                bonus_cell=self.bonus_cell,
                                board=eval_board,
                                path=path_v,
                                return_details=True,
                                strict_path=True
                            )
                        self.solved_words_with_scores = fast_scores

                # Snapshot board and words for intermission (Detailed Scoring Preservation)
                if self.game_type == '3d' or (self.board and len(self.board) == 6 and isinstance(self.board[0], list) and isinstance(self.board[0][0], list)):
                     self.previous_board = [[list(row) for row in face] for face in self.board]
                else:
                     self.previous_board = [list(row) for row in self.board] if self.board else None
                
                # USER REQUEST: Ensure 'All Words' list has full math breakdown in history
                self.previous_all_words = list(self.all_words) if self.all_words else []
                self.previous_all_word_scores = dict(getattr(self, 'solved_words_with_scores', {})) if getattr(self, 'solved_words_with_scores', None) else {}
                self.previous_min_length = getattr(self, 'current_min_length', 3)
                if self.game_type == 'subanagrams':
                    self.previous_sequence_length = getattr(self, 'sequence_length', None) or (len(self.board[0]) if (self.board and len(self.board) > 0 and isinstance(self.board[0], list)) else None)
                # BUGFIX: Snapshot paths, bonus_cell, and board_format so word-highlight clicks
                # during intermission use the COMPLETED round's board, not the next board.
                self.previous_all_words_paths = dict(self.all_words_paths) if isinstance(self.all_words_paths, dict) else {}
                self.previous_bonus_cell = getattr(self, 'bonus_cell', None)
                self.previous_board_format = getattr(self, 'current_board_format', 'Normal')
                if hasattr(word_validator, 'word_validator'):
                    word_validator.word_validator.ensure_csw_loaded()
                    self.previous_csw_only_words = [w for w in (self.previous_all_words or []) if word_validator.word_validator.is_csw_only(w)]
                    self.previous_added_words = [w for w in (self.previous_all_words or []) if word_validator.word_validator.is_added_word(w)]
                else:
                    self.previous_csw_only_words = list(getattr(self, 'csw_only_words', []) or [])
                    self.previous_added_words = list(getattr(self, 'added_words', []) or [])
                self.previous_bonus_word = self.bonus_word
                self.previous_dictionary = getattr(self, 'current_dictionary', 'NWL')
                self.previous_use_added_words = getattr(self, 'use_added_words', False)
                
                # Snapshot for persistence
                self.update_counts_by_len()
                self.recalculate_total_points() # Authoritative sync before snapshot
                self.previous_total_points = getattr(self, 'total_points_count', 0)
                self.previous_total_words = getattr(self, 'total_words_count', 0)
                self.previous_total_counts_by_len = dict(getattr(self, 'total_counts_by_len', {}))
                
                # Capture snapshot of players who participated in the completed round
                intermission_player_snapshots = []
                all_candidate_players = list(self.players)
                existing_uids = {p.user_id for p in self.players}
                for p in self.past_players.values():
                    if p.user_id not in existing_uids:
                        all_candidate_players.append(p)
                
                if self.game_type == 'split':
                    self.calculate_split_scores()

                try:
                    self.update_live_pe()
                except Exception as _pe_err:
                    print(f"[GameRoom] Error updating live PE on intermission: {_pe_err}")

                for p in all_candidate_players:
                    if (p.is_registered or p.is_guest) and (p.score > 0 or p.submitted_words or p.invalid_words):
                        intermission_player_snapshots.append({
                            'user_id': p.user_id,
                            'username': p.username,
                            'score': p.score,
                            'submitted_words': [dict(w) for w in p.submitted_words],
                            'invalid_words': list(p.invalid_words),
                            'rating': getattr(p, 'rating', 1200),
                            'performance_efficiency': getattr(p, 'performance_efficiency', 0),
                            'joined_mid_round': getattr(p, 'joined_mid_round', False)
                        })
                self.last_round_human_participants = len([p for p in all_candidate_players if not getattr(p, 'is_ai', False)])
                
                # Authoritative round placements for chat and results (participants sorted by score desc, rating desc)
                sorted_snapshots = sorted(
                    intermission_player_snapshots,
                    key=lambda x: (x.get('score', 0), x.get('rating', 1200)),
                    reverse=True
                )
                self.last_round_placements = {
                    p['username'].lower(): idx + 1 for idx, p in enumerate(sorted_snapshots)
                }

                # Winners History (Calculated synchronously so results/winners are instantly available)
                active_pool = [p for p in self.players if getattr(p, 'score', 0) > 0 or (getattr(p, 'submitted_words', []) and any(w.get('points', 0) > 0 for w in p.submitted_words))]
                max_score = max([p.score for p in active_pool]) if active_pool else 0
                
                winners_data = []
                winner_words = []
                if active_pool:
                    winners_data = [{'username': p.username, 'rating': p.rating} for p in active_pool if p.score == max_score]
                    for p in active_pool:
                        if p.score == max_score:
                            # Store all words in chronological order with timestamps so the
                            # replay can position them correctly in the timeline.
                            sorted_by_time = sorted(p.submitted_words, key=lambda x: x.get('time', 0))
                            winner_words = [
                                {'word': w['word'], 'points': w.get('points', 0), 'time': w.get('time', 0)}
                                for w in sorted_by_time
                            ]
                            break
                
                if max_score > 0:
                    # Ensure exactly one entry per round number by removing any prior entry for self.current_round
                    self.winners_history = [h for h in self.winners_history if h.get('round') != self.current_round]
                    self.winners_history.insert(0, {
                        'round': self.current_round,
                        'game_type': getattr(self, 'game_type', 'accumulative'),
                        'winners': winners_data,
                        'all_players': sorted([{'username': p.username, 'score': p.score, 'rating': getattr(p, 'rating', 1200)} for p in (active_pool or self.players)], key=lambda x: (1 if (x['score'] > 0 or any(getattr(p, 'submitted_words', []) or getattr(p, 'invalid_words', []) for p in (active_pool or self.players) if p.username == x['username'])) else 0, x['score'], x['rating']), reverse=True),
                        'score': max_score,
                        'board': [list(row) for row in self.board] if self.board else [],
                        'words': winner_words,
                        'all_words': sorted(list(getattr(self, 'all_words', []))) if getattr(self, 'all_words', None) else [],
                        'bonus_word': getattr(self, 'bonus_word', ''),
                        'timestamp': int(time.time() * 1000),
                        'round_duration': self.time_limit,
                        'round_start_time': self.round_start_time,
                    })
                    if len(self.winners_history) > 25: self.winners_history = self.winners_history[:25]
                    self.last_round_had_winner = True
                else:
                    # If no one entered words or scored > 0, ensure no winner record exists for this round
                    self.winners_history = [h for h in self.winners_history if h.get('round') != self.current_round]
                    self.last_round_had_winner = False

                # SYNCHRONOUS RATINGS CALCULATION:
                # Calculate and apply rating changes immediately so the change (e.g. "+7") is visible
                # without delay on the very first state response as soon as the round ends.
                try:
                    is_24h = (self.time_limit >= 7200)
                    if is_24h or self.game_type == 'subanagrams':
                        for p in self.players + quitters_snapshot:
                            p.rating_change = 0
                        print(f"[GameRoom] {self.game_type} room: skipping rating updates.")
                    else:
                        from rating_logic import calculate_proportional_rating_change
                        # USER MANDATE: Only affect ratings of people who played from the beginning of the round and had a score >= 1
                        round_starters = getattr(self, 'round_starting_player_ids', None)
                        participants = [
                            p for p in self.players + quitters_snapshot 
                            if getattr(p, 'score', 0) >= 1
                            and not getattr(p, 'is_ai', False)
                            and not is_player_guest(p)
                            and not getattr(p, 'joined_mid_round', False)
                            and (not round_starters or str(p.user_id) in round_starters)
                        ]
                        rating_changes = calculate_proportional_rating_change(participants, is_private=self.is_private, board_format=self.current_board_format)
                        
                        display_game_type = self.game_type.replace('solo_', '')
                        config_key = f"{display_game_type}|{self.board_dimensions}|{self.time_limit}"
                        active_humans = [
                            p for p in participants
                            if getattr(p, 'score', 0) >= 1
                            and not getattr(p, 'is_ai', False) and not getattr(p, 'is_guest', False) and not is_player_guest(p)
                        ]

                        def _is_established_opponent(o):
                            if getattr(o, 'is_unrated', False) or (0 < getattr(o, 'provisional_games', 0) < 10 and getattr(o, 'rating', 1200) == 1200):
                                        return False
                            o_rating = getattr(o, 'rating', 1200)
                            if o_rating == 1200:
                                o_cfg_games = getattr(o, 'config_games_played', 0) or 0
                                o_prov_games = getattr(o, 'provisional_games', 0) or 0
                                if o_cfg_games <= 10 and o_prov_games < 10:
                                    return False
                            return True

                        with get_db() as conn_p:
                            for p in self.players + quitters_snapshot:
                                # Strict User Mandate: Only affect ratings of people who played from the beginning of the round and had a score >= 1
                                played_from_start = (
                                    not getattr(p, 'joined_mid_round', False) and
                                    (not round_starters or str(p.user_id) in round_starters)
                                )
                                has_score = (getattr(p, 'score', 0) >= 1)
                                is_eligible_human = (not getattr(p, 'is_ai', False) and not getattr(p, 'is_guest', False) and not is_player_guest(p))

                                if not (played_from_start and has_score and is_eligible_human):
                                    p.rating_change = 0
                                    continue

                                if p.user_id in rating_changes:
                                    simulated_change = rating_changes[p.user_id]
                                    is_p_unrated = getattr(p, 'is_unrated', False) or (
                                        0 < getattr(p, 'provisional_games', 0) < 10 and getattr(p, 'rating', 1200) == 1200
                                    )
                                    if is_p_unrated:
                                        # 10-Game Unrated Placement:
                                        # User starts unrated and rating remains unchanged for first 10 games (with white square).
                                        # Only rounds played alongside at least one established rated opponent count toward placement.
                                        p.is_unrated = True
                                        p.rating_change = 0
                                        opponents = [
                                            o for o in active_humans
                                            if o.user_id != p.user_id and _is_established_opponent(o)
                                        ] if p in active_humans else []
                                        # User Mandate: Don't count a user's play if nobody is in the room as well as if no rating-established player playing along with them.
                                        if len(active_humans) >= 2 and opponents:
                                            opp_score = sum(getattr(o, 'score', 0) for o in opponents)
                                            opp_rating = sum(float(getattr(o, 'rating', 1200)) for o in opponents)
                                            user_score = getattr(p, 'score', 0)

                                            if opp_score > 0 and user_score > 0:
                                                r_round = opp_rating * (float(user_score) / float(opp_score))
                                            elif opp_score == 0 and user_score > 0:
                                                r_round = (opp_rating / len(opponents)) * 1.5
                                            elif user_score == 0:
                                                r_round = 400.0
                                            else:
                                                r_round = 1200.0

                                            r_round = max(1.0, min(3000.0, r_round))
                                            p.provisional_games = getattr(p, 'provisional_games', 0) + 1
                                            p.provisional_ratings_sum = getattr(p, 'provisional_ratings_sum', 0.0) + r_round
                                            p.provisional_rating = p.provisional_ratings_sum / p.provisional_games
                                            p.config_games_played = (getattr(p, 'config_games_played', 0) or 0) + 1

                                            print(f"[RatingPlacement] {p.username} placement game {p.provisional_games}/10: round_equilibrium={r_round:.1f}, running_avg={p.provisional_rating:.1f}")

                                            if p.provisional_games >= 10:
                                                # 10 games completed: establish approximate rating
                                                raw_approx = int(round(p.provisional_ratings_sum / 10.0))
                                                # User mandate: In the event the user cheats, make highest rating establishment 2300.
                                                established_rating = max(1, min(2300, raw_approx))
                                                p.rating = established_rating
                                                p.is_unrated = False

                                                conn_p.execute('''
                                                    INSERT INTO user_ratings (user_id, config_key, rating, is_unrated, provisional_games, provisional_ratings_sum, provisional_rating, games_played)
                                                    VALUES (?, ?, ?, 0, 10, ?, ?, 1)
                                                    ON CONFLICT(user_id, config_key) DO UPDATE SET
                                                        rating = excluded.rating,
                                                        is_unrated = 0,
                                                        provisional_games = 10,
                                                        provisional_ratings_sum = excluded.provisional_ratings_sum,
                                                        provisional_rating = excluded.provisional_rating,
                                                        games_played = COALESCE(user_ratings.games_played, 0) + 1
                                                ''', (p.user_id, config_key, established_rating, p.provisional_ratings_sum, established_rating))

                                                conn_p.execute('UPDATE users SET rating = MAX(400, ?) WHERE id = ?', (established_rating, p.user_id))
                                                print(f"[RatingPlacement] 🎯 Established rating for {p.username} in {config_key}: {established_rating} (raw was {raw_approx})")
                                            else:
                                                conn_p.execute('''
                                                    INSERT INTO user_ratings (user_id, config_key, rating, is_unrated, provisional_games, provisional_ratings_sum, provisional_rating, games_played)
                                                    VALUES (?, ?, 1200, 1, ?, ?, ?, 1)
                                                    ON CONFLICT(user_id, config_key) DO UPDATE SET
                                                        is_unrated = 1,
                                                        provisional_games = excluded.provisional_games,
                                                        provisional_ratings_sum = excluded.provisional_ratings_sum,
                                                        provisional_rating = excluded.provisional_rating,
                                                        games_played = COALESCE(user_ratings.games_played, 0) + 1
                                                ''', (p.user_id, config_key, p.provisional_games, p.provisional_ratings_sum, p.provisional_rating))
                                        else:
                                            print(f"[RatingPlacement] {p.username} round skipped for placement in {config_key} (nobody in room or no established rated opponents).")
                                    else:
                                        p.rating_change = simulated_change
                                        p.rating += p.rating_change

                                        # User Mandate: Don't count a user's play if nobody is in the room as well as if no rating-established player playing along with them.
                                        has_established_opponent = any(
                                            o.user_id != p.user_id and _is_established_opponent(o)
                                            for o in active_humans
                                        ) if p in active_humans else False

                                        if len(active_humans) >= 2 and has_established_opponent:
                                            p.config_games_played = (getattr(p, 'config_games_played', 0) or 0) + 1
                                            # Update Global Rating
                                            if p.rating_change != 0:
                                                conn_p.execute('UPDATE users SET rating = MAX(400, rating + ?) WHERE id = ?', (p.rating_change, p.user_id))
                                            
                                            # Update Config-Specific Rating
                                            conn_p.execute('''
                                                INSERT INTO user_ratings (user_id, config_key, rating, is_unrated, games_played)
                                                VALUES (?, ?, MAX(400, 1200 + ?), 0, 1)
                                                ON CONFLICT(user_id, config_key) DO UPDATE SET 
                                                    rating = MAX(400, rating + ?),
                                                    games_played = COALESCE(user_ratings.games_played, 0) + 1
                                            ''', (p.user_id, config_key, p.rating_change, p.rating_change))
                                        else:
                                            p.rating_change = 0
                                else:
                                    p.rating_change = 0

                            # 5. Distribute Abandonment Bounty (User Request: At the end when results are shown)
                            if self.abandonment_bounty > 0:
                                eligible_receivers = [
                                    p for p in self.players 
                                    if not p.is_ai and not getattr(p, 'is_guest', False) and not is_player_guest(p)
                                    and not getattr(p, 'joined_mid_round', False)
                                    and (not round_starters or str(p.user_id) in round_starters)
                                    and getattr(p, 'score', 0) >= 1
                                    and not getattr(p, 'is_unrated', False)
                                ]
                                if eligible_receivers:
                                    count = len(eligible_receivers)
                                    share = self.abandonment_bounty // count
                                    remainder = self.abandonment_bounty % count

                                    # Determine the per-format rating cap so bounty never busts the ceiling
                                    fmt = (self.current_board_format or '').lower()
                                    if 'triple' in fmt:
                                        rating_cap = 48
                                    elif 'double' in fmt:
                                        rating_cap = 32
                                    else:
                                        rating_cap = 16

                                    config_key = f"{self.game_type.replace('solo_', '')}|{self.board_dimensions}|{self.time_limit}"

                                    for i, target in enumerate(eligible_receivers):
                                        bonus = share + (1 if i < remainder else 0)
                                        if bonus <= 0: continue

                                        # Cap: only give as much bonus as keeps total_change <= rating_cap
                                        current_change = getattr(target, 'rating_change', 0)
                                        headroom = max(0, rating_cap - current_change)
                                        bonus = min(bonus, headroom)
                                        if bonus <= 0:
                                            with open(RATING_AUDIT_PATH, 'a') as log:
                                                log.write(f"[{time.time()}] Round-End Bounty SKIPPED for {target.username}: already at cap ({current_change}/{rating_cap})\n")
                                            continue

                                        # Apply to DB
                                        conn_p.execute('UPDATE users SET rating = rating + ? WHERE id = ?', (bonus, target.user_id))
                                        conn_p.execute('''
                                            INSERT INTO user_ratings (user_id, config_key, rating)
                                            VALUES (?, ?, 1200 + ?)
                                            ON CONFLICT(user_id, config_key) DO UPDATE SET rating = rating + ?
                                        ''', (target.user_id, config_key, bonus, bonus))

                                        # Apply in-memory
                                        target.rating += bonus
                                        if not hasattr(target, 'rating_change'): target.rating_change = 0
                                        target.rating_change = getattr(target, 'rating_change', 0) + bonus

                                        if not hasattr(target, 'bonus_notices'): target.bonus_notices = []
                                        target.bonus_notices.append(f"Received +{bonus} from round abandonment pool")

                                        with open(RATING_AUDIT_PATH, 'a') as log:
                                            log.write(f"[{time.time()}] Round-End Bounty Payout: +{bonus} to {target.username} (Room: {self.room_id}, Pool: {self.abandonment_bounty}, Cap: {rating_cap})\n")

                                    # Reset pool AFTER successful distribution
                                    self.abandonment_bounty = 0
                except Exception as e:
                    print(f"[GameRoom] Rating error: {e}")

                # Store data for results screen and RESET mid-round flag for NEXT round
                for p in self.players:
                    p.previous_round_score = p.score
                    p.previous_submitted_words = list(p.submitted_words)
                    p.joined_mid_round = False

                if active_pool and max_score > 0:
                    winner_names = [p.username for p in active_pool if p.score == max_score]
                    msg = f"🏆 {winner_names[0]} won!" if len(winner_names) == 1 else f"🏆 Tie!"
                    self.add_chat_message("SYSTEM", msg, is_system=True, color='#ffd700', is_winner=True)

                # Asynchronous Post-Round Processing (Heavy persistence operations)
                def process_results_async():
                    try:
                        # Save history and word tally immediately at the start of intermission
                        global _room_manager_instance
                        rm = _room_manager_instance
                        if not rm:
                            try:
                                import app
                                rm = getattr(app, 'room_manager', None)
                            except Exception:
                                pass
                        if rm and self.game_type != 'subanagrams' and not self.is_solo:
                            try:
                                rm.save_round_history(
                                    self,
                                    board=[list(row) for row in self.board] if self.board else None,
                                    all_words=list(self.complete_words) if (getattr(self, 'complete_words', None) and len(self.complete_words) > 0) else list(self.all_words),
                                    bonus_word=(self.bonus_word.upper() if self.bonus_word else None),
                                    player_snapshots=intermission_player_snapshots,
                                    round_num=self.current_round,
                                    all_words_paths=dict(getattr(self, 'all_words_paths', {})),
                                    round_start_time=self.round_start_time,
                                    board_format=self.current_board_format
                                )
                                
                                # Log word tally (strictly valid words with positive score only)
                                player_words = {
                                    p['username']: [
                                        w['word'] for w in p['submitted_words'] 
                                        if (w.get('points', 0) if isinstance(w, dict) else 0) > 0
                                    ] for p in intermission_player_snapshots
                                }
                                rm.log_word_tally(self, player_words, round_num=self.current_round, round_start_time=self.round_start_time)
                            except Exception as db_save_err:
                                print(f"[GameRoom] Error in immediate database save: {db_save_err}")

                    except Exception as e:
                        print(f"[GameRoom] Post-round error: {e}")

                threading.Thread(target=process_results_async, daemon=True).start()
                return True

        # 3. Intermission state check (milestones handled by RoomManager)
        if self.state == 'intermission':
            return True

        return False

    def trigger_early_fallback_rescue(self):
        """USER REQUEST: If no board is ready by 0:50 remaining time during intermission,
        grab a pregenerated board stored at random, assign its parameters, and stage it immediately."""
        now = time.time()
        import random
        
        # Pop compatible board from cache if available
        from board_generator import pop_compatible_cached_board
        sp = getattr(self, 'frozen_revealed_params', None) or self.next_spinner_params or self.spinner_params or {}
        dict_val = sp.get('dictionary', 'NWL')
        fmt_val = sp.get('board_format', 'Normal')
        min_l_val = sp.get('min_word_length', 3)
        use_aw_val = sp.get('use_added_words', False) or '+ AW' in str(dict_val).upper() or '+AW' in str(dict_val).upper()
        bonus_word_len = sp.get('bonus_word_length')
        sp_wc_range = sp.get('word_count_range')
        sp_diff = sp.get('difficulty')

        if getattr(self, 'is_solo', False) and getattr(self, 'initial_solo_params', None):
            if self.initial_solo_params.get('difficulty') and str(self.initial_solo_params.get('difficulty')).lower() != 'random':
                sp_diff = self.initial_solo_params['difficulty']
            if self.initial_solo_params.get('board_format') and str(self.initial_solo_params.get('board_format')).lower() != 'random':
                fmt_val = self.initial_solo_params['board_format']
            if self.initial_solo_params.get('min_word_length') and str(self.initial_solo_params.get('min_word_length')).lower() != 'random':
                min_l_val = self.initial_solo_params['min_word_length']
            if self.initial_solo_params.get('dictionary') and str(self.initial_solo_params.get('dictionary')).lower() != 'random':
                dict_val = self.initial_solo_params['dictionary']
            if 'use_added_words' in self.initial_solo_params:
                use_aw_val = self.initial_solo_params['use_added_words']

        fallback = pop_compatible_cached_board(
            self.board_dimensions,
            dict_val,
            fmt_val,
            min_l_val,
            use_aw_val,
            bonus_word_len=bonus_word_len,
            word_count_range=sp_wc_range,
            difficulty=sp_diff
        )
        
        if fallback:
            fb, fw, fc, ff, fp, fr, fbw, fparams = fallback
            print(f"[Fallback-Rescue] Early rescue: Popped board from cache with {len(fw)} words!")
        else:
            print(f"[Fallback-Rescue] Early rescue: Cache empty. Using emergency fallback board.")
            # Get emergency fallback board matching current spinner params
            e_format = fmt_val
            e_dict = dict_val
            e_use_aw = use_aw_val
            e_wc = sp_wc_range
            if not e_wc:
                e_wc = '100-200'
            e_min_len = min_l_val
            e_diff = sp_diff or 'Medium'
            
            fallback = get_emergency_fallback_board(
                self.board_dimensions, e_format, self.time_limit,
                dictionary=e_dict, use_added_words=e_use_aw, target_range=e_wc, min_word_length=e_min_len, difficulty=e_diff
            )
            if len(fallback) >= 9:
                fb, fw, fc, ff, fp, fr, fbw, _, fparams = fallback
            else:
                fb, fw, fc, ff, fp, fr, fbw, _ = fallback
                fparams = {}

        # Enforce Easy sanitization if requested for Solo
        if getattr(self, 'is_solo', False) and getattr(self, 'initial_solo_params', None):
            _s_d_req = self.initial_solo_params.get('difficulty')
            if _s_d_req and SpinnerSet.strip_difficulty_pct(_s_d_req) == 'Easy' and fr > 0.15:
                _bg_inst = _room_manager_instance.board_generator if (_room_manager_instance and hasattr(_room_manager_instance, 'board_generator')) else BoardGenerator()
                dims_p = str(self.board_dimensions).lower().split('x')
                d_p = int(dims_p[0]) if len(dims_p) == 3 else 1
                _bg_inst._sanitize_uniqueness(fb, dict_val, max_ratio=0.15, depth=d_p, bonus_cell=fc, min_word_length=min_l_val, use_added_words=use_aw_val)
                _re_s = _bg_inst._solve_board(fb, dict_val, (0, 99999), min_l_val, max_depth=20, store_paths=True, timeout=5.0, bonus_cell=fc, use_added_words=use_aw_val)
                fp = {w: p for w, p in _re_s.items() if len(w) >= min_l_val}
                fw = list(fp.keys())
                fr = _bg_inst._calculate_uniqueness_ratio(fb, dict_val, d_p, fc, min_l_val, use_added_words=use_aw_val)

        # 2. Sync spinner params — strictly prioritize the room's spun dictionary
        fparams = dict(fparams) if fparams else {}
        dict_val = sp.get('dictionary') or fparams.get('dictionary', 'NWL')
        use_aw_val = sp.get('use_added_words', False) or ('+ AW' in str(dict_val).upper()) or fparams.get('use_added_words', False)
        clean_d = str(dict_val).replace('+ AW', '').replace('+AW', '').strip()
        if use_aw_val:
            dict_val = f"{clean_d} + AW"
        else:
            dict_val = clean_d

        declared_target_wc = None
        if getattr(self, 'frozen_revealed_params', None) and isinstance(self.frozen_revealed_params, dict):
            declared_target_wc = self.frozen_revealed_params.get('word_count_range')
        if not declared_target_wc and sp:
            declared_target_wc = sp.get('word_count_range')

        max_cap = None
        if declared_target_wc == '50-100': max_cap = 99
        elif declared_target_wc == '100-200': max_cap = 199
        elif declared_target_wc == '200-300': max_cap = 299
        elif declared_target_wc == '300-400': max_cap = 399
        elif declared_target_wc == '400-500': max_cap = 499

        max_cap = None  # PREFACE FIX (#13/#14): never truncate valid board words
        if max_cap and len(fw) > max_cap:
            sampled_set = proportionally_sample_words(fw, max_cap)
            if fbw and fbw in fw and fbw not in sampled_set:
                sampled_set.add(fbw)
            fw = [w for w in fw if w in sampled_set]
            fp = {w: p for w, p in fp.items() if w in sampled_set}

        actual_wc = len(fw)
        if declared_target_wc and is_board_count_valid(actual_wc, declared_target_wc):
            wc_label = declared_target_wc
        else:
            wc_label = SpinnerSet.get_factchecked_wc_range(actual_wc, use_added_words=use_aw_val, dictionary=dict_val)

        new_sp = {
            'dictionary': sp.get('dictionary') or dict_val,
            'difficulty': sp.get('difficulty') or fparams.get('difficulty', 'Medium'),
            'word_count_range': wc_label,
            'board_format': sp.get('board_format') or ff or 'Normal',
            'min_word_length': sp.get('min_word_length') or fparams.get('min_word_length', 3),
            'bonus_word_length': sp.get('bonus_word_length') or (len(fbw) if fbw else (fparams.get('bonus_word_len') or 6)),
            'use_added_words': sp.get('use_added_words', use_aw_val),
            'board_dimensions': self.board_dimensions,
            'time_limit': self.time_limit,
            'generated_at': now,
            'uniqueness': fr
        }
        
        # Enforce sanitization
        new_sp = SpinnerSet.sanitize_params(new_sp, self.board_dimensions, self.time_limit >= 7200)

        if getattr(self, 'is_solo', False) and getattr(self, 'initial_solo_params', None):
            if self.initial_solo_params.get('difficulty') and str(self.initial_solo_params.get('difficulty')).lower() != 'random':
                new_sp['difficulty'] = SpinnerSet.format_difficulty_with_pct(SpinnerSet.strip_difficulty_pct(self.initial_solo_params['difficulty']), fr)
            if self.initial_solo_params.get('board_format') and str(self.initial_solo_params.get('board_format')).lower() != 'random':
                new_sp['board_format'] = self.initial_solo_params['board_format']

        # Update next spinner params for the upcoming round
        self.next_spinner_params = new_sp
        self.next_round_spinner_params = new_sp
        if not getattr(self, '_spinner_params_locked', False) and self.state != 'active':
            self.spinner_params = new_sp
            if getattr(self, 'frozen_revealed_params', None) and isinstance(self.frozen_revealed_params, dict):
                self.frozen_revealed_params['word_count_range'] = new_sp['word_count_range']
                self.frozen_revealed_params['bonus_word_length'] = new_sp['bonus_word_length']
        self.spinner_params_generated = True
        self.spinner_params_revealed = True
        self._reveal_sync_complete = True

        # Stage for the next round transition
        self.next_round_board = fb
        self.next_round_words = fw
        self.next_round_word_paths = fp
        self.next_round_bonus_cell = fc
        self.next_round_bonus = fbw
        self.next_round_format = ff
        self.next_round_uniqueness = fr
        self.next_round_total_words_count = len(fw)
        if hasattr(word_validator, 'word_validator'):
            word_validator.word_validator.ensure_csw_loaded()
            self.next_round_csw_only_words = [w for w in fw if word_validator.word_validator.is_csw_only(w)]
            self.next_round_added_words = [w for w in fw if word_validator.word_validator.is_added_word(w)]
        else:
            self.next_round_csw_only_words = []
            self.next_round_added_words = []

        # Pre-calculate next scores
        is_valued = ('valued' in str(ff).lower())
        scored_dict = {}
        for w in fw:
            if is_valued:
                scored_dict[w] = {'total': get_valued_word_score(w), 'base': get_valued_word_score(w)}
            else:
                length = len(w)
                s = 0
                if length <= 2: s = 0
                elif length <= 4: s = 1
                elif length == 5: s = 2
                elif length == 6: s = 3
                elif length == 7: s = 5
                elif length == 8: s = 9
                elif length == 9: s = 13
                elif length == 10: s = 17
                elif length == 11: s = 21
                elif length == 12: s = 25
                elif length == 13: s = 29
                elif length == 14: s = 33
                elif length >= 15: s = 37
                scored_dict[w] = {'total': s, 'base': s}
        self.next_round_word_scores = scored_dict
        self.next_round_total_points = sum(pts['total'] for pts in scored_dict.values())

        # Pre-calculate counts by length
        next_counts = {i: 0 for i in range(1, 31)}
        display_min = new_sp.get('min_word_length', 3)
        for w in fw:
            l = len(w)
            if display_min <= l <= 30:
                next_counts[l] += 1
        next_counts['_round'] = self.current_round + 1
        self.next_round_counts_by_len = next_counts

        # Reset search states
        self.board_search_started = True
        self.board_search_loading = False
        self.board_search_started_actual = False
        self.solving_complete = True

    def calculate_split_scores(self):
        """
        Calculate scores for Split Points mode.
        - Unique word: Full points
        - Shared word: Points split among finders.
        - Remainders given to earlier finders to ensure point pool is constant (Gain == Total Loss).
        """
        print(f"[GameRoom] Calculating Split Points for room {self.room_id}")
        
        # 1. Group players (and their word objects) by word
        word_finders = {} # {word: [(player, time, w_obj), ...]}
        
        for p in self.players:
            for w_obj in p.submitted_words:
                w = w_obj['word']
                if w not in word_finders:
                    word_finders[w] = []
                word_finders[w].append((p, w_obj.get('time', 0), w_obj))
                
            # 2. For each word, distribute points fairly
        for word, finders in word_finders.items():
            # Sort finders by submission time for consistency
            finders.sort(key=lambda x: x[1])
            
            count = len(finders)
            
            # PENALTY CHECK: If this word is a penalty word, it preserves its -3 value
            # Note: All finders of an invalid word get -3 (or split? User suggests "entry" implies per-entry)
            # Default: Current engine sets points: -3 at entry.
            is_any_penalty = any(f[2].get('is_penalty') for f in finders)
            
            if is_any_penalty:
                # Penalty words are NOT split or recalculated; they are strictly -3 for every finder
                for player, timestamp, w_obj in finders:
                    w_obj['points'] = -3
                continue

            # Standard Split Logic
            # OPTIMIZATION: Use path from the first finder's submission to skip redundant DFS search
            cached_path = finders[0][2].get('path') if finders else None
            
            res = calculate_word_score(
                word, 
                self.bonus_word, 
                board_format=self.current_board_format, 
                path=cached_path, 
                bonus_cell=self.bonus_cell, 
                board=self.board, 
                return_details=True,
                is_private=self.is_private,
                strict_path=True
            )
            base_points = res['total']
            
            # Divide points EQUALLY (User Request: "ensure they both get the same point value")
            # Strategy: Round Up (User left choice to me)
            # Formula: ceil(a/b) = (a + b - 1) // b
            final_points = (base_points + count - 1) // count
            
            # SPLIT THE DETAILS for frontend breakdown accuracy
            # Scaling original values by 1/count (rounded up)
            split_details = {
                'total': final_points,
                'base': (res.get('base', 0) + count - 1) // count,
                'bonus_word_points': (res.get('bonus_word_points', 0) + count - 1) // count,
                'bonus_letter_points': (res.get('bonus_letter_points', 0) + count - 1) // count,
                'either_or_points': (res.get('either_or_points', 0) + count - 1) // count
            }
            
            for i, (player, timestamp, w_obj) in enumerate(finders):
                # No remainder distribution - everyone gets the same rounded-up value
                
                # Update word object with split metadata for frontend
                
                # Update word object with split metadata for frontend
                w_obj['split_points'] = final_points
                w_obj['shared_count'] = count
                w_obj['is_unique'] = (count == 1)
                w_obj['points'] = final_points
                w_obj['base_points'] = base_points
                w_obj['score_details'] = split_details
                w_obj['is_bonus'] = (word == self.bonus_word)
                
                # Ensure found_bonus_word is set if this word is the bonus word
                if word == self.bonus_word:
                    player.found_bonus_word = True
            
            # Sync authoritative bonus score into room's solved_words_with_scores
            if hasattr(self, 'solved_words_with_scores') and isinstance(self.solved_words_with_scores, dict):
                if res.get('bonus_word_points', 0) > 0 or res.get('bonus_letter_points', 0) > 0 or res.get('either_or_points', 0) > 0:
                    self.solved_words_with_scores[word] = res
                
        # 3. Update scores for each player
        for p in self.players:
            self._recalculate_player_score(p)

    def generate_ai_turns(self):
        """Pre-calculate BOT behavior at the START of a round for incremental scoring"""
        ais = [p for p in self.players if p.is_ai]
        if not ais:
            return
            
        print(f"[GameRoom] Pre-generating turns for {len(ais)} bots in room {self.room_id}")
        
        # Use existing logic from PrivateMatchManager
        from private_match_logic import private_match_manager
        
        for ai in ais:
            # Clear previous if any (should already be cleared by reset logic)
            ai.submitted_words = []
            ai.score = 0
                
            # Filter all_words by min_word_length
            min_len = self.current_min_length
            possible_words = [w for w in self.all_words if len(w) >= min_len]
            
            words_data, total_score = private_match_manager.generate_ai_submission(
                ai.ai_rating, 
                self.all_words_paths, # Use paths dict to avoid slow DFS in scorer
                self.bonus_word,
                board_format=self.current_board_format,
                bonus_cell=self.bonus_cell,
                duration=self.time_limit
            )
            
            # Record words with absolute timestamps
            start = self.round_start_time
            if start <= 0: start = time.time()
            
            for wd in words_data:
                # Use randomized time_offset provided by AI logic
                wd['time'] = start + wd.pop('time_offset', 0)
                ai.submitted_words.append(wd)
                
                # USER: Density decrement for bots
                self.update_density_for_word(ai, wd['word'])
                
                # FCFS Sync for bots: Also add to shared room lists
                if self.game_type == 'fcfs':
                    # Check if another bot already picked this word (rare but possible in generation)
                    if wd['word'].upper() not in self._fcfs_found_words_set:
                        bot_meta = dict(wd)
                        bot_meta['finder'] = ai.username
                        self.fcfs_found_words.append(bot_meta)
                        self._fcfs_found_words_set.add(wd['word'].upper())
            
            print(f"[GameRoom] Bot {ai.username} pre-generated {len(ai.submitted_words)} words (Synced for FCFS: {self.game_type == 'fcfs'})")
            
    def update_counts_by_len(self):
        """Authoritative refresher for word distribution metadata. 
        Calculates counts by length for all words present on the board."""
        min_l = int(getattr(self, 'current_min_length', 3) or 3)
        valid_words = [w for w in (self.all_words or []) if len(w) >= min_l]
        if len(valid_words) != len(self.all_words or []) and self.all_words:
            self.all_words = set(valid_words)
            if isinstance(getattr(self, 'all_words_paths', None), dict):
                self.all_words_paths = {w: p for w, p in self.all_words_paths.items() if w in self.all_words}
        self.total_counts_by_len = {
            '_round': self.current_round,
            '_total': len(valid_words),
            '_min_len': min_l,
            **{str(l): sum(1 for w in valid_words if len(w) == l) for l in range(1, 31)}
        }
        self.total_words_count = len(valid_words)
        self.initial_total_words = self.total_words_count
        if hasattr(word_validator, 'word_validator'):
            word_validator.word_validator.ensure_csw_loaded()
            self.csw_only_words = [w for w in valid_words if word_validator.word_validator.is_csw_only(w)]
            self.added_words = [w for w in valid_words if word_validator.word_validator.is_added_word(w)]
        
    def recalculate_total_points(self):
        """Aggregate total attainable points for the active round's word list"""
        try:
            active_words = set(self.all_words or [])
            scores = getattr(self, 'solved_words_with_scores', None)
            if active_words and (not isinstance(scores, dict) or set(scores.keys()) != active_words):
                from scoring import calculate_word_score
                new_scores = {}
                paths_map = getattr(self, 'all_words_paths', {}) or {}
                b_fmt = 'Normal' if getattr(self, 'game_type', None) == 'subanagrams' else getattr(self, 'current_board_format', 'Normal')
                b_cell = getattr(self, 'bonus_cell', None)
                b_grid = getattr(self, 'board', None)
                b_word = getattr(self, 'bonus_word', None)
                for w in active_words:
                    new_scores[w] = calculate_word_score(
                        w,
                        bonus_word=b_word,
                        board_format=b_fmt,
                        bonus_cell=b_cell,
                        board=b_grid,
                        path=_get_word_path(paths_map, w),
                        return_details=True,
                        strict_path=True
                    )
                self.solved_words_with_scores = new_scores
                scores = new_scores
            sync_bonus_word_scores(self)
            attainable = 0
            scores = getattr(self, 'solved_words_with_scores', None)
            
            if scores:
                for pts in scores.values():
                    if isinstance(pts, dict):
                        attainable += pts.get('total', 0)
                    elif isinstance(pts, int):
                        attainable += pts
            
            # FALLBACK: If scores are missing/empty but words exist, use fast length-based estimate
            if attainable == 0 and getattr(self, 'all_words', None):
                fmt = str(getattr(self, 'current_board_format', '')).lower()
                is_valued = 'valued' in fmt
                for w in self.all_words:
                    length = len(w)
                    if is_valued:
                        attainable += get_valued_word_score(w)
                    else:
                        if length <= 2:   attainable += 0
                        elif length <= 4: attainable += 1
                        elif length == 5: attainable += 2
                        elif length == 6: attainable += 3
                        elif length == 7: attainable += 5
                        elif length == 8: attainable += 9
                        elif length == 9: attainable += 13
                        elif length == 10: attainable += 17
                        elif length == 11: attainable += 21
                        elif length == 12: attainable += 25
                        elif length == 13: attainable += 29
                        elif length == 14: attainable += 33
                        else:             attainable += 37

            self.total_points_count = attainable
            if attainable == 0 and len(scores or {}) > 0:
                print(f"[RECALC-DEBUG] Room {self.room_id}: Attainable=0 despite {len(scores)} scores! First value: {list(scores.values())[0]}")
            return attainable
        except Exception as e:
            print(f"[GameRoom] Error recalculating points for {self.room_id}: {e}")
            self.total_points_count = 0
            return 0

    def _get_bonus_word(self, length=8, dictionary='NWL', alternating=False, difficulty='Medium', exclude=None):
        global _room_manager_instance
        if _room_manager_instance and hasattr(_room_manager_instance, '_get_bonus_word'):
            return _room_manager_instance._get_bonus_word(length=length, dictionary=dictionary, alternating=alternating, difficulty=difficulty, exclude=exclude)
        return 'PLANETS'

_GLOBAL_RECENT_BONUS_WORDS = []
_GLOBAL_BONUS_LOCK = threading.Lock()

def _init_global_bonus_words():
    global _GLOBAL_RECENT_BONUS_WORDS
    try:
        from db import get_db
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT bonus_word FROM round_history WHERE bonus_word IS NOT NULL AND bonus_word != '' ORDER BY id DESC LIMIT 50")
            rows = cursor.fetchall()
            words = []
            for r in rows:
                bw = (r['bonus_word'] if isinstance(r, dict) else r[0])
                if bw and str(bw).strip().upper() not in ['NONE', 'NULL', 'PLANETS']:
                    words.append(str(bw).strip().upper())
            with _GLOBAL_BONUS_LOCK:
                _GLOBAL_RECENT_BONUS_WORDS = list(dict.fromkeys(reversed(words)))[-50:]
            print(f"[BonusWord] Initialized global bonus word history with {len(_GLOBAL_RECENT_BONUS_WORDS)} words: {_GLOBAL_RECENT_BONUS_WORDS[-10:]}")
    except Exception as e:
        print(f"[BonusWord] Could not initialize global bonus words from DB: {e}")

_init_global_bonus_words()

def _record_global_bonus_word(bw):
    if not bw: return
    bw_up = str(bw).upper().strip()
    if not bw_up or bw_up in ['NONE', 'NULL', 'PLANETS']: return
    with _GLOBAL_BONUS_LOCK:
        if bw_up in _GLOBAL_RECENT_BONUS_WORDS:
            _GLOBAL_RECENT_BONUS_WORDS.remove(bw_up)
        _GLOBAL_RECENT_BONUS_WORDS.append(bw_up)
        if len(_GLOBAL_RECENT_BONUS_WORDS) > 50:
            del _GLOBAL_RECENT_BONUS_WORDS[:-50]

def _record_room_bonus_word(room, bw):
    """Record a bonus word in the room's rolling history and global history to prevent back-to-back repeats."""
    if not bw:
        return
    bw_up = str(bw).upper().strip()
    if not bw_up or bw_up in ['NONE', 'NULL', 'PLANETS']:
        return
    _record_global_bonus_word(bw_up)
    if room:
        if not hasattr(room, 'recent_bonus_words') or room.recent_bonus_words is None:
            room.recent_bonus_words = []
        if bw_up not in room.recent_bonus_words:
            room.recent_bonus_words.append(bw_up)
            if len(room.recent_bonus_words) > 20:
                room.recent_bonus_words = room.recent_bonus_words[-20:]
        if not hasattr(room, 'bonus_word_history') or room.bonus_word_history is None:
            room.bonus_word_history = []
        if bw_up not in room.bonus_word_history:
            room.bonus_word_history.append(bw_up)
            if len(room.bonus_word_history) > 20:
                room.bonus_word_history = room.bonus_word_history[-20:]

def pick_distinct_bonus_word(room, candidate, target_words, target_len, dictionary='NWL', is_checkerboard=False, room_manager=None, extra_exclude=None):
    """
    Selects a bonus word guaranteeing:
    1. It NEVER matches the immediately preceding round's bonus word (room.previous_bonus_word or ghost_bonus).
    2. It NEVER repeats any recent bonus words for the room or globally across the server (_GLOBAL_RECENT_BONUS_WORDS).
    3. It NEVER ends in ING or INGS.
    4. It strictly matches target_len (between 6 and 10).
    5. It is present on the board (in target_words).
    """
    import random
    immediate_exclude = set()
    if extra_exclude:
        if isinstance(extra_exclude, (list, tuple, set)):
            for e in extra_exclude:
                if e: immediate_exclude.add(str(e).upper().strip())
        else:
            immediate_exclude.add(str(extra_exclude).upper().strip())

    if room:
        if getattr(room, 'previous_bonus_word', None):
            immediate_exclude.add(str(room.previous_bonus_word).upper().strip())
        if getattr(room, 'bonus_word', None):
            immediate_exclude.add(str(room.bonus_word).upper().strip())

    for bad_tok in ('', 'NONE', 'NULL', 'PLANETS'):
        immediate_exclude.discard(bad_tok)

    exclude_set = set(immediate_exclude)
    with _GLOBAL_BONUS_LOCK:
        for gw in _GLOBAL_RECENT_BONUS_WORDS[-30:]:
            if gw: exclude_set.add(gw)

    if room:
        for rb in getattr(room, 'recent_bonus_words', []) or []:
            if rb: exclude_set.add(str(rb).upper().strip())
        for bh in getattr(room, 'bonus_word_history', []) or []:
            if bh: exclude_set.add(str(bh).upper().strip())

    for bad_tok in ('', 'NONE', 'NULL', 'PLANETS'):
        exclude_set.discard(bad_tok)

    try:
        target_len = int(target_len)
    except (ValueError, TypeError):
        target_len = 8
    target_len = max(6, min(10, target_len))

    words_list = list(target_words) if target_words else []
    words_upper_set = {str(w).upper().strip() for w in words_list if w}
    rm = room_manager or _room_manager_instance

    # 1. Candidate check: if valid, matching exact target_len, in words_list (if provided), and not excluded.
    # Also accept if candidate was already staged as room.next_round_bonus for THIS upcoming round and is not in immediate_exclude!
    cand_str = str(candidate or '').upper().strip()
    is_staged_for_this_round = (room and cand_str and cand_str == str(getattr(room, 'next_round_bonus', '') or '').upper().strip())
    if cand_str and cand_str not in ('NONE', 'NULL', 'PLANETS') and len(cand_str) == target_len and not cand_str.endswith(('ING', 'INGS')) and 'ING' not in cand_str:
        if not words_upper_set or cand_str in words_upper_set:
            if cand_str not in exclude_set or (is_staged_for_this_round and cand_str not in immediate_exclude):
                _record_room_bonus_word(room, cand_str)
                return cand_str

    # 2. Check exact matches of target_len on the board (strictly excluding full history first, then immediate_exclude, then relaxing immediate_exclude)
    valid_exact = [w for w in words_list if len(w) == target_len and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper() and w.upper() not in exclude_set]
    if not valid_exact and words_list:
        valid_exact = [w for w in words_list if len(w) == target_len and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper() and w.upper() not in immediate_exclude]
    if not valid_exact and words_list:
        valid_exact = [w for w in words_list if len(w) == target_len and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
    if valid_exact:
        chosen = random.choice(valid_exact).upper()
        _record_room_bonus_word(room, chosen)
        return chosen

    # 3. Fallback on board: closest length (6..10) from words_list (never return a word not on the board when words_list exists!)
    valid_range = [w for w in words_list if 6 <= len(w) <= 10 and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper() and w.upper() not in exclude_set]
    if not valid_range and words_list:
        valid_range = [w for w in words_list if 6 <= len(w) <= 10 and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper() and w.upper() not in immediate_exclude]
    if not valid_range and words_list:
        valid_range = [w for w in words_list if 6 <= len(w) <= 10 and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
    if valid_range:
        valid_range.sort(key=lambda w: abs(len(w) - target_len))
        closest_l = len(valid_range[0])
        pool = [w for w in valid_range if len(w) == closest_l]
        if pool:
            chosen = random.choice(pool).upper()
            _record_room_bonus_word(room, chosen)
            return chosen

    # 4. Any valid word on board not ending in ING/INGS and not containing ING, then literally any word on board
    any_valid = [w for w in words_list if not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper() and w.upper() not in immediate_exclude]
    if not any_valid and words_list:
        any_valid = [w for w in words_list if not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
    if not any_valid and words_list:
        any_valid = list(words_list)
    if any_valid:
        any_valid.sort(key=lambda w: abs(len(w) - target_len))
        chosen = any_valid[0].upper()
        _record_room_bonus_word(room, chosen)
        return chosen

    # 5. Only if words_list was empty (before board exists): pick a fresh target_len word from dictionary
    if rm and hasattr(rm, '_get_bonus_word'):
        chosen = rm._get_bonus_word(length=target_len, dictionary=dictionary, alternating=is_checkerboard, exclude=list(exclude_set))
        if chosen and chosen.upper() not in immediate_exclude and 'ING' not in chosen.upper() and not chosen.upper().endswith(('ING', 'INGS')):
            chosen = chosen.upper()
            _record_room_bonus_word(room, chosen)
            return chosen

    from word_validator import word_validator
    cand_dict = word_validator.csw_by_len.get(target_len, []) if 'CSW' in str(dictionary).upper() else word_validator.nwl_by_len.get(target_len, [])
    fallback_pool = [w for w in cand_dict if not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper() and w.upper() not in exclude_set]
    if fallback_pool:
        chosen = random.choice(fallback_pool).upper()
        _record_room_bonus_word(room, chosen)
        return chosen

    return 'PLANETS'

def ensure_bonus_word_on_board(room, target_len=8):
    """
    Ironclad guarantee: room.bonus_word MUST be a word that exists on room.board (in room.all_words).
    If room.bonus_word is empty, 'PLANETS', ends in ING/INGS, or is not in room.all_words,
    this function selects the best valid word from room.all_words (prioritizing exact length target_len,
    then closest length in 6..10, then any word not ending in ING/INGS) and updates
    room.bonus_word, room.spinner_params['bonus_word_length'], room.frozen_revealed_params, etc.
    """
    if not room or not getattr(room, 'all_words', None):
        return
    import random
    all_words_upper_map = {str(w).upper().strip(): str(w) for w in room.all_words if w}
    if not all_words_upper_map:
        return

    bw_up = str(getattr(room, 'bonus_word', '') or '').upper().strip()
    is_valid = (
        bw_up and 
        bw_up in all_words_upper_map and 
        bw_up not in ('NONE', 'NULL', 'PLANETS') and 
        not bw_up.endswith(('ING', 'INGS')) and
        'ING' not in bw_up
    )
    if is_valid:
        return

    try:
        target_len = int(target_len or 8)
    except (ValueError, TypeError):
        target_len = 8

    prev_bw = str(getattr(room, 'previous_bonus_word', '') or '').upper().strip()

    # 1. Exact length matches on board (not ending in ING/INGS, not containing ING, not previous)
    exact_cands = [w for w in all_words_upper_map.keys() if len(w) == target_len and not w.endswith(('ING', 'INGS')) and 'ING' not in w and w != prev_bw]
    if not exact_cands:
        exact_cands = [w for w in all_words_upper_map.keys() if len(w) == target_len and not w.endswith(('ING', 'INGS')) and 'ING' not in w]
    if exact_cands:
        room.bonus_word = random.choice(exact_cands)
    else:
        # 2. Closest length in [6..10] on board (not ending in ING/INGS, not containing ING, not previous)
        range_cands = [w for w in all_words_upper_map.keys() if 6 <= len(w) <= 10 and not w.endswith(('ING', 'INGS')) and 'ING' not in w and w != prev_bw]
        if not range_cands:
            range_cands = [w for w in all_words_upper_map.keys() if 6 <= len(w) <= 10 and not w.endswith(('ING', 'INGS')) and 'ING' not in w]
        if range_cands:
            range_cands.sort(key=lambda w: abs(len(w) - target_len))
            closest_len = len(range_cands[0])
            pool = [w for w in range_cands if len(w) == closest_len]
            room.bonus_word = random.choice(pool)
        else:
            # 3. Any non-ING word on board
            any_cands = [w for w in all_words_upper_map.keys() if not w.endswith(('ING', 'INGS')) and 'ING' not in w]
            if any_cands:
                any_cands.sort(key=lambda w: abs(len(w) - target_len))
                room.bonus_word = any_cands[0]
            else:
                room.bonus_word = random.choice(list(all_words_upper_map.keys()))

    actual_len = len(room.bonus_word)
    if isinstance(getattr(room, 'spinner_params', None), dict):
        room.spinner_params['bonus_word_length'] = actual_len
        room.spinner_params['bonus_word'] = room.bonus_word
    if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
        room.frozen_revealed_params['bonus_word_length'] = actual_len
        room.frozen_revealed_params['bonus_word'] = room.bonus_word
    if getattr(room, 'next_spinner_params', None) and isinstance(room.next_spinner_params, dict):
        room.next_spinner_params['bonus_word_length'] = actual_len
        room.next_spinner_params['bonus_word'] = room.bonus_word

    print(f"[BonusWordIntegrity] Room {getattr(room, 'room_id', '?')} bonus word guaranteed: '{room.bonus_word}' ({actual_len}L) present on active board.")

def sync_bonus_word_scores(room):
    """
    Ensure that ONLY room.bonus_word has bonus_word_points in room.solved_words_with_scores,
    and all other words have bonus_word_points = 0 and corrected total points.
    If room.solved_words_with_scores is missing, empty, or missing room.bonus_word,
    synchronously populate it from active board words.
    """
    if not room:
        return
    bw_up = str(getattr(room, 'bonus_word', '') or '').upper().strip()
    active_words = getattr(room, 'all_words', None)
    if active_words and (
        not isinstance(getattr(room, 'solved_words_with_scores', None), dict)
        or not room.solved_words_with_scores
        or (bw_up and bw_up in {w.upper() for w in active_words} and bw_up not in room.solved_words_with_scores)
    ):
        from scoring import calculate_word_score
        paths_map = getattr(room, 'all_words_paths', {}) or {}
        b_fmt = 'Normal' if getattr(room, 'game_type', None) == 'subanagrams' else getattr(room, 'current_board_format', 'Normal')
        b_cell = getattr(room, 'bonus_cell', None)
        b_grid = getattr(room, 'board', None)
        room.solved_words_with_scores = {
            w: calculate_word_score(
                w,
                bonus_word=room.bonus_word,
                board_format=b_fmt,
                bonus_cell=b_cell,
                board=b_grid,
                path=_get_word_path(paths_map, w),
                return_details=True,
                strict_path=True
            )
            for w in active_words
        }
    scores = getattr(room, 'solved_words_with_scores', None)
    if not scores or not isinstance(scores, dict):
        return

    for w, s_info in list(scores.items()):
        if not isinstance(s_info, dict):
            continue
        w_up = str(w).upper().strip()
        if bw_up and w_up == bw_up:
            needed_bwp = len(w_up)
            cur_bwp = s_info.get('bonus_word_points', 0)
            if cur_bwp != needed_bwp:
                base = s_info.get('base', s_info.get('total', 0) - cur_bwp)
                s_info['base'] = base
                s_info['bonus_word_points'] = needed_bwp
                s_info['total'] = base + needed_bwp + s_info.get('bonus_letter_points', 0) + s_info.get('either_or_points', 0)
        else:
            cur_bwp = s_info.get('bonus_word_points', 0)
            if cur_bwp > 0:
                s_info['bonus_word_points'] = 0
                base = s_info.get('base', max(0, s_info.get('total', 0) - cur_bwp))
                s_info['base'] = base
                s_info['total'] = base + s_info.get('bonus_letter_points', 0) + s_info.get('either_or_points', 0)

def _get_word_path(paths_dict, word):
    if not isinstance(paths_dict, dict) or not word:
        return None
    w_str = str(word)
    return paths_dict.get(w_str) or paths_dict.get(w_str.upper()) or paths_dict.get(w_str.lower())

def calculate_word_score(word, bonus_word, board_format='Normal', path=None, bonus_cell=None, **kwargs):
    """Calculate points for a word using shared utility"""
    from scoring import calculate_word_score as shared_calc
    return shared_calc(word, bonus_word, board_format=board_format, path=path, bonus_cell=bonus_cell, **kwargs)

_STATIC_FALLBACKS_CACHE = None
# Track recently used static fallback board hashes to prevent board repetition.
# Holds up to 20 board hashes; oldest are removed once the limit is hit.
_RECENTLY_USED_FALLBACK_HASHES = []  # ordered list, most recent last
_MAX_RECENT_FALLBACK_HASHES = 20

def _record_fallback_hash_used(board_hash):
    """Record a fallback board hash as recently used, evicting oldest if at capacity."""
    global _RECENTLY_USED_FALLBACK_HASHES
    if board_hash in _RECENTLY_USED_FALLBACK_HASHES:
        _RECENTLY_USED_FALLBACK_HASHES.remove(board_hash)
    _RECENTLY_USED_FALLBACK_HASHES.append(board_hash)
    if len(_RECENTLY_USED_FALLBACK_HASHES) > _MAX_RECENT_FALLBACK_HASHES:
        _RECENTLY_USED_FALLBACK_HASHES.pop(0)

def get_emergency_fallback_board(dimensions, board_format='Normal', time_limit=60, dictionary='NWL', use_added_words=False, target_range=None, min_word_length=None, difficulty=None, sequence_length=None, bonus_word_len=None):
    """Dynamically generate a valid emergency fallback board that matches room dimensions and spells correct words."""
    import random
    from board_generator import BoardGenerator, serialize_param_key, pop_cached_board, refill_board_cache_bg, pop_any_cached_board
    global _room_manager_instance
    if _room_manager_instance and hasattr(_room_manager_instance, 'board_generator'):
        bg = _room_manager_instance.board_generator
    else:
        bg = BoardGenerator()

    if str(dimensions).lower() == 'subanagrams':
        from subanagrams_generator import generate_subanagrams_board_and_words
        clean_fmt = board_format if board_format in ['Word Guaranteed', 'Totally Random'] else 'Word Guaranteed'
        target_seq_len = max(6, min(10, int(sequence_length))) if sequence_length else random.choice([6, 7, 8, 9, 10])
        sp = {
            'board_dimensions': 'subanagrams',
            'time_limit': 120,
            'min_word_length': min_word_length or 3,
            'sequence_length': target_seq_len,
            'dictionary': dictionary or 'NWL',
            'base_dictionary': str(dictionary or 'NWL').replace('+ AW', '').replace('+AW', '').strip(),
            'use_added_words': use_added_words,
            'board_format': clean_fmt,
            'mode': 'word' if clean_fmt == 'Word Guaranteed' else 'random',
            'word_count_range': target_range or '100-200',
            'difficulty': difficulty or 'Practice',
        }
        res = generate_subanagrams_board_and_words(sp)
        if res:
            b_board, b_words, b_bonus, fparams = res
            return (
                b_board,
                b_words,
                (0, 0),
                fparams.get('board_format', clean_fmt),
                {w: [] for w in b_words},
                0.5,
                b_bonus,
                fparams.get('word_count_range', '100-200'),
                fparams
            )
        else:
            # Absolute hardcoded fallback for Subanagrams: Strictly a 1D sequence of 8 letters (never a 4x4 grid!)
            fallback_seq = ['S', 'T', 'A', 'R', 'L', 'I', 'N', 'G']
            from subanagrams_generator import solve_subanagrams, get_subanagrams_dict
            dict_data = get_subanagrams_dict('NWL', False)
            fb_words = solve_subanagrams('STARLING', dict_data, min_length=3)
            fb_params = {
                'board_dimensions': 'subanagrams',
                'time_limit': 120,
                'min_word_length': 3,
                'sequence_length': 8,
                'dictionary': 'NWL',
                'board_format': 'Word Guaranteed',
                'mode': 'word',
                'word_count_range': '200-300'
            }
            return (
                [fallback_seq],
                fb_words,
                (0, 0),
                'Word Guaranteed',
                {w: [] for w in fb_words},
                0.5,
                'STARLING',
                '200-300',
                fb_params
            )
        
    from spinner_set import SpinnerSet
    floor_l = 3
    if '4x6' in dimensions:
        floor_l = 4
    elif '5x7' in dimensions:
        floor_l = 5
    elif '6x8' in dimensions or '3x3x3' in dimensions:
        floor_l = 6
        
    if min_word_length is None:
        min_word_length = floor_l
    else:
        min_word_length = max(floor_l, int(min_word_length))
        
    min_accept = 50
    max_accept = 99999
    if target_range:
        try:
            parts = str(target_range).split('-')
            min_accept = int(parts[0])
            if len(parts) > 1:
                max_accept = int(parts[1]) - 1
            elif str(target_range).endswith('+'):
                max_accept = 99999
        except:
            if '50' in str(target_range): min_accept, max_accept = 50, 99
            elif '100' in str(target_range): min_accept, max_accept = 100, 199
            elif '200' in str(target_range): min_accept, max_accept = 200, 299
            elif '300' in str(target_range): min_accept, max_accept = 300, 399
            elif '400' in str(target_range): min_accept, max_accept = 400, 499
            elif '500' in str(target_range): min_accept, max_accept = 500, 99999
            
    is_aw = use_added_words or '+ AW' in str(dictionary).upper() or '+AW' in str(dictionary).upper()
    if is_aw:
        min_accept = max(300, min_accept)
        if max_accept < 399:
            max_accept = 399
    elif min_word_length >= 6:
        min_accept = min(min_accept, 30)
    else:
        min_accept = max(30, min_accept)
    
    parts = dimensions.split("x")
    is_3d = len(parts) == 3
    is_24h = time_limit >= 7200
    
    if is_24h:
        use_added_words = False
        dictionary = str(dictionary or 'NWL').upper().replace('+ AW', '').replace('+AW', '').replace('ADDED_WORDS', '').replace('AW', '').strip()
        if dictionary not in ['NWL', 'CSW']:
            dictionary = 'NWL'
        target_range = '300-400'
        target_range_resolved = '300-400'
        min_accept, max_accept = 300, 399
        fmt = 'Valued Letters'
        min_word_length = floor_l
    else:
        fmt = board_format

    try:
        # Determine starting min length based on requested min_word_length (already clamped >= floor_l)
        min_l = min_word_length
        
        # Determine target range - strictly preserve provided target_range
        if is_aw:
            use_added_words = True
            if '4x4' in dimensions:
                if min_word_length >= 5:
                    target_range_resolved = '100-200'
                    min_accept, max_accept = 100, 200
                elif min_word_length == 4:
                    target_range_resolved = '200-300'
                    min_accept, max_accept = 200, 300
                else:
                    target_range_resolved = '300-400'
                    min_accept, max_accept = 300, 399
            elif '4x6' in dimensions:
                target_range_resolved = '400-500' if target_range == '400-500' else '300-400'
                min_accept, max_accept = (400, 499) if target_range_resolved == '400-500' else (300, 399)
            elif target_range and target_range in ['300-400', '400-500', '500+']:
                target_range_resolved = target_range
                if target_range_resolved == '300-400':
                    min_accept, max_accept = 300, 399
                elif target_range_resolved == '400-500':
                    min_accept, max_accept = 400, 499
                elif target_range_resolved == '500+':
                    min_accept, max_accept = 500, 99999
            else:
                import random
                target_range_resolved = random.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
                if target_range_resolved == '300-400':
                    min_accept, max_accept = 300, 399
                elif target_range_resolved == '400-500':
                    min_accept, max_accept = 400, 499
                elif target_range_resolved == '500+':
                    min_accept, max_accept = 500, 99999
        elif target_range:
            target_range_resolved = target_range
            if '4x4' in dimensions and target_range_resolved in ['400-500', '500+']:
                target_range_resolved = '200-300'
                min_accept, max_accept = 200, 299
            elif '4x6' in dimensions and target_range_resolved == '500+':
                target_range_resolved = '300-400'
                min_accept, max_accept = 300, 399
        elif not is_24h:
            target_range_resolved = '100-200'
        
        from spinner_set import SpinnerSet
        floor_l = 3 if '4x4' in dimensions else (4 if '4x6' in dimensions else (5 if '5x7' in dimensions else 6))

        # --- CACHE-FIRST FAST PATH ---
        # Try progressively relaxed param keys against the DB cache before doing any live generation.
        # A cache hit is instantaneous — avoids the 5-10s blocking generation.
        base_dict = str(dictionary or 'NWL').upper().replace('+ AW', '').replace('+AW', '').strip() or 'NWL'
        if base_dict not in ['NWL', 'CSW']:
            base_dict = 'NWL'
        
        if is_aw:
            ranges_order = [target_range_resolved] if target_range_resolved in ['300-400', '400-500', '500+'] else ['300-400', '400-500', '500+']
        elif target_range_resolved:
            ranges_order = [target_range_resolved]
        elif min_accept >= 300:
            ranges_order = ['300-400', '400-500', '500+']
        else:
            ranges_order = ['200-300', '100-200', '300-400', '50-100']

        cache_candidates = []
        for r_item in ranges_order:
            cache_candidates.append((min_l, r_item, fmt, base_dict, use_added_words))
            if str(fmt).lower() == 'normal':
                cache_candidates.append((min_l, r_item, 'Normal', base_dict, use_added_words))
                if min_l == floor_l:
                    cache_candidates.append((floor_l, r_item, 'Normal', base_dict, use_added_words))
                    if not use_added_words:
                        cache_candidates.append((floor_l, r_item, 'Normal', base_dict, False))
        
        norm_target_diff = SpinnerSet.strip_difficulty_pct(difficulty) if difficulty else None
        diff_order = [norm_target_diff] if norm_target_diff else ['Medium', 'Easy', 'Hard']
        _p_dims = str(dimensions).split("x")
        _d_fb = int(_p_dims[0]) if len(_p_dims) == 3 else 1
        _r_fb = int(_p_dims[1] if len(_p_dims) == 3 else _p_dims[0])
        _c_fb = int(_p_dims[2] if len(_p_dims) == 3 else _p_dims[1])
        for (cml, ctr, cfmt, cdict, caw) in cache_candidates:
            for cdiff in diff_order:
                try:
                    cache_key = serialize_param_key(dimensions, '', ctr, cdict, cfmt, cml, cdiff, use_added_words=caw)
                    cached = pop_cached_board(cache_key)
                    if cached and len(cached) >= 7:
                        cboard, cwords, cbonus_cell, cfmt_ret, cpaths, cratio, cbonus_word = cached[:7]
                        if norm_target_diff and not is_aw:
                            act_diff = bg.get_difficulty_label(cratio, _r_fb, _c_fb, base_dict, depth=_d_fb, min_word_length=cml)
                            if act_diff != norm_target_diff:
                                print(f"[get_emergency_fallback_board] Rejecting cached board with actual difficulty {act_diff} != target {norm_target_diff} (ratio {cratio})")
                                continue
                        if cwords and len(cwords) >= 20:
                            print(f"[get_emergency_fallback_board] INSTANT CACHE HIT: min={cml}, range={ctr}, fmt={cfmt}, dict={cdict} → {len(cwords)} words")
                            # Kick off background refill for the popped key
                            refill_board_cache_bg(bg, cache_key, target_count=3)
                            
                            final_min = max(floor_l, min_word_length)
                            cwords_filtered = [w for w in cwords if len(w) >= final_min]
                            cpaths_filtered = {w: p for w, p in cpaths.items() if len(w) >= final_min}
                            
                            matching_final_min = sum(1 for w in cwords_filtered if len(w) == final_min)
                            if matching_final_min < 3:
                                print(f"[get_emergency_fallback_board] Cached board had only {matching_final_min} words of length {final_min} (needed >= 3). REJECTING cached board...")
                                continue
                                
                            actual_wc = len(cwords_filtered)
                            if is_aw and actual_wc < 300:
                                print(f"[get_emergency_fallback_board] Cached board had only {actual_wc} words for +AW (needed >= 300). REJECTING cached board...")
                                continue
                            if actual_wc < min_accept or actual_wc > max_accept:
                                print(f"[get_emergency_fallback_board] Cached board had {actual_wc} words for min_word_length={final_min} (needed {min_accept}-{max_accept}). REJECTING cached board...")
                                continue
                            if target_range_resolved and not is_board_count_valid(actual_wc, target_range_resolved):
                                print(f"[get_emergency_fallback_board] Cached board count {actual_wc} outside target_range_resolved {target_range_resolved}. REJECTING cached board...")
                                continue

                            import word_validator
                            if hasattr(word_validator, 'word_validator'):
                                word_validator.word_validator.ensure_csw_loaded()
                                if base_dict == 'CSW':
                                    csw_cnt = sum(1 for w in cwords_filtered if word_validator.word_validator.is_csw_only(w))
                                    if csw_cnt == 0:
                                        print(f"[get_emergency_fallback_board] Cached board has 0 CSW-only words for CSW dictionary. REJECTING...")
                                        continue
                                if is_aw:
                                    aw_cnt = sum(1 for w in cwords_filtered if word_validator.word_validator.is_added_word(w))
                                    if aw_cnt == 0:
                                        print(f"[get_emergency_fallback_board] Cached board has 0 Added Words for +AW dictionary. REJECTING...")
                                        continue
                                
                            ctr_resolved = SpinnerSet.get_factchecked_wc_range(actual_wc, use_added_words=(is_aw or caw), dictionary=cdict)
                            caw_resolved = bool(caw or is_aw)
                            cdict_resolved = str(cdict or 'NWL').replace('+ AW', '').replace('+AW', '').strip()
                            if cdict_resolved == 'AW': cdict_resolved = 'NWL'
                            if caw_resolved:
                                cdict_resolved = f"{cdict_resolved} + AW"

                            eparams_dict = {
                                'min_word_length': final_min,
                                'word_count_range': ctr_resolved,
                                'board_format': cfmt,
                                'dictionary': cdict_resolved,
                                'use_added_words': caw_resolved,
                                'difficulty': cdiff,
                                'bonus_word_len': len(cbonus_word) if cbonus_word else 6
                            }
                            _rep, _rw, _rp = sync_board_checkerboard_and_resolve(bg, cboard, cfmt, final_min, cdict_resolved, caw_resolved, cbonus_cell)
                            if _rep and _rw:
                                if target_range_resolved and not is_board_count_valid(len(_rw), target_range_resolved):
                                    print(f"[get_emergency_fallback_board] sync_board re-solved words {len(_rw)} outside target_range_resolved {target_range_resolved}. REJECTING...")
                                    continue
                                cwords_filtered, cpaths_filtered = _rw, _rp
                            return cboard, cwords_filtered, cbonus_cell, cfmt_ret, cpaths_filtered, cratio, cbonus_word, ctr_resolved, eparams_dict
                except Exception:
                    continue

        # --- ULTIMATE CACHE FALLBACK (Pop compatible board matching dimensions and parameters) ---
        try:
            from board_generator import pop_compatible_cached_board
            relaxed_res = pop_compatible_cached_board(
                dimensions,
                dictionary,
                board_format,
                min_word_length,
                use_added_words,
                bonus_word_len=bonus_word_len,
                word_count_range=target_range,
                difficulty=norm_target_diff
            )
            if relaxed_res:
                board, words, bonus_cell, updated_format, paths, ratio, bonus_word, params = relaxed_res
                if words and len(words) >= 20:
                    params = dict(params) if params else {}
                    final_min = max(floor_l, min_word_length)
                    
                    words_filtered = [w for w in words if len(w) >= final_min]
                    paths_filtered = {w: p for w, p in paths.items() if len(w) >= final_min}
                    
                    actual_wc = len(words_filtered)
                    matching_final_min = sum(1 for w in words_filtered if len(w) == final_min)
                    import word_validator
                    if hasattr(word_validator, 'word_validator'):
                        word_validator.word_validator.ensure_csw_loaded()
                    csw_cnt_ult = sum(1 for w in words_filtered if word_validator.word_validator.is_csw_only(w)) if ('CSW' in str(dictionary).upper() and hasattr(word_validator, 'word_validator')) else 1
                    aw_cnt_ult = sum(1 for w in words_filtered if word_validator.word_validator.is_added_word(w)) if (is_aw and hasattr(word_validator, 'word_validator')) else 1

                    is_impossible_outlier = ('4x4' in dimensions and min_accept >= 400) or ('4x6' in dimensions and min_accept >= 500)
                    if is_24h:
                        is_impossible_outlier = False
                    if matching_final_min < 3:
                        print(f"[get_emergency_fallback_board] Ultimate cached board had only {matching_final_min} words of length {final_min} (needed >= 3). REJECTING cached board...")
                    elif is_24h and (actual_wc < 300 or actual_wc > 399):
                        print(f"[get_emergency_fallback_board] Ultimate cached board had {actual_wc} words for 24h match (strictly needed 300-399). REJECTING cached board...")
                    elif is_aw and actual_wc < 300:
                        print(f"[get_emergency_fallback_board] Ultimate cached board had only {actual_wc} words for +AW (needed >= 300). REJECTING cached board...")
                    elif (actual_wc < min_accept or actual_wc > max_accept) and not is_impossible_outlier:
                        print(f"[get_emergency_fallback_board] Ultimate cached board had {actual_wc} words for min_word_length={final_min} (needed {min_accept}-{max_accept}). REJECTING cached board...")
                    elif target_range and not is_board_count_valid(actual_wc, target_range) and not is_impossible_outlier:
                        print(f"[get_emergency_fallback_board] Ultimate cached board had {actual_wc} words outside target_range {target_range}. REJECTING cached board...")
                    elif 'CSW' in str(dictionary).upper() and csw_cnt_ult == 0:
                        print(f"[get_emergency_fallback_board] Ultimate cached board has 0 CSW-only words for CSW dictionary. REJECTING...")
                    elif is_aw and aw_cnt_ult == 0:
                        print(f"[get_emergency_fallback_board] Ultimate cached board has 0 Added Words for +AW dictionary. REJECTING...")
                    elif board_format and str(board_format).lower() not in ['normal', 'random'] and str(updated_format).lower() != str(board_format).lower():
                        print(f"[get_emergency_fallback_board] Ultimate cached board format {updated_format} != target {board_format}. REJECTING cached board...")
                    elif norm_target_diff and not is_aw and bg.get_difficulty_label(ratio, _r_fb, _c_fb, dictionary, depth=_d_fb, min_word_length=final_min) != norm_target_diff:
                        act_diff_ult = bg.get_difficulty_label(ratio, _r_fb, _c_fb, dictionary, depth=_d_fb, min_word_length=final_min)
                        print(f"[get_emergency_fallback_board] Ultimate cached board had actual difficulty {act_diff_ult} != target {norm_target_diff}. REJECTING cached board...")
                    else:
                        is_aw_ult = use_added_words or ('+ AW' in str(dictionary).upper()) or ('+AW' in str(dictionary).upper())
                        orig_base_dict = str(dictionary or 'NWL').replace('+ AW', '').replace('+AW', '').strip()
                        final_dict_name = f"{orig_base_dict} + AW" if is_aw_ult else orig_base_dict
                        ctr = SpinnerSet.get_factchecked_wc_range(actual_wc, use_added_words=is_aw_ult, dictionary=final_dict_name)
                        params['use_added_words'] = is_aw_ult
                        params['dictionary'] = final_dict_name
                        params['min_word_length'] = final_min
                        params['word_count_range'] = ctr
                        
                        print(f"[get_emergency_fallback_board] ULTIMATE CACHE HIT: popped any board of dimensions {dimensions} with {len(words_filtered)} words")
                        return board, words_filtered, bonus_cell, updated_format, paths_filtered, ratio, bonus_word, ctr, params
        except Exception as e:
            print(f"[get_emergency_fallback_board] Ultimate cache fallback error: {e}")
    except Exception as e:
        print(f"[get_emergency_fallback_board] General cache fallback error: {e}")

    # --- FAST LIVE GENERATION ATTEMPT BEFORE RANDOM GRID ---
    try:
        target_emergency_bw_len = int(bonus_word_len) if bonus_word_len else (7 if min_word_length < 8 else 8)
        target_emergency_bw_len = max(6, min(10, target_emergency_bw_len))
        if _room_manager_instance and hasattr(_room_manager_instance, '_get_bonus_word'):
            emergency_bw = _room_manager_instance._get_bonus_word(length=target_emergency_bw_len, dictionary=dictionary)
        else:
            emergency_bw = random.choice(['DIAMOND', 'PAINTER', 'CAPTIVE', 'SILVERS', 'WEATHER', 'MONSTER', 'STATION', 'JOURNEY'])
        live_res = bg.generate_board(
            dimensions=dimensions,
            bonus_word=emergency_bw,
            word_count_range=target_range_resolved,
            board_format=fmt,
            dictionary=dictionary,
            min_word_length=min_word_length,
            difficulty=difficulty or "Medium",
            timeout=3.0,
            use_added_words=use_added_words
        )
        if live_res and len(live_res) >= 6:
            l_b, l_w, l_c, l_f, l_p, l_r = live_res[:6]
            norm_diff_req = SpinnerSet.strip_difficulty_pct(difficulty or "Medium")
            if norm_diff_req == 'Easy' and l_r > 0.15:
                bg._sanitize_uniqueness(l_b, dictionary, max_ratio=0.15, depth=6 if is_3d else 1, bonus_cell=l_c, min_word_length=min_word_length, use_added_words=use_added_words)
                _re_s = bg._solve_board(l_b, dictionary, (0, 99999), min_word_length, max_depth=20, store_paths=True, timeout=5.0, bonus_cell=l_c, use_added_words=use_added_words)
                l_w = [w for w in _re_s if len(w) >= min_word_length]
                l_p = {w: p for w, p in _re_s.items() if len(w) >= min_word_length}
                l_r = bg._calculate_uniqueness_ratio(l_b, dictionary, 6 if is_3d else 1, l_c, min_word_length, use_added_words=use_added_words)
            l_bw = live_res[6] if len(live_res) >= 7 else (l_w[0] if l_w else 'PLANETS')
            l_w_filt = [w for w in (l_w or []) if len(w) >= min_word_length]
            min_required = 300 if (is_aw or is_24h) else 15
            if len(l_w_filt) >= min_required and (not is_24h or len(l_w_filt) <= 399):
                # If target_range_resolved was provided and generated board exceeded max_accept, sample down
                if False and max_accept < 99999 and len(l_w_filt) > max_accept:  # never truncate valid words
                    sampled_set = proportionally_sample_words(l_w_filt, max_accept)
                    if l_bw and l_bw in l_w_filt and l_bw not in sampled_set:
                        sampled_set.add(l_bw)
                    l_w_filt = [w for w in l_w_filt if w in sampled_set]
                    l_p = {w: p for w, p in l_p.items() if w in sampled_set}
                l_p_filt = {w: p for w, p in l_p.items() if w in l_w_filt}

                if is_24h:
                    target_range_resolved = '300-400'
                else:
                    target_range_resolved = SpinnerSet.get_factchecked_wc_range(len(l_w_filt), use_added_words=is_aw, dictionary=dictionary)

                eparams = {
                    'min_word_length': min_word_length,
                    'word_count_range': target_range_resolved,
                    'board_format': 'Valued Letters' if is_24h else l_f,
                    'dictionary': dictionary,
                    'use_added_words': False if is_24h else use_added_words,
                    'difficulty': norm_diff_req,
                    'bonus_word_len': len(l_bw)
                }
                _rep, _rw, _rp = sync_board_checkerboard_and_resolve(bg, l_b, l_f, min_word_length, dictionary, use_added_words, l_c)
                if _rep and _rw:
                    l_w_filt, l_p_filt = _rw, _rp
                print(f"[get_emergency_fallback_board] Fast live generator succeeded: {len(l_w_filt)} words (range: {target_range_resolved})")
                return l_b, l_w_filt, l_c, l_f, l_p_filt, l_r, l_bw, target_range_resolved, eparams
    except Exception as live_err:
        print(f"[get_emergency_fallback_board] Fast live generation exception: {live_err}")

    # --- INSTANT EMERGENCY FALLBACK (Random grid solve) ---
    print(f"[get_emergency_fallback_board] Delivering instant pre-built emergency board for {dimensions}...")
    parts = dimensions.split("x")
    is_3d = len(parts) == 3
    is_bl_fmt = ('bonus letter' in str(fmt).lower() or str(fmt).lower().strip() == 'bonus')
    is_eo_fmt = ('either' in str(fmt).lower())
    is_chk_fmt = ('checkerboard' in str(fmt).lower())
    is_mania_fmt = ('mania' in str(fmt).lower())

    if is_3d:
        board = [
            [['S', 'T', 'A'], ['R', 'E', 'D'], ['L', 'I', 'N']],
            [['E', 'R', 'S'], ['A', 'N', 'T'], ['I', 'C', 'S']],
            [['T', 'R', 'A'], ['I', 'N', 'S'], ['C', 'A', 'P']],
            [['P', 'A', 'R'], ['T', 'I', 'E'], ['S', 'E', 'T']],
            [['S', 'O', 'U'], ['N', 'D', 'S'], ['F', 'A', 'R']],
            [['W', 'O', 'R'], ['D', 'S', 'E'], ['T', 'S', 'S']]
        ]
        bonus_cell = (0, 0, 0) if is_bl_fmt else None
    else:
        try:
            rows, cols = map(int, parts[:2])
        except (ValueError, TypeError):
            rows, cols = 4, 4
        # Use high-density distribution for high-target or AW rounds to ensure word counts match requirements
        from board_generator import LETTER_FREQ_SUPER_DENSITY
        if (is_aw or min_accept >= 200) and hasattr(bg, '_create_normal_board'):
            board = bg._create_normal_board(
                rows, cols, LETTER_FREQ_SUPER_DENSITY, depth=1 if not is_3d else 6,
                difficulty='Easy', dictionary=dictionary, word_count_range=target_range_resolved, use_added_words=use_added_words
            )
        else:
            # Randomly shuffled standard letter distribution to ensure unique boards
            std_letters = ["S","T","A","R","E","D","L","I","N","E","R","S","A","N","T","I","C","S","T","R","A","I","N","S","C","A","P","T","U","R","E","S","O","U","N","D","S","F","A","R","W","O","R","D","S","E","T","S"]
            offset = random.randint(0, len(std_letters) - 1)
            shuffled = std_letters[offset:] + std_letters[:offset]
            random.shuffle(shuffled)
            board = [[shuffled[(r*cols + c) % len(shuffled)] for c in range(cols)] for r in range(rows)]
        bonus_cell = (random.randint(0, rows-1), random.randint(0, cols-1)) if is_bl_fmt else None

    if is_eo_fmt:
        bonus_cell = None
        if not is_3d and rows > 0 and cols > 0:
            board[0][0] = f"{board[0][0]}/E" if '/' not in str(board[0][0]) else board[0][0]
    elif is_chk_fmt:
        bg._verify_checkerboard_safeguard(board)
    elif is_mania_fmt:
        parts_m = str(fmt).strip().upper().split()
        m_l = parts_m[0] if (len(parts_m) >= 2 and len(parts_m[0]) == 1) else 'E'
        bg._apply_mania_to_board(board, m_l, set(), is_checkerboard=False)
    else:
        if bg._is_checkerboard_layout(board):
            bg._break_accidental_checkerboard(board)

    # Solve emergency board with 5s timeout
    emergency_solve = bg._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=15, store_paths=True, timeout=5.0, bonus_cell=bonus_cell, use_added_words=use_added_words)
    words_filtered = [w for w in emergency_solve if len(w) >= min_word_length]
    paths_filtered = {w: p for w, p in emergency_solve.items() if len(w) >= min_word_length}

    # CRITICAL SAFEGUARD: If min_word_length on random grid yields fewer than 15 words, solve with 3L floor
    if len(words_filtered) < 15:
        emergency_solve_relaxed = bg._solve_board(board, dictionary, (0, 99999), 3, max_depth=15, store_paths=True, timeout=5.0, bonus_cell=bonus_cell, use_added_words=use_added_words)
        words_filtered = [w for w in emergency_solve_relaxed if len(w) >= 3]
        paths_filtered = {w: p for w, p in emergency_solve_relaxed.items() if len(w) >= 3}
        min_word_length = 3

    # CRITICAL SAFEGUARD FOR WORD COUNT (Strictly adhere to min_accept):
    if len(words_filtered) < min_accept:
        print(f"[get_emergency_fallback_board] Emergency board had only {len(words_filtered)} words (needed >= {min_accept}). Rescuing to guarantee word count...")
        from board_generator import LETTER_FREQ_SUPER_DENSITY
        for _dense_try in range(1):
            dense_b = bg._create_normal_board(
                rows, cols, LETTER_FREQ_SUPER_DENSITY, depth=1 if not is_3d else 6,
                difficulty='Easy', dictionary=dictionary, word_count_range=target_range_resolved, use_added_words=use_added_words
            )
            if is_mania_fmt:
                bg._apply_mania_to_board(dense_b, m_l, set(), is_checkerboard=False)
            dense_s = bg._solve_board(dense_b, dictionary, (0, 99999), min_word_length, max_depth=20, store_paths=True, timeout=2.0, bonus_cell=bonus_cell, use_added_words=use_added_words)
            dense_w = [w for w in dense_s if len(w) >= min_word_length]
            if len(dense_w) >= min_accept or len(dense_w) > len(words_filtered):
                board = dense_b
                emergency_solve = dense_s
                words_filtered = dense_w
                paths_filtered = {w: p for w, p in dense_s.items() if len(w) >= min_word_length}
                print(f"[get_emergency_fallback_board] Super-density emergency board reached {len(words_filtered)} words.")
                break

    # CRITICAL SAFEGUARD FOR CSW & ADDED WORDS:
    import word_validator
    if hasattr(word_validator, 'word_validator'):
        word_validator.word_validator.ensure_csw_loaded()
        if 'CSW' in str(dictionary).upper():
            csw_cnt = sum(1 for w in words_filtered if word_validator.word_validator.is_csw_only(w))
            if csw_cnt == 0:
                print(f"[get_emergency_fallback_board] Emergency board has 0 CSW-only words. Injecting CSW letters...")
                csw_to_inject = "CWTCH" if min_word_length <= 5 else "CRWTHS"
                for idx_c, ch in enumerate(csw_to_inject):
                    if is_3d:
                        if idx_c < len(board[0]) * len(board[0][0]):
                            board[0][idx_c // len(board[0][0])][idx_c % len(board[0][0])] = ch
                    else:
                        if idx_c < len(board) * len(board[0]):
                            board[idx_c // len(board[0])][idx_c % len(board[0])] = ch
                emergency_solve = bg._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=25, store_paths=True, timeout=10.0, bonus_cell=bonus_cell, use_added_words=use_added_words)
                words_filtered = [w for w in emergency_solve if len(w) >= min_word_length]
                paths_filtered = {w: p for w, p in emergency_solve.items() if len(w) >= min_word_length}

    # FINAL MANIA GUARD FOR EMERGENCY BOARDS
    if is_mania_fmt:
        total_cells_m = (rows * cols * 6) if is_3d else (rows * cols)
        if is_3d:
            m_cnt = sum(1 for f in range(len(board)) for r in range(len(board[f])) for c in range(len(board[f][r])) if board[f][r][c] == m_l)
        else:
            m_cnt = sum(1 for r in range(rows) for c in range(cols) if board[r][c] == m_l)
        UNCOMMON_MANIA_LETTERS = {'Q', 'Z', 'J', 'X', 'K'}
        target_ratio_m = 0.20 if m_l in UNCOMMON_MANIA_LETTERS else 0.33
        target_m_cnt = max(2, round(total_cells_m * target_ratio_m))
        if m_cnt < target_m_cnt:
            print(f"[get_emergency_fallback_board] Final Mania audit: board had only {m_cnt} '{m_l}' (needed {target_m_cnt}). Applying abundance...")
            bg._apply_mania_to_board(board, m_l, set(), is_checkerboard=False)
            emergency_solve = bg._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=25, store_paths=True, timeout=10.0, bonus_cell=bonus_cell, use_added_words=use_added_words)
            words_filtered = [w for w in emergency_solve if len(w) >= min_word_length]
            paths_filtered = {w: p for w, p in emergency_solve.items() if len(w) >= min_word_length}

        if is_aw:
            aw_cnt = sum(1 for w in words_filtered if word_validator.word_validator.is_added_word(w))
            if aw_cnt == 0:
                print(f"[get_emergency_fallback_board] Emergency board has 0 Added Words. Injecting AW letters...")
                aw_to_inject = "AALII" if min_word_length <= 5 else "AALIIS"
                for idx_c, ch in enumerate(aw_to_inject):
                    if is_3d:
                        if idx_c < len(board[0]) * len(board[0][0]):
                            board[0][idx_c // len(board[0][0])][idx_c % len(board[0][0])] = ch
                    else:
                        if idx_c < len(board) * len(board[0]):
                            board[idx_c // len(board[0])][idx_c % len(board[0])] = ch
                emergency_solve = bg._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=25, store_paths=True, timeout=10.0, bonus_cell=bonus_cell, use_added_words=use_added_words)
                words_filtered = [w for w in emergency_solve if len(w) >= min_word_length]
                paths_filtered = {w: p for w, p in emergency_solve.items() if len(w) >= min_word_length}

    # FINAL MANIA GUARD FOR EMERGENCY BOARDS
    if is_mania_fmt:
        total_cells_m = (rows * cols * 6) if is_3d else (rows * cols)
        if is_3d:
            m_cnt = sum(1 for f in range(len(board)) for r in range(len(board[f])) for c in range(len(board[f][r])) if board[f][r][c] == m_l)
        else:
            m_cnt = sum(1 for r in range(rows) for c in range(cols) if board[r][c] == m_l)
        UNCOMMON_MANIA_LETTERS = {'Q', 'Z', 'J', 'X', 'K'}
        target_ratio_m = 0.20 if m_l in UNCOMMON_MANIA_LETTERS else 0.33
        target_m_cnt = max(2, round(total_cells_m * target_ratio_m))
        if m_cnt < target_m_cnt:
            print(f"[get_emergency_fallback_board] Final Mania audit: board had only {m_cnt} '{m_l}' (needed {target_m_cnt}). Applying abundance...")
            bg._apply_mania_to_board(board, m_l, set(), is_checkerboard=False, difficulty=difficulty or "Medium")
            emergency_solve = bg._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=25, store_paths=True, timeout=10.0, bonus_cell=bonus_cell, use_added_words=use_added_words)
            words_filtered = [w for w in emergency_solve if len(w) >= min_word_length]
            paths_filtered = {w: p for w, p in emergency_solve.items() if len(w) >= min_word_length}

    # Strict Medium/Hard guard for emergency fallback boards
    is_3d = is_3d or (len(board) == 6 and isinstance(board[0], list) and isinstance(board[0][0], list))
    norm_diff_em = SpinnerSet.strip_difficulty_pct(difficulty or "Medium")
    if norm_diff_em in ["Medium", "Hard"]:
        d_em = 6 if is_3d else 1
        if bg._has_ing_sequence(board, d_em):
            bg._guarantee_no_ing(board, d_em, protected_positions=None)
            emergency_solve = bg._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=25, store_paths=True, timeout=10.0, bonus_cell=bonus_cell, use_added_words=use_added_words)
            words_filtered = [w for w in emergency_solve if len(w) >= min_word_length]
            paths_filtered = {w: p for w, p in emergency_solve.items() if len(w) >= min_word_length}

    # Strict uniqueness sanitization for Easy fallback boards
    cur_em_ratio = bg._calculate_uniqueness_ratio(board, dictionary, 6 if is_3d else 1, bonus_cell, min_word_length, use_added_words=use_added_words)
    if norm_diff_em == 'Easy' and cur_em_ratio > 0.15:
        bg._sanitize_uniqueness(board, dictionary, max_ratio=0.15, depth=6 if is_3d else 1, bonus_cell=bonus_cell, min_word_length=min_word_length, use_added_words=use_added_words)
        emergency_solve = bg._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=20, store_paths=True, timeout=5.0, bonus_cell=bonus_cell, use_added_words=use_added_words)
        words_filtered = [w for w in emergency_solve if len(w) >= min_word_length]
        paths_filtered = {w: p for w, p in emergency_solve.items() if len(w) >= min_word_length}
        cur_em_ratio = bg._calculate_uniqueness_ratio(board, dictionary, 6 if is_3d else 1, bonus_cell, min_word_length, use_added_words=use_added_words)
    elif norm_diff_em == 'Hard' and cur_em_ratio < 0.28:
        bg._boost_uniqueness(board, dictionary, min_ratio=0.28, depth=6 if is_3d else 1, bonus_cell=bonus_cell, min_word_length=min_word_length, use_added_words=use_added_words)
        emergency_solve = bg._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=20, store_paths=True, timeout=5.0, bonus_cell=bonus_cell, use_added_words=use_added_words)
        words_filtered = [w for w in emergency_solve if len(w) >= min_word_length]
        paths_filtered = {w: p for w, p in emergency_solve.items() if len(w) >= min_word_length}
        cur_em_ratio = bg._calculate_uniqueness_ratio(board, dictionary, 6 if is_3d else 1, bonus_cell, min_word_length, use_added_words=use_added_words)

    # MANDATE: Bonus Word MUST match declared bonus_word_length if specified, or in [6..10], and never contain/end in ING/INGS
    target_bw_len = sequence_length if sequence_length else None
    bw_candidates = [w for w in words_filtered if (len(w) == target_bw_len if target_bw_len else (6 <= len(w) <= 10)) and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
    if not bw_candidates:
        bw_candidates = [w for w in words_filtered if 6 <= len(w) <= 10 and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
    if bw_candidates:
        bonus_word = bw_candidates[0]
    else:
        if _room_manager_instance and hasattr(_room_manager_instance, '_get_bonus_word'):
            bonus_word = _room_manager_instance._get_bonus_word(length=target_bw_len or 6, dictionary=dictionary)
        else:
            bonus_word = 'PLANETS'
        if bonus_word and bonus_word not in words_filtered:
            words_filtered.append(bonus_word)
            paths_filtered[bonus_word] = [bonus_cell or (0, 0)]

    actual_cnt = len(words_filtered)
    if is_24h:
        target_range_resolved = '300-400'
        if len(words_filtered) > 399:
            bw_set = {bonus_word} if bonus_word else set()
            words_filtered = list(bw_set | set(words_filtered[:399]))
            paths_filtered = {w: p for w, p in paths_filtered.items() if w in words_filtered}
    else:
        target_range_resolved = SpinnerSet.get_factchecked_wc_range(actual_cnt, use_added_words=use_added_words, dictionary=dictionary)

    eparams = {
        'min_word_length': floor_l if is_24h else min_word_length,
        'word_count_range': '300-400' if is_24h else target_range_resolved,
        'board_format': 'Valued Letters' if is_24h else fmt,
        'dictionary': dictionary,
        'use_added_words': False if is_24h else use_added_words,
        'difficulty': norm_diff_em,
        'bonus_word_len': len(bonus_word),
        '_exact_wc_calculated': True,
        '_lock_word_count_range': True
    }

    # Trigger background refill to populate SQLite cache for next rounds
    def _async_bg_generate():
        try:
            bg.generate_board(
                dimensions=dimensions,
                bonus_word=None,
                word_count_range=target_range_resolved,
                dictionary=dictionary,
                board_format=fmt,
                min_word_length=min_word_length,
                difficulty=norm_diff_em,
                is_emergency=True,
                use_added_words=use_added_words
            )
        except Exception as async_err:
            print(f"[get_emergency_fallback_board] Background board generation error: {async_err}")

    threading.Thread(target=_async_bg_generate, daemon=True).start()
    return board, words_filtered, bonus_cell, fmt, paths_filtered, cur_em_ratio, bonus_word, target_range_resolved, eparams

    # Last resort if no static fallback exists in cache
    print(f"[get_emergency_fallback_board] CRITICAL: Static fallback cache empty! Returning empty mock board.")
    parts = dimensions.split("x")
    is_3d = len(parts) == 3
    if is_3d:
        board = [[['E' for _ in range(3)] for _ in range(3)] for _ in range(6)]
        bonus_cell = (0, 0, 0)
    else:
        rows, cols = map(int, parts)
        board = [['E' for _ in range(cols)] for _ in range(rows)]
        bonus_cell = (0, 0)
    return board, ["EAR"], bonus_cell, board_format, {"EAR": [bonus_cell]}, 0.5, "EAR", "50-100", {}


class RoomManager:
    def __init__(self):
        global _room_manager_instance
        _room_manager_instance = self
        self.rooms: Dict[str, GameRoom] = {}
        self.user_presence: Dict[str, float] = {} # {user_id_str: last_active_timestamp}
        self.lock = threading.RLock() # USE RLOCK: Prevents deadlocks when start_next_round calls start_round
        self.board_generator = BoardGenerator()
        
        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._bg_cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        print("[RoomManager] Background cleanup thread started")
    
    def _bg_cleanup_loop(self):
        """Authoritative Heartbeat: State transitions, cleanup, and milestones."""
        print("[RoomManager] Heartbeat entering main loop...")
        loop_counter = 0
        while True:
            loop_counter += 1
            try:
                # 1. State Advancing: iterate room-by-room
                rooms_to_process = list(self.rooms.items())
                for room_id, room in rooms_to_process:
                    try:
                        # Inactivity pause: if room is empty of human players and not a 24h daily room,
                        # pause it by setting state to 'waiting' and skipping milestone transition ticks.
                        # This prevents empty public rooms from looping rounds and pegging CPU in background.
                        is_daily = (room.time_limit >= 7200)
                        humans = [p for p in room.players if not getattr(p, 'is_ai', False)]
                        spectators = [s for s in room.spectators if not getattr(s, 'is_ai', False)]
                        all_humans = humans + spectators
                        now_t = time.time()
                        active_humans = [p for p in all_humans if (now_t - getattr(p, 'last_active', 0)) < 120.0]
                        if len(active_humans) == 0 and not is_daily:
                            # NEVER pause or wipe a room during an active round, intermission, or initial loading!
                            # Let active play, intermission, and round start run cleanly to completion.
                            if room.state in ['active', 'intermission', 'loading'] or getattr(room, 'starting_round', False):
                                pass
                            elif room.state != 'waiting':
                                print(f"[Heartbeat] Pausing inactive room {room_id} (0 active humans for >120s). Setting state to 'waiting'.")
                                with room._state_lock:
                                    room.state = 'waiting'
                                    room.starting_round = False
                                    room.board_search_started = False
                                    room.board_search_loading = False
                                    room.spinner_params_generated = False
                                    room.next_round_board = None
                                    room.board = None
                                continue

                        # timers/transitions
                        room.check_and_update_state()
                        self.check_6x8_rescue(room)
                        
                        # Milestones (Spinner, Search, Reveal, Round Start)
                        milestone = room.get_next_round_milestone()
                        if milestone == 'spinner':
                            # Use a temporary flag to prevent double-spinner launches during thread startup
                            if not getattr(room, '_transition_spinner_launched', False):
                                room._transition_spinner_launched = True
                                threading.Thread(target=self.generate_spinner_params, args=(room_id, False), daemon=True).start()
                        elif milestone == 'reveal':
                            if not getattr(room, 'spinner_params_revealed', False):
                                threading.Thread(target=self.generate_spinner_params, args=(room_id, True), daemon=True).start()
                        elif milestone == 'search':
                            # ATOMIC LOCK: Set flag IMMEDIATELY in this main heartbeat thread
                            with room._state_lock:
                                if not getattr(room, 'board_search_started', False) and not getattr(room, 'board_search_loading', False):
                                    room.board_search_loading = True 
                                    threading.Thread(target=self.start_board_search, args=(room_id,), daemon=True).start()
                        elif milestone == 'start':
                            # ATOMIC LOCK check only; start_next_round handles state management
                            if not getattr(room, 'starting_round', False):
                                threading.Thread(target=self.start_next_round, args=(room_id,), daemon=True).start()
                    except Exception as tick_err:
                        import traceback
                        print(f"[Heartbeat] Error on {room_id}: {tick_err}\n{traceback.format_exc()}")
                
                # 2. Inactivity Cleanup (Every 5s: 20 iterations at 0.25s)
                if loop_counter % 20 == 0:
                    self.cleanup_rooms(timeout=600, spec_timeout=600)
                    now = time.time()
                    with self.lock:
                        self.user_presence = {uid: ts for uid, ts in self.user_presence.items() if (now - ts) < 600}
                
                time.sleep(0.25)
            except Exception as e:
                import traceback
                print(f"[Heartbeat] CRITICAL: {e}\n{traceback.format_exc()}")
                time.sleep(5)
                
    def load_daily_score_sums(self, room):
        """
        Query daily_score_sums for this 24h room and cache on room.daily_score_sums
        so it is instantly available in the room state snapshot.
        """
        if not room or getattr(room, 'time_limit', 0) < 7200:
            return []
        try:
            canonical_key = normalize_24h_room_key(getattr(room, 'room_id', ''), getattr(room, 'board_dimensions', ''))
            dims = canonical_key.replace('24h_', '')
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT u.username, d.user_id, SUM(d.score_sum) as score_sum
                    FROM daily_score_sums d
                    JOIN users u ON d.user_id = u.id
                    WHERE (d.room_id = ? OR d.room_id = ? OR d.room_id LIKE ?)
                      AND d.score_sum > 0
                    GROUP BY d.user_id, u.username
                    HAVING SUM(d.score_sum) > 0
                    ORDER BY score_sum DESC
                ''', (canonical_key, str(room.room_id), f"%{dims}%"))
                rows = cursor.fetchall()

                # If no rows found, try automatic recovery from round_history
                if not rows:
                    try:
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
                        conn.commit()

                        cursor.execute('''
                            SELECT u.username, d.user_id, SUM(d.score_sum) as score_sum
                            FROM daily_score_sums d
                            JOIN users u ON d.user_id = u.id
                            WHERE (d.room_id = ? OR d.room_id = ? OR d.room_id LIKE ?)
                              AND d.score_sum > 0
                            GROUP BY d.user_id, u.username
                            HAVING SUM(d.score_sum) > 0
                            ORDER BY score_sum DESC
                        ''', (canonical_key, str(room.room_id), f"%{dims}%"))
                        rows = cursor.fetchall()
                    except Exception as _bf_err:
                        print(f"[load_daily_score_sums] Non-fatal backfill error: {_bf_err}")

                players = [{'username': r[0], 'score_sum': r[2]} for r in rows if r[2] and r[2] > 0]
                room.daily_score_sums = players
                return players
        except Exception as e:
            print(f"[RoomManager] Error loading daily_score_sums for {getattr(room, 'room_id', '')}: {e}")
            if not hasattr(room, 'daily_score_sums') or room.daily_score_sums is None:
                room.daily_score_sums = []
            return room.daily_score_sums

    def load_previous_day_data(self, room):
        """
        Load previous day (yesterday) data from the SQLite database
        if the in-memory variables are empty/vanished (e.g. after server restart).
        """
        if getattr(room, 'time_limit', 0) >= 7200 and not getattr(room, 'daily_score_sums', None):
            self.load_daily_score_sums(room)

        if getattr(room, 'previous_board', None) and getattr(room, 'previous_day_history', None):
            return # Already has in-memory data
            
        import json
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                # 1. Find the last completed round number
                if room.current_round <= 2:
                    # On server startup/reset (round <= 2), find the absolute latest round in history
                    cursor.execute("SELECT MAX(round_number) FROM round_history WHERE room_id = ?", (room.room_id,))
                else:
                    cursor.execute("SELECT MAX(round_number) FROM round_history WHERE room_id = ? AND round_number < ?", (room.room_id, room.current_round))
                row = cursor.fetchone()
                if not row or row[0] is None:
                    return
                    
                last_round = row[0]
            
                # 2. Query all player entries for this round
                cursor.execute('''
                    SELECT rh.user_id, u.username, rh.words_json, rh.board_json, rh.bonus_word, rh.bonus_cell, rh.board_format, rh.all_solutions_json, rh.all_words_paths, rh.board_dimensions, rh.total_words_avail
                    FROM round_history rh
                    LEFT JOIN users u ON rh.user_id = u.id
                    WHERE rh.room_id = ? AND rh.round_number = ?
                ''', (room.room_id, last_round))
                
                rows = cursor.fetchall()
                if not rows:
                    return
                
            # Parse common board/round attributes from the first entry
            first_row = rows[0]
            user_id, username, words_json, board_json, bonus_word, bonus_cell, board_format, all_solutions_json, all_words_paths, board_dimensions, total_words_avail = first_row
            
            # Restore previous board
            try:
                room.previous_board = json.loads(board_json)
            except Exception as e:
                print(f"[Restore-Error] previous_board: {e}")
                room.previous_board = []
                
            room.previous_bonus_word = bonus_word or ''
            try:
                room.previous_bonus_cell = json.loads(bonus_cell) if bonus_cell else None
            except:
                room.previous_bonus_cell = None
                
            # Restore solutions/all words
            restored_solutions = []
            restored_paths = {}
            for r in rows:
                # Look for the record that has solutions and paths
                if r[7]: # all_solutions_json
                    try:
                        restored_solutions = json.loads(r[7])
                    except: pass
                if r[8]: # all_words_paths
                    try:
                        restored_paths = json.loads(r[8])
                    except: pass
                    
            room.previous_all_words = restored_solutions
            room.previous_total_words = total_words_avail if total_words_avail else len(restored_solutions)
            
            # Reconstruct word scores for history
            from scoring import calculate_word_score
            restored_scores = {}
            for w in restored_solutions:
                w_upper = w.upper()
                w_path = restored_paths.get(w_upper) or restored_paths.get(w)
                restored_scores[w_upper] = calculate_word_score(
                    w_upper,
                    room.previous_bonus_word,
                    board_format='Valued Letters' if room.time_limit >= 7200 else (board_format or 'Normal'),
                    bonus_cell=room.previous_bonus_cell,
                    board=room.previous_board,
                    path=w_path,
                    return_details=True,
                    strict_path=True
                )
            room.previous_all_word_scores = restored_scores
            
            # Restore total points count by summing restored scores
            total_pts = 0
            for pts in restored_scores.values():
                if isinstance(pts, dict):
                    total_pts += pts.get('total', 0)
                elif isinstance(pts, int):
                    total_pts += pts
            room.previous_total_points = total_pts
            
            # Filter CSW / Added words for yesterday
            import word_validator
            if str(getattr(room, 'current_dictionary', 'NWL')).upper() in ['CSW', 'AW', 'ALL', 'ADDED_WORDS']:
                word_validator.word_validator.ensure_csw_loaded()
            room.previous_csw_only_words = [w for w in restored_solutions if word_validator.word_validator.is_csw_only(w)]
            room.previous_added_words = [w for w in restored_solutions if word_validator.word_validator.is_added_word(w)]
            
            # Restore previous day history (who found what)
            history = {}
            for r in rows:
                u_id = r[0]
                u_name = r[1] or ('System' if u_id == -1 else f"User_{u_id}")
                w_json = r[2]
                try:
                    words_data = json.loads(w_json) if w_json else []
                    if room.time_limit >= 7200 and words_data and room.previous_board:
                        from scoring import calculate_word_score
                        recalculated_words = []
                        for w_item in words_data:
                            if isinstance(w_item, dict):
                                word_str = w_item.get('word', '')
                            else:
                                word_str = str(w_item)
                            
                            word_upper = word_str.upper()
                            w_path = None
                            if restored_paths:
                                w_path = restored_paths.get(word_upper) or restored_paths.get(word_str)
                            
                            details = calculate_word_score(
                                word_upper,
                                room.previous_bonus_word,
                                board_format='Valued Letters',
                                bonus_cell=room.previous_bonus_cell,
                                board=room.previous_board,
                                path=w_path,
                                return_details=True,
                                strict_path=True
                            )
                            
                            recalculated_words.append({
                                'word': word_str,
                                'points': details.get('total', 0) if details else 0,
                                'timestamp': w_item.get('timestamp') if isinstance(w_item, dict) else time.time(),
                                'score_details': details
                            })
                        history[str(u_id)] = {
                            'username': u_name,
                            'found_words': recalculated_words
                        }
                    else:
                        found = []
                        for w_item in words_data:
                            if isinstance(w_item, dict):
                                found.append(w_item.get('word', ''))
                            else:
                                found.append(str(w_item))
                        history[str(u_id)] = {
                            'username': u_name,
                            'found_words': [w.upper() for w in found if w]
                        }
                except Exception as pe:
                    print(f"[Restore-Error] player {u_name} history: {pe}")
                    
            room.previous_day_history = history
            print(f"[RoomManager] Successfully restored previous day history for {room.room_id} Round {last_round} ({len(history)} players)")
            
        except Exception as ex:
            import traceback
            print(f"[RoomManager] Error restoring previous day data: {ex}\n{traceback.format_exc()}")

    def create_room(self, room_id, game_type, time_limit, board_dimensions, min_rating=0, max_rating=9999, is_private=False, is_solo=False, initial_solo_params=None):
        """Create a new game room or return an existing singleton for the configuration"""
        import threading
        try:
            with self.lock:
                # Singleton Logic ONLY for default pub_v2_ hubs (User-created rooms generate unique instances)
                if not is_private and str(room_id).startswith('pub_v2_'):
                    for existing_room in list(self.rooms.values()):
                        if (str(existing_room.game_type).lower() == str(game_type).lower() and 
                            str(existing_room.board_dimensions).lower() == str(board_dimensions).lower() and
                            int(existing_room.time_limit) == int(time_limit) and
                            int(existing_room.min_rating) == int(min_rating) and
                            int(existing_room.max_rating) == int(max_rating) and
                            not existing_room.is_solo and
                            not existing_room.is_private and
                            not str(existing_room.room_id).startswith('practice_')):
                            # WAKE UP CHECK: If room is in waiting state or empty of active human players, wake it up to active Round 1!
                            humans = [p for p in existing_room.players if not p.is_ai]
                            now_t = time.time()
                            active_humans = [p for p in humans if (now_t - getattr(p, 'last_active', 0)) < 120.0]
                            is_empty_or_inactive = (len(active_humans) == 0)
                            needs_wakeup = (existing_room.state == 'waiting' or not existing_room.board or (is_empty_or_inactive and existing_room.state == 'intermission' and (existing_room.game_type != 'subanagrams' or existing_room.time_remaining <= 0)))
                            if needs_wakeup and existing_room.time_limit < 7200:
                                if existing_room.state in ['waiting', 'intermission'] or not existing_room.board:
                                    if existing_room.game_type == 'subanagrams':
                                        # If room was sitting in waiting state or board is missing, refresh with newly randomized board
                                        if existing_room.state == 'waiting' or not existing_room.board:
                                            existing_room.board = None
                                            existing_room.next_round_board = None
                                        elif existing_room.board and len(existing_room.board) > 0 and existing_room.state == 'active':
                                            print(f"[RoomManager] Subanagrams singleton already has active board: {existing_room.room_id}")
                                            return existing_room
                                        # Fast-path 2: If next_round_board is already staged, promote it instantly in <1ms!
                                        if getattr(existing_room, 'next_round_board', None) and getattr(existing_room, 'next_round_words', None):
                                            print(f"[RoomManager] Subanagrams singleton promoting staged next_round_board instantly (<1ms): {existing_room.room_id}")
                                            b_board = existing_room.next_round_board
                                            b_words = existing_room.next_round_words
                                            b_bonus = getattr(existing_room, 'next_round_bonus', None)
                                            fparams = getattr(existing_room, 'next_spinner_params', None) or dict(existing_room.spinner_params or {})
                                            existing_room.next_round_board = None
                                            existing_room.next_round_words = None
                                            existing_room.next_round_bonus = None
                                        else:
                                            from subanagrams_generator import generate_subanagrams_board_and_words
                                            sp = SpinnerSet.generate_subanagrams_params()
                                            res = generate_subanagrams_board_and_words(sp)
                                            if res:
                                                b_board, b_words, b_bonus, fparams = res
                                            else:
                                                b_board = None
                                        if b_board:
                                            existing_room.board = b_board
                                            existing_room.sequence_length = fparams.get('sequence_length', len(b_board[0]))
                                            existing_room.all_words = set(b_words)
                                            existing_room.all_words_paths = {w: [] for w in b_words}
                                            existing_room.bonus_word = b_bonus
                                            seq_len_actual = len(b_board[0]) if (b_board and len(b_board) > 0 and isinstance(b_board[0], list)) else len(b_board)
                                            has_full = any(len(w) == seq_len_actual for w in b_words)
                                            final_fmt = 'Word Guaranteed' if (has_full and fparams.get('mode') == 'word') else 'Totally Random'
                                            existing_room.current_board_format = final_fmt
                                            fparams['board_format'] = final_fmt
                                            fparams['mode'] = 'word' if final_fmt == 'Word Guaranteed' else 'random'
                                            existing_room.current_min_length = fparams.get('min_word_length', 3)
                                            existing_room.current_dictionary = fparams.get('dictionary', 'CSW')
                                            existing_room.use_added_words = fparams.get('use_added_words', False)
                                            existing_room.spinner_params = dict(fparams)
                                            existing_room.frozen_revealed_params = dict(fparams)
                                            existing_room.current_word_count_range = fparams.get('word_count_range', '100-200')
                                            existing_room.spinner_params_generated = True
                                            from scoring import calculate_word_score
                                            existing_room.solved_words_with_scores = {
                                                w: calculate_word_score(w, b_bonus, board_format='Normal', return_details=True)
                                                for w in b_words
                                            }
                                            existing_room.total_words_count = len(b_words)
                                            existing_room.initial_total_words = len(b_words)
                                            existing_room.recalculate_total_points()
                                            existing_room.state = 'active'
                                            existing_room.round_start_time = time.time()
                                            existing_room.current_round = max(1, getattr(existing_room, 'current_round', 0) + 1)
                                            existing_room._initial_board_delivered = True
                                            self.pre_generate_next_round(existing_room.room_id)
                                            print(f"[RoomManager] Subanagrams singleton woken up: {existing_room.room_id}")
                                            return existing_room

                                    pre_pop = None
                                    # Fast-path 1: If next_round_board is already staged, promote it instantly in <1ms!
                                    if getattr(existing_room, 'next_round_board', None) and getattr(existing_room, 'next_round_words', None):
                                        print(f"[RoomManager] Non-subanagram singleton promoting staged next_round_board instantly (<1ms): {existing_room.room_id}")
                                        pre_pop = (
                                            existing_room.next_round_board,
                                            existing_room.next_round_words,
                                            getattr(existing_room, 'next_round_bonus_cell', None),
                                            getattr(existing_room, 'next_round_format', 'Normal'),
                                            getattr(existing_room, 'next_round_word_paths', {}),
                                            getattr(existing_room, 'next_round_uniqueness', 0.0),
                                            getattr(existing_room, 'next_round_bonus', ''),
                                            None,
                                            getattr(existing_room, 'next_spinner_params', None) or getattr(existing_room, 'next_round_spinner_params', None) or existing_room.spinner_params or {}
                                        )
                                        existing_room.next_round_board = None
                                        existing_room.next_round_words = []
                                        existing_room.next_round_word_paths = {}
                                        existing_room.next_round_bonus = None
                                        existing_room.next_spinner_params = None
                                        existing_room.next_round_spinner_params = None

                                    if not pre_pop:
                                        print(f"[RoomManager] Wakeup empty singleton {existing_room.room_id} to Active Round 1 with fresh board...")
                                        from board_generator import pop_any_cached_board, pop_compatible_cached_board
                                        sp_fresh = SpinnerSet.generate_params(existing_room.board_dimensions, is_24h=(existing_room.time_limit >= 7200))
                                        f_dict = sp_fresh.get('dictionary', 'NWL')
                                        f_aw = sp_fresh.get('use_added_words', False)
                                        f_fmt = sp_fresh.get('board_format', 'Normal')
                                        f_min = sp_fresh.get('min_word_length', 3)
                                        f_wc = sp_fresh.get('word_count_range')
                                        pre_pop = pop_compatible_cached_board(existing_room.board_dimensions, f_dict, f_fmt, f_min, f_aw, word_count_range=f_wc)
                                        if not pre_pop:
                                            pre_pop = pop_any_cached_board(existing_room.board_dimensions, min_word_length=f_min, target_format=f_fmt, dictionary=f_dict, use_added_words=f_aw, word_count_range=f_wc)
                                        if not pre_pop:
                                            pre_pop = pop_any_cached_board(existing_room.board_dimensions, min_word_length=f_min, target_format=f_fmt, dictionary=f_dict, use_added_words=f_aw)
                                        if not pre_pop:
                                            pre_pop = pop_any_cached_board(existing_room.board_dimensions, min_word_length=f_min)
                                        if not pre_pop:
                                            pre_pop = pop_any_cached_board(existing_room.board_dimensions)
                                        if not pre_pop:
                                            pre_pop = get_emergency_fallback_board(existing_room.board_dimensions, f_fmt, existing_room.time_limit, dictionary=f_dict, use_added_words=f_aw, target_range=f_wc, min_word_length=f_min)
                                    if pre_pop:
                                        if len(pre_pop) >= 9:
                                            r_board, r_words, r_bonus_c, r_fmt, r_dict, r_ratio, r_bonus_word, _, r_params = pre_pop
                                        else:
                                            r_board, r_words, r_bonus_c, r_fmt, r_dict, r_ratio, r_bonus_word, r_params = pre_pop
                                        
                                        r_params = dict(r_params) if isinstance(r_params, dict) else {}
                                        board_min_l = r_params.get('min_word_length', 3)
                                        grid_floor = 3
                                        if '4x6' in existing_room.board_dimensions: grid_floor = 4
                                        elif '5x7' in existing_room.board_dimensions: grid_floor = 5
                                        elif '6x8' in existing_room.board_dimensions or '3x3x3' in existing_room.board_dimensions: grid_floor = 6
                                        board_min_l = max(grid_floor, int(board_min_l) if board_min_l is not None else grid_floor)

                                        raw_dict = r_params.get('dictionary', 'NWL')
                                        raw_aw = r_params.get('use_added_words', False)
                                        existing_room.all_words_paths = {w: p for w, p in (r_dict or {}).items() if len(w) >= board_min_l and word_validator.word_validator.is_valid_word(w, raw_dict, use_added_words=raw_aw)}
                                        existing_room.all_words = set(existing_room.all_words_paths.keys())
                                        if existing_room.all_words:
                                            actual_shortest = min(len(w) for w in existing_room.all_words)
                                            if actual_shortest > board_min_l:
                                                board_min_l = actual_shortest

                                        wc_cnt = len(existing_room.all_words)
                                        wc_lbl = SpinnerSet.get_factchecked_wc_range(wc_cnt, use_added_words=raw_aw, dictionary=raw_dict)
                                        diff_lbl_wake = SpinnerSet.format_difficulty_with_pct(r_params.get('difficulty', 'Medium'), r_ratio)

                                        existing_room.spinner_params = {
                                            'dictionary': raw_dict,
                                            'board_format': r_fmt or 'Normal',
                                            'min_word_length': board_min_l,
                                            'word_count_range': wc_lbl,
                                            'difficulty': diff_lbl_wake,
                                            'uniqueness': r_ratio,
                                            'use_added_words': raw_aw,
                                            'bonus_word_length': len(r_bonus_word) if r_bonus_word else 8,
                                            'board_dimensions': existing_room.board_dimensions,
                                            'time_limit': existing_room.time_limit,
                                            'generated_at': time.time(),
                                            '_exact_wc_calculated': True
                                        }
                                        existing_room.spinner_params = SpinnerSet.sanitize_params(existing_room.spinner_params, existing_room.board_dimensions, existing_room.time_limit >= 7200)
                                        wc_lbl = existing_room.spinner_params.get('word_count_range', wc_lbl)

                                        existing_room.current_min_length = board_min_l
                                        existing_room.current_difficulty = existing_room.spinner_params.get('difficulty', diff_lbl_wake)
                                        existing_room.current_board_format = r_fmt or 'Normal'
                                        existing_room.current_word_count_range = wc_lbl
                                        existing_room.current_dictionary = existing_room.spinner_params.get('dictionary', raw_dict)
                                        existing_room.current_uniqueness = r_ratio
                                        existing_room.use_added_words = existing_room.spinner_params.get('use_added_words', raw_aw)

                                        existing_room.bonus_word = pick_distinct_bonus_word(
                                            existing_room, r_bonus_word, existing_room.all_words, len(r_bonus_word) if r_bonus_word else 8,
                                            dictionary=raw_dict, is_checkerboard=('checkerboard' in str(r_fmt).lower()),
                                            room_manager=self
                                        )
                                        ensure_bonus_word_on_board(existing_room, len(r_bonus_word) if r_bonus_word else 8)
                                        existing_room.previous_bonus_word = ''
                                        # CRITICAL: Always keep spinner_params bonus_word_length, min_word_length, and word_count_range strictly synchronized with the actual board!
                                        existing_room.spinner_params['bonus_word_length'] = len(existing_room.bonus_word)
                                        existing_room.spinner_params['word_count_range'] = wc_lbl
                                        existing_room.spinner_params['min_word_length'] = board_min_l
                                        import copy
                                        existing_room.frozen_revealed_params = copy.deepcopy(existing_room.spinner_params)
                                        existing_room.board = r_board
                                        existing_room.bonus_cell = r_bonus_c
                                        existing_room.total_words_count = len(existing_room.all_words)
                                        existing_room.initial_total_words = existing_room.total_words_count
                                        existing_room.current_round = 1
                                        existing_room._initial_board_delivered = True
                                        existing_room.last_round_had_winner = False
                                        existing_room.winners_history = [h for h in getattr(existing_room, 'winners_history', []) if h.get('round') != 1]
                                        existing_room.round_start_time = time.time()
                                        existing_room.state = 'active'
                                        existing_room.solved_words_with_scores = {}
                                        existing_room.update_counts_by_len()
                                        existing_room.initialize_density(r_board, existing_room.all_words_paths, r_fmt)
                                        existing_room.recalculate_total_points()
                                        # Clear any stale staging from previous idle session and kick off Round 2 pre-generation
                                        existing_room.next_round_board = None
                                        existing_room.next_round_words = []
                                        existing_room.next_round_word_paths = {}
                                        existing_room.next_round_word_scores = {}
                                        existing_room.next_round_csw_only_words = []
                                        existing_room.next_round_added_words = []
                                        existing_room.next_spinner_params = None
                                        existing_room.next_round_spinner_params = None
                                        existing_room.spinner_params_generated = False
                                        existing_room.spinner_params_revealed = False
                                        existing_room._fallback_at_50s_done = False
                                        existing_room.board_search_started = False
                                        existing_room.board_search_loading = False
                                        existing_room.board_search_started_actual = False
                                        self.pre_generate_next_round(existing_room.room_id)

                            print(f"[RoomManager] Singleton: Returning existing {game_type} room {existing_room.room_id}")
                            return existing_room

                print(f"[RoomManager] Creating NEW room {room_id} for {game_type} ({board_dimensions})")
                room = GameRoom(
                    room_id=room_id,
                    game_type=game_type,
                    time_limit=time_limit,
                    board_dimensions=board_dimensions,
                    min_rating=min_rating,
                    max_rating=max_rating,
                    is_solo=(is_solo or game_type == 'practice' or (room_id and room_id.startswith('practice_')) or (game_type and 'solo' in str(game_type).lower())),
                    is_private=is_private
                )
                if initial_solo_params:
                    room.initial_solo_params = dict(initial_solo_params)
                
                # Capacity Check
                if room.game_type in ['accumulative', 'solo_accumulative', 'subanagrams']:
                    room.max_players = 9999
                else:
                    room.max_players = 8

                self.rooms[room_id] = room
                
                # Chronological sequence synchronization: Query absolute max completed round from history
                max_round = 0
                try:
                    with get_db() as conn_r:
                        cursor_r = conn_r.cursor()
                        cursor_r.execute("SELECT MAX(round_number) FROM round_history WHERE room_id = ?", (room_id,))
                        last_round_row = cursor_r.fetchone()
                        if last_round_row and last_round_row[0] is not None:
                            max_round = last_round_row[0]
                except Exception as r_err:
                    print(f"[RoomManager] Error querying max round for initialization: {r_err}")

                room.current_round = max_round
                
                # PERSISTENCE FOR 24H ROOMS ON CREATION
                restored_active = False
                is_24h = (room.time_limit >= 7200)
                is_split = (room.game_type == 'split')
                
                if is_24h:
                    import json
                    import datetime
                    from zoneinfo import ZoneInfo
                    tz = DAILY_TZ
                    
                    try:
                        with get_db() as conn:
                            cursor = conn.execute('''
                                SELECT board_data, all_words, dictionary, min_length, updated_at,
                                       bonus_word, bonus_cell_json, board_format, uniqueness, word_count_range,
                                       active_players_json
                                FROM active_boards WHERE room_id = ?
                            ''', (room_id,))
                            row = cursor.fetchone()
                        if row:
                            board_data_json, all_words_json, dictionary, min_length, updated_at, bonus_word, bonus_cell_json, board_format, uniqueness, word_count_range, active_players_json = row
                            
                            # Check if the board is from the same day and complies with strict 300-400 range for 24h
                            saved_dt = datetime.datetime.fromtimestamp(updated_at, tz)
                            now_dt = datetime.datetime.fromtimestamp(time.time(), tz)
                            loaded_words = set(json.loads(all_words_json)) if all_words_json else set()
                            is_wc_compliant = (300 <= len(loaded_words) <= 399) if is_24h else True
                            if saved_dt.date() == now_dt.date() and is_wc_compliant:
                                print(f"[RoomManager] Restoring active 24h board for {room_id} from DB (saved at {saved_dt}, {len(loaded_words)} words)")
                                room.board = json.loads(board_data_json)
                                room.all_words = loaded_words
                                room.current_dictionary = dictionary or 'NWL'
                                room.current_min_length = min_length or 3
                                room.bonus_word = bonus_word or ''
                                room.bonus_cell = json.loads(bonus_cell_json) if bonus_cell_json else None
                                room.current_board_format = 'Valued Letters' if is_24h else (board_format or 'Normal')
                                room.current_uniqueness = uniqueness or 0.0
                                room.current_word_count_range = '300-400' if is_24h else (word_count_range or '100-200')
                                room.update_counts_by_len()
                                
                                # Deserialize and restore active players
                                if active_players_json:
                                    try:
                                        players_data = json.loads(active_players_json)
                                        restored_players = []
                                        for d in players_data:
                                            p = Player(
                                                user_id=d.get('user_id'),
                                                username=d.get('username'),
                                                rating=d.get('rating', 1200)
                                            )
                                            p.submitted_words = d.get('submitted_words', [])
                                            p.invalid_words = d.get('invalid_words', [])
                                            p.score = d.get('score', 0)
                                            p.previous_round_score = d.get('previous_round_score', 0)
                                            p.games_played = d.get('games_played', 0)
                                            p.previous_submitted_words = d.get('previous_submitted_words', [])
                                            p.found_bonus_word = d.get('found_bonus_word', False)
                                            p.last_active = d.get('last_active', time.time())
                                            p.input_method = d.get('input_method', 'mouse')
                                            p.country_flag = d.get('country_flag', '🏳️')
                                            p.joined_mid_round = d.get('joined_mid_round', False)
                                            p.has_exceptional_round = d.get('has_exceptional_round', False)
                                            p.is_guest = d.get('is_guest', False)
                                            p.is_ai = d.get('is_ai', False)
                                            p.ai_rating = d.get('ai_rating', 1200)
                                            p.has_abandoned = d.get('has_abandoned', False)
                                            p.game_room_chat = d.get('game_room_chat', True)
                                            restored_players.append(p)
                                        room.players = restored_players
                                        
                                        # Sync past_players and other mappings
                                        for rp in restored_players:
                                            room.past_players[str(rp.user_id)] = rp
                                        print(f"[RoomManager] Restored {len(restored_players)} active players/words for 24h room {room_id}")
                                    except Exception as p_err:
                                        print(f"[RoomManager] Error restoring active players for {room_id}: {p_err}")
                                
                                # Set state to active
                                room.round_start_time = updated_at # Preserve round start time!
                                room.state = 'active'
                                room.current_round = max_round + 1
                                
                                room.spinner_params = {
                                    'board_dimensions': room.board_dimensions,
                                    'dictionary': room.current_dictionary,
                                    'min_word_length': room.current_min_length,
                                    'board_format': room.current_board_format,
                                    'word_count_range': room.current_word_count_range,
                                    'difficulty': getattr(room, 'current_difficulty', 'Medium'),
                                    'bonus_word_length': len(room.bonus_word) if room.bonus_word else 8
                                }
                                room.spinner_params_generated = True
                                
                                # Re-generate word paths and scoring in background
                                def async_rebuild_active_scoring():
                                    try:
                                        from board_generator import BoardGenerator
                                        bg = BoardGenerator()
                                        flat_board = room.board
                                        # Re-solve the board to get paths
                                        words_dict = bg._solve_board(flat_board, room.current_dictionary, room.current_min_length)
                                        # Keep only words that are in all_words
                                        room.all_words_paths = {w: p for w, p in words_dict.items() if w in room.all_words}
                                        
                                        # Recalculate scores
                                        from scoring import calculate_word_score
                                        refined = {}
                                        for word in room.all_words:
                                            refined[word] = calculate_word_score(
                                                word, room.bonus_word, path=room.all_words_paths.get(word),
                                                board_format=room.current_board_format, bonus_cell=room.bonus_cell,
                                                board=room.board, return_details=True, strict_path=True
                                            )
                                        room.solved_words_with_scores = refined
                                        room.recalculate_total_points()
                                        print(f"[RoomManager] Active 24h board scoring/paths rebuilt successfully for {room_id}")
                                    except Exception as ex:
                                        print(f"[RoomManager] Error rebuilding scoring/paths for restored 24h room {room_id}: {ex}")
                                        
                                import threading
                                threading.Thread(target=async_rebuild_active_scoring, daemon=True).start()
                                self.pre_generate_next_round(room_id)
                                restored_active = True
                            elif saved_dt.date() == now_dt.date():
                                print(f"[RoomManager] Saved 24h board for {room_id} has {len(loaded_words)} words (violates strict 300-400 range). Rejecting saved board so fresh compliant board generates.")
                            else:
                                print(f"[RoomManager] Outdated 24h board found for {room_id} from {saved_dt} (Today is {now_dt}). Archiving to round_history.")
                                try:
                                    old_board = json.loads(board_data_json)
                                    old_all_words = set(json.loads(all_words_json))
                                    old_bonus_word = bonus_word or ''
                                    old_bonus_cell = json.loads(bonus_cell_json) if bonus_cell_json else None
                                    old_board_format = board_format or 'Normal'
                                    old_min_len = min_length or 3
                                    
                                    # Solve board to reconstruct paths for history validation
                                    from board_generator import BoardGenerator
                                    bg = BoardGenerator()
                                    old_paths = bg._solve_board(old_board, dictionary or 'NWL', old_min_len)
                                    
                                    # Restore active player snapshot data
                                    old_players = []
                                    if active_players_json:
                                        try:
                                            players_data = json.loads(active_players_json)
                                            for d in players_data:
                                                old_players.append({
                                                    'user_id': d.get('user_id'),
                                                    'username': d.get('username'),
                                                    'score': d.get('score', 0),
                                                    'submitted_words': d.get('submitted_words', []),
                                                    'invalid_words': d.get('invalid_words', []),
                                                    'rating': d.get('rating', 1200),
                                                    'performance_efficiency': d.get('performance_efficiency', 0.0),
                                                    'joined_mid_round': d.get('joined_mid_round', False)
                                                })
                                        except Exception as pe:
                                            print(f"[RoomManager] Error parsing old active players: {pe}")
                                            
                                    # Asynchronously archive to round_history
                                    archive_round_num = max_round + 1
                                    
                                    def archive_old_board_async():
                                        try:
                                            self.save_round_history(
                                                room,
                                                board=old_board,
                                                all_words=old_all_words,
                                                bonus_word=old_bonus_word,
                                                player_snapshots=old_players,
                                                round_num=archive_round_num,
                                                all_words_paths=old_paths,
                                                round_start_time=updated_at,
                                                board_format=old_board_format
                                            )
                                            print(f"[RoomManager] Successfully archived outdated 24h board as Round {archive_round_num}")
                                        except Exception as archive_err:
                                            print(f"[RoomManager] Error archiving outdated 24h board: {archive_err}")
                                            
                                    import threading
                                    threading.Thread(target=archive_old_board_async, daemon=True).start()
                                    
                                    # Advance current round index to accommodate the archived round
                                    room.current_round = max_round + 1
                                    
                                except Exception as archive_outer_err:
                                    print(f"[RoomManager] Error setting up archive for outdated board: {archive_outer_err}")
                    except Exception as db_err:
                        print(f"[RoomManager] Error checking/restoring active board from DB: {db_err}")
                    finally:
                        conn.close()
                
                if not restored_active:
                    # INSTANT START: Kickstart room immediately by popping/generating board first
                    if (not is_private or room.is_solo or is_24h):
                        if room.game_type == 'subanagrams':
                            print(f"[RoomManager] {room_id}: Kickstarting Subanagrams room immediately...")
                            from subanagrams_generator import generate_subanagrams_board_and_words
                            sp = SpinnerSet.generate_subanagrams_params()
                            res = generate_subanagrams_board_and_words(sp)
                            if res:
                                b_board, b_words, b_bonus, fparams = res
                                room.board = b_board
                                room.sequence_length = fparams.get('sequence_length', len(b_board[0]))
                                room.all_words = set(b_words)
                                room.all_words_paths = {w: [] for w in b_words}
                                room.bonus_word = b_bonus
                                seq_len_actual = len(b_board[0]) if (b_board and len(b_board) > 0 and isinstance(b_board[0], list)) else len(b_board)
                                has_full = any(len(w) == seq_len_actual for w in b_words)
                                final_fmt = 'Word Guaranteed' if (has_full and fparams.get('mode') == 'word') else 'Totally Random'
                                room.current_board_format = final_fmt
                                fparams['board_format'] = final_fmt
                                fparams['mode'] = 'word' if final_fmt == 'Word Guaranteed' else 'random'
                                room.current_min_length = fparams.get('min_word_length', 3)
                                room.current_dictionary = fparams.get('dictionary', 'CSW')
                                room.use_added_words = fparams.get('use_added_words', False)
                                room.spinner_params = dict(fparams)
                                room.frozen_revealed_params = dict(fparams)
                                room.current_word_count_range = fparams.get('word_count_range', '100-200')
                                room.spinner_params_generated = True
                                from scoring import calculate_word_score
                                room.solved_words_with_scores = {
                                    w: calculate_word_score(w, b_bonus, board_format='Normal', return_details=True)
                                    for w in b_words
                                }
                                room.total_words_count = len(b_words)
                                room.initial_total_words = len(b_words)
                                room.recalculate_total_points()
                                room.state = 'active'
                                room.round_start_time = time.time()
                                room.current_round = max(1, getattr(room, 'current_round', 0) + 1)
                                room._initial_board_delivered = True
                                self.pre_generate_next_round(room_id)
                                print(f"[RoomManager] Subanagrams room kickstarted successfully: {room_id}")
                                return room

                        print(f"[RoomManager] {room_id}: Kickstarting room immediately by popping board first...")
                        from board_generator import pop_cached_board, pop_any_cached_board, pop_compatible_cached_board, serialize_param_key
                        
                        is_24h_room = (room.time_limit >= 7200) or ('24h' in str(room.room_id)) or ('86400' in str(room.room_id))
                        grid_floor = 3
                        if '4x6' in str(room.board_dimensions): grid_floor = 4
                        elif '5x7' in str(room.board_dimensions): grid_floor = 5
                        elif '6x8' in str(room.board_dimensions) or '3x3x3' in str(room.board_dimensions): grid_floor = 6

                        pre_pop = None
                        if room.is_solo and getattr(room, 'initial_solo_params', None):
                            isp = room.initial_solo_params
                            s_sp = room.build_solo_spinner_params()
                            s_dict = s_sp.get('dictionary', 'NWL')
                            s_fmt = s_sp.get('board_format', 'Normal')
                            s_min_len = int(s_sp.get('min_word_length', 3))
                            s_use_aw = bool(s_sp.get('use_added_words', False))
                            if '+ AW' in str(s_dict) or '+AW' in str(s_dict) or s_dict == 'AW':
                                s_use_aw = True
                                s_dict = str(s_dict).replace('+ AW', '').replace('+AW', '').strip()
                                if s_dict == 'AW': s_dict = 'NWL'
                            s_bw_len = int(s_sp.get('bonus_word_length', 8) or 8)
                            s_diff = SpinnerSet.strip_difficulty_pct(s_sp.get('difficulty', 'Medium'))
                            s_wc = s_sp.get('word_count_range', '100-200')
                            
                            if is_24h_room:
                                s_fmt = 'Valued Letters'
                                s_wc = '300-400'
                                s_use_aw = False
                                s_min_len = grid_floor
                                s_dict = 'CSW' if 'CSW' in str(s_dict).upper() else 'NWL'
                                is_explicit_fmt = True

                            is_explicit_diff = bool(isp.get('difficulty') and str(isp.get('difficulty')).lower() != 'random')
                            is_explicit_fmt = bool(is_24h_room or (isp.get('board_format') and str(isp.get('board_format')).lower() != 'random'))

                            if is_24h_room:
                                pre_pop = pop_compatible_cached_board(room.board_dimensions, s_dict, 'Valued Letters', grid_floor, False, bonus_word_len=s_bw_len, difficulty=s_diff, word_count_range='300-400')
                                if not pre_pop:
                                    pre_pop = pop_compatible_cached_board(room.board_dimensions, s_dict, 'Valued Letters', grid_floor, False, difficulty=s_diff, word_count_range='300-400')
                                if not pre_pop:
                                    pre_pop = pop_compatible_cached_board(room.board_dimensions, s_dict, 'Valued Letters', grid_floor, False, word_count_range='300-400')
                                if not pre_pop:
                                    pre_pop = pop_any_cached_board(room.board_dimensions, min_word_length=grid_floor, target_format='Valued Letters', dictionary=s_dict, use_added_words=False, word_count_range='300-400')
                                if not pre_pop:
                                    pre_pop = pop_any_cached_board(room.board_dimensions, min_word_length=grid_floor, word_count_range='300-400')
                                if not pre_pop:
                                    pre_pop = pop_any_cached_board(room.board_dimensions, word_count_range='300-400')
                                if pre_pop:
                                    _cand_words = pre_pop[1] if len(pre_pop) > 1 else []
                                    _cand_cnt = sum(1 for w in _cand_words if len(w) >= grid_floor)
                                    if not (300 <= _cand_cnt <= 399):
                                        print(f"[RoomManager] Popped candidate discarded for 24h room {room_id}: had {_cand_cnt} words (needed 300-399).")
                                        pre_pop = None
                            else:
                                # Fast non-blocking cache cascade for instant solo room creation
                                # Tier 1: Strict match with target difficulty and word count
                                pre_pop = pop_compatible_cached_board(room.board_dimensions, s_dict, s_fmt, s_min_len, s_use_aw, bonus_word_len=s_bw_len, difficulty=s_diff, word_count_range=s_wc)
                                if not pre_pop:
                                    # Tier 2: Match target difficulty without strict word count
                                    pre_pop = pop_compatible_cached_board(room.board_dimensions, s_dict, s_fmt, s_min_len, s_use_aw, bonus_word_len=s_bw_len, difficulty=s_diff)
                                if not pre_pop:
                                    # Tier 3: Match target difficulty without bonus word len restriction
                                    pre_pop = pop_compatible_cached_board(room.board_dimensions, s_dict, s_fmt, s_min_len, s_use_aw, difficulty=s_diff)
                            
                                if not pre_pop and not is_explicit_fmt:
                                    # If format was random, try Normal format with target difficulty
                                    pre_pop = pop_compatible_cached_board(room.board_dimensions, s_dict, 'Normal', s_min_len, s_use_aw, difficulty=s_diff)

                                # ONLY if difficulty was random can we drop difficulty to pop any cached board
                                if not pre_pop and not is_explicit_diff:
                                    if not is_explicit_fmt:
                                        pre_pop = pop_compatible_cached_board(room.board_dimensions, s_dict, s_fmt, s_min_len, s_use_aw)
                                        if not pre_pop:
                                            pre_pop = pop_any_cached_board(room.board_dimensions, min_word_length=s_min_len, target_format=s_fmt, dictionary=s_dict, use_added_words=s_use_aw, word_count_range=s_wc)
                                        if not pre_pop:
                                            pre_pop = pop_any_cached_board(room.board_dimensions, min_word_length=s_min_len, target_format=s_fmt, dictionary=s_dict, use_added_words=s_use_aw)
                                        if not pre_pop:
                                            pre_pop = pop_any_cached_board(room.board_dimensions, min_word_length=s_min_len)
                                        if not pre_pop:
                                            pre_pop = pop_any_cached_board(room.board_dimensions)

                            _dims_parts = str(room.board_dimensions).lower().split('x')
                            _d0 = int(_dims_parts[0]) if len(_dims_parts) == 3 else 1
                            _r0 = int(_dims_parts[1] if len(_dims_parts) == 3 else _dims_parts[0])
                            _c0 = int(_dims_parts[2] if len(_dims_parts) == 3 else _dims_parts[1])

                            # STRICT AUDIT: If difficulty was explicitly requested, ensure popped board's actual uniqueness ratio matches!
                            if pre_pop and is_explicit_diff and not is_24h_room:
                                _chk_ratio = pre_pop[5]
                                _act_diff = self.board_generator.get_difficulty_label(_chk_ratio, _r0, _c0, s_dict, depth=_d0, min_word_length=s_min_len)
                                if _act_diff != s_diff:
                                    print(f"[RoomManager] Popped cached board actual difficulty {_act_diff} != requested {s_diff}. Discarding pop...")
                                    pre_pop = None

                            if pre_pop and is_explicit_fmt and not is_24h_room:
                                _chk_fmt = pre_pop[3]
                                if str(_chk_fmt).lower() != str(s_fmt).lower():
                                    print(f"[RoomManager] Popped cached board actual format {_chk_fmt} != requested {s_fmt}. Discarding pop...")
                                    pre_pop = None

                            # If cache had no compliant board, perform synchronous live generation with exact parameters
                            if not pre_pop and (is_explicit_diff or is_explicit_fmt or is_24h_room):
                                print(f"[RoomManager] No cached board strictly matching {s_diff} / {s_fmt} for {room_id}. Performing live generation...")
                                bonus_w_gen = self._get_bonus_word(length=s_bw_len, dictionary=s_dict) if hasattr(self, '_get_bonus_word') else 'PLANETS'
                                live_res = self.board_generator.generate_board(
                                    dimensions=room.board_dimensions,
                                    bonus_word=bonus_w_gen,
                                    word_count_range='300-400' if is_24h_room else s_wc,
                                    board_format='Valued Letters' if is_24h_room else s_fmt,
                                    dictionary=s_dict,
                                    min_word_length=grid_floor if is_24h_room else s_min_len,
                                    difficulty=s_diff,
                                    timeout=8.0 if is_24h_room else 4.0,
                                    use_added_words=False if is_24h_room else s_use_aw
                                )
                                if live_res and len(live_res) >= 6:
                                    lb_b, lb_w, lb_c, lb_f, lb_p, lb_r = live_res[:6]
                                    lb_bw = live_res[6] if len(live_res) >= 7 else bonus_w_gen
                                    effective_min = grid_floor if is_24h_room else s_min_len
                                    lb_w_filt = [w for w in lb_w if len(w) >= effective_min]
                                    if not is_24h_room or (300 <= len(lb_w_filt) <= 399):
                                        lb_params = {
                                            'dictionary': s_dict,
                                            'board_format': 'Valued Letters' if is_24h_room else (lb_f or s_fmt),
                                            'min_word_length': effective_min,
                                            'word_count_range': '300-400' if is_24h_room else s_wc,
                                            'difficulty': s_diff,
                                            'use_added_words': False if is_24h_room else s_use_aw,
                                            'bonus_word_len': len(lb_bw),
                                            'dimensions': room.board_dimensions
                                        }
                                        pre_pop = (lb_b, lb_w_filt, lb_c, lb_params['board_format'], {w: p for w, p in lb_p.items() if w in lb_w_filt}, lb_r, lb_bw, lb_params)
                                        print(f"[RoomManager] Live generation succeeded for {room_id}: {lb_params['board_format']} {s_diff} ({lb_r:.2%}, {len(lb_w_filt)} words)")

                            if not pre_pop:
                                pre_pop = get_emergency_fallback_board(
                                    room.board_dimensions, 'Valued Letters' if is_24h_room else s_fmt, room.time_limit,
                                    dictionary=s_dict, use_added_words=False if is_24h_room else s_use_aw, target_range='300-400' if is_24h_room else s_wc, min_word_length=grid_floor if is_24h_room else s_min_len, difficulty=s_diff
                                )
                        else:
                            sp_kick = SpinnerSet.generate_params(room.board_dimensions, is_24h=is_24h_room)
                            k_dict = sp_kick.get('dictionary', 'NWL')
                            k_aw = sp_kick.get('use_added_words', False)
                            k_fmt = sp_kick.get('board_format', 'Normal')
                            k_min = sp_kick.get('min_word_length', 3)
                            k_wc = sp_kick.get('word_count_range')
                            if is_24h_room:
                                k_dict = 'CSW' if 'CSW' in str(k_dict).upper() else 'NWL'
                                k_aw = False
                                k_fmt = 'Valued Letters'
                                k_min = grid_floor
                                k_wc = '300-400'
                                pre_pop = pop_compatible_cached_board(room.board_dimensions, k_dict, 'Valued Letters', grid_floor, False, word_count_range='300-400')
                                if not pre_pop:
                                    pre_pop = pop_any_cached_board(room.board_dimensions, min_word_length=grid_floor, target_format='Valued Letters', dictionary=k_dict, use_added_words=False, word_count_range='300-400')
                                if not pre_pop:
                                    pre_pop = pop_any_cached_board(room.board_dimensions, min_word_length=grid_floor, word_count_range='300-400')
                                if not pre_pop:
                                    pre_pop = pop_any_cached_board(room.board_dimensions, word_count_range='300-400')
                                if pre_pop:
                                    _cand_words = pre_pop[1] if len(pre_pop) > 1 else []
                                    _cand_cnt = sum(1 for w in _cand_words if len(w) >= grid_floor)
                                    if not (300 <= _cand_cnt <= 399):
                                        print(f"[RoomManager] Popped candidate discarded for 24h room {room_id}: had {_cand_cnt} words (needed 300-399).")
                                        pre_pop = None
                                if not pre_pop:
                                    pre_pop = get_emergency_fallback_board(
                                        room.board_dimensions, 'Valued Letters', room.time_limit,
                                        dictionary=k_dict, use_added_words=False, target_range='300-400', min_word_length=grid_floor, difficulty=sp_kick.get('difficulty', 'Medium')
                                    )
                            else:
                                pre_pop = pop_compatible_cached_board(room.board_dimensions, k_dict, k_fmt, k_min, k_aw, word_count_range=k_wc)
                                if not pre_pop:
                                    pre_pop = pop_any_cached_board(room.board_dimensions, min_word_length=k_min, target_format=k_fmt, dictionary=k_dict, use_added_words=k_aw, word_count_range=k_wc)
                                if not pre_pop:
                                    pre_pop = pop_any_cached_board(room.board_dimensions, min_word_length=k_min, target_format=k_fmt, dictionary=k_dict, use_added_words=k_aw)
                                if not pre_pop:
                                    pre_pop = pop_any_cached_board(room.board_dimensions, min_word_length=k_min)
                                if not pre_pop:
                                    pre_pop = pop_any_cached_board(room.board_dimensions)
                                if not pre_pop:
                                    pre_pop = get_emergency_fallback_board(
                                        room.board_dimensions, k_fmt, room.time_limit,
                                        dictionary=k_dict, use_added_words=k_aw, target_range=k_wc, min_word_length=k_min, difficulty=sp_kick.get('difficulty', 'Medium')
                                    )

                        if pre_pop:
                            if len(pre_pop) >= 9:
                                r_board, r_words, r_bonus_c, r_fmt, r_dict, r_ratio, r_bonus_word, _, r_params = pre_pop
                            else:
                                r_board, r_words, r_bonus_c, r_fmt, r_dict, r_ratio, r_bonus_word, r_params = pre_pop
                            
                            r_params = dict(r_params) if isinstance(r_params, dict) else {}
                            board_min_l = r_params.get('min_word_length', 3)
                            grid_floor = 3
                            if '4x6' in room.board_dimensions: grid_floor = 4
                            elif '5x7' in room.board_dimensions: grid_floor = 5
                            elif '6x8' in room.board_dimensions or '3x3x3' in room.board_dimensions: grid_floor = 6
                            board_min_l = max(grid_floor, int(board_min_l) if board_min_l is not None else grid_floor)
                            if is_24h_room:
                                board_min_l = grid_floor

                            # Derive scorable words & paths
                            raw_dict = 'NWL' if (is_24h_room and 'CSW' not in str(r_params.get('dictionary', '')).upper()) else r_params.get('dictionary', 'NWL')
                            raw_aw = False if is_24h_room else r_params.get('use_added_words', False)
                            room.all_words_paths = {w: p for w, p in (r_dict or {}).items() if len(w) >= board_min_l and word_validator.word_validator.is_valid_word(w, raw_dict, use_added_words=raw_aw)}
                            room.all_words = set(room.all_words_paths.keys())

                            # Guarantee factual min_word_length: Align to actual shortest word on this board
                            if room.all_words and not is_24h_room:
                                actual_shortest = min(len(w) for w in room.all_words)
                                if actual_shortest > board_min_l:
                                    print(f"[RoomManager] create_room aligned min_word_length from {board_min_l} to factual {actual_shortest}.")
                                    board_min_l = actual_shortest

                            # 24H IRONCLAD RESCUE: If actual words count is not strictly in 300..399, rescue!
                            if is_24h_room and not (300 <= len(room.all_words) <= 399):
                                print(f"[RoomManager] {room_id}: 24h room board has {len(room.all_words)} words (strictly requires 300-399). Rescuing with compliant fallback...")
                                h_fallback = get_emergency_fallback_board(
                                    room.board_dimensions, 'Valued Letters', room.time_limit,
                                    dictionary=raw_dict, use_added_words=False, target_range='300-400', min_word_length=grid_floor
                                )
                                if h_fallback and len(h_fallback) >= 7:
                                    r_board, r_words, r_bonus_c, r_fmt, r_p, r_ratio, r_bonus_word = h_fallback[:7]
                                    room.all_words_paths = {w: p for w, p in r_p.items() if len(w) >= grid_floor}
                                    room.all_words = set(room.all_words_paths.keys())
                                    if len(room.all_words) > 399:
                                        bw_keep = {r_bonus_word} if r_bonus_word else set()
                                        kept_w = list(bw_keep | set(list(room.all_words)[:399]))
                                        room.all_words = set(kept_w)
                                        room.all_words_paths = {w: p for w, p in room.all_words_paths.items() if w in room.all_words}

                            # Derive truthful word count range label
                            wc_cnt = len(room.all_words)
                            if is_24h_room:
                                wc_lbl = '300-400'
                            else:
                                wc_lbl = SpinnerSet.get_factchecked_wc_range(wc_cnt, use_added_words=raw_aw, dictionary=raw_dict)

                            # Construct spinner params ONCE from exact board parameters
                            _dims_parts = str(room.board_dimensions).lower().split('x')
                            _d0 = int(_dims_parts[0]) if len(_dims_parts) == 3 else 1
                            _r0 = int(_dims_parts[1] if len(_dims_parts) == 3 else _dims_parts[0])
                            _c0 = int(_dims_parts[2] if len(_dims_parts) == 3 else _dims_parts[1])
                            _actual_diff_base = self.board_generator.get_difficulty_label(r_ratio, _r0, _c0, raw_dict, depth=_d0, min_word_length=board_min_l)
                            if room.is_solo and getattr(room, 'initial_solo_params', None):
                                _s_d_param = room.initial_solo_params.get('difficulty')
                                if _s_d_param and str(_s_d_param).lower() != 'random':
                                    _actual_diff_base = SpinnerSet.strip_difficulty_pct(_s_d_param)
                            _create_diff_lbl = SpinnerSet.format_difficulty_with_pct(_actual_diff_base, r_ratio)
                            _target_bw_create = len(r_bonus_word) if r_bonus_word else 8
                            if room.is_solo and getattr(room, 'initial_solo_params', None):
                                if 's_bw_len' in locals():
                                    _target_bw_create = s_bw_len
                            elif 'sp_kick' in locals() and isinstance(sp_kick, dict):
                                _target_bw_create = int(sp_kick.get('bonus_word_length', 8) or 8)

                            room.spinner_params = {
                                'dictionary': raw_dict,
                                'board_format': r_fmt or 'Normal',
                                'min_word_length': board_min_l,
                                'word_count_range': wc_lbl,
                                'difficulty': _create_diff_lbl,
                                'use_added_words': raw_aw,
                                'bonus_word_length': _target_bw_create,
                                'board_dimensions': room.board_dimensions,
                                'time_limit': room.time_limit,
                                'generated_at': time.time(),
                                '_exact_wc_calculated': True
                            }
                            room.spinner_params = SpinnerSet.sanitize_params(room.spinner_params, room.board_dimensions, room.time_limit >= 7200)
                            if is_24h_room:
                                wc_lbl = '300-400'
                                board_min_l = grid_floor
                                room.spinner_params['word_count_range'] = '300-400'
                                room.spinner_params['board_format'] = 'Valued Letters'
                                room.spinner_params['min_word_length'] = grid_floor
                                room.spinner_params['use_added_words'] = False
                                room.current_min_length = grid_floor
                                room.current_word_count_range = '300-400'
                                room.current_board_format = 'Valued Letters'
                                room.use_added_words = False
                            else:
                                wc_lbl = room.spinner_params.get('word_count_range', wc_lbl)
                                room.current_min_length = board_min_l
                                room.current_difficulty = _create_diff_lbl
                                room.current_board_format = r_fmt or 'Normal'
                                room.current_word_count_range = wc_lbl
                                room.current_dictionary = room.spinner_params.get('dictionary', raw_dict)
                                room.current_uniqueness = r_ratio
                                room.use_added_words = room.spinner_params.get('use_added_words', raw_aw)

                            room.bonus_word = pick_distinct_bonus_word(
                                room, r_bonus_word, room.all_words, _target_bw_create,
                                dictionary=raw_dict, is_checkerboard=('checkerboard' in str(r_fmt).lower()),
                                room_manager=self
                            )
                            ensure_bonus_word_on_board(room, _target_bw_create)
                            room.previous_bonus_word = ''
                            # CRITICAL: Always keep spinner_params bonus_word_length, min_word_length, and word_count_range strictly synchronized with the actual board!
                            room.spinner_params['bonus_word_length'] = len(room.bonus_word)
                            room.spinner_params['word_count_range'] = wc_lbl
                            room.spinner_params['min_word_length'] = board_min_l
                            room.current_min_length = board_min_l
                            import copy
                            room.frozen_revealed_params = copy.deepcopy(room.spinner_params)
                            room.board = r_board
                            room.bonus_cell = r_bonus_c
                            room.total_words_count = len(room.all_words)
                            room.initial_total_words = room.total_words_count

                            room.round_start_time = time.time()
                            room.state = 'active'
                            room.current_round = 1
                            room._initial_board_delivered = True
                            room.last_round_had_winner = False
                            room.winners_history = [h for h in getattr(room, 'winners_history', []) if h.get('round') != 1]
                            room.last_saved_round = -1

                            room.update_counts_by_len()
                            room.initialize_density(r_board, room.all_words_paths, r_fmt)
                            room.recalculate_total_points()
                            
                            # Proactive background search for Round 2 (skip for solo since create_solo_match runs start_round immediately)
                            if not getattr(room, 'is_solo', False):
                                self.pre_generate_next_round(room_id)
                            print(f"[RoomManager] {room_id} kickstarted ACTIVE in single-pass with truthful board & params.")
                    else:
                        if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                            room.spinner_params = room.build_solo_spinner_params()
                            room.state = 'loading'
                        else:
                            room.spinner_params = SpinnerSet.generate_params(room.board_dimensions, is_24h, is_split)
                            room.state = 'intermission'
                            room.intermission_start_time = time.time()
                        if room_id.startswith('pub_') and not is_24h:
                             threading.Thread(target=self.start_board_search, args=(room_id,), daemon=True).start()

                return room
        except Exception as e:
            import traceback
            print(f"[RoomManager] CRITICAL ERROR in create_room: {e}\n{traceback.format_exc()}")
            raise
    
    def _apply_kickstart_results(self, room_id, e_results, m_len, is_24h):
        """Apply board generation results to a room that was in 'loading' state."""
        import threading
        room = self.get_room(room_id)
        if not room:
            return
        
        with room._state_lock:
            if room.state == 'active' and room.board and len(room.board) > 0:
                print(f"[RoomManager] Aborting _apply_kickstart_results for {room_id}: Room is ALREADY active with board!")
                return
        
        e_board, e_words, e_bonus_c, e_fmt, e_dict, e_ratio, e_bonus_word = e_results[:7]
        e_params = e_results[8] if len(e_results) > 8 else None
        
        room.board = e_board
        room.bonus_cell = e_bonus_c
        room.bonus_word = e_bonus_word or getattr(room, 'bonus_word', '')
        
        if e_params:
            print(f"[RoomManager] Aligning ultimate kickstart spinner parameters to match fallback board: {e_params}")
            room.spinner_params['dictionary'] = e_params.get('dictionary', 'NWL')
            room.spinner_params['difficulty'] = e_params.get('difficulty', 'Medium')
            room.spinner_params['word_count_range'] = e_params.get('word_count_range', '100-200')
            room.spinner_params['board_format'] = e_params.get('board_format', 'Normal')
            room.spinner_params['min_word_length'] = e_params.get('min_word_length', 3)
            room.spinner_params['use_added_words'] = e_params.get('use_added_words', False)
            room.spinner_params['bonus_word_length'] = e_params.get('bonus_word_len', 6)
            m_len = int(e_params.get('min_word_length', 3))
        else:
            if e_words:
                actual_shortest = min(len(w) for w in e_words)
                room.spinner_params['min_word_length'] = actual_shortest
                m_len = actual_shortest

        # Guarantee factual min_word_length: Align to actual shortest word on this board
        if e_words:
            actual_shortest = min(len(w) for w in e_words)
            if actual_shortest > m_len:
                print(f"[RoomManager] _apply_kickstart_results aligned min_word_length from {m_len} to factual {actual_shortest}.")
                m_len = actual_shortest
                room.spinner_params['min_word_length'] = actual_shortest

        room.current_min_length = m_len
        room.current_board_format = 'Valued Letters' if is_24h else (e_params.get('board_format') if e_params else e_fmt)
        room.current_word_count_range = room.spinner_params.get('word_count_range', '100-200')
        room.current_dictionary = room.spinner_params.get('dictionary', 'NWL')
        room.current_uniqueness = e_ratio
        room.use_added_words = room.spinner_params.get('use_added_words', False)
        room.all_words = {w for w in (e_words or []) if len(w) >= m_len}
        room.all_words_paths = {w: p for w, p in (e_dict or {}).items() if len(w) >= m_len}
        
        if hasattr(word_validator, 'word_validator'):
            room.csw_only_words = [w for w in room.all_words if word_validator.word_validator.is_csw_only(w)]
            room.added_words = [w for w in room.all_words if word_validator.word_validator.is_added_word(w)]
        else:
            room.csw_only_words = []
            room.added_words = []
        
        is_valued_kick = ('valued' in str(e_fmt).lower())
        kick_scores = {}
        for w in room.all_words:
            if is_valued_kick:
                kick_scores[w] = {'total': get_valued_word_score(w), 'base': get_valued_word_score(w)}
            else:
                length = len(w)
                if length <= 4: s = 1
                elif length == 5: s = 2
                elif length == 6: s = 3
                elif length == 7: s = 5
                elif length == 8: s = 9
                elif length == 9: s = 13
                elif length == 10: s = 17
                elif length == 11: s = 21
                elif length == 12: s = 25
                elif length == 13: s = 29
                elif length == 14: s = 33
                else: s = 37
                kick_scores[w] = {'total': s, 'base': s}
        room.solved_words_with_scores = kick_scores
        # Sync counts and word_count_range label to the actual filtered set
        room.update_counts_by_len()
        _akc = room.total_words_count
        room.current_word_count_range = self._get_factchecked_wc_range(_akc, use_added_words=room.use_added_words, dictionary=room.current_dictionary)

        room.round_start_time = time.time()
        room.state = 'active'
        room.current_round = 1
        room.last_round_had_winner = False
        room.winners_history = [h for h in getattr(room, 'winners_history', []) if h.get('round') != 1]
        room.last_saved_round = -1
        room.initialize_density(e_board, room.all_words_paths, e_fmt)
        room.recalculate_total_points()
        room.spinner_params_generated = True
        
        def refine_kickstart_scores():
            try:
                from scoring import calculate_word_score
                refined = {}
                for word in room.all_words:
                    refined[word] = calculate_word_score(
                        word, room.bonus_word, path=room.all_words_paths.get(word),
                        board_format=room.current_board_format, bonus_cell=room.bonus_cell,
                        board=room.board, return_details=True, strict_path=True
                    )
                room.solved_words_with_scores = refined
                room.recalculate_total_points()
            except Exception as e:
                print(f"[RoomManager] Kickstart refinement error for {room_id}: {e}")
        
        threading.Thread(target=refine_kickstart_scores, daemon=True).start()
        threading.Thread(target=self.start_board_search, args=(room_id,), daemon=True).start()
        print(f"[RoomManager] {room_id}: Async kickstart applied. Room now ACTIVE (Round 1, {m_len}L+).")
    
    def get_yesterdays_history(self, room, current_round):
        """Recover history for a 24h room from the database (Fallback)"""
        if not room: return {}
        
        # 1. OPTIMIZATION: If in-memory state is already populated, return it immediately
        if room.previous_day_history and len(room.previous_day_history) > 0:
            return room.previous_day_history
            
        import json
        import datetime
        try:
            room_id = room.room_id
            curr_round = room.current_round
            thirty_six_hours_ago = (datetime.datetime.now() - datetime.timedelta(hours=36)).strftime('%Y-%m-%d %H:%M:%S')

            with get_db() as conn:
                # If current_round is 1 or 2 (new server session), find the latest from history
                if curr_round <= 2:
                    cursor = conn.execute('''
                        SELECT rh.user_id, rh.words_json, rh.round_number, rh.timestamp, rh.board_json, rh.bonus_word, rh.bonus_cell, rh.board_format, rh.all_solutions_json, rh.all_words_paths, u.username
                        FROM round_history rh
                        LEFT JOIN users u ON rh.user_id = u.id
                        WHERE rh.room_id = ? AND rh.timestamp >= ?
                        ORDER BY rh.timestamp DESC, rh.id DESC
                    ''', (room_id, thirty_six_hours_ago))
                else:
                    cursor = conn.execute('''
                        SELECT rh.user_id, rh.words_json, rh.round_number, rh.timestamp, rh.board_json, rh.bonus_word, rh.bonus_cell, rh.board_format, rh.all_solutions_json, rh.all_words_paths, u.username
                        FROM round_history rh
                        LEFT JOIN users u ON rh.user_id = u.id
                        WHERE rh.room_id = ? AND rh.round_number < ? AND rh.timestamp >= ?
                        ORDER BY rh.timestamp DESC, rh.round_number DESC
                    ''', (room_id, curr_round, thirty_six_hours_ago))
                
                rows = cursor.fetchall()
                
                if not rows:
                    parts = room_id.split('_')
                    dims = parts[2] if len(parts) >= 3 else room.board_dimensions
                    cursor = conn.execute('''
                        SELECT rh.user_id, rh.words_json, rh.round_number, rh.timestamp, rh.board_json, rh.bonus_word, rh.bonus_cell, rh.board_format, rh.all_solutions_json, rh.all_words_paths, u.username
                        FROM round_history rh
                        LEFT JOIN users u ON rh.user_id = u.id
                        WHERE rh.board_dimensions = ? AND rh.game_type = ? AND rh.timestamp >= ?
                        ORDER BY rh.timestamp DESC
                    ''', (dims, room.game_type, thirty_six_hours_ago))
                    rows = cursor.fetchall()

            # INITIALIZE correctly to avoid UnboundLocalError if no rows exist
            recovered_board = None
            recovered_bonus_word = None
            recovered_bonus_cell = None
            recovered_format = 'Valued Letters' if room.time_limit >= 7200 else 'Normal'
            recovered_solutions = None
            recovered_paths = None
            history = {}

            # First pass: extract board metadata
            for row in rows:
                uid, words_json, round_num, ts, b_json, b_word, b_cell_json, b_format, sols_json, paths_json = row[:10]
                if b_json and not recovered_board:
                    recovered_board = json.loads(b_json)
                    recovered_bonus_word = b_word
                    recovered_bonus_cell = json.loads(b_cell_json) if b_cell_json else None
                    recovered_format = b_format
                
                if sols_json and not recovered_solutions:
                    try:
                        recovered_solutions = json.loads(sols_json)
                    except: pass
                    
                if paths_json and not recovered_paths:
                    try:
                        recovered_paths = json.loads(paths_json)
                    except: pass

            # Second pass: construct player history with dynamic recalculation for 24h rooms
            for row in rows:
                uid, words_json, round_num, ts, b_json, b_word, b_cell_json, b_format, sols_json, paths_json = row[:10]
                db_username = row[10] if len(row) > 10 else None
                uid_str = str(uid)
                if uid_str not in history:
                    if uid == -1:
                        uname = "System"
                    elif uid < 0:
                        uname = f"Guest_{abs(uid)}"
                    else:
                        uname = db_username if db_username else f"User {uid}"
                    
                    parsed_words = json.loads(words_json) if words_json else []
                    if room.time_limit >= 7200 and parsed_words and recovered_board:
                        from scoring import calculate_word_score
                        recalculated_words = []
                        for w_item in parsed_words:
                            if isinstance(w_item, dict):
                                word_str = w_item.get('word', '')
                            else:
                                word_str = str(w_item)
                            
                            word_upper = word_str.upper()
                            w_path = None
                            if recovered_paths:
                                w_path = recovered_paths.get(word_upper) or recovered_paths.get(word_str)
                            
                            details = calculate_word_score(
                                word_upper,
                                recovered_bonus_word,
                                board_format='Valued Letters',
                                bonus_cell=recovered_bonus_cell,
                                board=recovered_board,
                                path=w_path,
                                return_details=True,
                                strict_path=True
                            )
                            
                            recalculated_words.append({
                                'word': word_str,
                                'points': details.get('total', 0) if details else 0,
                                'timestamp': w_item.get('timestamp') if isinstance(w_item, dict) else time.time(),
                                'score_details': details
                            })
                        history[uid_str] = {
                            'username': uname,
                            'found_words': recalculated_words
                        }
                    else:
                        history[uid_str] = {
                            'username': uname,
                            'found_words': parsed_words
                        }
                    
                    # BACKWARD COMPATIBILITY: Also populate player objects if they are currently in the room
                    for p in room.players:
                        if p.user_id == uid:
                            p.previous_submitted_words = history[uid_str]['found_words']
            
            # 2. POPULATE ROOM STATE: Reconstruct full previous round state if board recovered
            if recovered_board:
                print(f"[RoomManager] Recovering board for room {room_id} from DB Fallback")
                room.previous_board = recovered_board
                room.previous_day_history = history # Cache for next call
                room.previous_bonus_word = recovered_bonus_word
                
                # USE STORED SOLUTIONS (Prevents dictionary mismatch issues)
                if recovered_solutions:
                     if isinstance(recovered_solutions, dict):
                         room.previous_all_words = list(recovered_solutions.keys())
                         if room.time_limit >= 7200:
                             from scoring import calculate_word_score
                             recalc_scores = {}
                             for w in room.previous_all_words:
                                 w_upper = w.upper()
                                 w_path = recovered_paths.get(w_upper) or recovered_paths.get(w)
                                 recalc_scores[w_upper] = calculate_word_score(
                                     w_upper,
                                     room.previous_bonus_word,
                                     board_format='Valued Letters',
                                     bonus_cell=recovered_bonus_cell,
                                     board=room.previous_board,
                                     path=w_path,
                                     return_details=True,
                                     strict_path=True
                                 )
                             room.previous_all_word_scores = recalc_scores
                         else:
                             room.previous_all_word_scores = recovered_solutions
                     elif isinstance(recovered_solutions, list):
                         room.previous_all_words = list(recovered_solutions)
                         if room.time_limit >= 7200:
                             from scoring import calculate_word_score
                             recalc_scores = {}
                             for w in room.previous_all_words:
                                 w_upper = w.upper()
                                 w_path = recovered_paths.get(w_upper) or recovered_paths.get(w)
                                 recalc_scores[w_upper] = calculate_word_score(
                                     w_upper,
                                     room.previous_bonus_word,
                                     board_format='Valued Letters',
                                     bonus_cell=recovered_bonus_cell,
                                     board=room.previous_board,
                                     path=w_path,
                                     return_details=True,
                                     strict_path=True
                                 )
                             room.previous_all_word_scores = recalc_scores
                         else:
                             room.previous_all_word_scores = {w: {} for w in recovered_solutions}
                     
                     # Recover min length from solutions
                     word_lengths = [len(w) for w in room.previous_all_words if len(w) >= 3]
                     room.previous_min_length = min(word_lengths) if word_lengths else (4 if room.board_dimensions == '6x8' else 3)
                     
                     if hasattr(word_validator, 'word_validator'):
                         room.previous_csw_only_words = [w for w in room.previous_all_words if word_validator.word_validator.is_csw_only(w)]
                         room.previous_added_words = [w for w in room.previous_all_words if word_validator.word_validator.is_added_word(w)]
                     else:
                         room.previous_csw_only_words = []
                         room.previous_added_words = []
                         
                     print(f"[RoomManager] Recovered {len(room.previous_all_words)} words from DB solutions.")
                else:
                    # Fallback solve (NWL)
                    from board_generator import solve_board
                    min_len = 4 if room.board_dimensions == '6x8' else 3
                    room.previous_min_length = min_len
                    dictionary = 'NWL'
                    try:
                        all_solutions = solve_board(recovered_board, dictionary)
                        bonus_upper = recovered_bonus_word.upper() if recovered_bonus_word else None
                        room.previous_all_words = [w for w in all_solutions if (len(w) >= min_len or (bonus_upper and w.upper() == bonus_upper))]
                        if bonus_upper and bonus_upper not in room.previous_all_words:
                            room.previous_all_words.append(bonus_upper)
                        room.previous_all_word_scores = {w: {} for w in room.previous_all_words}
                        
                        if hasattr(word_validator, 'word_validator'):
                            room.previous_csw_only_words = [w for w in room.previous_all_words if word_validator.word_validator.is_csw_only(w)]
                            room.previous_added_words = [w for w in room.previous_all_words if word_validator.word_validator.is_added_word(w)]
                        else:
                            room.previous_csw_only_words = []
                            room.previous_added_words = []
                    except Exception as e:
                        print(f"[RoomManager] Error solving recovered board: {e}")
                        room.previous_all_words = []
                        room.previous_all_word_scores = {}
                        room.previous_csw_only_words = []
                        room.previous_added_words = []

            return history
        except Exception as e:
            print(f"[RoomManager] Error fetching yesterday's history for {room.room_id}: {e}")
            return {}

    def get_room(self, room_id):
        """Get room by ID"""
        with self.lock:
            return self.rooms.get(room_id)
    
    def get_online_count(self):
        """Returns the number of users active in the last 60 seconds"""
        with self.lock:
            now = time.time()
            return sum(1 for ts in self.user_presence.values() if (now - ts) < 60)

    def update_presence(self, user_id):
        """Update global heartbeat for any user interaction"""
        if user_id:
            with self.lock:
                self.user_presence[str(user_id)] = time.time()

    def remove_presence(self, user_id):
        """Immediately mark user as offline and remove from all rooms (for logout/beacon)"""
        if not user_id: return
        uid_str = str(user_id)
        
        with self.lock:
            if uid_str in self.user_presence:
                del self.user_presence[uid_str]
        
        try:
            if 'lobby_manager' in globals():
                lobby_manager.remove_user(user_id)
        except Exception:
            pass
        
        # USER REQUEST: Immediate removal from all rooms to prevent "zombie" rooms
        rooms_to_delete = []
        for ri, room in list(self.rooms.items()):
             # Only remove and check for deletion if it's NOT a 24h room
             is_daily = (room.time_limit >= 7200)
             
             # Attempt removal first
             room.remove_player(user_id, force=True)
             
             # Re-check room occupancy
             humans = [p for p in room.players if not p.is_ai]
             is_public = ri.startswith('pub_')
             if len(humans) == 0 and len(room.spectators) == 0 and not is_daily and not is_public:
                  print(f"[RoomManager] Immediate cleanup: Room {ri} is empty after user {user_id} left. Deleting.")
                  rooms_to_delete.append(ri)
                  
        for ri in rooms_to_delete:
             self.delete_room(ri)

    def find_user_session(self, user_id):
        """Find user's current room and online status. Strictly prioritizes non-24h rooms."""
        uid_str = str(user_id)
        now = time.time()
        
        # Check global presence first
        last_seen = self.user_presence.get(uid_str, 0)
        is_online = (now - last_seen) < 75 # 75 seconds (reduced for better accuracy)
        
        # Search for active room:
        # 1. Non-24h standard active room (highest priority)
        # 2. 24h room (only if user is in NO other room)
        non_24h_match = None
        non_24h_max_active = -1
        
        fallback_24h_match = None
        fallback_24h_max_active = -1
        
        for room in self.rooms.values():
            is_24h = (getattr(room, 'time_limit', 0) >= 7200)

            # Check players
            for p in room.players:
                if str(p.user_id) == uid_str:
                    if is_24h:
                        if p.last_active > fallback_24h_max_active:
                            fallback_24h_max_active = p.last_active
                            fallback_24h_match = {
                                'room_id': room.room_id,
                                'is_online': True,
                                'is_spectator': False,
                                'game_type': room.game_type,
                                'board_dimensions': room.board_dimensions,
                                'time_limit': room.time_limit,
                                'is_24h': True
                            }
                    else:
                        if p.last_active > non_24h_max_active:
                            non_24h_max_active = p.last_active
                            non_24h_match = {
                                'room_id': room.room_id,
                                'is_online': True,
                                'is_spectator': False,
                                'game_type': room.game_type,
                                'board_dimensions': room.board_dimensions,
                                'time_limit': room.time_limit,
                                'is_24h': False
                            }

            # Check spectators
            for s in room.spectators:
                if str(s.user_id) == uid_str:
                    if is_24h:
                        if s.last_active > fallback_24h_max_active:
                            fallback_24h_max_active = s.last_active
                            fallback_24h_match = {
                                'room_id': room.room_id,
                                'is_online': True,
                                'is_spectator': True,
                                'game_type': room.game_type,
                                'board_dimensions': room.board_dimensions,
                                'time_limit': room.time_limit,
                                'is_24h': True
                            }
                    else:
                        if s.last_active > non_24h_max_active:
                            non_24h_max_active = s.last_active
                            non_24h_match = {
                                'room_id': room.room_id,
                                'is_online': True,
                                'is_spectator': True,
                                'game_type': room.game_type,
                                'board_dimensions': room.board_dimensions,
                                'time_limit': room.time_limit,
                                'is_24h': False
                            }
        
        if non_24h_match:
            return non_24h_match
            
        if fallback_24h_match:
            return fallback_24h_match
        
        # Not in an active room, but might still be online (Lobby/Profile)
        if is_online:
            return {
                'room_id': None,
                'is_online': True,
                'is_spectator': False
            }
            
        return None

    def delete_room(self, room_id):
        """Delete room if found."""
        with self.lock:
            if room_id in self.rooms:
                print(f"[RoomManager] Deleting room {room_id} (requested)")
                del self.rooms[room_id]
            else:
                print(f"[RoomManager] delete_room called for {room_id} but not found")
    
    def cleanup_rooms(self, timeout=600, spec_timeout=600):
        """Clean up empty or inactive rooms (defaults: 10m players, 10m spectators)"""
        rooms_to_delete = []
        
        # Iterate over a copy of keys to avoid modification issues
        for room_id, room in list(self.rooms.items()):
            try:
                # Inactivity pause: if room is empty of human players and not a 24h daily room,
                # pause it by setting state to 'waiting' and skipping transition/milestone processing.
                # This prevents empty public rooms from looping rounds and pegging CPU in background.
                is_daily = (room.time_limit >= 7200)
                humans = [p for p in room.players if not getattr(p, 'is_ai', False)]
                spectators = [s for s in room.spectators if not getattr(s, 'is_ai', False)]
                all_humans = humans + spectators
                now_t = time.time()
                active_humans = [p for p in all_humans if (now_t - getattr(p, 'last_active', 0)) < 120.0]
                if len(active_humans) == 0 and not is_daily:
                    # NEVER pause or wipe a room during an active round or intermission!
                    if room.state in ['active', 'intermission']:
                        pass  # Let active round and intermission expire cleanly via check_and_update_state()
                    elif room.state != 'waiting':
                        print(f"[BG-Cleanup] Pausing inactive room {room_id} (0 active humans for >120s). Setting state to 'waiting'.")
                        with room._state_lock:
                            room.state = 'waiting'
                            room.board = None
                            room.all_words = set()
                            room.all_words_paths = {}
                            room.starting_round = False
                            room.board_search_started = False
                            room.board_search_loading = False
                            room.spinner_params_generated = False
                            room.next_round_board = None
                        continue

                # 1. Update Game State (Transitions)
                # Ensure the transition itself works (logs errors specifically for the room)
                try:
                    state_changed = room.check_and_update_state()
                except Exception as transition_error:
                    print(f"[BG-Cleanup] CRITICAL ERROR updating room {room_id}: {transition_error}")
                    import traceback
                    traceback.print_exc()
                    state_changed = False
                
                # Handle proactive milestones (Search, Spinner, Reveal)
                milestone = room.get_next_round_milestone()
                
                if milestone == 'spinner':
                    # Proactive generation
                    threading.Thread(target=self.generate_spinner_params, args=(room_id, False), daemon=True).start()

                elif milestone == 'reveal':
                    # Reveal (Intermission only)
                    if not getattr(room, 'spinner_params_revealed', False):
                        threading.Thread(target=self.generate_spinner_params, args=(room_id, True), daemon=True).start()

                elif milestone == 'search':
                    # Proactive board search
                    if not getattr(room, 'board_search_started', False):
                        room.board_search_started = True
                        print(f"[TRANSITION] Room {room_id}: PROACTIVE SEARCH TRIGGERED")
                        threading.Thread(target=self.start_board_search, args=(room_id,), daemon=True).start()
                
                elif milestone == 'start' and room.state == 'intermission':
                    # Start next round
                    if not getattr(room, 'starting_round', False):
                        threading.Thread(target=self.start_next_round, args=(room_id,), daemon=True).start()

                # 2. Check for inactive players
                room.check_inactivity(timeout, spec_timeout)
                
                # Close room if empty of HUMAN players (zombie room prevention)
                humans = [p for p in room.players if not p.is_ai]
                is_empty_of_humans = (len(humans) == 0)
                is_daily = (room.time_limit >= 7200)
                
                is_public = room_id.startswith('pub_')
                if is_empty_of_humans and not is_daily and not is_public:
                    # If remove_player already flagged this room as closing, delete immediately.
                    if getattr(room, 'is_closing', False):
                        print(f"[RoomManager] Room {room_id} flagged as closing (last player left). Deleting now.")
                        rooms_to_delete.append(room_id)
                    else:
                        # Grace Period: 2 minutes fallback for rooms not caught by remove_player
                        room_uptime = time.time() - getattr(room, 'creation_time', time.time())
                        if room_uptime > 120:
                            print(f"[RoomManager] Marking room {room_id} for deletion (No human players, uptime: {int(room_uptime)}s)")
                            rooms_to_delete.append(room_id)
                    
            except Exception as e:
                import traceback
                print(f"[RoomManager] Error cleaning up/ticking room {room_id}: {e}\n{traceback.format_exc()}")
        
        # Delete marked rooms
        for room_id in rooms_to_delete:
            self.delete_room(room_id)
        
        if rooms_to_delete:
            print(f"[RoomManager] Cleanup complete. Removed {len(rooms_to_delete)} rooms.")
    
    def start_round(self, room_id, bonus_word_override=None):
        """Start a new round with spinner and board generation (Override: {bonus_word_override})"""
        room = self.get_room(room_id)
        if not room:
             return False
             
        room.starting_round = True
        room._round_start_init_time = time.time()
        
        # SYNC ADDED WORDS CONFIG: Ensure all processes reload the use_added_words state from disk
        if hasattr(word_validator, 'word_validator'):
            word_validator.word_validator.get_use_added_words()
            
        try:
            # Save previous round data before generating new one
            has_prev = hasattr(room, 'previous_all_words') and room.previous_all_words
            if not has_prev and room.all_words and room.board:
                # USER REQUEST: Absolute filter for history. 
                # Display Floor is 4L, but must also respect round minimum (e.g. 5L).
                display_min_hist = getattr(room, 'current_min_length', 3)
                words_list = [w for w in room.all_words if len(w) >= display_min_hist]
                room.previous_all_words = {w: {} for w in words_list} # Dict format for scores compatibility
                room.previous_board = [list(row) for row in room.board]
                print(f"[RoomManager] Saved {len(room.previous_all_words)} words to history (Fallback/Round {room.current_round})")
            elif has_prev:
                print(f"[RoomManager] Using existing history snapshot (intermission) for Round {room.current_round}")

            # SUBANAGRAMS ROUND INITIALIZATION: Dedicated fast path (< 5ms)
            if getattr(room, 'game_type', None) == 'subanagrams':
                print(f"[RoomManager] Starting Subanagrams round for {room_id}...")
                b_board = None
                b_words = None
                b_bonus = None
                fparams = None

                # Fast-path 1: Promote next_round_board if already staged
                if getattr(room, 'next_round_board', None) and getattr(room, 'next_round_words', None):
                    print(f"[RoomManager] Promoting staged Subanagrams board for {room_id}")
                    b_board = room.next_round_board
                    b_words = room.next_round_words
                    b_bonus = getattr(room, 'next_round_bonus', None)
                    fparams = getattr(room, 'next_spinner_params', None) or dict(room.spinner_params or {})
                    room.next_round_board = None
                    room.next_round_words = None
                    room.next_round_bonus = None
                else:
                    from subanagrams_generator import generate_subanagrams_board_and_words
                    sp = getattr(room, 'next_spinner_params', None) or getattr(room, 'frozen_revealed_params', None) or room.spinner_params
                    if not sp:
                        sp = SpinnerSet.generate_subanagrams_params(previous_params=getattr(room, 'spinner_params', None))
                    res = generate_subanagrams_board_and_words(sp)
                    if res:
                        b_board, b_words, b_bonus, fparams = res
                    else:
                        from subanagrams_generator import solve_subanagrams, get_subanagrams_dict
                        fallback_seq = ['S', 'T', 'A', 'R', 'L', 'I', 'N', 'G']
                        dict_data = get_subanagrams_dict('NWL', False)
                        b_words = solve_subanagrams('STARLING', dict_data, min_length=3)
                        b_board = [fallback_seq]
                        b_bonus = 'STARLING'
                        fparams = {
                            'board_dimensions': 'subanagrams',
                            'time_limit': 120,
                            'min_word_length': 3,
                            'sequence_length': 8,
                            'dictionary': 'NWL',
                            'board_format': 'Word Guaranteed',
                            'mode': 'word',
                            'word_count_range': '200-300',
                            'difficulty': 'Practice'
                        }

                if b_board and b_words:
                    room.board = b_board
                    room.sequence_length = fparams.get('sequence_length', len(b_board[0]))
                    room.all_words = set(b_words)
                    room.all_words_paths = {w: [] for w in b_words}
                    room.bonus_word = b_bonus
                    seq_len_actual = len(b_board[0]) if (b_board and len(b_board) > 0 and isinstance(b_board[0], list)) else len(b_board)
                    has_full = any(len(w) == seq_len_actual for w in b_words)
                    final_fmt = 'Word Guaranteed' if (has_full and fparams.get('mode') == 'word') else 'Totally Random'
                    room.current_board_format = final_fmt
                    fparams['board_format'] = final_fmt
                    fparams['mode'] = 'word' if final_fmt == 'Word Guaranteed' else 'random'
                    room.current_min_length = fparams.get('min_word_length', 3)
                    room.current_dictionary = fparams.get('dictionary', 'CSW')
                    room.use_added_words = fparams.get('use_added_words', False)
                    room.spinner_params = dict(fparams)
                    room.frozen_revealed_params = dict(fparams)
                    room.current_word_count_range = fparams.get('word_count_range', '100-200')
                    room.spinner_params_generated = True
                    from scoring import calculate_word_score
                    room.solved_words_with_scores = {
                        w: calculate_word_score(w, b_bonus, board_format='Normal', return_details=True)
                        for w in b_words
                    }
                    room.total_words_count = len(b_words)
                    room.initial_total_words = len(b_words)
                    room.recalculate_total_points()
                    room.state = 'active'
                    room.round_start_time = time.time()
                    room.current_round = max(1, getattr(room, 'current_round', 0) + 1)
                    room._initial_board_delivered = True
                    room.starting_round = False
                    room.board_search_loading = False
                    room.board_search_started = False
                    self.pre_generate_next_round(room.room_id)
                    print(f"[RoomManager] Subanagrams start_round completed successfully for {room_id} (Round {room.current_round})")
                    return True
            
            # CLEAR BOARD & WORDS IMMEDIATELY for 500+ (IO) rounds to avoid flickering
            wc_range = room.spinner_params.get('word_count_range', '100-200')
            wc_tuple = room._get_wc_tuple(wc_range)
            is_500plus = wc_tuple[0] >= 500
            if is_500plus:
                dims = room.board_dimensions.split('x')
                if len(dims) == 3:
                    # 3D Cube: 6 faces of NxM
                    f_num, r_num, c_num = map(int, dims)
                    room.board = [[['' for _ in range(c_num)] for _ in range(r_num)] for _ in range(6)]
                else:
                    rows_num, cols_num = map(int, dims)
                    room.board = [['' for _ in range(cols_num)] for _ in range(rows_num)]
                room.all_words = []
                room.complete_words = []
            
            # Generate spinner parameters (Use existing if already generated during intermission)
            is_24h = room.time_limit >= 7200
            if not room.spinner_params:
                is_split = (room.game_type == 'split')
                room.spinner_params = SpinnerSet.generate_params(room.board_dimensions, is_24h, is_split)
                room.spinner_params_generated = True
            # Issue 5: Only sanitize if params have NOT been revealed yet.
            # Once params are locked (revealed during intermission), do NOT mutate them.
            if not getattr(room, 'spinner_params_revealed', False):
                room.spinner_params = SpinnerSet.sanitize_params(room.spinner_params, room.board_dimensions, is_24h)
            
            # GET BONUS WORD (Use override if available, else roll new)
            if bonus_word_override:
                bonus_word = bonus_word_override
            else:
                min_length_requirement = room.spinner_params.get('min_word_length', 3)
                bw_l = max(room.spinner_params.get('bonus_word_length', 6), min_length_requirement)
                
                is_checkerboard = 'checkerboard' in str(room.spinner_params.get('board_format', '')).lower()
                bonus_word = self._get_bonus_word(
                    bw_l, 
                    room.spinner_params['dictionary'],
                    alternating=is_checkerboard,
                    exclude=room.bonus_word # EXCLUDE current
                )
            
            room.bonus_word = bonus_word
            
            # MANDATORY BONUS WORD LOCKDOWN: Every board in every format in Public Rooms MUST have a bonus word.
            if not bonus_word:
                print(f"[RoomManager] ! Emergency: bonus_word missing in start_round (fallback) for room {room_id}, rolling 6-letter fallback.")
                # At 45s remaining, we pick a bonus word based on the SPINNER parameters
                b_diff = room.spinner_params.get('difficulty', 'Medium')
                min_l = room.spinner_params.get('min_word_length', 3)
                bw_l = max(room.spinner_params.get('bonus_word_length', 6), min_l)
                
                bonus_word = self._get_bonus_word(
                    bw_l, 
                    room.spinner_params.get("dictionary", "NWL"),
                    difficulty=b_diff,
                    exclude=room.bonus_word # EXCLUDE current
                )
            
            # Synchronize room object early to avoid any desync in background/async layers
            room.bonus_word = bonus_word
            
            # Anti-duplicate board protection loop
            board_attempts = 0
            while board_attempts < 4:
                random.seed()
                
                _sp_dict = str(room.spinner_params.get('dictionary', 'NWL')).upper()
                use_aw_flag = room.spinner_params.get('use_added_words', False) or ('+ AW' in _sp_dict) or ('+AW' in _sp_dict) or (_sp_dict in ['AW', 'ADDED_WORDS'])
                
                # Calculate min_accept for safety check
                m_len = room.spinner_params.get('min_word_length', 3)
                min_accept = 50
                target_range = room.spinner_params.get('word_count_range')
                if target_range:
                    try:
                        min_accept = int(str(target_range).split('-')[0])
                    except:
                        if '50' in str(target_range): min_accept = 50
                        elif '100' in str(target_range): min_accept = 100
                        elif '200' in str(target_range): min_accept = 200
                        elif '300' in str(target_range): min_accept = 300
                        elif '400' in str(target_range): min_accept = 400
                        elif '500' in str(target_range): min_accept = 500
                if use_aw_flag:
                    min_accept = max(300, min_accept)
                elif m_len >= 6:
                    min_accept = min(min_accept, 30)
                else:
                    min_accept = max(30, min_accept)

                # OPTIMIZATION: Try to get a cached board instantly for all rooms to avoid any creation delay
                res = None
                from board_generator import serialize_param_key, pop_cached_board, pop_any_cached_board

                # SOLO MODE FAST PATH: Try popping compatible cached board matching room format for 1ms instant board!
                if getattr(room, 'is_solo', False):
                    target_fmt = room.spinner_params.get('board_format', 'Normal')
                    target_dict_raw = room.spinner_params.get('dictionary', 'NWL')
                    target_bw_len = int(room.spinner_params.get('bonus_word_length', 8) or 8)
                    target_diff_solo = room.spinner_params.get('difficulty', 'Medium')
                    if getattr(room, 'initial_solo_params', None):
                        if room.initial_solo_params.get('board_format') and str(room.initial_solo_params.get('board_format')).lower() != 'random':
                            target_fmt = room.initial_solo_params['board_format']
                        if room.initial_solo_params.get('difficulty') and str(room.initial_solo_params.get('difficulty')).lower() != 'random':
                            target_diff_solo = room.initial_solo_params['difficulty']
                        if room.initial_solo_params.get('dictionary') and str(room.initial_solo_params.get('dictionary')).lower() != 'random':
                            target_dict_raw = room.initial_solo_params['dictionary']
                    from board_generator import pop_compatible_cached_board
                    solo_pop = pop_compatible_cached_board(
                        room.board_dimensions,
                        target_dict_raw,
                        target_fmt,
                        room.spinner_params.get('min_word_length', 3),
                        use_aw_flag,
                        bonus_word_len=target_bw_len,
                        difficulty=target_diff_solo
                    )
                    if not solo_pop and str(target_fmt).lower() == 'normal':
                        solo_pop = pop_compatible_cached_board(
                            room.board_dimensions,
                            target_dict_raw,
                            target_fmt,
                            room.spinner_params.get('min_word_length', 3),
                            use_aw_flag,
                            difficulty=target_diff_solo
                        )
                    
                    if solo_pop:
                        if len(solo_pop) >= 9:
                            sboard, swords, sbonus_cell, sfmt, spaths, sratio, sbonus_word, _, sparams = solo_pop
                        else:
                            sboard, swords, sbonus_cell, sfmt, spaths, sratio, sbonus_word, sparams = solo_pop
                        
                        # Strict format compliance for solo mode
                        if str(target_fmt).lower() != 'normal' and str(sfmt).lower() != str(target_fmt).lower():
                            print(f"[RoomManager] Solo cached board format {sfmt} mismatch with {target_fmt}. Discarding pop...")
                            solo_pop = None
                        else:
                            _s_dims = room.board_dimensions.split('x')
                            _s_d0 = int(_s_dims[0]) if len(_s_dims) == 3 else 1
                            _s_r0 = int(_s_dims[1] if len(_s_dims) == 3 else _s_dims[0])
                            _s_c0 = int(_s_dims[2] if len(_s_dims) == 3 else _s_dims[1])
                            _act_s_diff = self.board_generator.get_difficulty_label(sratio, _s_r0, _s_c0, target_dict_raw, depth=_s_d0, min_word_length=m_len)
                            if _act_s_diff != SpinnerSet.strip_difficulty_pct(target_diff_solo):
                                print(f"[RoomManager] Solo cached board actual difficulty {_act_s_diff} != target {target_diff_solo}. Discarding pop...")
                                solo_pop = None
                            else:
                                _rep, _rw, _rp = sync_board_checkerboard_and_resolve(self.board_generator, sboard, target_fmt, m_len, room.spinner_params.get('dictionary', 'NWL'), use_aw_flag, sbonus_cell)
                                if _rep and _rw:
                                    swords, spaths = _rw, _rp

                        if solo_pop:
                            swords_filtered = [w for w in swords if len(w) >= m_len]
                            spaths_filtered = {w: p for w, p in (spaths or {}).items() if len(w) >= m_len}
                            if not sbonus_word or len(sbonus_word) != target_bw_len or sbonus_word not in swords_filtered or sbonus_word.upper().endswith(('ING', 'INGS')) or 'ING' in sbonus_word.upper():
                                _exact_bw_s = [w for w in swords_filtered if len(w) == target_bw_len and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
                                if _exact_bw_s:
                                    sbonus_word = random.choice(_exact_bw_s).upper()
                                else:
                                    sbonus_word = None
                            if len(swords_filtered) >= 10 and sbonus_word and len(sbonus_word) == target_bw_len:
                                print(f"[RoomManager] INSTANT Solo cached board popped for room {room_id} in 1ms! (format={target_fmt}, bonus={sbonus_word})")
                                res = (sboard, swords_filtered, sbonus_cell, target_fmt, spaths_filtered, sratio, sbonus_word)

                cached_res = None
                if not res:
                    param_key_str = serialize_param_key(
                        room.board_dimensions, bonus_word, room.spinner_params['word_count_range'],
                        room.spinner_params['dictionary'], room.spinner_params['board_format'],
                        room.spinner_params.get('min_word_length', 3), room.spinner_params.get('difficulty', 'Medium'),
                        use_added_words=use_aw_flag
                    )
                    cached_res = pop_cached_board(param_key_str)
                if cached_res:
                    cwords_exact = cached_res[1]
                    cwords_filtered = [w for w in cwords_exact if len(w) >= m_len]
                    if len(cwords_filtered) >= min_accept:
                        print(f"[RoomManager] Exact cache hit for room {room_id} start!")
                        cboard, cwords, cbonus_cell, cfmt, cpaths, cratio, cbonus_word = cached_res[:7]
                        cwords = cwords_filtered
                        cpaths = {w: p for w, p in cpaths.items() if len(w) >= m_len}
                        res = (cboard, cwords, cbonus_cell, cfmt, cpaths, cratio, cbonus_word)
                    else:
                        print(f"[RoomManager] Exact cache hit discarded because it had only {len(cwords_filtered)} words of length >= {m_len} (needed {min_accept}).")
                
                if not res:
                    print(f"[RoomManager] Exact cache miss or discarded for room {room_id}. Trying pop_compatible_cached_board...")
                    # Try popping up to 10 candidates to find one with enough words
                    max_limit = 99999
                    if target_range:
                        try:
                            parts = str(target_range).split('-')
                            if len(parts) == 2:
                                max_limit = int(parts[1]) - 1
                        except:
                            pass

                    _solo_target_diff = room.initial_solo_params.get('difficulty') if (getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None) and room.initial_solo_params.get('difficulty') and str(room.initial_solo_params.get('difficulty')).lower() != 'random') else room.spinner_params.get('difficulty', 'Medium')
                    _solo_target_fmt = room.initial_solo_params.get('board_format') if (getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None) and room.initial_solo_params.get('board_format') and str(room.initial_solo_params.get('board_format')).lower() != 'random') else room.spinner_params['board_format']

                    for _ in range(10):
                        from board_generator import pop_compatible_cached_board
                        relaxed_res = pop_compatible_cached_board(
                            room.board_dimensions,
                            room.spinner_params['dictionary'],
                            'Valued Letters' if is_24h else _solo_target_fmt,
                            room.spinner_params.get('min_word_length', 3),
                            use_aw_flag,
                            bonus_word_len=len(bonus_word) if bonus_word else None,
                            word_count_range='300-400' if is_24h else target_range,
                            difficulty=_solo_target_diff
                        )
                        if not relaxed_res:
                            break
                        board, all_words, bonus_cell, board_format_ret, all_words_dict, ratio, final_bonus_word, params = relaxed_res
                        p_min = params.get('min_word_length') if params else None
                        try:
                            act_min = int(p_min) if p_min is not None else m_len
                        except:
                            act_min = m_len
                        raw_dict = room.spinner_params.get('dictionary', 'NWL')
                        raw_aw = room.spinner_params.get('use_added_words', False)
                        all_words_filtered = [w for w in all_words if len(w) >= act_min and word_validator.word_validator.is_valid_word(w, raw_dict, use_added_words=raw_aw)]
                        if is_24h and not (300 <= len(all_words_filtered) <= 399):
                            print(f"[RoomManager] Popped candidate relaxed board discarded for 24h room {room_id}: had {len(all_words_filtered)} words (strictly needed 300-399).")
                            continue
                        if len(all_words_filtered) >= min_accept:
                            # Accept board without discarding valid grid words
                            all_words = all_words_filtered
                            all_words_dict = {w: p for w, p in all_words_dict.items() if w in all_words}
                            print(f"[RoomManager] Popped relaxed cached board for room {room_id} with {len(all_words)} words. Keeping spun parameters.")
                            
                            # Preserve the spun spinner_params completely to stick with what was initially determined.
                            # Only set the current active properties of the room to match the spun spinner_params.
                            room.current_dictionary = room.spinner_params.get('dictionary', 'NWL')
                            room.current_difficulty = _solo_target_diff
                            room.current_board_format = _solo_target_fmt
                            room.current_min_length = room.spinner_params.get('min_word_length', 3)
                            room.use_added_words = room.spinner_params.get('use_added_words', False)

                            
                            # Issue 2: Use the embedded bonus word from the cached board, not a freshly-spun one.
                            # The embedded bonus_word was verified to exist on the board during generation.
                            # Only re-roll if the embedded word is absent or not in all_words.
                            _cached_bw = final_bonus_word
                            _target_bw_len = int(room.spinner_params.get('bonus_word_length', 8) or 8)
                            if _cached_bw and len(_cached_bw) == _target_bw_len and _cached_bw in all_words_filtered and not _cached_bw.upper().endswith(('ING', 'INGS')) and 'ING' not in _cached_bw.upper():
                                bonus_word = _cached_bw
                                room.bonus_word = bonus_word
                            else:
                                # Embedded bonus word does not match target length — pick matching length word from board
                                _bw_candidates = [w for w in all_words_filtered if len(w) == _target_bw_len and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
                                if not _bw_candidates:
                                    _bw_len = max(m_len, min(len(w) for w in all_words_filtered) if all_words_filtered else 6)
                                    _bw_candidates = [w for w in all_words_filtered if len(w) == _bw_len and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
                                if _bw_candidates:
                                    bonus_word = random.choice(_bw_candidates)
                                    room.bonus_word = bonus_word
                                    print(f"[RoomManager] Embedded bonus word '{_cached_bw}' re-selected to match {_target_bw_len}L: '{bonus_word}'")

                            res = (board, all_words, bonus_cell, board_format_ret, all_words_dict, ratio, bonus_word)
                            break
                        else:
                            print(f"[RoomManager] Popped candidate relaxed board discarded: had only {len(all_words_filtered)} words of length >= {act_min} (needed {min_accept}).")
                
                if not res:
                    print(f"[RoomManager] Cache miss for room {room_id}. Delivering instant emergency fallback board.")
                    res = get_emergency_fallback_board(
                        room.board_dimensions, _solo_target_fmt, room.time_limit,
                        dictionary=room.spinner_params.get('dictionary', 'NWL'), use_added_words=use_aw_flag,
                        target_range=room.spinner_params.get('word_count_range', '100-200'), min_word_length=m_len,
                        difficulty=_solo_target_diff
                    )
                
                board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio, final_bonus_word = res[:7]

                # PREFACE #18b: the format declared in the Spinner Set is final. A cached/fallback board may carry a
                # different SCORING-ONLY format (e.g. Bounce instead of Valued Letters); those need no board change,
                # so stamp the declared one. Board-shape formats (Mania, Checkerboard, ...) are left untouched.
                try:
                    _declared_fmt = str(room.spinner_params.get('board_format', '') or '')
                    _scoring_only = ('normal', 'valued letters', 'penalty', 'bounce 1x', 'bounce 2x', 'bounce 3x', 'bounce')
                    if is_24h:
                        _declared_fmt = 'Valued Letters'
                    if (_declared_fmt.lower().strip() in _scoring_only
                            and str(updated_format).lower().strip() in _scoring_only
                            and str(updated_format).lower().strip() != _declared_fmt.lower().strip()):
                        updated_format = _declared_fmt
                        bonus_cell = None
                except Exception as _fmt_err:
                    print(f"[RoomManager] format stamp error: {_fmt_err}")

                # PREFACE (first-board contract): the board MUST satisfy what the Spinner Set declared (word count
                # range + board-shape format). If a cache/fallback pop does not, generate a compliant one instead.
                try:
                    _sp0 = room.spinner_params if isinstance(room.spinner_params, dict) else {}
                    _decl_rng = _sp0.get('word_count_range')
                    _decl_fmt = str(_sp0.get('board_format', '') or '')
                    _decl_low = _decl_fmt.lower().strip()
                    _cnt0 = sum(1 for w in (all_words or []) if len(w) >= int(m_len or 3))

                    def _flat_cells(b):
                        for x in b:
                            if isinstance(x, list):
                                yield from _flat_cells(x)
                            else:
                                yield x
                    _cells = list(_flat_cells(board))
                    _has_eo = any('/' in str(c) for c in _cells)
                    _bad = []
                    if is_24h:
                        _decl_rng = '300-400'
                        _decl_fmt = 'Valued Letters'
                        use_aw_flag = False
                        if not (300 <= _cnt0 <= 399):
                            _bad.append(f"24h count {_cnt0} not in 300-399")
                    elif _decl_rng and not is_board_count_valid(_cnt0, _decl_rng):
                        _bad.append(f"count {_cnt0} not in {_decl_rng}")
                    if not use_aw_flag:
                        _decl_diff0 = SpinnerSet.strip_difficulty_pct(_sp0.get('difficulty', 'Medium'))
                        _dims0 = room.board_dimensions.split('x')
                        _d0 = int(_dims0[0]) if len(_dims0) == 3 else 1
                        _r0 = int(_dims0[1] if len(_dims0) == 3 else _dims0[0])
                        _c0 = int(_dims0[2] if len(_dims0) == 3 else _dims0[1])
                        _base_d0 = str(_sp0.get('dictionary', 'NWL')).replace('+ AW', '').replace('+AW', '').strip() or 'NWL'
                        _actual_diff0 = self.board_generator.get_difficulty_label(u_ratio, _r0, _c0, _base_d0, depth=_d0, min_word_length=m_len)
                        if _actual_diff0 != _decl_diff0:
                            _bad.append(f"difficulty {_actual_diff0} != {_decl_diff0}")
                    if 'either' in _decl_low and not _has_eo:
                        _bad.append("missing Either/Or tile")
                    if 'either' not in _decl_low and _has_eo:
                        _bad.append("unexpected Either/Or tile")
                    if 'bonus letter' in _decl_low and not bonus_cell:
                        _bad.append("missing bonus cell")
                    if 'bonus letter' not in _decl_low and bonus_cell:
                        bonus_cell = None
                    if 'either' in _decl_low and bonus_cell:
                        bonus_cell = None
                    if 'mania' in _decl_low and 'mania' not in str(updated_format).lower():
                        _bad.append("missing Mania")
                    if _decl_low not in _scoring_only and _decl_low not in str(updated_format).lower().strip():
                        _bad.append("format shape mismatch")
                    if _bad:
                        print(f"[RoomManager] First-board contract violated for {room_id}: {_bad}. Generating compliant board...")
                        _base_dict = str(_sp0.get('dictionary', 'NWL')).replace('+ AW', '').replace('+AW', '').strip() or 'NWL'
                        try:
                            _g = self.board_generator.generate_board(
                                dimensions=room.board_dimensions,
                                bonus_word=bonus_word,
                                word_count_range=_decl_rng,
                                dictionary=_base_dict,
                                board_format=_decl_fmt,
                                min_word_length=m_len,
                                difficulty=_sp0.get('difficulty', 'Medium'),
                                timeout=10.0,
                                use_added_words=use_aw_flag
                            )
                            if _g and len(_g) >= 7:
                                board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio, final_bonus_word = _g[:7]
                                print(f"[RoomManager] Compliant first board generated for {room_id}: {len(all_words)} words")
                        except Exception as _gen_err:
                            print(f"[RoomManager] Compliant first-board generation attempt failed: {_gen_err}")
                except Exception as _contract_err:
                    print(f"[RoomManager] first-board contract check error: {_contract_err}")
                break
                
                # Issue 1 & 7: Check against rolling 10-board fingerprint history, not just 1 previous board
                _fp = self._get_board_fingerprint(board)
                _fp_history = getattr(room, 'board_fingerprint_history', [])
                if _fp and _fp in _fp_history and board_attempts < 3:
                    print(f"[RoomManager] WARNING: Board fingerprint ALREADY IN HISTORY for room {room_id}. Retrying...")
                    board_attempts += 1
                    continue
                if (getattr(room, 'board', None) == board or getattr(room, 'previous_board', None) == board) and board_attempts < 3:
                    print(f"[RoomManager] WARNING: Generated board in start_round for room {room_id} is IDENTICAL to current/previous round board. Retrying...")
                    board_attempts += 1
                    continue
                break
            
            if board is None:
                print(f"[RoomManager] ERROR: Board generation failed!")
                if getattr(room, 'is_solo', False):
                    room.board_gen_failed = True
                    room.board_gen_error = "Finding a board that meets these specific criteria takes too long and may not be achievable.<br><br>We recommend adjusting your parameters—such as choosing a slightly lower word count range or a shorter minimum word length—to start your match right away."
                return False
                
            # ATOMICITY: Apply new round data with strict display filtering
            display_min_start = room.spinner_params.get('min_word_length', 3)
            raw_dict = room.spinner_params.get('dictionary', 'NWL')
            raw_aw = room.spinner_params.get('use_added_words', False)
            if all_words:
                actual_shortest = min(len(w) for w in all_words)
                if actual_shortest != display_min_start:
                    print(f"[RoomManager] Word generator aligned min_word_length from {display_min_start} to factual {actual_shortest}. Updating room param.")
                    room.spinner_params['min_word_length'] = actual_shortest
                    if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                        room.frozen_revealed_params['min_word_length'] = actual_shortest
                    display_min_start = actual_shortest
            
            all_words_dict = {w: p for w, p in (all_words_dict or {}).items() if len(w) >= display_min_start and word_validator.word_validator.is_valid_word(w, raw_dict, use_added_words=raw_aw)}
            room.all_words = set(all_words_dict.keys())
            room.all_words_paths = all_words_dict

            # Override with the ACTUAL word count from the generated board.
            _actual_wc  = len(room.all_words)
            _use_aw_val = (
                room.use_added_words
                or (room.spinner_params.get('use_added_words', False) if isinstance(room.spinner_params, dict) else False)
                or ('+ AW' in str(getattr(room, 'current_dictionary', '')).upper())
                or ('+AW' in str(getattr(room, 'current_dictionary', '')).upper())
                or ('+ AW' in str(room.spinner_params.get('dictionary', '')).upper() if isinstance(room.spinner_params, dict) else False)
            )
            _real_wc_sr = self._get_factchecked_wc_range(_actual_wc, use_added_words=_use_aw_val, dictionary=getattr(room, 'current_dictionary', None))

            _declared_sr = None
            if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                _declared_sr = room.frozen_revealed_params.get('word_count_range')
            elif isinstance(room.spinner_params, dict) and getattr(room, '_spinner_params_locked', False):
                _declared_sr = room.spinner_params.get('word_count_range')

            if is_24h:
                _grid_floor = 3
                if '4x6' in str(room.board_dimensions): _grid_floor = 4
                elif '5x7' in str(room.board_dimensions): _grid_floor = 5
                elif '6x8' in str(room.board_dimensions) or '3x3x3' in str(room.board_dimensions): _grid_floor = 6
                display_min_start = _grid_floor
                _final_wc_sr = '300-400'
                if not (300 <= _actual_wc <= 399):
                    print(f"[RoomManager] start_round word count {_actual_wc} outside 300-400 for 24h room {room_id}. Generating compliant board...")
                    try:
                        _gen_24h = self.board_generator.generate_board(
                            dimensions=room.board_dimensions,
                            bonus_word=bonus_word,
                            word_count_range='300-400',
                            dictionary=raw_dict,
                            board_format='Valued Letters',
                            min_word_length=_grid_floor,
                            difficulty=room.spinner_params.get('difficulty', 'Medium'),
                            timeout=15.0,
                            use_added_words=False
                        )
                        if _gen_24h and len(_gen_24h) >= 7:
                            board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio, final_bonus_word = _gen_24h[:7]
                            all_words_dict = {w: p for w, p in (all_words_dict or {}).items() if len(w) >= _grid_floor}
                            room.all_words = set(all_words_dict.keys())
                            room.all_words_paths = all_words_dict
                            _actual_wc = len(room.all_words)
                    except Exception as _e24:
                        print(f"[RoomManager] 24h compliant board generation error: {_e24}")
            else:
                _final_wc_sr = _declared_sr if (_declared_sr and is_board_count_valid(_actual_wc, _declared_sr)) else _real_wc_sr

            room.current_min_length = display_min_start
            room.current_dictionary = raw_dict
            room.current_word_count_range = _final_wc_sr
            if isinstance(room.spinner_params, dict):
                room.spinner_params['word_count_range'] = _final_wc_sr
            if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                room.frozen_revealed_params['word_count_range'] = _final_wc_sr
            room.update_counts_by_len()

            # CATEGORIZATION (Synchronous): Ensure these are available immediately for UI sync
            if hasattr(word_validator, 'word_validator'):
                word_validator.word_validator.ensure_csw_loaded()
                room.csw_only_words = [w for w in room.all_words if word_validator.word_validator.is_csw_only(w)]
                room.added_words = [w for w in room.all_words if word_validator.word_validator.is_added_word(w)]
                if room.added_words:
                    print(f"[GameRoom {room.room_id}] Round {room.current_round} generated {len(room.added_words)} custom added words: {room.added_words}")
            else:
                room.csw_only_words = []
                room.added_words = []
            
            # CRITICAL: Preserve special cell metadata (Bonus Letter / Either/Or)
            # generate_board returns 'bonus_cell' coordinate as the 3rd element.
            with room._state_lock:
                if room.state == 'active' and room.board and len(room.board) > 0:
                    print(f"[RoomManager] Aborting start_round board application for {room_id}: Room is ALREADY active with board!")
                    room.starting_round = False
                    return True
            candidate_words = list(room.all_words or all_words or [])
            candidate_paths = room.all_words_paths if room.all_words_paths is not None else (all_words_dict or {})
            board, s_words, s_paths, bonus_cell, bonus_word, _ = self.sanitize_board_for_room_difficulty(
                room, board, candidate_words, candidate_paths, bonus_cell, bonus_word,
                room.current_board_format, room.current_difficulty, dictionary=room.current_dictionary, min_len=room.current_min_length
            )
            if s_words:
                room.all_words = set(s_words)
                room.all_words_paths = s_paths
                room.total_words_count = len(room.all_words)
            else:
                room.all_words = set(candidate_words)
                room.all_words_paths = candidate_paths
                room.total_words_count = len(room.all_words)
            room.board = board
            room.bonus_cell = bonus_cell

            # Issue 1 & 7: Record fingerprint in rolling history (max 10 entries)
            _new_fp = self._get_board_fingerprint(board)
            if _new_fp:
                if not hasattr(room, 'board_fingerprint_history') or room.board_fingerprint_history is None:
                    room.board_fingerprint_history = []
                room.board_fingerprint_history.append(_new_fp)
                if len(room.board_fingerprint_history) > 10:
                    room.board_fingerprint_history = room.board_fingerprint_history[-10:]

            _revealed_bw_sr = getattr(room, 'frozen_revealed_params', {}).get('bonus_word_length') if getattr(room, 'frozen_revealed_params', None) else None
            _target_bw_sr = int(_revealed_bw_sr or room.spinner_params.get('bonus_word_length', 8) or 8) if isinstance(room.spinner_params, dict) else 8
            if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                _solo_bw_c = room.initial_solo_params.get('bonus_word_length', 'random')
                if _solo_bw_c and str(_solo_bw_c).lower() not in ['random', '0', 'none', 'null', 'nan']:
                    try:
                        _target_bw_sr = int(_solo_bw_c)
                    except Exception:
                        pass
            _dict_sr = room.spinner_params.get('dictionary', 'NWL') if isinstance(room.spinner_params, dict) else 'NWL'
            _is_chk_sr = 'checkerboard' in str(room.current_board_format).lower()
            room.bonus_word = pick_distinct_bonus_word(
                room, final_bonus_word, room.all_words, _target_bw_sr,
                dictionary=_dict_sr, is_checkerboard=_is_chk_sr,
                room_manager=self, extra_exclude=[getattr(room, 'previous_bonus_word', None)]
            )
            ensure_bonus_word_on_board(room, _target_bw_sr)
            if room.bonus_word:
                actual_bw_len = len(room.bonus_word)
                if isinstance(room.spinner_params, dict):
                    room.spinner_params['bonus_word_length'] = actual_bw_len
                if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                    room.frozen_revealed_params['bonus_word_length'] = actual_bw_len
            sync_bonus_word_scores(room)

            print(f"[RoomManager] ROUND {room.current_round} START for {room_id}. Words found: {len(all_words)}")
            with open(DEBUG_FLOW_PATH, 'a') as f:
                 f.write(f"[game_room.py] 24h START: {room_id} words={len(all_words)} dim={room.board_dimensions} at {time.time()}\n")
            
            room.current_round += 1
            # Derive ACTUAL difficulty from achieved uniqueness ratio
            dims = room.board_dimensions.split('x')
            d_num = int(dims[0]) if len(dims) == 3 else 1
            r_num = int(dims[1] if len(dims) == 3 else dims[0])
            c_num = int(dims[2] if len(dims) == 3 else dims[1])
            achieved_diff = self.board_generator.get_difficulty_label(u_ratio, r_num, c_num, room.spinner_params.get('dictionary', 'NWL'), depth=d_num, board=room.board, target_difficulty=room.spinner_params.get('difficulty'), min_word_length=room.spinner_params.get('min_word_length', 3))
            if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                _req_solo_d = room.initial_solo_params.get('difficulty')
                if _req_solo_d and str(_req_solo_d).lower() != 'random':
                    _target_diff_base = SpinnerSet.strip_difficulty_pct(_req_solo_d)
                    if _target_diff_base == 'Easy' and u_ratio > 0.15:
                        self.board_generator._sanitize_uniqueness(room.board, room.spinner_params.get('dictionary', 'NWL'), max_ratio=0.15, depth=d_num, bonus_cell=room.bonus_cell, min_word_length=display_min_start, use_added_words=room.use_added_words)
                        _re_s = self.board_generator._solve_board(room.board, room.spinner_params.get('dictionary', 'NWL'), (0, 99999), display_min_start, max_depth=20, store_paths=True, timeout=5.0, bonus_cell=room.bonus_cell, use_added_words=room.use_added_words)
                        all_words_dict = {w: p for w, p in _re_s.items() if len(w) >= display_min_start}
                        room.all_words = set(all_words_dict.keys())
                        room.all_words_paths = all_words_dict
                        room.total_words_count = len(room.all_words)
                        room.initial_total_words = room.total_words_count
                        u_ratio = self.board_generator._calculate_uniqueness_ratio(room.board, room.spinner_params.get('dictionary', 'NWL'), d_num, room.bonus_cell, display_min_start, use_added_words=room.use_added_words)
                    achieved_diff = _target_diff_base
                _req_solo_fmt = room.initial_solo_params.get('board_format')
                if _req_solo_fmt and str(_req_solo_fmt).lower() != 'random':
                    updated_format = _req_solo_fmt
            room.current_difficulty = SpinnerSet.format_difficulty_with_pct(achieved_diff, u_ratio)
            if not getattr(room, '_spinner_params_locked', False):
                room.spinner_params['difficulty'] = room.current_difficulty
                room.spinner_params['board_format'] = updated_format
                room.spinner_params['uniqueness'] = u_ratio
            room.spinner_params_revealed = True # Ensure they are shown
            
            print(f"[RoomManager] ROUND {room.current_round} START - Params: {room.current_difficulty}, {room.current_dictionary}, {room.current_word_count_range}")
            
            print(f"[RoomManager] ROUND {room.current_round} START for room {room_id}")
            print(f"[RoomManager]   > Difficulty: {room.current_difficulty}")
            print(f"[RoomManager]   > Dictionary: {room.current_dictionary}")
            print(f"[RoomManager]   > Word Range: {room.current_word_count_range}")
            
            room.current_min_length = room.spinner_params.get('min_word_length', 3)
            room.current_board_format = 'Valued Letters' if is_24h else updated_format
            room.use_added_words = room.spinner_params.get('use_added_words', False)
            # room.bonus_cell already set above
            room.all_words_paths = all_words_dict # ATOMIC SAVE: Crucial for optimized scoring
            
            # USER: Density Initialization for Solo/Immediate Rounds
            try:
                room.initialize_density(board, all_words_dict, updated_format)
            except Exception as e:
                print(f"[Density-Diag] Failed to initialize: {e}")
            
                
            # FAST INITIALIZATION: Length-based scores to avoid "0 point" flickering in UI
            # (Ensures all paths have points immediately while detailed solver runs)
            is_valued_init = ('valued' in str(updated_format).lower())
            init_scored_dict = {}
            for word in (all_words or []):
                if is_valued_init:
                    init_scored_dict[word] = {'total': get_valued_word_score(word), 'base': get_valued_word_score(word)}
                else:
                    length = len(word)
                    s = 0
                    if length <= 2: s = 0
                    elif length <= 4: s = 1
                    elif length == 5: s = 2
                    elif length == 6: s = 3
                    elif length == 7: s = 5
                    elif length == 8: s = 9
                    elif length == 9: s = 13
                    elif length == 10: s = 17
                    elif length == 11: s = 21
                    elif length == 12: s = 25
                    elif length == 13: s = 29
                    elif length == 14: s = 33
                    elif length >= 15: s = 37
                    else: s = 1 # Fallback for 3/4 if logic misses
                    init_scored_dict[word] = {'total': max(1, s), 'base': max(1, s)}
            
            room.solved_words_with_scores = init_scored_dict
            display_min_init = getattr(room, 'current_min_length', 3)
            room.complete_words = [w for w in (all_words or []) if len(w) >= display_min_init]
            room.solving_complete = False # Detailed refinement still pending
            
            # AUTHORITATIVE SYNC
            room.update_counts_by_len()
            room.recalculate_total_points() # Ensure total pts is non-zero from round start
            
            # --- ASYNCHRONOUS POST-START TASKS ---
            # Offload heavy scoring and next-round pre-gen to background threads
            import threading
            def finalize_start_round_data():
                try:
                    # 2. Scoring (Calculated in background to allow instant round start)
                    from scoring import calculate_word_score
                    final_scores = {}
                    for word in room.all_words:
                        final_scores[word] = calculate_word_score(
                            word, 
                            room.bonus_word, 
                            board_format=room.current_board_format,
                            bonus_cell=room.bonus_cell,
                            board=room.board,
                            path=room.all_words_paths.get(word),
                            return_details=True,
                            strict_path=True
                        )
                    room.solved_words_with_scores = final_scores
                    room.complete_words = room.all_words
                    room.solving_complete = True # Signal that missed words are ready
                    room.recalculate_total_points() # Sync refined points after background scoring
                    
                    # 4. Trigger Pre-Generation for Round 2
                    self.pre_generate_next_round(room_id)
                    
                    # 5. Pre-generate AI turns for this first round
                    room.generate_ai_turns()

                    # For 24h rooms, save the generated board to the database immediately
                    if is_24h:
                        try:
                            import json
                            # Serialize active players
                            players_data = []
                            for p in room.players:
                                players_data.append({
                                    'user_id': p.user_id,
                                    'username': p.username,
                                    'rating': p.rating,
                                    'submitted_words': p.submitted_words,
                                    'invalid_words': p.invalid_words,
                                    'score': p.score,
                                    'previous_round_score': p.previous_round_score,
                                    'games_played': p.games_played,
                                    'previous_submitted_words': p.previous_submitted_words,
                                    'found_bonus_word': p.found_bonus_word,
                                    'last_active': p.last_active,
                                    'input_method': p.input_method,
                                    'country_flag': p.country_flag,
                                    'joined_mid_round': p.joined_mid_round,
                                    'has_exceptional_round': p.has_exceptional_round,
                                    'is_guest': p.is_guest,
                                    'is_ai': p.is_ai,
                                    'ai_rating': p.ai_rating,
                                    'has_abandoned': p.has_abandoned,
                                    'game_room_chat': getattr(p, 'game_room_chat', True)
                                })
                            players_json = json.dumps(players_data)
                            
                            with get_db() as conn:
                                conn.execute('''
                                    INSERT OR REPLACE INTO active_boards (
                                        room_id, board_data, all_words, dictionary, min_length, updated_at,
                                        bonus_word, bonus_cell_json, board_format, uniqueness, word_count_range,
                                        active_players_json
                                    )
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    room.room_id,
                                    json.dumps(room.board),
                                    json.dumps(list(room.all_words)),
                                    room.current_dictionary,
                                    room.current_min_length,
                                    time.time(),
                                    room.bonus_word or '',
                                    json.dumps(room.bonus_cell) if room.bonus_cell else None,
                                    room.current_board_format or 'Normal',
                                    room.current_uniqueness or 0.0,
                                    room.current_word_count_range or '300-400',
                                    players_json
                                ))
                            print(f"[RoomManager] Reconstructed 24h room {room.room_id} board saved to active_boards DB successfully.")
                        except Exception as db_err:
                            print(f"[RoomManager] Error saving reconstructed 24h room board to DB: {db_err}")
                except Exception as e:
                    print(f"[RoomManager] Background start error for {room_id}: {e}")
                finally:
                    room.starting_round = False

            threading.Thread(target=finalize_start_round_data, daemon=True).start()
            
            # Reset players
            for p in room.players:
                p.rating_change = 0
                
                # SNAPSHOT PROTECTION: if already cleared by 24h reset, keep the existing snapshot
                if len(p.submitted_words) > 0:
                    p.previous_round_score = p.score
                    p.previous_submitted_words = [dict(w) for w in p.submitted_words]
                elif room.time_limit >= 7200 and (getattr(p, 'previous_round_score', 0) > 0 or (hasattr(p, 'previous_submitted_words') and len(p.previous_submitted_words) > 0)):
                    # Keep existing snapshot from midnight reset
                    pass
                else:
                    # Standard round end
                    p.previous_round_score = p.score
                    p.previous_submitted_words = [dict(w) for w in p.submitted_words]
                
                p.submitted_words = []
                p.invalid_words = []
                p.score = 0
                p.found_bonus_word = False
                p.joined_mid_round = False
                # User Request: At the start of the next round following the awarding of a trophy icon, remove it from every user
                p.has_exceptional_round = False
                p.trophy_rounds_left = 0
                p.has_abandoned = False # Reset penalty flag for new round
                p._last_round_seen = room.current_round
                
            # Record authoritative set of players present at the start of this round
            room.round_starting_player_ids = {str(p.user_id) for p in room.players}

            # Clear FCFS global list
            room.fcfs_found_words = []
            room._fcfs_found_words_set = set()
            
            # Activate the room
            # User Request: Do NOT wipe spinner_params here. 
            # They should hold the intent labels revealed during intermission.
            room.spinner_params_generated = False
            # CRITICAL: Set round_start_time BEFORE state='active'.
            # If state is set first, check_and_update_state() (called from the tick worker or
            # any state poll) may see state='active' with round_start_time=0, compute
            # elapsed = now - 0 = huge number, and immediately transition to 'intermission',
            # causing start_next_round to fire and replace room.board mid-round.
            room.custom_end_time = 0
            room.round_start_time = time.time()
            room.state = 'active'
            room._initial_board_delivered = True  # ISSUE 6: first board is now live
            
            # TRIGGER PRE-GENERATION: Start searching for the NEXT round immediately 
            # to hide generation latency behind the active gameplay.
            self.pre_generate_next_round(room_id)
            room.custom_end_time = 0
            
            # SPLIT POINTS RANDOMIATION
            if room.game_type == 'split':
                random.shuffle(room.players)
                
            return True
        except Exception as e:
            import traceback
            print(f"[RoomManager] CRITICAL EXCEPTION in start_round: {e}")
            traceback.print_exc()
            return False
        finally:
            room.starting_round = False
    
    def _apply_daily_reset(self, room):
        """Perform daily cleanup and metadata reset for 24h rooms"""
        print(f"[RoomManager] Daily Reset: Preparing metadata for room {room.room_id}")
        
        # NOTE: We NO LONGER clear room.players here. 
        # app.py handles the atomic player state reset to avoid kicking active users (403).
        
        # CLEAR metadata (Timer handled dynamically by property)
        room.custom_end_time = 0
        room.solving_complete = False
        room.complete_words = []
        room.midnight_reset_occurred = True # Signal for app.py to handle player reset
        
        print(f"[RoomManager] Daily room {room.room_id}: Metadata cleared for new day.")

    def generate_spinner_params(self, room_id, reveal=False):
        """Generate and optionally reveal next round's game parameters"""
        from spinner_set import SpinnerSet
        from scoring import calculate_word_score
        room = self.get_room(room_id)
        if not room:
            return False

        # PERF (#3/#6): load CSW BEFORE taking the room lock. Lazy-loading it inside the lock stalled
        # submit_word and the 0:00 transition for seconds.
        try:
            if hasattr(word_validator, 'word_validator'):
                word_validator.word_validator.ensure_csw_loaded()
        except Exception as _csw_err:
            print(f"[RoomManager] CSW preload error: {_csw_err}")

        with room._state_lock:
            # 1. Check if already revealed - if so, NOOP
            if getattr(room, 'spinner_params_revealed', False) or getattr(room, '_spinner_params_locked', False):
                return True
                
            # 2. Check if already generated but not yet revealed
            if reveal and getattr(room, 'spinner_params_generated', False):
                if room.game_type == 'subanagrams':
                    import copy
                    new_params = getattr(room, 'next_spinner_params', None)
                    prev_p = getattr(room, 'spinner_params', None) or {}
                    # Guard: ensure params exist, board exists, and are not repeating previous round
                    if not new_params or not new_params.get('sequence_length') or (
                        new_params.get('sequence_length') == prev_p.get('sequence_length') and 
                        new_params.get('word_count_range') == prev_p.get('word_count_range') and
                        new_params.get('dictionary') == prev_p.get('dictionary')
                    ) or not getattr(room, 'next_round_board', None):
                        from spinner_set import SpinnerSet
                        from subanagrams_generator import generate_subanagrams_board_and_words
                        from scoring import calculate_word_score
                        new_params = SpinnerSet.generate_subanagrams_params(previous_params=prev_p)
                        res = generate_subanagrams_board_and_words(new_params)
                        if res:
                            nb, nw, nbw, nparams = res
                            room.next_round_board = nb
                            room.next_round_words = nw
                            room.next_round_word_paths = {w: [] for w in nw}
                            room.next_round_bonus = nbw
                            room.next_round_bonus_cell = None
                            room.next_round_format = nparams.get('board_format', 'Word Guaranteed')
                            room.next_round_spinner_params = nparams
                            room.next_spinner_params = nparams
                            room.next_round_min_length = nparams.get('min_word_length', 3)
                            room.next_round_total_words_count = len(nw)
                            room.next_round_word_scores = {
                                w: calculate_word_score(w, nbw, board_format='Normal', return_details=True)
                                for w in nw
                            }
                            new_params = nparams

                    room.spinner_params = dict(new_params)
                    room.frozen_revealed_params = copy.deepcopy(new_params)
                    room.next_spinner_params = dict(new_params)
                    room.next_round_min_length = new_params.get('min_word_length', 3)
                    room.spinner_params_revealed = True
                    room.was_revealed_this_intermission = True
                    room._spinner_params_locked = True
                    room._reveal_sync_complete = True
                    print(f"[RoomManager] SUCCESS: Revealed pre-generated params for subanagrams room {room_id}: {new_params.get('sequence_length')}L | {new_params.get('dictionary')} | {new_params.get('word_count_range')}")
                    return True

                # Ensure board has been found before revealing parameters at 0:45
                if not getattr(room, 'next_round_board', None):
                    print(f"[RoomManager] At 0:45 reveal: board not yet found for {room_id}. Ensuring board is ready immediately...")
                    room.ensure_next_board_ready()

                new_params = getattr(room, 'next_spinner_params', None)
                if not new_params and getattr(room, 'next_round_spinner_params', None):
                    new_params = getattr(room, 'next_round_spinner_params', None)
                if not new_params:
                    new_params = getattr(room, 'spinner_params', None) or {}

                new_params = dict(new_params)

                # Ensure parameters match the found board and finalize bonus word BEFORE locking at 0:45
                if getattr(room, 'next_round_words', None):
                    actual_wc = len(room.next_round_words)
                    use_aw_val = (new_params.get('use_added_words', False) or '+ AW' in str(new_params.get('dictionary', '')).upper() or '+AW' in str(new_params.get('dictionary', '')).upper())
                    cur_wc_range = new_params.get('word_count_range')
                    if cur_wc_range and is_board_count_valid(actual_wc, cur_wc_range) and (not use_aw_val or actual_wc >= 300):
                        new_params['word_count_range'] = cur_wc_range
                    else:
                        new_params['word_count_range'] = self._get_factchecked_wc_range(actual_wc, use_added_words=use_aw_val, dictionary=new_params.get('dictionary'))
                    new_params['min_word_length'] = min(len(w) for w in room.next_round_words)
                    if room.game_type != 'subanagrams':
                        target_bw_rev = int(new_params.get('bonus_word_length', 8) or 8)
                        if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                            _s_bw_r = room.initial_solo_params.get('bonus_word_length', 'random')
                            if _s_bw_r and str(_s_bw_r).lower() not in ['random', '0', 'none', 'null', 'nan']:
                                try:
                                    target_bw_rev = int(_s_bw_r)
                                except Exception:
                                    pass
                        room.next_round_bonus = pick_distinct_bonus_word(
                            room, getattr(room, 'next_round_bonus', None), room.next_round_words, target_bw_rev,
                            dictionary=new_params.get('dictionary', 'NWL'),
                            is_checkerboard=('checkerboard' in str(getattr(room, 'next_round_format', new_params.get('board_format', ''))).lower()),
                            room_manager=self,
                            extra_exclude=[getattr(room, 'bonus_word', None), getattr(room, 'previous_bonus_word', None)]
                        )
                if getattr(room, 'next_round_format', None):
                    new_params['board_format'] = room.next_round_format
                if getattr(room, 'next_round_bonus', None):
                    new_params['bonus_word_length'] = len(room.next_round_bonus)
                staged_uniq = getattr(room, 'next_round_uniqueness', None)
                if staged_uniq is None:
                    staged_uniq = new_params.get('uniqueness')
                if staged_uniq is not None and room.game_type != 'subanagrams':
                    new_params['uniqueness'] = staged_uniq
                    new_params['difficulty'] = SpinnerSet.format_difficulty_with_pct(new_params.get('difficulty', 'Medium'), staged_uniq)

                if room.game_type == 'subanagrams' and getattr(room, 'next_round_board', None) and getattr(room, 'next_round_words', None):
                    staged_seq_len = len(room.next_round_board[0]) if (isinstance(room.next_round_board[0], list)) else len(room.next_round_board)
                    has_full_w = any(len(w) == staged_seq_len for w in room.next_round_words)
                    is_word_mode = (new_params.get('board_format') == 'Word Guaranteed' or new_params.get('mode') == 'word')
                    if is_word_mode:
                        new_params['board_format'] = 'Word Guaranteed'
                        new_params['mode'] = 'word'
                        room.next_round_format = 'Word Guaranteed'
                        if not has_full_w:
                            from subanagrams_generator import generate_subanagrams_board_and_words
                            heal_res = generate_subanagrams_board_and_words(new_params)
                            if heal_res:
                                hb, hw, hbw, hp = heal_res
                                room.next_round_board = hb
                                room.next_round_words = hw
                                room.next_round_word_paths = {w: [] for w in hw}
                                room.next_round_bonus = hbw
                                room.next_round_total_words_count = len(hw)
                    else:
                        new_params['board_format'] = 'Totally Random'
                        new_params['mode'] = 'random'
                        room.next_round_format = 'Totally Random'

                import copy
                if not getattr(room, '_spinner_params_locked', False):
                    room.spinner_params = dict(new_params)
                    room.frozen_revealed_params = copy.deepcopy(new_params)
                    room.next_spinner_params = dict(new_params)
                
                room.next_round_min_length = new_params.get('min_word_length', 3)
                room.spinner_params_revealed = True
                room.was_revealed_this_intermission = True
                room._spinner_params_locked = True
                room._reveal_sync_complete = True
                print(f"[RoomManager] SUCCESS: Revealed pre-generated params for room {room_id} (15s mark: board guaranteed found)")
                return True
            
            # 3. If we are currently loading, skip to avoid double generation
            if getattr(room, 'spinner_params_loading', False):
                return True
                
            room.spinner_params_loading = True

        try:
            is_24h = (room.time_limit >= 7200)
            is_split = (room.game_type == 'split')

            if room.game_type == 'subanagrams':
                need_generate = False
                heal_needed = False
                sp_to_use = None

                with room._state_lock:
                    is_staged = getattr(room, 'spinner_params_generated', False) and bool(getattr(room, 'next_spinner_params', None))
                    curr_p = getattr(room, 'spinner_params', None)
                    if is_staged:
                        sp_to_use = dict(room.next_spinner_params)
                        if curr_p and sp_to_use.get('sequence_length') == curr_p.get('sequence_length') and sp_to_use.get('word_count_range') == curr_p.get('word_count_range') and sp_to_use.get('dictionary') == curr_p.get('dictionary'):
                            need_generate = True
                            sp_to_use = SpinnerSet.generate_subanagrams_params(previous_params=curr_p)
                            sp_to_use['board_dimensions'] = room.board_dimensions
                            sp_to_use['time_limit'] = room.time_limit
                        elif getattr(room, 'next_round_board', None) and getattr(room, 'next_round_words', None):
                            staged_seq_len = len(room.next_round_board[0]) if (isinstance(room.next_round_board[0], list)) else len(room.next_round_board)
                            has_full_w = any(len(w) == staged_seq_len for w in room.next_round_words)
                            is_word_mode = (sp_to_use.get('board_format') == 'Word Guaranteed' or sp_to_use.get('mode') == 'word')
                            if is_word_mode:
                                sp_to_use['board_format'] = 'Word Guaranteed'
                                sp_to_use['mode'] = 'word'
                                room.next_round_format = 'Word Guaranteed'
                                if not has_full_w:
                                    heal_needed = True
                            else:
                                sp_to_use['board_format'] = 'Totally Random'
                                sp_to_use['mode'] = 'random'
                                room.next_round_format = 'Totally Random'
                        print(f"[RoomManager] Using EXISTING staged params for room {room_id} (Lock-protected)")
                    else:
                        need_generate = True
                        sp_to_use = SpinnerSet.generate_subanagrams_params(previous_params=room.spinner_params)
                        sp_to_use['board_dimensions'] = room.board_dimensions
                        sp_to_use['time_limit'] = room.time_limit

                # CRITICAL LOCK-CONTENTION FIX: Run Subanagrams board generation and scoring
                # completely outside room._state_lock so incoming player word submissions never stall or timeout.
                if need_generate or heal_needed:
                    from subanagrams_generator import generate_subanagrams_board_and_words
                    from scoring import calculate_word_score
                    res = generate_subanagrams_board_and_words(sp_to_use)
                    if res:
                        nb, nw, nbw, nparams = res
                        nscores = {
                            w: calculate_word_score(w, nbw, board_format='Normal', return_details=True)
                            for w in nw
                        }
                        with room._state_lock:
                            room.next_round_board = nb
                            room.next_round_words = nw
                            room.next_round_word_paths = {w: [] for w in nw}
                            room.next_round_bonus = nbw
                            room.next_round_bonus_cell = None
                            room.next_round_format = nparams.get('board_format', 'Word Guaranteed')
                            room.next_round_spinner_params = nparams
                            room.next_spinner_params = nparams
                            room.next_round_min_length = nparams.get('min_word_length', 3)
                            room.next_round_total_words_count = len(nw)
                            room.next_round_word_scores = nscores
                            room.spinner_params_generated = True
                            room.spinner_params_loading = False
                            sp_to_use = nparams
                    else:
                        with room._state_lock:
                            room.next_spinner_params = sp_to_use
                            room.spinner_params_generated = True
                            room.spinner_params_loading = False
                else:
                    with room._state_lock:
                        room.spinner_params_loading = False

                if reveal:
                    import copy
                    with room._state_lock:
                        room.spinner_params = dict(sp_to_use)
                        room.frozen_revealed_params = copy.deepcopy(sp_to_use)
                        room.spinner_params_revealed = True
                        room.was_revealed_this_intermission = True
                        room._spinner_params_locked = True
                return True
            
            with room._state_lock:
                # Atomic check: only reuse existing params if they were already generated for this specific upcoming round
                is_already_staged_for_round = getattr(room, 'spinner_params_generated', False) and bool(getattr(room, 'next_spinner_params', None))
                if is_already_staged_for_round:
                    new_params = dict(room.next_spinner_params)
                    print(f"[RoomManager] Using EXISTING staged params for room {room_id} (Lock-protected)")
                else:
                    # Generate new parameters
                    if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                        new_params = room.build_solo_spinner_params()
                    else:
                        new_params = SpinnerSet.generate_params(
                            room.board_dimensions, 
                            is_24h=is_24h, 
                            is_split=is_split, 
                            previous_params=room.spinner_params
                        )
                    # Metadata: Ensure dimensions and time limits are included for the reveal animation
                    new_params['board_dimensions'] = room.board_dimensions
                    new_params['time_limit'] = room.time_limit
                    
                    # Attempt to find a pregenerated board in the cache matching new_params
                    from board_generator import pop_any_cached_board, pop_cached_board, serialize_param_key, refill_board_cache_bg, BoardGenerator
                    bg = BoardGenerator()
                    cached_board_data = None
                    
                    # 1. If in Solo mode and we have non-random parameters, try exact match
                    is_exact_candidate = False
                    if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                        isp = room.initial_solo_params
                        if isp.get('dictionary', 'random') != 'random' or isp.get('word_count_range', 'random') != 'random' or isp.get('board_format', 'random') != 'random' or isp.get('difficulty', 'random') != 'random':
                            is_exact_candidate = True
                            
                    if is_exact_candidate:
                        try:
                            exact_key = serialize_param_key(
                                room.board_dimensions,
                                '',
                                new_params.get('word_count_range', '100-200'),
                                new_params.get('dictionary', 'NWL'),
                                new_params.get('board_format', 'Normal'),
                                new_params.get('min_word_length', 3),
                                new_params.get('difficulty', 'Medium'),
                                use_added_words=new_params.get('use_added_words', False)
                            )
                            exact_res = pop_cached_board(exact_key)
                            if exact_res and len(exact_res) >= 7:
                                board, all_words, bonus_cell, board_format_ret, all_words_dict, ratio, final_bonus_word = exact_res[:7]
                                if all_words and len(all_words) >= 20:
                                    f_min_l = new_params.get('min_word_length', 3)
                                    grid_floor = 3
                                    if '4x6' in room.board_dimensions: grid_floor = 4
                                    elif '5x7' in room.board_dimensions: grid_floor = 5
                                    elif '6x8' in room.board_dimensions or '3x3x3' in room.board_dimensions: grid_floor = 6
                                    f_min_l = max(grid_floor, int(f_min_l) if f_min_l is not None else 3)
                                    dims_parts_ex = str(room.board_dimensions).lower().split('x')
                                    r_ex = int(dims_parts_ex[0]) if len(dims_parts_ex) >= 2 else 4
                                    c_ex = int(dims_parts_ex[1]) if len(dims_parts_ex) >= 2 else 4
                                    achieved_ex_diff = self.board_generator.get_difficulty_label(ratio, rows=r_ex, cols=c_ex, dictionary=new_params.get('dictionary', 'NWL'), min_word_length=f_min_l)
                                    target_ex_diff = SpinnerSet.strip_difficulty_pct(new_params.get('difficulty', 'Medium'))
                                    if achieved_ex_diff == target_ex_diff:
                                        new_params['min_word_length'] = f_min_l
                                        fw_filtered = [w for w in all_words if len(w) >= f_min_l]
                                        actual_cnt = len(fw_filtered)
                                        
                                        f_wc = new_params.get('word_count_range', '100-200')
                                        min_accept = 50
                                        try:
                                            min_accept = int(str(f_wc).split('-')[0])
                                        except:
                                            if '50' in str(f_wc): min_accept = 50
                                            elif '100' in str(f_wc): min_accept = 100
                                            elif '200' in str(f_wc): min_accept = 200
                                            elif '300' in str(f_wc): min_accept = 300
                                            elif '400' in str(f_wc): min_accept = 400
                                            elif '500' in str(f_wc): min_accept = 500
                                        
                                        is_aw_effective = new_params.get('use_added_words', False) or '+ AW' in str(new_params.get('dictionary', '')).upper()
                                        if is_aw_effective:
                                            min_accept = max(300, min_accept)
                                        else:
                                            min_accept = max(50, min_accept)
                                            
                                        if actual_cnt >= min_accept:
                                            target_bw_exact = int(new_params.get('bonus_word_length', 8) or 8)
                                            if not final_bonus_word or len(final_bonus_word) != target_bw_exact or final_bonus_word not in fw_filtered or final_bonus_word.upper().endswith(('ING', 'INGS')) or 'ING' in final_bonus_word.upper():
                                                exact_bw_cands = [w for w in fw_filtered if len(w) == target_bw_exact and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
                                                if exact_bw_cands:
                                                    final_bonus_word = random.choice(exact_bw_cands).upper()
                                                else:
                                                    final_bonus_word = None
                                            if final_bonus_word:
                                                is_aw_eff = new_params.get('use_added_words', False) or '+ AW' in str(new_params.get('dictionary', '')).upper()
                                                actual_wc_range_exact = SpinnerSet.get_factchecked_wc_range(actual_cnt, use_added_words=is_aw_eff, dictionary=new_params.get('dictionary', 'NWL'))
                                                new_params['word_count_range'] = actual_wc_range_exact
                                                cached_board_data = (board, fw_filtered, bonus_cell, board_format_ret, {w: p for w, p in all_words_dict.items() if len(w) >= f_min_l}, ratio, final_bonus_word, new_params)
                                                print(f"[RoomManager] Solo exact cache hit for key: {exact_key[:80]}...")
                        except Exception as e:
                            print(f"[RoomManager] Error querying solo exact cache: {e}")
                            
                    if not cached_board_data:
                        from board_generator import pop_compatible_cached_board, pop_any_cached_board
                        dict_val = new_params.get('dictionary', 'NWL')
                        fmt_val = new_params.get('board_format', 'Normal')
                        use_aw_val = new_params.get('use_added_words', False) or '+ AW' in str(dict_val).upper() or '+AW' in str(dict_val).upper()
                        m_len = new_params.get('min_word_length', 3)
                        bw_len = int(new_params.get('bonus_word_length', 8) or 8)
                        target_sp_range = new_params.get('word_count_range')
                        target_diff_base = SpinnerSet.strip_difficulty_pct(new_params.get('difficulty', 'Medium'))

                        for _ in range(10):
                            candidate = pop_compatible_cached_board(
                                room.board_dimensions, dict_val, fmt_val, m_len, use_aw_val, bonus_word_len=bw_len, difficulty=target_diff_base
                            )
                            if not candidate:
                                break
                            c_b, c_w, c_c, c_f, c_p, c_r, c_bw, c_params = candidate
                            dims_parts_c = str(room.board_dimensions).lower().split('x')
                            r_c = int(dims_parts_c[0]) if len(dims_parts_c) >= 2 else 4
                            c_c_num = int(dims_parts_c[1]) if len(dims_parts_c) >= 2 else 4
                            cand_diff_base = self.board_generator.get_difficulty_label(c_r, rows=r_c, cols=c_c_num, dictionary=dict_val, min_word_length=m_len)
                            if target_diff_base != cand_diff_base:
                                print(f"[generate_spinner_params] Candidate cached board difficulty '{cand_diff_base}' != target '{target_diff_base}'. Discarding.")
                                continue
                            fw_filt = [w for w in c_w if len(w) >= m_len]
                            actual_wc = len(fw_filt)
                            if is_board_count_valid(actual_wc, target_sp_range):
                                if not c_bw or len(c_bw) != bw_len or c_bw not in fw_filt or c_bw.upper().endswith(('ING', 'INGS')) or 'ING' in c_bw.upper():
                                    _c_bw_cands = [w for w in fw_filt if len(w) == bw_len and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
                                    if _c_bw_cands:
                                        c_bw = random.choice(_c_bw_cands).upper()
                                    else:
                                        print(f"[generate_spinner_params] Candidate cached board lacked {bw_len}L bonus word. Discarding.")
                                        continue
                                fp_filt = {w: p for w, p in c_p.items() if w in fw_filt}
                                actual_wc_range = self._get_factchecked_wc_range(actual_wc, use_added_words=use_aw_val, dictionary=dict_val)
                                new_params['word_count_range'] = actual_wc_range
                                if fw_filt:
                                    _cache_actual_min = min(len(w) for w in fw_filt)
                                    if _cache_actual_min > m_len:
                                        new_params['min_word_length'] = _cache_actual_min
                                cached_board_data = (c_b, fw_filt, c_c, c_f, fp_filt, c_r, c_bw, new_params)
                                break
                            else:
                                print(f"[generate_spinner_params] Candidate cached board had {actual_wc} words, mismatching spun range '{target_sp_range}'. Discarding.")

                        if not cached_board_data:
                            print(f"[generate_spinner_params] Cache miss/mismatch for {room_id} — board gen delegated to start_board_search thread")

                    # 3. Stage board data and attach uniqueness ratio to new_params immediately
                    if cached_board_data:
                        board, all_words, bonus_cell, board_format_ret, all_words_dict, ratio, final_bonus_word, b_params = cached_board_data
                        
                        new_params['uniqueness'] = ratio
                        dims_parts = str(room.board_dimensions).lower().split('x')
                        r_cnt = int(dims_parts[0]) if len(dims_parts) >= 2 else 4
                        c_cnt = int(dims_parts[1]) if len(dims_parts) >= 2 else 4
                        final_diff_label = self.board_generator.get_difficulty_label(ratio, rows=r_cnt, cols=c_cnt, dictionary=new_params.get('dictionary', 'NWL'), min_word_length=new_params.get('min_word_length', 3))
                        new_params['difficulty'] = SpinnerSet.format_difficulty_with_pct(final_diff_label, ratio)
                        target_bw_l = int(new_params.get('bonus_word_length', 8) or 8)
                        if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                            _solo_bw = room.initial_solo_params.get('bonus_word_length', 'random')
                            if _solo_bw and str(_solo_bw).lower() not in ['random', '0', 'none', 'null', 'nan']:
                                try:
                                    target_bw_l = int(_solo_bw)
                                except Exception:
                                    pass
                        if not final_bonus_word or len(final_bonus_word) != target_bw_l or final_bonus_word not in all_words or final_bonus_word.upper().endswith(('ING', 'INGS')) or 'ING' in final_bonus_word.upper():
                            _bw_cands = [w for w in all_words if len(w) == target_bw_l and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
                            if _bw_cands:
                                final_bonus_word = random.choice(_bw_cands).upper()
                        if final_bonus_word:
                            new_params['bonus_word_length'] = len(final_bonus_word)
                        
                        # Stage the board data in the room immediately
                        board, all_words, all_words_dict, bonus_cell, final_bonus_word, board_format_ret = self.sanitize_board_for_room_difficulty(
                            room, board, all_words, all_words_dict, bonus_cell, final_bonus_word,
                            board_format_ret, new_params.get('difficulty'), dictionary=new_params.get('dictionary'), min_len=new_params.get('min_word_length')
                        )
                        room.next_round_board = board
                        room.next_round_words = all_words
                        room.next_round_word_paths = all_words_dict
                        room.next_round_bonus_cell = bonus_cell
                        room.next_round_bonus = final_bonus_word
                        room.next_round_format = board_format_ret
                        room.next_round_uniqueness = ratio
                        room.next_round_total_words_count = len(all_words)
                        
                        if hasattr(word_validator, 'word_validator'):
                            word_validator.word_validator.ensure_csw_loaded()
                            room.next_round_csw_only_words = [w for w in all_words if word_validator.word_validator.is_csw_only(w)]
                            room.next_round_added_words = [w for w in all_words if word_validator.word_validator.is_added_word(w)]
                        else:
                            room.next_round_csw_only_words = []
                            room.next_round_added_words = []
                        
                        # Calculate detailed word scores (including bonus letter +3 and bonus word points)
                        scored_dict = {}
                        for w in all_words:
                            path_v = _get_word_path(all_words_dict, w)
                            scored_dict[w] = calculate_word_score(
                                w,
                                bonus_word=final_bonus_word,
                                board_format=board_format_ret,
                                bonus_cell=bonus_cell,
                                board=board,
                                path=path_v,
                                return_details=True
                            )
                        room.next_round_word_scores = scored_dict
                        
                        # Mark room search states as completed
                        room.board_search_started = True
                        room.board_search_started_actual = False
                        room.board_search_loading = False
                        room.solving_complete = True
                        
                        # Trigger background refill for this popped key
                        try:
                            refill_key = serialize_param_key(
                                room.board_dimensions,
                                final_bonus_word,
                                b_params.get('word_count_range', '100-200'),
                                b_params.get('dictionary', 'NWL'),
                                board_format_ret,
                                b_params.get('min_word_length', 3),
                                b_params.get('difficulty', 'Medium'),
                                use_added_words=b_params.get('use_added_words', False)
                            )
                            refill_board_cache_bg(bg, refill_key, target_count=3)
                        except Exception as refill_err:
                            print(f"[RoomManager] Error triggering refill from generate_spinner_params: {refill_err}")

                    room.next_spinner_params = dict(new_params)
                    room.spinner_params_generated = True

                if reveal:
                    # 2. PERFORM THE REVEAL (Making them visible to players at 0:45)
                    if not getattr(room, 'next_round_board', None):
                        print(f"[RoomManager] At 0:45 reveal (slow-path): board not yet found for {room_id}. Ensuring board is ready immediately...")
                        room.ensure_next_board_ready()

                    if getattr(room, 'next_round_spinner_params', None):
                        new_params = dict(room.next_round_spinner_params)
                    elif getattr(room, 'next_spinner_params', None):
                        new_params = dict(room.next_spinner_params)
                    else:
                        new_params = dict(new_params)

                    if getattr(room, 'next_round_words', None):
                        actual_wc = len(room.next_round_words)
                        use_aw_val = (new_params.get('use_added_words', False) or '+ AW' in str(new_params.get('dictionary', '')).upper() or '+AW' in str(new_params.get('dictionary', '')).upper())
                        cur_wc_range = new_params.get('word_count_range')
                        if cur_wc_range and is_board_count_valid(actual_wc, cur_wc_range) and (not use_aw_val or actual_wc >= 300):
                            new_params['word_count_range'] = cur_wc_range
                        else:
                            new_params['word_count_range'] = self._get_factchecked_wc_range(actual_wc, use_added_words=use_aw_val, dictionary=new_params.get('dictionary'))
                        new_params['min_word_length'] = min(len(w) for w in room.next_round_words)
                        if room.game_type != 'subanagrams':
                            target_bw_rev = int(new_params.get('bonus_word_length', 8) or 8)
                            if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                                _s_bw_r = room.initial_solo_params.get('bonus_word_length', 'random')
                                if _s_bw_r and str(_s_bw_r).lower() not in ['random', '0', 'none', 'null', 'nan']:
                                    try:
                                        target_bw_rev = int(_s_bw_r)
                                    except Exception:
                                        pass
                            room.next_round_bonus = pick_distinct_bonus_word(
                                room, getattr(room, 'next_round_bonus', None), room.next_round_words, target_bw_rev,
                                dictionary=new_params.get('dictionary', 'NWL'),
                                is_checkerboard=('checkerboard' in str(getattr(room, 'next_round_format', new_params.get('board_format', ''))).lower()),
                                room_manager=self,
                                extra_exclude=[getattr(room, 'bonus_word', None), getattr(room, 'previous_bonus_word', None)]
                            )
                    if getattr(room, 'next_round_format', None):
                        new_params['board_format'] = room.next_round_format
                    if getattr(room, 'next_round_bonus', None):
                        new_params['bonus_word_length'] = len(room.next_round_bonus)
                    staged_uniq = getattr(room, 'next_round_uniqueness', None)
                    if staged_uniq is None:
                        staged_uniq = new_params.get('uniqueness')
                    if staged_uniq is not None and room.game_type != 'subanagrams':
                        new_params['uniqueness'] = staged_uniq
                        new_params['difficulty'] = SpinnerSet.format_difficulty_with_pct(new_params.get('difficulty', 'Medium'), staged_uniq)

                    import copy
                    if not getattr(room, '_spinner_params_locked', False):
                        room.spinner_params = copy.deepcopy(new_params)
                        room.frozen_revealed_params = copy.deepcopy(new_params)
                        room.next_spinner_params = copy.deepcopy(new_params)
                        room.next_round_spinner_params = copy.deepcopy(new_params)
                    
                    # Update authoritative labels
                    room.next_round_min_length = new_params.get('min_word_length', 3)
                    room.spinner_params_revealed = True
                    room.was_revealed_this_intermission = True
                    room._spinner_params_locked = True  # LOCK: no further overwrites until intermission ends!
                    room._reveal_sync_complete = True
                    print(f"[RoomManager] SUCCESS: Generated and Revealed params for room {room_id} (15s mark: board guaranteed found)")
                else:
                    print(f"[RoomManager] SUCCESS: Silent parameter generation complete for room {room_id}")
                
                # 3. PROACTIVE LEAD TIME: Start searching for the board IMMEDIATELY after params are decided
                # Only start background search if board is not already staged and we haven't locked reveal at 0:45!
                if not getattr(room, 'next_round_board', None) and not reveal:
                    threading.Thread(target=self.start_board_search, args=(room_id,), daemon=True).start()
            
            return True
            
        except Exception as e:
            print(f"[RoomManager] ERROR in generate_spinner_params for {room_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            with room._state_lock:
                room.spinner_params_loading = False

    def apply_easy_fallback_and_search(self, room_id):
        """If by 0:55 during intermission the board has not been found, grab a pregenerated board or use an easy configuration."""
        room = self.get_room(room_id)
        if not room:
            return
        
        with room._state_lock:
            if getattr(room, 'next_round_board', None):
                return
            
            print(f"[RoomManager] At 0:55 intermission: board not yet staged for {room_id}. Consulting pregenerated cache...")
            now = time.time()
            dims = room.board_dimensions

            # 1. First, consult pregenerated boards for an instant board grab
            from board_generator import pop_compatible_cached_board, pop_any_cached_board
            
            cur_p = dict(room.next_spinner_params) if getattr(room, 'next_spinner_params', None) else {}
            target_dict = cur_p.get('dictionary', 'NWL')
            target_fmt = cur_p.get('board_format', 'Normal')
            target_min = cur_p.get('min_word_length', 3)
            use_aw = cur_p.get('use_added_words', False) or '+ AW' in str(target_dict).upper() or '+AW' in str(target_dict).upper()
            bw_len = cur_p.get('bonus_word_length')
            
            target_wc_range = cur_p.get('word_count_range')
            target_diff = cur_p.get('difficulty')
            
            cached_candidate = None
            try:
                # Try compatible match with spun parameters
                cached_candidate = pop_compatible_cached_board(
                    dims, target_dict, target_fmt, target_min, use_aw,
                    bonus_word_len=bw_len, word_count_range=target_wc_range, difficulty=target_diff
                )
                if not cached_candidate and str(target_fmt).lower() == 'normal':
                    # Try any cached Normal board strictly matching dictionary and use_added_words
                    cached_candidate = pop_any_cached_board(
                        dims, min_word_length=target_min, target_format='Normal',
                        dictionary=target_dict, use_added_words=use_aw, word_count_range=target_wc_range, difficulty=target_diff
                    )
            except Exception as pop_err:
                print(f"[RoomManager] Error popping pregenerated board in easy fallback: {pop_err}")

            if cached_candidate:
                c_board, c_words, c_bonus_cell, c_format_ret, c_paths, c_ratio, c_bonus_word, c_params = cached_candidate
                actual_cnt = len(c_words)
                c_use_aw = bool(c_params.get('use_added_words', False)) if c_params else use_aw
                c_dict_val = target_dict or (c_params.get('dictionary') if c_params else 'NWL')
                actual_wc_range = self._get_factchecked_wc_range(actual_cnt, use_added_words=c_use_aw, dictionary=c_dict_val)
                c_min_len = c_params.get('min_word_length', target_min) if c_params else target_min
                if c_words:
                    real_min = min(len(w) for w in c_words)
                    c_min_len = max(c_min_len, real_min)
                
                c_dict = target_dict or (c_params.get('dictionary') if c_params else 'NWL')
                effective_aw = use_aw or c_use_aw
                clean_cd = str(c_dict).replace('+ AW', '').replace('+AW', '').strip()
                if effective_aw:
                    c_dict = f"{clean_cd} + AW"
                else:
                    c_dict = clean_cd

                staged_params = {
                    'difficulty': c_params.get('difficulty', 'Easy') if c_params else 'Easy',
                    'dictionary': c_dict,
                    'board_format': c_format_ret or 'Normal',
                    'min_word_length': c_min_len,
                    'word_count_range': actual_wc_range,
                    'bonus_word_length': len(c_bonus_word) if c_bonus_word else 6,
                    'use_added_words': effective_aw,
                    'board_dimensions': dims,
                    'time_limit': room.time_limit,
                    'generated_at': now
                }
                staged_params = SpinnerSet.sanitize_params(staged_params, dims, room.time_limit >= 7200)

                room.next_round_board = c_board
                room.next_round_words = c_words
                room.next_round_word_paths = c_paths
                room.next_round_bonus_cell = c_bonus_cell
                room.next_round_bonus = c_bonus_word
                room.next_round_format = c_format_ret
                room.next_round_uniqueness = c_ratio
                room.next_round_total_words_count = actual_cnt
                if hasattr(word_validator, 'word_validator'):
                    word_validator.word_validator.ensure_csw_loaded()
                    room.next_round_csw_only_words = [w for w in c_words if word_validator.word_validator.is_csw_only(w)]
                    room.next_round_added_words = [w for w in c_words if word_validator.word_validator.is_added_word(w)]
                else:
                    room.next_round_csw_only_words = []
                    room.next_round_added_words = []
                from scoring import calculate_word_score
                room.next_round_word_scores = {
                    w: calculate_word_score(
                        w, bonus_word=c_bonus_word, board_format=c_format_ret,
                        bonus_cell=c_bonus_cell, board=c_board, path=_get_word_path(c_paths, w), return_details=True
                    )
                    for w in c_words
                }
                room.next_round_total_points = sum((pts['total'] if isinstance(pts, dict) else pts) for pts in room.next_round_word_scores.values())
                room.next_round_counts_by_len = {
                    '_round': room.current_round + 1,
                    '_total': len(c_words),
                    '_min_len': c_min_len,
                    **{str(l): sum(1 for w in c_words if len(w) == l) for l in range(1, 31)}
                }

                room.next_spinner_params = dict(staged_params)
                room.next_round_spinner_params = dict(staged_params)
                if not getattr(room, '_spinner_params_locked', False) and room.state != 'active':
                    room.spinner_params = dict(staged_params)
                room.next_round_min_length = c_min_len
                room.board_search_loading = False
                room.board_search_started_actual = False
                room.solving_complete = True
                print(f"[RoomManager] Popped pregenerated cache board for {room_id} ({actual_cnt} words, range={actual_wc_range}, format={c_format_ret})")
                return

            # 2. If pregenerated cache had no board, generate easy fallback with randomized word count range
            base_dict = 'CSW' if 'CSW' in str(room.current_dictionary).upper() else 'NWL'
            if dims in ['3x3', '4x4', '4x3', '3x4']:
                easy_min = 3
            elif dims in ['5x5', '4x6']:
                easy_min = 4
            else:
                easy_min = 5
                
            # Randomize word count range according to Odds Window probabilities (never hardcode 100-200!)
            if room.time_limit >= 7200:
                easy_wc = '300-400'
            else:
                easy_wc = SpinnerSet._spin_word_count(base_dict, easy_min, 'Easy', dims, use_added_words=False)
            
            easy_params = {
                'difficulty': 'Easy',
                'dictionary': base_dict,
                'board_format': 'Normal',
                'min_word_length': easy_min,
                'word_count_range': easy_wc,
                'bonus_word_length': 6,
                'use_added_words': False,
                'board_dimensions': dims,
                'time_limit': room.time_limit,
                'generated_at': now
            }
            easy_params = SpinnerSet.sanitize_params(easy_params, dims, room.time_limit >= 7200)
            
            room.next_spinner_params = dict(easy_params)
            if not getattr(room, '_spinner_params_locked', False) and room.state != 'active':
                room.spinner_params = dict(easy_params)
            room.next_round_min_length = easy_min
            
            room.board_search_started = False
            room.board_search_started_actual = False
            room.board_search_loading = False
            
        print(f"[RoomManager] Triggering instant easy board retrieval for {room_id} (target_range={easy_wc})...")
        room.ensure_next_board_ready()

    def check_6x8_rescue(self, room):
        """Rescue watchdog for all rooms.
        Rescues rooms stuck in 'loading' for >20s, or in intermission at 50s remaining without a board staged."""
        if not room:
            return
            
        now = time.time()
        is_daily = room.time_limit >= 7200
        
        # 1. Loading state rescue (stuck loading for >25s)
        is_stuck_loading = False
        if room.state == 'loading':
            start_t = getattr(room, '_round_start_init_time', None) or getattr(room, 'creation_time', None) or getattr(room, 'intermission_start_time', None) or now
            elapsed = now - start_t
            if elapsed > 25.0 and not getattr(room, '_did_loading_rescue', False):
                is_stuck_loading = True
                
        # 2. Intermission 15s remaining rescue (normal countdown rooms only)
        # Disable intermission watchdog to prevent mid-intermission parameter changes.
        # Fallback is handled cleanly at start_next_round transition (0s remaining).
        is_stuck_intermission = False
                    
        if is_stuck_loading or is_stuck_intermission:
            with room._state_lock:
                # Re-verify conditions under lock
                if is_stuck_loading:
                    if room.state != 'loading' or getattr(room, '_did_loading_rescue', False):
                        return
                    room._did_loading_rescue = True
                    print(f"[Rescue] Stuck in loading state for {room.room_id}. Forcing immediate rescue...")
                else:
                    if room.next_round_board or getattr(room, '_did_6x8_fallback_rescue', False):
                        return
                    room._did_6x8_fallback_rescue = True
                    print(f"[Rescue] Intermission 50s mark reached without board for {room.room_id}. Pulling cache/fallback instantly...")
                
                # --- PERFORM IMMEDIATE RESCUE ---
                # Attempt to pop ANY pregenerated board from the database cache matching this dimension
                from board_generator import pop_any_cached_board, BoardGenerator
                bg = BoardGenerator()
                
                # Determine min_accept and target min_word_length
                e_format = room.spinner_params.get('board_format', 'Normal') if room.spinner_params else 'Normal'
                e_dict = room.spinner_params.get('dictionary', 'NWL') if room.spinner_params else 'NWL'
                e_use_aw = room.spinner_params.get('use_added_words', False) if room.spinner_params else False
                e_wc = room.spinner_params.get('word_count_range') if room.spinner_params else None
                if not e_wc:
                    e_wc = SpinnerSet._spin_word_count(e_dict, 6 if '6x8' in str(room.board_dimensions) else 3, 'Medium', room.board_dimensions, use_added_words=e_use_aw)
                e_min_len = room.spinner_params.get('min_word_length') if room.spinner_params else None
                e_diff = room.spinner_params.get('difficulty', 'Medium') if room.spinner_params else 'Medium'
                try:
                    search_min = int(e_min_len)
                except:
                    dims = str(room.board_dimensions).lower().replace(" ", "")
                    search_min = 4 if '4x6' in dims else (5 if '5x7' in dims else (6 if '6x8' in dims or '3x3x3' in dims else 3))
                
                min_accept = 50
                if e_wc:
                    try:
                        min_accept = int(str(e_wc).split('-')[0])
                    except:
                        if '50' in str(e_wc): min_accept = 50
                        elif '100' in str(e_wc): min_accept = 100
                        elif '200' in str(e_wc): min_accept = 200
                        elif '300' in str(e_wc): min_accept = 300
                        elif '400' in str(e_wc): min_accept = 400
                        elif '500' in str(e_wc): min_accept = 500
                if e_use_aw:
                    min_accept = max(300, min_accept)
                else:
                    min_accept = max(50, min_accept)

                cached_res = None
                for _ in range(10):
                    from board_generator import pop_compatible_cached_board
                    candidate = pop_compatible_cached_board(
                        room.board_dimensions,
                        e_dict,
                        e_format,
                        search_min,
                        e_use_aw,
                        bonus_word_len=room.spinner_params.get('bonus_word_length') if room.spinner_params else None
                    )
                    if not candidate:
                        break
                    _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _fparams = candidate
                    _fw_filtered = [w for w in _fw if len(w) >= search_min]
                    if len(_fw_filtered) >= min_accept:
                        # Found valid cached board! Pack as 9 elements
                        _fw = _fw_filtered
                        _fp = {w: p for w, p in _fp.items() if len(w) >= search_min}
                        _ctr = _fparams.get('word_count_range') if _fparams else None
                        cached_res = (_fb, _fw, _fc, _ff, _fp, _fr, _fbw, _ctr, _fparams)
                        break
                    else:
                        print(f"[Rescue] Candidate cached board had only {len(_fw_filtered)} words of length >= {search_min} (needed {min_accept}). Discarding and retrying...")
                
                if not cached_res:
                    print(f"[Rescue] Cache empty or no board met word count floor. Pulling emergency fallback board.")
                    e_format = room.spinner_params.get('board_format', 'Normal') if room.spinner_params else 'Normal'
                    e_dict = room.spinner_params.get('dictionary', 'NWL') if room.spinner_params else 'NWL'
                    e_use_aw = room.spinner_params.get('use_added_words', False) if room.spinner_params else False
                    e_wc = room.spinner_params.get('word_count_range') if room.spinner_params else None
                    if not e_wc:
                        e_wc = SpinnerSet._spin_word_count(e_dict, 6 if '6x8' in str(room.board_dimensions) else 3, 'Medium', room.board_dimensions, use_added_words=e_use_aw)
                    e_min_len = room.spinner_params.get('min_word_length') if room.spinner_params else None
                    e_diff = room.spinner_params.get('difficulty', 'Medium') if room.spinner_params else 'Medium'
                    cached_res = get_emergency_fallback_board(
                        room.board_dimensions, e_format, room.time_limit,
                        dictionary=e_dict, use_added_words=e_use_aw, target_range=e_wc, min_word_length=e_min_len, difficulty=e_diff
                    )
                    
                if cached_res:
                    if len(cached_res) >= 9:
                        cboard, cwords, cbonus_cell, cfmt, cpaths, cratio, cbonus_word, ctr, cparams = cached_res
                    else:
                        cboard, cwords, cbonus_cell, cfmt, cpaths, cratio, cbonus_word, ctr = cached_res
                        cparams = {}
                        
                    print(f"[Rescue-Success] Found fallback board with {len(cwords)} words!")
                    
                    # Force apply parameters
                    cparams = dict(cparams) if cparams else {}
                    dict_val = (room.spinner_params.get('dictionary') if room.spinner_params else None) or cparams.get('dictionary', 'NWL')
                    use_aw_val = (room.spinner_params.get('use_added_words') if room.spinner_params else None) or cparams.get('use_added_words', False)
                    clean_d = str(dict_val).replace('+ AW', '').replace('+AW', '').strip()
                    if use_aw_val:
                        dict_val = f"{clean_d} + AW"
                    else:
                        dict_val = clean_d
                        
                    actual_wc = len(cwords)
                    # AW boards use the same full scale — no '300-400' floor.
                    wc_label = self._get_factchecked_wc_range(actual_wc, use_added_words=use_aw_val, dictionary=dict_val)

                    if not getattr(room, '_spinner_params_locked', False) and room.state != 'active':
                        room.spinner_params['dictionary'] = dict_val
                        room.spinner_params['difficulty'] = cparams.get('difficulty') or (room.spinner_params.get('difficulty') if room.spinner_params else 'Medium')
                        room.spinner_params['word_count_range'] = wc_label
                        room.spinner_params['board_format'] = cfmt or (room.spinner_params.get('board_format') if room.spinner_params else 'Normal')
                        _res_min = cparams.get('min_word_length') or (room.spinner_params.get('min_word_length') if room.spinner_params else 3)
                        if cwords:
                            _act_min = min(len(w) for w in cwords)
                            if _act_min > _res_min:
                                _res_min = _act_min
                        room.spinner_params['min_word_length'] = _res_min
                        room.spinner_params['use_added_words'] = use_aw_val
                        room.spinner_params['bonus_word_length'] = cparams.get('bonus_word_len', len(cbonus_word) if cbonus_word else 6)
                        
                        # Enforce sanitization
                        room.spinner_params = SpinnerSet.sanitize_params(room.spinner_params, room.board_dimensions, room.time_limit >= 7200)

                        room.next_spinner_params = dict(room.spinner_params)
                        room.next_round_spinner_params = dict(room.spinner_params)
                        room.spinner_params_generated = True
                    room._reveal_sync_complete = True
                    
                    if is_stuck_loading:
                        # For loading state, apply to current round directly and start game
                        self._apply_kickstart_results(room.room_id, (cboard, cwords, cbonus_cell, cfmt, cpaths, cratio, cbonus_word), room.spinner_params['min_word_length'], room.time_limit >= 7200)
                    else:
                        # For intermission, stage for the next round transition
                        room.next_round_board = cboard
                        room.next_round_words = cwords
                        room.next_round_word_paths = cpaths
                        room.next_round_bonus_cell = cbonus_cell
                        room.next_round_bonus = cbonus_word
                        room.next_round_format = cfmt
                        room.next_round_uniqueness = cratio
                        room.next_round_total_words_count = len(cwords)
                        
                        # Generate detailed scores (including bonus letter +3 and bonus word points)
                        scored_dict = {}
                        for w in cwords:
                            path_v = _get_word_path(cpaths, w)
                            scored_dict[w] = calculate_word_score(
                                w,
                                bonus_word=cbonus_word,
                                board_format=cfmt,
                                bonus_cell=cbonus_cell,
                                board=cboard,
                                path=path_v,
                                return_details=True
                            )
                        room.next_round_word_scores = scored_dict
                        room.solving_complete = True
                        room.board_search_started = True
                        room.board_search_loading = False
                        room.board_search_started_actual = False
                    return
                
                # If cache is completely empty, use simplified fast-generating parameters
                fast_min = 6
                use_aw = False
                fast_params = {
                    'difficulty': 'Medium',
                    'dictionary': 'NWL',
                    'word_count_range': '300-400' if room.time_limit >= 7200 else '100-200',
                    'board_format': 'Valued Letters' if room.time_limit >= 7200 else 'Normal',
                    'min_word_length': fast_min,
                    'bonus_word_length': 6,
                    'use_added_words': use_aw,
                    'generated_at': now,
                    'board_dimensions': room.board_dimensions,
                    'time_limit': room.time_limit
                }
                fast_params = SpinnerSet.sanitize_params(fast_params, room.board_dimensions, room.time_limit >= 7200)
                # ISSUE 5: Only overwrite spinner_params if they are NOT already locked/revealed
                if not getattr(room, '_spinner_params_locked', False) and not getattr(room, 'spinner_params_revealed', False):
                    room.spinner_params = dict(fast_params)
                    room.next_spinner_params = dict(fast_params)
                    room.next_round_spinner_params = dict(fast_params)
                room.spinner_params_generated = True
                room.spinner_params_revealed = True
                room._reveal_sync_complete = True
                
                # Reset staging and search flags to force a quick live generation
                room.next_round_board = None
                room.next_round_words = None
                room.next_round_word_paths = None
                room.next_round_word_scores = None
                room.next_round_total_points = 0
                room.next_round_total_words_count = 0
                room.next_round_bonus = None
                room.next_round_bonus_cell = None
                room.next_round_format = None
                room.next_round_uniqueness = None
                
                room.board_search_started = False
                room.board_search_started_actual = False
                room.board_search_loading = False
                
                if is_stuck_loading:
                    # In loading state, generate live on this thread with a short timeout
                    # to unblock the room immediately
                    print("[Rescue] Cache empty. Generating fallback board synchronously...")
                    try:
                        b_word = self._get_bonus_word(length=6, dictionary='NWL')
                        gen_res = self.board_generator.generate_board(
                            dimensions=room.board_dimensions,
                            bonus_word=b_word,
                            word_count_range=room.spinner_params.get('word_count_range', '100-200'),
                            board_format=room.spinner_params.get('board_format', 'Normal'),
                            dictionary=room.spinner_params.get('dictionary', 'NWL'),
                            min_word_length=6,
                            is_emergency=True,
                            use_added_words=False
                        )
                        if gen_res and len(gen_res) >= 7:
                            self._apply_kickstart_results(room.room_id, gen_res, 6, room.time_limit >= 7200)
                        else:
                            # Emergency fallback: change state to intermission to try again
                            room.state = 'intermission'
                            room.intermission_start_time = now
                    except Exception as e:
                        print(f"[Rescue-Error] Synch generator failed: {e}")
                        room.state = 'intermission'
                        room.intermission_start_time = now
                else:
                    # In intermission, trigger background search
                    import threading
                    threading.Thread(target=self.start_board_search, args=(room.room_id,), daemon=True).start()
    
    def start_board_search(self, room_id):
        """Start board search using Spinner Set parameters (called at 15s remaining)"""
        room = self.get_room(room_id)
        if not room:
            print(f"[RoomManager] ERROR: Room {room_id} not found")
            return False
            
        # TRACK search start time to handle timeouts in start_next_round (if not already set)
        if not hasattr(room, '_last_search_start_time'):
            room._last_search_start_time = time.time()
            
        if not getattr(room, 'spinner_params_generated', False):
            print(f"[RoomManager] Search requested but spinner params missing for room {room_id}. Generating now...")
            self.generate_spinner_params(room_id)
            
        # ATOMIC GUARD: Prevent redundant threads while allowing the legitimate one to proceed
        with room._state_lock:
            # Check if ALREADY STARTED (To prevent redundant generation)
            if getattr(room, 'board_search_started_actual', False) or getattr(room, 'next_round_board', None):
                return False
            
            # Start the search process
            room.board_search_loading = True
            room.board_search_started_actual = True # New flag to track actual execution
            room.board_search_started = True 
            room._last_search_start_time = time.time()
            
        print(f"[RoomManager] Starting board search process for room {room_id}")
        
        try:
            # AUTHORITATIVE: Use the specific params intended for this background search.
            if room.game_type == 'subanagrams':
                def subanagrams_search_worker():
                    try:
                        from spinner_set import SpinnerSet
                        from subanagrams_generator import generate_subanagrams_board_and_words
                        # MULTIPLAYER SUBANAGRAMS: Strictly generate fresh random parameters for next round.
                        # Do NOT inherit frozen_revealed_params from the currently active round!
                        curr_p = getattr(room, 'spinner_params', None)
                        p = getattr(room, 'next_spinner_params', None)
                        if not p or not p.get('sequence_length') or (curr_p and p.get('sequence_length') == curr_p.get('sequence_length') and p.get('word_count_range') == curr_p.get('word_count_range') and p.get('dictionary') == curr_p.get('dictionary')):
                            p = SpinnerSet.generate_subanagrams_params(previous_params=curr_p)
                            room.next_spinner_params = p
                        res = generate_subanagrams_board_and_words(p)
                        if res:
                            b_board, b_words, b_bonus, fparams = res
                            is_word_mode = (p.get('mode') == 'word' or p.get('board_format') == 'Word Guaranteed')
                            final_fmt = 'Word Guaranteed' if is_word_mode else 'Totally Random'
                            fparams['board_format'] = final_fmt
                            fparams['mode'] = 'word' if is_word_mode else 'random'
                            from scoring import calculate_word_score
                            b_scores = {
                                w: calculate_word_score(w, b_bonus, board_format='Normal', return_details=True)
                                for w in b_words
                            }
                            with room._state_lock:
                                room.next_round_board = b_board
                                room.next_round_words = b_words
                                room.next_round_word_paths = {w: [] for w in b_words}
                                room.next_round_bonus = b_bonus
                                room.next_round_format = final_fmt
                                room.next_round_bonus_cell = None
                                room.next_round_uniqueness = 0.0
                                room.next_round_spinner_params = fparams
                                room.next_spinner_params = fparams
                                room.spinner_params_generated = True
                                room.next_round_min_length = fparams.get('min_word_length', 3)
                                if getattr(room, 'spinner_params_revealed', False) and getattr(room, 'state', None) == 'intermission':
                                    room.frozen_revealed_params = copy.deepcopy(fparams)
                                room.next_round_word_scores = b_scores
                                room.next_round_total_words_count = len(b_words)
                                room.solving_complete = True
                                room.board_search_loading = False
                                room.board_search_started_actual = False
                                print(f"[RoomManager] Subanagrams board generated for {room_id}: {b_board[0]} ({len(b_words)} words, format={final_fmt}, bonus={b_bonus})")
                    except Exception as ex:
                        print(f"[RoomManager] Error generating Subanagrams board: {ex}")
                        room.board_search_loading = False
                        room.board_search_started_actual = False

                threading.Thread(target=subanagrams_search_worker, daemon=True).start()
                return True

            params = getattr(room, 'next_spinner_params', None)
            if not params:
                if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                    room.next_spinner_params = room.build_solo_spinner_params()
                else:
                    is_24h = room.time_limit >= 7200
                    is_split = (room.game_type == 'split')
                    room.next_spinner_params = SpinnerSet.generate_params(room.board_dimensions, is_24h, is_split, previous_params=getattr(room, 'spinner_params', None))
                params = room.next_spinner_params
            params = SpinnerSet.sanitize_params(params, room.board_dimensions, room.time_limit >= 7200)
            if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                _solo_d = room.initial_solo_params.get('difficulty', 'random')
                if _solo_d and str(_solo_d).lower() != 'random':
                    params['difficulty'] = SpinnerSet.format_difficulty_with_pct(_solo_d)
            room.next_spinner_params = params
            launched_generated_at = params.get('generated_at') if params else None
            
            if not params:
                # If still no params, we must wait or fail to avoid bleeding from previous round
                print(f"[RoomManager] WARNING: No next_spinner_params for {room_id}. Waiting for generation...")
                return False
                
            if room.time_limit >= 7200:
                if params:
                    params['board_format'] = 'Valued Letters'
                    params['word_count_range'] = '300-400'
                fmt = 'Valued Letters'
                wc_range = '300-400'
            else:
                fmt = params.get('board_format', 'Normal')
                wc_range = params.get('word_count_range', '100-200')
            
            # AUTHORITATIVE INTEGER CASTING: User mandate - ensure lengths are never interpreted as strings
            try:
                min_l = int(params.get('min_word_length', 3))
            except:
                min_l = 3
            try:
                bw_l_raw = params.get('bonus_word_length', 8)
                bw_l = min(max(int(bw_l_raw), 6), 10)
                params['bonus_word_length'] = bw_l
            except:
                bw_l = 8
                params['bonus_word_length'] = bw_l
            
            # Snapshot for the upcoming round logic
            room.next_round_min_length = min_l
            
            # ALWAYS get a bonus word based on the spinner length
            is_checkerboard = 'checkerboard' in str(fmt).lower()
            
            # EXTREME EXCLUSION: Exclude current word, staged word, and recent history
            exclude_list = []
            if getattr(room, 'bonus_word', None):
                exclude_list.append(room.bonus_word)
            if getattr(room, 'next_round_bonus', None):
                exclude_list.append(room.next_round_bonus)
            
            # Maintain a rolling history of the last 20 words used to prevent re-rolls
            history = getattr(room, 'bonus_word_history', [])
            exclude_list.extend(history)

            # 1. Choose bonus word based on difficulty constraints
            bonus_word = self._get_bonus_word(
                bw_l, 
                params.get('dictionary', 'NWL'),
                alternating=is_checkerboard,
                difficulty=params.get('difficulty', 'Medium'),
                exclude=exclude_list
            )
            
            room.next_round_bonus = bonus_word
            
            print(f"[RoomManager] Bonus word selected for next round: '{bonus_word}'")
            room.board_search_started = True
            search_round = room.current_round
            
            # Start board generation in background thread
            def generate_in_background():
                nonlocal bonus_word, params, launched_generated_at
                from scoring import calculate_word_score
                
                # Capture params locally for thread safety
                search_wc = params.get('word_count_range')
                search_dict = params.get('dictionary')
                search_fmt = params.get('board_format')
                search_min = params.get('min_word_length')
                search_diff = params.get('difficulty')
                
                try:
                    print(f"[RoomManager] Background board generation started for {room_id}...")
                    
                    # STEP 0: Try pop_any_cached_board FIRST — instant if cache has anything
                    from board_generator import pop_any_cached_board as _pop_any
                    
                    # Calculate min_accept for the search
                    _sp_dict = str(params.get('dictionary', 'NWL')).upper()
                    use_aw_flag = params.get('use_added_words', False) or ('+ AW' in _sp_dict) or ('+AW' in _sp_dict) or (_sp_dict in ['AW', 'ADDED_WORDS'])
                    min_accept = 50
                    if search_wc:
                        try:
                            if str(search_wc).startswith('500'):
                                min_accept = 500  # 500+ means AT LEAST 500
                            else:
                                min_accept = int(str(search_wc).split('-')[0])
                        except:
                            if '500' in str(search_wc): min_accept = 500
                            elif '400' in str(search_wc): min_accept = 400
                            elif '300' in str(search_wc): min_accept = 300
                            elif '200' in str(search_wc): min_accept = 200
                            elif '100' in str(search_wc): min_accept = 100
                            elif '50' in str(search_wc):  min_accept = 50
                    if room.time_limit >= 7200:
                        min_accept = 300
                        max_limit = 399
                        search_wc = '300-400'
                        search_fmt = 'Valued Letters'
                        use_aw_flag = False
                    elif use_aw_flag:
                        min_accept = max(300, min_accept)
                        if str(search_wc) in ['50-100', '100-200', '200-300']:
                            search_wc = '300-400'
                    else:
                        min_accept = max(50, min_accept)
                    
                    max_limit = 99999
                    if search_wc:
                        try:
                            parts = str(search_wc).split('-')
                            if len(parts) == 2:
                                max_limit = int(parts[1]) - 1
                        except:
                            pass
                        
                    _pre = None
                    for _ in range(10):
                        from board_generator import pop_compatible_cached_board
                        candidate = pop_compatible_cached_board(
                            room.board_dimensions,
                            search_dict,
                            search_fmt,
                            search_min,
                            use_aw_flag,
                            bonus_word_len=len(bonus_word) if bonus_word else None,
                            word_count_range='300-400' if room.time_limit >= 7200 else search_wc,
                            difficulty=search_diff
                        )
                        if not candidate:
                            break
                        _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _fparams = candidate
                        _fw_filtered = [w for w in _fw if len(w) >= search_min]
                        search_diff_base = SpinnerSet.strip_difficulty_pct(search_diff)
                        is_diff_match = True
                        if search_diff_base and str(search_diff_base).lower() != 'random':
                            b_dims_c = room.board_dimensions.split('x')
                            d_val_c = int(b_dims_c[0]) if len(b_dims_c) == 3 else 1
                            rows_c = int(b_dims_c[1] if len(b_dims_c) == 3 else b_dims_c[0])
                            cols_c = int(b_dims_c[2] if len(b_dims_c) == 3 else b_dims_c[1])
                            _cand_diff = self.board_generator.get_difficulty_label(_fr, rows_c, cols_c, search_dict, depth=d_val_c, min_word_length=search_min)
                            if _cand_diff != search_diff_base:
                                is_diff_match = False

                        matching_min_cnt = sum(1 for w in _fw_filtered if len(w) == search_min)
                        is_24h_ok = (not (room.time_limit >= 7200)) or (300 <= len(_fw_filtered) <= 399)
                        if is_24h_ok and is_board_count_valid(len(_fw_filtered), search_wc) and len(_fw_filtered) >= min_accept and is_diff_match and matching_min_cnt >= 3:
                            # Accept board without discarding valid grid words
                            _fw = _fw_filtered
                            _fp = {w: p for w, p in _fp.items() if w in _fw}
                            _pre = (_fb, _fw, _fc, _ff, _fp, _fr, _fbw, _fparams)
                            break
                        else:
                            print(f"[start_board_search] Cache pop candidate (words={len(_fw_filtered)}, min_len_cnt={matching_min_cnt}, ratio={_fr}) did not match range '{search_wc}' / diff '{search_diff}'. Discarding and retrying...")
                            
                    if _pre:
                        if (
                            getattr(room, '_spinner_params_locked', False)
                            or getattr(room, 'spinner_params_revealed', False)
                            or getattr(room, '_fallback_at_50s_done', False)
                        ) and getattr(room, 'next_round_board', None):
                            return
                        _board, _words, _bonus_c, _fmt, _paths, _ratio, _bonus_word, _params = _pre
                        print(f"[RoomManager] INSTANT CACHE HIT in board search for {room_id}: {len(_words)} words")
                        # Preserve spun parameters generated by SpinnerSet — do not overwrite with cached board params
                        params = dict(params)
                        actual_wc = len(_words)
                        use_aw = params.get('use_added_words', False)
                        if room.time_limit >= 7200:
                            wc_label = '300-400'
                        else:
                            wc_label = self._get_factchecked_wc_range(actual_wc, use_added_words=use_aw, dictionary=params.get('dictionary'))
                        params['word_count_range'] = wc_label
                        params['uniqueness'] = _ratio
                        b_dims_p = room.board_dimensions.split('x')
                        d_val_p = int(b_dims_p[0]) if len(b_dims_p) == 3 else 1
                        rows_p = int(b_dims_p[1] if len(b_dims_p) == 3 else b_dims_p[0])
                        cols_p = int(b_dims_p[2] if len(b_dims_p) == 3 else b_dims_p[1])
                        achieved_hit_diff = self.board_generator.get_difficulty_label(_ratio, rows_p, cols_p, search_dict, depth=d_val_p, min_word_length=search_min)
                        params['difficulty'] = SpinnerSet.format_difficulty_with_pct(achieved_hit_diff, _ratio)
                        
                        if _words:
                            _actual_min_board = min(len(w) for w in _words)
                            if _actual_min_board > params.get('min_word_length', 3):
                                params['min_word_length'] = _actual_min_board
                        
                        _use_word = pick_distinct_bonus_word(
                            room, _bonus_word or bonus_word, _words, bw_l,
                            dictionary=params.get('dictionary', 'NWL'),
                            is_checkerboard=('checkerboard' in str(_fmt).lower()),
                            room_manager=self,
                            extra_exclude=[getattr(room, 'bonus_word', None), getattr(room, 'previous_bonus_word', None)]
                        )
                        if _use_word:
                            params['bonus_word_length'] = len(_use_word)

                        room.next_spinner_params = dict(params)
                        if not getattr(room, '_spinner_params_locked', False) and room.state != 'active':
                            room.spinner_params['word_count_range'] = wc_label
                            if 'min_word_length' in params:
                                room.spinner_params['min_word_length'] = params['min_word_length']
                            if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                                room.frozen_revealed_params['word_count_range'] = wc_label
                                if 'min_word_length' in params:
                                    room.frozen_revealed_params['min_word_length'] = params['min_word_length']

                        room.next_round_words = _words
                        room.next_round_word_paths = _paths
                        room.next_round_bonus_cell = _bonus_c
                        room.next_round_bonus = _use_word
                        room.next_round_format = _fmt
                        room.next_round_uniqueness = _ratio
                        if hasattr(word_validator, 'word_validator'):
                            word_validator.word_validator.ensure_csw_loaded()
                            room.next_round_csw_only_words = [w for w in _words if word_validator.word_validator.is_csw_only(w)]
                            room.next_round_added_words = [w for w in _words if word_validator.word_validator.is_added_word(w)]
                        else:
                            room.next_round_csw_only_words = []
                            room.next_round_added_words = []
                        room.next_round_word_scores = {
                            w: calculate_word_score(
                                w, bonus_word=_use_word, board_format=_fmt,
                                bonus_cell=_bonus_c, board=_board, path=_get_word_path(_paths, w), return_details=True
                            )
                            for w in _words
                        }
                        room.next_round_total_points = sum((pts['total'] if isinstance(pts, dict) else pts) for pts in room.next_round_word_scores.values())
                        room.next_round_spinner_params = dict(params)
                        room.next_round_spinner_params['board_format'] = _fmt
                        room.next_round_spinner_params['word_count_range'] = wc_label
                        room.next_round_total_words_count = actual_wc
                        room.next_round_counts_by_len = {
                            '_round': room.current_round + 1,
                            '_total': actual_wc,
                            '_min_len': params.get('min_word_length', 3),
                            **{str(l): sum(1 for w in (_words or []) if len(w) == l) for l in range(1, 31)}
                        }
                        room.next_round_board = _board
                        room.solving_complete = True
                        room.board_search_loading = False
                        room.board_search_started_actual = False
                        return
                    
                    # STEP 1: No cache hit — proceed with live generation

                    print(f"[RoomManager] [DEBUG-GEN] Room {room_id} calling generate_board with search_min={search_min}, range={search_wc}")
                    
                    # USER REQUEST: Zero-latency 0:00 loading.
                    # We MUST ensure background search times out BEFORE the round ends.
                    # Target finish: 10s before round ends.
                    search_timeout = max(10.0, float(room.time_limit) - 10.0)
                    if room.time_limit >= 7200: search_timeout = 180.0 # 24h rooms get 3 mins
                    else: search_timeout = min(search_timeout, 120.0) # Cap at 2 mins for standard rooms
                    
                    # Database board check to prevent duplicate boards across restarts
                    recent_boards = []
                    try:
                        with get_db() as conn_b:
                            cursor_b = conn_b.cursor()
                            cursor_b.execute("SELECT board_json FROM round_history WHERE room_id = ? ORDER BY id DESC LIMIT 5", (room_id,))
                            rows_b = cursor_b.fetchall()
                            for row_b in rows_b:
                                if row_b[0]:
                                    try:
                                        recent_boards.append(json.loads(row_b[0]))
                                    except:
                                        pass
                    except Exception as db_err:
                        print(f"[RoomManager] Error querying recent boards: {db_err}")

                    # Anti-duplicate board protection loop
                    board_attempts = 0
                    while board_attempts < 4:
                        import random
                        random.seed()
                        
                        _dict_str = str(params.get('dictionary', 'NWL')).upper()
                        use_aw_flag = params.get('use_added_words', False) or ('+ AW' in _dict_str) or ('+AW' in _dict_str) or (_dict_str in ['AW', 'ADDED_WORDS'])
                        token = use_added_words_ctx.set(use_aw_flag)
                        try:
                            # ROBUST CALL: Use keyword arguments to prevent positional mismatch
                            res = self.board_generator.generate_board(
                                dimensions=room.board_dimensions,
                                bonus_word=bonus_word,
                                word_count_range=search_wc,
                                dictionary=search_dict,
                                board_format=search_fmt,
                                min_word_length=search_min,
                                difficulty=search_diff,
                                timeout=search_timeout,
                                use_added_words=use_aw_flag
                            )
                        finally:
                            use_added_words_ctx.reset(token)
                        
                        # ROBUST UNPACKING: Support legacy 6-tuple or modern 7-tuple
                        if len(res) == 7:
                            board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio, final_bonus_word = res
                        else:
                            board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio = res
                            final_bonus_word = bonus_word
                        
                        # Compare to current board and previous round board to guarantee uniqueness
                        is_duplicate = False
                        if getattr(room, 'board', None) == board or getattr(room, 'previous_board', None) == board:
                            is_duplicate = True
                        else:
                            for rb in recent_boards:
                                if rb == board:
                                    is_duplicate = True
                                    break
                        fw_cnt = sum(1 for w in (all_words or []) if len(w) >= search_min)
                        if fw_cnt < min_accept and board_attempts < 3:
                            print(f"[RoomManager] Generated board has only {fw_cnt} words of len>={search_min} (needed >= {min_accept}). Retrying...")
                            board_attempts += 1
                            continue

                        # If Added Words game and count is below 300, perform rescue sweep to ensure >= 300 words
                        if use_aw_flag and fw_cnt < 300:
                            print(f"[RoomManager] Added Words board has only {fw_cnt} words (< 300). Attempting rescue fill...")
                            try:
                                b_dims = room.board_dimensions.split('x')
                                r_cnt = int(b_dims[1] if len(b_dims) == 3 else b_dims[0])
                                c_cnt = int(b_dims[2] if len(b_dims) == 3 else b_dims[1])
                                d_cnt = int(b_dims[0]) if len(b_dims) == 3 else 1
                                r_board = self.board_generator._perform_rescue_sweep(
                                    board, r_cnt, c_cnt, d_cnt, search_dict, search_min, 300, 400,
                                    set(), search_diff, rescue_depth=25, is_checkerboard=('checkerboard' in str(search_fmt).lower()),
                                    board_format=search_fmt, use_added_words=True
                                )
                                r_solve = self.board_generator._solve_board(
                                    r_board, search_dict, (0, 99999), search_min, max_depth=25, store_paths=True, timeout=10.0, use_added_words=True
                                )
                                r_cnt_w = len(r_solve)
                                if r_cnt_w >= 300:
                                    board = r_board
                                    all_words = sorted(list(r_solve.keys()))
                                    all_words_dict = r_solve
                                    fw_cnt = r_cnt_w
                                    print(f"[RoomManager] Rescue sweep successfully raised AW board words to {fw_cnt}.")
                                else:
                                    print(f"[RoomManager] Rescue sweep reached {r_cnt_w} words (< 300). Attempting high-density fill...")
                                    from board_generator import LETTER_FREQ_SUPER_DENSITY
                                    for _aw_try in range(10):
                                        dense_b = self.board_generator._create_normal_board(
                                            r_cnt, c_cnt, LETTER_FREQ_SUPER_DENSITY, depth=d_cnt,
                                            difficulty='Easy', dictionary=search_dict, word_count_range='300-400', use_added_words=True
                                        )
                                        dense_s = self.board_generator._solve_board(dense_b, search_dict, (0, 99999), search_min, max_depth=25, store_paths=True, timeout=10.0, use_added_words=True)
                                        if len(dense_s) >= 300:
                                            board = dense_b
                                            all_words = sorted(list(dense_s.keys()))
                                            all_words_dict = dense_s
                                            fw_cnt = len(dense_s)
                                            print(f"[RoomManager] High-density fallback reached {fw_cnt} words for AW board.")
                                            break
                                    if fw_cnt < 300 and board_attempts < 4:
                                        board_attempts += 1
                                        continue
                            except Exception as aw_resc_err:
                                print(f"[RoomManager] Error during AW rescue sweep: {aw_resc_err}")

                        # USER MANDATE: Ensure generated board difficulty strictly matches requested difficulty
                        search_diff_base = SpinnerSet.strip_difficulty_pct(search_diff)
                        if search_diff_base and str(search_diff_base).lower() != 'random':
                            b_dims = room.board_dimensions.split('x')
                            d_val = int(b_dims[0]) if len(b_dims) == 3 else 1
                            rows = int(b_dims[1] if len(b_dims) == 3 else b_dims[0])
                            cols = int(b_dims[2] if len(b_dims) == 3 else b_dims[1])
                            gen_diff = self.board_generator.get_difficulty_label(u_ratio, rows, cols, search_dict, depth=d_val, board=board, min_word_length=search_min)
                            if gen_diff != search_diff_base:
                                if search_diff_base == 'Easy':
                                    _, easy_max = self.board_generator._get_uniqueness_range('Easy', rows, cols, search_dict, d_val, min_word_length=search_min)
                                    board, all_words_dict, u_ratio = self.board_generator._sanitize_uniqueness(board, d_val, search_dict, search_min, easy_max, rows, cols)
                                    all_words = sorted(list(all_words_dict.keys()))
                                    gen_diff = self.board_generator.get_difficulty_label(u_ratio, rows, cols, search_dict, depth=d_val, board=board, min_word_length=search_min)
                                elif search_diff_base in ['Medium', 'Hard']:
                                    med_min, _ = self.board_generator._get_uniqueness_range(search_diff_base, rows, cols, search_dict, d_val, min_word_length=search_min)
                                    board, all_words_dict, u_ratio = self.board_generator._boost_uniqueness(board, d_val, search_dict, search_min, med_min, rows, cols)
                                    all_words = sorted(list(all_words_dict.keys()))
                                    gen_diff = self.board_generator.get_difficulty_label(u_ratio, rows, cols, search_dict, depth=d_val, board=board, min_word_length=search_min)
                                if gen_diff != search_diff_base and board_attempts < 4:
                                    print(f"[RoomManager] Generated board difficulty '{gen_diff}' (ratio {u_ratio:.3f}) != requested '{search_diff_base}'. Retrying generation attempt {board_attempts + 1}...")
                                    board_attempts += 1
                                    continue

                        if not is_duplicate:
                            break
                        
                        print(f"[RoomManager] WARNING: Generated board for room {room_id} is IDENTICAL to current/previous round board. Retrying...")
                        board_attempts += 1
                    
                    # RE-VERIFY: Ensure we are targeting the ACTIVE room instance
                    target_room = room
                    active_room = self.get_room(room_id)
                    if active_room and active_room is not room:
                        print(f"[RoomManager] Search complete for {room_id}, but room was reconstructed. Redirecting to active instance.")
                        target_room = active_room

                    # STALE BOARD SEARCH PROTECTION:
                    # If target_room's current_round is greater than search_round,
                    # then the round transition has already occurred and this background board is stale.
                    if target_room.current_round > search_round:
                        print(f"[RoomManager] Stale board search discarded for {room_id} (search_round: {search_round}, current_round: {target_room.current_round})")
                        return

                    # STALE PARAMETERS & LOCKED REVEAL CHECK:
                    # Once 0:50 fallback or 0:45 reveal has already staged a board and locked/revealed the Spinner Set,
                    # a late-finishing background search thread must NEVER overwrite the staged board or parameters!
                    if (
                        getattr(target_room, '_spinner_params_locked', False)
                        or getattr(target_room, 'spinner_params_revealed', False)
                        or getattr(target_room, '_fallback_at_50s_done', False)
                    ) and getattr(target_room, 'next_round_board', None):
                        print(f"[RoomManager] Background board search finished after 0:50 fallback / 0:45 reveal lock for {room_id}. Preserving locked staged board & params.")
                        return

                    if launched_generated_at is not None:
                        is_locked_or_revealed = (
                            getattr(target_room, '_spinner_params_locked', False)
                            or getattr(target_room, 'spinner_params_revealed', False)
                        )
                        curr_generated_at = None
                        if target_room and getattr(target_room, 'next_spinner_params', None):
                            curr_generated_at = target_room.next_spinner_params.get('generated_at')
                        time_drift = abs((curr_generated_at or 0) - launched_generated_at)
                        if time_drift > 1.0:
                            curr_params = getattr(target_room, 'next_spinner_params', {}) or {}
                            config_changed = False
                            for key in ('dictionary', 'difficulty', 'board_format', 'min_word_length', 'word_count_range', 'use_added_words'):
                                if str(curr_params.get(key)) != str(params.get(key)):
                                    config_changed = True
                                    break
                            if config_changed and is_locked_or_revealed:
                                print(f"[RoomManager] Stale board search discarded for {room_id} because parameters were already revealed/locked to player (launched: {launched_generated_at}, current: {curr_generated_at})")
                                return
                            elif config_changed:
                                print(f"[RoomManager] Parameters re-spun before reveal for {room_id}; accepting freshly generated authentic board (launched: {launched_generated_at}, current: {curr_generated_at})")
                            else:
                                print(f"[RoomManager] Parameter timestamps drifted by {time_drift:.4f}s but configuration is identical. Keeping board.")

                    # LOCAL PREPARATION: Calculate all derived metrics without touching target_room yet
                    # Update word to the ACTUAL embedded word if different (MANDATORY consistency)
                    if final_bonus_word and len(final_bonus_word) == bw_l and final_bonus_word in all_words and not final_bonus_word.upper().endswith(('ING', 'INGS')) and 'ING' not in final_bonus_word.upper():
                        bonus_word = final_bonus_word
                    else:
                        exact_bw_cands = [w for w in all_words if len(w) == bw_l and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
                        if exact_bw_cands:
                            bonus_word = random.choice(exact_bw_cands).upper()
                        elif final_bonus_word and len(final_bonus_word) >= 6 and final_bonus_word in all_words and not final_bonus_word.upper().endswith(('ING', 'INGS')) and 'ING' not in final_bonus_word.upper():
                            bonus_word = final_bonus_word
                        else:
                            scorable = [w for w in all_words if 6 <= len(w) <= 10 and not w.upper().endswith(('ING', 'INGS')) and 'ING' not in w.upper()]
                            if scorable:
                                scorable.sort(key=lambda w: abs(len(w) - bw_l))
                                bonus_word = scorable[0]
                            else:
                                bonus_word = self._get_bonus_word(length=bw_l, dictionary=search_dict)
                    
                    if bonus_word:
                        params['bonus_word_length'] = len(bonus_word)

                    # Check if generator relaxed min_word_length
                    if all_words:
                        actual_shortest = min(len(w) for w in all_words)
                        if actual_shortest < search_min:
                            print(f"[RoomManager] Word generator relaxed min_word_length from {search_min} to {actual_shortest} in background. Updating room param.")
                            params['min_word_length'] = actual_shortest

                    # Update metadata for the pending round
                    planned_wc = (
                        (target_room.frozen_revealed_params.get('word_count_range') if getattr(target_room, 'frozen_revealed_params', None) else None)
                        or (target_room.next_spinner_params.get('word_count_range') if getattr(target_room, 'next_spinner_params', None) else None)
                        or '100-200'
                    )
                    max_cap = None
                    if planned_wc == '50-100': max_cap = 99
                    elif planned_wc == '100-200': max_cap = 199
                    elif planned_wc == '200-300': max_cap = 299
                    elif planned_wc == '300-400': max_cap = 399
                    elif planned_wc == '400-500': max_cap = 499

                    if False and max_cap and len(all_words) > max_cap:  # never truncate valid words
                        _wl = list(all_words)
                        total_raw = len(_wl)
                        by_len = {}
                        for w in _wl: by_len.setdefault(len(w), []).append(w)
                        selected = set()
                        if bonus_word and bonus_word in all_words: selected.add(bonus_word)
                        for l in sorted(by_len.keys()):
                            bw2 = [w for w in by_len[l] if w not in selected]
                            if not bw2: continue
                            pc = max(1, min(len(bw2), int(round((len(by_len[l]) / float(total_raw)) * max_cap))))
                            selected.update(sorted(bw2, key=lambda w: (len(w), w), reverse=True)[:pc])
                        if len(selected) < max_cap:
                            lft = sorted([w for w in _wl if w not in selected], key=lambda w: (len(w), w), reverse=True)
                            selected.update(lft[:(max_cap - len(selected))])
                        elif len(selected) > max_cap:
                            nb = [w for w in selected if w != bonus_word]
                            selected = set(nb[:max_cap])
                            if bonus_word and bonus_word in all_words: selected.add(bonus_word)
                        all_words = list(selected)
                        all_words_dict = {w: all_words_dict[w] for w in all_words if w in all_words_dict}

                    _sb_min_len = (target_room.next_spinner_params.get('min_word_length', 3)
                                   if getattr(target_room, 'next_spinner_params', None) else 3)
                    if all_words:
                        _actual_min_board = min(len(w) for w in all_words)
                        if _actual_min_board > _sb_min_len:
                            print(f"[RoomManager] Min-length correction in start_board_search: "
                                  f"planned={_sb_min_len}, actual={_actual_min_board}. Updating params.")
                            _sb_min_len = _actual_min_board

                    _sb_filtered_cnt = sum(1 for w in (all_words or []) if len(w) >= _sb_min_len)
                    if room.time_limit >= 7200:
                        achieved_wc = '300-400'
                    else:
                        achieved_wc = self._get_factchecked_wc_range(_sb_filtered_cnt, use_added_words=use_aw_flag, dictionary=search_dict)

                    f_low = str(updated_format).lower()
                    final_bonus_cell = bonus_cell if ('bonus letter' in f_low or f_low.strip() == 'bonus') else None

                    # INITIALIZATION: Calculate detailed scores with bonus letter and bonus word points
                    scored_dict = {}
                    for word in (all_words or []):
                        path_v = _get_word_path(all_words_dict, word)
                        scored_dict[word] = calculate_word_score(
                            word,
                            bonus_word=bonus_word,
                            board_format=updated_format,
                            bonus_cell=final_bonus_cell,
                            board=board,
                            path=path_v,
                            return_details=True
                        )
                    fast_total_points = sum((pts['total'] if isinstance(pts, dict) else pts) for pts in scored_dict.values())
                    next_counts = {
                        '_round': target_room.current_round + 1,
                        **{str(l): sum(1 for w in (all_words or []) if len(w) == l) for l in range(1, 31)}
                    }

                    b_dims = target_room.board_dimensions.split('x')
                    d_val = int(b_dims[0]) if len(b_dims) == 3 else 1
                    rows = int(b_dims[1] if len(b_dims) == 3 else b_dims[0])
                    cols = int(b_dims[2] if len(b_dims) == 3 else b_dims[1])
                    achieved_diff = self.board_generator.get_difficulty_label(u_ratio, rows, cols, search_dict, depth=d_val, board=board, target_difficulty=target_room.next_spinner_params.get('difficulty') if getattr(target_room, 'next_spinner_params', None) else None, min_word_length=_sb_min_len)
                    if getattr(target_room, 'is_solo', False) and getattr(target_room, 'initial_solo_params', None):
                        _solo_d_req = target_room.initial_solo_params.get('difficulty')
                        if _solo_d_req and str(_solo_d_req).lower() != 'random':
                            _solo_d_base = SpinnerSet.strip_difficulty_pct(_solo_d_req)
                            if _solo_d_base == 'Easy' and u_ratio > 0.15:
                                self.board_generator._sanitize_uniqueness(board, search_dict, max_ratio=0.15, depth=d_val, bonus_cell=final_bonus_cell, min_word_length=_sb_min_len, use_added_words=use_aw_flag)
                                _re_s = self.board_generator._solve_board(board, search_dict, (0, 99999), _sb_min_len, max_depth=20, store_paths=True, timeout=5.0, bonus_cell=final_bonus_cell, use_added_words=use_aw_flag)
                                all_words_dict = {w: p for w, p in _re_s.items() if len(w) >= _sb_min_len}
                                all_words = list(all_words_dict.keys())
                                u_ratio = self.board_generator._calculate_uniqueness_ratio(board, search_dict, d_val, final_bonus_cell, _sb_min_len, use_added_words=use_aw_flag)
                            achieved_diff = _solo_d_base
                        _solo_fmt_req = target_room.initial_solo_params.get('board_format')
                        if _solo_fmt_req and str(_solo_fmt_req).lower() != 'random':
                            updated_format = _solo_fmt_req
                    achieved_diff = SpinnerSet.format_difficulty_with_pct(achieved_diff, u_ratio)

                    # CSW and AW word tagging
                    display_min = _sb_min_len
                    filtered_all = [w for w in (all_words or []) if len(w) >= display_min]
                    if hasattr(word_validator, 'word_validator'):
                        word_validator.word_validator.ensure_csw_loaded()
                        csw_words = [w for w in filtered_all if word_validator.word_validator.is_csw_only(w)]
                    else:
                        csw_words = []
                    aw_words = [w for w in filtered_all if word_validator.word_validator.is_added_word(w)] if use_aw_flag else []

                    # ATOMIC STAGING PROMOTION: Set board and ALL metadata together under state lock
                    with target_room._state_lock:
                        if target_room.current_round > search_round:
                            print(f"[RoomManager] Stale board search discarded inside lock for {room_id} (search_round: {search_round}, current_round: {target_room.current_round})")
                            return
                        if (
                            getattr(target_room, '_spinner_params_locked', False)
                            or getattr(target_room, 'spinner_params_revealed', False)
                            or getattr(target_room, '_fallback_at_50s_done', False)
                        ) and getattr(target_room, 'next_round_board', None):
                            print(f"[RoomManager] Late board search discarded inside lock for {room_id}: 0:50 fallback or 0:45 reveal already locked board & params.")
                            return

                        board, all_words, all_words_dict, final_bonus_cell, bonus_word, updated_format = self.sanitize_board_for_room_difficulty(
                            target_room, board, all_words, all_words_dict, final_bonus_cell, bonus_word,
                            updated_format, params.get('difficulty'), dictionary=params.get('dictionary'), min_len=params.get('min_word_length')
                        )
                        target_room.next_round_board = board
                        target_room.next_round_words = list(all_words)
                        target_room.next_round_word_paths = all_words_dict
                        target_room.next_round_bonus_cell = final_bonus_cell
                        target_room.next_round_bonus = bonus_word
                        target_room.next_round_format = updated_format
                        target_room.next_round_uniqueness = u_ratio
                        target_room.next_round_word_scores = scored_dict
                        target_room.next_round_total_points = fast_total_points
                        target_room.next_round_total_words_count = _sb_filtered_cnt
                        target_room.next_round_counts_by_len = next_counts
                        target_room.next_round_difficulty = achieved_diff
                        target_room.next_round_csw_only_words = csw_words
                        target_room.next_round_added_words = aw_words

                        staged_sp = dict(params)
                        staged_sp['board_format'] = updated_format
                        staged_sp['word_count_range'] = achieved_wc
                        staged_sp['min_word_length'] = _sb_min_len
                        if bonus_word:
                            staged_sp['bonus_word_length'] = len(bonus_word)
                        staged_sp['uniqueness'] = u_ratio
                        staged_sp['difficulty'] = achieved_diff
                        target_room.next_round_spinner_params = staged_sp

                        if getattr(target_room, 'next_spinner_params', None):
                            target_room.next_spinner_params.update(staged_sp)
                            if not getattr(target_room, '_spinner_params_locked', False) and target_room.state != 'active':
                                if getattr(target_room, 'frozen_revealed_params', None) and isinstance(target_room.frozen_revealed_params, dict):
                                    target_room.frozen_revealed_params.update(staged_sp)
                                target_room.spinner_params.update(staged_sp)

                        target_room.spinner_params_generated = True

                        if target_room.board == board:
                            target_room.solved_words_with_scores = scored_dict
                            target_room.recalculate_total_points()
                            print(f"[RoomManager] Board {room_id} fast-synced to ACTIVE round ({target_room.total_points_count} pts)")
                        else:
                            print(f"[RoomManager] Board {room_id} signal-ready (Fast metrics applied: {target_room.next_round_total_points} pts)")

                    # BACKGROUND REFINEMENT: Detailed scoring (Scoring bonuses, paths, etc.)
                    from scoring import calculate_word_score
                    def refine_scores():
                        try:
                            if launched_generated_at is not None:
                                curr_generated_at = None
                                if target_room and getattr(target_room, 'next_spinner_params', None):
                                    curr_generated_at = target_room.next_spinner_params.get('generated_at')
                                time_drift = abs((curr_generated_at or 0) - launched_generated_at)
                                if time_drift > 1.0:
                                    curr_params = getattr(target_room, 'next_spinner_params', {}) or {}
                                    config_changed = False
                                    for key in ('dictionary', 'difficulty', 'board_format', 'min_word_length', 'word_count_range', 'use_added_words'):
                                        if str(curr_params.get(key)) != str(params.get(key)):
                                            config_changed = True
                                            break
                                    if config_changed:
                                        return

                            if target_room.current_round > search_round:
                                return
                            refined = {}
                            for word in (all_words or []):
                                path_v = _get_word_path(all_words_dict, word)
                                refined[word] = calculate_word_score(
                                    word, bonus_word, path=path_v,
                                    board_format=updated_format, bonus_cell=final_bonus_cell,
                                    board=board, return_details=True, strict_path=True
                                )
                            with target_room._state_lock:
                                if target_room.next_round_board == board:
                                    target_room.next_round_word_scores = refined
                                    target_room.next_round_total_points = sum(
                                        (pts.get('total', 0) if isinstance(pts, dict) else pts) 
                                        for pts in refined.values()
                                    )
                                if target_room.board == board:
                                    target_room.solved_words_with_scores = refined
                                    target_room.recalculate_total_points()
                            print(f"[RoomManager] Board {room_id} scoring refinement complete. Next Round Total: {getattr(target_room, 'next_round_total_points', 0)}")
                        except Exception as e:
                            print(f"[RoomManager] Refinement error for {room_id}: {e}")
                    
                    threading.Thread(target=refine_scores, daemon=True).start()
                        
                    # 3. Density Initialization
                    try:
                        target_room.initialize_density(board, all_words_dict, updated_format, is_staging=True)
                    except Exception as e:
                        print(f"[Density-Diag] Failed to initialize staging: {e}")

                    if aw_words:
                        print(f"[RoomManager {target_room.room_id}] Pre-gen Round {target_room.current_round + 1} generated {len(aw_words)} custom added words: {aw_words}")
                    
                    print(f"[RoomManager] Background pre-gen complete for {room_id}: {achieved_diff} | {updated_format} (Count: {len(all_words)})")
                    print(f"[RoomManager] Board found and params revealed! Words: {len(filtered_all)}")
                except Exception as e:
                    import traceback
                    print(f"[RoomManager] ERROR in background search for {room_id}: {e}")
                    traceback.print_exc()
                    room.spinner_params_generated = True
                finally:
                    # ABSOLUTE SAFETY: The loading flag MUST be cleared so the transition can proceed.
                    with room._state_lock:
                         room.board_search_loading = False
                         # Mark search as 'started' (finished) only if we actually found a board or exhausted retries.
                         # This prevents pre_generate_next_round from spamming if we are failing.
                         room.board_search_started = True 
                         # If we don't have a board yet, start_next_round will see 'loading=False' and trigger Emergency.
            
            thread = threading.Thread(target=generate_in_background, daemon=True)
            thread.start()
            return True
        finally:
            pass # Flag is managed by the background thread lifecycle
    
    
    def pre_generate_next_round(self, room_id):
        """Utility to initiate next-round board search during an active round."""
        room = self.get_room(room_id)
        if not room: return
        
        # ISSUE 6: Do NOT pre-generate until the first board has been delivered.
        # Without this guard, multiple calls stack up before round 1 board is even shown,
        # causing the board to change 7+ times in the first 45 seconds.
        if not getattr(room, '_initial_board_delivered', False):
            print(f"[RoomManager] Skipping pre-generation: first board not yet delivered for {room_id}")
            return
        
        # ALWAYS pre-generate next round board while current round is active.
        # This ensures the 0:00 transition is instant, especially for large 6x8 boards.
        # (Skip only if already loading or started)
        if getattr(room, 'board_search_loading', False) or getattr(room, 'board_search_started', False):
            return
        print(f"[RoomManager] PRE-GENERATING next board for room {room_id} (Scheduled after delay)")
        
        # Atomic Guard: If we are already searching or have a board ready, DO NOT RE-ROLL.
        if getattr(room, 'board_search_started', False) or getattr(room, 'next_round_board', None):
             print(f"[RoomManager] Lead-time: Search already in progress or board ready for {room_id}")
             return
             
        def delayed_pre_generate():
            # Wait 0.1 seconds to allow the transition-phase network requests to finish first.
            # (Reduced from 0.5s — transition is already complete when this thread fires.)
            time.sleep(0.1)
            
            # Recheck conditions in case the room state changed during the sleep.
            # Allow pre-generation from both 'active' AND 'intermission' states so that
            # if a client polls during intermission and there's no board staged, we can still
            # kick off generation with the full intermission window (60s) available.
            room_check = self.get_room(room_id)
            if not room_check or room_check.state not in ('active', 'intermission'):
                return
                
            if getattr(room_check, 'board_search_loading', False) or getattr(room_check, 'board_search_started', False) or getattr(room_check, 'next_round_board', None):
                return
                
            print(f"[RoomManager] Executing pre-generation for {room_id} (state={room_check.state})")
            self.generate_spinner_params(room_id, reveal=False)
            self.start_board_search(room_id)

        threading.Thread(target=delayed_pre_generate, daemon=True).start()

    def start_next_round(self, room_id):
        """Start next round with pre-generated board (called at 0s remaining)"""
        import time
        from spinner_set import SpinnerSet
        from scoring import calculate_word_score
        room = self.get_room(room_id)
        if not room:
            print(f"[RoomManager] ERROR: Room {room_id} not found")
            return False
            
        # SAFETY: If room is in 'waiting' but somehow triggered, allow transition ONLY if
        # there is no active board already in play (i.e., it's a genuine lobby start).
        # This prevents the heartbeat's empty-room pause from wiping a live board when
        # a human player is present (transient 0-player moments, join latency, etc.).
        if room.state == 'active':
            # Room is already live — nothing to do here
            print(f"[RoomManager] Aborting start_next_round for {room_id}: room is already active.")
            return False
        if room.state == 'waiting':
            board = getattr(room, 'board', None)
            has_active_board = isinstance(board, list) and len(board) > 0
            if has_active_board:
                # Board already exists — abort to prevent board re-roll.
                print(f"[RoomManager] Aborting start_next_round for {room_id}: state='waiting' but board already active (heartbeat false-positive).")
                return False
            # Allow direct transition: waiting -> active
            pass
            
        # 0. ATOMIC GUARD: Ensure only ONE thread/request triggers the round start transition
        # This prevents stacking up identical wait loops on a single slow-loading board.
        with room._state_lock:
            # ONLY ALLOW transition if room is currently in 'intermission' or 'waiting' (Lobby Start)
            # This prevents duplicate transitions or state-corrupting re-runs if watchdog triggers late.
            if room.state not in ['intermission', 'waiting']:
                 # print(f"[RoomManager] Skipping transition for {room_id}: State is {room.state}")
                 return False
                 
            if getattr(room, 'starting_round', False):
                curr_init = getattr(room, '_round_start_init_time', 0)
                timeout = 3.0
                if curr_init > 0 and (time.time() - curr_init > timeout):
                     print(f"[RoomManager] Stale start detected (>{timeout}s) for {room_id}, resetting guard.")
                     room.starting_round = False
                else:
                     print(f"[RoomManager] Already starting a round for {room_id}. Skipping duplicate start.")
                     return False
            
            room.starting_round = True
            room._round_start_init_time = time.time()
             
        print(f"[RoomManager] start_next_round processing for room {room_id}")
        room._board_promoted_this_transition = False
        
        try:
            # Calculate min_accept for safety check
            m_len = room.spinner_params.get('min_word_length', 3)
            use_aw_flag = room.spinner_params.get('use_added_words', False) or ('+ AW' in str(room.spinner_params.get('dictionary', 'NWL')).upper())
            min_accept = 50
            target_range = room.spinner_params.get('word_count_range')
            if target_range:
                try:
                    if str(target_range).startswith('500'):
                        min_accept = 500  # 500+ means AT LEAST 500
                    else:
                        min_accept = int(str(target_range).split('-')[0])
                except:
                    if '500' in str(target_range): min_accept = 500
                    elif '400' in str(target_range): min_accept = 400
                    elif '300' in str(target_range): min_accept = 300
                    elif '200' in str(target_range): min_accept = 200
                    elif '100' in str(target_range): min_accept = 100
            min_accept = max(300, min_accept) if use_aw_flag else max(30, min_accept)

            # INSTANT 0:00 TRANSITION: Always deliver staged board or popped cached board in <1ms!
            if not getattr(room, 'next_round_board', None):
                if room.game_type == 'subanagrams':
                    from subanagrams_generator import generate_subanagrams_board_and_words
                    p = getattr(room, 'next_spinner_params', None) or SpinnerSet.generate_subanagrams_params()
                    res = generate_subanagrams_board_and_words(p)
                    if res:
                        _fb, _fw, _fbw, _fparams = res
                        room.next_round_board = _fb
                        room.next_round_words = _fw
                        room.next_round_word_paths = {w: [] for w in _fw}
                        room.next_round_bonus = _fbw
                        seq_len_actual = len(_fb[0]) if (_fb and len(_fb) > 0 and isinstance(_fb[0], list)) else len(_fb)
                        has_full_len = any(len(w) == seq_len_actual for w in _fw)
                        final_fmt = 'Word Guaranteed' if (has_full_len and _fparams.get('mode') == 'word') else 'Totally Random'
                        _fparams['board_format'] = final_fmt
                        _fparams['mode'] = 'word' if final_fmt == 'Word Guaranteed' else 'random'
                        room.next_round_format = final_fmt
                        room.next_round_bonus_cell = None
                        room.next_round_uniqueness = 0.0
                        room.next_round_spinner_params = _fparams
                        room.next_spinner_params = _fparams
                        room.next_round_min_length = _fparams.get('min_word_length', 3)
                        room.spinner_params = dict(_fparams)
                else:
                    from board_generator import pop_compatible_cached_board, pop_any_cached_board
                    sp = room.spinner_params or {}
                    dict_val = sp.get('dictionary', 'NWL')
                    fmt_val = sp.get('board_format', 'Normal')
                    use_aw_val = sp.get('use_added_words', False) or '+ AW' in str(dict_val).upper() or '+AW' in str(dict_val).upper()
                    bonus_word_len = sp.get('bonus_word_length')
                    sp_diff = sp.get('difficulty')

                    if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                        isp = room.initial_solo_params
                        _solo_d = isp.get('difficulty')
                        if _solo_d and str(_solo_d).lower() != 'random':
                            sp_diff = SpinnerSet.strip_difficulty_pct(_solo_d)
                        _solo_f = isp.get('board_format')
                        if _solo_f and str(_solo_f).lower() != 'random':
                            fmt_val = _solo_f
                    
                    candidate = pop_compatible_cached_board(
                        room.board_dimensions, dict_val, fmt_val, m_len, use_aw_val,
                        bonus_word_len=bonus_word_len, word_count_range=target_range, difficulty=sp_diff
                    )
                    if not candidate:
                        candidate = pop_any_cached_board(
                            room.board_dimensions, min_word_length=m_len, target_format=fmt_val,
                            dictionary=dict_val, use_added_words=use_aw_val, word_count_range=target_range, difficulty=sp_diff
                        )
                    # Only fall back to unconstrained pop if difficulty and format were random/flexible
                    if not candidate and (not sp_diff or str(sp_diff).lower() == 'random') and (not fmt_val or str(fmt_val).lower() == 'normal'):
                        candidate = pop_any_cached_board(room.board_dimensions, min_word_length=m_len)
                    if not candidate and (not sp_diff or str(sp_diff).lower() == 'random') and (not fmt_val or str(fmt_val).lower() == 'normal'):
                        candidate = pop_any_cached_board(room.board_dimensions)
                    if not candidate:
                        print(f"[start_next_round] Cache miss for {room_id}. Generating instant emergency fallback board for format '{fmt_val}', diff '{sp_diff}'.")
                        candidate = get_emergency_fallback_board(
                            room.board_dimensions, fmt_val, room.time_limit,
                            dictionary=dict_val, use_added_words=use_aw_val, target_range=target_range, min_word_length=m_len,
                            difficulty=sp_diff,
                            bonus_word_len=bonus_word_len
                        )
                    
                    if candidate:
                        if len(candidate) >= 9:
                            _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _, _fparams = candidate
                        else:
                            _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _fparams = candidate
                        
                        # Strict format and difficulty compliance: candidate MUST match fmt_val and sp_diff
                        is_fmt_mismatch = bool(fmt_val and str(fmt_val).lower() != 'random' and str(_ff).lower() != str(fmt_val).lower())
                        is_diff_mismatch = False
                        if sp_diff and str(sp_diff).lower() != 'random':
                            _b_dims_ck = str(room.board_dimensions).split('x')
                            _d_ck = int(_b_dims_ck[0]) if len(_b_dims_ck) == 3 else 1
                            _r_ck = int(_b_dims_ck[1] if len(_b_dims_ck) == 3 else _b_dims_ck[0])
                            _c_ck = int(_b_dims_ck[2] if len(_b_dims_ck) == 3 else _b_dims_ck[1])
                            _cand_d_lbl = self.board_generator.get_difficulty_label(_fr, _r_ck, _c_ck, dict_val, depth=_d_ck, min_word_length=m_len)
                            if _cand_d_lbl != SpinnerSet.strip_difficulty_pct(sp_diff):
                                is_diff_mismatch = True

                        if is_fmt_mismatch or is_diff_mismatch:
                            print(f"[start_next_round] Compliance mismatch (candidate fmt={_ff} vs spun={fmt_val}, diff_mismatch={is_diff_mismatch}). Regenerating exact compliant board...")
                            candidate = get_emergency_fallback_board(
                                room.board_dimensions, fmt_val, room.time_limit,
                                dictionary=dict_val, use_added_words=use_aw_val, target_range=target_range, min_word_length=m_len,
                                difficulty=sp_diff,
                                bonus_word_len=bonus_word_len
                            )
                            if candidate:
                                if len(candidate) >= 9:
                                    _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _, _fparams = candidate
                                else:
                                    _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _fparams = candidate

                        _fw_filtered = [w for w in _fw if len(w) >= m_len]
                        if not _fw_filtered: _fw_filtered = _fw
                        if 'CSW' not in str(dict_val).upper() and hasattr(word_validator, 'word_validator'):
                            word_validator.word_validator.ensure_csw_loaded()
                            _fw_filtered = [w for w in _fw_filtered if not word_validator.word_validator.is_csw_only(w)]
                        _fp_filtered = {w: p for w, p in (_fp or {}).items() if w in _fw_filtered}
                        
                        # Strict word count compliance: candidate MUST match target_range if provided
                        if target_range and not is_board_count_valid(len(_fw_filtered), target_range):
                            print(f"[start_next_round] Word count mismatch (candidate {len(_fw_filtered)} not in {target_range}). Regenerating...")
                            candidate = get_emergency_fallback_board(
                                room.board_dimensions, fmt_val, room.time_limit,
                                dictionary=dict_val, use_added_words=use_aw_val, target_range=target_range, min_word_length=m_len,
                                difficulty=sp_diff,
                                bonus_word_len=bonus_word_len
                            )
                            if candidate:
                                if len(candidate) >= 9:
                                    _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _, _fparams = candidate
                                else:
                                    _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _fparams = candidate
                                _fw_filtered = [w for w in _fw if len(w) >= m_len]
                                if not _fw_filtered: _fw_filtered = _fw
                                if 'CSW' not in str(dict_val).upper() and hasattr(word_validator, 'word_validator'):
                                    word_validator.word_validator.ensure_csw_loaded()
                                    _fw_filtered = [w for w in _fw_filtered if not word_validator.word_validator.is_csw_only(w)]
                                _fp_filtered = {w: p for w, p in (_fp or {}).items() if w in _fw_filtered}
                        
                        # SAFEGUARD: If Checkerboard format, enforce strict C/V alternation
                        _rep, _rw, _rp = sync_board_checkerboard_and_resolve(self.board_generator, _fb, fmt_val, m_len, dict_val, use_aw_val, _fc)
                        if _rep and _rw:
                            _fw_filtered, _fp_filtered = _rw, _rp

                        print(f"[start_next_round] Instant 1-ms cache pop for {room_id}: {len(_fw_filtered)} words")
                        room.next_round_board = _fb
                        room.next_round_words = _fw_filtered
                        room.next_round_bonus_cell = _fc
                        room.next_round_bonus = _fbw or ''
                        room.next_round_word_paths = _fp_filtered
                        if hasattr(word_validator, 'word_validator'):
                            word_validator.word_validator.ensure_csw_loaded()
                            room.next_round_csw_only_words = [w for w in _fw_filtered if word_validator.word_validator.is_csw_only(w)]
                            room.next_round_added_words = [w for w in _fw_filtered if word_validator.word_validator.is_added_word(w)]
                        else:
                            room.next_round_csw_only_words = []
                            room.next_round_added_words = []
                        bw_l_val = int(room.spinner_params.get('bonus_word_length', 8) or 8) if isinstance(room.spinner_params, dict) else 8
                        if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                            _s_bw_v = room.initial_solo_params.get('bonus_word_length', 'random')
                            if _s_bw_v and str(_s_bw_v).lower() not in ['random', '0', 'none', 'null', 'nan']:
                                try:
                                    bw_l_val = int(_s_bw_v)
                                except Exception:
                                    pass
                        dict_val = room.spinner_params.get('dictionary', 'NWL') if isinstance(room.spinner_params, dict) else 'NWL'
                        _is_chk_p = 'checkerboard' in str(room.spinner_params.get('board_format', '')).lower()
                        room.next_round_bonus = pick_distinct_bonus_word(
                            room, _fbw, _fw_filtered, bw_l_val,
                            dictionary=dict_val, is_checkerboard=_is_chk_p,
                            room_manager=self, extra_exclude=[getattr(room, 'bonus_word', None), getattr(room, 'previous_bonus_word', None)]
                        )
                        room.next_round_format = _ff
                        room.next_round_uniqueness = _fr
                        if _fparams:
                            wc_label = target_range or _fparams.get('word_count_range', '100-200')
                            _fparams['word_count_range'] = wc_label
                            room.next_round_spinner_params = _fparams
                            
                            dict_val = _fparams.get('dictionary', 'NWL')
                            use_aw_val = _fparams.get('use_added_words', False)
                            if use_aw_val and '+ AW' not in str(dict_val) and '+AW' not in str(dict_val):
                                dict_val = f"{dict_val} + AW"
                            
                            # Do NOT overwrite room.spinner_params with cached board defaults!
                            # The spun parameters generated by SpinnerSet are authoritative.
                            pass
                            # Trigger background refill for this popped key
                            try:
                                from board_generator import BoardGenerator, serialize_param_key, refill_board_cache_bg
                                bg = BoardGenerator()
                                refill_key = serialize_param_key(
                                    room.board_dimensions,
                                    _fbw or '',
                                    wc_label,
                                    _fparams.get('dictionary', 'NWL'),
                                    _ff,
                                    _fparams.get('min_word_length', 3),
                                    _fparams.get('difficulty', 'Medium'),
                                    use_added_words=_fparams.get('use_added_words', False)
                                )
                                refill_board_cache_bg(bg, refill_key, target_count=3)
                            except Exception as refill_err:
                                print(f"[start_next_round] Error triggering refill from last-chance pop: {refill_err}")

            # --- START TRANSITION ---
            # ATOMIC REFERENCE CAPTURE: Since we replace the board object, a reference is safe and instant.
            ghost_prev_board = room.board 
            ghost_round_start_time = room.round_start_time
            ghost_board_format = room.current_board_format
            
            ghost_source_words = list(room.complete_words) if (getattr(room, 'complete_words', None) and len(room.complete_words) > 0) else list(room.all_words)
            ghost_bonus = (room.bonus_word.upper() if room.bonus_word else None)
            ghost_min_len = getattr(room, 'current_min_length', 3)
            ghost_round_num = room.current_round # CAPTURE NOW before it increments
            ghost_all_words_paths = dict(getattr(room, 'all_words_paths', {}))
            
            # SNAPSHOT PLAYERS: We MUST deep-copy the data because player objects are reset in the main thread
            # while the history saver runs in the background.
            ghost_player_snapshots = []
            
            # Combine current players and past players who played in the round to capture those who left during intermission
            all_candidate_players = list(room.players)
            existing_uids = {p.user_id for p in room.players}
            for p in room.past_players.values():
                if p.user_id not in existing_uids:
                    all_candidate_players.append(p)
                    
            try:
                room.update_live_pe()
            except Exception as _pe_err:
                print(f"[RoomManager] Error updating live PE before snapshot: {_pe_err}")

            for p in all_candidate_players:
                if (p.is_registered or p.is_guest) and (p.score > 0 or p.submitted_words or p.invalid_words):
                    ghost_player_snapshots.append({
                        'user_id': p.user_id,
                        'username': p.username,
                        'score': p.score,
                        'submitted_words': [dict(w) for w in p.submitted_words],
                        'invalid_words': list(p.invalid_words),
                        'rating': getattr(p, 'rating', 1200),
                        'performance_efficiency': getattr(p, 'performance_efficiency', 0),
                        'joined_mid_round': getattr(p, 'joined_mid_round', False)
                    })
            
            # USER REQUEST: Word Tally. Capture unique words found by each player in this round.
            # Strictly filter for valid words with positive score only (exclude invalid guesses/penalties)
            ghost_player_words = {
                p['username']: [
                    w['word'] for w in p['submitted_words']
                    if (w.get('points', 0) if isinstance(w, dict) else 0) > 0
                ] for p in ghost_player_snapshots
            }
            
            # DIAGNOSTIC: Log the capture
            total_captured = sum(len(ws) for ws in ghost_player_words.values())
            print(f"[RoomManager] Snapshot captured for {room.room_id}: {total_captured} words, {len(ghost_player_snapshots)} players, Round {ghost_round_num}")
            for u, ws in ghost_player_words.items():
                # ws is a list of strings (the words)
                submitted_strings = [w.upper() for w in ws]
            
            # 2. STATE TRANSITION LOCK: Perform the atomic board swap
            with room._state_lock:
                if room.state == 'active':
                    print(f"[RoomManager] Aborting start_next_round promotion for {room_id}: room already transitioned to active round {room.current_round}!")
                    room.starting_round = False
                    return False
                room._did_6x8_fallback_rescue = False
                room._did_050_fallback_rescue = False
                room._fallback_at_50s_done = False
                # Reset players active stats and roster for the next round BEFORE database save
                # FCFS: Clear shared found lists for the upcoming round
                room.fcfs_found_words = []
                room._fcfs_found_words_set = set()
                
                next_round_val = room.current_round + 1
                if room.time_limit >= 7200:
                    # Clear active stats and snapshot for all active players
                    for p in room.players:
                        if len(p.submitted_words) > 0:
                            p.previous_round_score = p.score
                            p.previous_submitted_words = [dict(w) for w in p.submitted_words]
                        p.submitted_words = []
                        p.invalid_words = []
                        p.score = 0
                        p.found_bonus_word = False
                        p.joined_mid_round = False
                        p.has_exceptional_round = False
                        p.trophy_rounds_left = 0
                        p.has_abandoned = False
                        p._last_round_seen = next_round_val
                        p.rating_change = 0
                    
                    # Clear active stats and snapshot for all archive players in past_players
                    for p in room.past_players.values():
                        if len(p.submitted_words) > 0:
                            p.previous_round_score = p.score
                            p.previous_submitted_words = [dict(w) for w in p.submitted_words]
                        p.submitted_words = []
                        p.invalid_words = []
                        p.score = 0
                        p.found_bonus_word = False
                        p.joined_mid_round = False
                        p.has_exceptional_round = False
                        p.trophy_rounds_left = 0
                        p.has_abandoned = False
                        p._last_round_seen = next_round_val
                        p.rating_change = 0
                    
                    room.players = []
                    room.spectators = []
                else:
                    room.round_starting_player_ids = {str(p.user_id) for p in room.players}
                    for p in room.players:
                        p.submitted_words, p.invalid_words, p.score = [], [], 0
                        p.found_bonus_word, p.has_abandoned = False, False
                        p.joined_mid_round = False
                        p._last_round_seen = next_round_val
                        p.rating_change = 0
                    for p in room.past_players.values():
                        p.submitted_words, p.invalid_words, p.score = [], [], 0
                        p.found_bonus_word, p.has_abandoned = False, False
                        p.joined_mid_round = False
                        p._last_round_seen = next_round_val
                        p.rating_change = 0

                # CLEAR BOARD & WORDS IMMEDIATELY if we are about to generate a new one
                # This handles the fallback Case or any state where we want to avoid stale data
                # ONLY DO THIS FOR 500+ (IO) rounds though, to avoid flickering for normal ones
                wc_range = room.spinner_params.get('word_count_range', (0, 0))
                wc_tuple = room._get_wc_tuple(wc_range)
                is_500plus = wc_tuple[0] >= 500
                
                if (not room.next_round_board or not room.next_round_words) and is_500plus:
                    dims = room.board_dimensions.split('x')
                    if len(dims) == 3:
                         # 3D Cube
                         f_num, r_num, c_num = map(int, dims)
                         room.board = [[['' for _ in range(c_num)] for _ in range(r_num)] for _ in range(6)]
                    else:
                        rows_num, cols_num = map(int, dims)
                        room.board = [['' for _ in range(cols_num)] for _ in range(rows_num)]
                    room.all_words = []
                    room.complete_words = []

                # --- 1. PARAMETER PROMOTION ---
                # We strictly favor revealed parameters to ensure the promise made at 45s (reveal) is kept at 0s (active)
                if not getattr(room, 'spinner_params_revealed', False) and hasattr(room, 'next_spinner_params') and room.next_spinner_params:
                    # Fallback for hidden params
                    room.spinner_params = room.next_spinner_params
                    room.spinner_params_revealed = True
                
                # USER REQUEST: Ensure UI range matches board EXACTLY by using the params used for generation
                # CRITICAL: When params have been revealed to players, the revealed next_spinner_params is authoritative.
                # next_round_spinner_params (from board search) may have stale params from a previous search cycle;
                # it should only win if reveal never happened (no promise was made to the player yet).
                if (getattr(room, 'was_revealed_this_intermission', False) or getattr(room, 'spinner_params_revealed', False)) and (getattr(room, 'frozen_revealed_params', None) or room.spinner_params):
                    # Revealed params shown on the Spinner Set at 0:45 timer are ironclad & authoritative!
                    active_params = dict(getattr(room, 'frozen_revealed_params', None) or room.spinner_params)
                elif getattr(room, 'next_spinner_params', None):
                    active_params = dict(room.next_spinner_params)
                else:
                    active_params = getattr(room, 'next_round_spinner_params', None) or room.spinner_params or {}
                room.current_board_format = 'Valued Letters' if room.time_limit >= 7200 else active_params.get('board_format', 'Normal')
                # NOTE: current_word_count_range is intentionally NOT set here from active_params.
                # It will be computed from the actual board word count at the accuracy enforcement
                # step below (real_wc). Setting it prematurely here to the target value causes a
                # brief flash of the wrong label if a heartbeat fires before real_wc is ready.
                room.current_difficulty = active_params.get('difficulty', 'Medium')
                room.current_dictionary = active_params.get('dictionary', 'NWL')
                room.use_added_words = bool(active_params.get('use_added_words', False)) or ('+ AW' in str(room.current_dictionary).upper()) or ('+AW' in str(room.current_dictionary).upper()) or (str(room.current_dictionary).upper() in ['AW', 'ADDED_WORDS'])
                if room.use_added_words and '+ AW' not in str(room.current_dictionary) and '+AW' not in str(room.current_dictionary):
                    _base_d = 'CSW' if 'CSW' in str(room.current_dictionary).upper() else 'NWL'
                    room.current_dictionary = f"{_base_d} + AW"
                raw_min = active_params.get('min_word_length', 3)
                
                # Update spinner_params to match the actual board being used
                room.spinner_params = dict(active_params) if active_params else {}
                room.spinner_params['dictionary'] = room.current_dictionary
                room.spinner_params['use_added_words'] = room.use_added_words

                active_uniq = active_params.get('uniqueness') if isinstance(active_params, dict) else None
                next_uniq = active_uniq if active_uniq is not None else getattr(room, 'next_round_uniqueness', None)
                if next_uniq is not None:
                    room.current_uniqueness = float(next_uniq)
                    if isinstance(room.spinner_params, dict):
                        room.spinner_params['uniqueness'] = room.current_uniqueness
                    dims_parts = str(room.board_dimensions).lower().split('x')
                    d_cnt = int(dims_parts[0]) if len(dims_parts) == 3 else 1
                    r_cnt = int(dims_parts[1] if len(dims_parts) == 3 else dims_parts[0]) if len(dims_parts) >= 2 else 4
                    c_cnt = int(dims_parts[2] if len(dims_parts) == 3 else dims_parts[1]) if len(dims_parts) >= 2 else 4
                    _actual_diff = self.board_generator.get_difficulty_label(room.current_uniqueness, rows=r_cnt, cols=c_cnt, dictionary=room.current_dictionary, depth=d_cnt, min_word_length=getattr(room, 'current_min_length', 3))
                    if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                        _req_solo_d = room.initial_solo_params.get('difficulty')
                        if _req_solo_d and str(_req_solo_d).lower() != 'random':
                            _actual_diff = SpinnerSet.strip_difficulty_pct(_req_solo_d)
                            if _actual_diff == 'Easy' and room.current_uniqueness > 0.15 and getattr(room, 'board', None):
                                self.board_generator._sanitize_uniqueness(room.board, room.current_dictionary, max_ratio=0.15, depth=d_cnt, bonus_cell=room.bonus_cell, min_word_length=getattr(room, 'current_min_length', 3), use_added_words=room.use_added_words)
                                _re_s = self.board_generator._solve_board(room.board, room.current_dictionary, (0, 99999), getattr(room, 'current_min_length', 3), max_depth=20, store_paths=True, timeout=5.0, bonus_cell=room.bonus_cell, use_added_words=room.use_added_words)
                                room.all_words_paths = {w: p for w, p in _re_s.items() if len(w) >= getattr(room, 'current_min_length', 3)}
                                room.all_words = set(room.all_words_paths.keys())
                                room.total_words_count = len(room.all_words)
                                room.initial_total_words = room.total_words_count
                                room.current_uniqueness = self.board_generator._calculate_uniqueness_ratio(room.board, room.current_dictionary, d_cnt, room.bonus_cell, getattr(room, 'current_min_length', 3), use_added_words=room.use_added_words)
                        _req_solo_fmt = room.initial_solo_params.get('board_format')
                        if _req_solo_fmt and str(_req_solo_fmt).lower() != 'random':
                            room.current_board_format = _req_solo_fmt
                            if isinstance(room.spinner_params, dict):
                                room.spinner_params['board_format'] = _req_solo_fmt
                            if isinstance(active_params, dict):
                                active_params['board_format'] = _req_solo_fmt
                    room.current_difficulty = SpinnerSet.format_difficulty_with_pct(
                        _actual_diff, room.current_uniqueness
                    )
                    if isinstance(room.spinner_params, dict):
                        room.spinner_params['difficulty'] = room.current_difficulty
                    if isinstance(active_params, dict):
                        active_params['difficulty'] = room.current_difficulty
                try:
                    room.current_min_length = int(raw_min)
                except:
                    room.current_min_length = 3

                # --- 2. BOARD & WORD PROMOTION ---
                # EMERGENCY SAFETY: If for any reason staging is empty or desynchronized, force a fast fallback board NOW.
                # CRITICAL: Never call generate_board() synchronously here — it blocks for 10-30s.
                # get_emergency_fallback_board() is always instant: cache-first, then pre-built board.
                is_staging_consistent = False
                if room.next_round_board and room.next_round_words:
                    is_staging_consistent = verify_board_words_consistency(
                        room.next_round_board, room.next_round_words, getattr(room, 'next_round_word_paths', None)
                    )

                if not room.next_round_board or not room.next_round_words or not is_staging_consistent:
                    if getattr(room, 'game_type', None) == 'subanagrams':
                        from subanagrams_generator import generate_subanagrams_board_and_words
                        sp = dict(active_params or getattr(room, 'frozen_revealed_params', None) or room.spinner_params or SpinnerSet.generate_subanagrams_params())
                        sub_res = generate_subanagrams_board_and_words(sp)
                        if sub_res:
                            b_board, b_words, b_bonus, fparams = sub_res
                            room.next_round_board = b_board
                            room.next_round_words = b_words
                            room.next_round_word_paths = {w: [] for w in b_words}
                            room.next_round_total_words_count = len(b_words)
                            room.next_round_bonus = b_bonus
                            room.next_round_bonus_word = b_bonus
                            room.next_round_format = sp.get('board_format', 'Word Guaranteed')
                            room.next_round_bonus_cell = None
                            room.next_round_uniqueness = 0.0
                            from scoring import calculate_word_score
                            room.next_round_word_scores = {
                                w: calculate_word_score(w, b_bonus, board_format='Normal', return_details=True)
                                for w in b_words
                            }
                            if isinstance(room.spinner_params, dict) and isinstance(fparams, dict):
                                room.spinner_params.update(fparams)
                                room.spinner_params['board_format'] = sp.get('board_format', 'Word Guaranteed')
                                room.spinner_params['mode'] = 'word' if sp.get('board_format') == 'Word Guaranteed' else 'random'
                    else:
                        if not is_staging_consistent and room.next_round_board and room.next_round_words:
                            print(f"[REMAINING-STABILIZER] CRITICAL: Board and words desynchronization detected for {room_id} at promotion! Discarding mismatched staging and forcing emergency fallback.")
                        else:
                            print(f"[REMAINING-STABILIZER] Staging empty for {room_id} at promotion. Forcing INSTANT emergency fallback (cache or pre-built).")
                        if room.time_limit >= 7200:
                            room.current_board_format = 'Valued Letters'
                            room.current_dictionary = room.spinner_params.get('dictionary', 'NWL')
                            room.current_min_length = int(room.spinner_params.get('min_word_length', 3))

                        e_min_len = room.spinner_params.get('min_word_length') if room.spinner_params else None
                        e_diff = room.spinner_params.get('difficulty', 'Medium') if room.spinner_params else 'Medium'
                        e_fmt_target = room.spinner_params.get('board_format', 'Normal') if room.spinner_params else 'Normal'
                        e_target_range = room.spinner_params.get('word_count_range') if room.spinner_params else None
                        e_bw_len = room.spinner_params.get('bonus_word_length') if room.spinner_params else None
                        if not e_target_range:
                            e_target_range = random.choice(['100-200', '200-300', '300-400'])
                        # Use the always-instant emergency fallback (cache → pre-built). Never blocks.
                        e_fallback = get_emergency_fallback_board(
                            room.board_dimensions, e_fmt_target, room.time_limit,
                            dictionary=room.current_dictionary,
                            use_added_words=getattr(room, 'use_added_words', False),
                            target_range=e_target_range,
                            min_word_length=e_min_len or 3,
                            difficulty=e_diff,
                            bonus_word_len=e_bw_len
                        )
                        if e_fallback and len(e_fallback) >= 8:
                            if len(e_fallback) >= 9:
                                e_board, e_words, e_bonus_c, e_fmt, e_paths, e_ratio, e_bonus_word, e_tr, e_params = e_fallback
                            else:
                                e_board, e_words, e_bonus_c, e_fmt, e_paths, e_ratio, e_bonus_word, e_tr = e_fallback
                                e_params = {}
                        else:
                            # Absolute last resort: tiny pre-built board (should never reach here)
                            print(f"[REMAINING-STABILIZER] get_emergency_fallback_board returned nothing for {room_id}. Using hardcoded micro-board.")
                            dims_parts = room.board_dimensions.split('x')
                            rows_n, cols_n = int(dims_parts[0]), int(dims_parts[1]) if len(dims_parts) >= 2 else (4, 4)
                            e_board = [['S','T','A','R'],['E','D','L','I'],['N','E','R','S'],['A','N','T','S']][:rows_n]
                            e_board = [row[:cols_n] for row in e_board]
                            e_words = ['ANTS', 'RANT', 'RANT', 'STAR', 'TARS', 'LEND', 'REND']
                            e_bonus_c = (0, 0)
                            e_fmt = e_fmt_target
                            e_paths = {}
                            e_ratio = 0.2
                            e_bonus_word = 'LANDERS'
                            e_tr = e_target_range
                            e_params = {}
                            
                        if e_params and not getattr(room, 'was_revealed_this_intermission', False) and not getattr(room, 'spinner_params_revealed', False):
                            print(f"[REMAINING-STABILIZER] Aligning room spinner parameters to match popped fallback board: {e_params}")
                            orig_dict = room.current_dictionary or room.spinner_params.get('dictionary')
                            orig_aw = getattr(room, 'use_added_words', False) or room.spinner_params.get('use_added_words', False)
                            final_dict = orig_dict or e_params.get('dictionary', 'NWL')
                            final_aw = orig_aw or e_params.get('use_added_words', False) or ('+ AW' in str(final_dict).upper())
                            clean_fd = str(final_dict).replace('+ AW', '').replace('+AW', '').strip()
                            final_dict = f"{clean_fd} + AW" if final_aw else clean_fd

                            _e_min = e_params.get('min_word_length', 3)
                            if e_words:
                                _act_s = min(len(w) for w in e_words)
                                if _act_s > _e_min:
                                    _e_min = _act_s

                            room.spinner_params['dictionary'] = final_dict
                            room.spinner_params['board_format'] = e_params.get('board_format', 'Normal')
                            room.spinner_params['min_word_length'] = _e_min
                            room.spinner_params['use_added_words'] = final_aw
                            room.spinner_params['bonus_word_length'] = len(e_bonus_word) if e_bonus_word else e_params.get('bonus_word_len', 6)
                            
                            room.current_dictionary = final_dict
                            room.current_board_format = e_params.get('board_format', 'Normal')
                            room.current_min_length = _e_min
                            room.use_added_words = final_aw
                        
                        room.next_round_board = e_board
                        room.next_round_words = e_words
                        room.next_round_word_paths = e_paths
                        room.next_round_total_words_count = len(e_words)
                        room.next_round_bonus = e_bonus_word
                        room.next_round_format = e_fmt
                        room.next_round_bonus_cell = e_bonus_c
                        room.next_round_uniqueness = e_ratio
                        room.initialize_density(e_board, e_paths, e_fmt, is_staging=True)
                        
                        if hasattr(word_validator, 'word_validator'):
                            word_validator.word_validator.ensure_csw_loaded()
                            room.next_round_csw_only_words = [w for w in e_words if word_validator.word_validator.is_csw_only(w)]
                            room.next_round_added_words = [w for w in e_words if word_validator.word_validator.is_added_word(w)]
                        else:
                            room.next_round_csw_only_words = []
                            room.next_round_added_words = []
                    
                    # USER REQUEST: Ensure Total Points is never 0.
                    # Fast-apply length based scores for the emergency board immediately.
                    # Detailed scores for the emergency board immediately
                    e_scores = {}
                    for w in e_words:
                        p_v = _get_word_path(e_paths, w)
                        e_scores[w] = calculate_word_score(
                            w,
                            bonus_word=e_bonus_word,
                            board_format=e_fmt,
                            bonus_cell=e_bonus_c,
                            board=e_board,
                            path=p_v,
                            return_details=True
                        )
                    room.next_round_word_scores = e_scores
                
                # --- 3. FINAL COUNT SYNC: Ensure factual counts are promoted even if truncation was skipped ---
                # USER REQUEST: Total count should reflect scorable words only.
                if hasattr(room, 'next_round_total_words_count') and room.next_round_total_words_count > 0:
                    room.total_words_count = room.next_round_total_words_count
                    room.initial_total_words = room.total_words_count
                else:
                    room.total_words_count = sum(1 for w in (room.next_round_words or []) if len(w) >= room.current_min_length)
                
                # --- 4. HISTORY PROMOTION ---
                # For 24-hour rooms, we do NOT overwrite these variables since the midnight transition
                # in check_and_update_state already captured the precise yesterday snapshots.
                if room.time_limit < 7200:
                    room.previous_min_length = ghost_min_len
                    room.previous_bonus_word = (ghost_bonus or '')
                    room.previous_bonus_cell = getattr(room, 'bonus_cell', None)
                    room.previous_board_format = ghost_board_format
                    room.previous_dictionary = getattr(room, 'current_dictionary', 'NWL')
                    room.previous_use_added_words = bool(getattr(room, 'use_added_words', False)) or ('+ AW' in str(room.previous_dictionary).upper()) or ('+AW' in str(room.previous_dictionary).upper())
                    # NOTE: `copy` is imported at module level. Do NOT re-import it here: a local
                    # `import copy` makes it a function-local name for ALL of start_next_round and
                    # breaks 24h rooms (which skip this branch) with UnboundLocalError.
                    room.previous_board = copy.deepcopy(ghost_prev_board) if ghost_prev_board else []
                    room.previous_all_words = [w for w in (ghost_source_words or []) if len(w) >= ghost_min_len]
                    room.previous_all_words_paths = dict(ghost_all_words_paths)
                    room.previous_all_word_scores = dict(getattr(room, 'solved_words_with_scores', {}))
                    if hasattr(word_validator, 'word_validator'):
                        word_validator.word_validator.ensure_csw_loaded()
                        room.previous_csw_only_words = [w for w in (room.previous_all_words or []) if word_validator.word_validator.is_csw_only(w)]
                        room.previous_added_words = [w for w in (room.previous_all_words or []) if word_validator.word_validator.is_added_word(w)]
                    else:
                        room.previous_csw_only_words = list(room.csw_only_words) if getattr(room, 'csw_only_words', None) else []
                        room.previous_added_words = list(room.added_words) if getattr(room, 'added_words', None) else []
                
                # Update current active counts synchronously (takes < 0.05ms via set lookup; no background race)
                room.csw_only_words = list(getattr(room, 'next_round_csw_only_words', []) or [])
                room.added_words = list(getattr(room, 'next_round_added_words', []) or [])

                # ATOMIC PROMOTION: Carry staging data to active room state
                if getattr(room, 'game_type', None) != 'subanagrams' and room.next_round_board:
                    san_b, san_w, san_p, san_c, san_bw, _ = self.sanitize_board_for_room_difficulty(
                        room, room.next_round_board, getattr(room, 'next_round_words', []),
                        getattr(room, 'next_round_word_paths', {}), getattr(room, 'next_round_bonus_cell', None),
                        getattr(room, 'next_round_bonus', None), getattr(room, 'next_round_format', None),
                        getattr(room, 'current_difficulty', None) or (room.spinner_params.get('difficulty') if isinstance(room.spinner_params, dict) else 'Medium')
                    )
                    room.next_round_board = san_b
                    if san_w:
                        room.next_round_words = san_w
                        room.next_round_word_paths = san_p
                        room.next_round_bonus = san_bw
                        room.next_round_bonus_cell = san_c
                room.board = room.next_round_board
                room._board_promoted_this_transition = True
                if getattr(room, 'game_type', None) == 'subanagrams':
                    is_invalid_sub = (
                        not room.board or
                        not isinstance(room.board, list) or
                        len(room.board) != 1 or
                        not isinstance(room.board[0], list) or
                        not (6 <= len(room.board[0]) <= 10)
                    )
                    if is_invalid_sub:
                        print(f"[RoomManager] Promoted Subanagrams board was non-1D or not 6-10 letters ({room.board}). Healing...")
                        from subanagrams_generator import generate_subanagrams_board_and_words
                        heal_p = dict(getattr(room, 'frozen_revealed_params', None) or room.spinner_params or {})
                        heal_p['board_dimensions'] = 'subanagrams'
                        heal_p['sequence_length'] = max(6, min(10, int(heal_p.get('sequence_length', 8))))
                        heal_res = generate_subanagrams_board_and_words(heal_p)
                        if heal_res:
                            hb, hw, hbw, hp = heal_res
                            room.board = hb
                            room.next_round_words = hw
                            room.all_words = set(hw)
                            room.all_words_paths = {w: [] for w in hw}
                            room.sequence_length = len(hb[0])
                            room.next_round_bonus = hbw
                            room.bonus_word = hbw
                            if isinstance(room.spinner_params, dict):
                                room.spinner_params.update(hp)
                            if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                                room.frozen_revealed_params.update(hp)
                    else:
                        actual_seq_len = len(room.board[0])
                        room.sequence_length = actual_seq_len
                        if isinstance(room.spinner_params, dict):
                            room.spinner_params['sequence_length'] = actual_seq_len
                        if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                            room.frozen_revealed_params['sequence_length'] = actual_seq_len
                room.current_board_format = 'Valued Letters' if room.time_limit >= 7200 else active_params.get('board_format', 'Normal')
                
                # USER REQUEST: Absolute consistency. Only include words that meet the round's scorable minimum.
                # HARD FLOOR: Always exclude 3-letter words from the 'All Words' list (User Request: "NOT 3 letter wrods")
                min_l = room.current_min_length if hasattr(room, 'current_min_length') else 3
                display_min = min_l
                room.all_words_paths = {w: p for w, p in (room.next_round_word_paths or {}).items() if len(w) >= display_min}
                
                # DICTIONARY ENFORCEMENT: Never allow CSW-only words to leak into an NWL room
                if 'CSW' not in str(getattr(room, 'current_dictionary', 'NWL')).upper() and hasattr(word_validator, 'word_validator'):
                    word_validator.word_validator.ensure_csw_loaded()
                    room.all_words_paths = {w: p for w, p in room.all_words_paths.items() if not word_validator.word_validator.is_csw_only(w)}
                
                room.all_words = set(room.all_words_paths.keys())
                room.total_words_count = len(room.all_words)
                room.initial_total_words = room.total_words_count
                if hasattr(word_validator, 'word_validator'):
                    word_validator.word_validator.ensure_csw_loaded()
                    room.csw_only_words = [w for w in room.all_words if word_validator.word_validator.is_csw_only(w)]
                    room.added_words = [w for w in room.all_words if word_validator.word_validator.is_added_word(w)] if room.use_added_words else []
                
                # ACTIVE INTEGRITY SHIELD: Auto-heal if active words ever mismatch active board
                if getattr(room, 'game_type', None) != 'subanagrams' and room.board and room.all_words and not verify_board_words_consistency(room.board, room.all_words, room.all_words_paths):
                    print(f"[RoomManager] CRITICAL WARNING: Active board and words mismatch detected for {room_id} post-promotion! Auto-healing word list...")
                    _bg = self.board_generator
                    _is_heal_aw = bool(getattr(room, 'use_added_words', False)) or ('+ AW' in str(getattr(room, 'current_dictionary', '')).upper()) or ('+AW' in str(getattr(room, 'current_dictionary', '')).upper())
                    solved_dict = _bg._solve_board(room.board, dictionary=getattr(room, 'current_dictionary', 'NWL'), min_word_length=display_min, use_added_words=_is_heal_aw)
                    room.all_words_paths = {w: p for w, p in solved_dict.items() if len(w) >= display_min}
                    room.all_words = set(room.all_words_paths.keys())
                    room.total_words_count = len(room.all_words)
                    room.initial_total_words = room.total_words_count
                    room.next_round_words = list(room.all_words)
                    room.next_round_word_paths = dict(room.all_words_paths)
                    if _is_heal_aw and hasattr(word_validator, 'word_validator'):
                        room.added_words = [w for w in room.all_words if word_validator.word_validator.is_added_word(w)]
                    if hasattr(word_validator, 'word_validator'):
                        word_validator.word_validator.ensure_csw_loaded()
                        room.csw_only_words = [w for w in room.all_words if word_validator.word_validator.is_csw_only(w)]

                # ACTIVE VALIDATION SHIELD: Prevent 0 CSW-only or 0 Added Words in CSW/AW rooms
                if getattr(room, 'game_type', None) != 'subanagrams':
                    needs_dict_healing = False
                    if 'CSW' in str(getattr(room, 'current_dictionary', '')).upper() and len(getattr(room, 'csw_only_words', [])) == 0:
                        needs_dict_healing = True
                    if getattr(room, 'use_added_words', False) and len(getattr(room, 'added_words', [])) == 0:
                        needs_dict_healing = True
                    if needs_dict_healing and room.board:
                        print(f"[RoomManager] CRITICAL WARNING: Active board for {room_id} has 0 CSW/AW words in {room.current_dictionary}! Re-solving...")
                        _bg = self.board_generator
                        _is_heal_aw = bool(getattr(room, 'use_added_words', False)) or ('+ AW' in str(getattr(room, 'current_dictionary', '')).upper())
                        _heal_dict = 'CSW' if 'CSW' in str(room.current_dictionary).upper() else 'NWL'
                        solved_dict = _bg._solve_board(room.board, dictionary=_heal_dict, min_word_length=display_min, use_added_words=_is_heal_aw)
                        if solved_dict:
                            room.all_words_paths = {w: p for w, p in solved_dict.items() if len(w) >= display_min}
                            room.all_words = set(room.all_words_paths.keys())
                            room.total_words_count = len(room.all_words)
                            room.initial_total_words = room.total_words_count
                            room.next_round_words = list(room.all_words)
                            room.next_round_word_paths = dict(room.all_words_paths)
                            if hasattr(word_validator, 'word_validator'):
                                word_validator.word_validator.ensure_csw_loaded()
                                room.csw_only_words = [w for w in room.all_words if word_validator.word_validator.is_csw_only(w)]
                                room.added_words = [w for w in room.all_words if word_validator.word_validator.is_added_word(w)]

                # ACTIVE WORD COUNT SHIELD: Guarantee board word count strictly satisfies announced Spinner range
                is_24h_room = (room.time_limit >= 7200) or ('24h' in str(room.room_id)) or ('86400' in str(room.room_id))
                if getattr(room, 'game_type', None) != 'subanagrams' and (getattr(room, 'was_revealed_this_intermission', False) or getattr(room, 'spinner_params_revealed', False) or is_24h_room):
                    target_announced_wc = (getattr(room, 'frozen_revealed_params', {}) or {}).get('word_count_range') or (room.spinner_params or {}).get('word_count_range')
                    if is_24h_room:
                        target_announced_wc = '300-400'
                    if target_announced_wc and not is_board_count_valid(len(room.all_words), target_announced_wc):
                        print(f"[RoomManager] CRITICAL WARNING: Active board word count {len(room.all_words)} violates announced range {target_announced_wc} for {room_id}! Healing...")
                        h_fallback = get_emergency_fallback_board(
                            room.board_dimensions, room.current_board_format, room.time_limit,
                            dictionary=room.current_dictionary,
                            use_added_words=getattr(room, 'use_added_words', False),
                            target_range=target_announced_wc,
                            min_word_length=display_min
                        )
                        if h_fallback and len(h_fallback) >= 8:
                            h_b, h_w, h_c, h_f, h_p, h_r, h_bw = h_fallback[:7]
                            h_words_filt = [w for w in h_w if len(w) >= display_min]
                            if is_board_count_valid(len(h_words_filt), target_announced_wc):
                                room.board = h_b
                                room.all_words = set(h_words_filt)
                                room.all_words_paths = {w: p for w, p in h_p.items() if w in room.all_words}
                                room.total_words_count = len(room.all_words)
                                room.initial_total_words = room.total_words_count
                                room.bonus_cell = h_c
                                room.next_round_board = h_b
                                room.next_round_words = list(room.all_words)
                                room.next_round_word_paths = dict(room.all_words_paths)
                                room.next_round_bonus = h_bw
                                if hasattr(word_validator, 'word_validator'):
                                    word_validator.word_validator.ensure_csw_loaded()
                                    room.csw_only_words = [w for w in room.all_words if word_validator.word_validator.is_csw_only(w)]
                                    room.added_words = [w for w in room.all_words if word_validator.word_validator.is_added_word(w)]
                                print(f"[RoomManager] Word count successfully healed to {len(room.all_words)} words matching {target_announced_wc}.")
                        if is_24h_room and not (300 <= len(room.all_words) <= 399):
                            try:
                                print(f"[RoomManager] Emergency fallback could not hit 300-400 for 24h room {room_id}. Running synchronous generator...")
                                _sync_24h = self.board_generator.generate_board(
                                    dimensions=room.board_dimensions,
                                    bonus_word=getattr(room, 'bonus_word', None) or 'PLANETS',
                                    word_count_range='300-400',
                                    dictionary=room.current_dictionary,
                                    board_format='Valued Letters',
                                    min_word_length=display_min,
                                    difficulty=room.current_difficulty or "Medium",
                                    timeout=15.0,
                                    use_added_words=False
                                )
                                if _sync_24h and len(_sync_24h) >= 7:
                                    sb, sw, sc, sf, sp, sr, sbw = _sync_24h[:7]
                                    sw_filt = [w for w in sw if len(w) >= display_min]
                                    if 300 <= len(sw_filt) <= 399:
                                        room.board = sb
                                        room.all_words = set(sw_filt)
                                        room.all_words_paths = {w: p for w, p in sp.items() if w in room.all_words}
                                        room.total_words_count = len(room.all_words)
                                        room.initial_total_words = room.total_words_count
                                        room.bonus_cell = sc
                                        room.next_round_board = sb
                                        room.next_round_words = list(room.all_words)
                                        room.next_round_word_paths = dict(room.all_words_paths)
                                        room.next_round_bonus = sbw
                            except Exception as _sync_err:
                                print(f"[RoomManager] 24h synchronous generator error: {_sync_err}")

                room.solved_words_with_scores = getattr(room, 'next_round_word_scores', {})
                if getattr(room, 'game_type', None) == 'subanagrams' and (not room.solved_words_with_scores or len(room.solved_words_with_scores) == 0):
                    from scoring import calculate_word_score
                    bw = getattr(room, 'bonus_word', None) or getattr(room, 'next_round_bonus', None)
                    room.solved_words_with_scores = {
                        w: calculate_word_score(w, bw, board_format='Normal', return_details=True)
                        for w in (room.all_words or [])
                    }
                
                # Save to DB will occur at the end of promotion after all parameters are finalized
                
                if getattr(room, 'game_type', None) == 'subanagrams':
                    # MULTIPLAYER SUBANAGRAMS: Strictly preserve the bonus word generated and declared at 0:45 intermission.
                    # Never call pick_distinct_bonus_word for Subanagrams: it clamps target_len to [6..10],
                    # rejects ING/INGS words, and mutates declared bonus_word_length.
                    declared_bw = getattr(room, 'next_round_bonus', None) or (room.frozen_revealed_params.get('bonus_word') if getattr(room, 'frozen_revealed_params', None) else None)
                    declared_bw_len = getattr(room, 'frozen_revealed_params', {}).get('bonus_word_length') if getattr(room, 'frozen_revealed_params', None) else None
                    if not declared_bw_len and isinstance(room.spinner_params, dict):
                        declared_bw_len = room.spinner_params.get('bonus_word_length')

                    if declared_bw and (not declared_bw_len or len(declared_bw) == int(declared_bw_len)):
                        room.bonus_word = declared_bw
                    elif declared_bw_len and room.all_words:
                        matching = [w for w in room.all_words if len(w) == int(declared_bw_len)]
                        room.bonus_word = random.choice(matching) if matching else (declared_bw or random.choice(list(room.all_words)))
                    else:
                        room.bonus_word = declared_bw or (random.choice(list(room.all_words)) if room.all_words else '')

                    actual_bw_len = len(room.bonus_word) if room.bonus_word else int(declared_bw_len or 8)
                    if isinstance(room.spinner_params, dict):
                        room.spinner_params['bonus_word_length'] = actual_bw_len
                        room.spinner_params['bonus_word'] = room.bonus_word
                    if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                        room.frozen_revealed_params['bonus_word_length'] = actual_bw_len
                        room.frozen_revealed_params['bonus_word'] = room.bonus_word
                    if getattr(room, 'next_spinner_params', None) and isinstance(room.next_spinner_params, dict):
                        room.next_spinner_params['bonus_word_length'] = actual_bw_len
                        room.next_spinner_params['bonus_word'] = room.bonus_word
                    sync_bonus_word_scores(room)
                else:
                    current_bw = getattr(room, 'next_round_bonus', '')
                    revealed_bw_l = getattr(room, 'frozen_revealed_params', {}).get('bonus_word_length') if getattr(room, 'frozen_revealed_params', None) else None
                    bw_l = int(revealed_bw_l or room.spinner_params.get('bonus_word_length', 8) or 8) if isinstance(room.spinner_params, dict) else 8
                    if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                        _solo_bw_l = room.initial_solo_params.get('bonus_word_length', 'random')
                        if _solo_bw_l and str(_solo_bw_l).lower() not in ['random', '0', 'none', 'null', 'nan']:
                            try:
                                bw_l = int(_solo_bw_l)
                            except Exception:
                                pass
                    dict_val = room.spinner_params.get('dictionary', 'NWL') if isinstance(room.spinner_params, dict) else 'NWL'
                    target_words = room.all_words or room.next_round_words or []
                    is_chk_start = 'checkerboard' in str(room.current_board_format).lower()
                    room.bonus_word = pick_distinct_bonus_word(
                        room, current_bw, target_words, bw_l,
                        dictionary=dict_val, is_checkerboard=is_chk_start,
                        room_manager=self, extra_exclude=[ghost_bonus, getattr(room, 'previous_bonus_word', None)]
                    )
                    ensure_bonus_word_on_board(room, bw_l)
                    if room.bonus_word:
                        actual_bw_len = len(room.bonus_word)
                        if isinstance(room.spinner_params, dict):
                            room.spinner_params['bonus_word_length'] = actual_bw_len
                        if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                            room.frozen_revealed_params['bonus_word_length'] = actual_bw_len
                        if getattr(room, 'next_spinner_params', None) and isinstance(room.next_spinner_params, dict):
                            room.next_spinner_params['bonus_word_length'] = actual_bw_len
                    sync_bonus_word_scores(room)
                
                room.bonus_cell = getattr(room, 'next_round_bonus_cell', None)

                # BONUS LETTER GUARD: The spun format can say "Bonus Letter" while the staged board
                # (cache/fallback pop) carries no bonus cell. Ensure a cell exists so the green tile shows.
                try:
                    _fmt_l = str(room.current_board_format or '').lower()
                    if room.time_limit < 7200 and ('bonus letter' in _fmt_l or _fmt_l.strip() == 'bonus') and not room.bonus_cell and room.board:
                        _b = room.board
                        if isinstance(_b[0][0], list):  # 3D board: (f, r, c)
                            _f = random.randint(0, len(_b) - 1)
                            room.bonus_cell = (_f, random.randint(0, len(_b[_f]) - 1), random.randint(0, len(_b[_f][0]) - 1))
                        else:
                            room.bonus_cell = (random.randint(0, len(_b) - 1), random.randint(0, len(_b[0]) - 1))
                        print(f"[RoomManager] Bonus Letter format had no bonus cell for {room_id}; assigned {room.bonus_cell}")
                        try:
                            _is_bl_aw = bool(getattr(room, 'use_added_words', False)) or ('+ AW' in str(getattr(room, 'current_dictionary', '')).upper()) or ('+AW' in str(getattr(room, 'current_dictionary', '')).upper())
                            _solved = self.board_generator._solve_board(room.board, getattr(room, 'current_dictionary', 'NWL'), (0, 99999), display_min, max_depth=25, store_paths=True, timeout=5.0, bonus_cell=room.bonus_cell, use_added_words=_is_bl_aw)
                            if _solved:
                                _upd = {w: p for w, p in _solved.items() if w in room.all_words_paths}
                                if len(_upd) == len(room.all_words_paths):
                                    room.all_words_paths = _upd
                        except Exception as _se:
                            print(f"[RoomManager] Bonus cell re-solve failed (keeping existing paths): {_se}")
                except Exception as _bg_err:
                    print(f"[RoomManager] Bonus cell guard error: {_bg_err}")

                # MANIA GUARD: Ensure promoted board strictly contains the required number of Mania letters
                try:
                    _fmt_cur = str(room.current_board_format or '').strip().upper()
                    if "MANIA" in _fmt_cur and room.board:
                        _parts_m = _fmt_cur.split()
                        if len(_parts_m) >= 2 and len(_parts_m[0]) == 1 and _parts_m[0].isalpha():
                            _m_letter = _parts_m[0]
                            _is_3d_m = isinstance(room.board, list) and len(room.board) > 0 and isinstance(room.board[0], list) and isinstance(room.board[0][0], list)
                            if _is_3d_m:
                                _m_cnt = sum(1 for f in range(len(room.board)) for r in range(len(room.board[f])) for c in range(len(room.board[f][r])) if room.board[f][r][c] == _m_letter)
                                _tot_cells = len(room.board) * len(room.board[0]) * len(room.board[0][0])
                            else:
                                _m_cnt = sum(1 for r in range(len(room.board)) for c in range(len(room.board[r])) if room.board[r][c] == _m_letter)
                                _tot_cells = len(room.board) * len(room.board[0])
                            
                            _UNCOMMON_LETTERS = {'Q', 'Z', 'J', 'X', 'K'}
                            _target_ratio = 0.20 if _m_letter in _UNCOMMON_LETTERS else 0.33
                            _target_m_cnt = max(2, round(_tot_cells * _target_ratio))
                            
                            if _m_cnt < _target_m_cnt:
                                print(f"[RoomManager] Mania Guard: Room {room_id} has format {_fmt_cur} but board only has {_m_cnt} '{_m_letter}' (needed {_target_m_cnt}). Applying mania letters...")
                                self.board_generator._apply_mania_to_board(room.board, _m_letter, set(), is_checkerboard=False)
                                _is_m_aw = bool(getattr(room, 'use_added_words', False)) or ('+ AW' in str(getattr(room, 'current_dictionary', '')).upper()) or ('+AW' in str(getattr(room, 'current_dictionary', '')).upper())
                                _re_solved = self.board_generator._solve_board(room.board, getattr(room, 'current_dictionary', 'NWL'), (0, 99999), display_min, max_depth=25, store_paths=True, timeout=5.0, bonus_cell=room.bonus_cell, use_added_words=_is_m_aw)
                                if _re_solved:
                                    room.all_words = set(_re_solved.keys())
                                    room.all_words_paths = _re_solved
                                    room.total_words_count = len(room.all_words)
                                    room.initial_total_words = room.total_words_count
                except Exception as _mania_guard_err:
                    print(f"[RoomManager] Mania guard error: {_mania_guard_err}")

                # --- 4. ACCURACY ENFORCEMENT (inside lock — all_words must be consistent before state='active') ---
                # These loops run over ~100-500 words and take < 5ms — not the bottleneck.
                _sp     = dict(room.spinner_params) if isinstance(room.spinner_params, dict) else {}
                _bw_up  = str(getattr(room, 'bonus_word', '')).upper().strip()
                _dict   = room.current_dictionary
                _use_aw = getattr(room, 'use_added_words', False)

                # Min-length filter
                room.all_words = {w for w in room.all_words if len(w) >= room.current_min_length}

                # Range cap truncation
                declared_wc = None
                if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                    declared_wc = room.frozen_revealed_params.get('word_count_range')
                if not declared_wc and active_params:
                    declared_wc = active_params.get('word_count_range')

                target_sp_range = declared_wc or _sp.get('word_count_range', '100-200')
                max_cap = None
                if target_sp_range == '50-100':   max_cap = 99
                elif target_sp_range == '100-200': max_cap = 199
                elif target_sp_range == '200-300': max_cap = 299
                elif target_sp_range == '300-400': max_cap = 399
                elif target_sp_range == '400-500': max_cap = 499

                if False and getattr(room, 'game_type', None) != 'subanagrams' and max_cap and len(room.all_words) > max_cap:  # never truncate valid words
                    _wl       = list(room.all_words)
                    total_raw = len(_wl)
                    by_len    = {}
                    for w in _wl:
                        by_len.setdefault(len(w), []).append(w)
                    selected = set()
                    if _bw_up and _bw_up in room.all_words:
                        selected.add(_bw_up)
                    for l in sorted(by_len.keys()):
                        bw2 = [w for w in by_len[l] if w not in selected]
                        if not bw2: continue
                        pc = max(1, min(len(bw2), int(round((len(by_len[l]) / float(total_raw)) * max_cap))))
                        selected.update(sorted(bw2, key=lambda w: (len(w), w), reverse=True)[:pc])
                    if len(selected) < max_cap:
                        lft = sorted([w for w in _wl if w not in selected], key=lambda w: (len(w), w), reverse=True)
                        selected.update(lft[:(max_cap - len(selected))])
                    elif len(selected) > max_cap:
                        nb = [w for w in selected if w != _bw_up]
                        selected = set(nb[:max_cap])
                        if _bw_up and _bw_up in room.all_words: selected.add(_bw_up)
                    room.all_words = selected

                # Keep paths and scores in sync with the final word set
                room.all_words_paths = {w: room.all_words_paths.get(w, []) for w in room.all_words}
                room.solved_words_with_scores = {w: v for w, v in (room.solved_words_with_scores or {}).items() if w in room.all_words}

                # Word count and range label — AW boards use the same full scale as non-AW.
                # 319 words on a board is '300-400', aligning header with board count.
                wc_cnt  = len(room.all_words)
                if getattr(room, 'game_type', None) == 'subanagrams':
                    real_wc = SpinnerSet.get_subanagrams_factchecked_wc_range(wc_cnt)
                else:
                    real_wc = self._get_factchecked_wc_range(wc_cnt, use_added_words=_use_aw, dictionary=room.current_dictionary)

                if room.all_words and getattr(room, 'game_type', None) != 'subanagrams':
                    actual_shortest = min(len(w) for w in room.all_words)
                    if actual_shortest != room.current_min_length:
                        print(f"[RoomManager] start_next_round aligned min_word_length from {room.current_min_length} to factual {actual_shortest}.")
                        room.current_min_length = actual_shortest
                        if isinstance(room.spinner_params, dict):
                            room.spinner_params['min_word_length'] = actual_shortest
                        if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                            room.frozen_revealed_params['min_word_length'] = actual_shortest
                        if getattr(room, 'next_spinner_params', None) and isinstance(room.next_spinner_params, dict):
                            room.next_spinner_params['min_word_length'] = actual_shortest

                room.total_words_count    = wc_cnt
                room.initial_total_words  = wc_cnt

                # USER REQUIREMENT: When the Spinner Set declares parameters at 0:45 during intermission,
                # do NOT change them at the start of the following round.
                declared_wc = None
                if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                    declared_wc = room.frozen_revealed_params.get('word_count_range')
                if not declared_wc and active_params:
                    declared_wc = active_params.get('word_count_range')

                if is_24h_room:
                    final_wc = '300-400'
                else:
                    final_wc = declared_wc if (declared_wc and is_board_count_valid(wc_cnt, declared_wc)) else real_wc

                room.current_word_count_range = final_wc
                if isinstance(room.spinner_params, dict):
                    room.spinner_params['word_count_range'] = final_wc
                if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                    room.frozen_revealed_params['word_count_range'] = final_wc
                if getattr(room, 'next_spinner_params', None) and isinstance(room.next_spinner_params, dict):
                    room.next_spinner_params['word_count_range'] = final_wc

                # SUBANAGRAMS ACCURACY ENFORCEMENT: Strictly adhere to parameters declared at 0:45 intermission
                if getattr(room, 'game_type', None) == 'subanagrams':
                    declared_fmt = (getattr(room, 'frozen_revealed_params', None) or active_params).get('board_format', 'Word Guaranteed')
                    seq_len_chk = getattr(room, 'sequence_length', None) or (len(room.board[0]) if (room.board and len(room.board) > 0 and isinstance(room.board[0], list)) else 8)
                    has_full_word = any(len(w) == seq_len_chk for w in room.all_words)
                    if declared_fmt == 'Word Guaranteed' and not has_full_word:
                        from subanagrams_generator import generate_subanagrams_board_and_words
                        heal_p = dict(getattr(room, 'frozen_revealed_params', None) or active_params)
                        heal_p['board_format'] = 'Word Guaranteed'
                        heal_p['mode'] = 'word'
                        heal_res = generate_subanagrams_board_and_words(heal_p)
                        if heal_res:
                            hb, hw, hbw, hp = heal_res
                            room.board = hb
                            room.all_words = set(hw)
                            room.all_words_paths = {w: [] for w in hw}
                            room.sequence_length = len(hb[0])
                            room.bonus_word = hbw
                            room.total_words_count = len(hw)
                            room.initial_total_words = len(hw)
                            room.current_board_format = 'Word Guaranteed'
                            room.current_word_count_range = hp.get('word_count_range') or SpinnerSet.get_subanagrams_factchecked_wc_range(len(hw))
                            if isinstance(room.spinner_params, dict):
                                room.spinner_params['word_count_range'] = room.current_word_count_range
                                room.spinner_params['bonus_word'] = hbw
                                room.spinner_params['bonus_word_length'] = len(hbw)
                            if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                                room.frozen_revealed_params['word_count_range'] = room.current_word_count_range
                                room.frozen_revealed_params['bonus_word'] = hbw
                                room.frozen_revealed_params['bonus_word_length'] = len(hbw)
                            sync_bonus_word_scores(room)
                    else:
                        room.current_board_format = declared_fmt

                    sub_mode = 'word' if room.current_board_format == 'Word Guaranteed' else 'random'
                    if isinstance(room.spinner_params, dict):
                        room.spinner_params['board_format'] = room.current_board_format
                        room.spinner_params['mode'] = sub_mode
                        room.spinner_params['sequence_length'] = room.sequence_length
                        if getattr(room, 'bonus_word', None):
                            room.spinner_params['bonus_word'] = room.bonus_word
                            room.spinner_params['bonus_word_length'] = len(room.bonus_word)
                    if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                        room.frozen_revealed_params['board_format'] = room.current_board_format
                        room.frozen_revealed_params['mode'] = sub_mode
                        room.frozen_revealed_params['sequence_length'] = room.sequence_length
                        if getattr(room, 'bonus_word', None):
                            room.frozen_revealed_params['bonus_word'] = room.bonus_word
                            room.frozen_revealed_params['bonus_word_length'] = len(room.bonus_word)
                    if getattr(room, 'next_spinner_params', None) and isinstance(room.next_spinner_params, dict):
                        room.next_spinner_params['board_format'] = room.current_board_format
                        room.next_spinner_params['mode'] = sub_mode
                        room.next_spinner_params['sequence_length'] = room.sequence_length
                        if getattr(room, 'bonus_word', None):
                            room.next_spinner_params['bonus_word'] = room.bonus_word
                            room.next_spinner_params['bonus_word_length'] = len(room.bonus_word)

                # Increment round counter BEFORE stamping total_counts_by_len['_round']
                room.current_round += 1

                room.complete_words = list(room.all_words)
                room.update_counts_by_len()
                room.recalculate_total_points()

                room.cell_density = getattr(room, 'next_round_cell_density', [])
                room.initial_cell_density = getattr(room, 'next_round_initial_cell_density', [])
                room.max_cell_density = getattr(room, 'next_round_max_cell_density', 0)
                room.global_round_found_words = set()
                room.initialize_player_densities()
                
                room.solving_complete = True
                # complete_words, update_counts_by_len, recalculate_total_points run AFTER lock
                
                # Save to DB for cheat prevention across workers and 24h room persistence
                if room.board and len(room.board) > 0:
                    def save_board_db_async():
                        try:
                            import json, time
                            players_data = []
                            for p in room.players:
                                players_data.append({
                                    'user_id': p.user_id, 'username': p.username, 'rating': p.rating,
                                    'submitted_words': p.submitted_words, 'invalid_words': p.invalid_words,
                                    'score': p.score, 'previous_round_score': p.previous_round_score,
                                    'games_played': p.games_played, 'previous_submitted_words': p.previous_submitted_words,
                                    'found_bonus_word': p.found_bonus_word, 'last_active': p.last_active,
                                    'input_method': p.input_method, 'country_flag': p.country_flag,
                                    'joined_mid_round': p.joined_mid_round, 'has_exceptional_round': p.has_exceptional_round,
                                    'is_guest': p.is_guest, 'is_ai': p.is_ai, 'ai_rating': p.ai_rating, 'has_abandoned': p.has_abandoned,
                                    'game_room_chat': getattr(p, 'game_room_chat', True)
                                })
                            players_json = json.dumps(players_data)
                            with get_db() as conn:
                                conn.execute('''
                                    INSERT OR REPLACE INTO active_boards (
                                        room_id, board_data, all_words, dictionary, min_length, updated_at,
                                        bonus_word, bonus_cell_json, board_format, uniqueness, word_count_range,
                                        active_players_json
                                    )
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    room.room_id, json.dumps(room.board), json.dumps(list(room.all_words)),
                                    room.current_dictionary, room.current_min_length, time.time(),
                                    room.bonus_word or '', json.dumps(room.bonus_cell) if room.bonus_cell else None,
                                    room.current_board_format or 'Normal', room.current_uniqueness or 0.0,
                                    room.current_word_count_range or ('300-400' if room.time_limit >= 7200 else '100-200'),
                                    players_json
                                ))
                        except Exception as db_err:
                            print(f"[RoomManager] Error saving board to DB async: {db_err}")

                    threading.Thread(target=save_board_db_async, daemon=True).start()
                
                # FINAL VALIDATION: If the board is STILL empty, we cannot start the round.
                # Revert to a 10-second emergency intermission to try again.
                if not room.board or len(room.board) == 0:
                    print(f"[RoomManager] CRITICAL: Room {room_id} failed to secure a board. Reverting to emergency intermission.")
                    room.state = 'intermission'
                    room.intermission_start_time = time.time() - 50 # 10s remaining
                    room.starting_round = False
                    return False
                
                # Double-check: If density data is missing in staging but format says 'Density', regenerate it now
                f_low_promo = str(room.current_board_format).lower()
                if 'density' in f_low_promo and (not room.cell_density or len(room.cell_density) == 0):
                    print(f"[Density-Diag] Staging density missing for {room_id} (Format: {f_low_promo}). Re-calculating mid-promotion.")
                    room.initialize_density(room.board, room.all_words_paths, f_low_promo)
                
                # Clear staging data immediately to prevent stale exclusion or duplicate promotion
                room.next_round_bonus = None
                
                # Clear staging data immediately to prevent stale exclusion or duplicate promotion
                room.next_round_bonus = None
                
                # --- 4. DRACONIAN STAGING CLEANUP ---
                # EXPLICITLY nullify all next_round attributes to ensure NO stale data bleeds into the future.
                # If the next searcher is slow, we want fresh/empty counts, not previous ones.
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
                room.board_search_started = False
                room.board_search_loading = False
                room.spinner_params_generated = False
                room.spinner_params_revealed = False
                room.was_revealed_this_intermission = False
                room.frozen_revealed_params = copy.deepcopy(room.spinner_params) if isinstance(room.spinner_params, dict) else None
                room.spinner_params_loading = False
                room.next_spinner_params = None
                room.next_round_spinner_params = None
                room.next_round_difficulty = None
                room.next_round_uniqueness = None
                room.board_search_started_actual = False
                room._spinner_params_locked = True   # Keep LOCKED during active round!
                room._initial_board_delivered = True  # ISSUE 6: mark first board has been delivered
                room._fallback_at_50s_done = False
                
                # --- FINAL CLEARANCE & NEXT LOG CHAIN ---
                # Clear staging data immediately to prevent stale exclusion or duplicate promotion
                room.next_round_board = None 
                room.next_round_words = []
                room.next_round_word_paths = {}
                room.next_round_word_scores = {}
                room.next_round_csw_only_words = []
                room.next_round_added_words = []
                room.next_round_bonus = None
                room.next_round_format = None
                room.next_round_difficulty = None
                room.next_round_uniqueness = None
                room.next_round_total_words_count = 0
                room.next_round_counts_by_len = {}
                room.next_round_total_points = 0
                room.next_round_cell_density = None
                room.next_round_initial_cell_density = None
                
                room.custom_end_time = 0
                # PRECISE START: don't flip to active until the intermission clock truly hits 0:00.
                _ist = getattr(room, 'intermission_start_time', 0) or 0
                _ilimit = 5 if room.time_limit >= 7200 else 60
                _target_start = (_ist + _ilimit) if _ist > 0 else time.time()
                _wait = _target_start - time.time()
                if 0 < _wait <= 2.0:
                    time.sleep(_wait)
                room.round_start_time = time.time()
                room.state = 'active'
                if room.time_limit < 7200:
                    room.round_starting_player_ids = {str(p.user_id) for p in room.players}
                if room.time_limit >= 7200:
                    if not hasattr(room, 'evicted_users') or room.evicted_users is None:
                        room.evicted_users = {}
                    for p in (room.players or []):
                        room.evicted_users[str(p.user_id)] = 'daily_reset'
                    for s in (room.spectators or []):
                        room.evicted_users[str(s.user_id)] = 'daily_reset'
                    room.players = []
                    room.past_players = {}
                    room.spectators = []
                    self.load_daily_score_sums(room)
                    print(f"[RoomManager] Reset 24h room {room_id} players for fresh new daily round and marked evicted_users for daily_reset notice.")
                room.midnight_reset_occurred = False
                if hasattr(room, 'intermission_stuck_start_time'):
                    delattr(room, 'intermission_stuck_start_time')
                if hasattr(room, 'intermission_stuck_time'):
                    delattr(room, 'intermission_stuck_time')

                room.custom_end_time = 0

                # IMPORTANT: CLEAR STARTING LOCK — do this right before exiting
                # so submit_word can proceed the instant the lock is released
                room._transition_spinner_launched = False
                room.starting_round = False

                print(f"[TRANSITION] Room {room_id}: INTERMISSION -> ACTIVE (Round {room.current_round}, Time: {room.round_start_time})")

            # Lock released — submit_word can now proceed immediately
            # AI turns and pre-generation run outside the lock
            room.generate_ai_turns()
            if getattr(room, 'game_type', None) == 'subanagrams':
                def subanagrams_instant_pregen():
                    try:
                        from spinner_set import SpinnerSet
                        from subanagrams_generator import generate_subanagrams_board_and_words
                        from scoring import calculate_word_score
                        next_p = SpinnerSet.generate_subanagrams_params(previous_params=room.spinner_params)
                        res = generate_subanagrams_board_and_words(next_p)
                        if res:
                            nb, nw, nbw, nparams = res
                            with room._state_lock:
                                room.next_round_board = nb
                                room.next_round_words = nw
                                room.next_round_word_paths = {w: [] for w in nw}
                                room.next_round_bonus = nbw
                                room.next_round_bonus_cell = None
                                room.next_round_format = nparams.get('board_format', 'Word Guaranteed')
                                room.next_round_spinner_params = nparams
                                room.next_spinner_params = nparams
                                room.next_round_min_length = nparams.get('min_word_length', 3)
                                room.next_round_total_words_count = len(nw)
                                room.next_round_word_scores = {
                                    w: calculate_word_score(w, nbw, board_format='Normal', return_details=True)
                                    for w in nw
                                }
                                room.spinner_params_generated = True
                                room.board_search_started = True
                                room.board_search_loading = False
                                room.board_search_started_actual = False
                                print(f"[RoomManager] Subanagrams next round staged immediately: {nparams.get('sequence_length')}L | {nparams.get('dictionary')} | {nparams.get('word_count_range')} | {nparams.get('board_format')}")
                    except Exception as e:
                        print(f"[RoomManager] Error pre-generating subanagrams next round: {e}")
                threading.Thread(target=subanagrams_instant_pregen, daemon=True).start()
            else:
                threading.Thread(target=self.pre_generate_next_round, args=(room_id,), daemon=True).start()

            print(f"[RoomManager] SUCCESS: Transitioned room {room_id} to Round {room.current_round}")
            
            # --- ASYNCHRONOUS POST-TRANSITION TASKS ---
            def finalize_results():
                # This is offloaded to avoid blocking the main server thread
                try:
                    if getattr(room, 'game_type', None) != 'subanagrams' and not getattr(room, 'is_solo', False):
                        # Save history to DB
                        self.save_round_history(
                            room, 
                            board=ghost_prev_board, 
                            all_words=ghost_source_words, 
                            bonus_word=ghost_bonus, 
                            player_snapshots=ghost_player_snapshots,
                            round_num=ghost_round_num,
                            all_words_paths=ghost_all_words_paths,
                            round_start_time=ghost_round_start_time,
                            board_format=ghost_board_format
                        )
                        
                        # USER REQUEST: Word Tally logging (CSW words only)
                        self.log_word_tally(room, ghost_player_words, round_num=ghost_round_num, round_start_time=ghost_round_start_time)
                    
                    # Update moderator-only boards or tournament stats if needed
                    # (Standard rooms just move on)
                except Exception as post_err:
                    print(f"[RoomManager] Event Error for {room_id}: {post_err}")

            threading.Thread(target=finalize_results, daemon=True).start()
            return True

        except Exception as transition_err:
            print(f"[RoomManager] CRITICAL ERROR during start_next_round for room {room_id}: {transition_err}")
            import traceback
            traceback.print_exc()
            try:
                with open(DEBUG_FLOW_PATH, 'a') as f:
                    f.write(f"[CRITICAL] start_next_round failed for {room_id}: {transition_err}\n{traceback.format_exc()}\n")
            except Exception as log_err:
                print(f"[RoomManager] Failed to write to debug_flow.log: {log_err}")
            # LOOP GUARD: the new board is already live. Finish the transition so the heartbeat does not
            # re-run start_next_round (re-rolling the board / re-evicting players) on every tick.
            try:
                with room._state_lock:
                    if getattr(room, '_board_promoted_this_transition', False) and room.state != 'active' and room.board:
                        print(f"[RoomManager] Recovering {room_id}: board promoted but transition errored. Forcing ACTIVE to stop retry loop.")
                        room.custom_end_time = 0
                        room.round_start_time = time.time()
                        room.state = 'active'
                        room.midnight_reset_occurred = False
                        room._transition_spinner_launched = False
            except Exception as rec_err:
                print(f"[RoomManager] Recovery after failed transition also failed for {room_id}: {rec_err}")
            return False

        finally:
            # ABSOLUTE SAFETY: The flag MUST be cleared so the room can try again if we failed.
            with room._state_lock:
                 room.starting_round = False
    
    def force_instant_board(self, room_id):
        """Force an instant pregenerated board delivery for a room to break any stall or loading hang (<1ms)."""
        import random, time
        room = self.get_room(room_id)
        if not room:
            return None
        with room._state_lock:
            from board_generator import pop_any_cached_board, pop_compatible_cached_board
            from word_validator import word_validator
            
            # IMMUTABLE ACTIVE GUARD: Never touch room if already active
            if room.state == 'active':
                return room

            # Handle subanagrams mode
            if room.game_type == 'subanagrams':
                from subanagrams_generator import generate_subanagrams_board_and_words
                sp = SpinnerSet.generate_subanagrams_params()
                res = generate_subanagrams_board_and_words(sp)
                if res:
                    b_board, b_words, b_bonus, fparams = res
                    room.board = b_board
                    room.sequence_length = fparams.get('sequence_length', len(b_board[0]))
                    room.all_words = set(b_words)
                    room.all_words_paths = {w: [] for w in b_words}
                    room.bonus_word = b_bonus
                    seq_len_actual = len(b_board[0]) if (b_board and len(b_board) > 0 and isinstance(b_board[0], list)) else len(b_board)
                    has_full = any(len(w) == seq_len_actual for w in b_words)
                    final_fmt = 'Word Guaranteed' if (has_full and fparams.get('mode') == 'word') else 'Totally Random'
                    room.current_board_format = final_fmt
                    fparams['board_format'] = final_fmt
                    fparams['mode'] = 'word' if final_fmt == 'Word Guaranteed' else 'random'
                    room.current_min_length = fparams.get('min_word_length', 3)
                    room.current_dictionary = fparams.get('dictionary', 'CSW')
                    room.use_added_words = fparams.get('use_added_words', False)
                    room.current_word_count_range = fparams.get('word_count_range', '100-200')
                    room.spinner_params = dict(fparams)
                    room.frozen_revealed_params = dict(fparams)
                    room.spinner_params_generated = True
                    from scoring import calculate_word_score
                    room.solved_words_with_scores = {
                        w: calculate_word_score(w, b_bonus, board_format='Normal', return_details=True)
                        for w in b_words
                    }
                    room.total_words_count = len(b_words)
                    room.initial_total_words = len(b_words)
                    room.recalculate_total_points()
                    room.state = 'active'
                    return room

            # Pop from cached board pool or emergency fallback matching room parameters
            sp_target = room.spinner_params or getattr(room, 'frozen_revealed_params', None) or {}
            d_target = sp_target.get('dictionary', 'NWL')
            use_aw_target = sp_target.get('use_added_words', False) or ('+ AW' in str(d_target).upper())
            fmt_target = sp_target.get('board_format', 'Normal')
            m_len_target = sp_target.get('min_word_length', 3)
            wc_target = sp_target.get('word_count_range', '300-400' if use_aw_target else '100-200')
            diff_target = sp_target.get('difficulty', 'Medium')

            is_24h_room = (room.time_limit >= 7200) or ('24h' in str(room.room_id).lower()) or ('86400' in str(room.room_id)) or (str(getattr(room, 'game_type', '')).lower() == '24h')
            if is_24h_room:
                fmt_target = 'Valued Letters'
                wc_target = '300-400'
                diff_target = 'Medium'
                grid_floor = 3
                if '4x6' in str(room.board_dimensions): grid_floor = 4
                elif '5x7' in str(room.board_dimensions): grid_floor = 5
                elif '6x8' in str(room.board_dimensions) or '3x3x3' in str(room.board_dimensions): grid_floor = 6
                m_len_target = grid_floor
                use_aw_target = False

            if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                if room.initial_solo_params.get('difficulty') and str(room.initial_solo_params.get('difficulty')).lower() != 'random':
                    diff_target = room.initial_solo_params['difficulty']
                if room.initial_solo_params.get('board_format') and str(room.initial_solo_params.get('board_format')).lower() != 'random':
                    fmt_target = room.initial_solo_params['board_format']
                if room.initial_solo_params.get('min_word_length') and str(room.initial_solo_params.get('min_word_length')).lower() != 'random':
                    m_len_target = room.initial_solo_params['min_word_length']
                if room.initial_solo_params.get('dictionary') and str(room.initial_solo_params.get('dictionary')).lower() != 'random':
                    d_target = room.initial_solo_params['dictionary']
                if 'use_added_words' in room.initial_solo_params:
                    use_aw_target = room.initial_solo_params['use_added_words']

            pre = pop_compatible_cached_board(
                room.board_dimensions, d_target, fmt_target, m_len_target, use_aw_target, word_count_range=wc_target, difficulty=diff_target
            )
            if not pre:
                pre = pop_any_cached_board(room.board_dimensions, min_word_length=m_len_target, target_format=fmt_target, dictionary=d_target, use_added_words=use_aw_target, word_count_range=wc_target, difficulty=diff_target)
            if not pre:
                pre = get_emergency_fallback_board(
                    room.board_dimensions, fmt_target, room.time_limit,
                    dictionary=d_target, use_added_words=use_aw_target, target_range=wc_target, min_word_length=m_len_target, difficulty=diff_target
                )
            if pre:
                if len(pre) >= 9:
                    r_board, r_words, r_bonus_c, r_fmt, r_dict, r_ratio, r_bonus_word, _, r_params = pre
                else:
                    r_board, r_words, r_bonus_c, r_fmt, r_dict, r_ratio, r_bonus_word, r_params = pre
                
                r_params = dict(r_params) if isinstance(r_params, dict) else {}
                board_min_l = r_params.get('min_word_length', 3)
                grid_floor = 3
                if '4x6' in str(room.board_dimensions): grid_floor = 4
                elif '5x7' in str(room.board_dimensions): grid_floor = 5
                elif '6x8' in str(room.board_dimensions) or '3x3x3' in str(room.board_dimensions): grid_floor = 6
                board_min_l = max(grid_floor, int(board_min_l) if board_min_l is not None else grid_floor)

                raw_dict = d_target or r_params.get('dictionary', 'NWL')
                raw_aw = use_aw_target if use_aw_target is not None else r_params.get('use_added_words', False)
                _val = getattr(word_validator, 'word_validator', word_validator)
                if 'CSW' in str(d_target).upper() or use_aw_target:
                    _bg = getattr(self, 'board_generator', None) or BoardGenerator()
                    _solved = _bg._solve_board(r_board, dictionary=raw_dict, word_count_range=(0, 99999), min_word_length=board_min_l, store_paths=True, timeout=5.0, bonus_cell=r_bonus_c, use_added_words=raw_aw)
                    if _solved:
                        room.all_words_paths = {w: p for w, p in _solved.items() if len(w) >= board_min_l}
                    else:
                        room.all_words_paths = {w: p for w, p in (r_dict or {}).items() if len(w) >= board_min_l and _val.is_valid_word(w, raw_dict, use_added_words=raw_aw)}
                else:
                    room.all_words_paths = {w: p for w, p in (r_dict or {}).items() if len(w) >= board_min_l and _val.is_valid_word(w, raw_dict, use_added_words=raw_aw)}
                room.all_words = set(room.all_words_paths.keys())
                room.complete_words = list(room.all_words)
                if hasattr(word_validator, 'word_validator'):
                    word_validator.word_validator.ensure_csw_loaded()
                    room.csw_only_words = [w for w in room.all_words if word_validator.word_validator.is_csw_only(w)]
                    room.added_words = [w for w in room.all_words if word_validator.word_validator.is_added_word(w)]
                if room.all_words:
                    actual_shortest = min(len(w) for w in room.all_words)
                    if actual_shortest > board_min_l:
                        print(f"[RoomManager] force_instant_board aligned min_word_length from {board_min_l} to factual {actual_shortest}.")
                        board_min_l = actual_shortest
                
                bw_candidate = r_bonus_word
                _fib_bw_l = int(room.spinner_params.get('bonus_word_length', 8) or 8) if isinstance(getattr(room, 'spinner_params', None), dict) else 8
                if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                    _s_fib_bw = room.initial_solo_params.get('bonus_word_length', 'random')
                    if _s_fib_bw and str(_s_fib_bw).lower() not in ['random', '0', 'none', 'null', 'nan']:
                        try:
                            _fib_bw_l = int(_s_fib_bw)
                        except Exception:
                            pass
                _is_chk_fib = 'checkerboard' in str(r_fmt).lower()
                _prev_bw_saved = getattr(room, 'bonus_word', '')
                room.bonus_word = pick_distinct_bonus_word(
                    room, bw_candidate, room.all_words, _fib_bw_l,
                    dictionary=raw_dict, is_checkerboard=_is_chk_fib,
                    room_manager=self, extra_exclude=[_prev_bw_saved, getattr(room, 'previous_bonus_word', None)]
                )
                ensure_bonus_word_on_board(room, _fib_bw_l)
                if _prev_bw_saved:
                    room.previous_bonus_word = _prev_bw_saved
                sync_bonus_word_scores(room)
                final_dict = d_target if d_target else raw_dict
                final_aw = use_aw_target if use_aw_target is not None else raw_aw
                final_fmt = fmt_target if fmt_target else r_fmt
                wc_lbl_fib = SpinnerSet.get_factchecked_wc_range(len(room.all_words), use_added_words=final_aw, dictionary=final_dict)
                room.current_word_count_range = wc_lbl_fib
                if isinstance(getattr(room, 'spinner_params', None), dict):
                    room.spinner_params['bonus_word_length'] = len(room.bonus_word)
                    room.spinner_params['word_count_range'] = wc_lbl_fib
                    room.spinner_params['dictionary'] = final_dict
                    room.spinner_params['use_added_words'] = final_aw
                if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                    room.frozen_revealed_params['bonus_word_length'] = len(room.bonus_word)
                    room.frozen_revealed_params['word_count_range'] = wc_lbl_fib
                    room.frozen_revealed_params['dictionary'] = final_dict
                    room.frozen_revealed_params['use_added_words'] = final_aw
                room.board = r_board
                room.bonus_cell = r_bonus_c
                room.total_words_count = len(room.all_words)
                room.initial_total_words = room.total_words_count
                room.current_board_format = final_fmt
                room.current_dictionary = final_dict
                room.current_min_length = board_min_l
                room.use_added_words = final_aw
                room.current_difficulty = r_params.get('difficulty', diff_target or 'Medium')
                if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                    _s_d_req = room.initial_solo_params.get('difficulty')
                    if _s_d_req and str(_s_d_req).lower() != 'random':
                        room.current_difficulty = SpinnerSet.strip_difficulty_pct(_s_d_req)
                if isinstance(getattr(room, 'spinner_params', None), dict):
                    room.spinner_params['difficulty'] = room.current_difficulty
                room.current_round = max(1, room.current_round)
                room.round_start_time = time.time()
                room.state = 'active'
                if room.time_limit < 7200:
                    room.round_starting_player_ids = {str(p.user_id) for p in room.players}
                room.solving_complete = True
                room.board_search_loading = False
                room.starting_round = False
                room.update_counts_by_len()
                room.recalculate_total_points()
                room.initialize_density(r_board, room.all_words_paths, r_fmt)
                print(f"[force_instant_board] Force-loaded instant board for {room_id} ({len(room.all_words)} words)")
                return room
        return room
    
    def _get_factchecked_wc_range(self, count, use_added_words=False, dictionary=None):
        return SpinnerSet.get_factchecked_wc_range(count, use_added_words=use_added_words, dictionary=dictionary)

    def sanitize_board_for_room_difficulty(self, room, board, words, paths, bonus_cell, bonus_word, board_format, difficulty, dictionary=None, min_len=None):
        """Ironclad safeguard ensuring no Medium/Hard room ever serves an ING sequence."""
        try:
            if not board or not isinstance(board, list):
                return board, words, paths, bonus_cell, bonus_word, board_format
            diff_base = SpinnerSet.strip_difficulty_pct(difficulty or getattr(room, 'current_difficulty', 'Medium'))
            if diff_base in ["Medium", "Hard"]:
                is_3d = len(board) == 6 and isinstance(board[0], list) and isinstance(board[0][0], list)
                depth = 6 if is_3d else 1
                if self.board_generator._has_ing_sequence(board, depth):
                    print(f"[RoomManager] 🛡️ SANITIZE GUARD: Room {getattr(room, 'room_id', '')} (diff={diff_base}) board had 'ING' sequence. Force breaking...")
                    self.board_generator._guarantee_no_ing(board, depth, protected_positions=None)
                    use_aw = bool(getattr(room, 'use_added_words', False))
                    d_target = dictionary or getattr(room, 'current_dictionary', 'NWL')
                    m_target = int(min_len or getattr(room, 'current_min_length', 3) or 3)
                    final_depth = 25 if (not is_3d and len(board)*len(board[0]) <= 16) else 14
                    new_paths = self.board_generator._solve_board(
                        board, d_target, (0, 99999), m_target, max_depth=final_depth, store_paths=True, timeout=15.0, bonus_cell=bonus_cell, use_added_words=use_aw
                    )
                    new_words = sorted(list(new_paths.keys()))
                    if bonus_word and (bonus_word not in new_words or 'ING' in bonus_word.upper() or bonus_word.upper().endswith(('ING', 'INGS'))):
                        bw_cands = [w for w in new_words if len(w) == len(bonus_word) and 'ING' not in w.upper() and not w.upper().endswith(('ING', 'INGS'))]
                        if bw_cands:
                            bonus_word = random.choice(bw_cands).upper()
                    return board, new_words, new_paths, bonus_cell, bonus_word, board_format
        except Exception as err:
            print(f"[RoomManager] Error in sanitize_board_for_room_difficulty: {err}")
        return board, words, paths, bonus_cell, bonus_word, board_format

    def _get_board_fingerprint(self, board):
        """Return a deterministic string fingerprint of a 2D or 3D board for dedup tracking."""
        try:
            if not board:
                return ''
            if isinstance(board[0][0], list):
                # 3D board: flatten all faces
                flat = ''.join(cell for face in board for row in face for cell in row)
            else:
                flat = ''.join(cell for row in board for cell in row)
            return flat
        except Exception:
            return ''

    def _get_bonus_word(self, length=8, dictionary='NWL', alternating=False, difficulty='Medium', exclude=None):
        """Get a bonus word of specified length, optionally enforcing C/V alternating pattern for Checkerboard"""
        import time
        from word_validator import word_validator
        
        # USER MANDATE: Bonus word length MUST be strictly between 6 and 10 letters!
        try:
            length = int(length)
        except (ValueError, TypeError):
            length = 8
        length = max(6, min(10, length))
        
        # Determine if we should exclude ING (Medium/Hard)
        diff_upper = str(difficulty).upper()
        exclude_ing = (diff_upper in ['MEDIUM', 'HARD', 'EXPERT', 'DIFFICULT', 'MASTERS', 'NORMAL'])
        
        # Get all words of the specified length (using cache if available)
        d_upper = str(dictionary).upper()
        if d_upper == 'AW' or d_upper == 'ADDED_WORDS':
            words = [w for w in word_validator.added_words if len(w) == length]
            if not words:
                words = word_validator.nwl_by_len.get(length, [])
                if not words: words = [w for w in word_validator.nwl_words if len(w) == length]
        elif d_upper == 'CSW':
            words = word_validator.csw_by_len.get(length, [])
            if not words: words = [w for w in word_validator.csw_words if len(w) == length]
        else:
            words = word_validator.nwl_by_len.get(length, [])
            if not words: words = [w for w in word_validator.nwl_words if len(w) == length]
        
        # Filter for alternating pattern if requested (MANDATORY for Checkerboard)
        if alternating:
            words = [w for w in words if self.board_generator._is_alternating_word(w)]
            if not words:
                if dictionary == 'CSW': words = [w for w in word_validator.csw_words if len(w) == length]
                else: words = [w for w in word_validator.nwl_words if len(w) == length]

        # USER MANDATE: Never allow a word ending in ING or INGS (or containing ING on Medium/Hard) to be a Bonus Word!
        import random
        if exclude_ing:
            valid_words = [w for w in words if not w.upper().endswith("ING") and not w.upper().endswith("INGS") and "ING" not in w.upper()]
            if not valid_words:
                all_dict_words = word_validator.csw_by_len.get(length, []) if dictionary == 'CSW' else word_validator.nwl_by_len.get(length, [])
                valid_words = [w for w in all_dict_words if not w.upper().endswith("ING") and not w.upper().endswith("INGS") and "ING" not in w.upper()]
        else:
            valid_words = [w for w in words if not w.upper().endswith("ING") and not w.upper().endswith("INGS")]
            if not valid_words:
                all_dict_words = word_validator.csw_by_len.get(length, []) if dictionary == 'CSW' else word_validator.nwl_by_len.get(length, [])
                valid_words = [w for w in all_dict_words if not w.upper().endswith("ING") and not w.upper().endswith("INGS")]
        
        # PREVENT REPEATS: Filter out words in history or currently active in the room
        final_exclude = set()
        with _GLOBAL_BONUS_LOCK:
            for gw in _GLOBAL_RECENT_BONUS_WORDS[-30:]:
                if gw: final_exclude.add(gw)
        
        # Explicitly passed exclusions (e.g. current/staged words or history list)
        if exclude:
            if isinstance(exclude, list):
                for e in exclude:
                    if e: final_exclude.add(str(e).upper())
            else:
                final_exclude.add(str(exclude).upper())
        
        # Pool filtered results
        if valid_words and final_exclude:
            filtered = [w for w in valid_words if w.upper() not in final_exclude]
            if len(filtered) >= 1:
                valid_words = filtered
                
        # Shuffle for maximum randomness instead of simple choice from potentially biased list
        if len(valid_words) > 1:
            random.shuffle(valid_words)
        
        result = random.choice(valid_words).upper() if valid_words else 'A' * length
        if result and not result.startswith('A' * length):
            _record_global_bonus_word(result)
        return result
    
    
    def save_round_history(self, room, board=None, all_words=None, bonus_word=None, player_snapshots=None, round_num=None, all_words_paths=None, round_start_time=None, board_format=None):
        """Save the results of the JUST COMPLETED round to the database"""
        # Determine target round number (use snapshot if provided, otherwise room's current)
        target_round = round_num if round_num is not None else room.current_round
        debug_log = f"[SAVE-ROUND-{room.room_id}-R{target_round}]"

        if room.is_solo or getattr(room, 'game_type', None) == 'subanagrams':
            print(f"[RoomManager] SKIPPING history save for {getattr(room, 'game_type', 'unknown')} room {room.room_id}")
            with open(DEBUG_FLOW_PATH, 'a') as f:
                f.write(f"{debug_log} - ABORT (Solo or Subanagrams)\n")
            return
            
        import json
        
        # Guard against double saving (Exact match check based on room, round, and start timestamp)
        save_key = (room.room_id, target_round, round_start_time or getattr(room, 'round_start_time', 0))
        if getattr(room, '_last_saved_round_key', None) == save_key:
            print(f"[RoomManager] History for {room.room_id} Round {target_round} already saved. Skipping.")
            with open(DEBUG_FLOW_PATH, 'a') as f:
                f.write(f"{debug_log} - ABORT (Already saved)\n")
            return
        
        try:
            with get_db() as conn:
                with open(DEBUG_FLOW_PATH, 'a') as f:
                    f.write(f"{debug_log} - DB CONNECTED\n")
                
                # Use passed-in snapshots if provided (prevents stale data from being saved)
                actual_board = board if board is not None else room.board
                board_json = json.dumps(actual_board)
                
                # Robust Timestamping for 24h rooms in America/Chicago timezone
                from zoneinfo import ZoneInfo
                tz = DAILY_TZ
                now = datetime.datetime.now(tz)
                # If a daily room ended just after midnight, the results belong to "Yesterday"
                if room.time_limit >= 7200 and now.hour == 0 and now.minute < 10:
                    yesterday = now - datetime.timedelta(days=1)
                    timestamp = yesterday.strftime('%Y-%m-%d 23:59:59')
                else:
                    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
                
                board_format = board_format if board_format is not None else room.current_board_format
                wc_range = room.spinner_params.get('word_count_range', (0, 0))
                wc_tuple = room._get_wc_tuple(wc_range)
                is_500plus = wc_tuple[0] >= 500
                
                # Board formats and 500+ unranked boards are allowed for history
                # so that words found are permanently preserved for Find Count.
                if is_500plus:
                    print(f"[RoomManager] Note: Room {room.room_id} Round {target_round} is 500+ (unranked for ratings, but history and word finds preserved).")
                     
                # Determine best data source (Room state might have already advanced to Next Round)
                actual_all_words = all_words if all_words is not None else room.all_words
                actual_bonus_word = bonus_word if bonus_word is not None else getattr(room, 'bonus_word', '')
                actual_all_words_paths = all_words_paths if all_words_paths is not None else getattr(room, 'all_words_paths', {})
                
                # Identify registered players who actually made any attempt
                # Use snapshots if available, otherwise fallback to current room players
                if player_snapshots is not None:
                    participating_registered = player_snapshots
                else:
                    participating_registered = [p for p in room.players if (p.is_registered or p.is_guest) and (p.score > 0 or p.submitted_words or p.invalid_words)]
                
                if not participating_registered:
                    if room.time_limit >= 7200:
                        print(f"[RoomManager] No registered players participated in 24h room {room.room_id}. Creating system placeholder to preserve board & solutions.")
                        participating_registered = [{
                            'user_id': -1,
                            'username': 'System',
                            'score': 0,
                            'submitted_words': [],
                            'rating': 1200,
                            'performance_efficiency': 0.0
                        }]
                    else:
                        print(f"[RoomManager] SKIPPING history save for room {room.room_id} Round {target_round} - no participating registered users.")
                        with open(DEBUG_FLOW_PATH, 'a') as f:
                            p_details = [{'name': (p.username if hasattr(p, 'username') else p.get('username')), 'uid': (p.user_id if hasattr(p, 'user_id') else p.get('user_id')), 'score': (p.score if hasattr(p, 'score') else p.get('score'))} for p in room.players]
                            f.write(f"{debug_log} - ABORT (No registered players). Details: {p_details}\n")
                        return

                print(f"[RoomManager] Saving history for room {room.room_id} Round {target_round} ({len(participating_registered)} players)")

                solutions_saved = False
                for p in participating_registered:
                    # p is either a Player object or a dictionary snapshot
                    u_id = p.user_id if hasattr(p, 'user_id') else p['user_id']
                    u_name = p.username if hasattr(p, 'username') else p['username']
                    u_score = p.score if hasattr(p, 'score') else p['score']
                    u_submitted = p.submitted_words if hasattr(p, 'submitted_words') else p.get('submitted_words', [])

                    # Only skip if user earned 0 points AND submitted no words
                    # (Unless it is the System placeholder for 24-hour rooms)
                    u_mid = getattr(p, 'joined_mid_round', False) if hasattr(p, 'joined_mid_round') else p.get('joined_mid_round', False)
                    if u_id != -1 and u_name != 'System' and u_score <= 0 and not u_submitted:
                        print(f"[RoomManager] Skipping saving round history for {u_name} because score is {u_score} and no words submitted")
                        continue

                    u_submitted = p.submitted_words if hasattr(p, 'submitted_words') else p['submitted_words']
                    u_rating = getattr(p, 'rating', 1200) if hasattr(p, 'rating') else p.get('rating', 1200)
                    u_perf = getattr(p, 'performance_efficiency', 0) if hasattr(p, 'performance_efficiency') else p.get('performance_efficiency', 0)
                    if not u_perf or u_perf <= 0.0:
                        reg_pool = [pl for pl in participating_registered if (getattr(pl, 'score', 0) if hasattr(pl, 'score') else pl.get('score', 0)) > 0]
                        if len(reg_pool) > 1:
                            tot_s = sum((pl.score if hasattr(pl, 'score') else pl.get('score', 0)) for pl in reg_pool)
                            tot_r = sum((getattr(pl, 'rating', 1200) if hasattr(pl, 'rating') else pl.get('rating', 1200)) for pl in reg_pool)
                            if tot_r > 0:
                                exp_s = (u_rating / tot_r) * tot_s
                                u_perf = round(u_score / exp_s, 2) if exp_s > 0 else 1.0
                            else:
                                u_perf = round(u_score / (tot_s / len(reg_pool)), 2) if tot_s > 0 else 1.0
                        else:
                            u_perf = 1.0
                    
                    # NORMALIZE TIMESTAMPS: Ensure numeric s for replay
                    words_data = []
                    actual_start_time = round_start_time if round_start_time is not None else (room.round_start_time or time.time())
                    for w in u_submitted:
                        # Get raw time or fallback
                        raw_time = w.get('time')
                        if not raw_time or isinstance(raw_time, str):
                            raw_time = actual_start_time
                        
                        w_im = w.get('input_method') if isinstance(w, dict) else None
                        if not w_im:
                            w_im = getattr(p, 'input_method', 'mouse') if hasattr(p, 'input_method') else p.get('input_method', 'mouse')

                        words_data.append({
                            'word': w['word'],
                            'points': w.get('points', 0),
                            'timestamp': raw_time,
                            'input_method': w_im or 'mouse'
                        })
                    
                    # Calculate Best Word (Using Base Points Only - strictly excluding bonus word extra points)
                    # Defense-in-depth: only consider valid words with positive score and present on the board
                    def _get_u_base_score(item):
                        if not item: return 0
                        w_txt = item.get('word', '') if isinstance(item, dict) else str(item)
                        return calculate_word_score(w_txt, bonus_word=None, board_format=board_format)

                    valid_submitted = [
                        w for w in u_submitted
                        if (w.get('points', 0) if isinstance(w, dict) else 0) > 0
                        and (not actual_all_words or (w.get('word', '') if isinstance(w, dict) else str(w)).upper() in actual_all_words)
                    ]
                    best_w_entry = max(valid_submitted, key=lambda x: (_get_u_base_score(x), len(x.get('word', '')))) if valid_submitted else None
                    best_word_text = best_w_entry['word'] if best_w_entry else None
                    best_word_val = _get_u_base_score(best_w_entry) if best_w_entry else 0

                    # Calculate WPM (Words Per Minute)
                    final_wpm = 0.0
                    if len(words_data) >= 5:
                        sorted_entries = sorted(words_data, key=lambda x: x['timestamp'])
                        if len(sorted_entries) >= 20:
                            peak_wpm = 0.0
                            for i in range(len(sorted_entries) - 19):
                                t_first = sorted_entries[i]['timestamp']
                                t_last = sorted_entries[i+19]['timestamp']
                                dt = t_last - t_first
                                if dt > 0.001:
                                    current_burst_wpm = (20.0 * 60.0) / dt
                                    peak_wpm = max(peak_wpm, current_burst_wpm)
                            final_wpm = peak_wpm
                        else:
                            t_first = sorted_entries[0]['timestamp']
                            t_last = sorted_entries[-1]['timestamp']
                            dt = t_last - t_first
                            if dt > 0.001:
                                final_wpm = (len(sorted_entries) * 60.0) / dt
                    
                    # 2. SAVE: Optimization - Only store full solutions/paths for the FIRST player saved in the batch
                    is_first_saved = not solutions_saved
                    solutions_payload = json.dumps(list(actual_all_words)) if is_first_saved else None
                    paths_payload = json.dumps(actual_all_words_paths) if is_first_saved else None 
                    if is_first_saved:
                        solutions_saved = True

                    cur_rh = conn.execute('''
                        INSERT INTO round_history (user_id, room_id, game_type, round_number, board_json, words_json, total_score, round_start_time, round_duration, timestamp, user_rating, performance_ratio, best_word, best_word_score, board_dimensions, wpm, total_words_avail, bonus_word, bonus_cell, board_format, all_solutions_json, all_words_paths)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        u_id, room.room_id, room.game_type, target_round, board_json, 
                        json.dumps(words_data), u_score, actual_start_time, room.time_limit, 
                        timestamp, u_rating, u_perf, best_word_text, best_word_val,
                        room.board_dimensions, final_wpm, len(actual_all_words), 
                        actual_bonus_word, json.dumps(room.bonus_cell), board_format,
                        solutions_payload, paths_payload
                    ))
                    rh_id = cur_rh.lastrowid

                    # Also persist player word finds into lightweight word_finds table for Find Count
                    # Find Count rules: NEVER record Solo rooms, 24h rooms, or Guest finds, and strictly ignore invalid/penalty guesses
                    is_solo_rh = getattr(room, 'is_solo', False) or ('solo' in str(room.room_id).lower()) or str(getattr(room, 'game_type', '')).startswith('solo')
                    is_24h_rh = (getattr(room, 'time_limit', 0) >= 7200) or ('24h' in str(room.room_id).lower()) or ('86400' in str(room.room_id)) or (getattr(room, 'game_type', '') == '24h')
                    if u_id != -1 and u_name != 'System' and not is_solo_rh and not is_24h_rh and not str(u_name).startswith('Guest_'):
                        try:
                            for entry in words_data:
                                pts = entry.get('points', 0) if isinstance(entry, dict) else 0
                                if pts <= 0:
                                    continue
                                w_text = entry.get('word', '').upper() if isinstance(entry, dict) else str(entry).upper()
                                if w_text:
                                    if actual_all_words and w_text not in actual_all_words:
                                        continue
                                    conn.execute('''
                                        INSERT INTO word_finds (word, user_id, timestamp, room_id, round_number, round_history_id)
                                        VALUES (?, ?, ?, ?, ?, ?)
                                    ''', (w_text, u_id, timestamp, room.room_id, target_round, rh_id))
                        except Exception:
                            pass

                    # Increment user input method statistics based on words submitted in this round (valid words only)
                    if u_id > 0 and u_name != 'System' and not str(u_name).startswith('Guest_'):
                        m_cnt = 0
                        k_cnt = 0
                        t_cnt = 0
                        player_im = getattr(p, 'input_method', 'mouse') if hasattr(p, 'input_method') else p.get('input_method', 'mouse')
                        for entry in words_data:
                            pts = entry.get('points', 0) if isinstance(entry, dict) else 0
                            if pts <= 0:
                                continue
                            im = (entry.get('input_method') if isinstance(entry, dict) else None) or player_im
                            im_str = str(im).lower()
                            if im_str == 'keyboard':
                                k_cnt += 1
                            elif im_str == 'touch':
                                t_cnt += 1
                            else:
                                m_cnt += 1

                        if m_cnt > 0 or k_cnt > 0 or t_cnt > 0:
                            try:
                                conn.execute('''
                                    UPDATE users 
                                    SET input_mouse_count = COALESCE(input_mouse_count, 0) + ?,
                                        input_keyboard_count = COALESCE(input_keyboard_count, 0) + ?,
                                        input_touch_count = COALESCE(input_touch_count, 0) + ?
                                    WHERE id = ?
                                ''', (m_cnt, k_cnt, t_cnt, u_id))
                            except Exception as _im_err:
                                print(f"[save_round_history] Error updating input method counts: {_im_err}")

                    if room.time_limit >= 7200 and u_id != -1 and u_name != 'System' and u_score > 0:
                        canonical_id = f"24h_{room.board_dimensions}"
                        conn.execute('''
                            INSERT INTO daily_score_sums (user_id, room_id, score_sum)
                            VALUES (?, ?, ?)
                            ON CONFLICT(user_id, room_id) DO UPDATE SET score_sum = score_sum + excluded.score_sum
                        ''', (u_id, canonical_id, u_score))
            
            if room.time_limit >= 7200:
                try:
                    self.load_daily_score_sums(room)
                except Exception as _ls_err:
                    print(f"[save_round_history] Error reloading daily_score_sums: {_ls_err}")
            
            # Also mark this board's hash as permanently used (survives PM2 restarts)
            try:
                from board_generator import get_board_hash, mark_board_hash_used
                _bh = get_board_hash(actual_board)
                if _bh:
                    mark_board_hash_used(_bh)
            except Exception as _bh_err:
                print(f"[RoomManager] Non-fatal: Could not mark board hash after save_round_history: {_bh_err}")
            
            room._last_saved_round_key = save_key
            room.last_saved_round = target_round
            print(f"[RoomManager] SUCCESS: Saved round history for room {room.room_id} Round {target_round}")

            try:
                import app
                if hasattr(app, '_WORD_FINDS_CACHE'):
                    app._WORD_FINDS_CACHE.clear()
            except Exception:
                pass

            with open(DEBUG_FLOW_PATH, 'a') as f:
                f.write(f"{debug_log} - SUCCESS: Saved to DB\n")
        except Exception as e:
            print(f"[RoomManager] Error saving round history: {e}")
            with open(DEBUG_FLOW_PATH, 'a') as f:
                f.write(f"{debug_log} - FATAL ERROR: {e}\n")

    def log_word_tally(self, room, player_words, round_num=None, round_start_time=None):
        """
        Tally how many users found each CSW word and log it to a central file.
        Also maintains a global cumulative tally in word_stats.json.
        """
        try:
            # USER REQUEST: Do not include words found in 24h rooms, Solo rooms, or Subanagrams in the word tally file
            is_solo_tally = getattr(room, 'is_solo', False) or ('solo' in str(getattr(room, 'room_id', '')).lower()) or str(getattr(room, 'game_type', '')).startswith('solo')
            is_24h_tally = (getattr(room, 'time_limit', 0) >= 7200) or ('24h' in str(getattr(room, 'room_id', '')).lower()) or ('86400' in str(getattr(room, 'room_id', ''))) or (getattr(room, 'game_type', None) in ('subanagrams', '24h'))
            if is_solo_tally or is_24h_tally:
                print(f"[WordTally] Skipping tally for {getattr(room, 'game_type', 'unknown')} room: {getattr(room, 'room_id', '')}")
                return

            target_round = round_num if round_num is not None else getattr(room, 'current_round', 0)
            target_start = round_start_time or getattr(room, 'round_start_time', 0)
            tally_key = (getattr(room, 'room_id', ''), target_round, target_start)
            if getattr(room, '_last_tallied_round_key', None) == tally_key:
                print(f"[WordTally] Room {getattr(room, 'room_id', '')} Round {target_round} already tallied. Skipping duplicate.")
                return
            room._last_tallied_round_key = tally_key

            import collections
            import json
            import datetime
            import os
            import word_validator
            
            # 1. Efficient Tally: Number of unique USERS who found each word in THIS round
            word_counts = collections.Counter()
            user_valid_words = {}
            for _uname, words in player_words.items():
                if str(_uname).startswith('Guest_'):
                    continue  # Guest finds are never counted in Find Count
                unique_words_for_user = set()
                for entry in words:
                    w = entry.get('word', '').upper() if isinstance(entry, dict) else str(entry).upper()
                    active_dict = getattr(room, 'current_dictionary', 'NWL')
                    if w and word_validator.is_valid_word(w, active_dict, use_added_words=getattr(room, 'use_added_words', False)):
                        unique_words_for_user.add(w)
                
                if unique_words_for_user:
                    user_valid_words[_uname] = unique_words_for_user
                for w in unique_words_for_user:
                    word_counts[w] += 1
            
            if not word_counts:
                return

            # Record per-user tallies in user_word_tallies table
            if user_valid_words:
                try:
                    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    with get_db() as conn:
                        for uname, uwords in user_valid_words.items():
                            c = conn.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (uname,))
                            row = c.fetchone()
                            if row:
                                uid = row['id'] if isinstance(row, dict) or hasattr(row, 'keys') else row[0]
                                for w in uwords:
                                    conn.execute("""
                                        INSERT INTO user_word_tallies (user_id, word, find_count, last_found_at)
                                        VALUES (?, ?, 1, ?)
                                        ON CONFLICT(user_id, word) DO UPDATE SET
                                            find_count = find_count + 1,
                                            last_found_at = excluded.last_found_at
                                    """, (uid, w, now_iso))
                        conn.commit()
                except Exception as uwt_err:
                    print(f"[WordTally] Error saving user_word_tallies: {uwt_err}")

            # 2. Append per-round entry to audit log
            log_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'room_id': room.room_id,
                'round': target_round,
                'tally': dict(word_counts)
            }
            
            log_path = WORD_TALLY_PATH
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
            # 3. Update GLOBAL Cumulative Tally (word_stats.json)
            # Use File-based Lock (fcntl) to prevent cross-process race conditions in Gunicorn
            try:
                stats_file = open(STATS_PATH, 'r+')
                fcntl.flock(stats_file, fcntl.LOCK_EX) # Exclusive Lock
                try:
                    global_stats = json.load(stats_file)
                except:
                    global_stats = {}
                
                # Merge new counts into global totals
                for word, count in word_counts.items():
                    if word in ["STAR", "MICE", "ARITIES"]:
                        print(f"[WordTally-Diag] Writing target word '{word}' (Count: {count}) to global stats.")
                    global_stats[word] = global_stats.get(word, 0) + count
                
                # Write back the updated totals
                stats_file.seek(0)
                stats_file.truncate()
                json.dump(global_stats, stats_file)
                stats_file.flush()
                os.fsync(stats_file.fileno()) # Force write to disk
                fcntl.flock(stats_file, fcntl.LOCK_UN) # Release
                stats_file.close()
                
                # USER REQUEST: Track in trace log as well
                with open(TRACE_PATH, 'a') as trace:
                    trace.write(f"[{datetime.datetime.now()}] ROUND_SYNC: Room {room.room_id} added {len(word_counts)} unique words\n")
                
                print(f"[WordTally] SUCCESS: Updated stats for room {room.room_id} (Words: {len(word_counts)})")
            except Exception as stats_err:
                print(f"[WordTally] File Lock Error: {stats_err}")
                
        except Exception as e:
            print(f"[WordTally] Error logging word tally for {room.room_id}: {e}")

    def start_complete_solving(self, room_id):
        """
        Mark solving as complete immediately - words already found during generation.
        """
        room = self.get_room(room_id)
        if not room:
            return
        
        print(f"[RoomManager] Words already found, marking as complete")
        room.complete_words = list(room.all_words)
        room.solving_complete = True



    def propose_board(self, room_id, proposed_board, username):
        """
        [DBG] Allows a client to submit a potentially target-meeting board for verification.
        If the board meets the current next_round_intent criteria, it is promoted immediately.
        """
        room = self.get_room(room_id)
        if not room: return {"error": "Room not found", "success": False}
        
        # 1. State Guard: Only accept if active search is in progress
        if not getattr(room, 'board_search_started', False) or getattr(room, 'solving_complete', False):
            return {"error": "Search not active", "success": False}
            
        # 2. Rate Limiting: Prevent Spam (1 proposal per 3s per player)
        now = time.time()
        last_prop = getattr(room, '_last_proposal_times', {})
        if username in last_prop and now - last_prop[username] < 3:
            return {"error": "Rate limited", "success": False}
        last_prop[username] = now
        room._last_proposal_times = last_prop
        
        target = getattr(room, 'next_spinner_params', {})
        if not target: return {"error": "No target params", "success": False}
        
        try:
            # 3. Authoritative Verification (Server-side Solve)
            # We wrap this in a timeout to avoid a client sending a "poison" board
            print(f"[DBG] Verifying board proposed by {username} for room {room_id}...")
            
            b_dims = room.board_dimensions.split('x')
            r_num = int(b_dims[1] if len(b_dims) == 3 else b_dims[0])
            c_num = int(b_dims[2] if len(b_dims) == 3 else b_dims[1])
            
            # Validate proposed dimensions
            if len(proposed_board) != r_num or len(proposed_board[0]) != c_num:
                 return {"error": "Dimension mismatch", "success": False}
                 
            # 4. Enforce Ironclad Sanitization and Abundance Rules for Client Proposals
            board_format = target.get('board_format', 'Normal')
            safe_format = str(board_format or "Normal").strip().upper()
            is_mania = "MANIA" in safe_format
            mania_letter = None
            if is_mania:
                parts = safe_format.split()
                if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isalpha():
                    mania_letter = parts[0]

            total_cells = r_num * c_num
            rare_letters = {"Q", "Z", "J", "X", "K"}
            rare_counts = {rl: 0 for rl in rare_letters}
            total_rares = 0
            
            letter_counts = {}
            for row in proposed_board:
                for cell in row:
                    cell_str = str(cell).upper()
                    # Handle Either/Or cell slash formats
                    parts = cell_str.split('/')
                    for part in parts:
                        if part:
                            letter_counts[part] = letter_counts.get(part, 0) + 1
                            if part in rare_letters:
                                rare_counts[part] += 1
                                if not (is_mania and part == mania_letter):
                                    total_rares += 1
                                
            # Check rare limits (max 1 of each, max 3 total) - Mania letter is exempt
            for rl, rc in rare_counts.items():
                if is_mania and rl == mania_letter:
                    continue
                if rc > 1:
                    return {"error": f"Too many rare letters: '{rl}' occurs {rc} times (Max 1)", "success": False}
            if total_rares > 3:
                return {"error": f"Too many total rare letters: {total_rares} (Max 3)", "success": False}

            # If Mania format, enforce exact required abundance (20% uncommon, 33% common)
            if is_mania and mania_letter:
                UNCOMMON_MANIA_LETTERS = {'Q', 'Z', 'J', 'X', 'K'}
                target_ratio = 0.20 if mania_letter in UNCOMMON_MANIA_LETTERS else 0.33
                target_m_cnt = max(2, round(total_cells * target_ratio))
                actual_m_cnt = letter_counts.get(mania_letter, 0)
                if actual_m_cnt < target_m_cnt:
                    return {"error": f"Mania letter '{mania_letter}' occurs only {actual_m_cnt} times (needed {target_m_cnt})", "success": False}
                
            # Check standard abundance limits on non-Mania letters
            VOWELS = {"A", "E", "I", "O", "U"}
            COMMON_CONSONANTS = {"S", "T", "R", "N", "L", "D"}
            
            for char, count in letter_counts.items():
                if is_mania and char == mania_letter:
                    continue  # Mania letter has no limit
                    
                if char in VOWELS:
                    limit = max(4, int(total_cells * 0.18))
                elif char in COMMON_CONSONANTS:
                    limit = max(3, int(total_cells * 0.12))
                else:
                    limit = max(2, int(total_cells * 0.09))
                    
                if count > limit:
                    return {"error": f"Letter '{char}' exceeded abundance cap ({count}/{limit})", "success": False}
                 
            # Extract criteria
            dict_name = target.get('dictionary', 'NWL')
            wc_range = room._get_wc_tuple(target.get('word_count_range', '100-200'))
            min_len = target.get('min_word_length', 3)
            bonus_word = target.get('bonus_word', '')
            
            # AUTHORITATIVE SOLVE
            # Use a slightly shallower depth (12) for fast verification
            all_words_dict = self.board_generator._solve_board(
                proposed_board, dict_name, (0, 9999), min_len, max_depth=12, store_paths=True
            )
            
            all_words = list(all_words_dict.keys())
            total_count = len(all_words)
            
            d_num = int(b_dims[0]) if len(b_dims) == 3 else 1
            u_ratio = self.board_generator.get_uniqueness_ratio(
                proposed_board, all_words, r_num, c_num, dict_name, depth=d_num
            )
            
            # Check Compliance
            is_compliant = wc_range[0] <= total_count <= wc_range[1]
            # Must contain bonus word if one exists
            if bonus_word and bonus_word.upper() not in all_words_dict:
                 is_compliant = False
                 
            if is_compliant:
                with room._state_lock:
                    # Double-check search hasn't finished already
                    if getattr(room, 'solving_complete', False):
                        return {"error": "Search already finished", "success": False}
                        
                    print(f"[DBG] SUCCESS! Prosed board by {username} is compliant ({total_count} words). Promoting.")
                    
                    d_num = int(b_dims[0]) if len(b_dims) == 3 else 1
                    r_num = int(b_dims[1] if len(b_dims) == 3 else b_dims[0])
                    c_num = int(b_dims[2] if len(b_dims) == 3 else b_dims[1])
                    
                    # Score and Uniqueness
                    scored_dict = self.board_generator.scoring.score_words(all_words, dict_name)
                    target_min_len = target.get('min_word_length', 3)
                    target_diff = target.get('difficulty', 'Medium')
                    achieved_diff = self.board_generator.get_difficulty_label(u_ratio, r_num, c_num, dict_name, depth=d_num, board=proposed_board, target_difficulty=target_diff, min_word_length=target_min_len)
                    
                    # PROMOTE DATA
                    room.next_round_board = proposed_board
                    room.next_round_words = all_words
                    room.next_round_word_paths = all_words_dict
                    room.next_round_word_scores = scored_dict
                    room.next_round_uniqueness = u_ratio
                    room.next_round_difficulty = achieved_diff
                    room.next_round_format = board_format
                    room.next_round_spinner_params = dict(target)
                    room.next_round_spinner_params['board_format'] = board_format
                    room.next_round_bonus = bonus_word
                    if bonus_word and bonus_word.upper() in all_words_dict:
                        room.next_round_bonus_cell = all_words_dict[bonus_word.upper()][0]
                    room.solving_complete = True # STOPS SERVER SEARCH
                    
                    # Sync UI
                    # Never mutate active/declared params mid-round or after 0:45 reveal.
                    room.next_round_difficulty = SpinnerSet.format_difficulty_with_pct(achieved_diff, u_ratio)
                    
                    self.trigger_room_update(room_id)
                    return {"success": True, "words_found": total_count, "uniqueness": u_ratio}
            else:
                return {"success": False, "error": f"Board not compliant ({total_count} words found)"}
                
        except Exception as e:
            print(f"[DBG] Error verifying proposal: {e}")
            return {"success": False, "error": str(e)}
    def trigger_room_update(self, room_id):
        # Implementation of global room update trigger if needed (e.g. for SocketIO or cache busting)
        pass # Placeholder for existing mechanism

# Global instances
room_manager = RoomManager()

from collections import deque

class LobbyManager:
    def __init__(self, room_mgr):
        self.room_manager = room_mgr
        self.lock = threading.RLock()
        self.lobby_users = {} # { user_id_str: { 'user_id': uid, 'username': uname, 'rating': r, 'avatar_url': url, 'last_seen': float } }
        self.chat_messages = deque(maxlen=100)
        self._msg_id_seq = 0

    def is_user_in_any_room(self, user_id):
        if not user_id:
            return False
        uid_str = str(user_id)
        now = time.time()
        for r in list(self.room_manager.rooms.values()):
            r_state = getattr(r, 'state', '')
            if r_state not in ('playing', 'intermission', 'countdown'):
                continue
            is_daily = (getattr(r, 'time_limit', 0) >= 7200)
            for p in getattr(r, 'players', []):
                if getattr(p, 'is_ai', False):
                    continue
                if str(getattr(p, 'user_id', '')) == uid_str:
                    timeout = 60 if is_daily else 25
                    if (now - getattr(p, 'last_active', 0)) < timeout:
                        return True
            for s in getattr(r, 'spectators', []):
                if str(getattr(s, 'user_id', '')) == uid_str:
                    if (now - getattr(s, 'last_active', 0)) < 25:
                        return True
        return False

    def update_presence(self, user_id, username, rating=1200, avatar_url=None, country_flag=None):
        if not user_id or not username:
            return
        uid_str = str(user_id)
        if self.is_user_in_any_room(user_id):
            with self.lock:
                self.lobby_users.pop(uid_str, None)
            return

        with self.lock:
            self.lobby_users[uid_str] = {
                'user_id': user_id,
                'username': username,
                'rating': rating if rating is not None else 1200,
                'avatar_url': avatar_url,
                'country_flag': country_flag or '🏳️',
                'last_seen': time.time()
            }

    def remove_user(self, user_id):
        if not user_id:
            return
        with self.lock:
            self.lobby_users.pop(str(user_id), None)
            for k, data in list(self.lobby_users.items()):
                if str(data.get('user_id')) == str(user_id) or str(data.get('username')) == str(user_id):
                    self.lobby_users.pop(k, None)

    def get_lobby_state(self):
        now = time.time()
        with self.lock:
            active_users = []
            stale_keys = []
            for uid_str, data in list(self.lobby_users.items()):
                if (now - data['last_seen']) > 15 or self.is_user_in_any_room(data['user_id']):
                    stale_keys.append(uid_str)
                else:
                    active_users.append({
                        'user_id': data['user_id'],
                        'username': data['username'],
                        'rating': data['rating'],
                        'avatar_url': data.get('avatar_url'),
                        'country_flag': data.get('country_flag', '🏳️')
                    })
            for k in stale_keys:
                self.lobby_users.pop(k, None)

            active_users.sort(key=lambda u: u['username'].lower())
            messages_list = list(self.chat_messages)

            return {
                'players': active_users,
                'count': len(active_users),
                'messages': messages_list
            }

    def add_message(self, user_id, username, rating, message, country_flag=None):
        if not message or not username:
            return None
        with self.lock:
            self._msg_id_seq += 1
            now = time.time()
            import datetime
            iso_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            msg_entry = {
                'id': self._msg_id_seq,
                'user_id': user_id,
                'username': username,
                'rating': rating if rating is not None else 1200,
                'country_flag': country_flag or '🏳️',
                'message': message.strip()[:1000],
                'time': now,
                'timestamp': iso_ts
            }
            self.chat_messages.append(msg_entry)
            return msg_entry

lobby_manager = LobbyManager(room_manager)
