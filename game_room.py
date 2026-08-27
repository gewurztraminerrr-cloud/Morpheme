"""
Game Room Management for Multiplayer Boggle
Handles room state, players, timers, and game logic
"""

import time
import random
import datetime
import threading
from dataclasses import dataclass, field
from typing import List, Dict
import os
import fcntl

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
        
        if self.time_limit >= 7200:
            self.current_word_count_range = '200-300'
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
            
    def add_chat_message(self, username, message, is_system=False, image=None, color=None, is_winner=False, rating=None):
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

        self.chat_messages.append({
            'username': username,
            'message': message,
            'image': image,
            'is_system': is_system,
            'is_winner': is_winner,
            'color': color,
            'time': time.time(),
            'rating': rating
        })
        # Keep only last 30 messages
        if len(self.chat_messages) > 30:
            self.chat_messages.pop(0)
    
    def add_player(self, user_id, username, rating, games_played=0, country_flag='🏳️', manual_accessed=False, is_guest=False, is_ai=False, ai_rating=1200):
        """Add player to room"""
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
                
            # UNPAUSE: If human player joins a paused 'waiting' room, unpause it immediately
            if self.state == 'waiting' and not is_ai and self.time_limit < 7200:
                print(f"[GameRoom] Human player {username} joined waiting room {self.room_id}. Unpausing room...")
                self.state = 'intermission'
                self.intermission_start_time = time.time() - 60
                self.spinner_params_generated = False
                self.board_search_started = False
            
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
                # MID-ROUND DETECTION: Use round_start_time as authoritative truth.
                if not is_daily and self.state == 'active' and not existing_player.joined_mid_round:
                    round_started_at = getattr(self, 'round_start_time', 0)
                    was_present_at_start = (
                        round_started_at > 0 and
                        prior_last_active >= round_started_at - 30
                    )
                    if not was_present_at_start or manual_accessed:
                        existing_player.joined_mid_round = True
                        print(f"[GameRoom] Mid-round flag set for {username} (existing player path). round_start={round_started_at:.0f}, prior_last_active={prior_last_active:.0f}")
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
                return True
            
            # Check if player exists in round_quitters (RESTORE mid-round state)
            quitter = next((q for q in self.round_quitters if str(q.user_id) == uid_str or str(q.username).lower() == uname_lower), None)
            if quitter:
                print(f"[GameRoom] Restoring quitter {username} ({user_id}) to active players with {len(quitter.submitted_words)} words, score={quitter.score}.")
                quitter.last_active = time.time()
                quitter.country_flag = country_flag
                quitter.is_guest = is_guest
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
                
                # MID-ROUND DETECTION: Use round_start_time as authoritative truth.
                if not is_daily and self.state == 'active' and not existing_player.joined_mid_round:
                    round_started_at = getattr(self, 'round_start_time', 0)
                    was_present_at_start = (
                        round_started_at > 0 and
                        prior_last_active >= round_started_at - 30
                    )
                    if not was_present_at_start or manual_accessed:
                        existing_player.joined_mid_round = True
                        print(f"[GameRoom] Mid-round flag set for {username} (past_player path). round_start={round_started_at:.0f}, prior_last_active={prior_last_active:.0f}")
                
                if not getattr(existing_player, 'cell_density', None):
                    self._initialize_player_density_grid(existing_player)
                self.players = [p for p in self.players if str(p.user_id) != uid_str and str(p.username).lower() != uname_lower]
                self.players.append(existing_player)
                self.players.sort(key=lambda p: p.rating, reverse=True)
                if is_daily:
                    self.save_active_players()
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
                ai_rating=ai_rating
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

    def update_player_activity(self, user_id):
        """Update last_active timestamp for a player or spectator"""
        uid_str = str(user_id)
        player = self.get_player(user_id)
        if player:
            player.last_active = time.time()
            return

        # Check spectators too
        for p in self.spectators:
            if str(p.user_id) == uid_str:
                p.last_active = time.time()
                break

    def check_inactivity(self, timeout=600, spec_timeout=600): 
        """Remove players and spectators who haven't been active for their respective timeout seconds"""
        now = time.time()
        is_daily = self.time_limit >= 7200
        
        # Collect IDs to remove to avoid modifying list during iteration (10-minute standard timeout)
        to_remove_ids = []
        for p in self.players:
            age = now - p.last_active
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
            age = now - p.last_active
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
    
    @property
    def time_remaining(self):
        """Calculate time remaining in current state"""
        # PRIORITY: Intermission timer is always literal (60s)
        # 1. Intermission timer (Fixed 60s or 5s for Daily)
        if self.state == 'intermission':
            if not self.intermission_start_time or self.intermission_start_time <= 0:
                self.intermission_start_time = time.time()
            elapsed = time.time() - self.intermission_start_time
            intermission_limit = 2 if self.time_limit >= 7200 else 60
            return max(0.0, intermission_limit - elapsed)
            
        # 2. 24h Room ACTIVE: Align to real-world midnight boundary (America/Chicago)
        if self.state == 'active' and self.time_limit >= 7200:
            import datetime
            from zoneinfo import ZoneInfo
            tz = ZoneInfo("America/Chicago")
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
            tz = ZoneInfo("America/Chicago")
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
                    'has_abandoned': p.has_abandoned
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
        if self.state != 'active':
            return False, "Round is not active", 0, None
            
        player = self.get_player(user_id)
        if not player:
            if self.get_spectator(user_id):
                return False, "Spectators cannot submit words", 0, None
            return False, "Player not in room", 0, None

        # Clean spectator list if they are an active player
        self.spectators = [s for s in self.spectators if str(s.user_id) != str(user_id)]
        
        # Update activity on submission
        self.update_player_activity(user_id)
        
        if not word or not word.strip():
             return False, "Empty word", 0, None
        
        word = word.strip().upper()
        
        # Check if word is valid
        matched_word = None
        
        # For case-insensitive checks
        fmt_lower = self.current_board_format.lower()
        is_eo = 'either' in fmt_lower
        is_bonus_mode = ('bonus' in fmt_lower or is_eo)

        # In Either/Or format, mousing a path blindly sends the first letter of the L/T tile.
        # We must reconstruct the possible words from the provided path and find the one that is valid.
        if path and is_eo:
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
                            expanded_opt = 'QU' if opt == 'Q' else opt
                            new_words.append(prefix + expanded_opt)
                    possible_words = new_words
                else:
                    expanded_cell = 'QU' if cell_val == 'Q' else cell_val
                    for i in range(len(possible_words)):
                        possible_words[i] += expanded_cell
            
            if valid_path:
                # Find which of the possible interpreted words from the path actually exists on the board
                # USER REQUEST: Use dictionary validation instead of just all_words to match Solo mode!
                
                submitted_word_upper = word.upper()
                
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
                elif submitted_word_upper in possible_words:
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
        else:
            # Self-healing fallback for truncated/missing valid words on the board
            # Test both original word and QU variant
            candidates = [word]
            if 'Q' in word:
                candidates.append(word.replace('Q', 'QU'))
                
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
            
            # FIX: Use current_board_format instead of spinner_params to work correctly in With Friends / Private Match
            if 'penalty' in fmt_lower and not is_24h:
                # Is it on the board? If path was provided by dragging, we know it is!
                if path:
                    is_penalty = True
                else:
                    is_on_b, _ = word_validator.word_validator.find_word_on_board(self.board, word)
                    if is_on_b:
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
                    'is_penalty': True
                })
                
                # Update score (Sequential floor at 0 to avoid negative debt)
                self._recalculate_player_score(player)
                return True, f"{word} PENALTY (-3)", penalty_points, word
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
        word_metadata = {
            'word': final_word,
            'time': word_timestamp,
            'points': points,
            'score_details': points_data,
            'path': path
        }
        
        # USER: Update Density (Only after word is fully validated and scored)
        try:
            self.update_density_for_word(player, final_word, path)
        except Exception as e:
            print(f"[Density-Error] Failed to update density: {e}")
        
        player.submitted_words.append(word_metadata)
        
        # FCFS: Update shared found words list for Live Feed synchronization
        if self.game_type == 'fcfs':
            shared_meta = dict(word_metadata)
            shared_meta['finder'] = player.username
            self.fcfs_found_words.append(shared_meta)
            self._fcfs_found_words_set.add(final_word.upper())
            print(f"[FCFS-Sync] Shared word '{final_word}' added for {player.username}. Total shared: {len(self.fcfs_found_words)}")
        
        # Check if this is the bonus word
        if final_word == self.bonus_word:
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

                # User Directive: Display a trophy icon when PE is 2.0 or greater!
                if p.performance_efficiency >= 2.0:
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
            density_grid = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(depth)]
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
                            density_grid[f][r][c] += 1
                    except (IndexError, TypeError, ValueError): continue
        else:
            rows, cols = len(board), len(board[0])
            density_grid = [[0 for _ in range(cols)] for _ in range(rows)]
            for word, path in (all_words_paths or {}).items():
                if not path or not isinstance(path, (list, tuple)): continue
                for node in path:
                    try:
                        r, c = node[-2:]
                        if 0 <= r < rows and 0 <= c < cols:
                            density_grid[r][c] += 1
                    except (IndexError, TypeError, ValueError): continue
        
        # Find max density for heatmap normalization
        import math
        max_d = 0
        if is_3d:
            for f in range(len(density_grid)):
                for r in range(len(density_grid[f])):
                    for c in range(len(density_grid[f][r])):
                        max_d = max(max_d, density_grid[f][r][c])
        else:
            for r in range(len(density_grid)):
                for c in range(len(density_grid[r])):
                    max_d = max(max_d, density_grid[r][c])

        if is_staging:
            self.next_round_cell_density = [row[:] for row in density_grid] if not is_3d else [[row[:] for row in face] for face in density_grid]
            self.next_round_initial_cell_density = [row[:] for row in density_grid] if not is_3d else [[row[:] for row in face] for face in density_grid]
            self.next_round_max_cell_density = max_d
        else:
            self.cell_density = density_grid
            self.initial_cell_density = [row[:] for row in density_grid] if not is_3d else [[row[:] for row in face] for face in density_grid]
            self.max_cell_density = max_d
            self.initialize_player_densities()
        
        print(f"[Density] Initialization complete. Max density: {max_d}")
    
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
        if self.state == 'waiting':
            humans = [p for p in self.players if not getattr(p, 'is_ai', False)]
            if len(humans) > 0:
                print(f"[check_and_update_state] Unpausing waiting room {self.room_id} because {len(humans)} human player(s) present.")
                with self._state_lock:
                    self.state = 'intermission'
                    self.intermission_start_time = now - 60  # Force intermission to look expired
                    self.spinner_params_generated = False
                    self.board_search_started = False
                return 'start'
            return None
        
        # stuck watchdog: check if intermission is stuck for > 10s at 0:00:00 (timer at 0 and state == intermission)
        # Determine normal intermission duration
        intermission_limit = 2 if self.time_limit >= 7200 else 60
        is_at_or_past_zero = (self.state == 'intermission' and (now - self.intermission_start_time >= intermission_limit))

        
        # USER MANDATE: 0:50 Intermission Fallback Watchdog (10s remaining in intermission)
        # If no board has been secured by 0:50 of intermission (50s elapsed), pop a pregenerated board immediately,
        # update Spinner Set parameters (spinner_params, next_spinner_params), and reveal them DURING intermission!
        if self.state == 'intermission' and self.time_limit < 7200:
            elapsed_intermission = now - self.intermission_start_time
            if elapsed_intermission >= 10.0 and not getattr(self, 'next_round_board', None) and not getattr(self, '_fallback_at_50s_done', False):
                print(f"[Watchdog] 10s into intermission for {self.room_id} — no board ready. Running emergency fallback.")
                self._fallback_at_50s_done = True

                # Capture params immediately (fast, on Waitress thread)
                from board_generator import pop_compatible_cached_board
                _use_aw   = self.spinner_params.get('use_added_words', False) if isinstance(self.spinner_params, dict) else False
                _dict     = self.spinner_params.get('dictionary', 'NWL')       if isinstance(self.spinner_params, dict) else 'NWL'
                _fmt      = self.spinner_params.get('board_format', 'Normal')  if isinstance(self.spinner_params, dict) else 'Normal'
                _min_len  = self.spinner_params.get('min_word_length', 3)      if isinstance(self.spinner_params, dict) else 3
                _range    = self.spinner_params.get('word_count_range', '100-200') if isinstance(self.spinner_params, dict) else '100-200'
                _diff     = self.spinner_params.get('difficulty', 'Medium')    if isinstance(self.spinner_params, dict) else 'Medium'
                _bw_len   = self.spinner_params.get('bonus_word_length', 8)    if isinstance(self.spinner_params, dict) else 8
                _dims     = self.board_dimensions
                _bg       = _room_manager_instance.board_generator if (_room_manager_instance and hasattr(_room_manager_instance, 'board_generator')) else BoardGenerator()
                _room_ref = self

                def _stage_board(popped):
                    p_board, p_words, p_bonus_cell, p_format, p_paths, p_ratio, p_bonus_word, p_params = popped
                    fw = [w for w in p_words if len(w) >= _min_len]
                    fp = {w: v for w, v in p_paths.items() if len(w) >= _min_len}
                    _room_ref.next_round_board = p_board
                    _room_ref.next_round_words = fw
                    _room_ref.next_round_word_paths = fp
                    _room_ref.next_round_bonus_cell = p_bonus_cell
                    bw = p_bonus_word
                    if not bw or str(bw).upper() == 'NONE':
                        bw = _room_ref._get_bonus_word(length=_bw_len, dictionary=_dict, alternating=('checkerboard' in str(_fmt).lower()))
                    _room_ref.next_round_bonus = bw
                    _room_ref.next_round_format = _fmt
                    _room_ref.next_round_uniqueness = p_ratio

                # STEP 1: Try fast cache pop (< 1ms with WAL mode) — runs on Waitress thread
                popped = None
                for _ in range(10):
                    candidate = pop_compatible_cached_board(_dims, _dict, _fmt, _min_len, _use_aw)
                    if not candidate:
                        break
                    _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _fparams = candidate
                    _c_min = (_fparams.get('min_word_length') if isinstance(_fparams, dict) else None) or _min_len
                    _fw_f = [w for w in _fw if len(w) >= _c_min]
                    min_len_count = sum(1 for w in _fw_f if len(w) == _c_min)
                    if is_board_count_valid(len(_fw_f), _range) and min_len_count >= 5:
                        popped = candidate
                        break

                if popped:
                    # Cache hit: stage immediately (fast path)
                    _stage_board(popped)
                else:
                    # STEP 2: Cache miss — spawn background thread; do NOT block Waitress thread!
                    def _watchdog_gen():
                        try:
                            gen_res = _bg.generate_board(
                                dimensions=_dims,
                                bonus_word=None,
                                word_count_range=_range,
                                board_format=_fmt,
                                dictionary=_dict,
                                min_word_length=_min_len,
                                difficulty=_diff,
                                is_emergency=True,
                                use_added_words=_use_aw
                            )
                            if gen_res:
                                g_b, g_w, g_c, g_f, g_p, g_r, g_bw = gen_res[:7]
                                g_params = gen_res[8] if len(gen_res) > 8 else (_room_ref.spinner_params or {})
                                _stage_board((g_b, g_w, g_c, g_f, g_p, g_r, g_bw, g_params))
                        except Exception as ex_wd:
                            print(f"[Watchdog] Emergency generate_board error: {ex_wd}")
                    import threading
                    threading.Thread(target=_watchdog_gen, daemon=True).start()

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
            intermission_limit = 2 if self.time_limit >= 7200 else 60
            if elapsed >= intermission_limit - 0.5:
                return 'start'
            
        # 2. Parameter Reveal (15s into intermission)
        if self.state == 'intermission':
            if not self.intermission_start_time or self.intermission_start_time <= 0:
                self.intermission_start_time = now
            elapsed = now - self.intermission_start_time
            intermission_duration = 2 if self.time_limit >= 7200 else 60
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

    def ensure_next_board_ready(self):
        """Ensure that self.next_round_board is populated immediately."""
        if getattr(self, 'next_round_board', None):
            return
            
        print(f"[GameRoom] ensure_next_board_ready: next_round_board is missing for {self.room_id}. Resolving instantly...")
        
        # Pop compatible board matching dimensions and revealed parameters
        from board_generator import pop_compatible_cached_board
        target_params = getattr(self, 'next_spinner_params', None) or getattr(self, 'spinner_params', None) or {}
        dict_val = target_params.get('dictionary', 'NWL')
        fmt_val = target_params.get('board_format', 'Normal')
        min_l_val = target_params.get('min_word_length', 3)
        use_aw_val = target_params.get('use_added_words', False) or '+ AW' in str(dict_val).upper() or '+AW' in str(dict_val).upper()
        bonus_word_len = target_params.get('bonus_word_length')
        fallback = None
        for _ in range(10):
            candidate = pop_compatible_cached_board(
                self.board_dimensions,
                dict_val,
                fmt_val,
                min_l_val,
                use_aw_val,
                bonus_word_len=bonus_word_len
            )
            if not candidate:
                break
            fb, fw, fc, ff, fp, fr, fbw, fparams = candidate
            
            f_min_l = fparams.get('min_word_length', 3) if fparams else 3
            grid_floor = 3
            if '4x6' in self.board_dimensions: grid_floor = 4
            elif '5x7' in self.board_dimensions: grid_floor = 5
            elif '6x8' in self.board_dimensions or '3x3x3' in self.board_dimensions: grid_floor = 6
            f_min_l = max(grid_floor, int(f_min_l) if f_min_l is not None else 3)
            
            fw_filtered = [w for w in fw if len(w) >= f_min_l]
            actual_cnt = len(fw_filtered)
            
            f_wc = fparams.get('word_count_range', '100-200') if fparams else '100-200'
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
            
            is_aw_effective = (fparams.get('use_added_words', False) or '+ AW' in str(fparams.get('dictionary', '')).upper()) if fparams else False
            if is_aw_effective:
                min_accept = max(100, min_accept)
            elif f_min_l >= 6:
                min_accept = min(min_accept, 30)
            else:
                min_accept = max(30, min_accept)
                
            if actual_cnt >= min_accept:
                fallback = (fb, fw_filtered, fc, ff, {w: p for w, p in fp.items() if len(w) >= f_min_l}, fr, fbw, fparams)
                break
            else:
                print(f"[GameRoom] ensure_next_board_ready Pop Candidate had only {actual_cnt} words of length >= {f_min_l} (needed {min_accept}). Discarding...")
                
        if fallback:
            fb, fw, fc, ff, fp, fr, fbw, fparams = fallback
            print(f"[GameRoom] ensure_next_board_ready: Popped fallback cached board for {self.room_id}")
            self.next_round_board = fb
            self.next_round_words = fw
            self.next_round_word_paths = fp
            self.next_round_bonus_cell = fc
            self.next_round_bonus = fbw or ''
            self.next_round_format = ff
            self.next_round_uniqueness = fr
            if fparams:
                actual_wc = len(fw)
                self.next_round_spinner_params = fparams
                
                dict_val = fparams.get('dictionary', 'NWL')
                use_aw_val = fparams.get('use_added_words', False)
                clean_dict = str(dict_val).replace('+ AW', '').replace('+AW', '').strip()
                if use_aw_val:
                    dict_val = f"{clean_dict} + AW"
                else:
                    dict_val = clean_dict
                
                new_sp = {
                    'dictionary': dict_val,
                    'difficulty': fparams.get('difficulty', 'Medium'),
                    'word_count_range': wc_label,
                    'board_format': fparams.get('board_format', 'Normal'),
                    'min_word_length': fparams.get('min_word_length', 3),
                    'bonus_word_length': len(fbw) if fbw else fparams.get('bonus_word_len', 6),
                    'use_added_words': use_aw_val,
                    'board_dimensions': self.board_dimensions,
                    'time_limit': self.time_limit,
                    'generated_at': time.time()
                }
                self.next_spinner_params = new_sp
                # FIX: Do NOT overwrite self.spinner_params here.
                # spinner_params = CURRENT round's params (shown before reveal at 0:45).
                # next_spinner_params = NEXT round's params (shown after reveal at 0:45).
                # Overwriting spinner_params here would immediately change what the spinner shows
                # mid-round or at the start of intermission (issues 6, 8, 9, 10).
                self.spinner_params_generated = True
                
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
            return

        # If cache empty
        print(f"[GameRoom] ensure_next_board_ready: Cache empty. Using emergency fallback board.")
        source_sp = self.next_spinner_params or self.spinner_params or {}
        e_format = source_sp.get('board_format', 'Normal')
        e_dict = source_sp.get('dictionary', 'NWL')
        e_use_aw = source_sp.get('use_added_words', False)
        e_wc = source_sp.get('word_count_range', '100-200')
        
        e_min_len = source_sp.get('min_word_length')
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
            
        if e_params and not getattr(self, '_spinner_params_locked', False):
            if not self.spinner_params:
                self.spinner_params = {}
            self.spinner_params['dictionary'] = e_params.get('dictionary', 'NWL')
            self.spinner_params['difficulty'] = e_params.get('difficulty', 'Medium')
            self.spinner_params['board_format'] = e_params.get('board_format', 'Normal')
            self.spinner_params['min_word_length'] = e_params.get('min_word_length', 3)
            self.spinner_params['use_added_words'] = e_params.get('use_added_words', False)
            self.spinner_params['bonus_word_length'] = len(e_bonus_word) if e_bonus_word else e_params.get('bonus_word_len', 6)
            
            self.current_dictionary = e_params.get('dictionary', 'NWL')
            self.current_difficulty = e_params.get('difficulty', 'Medium')
            self.current_board_format = e_params.get('board_format', 'Normal')
            self.current_min_length = e_params.get('min_word_length', 3)
            self.use_added_words = e_params.get('use_added_words', False)
            
        if e_tr and not getattr(self, '_spinner_params_locked', False):
            self.current_word_count_range = e_tr
            
        self.next_round_board = e_board
        self.next_round_words = e_words
        self.next_round_word_paths = e_paths
        self.next_round_total_words_count = len(e_words)
        self.next_round_bonus = e_bonus_word
        self.next_round_format = e_fmt
        self.next_round_bonus_cell = e_bonus_c
        self.next_round_uniqueness = e_ratio
        
        is_valued_e = ('valued' in str(self.current_board_format).lower())
        e_scores = {}
        for w in e_words:
            if is_valued_e: e_scores[w] = {'total': get_valued_word_score(w), 'base': get_valued_word_score(w)}
            else:
                length = len(w)
                s = 0
                if length <= 2: s = 0
                elif length <= 4: s = 1
                elif length == 5: s = 2
                elif length == 6: s = 3
                elif length == 7: s = 5
                elif length >= 8: s = 11
                e_scores[w] = {'total': s, 'base': s}
        self.next_round_word_scores = e_scores
        
        # Build next_spinner_params from the actual emergency board params
        esp = {
            'dictionary': e_params.get('dictionary', e_dict) if e_params else e_dict,
            'difficulty': e_params.get('difficulty', 'Medium') if e_params else 'Medium',
            'word_count_range': e_tr or '100-200',
            'board_format': e_fmt or 'Normal',
            'min_word_length': e_params.get('min_word_length', 3) if e_params else 3,
            'bonus_word_length': len(e_bonus_word) if e_bonus_word else 6,
            'use_added_words': e_use_aw,
            'board_dimensions': self.board_dimensions,
            'time_limit': self.time_limit,
            'generated_at': time.time()
        }
        self.next_spinner_params = esp
        # FIX: Do NOT overwrite self.spinner_params (current round's params).

        self.spinner_params_generated = True
        
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
            humans = [p for p in self.players if not p.is_ai]
            if len(humans) > 0:
                # ISSUE 6 FIX: Guard against re-entry. This block runs on every heartbeat tick
                # (every 0.1s), which wiped and regenerated the board 7+ times in 45 seconds.
                # _wakeup_in_progress ensures we only run the wake-up sequence once.
                if getattr(self, '_wakeup_in_progress', False):
                    return False  # Already waking up — don't wipe/regenerate again
                self._wakeup_in_progress = True
                print(f"[GameRoom] Waking up paused room {self.room_id} instantly because human player joined. Generating/popping fresh board.")
                self.board = None
                self.all_words = set()
                self.all_words_paths = {}

                # 1. Pop a compatible board from cache immediately
                from board_generator import pop_compatible_cached_board
                sp = self.spinner_params or {}
                dict_val = sp.get('dictionary', 'NWL')
                fmt_val = sp.get('board_format', 'Normal')
                min_l_val = sp.get('min_word_length', 3)
                use_aw_val = sp.get('use_added_words', False) or '+ AW' in str(dict_val).upper() or '+AW' in str(dict_val).upper()
                bonus_word_len = sp.get('bonus_word_length')
                target_range = sp.get('word_count_range', '100-200')
                fallback = None
                for _ in range(10):
                    candidate = pop_compatible_cached_board(
                        self.board_dimensions,
                        dict_val,
                        fmt_val,
                        min_l_val,
                        use_aw_val,
                        bonus_word_len=bonus_word_len
                    )
                    if not candidate:
                        break
                    _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _fparams = candidate
                    _c_min_len = (_fparams.get('min_word_length') if isinstance(_fparams, dict) else None) or min_l_val
                    _fw_filtered = [w for w in _fw if len(w) >= _c_min_len]
                    min_len_count = sum(1 for w in _fw_filtered if len(w) == _c_min_len)
                    if is_board_count_valid(len(_fw_filtered), target_range) and min_len_count >= 5:
                        fallback = (_fb, _fw_filtered, _fc, _ff, {w: p for w, p in _fp.items() if w in _fw_filtered}, _fr, _fbw, _fparams)
                        break
                
                if fallback:
                    fb, fw, fc, ff, fp, fr, fbw, fparams = fallback
                    print(f"[GameRoom] Wakeup: Popped board from cache with {len(fw)} words for range '{target_range}'!")
                else:
                    print(f"[GameRoom] Wakeup: Cache empty. Using emergency fallback board.")
                    # Get emergency fallback board matching default spinner params
                    e_format = self.spinner_params.get('board_format', 'Normal') if self.spinner_params else 'Normal'
                    e_dict = self.spinner_params.get('dictionary', 'NWL') if self.spinner_params else 'NWL'
                    e_use_aw = self.spinner_params.get('use_added_words', False) if self.spinner_params else False
                    e_wc = target_range
                    e_min_len = self.spinner_params.get('min_word_length') if self.spinner_params else None
                    e_diff = self.spinner_params.get('difficulty', 'Medium') if self.spinner_params else 'Medium'
                    fallback = get_emergency_fallback_board(
                        self.board_dimensions, e_format, self.time_limit,
                        dictionary=e_dict, use_added_words=e_use_aw, target_range=e_wc, min_word_length=e_min_len, difficulty=e_diff
                    )
                    if len(fallback) >= 9:
                        fb, fw, fc, ff, fp, fr, fbw, _, fparams = fallback
                    else:
                        fb, fw, fc, ff, fp, fr, fbw, fparams = fallback

                # 2. Sync spinner params
                fparams = dict(fparams) if fparams else {}
                dict_val = fparams.get('dictionary') or (self.spinner_params.get('dictionary') if self.spinner_params else 'NWL')
                use_aw_val = fparams.get('use_added_words') or (self.spinner_params.get('use_added_words') if self.spinner_params else False)
                if use_aw_val and '+ AW' not in str(dict_val) and '+AW' not in str(dict_val):
                    dict_val = f"{dict_val} + AW"
                
                wc_label = target_range

                new_sp = {
                    'dictionary': dict_val,
                    'difficulty': fparams.get('difficulty') or (self.spinner_params.get('difficulty') if self.spinner_params else 'Medium'),
                    'word_count_range': wc_label,
                    'board_format': ff or (self.spinner_params.get('board_format') if self.spinner_params else 'Normal'),
                    'min_word_length': fparams.get('min_word_length') or (self.spinner_params.get('min_word_length') if self.spinner_params else 3),
                    'bonus_word_length': len(fbw) if fbw else (fparams.get('bonus_word_len') or 6),
                    'use_added_words': use_aw_val,
                    'board_dimensions': self.board_dimensions,
                    'time_limit': self.time_limit,
                    'generated_at': now,
                    'uniqueness': fr
                }
                self.board = fb
                calc_min = fparams.get('min_word_length') if (isinstance(fparams, dict) and fparams.get('min_word_length') is not None) else 3
                self.all_words_paths = {w: p for w, p in fp.items() if len(w) >= int(calc_min)}
                self.all_words = set(self.all_words_paths.keys())
                self.total_words_count = len(self.all_words)
                self.initial_total_words = self.total_words_count

                new_sp = SpinnerSet.sanitize_params(new_sp, self.board_dimensions, self.time_limit >= 7200)

                if not getattr(self, '_spinner_params_locked', False):
                    self.spinner_params = new_sp
                    self.next_spinner_params = new_sp
                    self.next_round_spinner_params = new_sp
                    self.spinner_params_generated = True
                    self.spinner_params_revealed = True
                    self.was_revealed_this_intermission = True
                    self._spinner_params_locked = True  # LOCK: no further spinner_params overwrites until round resets
                    import copy
                    self.frozen_revealed_params = copy.deepcopy(new_sp)
                    self._reveal_sync_complete = True
                
                # Set up active round variables
                self.current_board_format = new_sp['board_format']
                self.current_word_count_range = wc_label
                self.current_difficulty = new_sp['difficulty']
                self.current_dictionary = new_sp['dictionary']
                self.current_min_length = new_sp['min_word_length']
                self.use_added_words = new_sp['use_added_words']
                self.current_uniqueness = fr
                bw_l = new_sp.get('bonus_word_length', 8) if isinstance(new_sp, dict) else 8
                if not fbw or str(fbw).strip().upper() in ['', 'NONE'] or str(fbw).upper().endswith('ING') or str(fbw).upper().endswith('INGS'):
                    candidates = [w for w in (self.all_words or []) if len(w) == bw_l and not w.upper().endswith('ING') and not w.upper().endswith('INGS')]
                    if not candidates:
                        candidates = [w for w in (self.all_words or []) if len(w) >= 5 and not w.upper().endswith('ING') and not w.upper().endswith('INGS')]
                    if candidates:
                        import random
                        fbw = random.choice(list(candidates)).upper()
                    else:
                        fbw = self._get_bonus_word(
                            length=bw_l,
                            dictionary=self.current_dictionary,
                            alternating=('checkerboard' in str(self.current_board_format).lower())
                        )
                self.bonus_word = str(fbw or '').upper().strip()
                self.bonus_cell = fc
                self.solving_complete = True
                
                # Fast-score length-based scores instantly to eliminate HTTP request blocking latency!
                fast_scores = {}
                is_valued = ('valued' in str(self.current_board_format).lower())
                from scoring import get_valued_word_score
                for w in self.all_words:
                    if is_valued:
                        s = get_valued_word_score(w)
                    else:
                        length = len(w)
                        s = 1 if length <= 4 else (2 if length == 5 else (3 if length == 6 else (5 if length == 7 else 11)))
                    fast_scores[w] = {'total': s, 'base': s, 'bonus_word_points': 0, 'bonus_letter_points': 0, 'either_or_points': 0}
                self.solved_words_with_scores = fast_scores
                self.csw_only_words = getattr(self, 'next_round_csw_only_words', [])
                self.added_words = getattr(self, 'next_round_added_words', [])
                self.update_counts_by_len()

                # Set timestamps and transition state IMMEDIATELY (0ms latency!)
                self.state = 'active'
                self.round_start_time = now
                self.intermission_start_time = 0
                self.starting_round = False
                self._wakeup_in_progress = False
                self._initial_board_delivered = True
                self._transition_spinner_launched = False

                # Refine detailed scoring and dictionary categorization asynchronously in background daemon thread
                def refine_active_scoring_async():
                    try:
                        from scoring import calculate_word_score
                        scored_dict = {}
                        eval_board = self.board or getattr(self, 'previous_board', None)
                        eval_fmt = self.current_board_format
                        for w in self.all_words:
                            path_val = _get_word_path(self.all_words_paths, w)
                            scored_dict[w] = calculate_word_score(
                                w, self.bonus_word,
                                board_format=eval_fmt,
                                path=path_val,
                                bonus_cell=self.bonus_cell,
                                board=eval_board,
                                return_details=True
                            )
                        self.solved_words_with_scores = scored_dict
                    except Exception as e:
                        print(f"[GameRoom] Async active score refinement error: {e}")

                threading.Thread(target=refine_active_scoring_async, daemon=True).start()
                return True

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
                    if elapsed >= self.time_limit - 0.2:
                        # Ensure we don't end round 0.5s after start due to uninitialized time
                        if self.round_start_time > 0:
                            should_end = True
            else:
                # 24H Reset Logic (Midnight Boundary)
                if self.round_start_time > 0:
                    import datetime
                    from zoneinfo import ZoneInfo
                    tz = ZoneInfo("America/Chicago")
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
                    print(f"[GameRoom] Transitioning {self.room_id}: solved_words_with_scores missing. Scoring in background thread to prevent delay.")
                    fast_scores = {}
                    for word in (self.all_words or []):
                        l = len(word)
                        s = 1
                        if l <= 4: s = 1
                        elif l == 5: s = 2
                        elif l == 6: s = 3
                        elif l == 7: s = 5
                        elif l >= 8: s = 11
                        fast_scores[word] = {'total': s, 'base': s}
                    self.solved_words_with_scores = fast_scores
                    
                    # Spawn asynchronous thread to refine scores with full details (bonuses, paths)
                    def compute_fallback_scores_async():
                        try:
                            from scoring import calculate_word_score
                            refined_fallback = {}
                            eval_board = self.board or getattr(self, 'previous_board', None)
                            eval_fmt = self.current_board_format or getattr(self, 'previous_board_format', 'Normal')
                            for word in (self.all_words or []):
                                path_v = _get_word_path(self.all_words_paths, word)
                                refined_fallback[word] = calculate_word_score(
                                    word, 
                                    self.bonus_word, 
                                    board_format=eval_fmt,
                                    bonus_cell=self.bonus_cell,
                                    board=eval_board,
                                    path=path_v,
                                    return_details=True,
                                    strict_path=True
                                )
                            self.solved_words_with_scores = refined_fallback
                            self.previous_all_word_scores = dict(refined_fallback)
                            self.previous_all_words = list(self.all_words or [])
                            self.recalculate_total_points()
                            print(f"[GameRoom] Fallback scoring background refinement complete for {self.room_id}")
                        except Exception as e:
                            print(f"[GameRoom] Fallback scoring background refinement error: {e}")
                            
                    threading.Thread(target=compute_fallback_scores_async, daemon=True).start()

                # Snapshot board and words for intermission (Detailed Scoring Preservation)
                if self.game_type == '3d' or (self.board and len(self.board) == 6 and isinstance(self.board[0], list) and isinstance(self.board[0][0], list)):
                     self.previous_board = [[list(row) for row in face] for face in self.board]
                else:
                     self.previous_board = [list(row) for row in self.board] if self.board else None
                
                # USER REQUEST: Ensure 'All Words' list has full math breakdown in history
                self.previous_all_words = list(self.all_words) if self.all_words else []
                self.previous_all_word_scores = dict(getattr(self, 'solved_words_with_scores', {})) if getattr(self, 'solved_words_with_scores', None) else {}
                self.previous_min_length = getattr(self, 'current_min_length', 3)
                # BUGFIX: Snapshot paths, bonus_cell, and board_format so word-highlight clicks
                # during intermission use the COMPLETED round's board, not the next board.
                self.previous_all_words_paths = dict(self.all_words_paths) if isinstance(self.all_words_paths, dict) else {}
                self.previous_bonus_cell = getattr(self, 'bonus_cell', None)
                self.previous_board_format = getattr(self, 'current_board_format', 'Normal')
                if getattr(self, 'csw_only_words', None) and len(self.csw_only_words) > 0:
                    self.previous_csw_only_words = list(self.csw_only_words)
                else:
                    self.previous_csw_only_words = [w for w in (self.previous_all_words or []) if word_validator.word_validator.is_csw_only(w)]

                if getattr(self, 'added_words', None) and len(self.added_words) > 0:
                    self.previous_added_words = list(self.added_words)
                else:
                    self.previous_added_words = [w for w in (self.previous_all_words or []) if word_validator.word_validator.is_added_word(w)]
                self.previous_bonus_word = self.bonus_word
                self.previous_dictionary = getattr(self, 'current_dictionary', 'NWL')
                self.previous_use_added_words = getattr(self, 'use_added_words', False)
                
                # Snapshot for persistence
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
                            'performance_efficiency': getattr(p, 'performance_efficiency', 0)
                        })
                
                # Asynchronous Post-Round Processing
                def process_results_async():
                    try:
                        if self.game_type == 'split':
                            self.calculate_split_scores()
                        
                        # Winners History
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
                                'winners': winners_data,
                                'all_players': sorted([{'username': p.username, 'score': p.score} for p in (active_pool or self.players)], key=lambda x: x['score'], reverse=True),
                                'score': max_score,
                                'board': [list(row) for row in self.board] if self.board else [],
                                'words': winner_words,
                                'bonus_word': getattr(self, 'bonus_word', ''),
                                'timestamp': int(time.time() * 1000),
                                'round_duration': self.time_limit,
                                'round_start_time': self.round_start_time,
                            })
                            if len(self.winners_history) > 25: self.winners_history = self.winners_history[:25]

                        # Ratings logic...
                        try:
                            is_24h = (self.time_limit >= 7200)
                            if is_24h:
                                for p in self.players + quitters_snapshot:
                                    p.rating_change = 0
                                print(f"[GameRoom] 24-hour room: skipping rating updates.")
                            else:
                                from rating_logic import calculate_proportional_rating_change
                                # USER MANDATE: Only change ratings for players who started the round from the beginning
                                participants = [
                                    p for p in self.players + quitters_snapshot 
                                    if (getattr(p, 'score', 0) > 0 or not getattr(p, 'is_ai', False)) 
                                    and not getattr(p, 'joined_mid_round', False)
                                ]
                                rating_changes = calculate_proportional_rating_change(participants, is_private=self.is_private, board_format=self.current_board_format)
                                
                                with get_db() as conn_p:
                                    for p in self.players + quitters_snapshot:
                                        if p.user_id in rating_changes:
                                             p.rating_change = rating_changes[p.user_id]
                                             p.rating += p.rating_change
                                             # Update Global Rating
                                             conn_p.execute('UPDATE users SET rating = MAX(400, rating + ?) WHERE id = ?', (p.rating_change, p.user_id))
                                             
                                             # Update Config-Specific Rating (using INSERT OR ON CONFLICT UPDATE upsert)
                                             display_game_type = self.game_type.replace('solo_', '')
                                             config_key = f"{display_game_type}|{self.board_dimensions}|{self.time_limit}"
                                             conn_p.execute('''
                                                 INSERT INTO user_ratings (user_id, config_key, rating)
                                                 VALUES (?, ?, MAX(400, 1200 + ?))
                                                 ON CONFLICT(user_id, config_key) DO UPDATE SET rating = MAX(400, rating + ?)
                                             ''', (p.user_id, config_key, p.rating_change, p.rating_change))
                                        else:
                                             p.rating_change = 0

                                    # 5. Distribute Abandonment Bounty (User Request: At the end when results are shown)
                                    if self.abandonment_bounty > 0:
                                        eligible_receivers = [p for p in self.players if not p.is_ai and not getattr(p, 'is_guest', False) and not getattr(p, 'joined_mid_round', False)]
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

                        # Save history and word tally immediately at the start of intermission
                        global _room_manager_instance
                        rm = _room_manager_instance
                        if not rm:
                            try:
                                import app
                                rm = getattr(app, 'room_manager', None)
                            except Exception:
                                pass
                        if rm:
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
                                
                                # Log word tally
                                player_words = {p['username']: [w['word'] for w in p['submitted_words']] for p in intermission_player_snapshots}
                                rm.log_word_tally(self, player_words)
                            except Exception as db_save_err:
                                print(f"[GameRoom] Error in immediate database save: {db_save_err}")

                        # Store data for results screen and RESET mid-round flag for NEXT round
                        for p in self.players:
                            p.previous_round_score = p.score
                            p.previous_submitted_words = list(p.submitted_words)
                            p.joined_mid_round = False

                        if active_pool and max_score > 0:
                            winner_names = [p.username for p in active_pool if p.score == max_score]
                            msg = f"🏆 {winner_names[0]} won!" if len(winner_names) == 1 else f"🏆 Tie!"
                            self.add_chat_message("SYSTEM", msg, is_system=True, color='#ffd700', is_winner=True)

                    except Exception as e:
                        print(f"[GameRoom] Post-round error: {e}")

                threading.Thread(target=process_results_async, daemon=True).start()
                return True

        # 3. Intermission state check (milestones handled by RoomManager)
        if self.state == 'intermission':
            is_daily = (self.time_limit >= 7200)
            if not is_daily:
                elapsed = now - self.intermission_start_time
                # 10s elapsed corresponds to 50s remaining (0:50) of a 60s intermission
                if elapsed >= 10.0 and not getattr(self, '_did_050_fallback_rescue', False):
                    # If not staged yet, trigger early fallback rescue
                    if not self.next_round_board or not self.next_round_words:
                        self._did_050_fallback_rescue = True
                        print(f"[Fallback-Rescue] 0:50 remaining time reached (elapsed: {elapsed:.2f}s) without staged board for room {self.room_id}. Triggering early fallback board instantly.")
                        self.trigger_early_fallback_rescue()
            return True

        return False

    def trigger_early_fallback_rescue(self):
        """USER REQUEST: If no board is ready by 0:50 remaining time during intermission,
        grab a pregenerated board stored at random, assign its parameters, and stage it immediately."""
        now = time.time()
        import random
        
        # Pop compatible board from cache if available
        from board_generator import pop_compatible_cached_board
        sp = self.next_spinner_params or self.spinner_params or {}
        dict_val = sp.get('dictionary', 'NWL')
        fmt_val = sp.get('board_format', 'Normal')
        min_l_val = sp.get('min_word_length', 3)
        use_aw_val = sp.get('use_added_words', False) or '+ AW' in str(dict_val).upper() or '+AW' in str(dict_val).upper()
        bonus_word_len = sp.get('bonus_word_length')
        fallback = pop_compatible_cached_board(
            self.board_dimensions,
            dict_val,
            fmt_val,
            min_l_val,
            use_aw_val,
            bonus_word_len=bonus_word_len
        )
        
        if fallback:
            fb, fw, fc, ff, fp, fr, fbw, fparams = fallback
            print(f"[Fallback-Rescue] Early rescue: Popped board from cache with {len(fw)} words!")
        else:
            print(f"[Fallback-Rescue] Early rescue: Cache empty. Using emergency fallback board.")
            # Get emergency fallback board matching current spinner params
            e_format = self.spinner_params.get('board_format', 'Normal') if self.spinner_params else 'Normal'
            e_dict = self.spinner_params.get('dictionary', 'NWL') if self.spinner_params else 'NWL'
            e_use_aw = self.spinner_params.get('use_added_words', False) if self.spinner_params else False
            e_wc = self.spinner_params.get('word_count_range', '100-200') if self.spinner_params else '100-200'
            e_min_len = self.spinner_params.get('min_word_length') if self.spinner_params else None
            e_diff = self.spinner_params.get('difficulty', 'Medium') if self.spinner_params else 'Medium'
            
            fallback = get_emergency_fallback_board(
                self.board_dimensions, e_format, self.time_limit,
                dictionary=e_dict, use_added_words=e_use_aw, target_range=e_wc, min_word_length=e_min_len, difficulty=e_diff
            )
            if len(fallback) >= 9:
                fb, fw, fc, ff, fp, fr, fbw, _, fparams = fallback
            else:
                fb, fw, fc, ff, fp, fr, fbw, _ = fallback
                fparams = {}

        # 2. Sync spinner params
        fparams = dict(fparams) if fparams else {}
        dict_val = fparams.get('dictionary') or (self.spinner_params.get('dictionary') if self.spinner_params else 'NWL')
        use_aw_val = fparams.get('use_added_words') or (self.spinner_params.get('use_added_words') if self.spinner_params else False)
        if use_aw_val and '+ AW' not in str(dict_val) and '+AW' not in str(dict_val):
            dict_val = f"{dict_val} + AW"
            
        actual_wc = len(fw)
        if actual_wc < 100: wc_label = '50-100'
        elif actual_wc < 200: wc_label = '100-200'
        elif actual_wc < 300: wc_label = '200-300'
        elif actual_wc < 400: wc_label = '300-400'
        elif actual_wc < 500: wc_label = '400-500'
        else: wc_label = '500+'

        new_sp = {
            'dictionary': dict_val,
            'difficulty': fparams.get('difficulty') or (self.spinner_params.get('difficulty') if self.spinner_params else 'Medium'),
            'word_count_range': wc_label,
            'board_format': ff or (self.spinner_params.get('board_format') if self.spinner_params else 'Normal'),
            'min_word_length': fparams.get('min_word_length') or (self.spinner_params.get('min_word_length') if self.spinner_params else 3),
            'bonus_word_length': fparams.get('bonus_word_len') or len(fbw) if fbw else 6,
            'use_added_words': use_aw_val,
            'board_dimensions': self.board_dimensions,
            'time_limit': self.time_limit,
            'generated_at': now,
            'uniqueness': fr
        }
        
        # Enforce sanitization
        new_sp = SpinnerSet.sanitize_params(new_sp, self.board_dimensions, self.time_limit >= 7200)

        # Update next spinner params and current spinner params to make sure they match!
        self.spinner_params = new_sp
        self.next_spinner_params = new_sp
        self.next_round_spinner_params = new_sp
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
                elif length >= 8: s = 11
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
                self.update_density_for_word(wd['word'])
                
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
        valid_words = list(self.all_words or [])
        self.total_counts_by_len = {
            '_round': self.current_round,
            **{str(l): sum(1 for w in valid_words if len(w) == l) for l in range(1, 31)}
        }
        self.total_words_count = len(valid_words)
        self.initial_total_words = self.total_words_count
        
    def recalculate_total_points(self):
        """Aggregate total attainable points for the active round's word list"""
        try:
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
                        else:             attainable += 11

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

def get_emergency_fallback_board(dimensions, board_format='Normal', time_limit=60, dictionary='NWL', use_added_words=False, target_range=None, min_word_length=None, difficulty=None):
    """Dynamically generate a valid emergency fallback board that matches room dimensions and spells correct words."""
    import random
    from board_generator import BoardGenerator, serialize_param_key, pop_cached_board, refill_board_cache_bg, pop_any_cached_board
    global _room_manager_instance
    if _room_manager_instance and hasattr(_room_manager_instance, 'board_generator'):
        bg = _room_manager_instance.board_generator
    else:
        bg = BoardGenerator()
        
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
    if min_word_length < 7 and target_range:
        try:
            min_accept = int(str(target_range).split('-')[0])
        except:
            if '50' in str(target_range): min_accept = 50
            elif '100' in str(target_range): min_accept = 100
            elif '200' in str(target_range): min_accept = 200
            elif '300' in str(target_range): min_accept = 300
            elif '400' in str(target_range): min_accept = 400
            elif '500' in str(target_range): min_accept = 500
            
    is_aw = use_added_words or '+ AW' in str(dictionary).upper() or '+AW' in str(dictionary).upper()
    if is_aw:
        min_accept = max(100, min_accept)
    elif min_word_length >= 6:
        min_accept = min(min_accept, 30)
    else:
        min_accept = max(30, min_accept)
    
    parts = dimensions.split("x")
    is_24h = time_limit >= 7200
    
    if is_24h:
        use_added_words = False
        dictionary = str(dictionary or 'NWL').upper().replace('+ AW', '').replace('+AW', '').replace('ADDED_WORDS', '').replace('AW', '').strip()
        if dictionary not in ['NWL', 'CSW']:
            dictionary = 'NWL'
        target_range = '300-400'
        target_range_resolved = '300-400'
        fmt = 'Valued Letters'
    else:
        fmt = board_format

    try:
        # Determine starting min length based on dimensions
        min_l = 4 if '4x4' in dimensions else (5 if '4x6' in dimensions else (6 if '5x7' in dimensions else 7))
        
        # Determine target range
        if not is_24h and use_added_words:
            use_added_words = True
            import random
            target_range_resolved = random.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
        elif not is_24h:
            if target_range:
                target_range_resolved = target_range
            else:
                target_range_resolved = '100-200'
        
        floor_l = 3 if '4x4' in dimensions else (4 if '4x6' in dimensions else (5 if '5x7' in dimensions else 6))

        # --- CACHE-FIRST FAST PATH ---
        # Try progressively relaxed param keys against the DB cache before doing any live generation.
        # A cache hit is instantaneous — avoids the 5-10s blocking generation.
        base_dict = str(dictionary or 'NWL').upper().replace('+ AW', '').replace('+AW', '').strip() or 'NWL'
        if base_dict not in ['NWL', 'CSW']:
            base_dict = 'NWL'
        
        if min_accept >= 300:
            ranges_order = [target_range_resolved, '400-500', '300-400', '500+', '200-300', '100-200']
        else:
            ranges_order = [target_range_resolved, '200-300', '100-200', '300-400', '50-100']

        cache_candidates = []
        for r_item in ranges_order:
            cache_candidates.append((min_l, r_item, fmt, base_dict, use_added_words))
            cache_candidates.append((min_l, r_item, 'Normal', base_dict, use_added_words))
            cache_candidates.append((floor_l, r_item, 'Normal', base_dict, use_added_words))
            cache_candidates.append((floor_l, r_item, 'Normal', base_dict, False))
        
        diff_order = [difficulty] + [d for d in ['Hard', 'Medium', 'Easy'] if d != difficulty] if difficulty else ['Medium', 'Easy', 'Hard']
        for (cml, ctr, cfmt, cdict, caw) in cache_candidates:
            for cdiff in diff_order:
                try:
                    cache_key = serialize_param_key(dimensions, '', ctr, cdict, cfmt, cml, cdiff, use_added_words=caw)
                    cached = pop_cached_board(cache_key)
                    if cached and len(cached) >= 7:
                        cboard, cwords, cbonus_cell, cfmt_ret, cpaths, cratio, cbonus_word = cached[:7]
                        if difficulty == 'Hard' and cratio < 0.35:
                            print(f"[get_emergency_fallback_board] Rejecting cached board with low ratio {cratio} for Hard difficulty target")
                            continue
                        if cwords and len(cwords) >= 20:
                            print(f"[get_emergency_fallback_board] INSTANT CACHE HIT: min={cml}, range={ctr}, fmt={cfmt}, dict={cdict} → {len(cwords)} words")
                            # Kick off background refill for the popped key
                            refill_board_cache_bg(bg, cache_key, target_count=3)
                            
                            final_min = max(floor_l, min_word_length)
                            cwords_filtered = [w for w in cwords if len(w) >= final_min]
                            cpaths_filtered = {w: p for w, p in cpaths.items() if len(w) >= final_min}
                            
                            actual_wc = len(cwords_filtered)
                            if actual_wc < min_accept:
                                print(f"[get_emergency_fallback_board] Cached board had only {actual_wc} words for min_word_length={final_min} (needed >= {min_accept}). REJECTING cached board...")
                                continue
                                
                            if actual_wc < 100: ctr_resolved = '50-100'
                            elif actual_wc < 200: ctr_resolved = '100-200'
                            elif actual_wc < 300: ctr_resolved = '200-300'
                            elif actual_wc < 400: ctr_resolved = '300-400'
                            elif actual_wc < 500: ctr_resolved = '400-500'
                            else: ctr_resolved = '500+'
                            
                            # Enforce standard dictionaries for word counts < 300
                            caw_resolved = caw
                            cdict_resolved = cdict
                            if actual_wc < 300:
                                caw_resolved = False
                                cdict_resolved = str(cdict).replace('+ AW', '').replace('+AW', '').strip()
                                if cdict_resolved == 'AW': cdict_resolved = 'NWL'

                            eparams_dict = {
                                'min_word_length': final_min,
                                'word_count_range': ctr_resolved,
                                'board_format': cfmt,
                                'dictionary': cdict_resolved,
                                'use_added_words': caw_resolved,
                                'difficulty': cdiff,
                                'bonus_word_len': len(cbonus_word) if cbonus_word else 6
                            }
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
                use_added_words
            )
            if relaxed_res:
                board, words, bonus_cell, updated_format, paths, ratio, bonus_word, params = relaxed_res
                if words and len(words) >= 20:
                    params = dict(params) if params else {}
                    final_min = max(floor_l, min_word_length)
                    
                    words_filtered = [w for w in words if len(w) >= final_min]
                    paths_filtered = {w: p for w, p in paths.items() if len(w) >= final_min}
                    
                    actual_wc = len(words_filtered)
                    if actual_wc < min_accept:
                        print(f"[get_emergency_fallback_board] Ultimate cached board had only {actual_wc} words for min_word_length={final_min} (needed >= {min_accept}). REJECTING cached board...")
                    else:
                        if actual_wc < 100: ctr = '50-100'
                        elif actual_wc < 200: ctr = '100-200'
                        elif actual_wc < 300: ctr = '200-300'
                        elif actual_wc < 400: ctr = '300-400'
                        elif actual_wc < 500: ctr = '400-500'
                        else: ctr = '500+'
                        
                        # Enforce standard dictionaries for word counts < 300
                        use_aw_val = params.get('use_added_words', False)
                        dict_val = params.get('dictionary', 'NWL')
                        if actual_wc < 300:
                            use_aw_val = False
                            dict_val = str(dict_val).replace('+ AW', '').replace('+AW', '').strip()
                            if dict_val == 'AW': dict_val = 'NWL'
                        params['use_added_words'] = use_aw_val
                        params['dictionary'] = dict_val
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
        if _room_manager_instance and hasattr(_room_manager_instance, '_get_bonus_word'):
            emergency_bw = _room_manager_instance._get_bonus_word(length=7 if min_word_length < 8 else 8, dictionary=dictionary)
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
            l_bw = live_res[6] if len(live_res) >= 7 else (l_w[0] if l_w else 'PLANETS')
            l_w_filt = [w for w in l_w if len(w) >= min_word_length]
            if len(l_w_filt) >= 15:
                l_p_filt = {w: p for w, p in l_p.items() if w in l_w_filt}
                eparams = {
                    'min_word_length': min_word_length,
                    'word_count_range': target_range_resolved,
                    'board_format': l_f,
                    'dictionary': dictionary,
                    'use_added_words': use_added_words,
                    'difficulty': difficulty or "Medium",
                    'bonus_word_len': len(l_bw)
                }
                print(f"[get_emergency_fallback_board] Fast live generator succeeded: {len(l_w_filt)} words")
                return l_b, l_w_filt, l_c, l_f, l_p_filt, l_r, l_bw, target_range_resolved, eparams
    except Exception as live_err:
        print(f"[get_emergency_fallback_board] Fast live generation exception: {live_err}")

    # --- INSTANT EMERGENCY FALLBACK (Random grid solve) ---
    print(f"[get_emergency_fallback_board] Delivering instant pre-built emergency board for {dimensions}...")
    parts = dimensions.split("x")
    is_3d = len(parts) == 3
    if is_3d:
        board = [
            [['S', 'T', 'A'], ['R', 'E', 'D'], ['L', 'I', 'N']],
            [['E', 'R', 'S'], ['A', 'N', 'T'], ['I', 'C', 'S']],
            [['T', 'R', 'A'], ['I', 'N', 'S'], ['C', 'A', 'P']],
            [['P', 'A', 'R'], ['T', 'I', 'E'], ['S', 'E', 'T']],
            [['S', 'O', 'U'], ['N', 'D', 'S'], ['F', 'A', 'R']],
            [['W', 'O', 'R'], ['D', 'S', 'E'], ['T', 'S', 'S']]
        ]
        bonus_cell = (0, 0, 0)
    else:
        rows, cols = map(int, parts[:2])
        # Randomly shuffled standard letter distribution to ensure unique boards
        std_letters = ["S","T","A","R","E","D","L","I","N","E","R","S","A","N","T","I","C","S","T","R","A","I","N","S","C","A","P","T","U","R","E","S","O","U","N","D","S","F","A","R","W","O","R","D","S","E","T","S"]
        offset = random.randint(0, len(std_letters) - 1)
        shuffled = std_letters[offset:] + std_letters[:offset]
        random.shuffle(shuffled)
        board = [[shuffled[(r*cols + c) % len(shuffled)] for c in range(cols)] for r in range(rows)]
        bonus_cell = (random.randint(0, rows-1), random.randint(0, cols-1))

    # Solve emergency board instantly
    emergency_solve = bg._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=15, store_paths=True, timeout=1.0, use_added_words=use_added_words)
    words_filtered = [w for w in emergency_solve if len(w) >= min_word_length]
    paths_filtered = {w: p for w, p in emergency_solve.items() if len(w) >= min_word_length}

    # CRITICAL SAFEGUARD: If min_word_length on random grid yields fewer than 15 words, solve with 3L floor
    if len(words_filtered) < 15:
        emergency_solve_relaxed = bg._solve_board(board, dictionary, (0, 99999), 3, max_depth=15, store_paths=True, timeout=1.0, use_added_words=use_added_words)
        words_filtered = [w for w in emergency_solve_relaxed if len(w) >= 3]
        paths_filtered = {w: p for w, p in emergency_solve_relaxed.items() if len(w) >= 3}
        min_word_length = 3
    # MANDATE: Bonus Word MUST be between 6 and 10 letters!
    bw_candidates = [w for w in words_filtered if 6 <= len(w) <= 10]
    if bw_candidates:
        bonus_word = bw_candidates[0]
    else:
        if _room_manager_instance and hasattr(_room_manager_instance, '_get_bonus_word'):
            bonus_word = _room_manager_instance._get_bonus_word(length=6, dictionary=dictionary)
        else:
            bonus_word = 'PLANETS'
        if bonus_word and bonus_word not in words_filtered:
            words_filtered.append(bonus_word)
            paths_filtered[bonus_word] = [bonus_cell]
    
    eparams = {
        'min_word_length': min_word_length,
        'word_count_range': target_range_resolved,
        'board_format': fmt,
        'dictionary': dictionary,
        'use_added_words': use_added_words,
        'difficulty': difficulty or "Medium",
        'bonus_word_len': len(bonus_word)
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
                difficulty=difficulty or "Medium",
                is_emergency=True,
                use_added_words=use_added_words
            )
        except Exception as async_err:
            print(f"[get_emergency_fallback_board] Background board generation error: {async_err}")

    threading.Thread(target=_async_bg_generate, daemon=True).start()
    return board, words_filtered, bonus_cell, fmt, paths_filtered, 0.5, bonus_word, target_range_resolved, eparams

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
                        humans = [p for p in room.players if not p.is_ai]
                        if len(humans) == 0 and not is_daily:
                            # ISSUE 6 FIX: NEVER pause an ACTIVE round — only pause intermission/waiting.
                            # Pausing active rooms mid-board causes board wipes and re-rolls.
                            # Also enforce a 30-second grace period after the most recent round started
                            # to absorb transient 0-player moments (e.g., join latency on first load).
                            round_age = time.time() - getattr(room, 'round_start_time', 0)
                            is_newly_active = (room.state == 'active' and round_age < 30)
                            if room.state not in ['waiting'] and not is_newly_active:
                                if room.state == 'active':
                                    # Never wipe an active board — just skip milestones silently
                                    continue
                                if room.state != 'waiting':
                                    print(f"[Heartbeat] Pausing empty room {room_id}. Setting state to 'waiting'.")
                                    with room._state_lock:
                                        room.state = 'waiting'
                                        room.starting_round = False
                                        room.board_search_started = False
                                        room.board_search_loading = False
                                        room.spinner_params_generated = False
                                        room.next_round_board = None
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
                        print(f"[Heartbeat] Error on {room_id}: {tick_err}")
                
                # 2. Lazy Inactivity Cleanup (Every 30s: 120 iterations at 0.25s)
                if loop_counter % 120 == 0:
                    self.cleanup_rooms(timeout=600)
                    now = time.time()
                    with self.lock:
                        self.user_presence = {uid: ts for uid, ts in self.user_presence.items() if (now - ts) < 600}
                
                time.sleep(0.25)
            except Exception as e:
                import traceback
                print(f"[Heartbeat] CRITICAL: {e}\n{traceback.format_exc()}")
                time.sleep(5)
                
    def load_previous_day_data(self, room):
        """
        Load previous day (yesterday) data from the SQLite database
        if the in-memory variables are empty/vanished (e.g. after server restart).
        """
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
                conn.close()
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
                            # WAKE UP CHECK: If room is empty of human players and in waiting/intermission state, wake it up to active Round 1!
                            humans = [p for p in existing_room.players if not p.is_ai]
                            if len(humans) == 0 and existing_room.time_limit < 7200:
                                if existing_room.state in ['waiting', 'intermission']:
                                    print(f"[RoomManager] Wakeup empty singleton {existing_room.room_id} to Active Round 1 with fresh board...")
                                    from board_generator import pop_any_cached_board
                                    pre_pop = pop_any_cached_board(existing_room.board_dimensions)
                                    if not pre_pop:
                                        pre_pop = get_emergency_fallback_board(existing_room.board_dimensions, 'Normal', existing_room.time_limit)
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

                                        wc_cnt = len(existing_room.all_words)
                                        if wc_cnt < 100: wc_lbl = '50-100'
                                        elif wc_cnt < 200: wc_lbl = '100-200'
                                        elif wc_cnt < 300: wc_lbl = '200-300'
                                        elif wc_cnt < 400: wc_lbl = '300-400'
                                        elif wc_cnt < 500: wc_lbl = '400-500'
                                        else: wc_lbl = '500+'

                                        existing_room.spinner_params = {
                                            'dictionary': raw_dict,
                                            'board_format': r_fmt or 'Normal',
                                            'min_word_length': board_min_l,
                                            'word_count_range': wc_lbl,
                                            'difficulty': r_params.get('difficulty', 'Medium'),
                                            'use_added_words': raw_aw,
                                            'bonus_word_length': len(r_bonus_word) if r_bonus_word else 8,
                                            'board_dimensions': existing_room.board_dimensions,
                                            'time_limit': existing_room.time_limit,
                                            'generated_at': time.time(),
                                            '_exact_wc_calculated': True
                                        }

                                        existing_room.current_min_length = board_min_l
                                        existing_room.current_board_format = r_fmt or 'Normal'
                                        existing_room.current_word_count_range = wc_lbl
                                        existing_room.current_dictionary = raw_dict
                                        existing_room.current_uniqueness = r_ratio
                                        existing_room.use_added_words = raw_aw

                                        bw_candidate = r_bonus_word
                                        if not bw_candidate or bw_candidate not in existing_room.all_words or str(bw_candidate).strip().upper() in ['', 'NONE'] or str(bw_candidate).upper().endswith('ING') or str(bw_candidate).upper().endswith('INGS'):
                                            cand_list = [w for w in (existing_room.all_words or []) if len(w) >= 6 and not w.upper().endswith('ING') and not w.upper().endswith('INGS')]
                                            if not cand_list:
                                                cand_list = [w for w in (existing_room.all_words or []) if not w.upper().endswith('ING') and not w.upper().endswith('INGS')]
                                            if cand_list:
                                                import random
                                                bw_candidate = random.choice(list(cand_list)).upper()
                                            else:
                                                bw_candidate = self._get_bonus_word(length=8, dictionary=raw_dict)

                                        existing_room.bonus_word = str(bw_candidate or '').upper().strip()
                                        existing_room.previous_bonus_word = existing_room.bonus_word
                                        existing_room.board = r_board
                                        existing_room.bonus_cell = r_bonus_c
                                        existing_room.total_words_count = len(existing_room.all_words)
                                        existing_room.initial_total_words = existing_room.total_words_count
                                        existing_room.current_round = 1
                                        existing_room.round_start_time = time.time()
                                        existing_room.state = 'active'
                                        existing_room.initialize_density(r_board, existing_room.all_words_paths, r_fmt)
                                        existing_room.recalculate_total_points()

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
                if room.game_type in ['accumulative', 'solo_accumulative']:
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
                    tz = ZoneInfo("America/Chicago")
                    
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
                            
                            # Check if the board is from the same day
                            saved_dt = datetime.datetime.fromtimestamp(updated_at, tz)
                            now_dt = datetime.datetime.fromtimestamp(time.time(), tz)
                            if saved_dt.date() == now_dt.date():
                                print(f"[RoomManager] Restoring active 24h board for {room_id} from DB (saved at {saved_dt})")
                                room.board = json.loads(board_data_json)
                                room.all_words = set(json.loads(all_words_json))
                                room.current_dictionary = dictionary or 'NWL'
                                room.current_min_length = min_length or 3
                                room.bonus_word = bonus_word or ''
                                room.bonus_cell = json.loads(bonus_cell_json) if bonus_cell_json else None
                                room.current_board_format = 'Valued Letters' if is_24h else (board_format or 'Normal')
                                room.current_uniqueness = uniqueness or 0.0
                                room.current_word_count_range = word_count_range or ('300-400' if is_24h else '100-200')
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
                                                    'performance_efficiency': d.get('performance_efficiency', 0.0)
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
                    if not is_24h and not is_private:
                        print(f"[RoomManager] {room_id}: Kickstarting room immediately by popping board first...")
                        from board_generator import pop_cached_board, pop_any_cached_board, pop_compatible_cached_board, serialize_param_key
                        
                        pre_pop = None
                        if room.is_solo and getattr(room, 'initial_solo_params', None):
                            isp = room.initial_solo_params
                            s_dict = isp.get('dictionary', 'NWL')
                            s_fmt = isp.get('board_format', 'Normal')
                            s_min_len = int(isp.get('min_word_length', 3))
                            s_use_aw = False
                            if '+ AW' in str(s_dict) or '+AW' in str(s_dict) or s_dict == 'AW':
                                s_use_aw = True
                                s_dict = str(s_dict).replace('+ AW', '').replace('+AW', '').strip()
                                if s_dict == 'AW': s_dict = 'NWL'
                            s_bw_len = int(isp.get('bonus_word_length', 8)) if str(isp.get('bonus_word_length', '')).isdigit() else 8
                            
                            pre_pop = pop_compatible_cached_board(room.board_dimensions, s_dict, s_fmt, s_min_len, s_use_aw, bonus_word_len=s_bw_len)
                            if not pre_pop:
                                try:
                                    s_diff = isp.get('difficulty', 'Medium')
                                    s_wc = isp.get('word_count_range', '100-200')
                                    e_res = self.board_generator.generate_board(
                                        dimensions=room.board_dimensions,
                                        bonus_word=None,
                                        word_count_range=s_wc,
                                        board_format=s_fmt,
                                        dictionary=s_dict,
                                        min_word_length=s_min_len,
                                        difficulty=s_diff,
                                        is_emergency=True,
                                        use_added_words=s_use_aw
                                    )
                                    if e_res:
                                        g_b, g_w, g_c, g_f, g_p, g_r, g_bw = e_res[0], e_res[1], e_res[2], e_res[3], e_res[4], e_res[5], e_res[6]
                                        g_params = e_res[8] if len(e_res) > 8 else isp
                                        pre_pop = (g_b, g_w, g_c, g_f, g_p, g_r, g_bw, g_params)
                                except Exception as e_sgen:
                                    print(f"[RoomManager] Solo emergency generate error: {e_sgen}")
                        else:
                            pre_pop = pop_any_cached_board(room.board_dimensions)

                        if not pre_pop:
                            pre_pop = get_emergency_fallback_board(
                                room.board_dimensions, 'Normal', room.time_limit,
                                dictionary='NWL', use_added_words=False, target_range='100-200', min_word_length=3, difficulty='Medium'
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

                            # Derive scorable words & paths
                            raw_dict = r_params.get('dictionary', 'NWL')
                            raw_aw = r_params.get('use_added_words', False)
                            room.all_words_paths = {w: p for w, p in (r_dict or {}).items() if len(w) >= board_min_l and word_validator.word_validator.is_valid_word(w, raw_dict, use_added_words=raw_aw)}
                            room.all_words = set(room.all_words_paths.keys())

                            # Derive truthful word count range label
                            wc_cnt = len(room.all_words)
                            if wc_cnt < 100: wc_lbl = '50-100'
                            elif wc_cnt < 200: wc_lbl = '100-200'
                            elif wc_cnt < 300: wc_lbl = '200-300'
                            elif wc_cnt < 400: wc_lbl = '300-400'
                            elif wc_cnt < 500: wc_lbl = '400-500'
                            else: wc_lbl = '500+'

                            # Construct spinner params ONCE from exact board parameters
                            room.spinner_params = {
                                'dictionary': raw_dict,
                                'board_format': r_fmt or 'Normal',
                                'min_word_length': board_min_l,
                                'word_count_range': wc_lbl,
                                'difficulty': r_params.get('difficulty', 'Medium'),
                                'use_added_words': raw_aw,
                                'bonus_word_length': len(r_bonus_word) if r_bonus_word else 8,
                                'board_dimensions': room.board_dimensions,
                                'time_limit': room.time_limit,
                                'generated_at': time.time(),
                                '_exact_wc_calculated': True
                            }

                            room.current_min_length = board_min_l
                            room.current_board_format = r_fmt or 'Normal'
                            room.current_word_count_range = wc_lbl
                            room.current_dictionary = raw_dict
                            room.current_uniqueness = r_ratio
                            room.use_added_words = raw_aw

                            bw_candidate = r_bonus_word
                            if not bw_candidate or bw_candidate not in room.all_words or str(bw_candidate).strip().upper() in ['', 'NONE'] or str(bw_candidate).upper().endswith('ING') or str(bw_candidate).upper().endswith('INGS'):
                                cand_list = [w for w in (room.all_words or []) if len(w) >= 6 and not w.upper().endswith('ING') and not w.upper().endswith('INGS')]
                                if not cand_list:
                                    cand_list = [w for w in (room.all_words or []) if not w.upper().endswith('ING') and not w.upper().endswith('INGS')]
                                if cand_list:
                                    import random
                                    bw_candidate = random.choice(list(cand_list)).upper()
                                else:
                                    bw_candidate = self._get_bonus_word(length=8, dictionary=raw_dict)

                            room.bonus_word = str(bw_candidate or '').upper().strip()
                            room.previous_bonus_word = room.bonus_word
                            room.board = r_board
                            room.bonus_cell = r_bonus_c
                            room.total_words_count = len(room.all_words)
                            room.initial_total_words = room.total_words_count

                            room.round_start_time = time.time()
                            room.state = 'active'
                            room.current_round = 1
                            room.last_saved_round = -1

                            room.initialize_density(r_board, room.all_words_paths, r_fmt)
                            room.recalculate_total_points()

                            # Refine scoring asynchronously in background
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
                                    print(f"[RoomManager] Kickstart refinement error: {e}")

                            threading.Thread(target=refine_kickstart_scores, daemon=True).start()
                            
                            # Proactive background search for Round 2
                            room.spinner_params_generated = True
                            threading.Thread(target=self.start_board_search, args=(room_id,), daemon=True).start()
                            print(f"[RoomManager] {room_id} kickstarted ACTIVE in single-pass with truthful board & params.")
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
                else: s = 11
                kick_scores[w] = {'total': s, 'base': s}
        room.solved_words_with_scores = kick_scores
        # Sync counts and word_count_range label to the actual filtered set
        room.update_counts_by_len()
        _akc = room.total_words_count
        if _akc < 100: room.current_word_count_range = '50-100'
        elif _akc < 200: room.current_word_count_range = '100-200'
        elif _akc < 300: room.current_word_count_range = '200-300'
        elif _akc < 400: room.current_word_count_range = '300-400'
        elif _akc < 500: room.current_word_count_range = '400-500'
        else: room.current_word_count_range = '500+'

        room.round_start_time = time.time()
        room.state = 'active'
        room.current_round = 1
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
            with get_db() as conn:
                room_id = room.room_id
            
            # ROBUST HISTORY FETCH: Find the most recent round for this room that is NOT the current one
            # We search for the highest round_number < current_round, or just the most recent if current is 0/1.
            curr_round = room.current_round
            
            # Use timestamp to find "Recently finished" data (within last 36h to handle midnight rollovers)
            thirty_six_hours_ago = (datetime.datetime.now() - datetime.timedelta(hours=36)).strftime('%Y-%m-%d %H:%M:%S')
            
            # If current_round is 1 or 2 (new server session), find the latest from history
            if curr_round <= 2:
                # Get the absolute most recent round that ISN'T today's start if possible 
                # (handled by date/timestamp filtering)
                cursor = conn.execute('''
                    SELECT user_id, words_json, round_number, timestamp, board_json, bonus_word, bonus_cell, board_format, all_solutions_json, all_words_paths
                    FROM round_history 
                    WHERE room_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC, id DESC
                ''', (room_id, thirty_six_hours_ago))
            else:
                cursor = conn.execute('''
                    SELECT user_id, words_json, round_number, timestamp, board_json, bonus_word, bonus_cell, board_format, all_solutions_json, all_words_paths
                    FROM round_history 
                    WHERE room_id = ? AND round_number < ? AND timestamp >= ?
                    ORDER BY timestamp DESC, round_number DESC
                ''', (room_id, curr_round, thirty_six_hours_ago))
            
            rows = cursor.fetchall()
            
            if not rows:
                 # Fallback: if no matches for stable ID, search by dimensions + game_type for the most recent 24h round
                 parts = room_id.split('_')
                 dims = parts[2] if len(parts) >= 3 else room.board_dimensions
                 cursor = conn.execute('''
                    SELECT user_id, words_json, round_number, timestamp, board_json, bonus_word, bonus_cell, board_format, all_solutions_json, all_words_paths
                    FROM round_history 
                    WHERE board_dimensions = ? AND game_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
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
                uid, words_json, round_num, ts, b_json, b_word, b_cell_json, b_format, sols_json, paths_json = row
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
                uid, words_json, round_num, ts, b_json, b_word, b_cell_json, b_format, sols_json, paths_json = row
                uid_str = str(uid)
                if uid_str not in history:
                    if uid == -1:
                        uname = "System"
                    elif uid < 0:
                        uname = f"Guest_{abs(uid)}"
                    else:
                        u_cursor = conn.execute("SELECT username FROM users WHERE id = ?", (uid,))
                        u_row = u_cursor.fetchone()
                        uname = u_row[0] if u_row else f"User {uid}"
                    
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
    
    def cleanup_rooms(self, timeout=1200, spec_timeout=1800):
        """Clean up empty or inactive rooms (defaults: 20m players, 30m spectators)"""
        rooms_to_delete = []
        
        # Iterate over a copy of keys to avoid modification issues
        for room_id, room in list(self.rooms.items()):
            try:
                # Inactivity pause: if room is empty of human players and not a 24h daily room,
                # pause it by setting state to 'waiting' and skipping transition/milestone processing.
                # This prevents empty public rooms from looping rounds and pegging CPU in background.
                is_daily = (room.time_limit >= 7200)
                humans = [p for p in room.players if not p.is_ai]
                if len(humans) == 0 and not is_daily:
                    # ISSUE 6 FIX: NEVER wipe the board or reset an ACTIVE room.
                    # Active rooms may briefly show 0 humans due to join latency or roster timing.
                    # Wiping room.board causes the watchdog to immediately start_next_round, rolling
                    # a completely new board (the "7 boards in 45 seconds" bug).
                    if room.state == 'active':
                        continue  # Skip — active board is sacred, never wipe mid-round
                    if room.state != 'waiting':
                        print(f"[BG-Cleanup] Pausing empty room {room_id}. Setting state to 'waiting'.")
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
                    import threading
                    threading.Thread(target=self.generate_spinner_params, args=(room_id, False), daemon=True).start()
                
                elif milestone == 'reveal':
                    # Reveal (Intermission only)
                    if not getattr(room, 'spinner_params_revealed', False):
                        import threading
                        threading.Thread(target=self.generate_spinner_params, args=(room_id, True), daemon=True).start()

                elif milestone == 'search':
                    # Proactive board search
                    if not getattr(room, 'board_search_started', False):
                        room.board_search_started = True
                        print(f"[TRANSITION] Room {room_id}: PROACTIVE SEARCH TRIGGERED")
                        import threading
                        threading.Thread(target=self.start_board_search, args=(room_id,), daemon=True).start()
                
                elif milestone == 'start' and room.state == 'intermission':
                    # Start next round
                    if not getattr(room, 'starting_round', False):
                        import threading
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
            if not has_prev and room.all_words:
                # USER REQUEST: Absolute filter for history. 
                # Display Floor is 4L, but must also respect round minimum (e.g. 5L).
                display_min_hist = getattr(room, 'current_min_length', 3)
                words_list = [w for w in room.all_words if len(w) >= display_min_hist]
                room.previous_all_words = {w: {} for w in words_list} # Dict format for scores compatibility
                room.previous_board = [list(row) for row in room.board]
                print(f"[RoomManager] Saved {len(room.previous_all_words)} words to history (Fallback/Round {room.current_round})")
            elif has_prev:
                print(f"[RoomManager] Using existing history snapshot (intermission) for Round {room.current_round}")
            
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
            from spinner_set import SpinnerSet
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
                    min_accept = max(100, min_accept)
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
                    from board_generator import pop_compatible_cached_board
                    solo_pop = pop_compatible_cached_board(
                        room.board_dimensions,
                        room.spinner_params.get('dictionary', 'NWL'),
                        target_fmt,
                        room.spinner_params.get('min_word_length', 3),
                        use_aw_flag
                    )
                    if not solo_pop and 'checkerboard' not in str(target_fmt).lower():
                        solo_pop = pop_any_cached_board(room.board_dimensions)
                    
                    if solo_pop:
                        if len(solo_pop) >= 9:
                            sboard, swords, sbonus_cell, sfmt, spaths, sratio, sbonus_word, _, sparams = solo_pop
                        else:
                            sboard, swords, sbonus_cell, sfmt, spaths, sratio, sbonus_word, sparams = solo_pop
                        
                        # SAFEGUARD: If format is Checkerboard, enforce strict C/V alternation pattern!
                        if 'checkerboard' in str(target_fmt).lower():
                            self.board_generator._verify_checkerboard_safeguard(sboard)

                        swords_filtered = [w for w in swords if len(w) >= m_len]
                        spaths_filtered = {w: p for w, p in (spaths or {}).items() if len(w) >= m_len}
                        if len(swords_filtered) >= 10:
                            print(f"[RoomManager] INSTANT Solo cached board popped for room {room_id} in 1ms! (format={target_fmt})")
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

                    for _ in range(10):
                        from board_generator import pop_compatible_cached_board
                        relaxed_res = pop_compatible_cached_board(
                            room.board_dimensions,
                            room.spinner_params['dictionary'],
                            room.spinner_params['board_format'],
                            room.spinner_params.get('min_word_length', 3),
                            use_aw_flag,
                            bonus_word_len=len(bonus_word) if bonus_word else None
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
                        if len(all_words_filtered) >= min_accept:
                            # Accept board without discarding valid grid words
                            all_words = all_words_filtered
                            all_words_dict = {w: p for w, p in all_words_dict.items() if w in all_words}
                            print(f"[RoomManager] Popped relaxed cached board for room {room_id} with {len(all_words)} words. Keeping spun parameters.")
                            
                            # Preserve the spun spinner_params completely to stick with what was initially determined.
                            # Only set the current active properties of the room to match the spun spinner_params.
                            room.current_dictionary = room.spinner_params.get('dictionary', 'NWL')
                            room.current_difficulty = room.spinner_params.get('difficulty', 'Medium')
                            room.current_board_format = room.spinner_params.get('board_format', 'Normal')
                            room.current_min_length = room.spinner_params.get('min_word_length', 3)
                            room.use_added_words = room.spinner_params.get('use_added_words', False)

                            
                            # Issue 2: Use the embedded bonus word from the cached board, not a freshly-spun one.
                            # The embedded bonus_word was verified to exist on the board during generation.
                            # Only re-roll if the embedded word is absent or not in all_words.
                            _cached_bw = final_bonus_word
                            if _cached_bw and len(_cached_bw) >= m_len and _cached_bw in all_words_filtered:
                                bonus_word = _cached_bw
                                room.bonus_word = bonus_word
                            else:
                                # Embedded bonus word not found in filtered set — re-roll from actual words on board
                                _bw_candidates = [w for w in all_words_filtered if len(w) == room.spinner_params.get('bonus_word_length', 8)]
                                if not _bw_candidates:
                                    _bw_len = max(m_len, min(len(w) for w in all_words_filtered) if all_words_filtered else 6)
                                    _bw_candidates = [w for w in all_words_filtered if len(w) == _bw_len]
                                if _bw_candidates:
                                    # Using global random module
                                    bonus_word = random.choice(_bw_candidates)
                                    room.bonus_word = bonus_word
                                    print(f"[RoomManager] Embedded bonus word '{_cached_bw}' not in board words; re-rolled to '{bonus_word}'")

                            res = (board, all_words, bonus_cell, board_format_ret, all_words_dict, ratio, bonus_word)
                            break
                        else:
                            print(f"[RoomManager] Popped candidate relaxed board discarded: had only {len(all_words_filtered)} words of length >= {act_min} (needed {min_accept}).")
                
                if not res:
                    print(f"[RoomManager] Cache miss for room {room_id}. Delivering instant emergency fallback board.")
                    res = get_emergency_fallback_board(
                        room.board_dimensions, room.spinner_params.get('board_format', 'Normal'), room.time_limit,
                        dictionary=room.spinner_params.get('dictionary', 'NWL'), use_added_words=use_aw_flag,
                        target_range=room.spinner_params.get('word_count_range', '100-200'), min_word_length=m_len, difficulty='Medium'
                    )
                
                board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio, final_bonus_word = res[:7]
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
                return False
                
            # ATOMICITY: Apply new round data with strict display filtering
            display_min_start = room.spinner_params.get('min_word_length', 3)
            raw_dict = room.spinner_params.get('dictionary', 'NWL')
            raw_aw = room.spinner_params.get('use_added_words', False)
            if all_words:
                actual_shortest = min(len(w) for w in all_words)
                if actual_shortest < display_min_start:
                    print(f"[RoomManager] Word generator relaxed min_word_length from {display_min_start} to {actual_shortest}. Updating room param.")
                    room.spinner_params['min_word_length'] = actual_shortest
                    display_min_start = actual_shortest
            
            all_words_dict = {w: p for w, p in (all_words_dict or {}).items() if len(w) >= display_min_start and word_validator.word_validator.is_valid_word(w, raw_dict, use_added_words=raw_aw)}
            room.all_words = set(all_words_dict.keys())
            room.all_words_paths = all_words_dict

            
            # PRESERVE SPUN PARAMETER & KEEP 100% OF SCORABLE GRID WORDS INTACT
            target_range = room.spinner_params.get('word_count_range') if isinstance(room.spinner_params, dict) else '100-200'
            room.current_word_count_range = target_range
            room.update_counts_by_len()

            # FIX: Override with the ACTUAL word count from the generated board.
            # The target range (e.g. '300-400') is what was requested; the board may
            # deliver a different count (e.g. 253). Recompute the label from the real
            # all_words set so the Spinner Set and header show the correct range.
            # AW boards use the same full-scale buckets as non-AW — 221 words is
            # '200-300', not '300-400', regardless of dictionary.
            _actual_wc  = len(room.all_words)
            _use_aw_val = room.use_added_words or (room.spinner_params.get('use_added_words', False) if isinstance(room.spinner_params, dict) else False)
            _real_wc_sr = self._get_factchecked_wc_range(_actual_wc, use_added_words=_use_aw_val)
            room.current_word_count_range = _real_wc_sr
            if isinstance(room.spinner_params, dict):
                room.spinner_params['word_count_range'] = _real_wc_sr
            if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                room.frozen_revealed_params['word_count_range'] = _real_wc_sr


            
            # CATEGORIZATION (Synchronous): Ensure these are available immediately for UI sync
            if hasattr(word_validator, 'word_validator'):
                if str(getattr(room, 'current_dictionary', 'NWL')).upper() in ['CSW', 'AW', 'ALL', 'ADDED_WORDS']:
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

            if final_bonus_word:
                room.bonus_word = final_bonus_word
                room.spinner_params['bonus_word_length'] = len(final_bonus_word)

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
            room.current_difficulty = f"{achieved_diff} ({int(u_ratio * 100)}%)"
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
                    elif length >= 8: s = 11
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
                                    'has_abandoned': p.has_abandoned
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
                                    room.current_word_count_range or '200-300',
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
                elif getattr(p, 'previous_round_score', 0) > 0 or (hasattr(p, 'previous_submitted_words') and len(p.previous_submitted_words) > 0):
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
        room = self.get_room(room_id)
        if not room:
            return False
            
        with room._state_lock:
            # 1. Check if already revealed - if so, NOOP
            if reveal and getattr(room, 'spinner_params_revealed', False):
                return True
                
            # 2. Check if already generated but not yet revealed
            if reveal and getattr(room, 'spinner_params_generated', False):
                new_params = getattr(room, 'next_spinner_params', None)
                if new_params:
                    # PERFORM THE REVEAL — mirror exactly what the slow-path does at line 5008-5019
                    import copy
                    if not getattr(room, '_spinner_params_locked', False):
                        room.spinner_params = dict(new_params)
                        # FIX: set frozen_revealed_params so start_board_search can update word_count_range
                        # on it, and so start_next_round uses it as the authoritative active_params.
                        room.frozen_revealed_params = copy.deepcopy(new_params)
                    
                    # Update authoritative labels so they change ON THE DOT at 0s (start_next_round)
                    # We store them in spinner_params for reveal, but don't promote to 'current_' yet
                    room.next_round_min_length = new_params.get('min_word_length', 3)
                    room.spinner_params_revealed = True
                    # FIX: mark the intermission as revealed so start_next_round takes the frozen path
                    room.was_revealed_this_intermission = True
                    # FIX: lock spinner_params so board search thread cannot overwrite them post-reveal.
                    # Without this lock, the board search was overwriting room.spinner_params with the
                    # actual board params (format, wc), making previous_params predictable and causing
                    # the anti-repeat check to always converge to the same "different" pattern.
                    room._spinner_params_locked = True
                    room._reveal_sync_complete = True
                    print(f"[RoomManager] SUCCESS: Revealed pre-generated params for room {room_id} (15s mark)")
                    return True
            
            # 3. If we are currently loading, skip to avoid double generation
            if getattr(room, 'spinner_params_loading', False):
                return True
                
            room.spinner_params_loading = True

        try:
            is_24h = (room.time_limit >= 7200)
            is_split = (room.game_type == 'split')
            
            with room._state_lock:
                # USER REQUEST: Prevent re-rolling! Check lock state INSIDE the atomic block.
                # In Solo mode, we always prefer the user's initial settings over background pre-gen!
                if getattr(room, 'spinner_params_generated', False) and room.next_spinner_params:
                    new_params = dict(room.next_spinner_params)
                    print(f"[RoomManager] Using EXISTING staged params for room {room_id} (Lock-protected)")
                else:
                    # Generate new parameters
                    if getattr(room, 'is_solo', False) and getattr(room, 'initial_solo_params', None):
                        initial_solo_params = room.initial_solo_params
                        dict_choice = initial_solo_params.get('dictionary', 'random')
                        min_word_len = int(initial_solo_params.get('min_word_length', 3))
                        
                        # Safe-parse bonus word length: spin if random (equal weights for 6-10)
                        bonus_len_choice = initial_solo_params.get('bonus_word_length', 'random')
                        if bonus_len_choice == 'random' or not bonus_len_choice or str(bonus_len_choice) == '0':
                            bonus_word_len = random.choices([6, 7, 8, 9, 10], weights=[20, 20, 20, 20, 20])[0]
                        else:
                            try:
                                bonus_word_len = int(bonus_len_choice)
                            except:
                                bonus_word_len = 8

                        board_fmt = initial_solo_params.get('board_format', 'Normal')
                        difficulty_choice = initial_solo_params.get('difficulty', 'random')
                        wc_choice = initial_solo_params.get('word_count_range', 'random')
                        
                        # 1. Resolve Dictionary
                        if dict_choice == 'random':
                            dictionary = SpinnerSet._spin_dictionary()
                        else:
                            dictionary = dict_choice
                            
                        # Extract + AW from dictionary name
                        use_aw = False
                        if dictionary and ('+ AW' in str(dictionary) or '+AW' in str(dictionary)):
                            use_aw = True
                            dictionary = str(dictionary).replace('+ AW', '').replace('+AW', '').strip()
                        elif dictionary == 'AW':
                            use_aw = True
                            dictionary = 'NWL'
                            
                        # 2. Resolve Difficulty
                        if difficulty_choice == 'random':
                            difficulty = SpinnerSet._spin_difficulty(room.board_dimensions, min_word_len)
                        else:
                            difficulty = difficulty_choice
                            
                        # 3. Resolve Word Count Range
                        if wc_choice == 'random':
                            wc_range = SpinnerSet._spin_word_count(dictionary, min_word_len, difficulty, room.board_dimensions, use_added_words=use_aw)
                        else:
                            wc_range = wc_choice
                            
                        # 4. Resolve Board Format
                        if board_fmt == 'random':
                            resolved_board_format = SpinnerSet._spin_board_format(is_24h=False, dimensions=room.board_dimensions)
                        else:
                            resolved_board_format = board_fmt

                        new_params = {
                            'min_word_length': min_word_len,
                            'difficulty': difficulty,
                            'word_count_range': wc_range,
                            'dictionary': dictionary,
                            'board_format': resolved_board_format,
                            'bonus_word_length': bonus_word_len,
                            'use_added_words': use_aw,
                            'generated_at': time.time()
                        }
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
                                    # FIX: Write the floored min_length back into new_params so that
                                    # frozen_revealed_params and active_params at round start use the
                                    # SAME min_word_length that staging used for filtering.
                                    # Without this, staging filters with min_length=6 but the round
                                    # starts with min_length=5, causing more words to survive and the
                                    # label to flip (e.g. staged '100-200' → round plays '200-300').
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
                                        # FIX: Update new_params word_count_range with ACTUAL count (not target)
                                        # so frozen_revealed_params and the Spinner Set show the real board size.
                                        is_aw_eff = new_params.get('use_added_words', False) or '+ AW' in str(new_params.get('dictionary', '')).upper()
                                        if is_aw_eff:
                                            actual_wc_range_exact = '300-400' if actual_cnt < 400 else ('400-500' if actual_cnt < 500 else '500+')
                                        else:
                                            if actual_cnt < 100: actual_wc_range_exact = '50-100'
                                            elif actual_cnt < 200: actual_wc_range_exact = '100-200'
                                            elif actual_cnt < 300: actual_wc_range_exact = '200-300'
                                            elif actual_cnt < 400: actual_wc_range_exact = '300-400'
                                            elif actual_cnt < 500: actual_wc_range_exact = '400-500'
                                            else: actual_wc_range_exact = '500+'
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
                        bw_len = new_params.get('bonus_word_length', 8)
                        target_sp_range = new_params.get('word_count_range')

                        for _ in range(10):
                            candidate = pop_compatible_cached_board(
                                room.board_dimensions, dict_val, fmt_val, m_len, use_aw_val, bonus_word_len=bw_len
                            )
                            if not candidate:
                                break
                            c_b, c_w, c_c, c_f, c_p, c_r, c_bw, c_params = candidate
                            fw_filt = [w for w in c_w if len(w) >= m_len]
                            actual_wc = len(fw_filt)
                            if is_board_count_valid(actual_wc, target_sp_range):
                                fp_filt = {w: p for w, p in c_p.items() if w in fw_filt}
                                actual_wc_range = self._get_factchecked_wc_range(actual_wc, use_added_words=use_aw_val)
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
                        new_params['difficulty'] = self.board_generator.get_difficulty_label(ratio, rows=r_cnt, cols=c_cnt, dictionary=new_params.get('dictionary', 'NWL'), min_word_length=new_params.get('min_word_length', 3))
                        if final_bonus_word:
                            new_params['bonus_word_length'] = len(final_bonus_word)
                        
                        # Stage the board data in the room immediately
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
                        
                        # Calculate length-based scores to prevent UI flashing
                        is_valued = ('valued' in str(board_format_ret).lower())
                        scored_dict = {}
                        for w in all_words:
                            if is_valued:
                                v_score = get_valued_word_score(w)
                                scored_dict[w] = {'total': v_score, 'base': v_score}
                            else:
                                length = len(w)
                                s = 1 if length <= 4 else (2 if length == 5 else (3 if length == 6 else (5 if length == 7 else 11)))
                                scored_dict[w] = {'total': s, 'base': s}
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
                    import copy
                    if not getattr(room, '_spinner_params_locked', False):
                        room.spinner_params = copy.deepcopy(new_params)
                        room.frozen_revealed_params = copy.deepcopy(new_params)
                    
                    # Update authoritative labels
                    room.next_round_min_length = new_params.get('min_word_length', 3)
                    room.spinner_params_revealed = True
                    room.was_revealed_this_intermission = True
                    room._spinner_params_locked = True  # LOCK: no further overwrites until intermission ends!
                    room._reveal_sync_complete = True
                    print(f"[RoomManager] SUCCESS: Generated and Revealed params for room {room_id}")
                else:
                    print(f"[RoomManager] SUCCESS: Silent parameter generation complete for room {room_id}")
                
                # 3. PROACTIVE LEAD TIME: Start searching for the board IMMEDIATELY after params are decided
                # This gives us up to 45-60s of lead time instead of 15s.
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

    def check_6x8_rescue(self, room):
        """Rescue watchdog for all rooms.
        Rescues rooms stuck in 'loading' for >20s, or in intermission at 50s remaining without a board staged."""
        if not room:
            return
            
        now = time.time()
        is_daily = room.time_limit >= 7200
        
        # 1. Loading state rescue (stuck loading for >20s)
        is_stuck_loading = False
        if room.state == 'loading':
            start_t = getattr(room, 'intermission_start_time', None) or getattr(room, 'created_at', None) or now
            elapsed = now - start_t
            if elapsed > 10.0 and not getattr(room, '_did_loading_rescue', False):
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
                e_wc = room.spinner_params.get('word_count_range', '100-200') if room.spinner_params else '100-200'
                e_use_aw = room.spinner_params.get('use_added_words', False) if room.spinner_params else False
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
                    e_wc = room.spinner_params.get('word_count_range', '100-200') if room.spinner_params else '100-200'
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
                    dict_val = cparams.get('dictionary') or (room.spinner_params.get('dictionary') if room.spinner_params else 'NWL')
                    use_aw_val = cparams.get('use_added_words') or (room.spinner_params.get('use_added_words') if room.spinner_params else False)
                    if use_aw_val and '+ AW' not in str(dict_val) and '+AW' not in str(dict_val):
                        dict_val = f"{dict_val} + AW"
                        
                    actual_wc = len(cwords)
                    # AW boards use the same full scale — no '300-400' floor.
                    wc_label = self._get_factchecked_wc_range(actual_wc, use_added_words=use_aw_val)

                    if not getattr(room, '_spinner_params_locked', False):
                        room.spinner_params['dictionary'] = dict_val
                        room.spinner_params['difficulty'] = cparams.get('difficulty') or (room.spinner_params.get('difficulty') if room.spinner_params else 'Medium')
                        room.spinner_params['word_count_range'] = wc_label
                        room.spinner_params['board_format'] = cfmt or (room.spinner_params.get('board_format') if room.spinner_params else 'Normal')
                        room.spinner_params['min_word_length'] = cparams.get('min_word_length') or (room.spinner_params.get('min_word_length') if room.spinner_params else 3)
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
                        
                        # Generate basic scores
                        is_valued = ('valued' in str(cfmt).lower())
                        scored_dict = {}
                        for w in cwords:
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
                                else: s = 11
                                scored_dict[w] = {'total': s, 'base': s}
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
                    'word_count_range': '200-300' if room.time_limit >= 7200 else '100-200',
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
            # Ensure fresh next_spinner_params are generated using SpinnerSet
            params = getattr(room, 'next_spinner_params', None)
            if not params:
                is_24h = room.time_limit >= 7200
                is_split = (room.game_type == 'split')
                from spinner_set import SpinnerSet
                room.next_spinner_params = SpinnerSet.generate_params(room.board_dimensions, is_24h, is_split, previous_params=getattr(room, 'spinner_params', None))
                params = room.next_spinner_params
            from spinner_set import SpinnerSet
            params = SpinnerSet.sanitize_params(params, room.board_dimensions, room.time_limit >= 7200)
            room.next_spinner_params = params
            launched_generated_at = params.get('generated_at') if params else None
            
            if not params:
                # If still no params, we must wait or fail to avoid bleeding from previous round
                print(f"[RoomManager] WARNING: No next_spinner_params for {room_id}. Waiting for generation...")
                return False
                
            if room.time_limit >= 7200:
                if params:
                    params['board_format'] = 'Valued Letters'
                    params['word_count_range'] = '200-300'
                fmt = 'Valued Letters'
                wc_range = '200-300'
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
            
            # PRE-VENT REPEATS: Add to history immediately so the 200/500+ retry loops don't pick it again for other rooms
            if not hasattr(room, 'bonus_word_history'):
                room.bonus_word_history = []
            if bonus_word and bonus_word not in room.bonus_word_history:
                room.bonus_word_history.append(bonus_word)
                if len(room.bonus_word_history) > 20:
                    room.bonus_word_history.pop(0)
            
            print(f"[RoomManager] Bonus word selected for next round: '{bonus_word}'")
            room.board_search_started = True
            search_round = room.current_round
            
            # Start board generation in background thread
            def generate_in_background():
                nonlocal bonus_word, params, launched_generated_at
                
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
                    if use_aw_flag:
                        min_accept = max(300, min_accept)
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
                            bonus_word_len=len(bonus_word) if bonus_word else None
                        )
                        if not candidate:
                            break
                        _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _fparams = candidate
                        _fw_filtered = [w for w in _fw if len(w) >= search_min]
                        is_diff_match = True
                        if search_diff == 'Hard' and _fr < 0.35:
                            is_diff_match = False
                        elif search_diff == 'Easy' and _fr > 0.25:
                            is_diff_match = False

                        if is_board_count_valid(len(_fw_filtered), search_wc) and len(_fw_filtered) >= min_accept and is_diff_match:
                            # Accept board without discarding valid grid words
                            _fw = _fw_filtered
                            _fp = {w: p for w, p in _fp.items() if w in _fw}
                            _pre = (_fb, _fw, _fc, _ff, _fp, _fr, _fbw, _fparams)
                            break
                        else:
                            print(f"[start_board_search] Cache pop candidate (words={len(_fw_filtered)}, ratio={_fr}) did not match range '{search_wc}' / diff '{search_diff}'. Discarding and retrying...")
                            
                    if _pre:
                        _board, _words, _bonus_c, _fmt, _paths, _ratio, _bonus_word, _params = _pre
                        print(f"[RoomManager] INSTANT CACHE HIT in board search for {room_id}: {len(_words)} words")
                        # Preserve spun parameters generated by SpinnerSet — do not overwrite with cached board params
                        params = dict(params)
                        room.next_spinner_params = params
                        wc_label = search_wc or '100-200'
                        params['word_count_range'] = wc_label
                        if room.next_spinner_params:
                            room.next_spinner_params['word_count_range'] = wc_label
                        _use_word = _bonus_word or bonus_word
                        if _use_word:
                            params['bonus_word_length'] = len(_use_word)
                            if room.next_spinner_params:
                                room.next_spinner_params['bonus_word_length'] = len(_use_word)
                        room.next_round_words = _words
                        room.next_round_word_paths = _paths
                        room.next_round_bonus_cell = _bonus_c
                        room.next_round_bonus = _use_word
                        room.next_round_format = _fmt
                        room.next_round_uniqueness = _ratio
                        room.next_round_spinner_params = dict(params)
                        room.next_round_spinner_params['board_format'] = _fmt
                        room.next_round_spinner_params['word_count_range'] = wc_label
                        actual_wc = len(_words)
                        room.next_round_total_words_count = actual_wc
                        room.next_round_board = _board
                        room.solving_complete = True
                        room.board_search_loading = False
                        room.board_search_started_actual = False
                        self.pre_generate_next_round(room_id)
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

                        if not is_duplicate:
                            break
                        
                        print(f"[RoomManager] WARNING: Generated board for room {room_id} is IDENTICAL to current/previous round board. Retrying...")
                        board_attempts += 1
                    
                    # Update word to the ACTUAL embedded word if different (MANDATORY consistency)
                    if final_bonus_word and len(final_bonus_word) >= 6:
                        bonus_word = final_bonus_word
                    else:
                        scorable = [w for w in all_words if len(w) >= 6]
                        if scorable:
                            bonus_word = sorted(scorable, key=len, reverse=True)[0]
                        else:
                            bonus_word = self._get_bonus_word(length=8, dictionary=search_dict)
                    
                    if bonus_word:
                        params['bonus_word_length'] = len(bonus_word)
                        if getattr(room, 'next_spinner_params', None):
                            room.next_spinner_params['bonus_word_length'] = len(bonus_word)
                        if getattr(room, 'spinner_params', None):
                            room.spinner_params['bonus_word_length'] = len(bonus_word)
                    
                    # ATOMIC STAGING PROMOTION: Set metadata FIRST and board LAST to prevent stale data race
                    if room.state != 'intermission':
                        print(f"[RoomManager] Background board generation for {room_id} finished while room state is '{room.state}'. SKIPPING assignment to prevent mid-round swap!")
                        return
                    
                    room.next_round_words = all_words
                    room.next_round_word_paths = all_words_dict
                    room.next_round_bonus_cell = bonus_cell
                    room.next_round_bonus = bonus_word
                    room.next_round_format = updated_format
                    room.next_round_uniqueness = u_ratio
                    # Check if generator relaxed min_word_length
                    if all_words:
                        actual_shortest = min(len(w) for w in all_words)
                        if actual_shortest < search_min:
                            print(f"[RoomManager] Word generator relaxed min_word_length from {search_min} to {actual_shortest} in background. Updating room param.")
                            params['min_word_length'] = actual_shortest

                    # USER REQUEST: Absolute consistency. Bundle the EXACT params used for this board.
                    room.next_round_spinner_params = dict(params)
                    room.next_round_spinner_params['board_format'] = updated_format # In case generator changed it
                    # FAST INITIALIZATION: Length-based scores to avoid "0 point" flickering in UI
                    # (Refined in background scoring loop below)
                    is_valued = ('valued' in str(updated_format).lower())
                    is_valued = ('valued' in str(updated_format).lower())
                    scored_dict = {}
                    for word in (all_words or []):
                        if is_valued:
                            v_score = get_valued_word_score(word)
                            scored_dict[word] = {'total': v_score, 'base': v_score}
                        else:
                            length = len(word)
                            s = 1 if length <= 4 else (2 if length == 5 else (3 if length == 6 else (5 if length == 7 else 11)))
                            scored_dict[word] = {'total': s, 'base': s}
                    
                    room.next_round_word_scores = scored_dict
                    room.next_round_total_points = sum(pts['total'] for pts in scored_dict.values())
                    room.next_round_total_words_count = len(all_words or [])
                    
                    # USER REQUEST: Pre-calculate counts for the Remaining tab update
                    next_counts = {i: 0 for i in range(1, 31)}
                    display_min = params.get('min_word_length', 3)
                    for word in (all_words or []):
                        l = len(word)
                        if display_min <= l <= 30:
                            next_counts[l] += 1
                    next_counts['_round'] = room.current_round + 1
                    room.next_round_counts_by_len = next_counts
                    
                    # RE-VERIFY: Ensure we are still updating the ACTIVE room instance
                    # (Prevents data loss if the room was reconstructed during search)
                    target_room = room
                    active_room = self.get_room(room_id)
                    if active_room and active_room is not room:
                         print(f"[RoomManager] Search complete for {room_id}, but room was reconstructed. Redirecting to active instance.")
                         target_room = active_room

                    # STALE PARAMETERS CHECK:
                    # If parameters have been re-spun/changed since this search started, discard.
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
                                print(f"[RoomManager] Stale board search discarded for {room_id} because parameters were re-spun (launched: {launched_generated_at}, current: {curr_generated_at})")
                                return
                            else:
                                print(f"[RoomManager] Parameter timestamps drifted by {time_drift:.4f}s but configuration is identical. Keeping board.")


                    # STALE BOARD SEARCH PROTECTION:
                    # If target_room's current_round is greater than search_round,
                    # then the round transition has already occurred and this background board is stale.
                    if target_room.current_round > search_round:
                        print(f"[RoomManager] Stale board search discarded for {room_id} (search_round: {search_round}, current_round: {target_room.current_round})")
                        return
                    
                    # IMMUTABLE ACTIVE BOARD LOCK:
                    # If target_room is currently in 'active' state for search_round,
                    # its active board is locked and CANNOT be swapped mid-round!
                    if target_room.state == 'active' and target_room.current_round == search_round:
                        print(f"[RoomManager] Background board search finished for active round {search_round} of {room_id}. Active board is locked. Discarding mid-round swap.")
                        return

                    target_room.next_round_board = board # SIGNAL READY IMMEDIATELY!
                    
                    # FIRST ROUND / LATE SYNC: If the round started while we were searching,
                    # promote the fast metrics to the ACTIVE state immediately to avoid 0-point displays.
                    if room.board == board:
                        room.solved_words_with_scores = scored_dict
                        room.recalculate_total_points()
                        print(f"[RoomManager] Board {room_id} fast-synced to ACTIVE round ({room.total_points_count} pts)")
                    else:
                        print(f"[RoomManager] Board {room_id} signal-ready (Fast metrics applied: {room.next_round_total_points} pts)")

                    # BACKGROUND REFINEMENT: Detailed scoring (Scoring bonuses, paths, etc.)
                    from scoring import calculate_word_score
                    def refine_scores():
                        try:
                            # Stale parameter check
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


                            # Stale refinement check
                            if target_room.current_round > search_round:
                                return
                            refined = {}
                            for word in (all_words or []):
                                path_v = _get_word_path(all_words_dict, word)
                                refined[word] = calculate_word_score(
                                    word, bonus_word, path=path_v,
                                    board_format=updated_format, bonus_cell=bonus_cell,
                                    board=board, return_details=True, strict_path=True
                                )
                            # Sync both staging and active (if round started during refinement)
                            if room.next_round_board == board:
                                room.next_round_word_scores = refined
                                room.next_round_total_points = sum(
                                    (pts.get('total', 0) if isinstance(pts, dict) else pts) 
                                    for pts in refined.values()
                                )
                            if room.board == board:
                                room.solved_words_with_scores = refined
                                room.recalculate_total_points()
                            print(f"[RoomManager] Board {room_id} scoring refinement complete. Next Round Total: {getattr(room, 'next_round_total_points', 0)}")
                        except Exception as e:
                            print(f"[RoomManager] Refinement error for {room_id}: {e}")
                    
                    threading.Thread(target=refine_scores, daemon=True).start()
                        
                    # 3. Density Initialization
                    try:
                        room.initialize_density(board, all_words_dict, updated_format, is_staging=True)
                    except Exception as e:
                        print(f"[Density-Diag] Failed to initialize staging: {e}")
                    
                    # SYNC FACT TO INTENT: Update both the staging area AND the revealed UI slot.
                    if getattr(room, 'next_spinner_params', None):
                        b_dims = room.board_dimensions.split('x')
                        d_val = int(b_dims[0]) if len(b_dims) == 3 else 1
                        rows = int(b_dims[1] if len(b_dims) == 3 else b_dims[0])
                        cols = int(b_dims[2] if len(b_dims) == 3 else b_dims[1])
                        achieved_diff = self.board_generator.get_difficulty_label(u_ratio, rows, cols, search_dict, depth=d_val, board=board, target_difficulty=room.next_spinner_params.get('difficulty'), min_word_length=room.next_spinner_params.get('min_word_length', 3))
                        # Wait to calculate achieved_wc until AFTER authoritative truncation
                        
                        # Frontend handles appending uniqueness percentage to difficulty label
                        
                        # AUTHORITATIVE SYNC: Ensure factual counts are promoted.
                        # The generator already enforces compliance, so we use the full list to avoid biasing length distribution.
                        room.next_round_words = list(all_words)
                        room.next_round_word_paths = all_words_dict
                        room.next_round_total_words_count = len(all_words)
                        
                        # Re-calculate total points if already set
                        if hasattr(room, 'next_round_total_points'):
                            room.next_round_total_points = sum(
                                (pts.get('total', 0) if isinstance(pts, dict) else pts) 
                                for w, pts in room.next_round_word_scores.items() if w in all_words
                            )
                        
                        # Sync scores list to match the word list (Safety check)
                        if hasattr(room, 'next_round_word_scores'):
                            room.next_round_word_scores = {w: room.next_round_word_scores[w] for w in all_words if w in room.next_round_word_scores}
                            
                        # Update metadata for the pending round
                        # FIX: compute achieved_wc from the MIN_LEN-filtered count so it matches
                        # exactly what the round will play. Using len(all_words) (unfiltered) caused
                        # the Spinner Set to promise e.g. '200-300' while the round played '100-200'
                        # because short words were counted but later excluded by min_word_length.
                        planned_wc = room.next_spinner_params.get('word_count_range', '100-200') if getattr(room, 'next_spinner_params', None) else '100-200'
                        max_cap = None
                        if planned_wc == '50-100': max_cap = 99
                        elif planned_wc == '100-200': max_cap = 199
                        elif planned_wc == '200-300': max_cap = 299
                        elif planned_wc == '300-400': max_cap = 399
                        elif planned_wc == '400-500': max_cap = 499

                        if max_cap and len(all_words) > max_cap:
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
                            room.next_round_words = all_words
                            room.next_round_word_paths = {w: all_words_dict[w] for w in all_words if w in all_words_dict}
                            room.next_round_total_words_count = len(all_words)

                        _sb_min_len = (room.next_spinner_params.get('min_word_length', 3)
                                       if getattr(room, 'next_spinner_params', None) else 3)
                        _sb_filtered_cnt = sum(1 for w in (all_words or []) if len(w) >= _sb_min_len)
                        achieved_wc = self._get_factchecked_wc_range(_sb_filtered_cnt, use_added_words=use_aw_flag)

                        # FIX: Detect when cached board was built for a HIGHER min_word_length.
                        # e.g. board from cache has only 8L+ words, but Spinner Set promised 6L.
                        # Update _sb_min_len so all downstream param stores show the real minimum.
                        if all_words:
                            _actual_min_board = min(len(w) for w in all_words)
                            if _actual_min_board > _sb_min_len:
                                print(f"[RoomManager] Min-length correction in start_board_search: "
                                      f"planned={_sb_min_len}, actual={_actual_min_board}. Updating params.")
                                _sb_min_len = _actual_min_board
                                if getattr(room, 'next_spinner_params', None):
                                    room.next_spinner_params['min_word_length'] = _sb_min_len
                                if not getattr(room, '_spinner_params_locked', False):
                                    room.spinner_params['min_word_length'] = _sb_min_len

                        if getattr(room, 'next_spinner_params', None):
                            room.next_spinner_params['board_format'] = updated_format
                            
                            if getattr(room, 'frozen_revealed_params', None) and isinstance(room.frozen_revealed_params, dict):
                                room.frozen_revealed_params['min_word_length'] = _sb_min_len
                            
                            # Only sync to active spinner_params if not yet locked for the round
                            if not getattr(room, '_spinner_params_locked', False):
                                room.spinner_params['board_format'] = updated_format
                        # REVEAL SYNC: Pre-calculate counts by length for the revelation phase
                        # This avoids the "Remaining tab lag" where it shows previous round stats
                        # Always calculate 1-30 to ensure valid data regardless of min-length transitions
                        # TAG with next round ID so frontend knows this is the Teaser data
                        room.next_round_counts_by_len = {
                            '_round': room.current_round + 1,
                            **{str(l): sum(1 for w in (all_words or []) if len(w) == l) for l in range(1, 31)}
                        }
                        # SYNC: If the board generator picked a natural fallback bonus word of a different length, update the UI params.
                        if final_bonus_word and getattr(room, 'next_spinner_params', None):
                            room.next_spinner_params['bonus_word_length'] = len(final_bonus_word)
                            if not getattr(room, '_spinner_params_locked', False):
                                room.spinner_params['bonus_word_length'] = len(final_bonus_word)
                        if getattr(room, 'next_spinner_params', None):
                            room.next_spinner_params['uniqueness'] = u_ratio
                            room.next_spinner_params['difficulty'] = achieved_diff
                            room.next_round_spinner_params = dict(room.next_spinner_params)
                        room.next_round_difficulty = achieved_diff
                        
                        # Authoritative recount after truncation (if any)
                        # Uses the same _sb_min_len computed above for achieved_wc consistency.
                        room.next_round_total_words_count = _sb_filtered_cnt
                        print(f"[RoomManager] Background pre-gen complete for {room_id}: {achieved_diff} | {updated_format} (Count: {len(all_words)})")
                        
                    # NOTE: Do NOT update room.current_difficulty here!
                    # Updating it now would flicker the active round header for players still in the round.
                    # It will be applied atomically when the round transition occurs.

                    # Double Lockdown - Ensure bonus_cell is strictly None for non-special formats
                    f_low = str(updated_format).lower()
                    if 'bonus letter' not in f_low and 'either' not in f_low:
                         room.next_round_bonus_cell = None
                         
                    # User requirement: "When you show the Spinner Set Popup, that means you have found a board."
                    room.spinner_params_generated = True
                    
                    # PERFORMANCE CACHE: Categorize words once per round
                    # USER REQUEST: Filter by min_len strictly.
                    # HARD FLOOR: Always exclude 3-letter words from the final solution display (User Request: "NOT 3 letter wrods")
                    min_l = room.next_spinner_params.get('min_word_length', 3) if getattr(room, 'next_spinner_params', None) else 3
                    display_min = min_l
                    filtered_all = [w for w in (all_words or []) if len(w) >= display_min]
                    
                    if str(search_dict).upper() in ['CSW', 'AW', 'ALL', 'ADDED_WORDS']:
                        word_validator.word_validator.ensure_csw_loaded()
                    room.next_round_csw_only_words = [w for w in filtered_all if word_validator.word_validator.is_csw_only(w)]
                    room.next_round_added_words = [w for w in filtered_all if word_validator.word_validator.is_added_word(w)]
                    if room.next_round_added_words:
                        print(f"[RoomManager {room.room_id}] Pre-gen Round {room.current_round + 1} generated {len(room.next_round_added_words)} custom added words: {room.next_round_added_words}")
                    
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
                # Board already exists — this 'waiting' state was a false-positive from the
                # heartbeat's empty-room pause. Abort to prevent board re-roll.
                print(f"[RoomManager] Aborting start_next_round for {room_id}: state='waiting' but board already active (heartbeat false-positive).")
                return False
            room.state = 'intermission'  # Canonical path: waiting → intermission → active
            room.intermission_start_time = time.time() - 60  # Force it to look expired
            
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
                    elif '50' in str(target_range):  min_accept = 50
            min_accept = max(30, min_accept)

            # INSTANT 0:00 TRANSITION: Always deliver staged board or popped cached board in <1ms!
            if not getattr(room, 'next_round_board', None):
                from board_generator import pop_compatible_cached_board, pop_any_cached_board
                sp = room.spinner_params or {}
                dict_val = sp.get('dictionary', 'NWL')
                fmt_val = sp.get('board_format', 'Normal')
                use_aw_val = sp.get('use_added_words', False) or '+ AW' in str(dict_val).upper() or '+AW' in str(dict_val).upper()
                bonus_word_len = sp.get('bonus_word_length')
                
                candidate = pop_compatible_cached_board(
                    room.board_dimensions, dict_val, fmt_val, m_len, use_aw_val, bonus_word_len=bonus_word_len
                )
                if not candidate:
                    candidate = pop_any_cached_board(room.board_dimensions)
                if not candidate:
                    print(f"[start_next_round] Cache miss for {room_id}. Generating instant emergency fallback board.")
                    candidate = get_emergency_fallback_board(
                        room.board_dimensions, fmt_val, room.time_limit,
                        dictionary=dict_val, use_added_words=use_aw_val, target_range=target_range, min_word_length=m_len
                    )
                
                if candidate:
                    if len(candidate) >= 9:
                        _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _, _fparams = candidate
                    else:
                        _fb, _fw, _fc, _ff, _fp, _fr, _fbw, _fparams = candidate
                    
                    _fw_filtered = [w for w in _fw if len(w) >= m_len]
                    if not _fw_filtered: _fw_filtered = _fw
                    _fp_filtered = {w: p for w, p in (_fp or {}).items() if w in _fw_filtered}
                    
                    # SAFEGUARD: If Checkerboard format, enforce strict C/V alternation
                    if 'checkerboard' in str(fmt_val).lower() or 'checkerboard' in str(_ff).lower():
                        self.board_generator._verify_checkerboard_safeguard(_fb)

                    print(f"[start_next_round] Instant 1-ms cache pop for {room_id}: {len(_fw_filtered)} words")
                    room.next_round_board = _fb
                    room.next_round_words = _fw_filtered
                    room.next_round_bonus_cell = _fc
                    room.next_round_bonus = _fbw or ''
                    room.next_round_word_paths = _fp_filtered
                    bw_l_val = room.spinner_params.get('bonus_word_length', 8) if isinstance(room.spinner_params, dict) else 8
                    dict_val = room.spinner_params.get('dictionary', 'NWL') if isinstance(room.spinner_params, dict) else 'NWL'
                    if not _fbw or str(_fbw).upper() == 'NONE':
                        _fbw = self._get_bonus_word(length=bw_l_val, dictionary=dict_val)
                    room.next_round_bonus = _fbw
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
                        'performance_efficiency': getattr(p, 'performance_efficiency', 0)
                    })
            
            # USER REQUEST: Word Tally. Capture unique words found by each player in this round.
            # We must do this BEFORE promotion resets p.submitted_words.
            ghost_player_words = {p['username']: [w['word'] for w in p['submitted_words']] for p in ghost_player_snapshots}
            
            # DIAGNOSTIC: Log the capture
            total_captured = sum(len(ws) for ws in ghost_player_words.values())
            print(f"[RoomManager] Snapshot captured for {room.room_id}: {total_captured} words, {len(ghost_player_snapshots)} players, Round {ghost_round_num}")
            for u, ws in ghost_player_words.items():
                # ws is a list of strings (the words)
                submitted_strings = [w.upper() for w in ws]
            
            # 2. STATE TRANSITION LOCK: Perform the atomic board swap
            with room._state_lock:
                room._did_6x8_fallback_rescue = False
                room._did_050_fallback_rescue = False
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
                room.use_added_words = active_params.get('use_added_words', False)
                raw_min = active_params.get('min_word_length', 3)
                
                # Update spinner_params to match the actual board being used
                room.spinner_params = dict(active_params) if active_params else {}

                active_uniq = active_params.get('uniqueness') if isinstance(active_params, dict) else None
                next_uniq = active_uniq if active_uniq is not None else getattr(room, 'next_round_uniqueness', None)
                if next_uniq is not None:
                    room.current_uniqueness = float(next_uniq)
                    if isinstance(room.spinner_params, dict):
                        room.spinner_params['uniqueness'] = room.current_uniqueness
                    dims_parts = str(room.board_dimensions).lower().split('x')
                    r_cnt = int(dims_parts[0]) if len(dims_parts) >= 2 else 4
                    c_cnt = int(dims_parts[1]) if len(dims_parts) >= 2 else 4
                    room.current_difficulty = self.board_generator.get_difficulty_label(
                        room.current_uniqueness, rows=r_cnt, cols=c_cnt, dictionary=room.current_dictionary, min_word_length=raw_min
                    )
                try:
                    room.current_min_length = int(raw_min)
                except:
                    room.current_min_length = 3

                # --- 2. BOARD & WORD PROMOTION ---
                # EMERGENCY SAFETY: If for any reason staging is empty, force a fast fallback board NOW.
                # CRITICAL: Never call generate_board() synchronously here — it blocks for 10-30s.
                # get_emergency_fallback_board() is always instant: cache-first, then pre-built board.
                if not room.next_round_board or not room.next_round_words:
                    print(f"[REMAINING-STABILIZER] Staging empty for {room_id} at promotion. Forcing INSTANT emergency fallback (cache or pre-built).")
                    if room.time_limit >= 7200:
                        room.current_board_format = 'Valued Letters'
                        room.current_dictionary = room.spinner_params.get('dictionary', 'NWL')
                        room.current_min_length = int(room.spinner_params.get('min_word_length', 3))

                    e_min_len = room.spinner_params.get('min_word_length') if room.spinner_params else None
                    e_diff = room.spinner_params.get('difficulty', 'Medium') if room.spinner_params else 'Medium'
                    e_fmt_target = room.spinner_params.get('board_format', 'Normal') if room.spinner_params else 'Normal'
                    e_target_range = room.spinner_params.get('word_count_range', '100-200') if room.spinner_params else '100-200'
                    # Use the always-instant emergency fallback (cache → pre-built). Never blocks.
                    e_fallback = get_emergency_fallback_board(
                        room.board_dimensions, e_fmt_target, room.time_limit,
                        dictionary=room.current_dictionary,
                        use_added_words=getattr(room, 'use_added_words', False),
                        target_range=e_target_range,
                        min_word_length=e_min_len or 3,
                        difficulty=e_diff
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
                        room.spinner_params['dictionary'] = e_params.get('dictionary', 'NWL')
                        room.spinner_params['board_format'] = e_params.get('board_format', 'Normal')
                        room.spinner_params['min_word_length'] = e_params.get('min_word_length', 3)
                        room.spinner_params['use_added_words'] = e_params.get('use_added_words', False)
                        room.spinner_params['bonus_word_length'] = len(e_bonus_word) if e_bonus_word else e_params.get('bonus_word_len', 6)
                        
                        room.current_dictionary = e_params.get('dictionary', 'NWL')
                        room.current_board_format = e_params.get('board_format', 'Normal')
                        room.current_min_length = e_params.get('min_word_length', 3)
                        room.use_added_words = e_params.get('use_added_words', False)
                    
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
                        if str(getattr(room, 'current_dictionary', 'NWL')).upper() in ['CSW', 'AW', 'ALL', 'ADDED_WORDS']:
                            word_validator.word_validator.ensure_csw_loaded()
                        room.next_round_csw_only_words = [w for w in e_words if word_validator.word_validator.is_csw_only(w)]
                        room.next_round_added_words = [w for w in e_words if word_validator.word_validator.is_added_word(w)]
                    else:
                        room.next_round_csw_only_words = []
                        room.next_round_added_words = []
                    
                    # USER REQUEST: Ensure Total Points is never 0.
                    # Fast-apply length based scores for the emergency board immediately.
                    is_valued_e = ('valued' in str(room.current_board_format).lower())
                    e_scores = {}
                    for w in e_words:
                        if is_valued_e: e_scores[w] = {'total': get_valued_word_score(w), 'base': get_valued_word_score(w), 'bonus_word_points': 0, 'bonus_letter_points': 0, 'either_or_points': 0}
                        else:
                            length = len(w)
                            s = 0
                            if length <= 2: s = 0
                            elif length <= 4: s = 1
                            elif length == 5: s = 2
                            elif length == 6: s = 3
                            elif length == 7: s = 5
                            elif length >= 8: s = 11
                            e_scores[w] = {'total': s, 'base': s, 'bonus_word_points': 0, 'bonus_letter_points': 0, 'either_or_points': 0}
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
                    import copy
                    room.previous_board = copy.deepcopy(ghost_prev_board) if ghost_prev_board else []
                    room.previous_all_words = [w for w in (ghost_source_words or []) if len(w) >= ghost_min_len]
                    room.previous_all_words_paths = dict(ghost_all_words_paths)
                    room.previous_all_word_scores = dict(getattr(room, 'solved_words_with_scores', {}))
                    room.previous_csw_only_words = list(room.csw_only_words) if getattr(room, 'csw_only_words', None) else []
                    room.previous_added_words = list(room.added_words) if getattr(room, 'added_words', None) else []
                
                # Update current active counts
                room.csw_only_words = getattr(room, 'next_round_csw_only_words', [])
                room.added_words = getattr(room, 'next_round_added_words', [])

                # Defer heavy CSW/AW tagging so state promotion completes in < 0.1ms
                target_words_list = list(room.next_round_words or room.all_words or [])
                def _tag_supplemental_words(rm_ref, w_list, dict_str, use_aw):
                    try:
                        if ('CSW' in str(dict_str).upper() or 'ALL' in str(dict_str).upper()) and hasattr(word_validator, 'word_validator'):
                            word_validator.word_validator.ensure_csw_loaded()
                            rm_ref.csw_only_words = [w for w in w_list if word_validator.word_validator.is_csw_only(w)]
                        if (use_aw or '+ AW' in str(dict_str).upper()) and hasattr(word_validator, 'word_validator'):
                            rm_ref.added_words = [w for w in w_list if word_validator.word_validator.is_added_word(w)]
                    except Exception as tag_err:
                        print(f"[RoomManager] Background word tagging error: {tag_err}")
                        
                threading.Thread(target=_tag_supplemental_words, args=(room, target_words_list, getattr(room, 'current_dictionary', 'NWL'), getattr(room, 'use_added_words', False)), daemon=True).start()

                # ATOMIC PROMOTION: Carry staging data to active room state
                room.board = room.next_round_board
                room.current_board_format = 'Valued Letters' if room.time_limit >= 7200 else active_params.get('board_format', 'Normal')
                
                # USER REQUEST: Absolute consistency. Only include words that meet the round's scorable minimum.
                # HARD FLOOR: Always exclude 3-letter words from the 'All Words' list (User Request: "NOT 3 letter wrods")
                min_l = room.current_min_length if hasattr(room, 'current_min_length') else 3
                display_min = min_l
                room.all_words_paths = {w: p for w, p in (room.next_round_word_paths or {}).items() if len(w) >= display_min}
                room.all_words = set(room.all_words_paths.keys())
                
                room.solved_words_with_scores = getattr(room, 'next_round_word_scores', {})
                
                # Save to DB will occur at the end of promotion after all parameters are finalized
                
                current_bw = getattr(room, 'next_round_bonus', '')
                bw_l = room.spinner_params.get('bonus_word_length', 8) if isinstance(room.spinner_params, dict) else 8
                dict_val = room.spinner_params.get('dictionary', 'NWL') if isinstance(room.spinner_params, dict) else 'NWL'
                target_words = room.next_round_words or room.all_words or []
                if not current_bw or current_bw not in target_words or str(current_bw).strip().upper() in ['', 'NONE'] or str(current_bw).upper().endswith('ING') or str(current_bw).upper().endswith('INGS') or len(str(current_bw).strip()) < 6:
                    candidates = [w for w in target_words if len(w) == bw_l and not w.upper().endswith('ING') and not w.upper().endswith('INGS')]
                    if not candidates:
                        candidates = [w for w in target_words if 6 <= len(w) <= 10 and not w.upper().endswith('ING') and not w.upper().endswith('INGS')]
                    if candidates:
                        # Using global random module
                        current_bw = random.choice(list(candidates)).upper()
                    else:
                        current_bw = self._get_bonus_word(length=max(6, bw_l), dictionary=dict_val, alternating=('checkerboard' in str(room.current_board_format).lower()))
                room.bonus_word = str(current_bw or '').upper().strip()
                
                room.bonus_cell = getattr(room, 'next_round_bonus_cell', None)

                # --- 4. ACCURACY ENFORCEMENT (inside lock — all_words must be consistent before state='active') ---
                # These loops run over ~100-500 words and take < 5ms — not the bottleneck.
                _sp     = dict(room.spinner_params) if isinstance(room.spinner_params, dict) else {}
                _bw_up  = str(getattr(room, 'bonus_word', '')).upper().strip()
                _dict   = room.current_dictionary
                _use_aw = getattr(room, 'use_added_words', False)

                # Min-length filter
                room.all_words = {w for w in room.all_words if len(w) >= room.current_min_length}

                # Range cap truncation
                target_sp_range = _sp.get('word_count_range', '100-200')
                max_cap = None
                if target_sp_range == '50-100':   max_cap = 99
                elif target_sp_range == '100-200': max_cap = 199
                elif target_sp_range == '200-300': max_cap = 299
                elif target_sp_range == '300-400': max_cap = 399
                elif target_sp_range == '400-500': max_cap = 499

                if max_cap and len(room.all_words) > max_cap:
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
                real_wc = self._get_factchecked_wc_range(wc_cnt, use_added_words=_use_aw)

                room.total_words_count    = wc_cnt
                room.initial_total_words  = wc_cnt
                room.current_word_count_range = real_wc

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
                                    'is_guest': p.is_guest, 'is_ai': p.is_ai, 'ai_rating': p.ai_rating, 'has_abandoned': p.has_abandoned
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
                                    room.current_word_count_range or ('200-300' if room.time_limit >= 7200 else '100-200'),
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
                room.frozen_revealed_params = None
                room.spinner_params_loading = False
                room.next_spinner_params = None
                room.next_round_spinner_params = None
                room.next_round_difficulty = None
                room.next_round_uniqueness = None
                room.board_search_started_actual = False
                room._spinner_params_locked = True   # Keep LOCKED during active round!
                room._initial_board_delivered = True  # ISSUE 6: mark first board has been delivered
                
                # Reset Round counters
                room.current_round += 1
                
                # --- FINAL CLEARANCE & NEXT LOG CHAIN ---
                # Clear staging data immediately to prevent stale exclusion or duplicate promotion
                room.next_round_board = None 
                room.next_round_words = []
                room.next_round_word_paths = {}
                room.next_round_word_scores = {}
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
                room.round_start_time = time.time()
                room.state = 'active'
                if room.time_limit >= 7200:
                    room.players = []
                    room.past_players = {}
                    room.spectators = []
                    print(f"[RoomManager] Reset 24h room {room_id} players for fresh new daily round.")
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
            threading.Thread(target=self.pre_generate_next_round, args=(room_id,), daemon=True).start()

            print(f"[RoomManager] SUCCESS: Transitioned room {room_id} to Round {room.current_round}")
            
            # --- ASYNCHRONOUS POST-TRANSITION TASKS ---
            def finalize_results():
                # This is offloaded to avoid blocking the main server thread
                try:
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
                    self.log_word_tally(room, ghost_player_words)
                    
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
            return False

        finally:
            # ABSOLUTE SAFETY: The flag MUST be cleared so the room can try again if we failed.
            with room._state_lock:
                 room.starting_round = False
    
    def _get_factchecked_wc_range(self, count, use_added_words=False, dictionary=None):
        """Map actual word count to the exact corresponding standard range bucket.
           Standard buckets: 50-100, 100-200, 200-300, 300-400, 400-500, 500+
        """
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 0
            
        if count >= 500: return '500+'
        if count >= 400: return '400-500'
        if count >= 300: return '300-400'
        if count >= 200: return '200-300'
        if count >= 100: return '100-200'
        return '50-100'

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

        # USER MANDATE: Never allow a word ending in ING or INGS to be a Bonus Word across ANY difficulty or format!
        import random
        valid_words = [w for w in words if not w.upper().endswith("ING") and not w.upper().endswith("INGS")]
        if not valid_words:
            all_dict_words = word_validator.csw_by_len.get(length, []) if dictionary == 'CSW' else word_validator.nwl_by_len.get(length, [])
            valid_words = [w for w in all_dict_words if not w.upper().endswith("ING") and not w.upper().endswith("INGS")]
        
        # PREVENT REPEATS: Filter out words in history or currently active in the room
        final_exclude = set()
        
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
        return result
    
    
    def save_round_history(self, room, board=None, all_words=None, bonus_word=None, player_snapshots=None, round_num=None, all_words_paths=None, round_start_time=None, board_format=None):
        """Save the results of the JUST COMPLETED round to the database"""
        # Determine target round number (use snapshot if provided, otherwise room's current)
        target_round = round_num if round_num is not None else room.current_round
        debug_log = f"[SAVE-ROUND-{room.room_id}-R{target_round}]"

        if room.is_solo:
            print(f"[RoomManager] SKIPPING history save for SOLO room {room.room_id}")
            with open(DEBUG_FLOW_PATH, 'a') as f:
                f.write(f"{debug_log} - ABORT (Solo)\n")
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
                tz = ZoneInfo("America/Chicago")
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
                
                # Board formats (Normal, Cube, Mania, etc.) are allowed for history
                # (Validation for rank/stats can be done at display time if needed)
                if is_500plus:
                     print(f"[RoomManager] SKIPPING history save for room {room.room_id} - 500+ is unranked.")
                     return
                     
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

                    # If a user gets a score of 0 and submitted no words, do not save
                    # (Unless it is the System placeholder for 24-hour rooms)
                    if u_score <= 0 and not u_submitted and u_id != -1 and u_name != 'System':
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
                        
                        words_data.append({
                            'word': w['word'],
                            'points': w.get('points', 0),
                            'timestamp': raw_time
                        })
                    
                    # Calculate Best Word
                    best_w_entry = max(u_submitted, key=lambda x: x.get('points', 0)) if u_submitted else None
                    best_word_text = best_w_entry['word'] if best_w_entry else None
                    best_word_val = best_w_entry.get('points', 0) if best_w_entry else 0

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

                    conn.execute('''
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

                    if room.time_limit >= 7200 and u_id != -1 and u_name != 'System' and u_score > 0:
                        canonical_id = f"24h_{room.board_dimensions}"
                        conn.execute('''
                            INSERT INTO daily_score_sums (user_id, room_id, score_sum)
                            VALUES (?, ?, ?)
                            ON CONFLICT(user_id, room_id) DO UPDATE SET score_sum = score_sum + excluded.score_sum
                        ''', (u_id, canonical_id, u_score))
            
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

            with open(DEBUG_FLOW_PATH, 'a') as f:
                f.write(f"{debug_log} - SUCCESS: Saved to DB\n")
        except Exception as e:
            print(f"[RoomManager] Error saving round history: {e}")
            with open(DEBUG_FLOW_PATH, 'a') as f:
                f.write(f"{debug_log} - FATAL ERROR: {e}\n")

    def log_word_tally(self, room, player_words):
        """
        Tally how many users found each CSW word and log it to a central file.
        Also maintains a global cumulative tally in word_stats.json.
        """
        try:
            # USER REQUEST: Do not include words found in 24h rooms in the word tally file
            if room.time_limit >= 7200:
                print(f"[WordTally] Skipping tally for 24h room: {room.room_id}")
                return

            import collections
            import json
            import datetime
            import os
            from word_validator import word_validator
            
            # 1. Efficient Tally: Number of unique USERS who found each word in THIS round
            word_counts = collections.Counter()
            for words in player_words.values():
                unique_words_for_user = set()
                for entry in words:
                    w = entry.get('word', '').upper() if isinstance(entry, dict) else str(entry).upper()
                    active_dict = getattr(room, 'current_dictionary', 'NWL')
                    if w and word_validator.is_valid_word(w, active_dict, use_added_words=getattr(room, 'use_added_words', False)):
                        unique_words_for_user.add(w)
                
                for w in unique_words_for_user:
                    word_counts[w] += 1
            
            if not word_counts:
                return

            # 2. Append per-round entry to audit log
            log_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'room_id': room.room_id,
                'round': room.current_round,
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
                                total_rares += 1
                                
            # Check rare limits (max 1 of each, max 3 total)
            for rl, rc in rare_counts.items():
                if rc > 1:
                    return {"error": f"Too many rare letters: '{rl}' occurs {rc} times (Max 1)", "success": False}
            if total_rares > 3:
                return {"error": f"Too many total rare letters: {total_rares} (Max 3)", "success": False}
                
            # Check standard abundance limits on non-Mania formats
            board_format = target.get('board_format', 'Normal')
            safe_format = str(board_format or "Normal").strip().upper()
            is_mania = "MANIA" in safe_format
            mania_letter = None
            if is_mania:
                parts = safe_format.split()
                if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isalpha():
                    mania_letter = parts[0]
                    
            total_cells = r_num * c_num
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
                    room.current_difficulty = f"{achieved_diff} ({int(u_ratio * 100)}%)"
                    room.spinner_params['difficulty'] = room.current_difficulty
                    room.spinner_params['uniqueness'] = u_ratio
                    room.current_uniqueness = u_ratio
                    
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

# Global instance
room_manager = RoomManager()
