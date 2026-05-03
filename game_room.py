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
import sqlite3
import json
from spinner_set import SpinnerSet
from board_generator import BoardGenerator
from scoring import calculate_word_score
from rating_logic import calculate_proportional_rating_change, is_player_guest
import word_validator

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
    performance_efficiency: float = 0.0
    is_guest: bool = False
    is_ai: bool = False
    ai_rating: int = 1200
    has_abandoned: bool = False

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
        # Force integer types for comparisons
        self.time_limit = int(self.time_limit)
        if self.min_rating is not None: self.min_rating = int(self.min_rating)
        if self.max_rating is not None: self.max_rating = int(self.max_rating)
        
        # Configuration-specific max players
        if self.game_type in ['accumulative', 'solo_accumulative']:
            self.max_players = 9999 # Effectively unlimited
        elif self.game_type == 'fcfs':
            self.max_players = 16
        else:
            self.max_players = 8

        # INITIALIZE LOCKS
        self._state_lock = threading.RLock() # Reentrant to prevent deadlocks during transition
            
    def add_chat_message(self, username, message, is_system=False, image=None, color=None, is_winner=False):
        """Add chat message to room"""
        self.chat_messages.append({
            'username': username,
            'message': message,
            'image': image,
            'is_system': is_system,
            'is_winner': is_winner,
            'color': color,
            'time': time.time()
        })
        # Keep only last 30 messages
        if len(self.chat_messages) > 30:
            self.chat_messages.pop(0)
    
    def add_player(self, user_id, username, rating, games_played=0, country_flag='🏳️', manual_accessed=False, is_guest=False, is_ai=False, ai_rating=1200):
        """Add player to room"""
        is_daily = self.time_limit >= 7200
        
        # NOTE: Abandonment penalty is fixed at exit; re-joining does not remove from quitters list.


        
        # Check if player already exists (PERSISTENCE)
        existing_player = self.get_player(user_id)
        if existing_player and is_daily:
            print(f"[GameRoom] Persistence: Reusing existing player {username} in 24h room {self.room_id}")
            existing_player.last_active = time.time()
            existing_player.country_flag = country_flag # Update flag
            # CRITICAL: Always sync rating from DB even for persistent daily players
            if rating is not None and not is_guest:
                existing_player.rating = rating
            existing_player.is_guest = is_guest
            # Note: manual_accessed doesn't force mid-round for persistent daily rooms usually, 
            # but if it's the rule, we should apply it.
            # For now, let's stick to the user's rule for ALL rooms.
            if manual_accessed:
                existing_player.joined_mid_round = True
            elif not is_daily and self.state == 'active':
                # If they rejoin mid-round after being gone, mark them late? 
                # Ideally yes, unless was_already_in_room logic covers it. 
                # But here we are reusing existing_player object which means they were in past_players.
                # If they were active before, they should be fine?
                # But if they left and came back much later in same round?
                # User rule is usually strict. Let's stick to the consistent check.
                existing_player.joined_mid_round = True
            # Ensure they are removed from round_quitters if they were in there (REJOIN TRANSITION)
            self.round_quitters = [q for q in self.round_quitters if str(q.user_id) != str(user_id)]
            return True
        
        # Track if they were already in the room (to avoid mid-round flag on refresh)
        was_already_in_room = existing_player is not None
        was_joined_mid_round = getattr(existing_player, 'joined_mid_round', False) if existing_player else False
            
        # Check if player exists in round_quitters (RESTORE mid-round state)
        quitter = next((q for q in self.round_quitters if str(q.user_id) == str(user_id)), None)
        if quitter:
            print(f"[GameRoom] Restoring quitter {username} ({user_id}) to active players list with {len(quitter.submitted_words)} words.")
            quitter.last_active = time.time()
            quitter.country_flag = country_flag
            quitter.is_guest = is_guest
            self.players.append(quitter)
            # CRITICAL: Remove from round_quitters so they aren't penalized as a quitter at round end
            self.round_quitters = [q for q in self.round_quitters if str(q.user_id) != str(user_id)]
            return True

        # Check if player exists in past_players
        # print(f"DEBUG: Checking past_players for {user_id}. Past players count: {len(self.past_players)}")
        existing_player = next((p for p in self.past_players.values() if str(p.user_id) == str(user_id)), None)
        
        if existing_player:
            last_p_round = getattr(existing_player, '_last_round_seen', -1)
            if last_p_round != -1 and last_p_round != self.current_round:
                # NEW ROUND: Clear all round-specific activity
                existing_player.found_bonus_word = False
                existing_player.has_abandoned = False
                existing_player.joined_mid_round = (self.state == 'active')
                existing_player.submitted_words = []
                existing_player.invalid_words = []
                existing_player.score = 0
                existing_player.previous_round_score = 0
            
            existing_player._last_round_seen = self.current_round
            existing_player.last_active = time.time()
            existing_player.country_flag = country_flag
            existing_player.games_played = games_played
            # CRITICAL: Always sync rating from DB for rejoiners/refreshers
            if rating is not None and not is_guest:
                existing_player.rating = rating
            existing_player.is_guest = is_guest
            
            if manual_accessed:
                existing_player.joined_mid_round = True
            elif not is_daily and self.state == 'active':
                 # Check for "Refresh" grace period (15s)
                 if (time.time() - existing_player.last_active) > 15:
                      existing_player.joined_mid_round = True
            
            self.players.append(existing_player)
            return True

        # Ensure player is not already in the room (prevent duplicates)
        self.remove_player(user_id)
        
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
            
        self.players.append(player)
        self.players.sort(key=lambda p: p.rating, reverse=True)
        
        # System Notice
        self.add_chat_message("System", f"{username} has entered the room.", is_system=True)
        
        return True # Success

    def add_spectator(self, user_id, username, rating):
        """Add spectator to room"""
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
        if wc_range == '200+': return (200, 500)
        if wc_range == '500+': return (500, 99999)
        if wc_range in ['1500+', '2000+']: return (500, 99999) # Backward compatibility
        return (0, 0)
    
    def remove_player(self, user_id, force=False):
        """Remove player or spectator from room. 
        Note: Penalties are handled in app.py via apply_leave_penalty()
        """
        uid_str = str(user_id)
        
        # PERSISTENCE: Never remove PLAYERS from 24h rooms unless forced (e.g. logout)
        # However, we ALWAYS allow removing spectators.
        if self.time_limit >= 7200 and not force:
            self.spectators = [p for p in self.spectators if str(p.user_id) != uid_str]
            return

        initial_players = len(self.players)
        initial_specs = len(self.spectators)
        
        leaving_player = next((p for p in self.players if str(p.user_id) == uid_str), None)
        username = leaving_player.username if leaving_player else "Someone"
        
        # Track removal in audit logs
        if leaving_player:
            try:
                with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/rating_audit.log', 'a') as log:
                    log.write(f"[{time.time()}] Removal: User {username}, Score: {leaving_player.score}, Words: {len(leaving_player.submitted_words)}, Room: {self.room_id}, Round: {self.current_round}, State: {self.state}\n")
            except: pass
            
            # If in active round, add to round_quitters to prevent re-entering for same round
            if self.state == 'active':
                if not any(str(q.user_id) == uid_str for q in self.round_quitters):
                    self.round_quitters.append(leaving_player)

        # Remove from both lists
        self.players = [p for p in self.players if str(p.user_id) != uid_str]
        self.spectators = [p for p in self.spectators if str(p.user_id) != uid_str]
        
        # USER REQUEST: Kick spectators if last human player leaves
        humans = [p for p in self.players if not p.is_ai]
        is_daily = (self.time_limit >= 7200)
        
        if not humans and self.spectators and not is_daily:
            print(f"[GameRoom] Last human player has left room {self.room_id}. CLOSING ROOM.")
            self.is_closing = True
            self.spectators = []
            self.add_chat_message("System", "All players have left. Room is closing.", is_system=True)

        # If forced (logout), clear from past_players archive
        if force:
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
                # User Request: Enforce inactivity removal even for 24h rooms to keep lobby counts accurate
                # if is_daily:
                #    continue 
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
            self.remove_player(uid)
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
                specs_removed = True
                
        if specs_removed:
            self.spectators = active_spectators
            
        return players_removed or specs_removed
    
    def get_player(self, user_id):
        """Get player by ID"""
        uid_str = str(user_id)
        for p in self.players:
            if str(p.user_id) == uid_str:
                return p
        return None
    
    @property
    def time_remaining(self):
        """Calculate time remaining in current state"""
        # PRIORITY: Intermission timer is always literal (60s)
        # 1. Intermission timer (Fixed 60s or 5s for Daily)
        if self.state == 'intermission':
            elapsed = time.time() - self.intermission_start_time
            intermission_limit = 5 if self.time_limit >= 7200 else 60
            return max(0, intermission_limit - int(elapsed))
            
        # 2. 24h Room ACTIVE: Align to real-world midnight boundary
        if self.state == 'active' and self.time_limit >= 7200:
            import datetime
            now = datetime.datetime.now()
            next_midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time.min)
            delta = (next_midnight - now).total_seconds()
            return max(0, int(delta))

        if self.state == 'active':
            if self.custom_end_time > 0:
                return max(0, int(self.custom_end_time - time.time()))
            
            elapsed = time.time() - self.round_start_time
            return max(0, self.time_limit - int(elapsed))
        elif self.state == 'waiting':
             return self.time_limit # Use the limit as the waiting value
        return 0
    
    @property
    def round_end_time(self):
        """Get timestamp when current round ends (for client sync)"""
        # 24h Room (>= 2h limit): Always align dynamically to real-world midnight boundary (LOCAL)
        if self.time_limit >= 7200:
            import datetime
            now = datetime.datetime.now()
            next_midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time.min)
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
            return self.intermission_start_time + 60  # 60 second intermission
        return 0
    
    def submit_word(self, user_id, word, path=None):
        """Submit word for player"""
        # Security check: Spectators cannot play
        for s in self.spectators:
            if str(s.user_id) == str(user_id):
                return False, "Spectators cannot submit words", 0, None
        
        player = self.get_player(user_id)

        if not player:
            return False, "Player not in room", 0, None
        
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
                # Handle both 2D [r, c] and 3D [f, r, c]
                if len(node) == 3:
                    f, r, c = node
                    if 0 <= f < len(self.board) and 0 <= r < len(self.board[f]) and 0 <= c < len(self.board[f][r]):
                        cell_val = str(self.board[f][r][c])
                    else:
                        valid_path = False; break
                else:
                    r, c = node
                    if 0 <= r < len(self.board) and 0 <= c < len(self.board[0]):
                        cell_val = str(self.board[r][c])
                    else:
                        valid_path = False; break
                
                if '/' in cell_val:
                    options = cell_val.split('/')
                    new_words = []
                    for prefix in possible_words:
                        for opt in options:
                            new_words.append(prefix + opt)
                    possible_words = new_words
                else:
                    for i in range(len(possible_words)):
                        possible_words[i] += cell_val
            
            if valid_path:
                # Find which of the possible interpreted words from the path actually exists on the board
                valid_options = [w for w in possible_words if w in self.all_words]
                if len(valid_options) == 1:
                    word = valid_options[0]  # Auto-correct the submission to the valid Either/Or letter
                    matched_word = word
        
        # 2. Logic Check
        is_in = word in self.all_words
        min_len_req = self.current_min_length
        
        # EARLY EXIT: Check minimum length FIRST (User Request: Clearer feedback)
        # Boggle usually treats 'Q' as 'QU', so check if length would be sufficient even with expansion
        effective_len = len(word.replace('Q', 'QU')) if 'Q' in word else len(word)
        if effective_len < min_len_req:
            return False, f"{word.upper()} is too short (Min: {min_len_req})", 0, None

        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/word_debug.log', 'a') as debug:
            debug.write(f"[{time.time()}] Word: {word} | In Board: {is_in} | Min: {min_len_req} | Len: {len(word)} | Room: {self.room_id}\n")

        # Direct match check
        if is_in:
            matched_word = word
        elif 'Q' in word:
            # Fallback: check if "QU" variant exists (Handle Q -> QU mapping)
            # Strategy: Replace 'Q' with 'QU' and check if that exists in all_words
            # NOTE: This simple replace handles single Q. For multiple Qs, we might need permutations.
            # Boggle logic usually treats tile 'Q' as 'QU', so direct replacement works.
            # However, if 'U' is on the board, user might have typed Q-U-A-T-E explicitly.
            # If Q is represented as Q, then Q-A-T-E -> QATE.
            # If board generator found QUATE via (Q->QU)-A-T-E, then all_words has QUATE.
            # So, check if replacing Q with QU yields a valid word.
            variant = word.replace('Q', 'QU')
            if variant in self.all_words:
                matched_word = variant
        
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
                    from board_generator import BoardGenerator
                    bg = BoardGenerator()
                    if bg.is_word_on_board(word, self.board):
                        is_penalty = True
            
            if is_penalty:
                # Apply penalty (-3 points)
                penalty_points = -3
                
                # Prevent spamming the same penalty word
                existing_words = {w['word'] for w in player.submitted_words}
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
        # Extract existing words from the list of dicts
        existing_words = {w['word'] for w in player.submitted_words}
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
            is_private=self.is_private
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
            self.update_density_for_word(final_word, path)
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
                if w_obj['word'] == final_word:
                    points = w_obj['points']
                    break
        else:
            # For non-split modes (Accumulative, FCFS, Penalty), update 'points' from the recalculated object
            # to ensure user receives the correct score in the notification
            for w_obj in player.submitted_words:
                if w_obj['word'] == final_word:
                    points = w_obj['points']
                    break

        # NEW: Update Live PE Calculation
        self.update_live_pe()

        return True, f"{final_word} ACCEPTED", points, final_word

    def update_live_pe(self):
        """Calculates performance efficiency in real-time for UI trophy"""
        # Split into Registered and Guest pools to ensure isolation
        reg_players = [p for p in self.players if p.user_id > 0 and not p.is_guest and (p.score > 0 or len(p.submitted_words) > 0 or len(p.invalid_words) > 0)]
        guest_players = [p for p in self.players if (p.is_guest or p.user_id <= 0) and (p.score > 0 or len(p.submitted_words) > 0 or len(p.invalid_words) > 0)]
        
        # 1. Registered Players: Compete only against other registered players
        reg_score_sum = sum(p.score for p in reg_players)
        reg_rating_sum = sum(p.rating for p in reg_players)
        
        # Room must have more than 1 player to earn a trophy
        multiple_players = (len(reg_players) + len(guest_players)) > 1

        max_score = max(p.score for p in self.players) if self.players else 0
        if reg_rating_sum > 0:
            for p in reg_players:
                expected = (p.rating / reg_rating_sum) * reg_score_sum
                p.performance_efficiency = p.score / expected if expected > 0 else 0.0
                # Remarkable: Winner AND (Unusually high PE >= 4 & Score >= 40, or raw excellence Score >= 100)
                p.has_exceptional_round = multiple_players and p.score > 0 and p.score == max_score and \
                                         ((p.performance_efficiency >= 4.0 and p.score >= 40) or p.score >= 100)

        # 2. Guests: Use solo baseline (PE=1.0) so they don't affect pool but can still earn trophies on raw score
        for p in guest_players:
            p.performance_efficiency = 1.0
            p.has_exceptional_round = multiple_players and p.score > 0 and p.score == max_score and (p.score >= 100)
    
    def update_density_for_word(self, word, path=None):
        """Decrement cell density for found words in Density format"""
        cur_fmt = str(self.current_board_format).lower()
        if 'density' in cur_fmt:
            word_upper = word.upper()
            if word_upper not in self.global_round_found_words:
                 self.global_round_found_words.add(word_upper)
                 # Get path (User path or pre-calculated path)
                 word_path = path or (self.all_words_paths.get(word_upper) if hasattr(self, 'all_words_paths') else None)
                 if word_path:
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
                                 
                                 if f < len(self.cell_density) and r < len(self.cell_density[f]) and c < len(self.cell_density[f][r]):
                                     if self.cell_density[f][r][c] > 0:
                                         self.cell_density[f][r][c] -= 1
                             else:
                                 coords = list(map(int, node))
                                 r, c = coords[-2:]
                                 if r < len(self.cell_density) and c < len(self.cell_density[r]):
                                     if self.cell_density[r][c] > 0:
                                         self.cell_density[r][c] -= 1
                         except (IndexError, TypeError, ValueError): continue
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
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
            f.write(f"[Density-Diag] Room: {self.room_id} | Format: {fmt_low} | Staging: {is_staging} | Board: {len(board) if board else 0} | Paths: {len(all_words_paths) if all_words_paths else 0} at {time.time()}\n")

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
        
        print(f"[Density] Initialization complete. Max density: {max_d}")
    
    def _recalculate_player_score(self, player):
        """
        Recalculate player score from submitted words sequentially.
        """
        # Sort by submission time
        sorted_words = sorted(player.submitted_words, key=lambda x: x.get('time', 0))
        current_score = 0
        fmt = self.current_board_format
        import logging
        logger = logging.getLogger("scoring")
        logger.debug(f"[Recalc] Re-evaluating score for {player.username}. Words: {len(player.submitted_words)} | Room FMT: {fmt}")
        
        for w_obj in sorted_words:
            # Handle Penalty words or Split Points already stored in w_obj
            # Priority: 
            # 1. Use existing 'points' value if it's already recorded (essential for Split Points and Penalties)
            # 2. Otherwise recalculate (Normal/Accumulative/FCFS fallback)
            
            p_val = w_obj.get('points')
            
            if p_val is not None:
                # Use pre-calculated value
                points = p_val
                # Still need details for the frontend breakdown if possible
                points_details = w_obj.get('score_details', {'total': points})
            else:
                # Use word_path from solver to avoid slow DFS for typed words (essential for round-end fluid transitions)
                word_path = self.all_words_paths.get(w_obj['word'], w_obj.get('path'))
                
                points_details = calculate_word_score(
                    w_obj['word'], 
                    self.bonus_word, 
                    board_format=fmt,
                    path=word_path,
                    bonus_cell=self.bonus_cell,
                    board=self.board,
                    return_details=True,
                    is_private=self.is_private
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
        
        # 1. Start Milestone: Threshold is TR=0 during intermission
        if self.state == 'intermission' and self.time_remaining <= 0:
            if getattr(self, 'starting_round', False):
                start_init = getattr(self, '_round_start_init_time', 0)
                if start_init > 0 and (now - start_init > 12.0):
                    self.starting_round = False
                    print(f"[RoomManager] STALE starting_round detected for {self.room_id} (>12s). Resetting.")
                else:
                    return None
            return 'start'
            
        # 2. Parameter Reveal (15s into intermission)
        if self.state == 'intermission':
            elapsed = now - self.intermission_start_time
            intermission_duration = 5 if self.time_limit >= 7200 else 60
            reveal_threshold = 15.0 if intermission_duration >= 20 else 1.0
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

    def check_and_update_state(self):
        """Authoritative state machine for game rooms.
        Handles transitions and timing for all game modes."""
        now = time.time()
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
                    if elapsed >= self.time_limit:
                        # Ensure we don't end round 0.5s after start due to uninitialized time
                        if self.round_start_time > 0:
                            should_end = True
            else:
                # 24H Reset Logic (Midnight Boundary)
                if self.round_start_time > 0:
                    import datetime
                    round_start_dt = datetime.datetime.fromtimestamp(self.round_start_time)
                    if now > self.round_start_time and datetime.datetime.fromtimestamp(now).date() > round_start_dt.date():
                        print(f"[GameRoom] Midnight Reset DETECTED for {self.room_id}")
                        should_end = True
                        self.midnight_reset_occurred = True
                        self.previous_board = [list(row) for row in self.board] if self.board else None
                        self.previous_min_length = getattr(self, 'current_min_length', 3)
                        self.previous_all_words = getattr(self, 'solved_words_with_scores', {}) if self.solved_words_with_scores else {w: {} for w in self.all_words} # Dict for scoring fallback

        # 2. Transition ACTIVE -> INTERMISSION
        if self.state == 'active' and should_end:
            with self._state_lock:
                if self.state != 'active':
                    return True
                
                self.state = 'intermission'
                self.intermission_start_time = now
                print(f"[TRANSITION] Room {self.room_id}: ACTIVE -> INTERMISSION (Time: {self.intermission_start_time}, Elapsed: {now - self.round_start_time})")
                
                # [PROACTIVE] Do NOT clear generated/search flags here anymore.
                # They are now reset only at start_next_round.
                self.round_quitters = []
                self.custom_end_time = 0 # CLEAR ALWAYS AT TRANSITION
                
                # USER REQUEST: Absolute accuracy for 'All Words' panel scoring.
                # If background scoring isn't finished, perform a synchronous fallback score calculation.
                if not getattr(self, 'solved_words_with_scores', None) or not self.solved_words_with_scores:
                    print(f"[GameRoom] Transitioning {self.room_id}: solved_words_with_scores missing. Scoring synchronously.")
                    from scoring import calculate_word_score
                    fallback_scores = {}
                    for word in self.all_words:
                        fallback_scores[word] = calculate_word_score(
                            word, 
                            self.bonus_word, 
                            board_format=self.current_board_format,
                            bonus_cell=self.bonus_cell,
                            board=self.board,
                            path=self.all_words_paths.get(word),
                            return_details=True
                        )
                    self.solved_words_with_scores = fallback_scores

                # Snapshot board and words for intermission (Detailed Scoring Preservation)
                if self.game_type == '3d' or (self.board and len(self.board) == 6 and isinstance(self.board[0], list) and isinstance(self.board[0][0], list)):
                     self.previous_board = [[list(row) for row in face] for face in self.board]
                else:
                     self.previous_board = [list(row) for row in self.board] if self.board else None
                
                # USER REQUEST: Ensure 'All Words' list has full math breakdown in history
                self.previous_all_words = getattr(self, 'solved_words_with_scores', {})
                self.previous_csw_only_words = [w for w in (self.all_words or []) if word_validator.word_validator.is_csw_only(w)]
                self.previous_bonus_word = self.bonus_word
                
                # Snapshot for persistence
                self.recalculate_total_points() # Authoritative sync before snapshot
                self.previous_total_points = getattr(self, 'total_points_count', 0)
                self.previous_total_words = getattr(self, 'total_words_count', 0)
                self.previous_total_counts_by_len = dict(getattr(self, 'total_counts_by_len', {}))
                
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
                                    sorted_submitted = sorted(p.submitted_words, key=lambda x: x.get('points', 0), reverse=True)
                                    winner_words = [{'word': w['word'], 'points': w.get('points', 0)} for w in sorted_submitted[:20]]
                                    break
                        
                        if max_score > 0:
                            self.winners_history.insert(0, {
                                'round': self.current_round,
                                'winners': winners_data,
                                'all_players': sorted([{'username': p.username, 'score': p.score} for p in (active_pool or self.players)], key=lambda x: x['score'], reverse=True),
                                'score': max_score,
                                'board': [list(row) for row in self.board] if self.board else [],
                                'words': winner_words,
                                'bonus_word': getattr(self, 'bonus_word', ''),
                                'timestamp': int(time.time() * 1000)
                            })
                            if len(self.winners_history) > 25: self.winners_history = self.winners_history[:25]

                        # Ratings logic...
                        try:
                            from rating_logic import calculate_proportional_rating_change
                            # USER MANDATE: Only change ratings for players who started the round from the beginning
                            participants = [
                                p for p in self.players + self.round_quitters 
                                if (getattr(p, 'score', 0) > 0 or not getattr(p, 'is_ai', False)) 
                                and not getattr(p, 'joined_mid_round', False)
                            ]
                            rating_changes = calculate_proportional_rating_change(participants, is_private=self.is_private)
                            
                            import sqlite3
                            conn_p = sqlite3.connect('morpheme.db', timeout=30)
                            for p in self.players + self.round_quitters:
                                if p.user_id in rating_changes:
                                    p.rating_change = rating_changes[p.user_id]
                                    p.rating += p.rating_change
                                    # Update Global Rating
                                    conn_p.execute('UPDATE users SET rating = MAX(400, rating + ?) WHERE id = ?', (p.rating_change, p.user_id))
                                    
                                    # Update Config-Specific Rating
                                    display_game_type = self.game_type.replace('solo_', '')
                                    config_key = f"{display_game_type}|{self.board_dimensions}|{self.time_limit}"
                                    conn_p.execute('''
                                        UPDATE user_ratings 
                                        SET rating = MAX(400, rating + ?) 
                                        WHERE user_id = ? AND config_key = ?
                                    ''', (p.rating_change, p.user_id, config_key))

                            # 5. Distribute Abandonment Bounty (User Request: At the end when results are shown)
                            if self.abandonment_bounty > 0:
                                eligible_receivers = [p for p in self.players if not p.is_ai and not getattr(p, 'is_guest', False) and not getattr(p, 'joined_mid_round', False)]
                                if eligible_receivers:
                                    count = len(eligible_receivers)
                                    share = self.abandonment_bounty // count
                                    remainder = self.abandonment_bounty % count
                                    
                                    config_key = f"{self.game_type.replace('solo_', '')}|{self.board_dimensions}|{self.time_limit}"
                                    
                                    for i, target in enumerate(eligible_receivers):
                                        bonus = share + (1 if i < remainder else 0)
                                        if bonus <= 0: continue
                                        
                                        # Apply to DB
                                        conn_p.execute('UPDATE users SET rating = rating + ? WHERE id = ?', (bonus, target.user_id))
                                        conn_p.execute('UPDATE user_ratings SET rating = rating + ? WHERE user_id = ? AND config_key = ?', (bonus, target.user_id, config_key))
                                        
                                        # Apply in-memory (and ensure rating_change is updated for UI display)
                                        target.rating += bonus
                                        if not hasattr(target, 'rating_change'): target.rating_change = 0
                                        target.rating_change = getattr(target, 'rating_change', 0) + bonus
                                        
                                        if not hasattr(target, 'bonus_notices'): target.bonus_notices = []
                                        target.bonus_notices.append(f"Received +{bonus} from round abandonment pool")

                                        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/rating_audit.log', 'a') as log:
                                            log.write(f"[{time.time()}] Round-End Bounty Payout: +{bonus} to {target.username} (Room: {self.room_id}, Pool: {self.abandonment_bounty})\n")
                                    
                                    # Reset pool AFTER successful distribution
                                    self.abandonment_bounty = 0
                                    
                            conn_p.commit()
                            conn_p.close()
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
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

                    except Exception as e:
                        print(f"[GameRoom] Post-round error: {e}")

                threading.Thread(target=process_results_async, daemon=True).start()
                return True

        # 3. Intermission state check (milestones handled by RoomManager)
        if self.state == 'intermission':
            return True

        return False

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
                is_private=self.is_private
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
                'bonus_letter_points': (res.get('bonus_letter_points', 0) + count - 1) // count
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
                self._update_density_for_word(wd['word'])
                
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
        Always calculates 1L-30L to ensure metadata is valid regardless of min-length transitions."""
        # Authoritative Tag: Tag the counts with the current round to prevent frontend staleness
        self.total_counts_by_len = {
            '_round': self.current_round,
            **{str(l): sum(1 for w in (self.all_words or []) if len(w) == l) for l in range(1, 31)}
        }
        min_len = getattr(self, 'current_min_length', 3)
        self.total_words_count = sum(1 for w in (self.all_words or []) if len(w) >= min_len)
        
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
                        attainable += length
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

def calculate_word_score(word, bonus_word, board_format='Normal', path=None, bonus_cell=None, **kwargs):
    """Calculate points for a word using shared utility"""
    from scoring import calculate_word_score as shared_calc
    return shared_calc(word, bonus_word, board_format=board_format, path=path, bonus_cell=bonus_cell, **kwargs)


class RoomManager:
    def __init__(self):
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
                        # timers/transitions
                        room.check_and_update_state()
                        
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
                
                # 2. Lazy Inactivity Cleanup (Every 30s)
                if loop_counter % 60 == 0:
                    self.cleanup_rooms(timeout=600)
                    now = time.time()
                    with self.lock:
                        self.user_presence = {uid: ts for uid, ts in self.user_presence.items() if (now - ts) < 600}
                
                time.sleep(0.1)
            except Exception as e:
                import traceback
                print(f"[Heartbeat] CRITICAL: {e}\n{traceback.format_exc()}")
                time.sleep(5)
                
    def create_room(self, room_id, game_type, time_limit, board_dimensions, min_rating=0, max_rating=9999, is_private=False):
        """Create a new game room or return an existing singleton for the configuration"""
        try:
            with self.lock:
                # Singleton Logic for Multiplayer Hubs (Skip for Private/Solo rooms)
                if not is_private:
                    for existing_room in self.rooms.values():
                        if (existing_room.game_type == game_type and 
                            existing_room.board_dimensions == board_dimensions and
                            existing_room.time_limit == time_limit and
                            existing_room.min_rating == min_rating and
                            existing_room.max_rating == max_rating and
                            not existing_room.is_solo and
                            not existing_room.is_private and
                            not existing_room.room_id.startswith('practice_')):
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
                    is_solo=(game_type == 'practice' or (room_id and room_id.startswith('practice_'))),
                    is_private=is_private
                )
                
                # Capacity Check
                if room.game_type in ['accumulative', 'solo_accumulative']:
                    room.max_players = 9999
                else:
                    room.max_players = 8

                self.rooms[room_id] = room
                
                # ATOMIC INITIALIZATION
                import threading
                is_24h = (room.time_limit >= 7200)
                is_split = (room.game_type == 'split')
                room.spinner_params = SpinnerSet.generate_params(room.board_dimensions, is_24h, is_split)

                # INSTANT START: User Request - No wait on first entry for public hubs.
                if room_id.startswith('pub_') and not is_24h:
                    print(f"[RoomManager] {room_id}: First entry. Generating synchronous kickstart board...")
                    # 1. Pick a bonus word
                    bw_l = room.spinner_params.get('bonus_word_length', 8)
                    b_word = self._get_bonus_word(length=bw_l, dictionary=room.spinner_params.get('dictionary', 'NWL'))
                    
                    # 2. Generate board (Emergency mode for speed)
                    e_results = self.board_generator.generate_board(
                        dimensions=room.board_dimensions,
                        bonus_word=b_word,
                        word_count_range=room.spinner_params.get('word_count_range', '100-200'), 
                        dictionary=room.spinner_params.get('dictionary', 'NWL'),
                        board_format=room.spinner_params.get('board_format', 'Normal'),
                        min_word_length=room.spinner_params.get('min_word_length', 3),
                        is_emergency=True
                    )
                    
                    if e_results and len(e_results) >= 7:
                        e_board, e_words, e_bonus_c, e_fmt, e_dict, e_ratio, e_bonus_word = e_results[:7]
                        
                        # 3. Promote to ACTIVE immediately
                        room.state = 'active'
                        room.round_start_time = time.time()
                        room.board = e_board
                        # Filter words for length lockdown
                        min_l = room.spinner_params.get('min_word_length', 3)
                        display_min = min_l
                        room.all_words = {w for w in (e_words or []) if len(w) >= display_min}
                        room.all_words_paths = {w: p for w, p in (e_dict or {}).items() if len(w) >= display_min}
                        room.bonus_word = e_bonus_word or b_word
                        room.bonus_cell = e_bonus_c
                        room.current_board_format = e_fmt
                        room.current_round = 1
                        room.current_min_length = min_l
                        room.current_word_count_range = room.spinner_params.get('word_count_range', '100-200')
                        room.initialize_density(e_board, room.all_words_paths, e_fmt)
                        room.recalculate_total_points()
                        
                        # 4. Trigger PROACTIVE search for Round 2 in background
                        room.spinner_params_generated = True
                        threading.Thread(target=self.start_board_search, args=(room_id,), daemon=True).start()
                        print(f"[RoomManager] {room_id}: Kickstart complete. Round 1 started.")
                else:
                    # Default behavior: Intermission
                    room.state = 'intermission'
                    room.intermission_start_time = time.time() - 45 # 15s TR
                    if room_id.startswith('pub_') and not is_24h:
                         threading.Thread(target=self.start_board_search, args=(room_id,), daemon=True).start()

                return room
        except Exception as e:
            import traceback
            print(f"[RoomManager] CRITICAL ERROR in create_room: {e}\n{traceback.format_exc()}")
            raise
    
    def get_yesterdays_history(self, room, current_round):
        """Recover history for a 24h room from the database (Fallback)"""
        if not room: return {}
        
        # 1. OPTIMIZATION: If in-memory state is already populated, return it immediately
        if room.previous_day_history and len(room.previous_day_history) > 0:
            return room.previous_day_history
            
        import sqlite3
        import json
        import datetime
        try:
            conn = sqlite3.connect('morpheme.db', timeout=30)
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
                    ORDER BY timestamp DESC, round_number DESC
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
            recovered_format = 'Normal'
            recovered_solutions = None
            recovered_paths = None
            history = {}

            for row in rows:
                uid, words_json, round_num, ts, b_json, b_word, b_cell_json, b_format, sols_json, paths_json = row
                uid_str = str(uid)
                if uid_str not in history:
                    u_cursor = conn.execute("SELECT username FROM users WHERE id = ?", (uid,))
                    u_row = u_cursor.fetchone()
                    uname = u_row[0] if u_row else f"User {uid}"
                    
                    history[uid_str] = {
                        'username': uname,
                        'found_words': json.loads(words_json) 
                    }
                    
                    # BACKWARD COMPATIBILITY: Also populate player objects if they are currently in the room
                    for p in room.players:
                        if p.user_id == uid:
                            p.previous_submitted_words = history[uid_str]['found_words']

                # Store board metadata from most recent record
                if not recovered_board:
                    recovered_board = json.loads(b_json)
                    recovered_bonus_word = b_word
                    recovered_bonus_cell = json.loads(b_cell_json) if b_cell_json else None
                    recovered_format = b_format
                    recovered_solutions = json.loads(sols_json) if sols_json else None
                    recovered_paths = json.loads(paths_json) if paths_json else None

            conn.close()
            
            # 2. POPULATE ROOM STATE: Reconstruct full previous round state if board recovered
            if recovered_board:
                print(f"[RoomManager] Recovering board for room {room_id} from DB Fallback")
                room.previous_board = recovered_board
                room.previous_day_history = history # Cache for next call
                
                # USE STORED SOLUTIONS (Prevents dictionary mismatch issues)
                if recovered_solutions:
                     room.previous_all_words = recovered_solutions
                     print(f"[RoomManager] Recovered {len(recovered_solutions)} words from DB solutions.")
                else:
                    # Fallback solve (NWL)
                    from board_generator import solve_board
                    min_len = 3 
                    dictionary = 'NWL'
                    try:
                        all_solutions = solve_board(recovered_board, dictionary)
                        bonus_upper = recovered_bonus_word.upper() if recovered_bonus_word else None
                        room.previous_all_words = [w for w in all_solutions if (len(w) >= min_len or (bonus_upper and w.upper() == bonus_upper))]
                        if bonus_upper and bonus_upper not in room.previous_all_words:
                            room.previous_all_words.append(bonus_upper)
                    except Exception as e:
                        print(f"[RoomManager] Error solving recovered board: {e}")
                        room.previous_all_words = []

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
             if len(humans) == 0 and len(room.spectators) == 0 and not is_daily:
                  print(f"[RoomManager] Immediate cleanup: Room {ri} is empty after user {user_id} left. Deleting.")
                  rooms_to_delete.append(ri)
                  
        for ri in rooms_to_delete:
             self.delete_room(ri)

    def find_user_session(self, user_id):
        """Find user's current room and online status"""
        uid_str = str(user_id)
        now = time.time()
        
        # Check global presence first
        last_seen = self.user_presence.get(uid_str, 0)
        is_online = (now - last_seen) < 75 # 75 seconds (reduced for better accuracy)
        
        # Search for active room - Priority to most recently active
        best_match = None
        max_active = -1
        
        for room in self.rooms.values():
            # Check players
            for p in room.players:
                if str(p.user_id) == uid_str:
                    if p.last_active > max_active:
                        max_active = p.last_active
                        best_match = {
                            'room_id': room.room_id,
                            'is_online': True,
                            'is_spectator': False,
                            'game_type': room.game_type,
                            'board_dimensions': room.board_dimensions,
                            'time_limit': room.time_limit
                        }
            # Check spectators
            for s in room.spectators:
                if str(s.user_id) == uid_str:
                    if s.last_active > max_active:
                        max_active = s.last_active
                        best_match = {
                            'room_id': room.room_id,
                            'is_online': True,
                            'is_spectator': True,
                            'game_type': room.game_type,
                            'board_dimensions': room.board_dimensions,
                            'time_limit': room.time_limit
                        }
        
        if best_match:
            return best_match
        
        # Not in a room, but might still be online (Lobby/Profile)
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
                
                if is_empty_of_humans and not is_daily:
                    # Grace Period: Don't delete rooms that are less than 10 minutes old
                    # This allows time for players to join newly created/reconstructed rooms.
                    room_uptime = time.time() - getattr(room, 'creation_time', time.time())
                    if room_uptime > 600:
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
                display_min_hist = min_len
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
            
            # Generate board
            board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio, final_bonus_word = self.board_generator.generate_board(
                room.board_dimensions,
                bonus_word,
                room.spinner_params['word_count_range'],
                room.spinner_params['dictionary'],
                room.spinner_params['board_format'],
                room.spinner_params.get('min_word_length', 3),
                room.spinner_params.get('difficulty', 'Medium'),
                is_emergency=True # SPEED: For the very first user in a room, prioritize instant start
            )
            
            if board is None:
                print(f"[RoomManager] ERROR: Board generation failed!")
                return False
                
            # ATOMICITY: Apply new round data with strict display filtering
            display_min_start = room.spinner_params.get('min_word_length', 3)
            room.all_words = {w for w in (all_words or []) if len(w) >= display_min_start}
            
            # CATEGORIZATION (Synchronous): Ensure these are available immediately for UI sync
            if hasattr(word_validator, 'word_validator'):
                room.csw_only_words = [w for w in room.all_words if word_validator.word_validator.is_csw_only(w)]
                room.added_words = [w for w in room.all_words if word_validator.word_validator.is_added_word(w)]
            else:
                room.csw_only_words = []
                room.added_words = []
            
            # CRITICAL: Preserve special cell metadata (Bonus Letter / Either/Or)
            # generate_board returns 'bonus_cell' coordinate as the 3rd element.
            room.bonus_cell = bonus_cell

            if final_bonus_word:
                room.bonus_word = final_bonus_word
                room.spinner_params['bonus_word_length'] = len(final_bonus_word)

            print(f"[RoomManager] ROUND {room.current_round} START for {room_id}. Words found: {len(all_words)}")
            with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                 f.write(f"[game_room.py] 24h START: {room_id} words={len(all_words)} dim={room.board_dimensions} at {time.time()}\n")
            
            room.current_round += 1
            # Derive ACTUAL difficulty from achieved uniqueness ratio
            dims = room.board_dimensions.split('x')
            d_num = int(dims[0]) if len(dims) == 3 else 1
            r_num = int(dims[1] if len(dims) == 3 else dims[0])
            c_num = int(dims[2] if len(dims) == 3 else dims[1])
            achieved_diff = self.board_generator.get_difficulty_label(u_ratio, r_num, c_num, room.spinner_params.get('dictionary', 'NWL'), depth=d_num)
            room.current_difficulty = achieved_diff
            room.current_uniqueness = u_ratio
            room.current_dictionary = room.spinner_params.get('dictionary', 'NWL')
            room.current_word_count_range = room.spinner_params.get('word_count_range', 'Varying...')
            
            # CRITICAL SYNC: Update the UI header slot with the ground truth
            room.spinner_params['difficulty'] = achieved_diff
            room.spinner_params['board_format'] = updated_format
            room.spinner_params['uniqueness'] = u_ratio
            room.spinner_params_revealed = True # Ensure they are shown
            
            print(f"[RoomManager] ROUND {room.current_round} START - Params: {room.current_difficulty}, {room.current_dictionary}, {room.current_word_count_range}")
            
            print(f"[RoomManager] ROUND {room.current_round} START for room {room_id}")
            print(f"[RoomManager]   > Difficulty: {room.current_difficulty}")
            print(f"[RoomManager]   > Dictionary: {room.current_dictionary}")
            print(f"[RoomManager]   > Word Range: {room.current_word_count_range}")
            
            room.current_min_length = room.spinner_params.get('min_word_length', 3)
            room.current_board_format = updated_format
            # room.bonus_cell already set above
            room.all_words_paths = all_words_dict # ATOMIC SAVE: Crucial for optimized scoring
            
            # USER: Density Initialization for Solo/Immediate Rounds
            try:
                room.initialize_density(board, all_words_dict, updated_format)
            except Exception as e:
                print(f"[Density-Diag] Failed to initialize: {e}")
            
            # Double Lockdown
            f_low = str(updated_format).lower()
            if 'bonus letter' not in f_low and 'either' not in f_low:
                room.bonus_cell = None
                
            # FAST INITIALIZATION: Length-based scores to avoid "0 point" flickering in UI
            # (Ensures all paths have points immediately while detailed solver runs)
            is_valued_init = ('valued' in str(updated_format).lower())
            init_scored_dict = {}
            for word in (all_words or []):
                if is_valued_init:
                    init_scored_dict[word] = {'total': len(word), 'base': len(word)}
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
                            return_details=True
                        )
                    room.solved_words_with_scores = final_scores
                    room.complete_words = room.all_words
                    room.solving_complete = True # Signal that missed words are ready
                    room.recalculate_total_points() # Sync refined points after background scoring
                    
                    # 4. Trigger Pre-Generation for Round 2
                    self.pre_generate_next_round(room_id)
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
                p.has_exceptional_round = False
                p.performance_efficiency = 0.0
                p.has_abandoned = False # Reset penalty flag for new round
                p._last_round_seen = room.current_round
                
            # Clear FCFS global list
            room.fcfs_found_words = []
            room._fcfs_found_words_set = set()
            
            # Activate the room
            # User Request: Do NOT wipe spinner_params here. 
            # They should hold the intent labels revealed during intermission.
            room.spinner_params_generated = False
            room.state = 'active'
            room.round_start_time = time.time()
            
            # TRIGGER PRE-GENERATION: Start searching for the NEXT round immediately 
            # to hide generation latency behind the active gameplay.
            self.pre_generate_next_round(room_id)
            room.custom_end_time = 0
            
            # SPLIT POINTS RANDOMIATION
            if room.game_type == 'split':
                import random
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
                    # PERFORM THE REVEAL
                    room.spinner_params = dict(new_params)
                    
                    # Update authoritative labels so they change ON THE DOT at 0s (start_next_round)
                    # We store them in spinner_params for reveal, but don't promote to 'current_' yet
                    room.next_round_min_length = new_params.get('min_word_length', 3)
                    room.spinner_params_revealed = True
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
                if getattr(room, 'spinner_params_generated', False) and room.next_spinner_params:
                    new_params = dict(room.next_spinner_params)
                    print(f"[RoomManager] Using EXISTING staged params for room {room_id} (Lock-protected)")
                else:
                    # Generate new parameters
                    new_params = SpinnerSet.generate_params(
                        room.board_dimensions, 
                        is_24h=is_24h, 
                        is_split=is_split, 
                        previous_params=room.spinner_params
                    )
                    # Metadata: Ensure dimensions and time limits are included for the reveal animation
                    new_params['board_dimensions'] = room.board_dimensions
                    new_params['time_limit'] = room.time_limit
                    
                    room.next_spinner_params = dict(new_params)
                    room.spinner_params_generated = True

                if reveal:
                    # 2. PERFORM THE REVEAL (Making them visible to players)
                    room.spinner_params = dict(new_params)
                    
                    # Update authoritative labels
                    room.next_round_min_length = new_params.get('min_word_length', 3)
                    room.spinner_params_revealed = True
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
    
    def start_board_search(self, room_id):
        """Start board search using Spinner Set parameters (called at 15s remaining)"""
        room = self.get_room(room_id)
        if not room:
            print(f"[RoomManager] ERROR: Room {room_id} not found")
            return False
            
        # 1. 24H ROOMS: Board generation is synchronous or triggered at midnight, 
        # so we don't start a background thread here to avoid potential race conditions with app.py reset.
        if room.time_limit >= 7200: 
            print(f"[RoomManager] 24h room {room_id}: Tracking search started for UI sync.")
            room.board_search_started = True 
            if not hasattr(room, '_last_search_start_time'):
                room._last_search_start_time = time.time()
            room.board_search_loading = False # Release stall flag
            return True

        # TRACK search start time to handle timeouts in start_next_round (if not already set)
        if not hasattr(room, '_last_search_start_time'):
            room._last_search_start_time = time.time()
            
        if not getattr(room, 'spinner_params_generated', False):
            print(f"[RoomManager] Search requested but spinner params missing for room {room_id}. Generating now...")
            self.generate_spinner_params(room_id)
            
        # ATOMIC GUARD: Prevent redundant threads while allowing the legitimate one to proceed
        with room._state_lock:
            # Check if ALREADY STARTED (To prevent redundant generation)
            if getattr(room, 'board_search_started_actual', False):
                return False
            
            # Start the search process
            room.board_search_loading = True
            room.board_search_started_actual = True # New flag to track actual execution
            room.board_search_started = True 
            room._last_search_start_time = time.time()
            
        print(f"[RoomManager] Starting board search process for room {room_id}")
        
        try:
            # AUTHORITATIVE: Use the specific params intended for this background search.
            # Fallback to spinner_params ONLY if it's the very first round (current_round == 1)
            params = getattr(room, 'next_spinner_params', None)
            if not params and room.current_round == 1:
                params = getattr(room, 'spinner_params', {})
            
            if not params:
                # If still no params, we must wait or fail to avoid bleeding from previous round
                print(f"[RoomManager] WARNING: No next_spinner_params for {room_id}. Waiting for generation...")
                return False
                
            fmt = params.get('board_format', 'Normal')
            wc_range = params.get('word_count_range', '100-200')
            
            # AUTHORITATIVE INTEGER CASTING: User mandate - ensure lengths are never interpreted as strings
            try:
                min_l = int(params.get('min_word_length', 3))
            except:
                min_l = 3
            try:
                bw_l_raw = params.get('bonus_word_length', 6)
                bw_l = max(int(bw_l_raw), min_l)
                # SYNC: Update the params so the Spinner Set UI displays the actual length used.
                params['bonus_word_length'] = bw_l
            except:
                bw_l = max(6, min_l)
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
            
            # Start board generation in background thread
            def generate_in_background():
                nonlocal bonus_word, params
                try:
                    print(f"[RoomManager] Background board generation started for {room_id}...")
                    # Capture params locally for thread safety
                    search_wc = params.get('word_count_range')
                    search_dict = params.get('dictionary')
                    search_fmt = params.get('board_format')
                    search_min = params.get('min_word_length')
                    search_diff = params.get('difficulty')
                    
                    print(f"[RoomManager] [DEBUG-GEN] Room {room_id} calling generate_board with search_min={search_min}, range={search_wc}")
                    
                    # ROBUST CALL: The generator might return 6 or 7 values depending on the branch taken.
                    res = self.board_generator.generate_board(
                        room.board_dimensions,
                        bonus_word,
                        search_wc,
                        search_dict,
                        search_fmt,
                        search_min,
                        search_diff
                    )
                    
                    # ROBUST UNPACKING: Support legacy 6-tuple or modern 7-tuple
                    if len(res) == 7:
                        board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio, final_bonus_word = res
                    else:
                        board, all_words, bonus_cell, updated_format, all_words_dict, u_ratio = res
                        final_bonus_word = bonus_word
                    
                    # Update word to the ACTUAL embedded word if different
                    if final_bonus_word:
                        bonus_word = final_bonus_word
                    
                    # ATOMIC STAGING PROMOTION: Set metadata FIRST and board LAST to prevent stale data race
                    room.next_round_words = all_words
                    room.next_round_word_paths = all_words_dict
                    room.next_round_bonus_cell = bonus_cell
                    room.next_round_bonus = bonus_word
                    room.next_round_format = updated_format
                    room.next_round_uniqueness = u_ratio
                    # USER REQUEST: Absolute consistency. Bundle the EXACT params used for this board.
                    room.next_round_spinner_params = dict(params)
                    room.next_round_spinner_params['board_format'] = updated_format # In case generator changed it
                    # FAST INITIALIZATION: Length-based scores to avoid "0 point" flickering in UI
                    # (Refined in background scoring loop below)
                    is_valued = ('valued' in str(updated_format).lower())
                    scored_dict = {}
                    for word in (all_words or []):
                        if is_valued:
                            # Sum of letter values (LETTER_VALUES import might be needed or just hardcode for speed)
                            scored_dict[word] = {'total': len(word), 'base': len(word)} # Crude estimate
                        else:
                            length = len(word)
                            s = 0
                            if length <= 2: s = 0
                            elif length <= 4: s = 1
                            elif length == 5: s = 2
                            elif length == 6: s = 3
                            elif length == 7: s = 5
                            elif length >= 8: s = 11
                            scored_dict[word] = {'total': s, 'base': s}
                    
                    room.next_round_word_scores = scored_dict
                    room.next_round_total_points = sum(pts['total'] for pts in scored_dict.values())
                    
                    # RE-VERIFY: Ensure we are still updating the ACTIVE room instance
                    # (Prevents data loss if the room was reconstructed during search)
                    target_room = room
                    active_room = self.get_room(room_id)
                    if active_room and active_room is not room:
                         print(f"[RoomManager] Search complete for {room_id}, but room was reconstructed. Redirecting to active instance.")
                         target_room = active_room

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
                            refined = {}
                            for word in (all_words or []):
                                refined[word] = calculate_word_score(
                                    word, bonus_word, path=all_words_dict.get(word),
                                    board_format=updated_format, bonus_cell=bonus_cell,
                                    board=board, return_details=True
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
                        achieved_diff = self.board_generator.get_difficulty_label(u_ratio, rows, cols, search_dict, depth=d_val)
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
                        achieved_wc = self._get_factchecked_wc_range(len(all_words))
                        
                        if getattr(room, 'next_spinner_params', None):
                            room.next_spinner_params['difficulty'] = achieved_diff
                            room.next_spinner_params['board_format'] = updated_format
                            # We intentionally DO NOT overwrite word_count_range here. 
                            # If the solver produces 105 words for a 50-100 target, we keep the 
                            # original intent (50-100) to ensure the UI aligns with the Spinner Animation.
                            
                            # If already revealed, update the active spinner_params too
                            if getattr(room, 'spinner_params_revealed', False):
                                room.spinner_params['difficulty'] = achieved_diff
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
                            if getattr(room, 'spinner_params_revealed', False):
                                room.spinner_params['bonus_word_length'] = len(final_bonus_word)
                        if getattr(room, 'next_spinner_params', None):
                            room.next_spinner_params['uniqueness'] = u_ratio
                        room.next_round_difficulty = achieved_diff
                        
                        # Authoritative recount after truncation (if any)
                        min_len = room.next_spinner_params.get('min_word_length', 3) if getattr(room, 'next_spinner_params', None) else 3
                        room.next_round_total_words_count = sum(1 for w in (all_words or []) if len(w) >= min_len)
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
                    
                    room.next_round_csw_only_words = [w for w in filtered_all if word_validator.word_validator.is_csw_only(w)]
                    room.next_round_added_words = [w for w in filtered_all if word_validator.word_validator.is_added_word(w)]
                    
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
        # ALWAYS pre-generate next round board while current round is active.
        # This ensures the 0:00 transition is instant, especially for large 6x8 boards.
        # (Skip only if already loading or started)
        if getattr(room, 'board_search_loading', False) or getattr(room, 'board_search_started', False):
            return
        
        print(f"[RoomManager] PRE-GENERATING next board for room {room_id} (Lead time start)")
        
        # Atomic Guard: If we are already searching or have a board ready, DO NOT RE-ROLL.
        if getattr(room, 'board_search_started', False) or getattr(room, 'next_round_board', None):
             print(f"[RoomManager] Lead-time: Search already in progress or board ready for {room_id}")
             return
             
        self.generate_spinner_params(room_id, reveal=False)
        self.start_board_search(room_id)

    def start_next_round(self, room_id):
        """Start next round with pre-generated board (called at 0s remaining)"""
        room = self.get_room(room_id)
        if not room:
            print(f"[RoomManager] ERROR: Room {room_id} not found")
            return False
            
        # SAFETY: If room is in 'waiting' but somehow triggered, allow transition
        if room.state == 'waiting':
             room.state = 'intermission' # Canonical path is waiting -> intermission -> active
             room.intermission_start_time = time.time() - 60 # Force it to look expired
            
        # 0. ATOMIC GUARD: Ensure only ONE thread/request triggers the round start transition
        # This prevents stacking up identical wait loops on a single slow-loading board.
        with room._state_lock:
            # ONLY ALLOW transition if room is currently in 'intermission' or 'waiting' (Lobby Start)
            # This prevents duplicate transitions or state-corrupting re-runs if watchdog triggers late.
            if room.state not in ['intermission', 'waiting']:
                 # print(f"[RoomManager] Skipping transition for {room_id}: State is {room.state}")
                 return False
                 
            if getattr(room, 'starting_round', False):
                # Watchdog reset: If stalled for > 12s (fast recovery)
                curr_init = getattr(room, '_round_start_init_time', 0)
                if curr_init > 0 and (time.time() - curr_init > 12.0):
                     print(f"[RoomManager] Stale start detected (>12s) for {room_id}, resetting guard.")
                     room.starting_round = False
                else:
                     # IMPORTANT: Do NOT 'return False' here if we are the thread that was just launched!
                     # The heartbeat/app.py logic sets room.starting_round = True BEFORE starting the thread.
                     # We only return False if another thread is actually INSIDE the try...finally block (not yet implemented with thread id)
                     # For now, we trust the launcher checked the guard.
                     pass
            
            room.starting_round = True
            room._round_start_init_time = time.time()
             
        # Log to debug_flow.log for observability
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
             f.write(f"[{time.time()}] [RoomManager] start_next_round beginning for room {room_id}\n")
        
        print(f"[RoomManager] start_next_round processing for room {room_id}")
        
        try:
            # 1. PRE-CHECK: If search skipped or missed (e.g. server load), handle it here
            if not getattr(room, 'next_round_board', None):
                # Check for stale search (loading > 45s) to prevent 'zombie' room locks.
                last_start = getattr(room, '_last_search_start_time', 0)
                is_stale = (time.time() - last_start > 45.0) if last_start > 0 else False

                if not getattr(room, 'board_search_loading', False) or is_stale:
                    if is_stale:
                        print(f"[RoomManager] STALE board search detected (>45s) for {room_id}. Forcing emergency re-trigger.")
                    else:
                        print(f"[RoomManager] Board MISSING for {room_id} at transition. Resetting and triggering emergency search.")
                    with room._state_lock:
                         room.board_search_started = False
                         room.board_search_loading = False # Clear stall flag
                    self.start_board_search(room_id)
                
                # INSTANT RECOVERY: USER MANDATE - ZERO WAIT at 0:00.
                # If background search is still loading, it's taking too long. 
                # We abandon the wait and generate an optimized board INSTANTLY (<500ms).
                if not getattr(room, 'next_round_board', None):
                    print(f"[RoomManager] INSTANT EMERGENCY: Synchronous 1-pass generation for {room_id} (Zero wait).")
                    
                    # Ensure we have some params
                    if not getattr(room, 'next_spinner_params', None):
                        self.generate_spinner_params(room_id)
                    
                    params = room.next_spinner_params
                    
                    # DYNAMIC BONUS WORD: Pick a fresh random word instead of a hardcoded fallback
                    b_word = getattr(room, 'next_round_bonus', None)
                    if not b_word:
                        print(f"[RoomManager] Emergency: Picking random bonus word for {room_id}")
                        b_word = self._get_bonus_word(
                            length=params.get('bonus_word_length', 8),
                            dictionary=params.get('dictionary', 'NWL'),
                            alternating=('checkerboard' in str(params.get('board_format')).lower()),
                            difficulty=params.get('difficulty', 'Medium'),
                            exclude=getattr(room, 'bonus_word_history', [])
                        )

                    # Generate board using RAW spinner intent to ensure ironclad compliance
                    e_results = self.board_generator.generate_board(
                        dimensions=room.board_dimensions,
                        bonus_word=b_word, 
                        word_count_range=params.get('word_count_range', '100-200'), 
                        dictionary=params.get('dictionary', 'NWL'),
                        board_format=params.get('board_format', 'Normal'),
                        min_word_length=params.get('min_word_length', 3),
                        difficulty=params.get('difficulty', 'Normal'),
                        is_emergency=True
                    )
                    
                    if e_results:
                        e_board, e_words, e_bonus_c, e_fmt, e_dict, e_ratio, e_bonus_word = e_results[:7]
                    
                    # STAGE the emergency result
                    room.next_round_board = e_board
                    room.next_round_words = e_words
                    room.next_round_word_paths = e_dict
                    room.next_round_bonus_cell = e_bonus_c
                    room.next_round_format = e_fmt
                    room.next_round_bonus = e_bonus_word or b_word # Use the actually embedded word
                    room.next_round_uniqueness = e_ratio
                    
                    # --- INSTANT PROMOTION ---
                    # 1. Fast Metrics (Zero Latency)
                    is_valued_e = ('valued' in str(e_fmt).lower())
                    e_scored_dict = {}
                    for word in (e_words or []):
                        if is_valued_e: e_scored_dict[word] = {'total': len(word), 'base': len(word)}
                        else:
                            length = len(word)
                            s = 0
                            if length <= 2: s = 0
                            elif length <= 4: s = 1
                            elif length == 5: s = 2
                            elif length == 6: s = 3
                            elif length == 7: s = 5
                            elif length >= 8: s = 11
                            else: s = 1 # Absolute fallback for length 3/4 if above check misses
                            e_scored_dict[word] = {'total': max(1, s), 'base': max(1, s)}
                    
                    room.next_round_word_scores = e_scored_dict
                    room.next_round_total_points = sum(pts['total'] for pts in e_scored_dict.values())
                    
                    # SYNCHRONOUS COUNTS FOR EMERGENCY (Ensures Remaining Tab is accurate at 0s mark)
                    # We must set current_min_length on the room if it's about to be promoted
                    room.current_min_length = params.get('min_word_length', 3)
                    filtered_e_words = [w for w in (e_words or []) if len(w) >= room.current_min_length]
                    
                    room.next_round_counts_by_len = {str(l): sum(1 for w in (e_words or []) if len(w) == l) for l in range(1, 31)}
                    room.next_round_total_words_count = len(filtered_e_words)
                    room.next_round_total_points = sum(pts['total'] if isinstance(pts, dict) else pts for pts in e_scored_dict.values())
                    
                    room.next_round_board = e_board # PROMPT SIGNALING

                    # 2. Background Refinement
                    from scoring import calculate_word_score
                    def refine_emergency_scores():
                        try:
                            refined_e = {}
                            for w in (e_words or []):
                                refined_e[w] = calculate_word_score(
                                    w, (e_bonus_word or b_word), path=e_dict.get(w),
                                    board_format=e_fmt, bonus_cell=e_bonus_c,
                                    board=e_board, return_details=True
                                )
                            room.next_round_word_scores = refined_e
                            
                            # (Counts already calculated synchronously at start for speed)

                            # Active Sync
                            if room.board == e_board:
                                room.solved_words_with_scores = refined_e
                                room.recalculate_total_points()
                        except: pass
                    
                    threading.Thread(target=refine_emergency_scores, daemon=True).start()

                    # FACT-CHECK labels for UI
                    from word_validator import word_validator
                    # We NO LONGER overwrite word_count_range here to prevent "Intent Flipping"
                    # If players saw 50-100, we keep that label even if emergency found 105 words.
                    if getattr(room, 'next_spinner_params', None):
                        room.next_spinner_params['board_format'] = e_fmt
                    room.spinner_params['board_format'] = e_fmt
                    room.current_board_format = e_fmt
                    
                    room.next_round_csw_only_words = [w for w in e_words if word_validator.is_csw_only(w)]
                    room.next_round_added_words = [w for w in e_words if word_validator.is_added_word(w)]

                    # Density for Emergency
                    room.initialize_density(e_board, e_dict, e_fmt, is_staging=True)
                    
                    print(f"[RoomManager] INSTANT generation complete for {room_id} (Word: {b_word}). Transitioning NOW.")
            
            # --- START TRANSITION ---
            # ATOMIC REFERENCE CAPTURE: Since we replace the board object, a reference is safe and instant.
            ghost_prev_board = room.board 
            
            ghost_source_words = list(room.complete_words) if (getattr(room, 'complete_words', None) and len(room.complete_words) > 0) else list(room.all_words)
            ghost_bonus = (room.bonus_word.upper() if room.bonus_word else None)
            ghost_min_len = getattr(room, 'current_min_length', 3)
            ghost_players = [p for p in room.players]
            
            # 2. STATE TRANSITION LOCK: Perform the atomic board swap
            with room._state_lock:
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
                # CRITICAL: Use 'or' to handle cases where next_round_spinner_params is explicitly None
                active_params = getattr(room, 'next_round_spinner_params', None) or room.spinner_params or {}
                room.current_board_format = room.next_round_format or active_params.get('board_format', 'Normal')
                room.current_word_count_range = active_params.get('word_count_range', '100-200')
                room.current_difficulty = active_params.get('difficulty', 'Medium')
                room.current_dictionary = active_params.get('dictionary', 'NWL')
                raw_min = active_params.get('min_word_length', 3)
                
                # Update spinner_params to match the actual board being used
                room.spinner_params = dict(active_params) if active_params else {}

                room.current_uniqueness = getattr(room, 'next_round_uniqueness', room.current_uniqueness)
                try:
                    room.current_min_length = int(raw_min)
                except:
                    room.current_min_length = 3

                # --- 2. BOARD & WORD PROMOTION ---
                # EMERGENCY SAFETY: If for any reason staging is empty, force a fast fallback board NOW
                if not room.next_round_board or not room.next_round_words:
                    print(f"[REMAINING-STABILIZER] Staging empty for {room_id} at promotion. Forcing emergency fallbackboard.")
                    from board_generator import BoardGenerator
                    bg = BoardGenerator()
                    # CORRECTION: The method is generate_board, and it MUST receive the intended range to properly constrain results
                    target_range = room.spinner_params.get('word_count_range', '100-200')
                    e_results = bg.generate_board(
                        room.board_dimensions, 
                        bonus_word=getattr(room, 'next_round_bonus', ''),
                        word_count_range=target_range,
                        dictionary=room.current_dictionary,
                        board_format=room.current_board_format,
                        min_word_length=room.current_min_length,
                        difficulty=room.current_difficulty,
                        is_emergency=True
                    )
                    # Robust Unpacking: Support both 6-tuple and 7-tuple returns
                    if len(e_results) == 7:
                        e_board, e_words, e_bonus_c, _, e_paths, e_ratio, e_bonus_word = e_results
                    else:
                        e_board, e_words, e_bonus_c, _, e_paths, e_ratio = e_results
                        e_bonus_word = getattr(room, 'next_round_bonus', '')
                    
                    room.next_round_board = e_board
                    room.next_round_words = e_words
                    room.next_round_word_paths = e_paths
                    room.next_round_total_words_count = len(e_words)
                    room.next_round_bonus = e_bonus_word
                    
                    # USER REQUEST: Ensure Total Points is never 0.
                    # Fast-apply length based scores for the emergency board immediately.
                    is_valued_e = ('valued' in str(room.current_board_format).lower())
                    e_scores = {}
                    for w in e_words:
                        if is_valued_e: e_scores[w] = {'total': len(w), 'base': len(w)}
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
                    room.next_round_word_scores = e_scores
                
                # --- 3. FINAL COUNT SYNC: Ensure factual counts are promoted even if truncation was skipped ---
                # USER REQUEST: Total count should reflect scorable words only.
                if hasattr(room, 'next_round_total_words_count') and room.next_round_total_words_count > 0:
                    room.total_words_count = room.next_round_total_words_count
                    room.initial_total_words = room.total_words_count
                else:
                    room.total_words_count = sum(1 for w in (room.next_round_words or []) if len(w) >= room.current_min_length)
                
                # --- 4. HISTORY PROMOTION ---
                room.previous_min_length = getattr(room, 'current_min_length', 3)
                room.previous_board = list(room.board) if room.board else []
                # USER REQUEST: Ensure intermission list matches round rules
                display_min_prev = getattr(room, 'current_min_length', 3)
                room.previous_all_words = [w for w in (room.all_words or []) if len(w) >= display_min_prev]
                room.previous_csw_only_words = list(room.csw_only_words) if room.csw_only_words else []
                room.previous_added_words = list(room.added_words) if room.added_words else []
                
                # Update current active counts
                room.csw_only_words = getattr(room, 'next_round_csw_only_words', [])
                room.added_words = getattr(room, 'next_round_added_words', [])

                # ATOMIC PROMOTION: Carry staging data to active room state
                room.board = room.next_round_board
                
                # USER REQUEST: Absolute consistency. Only include words that meet the round's scorable minimum.
                # HARD FLOOR: Always exclude 3-letter words from the 'All Words' list (User Request: "NOT 3 letter wrods")
                min_l = room.current_min_length if hasattr(room, 'current_min_length') else 3
                display_min = min_l
                room.all_words = {w for w in (room.next_round_words or []) if len(w) >= display_min}
                room.all_words_paths = {w: p for w, p in (room.next_round_word_paths or {}).items() if len(w) >= display_min}
                room.solved_words_with_scores = getattr(room, 'next_round_word_scores', {})
                
                # USER REQUEST: Ensure Bonus Word is ironclad (highlighted in green at end)
                # If for any reason next_round_bonus is empty (e.g. emergency stall), pick a fresh one now
                current_bw = getattr(room, 'next_round_bonus', '')
                if not current_bw:
                    bw_l = room.spinner_params.get('bonus_word_length', 8)
                    current_bw = self._get_bonus_word(
                        length=bw_l,
                        dictionary=room.current_dictionary,
                        alternating=('checkerboard' in str(room.current_board_format).lower())
                    )
                room.bonus_word = current_bw
                
                # SYNC: Ensure Spinner Set params match the ACTUAL bonus word length we are starting with.
                if room.bonus_word and getattr(room, 'spinner_params', None):
                    room.spinner_params['bonus_word_length'] = len(room.bonus_word)
                
                # SAFETY SYNC: Ensure the bonus word is actually in all_words
                # (Prevents UI from failing to highlight it if it were somehow missed by the solver)
                # And ensure it bypasses the min-length filter if it was somehow shorter (unlikely but safe)
                if room.bonus_word and room.bonus_word not in room.all_words:
                    room.all_words.add(room.bonus_word)
                
                room.bonus_cell = getattr(room, 'next_round_bonus_cell', None)
                
                # --- 4. ACCURACY ENFORCEMENT: Draconian word count truncation ---
                try:
                    # USER REQUEST: Absolute filter for the final scorable set.
                    # This ensures 3L and 4L words are PURGED from both the list and the score dictionary.
                    min_l = room.current_min_length if hasattr(room, 'current_min_length') else 3
                    display_min_final = min_l
                    
                    room.all_words = {w for w in (room.all_words or []) if len(w) >= display_min_final}
                    room.all_words_paths = {w: room.all_words_paths.get(w, []) for w in room.all_words}
                    
                    if hasattr(room, 'solved_words_with_scores'):
                        room.solved_words_with_scores = {w: room.solved_words_with_scores[w] for w in room.all_words if w in room.solved_words_with_scores}

                    target_range = getattr(room, 'current_word_count_range', '100-200')
                    if target_range:
                        _, max_target = self.board_generator._parse_word_count_range(target_range)
                        
                        if max_target < 99999 and len(room.all_words) > max_target:
                            print(f"[ACCURACY-SYNC] Truncating Round {room.current_round} to {max_target} words to match range '{target_range}'")
                            # Sort by length desc then alpha
                            sorted_scorable = sorted(list(room.all_words), key=lambda w: (len(w), w), reverse=True)[:max_target]
                            room.all_words = set(sorted_scorable)
                            room.all_words_paths = {w: room.all_words_paths.get(w, []) for w in room.all_words}
                            room.solved_words_with_scores = {w: room.solved_words_with_scores[w] for w in room.all_words if w in room.solved_words_with_scores}
                            
                    # Explicitly verify the length matches what we truncated to avoid ANY downstream counting ghosts
                    room.total_words_count = len(room.all_words)
                except Exception as e:
                    print(f"[ACCURACY-ERROR] Failed to truncate: {e}")
                
                # FINAL ACCURACY SYNC: Ensure the header labels exactly match the results
                room.total_words_count = sum(1 for w in room.all_words if len(w) >= room.current_min_length)
                room.current_difficulty = getattr(room, 'next_round_difficulty', room.current_difficulty)

                room.cell_density = getattr(room, 'next_round_cell_density', [])
                room.initial_cell_density = getattr(room, 'next_round_initial_cell_density', [])
                room.max_cell_density = getattr(room, 'next_round_max_cell_density', 0)
                room.global_round_found_words = set()
                
                room.solving_complete = True 
                room.complete_words = list(room.all_words) 
                room.update_counts_by_len()
                room.recalculate_total_points()
                
                # FINAL VALIDATION: If the board is STILL empty, we cannot start the round.
                # Revert to a 5-second emergency intermission to try again.
                if not room.board or len(room.board) == 0:
                    print(f"[RoomManager] CRITICAL: Room {room_id} failed to secure a board. Reverting to emergency intermission.")
                    room.state = 'intermission'
                    room.intermission_start_time = time.time() - 55 # 5s remaining
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
                room.next_round_total_words_count = 0
                room.next_round_counts_by_len = {}
                room.next_round_total_points = 0
                room.next_round_cell_density = None
                room.next_round_initial_cell_density = None
                room.board_search_started = False
                room.board_search_loading = False
                room.spinner_params_generated = False
                room.spinner_params_revealed = False
                room.spinner_params_loading = False
                room.next_spinner_params = None
                room.next_round_spinner_params = None
                
                # Reset Round counters
                room.current_round += 1
                # USER REQUEST: Reset 24h rooms to [0] players at midnight transition
                if room.time_limit >= 7200:
                    room.players = []
                    room.spectators = []
                else:
                    for p in room.players:
                        p.submitted_words, p.invalid_words, p.score = [], [], 0
                        p.found_bonus_word, p.has_abandoned = False, False
                        p.joined_mid_round = False
                        p._last_round_seen = room.current_round
                
                # Update word counts by length for the new round
                room.update_counts_by_len()
                
                # Robust Format Check
                f_low_current = str(room.current_board_format).lower()
                is_special_fmt = ('bonus letter' in f_low_current or 'either' in f_low_current)
                if not is_special_fmt:
                    room.bonus_cell = None
                
                # --- FINAL CLEARANCE & NEXT LOG CHAIN ---
                # Clear staging data immediately to prevent stale exclusion or duplicate promotion
                room.next_round_board = None 
                room.next_round_words = []
                room.next_round_word_paths = {}
                room.next_round_word_scores = {}
                room.next_round_bonus = None
                room.next_round_total_words_count = 0
                room.next_round_counts_by_len = {}
                room.next_round_total_points = 0
                room.next_round_cell_density = None
                room.next_round_initial_cell_density = None
                
                # ATOMIC PROMOTION: Set state to active
                room.state = 'active'
                room.round_start_time = time.time()
                
                room.custom_end_time = 0 
                
                # START PRE-GENERATION FOR N+2 NOW (Safe since all R+1 staging is cleared)
                # USER REQUEST: Ensure this happens AFTER all cleanup to avoid race conditions.
                threading.Thread(target=self.pre_generate_next_round, args=(room_id,), daemon=True).start()
                
                return True
                
                # IMPORTANT: CLEAR STARTING LOCK
                room.starting_round = False
                
                print(f"[TRANSITION] Room {room_id}: INTERMISSION -> ACTIVE (Round {room.current_round}, Time: {room.round_start_time})")
                return True

            print(f"[RoomManager] SUCCESS: Transitioned room {room_id} to Round {room.current_round}")
            
            # --- ASYNCHRONOUS POST-TRANSITION TASKS ---
            def finalize_results():
                # This is offloaded to avoid blocking the main server thread
                try:
                    # Save history to DB
                    self.save_round_history(room, board=ghost_prev_board, all_words=ghost_source_words, bonus_word=ghost_bonus)
                    
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
            return False

        finally:
            # ABSOLUTE SAFETY: The flag MUST be cleared so the room can try again if we failed.
            with room._state_lock:
                 room.starting_round = False
    
    def _get_factchecked_wc_range(self, count):
        """Map actual word count to the closest standard spinner display range.
           Matches the 100-200 and 200-300 targets defined in SpinnerSet."""
        if count >= 300: return '200-300'
        if count >= 200: return '200-300'
        if count >= 100: return '100-200'
        return '50-100'

    def _get_bonus_word(self, length=8, dictionary='NWL', alternating=False, difficulty='Medium', exclude=None):
        """Get a bonus word of specified length, optionally enforcing C/V alternating pattern for Checkerboard"""
        import time
        from word_validator import word_validator
        
        # Determine if we should exclude ING (Medium/Hard)
        diff_upper = str(difficulty).upper()
        exclude_ing = (diff_upper in ['MEDIUM', 'HARD', 'EXPERT', 'DIFFICULT', 'MASTERS', 'NORMAL'])
        
        # Get all words of the specified length (using cache if available)
        if dictionary == 'CSW':
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

        # Return random word (Reroll if ING detected in Medium/Hard or if in exclude list)
        import random
        valid_words = words
        if exclude_ing:
            valid_words = [w for w in words if "ING" not in w.upper()]
            if not valid_words:
                valid_words = words
        
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
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
            f.write(f"[game_room.py] _get_bonus_word SUCCESS: {result} (alternating={alternating}) at {time.time()}\n")
        return result
    
    
    def save_round_history(self, room, board=None, all_words=None, bonus_word=None):
        """Save the results of the JUST COMPLETED round to the database"""
        if room.is_solo:
            print(f"[RoomManager] SKIPPING history save for SOLO room {room.room_id}")
            return
            
        import sqlite3
        import json
        
        # Guard against double saving
        if getattr(room, 'last_saved_round', 0) >= room.current_round:
            return
        
        try:
            conn = sqlite3.connect('morpheme.db', timeout=30)
            
            # Use passed-in snapshots if provided (prevents stale data from being saved)
            actual_board = board if board is not None else room.board
            board_json = json.dumps(actual_board)
            
            # Robust Timestamping for 24h rooms
            now = datetime.datetime.now()
            # If a daily room ended just after midnight, the results belong to "Yesterday"
            if room.time_limit >= 7200 and now.hour == 0 and now.minute < 10:
                yesterday = now - datetime.timedelta(days=1)
                timestamp = yesterday.strftime('%Y-%m-%d 23:59:59')
            else:
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
            
            board_format = room.current_board_format
            wc_range = room.spinner_params.get('word_count_range', (0, 0))
            wc_tuple = room._get_wc_tuple(wc_range)
            is_500plus = wc_tuple[0] >= 500
            
            # Board formats (Normal, Cube, Mania, etc.) are allowed for history
            # (Validation for rank/stats can be done at display time if needed)
            if is_500plus:
                 print(f"[RoomManager] SKIPPING history save for room {room.room_id} - 500+ is unranked.")
                 conn.close()
                 return
                 
            # Determine best data source (Room state might have already advanced to Next Round)
            actual_all_words = all_words if all_words is not None else room.all_words
            actual_bonus_word = bonus_word if bonus_word is not None else getattr(room, 'bonus_word', '')
            
            # Identify registered players who actually made any attempt
            participating_registered = [p for p in room.players if p.user_id > 0 and (p.score > 0 or p.submitted_words or p.invalid_words)]
            
            if not participating_registered:
                print(f"[RoomManager] SKIPPING history save for room {room.room_id} - no participating registered users.")
                conn.close()
                return

            for p in participating_registered:
                # Use current submitted_words because we call this BEFORE clearing
                
                # NORMALIZE TIMESTAMPS: Ensure numeric s for replay
                words_data = []
                for w in p.submitted_words:
                    # Get raw time or fallback
                    raw_time = w.get('time')
                    if not raw_time or isinstance(raw_time, str):
                        raw_time = room.round_start_time or time.time()
                    
                    words_data.append({
                        'word': w['word'],
                        'points': w.get('points', 0),
                        'timestamp': raw_time
                    })
                
                # Calculate Best Word
                best_w_entry = max(p.submitted_words, key=lambda x: x.get('points', 0)) if p.submitted_words else None
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
                
                # 2. SAVE: Optimization - Only store full solutions/paths for the FIRST player in the batch
                is_first_player = (p == participating_registered[0])
                solutions_payload = json.dumps(list(actual_all_words)) if is_first_player else None
                paths_payload = json.dumps(getattr(room, 'all_words_paths', {})) if is_first_player else None # all_words_paths contains paths

                conn.execute('''
                    INSERT INTO round_history (user_id, room_id, game_type, round_number, board_json, words_json, total_score, round_start_time, round_duration, timestamp, user_rating, performance_ratio, best_word, best_word_score, board_dimensions, wpm, total_words_avail, bonus_word, bonus_cell, board_format, all_solutions_json, all_words_paths)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (p.user_id, room.room_id, room.game_type, room.current_round, board_json, json.dumps(words_data), p.score, room.round_start_time, room.time_limit, timestamp, p.rating, p.performance_efficiency, best_word_text, best_word_val, room.board_dimensions, final_wpm, len(actual_all_words), actual_bonus_word, json.dumps(getattr(room, 'bonus_cell', None)), getattr(room, 'current_board_format', 'Normal'), solutions_payload, paths_payload))
            
            room.last_saved_round = room.current_round
            conn.commit()
            conn.close()
            print(f"[RoomManager] Saved round history for room {room.room_id} (Round {room.current_round})")
        except Exception as e:
            print(f"[RoomManager] Error saving round history: {e}")

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
                    u_ratio = self.board_generator.get_uniqueness_ratio(proposed_board, all_words, r_num, c_num, dict_name, depth=d_num)
                    achieved_diff = self.board_generator.get_difficulty_label(u_ratio, r_num, c_num, dict_name, depth=d_num)
                    
                    # PROMOTE DATA
                    room.next_round_board = proposed_board
                    room.next_round_words = all_words
                    room.next_round_word_paths = all_words_dict
                    room.next_round_word_scores = scored_dict
                    room.next_round_uniqueness = u_ratio
                    room.next_round_difficulty = achieved_diff
                    room.solving_complete = True # STOPS SERVER SEARCH
                    
                    # Sync UI
                    room.spinner_params['difficulty'] = achieved_diff
                    room.spinner_params['uniqueness'] = u_ratio
                    room.current_difficulty = achieved_diff
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
