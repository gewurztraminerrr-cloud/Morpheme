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
    fcfs_found_words: set = field(default_factory=set)
    
    # Spinner parameters
    current_min_length: int = 3
    current_board_format: str = 'Normal'
    current_word_count_range: str = '100-200'
    current_dictionary: str = 'NWL'
    current_difficulty: str = 'Normal'
    current_bonus_word_length: int = 0
    spinner_params: Dict = field(default_factory=dict)
    
    # Next round pre-generation (for Accumulative timing)
    spinner_params_generated: bool = False  # Track if spinner set generated for next round
    board_search_started: bool = False      # Track if board search started
    next_round_board: List[List[str]] = field(default_factory=list)  # Store pre-generated board
    next_round_words: List[str] = field(default_factory=list)  # Store pre-generated word list
    next_round_bonus: str = ''  # Store bonus word for next round
    next_round_bonus_cell: tuple = None # Store bonus cell for next round
    
    # Players
    players: List[Player] = field(default_factory=list)
    past_players: Dict[str, Player] = field(default_factory=dict) # Archive of players for persistence
    round_quitters: List[Player] = field(default_factory=list) # Players who left mid-round after playing
    abandonment_bounty: int = 0 # Points collected from quitters for distribution at round end
    
    # Chat
    chat_messages: List[Dict] = field(default_factory=list)
    
    # History of winners
    winners_history: List[Dict] = field(default_factory=list) # [{'round': N, 'winners': [names], 'score': S}]

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
            
    def add_chat_message(self, username, message, is_system=False, image=None):
        """Add chat message to room"""
        self.chat_messages.append({
            'username': username,
            'message': message,
            'image': image,
            'is_system': is_system,
            'time': time.time()
        })
        # Keep only last 30 messages
        if len(self.chat_messages) > 30:
            self.chat_messages.pop(0)
    
    def add_player(self, user_id, username, rating, games_played=0, country_flag='🏳️', manual_accessed=False, is_guest=False):
        """Add player to room"""
        is_daily = self.time_limit >= 7200
        
        # NOTE: Abandonment penalty is fixed at exit; re-joining does not remove from quitters list.


        
        # Check if player already exists (PERSISTENCE)
        existing_player = self.get_player(user_id)
        if existing_player and is_daily:
            print(f"[GameRoom] Persistence: Reusing existing player {username} in 24h room {self.room_id}")
            existing_player.last_active = time.time()
            existing_player.country_flag = country_flag # Update flag
            # Update guest status if it changed (unlikely but safe)
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
            else:
                # If they join during intermission or early enough, reset the flag
                existing_player.joined_mid_round = False
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
            return True

        # Check if player exists in past_players
        # print(f"DEBUG: Checking past_players for {user_id}. Past players count: {len(self.past_players)}")
        existing_player = next((p for p in self.past_players.values() if str(p.user_id) == str(user_id)), None)
        
        if existing_player:
            print(f"DEBUG: RESTORING player {user_id} from past_players. History len: {len(existing_player.previous_submitted_words)}")
            print(f"DEBUG: Restored words: {[w['word'] for w in existing_player.previous_submitted_words]}")
            existing_player.last_active = time.time()
            existing_player.country_flag = country_flag # Update flag
            existing_player.games_played = games_played # Update games played (if changed)
            existing_player.is_guest = is_guest # Update guest status
            if manual_accessed:
                existing_player.joined_mid_round = True
            elif not is_daily and self.state == 'active':
                 # Check for "Refresh" grace period (15s)
                 # If they were gone for > 15s, mark as late joiner even if restoring
                 if (time.time() - existing_player.last_active) > 15:
                      print(f"[GameRoom] Restored player {username} marked as LATE JOINER (Inactive for {time.time() - existing_player.last_active:.1f}s)")
                      existing_player.joined_mid_round = True
            self.players.append(existing_player)
            return True

        # Ensure player is not already in the room (prevent duplicates)
        self.remove_player(user_id)
        
        # Check max players specific to room
        if len(self.players) >= self.max_players:
            return False # Room full
            
        player = Player(user_id, username, rating, games_played=games_played, country_flag=country_flag, is_guest=is_guest)
        if manual_accessed:
            player.joined_mid_round = True
        elif was_already_in_room:
            player.joined_mid_round = was_joined_mid_round
        elif self.state == 'active' and not is_daily:
            player.joined_mid_round = True
            
        self.players.append(player)
        self.players.sort(key=lambda p: p.rating, reverse=True)
        
        # System Notice
        self.add_chat_message("System", f"{username} has entered the room.", is_system=True)
        
        return True # Success

    def add_spectator(self, user_id, username, rating):
        """Add spectator to room"""
        # Disable spectating for 24h rooms (>= 2h)
        if self.time_limit >= 7200:
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
        """Remove player or spectator from room"""
        # PERSISTENCE: Never remove players from 24h rooms unless forced (e.g. logout)
        if self.time_limit >= 7200 and not force:
            # We still allow removing from spectators if they were accidentally added there
            initial_specs = len(self.spectators)
            self.spectators = [p for p in self.spectators if str(p.user_id) != str(user_id)]
            return

        # Remove from players - Use string comparison to be safe against type mismatches
        initial_players = len(self.players)
        # Find player to get username for notice
        leaving_player = next((p for p in self.players if str(p.user_id) == str(user_id)), None)
        username = leaving_player.username if leaving_player else "Someone"
        
        # Track abandonment (if in active round and played and NOT mid-round joiner)
        if leaving_player and self.state == 'active' and not getattr(leaving_player, 'joined_mid_round', False):
             # ENFORCE RULE: Only apply abandonment penalty if there is at least one OTHER player who started from the beginning.
             other_starters = [
                 p for p in self.players + self.round_quitters 
                 if str(p.user_id) != str(user_id) and not getattr(p, 'is_ai', False) and not is_player_guest(p) and not getattr(p, 'joined_mid_round', False)
             ]
             
             if len(other_starters) >= 1:
                 print(f"[DEBUG-PENALTY] Player {username} leaving during active round. Activity check: words={len(leaving_player.submitted_words)}, invalid={len(leaving_player.invalid_words)}, score={leaving_player.score}")
                 if leaving_player.submitted_words or leaving_player.invalid_words:
                      # Only add if not already in quitters
                      if not any(q.user_id == leaving_player.user_id for q in self.round_quitters):
                           print(f"[GameRoom] Player {username} ({user_id}) abandoned mid-round. Logging as Quitter.")
                           self.round_quitters.append(leaving_player)
                           
                           # AUTOMATIC INSTANT PENALTY - Deduct 16 immediately to prevent "cheating" re-joins
                           is_guest = is_player_guest(leaving_player)
                           print(f"[DEBUG-PENALTY] Quitter {username} is_guest={is_guest}, user_id={leaving_player.user_id}")
                           
                           if not is_guest and leaving_player.user_id > 0:
                                 print(f"[DEBUG-PENALTY] Applying -16 to {username} (Current Rating: {leaving_player.rating})")
                                 leaving_player.rating = max(0, leaving_player.rating - 16)
                                 leaving_player.rating_change -= 16
                                 self.abandonment_bounty += 16
                                 print(f"[GameRoom] INSTANT Penalty applied to {username}. Bounty now: {self.abandonment_bounty}")
                                 
                                 # Apply to Global Rank immediately
                                 try:
                                     db_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/morpheme.db'
                                     db_conn = sqlite3.connect(db_path, timeout=30)
                                     
                                     # 1. Update Global users table
                                     cur = db_conn.execute('UPDATE users SET rating = rating - 16 WHERE id = ?', (leaving_player.user_id,))
                                     print(f"[DEBUG-PENALTY] users table update for {username}: rows modified = {cur.rowcount}")
                                     
                                     # 2. Update config-specific user_ratings table
                                     current_config = f"{self.game_type}|{self.board_dimensions}|{self.time_limit}"
                                     cur2 = db_conn.execute('UPDATE user_ratings SET rating = rating - 16 WHERE user_id = ? AND config_key = ?', (leaving_player.user_id, current_config))
                                     print(f"[DEBUG-PENALTY] user_ratings table update for {username} (config={current_config}): rows modified = {cur2.rowcount}")
                                     
                                     db_conn.commit()
                                     db_conn.close()
                                     print(f"[DEBUG-PENALTY] DB Commit successful for {username}")
                                 except Exception as e:
                                     print(f"[GameRoom] Instant Penalty DB update Error for {username}: {e}")
                           else:
                                 print(f"[DEBUG-PENALTY] SKIPPING DB update for {username} (Guest or Invalid ID)")
                      else:
                           print(f"[DEBUG-PENALTY] Player {username} already in round_quitters. Skipping duplicate penalty.")
                 else:
                      print(f"[DEBUG-PENALTY] Player {username} left but had NO activity. (words=0, invalid=0)")
        else:
             print(f"[DEBUG-PENALTY] Player {username} removed outside active round or state. (state={self.state})")

        self.players = [p for p in self.players if str(p.user_id) != str(user_id)]
        if len(self.players) < initial_players:
            print(f"[GameRoom] Removed player {user_id} ({username}) from room {self.room_id} (force={force})")
            self.add_chat_message("System", f"{username} has left the room.", is_system=True)

        # If forced (logout), also clear from past_players archive so they don't auto-restore if they rejoin
        if force:
            uid_str = str(user_id)
            if uid_str in self.past_players:
                del self.past_players[uid_str]
                print(f"[GameRoom] Cleared {username} from past_players in room {self.room_id}")

        # Remove from spectators (just in case)
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
            if not is_daily and age >= timeout and not p.is_ai:
                to_remove_ids.append(p.user_id)

        players_removed = False
        for uid in to_remove_ids:
            # remove_player handles the abandonment rating penalty logic!
            self.remove_player(uid)
            players_removed = True

        # Check spectators
        active_spectators = []
        specs_removed = False
        for p in self.spectators:
            age = now - p.last_active
            # Spectators get a longer timeout (default 30 mins)
            # 24h rooms never kick spectators either
            if is_daily or age < spec_timeout:
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
        if self.state == 'active':
            if self.custom_end_time > 0:
                return max(0, int(self.custom_end_time - time.time()))
            
            elapsed = time.time() - self.round_start_time
            return max(0, self.time_limit - int(elapsed))
        elif self.state == 'intermission':
            elapsed = time.time() - self.intermission_start_time
            return max(0, 60 - int(elapsed))  # 60 second intermission
        return 0
    
    @property
    def round_end_time(self):
        """Get timestamp when current round ends (for client sync)"""
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
        
        word = word.upper()
        
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
            for r, c in path:
                if 0 <= r < len(self.board) and 0 <= c < len(self.board[0]):
                    cell_val = str(self.board[r][c])
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
                else:
                    valid_path = False
                    break
            
            if valid_path:
                # Find which of the possible interpreted words from the path actually exists on the board
                valid_options = [w for w in possible_words if w in self.all_words]
                if len(valid_options) == 1:
                    word = valid_options[0]  # Auto-correct the submission to the valid Either/Or letter
                    matched_word = word
        
        # Direct match check
        if word in self.all_words:
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
            min_len = self.spinner_params.get('min_word_length', 3)
            
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
        
        # Check minimum length (use the final word length, e.g., QUATE is 5, QATE is 4)
        min_len = self.spinner_params.get('min_word_length', 3)
        if len(final_word) < min_len:
            return False, f"{final_word} is too short", 0, None
        
        # Check if already submitted (by this player)
        # Extract existing words from the list of dicts
        existing_words = {w['word'] for w in player.submitted_words}
        if final_word in existing_words:
            return False, f"{final_word} ALREADY FOUND", 0, None
        
        # FCFS Mode: Check if word found by ANYONE
        if self.game_type == 'fcfs':
            if final_word in self.fcfs_found_words:
                return False, f"{final_word} FOUND BY ANOTHER", 0, None
            self.fcfs_found_words.add(final_word)
        
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
        
        # Add word as structured data
        player.submitted_words.append({
            'word': final_word,
            'time': time.time(),
            'points': points,
            'path': path
        })
        
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
    
    def _recalculate_player_score(self, player):
        """
        Recalculate player score from submitted words sequentially.
        """
        # Sort by submission time
        sorted_words = sorted(player.submitted_words, key=lambda x: x.get('time', 0))
        current_score = 0
        fmt = self.current_board_format
        from scoring import score_logger
        score_logger.debug(f"[Recalc] Re-evaluating score for {player.username}. Words: {len(player.submitted_words)} | Room FMT: {fmt}")
        
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
                # Fallback: Recalculate (Standard modes or first-time submission)
                points_details = calculate_word_score(
                    w_obj['word'], 
                    self.bonus_word, 
                    board_format=fmt,
                    path=w_obj.get('path'),
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
            
            score_logger.debug(f"[Recalc]   Word: {w_obj['word']} | Points: {points} | Running Total: {current_score}")
                
        player.score = current_score
        return current_score
    
    def get_intermission_milestone(self):
        """Returns which milestone we're at during intermission.
        
        Returns:
            'spinner' - At 45s remaining, time to generate Spinner Set
            'search' - At 15s remaining, time to start board search
            'start' - At 0s remaining, time to start next round
            None - No milestone reached yet
        """
        if self.state != 'intermission':
            return None
        
        time_remaining = self.time_remaining
        
        # Check milestones in order (most urgent first)
        if time_remaining <= 0:
            return 'start'
        elif time_remaining <= 15:
            if not self.board_search_started:
                return 'search'
        elif not self.spinner_params_generated:
            # Trigger Spinner Generation as soon as intermission begins (if not already done)
            return 'spinner'
        
        return None
    
    def check_and_update_state(self):
        """Check timers and update game state accordingly"""
        # Check if active round has expired
        if self.state == 'active' and self.time_remaining == 0:
            # AI turns are now generated at START of round for incremental scoring
            
            self.state = 'intermission'
            self.intermission_start_time = time.time()
            board_format = self.current_board_format
            
            # IMMEDIATE SNAPSHOT (24h Rooms): Save history now so it is available during intermission
            if self.time_limit >= 7200:
                print(f"[GameRoom] Snapshotting history at start of intermission for room {self.room_id}")
                # Filter previous_all_words by the CURRENT round's min length
                min_len = getattr(self, 'current_min_length', 3)
                self.previous_all_words = [w for w in self.all_words if len(w) >= min_len]
                self.previous_board = [list(row) for row in self.board] # Deep copy
                self.previous_day_history = {}
                for p in self.players:
                    self.previous_day_history[str(p.user_id)] = {
                        'username': p.username,
                        'found_words': [w['word'] for w in p.submitted_words]
                    }
            
            # SPLIT POINTS LOGIC
            if self.game_type == 'split':
                self.calculate_split_scores()
            else:
                # Recalculate AI scores (since they were playing incrementally)
                for p in self.players:
                    if p.is_ai:
                        self._recalculate_player_score(p)

            # UPDATE RATINGS (Immediately at round end)
            # Calculate Proportional Rating changes based on FINAL scores
            print(f"[GameRoom] Calculating Proportional ratings at end of Round {self.current_round}")
            
            # Performance Efficiency (PE) & History Logic
            # Split into Registered and Guest pools to ensure isolation
            active_competitors = [p for p in self.players if (p.score > 0 or len(p.submitted_words) > 0 or len(p.invalid_words) > 0)]
            reg_players = [p for p in active_competitors if p.user_id > 0 and not p.is_guest]
            guest_players = [p for p in active_competitors if (p.is_guest or p.user_id <= 0)]
            
            # Calculate PE for everyone with isolation
            max_pe = 0.0
            
            # Determine if there are at least two active competitors to allow trophies
            multiple_active = len(active_competitors) > 1
            max_score = max(p.score for p in active_competitors) if active_competitors else 0

            # 1. Registered Players Pool
            reg_score_sum = sum(p.score for p in reg_players)
            reg_rating_sum = sum(p.rating for p in reg_players)
            
            if reg_rating_sum > 0:
                for p in reg_players:
                    expected = (p.rating / reg_rating_sum) * reg_score_sum
                    p.performance_efficiency = p.score / expected if expected > 0 else 0
                    max_pe = max(max_pe, p.performance_efficiency)
                    # Exceptional Round: 
                    # 1. Beating your fair share significantly (PE >= 1.5)
                    # 2. Or pure excellence (Score >= 100) even if solo/average
                    p.has_exceptional_round = (multiple_active and p.performance_efficiency >= 1.5) or (p.score >= 100)
            else:
                for p in reg_players:
                    p.performance_efficiency = 1.0
                    max_pe = max(max_pe, 1.0)
                    p.has_exceptional_round = (p.score >= 100)

            # 2. Guests Pool
            for p in guest_players:
                p.performance_efficiency = 1.0
                max_pe = max(max_pe, 1.0)
                p.has_exceptional_round = False # PE is 1.0 (Average)

            # Determine Notable Winners for Replay Tab
            # The user wants "enormous wins" to determine replay listing.
            # We list the round if max_pe >= 1.5 (Significant overperformance)
            # Determine Notable Winners for Replay Tab
            # The user previously wanted "enormous wins", but now wants "most recent 10 rounds".
            # So we remove the PE >= 1.5 restriction and log active rounds.
            # We still need at least 1 competitor to have a "winner" or purpose.
            if active_competitors:
                winners_data = [{'username': p.username, 'rating': p.rating, 'pe': p.performance_efficiency} for p in active_competitors if p.score == max_score]
                
                # Capture winners' words
                winner_words = []
                for p in active_competitors:
                    if p.score == max_score:
                        winner_words = [{'word': w['word'], 'points': w.get('points', 0), 'timestamp': w.get('time', time.time())} for w in p.submitted_words]
                        break

                self.winners_history.insert(0, {
                    'round': self.current_round,
                    'winners': winners_data,
                    'all_players': sorted([{'username': p.username, 'score': p.score, 'rating': p.rating, 'pe': p.performance_efficiency} for p in active_competitors], key=lambda x: x['score'], reverse=True),
                    'score': max_score,
                    'max_pe': max_pe,
                    'board': self.board,
                    'words': winner_words,
                    'bonus_word': self.bonus_word,
                    'round_duration': self.time_limit,
                    'round_start_time': self.round_start_time,
                    'game_type': self.game_type,
                    'timestamp': int(time.time() * 1000)
                })

                # Keep last 50 (or 10 as user mentioned? User said "most recent 10 rounds played no longer documented". 
                # This implies they expect to see recent ones. 50 cover 10.)
                if len(self.winners_history) > 50:
                    self.winners_history = self.winners_history[:50]
            # ABANDONMENT PENALTY (Requested Rule):
            # Track quitters and their bounty independently of the ranked/unranked format check.
            # 1. Real Quitters (those who explicitly left mid-round)
            active_quitters = [q for q in self.round_quitters if not is_player_guest(q) and not q.is_ai and (q.score > 0 or q.submitted_words or q.invalid_words)]
            
            # 2. Zombies (Soft Abandoners: humans who closed the tab but haven't been purged by the 10-minute timeout yet)
            # Threshold: 120 seconds since last poll during an active round.
            # Must exclude joined_mid_round to prevent penalizing mid-round joiners!
            now = time.time()
            soft_quitters = [p for p in self.players if not is_player_guest(p) and not p.is_ai and not getattr(p, 'joined_mid_round', False) and (now - p.last_active) > 120 and (p.score > 0 or p.submitted_words or p.invalid_words)]
            
            # Record soft-quitter IDs for penalty and distribution exclusion
            soft_quitter_ids = {p.user_id for p in soft_quitters}
            active_quitters.extend(soft_quitters) # Combine for uniform penalty reporting
            active_quitter_ids = {q.user_id for q in active_quitters}
            
            # Calculate total bounty (accumulated instant penalties + pending soft penalties)
            total_bounty = self.abandonment_bounty + (len(soft_quitters) * 16)
            
            # Skip rating updates for 500+ modes (unranked)
            wc_range = getattr(self, 'current_word_count_range', '100-200')
            wc_tuple = self._get_wc_tuple(wc_range)
            is_500plus = wc_tuple[0] >= 500
            
            competitive_human_starters = [
                p for p in self.players + self.round_quitters 
                if not getattr(p, 'is_ai', False) and not is_player_guest(p) and not getattr(p, 'joined_mid_round', False)
            ]
            
            # Ranked vs Unranked check, ENFORCING >= 2 competitive starters for Lobby Rooms
            is_strictly_ranked = True
            if is_500plus:
                is_strictly_ranked = False
                print(f"[GameRoom] Round {self.current_round} is inherently UNRANKED (Format: {board_format}, 500+: {is_500plus}).")
            elif not self.is_private and len(competitive_human_starters) <= 1:
                is_strictly_ranked = False
                print(f"[GameRoom] Round {self.current_round} UNRANKED: Only {len(competitive_human_starters)} human player(s) started from the beginning.")
                
            if not is_strictly_ranked:
                rating_changes = {p.user_id: 0 for p in self.players + self.round_quitters}
                
                # Zero out bounties to definitively block any ratings from artificially changing
                total_bounty = 0
                active_quitters = []
                self.abandonment_bounty = 0
            else:
                rating_changes = calculate_proportional_rating_change(self.players, is_private=self.is_private)
                
            # APPLY ABANDONMENT PENALTY & BOUNTY (Regardless of rank status - Discourages quitting everywhere)
            if total_bounty > 0 or active_quitters:
                # 1. Distribute the 'bounty' collected from all quitters.
                if total_bounty > 0:
                    # Remaining human players who actually played and stayed throughout (didn't exit and aren't zombies)
                    # ONLY competitive starters can collect the bounty!
                    active_humans_remaining = [
                        p for p in self.players 
                        if not p.is_ai and not is_player_guest(p) and not getattr(p, 'joined_mid_round', False) 
                        and p.user_id not in active_quitter_ids and (p.score > 0 or p.submitted_words or p.invalid_words)
                    ]

                    if active_humans_remaining:
                        bonus = total_bounty // len(active_humans_remaining)
                        for p in active_humans_remaining:
                            # Apply bonus and hard cap the total change at 16 (just like a normal great win)
                            total_change = rating_changes.get(p.user_id, 0) + bonus
                            rating_changes[p.user_id] = max(-16, min(16, total_change))
                            print(f"[GameRoom] Awarding {bonus} bonus points to {p.username} due to {total_bounty} penalty points from quitters. Final Change (capped): {rating_changes[p.user_id]}")
                    
                # 2. Ensure ALL quitters (real and soft) show as -16 in the UI report
                for q in active_quitters:
                    rating_changes[q.user_id] = -16

            # Connect for stats check - AND NOW PERSISTENCE
            try:
                db_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/morpheme.db'
                conn = sqlite3.connect(db_path, timeout=30)
            except:
                conn = None

            # Process unique participants for DB updates (prioritize current players for memory persistence)
            involved_ids = set()
            all_involved = []
            for p in self.players:
                if p.user_id not in involved_ids:
                    all_involved.append(p)
                    involved_ids.add(p.user_id)
            for q in self.round_quitters:
                if q.user_id not in involved_ids:
                    all_involved.append(q)
                    involved_ids.add(q.user_id)

            # Identification of quitters to avoid double-subtracting their instant penalty
            # Only those who were ALREADY penalized on exit should be skipped in the DB delta.
            # Soft-quitters (zombies) STILL need their DB penalty applied now.
            already_penalized_ids = {q.user_id for q in self.round_quitters if not any(sq.user_id == q.user_id for sq in soft_quitters)}

            for player in all_involved:
                change = int(rating_changes.get(player.user_id, 0))
                
                # AVOID DOUBLE-DIPPING: Penny was already deducted from DB and object if they were a REAL quitter.
                actual_db_delta = change
                if player.user_id in already_penalized_ids:
                    # Penalty already applied at exit, so don't apply it again here in the DB update.
                    actual_db_delta = 0 
                
                player.rating += actual_db_delta
                player.rating_change = change # Use full change (including -16) for UI display
                print(f"[GameRoom] Rating Update Applied: {player.username} ({player.user_id}) -> Change: {change}, New Rating: {player.rating}")
                
                # Skip saving history if player was inactive (0 score, no attempts)
                if player.score == 0 and not player.submitted_words and not player.invalid_words:
                    print(f"[GameRoom] Skipping DB history for inactive player {player.username} in round {self.current_round}")
                    continue

                # Persist Rating to DB - ALLOW SOLO FOR PRIVATE AND CUBE
                is_ranked_mode = (not self.is_solo or self.is_private or self.game_type == '3d') and not is_500plus
                if conn and player.user_id > 0 and is_ranked_mode:
                    try:
                        # 1. Update Global Profile Rating
                        # 2. Update Config-Specific Rating (GUARDED BY RANKED MODE)
                        if is_ranked_mode:
                            dims = self.board_dimensions
                            config_key = f"{self.game_type.replace('solo_', '')}|{dims}|{self.time_limit}"
                            
                            conn.execute('''
                                INSERT INTO user_ratings (user_id, config_key, rating) VALUES (?, ?, ?)
                                ON CONFLICT(user_id, config_key) DO UPDATE SET rating = rating + ?
                            ''', (player.user_id, config_key, player.rating, actual_db_delta))
                        
                        
                        # Competitive if at least 2 registered humans participated (EXCLUDE BOTS as human equivalents)
                        # EXCLUDE MID-ROUND JOINERS from competition pool
                        # INCORPORATE QUITTERS into competition count
                        human_participants = [p for p in all_involved if (not getattr(p, 'is_ai', False) and not getattr(p, 'is_guest', False) and not getattr(p, 'joined_mid_round', False) and (p.score > 0 or p.submitted_words or p.invalid_words))]
                        is_competitive = len(human_participants) >= 2
                        
                        # Allow global rank update if competitive AND (is ranked format OR there's an abandonment bonus to distribute)
                        if is_competitive and (is_ranked_mode or self.abandonment_bounty > 0):
                            conn.execute('UPDATE users SET rating = ?, games_played = games_played + 1 WHERE id = ?', (player.rating, player.user_id))
                            print(f"[GameRoom] DB Updated: User {player.username} global rating/stats (Competitive: {is_competitive}, Format: {board_format})")
                        else:
                            # Still update in-memory but skip global rank if not competitive or not normal
                            print(f"[GameRoom] DB SKIP: User {player.username} global rating update (Competitive: {is_competitive}, Format: {board_format})")
                        
                        if is_competitive and player.score > 0 and board_format in ['Normal', 'Cube']:
                             # Find max score among ALL participants (even bots) to see if this human actually won
                             max_all_score = max([p.score for p in all_involved]) if all_involved else 0
                             if player.score == max_all_score:
                                 conn.execute('UPDATE users SET wins = wins + 1 WHERE id = ?', (player.user_id,))
                        if is_competitive:
                            conn.execute('''
                                UPDATE users 
                                SET pe_count = pe_count + 1,
                                    avg_pe = ((avg_pe * pe_count) + ?) / (pe_count + 1),
                                    max_pe = MAX(max_pe, ?)
                                WHERE id = ?
                            ''', (player.performance_efficiency, player.performance_efficiency, player.user_id))
                        
                    except Exception as e:
                        print(f"[GameRoom] DB Save Error for {player.username}: {e}")

            if conn:
                conn.commit()
                conn.close()

            # CLEAR QUITTERS AND BOUNTY FOR NEXT ROUND
            self.round_quitters = []
            self.abandonment_bounty = 0

            # Reset timing flags for next intermission
            self.spinner_params_generated = False
            self.board_search_started = False
            return True
        
        # Check if intermission has expired
        if self.state == 'intermission' and self.time_remaining == 0:
            if self.game_type in ['accumulative', 'solo_accumulative', 'fcfs', 'split', 'standard', '3d']:
                # Signal that new round should start
                # This will be handled by RoomManager
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
            res = calculate_word_score(
                word, 
                self.bonus_word, 
                board_format=self.current_board_format, 
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
            
            for i, (player, timestamp, w_obj) in enumerate(finders):
                # No remainder distribution - everyone gets the same rounded-up value
                
                # Update word object with split metadata for frontend
                
                # Update word object with split metadata for frontend
                w_obj['split_points'] = final_points
                w_obj['shared_count'] = count
                w_obj['is_unique'] = (count == 1)
                w_obj['points'] = final_points
                w_obj['base_points'] = base_points
                
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
                possible_words, 
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
            
            print(f"[GameRoom] Bot {ai.username} pre-generated {len(ai.submitted_words)} words")

def calculate_word_score(word, bonus_word, board_format='Normal', path=None, bonus_cell=None, **kwargs):
    """Calculate points for a word using shared utility"""
    from scoring import calculate_word_score as shared_calc
    return shared_calc(word, bonus_word, board_format=board_format, path=path, bonus_cell=bonus_cell, **kwargs)


class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, GameRoom] = {}
        self.user_presence: Dict[str, float] = {} # {user_id_str: last_active_timestamp}
        self.lock = threading.Lock()
        self.board_generator = BoardGenerator()
        
        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._bg_cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        print("[RoomManager] Background cleanup thread started")
    
    def _bg_cleanup_loop(self):
        """Periodically clean up inactive rooms and players"""
        while True:
            try:
                time.sleep(60) # Run every minute
                # Routine 10-minute inactivity cleanup
                self.cleanup_rooms(timeout=600) 
                
                # Cleanup presence map
                with self.lock:
                    now = time.time()
                    self.user_presence = {uid: ts for uid, ts in self.user_presence.items() if (now - ts) < 600} # 10 min
            except Exception as e:
                import traceback
                print(f"[RoomManager] Error in background cleanup loop: {e}\n{traceback.format_exc()}")
                
    def create_room(self, room_id, game_type, time_limit, board_dimensions, min_rating=0, max_rating=9999, is_private=False):
        """Create a new game room or return an existing singleton for the configuration"""
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
            
            # Unlimited players for Accumulative, 8 for others
            if game_type in ['accumulative', 'solo_accumulative']:
                room.max_players = 9999
            else:
                room.max_players = 8

            self.rooms[room_id] = room
            return room
    
    def get_room(self, room_id):
        """Get room by ID"""
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
        """Immediately mark user as offline (for logout/beacon)"""
        if user_id:
            with self.lock:
                uid = str(user_id)
                if uid in self.user_presence:
                    del self.user_presence[uid]

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
        """Delete room"""
        with self.lock:
            if room_id in self.rooms:
                print(f"[RoomManager] Deleting room {room_id} (requested)")
                del self.rooms[room_id]
            else:
                print(f"[RoomManager] delete_room called for {room_id} but not found")
    
    def cleanup_rooms(self, timeout=600, spec_timeout=600):
        """Clean up empty or inactive rooms (defaults: 10m players, 30m spectators)"""
        rooms_to_delete = []
        
        # Iterate over a copy of keys to avoid modification issues
        for room_id, room in list(self.rooms.items()):
            try:
                # Check for inactive players
                room.check_inactivity(timeout, spec_timeout)
                
                # Close room if empty (excludes AI bots)
                # If a room only has bots and No humans/no spectators, it's considered empty.
                humans = [p for p in room.players if not p.is_ai]
                is_empty = (len(humans) == 0 and len(room.spectators) == 0)
                is_daily = (room.time_limit >= 7200)
                
                if is_empty and not is_daily:
                    print(f"[RoomManager] Marking room {room_id} for deletion (Empty)")
                    rooms_to_delete.append(room_id)
                    
            except Exception as e:
                print(f"[RoomManager] Error cleaning up room {room_id}: {e}")
        
        # Delete marked rooms
        for room_id in rooms_to_delete:
            self.delete_room(room_id)
        
        if rooms_to_delete:
            print(f"[RoomManager] Cleanup complete. Removed {len(rooms_to_delete)} rooms.")
    
    def start_round(self, room_id):
        """Start a new round with spinner and board generation"""
        log_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log'
        with open(log_path, 'a') as f:
            f.write(f"[START_ROUND] Enterprise for {room_id} at {time.time()}\n")
        
        room = self.get_room(room_id)
        if not room:
            with open(log_path, 'a') as f:
                f.write(f"[START_ROUND] Error: Room {room_id} NOT FOUND\n")
            return False
            
        # FORCE RESET FOR DEBUGGING - allow re-entry
        if room.starting_round:
            with open(log_path, 'a') as f:
                f.write(f"[START_ROUND] Resetting stuck starting_round flag for {room_id}\n")
            room.starting_round = False
            
        room.starting_round = True
        try:
            with open(log_path, 'a') as f:
                f.write(f"[START_ROUND] Pre-spinner for {room_id}\n")
            
            # Save previous round data before generating new one
            # SAFEGUARD: If intermission already snapped it, keep it!
            # Check if it's genuinely populated, not just an empty list
            has_prev = hasattr(room, 'previous_all_words') and room.previous_all_words
            if not has_prev and room.all_words:
                # Filter by the CURRENT round's min length
                min_len = getattr(room, 'current_min_length', 3)
                room.previous_all_words = [w for w in room.all_words if len(w) >= min_len]
                room.previous_board = [list(row) for row in room.board]
                print(f"[RoomManager] Saved {len(room.previous_all_words)} words to history (Fallback/Round {room.current_round})")
            elif has_prev:
                print(f"[RoomManager] Using existing history snapshot (intermission) for Round {room.current_round}")
            else:
                print(f"[RoomManager] WARNING: No words to save to history (Round {room.current_round})")
            
            print(f"[RoomManager] Generating spinner parameters for {room.board_dimensions}")
            
            # CLEAR BOARD & WORDS IMMEDIATELY to prevent "deceiving" the user during sync generation
            # This ensures they see an empty board until the optimized one is ready.
            # ONLY DO THIS FOR 500+ (IO) rounds though, to avoid flickering for normal ones
            wc_range = room.spinner_params.get('word_count_range', (0, 0))
            wc_tuple = room._get_wc_tuple(wc_range)
            is_500plus = wc_tuple[0] >= 500
            if is_500plus:
                rows_num, cols_num = map(int, room.board_dimensions.split('x'))
                room.board = [['' for _ in range(cols_num)] for _ in range(rows_num)]
                room.all_words = []
                room.complete_words = []
            
            # Generate spinner parameters (Use existing if already generated during intermission)
            if not room.spinner_params:
                is_24h = room.time_limit >= 7200
                room.spinner_params = SpinnerSet.generate_params(room.board_dimensions, is_24h)
                room.spinner_params_generated = True
                with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                    f.write(f"[game_room.py] start_round: Params generated at {time.time()}\n")
            else:
                with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                    f.write(f"[game_room.py] start_round: Using existing params at {time.time()}\n")
            
            # Get bonus word from dictionary
            fmt = room.spinner_params['board_format']
            wc_range = room.spinner_params.get('word_count_range', (0, 0))
            fmt = room.spinner_params['board_format']
            wc_range = room.spinner_params.get('word_count_range', '100-200')
            wc_tuple = room._get_wc_tuple(wc_range)
            
            # ALWAYS get a bonus word based on the spinner length
            bonus_word = self._get_bonus_word(room.spinner_params['bonus_word_length'], 
                                              room.spinner_params['dictionary'])
            
            # User Request: NO BONUS WORDS on Checkerboard boards
            if 'checkerboard' in str(room.spinner_params.get('board_format', '')).lower():
                bonus_word = "" # Explicitly clear it for this format
                room.spinner_params['bonus_word_length'] = 0
                room.current_bonus_word_length = 0
                print(f"[GameRoom] FORCED bonus word to 0 for Checkerboard format")
                
            room.bonus_word = bonus_word
            with open(log_path, 'a') as f:
                f.write(f"[START_ROUND] Calling generate_board for {room_id} (bonus={bonus_word})\n")
            
            # Generate board
            with open(log_path, 'a') as f:
                f.write(f"[START_ROUND] board_generator instance: {self.board_generator}\n")
            board, all_words, bonus_cell, updated_format = self.board_generator.generate_board(
                room.board_dimensions,
                bonus_word,
                room.spinner_params['word_count_range'],
                room.spinner_params['dictionary'],
                room.spinner_params['board_format'],
                room.spinner_params['min_word_length'],  # Only count words meeting min length
                room.spinner_params['difficulty']
            )
            # Update the room's format to include the mania letter if it was generated
            room.spinner_params['board_format'] = updated_format
            room.current_board_format = updated_format
            room.bonus_cell = bonus_cell
            print(f"[RoomManager] Board generation complete! Board: {board is not None}, Words: {len(all_words) if all_words else 0}")
            
            if board is None:
                print(f"[RoomManager] ERROR: Board generation failed!")
                return False
            
            # ATOMICITY FIX: Do not assign to room.all_words yet
            # room.board = board (Safe to assign)
            room.board = board
            
            # Store in temp var
            new_all_words = all_words
            print(f"DEBUG: Valid words sample (hidden): {new_all_words[:10]}")
            
            # Start the round immediately with timer
            room.current_round += 1
            
            # Update data atomically (words first)
            room.all_words = new_all_words
            room.current_min_length = room.spinner_params.get('min_word_length', 3)
            room.current_board_format = room.spinner_params.get('board_format', 'Normal')
            room.current_word_count_range = room.spinner_params.get('word_count_range', '100-200')
            room.current_dictionary = room.spinner_params.get("dictionary", "NWL")
            room.current_difficulty = room.spinner_params.get("difficulty", "Normal")
            room.current_bonus_word_length = room.spinner_params.get("bonus_word_length", 0)
            
            # Clear stale params so they don't leak into the next intermission display prematurely
            room.spinner_params = {}
            room.spinner_params_generated = False
            
            # ATOMICITY FIX: Everyone in the room at start_round is NO LONGER mid-round joined
            for p in room.players:
                p.joined_mid_round = False
            
            # SET ACTIVE STATE LAST to ensure all parameters are written before polling returns
            room.state = 'active'
            print(f"[RoomManager] Round {room.current_round} started and state set to ACTIVE")
            
            # Default custom end time
            room.custom_end_time = 0
            
            # SPLIT POINTS RANDOMIATION
            if room.game_type == 'split':
                import random
                random.shuffle(room.players)
                print(f"[RoomManager] Randomized player order for Split Points round")
            
            print(f"[RoomManager] Round {room.current_round} started - timer active!")
            
            # Clear previous words and scores
            for player in room.players:
                player.rating_change = 0 # Reset change display for new round start
                # Store current score for next round's comparison
                player.previous_round_score = player.score
                
                # Save submitted words to history
                player.previous_submitted_words = list(player.submitted_words)
                
                # Clear for new round
                player.submitted_words = []
                player.invalid_words = []
                player.score = 0
                player.found_bonus_word = False
                player.joined_mid_round = False
                player.has_exceptional_round = False
                player.performance_efficiency = 0.0
                
            # Daily Room Logic (>= 24h) - Reset at Midnight
            if room.time_limit >= 7200 and room.current_round > 1:
                self._apply_daily_reset(room)
                
            # Clear FCFS global list
                
            # Clear FCFS global list
            room.fcfs_found_words.clear()
            
            # DISABLED: Background complete solve is too slow and never finishes
            # Just use the fast solve results (30 words) for scoring and intermission
            print(f"[RoomManager] Using fast solve results for scoring")
            
            # Pre-calculate scores for the words we found
            room.solved_words_with_scores = {}
            for word in all_words:
                room.solved_words_with_scores[word] = calculate_word_score(
                    word, 
                    bonus_word, 
                    board_format=fmt, 
                    bonus_cell=bonus_cell,
                    board=board,
                    return_details=True
                )
            room.complete_words = all_words
            room.solving_complete = True
            print(f"[RoomManager] Scored {len(all_words)} words")
            
            # Generate AI turns before activating
            room.generate_ai_turns()

            # FINAL STEP: Activate the room
            # We do this LAST to avoid race conditions where state='active' but custom_end_time isn't set yet.
            room.state = 'active'
            with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                f.write(f"[game_room.py] start_round SUCCESS: state set to ACTIVE for {room_id} at {time.time()}\n")
            room.round_start_time = time.time()
            print(f"[RoomManager] Round {room.current_round} ACTIVATED at {room.round_start_time}. Custom End: {room.custom_end_time}")
            
            return True
            
        except Exception as e:
            import traceback
            with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                f.write(f"[game_room.py] start_round CRITICAL EXCEPTION: {e} at {time.time()}\n")
            print(f"[RoomManager] CRITICAL EXCEPTION in start_round: {e}")
            traceback.print_exc()
            return False
        finally:
            with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                f.write(f"[game_room.py] start_round FINALLY for {room_id} at {time.time()}\n")
            print(f"[RoomManager] start_round finally block for room {room_id}")
            # Always reset the flag, even if board generation fails
            room.starting_round = False
    
    def _apply_daily_reset(self, room):
        """Perform daily cleanup and midnight alignment for 24h rooms"""
        print(f"[RoomManager] Daily Reset: Clearing ALL players and spectators for room {room.room_id}")
        
        # Store existing players for rating reference
        for p in room.players:
            room.past_players[str(p.user_id)] = p
        
        # CLEAR THE LISTS: Forces all clients to re-join
        room.players = []
        room.spectators = []
        
        # RE-ALIGN TO NEXT MIDNIGHT
        now_dt = datetime.datetime.now()
        midnight = (now_dt + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        delta = (midnight - now_dt).total_seconds()
        room.custom_end_time = time.time() + delta
        print(f"[RoomManager] Daily room: re-aligned to next midnight ({midnight})")

    def generate_spinner_params(self, room_id):
        """Generate Spinner Set parameters for next round (called immediately on round end)"""
        room = self.get_room(room_id)
        if not room:
            print(f"[RoomManager] ERROR: Room {room_id} not found")
            return False
            
        # TWO-STAGE GUARD: Check if already done OR if currently in progress
        if getattr(room, 'spinner_params_generated', False) or getattr(room, 'spinner_params_loading', False):
            return True
            
        # ATOMICITY GUARD: Set loading flag immediately
        room.spinner_params_loading = True
        print(f"[RoomManager] Starting generation of Spinner Set parameters for room {room_id}")
        
        try:
            # Generate spinner parameters for next round
            is_24h = room.time_limit >= 7200
            new_params = SpinnerSet.generate_params(room.board_dimensions, is_24h)
            
            # ATOMIC SWAP: New params applied first, then final flag set
            room.spinner_params = new_params
            room.spinner_params_generated = True
            
            print(f"[RoomManager] Spinner params generated and assigned: {room.spinner_params}.")
            return True
        finally:
            room.spinner_params_loading = False
    
    def start_board_search(self, room_id):
        """Start board search using Spinner Set parameters (called at 15s remaining)"""
        room = self.get_room(room_id)
        if not room:
            print(f"[RoomManager] ERROR: Room {room_id} not found")
            return False
            
        # TWO-STAGE GUARD: Check if already done OR if currently in progress
        if getattr(room, 'board_search_started', False) or getattr(room, 'board_search_loading', False):
            return True
            
        if not room.spinner_params_generated:
            print(f"[RoomManager] WARNING: Spinner params not generated yet - cannot start search.")
            return False
            
        # ATOMICITY GUARD: Set loading flag immediately
        room.board_search_loading = True
        print(f"[RoomManager] Starting board search process at 15s remaining for room {room_id}")
        
        try:
            fmt = room.spinner_params['board_format']
            wc_range = room.spinner_params.get('word_count_range', '100-200')
            wc_tuple = room._get_wc_tuple(wc_range)
            
            # ALWAYS get a bonus word based on the spinner length
            bonus_word = self._get_bonus_word(room.spinner_params['bonus_word_length'], 
                                              room.spinner_params['dictionary'])
            
            # User Request: NO BONUS WORDS on Checkerboard boards
            if 'checkerboard' in str(room.spinner_params.get('board_format', '')).lower():
                bonus_word = "" # Explicitly clear it for this format
                room.spinner_params['bonus_word_length'] = 0
                room.next_round_bonus_length = 0
                print(f"[GameRoom] FORCED bonus word length to 0 for Checkerboard search")
                
            room.next_round_bonus = bonus_word
            print(f"[RoomManager] Bonus word selected: '{bonus_word}'")
            
            # If 500+, CLEAR THE CURRENT BOARD NOW so user doesn't see a "deceptive" board
            # while optimization is running in the background.
            if is_500plus:
                print(f"[RoomManager] 500+ detected: Clearing active board ahead of time")
                rows_num, cols_num = map(int, room.board_dimensions.split('x'))
                room.board = [['' for _ in range(cols_num)] for _ in range(rows_num)]
                room.all_words = []
                room.complete_words = []
            
            room.board_search_started = True
            
            # Start board generation in background thread
            def generate_in_background():
                print(f"[RoomManager] Background board generation started...")
                board, all_words, bonus_cell, updated_format = self.board_generator.generate_board(
                    room.board_dimensions,
                    bonus_word,
                    room.spinner_params['word_count_range'],
                    room.spinner_params['dictionary'],
                    room.spinner_params['board_format'],
                    room.spinner_params['min_word_length'],
                    room.spinner_params['difficulty']
                )
                
                if board is None:
                    print(f"[RoomManager] ERROR: Board generation failed!")
                    room.board_search_started = False # Reset to allow retry
                    return
                
                # Update the room's format for the next round
                room.spinner_params['board_format'] = updated_format
                room.next_round_board = board
                room.next_round_words = all_words
                room.next_round_bonus_cell = bonus_cell
                room.next_round_format = updated_format # Cache for when the round actually starts
                
                # User requirement: "When you show the Spinner Set Popup, that means you have found a board."
                # So we only set generated=True AFTER the background thread succeeds!
                room.spinner_params_generated = True
                
                print(f"[RoomManager] Board found and params revealed! Words: {len(all_words) if all_words else 0}")
            
            thread = threading.Thread(target=generate_in_background, daemon=True)
            thread.start()
            return True
        finally:
            room.board_search_loading = False
    
    
    def start_next_round(self, room_id):
        """Start next round with pre-generated board (called at 0s remaining)"""
        room = self.get_room(room_id)
        if not room:
            print(f"[RoomManager] ERROR: Room {room_id} not found")
            return False
            
        if getattr(room, 'starting_round', False):
            print(f"[RoomManager] Round start already in progress for room {room_id}, skipping duplicate call.")
            return False
            
        room.starting_round = True
        print(f"[RoomManager] start_next_round called for room {room_id}")
        
        try:
            
            print(f"[RoomManager] checking fallback for room {room_id} (game={room.game_type})")
            # Check if board is ready
            # Fallback if next_round_board is empty
            if not room.next_round_board or not room.next_round_words:
                print(f"[RoomManager] WARNING: Board not ready, falling back to start_round")
                return self.start_round(room_id)
            
            # CLEAR BOARD & WORDS IMMEDIATELY if we are about to generate a new one
            # This handles the fallback Case or any state where we want to avoid stale data
            # ONLY DO THIS FOR 500+ (IO) rounds though, to avoid flickering for normal ones
            wc_range = room.spinner_params.get('word_count_range', (0, 0))
            wc_tuple = room._get_wc_tuple(wc_range)
            is_500plus = wc_tuple[0] >= 500
            
            if (not room.next_round_board or not room.next_round_words) and is_500plus:
                rows_num, cols_num = map(int, room.board_dimensions.split('x'))
                room.board = [['' for _ in range(cols_num)] for _ in range(rows_num)]
                room.all_words = []
                room.complete_words = []

            # SAVE PREVIOUS ROUND DATA (Persistence for "Previous Day" tab)
            # CHECK if we already snapshotted during intermission. If so, KEEP IT!
            # Use getattr with explicit check for None and emptiness to avoid bugs
            has_prev_all = getattr(room, 'previous_all_words', None) is not None and len(room.previous_all_words) > 0
            has_prev_hist = getattr(room, 'previous_day_history', None) is not None and len(room.previous_day_history) > 0
            
            print(f"DEBUG: start_next_round persistence check. HasHist={has_prev_hist}, HasAll={has_prev_all}")
            if has_prev_hist:
                 print(f"DEBUG: History keys: {list(room.previous_day_history.keys())}")

            if not has_prev_all or not has_prev_hist:
                print("DEBUG: Snapshot IS missing or empty. Creating new snapshot now.")
                
                # Filter by min_word_length (fallback) - Use current active round's min length
                min_len = getattr(room, 'current_min_length', 3)
                source_words = list(room.complete_words) if (getattr(room, 'complete_words', None) and len(room.complete_words) > 0) else list(room.all_words)
                
                if not has_prev_all:
                    room.previous_all_words = [w for w in source_words if len(w) >= min_len]
                
                # SNAPSHOT HISTORY BEFORE OVERWRITING BOARD
                if room.time_limit >= 7200 and not has_prev_hist:
                    print(f"[RoomManager] Daily Reset: Snapshotting history (fallback/start_next) before new round")
                    room.previous_day_history = {}
                    for p in room.players:
                        room.previous_day_history[str(p.user_id)] = {
                            'username': p.username,
                            'found_words': [w['word'] for w in p.submitted_words]
                        }
                    print(f"[RoomManager] Saved daily history for {len(room.players)} players")
            else:
                print(f"[RoomManager] SKIPPING snapshot - Using existing history from intermission")

            # Use pre-generated board and words
            room.board = room.next_round_board
            room.all_words = room.next_round_words
            # Snapshot parameters for the round
            room.current_min_length = room.spinner_params.get('min_word_length', 3)
            room.current_board_format = room.spinner_params.get('board_format', 'Normal')
            room.current_word_count_range = room.spinner_params.get('word_count_range', '100-200')
            room.current_dictionary = room.spinner_params.get("dictionary", "NWL")
            room.current_difficulty = room.spinner_params.get("difficulty", "Normal")
            room.current_bonus_word_length = room.spinner_params.get("bonus_word_length", 0)
            
            # ATOMICITY FIX: SET ACTIVE STATE NOW that parameters are fully populated
            room.state = 'active'
            room.round_start_time = time.time()
            
            # Reset flags for the NEW round's next intermission
            room.spinner_params_generated = False
            room.board_search_started = False
            room.spinner_params = {} # CLEAR STALE PARAMS
            
            # SPLIT POINTS RANDOMIATION
            if room.game_type == 'split':
                import random
                random.shuffle(room.players)
                print(f"[RoomManager] Randomized player order for Split Points round")
                
            print(f"[RoomManager] Round {room.current_round} started with pre-generated board!")

            # 1. Calculate ELO changes based on FINAL scores of previous round
            # MOVED TO check_and_update_state (Start of Intermission)

            # Reset all players FIRST, then generate AI turns so bot words aren't wiped.
            for player in room.players:
                player.rating_change = 0

                # Store current score for next round's comparison
                player.previous_round_score = player.score

                # SAVE PREVIOUS WORDS
                player.previous_submitted_words = list(player.submitted_words)

                # Clear for new round
                player.submitted_words = []
                player.invalid_words = []
                player.score = 0
                player.found_bonus_word = False
                player.joined_mid_round = False
                player.has_exceptional_round = False
                player.performance_efficiency = 0.0
            print(f"[RoomManager] Round {room.current_round} started with pre-generated board!")

            # Generate AI turns AFTER player reset so bot pre-generated words are not immediately erased.
            room.generate_ai_turns()

            print(f"[RoomManager] Round {room.current_round} started with pre-generated board!")
                
            # PERSISTENCE: If this is a 24h room, clear the player list for the new day
            if room.time_limit >= 7200:
                self._apply_daily_reset(room)
                
            # Clear FCFS global list
            if hasattr(room, 'fcfs_found_words'):
                room.fcfs_found_words.clear()
            
            # Pre-calculate scores with breakdown
            room.solved_words_with_scores = {}
            for word in room.all_words:
                room.solved_words_with_scores[word] = calculate_word_score(
                    word, 
                    room.bonus_word, 
                    board_format=room.current_board_format,
                    bonus_cell=room.bonus_cell,
                    board=room.board,
                    return_details=True
                )
            room.complete_words = room.all_words
            room.solving_complete = True
            print(f"[RoomManager] Scored {len(room.all_words)} words")
            
            # Clear next round data
            room.next_round_board = []
            room.next_round_words = []
            room.next_round_bonus = ''
            
            return True
            
        except Exception as e:
            import traceback
            print(f"[RoomManager] CRITICAL ERROR in start_next_round: {e}")
            traceback.print_exc()
            return False
        finally:
            if room:
                room.starting_round = False
    
    def _get_bonus_word(self, length, dictionary):
        """Get a bonus word of specified length"""
        import time
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
            f.write(f"[game_room.py] _get_bonus_word ENTERED (len={length}, dict={dictionary}) at {time.time()}\n")
        from word_validator import word_validator
        
        # Get all words of the specified length
        if dictionary == 'CSW':
            words = [w for w in word_validator.csw_words if len(w) == length]
        else:
            words = [w for w in word_validator.nwl_words if len(w) == length]
        
        # Return random word
        import random
        result = random.choice(words) if words else 'A' * length
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
            f.write(f"[game_room.py] _get_bonus_word SUCCESS: {result} at {time.time()}\n")
        return result
    
    
    def save_round_history(self, room):
        """Save the results of the JUST COMPLETED round to the database"""
        if room.is_solo:
            print(f"[RoomManager] SKIPPING history save for SOLO room {room.room_id}")
            return
            
        import sqlite3
        import json
        
        # Guard against double saving
        if room.last_saved_round >= room.current_round:
            return
        
        try:
            conn = sqlite3.connect('morpheme.db', timeout=30)
            board_json = json.dumps(room.board)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
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
                # User Objective: Fastest sequence of 20 valid words in a row.
                # Fallback: Average WPM of entire round if at least 5 words found.
                final_wpm = 0.0
                if len(words_data) >= 5:
                    # Ensure chronological order
                    sorted_entries = sorted(words_data, key=lambda x: x['timestamp'])
                    if len(sorted_entries) >= 20:
                        peak_wpm = 0.0
                        for i in range(len(sorted_entries) - 19):
                            # Window of 20 words: from index i to i+19
                            t_first = sorted_entries[i]['timestamp']
                            t_last = sorted_entries[i+19]['timestamp']
                            dt = t_last - t_first
                            if dt > 0.001:
                                current_burst_wpm = (20.0 * 60.0) / dt
                                if current_burst_wpm > peak_wpm:
                                    peak_wpm = current_burst_wpm
                        final_wpm = peak_wpm
                    else:
                        # Full sequence average for < 20 words
                        t_first = sorted_entries[0]['timestamp']
                        t_last = sorted_entries[-1]['timestamp']
                        dt = t_last - t_first
                        if dt > 0.001:
                            final_wpm = (len(sorted_entries) * 60.0) / dt
                
                conn.execute('''
                    INSERT INTO round_history (user_id, room_id, game_type, round_number, board_json, words_json, total_score, round_start_time, round_duration, timestamp, user_rating, performance_ratio, best_word, best_word_score, board_dimensions, wpm, total_words_avail)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (p.user_id, room.room_id, room.game_type, room.current_round, board_json, json.dumps(words_data), p.score, room.round_start_time, room.time_limit, timestamp, p.rating, p.performance_efficiency, best_word_text, best_word_val, room.board_dimensions, final_wpm, len(room.all_words)))
            
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



# Global instance
room_manager = RoomManager()

# DEBUG PATCH for submit_word
# Appending print statement to verify persistence
