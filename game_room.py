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
    pending_ai_submissions: List[Dict] = field(default_factory=list)

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
    previous_board: List[List[str]] = field(default_factory=list) # Snapshot of yesterday's board
    all_words: List[str] = field(default_factory=list)  # Fast initial word list
    previous_all_words: List[str] = field(default_factory=list) # Previous round words
    previous_day_history: Dict = field(default_factory=dict) # Snapshot of yesterday's game (Found/Missed)
    complete_words: List[str] = field(default_factory=list)  # Complete word list from background solving
    solved_words_with_scores: Dict[str, int] = field(default_factory=dict)  # Pre-computed word scores
    bonus_word: str = ''
    bonus_letter_cell: tuple = None # (r, c) for the Bonus Letter format
    solving_complete: bool = False  # Track if background solving is done
    
    # FCFS Mode specific
    fcfs_found_words: set = field(default_factory=set)
    
    # Spinner parameters
    spinner_params: Dict = field(default_factory=dict)
    current_min_length: int = 3  # Stores active round's min length (decoupled from spinner_params which updates early)
    
    # Next round pre-generation (for Accumulative timing)
    spinner_params_generated: bool = False  # Track if spinner set generated for next round
    board_search_started: bool = False      # Track if board search started
    next_round_board: List[List[str]] = field(default_factory=list)  # Store pre-generated board
    next_round_words: List[str] = field(default_factory=list)  # Store pre-generated word list
    next_round_bonus: str = ''  # Store bonus word for next round
    
    # Players
    players: List[Player] = field(default_factory=list)
    past_players: Dict[str, Player] = field(default_factory=dict) # Archive of players for persistence
    
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
            elif not is_daily and self.state == 'active' and (self.time_limit - self.time_remaining) > 5:
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
            elif not is_daily and self.state == 'active' and (self.time_limit - self.time_remaining) > 5:
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
        elif self.state == 'active' and (self.time_limit - self.time_remaining) > 5 and not is_daily and not was_already_in_room:
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

    def check_inactivity(self, timeout=420, spec_timeout=1800): 
        """Remove players and spectators who haven't been active for their respective timeout seconds"""
        now = time.time()
        active_players = []
        players_removed = False
        is_daily = self.time_limit >= 7200
        
        for p in self.players:
            age = now - p.last_active
            # Keep active players OR keep all players if 24h room (daily persistence)
            if is_daily or (age < timeout):
                active_players.append(p)
            else:
                log_msg = f"[GameRoom] Removing inactive player {p.username} (ID={p.user_id}) in room {self.room_id} (inactive for {age:.1f}s)\n"
                print(log_msg.strip())
                with open('inactivity_debug.log', 'a') as f:
                    f.write(f"{datetime.datetime.now()} {log_msg}")
                players_removed = True
        
        if players_removed:
            self.players = active_players

        # Check spectators
        active_spectators = []
        specs_removed = False
        for p in self.spectators:
            age = now - p.last_active
            # Spectators get a longer timeout (default 30 mins)
            if is_daily or (age < spec_timeout):
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
    
    def submit_word(self, user_id, word):
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
            
            if self.spinner_params.get('board_format') == 'Penalty' and not is_24h and len(word) >= min_len:
                # Is it on the board?
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
                    'points': penalty_points
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
        
        # Calculate score for this word
        # 4. Calculate score
        points = calculate_word_score(final_word, self.bonus_word)
        
        # Check Bonus Letter tile usage (server-side path check)
        uses_bonus_cell = False
        if self.board_format == 'Bonus Letter' and self.bonus_letter_cell:
            # Check if target cell is potentially part of this word
            from board_generator import BoardGenerator
            bg = BoardGenerator()
            # If the cell LETTER is in the word, then we can check if it's reachable.
            # But the simplest is: does the letter at that cell match any letter in final_word?
            # And then: is there a path using that cell?
            # For performance, we check if the cell's letter is even in the word first.
            r_b, c_b = self.bonus_letter_cell
            target_char = self.board[r_b][c_b].upper()
            if target_char in final_word.upper() or ('Q' in target_char and 'QU' in final_word.upper()):
                 # Exhaustive path check for this specific word using this specific cell
                 if self.is_cell_on_word_path(final_word, r_b, c_b):
                     uses_bonus_cell = True

        # Add word as structured data
        player.submitted_words.append({
            'word': final_word,
            'points': points,
            'time': time.time(),
            'uses_bonus_cell': uses_bonus_cell
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

        if reg_rating_sum > 0:
            for p in reg_players:
                expected = (p.rating / reg_rating_sum) * reg_score_sum
                p.performance_efficiency = p.score / expected if expected > 0 else 0.0
                # Trophy only if more than one player is playing, and performance is exceptionally high
                p.has_exceptional_round = multiple_players and p.performance_efficiency >= 1.5
        else:
            for p in reg_players:
                p.performance_efficiency = 1.0
                p.has_exceptional_round = False

        # 2. Guests: Use solo baseline (PE=1.0) so they don't affect pool but can still earn trophies on raw score
        for p in guest_players:
            p.performance_efficiency = 1.0
            p.has_exceptional_round = False
    
    def _recalculate_player_score(self, player):
        """
        Recalculate player score from submitted words sequentially.
        """
        sorted_words = sorted(player.submitted_words, key=lambda x: x.get('time', 0))
        current_score = 0
        
        # Scoring Weights for Valued Letters
        LETTER_VALUES = {
            'A':2, 'B':4, 'C':4, 'D':3, 'E':1, 'F':5, 'G':3, 'H':5, 'I':2, 'J':10,
            'K':6, 'L':3, 'M':4, 'N':2, 'O':2, 'P':4, 'Q':10, 'R':2, 'S':2, 'T':2,
            'U':4, 'V':5, 'W':5, 'X':9, 'Y':5, 'Z':9
        }

        for w_obj in sorted_words:
            word = w_obj['word']
            points = w_obj.get('points', 0)
            
            # Apply Special Format Scoring
            if self.board_format == 'Bonus Letter' and self.bonus_letter_cell:
                # Check if this word used the bonus cell
                # We need to know the path. If path is missing, we use a simple "contains"
                # as a fallback, but the BoardGen only returns words on board.
                # Realistically we need the path. Let's assume +5 if it uses it.
                # For now, let's assume word-based bonus if word has the letter?
                # No, user said "Take a letter from the board" (cell).
                # If path isn't tracked in submitted_words, we might need to add it.
                if w_obj.get('uses_bonus_cell'):
                    points += 5
            
            elif self.board_format == 'Valued Letters':
                # Scrabble-style scoring
                points = sum(LETTER_VALUES.get(c, 0) for c in word.upper())
                if word.upper() == self.bonus_word.upper():
                    points += len(word) # Bonus word still adds length

            current_score = max(0, current_score + points)
            w_obj['points'] = points # Update points in object for UI

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
        elif time_remaining <= 15 and not self.board_search_started:
            return 'search'
        elif not self.spinner_params_generated:
            return 'spinner'
        
        return None
    
    def is_cell_on_word_path(self, word, target_r, target_c):
        """Check if target_r, target_c is part of ANY path that forms 'word'"""
        rows, cols = len(self.board), len(self.board[0])
        word = word.upper()
        
        def dfs(r, c, index, visited):
            if index >= len(word):
                return True
            
            # Check neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        cell_content = self.board[nr][nc].upper()
                        # Handle Q/QU and Either/Or
                        options = cell_content.split('/') if '/' in cell_content else [cell_content]
                        
                        for letter in options:
                            m_len = 0
                            if letter == 'Q':
                                if word[index:index+2] == 'QU': m_len = 2
                                elif word[index] == 'Q': m_len = 1
                            elif word[index] == letter:
                                m_len = 1
                            
                            if m_len > 0:
                                if dfs(nr, nc, index + m_len, visited | {(nr, nc)}):
                                    return True
            return False

        # We must find if target cell is on ANY valid path
        # So we start DFS from everywhere? No, that's regular search.
        # Efficient: DFS from every start cell, but if we pass through target_r, target_c, mark it.
        
        def dfs_with_target(r, c, index, visited, hit_target):
            hit_target = hit_target or (r == target_r and c == target_c)
            
            # Match current cell
            cell_content = self.board[r][c].upper()
            options = cell_content.split('/') if '/' in cell_content else [cell_content]
            
            found = False
            for letter in options:
                m_len = 0
                if letter == 'Q':
                    if word[index:index+2] == 'QU': m_len = 2
                    elif word[index] == 'Q': m_len = 1
                elif word[index] == letter:
                    m_len = 1
                
                if m_len > 0:
                    next_idx = index + m_len
                    # Base case: word complete
                    if next_idx == len(word):
                        if hit_target: return True
                        continue
                    
                    # Continue DFS
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                                if dfs_with_target(nr, nc, next_idx, visited | {(r, c)}, hit_target):
                                    return True
            return False

        for r in range(rows):
            for c in range(cols):
                if dfs_with_target(r, c, 0, set(), False):
                    return True
        return False

    def check_and_update_state(self):
        """Check timers and update game state accordingly"""
        # Check if active round has expired
        if self.state == 'active':
            # NEW: Process pending AI moves incrementally
            self.update_ai_turns()
            
            if self.time_remaining == 0:
                # FINAL check for any remaining AI turns (unlikely but safe)
                self.update_ai_turns()
            
            self.state = 'intermission'
            self.intermission_start_time = time.time()
            
            # IMMEDIATE SNAPSHOT (24h Rooms): Save history now so it is available during intermission
            if self.time_limit >= 7200:
                print(f"[GameRoom] Snapshotting history at start of intermission for room {self.room_id}")
                # Filter previous_all_words by the CURRENT round's min length (not next round's!)
                min_len = getattr(self, 'current_min_length', 3)
                self.previous_all_words = [w for w in self.all_words if len(w) >= min_len]
                self.previous_board = [row[:] for row in self.board] # Deep copy current board
                self.previous_day_history = {
                    'board': self.previous_board,
                    'all_words': self.previous_all_words,
                    'players': {}
                }
                for p in self.players:
                    self.previous_day_history['players'][str(p.user_id)] = {
                        'username': p.username,
                        'found_words': [w['word'] for w in p.submitted_words]
                    }
            
            # SPLIT POINTS LOGIC
            if self.game_type == 'split':
                self.calculate_split_scores()

            # UPDATE RATINGS (Immediately at round end)
            # User Request: "do not include any board from any Format other than Normal in Skill Rankings and Leaderboards."
            if self.board_format == 'Normal':
                print(f"[GameRoom] Calculating Proportional ratings at end of Round {self.current_round}")
                rating_changes = calculate_proportional_rating_change(self.players)
            else:
                print(f"[GameRoom] Round format '{self.board_format}' is NOT Normal. Ratings unchanged.")
                rating_changes = {p.user_id: 0 for p in self.players}
            
            # Performance Efficiency (PE) & History Logic
            # Split into Registered and Guest pools to ensure isolation
            active_competitors = [p for p in self.players if (p.score > 0 or len(p.submitted_words) > 0 or len(p.invalid_words) > 0)]
            reg_players = [p for p in active_competitors if p.user_id > 0 and not p.is_guest]
            guest_players = [p for p in active_competitors if (p.is_guest or p.user_id <= 0)]
            
            # Calculate PE for everyone with isolation
            max_pe = 0.0
            
            # Determine if there are at least two active competitors to allow trophies
            multiple_active = len(active_competitors) > 1

            # 1. Registered Players Pool
            reg_score_sum = sum(p.score for p in reg_players)
            reg_rating_sum = sum(p.rating for p in reg_players)
            
            if reg_rating_sum > 0:
                for p in reg_players:
                    expected = (p.rating / reg_rating_sum) * reg_score_sum
                    p.performance_efficiency = p.score / expected if expected > 0 else 0
                    max_pe = max(max_pe, p.performance_efficiency)
                    # Trophy only if more than one active player, and performance is exceptionally high
                    p.has_exceptional_round = multiple_active and p.performance_efficiency >= 1.5
            else:
                for p in reg_players:
                    p.performance_efficiency = 1.0
                    max_pe = max(max_pe, 1.0)
                    p.has_exceptional_round = False

            # 2. Guests Pool
            for p in guest_players:
                p.performance_efficiency = 1.0
                max_pe = max(max_pe, 1.0)
                p.has_exceptional_round = False

            # Determine Notable Winners for Replay Tab
            # The user wants "enormous wins" to determine replay listing.
            # We list the round if max_pe >= 1.5 (Significant overperformance)
            # Determine Notable Winners for Replay Tab
            # The user previously wanted "enormous wins", but now wants "most recent 10 rounds".
            # So we remove the PE >= 1.5 restriction and log active rounds.
            # We still need at least 1 competitor to have a "winner" or purpose.
            if active_competitors:
                max_score = max(p.score for p in active_competitors)
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
            else:
                print(f"[GameRoom] Round {self.current_round} skipped for history (No active competitors)")

            # Connect for stats check - AND NOW PERSISTENCE
            try:
                conn = sqlite3.connect('morpheme.db', timeout=30)
            except:
                conn = None

            for player in self.players:
                change = int(rating_changes.get(player.user_id, 0))
                player.rating += change
                player.rating_change = change
                print(f"[GameRoom] Rating Update Applied: {player.username} ({player.user_id}) -> Change: {change}, New Rating: {player.rating}")
                
                 # Skip saving history if player was inactive (0 score, no attempts)
                if player.score == 0 and not player.submitted_words and not player.invalid_words:
                    print(f"[GameRoom] Skipping DB history for inactive player {player.username} in round {self.current_round}")
                    continue

                # Persist Rating to DB - SKIP FOR SOLO
                if conn and player.user_id > 0 and not self.is_solo:
                    try:
                        # 1. Update Global Profile Rating
                        if self.game_type in ['accumulative', 'solo_accumulative']: # Only standard mode affects global rating? Or all? Usually all.
                             # But let's check user_ratings table for specific config persistence
                             pass

                        # 2. Update Config-Specific Rating
                        dims = f"{len(self.board)}x{len(self.board[0])}"
                        config_key = f"{self.game_type}|{dims}|{self.time_limit}"
                        
                        conn.execute('''
                            INSERT INTO user_ratings (user_id, config_key, rating) VALUES (?, ?, ?)
                            ON CONFLICT(user_id, config_key) DO UPDATE SET rating = rating + ?
                        ''', (player.user_id, config_key, player.rating, change))
                        
                        active_scorers = [p for p in self.players if p.score > 0 and not p.is_ai]
                        is_competitive = len(active_scorers) >= 2

                        if is_competitive:
                            conn.execute('UPDATE users SET rating = ?, games_played = games_played + 1 WHERE id = ?', (player.rating, player.user_id))
                        else:
                            conn.execute('UPDATE users SET rating = ? WHERE id = ?', (player.rating, player.user_id))
                        print(f"[GameRoom] DB Updated: User {player.username} rating/stats (Competitive: {is_competitive})")
                        
                        if is_competitive and player.score > 0 and player.score == max([p.score for p in active_competitors]):
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



            # Reset timing flags for next intermission
            self.spinner_params_generated = False
            self.board_search_started = False
            return True
        
        # Check if intermission has expired
        if self.state == 'intermission' and self.time_remaining == 0:
            if self.game_type in ['accumulative', 'solo_accumulative', 'fcfs', 'split']:
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
            # Sort finders by submission time to break ties in remainder distribution
            finders.sort(key=lambda x: x[1])
            
            count = len(finders)
            base_points = calculate_word_score(word, self.bonus_word)
            
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

    def prepare_ai_turns(self):
        """Generate AI word submissions at the start of the round to play incrementally"""
        ais = [p for p in self.players if p.is_ai]
        if not ais:
            return
            
        print(f"[GameRoom] Preparing turns for {len(ais)} bots in room {self.room_id}")
        
        # Use existing logic from PrivateMatchManager (adapted)
        from private_match_logic import private_match_manager
        
        for ai in ais:
            ai.pending_ai_submissions = [] # Reset
            
            # Filter all_words by min_word_length
            min_len = self.spinner_params.get('min_word_length', 3)
            possible_words = [w for w in self.all_words if len(w) >= min_len]
            
            words_data, _ = private_match_manager.generate_ai_submission(
                ai.ai_rating, 
                possible_words, 
                self.bonus_word,
                duration=self.time_limit
            )
            
            # Convert time_offsets to absolute round timestamps
            start = self.round_start_time
            if start == 0: start = time.time() # Handle edge case
            
            for wd in words_data:
                wd['time'] = start + wd.pop('time_offset', 0)
                ai.pending_ai_submissions.append(wd)
            
            # Sort by time so we process them in sequence
            ai.pending_ai_submissions.sort(key=lambda x: x['time'])
            print(f"[GameRoom] Bot {ai.username} pre-loaded with {len(ai.pending_ai_submissions)} pending words.")

    def update_ai_turns(self):
        """Process pending AI moves based on current time. Called during room polling."""
        now = time.time()
        ais = [p for p in self.players if p.is_ai]
        
        for ai in ais:
            if not ai.pending_ai_submissions:
                continue
                
            submissions_this_tick = []
            remaining = []
            
            for wd in ai.pending_ai_submissions:
                if wd['time'] <= now:
                    submissions_this_tick.append(wd)
                else:
                    remaining.append(wd)
            
            if submissions_this_tick:
                # Process these words as if they were typed now
                for wd in submissions_this_tick:
                    # Logic similar to submit_word but simplified since AI words are pre-validated as "on board"
                    # However, we must check FCFS collision
                    word = wd['word']
                    
                    if self.game_type == 'fcfs' and word in self.fcfs_found_words:
                        # Someone else got it first! Bot "loses" the word.
                        continue
                    
                    if self.game_type == 'fcfs':
                        self.fcfs_found_words.add(word)

                    ai.submitted_words.append(wd)
                    if word == self.bonus_word:
                        ai.found_bonus_word = True
                
                # Update score
                self._recalculate_player_score(ai)
                ai.pending_ai_submissions = remaining
                print(f"[GameRoom] UI Update: Bot {ai.username} revealed {len(submissions_this_tick)} words. Score: {ai.score}")
            
            # Also calculate invalid words points (0, but we might want to track count)


def calculate_proportional_rating_change(players):
    """
    Calculate rating changes based on Proportional Share system (from rating.java).
    Expected Score = (TotalScore / TotalRating) * PlayerRating
    Change based on deviation from Expected Score relative to a 75% baseline.
    """
    
    # 1. Check for integrity (User Rule: If any late joiner exists mixed with full players, void the round ratings)
    # Late joiners in Split/FCFS steal points/words, unfairly lowering full players' scores.
    # Even in Accumulative, the user requested strict invalidation ("don't change any score").
    has_late_joiner = any(getattr(p, 'joined_mid_round', False) for p in players)
    
    changes = {p.user_id: 0 for p in players}

    if has_late_joiner:
        late_players = [p.username for p in players if getattr(p, 'joined_mid_round', False)]
        print(f"[Rating] Late joiner detected in room: {late_players}. Voiding rating updates for ALL players (Cnt: {len(players)}) to ensure fairness.")
        return changes

    # 2. Check if a Guest is present (User Rule: "My rating should still not go up nor down playing with a guest NO MATTER WHAT")
    # We check both the is_guest flag AND the username prefix for robustness.
    def is_player_guest(p):
        return getattr(p, 'is_guest', False) or (p.username and p.username.startswith('Guest_'))

    has_guest = any(is_player_guest(p) for p in players)
    if has_guest:
        # Log the specific reason for voiding
        guests = [p.username for p in players if is_player_guest(p)]
        print(f"[Rating] Guest(s) {guests} detected in room. Round is UNRANKED for everyone.")
        return changes

    # 3. Identify active registered players (user_id > 0 OR is_ai, AND not a guest)
    # Bots are included to ensure ratings reflect performance relative to ALL competitors.
    active_players = [p for p in players if (p.user_id > 0 or p.is_ai) and not is_player_guest(p)]

    # Rating change requires at least two competing registered players WHO SCORED.
    # If it's just one player scoring vs ghosts/AFK, they should not gain or lose anything.
    active_scorers = [p for p in active_players if p.score > 0]
    
    if len(active_scorers) < 2:
        if len(active_scorers) == 1:
            print(f"[Rating] Only one active scorer ({active_scorers[0].username}) present. Round is unranked.")
        else:
            print(f"[Rating] No active scorers present. Round is unranked.")
        return changes
        
    # 3. Sum Totals
    score_sum = sum(p.score for p in active_players)
    rating_sum = sum(p.rating for p in active_players)
    
    if rating_sum == 0:
        return changes # Prevent division by zero
        
    print(f"[Rating] Proportional Calc: ScoreSum={score_sum}, RatingSum={rating_sum}, Players={len(active_players)}")
    
    # 3. Calculate Changes
    for p in active_players:
        # Expected score: logic from Java line 33: ((double)scoreSum/ratingSum) * theRatingInt
        expected_score = (score_sum / rating_sum) * p.rating
        
        # Sixteen Value: logic from Java line 36: expectedScore * ((double) 75/100)
        sixteen_score_value = expected_score * 0.75
        
        # Increment unit: logic from Java line 41: sixteenScoreValue / 16
        increment = sixteen_score_value / 16.0
        
        change = 0
        
        if increment <= 0.05:
            # If increment is effectively zero, check raw deviation?
            # Or enforce a minimum increment?
            # Java says: double sixteenScoreValue = expectedScore * ((double) 75/100);
            # increment = sixteenScoreValue / 16;
            # If expected score is 10, sixteen=7.5, increment=0.46.
            # Difference of 10 points: 10 > 0.46 * d. d=1 -> 0.46. d=16 -> 7.36.
            # 10 > 7.36. d=17 (max). Change = +-16.
            # Wait, `while d <= 16`. If logic loop exhausts: change = +-16.
            # If increment is small, d can reach 16 easily?
            # Wait, lines 799: `if d > 16: change = -16`.
            # So if increment is tiny, loop runs until d=17, forcing MAX change (-16).
            # This is OPPOSITE of "not changing".
            
            # BUT if increment is 0 (score_sum=0? No, score_sum > 0).
            pass

        if increment > 0:
            if p.score < expected_score:
                difference = expected_score - p.score
                # Logic loop lines 44-57
                # Basically: find d such that increment * d >= difference
                # Mathematical equivalent: ceil(difference / increment)
                # But let's follow the loop logic to be precise to the Java implementation
                # The loop breaks at the first match.
                d = 1
                while d <= 16:
                    if increment * d >= difference:
                        change = -d
                        break
                    d += 1
                if d > 16:
                    change = -16
                    
            elif p.score > expected_score:
                difference = p.score - expected_score
                # Logic loop lines 64-77
                f = 1
                while f <= 16:
                    if increment * f >= difference:
                        change = f
                        break
                    f += 1
                if f > 16:
                    change = 16
        
        changes[p.user_id] = change
        print(f"[Rating] Player {p.username} (R:{p.rating}, S:{p.score}): Exp={expected_score:.2f}, Diff={p.score-expected_score:.2f}, Change={change}")
        
    return changes

def calculate_word_score(word, bonus_word):
    """Calculate points for a word using standard Boggle scoring"""
    length = len(word)
    
    # Redirect to shared utility
    from scoring import calculate_word_score as shared_calc
    return shared_calc(word, bonus_word)
    
    # Bonus word gets extra points
    if word == bonus_word:
        bonus_points = length  # Extra points equal to word length
        return base_score + bonus_points
    
    return base_score


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
                # Routine 7-minute inactivity cleanup
                self.cleanup_rooms(timeout=420) 
                
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
    
    def cleanup_rooms(self, timeout=420, spec_timeout=1800):
        """Clean up empty or inactive rooms (defaults: 7m players, 30m spectators)"""
        rooms_to_delete = []
        
        # Iterate over a copy of keys to avoid modification issues
        for room_id, room in list(self.rooms.items()):
            try:
                # 1. Update Room State (transitions to intermission at midnight etc)
                room.check_and_update_state()

                # 2. Process Milestones (Spinner -> Search -> Next Round) autonomously
                if room.state == 'intermission' and (room.game_type in ['accumulative', 'solo_accumulative', 'fcfs', 'split'] or room.time_limit >= 7200):
                    milestone = room.get_intermission_milestone()
                    if milestone == 'spinner':
                        self.generate_spinner_params(room_id)
                    elif milestone == 'search':
                        self.start_board_search(room_id)
                    elif milestone == 'start':
                        print(f"[RoomManager/BG] Milestone 'start' hit for {room_id}. Transitioning.")
                        self.start_next_round(room_id)

                # 3. Check for inactive players
                room.check_inactivity(timeout, spec_timeout)
                
                # Close room if empty (except for 24h persistent rooms)
                is_empty = (len(room.players) == 0 and len(room.spectators) == 0)
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
        print(f"[RoomManager] start_round called for room {room_id}")
        room = self.get_room(room_id)
        if not room:
            print(f"[RoomManager] ERROR: Room {room_id} not found")
            return False
        
        # Prevent concurrent start_round calls
        if room.starting_round:
            print(f"[RoomManager] Round already starting, skipping...")
            return False
        
        room.starting_round = True
        
        # Save previous round data before generating new one
        # SAFEGUARD: If intermission already snapped it, keep it!
        # Check if it's genuinely populated, not just an empty list
        has_prev = hasattr(room, 'previous_all_words') and room.previous_all_words
        if not has_prev and room.all_words:
            # Filter by the CURRENT round's min length (stored in room.current_min_length)
            # This avoids using tomorrow's min length to filter today's words.
            min_len = getattr(room, 'current_min_length', 3)
            room.previous_all_words = [w for w in room.all_words if len(w) >= min_len]
            print(f"[RoomManager] Saved {len(room.previous_all_words)} words to history (Fallback/Round {room.current_round})")
        elif has_prev:
            print(f"[RoomManager] Using existing history snapshot (intermission) for Round {room.current_round}")
        else:
            print(f"[RoomManager] WARNING: No words to save to history (Round {room.current_round})")
        
        try:
            print(f"[RoomManager] Generating spinner parameters for {room.board_dimensions}")
            # Generate spinner parameters (Skip if solo and already configured)
            if not (room.is_solo and room.spinner_params):
                is_24h = room.time_limit >= 7200
                room.spinner_params = SpinnerSet.generate_params(room.board_dimensions, is_24h)
                print(f"[RoomManager] Solo/Practice: Using existing parameters")
            else:
                print(f"[RoomManager] Spinner params: {room.spinner_params}")
            
            # Get bonus word from dictionary
            if room.spinner_params['board_format'] in ['Checkerboard', 'Mania'] or 'Mania' in room.spinner_params['board_format']:
                print(f"[RoomManager] {room.spinner_params['board_format']} format selected - disabling bonus word")
                bonus_word = ''
            else:
                print(f"[RoomManager] Getting bonus word (length={room.spinner_params['bonus_word_length']}, dict={room.spinner_params['dictionary']})")
                bonus_word = self._get_bonus_word(room.spinner_params['bonus_word_length'], 
                                                  room.spinner_params['dictionary'])
            room.bonus_word = bonus_word
            print(f"[RoomManager] Bonus word selected: '{bonus_word}'")
            
            # Generate board
            print(f"[RoomManager] Starting board generation...")
            board, all_words, bonus_tile = self.board_generator.generate_board(
                room.board_dimensions,
                bonus_word,
                room.spinner_params['word_count_range'],
                room.spinner_params['dictionary'],
                room.spinner_params['board_format'],
                room.spinner_params['min_word_length'],  # Only count words meeting min length
                room.spinner_params['difficulty']
            )
            room.bonus_letter_cell = bonus_tile
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
            # room.state = 'active'  <-- MOVED TO END
            # room.round_start_time = time.time() <-- MOVED TO END
            
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
            # LOGIC FIX: We must process persistence BEFORE clearing data
            if room.time_limit >= 7200 and room.current_round > 1:
                print(f"[RoomManager] Daily Reset: Archiving & Wiping players in room {room_id}")
                
                # Ensure structure exists
                if not isinstance(room.previous_day_history, dict) or 'players' not in room.previous_day_history:
                    room.previous_day_history = {
                        'board': [row[:] for row in room.previous_board] if room.previous_board else None,
                        'all_words': room.previous_all_words,
                        'players': {}
                    }

                # Archive Player Data for 'Found/Missed' checks
                for p in room.players:
                    room.past_players[str(p.user_id)] = p
                    if str(p.user_id) not in room.previous_day_history['players']:
                        room.previous_day_history['players'][str(p.user_id)] = {
                            'username': p.username,
                            'found_words': [w['word'] for w in p.previous_submitted_words]
                        }

                # 3. Wipe current players
                room.players = []
                room.spectators = []
            
            # Clear FCFS global list
            room.fcfs_found_words.clear()
            
            # Only align to midnight if it's genuinely a long-duration room
            if room.time_limit >= 7200:
                now = datetime.datetime.now()
                now_dt = datetime.datetime.now()
                # Next midnight
                midnight = (now_dt + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                delta = (midnight - now_dt).total_seconds()
                room.custom_end_time = time.time() + delta
                print(f"[RoomManager] Daily room: aligned to next midnight ({midnight})")
                
            # Clear FCFS global list
            room.fcfs_found_words.clear()
            
            # DISABLED: Background complete solve is too slow and never finishes
            # Just use the fast solve results (30 words) for scoring and intermission
            print(f"[RoomManager] Using fast solve results for scoring")
            
            # Pre-calculate scores for the words we found
            room.solved_words_with_scores = {}
            for word in all_words:
                room.solved_words_with_scores[word] = calculate_word_score(word, bonus_word)
            room.complete_words = all_words
            room.solving_complete = True
            print(f"[RoomManager] Scored {len(all_words)} words")
            
            # FINAL STEP: Activate the room
            # We do this LAST to avoid race conditions where state='active' but custom_end_time isn't set yet.
            room.state = 'active'
            room.round_start_time = time.time()
            print(f"[RoomManager] Round {room.current_round} ACTIVATED at {room.round_start_time}. Custom End: {room.custom_end_time}")
            
            # Initialize AI Turns if any bots are present
            room.prepare_ai_turns()
            
            return True
            
        except Exception as e:
            import traceback
            print(f"[RoomManager] CRITICAL EXCEPTION in start_round: {e}")
            traceback.print_exc()
            return False
        finally:
            # Always reset the flag, even if board generation fails
            room.starting_round = False
    
    def generate_spinner_params(self, room_id):
        """Generate Spinner Set parameters for next round (called at 45s remaining)"""
        print(f"[RoomManager] Generating Spinner Set parameters at 45s remaining for room {room_id}")
        room = self.get_room(room_id)
        if not room:
            print(f"[RoomManager] ERROR: Room {room_id} not found")
            return False
        
        # Generate spinner parameters for next round
        if getattr(room, 'is_solo', False):
            print(f"[RoomManager] Solo match detected. Retaining existing spinner parameters.")
            room.spinner_params_generated = True
            return True
            
        is_24h = room.time_limit >= 7200
        room.spinner_params = SpinnerSet.generate_params(room.board_dimensions, is_24h)
        room.spinner_params_generated = True
        print(f"[RoomManager] Spinner params generated: {room.spinner_params}")
        return True
    
    def start_board_search(self, room_id):
        """Start board search using Spinner Set parameters (called at 15s remaining)"""
        print(f"[RoomManager] Starting board search at 15s remaining for room {room_id}")
        room = self.get_room(room_id)
        if not room:
            print(f"[RoomManager] ERROR: Room {room_id} not found")
            return False
        
        if not room.spinner_params_generated:
            print(f"[RoomManager] WARNING: Spinner params not generated yet, generating now")
            self.generate_spinner_params(room_id)
        
        fmt = room.spinner_params['board_format']
        if fmt == 'Checkerboard':
            print(f"[RoomManager] Checkerboard format selected - disabling bonus word")
            bonus_word = ''
        else:
            print(f"[RoomManager] Getting bonus word (length={room.spinner_params['bonus_word_length']}, dict={room.spinner_params['dictionary']})")
            bonus_word = self._get_bonus_word(room.spinner_params['bonus_word_length'], 
                                              room.spinner_params['dictionary'])
        room.next_round_bonus = bonus_word
        print(f"[RoomManager] Bonus word selected: '{bonus_word}'")
        
        # Start board generation in background thread
        def generate_in_background():
            print(f"[RoomManager] Background board generation started...")
            board, all_words = self.board_generator.generate_board(
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
                return
            
            room.next_round_board = board
            room.next_round_words = all_words
            print(f"[RoomManager] Board generated! Board: {board is not None}, Words: {len(all_words) if all_words else 0}")
        
        thread = threading.Thread(target=generate_in_background, daemon=True)
        thread.start()
        room.board_search_started = True
        return True
    
    
    def start_next_round(self, room_id):
        """Start next round with pre-generated board (called at 0s remaining)"""
        print(f"[RoomManager] start_next_round called for room {room_id}")
        
        try:
            room = self.get_room(room_id)
            if not room:
                print(f"[RoomManager] ERROR: Room {room_id} not found")
                return False
            
            print(f"[RoomManager] checking fallback for room {room_id} (game={room.game_type})")
            # Check if board is ready
            # Fallback if next_round_board is empty
            if not room.next_round_board or not room.next_round_words:
                print(f"[RoomManager] WARNING: Board not ready, falling back to start_round")
                return self.start_round(room_id)
            
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
                
                if not room.previous_board and room.board:
                    room.previous_board = [row[:] for row in room.board]

                # SNAPSHOT HISTORY BEFORE OVERWRITING BOARD
                if room.time_limit >= 7200 and not has_prev_hist:
                    print(f"[RoomManager] Daily Reset: Snapshotting history (fallback/start_next) before new round")
                    room.previous_day_history = {
                        'board': room.previous_board,
                        'all_words': room.previous_all_words,
                        'players': {}
                    }
                    for p in room.players:
                        room.previous_day_history['players'][str(p.user_id)] = {
                            'username': p.username,
                            'found_words': [w['word'] for w in p.submitted_words]
                        }
                    print(f"[RoomManager] Saved daily history for {len(room.players)} players")
            else:
                print(f"[RoomManager] SKIPPING snapshot - Using existing history from intermission")

            # Use pre-generated board and words
            room.board = room.next_round_board
            room.all_words = room.next_round_words
            room.bonus_word = room.next_round_bonus
            
            # Start the round
            room.current_round += 1
            room.state = 'active'
            room.current_min_length = room.spinner_params.get('min_word_length', 3)
            room.round_start_time = time.time()
            
            # SPLIT POINTS RANDOMIATION
            if room.game_type == 'split':
                import random
                random.shuffle(room.players)
                print(f"[RoomManager] Randomized player order for Split Points round")
                
            print(f"[RoomManager] Round {room.current_round} started with pre-generated board!")
            
            print(f"[RoomManager] Round {room.current_round} started with pre-generated board!")
            
            # 1. Calculate ELO changes based on FINAL scores of previous round
            # MOVED TO check_and_update_state (Start of Intermission)
            
            for player in room.players:
                # Reset Rating Change (so it doesn't persist forever)
                # But we want to show it during intermission...
                # Actually, strictly speaking, start_next_round is AFTER intermission.
                # So we should probably reset it here so it doesn't show during the *next* active round?
                # The user said "Change ratings as soon as round is over".
                # If we reset it here, it clears the green/red indicators for the new round.
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
                
            # PERSISTENCE: If this is a 24h room, clear the player list for the new day
            # (Users who enter will be added fresh)
            if room.time_limit >= 7200:
                print(f"[RoomManager] Daily Reset: Archiving player list for 24h room {room_id}")
                
                 # HISTORY ALREADY SNAPSHOTTED AT START OF FUNCTION
                
                # Move current players to past_players archive
                for p in room.players:
                    room.past_players[str(p.user_id)] = p
                
                room.players = []
                room.spectators = []
                
            # Clear FCFS global list
            if hasattr(room, 'fcfs_found_words'):
                room.fcfs_found_words.clear()
            
            # Pre-calculate scores
            room.solved_words_with_scores = {}
            for word in room.all_words:
                room.solved_words_with_scores[word] = calculate_word_score(word, room.bonus_word)
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
    
    def _get_bonus_word(self, length, dictionary):
        """Get a bonus word of specified length"""
        from word_validator import word_validator
        
        # Get all words of the specified length
        if dictionary == 'CSW':
            words = [w for w in word_validator.csw_words if len(w) == length]
        else:
            words = [w for w in word_validator.nwl_words if len(w) == length]
        
        # Return random word
        import random
        return random.choice(words) if words else 'A' * length
    
    
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
            
            # Statistics Rule: Only count rounds where at least 2 humans actually played (score > 0)
            # This prevents padding stats and leaderboard by playing in empty or non-competitive rooms.
            # We exclude AI bots from this count since they don't count as "another person".
            active_scorers = [p for p in room.players if p.score > 0 and not p.is_ai]
            
            if len(active_scorers) < 2:
                print(f"[RoomManager] SKIPPING history save for room {room.room_id} - only {len(active_scorers)} human player(s) reached a positive score.")
                conn.close()
                return

            playing_players = [p for p in room.players if p.user_id > 0]
            # No need for another check on playing_players as active_scorers implies human players exists.

            for p in playing_players:
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
                
                conn.execute('''
                    INSERT INTO round_history (user_id, room_id, game_type, round_number, board_json, words_json, total_score, round_start_time, round_duration, timestamp, user_rating, performance_ratio, best_word, best_word_score, board_dimensions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (p.user_id, room.room_id, room.game_type, room.current_round, board_json, json.dumps(words_data), p.score, room.round_start_time, room.time_limit, timestamp, p.rating, p.performance_efficiency, best_word_text, best_word_val, room.board_dimensions))
            
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
