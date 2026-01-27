"""
Game Room Management for Multiplayer Boggle
Handles room state, players, timers, and game logic
"""

import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict
from spinner_set import SpinnerSet
from board_generator import BoardGenerator

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
    found_bonus_word: bool = False
    last_active: float = field(default_factory=time.time)

@dataclass
class GameRoom:
    room_id: str
    game_type: str  # 'accumulative', 'fcfs', 'split'
    time_limit: int  # seconds per round
    board_dimensions: str  # '4x4', '4x6', etc.
    
    # Rating limits
    min_rating: int = 0
    max_rating: int = 9999
    
    # Spectators
    spectators: List[Player] = field(default_factory=list)
    max_players: int = 8
    

    
    # Game state
    creation_time: float = field(default_factory=time.time)
    state: str = 'waiting'  # 'waiting', 'active', 'intermission', 'finished'
    current_round: int = 0
    starting_round: bool = False  # Prevents concurrent round starts
    
    # Timer
    round_start_time: float = 0
    intermission_start_time: float = 0
    
    # Current board data
    board: List[List[str]] = field(default_factory=list)
    all_words: List[str] = field(default_factory=list)  # Fast initial word list
    complete_words: List[str] = field(default_factory=list)  # Complete word list from background solving
    solved_words_with_scores: Dict[str, int] = field(default_factory=dict)  # Pre-computed word scores
    bonus_word: str = ''
    solving_complete: bool = False  # Track if background solving is done
    
    # FCFS Mode specific
    fcfs_found_words: set = field(default_factory=set)
    
    # Spinner parameters
    spinner_params: Dict = field(default_factory=dict)
    
    # Next round pre-generation (for Accumulative timing)
    spinner_params_generated: bool = False  # Track if spinner set generated for next round
    board_search_started: bool = False      # Track if board search started
    next_round_board: List[List[str]] = field(default_factory=list)  # Store pre-generated board
    next_round_words: List[str] = field(default_factory=list)  # Store pre-generated word list
    next_round_bonus: str = ''  # Store bonus word for next round
    
    # Players
    players: List[Player] = field(default_factory=list)
    
    # Chat
    chat_messages: List[Dict] = field(default_factory=list)
    
    def add_chat_message(self, username, message):
        """Add chat message to room"""
        self.chat_messages.append({
            'username': username,
            'message': message,
            'time': time.time()
        })
        # Keep only last 50 messages
        if len(self.chat_messages) > 50:
            self.chat_messages.pop(0)
    
    def add_player(self, user_id, username, rating):
        """Add player to room"""
        # Ensure player is not already in the room (prevent duplicates)
        self.remove_player(user_id)
        
        # Check max players specific to room
        if len(self.players) >= self.max_players:
            return False # Room full
            
        player = Player(user_id, username, rating)
        self.players.append(player)
        self.players.sort(key=lambda p: p.rating, reverse=True)
        return True # Success

    def add_spectator(self, user_id, username, rating):
        """Add spectator to room"""
        # Ensure user is not already a player or spectator
        self.remove_player(user_id)
        
        player = Player(user_id, username, rating)
        self.spectators.append(player)
        return True
    
    def remove_player(self, user_id):
        """Remove player or spectator from room"""
        # Remove from players - Use string comparison to be safe against type mismatches
        initial_players = len(self.players)
        self.players = [p for p in self.players if str(p.user_id) != str(user_id)]
        if len(self.players) < initial_players:
            print(f"[GameRoom] Removed player {user_id} from room {self.room_id}")

        # Remove from spectators (just in case)
        initial_specs = len(self.spectators)
        self.spectators = [p for p in self.spectators if str(p.user_id) != str(user_id)]
        if len(self.spectators) < initial_specs:
            print(f"[GameRoom] Removed spectator {user_id} from room {self.room_id}")

    def update_player_activity(self, user_id):
        """Update last_active timestamp for a player or spectator"""
        player = self.get_player(user_id)
        if player:
            player.last_active = time.time()
            return
            
        # Check spectators too
        for p in self.spectators:
            if p.user_id == user_id:
                p.last_active = time.time()
                break

    def check_inactivity(self, timeout=60): # 1 minute default for aggressive cleanup
        """Remove players and spectators who haven't been active for 'timeout' seconds"""
        now = time.time()
        active_players = []
        removed = False
        
        for p in self.players:
            if now - p.last_active < timeout:
                active_players.append(p)
            else:
                print(f"[GameRoom] Removing inactive player {p.username} (last active {now - p.last_active:.1f}s ago)")
                removed = True
        
        if removed:
            self.players = active_players

        # Check spectators
        active_spectators = []
        for p in self.spectators:
            if now - p.last_active < timeout:
                active_spectators.append(p)
            else:
                print(f"[GameRoom] Removing inactive spectator {p.username}")
                removed = True
                
        if len(active_spectators) != len(self.spectators):
            self.spectators = active_spectators
            
        return removed
    
    def get_player(self, user_id):
        """Get player by ID"""
        for p in self.players:
            if p.user_id == user_id:
                return p
        return None
    
    @property
    def time_remaining(self):
        """Calculate time remaining in current state"""
        if self.state == 'active':
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
            # Handle invalid word tracking for Split Points (or general reference)
            player.invalid_words.append(word)
            return False, "Word not on board", 0, None
        
        # Use the matched word (which might be the QU variant) for scoring/display
        final_word = matched_word
        
        # Check minimum length (use the final word length, e.g., QUATE is 5, QATE is 4)
        min_len = self.spinner_params.get('min_word_length', 3)
        if len(final_word) < min_len:
            return False, f"Word must be at least {min_len} letters", 0, None
        
        # Check if already submitted (by this player)
        # Extract existing words from the list of dicts
        existing_words = {w['word'] for w in player.submitted_words}
        if final_word in existing_words:
            return False, "Word already submitted", 0, None
            
        # FCFS Mode: Check if word found by ANYONE
        if self.game_type == 'fcfs':
            if final_word in self.fcfs_found_words:
                return False, "Word already found by another player", 0, None
            self.fcfs_found_words.add(final_word)
        
        # Calculate score for this word
        points = sum(self.solved_words_with_scores.get(w, 0) for w in [final_word])
        
        # Add word as structured data
        player.submitted_words.append({
            'word': final_word,
            'time': time.time(),
            'points': points
        })
        
        # Check if this is the bonus word
        if final_word == self.bonus_word:
            player.found_bonus_word = True
            print(f"[GameRoom] Player {player.username} found the BONUS WORD: {final_word}!")
        
        # Update player score immediately
        player.score = sum(w['points'] for w in player.submitted_words)
        
        return True, "Word accepted", points, final_word
    
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
    
    def check_and_update_state(self):
        """Check timers and update game state accordingly"""
        # Check if active round has expired
        if self.state == 'active' and self.time_remaining == 0:
            self.state = 'intermission'
            self.intermission_start_time = time.time()
            
            # SPLIT POINTS LOGIC
            if self.game_type == 'split':
                self.calculate_split_scores()
            
            # Reset timing flags for next intermission
            self.spinner_params_generated = False
            self.board_search_started = False
            return True
        
        # Check if intermission has expired
        if self.state == 'intermission' and self.time_remaining == 0:
            if self.game_type in ['accumulative', 'fcfs', 'split']:
                # Signal that new round should start
                # This will be handled by RoomManager
                return True
        
        return False

    def calculate_split_scores(self):
        """
        Calculate scores for Split Points mode.
        - Unique word: Full points
        - Shared word: Points / Count (rounded down)
        """
        print(f"[GameRoom] Calculating Split Points for room {self.room_id}")
        
        # 1. Count occurrences of each word (VALID words only)
        word_counts = {}
        
        for p in self.players:
            for w_obj in p.submitted_words:
                w = w_obj['word']
                # Only count valid words (which they should be if in submitted_words)
                word_counts[w] = word_counts.get(w, 0) + 1
                
        # 2. Update scores for each player
        for p in self.players:
            new_total_score = 0
            for w_obj in p.submitted_words:
                w = w_obj['word']
                count = word_counts.get(w, 1)
                
                # Base points were calculated on submission
                base_points = calculate_word_score(w, self.bonus_word)
                
                # Split points
                final_points = base_points // count
                
                # Update word object with split metadata for frontend
                w_obj['split_points'] = final_points
                w_obj['shared_count'] = count
                w_obj['is_unique'] = (count == 1)
                w_obj['points'] = final_points # Update the main points field to the split value
                w_obj['base_points'] = base_points # Keep track of what it was worth
                
                new_total_score += final_points
            
            # Update player total score
            print(f"[GameRoom] Player {p.username}: Old Score={p.score}, New Split Score={new_total_score}")
            p.score = new_total_score
            
            # Update player total score
            print(f"[GameRoom] Player {p.username}: Old Score={p.score}, New Split Score={new_total_score}")
            p.score = new_total_score
            for w in p.submitted_words:
                 print(f"  Word: {w.get('word')} Unique: {w.get('is_unique')} Points: {w.get('points')}")
            
            # Also calculate invalid words points (0, but we might want to track count)


def calculate_pairwise_elo(players):
    """
    Calculate rating changes based on pairwise comparisons.
    For each pair of players (A, B):
        Calculate expected score for A vs B
        Calculate actual score (1=win, 0.5=draw, 0=loss)
        Delta = K * (Actual - Expected)
    
    Final Elo Change for A = Sum of Deltas vs all opponents
    """
    K = 32
    changes = {p.user_id: 0 for p in players}
    
    if len(players) < 2:
        return changes # No changes if solo
        
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pA = players[i]
            pB = players[j]
            
            # Expected score for A
            # Ra = pA.rating, Rb = pB.rating
            expA = 1 / (1 + 10 ** ((pB.rating - pA.rating) / 400))
            expB = 1 / (1 + 10 ** ((pA.rating - pB.rating) / 400))
            
            # Actual score based on points
            if pA.score > pB.score:
                actA = 1.0
                actB = 0.0
            elif pA.score < pB.score:
                actA = 0.0
                actB = 1.0
            else:
                actA = 0.5
                actB = 0.5
            
            # Calculate Delta
            deltaA = K * (actA - expA)
            deltaB = K * (actB - expB)
            
            changes[pA.user_id] += deltaA
            changes[pB.user_id] += deltaB
            
    return changes

def calculate_word_score(word, bonus_word):
    """Calculate points for a word using standard Boggle scoring"""
    length = len(word)
    
    # Base score by word length
    if length <= 2:
        base_score = 0
    elif length <= 4:
        base_score = 1
    elif length == 5:
        base_score = 2
    elif length == 6:
        base_score = 3
    elif length == 7:
        base_score = 5
    else:  # 8+ letters
        base_score = 11
    
    # Bonus word gets extra points
    if word == bonus_word:
        bonus_points = length  # Extra points equal to word length
        return base_score + bonus_points
    
    return base_score


class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, GameRoom] = {}
        self.lock = threading.Lock()
        self.board_generator = BoardGenerator()
    
    def create_room(self, room_id, game_type, time_limit, board_dimensions):
        """Create a new game room"""
        with self.lock:
            room = GameRoom(
                room_id=room_id,
                game_type=game_type,
                time_limit=time_limit,
                board_dimensions=board_dimensions
            )
            self.rooms[room_id] = room
            return room
    
    def get_room(self, room_id):
        """Get room by ID"""
        return self.rooms.get(room_id)
    
    def delete_room(self, room_id):
        """Delete room"""
        with self.lock:
            if room_id in self.rooms:
                print(f"[RoomManager] Deleting room {room_id} (requested)")
                del self.rooms[room_id]
            else:
                print(f"[RoomManager] delete_room called for {room_id} but not found")
    
    def cleanup_rooms(self, timeout=1200):
        """Clean up empty or inactive rooms"""
        rooms_to_delete = []
        
        # Iterate over a copy of keys to avoid modification issues
        for room_id, room in list(self.rooms.items()):
            # Check for inactive players
            room.check_inactivity(timeout)
            
            # If room is empty, mark for deletion
            if len(room.players) == 0:
                print(f"[RoomManager] Room {room_id} is empty (after cleanup), marking for deletion")
                rooms_to_delete.append(room_id)
        
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
        
        try:
            print(f"[RoomManager] Generating spinner parameters for {room.board_dimensions}")
            # Generate spinner parameters
            room.spinner_params = SpinnerSet.generate_params(room.board_dimensions)
            print(f"[RoomManager] Spinner params: {room.spinner_params}")
            
            # Get bonus word from dictionary
            if room.spinner_params['board_format'] == 'Checkerboard':
                print(f"[RoomManager] Checkerboard format selected - disabling bonus word")
                bonus_word = ''
            else:
                print(f"[RoomManager] Getting bonus word (length={room.spinner_params['bonus_word_length']}, dict={room.spinner_params['dictionary']})")
                bonus_word = self._get_bonus_word(room.spinner_params['bonus_word_length'], 
                                                  room.spinner_params['dictionary'])
            room.bonus_word = bonus_word
            print(f"[RoomManager] Bonus word selected: '{bonus_word}'")
            
            # Generate board
            print(f"[RoomManager] Starting board generation...")
            board, all_words = self.board_generator.generate_board(
                room.board_dimensions,
                bonus_word,
                room.spinner_params['word_count_range'],
                room.spinner_params['dictionary'],
                room.spinner_params['board_format'],
                room.spinner_params['min_word_length'],  # Only count words meeting min length
                room.spinner_params['difficulty']
            )
            print(f"[RoomManager] Board generation complete! Board: {board is not None}, Words: {len(all_words) if all_words else 0}")
            
            if board is None:
                print(f"[RoomManager] ERROR: Board generation failed!")
                return False
            
            room.board = board
            room.all_words = all_words
            
            # Start the round immediately with timer
            room.current_round += 1
            room.state = 'active'
            room.round_start_time = time.time()
            
            # SPLIT POINTS RANDOMIATION
            if room.game_type == 'split':
                import random
                random.shuffle(room.players)
                print(f"[RoomManager] Randomized player order for Split Points round")
            
            print(f"[RoomManager] Round {room.current_round} started - timer active!")
            
            # Clear previous words and scores
            
            # Calculate ELO only after first round
            elo_changes = {}
            if room.current_round > 1:
                elo_changes = calculate_pairwise_elo(room.players)

            for player in room.players:
                if room.current_round > 1:
                     change = int(elo_changes.get(player.user_id, 0))
                     player.rating += change
                     player.rating_change = change
                else:
                     player.rating_change = 0
                # Store current score for next round's comparison
                player.previous_round_score = player.score
                # Clear for new round
                player.submitted_words = []
                player.invalid_words = []
                player.score = 0
                player.score = 0
                player.found_bonus_word = False
                
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
        room.spinner_params = SpinnerSet.generate_params(room.board_dimensions)
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
        
        if room.spinner_params['board_format'] == 'Checkerboard':
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
            
            # Use pre-generated board and words
            room.board = room.next_round_board
            room.all_words = room.next_round_words
            room.bonus_word = room.next_round_bonus
            
            # Start the round
            room.current_round += 1
            room.state = 'active'
            room.round_start_time = time.time()
            
            # SPLIT POINTS RANDOMIATION
            if room.game_type == 'split':
                import random
                random.shuffle(room.players)
                print(f"[RoomManager] Randomized player order for Split Points round")
                
            print(f"[RoomManager] Round {room.current_round} started with pre-generated board!")
            
            # Clear previous words and scores
            # Clear previous words and scores
            
            # 1. Calculate ELO changes based on FINAL scores of previous round
            # Do this BEFORE resetting scores
            elo_changes = calculate_pairwise_elo(room.players)
            
            for player in room.players:
                # Update Rating
                change = int(elo_changes.get(player.user_id, 0))
                player.rating += change
                player.rating_change = change
                
                # Store current score for next round's comparison
                player.previous_round_score = player.score
                
                # Clear for new round
                player.submitted_words = []
                player.invalid_words = []
                player.score = 0
                player.found_bonus_word = False
                
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
    
    def start_complete_solving(self, room_id):
        """
        Mark solving as complete immediately - words already found during generation.
        """
        room = self.get_room(room_id)
        if not room:
            return
        
        print(f"[RoomManager] Words already found, marking as complete")
        room.solving_complete = True



# Global instance
room_manager = RoomManager()
