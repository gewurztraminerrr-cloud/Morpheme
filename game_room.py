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
    submitted_words: List[str] = field(default_factory=list)
    score: int = 0
    previous_round_score: int = 0
    rating_change: int = 0
    found_bonus_word: bool = False

@dataclass
class GameRoom:
    room_id: str
    game_type: str  # 'accumulative', 'fcfs', 'split'
    time_limit: int  # seconds per round
    board_dimensions: str  # '4x4', '4x6', etc.
    
    # Game state
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
    
    def add_player(self, user_id, username, rating):
        """Add player to room"""
        player = Player(user_id, username, rating)
        self.players.append(player)
        self.players.sort(key=lambda p: p.rating, reverse=True)
    
    def remove_player(self, user_id):
        """Remove player from room"""
        self.players = [p for p in self.players if p.user_id != user_id]
    
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
        player = self.get_player(user_id)
        if not player:
            return False, "Player not in room"
        
        word = word.upper()
        
        # Check if word is valid
        if word not in self.all_words:
            return False, "Word not on board"
        
        # Check minimum length
        min_len = self.spinner_params.get('min_word_length', 3)
        if len(word) < min_len:
            return False, f"Word must be at least {min_len} letters"
        
        # Check if already submitted
        if word in player.submitted_words:
            return False, "Word already submitted"
        
        # Add word
        player.submitted_words.append(word)
        
        # Check if this is the bonus word
        if word == self.bonus_word:
            player.found_bonus_word = True
            print(f"[GameRoom] Player {player.username} found the BONUS WORD: {word}!")
        
        # Update player score immediately
        player.score = sum(self.solved_words_with_scores.get(w, 0) for w in player.submitted_words)
        
        return True, "Word accepted"
    
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
        elif time_remaining <= 45 and not self.spinner_params_generated:
            return 'spinner'
        
        return None
    
    def check_and_update_state(self):
        """Check timers and update game state accordingly"""
        # Check if active round has expired
        if self.state == 'active' and self.time_remaining == 0:
            self.state = 'intermission'
            self.intermission_start_time = time.time()
            # Reset timing flags for next intermission
            self.spinner_params_generated = False
            self.board_search_started = False
            return True
        
        # Check if intermission has expired (for Accumulative games)
        if self.state == 'intermission' and self.time_remaining == 0:
            if self.game_type == 'accumulative':
                # Signal that new round should start
                # This will be handled by RoomManager
                return True
        
        return False

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
                del self.rooms[room_id]
    
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
            print(f"[RoomManager] Getting bonus word (length={room.spinner_params['bonus_word_length']}, dict={room.spinner_params['dictionary']})")
            bonus_word = self._get_bonus_word(room.spinner_params['bonus_word_length'], 
                                              room.spinner_params['dictionary'])
            room.bonus_word = bonus_word
            print(f"[RoomManager] Bonus word selected: {bonus_word}")
            
            # Generate board
            print(f"[RoomManager] Starting board generation...")
            board, all_words = self.board_generator.generate_board(
                room.board_dimensions,
                bonus_word,
                room.spinner_params['word_count_range'],
                room.spinner_params['dictionary'],
                room.spinner_params['board_format'],
                room.spinner_params['min_word_length']  # Only count words meeting min length
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
            print(f"[RoomManager] Round {room.current_round} started - timer active!")
            
            # Clear previous words and scores
            for player in room.players:
                # Calculate rating change (current score - previous round score)
                player.rating_change = player.score - player.previous_round_score
                # Store current score for next round's comparison
                player.previous_round_score = player.score
                # Clear for new round
                player.submitted_words = []
                player.score = 0
                player.found_bonus_word = False
            
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
        
        # Get bonus word from dictionary
        print(f"[RoomManager] Getting bonus word (length={room.spinner_params['bonus_word_length']}, dict={room.spinner_params['dictionary']})")
        bonus_word = self._get_bonus_word(room.spinner_params['bonus_word_length'], 
                                          room.spinner_params['dictionary'])
        room.next_round_bonus = bonus_word
        print(f"[RoomManager] Bonus word selected: {bonus_word}")
        
        # Start board generation in background thread
        def generate_in_background():
            print(f"[RoomManager] Background board generation started...")
            board, all_words = self.board_generator.generate_board(
                room.board_dimensions,
                bonus_word,
                room.spinner_params['word_count_range'],
                room.spinner_params['dictionary'],
                room.spinner_params['board_format'],
                room.spinner_params['min_word_length']
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
        print(f"[RoomManager] Starting next round with pre-generated board for room {room_id}")
        room = self.get_room(room_id)
        if not room:
            print(f"[RoomManager] ERROR: Room {room_id} not found")
            return False
        
        # Check if board is ready
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
        print(f"[RoomManager] Round {room.current_round} started with pre-generated board!")
        
        # Clear previous words and scores
        for player in room.players:
            # Calculate rating change (current score - previous round score)
            player.rating_change = player.score - player.previous_round_score
            # Store current score for next round's comparison
            player.previous_round_score = player.score
            # Clear for new round
            player.submitted_words = []
            player.score = 0
            player.found_bonus_word = False
        
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
