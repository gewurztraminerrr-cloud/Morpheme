"""
Board Generator for Morpheme Boggle Game
Generates boards with bonus word embedding and validation
"""

import random
from word_validator import word_validator

# Letter frequency (A-Z)
# Letter frequency (A-Z)
# Medium/Hard weights
LETTER_FREQ_MH = [343, 100, 157, 161, 455, 64, 106, 108, 326, 11, 64, 236,
                  131, 232, 266, 123, 8, 272, 283, 224, 168, 40, 49, 15, 92, 22]

# Default/Easy weights
LETTER_FREQ_DEFAULT = [190, 45, 99, 82, 278, 29, 69, 61, 222, 4, 23, 129,
                       71, 165, 163, 74, 4, 172, 237, 161, 81, 23, 19, 7, 40, 12]

VOWELS = 'AEIOU'
CONSONANTS = 'BCDFGHJKLMNPQRSTVWXYZ'

class BoardGenerator:
    # Class-level cache for optimal board generation method per parameter set
    method_cache = {}
    
    def __init__(self):
        # Store letters
        self.letters = [chr(65 + i) for i in range(26)]  # A-Z
        
    def _get_weights(self, difficulty):
        if difficulty in ['Medium', 'Hard']:
            return LETTER_FREQ_MH
        return LETTER_FREQ_DEFAULT
    
    def generate_board(self, dimensions, bonus_word, word_count_range, dictionary, board_format, min_word_length=3, difficulty='Normal'):
        """
        Generate a valid board that meets word count requirements.
        Uses cached optimal method or tests both formats on first use.
        Only counts words >= min_word_length.
        Returns: (board, all_words) or None if unable to generate
        """
        print(f"[BoardGen] generate_board called: {dimensions}, bonus={bonus_word}, range={word_count_range}, format={board_format}, dict={dictionary}")
        
        
        # Parse word count requirements
        min_words, max_words = self._parse_word_count_range(word_count_range)
        print(f"[BoardGen] Target word count: {min_words}-{max_words if max_words != float('inf') else '∞'}")
        
        # REMOVED: Cache lookup that overrode user format preference
        # We now strictly respect the board_format passed in arguments
        
        # Try to generate valid board (max 10 attempts)
        max_attempts = 10
        for attempt in range(1, max_attempts + 1):
            print(f"[BoardGen] Attempt {attempt}/{max_attempts}")
            
            rows, cols = map(int, dimensions.split('x'))
            
            # Get weights for this attempt
            weights = self._get_weights(difficulty)
            
            # Create board
            if board_format == 'Checkerboard':
                board = self._create_checkerboard(rows, cols, weights)
            else:
                board = self._create_normal_board(rows, cols, weights)
            
            # IMPORTANT: Embed bonus word before solving
            if bonus_word:
                if not self._embed_bonus_word(board, bonus_word):
                    print(f"[BoardGen] ✗ Failed to embed bonus word, retrying...")
                    continue  # Try again with new board
                print(f"[BoardGen] ✓ Bonus word '{bonus_word}' embedded successfully")
            else:
                 print(f"[BoardGen] - Skipping bonus word embedding (bonus word not provided)")
            
            # Solve board to find words (filtered by min length)
            all_words = self._solve_board(board, dictionary, word_count_range, min_word_length)
            word_count = len(all_words)
            
            # Validate word count
            if self._validate_word_count(word_count, min_words, max_words):
                print(f"[BoardGen] ✓ Board valid: {word_count} words")
                return board, all_words
            else:
                print(f"[BoardGen] ✗ Rejected: {word_count} words")
        
        # Max attempts reached - use last board as fallback
        print(f"[BoardGen] ⚠ Max attempts reached: {word_count} words")
        return board, all_words
    
    def _parse_word_count_range(self, word_count_range):
        """Parse word count range (tuple or string) into (min, max) tuple"""
        # Handle tuple format from spinner_set: (30, 60)
        if isinstance(word_count_range, tuple):
            return word_count_range
        
        # Handle string format: "50-100", "100-200", "200+"
        if word_count_range == '50-100':
            return (50, 100)
        elif word_count_range == '100-200':
            return (100, 200)
        elif word_count_range == '200+':
            return (200, float('inf'))
        else:
            # Default to no restrictions
            return (0, float('inf'))
    
    def _validate_word_count(self, word_count, min_words, max_words):
        """Check if word count falls within the required range"""
        return min_words <= word_count <= max_words
    
    def _test_board_formats(self, dimensions, bonus_word, word_count_range, dictionary, min_words, max_words, min_word_length=3):
        """Test both board formats and return the faster one that meets requirements"""
        import time
        
        rows, cols = map(int, dimensions.split('x'))
        results = {}
        
        # Test Checkerboard format
        print(f"[BoardGen] Testing Checkerboard format...")
        start = time.time()
        board_cb = self._create_checkerboard(rows, cols)
        words_cb = self._solve_board(board_cb, dictionary, word_count_range, min_word_length)
        time_cb = time.time() - start
        valid_cb = self._validate_word_count(len(words_cb), min_words, max_words)
        results['Checkerboard'] = (time_cb, len(words_cb), valid_cb)
        print(f"[BoardGen] Checkerboard: {time_cb:.2f}s, {len(words_cb)} words, {'VALID' if valid_cb else 'INVALID'}")
        
        # Test Normal format
        print(f"[BoardGen] Testing Normal format...")
        start = time.time()
        board_normal = self._create_normal_board(rows, cols)
        words_normal = self._solve_board(board_normal, dictionary, word_count_range, min_word_length)
        time_normal = time.time() - start
        valid_normal = self._validate_word_count(len(words_normal), min_words, max_words)
        results['Normal'] = (time_normal, len(words_normal), valid_normal)
        print(f"[BoardGen] Normal: {time_normal:.2f}s, {len(words_normal)} words, {'VALID' if valid_normal else 'INVALID'}")
        
        # Choose faster valid method, or just faster method if neither is valid
        if valid_cb and valid_normal:
            return 'Checkerboard' if time_cb < time_normal else 'Normal'
        elif valid_cb:
            return 'Checkerboard'
        elif valid_normal:
            return 'Normal'
        else:
            # Neither is valid, return faster one
            return 'Checkerboard' if time_cb < time_normal else 'Normal'
    
    def _create_normal_board(self, rows, cols, weights):
        """Create board with weighted random letters"""
        board = []
        for r in range(rows):
            row = []
            for c in range(cols):
                # Draw a random letter using frequency weights
                row.append(random.choices(self.letters, weights=weights, k=1)[0])
            board.append(row)
        return board
    
    def _create_checkerboard(self, rows, cols, weights):
        """Create checkerboard pattern (consonants/vowels) with weighted letters"""
        # separate weights for consonants and vowels
        # VOWELS = 'AEIOU'
        # CONSONANTS = 'BCDFGHJKLMNPQRSTVWXYZ'
        
        vowel_indices = [self.letters.index(c) for c in VOWELS]
        consonant_indices = [self.letters.index(c) for c in CONSONANTS]
        
        vowel_weights = [weights[i] for i in vowel_indices]
        consonant_weights = [weights[i] for i in consonant_indices]
        
        board = []
        for r in range(rows):
            row = []
            for c in range(cols):
                if (r + c) % 2 == 0:
                    # Consonant
                    row.append(random.choices(CONSONANTS, weights=consonant_weights, k=1)[0])
                else:
                    # Vowel
                    row.append(random.choices(VOWELS, weights=vowel_weights, k=1)[0])
            board.append(row)
        return board
    
    def _embed_bonus_word(self, board, bonus_word):
        """Embed bonus word using backtracking to find a valid path"""
        rows, cols = len(board), len(board[0])
        word_len = len(bonus_word)
        
        # Create list of all cells and shuffle for randomness
        start_cells = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(start_cells)
        
        def get_valid_neighbors(r, c, visited):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited):
                        neighbors.append((nr, nc))
            random.shuffle(neighbors) # Randomize direction
            return neighbors

        def backtrack(current_path):
            if len(current_path) == word_len:
                return current_path
            
            r, c = current_path[-1]
            visited = set(current_path)
            
            for nr, nc in get_valid_neighbors(r, c, visited):
                result = backtrack(current_path + [(nr, nc)])
                if result:
                    return result
            return None

        # Try to find a path from any random starting cell
        for start_r, start_c in start_cells:
            path = backtrack([(start_r, start_c)])
            if path:
                # Embed the word
                for i, (r, c) in enumerate(path):
                    board[r][c] = bonus_word[i]
                return True
                
        return False
    
    def _solve_board(self, board, dictionary, word_count_range, min_word_length=3):
        """Find all valid words on the board that meet minimum length requirement"""
        rows, cols = len(board), len(board[0])
        found_words = set()
        
        # Calculate search limits based on board size
        board_size = rows * cols
        
        # Exhaustive search - find ALL words (max 15 letters)
        max_word_length = 15
        
        # Higher targets to find ALL words (not just 30)
        if board_size <= 16:  # 4x4
            target_count = 300  # Find up to 300 words
        else:
            target_count = 500  # Find up to 500 words for larger boards
        
        print(f"[BoardGen] Solver config: max_len={max_word_length}, min_len={min_word_length}, target={target_count} words")
        
        def dfs(r, c, path, visited, word):
            # Add word if it's valid and meets minimum length requirement
            if len(word) >= min_word_length and word_validator.is_valid_word(word, dictionary):
                found_words.add(word)
                # NO early termination - find ALL words
            
            # Stop if word is getting too long
            if len(word) >= max_word_length:
                return
            
            # Try all 8 adjacent cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = r + dr, c + dc
                    
                    if (0 <= nr < rows and 0 <= nc < cols and 
                        (nr, nc) not in visited):
                        
                        visited.add((nr, nc))
                        
                        # Handle Q/QU branching
                        cell_letter = board[nr][nc]
                        
                        # Branch 1: Treat as regular letter (includes Q as just Q)
                        next_word_1 = word + cell_letter
                        if word_validator.has_valid_prefix(next_word_1, dictionary):
                            dfs(nr, nc, path + [(nr, nc)], visited, next_word_1)
                            
                        # Branch 2: Specific QU logic
                        if cell_letter == 'Q':
                            next_word_2 = word + 'QU'
                            if word_validator.has_valid_prefix(next_word_2, dictionary):
                                dfs(nr, nc, path + [(nr, nc)], visited, next_word_2)
                                
                        visited.remove((nr, nc))
        
        # Start from every cell - search exhaustively
        for r in range(rows):
            for c in range(cols):
                visited = {(r, c)}
                # Initial cell can also be Q or QU
                start_letter = board[r][c]
                
                # Branch 1: Regular
                dfs(r, c, [(r, c)], visited, start_letter)
                
                # Branch 2: Q -> QU
                if start_letter == 'Q':
                    dfs(r, c, [(r, c)], visited, 'QU')
        
        return sorted(list(found_words))
    
    def complete_solve_board(self, board, dictionary):
        """
        Exhaustively find ALL valid words on the board without limits.
        Used for background solving during intermission.
        """
        rows, cols = len(board), len(board[0])
        found_words = set()
        
        print(f"[BoardGen] Complete solver: searching exhaustively (max_len=10, no early termination)")
        
        def dfs(r, c, path, visited, word):
            # Add word if it's valid and long enough
            if len(word) >= 3 and word_validator.is_valid_word(word, dictionary):
                found_words.add(word)
            
            # Stop if word is getting too long (reduced to 10 for speed)
            if len(word) >= 10:
                return
            
            # Try all 8 adjacent cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = r + dr, c + dc
                    
                    if (0 <= nr < rows and 0 <= nc < cols and 
                        (nr, nc) not in visited):
                        
                        visited.add((nr, nc))
                        
                        # Handle Q/QU branching
                        cell_letter = board[nr][nc]
                        
                        # Branch 1: Treat as regular letter
                        dfs(nr, nc, path + [(nr, nc)], visited, word + cell_letter)
                        
                        # Branch 2: Specific QU logic
                        if cell_letter == 'Q':
                            dfs(nr, nc, path + [(nr, nc)], visited, word + 'QU')
                        
                        visited.remove((nr, nc))
        
        # Start from every cell - no early termination
        for r in range(rows):
            for c in range(cols):
                visited = {(r, c)}
                start_letter = board[r][c]
                
                # Branch 1
                dfs(r, c, [(r, c)], visited, start_letter)
                
                # Branch 2
                if start_letter == 'Q':
                     dfs(r, c, [(r, c)], visited, 'QU')
        
        print(f"[BoardGen] Complete solver finished: found {len(found_words)} total words")
        return sorted(list(found_words))


# Test
if __name__ == '__main__':
    gen = BoardGenerator()
    board, words = gen.generate_board('4x4', 'BACKWARD', (50, 150), 'NWL', 'Normal')
    if board:
        print("Board generated!")
        for row in board:
            print(' '.join(row))
        print(f"\\nFound {len(words)} words")
        print(f"Bonus word: BACKWARD")
    else:
        print("Failed to generate board")
