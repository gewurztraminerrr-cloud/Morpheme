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

# IO Base weights (Sum = 10000)
LETTER_FREQ_IO_BASE = [800, 230, 360, 410, 1180, 150, 300, 240, 750, 20, 140, 560, 
                       280, 580, 610, 290, 20, 730, 940, 570, 370, 100, 120, 30, 180, 40]

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
        Returns: (board, all_words, bonus_cell)
        """
        import time
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
            f.write(f"[board_generator.py] generate_board ENTERED for {dimensions} at {time.time()}\n")
            
        # Initialize defaults to prevent NameError in return paths
        board = None
        all_words = []
        bonus_cell = None
        word_count = 0
        
        # FOR UNCONDITIONAL UNIQUENESS: Re-seed random from system randomness
        # This breaks any process-level determinism from forks/seeds
        import random
        random.seed()
        
        # For unconditional Checkerboard pattern accuracy (User Request: No bonus words on Checkerboard)
        if 'checkerboard' in str(board_format).lower():
            bonus_word = ""
            actual_bonus_word = ""
            print(f"[BoardGen] FORCE CLEAR bonus word for Checkerboard format")
            
        print(f"[BoardGen] generate_board called: {dimensions}, bonus={bonus_word}, range={word_count_range}, format={board_format}, dict={dictionary}")
        
        
        # 0. Handle 3x3x3 Cube Generation
        if dimensions == '3x3x3':
            min_words, max_words = self._parse_word_count_range(word_count_range)
            print(f"[BoardGen] Generating 3x3x3 Cube Board (Iterative Search: {min_words}-{max_words})...")
            
            for attempt in range(1, 21): # Up to 20 attempts for Cube
                board = self._create_cube_board(difficulty)
                if bonus_word:
                    self._embed_bonus_word_cube(board, bonus_word)
                
                # Enforce vowel minimum (33%)
                self._enforce_vowel_minimum(board, self._get_weights(difficulty))
                
                all_words = self._solve_cube_board(board, dictionary, min_word_length)
                word_count = len(all_words)
                
                if min_words <= word_count <= max_words:
                    print(f"[BoardGen] ✓ Cube Board valid on attempt {attempt} (Words: {word_count})")
                    return board, all_words, None, board_format
                
                # If we are failing, maybe try adjusting weights?
                # Optimization: if too few words, keep trying.
                if attempt == 20:
                    print(f"[BoardGen] WARNING: Cube generation hit max attempts, returning best effort (Words: {word_count})")
                    return board, all_words, None, board_format

        # Parse word count requirements
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
            f.write(f"[board_generator.py] generate_board: Parsing word count range {word_count_range} at {time.time()}\n")
        min_words, max_words = self._parse_word_count_range(word_count_range)
        print(f"[BoardGen] Target word count: {min_words}-{max_words if max_words != float('inf') else '∞'}")
        
        # REMOVED: Cache lookup that overrode user format preference
        # We now strictly respect the board_format passed in arguments
        
        # 0. Handle "Mania" without a prefix (e.g. from user dropdown selection)
        if board_format.strip() == 'Mania':
            # User Request: 30% vowels, 70% consonants for Mania formats
            import random
            if random.random() < 0.30:
                mania_letter = random.choice('AEIOU')
            else:
                mania_letter = random.choice('BCDFGHJKLMNPQRSTVWXYZ')
            board_format = f"{mania_letter} Mania"
            # Update fmt_clean/fmt_lower for subsequent steps
            fmt_clean = board_format.strip()
            fmt_lower = fmt_clean.lower()
            print(f"[BoardGen] Mania: Picked letter '{mania_letter}', new format: '{board_format}'")

        # Try to generate valid board (max 15 attempts)
        max_attempts = 15 

        # 0.1 Handle 500+ mode (Iterative Optimization)
        if min_words >= 500:
            print(f"[BoardGen] Entering 500+ Mode (Iterative Optimization with Bonus Word)")
            fmt_clean = board_format.strip()
            fmt_lower = fmt_clean.lower()
            is_checkerboard_fmt = 'checkerboard' in fmt_lower
            
            rows, cols = map(int, dimensions.split('x'))
            if is_checkerboard_fmt:
                board = self._create_checkerboard(rows, cols, self._get_weights(difficulty))
                # Now optimize it while FOLLOWING the pattern
                board = self._create_2000plus_board(rows, cols, dictionary, is_checkerboard=True)
                # NO BONUS WORD EMBEDDING for Checkerboard
                bonus_word = None
            else:
                board = self._create_2000plus_board(rows, cols, dictionary)
            
            # Embed bonus word into the dense board
            bonus_cells_set = set()
            if bonus_word and not is_checkerboard_fmt:
                path = self._embed_bonus_word(board, bonus_word, is_checkerboard=is_checkerboard_fmt)
                if path:
                    bonus_cells_set = set(path)
            
            # 4.2 Enforce vowel minimum (30-33%) - Dense Mode (Skip for Checkerboard)
            self._enforce_vowel_minimum(board, self._get_weights(difficulty), is_checkerboard=is_checkerboard_fmt)
                
            # 4.3 Final Checkerboard Safeguard (Clean up any remaining mismatches)
            if is_checkerboard_fmt:
                self._verify_checkerboard_safeguard(board, self._get_weights(difficulty), bonus_cells_set)
                
            all_words = self._solve_board(board, dictionary, (min_words, max_words), min_word_length)
            return board, all_words, None, board_format

        for attempt in range(1, max_attempts + 1):
            with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                f.write(f"[board_generator.py] generate_board: Attempt {attempt} for {dimensions} at {time.time()}\n")
            print(f"[BoardGen] Attempt {attempt}/{max_attempts}")
            
            rows, cols = map(int, dimensions.split('x'))
            weights = self._get_weights(difficulty)
            
            # 1. Create base board
            fmt_clean = board_format.strip()
            fmt_lower = fmt_clean.lower()
            is_checkerboard_fmt = 'checkerboard' in fmt_lower
            
            # User Request: NO BONUS WORDS on Checkerboard boards to ensure perfect alternation
            if is_checkerboard_fmt:
                actual_bonus_word = None
                bonus_word = None
                
            if is_checkerboard_fmt:
                board = self._create_checkerboard(rows, cols, weights)
            else:
                board = self._create_normal_board(rows, cols, weights)
            
            # 2. Add extra markers for formats that need them
            bonus_cell = None
            # Note: For 'Bonus Letter', bonus_cell is selected AFTER embedding the bonus word
            # to guarantee no overlap. See below after step 3.
            
            # 3. Embed bonus word (Overlay first)
            actual_bonus_word = bonus_word if bonus_word else ""
            
            bonus_cells_set = set()
            if actual_bonus_word:
                path = self._embed_bonus_word(board, actual_bonus_word, is_checkerboard=is_checkerboard_fmt)
                if not path:
                    print(f"[BoardGen] ✗ Failed to embed bonus word, retrying...")
                    continue
                bonus_cells_set = set(path)
                print(f"[BoardGen] ✓ Bonus word '{actual_bonus_word}' embedded successfully")
            
            # Now pick bonus_cell for 'Bonus Letter' format (AFTER bonus word path is known)
            if 'bonus letter' in fmt_lower:
                # Allow overlap with bonus word (User request: Bonus Word may randomly use Bonus Letter)
                selectable_cells = [(r, c) for r in range(rows) for c in range(cols)]
                if selectable_cells:
                    bonus_cell = random.choice(selectable_cells)
                else:
                    # Fallback: pick any cell if the bonus word filled the entire board somehow
                    bonus_cell = (random.randint(0, rows-1), random.randint(0, cols-1))
                print(f"[BoardGen] * Bonus Letter cell selected: {bonus_cell}")
            
            # Now set bonus_cell for 'Either/Or' format, creating the tile AFTER bonus word
            if 'either/or' in fmt_lower or 'either' in fmt_lower:
                # Allow overlap with bonus word
                selectable_cells = [(r, c) for r in range(rows) for c in range(cols)]
                if selectable_cells:
                    bonus_cell = random.choice(selectable_cells)
                else:
                    bonus_cell = (random.randint(0, rows-1), random.randint(0, cols-1))
                
                # Make it an Either/Or tile
                r, c = bonus_cell
                orig = board[r][c]
                others = [l for l in self.letters if l != orig]
                other_weights = [weights[self.letters.index(l)] for l in others]
                other = random.choices(others, weights=other_weights, k=1)[0]
                pair = sorted([orig, other])
                board[r][c] = f"{pair[0]}/{pair[1]}"
                print(f"[BoardGen] * Either/Or cell identified: {bonus_cell}")
            
            # 4. Apply extra effects
            # 4.1 Enforce vowel minimum (30-33%) - All boards (Skip logic inside method for Checkerboard)
            self._enforce_vowel_minimum(board, weights, is_checkerboard=is_checkerboard_fmt)

            # APPLY MANIA AFTER VOWEL BALANCING so it doesn't get overwritten
            if 'mania' in fmt_lower:
                parts = board_format.split(' ')
                if len(parts) >= 2 and len(parts[0]) == 1:
                    mania_letter = parts[0].upper()
                else:
                    # Fallback or initialization error - already handled at top but be safe
                    import random
                    if random.random() < 0.30: mania_letter = random.choice('AEIOU')
                    else: mania_letter = random.choice('BCDFGHJKLMNPQRSTVWXYZ')
                    board_format = f"{mania_letter} Mania"
                    
                self._apply_mania_to_board(board, mania_letter, exclude_cells=bonus_cells_set, is_checkerboard=is_checkerboard_fmt)
            
            # 5. Solve and Validate
            with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                f.write(f"[board_generator.py] generate_board: Solving board for attempt {attempt} at {time.time()}\n")
            all_words = self._solve_board(board, dictionary, word_count_range, min_word_length)
            with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                f.write(f"[board_generator.py] generate_board: Solve complete for attempt {attempt} at {time.time()} ({len(all_words)} words)\n")
            if 'either/or' in fmt_lower or 'either' in fmt_lower:
                if self._has_either_or_ambiguity(board, dictionary):
                    print(f"[BoardGen] ✗ Either/Or ambiguity detected, retrying...")
                    continue
            
            scorable_words = [w for w in all_words if len(w) >= min_word_length]
            word_count = len(scorable_words)
            
            if self._validate_word_count(word_count, min_words, max_words):
                # 6. Final Checkerboard Safeguard
                if is_checkerboard_fmt:
                    self._verify_checkerboard_safeguard(board, weights, bonus_cells_set)
                
                print(f"[BoardGen] ✓ Board valid: {word_count} scorable words")
                return board, all_words, bonus_cell, board_format
                
        print(f"[BoardGen] ⚠ Max attempts reached: {word_count} words")
        return board, all_words, bonus_cell, board_format
    
    def _parse_word_count_range(self, word_count_range):
        """Parse word count range (tuple or string) into (min, max) tuple"""
        # Handle tuple format from spinner_set: (30, 60)
        if isinstance(word_count_range, tuple):
            return word_count_range
        
        # Handle string format: "50-100", "100-200", "200+", "500+"
        if word_count_range == '50-100':
            return (50, 100)
        elif word_count_range == '100-200':
            return (100, 200)
        elif word_count_range == '200+':
            return (200, 500)
        elif word_count_range == '500+':
            return (500, 99999)
        elif word_count_range in ['1500+', '2000+']:
            return (500, 99999) # Backward compatibility
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
        """Create board with weighted random letters, avoiding redundant U next to Q"""
        board = [[None for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                # Check neighbors for a 'Q'
                has_q_neighbor = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if board[nr][nc] == 'Q':
                                has_q_neighbor = True
                                break
                    if has_q_neighbor: break
                
                if has_q_neighbor:
                    # Filter out 'U' (index 20) from weights
                    safe_weights = list(weights)
                    safe_weights[20] = 0
                    board[r][c] = random.choices(self.letters, weights=safe_weights, k=1)[0]
                else:
                    board[r][c] = random.choices(self.letters, weights=weights, k=1)[0]
        return board
    
    def _create_checkerboard(self, rows, cols, weights):
        """Create checkerboard pattern (consonants/vowels) with weighted letters.
        To ensure it alternates 'diagonally', we use row % 2."""
        vowel_indices = [self.letters.index(c) for c in VOWELS]
        consonant_indices = [self.letters.index(c) for c in CONSONANTS]
        
        vowel_weights = [weights[i] for i in vowel_indices]
        consonant_weights = [weights[i] for i in consonant_indices]
        
        board = [[None for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                # Alternate types: Row 0, Col 0: C, Col 1: V...
                # Using (r + c) % 2 ensures diagonal neighbors are same type, and orthogonal are different.
                if (r + c) % 2 == 0:
                    board[r][c] = random.choices(CONSONANTS, weights=consonant_weights, k=1)[0]
                else:
                    board[r][c] = random.choices(VOWELS, weights=vowel_weights, k=1)[0]
        return board
    
    def _create_2000plus_board(self, rows, cols, dictionary, is_checkerboard=False):
        """
        Iterative Optimization (IO)
        1. Start with a random board using custom IO Base weights.
        2. Scan every position. For each, test A-Z and pick the best letter.
        """
        weights = LETTER_FREQ_IO_BASE
        if is_checkerboard:
            board = self._create_checkerboard(rows, cols, weights)
        else:
            board = self._create_normal_board(rows, cols, weights)
        
        print(f"[BoardGen] Initializing IO Optimization for {rows}x{cols} board (Checkerboard: {is_checkerboard})")
        
        for r in range(rows):
            for c in range(cols):
                best_char = board[r][c]
                max_words = 0
                
                # Test each letter in the alphabet
                # If checkerboard, ONLY test letters that match the required type
                required_is_vowel = ((r + c) % 2 != 0)
                
                for char in self.letters:
                    if is_checkerboard:
                        if self._is_vowel(char) != required_is_vowel:
                            continue
                            
                    board[r][c] = char
                    # Quick solve for this configuration
                    words = self._solve_board(board, dictionary, (0, 99999), 3)
                    if len(words) > max_words:
                        max_words = len(words)
                        best_char = char
                
                board[r][c] = best_char
                print(f"[BoardGen] Optimized ({r},{c}) -> {best_char} (Words: {max_words})")
        
        return board

    def _create_either_or_board(self, rows, cols, weights):
        """Create a board where some tiles contain two letters (e.g. L/T)."""
        board = self._create_normal_board(rows, cols, weights)
        
        # Determine number of Either/Or tiles (User Request: Exactly one per board)
        count = 1
        
        # Pick positions
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(cells)
        
        for i in range(count):
            r, c = cells[i]
            orig = board[r][c]
            # Pick a second letter. 
            others = [l for l in self.letters if l != orig]
            other_weights = [weights[self.letters.index(l)] for l in others]
            other = random.choices(others, weights=other_weights, k=1)[0]
            
            # Store as "L/T"
            pair = sorted([orig, other])
            board[r][c] = f"{pair[0]}/{pair[1]}"
            
        return board
    
    def _apply_mania_to_board(self, board, mania_letter, exclude_cells, is_checkerboard=False):
        """Fill approx 31% of cells with the mania letter (5/16 ratio)."""
        if not mania_letter or len(mania_letter) != 1:
             print(f"[BoardGen] Mania: INVALID letter '{mania_letter}', skipping abundance")
             return
             
        rows, cols = len(board), len(board[0])
        total_cells = rows * cols
        
        # Determine mania type
        is_mania_vowel = self._is_vowel(mania_letter)
        
        # Target ratio: 5/16 (31.25%)
        target_ratio = 5.0 / 16.0
        target_count = max(3, round(total_cells * target_ratio))
        
        current_count = sum(1 for r in range(rows) for c in range(cols) if board[r][c] == mania_letter)
        needed = target_count - current_count
        
        if needed <= 0: return
            
        all_positions = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) in exclude_cells: continue
                if '/' in str(board[r][c]): continue
                
                # If checkerboard, only allow positions that match the mania letter's type
                if is_checkerboard:
                    is_cell_vowel_expected = ((r + c) % 2 != 0)
                    if is_mania_vowel != is_cell_vowel_expected:
                        continue
                        
                all_positions.append((r, c))

        random.shuffle(all_positions)
        
        filled = 0
        for r, c in all_positions:
            if filled >= needed: break
            board[r][c] = mania_letter
            filled += 1
    
    def _embed_bonus_word(self, board, bonus_word, is_checkerboard=False):
        """Embed bonus word using backtracking to find a valid path.
        Returns the path (list of cells) if successful, else None."""
        rows, cols = len(board), len(board[0])
        
        # Pre-process word to treat 'QU' as a single unit
        processed_word = []
        i = 0
        while i < len(bonus_word):
            if i < len(bonus_word) - 1 and bonus_word[i:i+2].upper() == 'QU':
                processed_word.append('Q')
                i += 2
            else:
                processed_word.append(bonus_word[i].upper())
                i += 1
        
        word_len = len(processed_word)
        
        # If checkerboard, NO BONUS WORDS allowed per user request
        if is_checkerboard:
            with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                f.write(f"[BoardGen] Strictly skipping bonus word embedding for Checkerboard.\n")
            return None
        
        # Pre-calculate C/V status for each letter in word
        word_vowel_map = [self._is_vowel(letter) for letter in processed_word]
        
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
            
            idx = len(current_path)
            r, c = current_path[-1]
            visited = set(current_path)
            
            for nr, nc in get_valid_neighbors(r, c, visited):
                # If checkerboard, the next cell (nr, nc) must match the type of processed_word[idx]
                if is_checkerboard:
                    is_expected_vowel = ((nr + nc) % 2 != 0)
                    if word_vowel_map[idx] != is_expected_vowel:
                        continue
                        
                result = backtrack(current_path + [(nr, nc)])
                if result:
                    return result
            return None

        # Try to find a path from any random starting cell
        # Filter starts based on checkerboard if needed
        possible_starts = []
        for r in range(rows):
            for c in range(cols):
                if is_checkerboard:
                    is_expected_vowel = ((r + c) % 2 != 0)
                    if word_vowel_map[0] == is_expected_vowel:
                        possible_starts.append((r, c))
                else:
                    possible_starts.append((r, c))
                    
        random.shuffle(possible_starts)
        
        import time
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
            f.write(f"[board_generator.py] _embed_bonus_word: Attempting to embed '{bonus_word}' at {time.time()}\n")
        
        for start_r, start_c in possible_starts:
            path = backtrack([(start_r, start_c)])
            if path:
                # Embed the processed letters
                for i, (r, c) in enumerate(path):
                    board[r][c] = processed_word[i]
                with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                    f.write(f"[board_generator.py] _embed_bonus_word: SUCCESS at {time.time()}\n")
                return path
        
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
            f.write(f"[board_generator.py] _embed_bonus_word: FAILED at {time.time()}\n")
        return None
    
    def _has_either_or_ambiguity(self, board, dictionary):
        """Check if any path in the board passing through the E/O tile could represent two different valid words."""
        rows, cols = len(board), len(board[0])
        eo_pos = None
        for r in range(rows):
            for c in range(cols):
                if '/' in str(board[r][c]):
                    eo_pos = (r, c)
                    break
        if not eo_pos: return False
        
        def dfs_check(r, c, visited, word_so_far):
            # word_so_far is a list of lists of possible letters at each step
            # e.g. [['E'], ['L', 'T'], ['U'], ['D'], ['E']]
            
            # Convert to possible words
            from itertools import product
            possible_words = [''.join(p) for p in product(*word_so_far)]
            
            # Optimization: If no tile so far has multiple letters, ambiguity is impossible
            has_multi = any(len(l) > 1 for l in word_so_far)
            
            if has_multi:
                valid_words = [w for w in possible_words if word_validator.is_valid_word(w, dictionary)]
                if len(valid_words) > 1:
                    # Ambiguity detected!
                    return True
            
            # Pruning: if NO possible word is a valid prefix, stop
            if not any(word_validator.has_valid_prefix(w, dictionary) for w in possible_words):
                return False
                
            # Geographic Pruning: If we can't reach eo_pos within the remaining steps, stop.
            er, ec = eo_pos
            dist = max(abs(r - er), abs(c - ec))
            remaining_steps = 8 - len(word_so_far)
            if dist > remaining_steps:
                return False
            
            # Continue
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited):
                        cell = board[nr][nc]
                        letters = cell.split('/') if '/' in cell else [cell]
                        if dfs_check(nr, nc, visited | {(nr, nc)}, word_so_far + [letters]):
                            return True
            return False

        # Since we only have one E/O tile, start DFS from eo_pos to find paths containing it
        # Actually, paths could START at eo_pos, END at eo_pos, or PASS THROUGH it.
        # Starting from every cell is fine as long as we prune those that haven't hit eo_pos yet in a smart way.
        # But even better: Since ambiguity ONLY exists if eo_pos is in the path, we can just start from eo_pos.
        # Wait, that only finds paths STARTING there.
        # But if there is a path A-B-C/H-D that is an ambiguity (ABCD, ABHD), 
        # then the path C/H-D-E... or B-A... would also be checked?
        # Actually, let's just start from every cell but return False if word_so_far length is 8 and still no eo_pos hit.
        
        for r in range(rows):
            for c in range(cols):
                cell = board[r][c]
                letters = cell.split('/') if '/' in cell else [cell]
                if dfs_check(r, c, {(r, c)}, [letters]):
                    return True
        return False

    def _solve_board(self, board, dictionary, word_count_range, min_word_length=3):
        """Find all valid words on the board. Handles Either/Or tiles."""
        rows, cols = len(board), len(board[0])
        found_words = set()
        max_word_length = 25
        
        def dfs(r, c, visited, word):
            # word is base string. If cell is Either/Or, we branch.
            cell = board[r][c]
            letters_to_try = cell.split('/') if '/' in cell else [cell]
            
            for char in letters_to_try:
                # 1. Regular
                w1 = word + char
                if word_validator.is_valid_word(w1, dictionary) and len(w1) >= min_word_length:
                    found_words.add(w1)
                
                if len(w1) < max_word_length and word_validator.has_valid_prefix(w1, dictionary):
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if dr == 0 and dc == 0: continue
                            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                                dfs(nr, nc, visited | {(nr, nc)}, w1)
                
                # 2. QU logic
                if char == 'Q':
                    w2 = word + 'QU'
                    if word_validator.is_valid_word(w2, dictionary) and len(w2) >= min_word_length:
                        found_words.add(w2)
                    if len(w2) < max_word_length and word_validator.has_valid_prefix(w2, dictionary):
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = r + dr, c + dc
                                if dr == 0 and dc == 0: continue
                                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                                    dfs(nr, nc, visited | {(nr, nc)}, w2)

        for r in range(rows):
            for c in range(cols):
                dfs(r, c, {(r, c)}, "")
        
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
                        
                        # Handle Either/Or and Q/QU branching
                        cell_val = str(board[nr][nc])
                        letters = cell_val.split('/') if '/' in cell_val else [cell_val]
                        
                        for char in letters:
                            # Branch 1: Treat as regular letter
                            dfs(nr, nc, path + [(nr, nc)], visited, word + char)
                            
                            # Branch 2: Specific QU logic
                            if char == 'Q':
                                dfs(nr, nc, path + [(nr, nc)], visited, word + 'QU')
                        
                        visited.remove((nr, nc))
        
        # Start from every cell - no early termination
        for r in range(rows):
            for c in range(cols):
                visited = {(r, c)}
                cell_val = str(board[r][c])
                letters = cell_val.split('/') if '/' in cell_val else [cell_val]
                
                for char in letters:
                    # Branch 1
                    dfs(r, c, [(r, c)], visited, char)
                    
                    # Branch 2
                    if char == 'Q':
                         dfs(r, c, [(r, c)], visited, 'QU')
        
        print(f"[BoardGen] Complete solver finished: found {len(found_words)} total words")
        return sorted(list(found_words))
    
    def is_word_on_board(self, word, board):
        """Check if a word exists on the board (2D or 3D Surface)"""
        if not board: return False
        is_3d = (len(board) == 6 and isinstance(board[0], list) and isinstance(board[0][0], list))
        word = word.upper()
        
        def dfs_find(f, r, c, index, visited):
            if index >= len(word): return True
            
            # Use appropriate neighbors based on dimension
            neighbors = []
            if not is_3d:
                rows, cols = len(board), len(board[0])
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols: neighbors.append((-1, nr, nc))
            else:
                neighbors = self._get_cube_neighbors(f, r, c)
            
            for nf, nr, nc in neighbors:
                if (nf, nr, nc) in visited: continue
                
                cell_val = str(board[nf][nr][nc] if is_3d else board[nr][nc]).upper()
                letters = cell_val.split('/') if '/' in cell_val else [cell_val]
                
                for char in letters:
                    match_length = 0
                    if char == 'Q' and word.startswith('QU', index): match_length = 2
                    elif word.startswith(char, index): match_length = len(char)
                    
                    if match_length > 0:
                        if index + match_length >= len(word): return True
                        if dfs_find(nf, nr, nc, index + match_length, visited | {(nf, nr, nc)}):
                            return True
            return False

        # Start from every cell
        if not is_3d:
            rows, cols = len(board), len(board[0])
            for r in range(rows):
                for c in range(cols):
                    cell_val = str(board[r][c]).upper()
                    # Initial check
                    for char in (cell_val.split('/') if '/' in cell_val else [cell_val]):
                        match_l = 0
                        if char == 'Q' and word.startswith('QU', 0): match_l = 2
                        elif word.startswith(char, 0): match_l = len(char)
                        if match_l > 0:
                            if match_l >= len(word): return True
                            if dfs_find(-1, r, c, match_l, {(-1, r, c)}): return True
        else:
            for f in range(6):
                for r in range(3):
                    for c in range(3):
                        cell_val = str(board[f][r][c]).upper()
                        if cell_val == 'Q' and word.startswith('QU', 0):
                            if 2 >= len(word): return True
                            if dfs_find(f, r, c, 2, {(f, r, c)}): return True
                        elif word.startswith(cell_val, 0):
                            if len(cell_val) >= len(word): return True
                            if dfs_find(f, r, c, len(cell_val), {(f, r, c)}): return True
        return False

    def can_word_hit_bonus(self, word, board, bonus_cell):
        """Check if a word can be formed on board such that its path contains bonus_cell"""
        if not bonus_cell: return False
        rows, cols = len(board), len(board[0])
        word = word.upper()
        target_r, target_c = tuple(bonus_cell)
        
        def dfs_find(r, c, index, visited, hit_target):
            # If current cell is target, mark as hit
            if r == target_r and c == target_c:
                hit_target = True
            
            if index >= len(word):
                return hit_target
            
            # Check all 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited):
                        cell_val = str(board[nr][nc]).upper()
                        letters = cell_val.split('/') if '/' in cell_val else [cell_val]
                        
                        for char in letters:
                            match_len = 0
                            if char == 'Q':
                                if word.startswith('QU', index): match_len = 2
                                elif word[index] == 'Q': match_len = 1
                            elif word[index] == char:
                                match_len = 1
                            
                            if match_len > 0:
                                if dfs_find(nr, nc, index + match_len, visited | {(nr, nc)}, hit_target):
                                    return True
            return False

        # Start from every cell
        for r in range(rows):
            for c in range(cols):
                cell_val = str(board[r][c]).upper()
                letters = cell_val.split('/') if '/' in cell_val else [cell_val]
                for char in letters:
                    match_len = 0
                    if char == 'Q':
                        if word.startswith('QU'): match_len = 2
                        elif word.startswith('Q'): match_len = 1
                    elif word.startswith(char):
                        match_len = 1
        return False
    def _create_cube_board(self, difficulty='Normal'):
        """Create a 3x3x3 cube board (6 faces, 3x3 each)"""
        weights = self._get_weights(difficulty)
        board = []
        for f in range(6):
            face = [[random.choices(self.letters, weights=weights, k=1)[0] for _ in range(3)] for _ in range(3)]
            board.append(face)
        return board

    def _get_cube_neighbors(self, f, r, c):
        """Standard 8-way adjacency for a 3x3x3 cube surface"""
        # (face, row, col)
        res = []
        # Intra-face
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < 3 and 0 <= nc < 3:
                    res.append((f, nr, nc))

        # Inter-face (Edges and Corners)
        # Face Layout (Standard Net):
        #      [4] (Top)
        #  [2] [0] [3] [1] (Left, Front, Right, Back)
        #      [5] (Bottom)
        
        # 0 (Front)
        if f == 0:
            if r == 0: # Top Edge
                res.extend([(4, 2, c), (4, 2, c-1), (4, 2, c+1)]) # Orthogonal + Corners
            if r == 2: # Bottom Edge
                res.extend([(5, 0, c), (5, 0, c-1), (5, 0, c+1)])
            if c == 0: # Left Edge
                res.extend([(2, r, 2), (2, r-1, 2), (2, r+1, 2)])
            if c == 2: # Right Edge
                res.extend([(3, r, 0), (3, r-1, 0), (3, r+1, 0)])

        # 1 (Back)
        elif f == 1:
            if r == 0: # Top Edge -> Top (4) Top
                res.extend([(4, 0, 2-c), (4, 0, 2-(c-1)), (4, 0, 2-(c+1))])
            if r == 2: # Bottom Edge -> Bottom (5) Bottom
                res.extend([(5, 2, 2-c), (5, 2, 2-(c-1)), (5, 2, 2-(c+1))])
            if c == 0: # Left Edge -> Right (3) Right
                res.extend([(3, r, 2), (3, r-1, 2), (3, r+1, 2)])
            if c == 2: # Right Edge -> Left (2) Left
                res.extend([(2, r, 0), (2, r-1, 0), (2, r+1, 0)])

        # 2 (Left)
        elif f == 2:
            if r == 0: # Top Edge -> Top (4) Left
                res.extend([(4, c, 0), (4, c-1, 0), (4, c+1, 0)])
            if r == 2: # Bottom Edge -> Bottom (5) Left
                res.extend([(5, 2-c, 0), (5, 2-(c-1), 0), (5, 2-(c+1), 0)])
            if c == 0: # Left Edge -> Back (1) Right
                res.extend([(1, r, 2), (1, r-1, 2), (1, r+1, 2)])
            if c == 2: # Right Edge -> Front (0) Left
                res.extend([(0, r, 0), (0, r-1, 0), (0, r+1, 0)])

        # 3 (Right)
        elif f == 3:
            if r == 0: # Top Edge -> Top (4) Right
                res.extend([(4, 2-c, 2), (4, 2-(c-1), 2), (4, 2-(c+1), 2)])
            if r == 2: # Bottom Edge -> Bottom (5) Right
                res.extend([(5, c, 2), (5, c-1, 2), (5, c+1, 2)])
            if c == 0: # Left Edge -> Front (0) Right
                res.extend([(0, r, 2), (0, r-1, 2), (0, r+1, 2)])
            if c == 2: # Right Edge -> Back (1) Left
                res.extend([(1, r, 0), (1, r-1, 0), (1, r+1, 0)])

        # 4 (Top)
        elif f == 4:
            if r == 0: # Top Edge -> Back (1) Top
                res.extend([(1, 0, 2-c), (1, 0, 2-(c-1)), (1, 0, 2-(c+1))])
            if r == 2: # Bottom Edge -> Front (0) Top
                res.extend([(0, 0, c), (0, 0, c-1), (0, 0, c+1)])
            if c == 0: # Left Edge -> Left (2) Top
                res.extend([(2, 0, r), (2, 0, r-1), (2, 0, r+1)])
            if c == 2: # Right Edge -> Right (3) Top
                res.extend([(3, 0, 2-r), (3, 0, 2-(r-1)), (3, 0, 2-(r+1))])

        # 5 (Bottom)
        elif f == 5:
            if r == 0: # Top Edge -> Front (0) Bottom
                res.extend([(0, 2, c), (0, 2, c-1), (0, 2, c+1)])
            if r == 2: # Bottom Edge -> Back (1) Bottom
                res.extend([(1, 2, 2-c), (1, 2, 2-(c-1)), (1, 2, 2-(c+1))])
            if c == 0: # Left Edge -> Left (2) Bottom
                res.extend([(2, 2, 2-r), (2, 2, 2-(r-1)), (2, 2, 2-(r+1))])
            if c == 2: # Right Edge -> Right (3) Bottom
                res.extend([(3, 2, r), (3, 2, r-1), (3, 2, r+1)])

        # Filter out invalid and duplicates
        clean = []
        seen = set()
        for nf, nr, nc in res:
            if 0 <= nf < 6 and 0 <= nr < 3 and 0 <= nc < 3 and (nf, nr, nc) not in seen and (nf, nr, nc) != (f, r, c):
                clean.append((nf, nr, nc))
                seen.add((nf, nr, nc))
        return clean

    def _enforce_vowel_minimum(self, board, weights, is_checkerboard=False):
        """Ensure 30%-33% of tiles are vowels (User Request: Strict range for all boards)
           FOR CHECKERBOARD: Always skip this balancer as it must stay at 50%."""
        if not board or is_checkerboard: return
        
        # Flatten board to get all cells
        flat_cells = []
        is_3d = (len(board) == 6 and isinstance(board[0], list) and isinstance(board[0][0], list))
        
        if is_3d:
            # 3x3x3 Cube Surface (54 total cells)
            for f in range(6):
                for r in range(3):
                    for c in range(3):
                        flat_cells.append((f, r, c))
        else:
            # Standard 2D Grid
            rows, cols = len(board), len(board[0])
            for r in range(rows):
                for c in range(cols):
                    flat_cells.append((r, c))
        
        total_cells = len(flat_cells)
        # Target: 30%-33%
        # 16 cells -> 5 (31.2%)
        # 25 cells -> 8 (32.0%)
        # 54 cells -> 17 (31.5%)
        target_vowels = (total_cells * 315 + 500) // 1000 
        
        v_indices = [self.letters.index(v) for v in VOWELS]
        v_w = [weights[v_idx] for v_idx in v_indices]
        c_indices = [self.letters.index(c) for c in CONSONANTS]
        c_w = [weights[c_idx] for c_idx in c_indices]
        
        # Count current vowels
        current_vowel_cells = []
        current_consonant_cells = []
        
        for pos in flat_cells:
            if is_3d:
                f, r, c = pos
                val = str(board[f][r][c])
            else:
                r, c = pos
                val = str(board[r][c])
            
            if self._is_vowel(val):
                current_vowel_cells.append(pos)
            else:
                current_consonant_cells.append(pos)
        
        current_count = len(current_vowel_cells)
        random.shuffle(current_vowel_cells)
        random.shuffle(current_consonant_cells)
        
        if current_count < target_vowels:
            # Need more vowels
            needed = target_vowels - current_count
            for i in range(min(needed, len(current_consonant_cells))):
                pos = current_consonant_cells[i]
                new_v = random.choices(list(VOWELS), weights=v_w, k=1)[0]
                if is_3d:
                    f, r, c = pos
                    board[f][r][c] = new_v
                else:
                    r, c = pos
                    board[r][c] = new_v
            print(f"[BoardGen] Enforced 30-33% vowels: Added {needed} vowels.")
            
        elif current_count > target_vowels:
            # Need fewer vowels (Too many can happen with weights)
            over = current_count - target_vowels
            for i in range(min(over, len(current_vowel_cells))):
                pos = current_vowel_cells[i]
                new_c = random.choices(list(CONSONANTS), weights=c_w, k=1)[0]
                if is_3d:
                    f, r, c = pos
                    board[f][r][c] = new_c
                else:
                    r, c = pos
                    board[r][c] = new_c
            print(f"[BoardGen] Enforced 30-33% vowels: Removed {over} vowels.")

    def _verify_checkerboard_safeguard(self, board, weights, bonus_cells_set):
        """Final check to ensure the board strictly alternates C/V in checkerboard mode."""
        if not board: return
        rows, cols = len(board), len(board[0])
        v_indices = [self.letters.index(v) for v in VOWELS]
        v_weights = [weights[v_idx] for v_idx in v_indices]
        c_indices = [self.letters.index(c) for c in CONSONANTS]
        c_weights = [weights[c_idx] for c_idx in c_indices]
        
        repaired = 0
        for r in range(rows):
            for c in range(cols):
                # We used to skip the bonus word path, but the USER requested strict layout 
                # for ALL spots on a 4x4 board, so we now repair EVERY spot.
                if '/' in str(board[r][c]): continue
                
                # Check expectation: (0,0) is C, (0,1) is V...
                is_expected_vowel = ((r + c) % 2 != 0)
                is_actual_vowel = self._is_vowel(board[r][c])
                
                if is_actual_vowel != is_expected_vowel:
                    # Swap it
                    if is_expected_vowel:
                        board[r][c] = random.choices(list(VOWELS), weights=v_weights, k=1)[0]
                    else:
                        board[r][c] = random.choices(list(CONSONANTS), weights=c_weights, k=1)[0]
                    repaired += 1
        if repaired > 0:
            print(f"[BoardGen] Checkerboard Safeguard: Forced {repaired} letters to maintain alternation pattern.")

    def _is_vowel(self, char):
        """Helper to check if a letter (or tile string) is a vowel"""
        if not char: return False
        # Handle Either/Or L/T - return True if either is a vowel
        letters = str(char).upper().split('/')
        for l in letters:
            if l in VOWELS:
                return True
        return False

    def _is_consonant(self, char):
        """Helper to check if a letter is a consonant"""
        if not char: return False
        letters = str(char).upper().split('/')
        for l in letters:
            if l in CONSONANTS:
                return True
        return False

    def _is_alternating_word(self, word_chars):
        """Check if a series of letters strictly alternates C/V"""
        if not word_chars: return True
        current_v = self._is_vowel(word_chars[0])
        for i in range(1, len(word_chars)):
            next_v = self._is_vowel(word_chars[i])
            if next_v == current_v:
                return False
            current_v = next_v
        return True

    def _solve_cube_board(self, board, dictionary, min_word_length=3):
        """Find words on a 3x3x3 cube surface using DFS"""
        found = set()
        max_len = 25
        
        def dfs(f, r, c, visited, word):
            char = board[f][r][c]
            # Branch 1: Normal
            w1 = word + char
            if word_validator.is_valid_word(w1, dictionary) and len(w1) >= min_word_length:
                found.add(w1)
            if len(w1) < max_len and word_validator.has_valid_prefix(w1, dictionary):
                for nf, nr, nc in self._get_cube_neighbors(f, r, c):
                    if (nf, nr, nc) not in visited:
                        dfs(nf, nr, nc, visited | {(nf, nr, nc)}, w1)
            # Branch 2: QU
            if char == 'Q':
                w2 = word + 'QU'
                if word_validator.is_valid_word(w2, dictionary) and len(w2) >= min_word_length:
                    found.add(w2)
                if len(w2) < max_len and word_validator.has_valid_prefix(w2, dictionary):
                    for nf, nr, nc in self._get_cube_neighbors(f, r, c):
                        if (nf, nr, nc) not in visited:
                            dfs(nf, nr, nc, visited | {(nf, nr, nc)}, w2)

        for f in range(6):
            for r in range(3):
                for c in range(3):
                    dfs(f, r, c, {(f, r, c)}, "")
        
        return sorted(list(found))

    def _embed_bonus_word_cube(self, board, bonus_word):
        """Backtracking embed on cube surface"""
        p_word = []
        i = 0
        while i < len(bonus_word):
            if i < len(bonus_word) - 1 and bonus_word[i:i+2].upper() == 'QU':
                p_word.append('Q'); i += 2
            else:
                p_word.append(bonus_word[i].upper()); i += 1
        
        cells = [(f, r, c) for f in range(6) for r in range(3) for c in range(3)]
        random.shuffle(cells)
        
        def backtrack(path):
            if len(path) == len(p_word): return path
            cf, cr, cc = path[-1]
            neighbors = self._get_cube_neighbors(cf, cr, cc)
            random.shuffle(neighbors)
            for nf, nr, nc in neighbors:
                if (nf, nr, nc) not in path:
                    res = backtrack(path + [(nf, nr, nc)])
                    if res: return res
            return None

        for sf, sr, sc in cells:
            path = backtrack([(sf, sr, sc)])
            if path:
                for idx, (f, r, c) in enumerate(path):
                    board[f][r][c] = p_word[idx]
                return path
        return None
if __name__ == '__main__':
    gen = BoardGenerator()
    board, words, bonus_cell = gen.generate_board('4x4', 'BACKWARD', (50, 150), 'NWL', 'Normal', 3, 'Normal')
    if board:
        print("Board generated!")
        for row in board:
            print(' '.join(row))
        print(f"\\nFound {len(words)} words")
        print(f"Bonus word: BACKWARD")
    else:
        print("Failed to generate board")
