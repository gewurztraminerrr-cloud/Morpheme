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
        Returns: (board, all_words) or None if unable to generate
        """
        import time
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
            f.write(f"[board_generator.py] generate_board ENTERED for {dimensions} at {time.time()}\n")
        # FOR UNCONDITIONAL UNIQUENESS: Re-seed random from system randomness
        # This breaks any process-level determinism from forks/seeds
        import random
        random.seed()
        
        print(f"[BoardGen] generate_board called: {dimensions}, bonus={bonus_word}, range={word_count_range}, format={board_format}, dict={dictionary}")
        
        
        # Parse word count requirements
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
            f.write(f"[board_generator.py] generate_board: Parsing word count range {word_count_range} at {time.time()}\n")
        min_words, max_words = self._parse_word_count_range(word_count_range)
        print(f"[BoardGen] Target word count: {min_words}-{max_words if max_words != float('inf') else '∞'}")
        
        # REMOVED: Cache lookup that overrode user format preference
        # We now strictly respect the board_format passed in arguments
        
        # 0. Handle "Mania" without a prefix (e.g. from user dropdown selection)
        if board_format == 'Mania':
            mania_letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            board_format = f"{mania_letter} Mania"

        # Try to generate valid board (max 15 attempts)
        max_attempts = 15 

        # 0.1 Handle 500+ mode (Iterative Optimization)
        if min_words >= 500:
            print(f"[BoardGen] Entering 500+ Mode (Iterative Optimization)")
            rows, cols = map(int, dimensions.split('x'))
            board = self._create_2000plus_board(rows, cols, dictionary)
            all_words = self._solve_board(board, dictionary, (min_words, max_words), min_word_length)
            return board, all_words, None

        for attempt in range(1, max_attempts + 1):
            with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                f.write(f"[board_generator.py] generate_board: Attempt {attempt} for {dimensions} at {time.time()}\n")
            print(f"[BoardGen] Attempt {attempt}/{max_attempts}")
            
            rows, cols = map(int, dimensions.split('x'))
            weights = self._get_weights(difficulty)
            
            # 1. Create base board
            if board_format == 'Checkerboard':
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
                path = self._embed_bonus_word(board, actual_bonus_word)
                if not path:
                    print(f"[BoardGen] ✗ Failed to embed bonus word, retrying...")
                    continue
                bonus_cells_set = set(path)
                print(f"[BoardGen] ✓ Bonus word '{actual_bonus_word}' embedded successfully")
            
            # Now pick bonus_cell for 'Bonus Letter' format (AFTER bonus word path is known)
            if board_format == 'Bonus Letter':
                # Allow overlap with bonus word (User request: Bonus Word may randomly use Bonus Letter)
                selectable_cells = [(r, c) for r in range(rows) for c in range(cols)]
                if selectable_cells:
                    bonus_cell = random.choice(selectable_cells)
                else:
                    # Fallback: pick any cell if the bonus word filled the entire board somehow
                    bonus_cell = (random.randint(0, rows-1), random.randint(0, cols-1))
                print(f"[BoardGen] ✓ Bonus Letter cell selected: {bonus_cell} (letter: {board[bonus_cell[0]][bonus_cell[1]]})")
            
            # Now set bonus_cell for 'Either/Or' format, creating the tile AFTER bonus word
            if board_format == 'Either/Or':
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
                print(f"[BoardGen] ✓ Either/Or cell identified: {bonus_cell} (letters: {board[r][c]})")
            
            # 4. Apply extra effects
            if board_format.endswith(' Mania'):
                mania_letter = board_format.split(' ')[0]
                self._apply_mania_to_board(board, mania_letter, exclude_cells=bonus_cells_set)
            
            # 5. Solve and Validate
            with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                f.write(f"[board_generator.py] generate_board: Solving board for attempt {attempt} at {time.time()}\n")
            all_words = self._solve_board(board, dictionary, word_count_range, min_word_length)
            with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
                f.write(f"[board_generator.py] generate_board: Solve complete for attempt {attempt} at {time.time()} ({len(all_words)} words)\n")
            if board_format == 'Either/Or':
                if self._has_either_or_ambiguity(board, dictionary):
                    print(f"[BoardGen] ✗ Either/Or ambiguity detected, retrying...")
                    continue
            
            scorable_words = [w for w in all_words if len(w) >= min_word_length]
            word_count = len(scorable_words)
            
            if self._validate_word_count(word_count, min_words, max_words):
                print(f"[BoardGen] ✓ Board valid: {word_count} scorable words")
                # If Bonus Letter, we return it as part of a metadata dictionary or something?
                # For now let's return (board, all_words, bonus_cell)
                return board, all_words, bonus_cell
                
        print(f"[BoardGen] ⚠ Max attempts reached: {word_count} words")
        return board, all_words, bonus_cell
    
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
    
    def _create_2000plus_board(self, rows, cols, dictionary):
        """
        Iterative Optimization (IO)
        1. Start with a random board using custom IO Base weights.
        2. Scan every position. For each, test A-Z and pick the best letter.
        """
        weights = LETTER_FREQ_IO_BASE
        board = self._create_normal_board(rows, cols, weights)
        
        print(f"[BoardGen] Initializing IO Optimization for {rows}x{cols} board")
        
        for r in range(rows):
            for c in range(cols):
                best_char = board[r][c]
                max_words = 0
                
                # Test each letter in the alphabet
                for char in self.letters:
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
    
    def _apply_mania_to_board(self, board, mania_letter, exclude_cells):
        """Fill approx 31% of cells with the mania letter (5/16 ratio)."""
        rows, cols = len(board), len(board[0])
        total_cells = rows * cols
        
        # Target ratio: 5/16 (31.25%)
        target_ratio = 5.0 / 16.0
        target_count = max(3, round(total_cells * target_ratio))
        
        current_count = sum(1 for r in range(rows) for c in range(cols) if board[r][c] == mania_letter)
        needed = target_count - current_count
        
        if needed <= 0: return
            
        all_positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in exclude_cells and not '/' in board[r][c]]
        random.shuffle(all_positions)
        
        filled = 0
        for r, c in all_positions:
            if filled >= needed: break
            board[r][c] = mania_letter
            filled += 1
    
    def _embed_bonus_word(self, board, bonus_word):
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
        import time
        with open('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log', 'a') as f:
            f.write(f"[board_generator.py] _embed_bonus_word: Attempting to embed '{bonus_word}' at {time.time()}\n")
        
        for start_r, start_c in start_cells:
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

    def is_word_on_board(self, word, board):
        """Check if a specific word exists on the board (ignoring dictionary)"""
        rows, cols = len(board), len(board[0])
        word = word.upper()
        
        def dfs_find(r, c, index, visited):
            if index >= len(word):
                return True
            
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
                                if dfs_find(nr, nc, index + match_len, visited | {(nr, nc)}):
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
                    
                    if match_len > 0:
                        if dfs_find(r, c, match_len, {(r, c)}):
                            return True
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
                    
                    if match_len > 0:
                        if dfs_find(r, c, match_len, {(r, c)}, False):
                            return True
        return False
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
