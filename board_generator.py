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
        # FOR UNCONDITIONAL UNIQUENESS: Re-seed random from system randomness
        # This breaks any process-level determinism from forks/seeds
        import random
        random.seed()
        
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
            
            # IMPORTANT: Embed bonus word before solving - this "locks in" the bonus word
            bonus_cells = set()
            if bonus_word:
                path = self._embed_bonus_word(board, bonus_word)
                if not path:
                    print(f"[BoardGen] ✗ Failed to embed bonus word, retrying...")
                    continue  # Try again with new board
                bonus_cells = set(path)
                print(f"[BoardGen] ✓ Bonus word '{bonus_word}' embedded successfully")
            else:
                print(f"[BoardGen] - Skipping bonus word embedding (bonus word not provided)")
            
            # After bonus word is locked in, apply special board formats
            if board_format.endswith(' Mania'):
                mania_letter = board_format.split(' ')[0]
                self._apply_mania_to_board(board, mania_letter, exclude_cells=bonus_cells)
            elif board_format == 'Checkerboard':
                # Re-apply strict checkerboard if bonus word skewed it
                self._apply_checkerboard_strict(board, exclude_cells=bonus_cells)
            elif board_format == 'Either/Or':
                # Apply Either/Or to one tile
                self._apply_either_or(board, exclude_cells=bonus_cells)
            elif board_format == 'Bonus Letter':
                # Select a bonus letter tile
                board_bonus_tile = self._apply_bonus_letter(board)
            elif board_format == 'Valued Letters':
                # Just a flag for scoring, nothing to change on board letters
                pass
            
            # Solve board to find ALL valid dictionary words (min length 2 for validation feedback)
            # We use 2 here so that if a player types a 2-letter word, we know it's on the board
            # and can say "too short" instead of "INVALID".
            all_words = self._solve_board(board, dictionary, word_count_range, 2)
            
            # For board acceptance (the min_words/max_words check), only count words meeting the ACTUAL min_word_length
            scorable_words = [w for w in all_words if len(w) >= min_word_length]
            word_count = len(scorable_words)
            
            # Validate word count based on scorable words
            if self._validate_word_count(word_count, min_words, max_words):
                # SPECIAL CHECK: Either/Or Ambiguity
                if board_format == 'Either/Or':
                    if self._check_either_or_ambiguity(scorable_words, board):
                        print(f"[BoardGen] ✗ Rejected: Either/Or Ambiguity detected")
                        continue

                print(f"[BoardGen] ✓ Board valid: {word_count} scorable words (of {len(all_words)} total)")
                # Return potential bonus tile for "Bonus Letter" format
                if board_format == 'Bonus Letter':
                    return board, all_words, board_bonus_tile
                return board, all_words, None
            else:
                print(f"[BoardGen] ✗ Rejected: {word_count} scorable words")
        
        # Max attempts reached - use last board as fallback
        print(f"[BoardGen] ⚠ Max attempts reached: {word_count} words")
        return board, all_words
    
    def _parse_word_count_range(self, word_count_range):
        """Parse word count range (tuple, list, or string) into (min, max) tuple"""
        # Handle tuple/list format from spinner_set or explicit pass: (30, 60) or [30, 60]
        if isinstance(word_count_range, (tuple, list)):
            return tuple(word_count_range)
        
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
        """Create checkerboard pattern (consonants/vowels) with weighted letters"""
        vowel_indices = [self.letters.index(c) for c in VOWELS]
        consonant_indices = [self.letters.index(c) for c in CONSONANTS]
        
        vowel_weights = [weights[i] for i in vowel_indices]
        consonant_weights = [weights[i] for i in consonant_indices]
        
        board = [[None for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    # Consonant
                    board[r][c] = random.choices(CONSONANTS, weights=consonant_weights, k=1)[0]
                else:
                    # Vowel
                    board[r][c] = random.choices(VOWELS, weights=vowel_weights, k=1)[0]
        return board

    def _apply_checkerboard_strict(self, board, exclude_cells):
        """Enforce strict consonant/vowel alternating while preserving bonus word cells if possible"""
        rows, cols = len(board), len(board[0])
        for r in range(rows):
            for c in range(cols):
                if (r, c) in exclude_cells: continue
                
                is_vowel = board[r][c] in VOWELS
                target_vowel = (r + c) % 2 != 0
                
                if is_vowel != target_vowel:
                    # Replace with appropriate type
                    if target_vowel:
                        board[r][c] = random.choice(VOWELS)
                    else:
                        board[r][c] = random.choice(CONSONANTS)
    
    def _apply_mania_to_board(self, board, mania_letter, exclude_cells):
        """Fill a significant percentage (~31%) of AVAILABLE cells with the mania letter."""
        rows, cols = len(board), len(board[0])
        total_cells = rows * cols
        
        # User requested 5/16 ratio (~31%)
        target_count = max(4, round(total_cells * (5/16)))
        
        # Check how many we already have (from bonus word or initial board)
        current_count = sum(1 for r in range(rows) for c in range(cols) if board[r][c] == mania_letter)
        needed = target_count - current_count
        
        if needed <= 0:
            print(f"[BoardGen] Mania '{mania_letter}': Already have {current_count} letters, target was {target_count}")
            return
            
        # Get list of all positions NOT part of the bonus word
        all_positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in exclude_cells]
        random.shuffle(all_positions)
        
        filled = 0
        for r, c in all_positions:
            if filled >= needed:
                break
            board[r][c] = mania_letter
            filled += 1
        
        final_count = current_count + filled
        print(f"[BoardGen] Mania '{mania_letter}': placed {filled} more, final {final_count}/{total_cells} cells ({final_count/total_cells*100:.0f}%)")

    def _apply_either_or(self, board, exclude_cells):
        """Pick one tile to be an 'Either/Or' tile (e.g. 'L/T')"""
        rows, cols = len(board), len(board[0])
        # Pick a cell NOT in bonus word
        candidates = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in exclude_cells]
        if not candidates: return
        r, c = random.choice(candidates)
        
        # Pick two distinct letters. Use Uniques Frequency roughly (weights).
        l1 = board[r][c]
        l2 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        while l2 == l1:
            l2 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        board[r][c] = f"{l1}/{l2}"
        print(f"[BoardGen] Either/Or applied at ({r},{c}): {board[r][c]}")

    def _check_either_or_ambiguity(self, words, board):
        """
        Check if using the Either/Or tile results in any ambiguity 
        (e.g. both ETUDE and ELUDE exist using the same tile).
        """
        # Find the either/or tile
        rows, cols = len(board), len(board[0])
        eo_tile = None
        for r in range(rows):
            for c in range(cols):
                if '/' in board[r][c]:
                    eo_tile = (r, c, board[r][c].split('/'))
                    break
            if eo_tile: break
        
        if not eo_tile: return False
        
        r_eo, c_eo, options = eo_tile
        
        # This is a bit complex as we need physical paths to be 100% sure.
        # But for 'find one in which there is no confusion', we can check if 
        # both words exist in the 'words' list and could share the same path.
        # Minimalist approach: if any two words are identical except for the EO letter at the same index
        # AND could be formed on this board.
        
        # For simplicity, let's just check if both words in the pair (e.g. ELUDE/ETUDE) are found.
        # This is a safe "scrap it" condition.
        word_set = set(words)
        for w in words:
            if options[0] in w:
                # Try replacing options[0] with options[1]
                # Note: this doesn't guarantee they share the same TILE, 
                # but if they both exist on an EO board, it's risky.
                alt_w = w.replace(options[0], options[1])
                if alt_w in word_set:
                    return True
        return False

    def _apply_bonus_letter(self, board):
        """Randomly select ONE cell to be the 'Bonus Letter' for the round."""
        rows, cols = len(board), len(board[0])
        r, c = random.randint(0, rows-1), random.randint(0, cols-1)
        print(f"[BoardGen] Bonus Letter tile selected at ({r},{c})")
        return (r, c)
    
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
        for start_r, start_c in start_cells:
            path = backtrack([(start_r, start_c)])
            if path:
                # Embed the processed letters
                for i, (r, c) in enumerate(path):
                    board[r][c] = processed_word[i]
                return path
                
        return None
    
    def _solve_board(self, board, dictionary, word_count_range, min_word_length=3):
        """Find all valid words on the board that meet minimum length requirement"""
        rows, cols = len(board), len(board[0])
        found_words = set()
        
        # Calculate search limits based on board size
        board_size = rows * cols
        
        # Exhaustive search - find ALL words (max 25 letters to cover 16+ supplementary list)
        max_word_length = 25
        
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
                        cell_content = board[nr][nc]
                        letters_to_try = []
                        if '/' in cell_content:
                            # Either/Or tile
                            letters_to_try = cell_content.split('/')
                        else:
                            letters_to_try = [cell_content]

                        for cell_letter in letters_to_try:
                            # Branch 1: Treat as regular letter
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
                # Either/Or at start
                cell_content = board[r][c]
                letters_to_try = cell_content.split('/') if '/' in cell_content else [cell_content]
                
                for start_letter in letters_to_try:
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

    def is_word_on_board(self, word, board):
        """Check if a specific word exists on the board (ignoring dictionary)"""
        rows, cols = len(board), len(board[0])
        word = word.upper()
        
        def dfs_find(r, c, remaining_word, visited):
            if not remaining_word:
                return True
            
            # Check all 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited):
                        cell_letter = board[nr][nc]
                        
                        # Branch 1: Regular match
                        if cell_letter == remaining_word[0]:
                            if dfs_find(nr, nc, remaining_word[1:], visited | {(nr, nc)}):
                                return True
                                
                        # Branch 2: Q -> QU match
                        if cell_letter == 'Q' and remaining_word.startswith('QU'):
                            if dfs_find(nr, nc, remaining_word[2:], visited | {(nr, nc)}):
                                return True
            return False

        # Start from every cell
        for r in range(rows):
            for c in range(cols):
                start_letter = board[r][c]
                # Branch 1: Regular
                if start_letter == word[0]:
                    if dfs_find(r, c, word[1:], {(r, c)}):
                        return True
                # Branch 2: Q -> QU
                if start_letter == 'Q' and word.startswith('QU'):
                    if dfs_find(r, c, word[2:], {(r, c)}):
                        return True
        return False


# Test
if __name__ == '__main__':
    gen = BoardGenerator()
    board, words = gen.generate_board('4x4', 'BACKWARD', (50, 150), 'NWL', 'Normal', 3, 'Normal')
    if board:
        print("Board generated!")
        for row in board:
            print(' '.join(row))
        print(f"\\nFound {len(words)} words")
        print(f"Bonus word: BACKWARD")
    else:
        print("Failed to generate board")
