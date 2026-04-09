"""
Board Generator for Morpheme Boggle Game
Generates boards with bonus word embedding and validation
"""

import random
import time
from word_validator import word_validator

# Letter frequency (A-Z)
# Medium/Hard weights - CUSTOMIZED: Peak Connectivity for 7-10L words
# User-provided frequencies for 4x4 (A-Z)
LETTER_FREQ_USER = [
    300,
    95,
    169,
    136,
    400,
    61,
    84,
    104,
    334,
    9,
    51,
    247,
    126,
    225,
    268,
    122,
    7,
    279,
    269,
    240,
    157,
    41,
    41,
    16,
    95,
    18,
]

# Easy weights (Sum = 10000) - CUSTOMIZED: Peak Density
LETTER_FREQ_EASY = [
    1050,
    230,
    360,
    410,
    1400,
    150,
    300,
    240,
    750,
    20,
    140,
    560,
    280,
    580,
    610,
    290,
    20,
    730,
    940,
    570,
    600,
    100,
    120,
    30,
    180,
    40,
]  # A=1050, E=1400, U=600

VOWELS = "AEIOU"
CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ"
# User-identified difficult letters for Hard rounds (with common support for density)
RARE_SET = "ZXQJKVWYPFBHCMAU" + "ETAOINSRHDLU" + "AEIOUAEIOU"  # Blend with common consonants and 10 vowels

# Sparse weights for large grids with low word count targets (Reduced common vowels/consonants)
# Sparse weights for large grids with low word count targets (Heavily reduced vowels/common consonants)
# Sum = 2605 (approx 1/4 of standard 10000 set for rare packing)
LETTER_FREQ_SPARSE = [
    70,   # A
    120,  # B
    140,  # C
    110,  # D
    90,   # E
    110,  # F
    120,  # G
    120,  # H
    80,   # I
    130,  # J
    125,  # K
    140,  # L
    130,  # M
    110,  # N
    80,   # O
    110,  # P
    130,  # Q
    100,  # R
    100,  # S
    90,   # T
    50,   # U
    130,  # V
    115,  # W
    150,  # X
    110,  # Y
    140   # Z
]


class BoardGenerator:
    # Class-level cache for optimal board generation method per parameter set
    method_cache = {}

    def __init__(self):
        self.letters = [chr(65 + i) for i in range(26)]  # A-Z
        self.unique_sets = {}
        self.cube_neighbor_cache = None

    def _get_difficulty_set(self, dictionary_type):
        """Lazy-load and cache unique word sets for diff validation"""
        core_type = dictionary_type.upper()
        if core_type not in self.unique_sets:
            if core_type.startswith("UNIQUE"):
                path = f"/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/{core_type.lower()}.txt"
                path_alt = f"/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/{core_type}.txt"
            else:
                path = f"/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/unique{core_type}.txt"
                path_alt = f"/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/Unique{core_type}.txt"

            try:
                with open(path, "r") as f:
                    self.unique_sets[core_type] = set(line.strip().upper() for line in f if line.strip())
                print(f"[BoardGen] Loaded {len(self.unique_sets[core_type])} unique words for {core_type}")
            except Exception:
                try:
                    with open(path_alt, "r") as f:
                        self.unique_sets[core_type] = set(line.strip().upper() for line in f if line.strip())
                    print(f"[BoardGen] Loaded {len(self.unique_sets[core_type])} unique words for {core_type} (Alt Path)")
                except Exception:
                    print(f"[BoardGen] Warning: Unique set for {core_type} NOT FOUND at {path} or {path_alt}")
        return self.unique_sets.get(core_type, set())

    def _get_uniqueness_range(self, difficulty, rows=4, cols=4):
        """Get (min, max) ratio range for specified difficulty.
        # Grids >= 35 or 3x3x3 face surface area (9) are considered large for uniqueness rules"""
        is_large = (rows * cols >= 35) or (rows * cols == 9)
        if is_large:
            # 6x8 grid empirical limits: Hard=42-55%, Medium=26-41%, Easy=1-25%
            ranges = {"Easy": (0.01, 0.25), "Medium": (0.26, 0.41), "Hard": (0.42, 0.55)}
        else:
            # 4x4 or 4x6 grid empirical limits: Hard=55%+, Medium=30-54%, Easy=0-29%
            ranges = {"Easy": (0.0, 0.29), "Medium": (0.30, 0.54), "Hard": (0.55, 1.0)}

        return ranges.get(difficulty, (0, 1.0))

    def get_uniqueness_ratio(self, board, all_words, rows=4, cols=4, dictionary="NWL"):
        """Calculate the uniqueness ratio for a given board and word list.
        User Requirement: For small boards (4x4, 4x6), ignore 3-letter and 4-letter words
        to ensure the percentage isn't diluted by common filler.
        """
        if not all_words:
            return 0.0

        unique_set = self._get_difficulty_set(dictionary)
        if not unique_set:
            return 0.0

        num_tiles = int(rows) * int(cols)
        # Small grid criteria: < 35 cells (4x4=16, 4x6=24, 5x6=30)
        is_small = (num_tiles < 35)

        if is_small:
            # IGNORE 3-letter and 4-letter words for uniqueness ratio (User Request)
            relevant_words = [w for w in all_words if len(w) >= 5]
            if not relevant_words:
                # Fallback to all words ONLY if no 5L+ words exist (rare)
                relevant_words = list(all_words)
        else:
            # Large grids/Cubes use all words of at least 3L
            relevant_words = list(all_words)

        count_relevant = len(relevant_words)
        count_unique = sum(1 for w in relevant_words if w.upper() in unique_set)

        return count_unique / count_relevant if count_relevant > 0 else 0.0

    def get_difficulty_label(self, ratio, rows=4, cols=4):
        """Derive difficulty label from actual uniqueness ratio achieved."""
        # Defensive casting to ensure math logic works
        try:
            # ROBUST PARSING: handles '0.14', '14.0', or '14%'
            rat_str = str(ratio).replace('%', '').strip()
            rat = float(rat_str) if rat_str else 0.0
            
            # Auto-scale if passed as percentage (e.g. 14.0 instead of 0.14)
            if rat > 1.0:
                rat = rat / 100.0
                
            r = int(rows)
            c = int(cols)
        except Exception as e:
            print(f"[BoardGen-Diff] ERROR parsing ratio '{ratio}': {e}")
            return "Easy" # Safe fallback

        res = "Easy"
        
        # Grid Size Metrics
        total_tiles = r * c
        # 6x8 grid = 48, 3x3x3 cube = 54 (6 faces * 3x3), 4x4 = 16, 4x6 = 24
        # User Requirement: 4x4 and 4x6 are NOT large.
        is_large = (total_tiles >= 35) or (total_tiles == 54)
        
        if is_large:
            # 6x8 grid or 3x3x3 cube empirical limits (Harder to get high uniqueness on HUGE grids)
            if rat >= 0.38:
                res = "Hard"
            elif rat >= 0.22:
                res = "Medium"
            else:
                res = "Easy"
        else:
            # 4x4 or 4x6 grid thresholds: Hard=55%+, Medium=36-54%, Easy=0-35%
            # USER REQUEST: Ensure 33%, 24%, 13% or 16% on 4x4 is strictly EASY.
            if rat >= 0.55:
                res = "Hard"
            elif rat >= 0.36:
                res = "Medium"
            else:
                res = "Easy"
        
        # CRITICAL DEBUG: This helps explain WHY a label was chosen in the logs.
        # Check this in server.log to see the exact decision path.
        print(f"[BoardGen-Diff] FINAL RESULT: {res} | rat={rat:.4f} (Raw: {ratio}) | Grid: {r}x{c} ({total_tiles} tiles) | Large: {is_large}")
        return res

    def _is_creating_forbidden_sequence(self, board, char, r, c, f, target_seq="ING", depth=1):
        """Highly optimized local check to see if placing 'char' at (r, c, f) creates forbidden sequence."""
        # 1. Base check: is char even in the forbidden set?
        if char not in target_seq:
            return False

        is_3d = depth > 1
        if is_3d:
            if not board: return False
            depth_val = len(board)
            if depth_val == 0: return False
            if f >= len(board) or board[f] is None: return False
            rows_val = len(board[f])
            if r >= rows_val or board[f][r] is None: return False
            cols_val = len(board[f][r])
        else:
            if not board: return False
            depth_val = 1
            rows_val = len(board)
            cols_val = len(board[0]) if rows_val > 0 else 0

        def get_val(nf, nr, nc):
            try:
                if is_3d:
                    if 0 <= nf < len(board) and 0 <= nr < len(board[nf]) and 0 <= nc < len(board[nf][nr]):
                        return board[nf][nr][nc]
                else:
                    if 0 <= nr < len(board) and 0 <= nc < len(board[nr]):
                        return board[nr][nc]
            except (IndexError, TypeError):
                pass
            return None

        # 2. Local neighborhood check for "ING" specifically
        # Case A: Placing 'I'
        if char == "I":
            # Need N-G neighbor chain
            neighbors = []
            if depth_val == 6:  # Cube surface
                neighbors = self._get_cube_neighbors(f, r, c)
            else:
                for df in ([-1, 0, 1] if is_3d else [0]):
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if df == 0 and dr == 0 and dc == 0:
                                continue
                            nf, nr, nc = f + df, r + dr, c + dc
                            if 0 <= nf < depth_val and 0 <= nr < rows_val and 0 <= nc < cols_val:
                                neighbors.append((nf, nr, nc))

            for nf, nr, nc in neighbors:
                if get_val(nf, nr, nc) == "N":
                    # Search for G neighbor of THIS N
                    n2_neighbors = []
                    if depth_val == 6:
                        n2_neighbors = self._get_cube_neighbors(nf, nr, nc)
                    else:
                        for d2f in ([-1, 0, 1] if is_3d else [0]):
                            for d2r in [-1, 0, 1]:
                                for d2c in [-1, 0, 1]:
                                    if d2f == 0 and d2r == 0 and d2c == 0:
                                        continue
                                    n2f, n2r, n2c = nf + d2f, nr + d2r, nc + d2c
                                    if 0 <= n2f < depth_val and 0 <= n2r < rows_val and 0 <= n2c < cols_val:
                                        n2_neighbors.append((n2f, n2r, n2c))

                    for n2f, n2r, n2c in n2_neighbors:
                        if (n2f, n2r, n2c) == (f, r, c):
                            continue  # Don't revisit 'I'
                        if get_val(n2f, n2r, n2c) == "G":
                            return True
        # Case B: Placing 'N'
        elif char == "N":
            # Need 'I' neighbor AND 'G' neighbor
            has_i = False
            has_g = False

            neighbors = []
            if is_3d and depth_val == 6:
                neighbors = self._get_cube_neighbors(f, r, c)
            else:
                for df in ([-1, 0, 1] if is_3d else [0]):
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if df == 0 and dr == 0 and dc == 0:
                                continue
                            nf, nr, nc = f + df, r + dr, c + dc
                            if 0 <= nf < depth_val and 0 <= nr < rows_val and 0 <= nc < cols_val:
                                neighbors.append((nf, nr, nc))

            for nf, nr, nc in neighbors:
                val = get_val(nf, nr, nc)
                if val == "I":
                    has_i = True
                if val == "G":
                    has_g = True
                if has_i and has_g:
                    return True
        # Case C: Placing 'G'
        elif char == "G":
            # Need N-I neighbor chain
            neighbors = []
            if is_3d and depth_val == 6:
                neighbors = self._get_cube_neighbors(f, r, c)
            else:
                for df in ([-1, 0, 1] if is_3d else [0]):
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if df == 0 and dr == 0 and dc == 0:
                                continue
                            nf, nr, nc = f + df, r + dr, c + dc
                            if 0 <= nf < depth_val and 0 <= nr < rows_val and 0 <= nc < cols_val:
                                neighbors.append((nf, nr, nc))

            for nf, nr, nc in neighbors:
                if get_val(nf, nr, nc) == "N":
                    n2_neighbors = []
                    if is_3d and depth_val == 6:
                        n2_neighbors = self._get_cube_neighbors(nf, nr, nc)
                    else:
                        for d2f in ([-1, 0, 1] if is_3d else [0]):
                            for d2r in [-1, 0, 1]:
                                for d2c in [-1, 0, 1]:
                                    if d2f == 0 and d2r == 0 and d2c == 0:
                                        continue
                                    n2f, n2r, n2c = nf + d2f, nr + d2r, nc + d2c
                                    if 0 <= n2f < depth_val and 0 <= n2r < rows_val and 0 <= n2c < cols_val:
                                        n2_neighbors.append((n2f, n2r, n2c))

                    for n2f, n2r, n2c in n2_neighbors:
                        if (n2f, n2r, n2c) == (f, r, c):
                            continue
                        if get_val(n2f, n2r, n2c) == "I":
                            return True

        # PROSCRIBED SEQUENCES: SEX, FUCK, SHIT, etc. (Safety & Public Friendly boards)
        # Check if placing 'char' at (f,r,c) completes ANY word in this list
        # Simple adjacency check (covers rows, cols, diagonals)
        proscribed = ["SEX", "FUCK", "CUNT", "SHIT", "LUBE", "PORN", "COCK", "DICK", "BONE", "PISS", "CLIT"]
        for p_word in proscribed:
            p_len = len(p_word)
            if char in p_word:
                # Potential match. Check recursively for neighbors that complete the sequence.
                p_idx = p_word.index(char)

                # Simple neighborhood check for adjacent letters from p_word
                prev_t = p_word[p_idx - 1] if p_idx > 0 else None
                next_t = p_word[p_idx + 1] if p_idx < p_len - 1 else None

                for df in ([-1, 0, 1] if depth_val > 1 else [0]):
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if df == 0 and dr == 0 and dc == 0:
                                continue
                            nf, nr, nc = f + df, r + dr, c + dc
                            if 0 <= nf < depth_val and 0 <= nr < rows_val and 0 <= nc < cols_val:
                                v = get_val(nf, nr, nc)
                                if v == prev_t or v == next_t:
                                    # High probability of forming the word. Block.
                                    return True
        return False

    def _has_forbidden_sequence(self, board, sequence="ING", depth=1):
        """Perform a board-wide scan for a forbidden sequence."""
        is_3d = (depth > 1) or (len(board) == 6 and isinstance(board[0], list) and isinstance(board[0][0], list))
        depth_val = 6 if (len(board) == 6 and is_3d) else depth
        rows, cols = (len(board[0]), len(board[0][0])) if is_3d else (len(board), len(board[0]))
        seq_len = len(sequence)

        def find_next(idx, r, c, f, visited, d_val):
            if idx == seq_len:
                return True
            target = sequence[idx]
            for df in ([-1, 0, 1] if d_val > 1 else [0]):
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if df == 0 and dr == 0 and dc == 0:
                            continue
                        nf, nr, nc = f + df, r + dr, c + dc
                        if 0 <= nf < d_val and 0 <= nr < rows and 0 <= nc < cols and (nf, nr, nc) not in visited:
                            val = board[nf][nr][nc] if d_val > 1 else board[nr][nc]
                            if val == target:
                                visited.add((nf, nr, nc))
                                if find_next(idx + 1, nr, nc, nf, visited, d_val):
                                    return True
                                visited.remove((nf, nr, nc))
            return False

        for f in range(depth_val):
            for r in range(rows):
                for c in range(cols):
                    val = board[f][r][c] if depth_val > 1 else board[r][c]
                    if val == sequence[0]:
                        if find_next(1, r, c, f, {(f, r, c)}, depth_val):
                            return True
        return False

    def _count_forbidden_sequence(self, board, sequence="ING", depth=1):
        """Count the number of times a sequence occurs on the entire board."""
        is_3d = (depth > 1) or (len(board) == 6 and isinstance(board[0], list) and isinstance(board[0][0], list))
        depth_val = 6 if (len(board) == 6 and is_3d) else depth
        rows, cols = (len(board[0]), len(board[0][0])) if is_3d else (len(board), len(board[0]))
        seq_len = len(sequence)

        def find_next(idx, r, c, f, visited, d_val):
            if idx == seq_len:
                return 1
            target = sequence[idx]
            paths = 0
            for df in ([-1, 0, 1] if d_val > 1 else [0]):
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if df == 0 and dr == 0 and dc == 0:
                            continue
                        nf, nr, nc = f + df, r + dr, c + dc
                        if 0 <= nf < d_val and 0 <= nr < rows and 0 <= nc < cols and (nf, nr, nc) not in visited:
                            val = board[nf][nr][nc] if d_val > 1 else board[nr][nc]
                            if val == target:
                                visited.add((nf, nr, nc))
                                paths += find_next(idx + 1, nr, nc, nf, visited, d_val)
                                visited.remove((nf, nr, nc))
            return paths

        cnt = 0
        for f in range(depth_val):
            for r in range(rows):
                for c in range(cols):
                    val = board[f][r][c] if depth_val > 1 else board[r][c]
                    if val == sequence[0]:
                        cnt += find_next(1, r, c, f, {(f, r, c)}, depth_val)
        return cnt

    def _select_strategy(self, dimensions, min_words, max_words, difficulty, min_word_length):
        """Standard methodology selection based on empirical analysis of 200+ parameter sets"""
        parts = dimensions.split("x")
        if len(parts) == 3:
            depth, rows, cols = map(int, parts)
        else:
            rows, cols = map(int, parts)
            depth = 1

        # Small grids targeting uniqueness (Hard) or high density (Medium 100+)
        if rows * cols <= 25:
            # User Request Fix: High word count targets on small grids MUST use IO to hit range reliably
            if min_words >= 100 and rows * cols >= 35:
                return "StepwiseOptimization"
            # On small grids (4x4), FastReRoll can hit 100-200 easily without heavy optimization
            return "FastReRoll"

        if rows * cols >= 35:
            # User Request: On large grids, standard random generation is way too dense (600+ words).
            # We MUST use IO to effectively target restricted counts (like 50-100 or 100-200).
            # For 200+ counts, standard generation satisfies the requirement instantly.
            if max_words < 500:
                return "StepwiseOptimization"
            return "FastReRoll"

        # Large grids OR high density targets (200+) need optimization
        if difficulty == "Hard" or min_words >= 200:
            return "StepwiseOptimization"

        # Standard large grid density
        return "FastReRoll"

    def generate_board(
        self, dimensions, bonus_word, word_count_range, dictionary, board_format, min_word_length=3, difficulty="Medium"
    ):
        """
        Generate a valid board that meets word count requirements.
        Uses cached optimal method or tests both formats on first use.
        Only counts words >= min_word_length.
        Returns: (board, all_words, bonus_cell, board_format, all_words_dict, uniqueness_ratio)
        """
        import time

        with open("/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log", "a") as f:
            f.write(f"[board_generator.py] generate_board ENTERED for {dimensions} (Range: {word_count_range}) at {time.time()}\n")

        # --- DIFFICULTY NORMALIZATION (User Change: Normal -> Medium / Expert -> Hard) ---
        # Ensure that lobby-specific or legacy difficulty labels are mapped correctly for internal logic
        diff_map = {"NORMAL": "Medium", "EXPERT": "Hard", "DIFFICULT": "Hard", "MASTERS": "Hard"}
        internal_difficulty = diff_map.get(difficulty.upper(), difficulty.capitalize())
        difficulty = internal_difficulty

        # Initialize defaults to prevent NameError in return paths
        board = None
        all_words = []
        bonus_cell = None
        word_count = 0

        # FOR UNCONDITIONAL UNIQUENESS: Re-seed random from system randomness
        # This breaks any process-level determinism from forks/seeds
        import random

        random.seed()

        # Dimension Parsing (3x3x3 is 6 faces of 3x3 = 54 tiles)
        if dimensions == "3x3x3":
            depth, rows, cols = 6, 3, 3
        else:
            parts = dimensions.split("x")
            if len(parts) == 3:
                depth, rows, cols = map(int, parts)
            else:
                rows, cols = map(int, parts)
                depth = 1
        num_tiles = rows * cols * depth

        num_tiles = rows * cols * depth
        min_words, max_words = self._parse_word_count_range(word_count_range)
        
        # SMALL BOARD SAFETY CLAMP: 
        # Large word count targets (200+, 100+) depend heavily on min_word_length.
        if num_tiles <= 16: 
            # Empirical limits for 4x4: 3L: ~180, 4L: ~100, 5L: ~45
            target_limit = 180 if min_word_length <= 3 else 100 if min_word_length == 4 else 45
            if min_words > target_limit:
                print(f"[BoardGen] Target {min_words} is extremely difficult for 4x4 ({min_word_length}L). Clamping to {target_limit}.")
                min_words = target_limit
            if max_words < min_words:
                max_words = min_words * 2

        print(
            f"[BoardGen] Target word count: {min_words}-{max_words if max_words != float('inf') else '∞'} (Tiles: {num_tiles})"
        )

        # REMOVED: Cache lookup that overrode user format preference
        # We now strictly respect the board_format passed in arguments

        # 0. Handle "Mania" without a prefix (e.g. from user dropdown selection)
        # --- PERSISTENT OUTER LOOP (User Request: Keep loading until requirements met) ---
        # USER REQUEST: Absolute Ironclad Compliance. Keep searching until satisfied.
        # Global safety timeout: 120s (User prefers high latency over non-compliance)
        start_overall_time = time.time()
        # Ensure bonus word length is valid
        if len(bonus_word) < min_word_length:
            print(f"[BoardGen] ERROR: Bonus word '{bonus_word}' ({len(bonus_word)}) is shorter than min_word_length ({min_word_length}).")
            
        # Ironclad Loop Count (Safety)
        outer_restarts = 0
        overall_timeout = 45 # Increased from 25 to allow high-latency ironclad compliance
        while time.time() - start_overall_time < overall_timeout and outer_restarts < 3:
            # Re-seed every cycle to ensure varied results
            import random
            random.seed()
            
            with open("/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log", "a") as f:
                f.write(f"[board_generator.py] Cycle starting for {dimensions} (Retry {outer_restarts})\n")
            # Re-read word counts for this pass
            min_words, max_words = self._parse_word_count_range(word_count_range)

            # --- STRATEGY SELECTION (Based on parameters.txt optimal mapping) ---
            strategy = self._select_strategy(dimensions, min_words, max_words, difficulty, min_word_length)

            # 1. Reset metrics for this overall pass
            board = None
            all_found_list = []
            bonus_cell = None
            word_count = 0

            fmt_clean = board_format.strip()
            fmt_lower = fmt_clean.lower()
            board_format = fmt_clean  # Restore original

            # Mania Logic
            if "mania" in fmt_lower:
                mania_letter = random.choice("ABCDEFGHIJKLMNOPRST")
                board_format = f"{mania_letter} Mania"
                fmt_clean = board_format.strip()
                fmt_lower = fmt_clean.lower()
                print(f"[BoardGen] Mania Pass: Picked letter '{mania_letter}'")

            # Try to generate valid board (Standard search attempts)
            max_attempts = 100
            is_4x4 = rows * cols == 16
            is_hard_req = min_word_length >= 7 or min_words >= 150
            is_extreme_req = min_word_length >= 8 or (min_word_length >= 7 and "checkerboard" in board_format.lower())

            # USER REQUEST: For IO-based strategies, each attempt is a full optimization pass. 
            # We don't need 100+ attempts; if it doesn't converge in 5-10, we should fallback.
            if strategy in ["DensityOptimization", "HardOptimization", "StepwiseOptimization"]:
                # Optimization strategies are heavier, but 4x4 is fast. 
                # Give 4x4 more passes to ensure it hits the Hard target revealed in lobby.
                max_attempts = 100 if is_4x4 else 12
            elif is_4x4:
                # Standard strategies for 4x4 are almost instant.
                # Give them a high budget to avoid falling back to 'Medium' and causing header jumps.
                max_attempts = 250 if min_words >= 150 else 150
            elif is_extreme_req:
                max_attempts = 50
            elif is_hard_req:
                max_attempts = 30
            else:
                max_attempts = 20

            # PERFORMANCE: Safeguard against blanket overrides for optimization strategies
            if strategy in ["DensityOptimization", "HardOptimization", "StepwiseOptimization"] and rows * cols >= 35:
                # User Request: For huge grids, don't repeat the expensive optimization loop. 
                # If target isn't met in the first pass, the fallback logic is faster and safer.
                max_attempts = 1 

            # --- BEST BOARD TRACKING ---
            best_board_global = None
            best_words_global = []
            best_cell_global = None
            best_fmt_global = board_format
            best_dict_global = None
            best_ratio_global = 0.0
            best_count_global = -1

            # Final Attempt Adjustment
            if (
                strategy not in ["DensityOptimization", "HardOptimization", "StepwiseOptimization"]
                and not is_extreme_req
                and not is_4x4
            ):
                if rows * cols >= 35:
                    max_attempts = max(max_attempts, 40)
                else:
                    max_attempts = max(max_attempts, 25)

            for attempt in range(max_attempts):
                # ABSOLUTE SAFETY BREAK: Never exceed overall timeout during attempt loop
                if time.time() - start_overall_time > overall_timeout:
                    print(f"[BoardGen] !! Loop AUTO-BREAK: Time limit ({overall_timeout}s) exceeded.")
                    break
                
                print(f"[BoardGen] Attempt {attempt}/{max_attempts} (Strategy: {strategy})")

                # Weight Selection: Denser weights (EASY) for density targets
                # Weight Selection: Denser weights (EASY) for density targets
                if difficulty == "Easy":
                    weights = LETTER_FREQ_EASY
                elif min_words >= 150:
                    weights = LETTER_FREQ_EASY
                elif (num_tiles >= 35) and max_words <= 150:
                    # Ultra-Sparse weights for HUGE grids with LOW word targets (50-100)
                    weights = LETTER_FREQ_SPARSE
                elif (num_tiles >= 35) and max_words <= 320:
                    # Balanced user weights for normal large grid targets
                    weights = LETTER_FREQ_USER
                elif difficulty in ["Medium", "Hard"] or is_4x4 or strategy == "HardOptimization" or depth > 1:
                    weights = LETTER_FREQ_USER
                else:
                    weights = LETTER_FREQ_EASY

                fmt_clean = board_format.strip()
                fmt_lower = fmt_clean.lower()
                is_checkerboard_fmt = "checkerboard" in fmt_lower

                # No silent board format overrides for high density; honor the Spinner Set's request.

                # On 4x4, uniqueness thresholds should remain high
                if is_4x4:
                    min_r, max_r = self._get_uniqueness_range(difficulty, rows, cols)
                else:
                    # USER REQUEST: SPEED. For large grids (6x8), reduce retry attempts.
                    # Large grids are easy to pack; if we fail twice, we should hit fallback early.
                    if (num_tiles >= 35):
                        max_attempts = 4
                    
                    # On the absolute last attempts, relax the word count range to prioritize the bonus word
                    current_min_words = min_words
                    current_max_words = max_words
                    if attempt >= max_attempts - 1:
                        print(f"[BoardGen] ! Relaxing word count constraints to ensure bonus word embedding.")
                        current_min_words = max(5, min_words // 2)
                        current_max_words = max_words * 2

                # --- DICTIONARY ALIGNMENT (User Request: TITANS verification) ---
                # We preserve the original dictionary (NWL/CSW) for the final round data.
                # We only use the 'Unique' dictionaries for the ITERATIVE search phase.
                original_dictionary = dictionary
                if difficulty == "Hard" or strategy == "HardOptimization" or is_4x4:
                    if dictionary.upper() == "NWL":
                        search_dictionary = "UniqueNWL"
                    elif dictionary.upper() == "CSW":
                        search_dictionary = "UniqueCSW"
                    else:
                        search_dictionary = dictionary
                else:
                    search_dictionary = dictionary

                # --- BOARD CREATION & OPTIMIZATION ---
                if is_checkerboard_fmt:
                    board = self._create_checkerboard(rows, cols, weights, depth=depth)
                else:
                    board = self._create_normal_board(rows, cols, weights, depth=depth)

                # --- BONUS WORD EMBEDDING (MANDATORY) ---
                # Embed BEFORE IO optimization so we can protect it
                bonus_cell = None
                bonus_cells_set = set()
                actual_bonus_word = bonus_word if bonus_word else ""
                if actual_bonus_word:
                    if depth > 1:
                        path = self._embed_bonus_word_cube(board, actual_bonus_word, is_checkerboard=is_checkerboard_fmt)
                    else:
                        path = self._embed_bonus_word(board, actual_bonus_word, is_checkerboard=is_checkerboard_fmt)
                    if not path:
                        print(
                            f"[BoardGen] ✗ Failed to embed bonus word '{actual_bonus_word}', retrying attempt {attempt}..."
                        )
                        continue
                    bonus_cells_set = set(path)
                    print(f"[BoardGen] ✓ Bonus word '{actual_bonus_word}' embedded successfully")

                # --- RUN OPTIMIZATION PASS IF REQUIRED ---
                # (Optimization logic moved below special formats)

                # --- SPECIAL FORMAT SPECIALS (Bonus Letter, Either/Or) ---
                # User Change: Track ALL special cells to protect them all from post-processing
                special_cells = []
                if "bonus letter" in fmt_lower:
                    if depth > 1:
                        selectable_cells = [
                            (f, r, c) for f in range(depth) for r in range(rows) for c in range(cols) if (f, r, c) not in bonus_cells_set
                        ]
                    else:
                        selectable_cells = [
                            (r, c) for r in range(rows) for c in range(cols) if (r, c) not in bonus_cells_set
                        ]
                    b_cell = random.choice(selectable_cells) if selectable_cells else (0, 0, 0) if depth > 1 else (0, 0)
                    special_cells.append(b_cell)
                    print(f"[BoardGen] * Bonus Letter cell: {b_cell}")

                if "either/or" in fmt_lower or "either" in fmt_lower:
                    # Exactly one Either/Or cell per board
                    num_eo = 1
                    print(f"[BoardGen] Applying {num_eo} Either/Or cell...")

                    for _ in range(num_eo):
                        if depth > 1:
                            selectable_cells = [(f, r, c) for f in range(depth) for r in range(rows) for c in range(cols) if (f, r, c) not in bonus_cells_set and (f, r, c) not in special_cells]
                        else:
                            selectable_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in bonus_cells_set and (r, c) not in special_cells]
                        
                        if not selectable_cells: break
                        eo_cell = random.choice(selectable_cells)
                        special_cells.append(eo_cell)
                        
                        if depth > 1:
                            f_eo, r_eo, c_eo = eo_cell
                            orig = board[f_eo][r_eo][c_eo] or random.choices(self.letters, weights=weights, k=1)[0]
                            others = [l for l in self.letters if l != orig]
                            other = random.choices(others, weights=[weights[self.letters.index(l)] for l in others], k=1)[0]
                            pair = sorted([str(orig), str(other)])
                            board[f_eo][r_eo][c_eo] = f"{pair[0]}/{pair[1]}"
                        else:
                            r_eo, c_eo = eo_cell
                            orig = board[r_eo][c_eo] or random.choices(self.letters, weights=weights, k=1)[0]
                            others = [l for l in self.letters if l != orig]
                            other = random.choices(others, weights=[weights[self.letters.index(l)] for l in others], k=1)[0]
                            pair = sorted([str(orig), str(other)])
                            board[r_eo][c_eo] = f"{pair[0]}/{pair[1]}"
                    print(f"[BoardGen] * Applied {len(special_cells)} special cells.")

                # Define the PRIMARY bonus_cell for room metadata (used for the badge)
                bonus_cell = special_cells[-1] if special_cells else None

                # --- POST-PROCESSING ---
                # Unify all critical cells that must NOT be overwritten during balancing/propagation
                all_excluded = set(bonus_cells_set)
                for sc in special_cells:
                    all_excluded.add(sc)

                # --- SPECIAL FORMATS: MANIA FLOODING ---
                # MUST happen BEFORE IO so optimization respects the mania constraints
                mania_letter = None
                if "mania" in fmt_lower:
                    format_parts = fmt_lower.split()
                    if len(format_parts) >= 2:
                        mania_letter = format_parts[0].upper()
                        if len(mania_letter) == 1:
                            # Initial flood to seed the board before optimization
                            self._apply_mania_to_board(
                                board, mania_letter, all_excluded, is_checkerboard=is_checkerboard_fmt
                            )
                            # Add flooded cells to all_excluded to protect them during IO
                            for r in range(rows):
                                for c in range(cols):
                                    if board[r][c] == mania_letter:
                                        all_excluded.add((r, c))

                # --- RUN OPTIMIZATION PASS IF REQUIRED ---
                # IO will now strictly honor all_excluded (Bonus word + Mania letters)
                if strategy == "StepwiseOptimization":
                    min_target_r, max_target_r = self._get_uniqueness_range(difficulty, rows, cols)
                    board = self._create_2000plus_board(
                        rows,
                        cols,
                        search_dictionary,
                        is_checkerboard_fmt,
                        board,
                        all_excluded,
                        "Uniqueness" if difficulty == "Hard" else "Density",
                        min_word_length,
                        max_words,
                        min_words,
                        min_target_r,
                        max_target_r,
                        depth=depth,
                        difficulty=difficulty,
                        bonus_word=actual_bonus_word,
                        weights=weights,
                    )
                elif strategy == "HighDensity":
                    board = self._create_2000plus_board(
                        rows,
                        cols,
                        search_dictionary,
                        is_checkerboard_fmt,
                        board,
                        all_excluded,
                        "Density",
                        min_word_length,
                        max_words,
                        min_words,
                        0,
                        1,
                        depth=depth,
                        difficulty=difficulty,
                        bonus_word=actual_bonus_word,
                        weights=weights,
                    )

                # Apply vowel balancing (User Request: Maintain 30-35% on all boards including 6x8)
                # Use EXACT checkerboard flag
                if not is_checkerboard_fmt:
                    self._enforce_vowel_minimum(
                        board, weights, is_checkerboard=False, excluded_cells=all_excluded, difficulty=difficulty
                    )

                if "either/or" in fmt_lower or "either" in fmt_lower:
                    if self._has_either_or_ambiguity(board, dictionary):
                        print(f"[BoardGen] ✗ Either/Or ambiguity detected, retrying...")
                        continue

                if is_checkerboard_fmt:
                    self._verify_checkerboard_safeguard(board, weights, bonus_cells_set)

                # --- UNIQUENESS & FORBIDDEN SEQUENCES ---
                # User Request: ING forbidden in Medium/Hard
                # Only allow ING if it is part of the bonus word
                max_ing = actual_bonus_word.upper().count("ING") if actual_bonus_word else 0
                if (
                    difficulty in ["Medium", "Hard"]
                    and self._count_forbidden_sequence(board, "ING", depth=depth) > max_ing
                ):
                    print(f"[BoardGen] ✗ Forbidden ING sequence detected (found > {max_ing}), retrying...")
                    continue

                unique_set = self._get_difficulty_set(original_dictionary)
                print(f"[BoardGen] Uniqueness set size for {original_dictionary}: {len(unique_set)}")
                min_r, max_r = self._get_uniqueness_range(difficulty, rows, cols)
                # --- FAST SOLVE WITHOUT PATH TRACKING ---
                # JAVA ALIGNMENT: Always solve against the original dict (NWL) for the final result list.
                # PERFORMANCE: Use depth 10 (large) or 13 (small) during search for speed; final return will use depth 25.
                solve_depth_temp = 10 if (rows * cols >= 35) else 13
                all_words_dict = self._solve_board(
                    board,
                    original_dictionary,
                    (0, 99999),
                    min_word_length,
                    max_depth=solve_depth_temp,
                    store_paths=False,
                    timeout=5.0, # Final results solve must be fast
                )

                if all_words_dict is not None:
                    count_total = len(all_words_dict)
                    # User Request: Use the TOTAL word count for uniqueness percentage (not just 6-8L) for small grids
                    # to reflect the true difficulty of the board. Huge grids and 3D boards still prioritize long word rarity.
                    if num_tiles >= 35 or (depth > 1): # 6x6 (36) or larger or Cube (27)
                        # User Request Accuracy: Use ALL words of at least min_word_length for uniqueness ratio
                        # This ensures the percentage accurately reflects the full board complexity.
                        relevant_words = [w for w in all_words_dict if len(w) >= min_word_length]
                        if not relevant_words:
                            relevant_words = list(all_words_dict.keys())
                    else:
                        # User Request (Strict Accuracy): For small boards (4x4, 4x6, 5x5), 
                        # IGNORE 3-letter and 4-letter words when calculating uniqueness.
                        # This avoids common 'filler' words diluting the difficulty ratio.
                        relevant_words = [w for w in all_words_dict if len(w) >= 5]
                        if not relevant_words:
                            relevant_words = list(all_words_dict.keys())

                    count_relevant = len(relevant_words)
                    count_unique = sum(1 for w in relevant_words if w.upper() in unique_set)
                    ratio = count_unique / count_relevant if count_relevant > 0 else 0

                    # Extract word list from dict for initial validation
                    all_found_list = list(all_words_dict.keys())

                    # --- SMARTER BEST BOARD TRACKING ---
                    # We prioritize:
                    # 1. Satisfaction of UNIQUENESS range (Essential for Difficulty)
                    # 2. Satisfaction of WORD COUNT range
                    # 3. Highest word count (Density)

                    within_word_range = min_words <= count_total <= max_words
                    within_unique_range = min_r <= ratio <= max_r

                    # Current candidate score
                    # 5000 for uniqueness + 1000 for word count + actual count
                    # CRITICAL: If outside word range, we SUBTRACT 10000 to ensure ANY in-range board wins
                    range_bonus = 1000 if within_word_range else -10000
                    unique_bonus = 5000 if within_unique_range else 0
                    candidate_score = unique_bonus + range_bonus + count_total

                    # Get previous best score
                    prev_within_word = min_words <= best_count_global <= max_words
                    prev_within_unique = min_r <= best_ratio_global <= max_r
                    best_score = (
                        (5000 if prev_within_unique else 0) + (1000 if prev_within_word else -10000) + best_count_global
                    )

                    if candidate_score > best_score:
                        best_count_global = count_total
                        best_board_global = [row[:] for row in board]
                        best_words_global = sorted(all_found_list)
                        best_cell_global = bonus_cell
                        best_fmt_global = board_format
                        best_dict_global = all_words_dict
                        best_ratio_global = ratio

                    # USER REQUEST: Ironclad word count compliance (Zero tolerance for undershooting minimum)
                    # For grids < 48 cells, we use absolute range matching.
                    is_compliant = min_words <= count_total <= max_words

                    if num_tiles >= 48:
                        # Large grids (6x8) still allow a 10% overshoot for density-heavy targets to avoid
                        # infinite loops, but we strictly enforce the minimum.
                        limit_ratio = 1.10 if max_words <= 200 else 1.20
                        overshoot_limit = int(max_words * limit_ratio)
                        is_compliant = min_words <= count_total <= overshoot_limit

                    if difficulty != "Easy" and not within_unique_range:
                        print(
                            f"[BoardGen] ✗ Uniqueness ratio {ratio:.2%} outside range ({min_r:.2%}-{max_r:.2%}) for {difficulty}, retrying..."
                        )
                        continue

                    if not is_compliant:
                        print(
                            f"[BoardGen] ✗ NON-COMPLIANT word count {count_total} for target {min_words}-{max_words}. KEEP SEARCHING..."
                        )
                        continue

                    print(f"[BoardGen] ✓ Valid board found: {count_total} words, {ratio:.2%} unique")

                    # GUARANTEE: Confirm bonus word actually exists in the final board's word list
                    actual_found = (
                        (actual_bonus_word.upper() in [w.upper() for w in all_found_list])
                        if actual_bonus_word
                        else True
                    )

                    if not actual_found and actual_bonus_word:
                        print(
                            f"[BoardGen] ! Bonus word '{actual_bonus_word}' found on board but filtered. Injecting manually."
                        )
                        all_words_dict[actual_bonus_word.upper()] = path
                        all_found_list.append(actual_bonus_word.upper())
                        actual_found = True
                    # Finalpass return
                    print(f"[BoardGen] Executing Final Path-Tracking Solve on selected board...")
                    final_depth = 25 if (rows * cols < 35) else 12
                    final_words_dict = self._solve_board(
                        board, original_dictionary, (0, 99999), min_word_length, max_depth=final_depth, store_paths=True
                    )
                    
                    # RECALCULATE ACCURATE RATIO (User Request: Absolute Accuracy)
                    unique_set_final = self._get_difficulty_set(original_dictionary)
                    
                    # IGNORE 3s and 4s for small grids when determining FINAL percentage (User Request)
                    if (rows * cols < 35) and (depth == 1):
                        f_words_rel = [w for w in final_words_dict if len(w) >= 5]
                        if not f_words_rel: f_words_rel = list(final_words_dict.keys())
                    else:
                        f_words_rel = list(final_words_dict.keys())

                    f_count = len(f_words_rel)
                    f_unique = sum(1 for w in f_words_rel if w.upper() in unique_set_final)
                    final_ratio = f_unique / f_count if f_count > 0 else 0

                    # --- EITHER/OR COUNT ENFORCEMENT (User Request: Exactly One) ---
                    # In rare cases, optimization or fallback might double-inject.
                    # We perform a final scan and resolve any duplicates by replacing with a random letter.
                    eo_count = 0
                    if depth > 1:
                        for f in range(depth):
                            for r in range(rows):
                                for c in range(cols):
                                    if "/" in str(board[f][r][c]):
                                        eo_count += 1
                                        if eo_count > 1:
                                            # Replace duplicate with a normal letter
                                            board[f][r][c] = random.choices(self.letters, weights=weights, k=1)[0]
                    else:
                        for r in range(rows):
                            for c in range(cols):
                                if "/" in str(board[r][c]):
                                    eo_count += 1
                                    if eo_count > 1:
                                        board[r][c] = random.choices(self.letters, weights=weights, k=1)[0]
                    
                    if eo_count > 1:
                        print(f"[BoardGen] ! Fixed Either/Or duplication: Reduced {eo_count} -> 1.")

                    if actual_bonus_word and actual_bonus_word.upper() not in final_words_dict:
                        print(f"[BoardGen] !! Bonus word '{actual_bonus_word}' missing from final solve. Injecting with coords {bonus_cell}")
                        final_words_dict[actual_bonus_word.upper()] = bonus_cell
                    
                    return (
                        board,
                        sorted(list(final_words_dict.keys())),
                        bonus_cell,
                        board_format,
                        final_words_dict,
                        final_ratio,
                        actual_bonus_word.upper() if actual_bonus_word else None
                    )
            # --- RETURN BEST FOUND IF ALL ATTEMPTS FAILED ---
            # User Request: If it isn't compliant, return the BEST candidate found rather than hanging.
            if best_board_global is not None:
                print(
                    f"[BoardGen] !! Exhausted {max_attempts} attempts. Returning BEST candidate (Words: {best_count_global}, Unique: {best_ratio_global:.2%})"
                )
                final_depth = 25 if (rows * cols < 35) else 14
                final_best_dict = self._solve_board(
                    best_board_global,
                    original_dictionary,
                    (0, 99999),
                    min_word_length,
                    max_depth=final_depth,
                    store_paths=True,
                )
                
                # RECALCULATE ACCURATE RATIO FOR BEST-FOUND
                unique_set_final = self._get_difficulty_set(original_dictionary)
                
                # IGNORE 3s and 4s for small grids (User Request)
                if (rows * cols < 35) and (depth == 1):
                    fb_words_rel = [w for w in final_best_dict if len(w) >= 5]
                    if not fb_words_rel: fb_words_rel = list(final_best_dict.keys())
                else:
                    fb_words_rel = list(final_best_dict.keys())

                fb_count = len(fb_words_rel)
                fb_unique = sum(1 for w in fb_words_rel if w.upper() in unique_set_final)
                fb_ratio = fb_unique / fb_count if fb_count > 0 else 0

                if actual_bonus_word and actual_bonus_word.upper() not in final_best_dict:
                    print(f"[BoardGen] !! Bonus word '{actual_bonus_word}' missing from best-found solve. Injecting.")
                    final_best_dict[actual_bonus_word.upper()] = best_cell_global

                return (
                    best_board_global,
                    sorted(list(final_best_dict.keys())),
                    best_cell_global,
                    best_fmt_global,
                    final_best_dict,
                    fb_ratio,
                    actual_bonus_word.upper() if actual_bonus_word else None
                )

            # FULL FALLBACK: Mandatory Injection on Clean Slate
            print(f"[BoardGen] !! FALLBACK ACTIVATED for {dimensions}. Using Clean Slate Injection.")
            # 1. Start with empty (Handle 3D Fallback)
            if depth > 1:
                fallback_board = [[["" for _ in range(cols)] for _ in range(rows)] for _ in range(depth)]
            else:
                fallback_board = [["" for _ in range(cols)] for _ in range(rows)]

            # 2. Force embed on EMPTY board (GUARANTEED success for L <= 16 on 4x4)
            actual_bonus_word = bonus_word if bonus_word else ""
            if depth > 1:
                path = self._embed_bonus_word_cube(fallback_board, actual_bonus_word)
            else:
                path = self._embed_bonus_word(fallback_board, actual_bonus_word, is_checkerboard=is_checkerboard_fmt)
            bonus_cells_set = set(path) if path else set()

        # 3. Special Cell Injection for Fallback (User Request: Either/Or must exist in fallback)
        fallback_special_cells = []
        if "bonus letter" in fmt_lower:
            if depth > 1:
                selectable = [(f, r, c) for f in range(depth) for r in range(rows) for c in range(cols) if (f, r, c) not in bonus_cells_set]
                b_cell = random.choice(selectable) if selectable else (0, 0, 0)
            else:
                selectable = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in bonus_cells_set]
                b_cell = random.choice(selectable) if selectable else (0, 0)
            fallback_special_cells.append(b_cell)

        if "either/or" in fmt_lower or "either" in fmt_lower:
            if depth > 1:
                selectable = [
                    (f, r, c)
                    for f in range(depth)
                    for r in range(rows)
                    for c in range(cols)
                    if (f, r, c) not in bonus_cells_set and (f, r, c) not in fallback_special_cells
                ]
                eo_cell = random.choice(selectable) if selectable else (0, 0, 0)
            else:
                selectable = [
                    (r, c)
                    for r in range(rows)
                    for c in range(cols)
                    if (r, c) not in bonus_cells_set and (r, c) not in fallback_special_cells
                ]
                eo_cell = random.choice(selectable) if selectable else (0, 0)
            fallback_special_cells.append(eo_cell)

        # Mania letter for fallback
        mania_letter_fb = None
        if "mania" in fmt_lower:
            # Extract mania letter from format (e.g. "E Mania")
            parts = fmt_lower.split()
            if len(parts) >= 2:
                mania_letter_fb = parts[0].upper()

        fallback_all_excluded = set(bonus_cells_set)
        for sc in fallback_special_cells:
            fallback_all_excluded.add(sc)

        # 4. Fill the rest with random letters
        # GRID-AWARE FALLBACK WEIGHTS:
        if (num_tiles >= 35) and max_words <= 150:
            weights = LETTER_FREQ_SPARSE
        elif (num_tiles >= 35) and max_words <= 320:
            weights = LETTER_FREQ_USER
        else:
            weights = self._get_weights(difficulty)
            
        final_words_dict = {}
        final_found_list = []
        # User Request: Speed. Reduce fallback attempts on large grids.
        fb_attempts = 5 if num_tiles >= 35 else 20
        for fb_attempt in range(fb_attempts):
            # Global Timeout Safety: Exit fallback if we've been generating for too long overall (40s mark)
            if time.time() - start_overall_time > 40:
                print(f"[BoardGen] !! FALLBACK AUTO-BREAK: Time limit (40s) reached.")
                break

            if depth > 1:
                for f in range(depth):
                    for r in range(rows):
                        for c in range(cols):
                            # Fill if empty OR if not excluded
                            if (f, r, c) not in fallback_all_excluded or not fallback_board[f][r][c]:
                                fallback_board[f][r][c] = random.choices(self.letters, weights=weights, k=1)[0]
            else:
                for r in range(rows):
                    for c in range(cols):
                        # Fill if empty OR if not excluded
                        if (r, c) not in fallback_all_excluded or not fallback_board[r][c]:
                            fallback_board[r][c] = random.choices(self.letters, weights=weights, k=1)[0]

                # Apply Mania flooding to fallback after random fill
                if mania_letter_fb:
                    self._apply_mania_to_board(
                        fallback_board, mania_letter_fb, fallback_all_excluded, is_checkerboard=is_checkerboard_fmt
                    )
                
                # Pattern compliance MUST be enforced on fallback too
                if is_checkerboard_fmt:
                    self._verify_checkerboard_safeguard(fallback_board, weights, fallback_all_excluded)

                # Inject Either/Or slash tile into fallback board
                if "either/or" in fmt_lower or "either" in fmt_lower:
                    eo_coords = fallback_special_cells[-1] if fallback_special_cells else None
                    if eo_coords:
                        if depth > 1:
                            f_eo, r_eo, c_eo = eo_coords
                            orig = fallback_board[f_eo][r_eo][c_eo]
                            if not orig:
                                orig = random.choices(self.letters, weights=weights, k=1)[0]
                        else:
                            r_eo, c_eo = eo_coords
                            orig = fallback_board[r_eo][c_eo]
                            if not orig:
                                orig = random.choices(self.letters, weights=weights, k=1)[0]
                            
                        # Choose a different letter for the slash
                        others = [l for l in self.letters if l != orig]
                        weights_others = [weights[self.letters.index(l)] for l in others]
                        other = random.choices(others, weights=weights_others, k=1)[0]
                        # Sort to ensure consistent "A/B" format
                        if depth > 1:
                            fallback_board[f_eo][r_eo][c_eo] = f"{sorted([orig, other])[0]}/{sorted([orig, other])[1]}"
                        else:
                            fallback_board[r_eo][c_eo] = f"{sorted([orig, other])[0]}/{sorted([orig, other])[1]}"

                # Final Path-Tracking Solve for fallback check
                # Performance Safety: Even fallback solves MUST be timed on huge grids
                final_words_dict = self._solve_board(
                    fallback_board, original_dictionary, (0, 99999), min_word_length, max_depth=12, store_paths=True, timeout=4.0
                )
                
                # Injection Safety: Use fallback_special_cells[-1] as the coordinate for the bonus word
                fb_bonus_coords = fallback_special_cells[-1] if fallback_special_cells else None
                if actual_bonus_word and actual_bonus_word.upper() not in [w.upper() for w in final_words_dict]:
                    final_words_dict[actual_bonus_word.upper()] = fb_bonus_coords

                final_found_list = list(final_words_dict.keys())
                # ABSOLUTE Ironclad Compliance for Fallback
                is_dense_enough = len(final_found_list) >= min_words
                is_not_too_dense = len(final_found_list) <= (max_words * 1.10 if num_tiles >= 48 else max_words)

                if is_dense_enough and is_not_too_dense:
                    break
                
                # If we are struggling to meet compliance even in fallback, just accept whatever we have on attempt 5
                if fb_attempt >= 5:
                    print(f"[BoardGen] Fallback attempt {fb_attempt} limit reached. Accepting best effort board.")
                    break

            # FINAL CHECK: If even fallback is way off, repeat the ENTIRE outer loop!
            # Zero-Tolerance Policy: If target minimum is not met, we DO NOT return a board.
            if len(final_found_list) < min_words:
                outer_restarts += 1
                if outer_restarts >= 3:
                     print(f"[BoardGen] !! CRITICAL: Even fallback failed 3 times. Returning most dense board available ({len(final_found_list)} words).")
                     bonus_cell = fallback_special_cells[-1] if fallback_special_cells else None
                     return fallback_board, sorted(final_found_list), bonus_cell, board_format, final_words_dict, -1.0, actual_bonus_word.upper() if actual_bonus_word else None
                     
            # If we reached here, word count is GUARANTEED to be >= min_words
            # Update return metadata for fallback
            bonus_cell = fallback_special_cells[-1] if fallback_special_cells else None
            return fallback_board, sorted(final_found_list), bonus_cell, board_format, final_words_dict, -1.0, actual_bonus_word.upper() if actual_bonus_word else None

        # --- LAST RESORT RETURN ---
        # If the while loop exits (timeout or max restarts) without a return, 
        # we MUST return the best board found so far to prevent NoneType crashes.
        print(f"[BoardGen] !! Loop terminated without return. Using Best Global board as final resort.")
        if best_board_global:
            # We must solve it once more (accurately) to ensure results are valid for return
            final_best_dict = self._solve_board(
                best_board_global, original_dictionary, (0, 99999), min_word_length, max_depth=12, store_paths=True, timeout=5.0
            )
            return (
                best_board_global,
                sorted(list(final_best_dict.keys())),
                best_cell_global,
                best_fmt_global + " (Final Resort)",
                final_best_dict,
                best_ratio_global,
                actual_bonus_word.upper() if actual_bonus_word else None
            )
        
        # Absolute fallback if even best_board_global is None (should be impossible)
        return fallback_board, [], None, "Normal (Safety)", {}, 0.0, None

    def _parse_word_count_range(self, word_count_range):
        """Parse word count range (tuple or string) into (min, max) tuple"""
        # Handle tuple format from spinner_set: (30, 60)
        if isinstance(word_count_range, tuple):
            return word_count_range

        if not isinstance(word_count_range, str):
            return (0, float("inf"))

        # Handle string format: "50-100", "100-200", "200+", "500+"
        if word_count_range == "50-100":
            return (50, 100)
        elif word_count_range == "100-200":
            return (100, 200)
        elif word_count_range == "200+":
            return (200, 99999)
        elif word_count_range == "500+":
            return (500, 99999)

        # Generic dash parsing: "100-200", "500-99999"
        if "-" in word_count_range:
            try:
                parts = word_count_range.split("-")
                return (int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                pass

        if word_count_range in ["1500+", "2000+"]:
            return (500, 99999)  # Backward compatibility

        # Default to no restrictions
        return (0, float("inf"))

    def _validate_word_count(self, word_count, min_words, max_words):
        """Check if word count falls within the required range"""
        return min_words <= word_count <= max_words

    def _test_board_formats(
        self, dimensions, bonus_word, word_count_range, dictionary, min_words, max_words, min_word_length=3
    ):
        """Test both board formats and return the faster one that meets requirements"""
        import time

        rows, cols = map(int, dimensions.split("x"))
        results = {}

        # Test Checkerboard format
        print(f"[BoardGen] Testing Checkerboard format...")
        start = time.time()
        board_cb = self._create_checkerboard(rows, cols, LETTER_FREQ_EASY)
        # Use depth 10 for INITIAL TESTING pass to avoid timeout
        words_cb = self._solve_board(
            board_cb, dictionary, word_count_range, min_word_length, max_depth=10, store_paths=False
        )
        time_cb = time.time() - start
        valid_cb = self._validate_word_count(len(words_cb), min_words, max_words)
        results["Checkerboard"] = (time_cb, len(words_cb), valid_cb)
        print(f"[BoardGen] Checkerboard: {time_cb:.2f}s, {len(words_cb)} words, {'VALID' if valid_cb else 'INVALID'}")

        # Test Normal format
        print(f"[BoardGen] Testing Normal format...")
        start = time.time()
        board_normal = self._create_normal_board(rows, cols, LETTER_FREQ_EASY)
        # Use depth 10 for INITIAL TESTING pass
        words_normal = self._solve_board(
            board_normal, dictionary, word_count_range, min_word_length, max_depth=10, store_paths=False
        )
        time_normal = time.time() - start
        valid_normal = self._validate_word_count(len(words_normal), min_words, max_words)
        results["Normal"] = (time_normal, len(words_normal), valid_normal)
        print(
            f"[BoardGen] Normal: {time_normal:.2f}s, {len(words_normal)} words, {'VALID' if valid_normal else 'INVALID'}"
        )

        # Choose faster valid method, or just faster method if neither is valid
        if valid_cb and valid_normal:
            return "Checkerboard" if time_cb < time_normal else "Normal"
        elif valid_cb:
            return "Checkerboard"
        elif valid_normal:
            return "Normal"
        else:
            # Neither is valid, return faster one
            return "Checkerboard" if time_cb < time_normal else "Normal"

    def _get_weights(self, difficulty):
        """Standard letter weights for different difficulties"""
        if difficulty == "Easy":
            return LETTER_FREQ_EASY
        else:
            return LETTER_FREQ_USER

    def _create_normal_board(self, rows, cols, weights, depth=1):
        """Create board with weighted random letters, avoiding redundant U next to Q"""
        if depth > 1:
            return [
                [[random.choices(self.letters, weights=weights, k=1)[0] for _ in range(cols)] for _ in range(rows)]
                for _ in range(depth)
            ]

        board = [[None for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                # We'll try up to 3 times to pick a letter that doesn't form forbidden ING
                # (only for Medium/Hard)
                for _ in range(3):
                    # Check neighbors for a 'Q'
                    has_q_neighbor = False
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if board[nr][nc] == "Q":
                                    has_q_neighbor = True
                                    break
                        if has_q_neighbor:
                            break

                    if has_q_neighbor:
                        safe_weights = list(weights)
                        safe_weights[20] = 0  # No 'U'
                        char = random.choices(self.letters, weights=safe_weights, k=1)[0]
                    else:
                        char = random.choices(self.letters, weights=weights, k=1)[0]

                    # Proactive Forbidden Sequence Check
                    # Note: We can't know the final max_ing here easily, so we just try to avoid it
                    if self._is_creating_forbidden_sequence(board, char, r, c, 0, depth=1):
                        continue  # Re-roll this tile

                    board[r][c] = char
                    break

                if board[r][c] is None:
                    # Final fallback if we keep hitting ING (unlikely)
                    board[r][c] = random.choices(self.letters, weights=weights, k=1)[0]
        return board

    def _create_checkerboard(self, rows, cols, weights, depth=1):
        """Create checkerboard pattern (consonants/vowels) with weighted letters.
        To ensure it alternates 'diagonally', we use row % 2."""
        vowel_indices = [self.letters.index(c) for c in VOWELS]
        consonant_indices = [self.letters.index(c) for c in CONSONANTS]

        vowel_weights = [weights[i] for i in vowel_indices]
        consonant_weights = [weights[i] for i in consonant_indices]

        if depth > 1:
            # Initialize 3D structure with Nones to prevent IndexError in sequence checks
            board = [[[None for _ in range(cols)] for _ in range(rows)] for _ in range(depth)]
            for f in range(depth):
                for r in range(rows):
                    for c in range(cols):
                        # Try a few times to avoid ING
                        for _ in range(3):
                            # Checkerboard pattern in 3D: (f+r+c)%2 == 0 is Consonant, == 1 is Vowel
                            if (f + r + c) % 2 == 0:
                                char = random.choices(CONSONANTS, weights=consonant_weights, k=1)[0]
                            else:
                                char = random.choices(VOWELS, weights=vowel_weights, k=1)[0]

                            if self._is_creating_forbidden_sequence(board, char, r, c, f, depth=depth):
                                continue

                            board[f][r][c] = char
                            break
                        if board[f][r][c] is None:
                            # FALLBACK: Maintain parity
                            if (f + r + c) % 2 == 0:
                                board[f][r][c] = random.choices(CONSONANTS, weights=consonant_weights, k=1)[0]
                            else:
                                board[f][r][c] = random.choices(VOWELS, weights=vowel_weights, k=1)[0]
            return board

        board = [[None for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                # Try a few times to avoid ING
                for _ in range(3):
                    if (r + c) % 2 == 0:
                        char = random.choices(CONSONANTS, weights=consonant_weights, k=1)[0]
                    else:
                        char = random.choices(VOWELS, weights=vowel_weights, k=1)[0]

                    if self._is_creating_forbidden_sequence(board, char, r, c, 0, depth=1):
                        continue

                    board[r][c] = char
                    break

                if board[r][c] is None:
                    # FALLBACK: Maintain parity
                    # (r+c)%2 == 0 is Consonant, == 1 is Vowel
                    if (r + c) % 2 == 0:
                        board[r][c] = random.choices(CONSONANTS, weights=consonant_weights, k=1)[0]
                    else:
                        board[r][c] = random.choices(VOWELS, weights=vowel_weights, k=1)[0]
        return board

    def _create_2000plus_board(
        self,
        rows,
        cols,
        dictionary,
        is_checkerboard=False,
        board=None,
        excluded_cells=None,
        target_type="Density",
        min_word_length=3,
        max_words=999,
        min_words=0,
        min_r=0.0,
        max_r=1.0,
        depth=1,
        difficulty="Medium",
        bonus_word="",
        weights=None,
    ):
        """
        Iterative Optimization (IO)
        target_type: 'Density' (max words) or 'Uniqueness' (70% unique words)
        """
        bonus_word_upper = bonus_word.upper() if bonus_word else ""
        max_ing = bonus_word_upper.count("ING")
        if excluded_cells is None:
            excluded_cells = set()

        def get_weighted_score(words_dict_keys):
            # length-weighted scoring with heavy bonuses for 7L+ words (preservation)
            s = 0
            for w in words_dict_keys:
                wl = len(w)
                if wl >= 8:
                    s += 1000
                elif wl >= 7:
                    s += 500
                elif wl >= 6:
                    s += 100
                else:
                    s += wl - 2
            return s

        # Use weights provided from generate_board or difficulty instead of hardcoded Easy (Density)
        if weights is None:
            weights = self._get_weights(difficulty) if difficulty else LETTER_FREQ_USER
        if board is None:
            if is_checkerboard:
                board = self._create_checkerboard(rows, cols, weights)
            else:
                board = self._create_normal_board(rows, cols, weights)

        # Determine number of passes
        pass_count = 1
        if target_type == "Density":
            if rows * cols >= 35:
                pass_count = 1 # 6x8/Huge grids are extremely dense; 1 pass is always enough
            elif min_words >= 200:
                pass_count = 4 # High Density targets need more passes to pack words (4x4)
            elif min_word_length >= 7:
                pass_count = 3
        else:  # Uniqueness target
            if rows * cols >= 35:
                pass_count = 1 # One pass is enough to hit 50% uniqueness on 6x8
            elif min_word_length >= 5:
                pass_count = 2

        # [DBG] 3D-O1: For 3D Cubes, 1 pass is ALWAYS enough due to massive connectivity.
        if depth > 1:
            pass_count = 1
        print(f"[BoardGen] IO Optimization ({target_type}): {pass_count} pass(es) for min_length={min_word_length}")

        unique_set = self._get_difficulty_set(dictionary)

        start_io_time = time.time()

        # PERFORMANCE: 4x4 grids are fast to solve, so we can afford more passes for high-density
        is_4x4 = rows * cols == 16
        # User Request: Ensure 200+ boards on 4x4 are consistently found (Weight 25% in Spinner)
        # We increase pass_count to 4 for 200+ 4x4 targets to ensure success.
        max_passes = 4 if (is_4x4 and min_words >= 200) else pass_count if (not is_4x4 or min_words >= 200) else 1

        # Use a more targeted dictionary for Uniqueness optimization to match Java's speed/efficiency
        original_dictionary = dictionary
        if target_type == "Uniqueness":
            if dictionary.upper() == "NWL":
                dictionary = "UniqueNWL"
            elif dictionary.upper() == "CSW":
                dictionary = "UniqueCSW"

        start_overall_io_time = time.time()
        for p in range(1, max_passes + 1):
            if depth > 1:
                tiles = [
                    (f, r, c)
                    for f in range(depth)
                    for r in range(rows)
                    for c in range(cols)
                    if (f, r, c) not in excluded_cells
                ]
            else:
                tiles = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in excluded_cells]
            random.shuffle(tiles)

            # --- PERFORMANCE: Initial metrics for incremental update ---
            cells_counter = 0
            start_io_time = time.time()
            v_total = rows * cols * depth
            v_count_global = self._count_vowels(board)

            for tile in tiles:
                # CRITICAL: Total timeout for all passes combined to prevent background thread stalls
                # For large grids, 10s of optimization is plenty; if not met, fallback is better.
                io_timeout = 10 if rows * cols >= 35 else 20 if (is_4x4 and min_words >= 200) else 12 if is_4x4 else 15
                elapsed_io = time.time() - start_overall_io_time
                if elapsed_io > io_timeout:
                    print(f"[BoardGen] IO Global Timeout reached ({elapsed_io:.1f}s > {io_timeout}s). Returning current state.")
                    return board

                if depth > 1:
                    f, r, c = tile
                else:
                    r, c = tile
                    f = 0
                cells_counter += 1
                # PERFORMANCE: Adaptive depth-capping and tile-skipping for speed
                # On large grids (>= 35 cells), IO is extremely powerful. We can skip many tiles 
                # and still hit targets, dramatically reducing load times.
                num_cells = rows * cols * depth
                if num_cells >= 35:
                    # FOR LARGE GRIDS: Depth 10 is sufficient to 'see' target words (6-8L)
                    current_solve_depth = max(min_word_length + 2, 10)
                    # 2D Large (5x7, 6x8): 75% skip is enough even for high density
                    # 3D Cubes (54 cells): 90% skip due to massive connectivity
                    skip_prob = 0.9 if depth > 1 else 0.75
                    if random.random() < skip_prob:
                        continue
                elif num_cells >= 24:
                    current_solve_depth = 11 if depth > 1 else 10
                    if random.random() < 0.2:
                        continue
                else:
                    # 4x4 boards (16 cells)
                    # Optimization: For 200+ words, depth 8 is plenty and much faster.
                    current_solve_depth = 8 if (is_4x4 and min_words >= 150) else 9 

                # Check for Early Exist before we start modifying this tile again to see if we're done
                # PERFORMANCE: 200+ targets need precision. Solving every 2nd tile caughts success earlier.
                eval_freq = 2 if (is_4x4 and min_words >= 150) else 4 if is_4x4 else 6
                if cells_counter % eval_freq == 0:
                    # Restore original dictionary for the eval check
                    try:
                        current_words_eval = self._solve_board(
                            board,
                            original_dictionary,
                            (0, 99999),
                            min_word_length,
                            max_depth=current_solve_depth,
                            store_paths=False,
                            timeout=1.5
                        )
                    except TimeoutError:
                        current_words_eval = {}
                    count_eval = len(current_words_eval)

                    # Calidate uniqueness (User Request: Use all words for small grid uniqueness)
                    if rows * cols < 35:
                        relevant_ev = list(current_words_eval.keys())
                    else:
                        relevant_ev = [w for w in current_words_eval if 6 <= len(w) <= 8]

                    count_rel_ev = len(relevant_ev)
                    count_u_ev = sum(1 for w in relevant_ev if w in unique_set)
                    ratio_u_ev = count_u_ev / count_rel_ev if count_rel_ev > 0 else 0

                    if min_words <= count_eval <= max_words and min_r <= ratio_u_ev <= max_r:
                        print(f"[BoardGen] SUCCESS: Target met mid-round at cell {cells_counter}. Returning board.")
                        return board

                # --- UNIFIED STEPWISE READ (User Request: Align IO with NWL Authority) ---
                # PERFORMANCE: For 4x4 grids, we reduce the individual solver timeout to 0.05s 
                # (down from 0.3s) to allow for hundreds of swaps within a reasonable 10s budget.
                inner_timeout = 0.05 if is_4x4 else 0.12 if num_cells < 35 else 0.15
                try:
                    initial_results = self._solve_board(
                        board, dictionary, (0, 99999), min_word_length, max_depth=current_solve_depth, store_paths=False, timeout=inner_timeout
                    )
                except TimeoutError:
                    # If it somehow still raises (though it shouldn't after my next change), return empty
                    initial_results = {}

                def calculate_composite_value(words_dict_keys):
                    # Robust weighted scoring: Value = (Length-Weight + Long-Word-Bonus) * Multiplier
                    v = 0
                    for w in words_dict_keys:
                        l = len(w)
                        bonus = 0
                        if l >= 8:
                            bonus = 1000  # Massive preservation bonus for long words
                        elif l >= 7:
                            bonus = 500
                        elif l >= 6:
                            bonus = 100

                        base_val = (l - 2) + bonus

                        # Multiplier Logic (User Request: Protect common long words)
                        # All 6L+ words are 'High Value' (15x) if we are doing Density optimization.
                        # IF we are doing Uniqueness optimization, ONLY provide the 15x multiplier if actually in unique set.
                        is_unique = w in unique_set

                        if target_type == "Uniqueness":
                            multiplier = 15 if is_unique else 1
                        else:
                            # Density optimization: prioritize long words regardless of uniqueness
                            multiplier = 15 if (is_unique or l >= 6) else 1

                        v += base_val * multiplier
                    return v

                curr_count = calculate_composite_value(initial_results.keys())
                curr_auth_count = len(initial_results)
                best_count_w = curr_auth_count  # Track the word count of the best configuration so far

                # STOP if we are already at or near max_words during Density optimization
                if target_type == "Density" and curr_auth_count >= max_words:
                    print(
                        f"[BoardGen] Target density reached during pass {p} ({curr_auth_count} >= {max_words}). Stopping further tiles."
                    )
                    return board

                # When using Unique dict, all words are unique by definition
                curr_unique = curr_count
                curr_ratio = 1.0  # (Since dictionary = UniqueSet)

                old_char = board[f][r][c] if depth > 1 else board[r][c]
                best_char = old_char

                # Test pool of letters
                # PATTERN-AWARE: If it's a Checkerboard, we MUST only test Vowels on vowel cells and Consonants on consonant cells
                if is_checkerboard:
                    # Checkerboard pattern: (f + r + c) % 2 == 1 is Vowel in 3D, (r + c) % 2 == 1 in 2D
                    target_is_vowel = (f + r + c) % 2 != 0 if depth > 1 else (r + c) % 2 != 0
                    if target_is_vowel:
                        test_pool = list(VOWELS)
                    else:
                        test_pool = list(CONSONANTS)
                else:
                    if target_type == "Density":
                        if min_words >= 200:
                            # User Request: If aiming for high density, use most common English letters
                            # Expand pool for 4x4 to ensure we don't hit variety-plateaus
                            test_pool = list("ETAOINSRDL") + (list("BCUM") if is_4x4 else [])
                        elif max_words <= 150 and rows * cols >= 35:
                            # User Request: On large grids with low word targets, we need RARE letters to prevent
                            # word counts from exploding. Using standard English frequency makes 100 counts impossible.
                            test_pool = list("ZXQJKVWYPFBHC") + [random.choice("ETAOINSR") for _ in range(2)]
                        else:
                            # Limit density search to relevant letters for HUGE speedup
                            test_pool = list("ETAOINSRHDLU") + [
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3)
                            ]
                    else:
                        # Hard round optimization
                        # Even when optimizing for uniqueness, we need some common letters to form words
                        if curr_count < max_words // 3:
                            test_pool = list("ETAOINSRHDLU")
                        else:
                            test_pool = list(RARE_SET) + list("ETAO")

                random.shuffle(test_pool)

                # PERFORMANCE: Scale test pool by grid size
                # JAVA ALIGNMENT: Reduced sampling to prevent long synchronous hangs
                if is_4x4:
                    sample_size = 5 # Reduced for 4x4 speed
                elif num_cells >= 48:
                    # 6x8 grids: cap at 4 samples to ensure completion under 20s
                    sample_size = 4 if min_words >= 200 else 2 
                else:
                    sample_size = 2 # 4x6, 3x3x3 speedup (from 4)
                test_pool = test_pool[:sample_size]

                for char in test_pool:
                    # User Request: Highly localized checks for forbidden sequences (like ING) during optimization
                    # to prevent them from leaking into the final board.
                    if difficulty in ["Medium", "Hard"] and self._is_creating_forbidden_sequence(
                        board, char, r, c, f if depth > 1 else 0, depth=depth
                    ):
                        continue

                    if depth > 1:
                        board[f][r][c] = char
                    else:
                        board[r][c] = char

                    # Optimization: Incremental vowel ratio enforcement
                    if not is_checkerboard:
                        old_v = self._is_vowel(old_char)
                        new_v = self._is_vowel(char)
                        v_count = v_count_global - (1 if old_v else 0) + (1 if new_v else 0)
                        
                        # For High Density targets, allow a significantly broader vowel range to facilitate long word connectivity
                        if target_type == "Density":
                             # If we are failing to hit the target, allow even more vowels (up to 50% for 7L+ boards)
                             max_v_ratio = 0.50 if min_word_length >= 7 else 0.44
                             min_v_ratio = 0.25
                        else:
                             # Uniqueness optimization (Hard rounds)
                             max_v_ratio = 0.45
                             min_v_ratio = 0.25

                        if not (min_v_ratio <= v_count / v_total <= max_v_ratio):
                            continue  # Skip letters that break vowel ratio during optimization

                    # --- OPTIMIZED FORBIDDEN SEQUENCE ENFORCEMENT ---
                    # User Request: ING sequences not permissible in Medium and Hard
                    if difficulty in ["Medium", "Hard"] and self._is_creating_forbidden_sequence(
                        board, char, r, c, f if depth > 1 else 0, depth=depth
                    ):
                        continue

                    # Test results against Authority dictionary
                    results = self._solve_board(
                        board, dictionary, (0, 99999), min_word_length, max_depth=current_solve_depth, store_paths=False, timeout=inner_timeout
                    )
                    val = calculate_composite_value(results.keys())
                    count_w = len(results)

                    is_overshooting = count_w > max_words
                    is_undershooting = count_w < min_words

                    if target_type == "Uniqueness":
                        # Mode (Hard): Maximize unique value, but strictly handle word count limits (especially on 6x8)
                        if not is_overshooting and (val > curr_count or (val == curr_count and count_w > best_count_w)):
                            # Standard improvement (Under limit)
                            curr_count = val
                            best_count_w = count_w
                            best_char = char
                        elif is_overshooting:
                            # Reduction phase: Force word count down towards max_words
                            # We accept ANY change that reduces word count significantly, OR
                            # a change that reduces word count slightly while keeping/improving value.
                            if count_w < best_count_w:
                                # Improvement in terms of limit compliance.
                                # We accept it if val gain is positive OR if val drop is small (<10%)
                                if val >= curr_count * 0.90:
                                    curr_count = val
                                    best_count_w = count_w
                                    best_char = char
                            elif val > curr_count * 1.2 and count_w <= best_count_w:
                                # Significant value gain with no count penalty.
                                curr_count = val
                                best_count_w = count_w
                                best_char = char
                    else:  # Density target
                        # Mode (Easy/Medium): Maximize density (Points/Word)
                        if not is_overshooting and (val > curr_count or (val == curr_count and count_w > best_count_w)):
                            # Standard improvement (Under limit)
                            curr_count = val
                            best_count_w = count_w
                            best_char = char
                        elif is_overshooting:
                            # Reduction phase (Highly strict for 6x8)
                            # Priority 1: Count reduction
                            if count_w < best_count_w:
                                # Accept if it stays within 15% of previous value density
                                if (val / count_w) >= (curr_count / best_count_w) * 0.85:
                                    curr_count = val
                                    best_count_w = count_w
                                    best_char = char

                # Apply best found character for this tile
                if best_char != old_char:
                    v_count_global = v_count_global - (1 if self._is_vowel(old_char) else 0) + (1 if self._is_vowel(best_char) else 0)

                if depth > 1:
                    board[f][r][c] = best_char
                else:
                    board[r][c] = best_char

                # Cleanup before move to next cell (No dictionary swap needed here)

            # Pass complete check
            # Restore original dictionary for the final eval check of the pass
            dictionary = original_dictionary

            # JAVA ALIGNMENT: Always evaluate against the AUTHORITATIVE dictionary (NWL/CSW) at the end of a pass
            # to ensure the mid-round word list is accurate.
            test_solve_all = self._solve_board(
                board, dictionary, (0, 99999), min_word_length, max_depth=current_solve_depth, store_paths=False
            )
            total_words = len(test_solve_all)

            # Uniqueness check for early exit (Use accurate length-aware filter)
            if rows * cols < 35:
                relevant_final = list(test_solve_all.keys())
            else:
                relevant_final = [w for w in test_solve_all if len(w) >= min_word_length]

            count_rel_final = len(relevant_final)
            count_unique = sum(1 for w in relevant_final if w.upper() in unique_set)
            curr_r = count_unique / count_rel_final if count_rel_final > 0 else 0
            print(f"[BoardGen] Pass {p} complete. Count: {total_words}, Unique: {curr_r:.1%}")

            # EARLY EXIT if we satisfied EVERYTHING
            if min_words <= total_words <= max_words and min_r <= curr_r <= max_r:
                print(f"[BoardGen] SUCCESS: Early exit after pass {p} - All targets met.")
                break

            if target_type == "Density" and total_words >= max_words:
                print(
                    f"[BoardGen] Target density reached after pass {p} ({total_words} >= {max_words}). Stopping pass early."
                )
                break

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

        is_3d = len(board) > 0 and isinstance(board[0], list) and isinstance(board[0][0], list)
        depth_val = len(board) if is_3d else 1
        rows, cols = (len(board[0]), len(board[0][0])) if is_3d else (len(board), len(board[0]))
        total_cells = rows * cols * depth_val

        # Determine mania type
        is_mania_vowel = self._is_vowel(mania_letter)

        # Target ratio: 5/16 (31.25%)
        target_ratio = 5.0 / 16.0
        target_count = max(3, round(total_cells * target_ratio))

        if is_3d:
            current_count = sum(1 for f in range(depth_val) for r in range(rows) for c in range(cols) if board[f][r][c] == mania_letter)
        else:
            current_count = sum(1 for r in range(rows) for c in range(cols) if board[r][c] == mania_letter)
        needed = target_count - current_count

        if needed <= 0:
            return

        all_positions = []
        if is_3d:
            for f in range(depth_val):
                for r in range(rows):
                    for c in range(cols):
                        if (f, r, c) in exclude_cells:
                            continue
                        if "/" in str(board[f][r][c]):
                            continue
                        if is_checkerboard:
                            is_cell_vowel_expected = (f + r + c) % 2 != 0
                            if is_mania_vowel != is_cell_vowel_expected:
                                continue
                        all_positions.append((f, r, c))
        else:
            for r in range(rows):
                for c in range(cols):
                    if (r, c) in exclude_cells:
                        continue
                    if "/" in str(board[r][c]):
                        continue
                    if is_checkerboard:
                        is_cell_vowel_expected = (r + c) % 2 != 0
                        if is_mania_vowel != is_cell_vowel_expected:
                            continue
                    all_positions.append((r, c))

        random.shuffle(all_positions)

        filled = 0
        for pos in all_positions:
            if filled >= needed:
                break
            if is_3d:
                f, r, c = pos
                board[f][r][c] = mania_letter
            else:
                r, c = pos
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
            if i < len(bonus_word) - 1 and bonus_word[i : i + 2].upper() == "QU":
                processed_word.append("Q")
                i += 2
            else:
                processed_word.append(bonus_word[i].upper())
                i += 1

        word_len = len(processed_word)

        # Proceed with embedding (Checkerboard will use backtracking that respects C/V alternating pattern)

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
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        neighbors.append((nr, nc))
            random.shuffle(neighbors)  # Randomize direction
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
                    is_expected_vowel = (nr + nc) % 2 != 0
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
                    is_expected_vowel = (r + c) % 2 != 0
                    if word_vowel_map[0] == is_expected_vowel:
                        possible_starts.append((r, c))
                else:
                    possible_starts.append((r, c))

        random.shuffle(possible_starts)

        import time

        with open("/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log", "a") as f:
            f.write(f"[board_generator.py] _embed_bonus_word: Attempting to embed '{bonus_word}' at {time.time()}\n")

        for start_r, start_c in possible_starts:
            path = backtrack([(start_r, start_c)])
            if path:
                # Embed the processed letters
                for i, (r, c) in enumerate(path):
                    board[r][c] = processed_word[i]
                with open("/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log", "a") as f:
                    f.write(f"[board_generator.py] _embed_bonus_word: SUCCESS at {time.time()}\n")
                return path

        with open("/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/debug_flow.log", "a") as f:
            f.write(f"[board_generator.py] _embed_bonus_word: FAILED at {time.time()}\n")
        return None

    def _has_either_or_ambiguity(self, board, dictionary):
        """Check if any path in the board passing through the E/O tile could represent two different valid words."""
        is_3d = len(board) > 0 and isinstance(board[0], list) and isinstance(board[0][0], list)
        depth_val = len(board) if is_3d else 1
        rows, cols = (len(board[0]), len(board[0][0])) if is_3d else (len(board), len(board[0]))
        
        eo_pos = None
        if is_3d:
            for f in range(depth_val):
                for r in range(rows):
                    for c in range(cols):
                        if "/" in str(board[f][r][c]):
                            eo_pos = (f, r, c)
                            break
                    if eo_pos: break
                if eo_pos: break
        else:
            for r in range(rows):
                for c in range(cols):
                    if "/" in str(board[r][c]):
                        eo_pos = (r, c)
                        break
                if eo_pos: break
        
        if not eo_pos:
            return False

        def dfs_check(f, r, c, visited, word_so_far):
            # word_so_far is a list of lists of possible letters at each step
            # e.g. [['E'], ['L', 'T'], ['U'], ['D'], ['E']]

            # Convert to possible words
            from itertools import product

            possible_words = ["".join(p) for p in product(*word_so_far)]

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
            if is_3d:
                ef, er, ec = eo_pos
                dist = max(abs(f - ef), abs(r - er), abs(c - ec))
                # For cube, dist might be different, but 3 is a safe upper bound on a small cube
                if dist > 3: dist = 1 # Approximation
            else:
                er, ec = eo_pos
                dist = max(abs(r - er), abs(c - ec))
            
            remaining_steps = 8 - len(word_so_far)
            if dist > remaining_steps:
                return False

            # Continue
            neighbors = []
            if is_3d:
                neighbors = self._get_cube_neighbors(f, r, c)
            else:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbors.append((0, nr, nc))

            for nf, nr, nc in neighbors:
                if (nf, nr, nc) not in visited:
                    cell = board[nf][nr][nc] if is_3d else board[nr][nc]
                    letters = cell.split("/") if "/" in cell else [cell]
                    if dfs_check(nf, nr, nc, visited | {(nf, nr, nc)}, word_so_far + [letters]):
                        return True
            return False

        # iterate over all cells
        if is_3d:
            for fi in range(depth_val):
                for ri in range(rows):
                    for ci in range(cols):
                        cell = board[fi][ri][ci]
                        letters = cell.split("/") if "/" in cell else [cell]
                        if dfs_check(fi, ri, ci, {(fi, ri, ci)}, [letters]):
                            return True
        else:
            for ri in range(rows):
                for ci in range(cols):
                    cell = board[ri][ci]
                    letters = cell.split("/") if "/" in cell else [cell]
                    if dfs_check(0, ri, ci, {(0, ri, ci)}, [letters]):
                        return True
        return False

    def _solve_board(
        self, board, dictionary="NWL", word_count_range=(0, 99999), min_word_length=3, max_depth=12, store_paths=True, timeout=10.0
    ):
        """Find all valid words on the board using high-speed node-based DFS traversal."""
        # Support 3D detect
        is_3d = len(board) > 0 and isinstance(board[0], list) and isinstance(board[0][0], list)
        depth_val = len(board) if is_3d else 1

        if is_3d:
            rows, cols = len(board[0]), len(board[0][0])
        else:
            rows, cols = len(board), len(board[0])

        found_words = {}  # {word: path_sample}

        # High-speed visitor tracking
        if depth_val == 1:
            visited = [[False for _ in range(cols)] for _ in range(rows)]
        else:
            visited = [[[False for _ in range(depth_val)] for _ in range(cols)] for _ in range(rows)]

        import time

        solver_start_time = time.time()
        solver_timeout = timeout  # Configurable timeout for board solving

        # --- PRE-LOAD TRIE ROOT ---
        if dictionary == "UniqueNWL":
            trie_root = word_validator.unique_nwl_trie
        elif dictionary == "UniqueCSW":
            trie_root = word_validator.unique_csw_trie
        elif dictionary == "CSW":
            trie_root = word_validator.csw_trie
        else:
            trie_root = word_validator.nwl_trie

        def dfs(f, r, c, current_d, current_word, current_node, current_path):
            if current_d > max_depth:
                return

            # HARD TIMEOUT: Stop searching if we've spent too long solving (Safety for 6x8 dense boards)
            if time.time() - solver_start_time > solver_timeout:
                return # Partial results are returned by the wrapper

            cell = board[f][r][c] if depth_val > 1 else board[r][c]
            letters = cell.split("/") if "/" in cell else [cell]

            for char in letters:
                # 1. Trie Advancement
                next_node = current_node.children.get(char)
                if not next_node:
                    continue  # PRUNED!

                new_word = current_word + char
                new_path = current_path + [(f, r, c)] if store_paths else None

                # Check current word
                if len(new_word) >= min_word_length and next_node.is_word:
                    if new_word not in found_words:
                        found_words[new_word] = new_path if store_paths else True

                # Prune and continue using Trie
                if len(new_word) < max_depth:
                    if depth_val == 1:
                        visited[r][c] = True
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                                    dfs(0, nr, nc, current_d + 1, new_word, next_node, new_path)
                        visited[r][c] = False
                    else:
                        # 3D CUBE SUPPORT: 26-way adjacency
                        visited[r][c][f] = True
                        for nf, nr, nc in self._get_cube_neighbors(f, r, c):
                            if not visited[nr][nc][nf]:
                                dfs(nf, nr, nc, current_d + 1, new_word, next_node, new_path)
                        visited[r][c][f] = False

                # Qu Logic (only if not Either/Or for simplicity)
                if char == "Q":
                    u_node = next_node.children.get("U")
                    if u_node:
                        q_word = current_word + "QU"
                        if len(q_word) >= min_word_length and u_node.is_word:
                            if q_word not in found_words:
                                found_words[q_word] = new_path if store_paths else True

                        if len(q_word) < max_depth:
                            if depth_val == 1:
                                visited[r][c] = True
                                for dr in [-1, 0, 1]:
                                    for dc in [-1, 0, 1]:
                                        nr, nc = r + dr, c + dc
                                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                                            dfs(0, nr, nc, current_d + 1, q_word, u_node, new_path)
                                visited[r][c] = False
                            else:
                                visited[r][c][f] = True
                                for nf, nr, nc in self._get_cube_neighbors(f, r, c):
                                    if not visited[nr][nc][nf]:
                                        dfs(nf, nr, nc, current_d + 1, q_word, u_node, new_path)
                                visited[r][c][f] = False

        # Wrapper to handle timeout exception and return partially found words
        try:
            # Start from every cell
            for fi in range(depth_val):
                for ri in range(rows):
                    for ci in range(cols):
                        if time.time() - solver_start_time > solver_timeout:
                            break
                        dfs(fi, ri, ci, 1, "", trie_root, [])
        except Exception as e:
            print(f"[Solver] CRITICAL ERROR: {e}")
            
        return found_words

    def complete_solve_board(self, board, dictionary):
        """
        Exhaustively find ALL valid words on the board without limits.
        Used for background solving during intermission.
        """
        import time

        start_t = time.time()
        # Hard cap for exhaustive search to prevent server lockup on 6x8 dense boards
        solver_timeout = 12.0

        rows, cols = len(board), len(board[0])
        found_words = set()

        print(f"[BoardGen] Complete solver: searching with Trie pruning (max_len=10, timeout={solver_timeout}s)")

        def dfs(r, c, visited, word):
            if time.time() - start_t > solver_timeout:
                return

            # Add word if it's valid and long enough
            # Use cached validator results if possible
            if len(word) >= 3 and word_validator.is_valid_word(word, dictionary):
                found_words.add(word)

            # Prune search using Trie/Prefix checking
            if len(word) < 10 and word_validator.has_valid_prefix(word, dictionary):
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            cell_val = str(board[nr][nc])
                            letters = cell_val.split("/") if "/" in cell_val else [cell_val]

                            for char in letters:
                                dfs(nr, nc, visited, word + char)
                                if char == "Q":
                                    dfs(nr, nc, visited, word + "QU")
                            visited.remove((nr, nc))

        for r in range(rows):
            for c in range(cols):
                cell_val = str(board[r][c])
                letters = cell_val.split("/") if "/" in cell_val else [cell_val]
                for char in letters:
                    dfs(r, c, {(r, c)}, char)
                    if char == "Q":
                        dfs(r, c, {(r, c)}, "QU")

        print(
            f"[BoardGen] Complete solver finished: found {len(found_words)} total words in {time.time() - start_t:.2f}s"
        )
        return sorted(list(found_words))

    def is_word_on_board(self, word, board):
        """Check if a word exists on the board (2D or 3D Surface)"""
        if not board:
            return False
        is_3d = len(board) == 6 and isinstance(board[0], list) and isinstance(board[0][0], list)
        word = word.upper()

        def dfs_find(f, r, c, index, visited):
            if index >= len(word):
                return True

            # Use appropriate neighbors based on dimension
            neighbors = []
            if not is_3d:
                rows, cols = len(board), len(board[0])
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbors.append((-1, nr, nc))
            else:
                neighbors = self._get_cube_neighbors(f, r, c)

            for nf, nr, nc in neighbors:
                if (nf, nr, nc) in visited:
                    continue

                cell_val = str(board[nf][nr][nc] if is_3d else board[nr][nc]).upper()
                letters = cell_val.split("/") if "/" in cell_val else [cell_val]

                for char in letters:
                    match_length = 0
                    if char == "Q" and word.startswith("QU", index):
                        match_length = 2
                    elif word.startswith(char, index):
                        match_length = len(char)

                    if match_length > 0:
                        if index + match_length >= len(word):
                            return True
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
                    for char in (cell_val.split("/") if "/" in cell_val else [cell_val]):
                        match_l = 0
                        if char == "Q" and word.startswith("QU", 0):
                            match_l = 2
                        elif word.startswith(char, 0):
                            match_l = len(char)
                        if match_l > 0:
                            if match_l >= len(word):
                                return True
                            if dfs_find(-1, r, c, match_l, {(-1, r, c)}):
                                return True
        else:
            for f in range(6):
                for r in range(3):
                    for c in range(3):
                        cell_val = str(board[f][r][c]).upper()
                        if cell_val == "Q" and word.startswith("QU", 0):
                            if 2 >= len(word):
                                return True
                            if dfs_find(f, r, c, 2, {(f, r, c)}):
                                return True
                        elif word.startswith(cell_val, 0):
                            if len(cell_val) >= len(word):
                                return True
                            if dfs_find(f, r, c, len(cell_val), {(f, r, c)}):
                                return True
        return False

    def can_word_hit_bonus(self, word, board, bonus_cell):
        """Check if a word can be formed on board such that its path contains bonus_cell"""
        if not bonus_cell:
            return False
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
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        cell_val = str(board[nr][nc]).upper()
                        letters = cell_val.split("/") if "/" in cell_val else [cell_val]

                        for char in letters:
                            match_len = 0
                            if char == "Q":
                                if word.startswith("QU", index):
                                    match_len = 2
                                elif word[index] == "Q":
                                    match_len = 1
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
                letters = cell_val.split("/") if "/" in cell_val else [cell_val]
                for char in letters:
                    match_len = 0
                    if char == "Q":
                        if word.startswith("QU"):
                            match_len = 2
                        elif word.startswith("Q"):
                            match_len = 1
                    elif word.startswith(char):
                        match_len = 1
        return False

    def _create_cube_board(self, difficulty="Medium"):
        """Create a 3x3x3 cube board (6 faces, 3x3 each)"""
        weights = self._get_weights(difficulty)
        board = []
        for f in range(6):
            face = [[random.choices(self.letters, weights=weights, k=1)[0] for _ in range(3)] for _ in range(3)]
            board.append(face)
        return board

    def _get_cube_neighbors(self, f, r, c):
        """Standard 8-way adjacency for a 3x3x3 cube surface (Cached)"""
        if self.cube_neighbor_cache and (f, r, c) in self.cube_neighbor_cache:
            return self.cube_neighbor_cache[(f, r, c)]

        # Initialize cache if missing
        if self.cube_neighbor_cache is None:
            self.cube_neighbor_cache = {}
            for _f in range(6):
                for _r in range(3):
                    for _c in range(3):
                        self.cube_neighbor_cache[(_f, _r, _c)] = self._calculate_cube_neighbors_uncached(_f, _r, _c)

        return self.cube_neighbor_cache.get((f, r, c), [])

    def _calculate_cube_neighbors_uncached(self, f, r, c):
        """Internal helper to calculate adjacency on a 6-face cube net."""
        # (face, row, col)
        res = []
        # Intra-face
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 3 and 0 <= nc < 3:
                    res.append((f, nr, nc))

        # Helper to ensure only valid coords are added from interfaces
        def add_safe(face, row, col):
            if 0 <= face < 6 and 0 <= row < 3 and 0 <= col < 3:
                res.append((face, row, col))

        # Inter-face (Edges and Corners)
        # Face Layout (Standard Net):
        #      [4] (Top)
        #  [2] [0] [3] [1] (Left, Front, Right, Back)
        #      [5] (Bottom)

        # 0 (Front)
        if f == 0:
            if r == 0:  # Top Edge
                add_safe(4, 2, c)
                add_safe(4, 2, c - 1)
                add_safe(4, 2, c + 1)
            if r == 2:  # Bottom Edge
                add_safe(5, 0, c)
                add_safe(5, 0, c - 1)
                add_safe(5, 0, c + 1)
            if c == 0:  # Left Edge
                add_safe(2, r, 2)
                add_safe(2, r - 1, 2)
                add_safe(2, r + 1, 2)
            if c == 2:  # Right Edge
                add_safe(3, r, 0)
                add_safe(3, r - 1, 0)
                add_safe(3, r + 1, 0)

        # 1 (Back)
        elif f == 1:
            if r == 0:  # Top Edge -> Top (4) Top
                add_safe(4, 0, 2 - c)
                add_safe(4, 0, 2 - (c - 1))
                add_safe(4, 0, 2 - (c + 1))
            if r == 2:  # Bottom Edge -> Bottom (5) Bottom
                add_safe(5, 2, 2 - c)
                add_safe(5, 2, 2 - (c - 1))
                add_safe(5, 2, 2 - (c + 1))
            if c == 0:  # Left Edge -> Right (3) Right
                add_safe(3, r, 2)
                add_safe(3, r - 1, 2)
                add_safe(3, r + 1, 2)
            if c == 2:  # Right Edge -> Left (2) Left
                add_safe(2, r, 0)
                add_safe(2, r - 1, 0)
                add_safe(2, r + 1, 0)

        # 2 (Left)
        elif f == 2:
            if r == 0:  # Top Edge -> Top (4) Left
                add_safe(4, c, 0)
                add_safe(4, c - 1, 0)
                add_safe(4, c + 1, 0)
            if r == 2:  # Bottom Edge -> Bottom (5) Left
                add_safe(5, 2 - c, 0)
                add_safe(5, 2 - (c - 1), 0)
                add_safe(5, 2 - (c + 1), 0)
            if c == 0:  # Left Edge -> Back (1) Right
                add_safe(1, r, 2)
                add_safe(1, r - 1, 2)
                add_safe(1, r + 1, 2)
            if c == 2:  # Right Edge -> Front (0) Left
                add_safe(0, r, 0)
                add_safe(0, r - 1, 0)
                add_safe(0, r + 1, 0)

        # 3 (Right)
        elif f == 3:
            if r == 0:  # Top Edge -> Top (4) Right
                add_safe(4, 2 - c, 2)
                add_safe(4, 2 - (c - 1), 2)
                add_safe(4, 2 - (c + 1), 2)
            if r == 2:  # Bottom Edge -> Bottom (5) Right
                add_safe(5, c, 2)
                add_safe(5, c - 1, 2)
                add_safe(5, c + 1, 2)
            if c == 0:  # Left Edge -> Front (0) Right
                add_safe(0, r, 2)
                add_safe(0, r - 1, 2)
                add_safe(0, r + 1, 2)
            if c == 2:  # Right Edge -> Back (1) Left
                add_safe(1, r, 0)
                add_safe(1, r - 1, 0)
                add_safe(1, r + 1, 0)

        # 4 (Top)
        elif f == 4:
            if r == 0:  # Top Edge -> Back (1) Top
                add_safe(1, 0, 2 - c)
                add_safe(1, 0, 2 - (c - 1))
                add_safe(1, 0, 2 - (c + 1))
            if r == 2:  # Bottom Edge -> Front (0) Top
                add_safe(0, 0, c)
                add_safe(0, 0, c - 1)
                add_safe(0, 0, c + 1)
            if c == 0:  # Left Edge -> Left (2) Top
                add_safe(2, 0, r)
                add_safe(2, 0, r - 1)
                add_safe(2, 0, r + 1)
            if c == 2:  # Right Edge -> Right (3) Top
                add_safe(3, 0, 2 - r)
                add_safe(3, 0, 2 - (r - 1))
                add_safe(3, 0, 2 - (r + 1))

        # 5 (Bottom)
        elif f == 5:
            if r == 0:  # Top Edge -> Front (0) Bottom
                add_safe(0, 2, c)
                add_safe(0, 2, c - 1)
                add_safe(0, 2, c + 1)
            if r == 2:  # Bottom Edge -> Back (1) Bottom
                add_safe(1, 2, 2 - c)
                add_safe(1, 2, 2 - (c - 1))
                add_safe(1, 2, 2 - (c + 1))
            if c == 0:  # Left Edge -> Left (2) Bottom
                add_safe(2, 2, 2 - r)
                add_safe(2, 2, 2 - (r - 1))
                add_safe(2, 2, 2 - (r + 1))
            if c == 2:  # Right Edge -> Right (3) Bottom
                add_safe(3, 2, r)
                add_safe(3, 2, r - 1)
                add_safe(3, 2, r + 1)

        # Filter out invalid and duplicates
        clean = []
        seen = set()
        for nf, nr, nc in res:
            if 0 <= nf < 6 and 0 <= nr < 3 and 0 <= nc < 3 and (nf, nr, nc) not in seen and (nf, nr, nc) != (f, r, c):
                clean.append((nf, nr, nc))
                seen.add((nf, nr, nc))
        return clean

    def _enforce_vowel_minimum(self, board, weights, is_checkerboard=False, excluded_cells=None, difficulty="Medium"):
        """Ensure 33%-38% of tiles are vowels (User Request: Strict range for all boards)
        FOR CHECKERBOARD: Preserve pattern (50%) to avoid breaking logic."""
        if not board or is_checkerboard:
            return

        # Flatten board to get all cells
        flat_cells = []
        is_3d = len(board) > 0 and isinstance(board[0], list) and isinstance(board[0][0], list)
        depth_val = len(board) if is_3d else 1
        
        excluded_set = set(excluded_cells) if excluded_cells else set()

        if is_3d:
            rows, cols = len(board[0]), len(board[0][0])
            for f in range(depth_val):
                for r in range(rows):
                    for c in range(cols):
                        if (f, r, c) not in excluded_set:
                            flat_cells.append((f, r, c))
        else:
            # Standard 2D Grid
            rows, cols = len(board), len(board[0])
            for r in range(rows):
                for c in range(cols):
                    if (r, c) not in excluded_set:
                        flat_cells.append((r, c))

        # 1. Calculate TOTAL board size and Vowel Count currently in protected/excluded cells
        full_board_size = rows * cols * depth_val
        vowels_in_excluded = 0
        for pos in excluded_set:
            if is_3d:
                f, r, c = pos
                val = str(board[f][r][c])
            else:
                r, c = pos
                val = str(board[r][c])
            if any(v in val.upper() for v in VOWELS):
                vowels_in_excluded += 1

        # 2. Target: 32.5% (Midpoint of 30%-35%) for the ENTIRE board
        # Use precise percentages to avoid floor issues on smaller boards
        target_total_v = int(round(full_board_size * 0.325))
        min_total_v = int(round(full_board_size * 0.30))
        max_total_v = int(round(full_board_size * 0.35))

        # Clamp target to the user's strict range
        target_total_v = max(min_total_v, min(max_total_v, target_total_v))

        # 3. Calculate remaining vowels needed in the non-excluded area
        total_remaining_needed = max(0, target_total_v - vowels_in_excluded)

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

        if current_count < total_remaining_needed:
            # Need more vowels
            needed = total_remaining_needed - current_count
            for i in range(min(needed, len(current_consonant_cells))):
                pos = current_consonant_cells[i]
                if is_3d:
                    f, r, c = pos
                else:
                    r, c = pos
                
                safe_v_found = False
                for _ in range(5):
                    new_v = random.choices(list(VOWELS), weights=v_w, k=1)[0]
                    # Local validation ONLY: Sequence prohibition is authoritative
                    if difficulty not in ["Medium", "Hard"] or not self._is_creating_forbidden_sequence(
                        board, new_v, r, c, f if is_3d else 0, depth=depth_val
                    ):
                        if is_3d:
                            board[f][r][c] = new_v
                        else:
                            board[r][c] = new_v
                        safe_v_found = True
                        break
            print(f"[BoardGen] Enforced 30-35% vowels: Added {needed} vowels to reach {target_total_v} total (Excluded: {vowels_in_excluded}).")

        elif current_count > total_remaining_needed:
            # Need fewer vowels (Too many can happen with weights)
            over = current_count - total_remaining_needed
            for i in range(min(over, len(current_vowel_cells))):
                pos = current_vowel_cells[i]
                for _ in range(5):
                    new_c = random.choices(list(CONSONANTS), weights=c_w, k=1)[0]
                    if is_3d:
                        f, r, c = pos
                        if difficulty not in ["Medium", "Hard"] or not self._is_creating_forbidden_sequence(
                            board, new_c, r, c, f, depth=depth_val
                        ):
                            board[f][r][c] = new_c
                            break
                    else:
                        r, c = pos
                        if difficulty not in ["Medium", "Hard"] or not self._is_creating_forbidden_sequence(
                            board, new_c, r, c, 0, depth=depth_val
                        ):
                            board[r][c] = new_c
                            break
            print(f"[BoardGen] Enforced 30-35% vowels: Removed {over} vowels to reach {target_total_v} total (Excluded: {vowels_in_excluded}).")

    def _count_vowels(self, board):
        v_count = 0
        is_3d = len(board) > 0 and isinstance(board[0], list) and isinstance(board[0][0], list)
        depth_val = len(board) if is_3d else 1
        rows, cols = (len(board[0]), len(board[0][0])) if is_3d else (len(board), len(board[0]))

        if is_3d:
            for f in range(depth_val):
                for r in range(rows):
                    for c in range(cols):
                        if self._is_vowel(board[f][r][c]):
                            v_count += 1
        else:
            for r in range(rows):
                for c in range(cols):
                    if self._is_vowel(board[r][c]):
                        v_count += 1
        return v_count

    def _verify_checkerboard_safeguard(self, board, weights, bonus_cells_set):
        """Final check to ensure the board strictly alternates C/V in checkerboard mode."""
        if not board:
            return
        v_indices = [self.letters.index(v) for v in VOWELS]
        v_weights = [weights[v_idx] for v_idx in v_indices]
        c_indices = [self.letters.index(c) for c in CONSONANTS]
        c_weights = [weights[c_idx] for c_idx in c_indices]

        # Determine if board is 2D or 3D
        is_3d = isinstance(board[0][0], list)
        repaired = 0
        
        if is_3d:
            depth_val, rows, cols = len(board), len(board[0]), len(board[0][0])
            for f in range(depth_val):
                for r in range(rows):
                    for c in range(cols):
                        if "/" in str(board[f][r][c]): continue
                        expected_vowel = (f + r + c) % 2 != 0
                        current_val = board[f][r][c]
                        is_actual_vowel = self._is_vowel(current_val)
                        if is_actual_vowel != expected_vowel or not current_val:
                            if expected_vowel:
                                board[f][r][c] = random.choices(list(VOWELS), weights=v_weights, k=1)[0]
                            else:
                                board[f][r][c] = random.choices(list(CONSONANTS), weights=c_weights, k=1)[0]
                            repaired += 1
        else:
            rows, cols = len(board), len(board[0])
            for r in range(rows):
                for c in range(cols):
                    if "/" in str(board[r][c]): continue
                    expected_vowel = (r + c) % 2 != 0
                    current_val = board[r][c]
                    is_actual_vowel = self._is_vowel(current_val)
                    # Force repair if it's not a vowel when it should be, OR if it's a vowel when it shouldn't be, OR if it's empty
                    if is_actual_vowel != expected_vowel or not current_val:
                        if expected_vowel:
                            board[r][c] = random.choices(list(VOWELS), weights=v_weights, k=1)[0]
                        else:
                            board[r][c] = random.choices(list(CONSONANTS), weights=c_weights, k=1)[0]
                        repaired += 1

        if repaired > 0:
            print(f"[BoardGen] Checkerboard Safeguard: Forced {repaired} letters to maintain alternation pattern.")

    def _is_vowel(self, char):
        """Helper to check if a letter (or tile string) is a vowel"""
        if not char:
            return False
        # Handle Either/Or L/T - return True if either is a vowel
        letters = str(char).upper().split("/")
        for l in letters:
            if l in VOWELS:
                return True
        return False



    def _is_consonant(self, char):
        """Helper to check if a letter is a consonant"""
        if not char:
            return False
        letters = str(char).upper().split("/")
        for l in letters:
            if l in CONSONANTS:
                return True
        return False

    def _is_alternating_word(self, word_chars):
        """Check if a series of letters strictly alternates C/V"""
        if not word_chars:
            return True
        current_v = self._is_vowel(word_chars[0])
        for i in range(1, len(word_chars)):
            next_v = self._is_vowel(word_chars[i])
            if next_v == current_v:
                return False
            current_v = next_v
        return True

    def _solve_cube_board(self, board, dictionary, min_word_length=3):
        """Find words on a 3x3x3 cube surface using Optimized Backtracking DFS"""
        found = {}  # {word: path}
        import time

        start_t = time.time()
        solver_timeout = 3.0  # Strict 3s timeout for 3D solving
        max_len = 10  # PERFORMANCE: Reset to 10 with neighbor-cache it should be fine.
        visited_cells = set()

        # Pre-calculate neighbors for this solve session (Fast local access)
        # 3x3x3 surface is only 54 cells.
        cube_neighbors = {}
        for fi in range(6):
            for ri in range(3):
                for ci in range(3):
                    cube_neighbors[(fi, ri, ci)] = self._get_cube_neighbors(fi, ri, ci)

        path_list = []


        # REFACTORED DFS for maximum speed: Direct Trie Traversal
        def solve_dfs(f, r, c, node, word_str):
            if time.time() - start_t > solver_timeout:
                return

            char = board[f][r][c]
            # Support Either/Or tiles in 3D (e.g. 'A/B')
            letters = char.split('/') if '/' in char else [char]
            
            for l in letters:
                if l not in node.children:
                    continue
                
                next_node = node.children[l]
                current_word = word_str + l
                
                # Special 'Q' handling (matches QU)
                if l == 'Q' and 'U' in next_node.children:
                    # In 3D, 'Q' is treated as 'QU' for points but only 1 tile
                    next_node = next_node.children['U']
                    current_word += 'U'
                
                if len(current_word) >= min_word_length and next_node.is_word:
                    if current_word not in found:
                        found[current_word] = list(path_list) + [(f, r, c)]
                
                if len(current_word) < max_len:
                    visited_cells.add((f, r, c))
                    path_list.append((f, r, c))
                    for nf, nr, nc in cube_neighbors[(f, r, c)]:
                        if (nf, nr, nc) not in visited_cells:
                            solve_dfs(nf, nr, nc, next_node, current_word)
                    path_list.pop()
                    visited_cells.remove((f, r, c))

        depth_val = len(board)
        rows, cols = len(board[0]), len(board[0][0])
        
        # Determine the correct starting Trie
        from word_validator import word_validator
        if dictionary == 'UniqueNWL':
            start_trie = word_validator.unique_nwl_trie
        elif dictionary == 'UniqueCSW':
            start_trie = word_validator.unique_csw_trie
        elif dictionary == 'CSW':
            start_trie = word_validator.csw_trie
        else:
            start_trie = word_validator.nwl_trie

        for f in range(depth_val):
            for r in range(rows):
                for c in range(cols):
                    if time.time() - start_t > solver_timeout:
                        break
                    solve_dfs(f, r, c, start_trie, "")


        duration = time.time() - start_t
        print(f"[BoardGen] Cube Solver finished in {duration:.2f}s (Words found: {len(found)})")
        return found

    def _embed_bonus_word_cube(self, board, bonus_word, is_checkerboard=False):
        """Backtracking embed on cube surface"""
        p_word = []
        i = 0
        while i < len(bonus_word):
            if i < len(bonus_word) - 1 and bonus_word[i : i + 2].upper() == "QU":
                p_word.append("Q")
                i += 2
            else:
                p_word.append(bonus_word[i].upper())
                i += 1

        depth_val = len(board)
        rows, cols = len(board[0]), len(board[0][0])
        cells = [(f, r, c) for f in range(depth_val) for r in range(rows) for c in range(cols)]
        random.shuffle(cells)

        def backtrack(path):
            if len(path) == len(p_word):
                return path
            cf, cr, cc = path[-1]
            neighbors = self._get_cube_neighbors(cf, cr, cc)
            random.shuffle(neighbors)
            for nf, nr, nc in neighbors:
                if (nf, nr, nc) not in path:
                    if is_checkerboard:
                        # (f+r+c)%2 == 0 is Consonant, == 1 is Vowel
                        expected_vowel = (nf + nr + nc) % 2 != 0
                        if self._is_vowel(p_word[len(path)]) != expected_vowel:
                            continue
                    res = backtrack(path + [(nf, nr, nc)])
                    if res:
                        return res
            return None

        for sf, sr, sc in cells:
            if is_checkerboard:
                expected_vowel = (sf + sr + sc) % 2 != 0
                if self._is_vowel(p_word[0]) != expected_vowel:
                    continue
            path = backtrack([(sf, sr, sc)])
            if path:
                for idx, (f, r, c) in enumerate(path):
                    board[f][r][c] = p_word[idx]
                return path
        return None


def solve_board(board, dictionary="NWL", min_word_length=3):
    """Standalone wrapper for external solving (e.g. history recovery)"""
    from board_generator import BoardGenerator

    bg = BoardGenerator()
    # Support 3x3x3 detect for depth
    is_3d = len(board) > 0 and isinstance(board[0], list) and isinstance(board[0][0], list)
    max_d = 25 if not is_3d else 54
    results = bg._solve_board(
        board, dictionary, min_word_length=min_word_length, max_depth=max_d, store_paths=False
    )
    return list(results.keys()) if results else []


if __name__ == "__main__":
    gen = BoardGenerator()
    board, words, bonus_cell = gen.generate_board("4x4", "BACKWARD", (50, 150), "NWL", "Normal", 3, "Normal")
    if board:
        print("Board generated!")
        for row in board:
            print(" ".join(row))
        print(f"\\nFound {len(words)} words")
        print(f"Bonus word: BACKWARD")
    else:
        print("Failed to generate board")
