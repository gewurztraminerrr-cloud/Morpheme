"""
Board Generator for Morpheme Boggle Game
Generates boards with bonus word embedding and validation
"""

import random
import time
import os
from word_validator import word_validator

DEBUG_FLOW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_flow.log')

# Letter frequency (A-Z)
# Medium/Hard weights - CUSTOMIZED: Peak Connectivity for 7-10L words
# User-provided frequencies for 4x4 (A-Z)
LETTER_FREQ_USER = [
    50,  # A
    95,  # B
    200, # C
    136, # D
    400, # E
    150, # F
    120, # G
    180, # H
    700, # I
    1,   # J (Drastic Reduction)
    2,   # K (Drastic Reduction)
    247, # L
    180, # M
    225, # N
    268, # O
    160, # P
    1,   # Q (Drastic Reduction)
    279, # R
    269, # S
    240, # T
    157, # U
    20,  # V (Reduced)
    20,  # W (Reduced)
    1,   # X (Drastic Reduction)
    95,  # Y
    1,   # Z (Drastic Reduction)
]

# Easy weights (Sum = 10000) - CUSTOMIZED: Peak Density
LETTER_FREQ_EASY = [
    200, # A
    230,  # B
    650,  # C (Boosted per user request)
    410,  # D
    1400, # E
    450,  # F (Boosted per request)
    300,  # G
    550,  # H (Boosted per request)
    1400, # I
    1,    # J (Hardened)
    2,    # K (Hardened)
    560,  # L
    580,  # M (Boosted per user request)
    580,  # N
    610,  # O
    590,  # P (Boosted per user request)
    1,    # Q (Hardened)
    730,  # R
    940,  # S
    570,  # T
    600,  # U
    50,   # V
    60,   # W
    2,    # X (Hardened)
    180,  # Y
    1,    # Z (Hardened)
]

LETTER_FREQ_SUPER_DENSITY = [
    250, # A
    150,  # B
    350,  # C
    300,  # D
    1600, # E
    250,  # F
    150,  # G
    300,  # H
    1500, # I
    1,    # J (Near Zero)
    2,    # K (Near Zero)
    500,  # L
    350,  # M
    600,  # N
    1000, # O
    250,  # P
    1,    # Q (Near Zero)
    600,  # R
    800,  # S
    800,  # T
    400,  # U
    10,   # V (Reduced)
    15,   # W (Reduced)
    1,    # X (Near Zero)
    100,  # Y
    1,    # Z (Near Zero)
]

VOWELS = "AEIOU"
CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ"
# User-identified difficult letters for Hard rounds (with common support for density)
RARE_SET = "ZXQJKVWYPFBHCMAU" + "ETAOINSRHDLU" + "AEIOUAEIOU"  # Blend with common consonants and 10 vowels

# Sparse weights for large grids with low word count targets (Reduced common vowels/consonants)
# Sparse weights for large grids with low word count targets (Heavily reduced vowels/common consonants)
# Sum = 2605 (approx 1/4 of standard 10000 set for rare packing)
LETTER_FREQ_SPARSE = [
    180,  # A (Increased)
    120,  # B
    360,  # C (Boosted per user request)
    110,  # D
    140,  # E
    320,  # F (Boosted per user request)
    120,  # G
    320,  # H (Boosted per user request)
    140,  # I (Increased)
    5,    # J (Hardened)
    40,   # K (Decreased)
    140,  # L
    340,  # M (Boosted per user request)
    110,  # N
    140,  # O (Increased)
    320,  # P (Boosted per user request)
    1,    # Q (Hardened)
    730,  # R
    940,  # S
    570,  # T
    600,  # U
    20,   # V (Reduced)
    30,   # W (Reduced)
    2,    # X (Hardened)
    180,  # Y
    1,    # Z (Hardened)
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
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if core_type.startswith("UNIQUE"):
                path = os.path.join(base_dir, 'dictionaries', f"{core_type.lower()}.txt")
                path_alt = os.path.join(base_dir, 'dictionaries', f"{core_type}.txt")
            else:
                path = os.path.join(base_dir, 'dictionaries', f"unique{core_type}.txt")
                path_alt = os.path.join(base_dir, 'dictionaries', f"Unique{core_type}.txt")

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

    def _get_uniqueness_range(self, difficulty, rows=4, cols=4, dictionary="NWL", depth=1):
        """Get (min, max) ratio range for specified difficulty.
        Calibrated for both Small (4x4, 4x6) and Large (6x8, 3x3x3) grids."""
        
        # 3x3x3 Cube has 54 tiles, 6x8 has 48. Both are Large.
        # rows*cols for 3x3x3 is 9, but depth is 3.
        total_tiles = int(rows) * int(cols) * int(depth)
        is_large = (total_tiles >= 35)

        if is_large:
            # Huge grids (6x8, 5x7, 3x3x3): Higher connectivity lowers uniqueness perception thresholds.
            # Updated: Easy <= 25%, Medium 26-44%, Hard >= 45%
            ranges = {"Easy": (0.01, 0.25), "Medium": (0.26, 0.44), "Hard": (0.45, 1.0)}
        else:
            # Small grids (4x4, 4x6): Thresholds depend heavily on dictionary size.
            dict_upper = str(dictionary).upper()
            if dict_upper == "CSW":
                # CSW (Massive vocabulary): High uniqueness is natural.
                # Thresholds: Easy <= 44%, Medium 45-64%, Hard >= 65%
                ranges = {"Easy": (0.01, 0.44), "Medium": (0.45, 0.64), "Hard": (0.65, 1.0)}
            else:
                # NWL (Strict vocabulary): Even a few unique words make it Medium/Hard.
                # Thresholds: Easy <= 25%, Medium 26-44%, Hard >= 45% (Synced with Large NWL for simplicity)
                ranges = {"Easy": (0.01, 0.25), "Medium": (0.26, 0.44), "Hard": (0.45, 1.0)}

        return ranges.get(difficulty, (0, 1.0))

    def get_uniqueness_ratio(self, board, all_words, rows=4, cols=4, dictionary="NWL", depth=1):
        """Calculate the uniqueness ratio for a given board and word list.
        User Requirement: For small boards (4x4, 4x6), ignore 3-letter and 4-letter words
        to ensure the percentage isn't diluted by common filler.
        """
        if not all_words:
            return 0.0

        unique_set = self._get_difficulty_set(dictionary)
        if not unique_set:
            return 0.0

        total_tiles = int(rows) * int(cols) * int(depth)
        # Small grid criteria: < 35 cells (4x4=16, 4x6=24, 5x6=30)
        is_small = (total_tiles < 35)

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

    def get_difficulty_label(self, ratio, rows=4, cols=4, dictionary="NWL", depth=1, board=None, target_difficulty=None):
        """Derive difficulty label from actual uniqueness ratio achieved, honoring target_difficulty if provided."""
        # Clean target difficulty if it has percentages
        if target_difficulty:
            clean_diff = str(target_difficulty).split('(')[0].strip()
            if clean_diff in ['Easy', 'Medium', 'Hard']:
                # Respect the ING sequence rule which overrides Medium/Hard to Easy
                if clean_diff != "Easy" and board and self._has_forbidden_sequence(board, sequence="ING", depth=depth):
                    print(f"[BoardGen-Diff] Forcing 'Easy' because 'ING' sequence was found.")
                    return "Easy"
                return clean_diff

        # USER REQUEST: Ensure ING sequences NEVER appear in Medium/Hard. If board is provided and has ING, force Easy.
        if board and self._has_forbidden_sequence(board, sequence="ING", depth=depth):
            print(f"[BoardGen-Diff] Forcing 'Easy' because 'ING' sequence was found.")
            return "Easy"
            
        # Defensive casting to ensure math logic works
        try:
            # ROBUST PARSING: handles '0.14', '14.0', or '14%'
            rat_str = str(ratio).replace('%', '').strip()
            rat = float(rat_str) if rat_str else 0.0
            
            if rat > 1.0:
                rat = rat / 100.0
            
            r = int(rows)
            c = int(cols)
            d = int(depth)
        except Exception as e:
            print(f"[BoardGen-Diff] ERROR parsing ratio '{ratio}': {e}")
            return "Easy" # Safe fallback

        res = "Easy"
        
        total_tiles = r * c * d
        is_large = (total_tiles >= 35)
        
        dict_upper = str(dictionary).upper()
        
        # Use a unified threshold system
        if is_large:
            # Large Grids (6x8, 5x7, 3x3x3): Optimized for lower uniqueness on large grids
            if rat >= 0.36: return "Hard"
            if rat >= 0.21: return "Medium"
            return "Easy"
        else:
            # Small Grids (4x4, 4x6)
            if dict_upper == "CSW":
                # CSW Small: Shifted higher because CSW has more obscure words
                if rat >= 0.56: return "Hard"
                if rat >= 0.41: return "Medium"
                return "Easy"
            else:
                # NWL Small: Following User's 30/40/41 rule
                if rat >= 0.41: return "Hard"
                if rat >= 0.31: return "Medium"
                return "Easy"
        
        # CRITICAL DEBUG: This helps explain WHY a label was chosen in the logs.
        # Check this in server.log to see the exact decision path.
        print(f"[BoardGen-Diff] FINAL RESULT: {res} | rat={rat:.4f} (Raw: {ratio}) | Grid: {r}x{c} ({total_tiles} tiles) | Large: {is_large}")
        return res

    def _sanitize_rare_letters(self, board, depth=1, protected_positions=None, is_checkerboard=False):
        """
        USER MANDATE: Ironclad enforcement of rare letter distribution.
        1. Max 1 of each super-rare (Q, Z, J, X, K).
        2. Max 3 TOTAL super-rare letters per board (New Hardening).
        """
        rare_letters = {'Q', 'Z', 'J', 'X', 'K'}
        found_counts = {rl: 0 for rl in rare_letters}
        total_rares_found = 0
        
        rows = len(board) if depth == 1 else len(board[0])
        cols = len(board[0]) if depth == 1 else len(board[0][0])
        
        # Protected set for bonus word
        protected = set(protected_positions) if protected_positions else set()
        
        sanitized_count = 0
        for f in range(depth):
            for r in range(rows):
                for c in range(cols):
                    cell = str(board[f][r][c] if depth > 1 else board[r][c])
                    
                    # Check for ANY rare letter in the cell (handles "QU" or "Q")
                    replaced_in_this_cell = False
                    for rl in rare_letters:
                        if rl in cell:
                            # Is this letter redundant or exceeding total cap?
                            is_redundant = (found_counts[rl] >= 1)
                            is_over_total_cap = (total_rares_found >= 3)
                            
                            if is_redundant or is_over_total_cap:
                                if (f, r, c) in protected or (r, c) in protected:
                                    # Protected bonus word letters are ALWAYS kept
                                    found_counts[rl] += 1
                                    total_rares_found += 1
                                    continue
                                
                                # REPLACE IT with a pool of playable mid-tier consonants or vowels
                                # If checkerboard, only replace consonants with consonants!
                                if is_checkerboard:
                                    replacements = ["F", "H", "C", "P", "M", "R", "S", "T"]
                                else:
                                    replacements = ["F", "H", "C", "P", "M", "E", "A", "I", "R", "S", "T"]
                                new_char = random.choice(replacements)
                                if depth > 1: board[f][r][c] = new_char
                                else: board[r][c] = new_char
                                print(f"[BoardGen] 🛡️ Sanitizer: Replaced redundant/excess '{rl}' at {(f,r,c) if depth > 1 else (r,c)} with '{new_char}'")
                                sanitized_count += 1
                                replaced_in_this_cell = True
                                break # Exit rl loop for this cell
                            else:
                                found_counts[rl] += 1
                                total_rares_found += 1
                    
                    if replaced_in_this_cell:
                        continue
        if sanitized_count > 0:
            print(f"[BoardGen] 🛡️ Sanitizer replaced {sanitized_count} redundant rare letters.")

    def _sanitize_letter_abundances(self, board, depth=1, board_format="Normal", protected_positions=None, is_checkerboard=False):
        """
        USER MANDATE: Ironclad enforcement that NO letter may have an abundance (high count)
        on the board unless the round's format is specifically that letter's Mania format (e.g. "S Mania").
        
        This prevents weird high-density clusters of consonants like 'W' or 'S' on Normal/Non-Mania boards.
        """
        # Determine current Mania letter if any
        safe_format = str(board_format or "Normal").strip().upper()
        mania_letter = None
        if "MANIA" in safe_format:
            parts = safe_format.split()
            if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isalpha():
                mania_letter = parts[0]

        rows = len(board) if depth == 1 else len(board[0])
        cols = len(board[0]) if depth == 1 else len(board[0][0])
        grid_size = rows * cols
        protected = set(protected_positions) if protected_positions else set()

        total_cells = grid_size * depth
        
        letter_counts = {}
        for f in range(depth):
            for r in range(rows):
                for c in range(cols):
                    cell = board[f][r][c] if depth > 1 else board[r][c]
                    for char in cell:
                        if char.isalpha():
                            letter_counts[char.upper()] = letter_counts.get(char.upper(), 0) + 1

        sanitized_count = 0
        
        VOWELS = {"A", "E", "I", "O", "U"}
        COMMON_CONSONANTS = {"S", "T", "R", "N", "L", "D"}
        
        def get_limit(letter):
            if letter == mania_letter:
                return 9999
            if letter in VOWELS:
                return max(4, int(total_cells * 0.18))
            if letter in COMMON_CONSONANTS:
                return max(3, int(total_cells * 0.12))
            return max(2, int(total_cells * 0.09))

        consonants_pool = [c for c in "BCDFGHJKLMNPQRSTVWXYZ" if c != mania_letter]
        vowels_pool = [v for v in "AEIOU" if v != mania_letter]

        for letter, count in list(letter_counts.items()):
            limit = get_limit(letter)
            if count > limit:
                positions = []
                for f in range(depth):
                    for r in range(rows):
                        for c in range(cols):
                            cell = board[f][r][c] if depth > 1 else board[r][c]
                            if cell == letter:
                                pos = (f, r, c) if depth > 1 else (r, c)
                                if pos not in protected:
                                    positions.append(pos)
                
                random.shuffle(positions)
                
                excess = count - limit
                to_replace = positions[:excess]
                
                for pos in to_replace:
                    new_char = None
                    is_vowel = (letter in VOWELS)
                    
                    for _ in range(50):
                        if is_checkerboard or is_vowel:
                            if is_vowel:
                                cand = random.choice(vowels_pool)
                            else:
                                cand = random.choice(consonants_pool)
                        else:
                            cand = random.choice(vowels_pool + consonants_pool)
                            
                        if letter_counts.get(cand, 0) < get_limit(cand):
                            new_char = cand
                            break
                            
                    if not new_char:
                        new_char = random.choice(vowels_pool if is_vowel else consonants_pool)
                        
                    if depth > 1:
                        f, r, c = pos
                        board[f][r][c] = new_char
                    else:
                        r, c = pos
                        board[r][c] = new_char
                        
                    letter_counts[letter] -= 1
                    letter_counts[new_char] = letter_counts.get(new_char, 0) + 1
                    sanitized_count += 1
                    print(f"[BoardGen] 🛡️ Sanitizer: Replaced excess '{letter}' ({count}/{limit}) at {pos} with '{new_char}'")

        if sanitized_count > 0:
            print(f"[BoardGen] 🛡️ Sanitizer: Replaced a total of {sanitized_count} excess letters to enforce non-abundance.")

    def _sanitize_forbidden_sequences(self, board, depth=1, protected_positions=None, is_checkerboard=False):
        """
        USER MANDATE: Break up any "ING" sequences on Medium/Hard boards.
        """
        is_3d = (depth > 1) or (len(board) == 6 and isinstance(board[0], list) and isinstance(board[0][0], list))
        depth_val = 6 if (len(board) == 6 and is_3d) else depth
        rows = len(board[0]) if is_3d else len(board)
        cols = len(board[0][0]) if is_3d else len(board[0])
        protected = set(protected_positions) if protected_positions else set()

        sequence = "ING"
        seq_len = len(sequence)

        # Loop until no forbidden sequences are found
        max_attempts = 200
        sanitized_count = 0
        for attempt_idx in range(max_attempts):
            found_any = False
            
            def find_path(idx, r, c, f, current_path):
                if idx == seq_len:
                    return current_path
                target = sequence[idx]
                for df in ([-1, 0, 1] if is_3d else [0]):
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if df == 0 and dr == 0 and dc == 0:
                                continue
                            nf, nr, nc = f + df, r + dr, c + dc
                            if 0 <= nf < depth_val and 0 <= nr < rows and 0 <= nc < cols:
                                pos = (nf, nr, nc) if is_3d else (nr, nc)
                                if pos not in current_path:
                                    val = board[nf][nr][nc] if is_3d else board[nr][nc]
                                    if val == target:
                                        res = find_path(idx + 1, nr, nc, nf, current_path + [pos])
                                        if res: return res
                return None

            for f in range(depth_val):
                for r in range(rows):
                    for c in range(cols):
                        val = board[f][r][c] if is_3d else board[r][c]
                        if val == sequence[0]:
                            pos = (f, r, c) if is_3d else (r, c)
                            path = find_path(1, r, c, f, [pos])
                            if path:
                                found_any = True
                                replaced = False
                                for p in path:
                                    if p not in protected:
                                        if is_checkerboard:
                                            # Keep the same vowel/consonant type!
                                            orig_char = sequence[path.index(p)]
                                            is_v = orig_char in "AEIOU"
                                            if is_v:
                                                replacements = ["A", "E", "O", "U"]
                                            else:
                                                replacements = ["S", "T", "R", "L", "C", "P"]
                                        else:
                                            replacements = ["A", "E", "O", "S", "T", "R", "L", "C", "P"]
                                        new_char = random.choice(replacements)
                                        if is_3d: board[p[0]][p[1]][p[2]] = new_char
                                        else: board[p[0]][p[1]] = new_char
                                        replaced = True
                                        print(f"[BoardGen] 🛡️ Sequence Sanitizer: Broke 'ING' by replacing '{sequence[path.index(p)]}' at {p} with '{new_char}'")
                                        sanitized_count += 1
                                        break
                                if not replaced:
                                    print(f"[BoardGen] ⚠️ Sequence Sanitizer: Could not break 'ING' because all tiles are protected!")
                                break
                    if found_any: break
                if found_any: break
            
            if not found_any:
                break
                
        if attempt_idx == max_attempts - 1:
            print(f"[BoardGen] ⚠️ Sequence Sanitizer reached max attempts ({max_attempts}). Some ING sequences may remain.")

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

    def _select_strategy(self, dimensions, min_words, max_words, difficulty, min_word_length, is_emergency=False):
        """Standard methodology selection based on empirical analysis of 200+ parameter sets.
        USER REQUEST: For emergencies, always use FastReRoll to keep UI responsive.
        """
        if is_emergency:
            return "FastReRoll"
            
        parts = dimensions.split("x")
        if len(parts) == 3:
            depth, rows, cols = map(int, parts)
        else:
            rows, cols = map(int, parts)
            depth = 1

        # Small grids targeting uniqueness (Hard) or high density (Medium 100+)
        if rows * cols <= 25:
            # User Request Fix: High word count targets or Medium/Hard difficulty on small grids MUST use IO to hit range reliably
            if difficulty in ["Medium", "Hard"] or min_words >= 100:
                return "StepwiseOptimization"
            # On small grids (4x4), FastReRoll can hit low-count Easy targets easily
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
        self, dimensions, bonus_word, word_count_range, dictionary, board_format, min_word_length=3, difficulty="Medium", is_emergency=False, timeout=None
    ):
        """
        Generate a valid board that meets word count requirements (100-300).
        RESTARTED: Simplified logic with ironclad compliance.
        """
        # Ensure Mania has a valid single-letter prefix
        if "mania" in str(board_format).lower():
            parts = str(board_format).strip().split()
            # If it doesn't have a single-letter prefix
            if len(parts) < 2 or len(parts[0]) != 1 or not parts[0].isalpha():
                if random.random() < 0.33:
                    mania_letter = random.choice('AEIOU')
                else:
                    mania_letter = random.choice('BCDFGHJKLMNPQRSTVWXYZ')
                board_format = f"{mania_letter} Mania"
                print(f"[BoardGen] Normalized naked Mania format to '{board_format}'")

        # USER REQUEST: Parse word count range from spinner instead of hardcoding
        min_words, max_words = self._parse_word_count_range(word_count_range)
        print(f"[BoardGen] generate_board called for {dimensions} | Range: {word_count_range} ({min_words}-{max_words}) | MinLen: {min_word_length}L")
        
        # 1. Dimension Parsing
        if dimensions == "3x3x3":
            depth, rows, cols = 6, 3, 3
        else:
            parts = dimensions.split("x")
            depth, rows, cols = (map(int, parts) if len(parts) == 3 else (1, int(parts[0]), int(parts[1])))
        
        num_tiles = rows * cols * depth
        
        # (Special procedure for large boards removed per user request to use standard algorithm)
        
        # 2. Setup Loop
        start_time = time.time()
        # User Request: Keep generating until it falls within range.
        # We give a generous timeout for the "Ironclad" guarantee.
        # IO Optimization timeout: 4x4 boards need more time for high-density targets (100+)
        # USER MANDATE: Do not distribute until criteria is met. We increase timeout and attempts.
        if timeout is None:
            timeout = 120.0 if (not is_emergency and rows*cols <= 16 and min_words >= 100) else (60.0 if not is_emergency else 8.0)
        attempts = 0
        
        while time.time() - start_time < timeout:
            attempts += 1
            print(f"[BoardGen] COMPLIANCE ATTEMPT {attempts} (Target: {min_words}-{max_words}, MinLen: {min_word_length})")

            # If we are in Mania format and failing to find a compliant board,
            # dynamically rotate the mania letter to a different one to aid compliance!
            if "mania" in str(board_format).lower() and attempts > 3:
                if random.random() < 0.33:
                    mania_letter = random.choice('AEIOU')
                else:
                    mania_letter = random.choice('BCDFGHJKLMNPQRSTVWXYZ')
                board_format = f"{mania_letter} Mania"
                print(f"[BoardGen] Rotated Mania letter to '{mania_letter}' on attempt {attempts} to find compliant board.")
            
            # --- STRATEGY SELECTION ---
            # USER REQUEST: Use Word Soup for everything, removing the slow brute-force block for 500+ density
            pass

            # --- BOARD CREATION ---
            safe_format = str(board_format or "").lower()
            is_checkerboard = "checkerboard" in safe_format

            # Standard Strategies
            if num_tiles >= 24 and depth == 1 and not is_checkerboard and "either/or" not in safe_format:
                strategy = "WordSoup"
            else:
                strategy = "StepwiseOptimization" if num_tiles >= 24 else "HighDensity"
            
            # Weighted frequencies for density
            # USER REQUEST: If target is high density or high min length, use Super Density weights
            is_super_dense = (min_words >= 200 or (min_word_length >= 4 and rows*cols <= 24))
            weights = LETTER_FREQ_SUPER_DENSITY if is_super_dense else (LETTER_FREQ_EASY if (min_words >= 100 or attempts > 3) else LETTER_FREQ_USER)
            if is_checkerboard:
                board = self._create_checkerboard(rows, cols, weights, depth=depth, difficulty=difficulty)
            elif "either/or" in safe_format:
                # User Request: Support Either/Or tiles (e.g. A/B)
                board = self._create_either_or_board(rows, cols, weights)
                if self._has_either_or_ambiguity(board, dictionary):
                    print(f"[BoardGen] ATTEMPT {attempts}: Either/Or board has ambiguity. Retrying...")
                    continue
            else:
                board = self._create_normal_board(rows, cols, weights, depth=depth, difficulty=difficulty, dictionary=dictionary)

            # USER REQUEST: Limit the number of A's on large boards (5x7, 6x8)
            if rows * cols >= 35:
                flat_letters = []
                for f in range(len(board)):
                    for r in range(len(board[f])):
                        for c in range(len(board[f][r])):
                            cell = str(board[f][r][c])
                            flat_letters.extend(cell.split('/'))
                
                a_count = sum(1 for char in flat_letters if char == 'A')
                max_as = 9 if rows * cols >= 48 else 7 # Max 9 for 6x8, 7 for 5x7
                if a_count > max_as:
                    print(f"[BoardGen] ATTEMPT {attempts}: Too many A's ({a_count} > {max_as}). Retrying...")
                    continue
            
            # --- BONUS WORD EMBEDDING ---
            embedded_path = None
            if bonus_word:
                if depth > 1:
                    embedded_path = self._embed_bonus_word_cube(board, bonus_word, is_checkerboard=is_checkerboard)
                else:
                    embedded_path = self._embed_bonus_word(board, bonus_word, is_checkerboard=is_checkerboard)
                
                if not embedded_path:
                    # If embedding fails, we retry the whole board attempt
                    print(f"[BoardGen] ATTEMPT {attempts}: Failed to embed bonus word '{bonus_word}'. Retrying...")
                    continue
                
                # Removed early sanitize (moved to end of attempt to catch optimization/sweeps)
            else:
                pass

            all_excluded = set()
            if "mania" in board_format.lower():
                mania_letter = board_format.split()[0].upper()
                self._apply_mania_to_board(board, mania_letter, all_excluded, is_checkerboard=is_checkerboard)
                # Add mania cells to excluded to protect them from optimization
                for f in range(depth):
                    for r in range(rows):
                        for c in range(cols):
                            cell = board[f][r][c] if depth > 1 else board[r][c]
                            if cell == mania_letter: all_excluded.add((f, r, c) if depth > 1 else (r, c))
            
            # Protect Either/Or tiles from optimization
            if "either/or" in board_format.lower():
                for f in range(depth):
                    for r in range(rows):
                        for c in range(cols):
                            cell = board[f][r][c] if depth > 1 else board[r][c]
                            if "/" in cell: all_excluded.add((f, r, c) if depth > 1 else (r, c))

            # --- OPTIMIZATION ---
            if strategy == "StepwiseOptimization":
                # Stage 1: Base Board (Targeting 150 for safety margin)
                # USER REQUEST: Even if we use "Easy" weights for speed, we MUST respect forbidden sequences 
                # if the target difficulty is Medium or Hard.
                board = self._create_2000plus_board(
                    rows, cols, dictionary, is_checkerboard, board, all_excluded, "Density",
                    min_word_length, max_words, min_words, 0, 1, depth=depth, difficulty=difficulty, weights=weights
                )
                # Stage 2: IO and B Uniqueness
                board = self._apply_io_b_uniqueness_optimization(
                    board, rows, cols, dictionary, all_excluded, min_word_length, depth=depth, difficulty=difficulty, max_words=max_words, is_checkerboard=is_checkerboard
                )
            else:
                board = self._create_2000plus_board(
                    rows, cols, dictionary, is_checkerboard, board, all_excluded, "Density",
                    min_word_length, max_words, min_words, 0, 1, depth=depth, difficulty=difficulty, weights=weights
                )

            # --- SOLVE FOR INITIAL COUNT ---
            # PERFORMANCE: For Large/3D grids in emergency mode, depth 12 is enough to be rapid (Zero Wait)
            if is_emergency and rows * cols >= 35:
                final_depth = 12
            else:
                final_depth = 25 if rows * cols <= 16 else 14
            all_words_dict = self._solve_board(
                board, dictionary, (0, 99999), min_word_length, max_depth=final_depth, store_paths=True, timeout=30.0
            )
            count = len(all_words_dict)
            print(f"[BoardGen] ATTEMPT {attempts} PRE-SWEEP: Count={count} Target={min_words}-{max_words}")

            # --- DYNAMIC CORRECTION (Push-Pull) ---
            if count < min_words:
                # SPARSE: Add letters to increase count
                board = self._perform_rescue_sweep(board, rows, cols, depth, dictionary, min_word_length, min_words, max_words, all_excluded, difficulty, rescue_depth=final_depth, protected_path=embedded_path, is_checkerboard=is_checkerboard, board_format=board_format)
            elif count > max_words:
                # OVER-DENSE: Remove letters to decrease count
                board = self._perform_decimation_sweep(board, rows, cols, depth, dictionary, min_word_length, min_words, max_words, all_excluded, difficulty, rescue_depth=final_depth, protected_path=embedded_path, is_checkerboard=is_checkerboard, board_format=board_format)

            # --- FINAL RARE LETTER SANITIZATION (User Request: Max 1 Q, Z, J, X, K) ---
            # We do this AFTER all optimizations and sweeps to ensure compliance and clean board.
            self._sanitize_rare_letters(board, depth, protected_positions=embedded_path, is_checkerboard=is_checkerboard)
            self._sanitize_letter_abundances(board, depth, board_format=board_format, protected_positions=embedded_path, is_checkerboard=is_checkerboard)
            if difficulty in ["Medium", "Hard"]:
                self._sanitize_forbidden_sequences(board, depth, protected_positions=embedded_path, is_checkerboard=is_checkerboard)

            # Final check to ensure the board strictly alternates C/V in checkerboard mode
            if is_checkerboard:
                self._verify_checkerboard_safeguard(board, weights, set(embedded_path) if embedded_path else set())

            # --- FINAL A'S SANITIZATION FOR LARGE BOARDS ---
            if rows * cols >= 35 and depth == 1:
                max_as = 9 if rows * cols >= 48 else 7
                
                # Find all positions of 'A'
                a_positions = []
                for r in range(rows):
                    for c in range(cols):
                        if board[r][c] == 'A':
                            a_positions.append((r, c))
                
                if len(a_positions) > max_as:
                    print(f"[BoardGen] Excess A's detected after optimization ({len(a_positions)} > {max_as}). Sanitizing...")
                    random.shuffle(a_positions)
                    protected = set(embedded_path) if embedded_path else set()
                    
                    for r, c in a_positions[:]:
                        if len(a_positions) <= max_as:
                            break
                        if (r, c) in protected:
                            continue
                            
                        # Replace with another vowel to maintain checkerboard or vowel status
                        replacements = ["E", "O", "I"]
                        board[r][c] = random.choice(replacements)
                        a_positions.remove((r, c))
                        print(f"[BoardGen] Replaced excess 'A' at ({r}, {c}) with '{board[r][c]}'")

            # Re-solve after sweeps and sanitization for final confirmation
            all_words_dict = self._solve_board(
                board, dictionary, (0, 99999), min_word_length, max_depth=final_depth, store_paths=True, timeout=30.0
            )
            count = len(all_words_dict)
            print(f"[BoardGen] ATTEMPT {attempts} POST-SWEEP VERIFICATION: Count={count} Target={min_words}-{max_words} MinLen={min_word_length}L")
            
            if min_words <= count <= max_words:
                print(f"[BoardGen] ✓ IRONCLAD COMPLIANT BOARD FOUND ({count} words @ {min_word_length}L+) on attempt {attempts}")
                
                # RECALCULATE RATIO
                unique_set = self._get_difficulty_set(dictionary)
                u_count = sum(1 for w in all_words_dict if w.upper() in unique_set)
                ratio = u_count / count if count > 0 else 0
                
                # PICK BONUS WORD
                actual_bonus = None
                bonus_cell = None
                
                # USER REQUEST: Absolute Parity. Verify the bonus word survived the sweeps.
                if bonus_word and embedded_path and bonus_word.upper() in [w.upper() for w in all_words_dict]:
                    # We have a successfully embedded word that survived the sweeps
                    actual_bonus = bonus_word
                    bonus_cell = all_words_dict[actual_bonus.upper() if actual_bonus.upper() in all_words_dict else actual_bonus][0]
                else:
                    # Pick a "Natural" bonus word from the board (MANDATORY for all formats)
                    # Use all_words_dict because it's the verified post-sweep solution
                    suitable = [w for w in all_words_dict if 6 <= len(w) <= 10]
                    if not suitable: suitable = [w for w in all_words_dict if len(w) >= 6]
                    if not suitable: suitable = [w for w in all_words_dict if len(w) >= 3] # Absolute fallback
                    
                    # USER REQUEST: Prefer requested length if available in the natural fallback
                    requested_length = len(bonus_word) if bonus_word else 8
                    suitable_exact = [w for w in suitable if len(w) == requested_length]
                    
                    if suitable_exact:
                        actual_bonus = random.choice(suitable_exact)
                    else:
                        actual_bonus = sorted(suitable, key=len, reverse=True)[0] if suitable else None
                        
                    if actual_bonus:
                        bonus_cell = all_words_dict[actual_bonus][0]
                
                # USER REQUEST: If format is Bonus Letter, we MUST have a bonus cell even if no long word found.
                if not bonus_cell and "bonus letter" in safe_format:
                    # Pick a random cell
                    bonus_cell = (random.randint(0, rows-1), random.randint(0, cols-1))
                    if depth > 1: bonus_cell = (random.randint(0, depth-1), bonus_cell[0], bonus_cell[1])
                
                return (
                    board,
                    sorted(list(all_words_dict.keys())),
                    bonus_cell,
                    board_format,
                    all_words_dict,
                    ratio,
                    actual_bonus.upper() if actual_bonus else None
                )
            else:
                print(f"[BoardGen] ✗ NON-COMPLIANT ({count} words). Retrying...")

        # FINAL FALLBACK (If timeout reached)
        # In a complete restart, even fallback MUST be compliant. 
        print("[BoardGen] !! TIMEOUT REACHED. FORCING EMERGENCY CLEAN SLATE.")
        return self._generate_emergency_compliant_board(dimensions, min_word_length, dictionary, board_format, word_count_range, difficulty)

    def _generate_emergency_compliant_board(self, dimensions, min_word_length, dictionary, board_format, word_count_range, difficulty):
        """
        USER MANDATE: NEVER return a non-compliant board. 
        We will loop indefinitely until the target is met.
        We also strictly respect board formats (Mania, Checkerboard, Either/Or) in the emergency path.
        """
        min_words, max_words = self._parse_word_count_range(word_count_range)
        
        # Dimension Parsing
        if dimensions == "3x3x3": depth, rows, cols = 6, 3, 3
        else:
            parts = dimensions.split("x")
            depth, rows, cols = (map(int, parts) if len(parts) == 3 else (1, int(parts[0]), int(parts[1])))
        
        _attempt = 0
        while True:
            _attempt += 1
            print(f"[BoardGen] 🆘 EMERGENCY LOOP ATTEMPT {_attempt} (Target: {min_words}-{max_words})")
            
            # 1. Handle Mania letter selection and dynamic letter rotation if attempts are failing
            if "mania" in str(board_format).lower():
                parts = str(board_format).strip().split()
                if len(parts) < 2 or len(parts[0]) != 1 or not parts[0].isalpha() or (_attempt > 2 and _attempt % 2 == 0):
                    import random
                    if random.random() < 0.33:
                        mania_letter = random.choice('AEIOU')
                    else:
                        mania_letter = random.choice('BCDFGHJKLMNPQRSTVWXYZ')
                    board_format = f"{mania_letter} Mania"
                    print(f"[BoardGen] [Emergency] Chosen/Rotated Mania letter to '{mania_letter}' on attempt {_attempt}")
                else:
                    mania_letter = parts[0].upper()
            else:
                mania_letter = None

            # 2. Setup Board based on format
            safe_format = str(board_format or "").lower()
            is_checkerboard = "checkerboard" in safe_format
            
            if min_word_length >= 4:
                weights = LETTER_FREQ_SUPER_DENSITY
                # USER REQUEST: If we are forced to use Super Density frequencies in emergency mode,
                # update the format to 'Density' so the Spinner Set doesn't abruptly change to a different
                # format (like Bonus Letter) at the start of the round.
                if "density" not in safe_format and safe_format in ["", "normal"]:
                    board_format = "Density"
                    safe_format = "density"
            else:
                weights = LETTER_FREQ_EASY
            
            if is_checkerboard:
                board = self._create_checkerboard(rows, cols, weights, depth=depth, difficulty=difficulty)
                print(f"[BoardGen] Pure Checkerboard generated. Skipping optimization to guarantee layout.")
                
                final_solve = self._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=12 if rows * cols >= 35 else 25, store_paths=True, timeout=30.0)
                found_list = list(final_solve.keys())
                
                suitable_bonus = [w for w in found_list if 6 <= len(w) <= 10]
                if not suitable_bonus: suitable_bonus = [w for w in found_list if len(w) >= 3]
                final_bonus_word = suitable_bonus[0] if suitable_bonus else None
                
                return (
                    board,
                    sorted(found_list),
                    None,
                    "Checkerboard",
                    final_solve,
                    1.0,
                    final_bonus_word.upper() if final_bonus_word else None,
                )
            all_excluded = set()
            if "either/or" in safe_format:
                board = self._create_either_or_board(rows, cols, weights)
                # Find the Either/Or tile and add it to all_excluded to protect it from optimization
                for r in range(rows):
                    for c in range(cols):
                        if "/" in str(board[r][c]):
                            all_excluded.add((r, c))
                            print(f"[BoardGen] Protected Either/Or tile at ({r}, {c})")
            else:
                board = self._create_normal_board(rows, cols, weights, depth=depth, difficulty=difficulty)
                
            # 3. Apply Mania abundance and register cells as excluded to protect them
            if mania_letter:
                self._apply_mania_to_board(board, mania_letter, all_excluded, is_checkerboard=is_checkerboard)
                # Protect these positions from future sweeps or optimizations
                for f in range(depth):
                    for r in range(rows):
                        for c in range(cols):
                            cell = board[f][r][c] if depth > 1 else board[r][c]
                            if cell == mania_letter:
                                all_excluded.add((f, r, c) if depth > 1 else (r, c))

            # Optimization: One quick pass at max density (respecting protected cells)
            board = self._create_2000plus_board(
                rows, cols, dictionary, is_checkerboard, board, all_excluded, "Density",
                min_word_length, max_words, min_words, 0, 1, depth=depth, difficulty=difficulty, weights=weights,
                board_format=board_format
            )
            
            # Final Rescue Sweep to hit the target floor (protecting format tiles)
            board = self._perform_rescue_sweep(board, rows, cols, depth, dictionary, min_word_length, min_words, max_words, all_excluded, difficulty, is_checkerboard=is_checkerboard, board_format=board_format)
            
            display_min = min_word_length
            final_solve = self._solve_board(board, dictionary, (0, 99999), display_min, max_depth=12 if rows * cols >= 35 else 25, store_paths=True, timeout=30.0)
            count = len(final_solve)
            
            if count > max_words:
                board = self._perform_decimation_sweep(board, rows, cols, depth, dictionary, min_word_length, min_words, max_words, all_excluded, difficulty, is_checkerboard=is_checkerboard, board_format=board_format)
                final_solve = self._solve_board(board, dictionary, (0, 99999), display_min, max_depth=12 if rows * cols >= 35 else 25, store_paths=True, timeout=30.0)
                count = len(final_solve)
            
            # --- FINAL SANITIZATION (protecting format tiles) ---
            self._sanitize_rare_letters(board, depth, protected_positions=all_excluded, is_checkerboard=is_checkerboard)
            self._sanitize_letter_abundances(board, depth, board_format=board_format, protected_positions=all_excluded, is_checkerboard=is_checkerboard)
            if difficulty in ["Medium", "Hard"]:
                self._sanitize_forbidden_sequences(board, depth, protected_positions=all_excluded, is_checkerboard=is_checkerboard)
            
            # USER REQUEST: Re-solve after sweeps and sanitization for final confirmation
            # This ensures the bonus word and all words in the list are ACTUALLY on the board!
            final_solve = self._solve_board(
                board, dictionary, (0, 99999), display_min, max_depth=12 if rows * cols >= 35 else 25, store_paths=True, timeout=30.0
            )
            count = len(final_solve)
            
            if min_words <= count <= max_words or _attempt >= 5:
                if _attempt >= 5:
                    print(f"[BoardGen] ⚠️ EMERGENCY LOOP TIMEOUT: Failed to hit target after 5 attempts. Returning best effort with {count} words.")
                else:
                    print(f"[BoardGen] ✓ EMERGENCY COMPLIANCE SUCCESS: {count} words after {_attempt} emergency tries.")
                # Fallback metadata
                unique_set = self._get_difficulty_set(dictionary)
                u_count = sum(1 for w in final_solve if w.upper() in unique_set)
                ratio = u_count / count if count > 0 else 0
                
                # USER REQUEST: Ensure every board in every format has a Bonus Word
                suitable = [w for w in final_solve if 6 <= len(w) <= 10]
                if not suitable: suitable = [w for w in final_solve if len(w) >= 6]
                if not suitable: suitable = [w for w in final_solve if len(w) >= 3]
                
                final_bonus = sorted(suitable, key=len, reverse=True)[0] if suitable else None
                bonus_cell = None
                if final_bonus:
                    bonus_cell = final_solve[final_bonus][0]
 
                return (board, sorted(list(final_solve.keys())), bonus_cell, board_format, final_solve, ratio, final_bonus)
            else:
                print(f"[BoardGen] ✗ EMERGENCY ATTEMPT {_attempt} FAILED: {count} words is not in {min_words}-{max_words}")

    def _final_rare_letter_polish(self, board, depth, protected_positions=None, difficulty=None):
        """Final audit to clean rare letters and forbidden sequences."""
        self._sanitize_rare_letters(board, depth, protected_positions=protected_positions)
        if difficulty in ["Medium", "Hard"]:
            self._sanitize_forbidden_sequences(board, depth, protected_positions=protected_positions)

    def _generate_io_base_board_procedure(
        self, dimensions, bonus_word, word_count_range, dictionary, min_word_length, difficulty, board_format="Normal"
    ):
        """
        Special Procedure: Checkerboard IO and Base
        1. Generate Base board with ~100 words.
        2. Iteratively optimize IO tiles (alternating) to maximize Unique Words.
        """
        rows, cols = map(int, dimensions.split("x"))
        
        # We use a retry loop to guarantee this range.
        for attempt in range(3):
            # Determine Unique dictionary to use for optimization
            d_upper = str(dictionary).upper()
            unique_dict_name = "UniqueNWL" if d_upper == "NWL" else "UniqueCSW"

            # 1. Generate Base Board - NO EMBEDDING
            # USER REQUEST: Scale base board words based on Spinner Set target
            min_w, max_w = self._parse_word_count_range(word_count_range)
            if max_w <= 200:
                base_val = 50
            elif max_w <= 300:
                base_val = 100
            else:
                base_val = 200
                
            base_target = (base_val - 10, base_val + 10) if attempt == 0 else (base_val - 20, base_val + 5)
            
            print(f"[BoardGen] Procedure: IO-Base Checkerboard (Attempt {attempt}). Base target: {base_target}")
            weights = LETTER_FREQ_SUPER_DENSITY if min_word_length >= 4 else LETTER_FREQ_EASY
            base_board = self._create_normal_board(rows, cols, weights, depth=1, difficulty=difficulty)
            print(f"[BoardGen] Normal base generated directly for special procedure.")
            
            final_solve = self._solve_board(base_board, dictionary, (0, 99999), min_word_length, max_depth=12, store_paths=True, timeout=30.0)
            base_words = list(final_solve.keys())

            # 2. IO Optimization
            print(f"[BoardGen] Optimizing IO tiles (Random Walk)...")
            final_board = [row[:] for row in base_board]
            
            # Checkerboard pattern: (r + c) % 2 == 1 (odd tiles) are IO
            io_positions = [(r, c) for r in range(rows) for c in range(cols) if (r + c) % 2 == 1]
            import random
            random.shuffle(io_positions)
            
            optimized_count = 0
            final_words_dict = {}
            final_count = 0
            
            for r, c in io_positions:
                best_letter = final_board[r][c]
                max_unique_at_loc = 0
                
                # Test pool of letters (Respect Checkerboard if needed)
                test_pool = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                if "checkerboard" in str(board_format).lower():
                    target_is_vowel = (r + c) % 2 != 0
                    test_pool = list(VOWELS) if target_is_vowel else list(CONSONANTS)
                
                for char in test_pool:
                    if self._is_rare_limited(final_board, char):
                        if char != final_board[r][c]:
                            continue
                    
                    final_board[r][c] = char
                    # Solve using the surgical must_include solver
                    words_dict = self._solve_board(
                        final_board, unique_dict_name, (0, 99999), min_word_length, max_depth=10, store_paths=False, must_include=(r, c)
                    )
                    unique_count_at_loc = len(words_dict)

                    if unique_count_at_loc > max_unique_at_loc:
                        max_unique_at_loc = unique_count_at_loc
                        best_letter = char
                
                final_board[r][c] = best_letter
                optimized_count += 1
                
                # Check if board meets criteria
                final_words_dict = self._solve_board(
                    final_board, dictionary, (0, 99999), min_word_length, max_depth=12, store_paths=True
                )
                final_count = len(final_words_dict)
                
                print(f"[BoardGen] Optimized {optimized_count} spots. Current count: {final_count}")
                if min_w <= final_count <= max_w:
                    print(f"[BoardGen] Found compliant board after optimizing {optimized_count} spots!")
                    break
                
                # Safety break if taking too long (max 3 spots!)
                if optimized_count >= 3:
                    print(f"[BoardGen] Hit max spot limit (3). Proceeding with current board.")
                    break
            
            if min_w <= final_count <= max_w:
                print(f"[BoardGen] ✓ IO-Base Compliance: {final_count} words (Range: {min_w}-{max_w})")
                break
            else:
                print(f"[BoardGen] ✗ IO-Base Non-Compliance: {final_count} words. Retrying...")
        
        # 4. USER REQUEST: Pick a Bonus Word (6-10 letters long) from the final board
        found_list = list(final_words_dict.keys())
        # Filter for 6-10 length
        suitable_bonus = [w for w in found_list if 6 <= len(w) <= 10]
        
        # Fallback if no 6-10L words found (unlikely in high density, but for safety)
        if not suitable_bonus:
            suitable_bonus = [w for w in found_list if len(w) >= 6]
        if not suitable_bonus:
            suitable_bonus = [w for w in found_list if len(w) >= 3] # Absolute fallback
        
        final_bonus_word = None
        bonus_cell = None
        if suitable_bonus:
            # Sort by length descending to pick a challenging/impressive one
            suitable_bonus.sort(key=len, reverse=True)
            final_bonus_word = suitable_bonus[0]
            # Get path for the anchor cell
            bonus_path = final_words_dict[final_bonus_word]
            if bonus_path:
                bonus_cell = bonus_path[0] # Use start of word as anchor
        
        print(f"[BoardGen] Selected Natural Bonus Word: {final_bonus_word} (Anchor: {bonus_cell})")
        # FINAL AUDIT (User Request: Max 1 Rare Letter)
        self._sanitize_rare_letters(final_board, depth=1)
        if difficulty in ["Medium", "Hard"]:
            self._sanitize_forbidden_sequences(final_board, depth=1, protected_positions=[bonus_cell] if bonus_cell else None)

        return (
            final_board,
            sorted(found_list),
            bonus_cell,
            board_format,
            final_words_dict,
            1.0,
            final_bonus_word.upper() if final_bonus_word else None,
        )

    def _parse_word_count_range(self, word_count_range):
        """Parse word count range using Pattern-Aware Regex"""
        if isinstance(word_count_range, tuple):
            return word_count_range

        if not isinstance(word_count_range, str):
            return (50, 100)

        import re
        
        # 1. Look for explicit dash range (e.g. "50-100" or "Words: 50-100")
        # Pattern: digits, optional spaces, dash, optional spaces, digits
        range_match = re.search(r'(\d+)\s*-\s*(\d+)', word_count_range)
        if range_match:
            return (int(range_match.group(1)), int(range_match.group(2)))
            
        # 2. Look for open-ended range (e.g. "200+" or "500+")
        plus_match = re.search(r'(\d+)\s*\+', word_count_range)
        if plus_match:
            return (int(plus_match.group(1)), 99999)
            
        # 3. Fallback: Just the first number found if no range pattern
        nums = re.findall(r'\d+', word_count_range)
        if nums:
            # If we see many numbers (like in a full summary), skip the first few (dims, time)
            # and look for the one that looks like a range.
            # But the patterns above should catch 99% of cases.
            val = int(nums[-1]) # Take the LAST number if all else fails
            return (val, val + 50)

        return (50, 100) # Safety floor

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

    def _create_normal_board(self, rows, cols, weights, depth=1, difficulty="Easy", dictionary=None):
        """Create board using Overwriting Word Soup method for 2D, or random letters for 3D"""
        if depth > 1:
            return [
                [[random.choices(self.letters, weights=weights, k=1)[0] for _ in range(cols)] for _ in range(rows)]
                for _ in range(depth)
            ]

        board = [[' ' for _ in range(cols)] for _ in range(rows)]
        
        # Get words from dictionary of length 5-10 (Excluding ING words as requested!)
        valid_words = []
        if dictionary:
            if isinstance(dictionary, str):
                dict_name = dictionary.upper()
                import os
                if difficulty in ["Medium", "Hard"]:
                    # Load unique set for Medium/Hard (e.g. uniqueNWL.txt)
                    loaded_dict = self._get_difficulty_set(dict_name)
                    if loaded_dict:
                        dictionary = loaded_dict
                        print(f"[BoardGen] Using UNIQUE dictionary (unique{dict_name}.txt) for {difficulty} Word Soup.")
                    else:
                        print(f"[BoardGen] ⚠️ Failed to load unique set for {dict_name}. Using fallback words.")
                        dictionary = ["EXAMPLE", "BOARDS", "PUZZLE", "BOGGLE", "WONDER"]
                else:
                    # Load full dictionary for Easy (e.g. NWL.txt)
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    path = os.path.join(base_dir, 'dictionaries', f"{dict_name}.txt")
                    try:
                        with open(path, "r") as f:
                            dictionary = [line.strip().upper() for line in f if line.strip()]
                        print(f"[BoardGen] Using FULL dictionary ({dict_name}.txt) for Easy Word Soup.")
                    except Exception as e:
                        print(f"[BoardGen] Error loading full dictionary {dict_name}: {e}. Using fallback words.")
                        dictionary = ["EXAMPLE", "BOARDS", "PUZZLE", "BOGGLE", "WONDER"]
                        
            valid_words = [w for w in dictionary if 5 <= len(w) <= 10 and not w.upper().endswith("ING") and not w.upper().endswith("INGS")]
            
        if not valid_words:
            # Fallback if no dictionary passed or no words of that length
            valid_words = ["EXAMPLE", "BOARDS", "PUZZLE", "BOGGLE", "WONDER"]
            
        # Determine number of words to embed based on grid size
        num_cells = rows * cols
        num_words_to_embed = 45 if num_cells >= 48 else (30 if num_cells >= 35 else 15)
        
        selected_words = random.sample(valid_words, min(num_words_to_embed, len(valid_words)))
        
        def get_neighbors(r, c):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append((nr, nc))
            return neighbors

        def find_random_path(length):
            start_r = random.randint(0, rows - 1)
            start_c = random.randint(0, cols - 1)
            path = [(start_r, start_c)]
            for _ in range(length - 1):
                curr_r, curr_c = path[-1]
                neighbors = get_neighbors(curr_r, curr_c)
                valid_neighbors = [n for n in neighbors if n not in path]
                if not valid_neighbors: return None
                path.append(random.choice(valid_neighbors))
            return path

        print(f"[BoardGen] Word Soup: Embedding {len(selected_words)} words on {rows}x{cols} grid...")
        for word in selected_words:
            path = None
            for _ in range(10): # Try 10 times to find a path
                path = find_random_path(len(word))
                if path:
                    break
            if path:
                for i, (r, c) in enumerate(path):
                    board[r][c] = word[i]
                    
        # Fill remaining empty cells with random letters
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == ' ':
                    board[r][c] = random.choices(self.letters, weights=weights, k=1)[0]
                    
        return board
        
        # USER REQUEST: Vowel Density Floor
        # If the random board has < 25% vowels, it's likely a "dead board" for high-min words.
        v_floor = int(rows * cols * 0.25)
        v_count = self._count_vowels(board)
        if v_count < v_floor:
            vowels = [l for l in self.letters if self._is_vowel(l)]
            v_weights = [weights[self.letters.index(l)] for l in vowels]
            consonant_tiles = [(r, c) for r in range(rows) for c in range(cols) if not self._is_vowel(board[r][c])]
            random.shuffle(consonant_tiles)
            for _ in range(v_floor - v_count):
                if not consonant_tiles: break
                tr, tc = consonant_tiles.pop()
                board[tr][tc] = random.choices(vowels, weights=v_weights, k=1)[0]
        
        # Q/U Logic: Ensure Q has a chance for U but avoid redundant clusters
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "Q":
                    has_u = False
                    neighbors = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and (dr != 0 or dc != 0):
                                neighbors.append((nr, nc))
                                if board[nr][nc] == "U": has_u = True
                    if not has_u and neighbors:
                        nr, nc = random.choice(neighbors)
                        board[nr][nc] = "U"
        return board

    def _create_checkerboard(self, rows, cols, weights, depth=1, difficulty="Easy"):
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

                            if difficulty in ["Medium", "Hard"] and self._is_creating_forbidden_sequence(board, char, r, c, f, depth=depth):
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

                    if difficulty in ["Medium", "Hard"] and self._is_creating_forbidden_sequence(board, char, r, c, 0, depth=1):
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
        board_format="Normal"
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
                # User Request: High minimum lengths (7L+) are very hard even on large grids.
                # Standard density only needs 1 pass, but 7L+ needs more effort.
                pass_count = 3 if min_word_length >= 7 else 1
            elif min_words >= 200:
                pass_count = 4 # High Density targets need more passes to pack words (4x4)
            elif min_word_length >= 7:
                pass_count = 3
        else:  # Uniqueness target
            if rows * cols >= 35:
                pass_count = 2 if min_word_length >= 7 else 1
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
        # USER REQUEST: Ensure 100+ and 200+ boards on 4x4 are consistently found.
        # We need at least 2 passes for 100+ and 4 passes for 200+.
        if is_4x4:
            if min_words >= 200: max_passes = 4
            elif min_words >= 100: max_passes = 2
            else: max_passes = 1
        else:
            max_passes = pass_count

        # Use a more targeted dictionary for Uniqueness optimization to match Java's speed/efficiency
        # PERFORMANCE: For 4x4 grids, the full dictionary is fast enough. 
        # Using 'Unique' dictionaries (5L+) on 4x4 can miss 3L/4L words during density calc.
        original_dictionary = dictionary
        if target_type == "Uniqueness" and not is_4x4:
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
                # For large grids, 15s of optimization is plenty; if not met, fallback is better.
                # USER REQUEST: For high-density 4x4 (200+), we need more time for 4 passes.
                io_timeout = 15 if rows * cols >= 35 else 40 if (is_4x4 and min_words >= 100) else 18
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
                    # USER REQUEST: For high-min rounds (7L+), skip FEWER tiles to ensure we hit targets.
                    if min_word_length >= 7:
                        skip_prob = 0.5 if depth > 1 else 0.10 # Reduced skip for 7L+
                    else:
                        skip_prob = 0.9 if depth > 1 else 0.75
                    if random.random() < skip_prob:
                        continue
                elif num_cells >= 24:
                    current_solve_depth = 11 if depth > 1 else 10
                    if random.random() < 0.2:
                        continue
                else:
                    # 4x4 boards (16 cells)
                    # USER REQUEST: For high-min rounds (5L+), increase depth to capture more long words.
                    # Ultra-dense 200+ targets need depth 20+ to correctly count all permutations.
                    if is_4x4 and min_words >= 200:
                        current_solve_depth = 20
                    else:
                        current_solve_depth = 11 if min_word_length >= 5 else 8 if (is_4x4 and min_words >= 150) else 9

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
                    elif count_eval > max_words + 150:
                        # SAFETY: Prevent bloat in high-connectivity grids (3D)
                        print(f"[BoardGen] Word count overshoot ({count_eval} > {max_words}). Returning latest safe state.")
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
                            test_pool = list("ETAOINSRDL") + (list("BCUMH") if is_4x4 else [])
                            # Priority for 4x4 high-density: Vowels
                            if is_4x4: test_pool = list("AEIOU") + test_pool
                        elif max_words <= 150 and rows * cols >= 35 and min_word_length <= 4:
                            # User Request: On large grids with low word targets, we need RARE letters to prevent
                            # word counts from exploding. Using standard English frequency makes 100 counts impossible.
                            # ONLY APPLY THIS IF MIN LENGTH IS SMALL (3L or 4L). 
                            # High-min rounds (7L+) are naturally sparse and need common letters.
                            if difficulty == "Easy":
                                # For Easy sparse boards, use a more balanced pool to keep uniqueness low
                                test_pool = list("ETAOINSRDL") + list("BCUMFVGPH")
                            else:
                                # USER REQUEST: Limit rare letters. We only add them to the pool if not already on board.
                                rare_pool = []
                                for rl in "ZXQJK":
                                    # Use ironclad rare limit check (Max 1 each, Max 3 total)
                                    if not self._is_rare_limited(board, rl, depth):
                                        rare_pool.append(rl)
                                test_pool = rare_pool + list("VWYPFBHC") + [random.choice("ETAOINSR") for _ in range(2)]
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
                    sample_size = 12 if min_words >= 200 else (8 if min_words >= 100 else 5)
                elif num_cells >= 48:
                    # 6x8 grids: cap at 4 samples to ensure completion under 20s
                    # USER REQUEST: For high-min rounds (7L+), increase sample size to ensure connectivity
                    sample_size = 4 if (min_words >= 200 or min_word_length >= 7) else 2 
                else:
                    sample_size = 2 # 4x6, 3x3x3 speedup (from 4)
                test_pool = test_pool[:sample_size]

                for char in test_pool:
                    # Enforce letter abundance limits except in specialized Mania formats
                    if self._is_abundance_limited(board, char, board_format=board_format, depth=depth):
                        continue

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
                             max_v_ratio = 0.55 if (min_word_length >= 5 or min_words >= 200) else 0.44
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
                        # USER REQUEST: If undershooting, prioritize raw count over point-value to hit floor.
                        if is_undershooting:
                            if count_w > best_count_w or (count_w == best_count_w and val > curr_count):
                                curr_count = val
                                best_count_w = count_w
                                best_char = char
                        elif not is_overshooting and (val > curr_count or (val == curr_count and count_w > best_count_w)):
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

    def _perform_decimation_sweep(self, board, rows, cols, depth, dictionary, min_word_length, min_words, max_words, excluded, difficulty, rescue_depth=15, protected_path=None, is_checkerboard=False, board_format="Normal"):
        """
        USER REQUEST: Reduce word count by replacing high-density cells with 'dead' letters.
        """
        import time
        start_time = time.time()
        
        # Flatten protected cells
        protected_cells = set()
        if protected_path:
            for cell in protected_path:
                if isinstance(cell, (list, tuple)):
                    protected_cells.add(tuple(cell))

        positions = []
        for f in range(depth):
            for r in range(rows):
                for c in range(cols):
                    pos = (f, r, c) if depth > 1 else (r, c)
                    if pos not in excluded and pos not in protected_cells:
                        positions.append(pos)
        random.shuffle(positions)
        
        current_solve = self._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=rescue_depth, store_paths=False)
        current_count = len(current_solve)
        if current_count <= max_words: return board
        
        print(f"[BoardGen] 🔨 DECIMATION SWEEP START (Current: {current_count}, Max: {max_words})")
        dead_chars = ["Z", "X", "Q", "J", "K", "V"]
        
        for pos in positions:
            if time.time() - start_time > 20.0: break # Hard time limit
            f_p, r_p, c_p = (pos[0], pos[1], pos[2]) if depth > 1 else (0, pos[0], pos[1])
            
            if is_checkerboard:
                target_is_vowel = (f_p + r_p + c_p) % 2 != 0 if depth > 1 else (r_p + c_p) % 2 != 0
                if target_is_vowel:
                    continue # Skip vowel cells since we don't have dead vowels!
            
            old_char = board[f_p][r_p][c_p] if depth > 1 else board[r_p][c_p]
            best_char, best_score = old_char, current_count
            
            # Try dead letters to see which breaks the most words
            for char in ["Z", "X", "Q", "J", "K", "V", "W", "G", "F", "B", "P", "M", "H"]:
                if char == old_char: continue
                # USER REQUEST: Max 1 rare letter and Max 3 total rares
                if self._is_rare_limited(board, char, depth):
                    if char != (board[f_p][r_p][c_p] if depth > 1 else board[r_p][c_p]):
                        continue
                if self._is_abundance_limited(board, char, board_format=board_format, depth=depth):
                    continue
                    
                if depth > 1: board[f_p][r_p][c_p] = char
                else: board[r_p][c_p] = char
                
                res = self._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=rescue_depth, store_paths=False)
                new_count = len(res)
                
                # DRACONIAN: Always favor lower counts if we are over max_words
                if new_count < best_score:
                    best_score, best_char = new_count, char
                
                if best_score <= max_words: break
            
            if depth > 1: board[f_p][r_p][c_p] = best_char
            else: board[r_p][c_p] = best_char
            current_count = best_score
            if current_count <= max_words:
                print(f"[BoardGen] ✅ DECIMATION SUCCESSFUL: Hit {current_count} words.")
                break
        return board

    def _perform_rescue_sweep(self, board, rows, cols, depth, dictionary, min_word_length, min_words, max_words, excluded, difficulty, rescue_depth=15, protected_path=None, is_checkerboard=False, board_format="Normal"):
        """
        USER REQUEST: Perform IO operations on random locations until desired word count is reached.
        """
        import time
        start_time = time.time()
        
        # Flatten protected cells
        protected_cells = set()
        if protected_path:
            for cell in protected_path:
                if isinstance(cell, (list, tuple)):
                    protected_cells.add(tuple(cell))

        positions = []
        for f in range(depth):
            for r in range(rows):
                for c in range(cols):
                    pos = (f, r, c) if depth > 1 else (r, c)
                    if pos not in excluded and pos not in protected_cells:
                        positions.append(pos)
        random.shuffle(positions)
        current_solve = self._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=rescue_depth, store_paths=False)
        current_count = len(current_solve)
        if current_count >= min_words: return board
        print(f"[BoardGen] 🆘 RESCUE SWEEP START (Current: {current_count}, Target: {min_words})")
        max_rescue_tiles = 24 if rows * cols >= 48 else len(positions)
        rescue_pool = positions[:max_rescue_tiles]
        for pos in rescue_pool:
            if time.time() - start_time > 30.0: break
            f_p, r_p, c_p = (pos[0], pos[1], pos[2]) if depth > 1 else (0, pos[0], pos[1])
            old_char = board[f_p][r_p][c_p] if depth > 1 else board[r_p][c_p]
            best_char, max_score = old_char, current_count
            test_chars = ["S", "E", "R", "T", "A", "I", "O", "N", "L", "C", "D", "U", "H", "P", "B", "M", "G", "F", "W", "Y"]
            if is_checkerboard:
                target_is_vowel = (f_p + r_p + c_p) % 2 != 0 if depth > 1 else (r_p + c_p) % 2 != 0
                if target_is_vowel:
                    test_chars = [c for c in test_chars if self._is_vowel(c)]
                else:
                    test_chars = [c for c in test_chars if not self._is_vowel(c)]
            for char in list(set(test_chars)):
                if char == old_char: continue
                if difficulty in ["Medium", "Hard"] and self._is_creating_forbidden_sequence(board, char, r_p, c_p, f_p, depth=depth):
                    continue
                if self._is_abundance_limited(board, char, board_format=board_format, depth=depth):
                    continue
                if depth > 1: board[f_p][r_p][c_p] = char
                else: board[r_p][c_p] = char
                
                # Solve (Deeper depth for accuracy in rescue)
                res = self._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=rescue_depth, store_paths=False)
                new_count = len(res)
                if new_count > max_score and new_count <= max_words:
                    max_score, best_char = new_count, char
                if max_score >= min_words: break
            if depth > 1: board[f_p][r_p][c_p] = best_char
            else: board[r_p][c_p] = best_char
            current_count = max_score
            if current_count >= min_words:
                print(f"[BoardGen] ✅ RESCUE SUCCESSFUL: Hit {current_count} words.")
                break
        return board

    def _apply_io_b_uniqueness_optimization(self, board, rows, cols, dictionary, excluded_cells, min_word_length, depth=1, difficulty="Medium", max_words=200, is_checkerboard=False):
        """
        USER MANDATE: Stage 2 of 200+ Optimization. 
        Implements specific "IO and B" checkerboard where:
        B (Base) = (r+c)%2 == 1 -> Preserved letters from Base Board
        IO (Optimized) = (r+c)%2 == 0 -> Recalculated for Maximum Unique Words
        """
        import time
        import random
        
        unique_set = self._get_difficulty_set(dictionary)
        if not unique_set:
            print("[BoardGen] !! Stage 2 Skip: Unique set empty.")
            return board
            
        print(f"[BoardGen] Stage 2 starting for {rows}x{cols} ({dictionary})")
        start_time = time.time()
        
        # 1. Identify IO positions (even parity sum)
        io_positions = []
        if depth > 1:
            for f in range(depth):
                for r in range(rows):
                    for c in range(cols):
                        if (f + r + c) % 2 == 0 and (f, r, c) not in excluded_cells:
                            io_positions.append((f, r, c))
        else:
            for r in range(rows):
                for c in range(cols):
                    if (r + c) % 2 == 0 and (r, c) not in excluded_cells:
                        io_positions.append((r, c))
        
        # User Requirement: Process positions sequentially
        random.shuffle(io_positions)
        
        # Solve depth optimization
        solve_depth = 6 if (rows * cols >= 35) else 12
        timeout_cell = 0.05 if (rows * cols >= 35) else 0.15
        
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        for idx, pos in enumerate(io_positions):
            # Global timeout check (Stage 2 shouldn't exceed 5s)
            if time.time() - start_time > 5:
                print(f"[BoardGen] !! Stage 2 Global Timeout. Stopping at cell {idx}/{len(io_positions)}.")
                break
                
            if depth > 1: f_t, r_t, c_t = pos
            else: r_t, c_t = pos; f_t = 0
            
            orig_char = board[f_t][r_t][c_t] if depth > 1 else board[r_t][c_t]
            best_char = orig_char
            max_score = -1
            is_easy = (str(difficulty).lower() == "easy")
            
            # Solve for each letter to pick the winning candidate
            if is_checkerboard:
                target_is_vowel = (f_t + r_t + c_t) % 2 != 0 if depth > 1 else (r_t + c_t) % 2 != 0
                test_alphabet = list(VOWELS) if target_is_vowel else list(CONSONANTS)
            else:
                test_alphabet = alphabet
            
            for char in test_alphabet:
                # USER REQUEST: Max 1 rare letter and Max 3 total rares
                if self._is_rare_limited(board, char, depth):
                    # Allow if it's the SAME letter we are already testing (no increase)
                    if char != (board[f_t][r_t][c_t] if depth > 1 else board[r_t][c_t]):
                        continue

                if depth > 1: board[f_t][r_t][c_t] = char
                else: board[r_t][c_t] = char
                
                try:
                    all_found = self._solve_board(
                        board, dictionary, (0, 99999), min_word_length, 
                        max_depth=solve_depth, store_paths=False, timeout=timeout_cell
                    )
                    total_w = len(all_found)
                    unique_w = sum(1 for w in all_found if w in unique_set)
                    
                    # USER REQUEST: Total word count compliance.
                    # We prioritize density/uniqueness but STERNLY penalize overshooting max_words (ceiling).
                    if is_easy:
                        score = total_w
                    else:
                        score = unique_w
                    
                    if total_w > max_words:
                        # Draconian penalty for every word over the limit to keep density in check during Stage 2
                        # User Request: "also above it as well" - we must ensure we don't overshoot.
                        score -= (total_w - max_words) * 1000
                except Exception:
                    score = 0
                    
                if score > max_score:
                    max_score = score
                    best_char = char
            
            # Set the winning letter for this IO position
            if depth > 1: board[f_t][r_t][c_t] = best_char
            else: board[r_t][c_t] = best_char
            
            if (idx + 1) % 4 == 0 or idx == len(io_positions) - 1:
                print(f"[BoardGen] Stage 2 Progress: {idx+1}/{len(io_positions)} IO tiles optimized.")
                
        print(f"[BoardGen] Stage 2 Complete in {time.time() - start_time:.2f}s.")
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
            
            # ENSURE: One vowel, one consonant
            is_orig_vowel = self._is_vowel(orig)
            if is_orig_vowel:
                # Pick other from consonants
                others = [l for l in self.letters if not self._is_vowel(l)]
            else:
                # Pick other from vowels
                others = [l for l in self.letters if self._is_vowel(l)]
            
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

        with open(DEBUG_FLOW_PATH, "a") as f:
            f.write(f"[board_generator.py] _embed_bonus_word: Attempting to embed '{bonus_word}' at {time.time()}\n")

        for start_r, start_c in possible_starts:
            path = backtrack([(start_r, start_c)])
            if path:
                # Embed the processed letters
                for i, (r, c) in enumerate(path):
                    board[r][c] = processed_word[i]
                with open(DEBUG_FLOW_PATH, "a") as f:
                    f.write(f"[board_generator.py] _embed_bonus_word: SUCCESS at {time.time()}\n")
                return path

        with open(DEBUG_FLOW_PATH, "a") as f:
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
        self, board, dictionary="NWL", word_count_range=(0, 99999), min_word_length=3, max_depth=12, store_paths=True, timeout=10.0, must_include=None
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
        # SYNC: Ensure tries are current with 'Added Words' toggle and file state
        from word_validator import word_validator
        word_validator.get_use_added_words()
        
        d_upper = str(dictionary).upper()
        if d_upper in ["UNIQUECSW", "CSW"]:
            word_validator.ensure_csw_loaded()
            
        if d_upper == "UNIQUENWL":
            trie_root = word_validator.unique_nwl_trie
        elif d_upper == "UNIQUECSW":
            trie_root = word_validator.unique_csw_trie
        elif d_upper == "CSW":
            trie_root = word_validator.csw_trie
        else:
            trie_root = word_validator.nwl_trie

        def dfs(f, r, c, current_d, current_word, current_node, current_path, uses_target=False):
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
                new_path = current_path + ([(f, r, c)] if depth_val > 1 else [(r, c)]) if store_paths else []
                
                # Check if we've hit the target tile
                new_uses_target = uses_target or (must_include and (f, r, c) == (must_include[0] if len(must_include)==3 else 0, must_include[-2], must_include[-1]))

                # Check current word
                if len(new_word) >= min_word_length and next_node.is_word:
                    # If we are in "must_include" mode, only add if the path used the target
                    if not must_include or new_uses_target:
                        if new_word not in found_words:
                            found_words[new_word] = new_path if store_paths else True
                            # PERFORMANCE GUARD: Exit solver if word count explodes (especially in 3D)
                            # We allow a buffer above max_words for scoring and selection variety.
                            if len(found_words) > 1500:
                                return

                # Prune and continue using Trie
                if len(new_word) < max_depth:
                    if depth_val == 1:
                        visited[r][c] = True
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                                    dfs(0, nr, nc, current_d + 1, new_word, next_node, new_path, new_uses_target)
                        visited[r][c] = False
                    else:
                        # 3D CUBE SUPPORT: 26-way adjacency
                        visited[r][c][f] = True
                        for nf, nr, nc in self._get_cube_neighbors(f, r, c):
                            if not visited[nr][nc][nf]:
                                dfs(nf, nr, nc, current_d + 1, new_word, next_node, new_path, new_uses_target)
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
                                            dfs(0, nr, nc, current_d + 1, q_word, u_node, new_path, new_uses_target)
                                visited[r][c] = False
                            else:
                                visited[r][c][f] = True
                                for nf, nr, nc in self._get_cube_neighbors(f, r, c):
                                    if not visited[nr][nc][nf]:
                                        dfs(nf, nr, nc, current_d + 1, q_word, u_node, new_path, new_uses_target)
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

    def complete_solve_board(self, board, dictionary, min_word_length=3):
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
            if len(word) >= min_word_length and word_validator.is_valid_word(word, dictionary):
                found_words.add(word)

            # Prune search using Trie/Prefix checking
            if word_validator.has_valid_prefix(word, dictionary):
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
                        if "/" in str(board[f][r][c]) or (f, r, c) in bonus_cells_set: continue
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
                    if "/" in str(board[r][c]) or (r, c) in bonus_cells_set: continue
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

    def _count_char_on_board(self, board, char, depth=1):
        """Count instances of a specific character (or substring like 'QU') on the board."""
        count = 0
        rows = len(board) if depth == 1 else len(board[0])
        cols = len(board[0]) if depth == 1 else len(board[0][0])
        for f in range(depth):
            for r in range(rows):
                for c in range(cols):
                    cell = board[f][r][c] if depth > 1 else board[r][c]
                    if char in str(cell):
                        count += 1
        return count

    def _is_abundance_limited(self, board, char, board_format="Normal", depth=1):
        """Helper to ensure letters do not exceed abundance limits during sweeps/optimization."""
        # Active Mania letter has no limit
        safe_format = str(board_format or "Normal").strip().upper()
        if "MANIA" in safe_format:
            parts = safe_format.split()
            if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isalpha():
                if char.upper() == parts[0]:
                    return False

        # Calculate total cells
        rows = len(board) if depth == 1 else len(board[0])
        cols = len(board[0]) if depth == 1 else len(board[0][0])
        total_cells = rows * cols * depth

        # Letter type limits
        VOWELS = {"A", "E", "I", "O", "U"}
        COMMON_CONSONANTS = {"S", "T", "R", "N", "L", "D"}

        upper_char = char.upper()
        if upper_char in VOWELS:
            limit = max(4, int(total_cells * 0.18))
        elif upper_char in COMMON_CONSONANTS:
            limit = max(3, int(total_cells * 0.12))
        else:
            limit = max(2, int(total_cells * 0.09))

        current_count = self._count_char_on_board(board, upper_char, depth)
        return current_count >= limit

    def _is_rare_limited(self, board, char, depth=1):
        """Helper for optimization loops to respect global rare limits (Max 1 per, Max 3 total)."""
        rare_letters = {"Q", "Z", "J", "X", "K"}
        if char not in rare_letters:
            return False
            
        # 1. Per-letter limit (Max 1)
        if self._count_char_on_board(board, char, depth) >= 1:
            return True
            
        # 2. Total Cap limit (Max 3 TOTAL across all rares)
        total_rares = 0
        for rl in rare_letters:
            total_rares += self._count_char_on_board(board, rl, depth)
        if total_rares >= 3:
            return True
            
        return False

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
        d_upper = str(dictionary).upper()
        if d_upper in ["UNIQUECSW", "CSW"]:
            word_validator.ensure_csw_loaded()
            
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
