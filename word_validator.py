"""
Word Validator for Morpheme Boggle Game
Loads NWL and CSW dictionaries and validates words.
Also loads a supplementary 16+ letter word list (16plus.txt) that is
always consulted in addition to whichever main dictionary is in use.
"""

import os
import contextvars

# Context variable to specify if added words are enabled for the current thread/context
use_added_words_ctx = contextvars.ContextVar('use_added_words', default=None)

class TrieNode:
    __slots__ = ('children', 'is_word')
    def __init__(self):
        self.children = {}
        self.is_word = False

class WordValidator:
    def __init__(self):
        self.nwl_words = set()
        self.csw_words = set()
        self.csw_only = set()
        self.long_words = set()   # 16+ letter supplementary list
        self.added_words = set()  # Custom moderator-added words
        self.nwl_trie = TrieNode()
        self.csw_trie = TrieNode()
        self.unique_nwl_trie = TrieNode()
        self.unique_csw_trie = TrieNode()
        self.long_trie = TrieNode()
        self.added_trie = TrieNode()
        
        self.use_added_words = True # Global toggle for moderator words
        self.base_path = os.path.join(os.path.dirname(__file__), 'dictionaries')
        self.config_path = os.path.join(self.base_path, 'added_words_config.json')
        
        self._load_config()
        self._load_dictionaries()

    def _load_config(self):
        """Load global config for added words"""
        if os.path.exists(self.config_path):
            try:
                import json
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.use_added_words = config.get('use_added_words', True)
            except Exception as e:
                import traceback
                print(f"[WordValidator] Error loading config: {e}\n{traceback.format_exc()}")
                self.use_added_words = True

    def _save_config(self):
        """Save global config for added words atomically"""
        try:
            import json
            import tempfile
            # Atomic write to prevent concurrent read/write corruption
            fd, temp_path = tempfile.mkstemp(dir=self.base_path, suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump({'use_added_words': self.use_added_words}, f)
                os.replace(temp_path, self.config_path)
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
        except Exception as e:
            import traceback
            print(f"[WordValidator] Error saving config: {e}\n{traceback.format_exc()}")
            raise e

    def toggle_added_words(self, enabled):
        """Toggle added words and REBUILD Tries for immediate game-wide effect"""
        self.use_added_words = enabled
        self._save_config()
        # Full Rebuild to ensure Tries and sets are clean
        self._load_dictionaries()
        return self.use_added_words

    def get_use_added_words(self, force=False):
        """Actively read config and check for word list changes to sync across multiple Gunicorn workers"""
        import time
        now = time.time()
        last_check = getattr(self, '_last_config_check', 0)
        if not force and (now - last_check < 2.0):
            return self.use_added_words
        self._last_config_check = now

        old_val = getattr(self, 'use_added_words', True)
        self._load_config()
        
        # Check added_words.txt timestamp for changes even if toggle didn't change
        added_path = os.path.join(self.base_path, 'added_words.txt')
        curr_mtime = 0
        if os.path.exists(added_path):
            curr_mtime = os.path.getmtime(added_path)
        
        old_mtime = getattr(self, '_added_words_mtime', 0)
        
        if old_val != self.use_added_words or curr_mtime != old_mtime:
            # Sync happened or file changed, we must rebuild locally too!
            self._added_words_mtime = curr_mtime
            if old_val != self.use_added_words:
                 # Full dictionary rebuild only if toggle changed
                 self._load_dictionaries()
            else:
                 # Just words list changed, lightweight reload
                 self.reload_added_words()
                 
        return self.use_added_words
    
    def _load_dictionaries(self):
        """Load NWL and 16+ supplementary list into memory. CSW loaded on demand."""
        base_dir = os.path.dirname(__file__)
        self.csw_loaded = False
        
        # Load NWL
        nwl_path = os.path.join(base_dir, 'dictionaries', 'NWL.txt')
        with open(nwl_path, 'r') as f:
            self.nwl_words = {line.strip().upper() for line in f if line.strip()}
            
        # Load custom_nwl if exists
        custom_nwl_path = os.path.join(base_dir, 'dictionaries', 'custom_nwl.txt')
        if os.path.exists(custom_nwl_path):
            with open(custom_nwl_path, 'r') as f:
                custom_nwl = {line.strip().upper() for line in f if line.strip()}
                self.nwl_words.update(custom_nwl)
        
        # Initialize empty sets for CSW
        self.csw_words = set()
        self.csw_only = set()
        self.unique_csw_words = set()

        # Load 16+ supplementary list
        long_path = os.path.join(base_dir, 'dictionaries', '16plus.txt')
        if os.path.exists(long_path):
            with open(long_path, 'r') as f:
                self.long_words = {line.strip().upper() for line in f if line.strip()}
            print(f"Loaded {len(self.long_words)} supplementary 16+ words")
        else:
            print("[WordValidator] Warning: 16plus.txt not found – skipping supplementary list")
            
        # Load custom added words
        self.reload_added_words()
        
        # Load Unique NWL
        un_path = os.path.join(base_dir, 'dictionaries', 'uniqueNWL.txt')
        if os.path.exists(un_path):
            with open(un_path, 'r') as f:
                self.unique_nwl_words = {line.strip().upper() for line in f if line.strip()}
        else:
            self.unique_nwl_words = set()

        print(f"Loaded {len(self.nwl_words)} NWL words and unique sets ({len(self.unique_nwl_words)})")
        
        # Pre-cache words by length for fast bonus word lookup
        self.nwl_by_len = {}
        self.csw_by_len = {}
        for w in self.nwl_words:
            length = len(w)
            if length not in self.nwl_by_len: self.nwl_by_len[length] = []
            self.nwl_by_len[length].append(w)
        
        # Reset tries for a fresh build
        self.nwl_trie = TrieNode()
        self.csw_trie = TrieNode()
        self.unique_nwl_trie = TrieNode()
        self.unique_csw_trie = TrieNode()
        
        # Build tries for fast prefix checking (NWL only for now)
        print("Building tries (clean build) for fast prefix checking (NWL)...")
        indices = [
            (self.nwl_trie, self.nwl_words),
            (self.unique_nwl_trie, self.unique_nwl_words)
        ]
        
        for trie, word_set in indices:
            for word in word_set:
                self._add_to_trie(trie, word)
            # Merge supplementary lists into main tries
            for word in self.long_words:
                self._add_to_trie(trie, word)
                
        print("NWL Tries built and merged successfully!")
        
        # Pre-calculate full sets for O(1) union performance
        self._recalculate_full_sets()

    def ensure_csw_loaded(self):
        """Lazy-load CSW dictionary if not already loaded"""
        if getattr(self, 'csw_loaded', False):
            return
            
        print("[WordValidator] Lazy-loading CSW dictionary...")
        base_dir = os.path.dirname(__file__)
        
        # Load CSW
        csw_path = os.path.join(base_dir, 'dictionaries', 'CSW.txt')
        if os.path.exists(csw_path):
            with open(csw_path, 'r') as f:
                self.csw_words = {line.strip().upper() for line in f if line.strip()}
        else:
            print("[WordValidator] Error: CSW.txt not found!")
            self.csw_words = set()
            
        # Load custom_csw if exists
        custom_csw_path = os.path.join(base_dir, 'dictionaries', 'custom_csw.txt')
        if os.path.exists(custom_csw_path):
            with open(custom_csw_path, 'r') as f:
                custom_csw = {line.strip().upper() for line in f if line.strip()}
                self.csw_words.update(custom_csw)
            
        # Calculate CSW-only words
        self.csw_only = self.csw_words - self.nwl_words

        # Load Unique CSW
        uc_path = os.path.join(base_dir, 'dictionaries', 'uniqueCSW.txt')
        if os.path.exists(uc_path):
            with open(uc_path, 'r') as f:
                self.unique_csw_words = {line.strip().upper() for line in f if line.strip()}
        else:
            self.unique_csw_words = set()
            
        # Pre-cache words by length
        for w in self.csw_words:
            length = len(w)
            if length not in self.csw_by_len: self.csw_by_len[length] = []
            self.csw_by_len[length].append(w)
            
        # Build tries for CSW
        print("[WordValidator] Building CSW tries...")
        indices = [
            (self.csw_trie, self.csw_words),
            (self.unique_csw_trie, self.unique_csw_words)
        ]
        
        for trie, word_set in indices:
            for word in word_set:
                self._add_to_trie(trie, word)
            for word in self.long_words:
                self._add_to_trie(trie, word)
                    
        self.csw_loaded = True
        self._filter_added_words()
        # Recalculate full sets to include CSW
        self._recalculate_full_sets()
        print(f"[WordValidator] CSW loaded successfully! ({len(self.csw_words)} words)")
    
    def reload_added_words(self):
        """Reload custom added words from file and rebuild their trie"""
        base_dir = os.path.dirname(__file__)
        added_path = os.path.join(base_dir, 'dictionaries', 'added_words.txt')
        
        self.added_words = set()
        self.added_words_list = []
        self.added_trie = TrieNode()
        
        if os.path.exists(added_path):
            with open(added_path, 'r') as f:
                raw_lines = [line.strip().upper() for line in f if line.strip()]
                seen = set()
                for w in raw_lines:
                    if w not in seen:
                        seen.add(w)
                        self.added_words_list.append(w)
                        self.added_words.add(w)

            print(f"Loaded {len(self.added_words)} custom added words as standalone trie")
            self._filter_added_words()
            self.added_words_list = [w for w in self.added_words_list if w in self.added_words]
            self._recalculate_full_sets()
            
            # Now build added_trie strictly using the filtered self.added_words!
            for w in self.added_words:
                self._add_to_trie(self.added_trie, w)

    def add_word_in_memory(self, word):
        """Add a word to the in-memory structures instantly and thread-safely."""
        word = word.upper().strip()
        if word not in self.added_words:
            self.added_words.add(word)
            self.added_words_list.insert(0, word)
            self._add_to_trie(self.added_trie, word)
            self._recalculate_full_sets()

    def remove_word_in_memory(self, word):
        """Remove a word from the in-memory structures instantly and thread-safely."""
        word = word.upper().strip()
        if word in self.added_words:
            self.added_words.remove(word)
            if word in self.added_words_list:
                self.added_words_list.remove(word)
            
            # Rebuild trie clean
            self.added_trie = TrieNode()
            for w in self.added_words:
                self._add_to_trie(self.added_trie, w)
            self._recalculate_full_sets()


    def _add_to_trie(self, root, word):
        """Add a word to the trie"""
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    def has_valid_prefix(self, prefix, dictionary='NWL', use_added_words=None):
        """Check if prefix could lead to a valid word using pre-merged tries."""
        if use_added_words is None:
            self.get_use_added_words()
            val = use_added_words_ctx.get()
            if val is None:
                val = False
        else:
            val = use_added_words

        d_upper = str(dictionary).upper()
        if d_upper == 'UNIQUENWL':
            trie = self.unique_nwl_trie
        elif d_upper == 'UNIQUECSW':
            self.ensure_csw_loaded()
            trie = self.unique_csw_trie
        elif d_upper == 'CSW':
            self.ensure_csw_loaded()
            trie = self.csw_trie
        elif d_upper == 'AW' or d_upper == 'ADDED_WORDS':
            self.ensure_csw_loaded()
            return self.has_valid_prefix(prefix, 'CSW', use_added_words=val) or self.has_valid_prefix(prefix, '_ONLY_ADDED_', use_added_words=val)
        elif d_upper == '_ONLY_ADDED_':
            trie = self.added_trie
        else:
            trie = self.nwl_trie
        
        node = trie
        for char in prefix:
            if char not in node.children:
                if val and d_upper in ['NWL', 'CSW']:
                    return self.has_valid_prefix(prefix, '_ONLY_ADDED_', use_added_words=False)
                return False
            node = node.children[char]
        return True
    
    def is_valid_word(self, word, dictionary='NWL', use_added_words=None):
        """Check if word is valid using pre-merged sets."""
        d_upper = str(dictionary).upper()
        has_aw = ('+ AW' in d_upper) or ('+AW' in d_upper) or (d_upper in ['AW', 'ADDED_WORDS', 'ALL', 'ALL + AW', 'ALL+AW'])
        if use_added_words is None:
            self.get_use_added_words()
            val = use_added_words_ctx.get()
            if val is None:
                val = has_aw or getattr(self, 'use_added_words', True)
        else:
            val = use_added_words or has_aw
            
        if 'CSW' in d_upper and 'NWL' not in d_upper and 'ALL' not in d_upper:
            self.ensure_csw_loaded()
            return word in self.csw_words or word in self.long_words or (val and word in self.added_words)
        elif d_upper in ['ALL', 'ALL + AW', 'ALL+AW', 'AW', 'ADDED_WORDS'] or ('ALL' in d_upper) or ('AW' in d_upper and 'NWL' not in d_upper and 'CSW' not in d_upper):
            self.ensure_csw_loaded()
            return word in self.nwl_words or word in self.csw_words or word in self.long_words or word in self.added_words
        else:  # NWL
            return word in self.nwl_words or word in self.long_words or (val and word in self.added_words)
    
    def is_csw_only(self, word):
        """Check if word is in CSW but not NWL"""
        # PERFORMANCE FIX: Do not force a 20-second lazy-load of the CSW dictionary 
        # just to check a word in an NWL room (which will never be CSW-only anyway).
        if not getattr(self, 'csw_loaded', False):
            return False
        return word.upper() in self.csw_only
        
    def is_added_word(self, word):
        """Check if word is in the custom moderator-added list and NOT in standard NWL or CSW"""
        return word.upper() in self.added_words

    def is_valid_word_authoritative(self, word):
        """Check if word is in ANY official dictionary (excluding Added Words)"""
        self.ensure_csw_loaded()
        w = word.upper()
        return w in self.nwl_words or w in self.csw_words or w in self.long_words
    
    def filter_valid_words(self, words, dictionary='NWL'):
        """Filter list to only valid words"""
        return [w for w in words if self.is_valid_word(w, dictionary)]
    
    def _recalculate_full_sets(self):
        """Update the pre-calculated full sets (union of main, long, and added words)"""
        if self.use_added_words:
            self.full_nwl_set = self.nwl_words | self.long_words | self.added_words
            self.full_csw_set = self.csw_words | self.long_words | self.added_words
        else:
            self.full_nwl_set = self.nwl_words | self.long_words
            self.full_csw_set = self.csw_words | self.long_words

    def _filter_added_words(self):
        """Filter out standard words from added_words set to keep it clean and fast"""
        if self.nwl_words:
            self.added_words = self.added_words - self.nwl_words
        if self.long_words:
            self.added_words = self.added_words - self.long_words
        if getattr(self, 'csw_loaded', False) and self.csw_words:
            self.added_words = self.added_words - self.csw_words

    def load_dictionary(self, dict_name, use_added_words=None):
        """Return the pre-calculated full set for the given dictionary."""
        if use_added_words is None:
            use_added_words = self.use_added_words
            
        d_upper = str(dict_name).upper()
        if d_upper == 'AW' or d_upper == 'ADDED_WORDS':
            return self.added_words
        elif d_upper == 'CSW':
            self.ensure_csw_loaded()
            if use_added_words == self.use_added_words:
                return self.full_csw_set
            return self.csw_words | self.long_words | (self.added_words if use_added_words else set())
        else:  # NWL (default)
            if use_added_words == self.use_added_words:
                return self.full_nwl_set
            return self.nwl_words | self.long_words | (self.added_words if use_added_words else set())

    def find_word_on_board(self, board, word, return_path=False):
        """Standard DFS search for a word on a Boggle board.
           Supports Either/Or cells (e.g. 'A/B') and Qu (Q matches QU).
           OPTIMIZED: First-letter index filtering for instant < 1ms search."""
        if not word or not board:
            return (False, None) if return_path else False
            
        word = word.upper()
        is_3d = isinstance(board[0], list) and isinstance(board[0][0], list)
        first_char = word[0]
        
        # 1. Quick initial scan to find valid starting coordinates for word[0]
        starting_cells = []
        if is_3d:
            for f in range(len(board)):
                for r in range(len(board[f])):
                    for c in range(len(board[f][r])):
                        val = str(board[f][r][c]).upper()
                        if first_char in val or (first_char == 'Q' and 'QU' in val) or ('/' in val and any(first_char == opt.strip()[0] for opt in val.split('/'))):
                            starting_cells.append((f, r, c))
        else:
            rows, cols = len(board), len(board[0])
            for r in range(rows):
                for c in range(cols):
                    val = str(board[r][c]).upper()
                    if first_char in val or (first_char == 'Q' and 'QU' in val) or ('/' in val and any(first_char == opt.strip()[0] for opt in val.split('/'))):
                        starting_cells.append((r, c))

        if not starting_cells:
            return (False, None) if return_path else False
            
        # 2. Run DFS ONLY starting from candidate cells
        if not is_3d:
            rows, cols = len(board), len(board[0])
            steps = 0
            max_steps = 25000
            
            def dfs_2d(r, c, index, visited_path):
                nonlocal steps
                steps += 1
                if steps > max_steps:
                    return None
                if r < 0 or r >= rows or c < 0 or c >= cols or (r, c) in visited_path:
                    return None
                cell_val = str(board[r][c]).upper()
                letters = cell_val.split('/') if '/' in cell_val else [cell_val]
                for char in letters:
                    match_len = 0
                    if char == 'Q' and word[index:index+2] == 'QU': 
                        match_len = 2
                    elif word.startswith(char, index): 
                        match_len = len(char)
                    
                    if match_len > 0:
                        current_path = visited_path + [(r, c)]
                        if index + match_len >= len(word):
                            return current_path
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                res_path = dfs_2d(r + dr, c + dc, index + match_len, current_path)
                                if res_path: return res_path
                return None

            for r, c in starting_cells:
                steps = 0
                path = dfs_2d(r, c, 0, [])
                if path:
                    return (True, path) if return_path else True
        else:
            # 3D Board DFS Search
            def get_neighbors_3d(f, r, c):
                res = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 3 and 0 <= nc < 3: res.append((f, nr, nc))
                if f == 0:
                    if r == 0: res.extend([(4, 2, c), (4, 2, c-1), (4, 2, c+1)])
                    if r == 2: res.extend([(5, 0, c), (5, 0, c-1), (5, 0, c+1)])
                    if c == 0: res.extend([(2, r, 2), (2, r-1, 2), (2, r+1, 2)])
                    if c == 2: res.extend([(3, r, 0), (3, r-1, 0), (3, r+1, 0)])
                elif f == 1:
                    if r == 0: res.extend([(4, 0, 2-c), (4, 0, 2-(c-1)), (4, 0, 2-(c+1))])
                    if r == 2: res.extend([(5, 2, 2-c), (5, 2, 2-(c-1)), (5, 2, 2-(c+1))])
                    if c == 0: res.extend([(3, r, 2), (3, r-1, 2), (3, r+1, 2)])
                    if c == 2: res.extend([(2, r, 0), (2, r-1, 0), (2, r+1, 0)])
                elif f == 2:
                    if r == 0: res.extend([(4, c, 0), (4, c-1, 0), (4, c+1, 0)])
                    if r == 2: res.extend([(5, 2-c, 0), (5, 2-(c-1), 0), (5, 2-(c+1), 0)])
                    if c == 0: res.extend([(1, r, 2), (1, r-1, 2), (1, r+1, 2)])
                    if c == 2: res.extend([(0, r, 0), (0, r-1, 0), (0, r+1, 0)])
                elif f == 3:
                    if r == 0: res.extend([(4, 2-c, 2), (4, 2-(c-1), 2), (4, 2-(c+1), 2)])
                    if r == 2: res.extend([(5, c, 2), (5, c-1, 2), (5, c+1, 2)])
                    if c == 0: res.extend([(0, r, 2), (0, r-1, 2), (0, r+1, 2)])
                    if c == 2: res.extend([(1, r, 0), (1, r-1, 0), (1, r+1, 0)])
                elif f == 4:
                    if r == 0: res.extend([(1, 0, 2-c), (1, 0, 2-(c-1)), (1, 0, 2-(c+1))])
                    if r == 2: res.extend([(0, 0, c), (0, 0, c-1), (0, 0, c+1)])
                    if c == 0: res.extend([(2, 0, r), (2, 0, r-1), (2, 0, r+1)])
                    if c == 2: res.extend([(3, 0, 2-r), (3, 0, 2-(r-1)), (3, 0, 2-(r+1))])
                elif f == 5:
                    if r == 0: res.extend([(0, 2, c), (0, 2, c-1), (0, 2, c+1)])
                    if r == 2: res.extend([(1, 2, 2-c), (1, 2, 2-(c-1)), (1, 2, 2-(c+1))])
                    if c == 0: res.extend([(2, 2, 2-r), (2, 2, 2-(r-1)), (2, 2, 2-(r+1))])
                    if c == 2: res.extend([(3, 2, r), (3, 2, r-1), (3, 2, r+1)])
                return [(nf, nr, nc) for nf, nr, nc in res if 0 <= nf < 6 and 0 <= nr < 3 and 0 <= nc < 3]

            def dfs_3d(f, r, c, index, visited_path):
                cell_val = str(board[f][r][c]).upper()
                letters = cell_val.split('/') if '/' in cell_val else [cell_val]
                for char in letters:
                    match_len = 0
                    if char == 'Q' and word[index:index+2] == 'QU': match_len = 2
                    elif word.startswith(char, index): match_len = len(char)
                    
                    if match_len > 0:
                        current_path = visited_path + [(f, r, c)]
                        if index + match_len >= len(word):
                            return current_path
                        for nf, nr, nc in get_neighbors_3d(f, r, c):
                            if (nf, nr, nc) not in visited_path:
                                res_path = dfs_3d(nf, nr, nc, index + match_len, current_path)
                                if res_path: return res_path
                return None

            for f, r, c in starting_cells:
                path = dfs_3d(f, r, c, 0, [])
                if path:
                    return (True, path) if return_path else True

        return (False, None) if return_path else False

# Global instance
word_validator = WordValidator()

# Pre-load CSW dictionary in a background thread on startup to prevent lobby transition delay
import threading
threading.Thread(target=word_validator.ensure_csw_loaded, daemon=True).start()

