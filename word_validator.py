"""
Word Validator for Morpheme Boggle Game
Loads NWL and CSW dictionaries and validates words.
Also loads a supplementary 16+ letter word list (16plus.txt) that is
always consulted in addition to whichever main dictionary is in use.
"""

import os

class TrieNode:
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
            except:
                self.use_added_words = True

    def _save_config(self):
        """Save global config for added words"""
        try:
            import json
            with open(self.config_path, 'w') as f:
                json.dump({'use_added_words': self.use_added_words}, f)
        except:
            pass

    def toggle_added_words(self, enabled):
        """Toggle added words and REBUILD Tries for immediate game-wide effect"""
        self.use_added_words = enabled
        self._save_config()
        # Full Rebuild to ensure Tries and sets are clean
        self._load_dictionaries()
        return self.use_added_words

    def get_use_added_words(self):
        """Actively read config and check for word list changes to sync across multiple Gunicorn workers"""
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
            
            if self.use_added_words:
                for word in self.added_words:
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
            if self.use_added_words:
                for word in self.added_words:
                    self._add_to_trie(trie, word)
                    
        self.csw_loaded = True
        # Recalculate full sets to include CSW
        self._recalculate_full_sets()
        print(f"[WordValidator] CSW loaded successfully! ({len(self.csw_words)} words)")
    
    def reload_added_words(self):
        """Reload custom added words from file and rebuild their trie"""
        base_dir = os.path.dirname(__file__)
        added_path = os.path.join(base_dir, 'dictionaries', 'added_words.txt')
        
        self.added_words = set()
        self.added_trie = TrieNode()
        
        if os.path.exists(added_path):
            with open(added_path, 'r') as f:
                for line in f:
                    word = line.strip().upper()
                    if word:
                        self.added_words.add(word)
                        self._add_to_trie(self.added_trie, word)
                        # CRITICAL: If enabled, inject these into the high-speed main tries so they appear on boards
                        if getattr(self, 'use_added_words', True):
                            self._add_to_trie(self.nwl_trie, word)
                            self._add_to_trie(self.csw_trie, word)
                            self._add_to_trie(self.unique_nwl_trie, word)
                            self._add_to_trie(self.unique_csw_trie, word)

            print(f"Loaded {len(self.added_words)} custom added words (Re-injected into main tries)")
            self._recalculate_full_sets()

    def _add_to_trie(self, root, word):
        """Add a word to the trie"""
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    def has_valid_prefix(self, prefix, dictionary='NWL'):
        """Check if prefix could lead to a valid word using pre-merged tries."""
        if dictionary == 'UniqueNWL':
            trie = self.unique_nwl_trie
        elif dictionary == 'UniqueCSW':
            self.ensure_csw_loaded()
            trie = self.unique_csw_trie
        elif dictionary == 'CSW':
            self.ensure_csw_loaded()
            trie = self.csw_trie
        else:
            trie = self.nwl_trie
        
        node = trie
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def is_valid_word(self, word, dictionary='NWL'):
        """Check if word is valid using pre-merged sets."""
        d_upper = str(dictionary).upper()
        if d_upper == 'UNIQUENWL':
            return word in self.unique_nwl_words or (self.use_added_words and word in self.added_words)
        elif d_upper == 'UNIQUECSW':
            self.ensure_csw_loaded()
            return word in self.unique_csw_words or (self.use_added_words and word in self.added_words)
        elif d_upper == 'CSW':
            self.ensure_csw_loaded()
            return word in self.csw_words or word in self.long_words or (self.use_added_words and word in self.added_words)
        else:  # NWL
            return word in self.nwl_words or word in self.long_words or (self.use_added_words and word in self.added_words)
    
    def is_csw_only(self, word):
        """Check if word is in CSW but not NWL"""
        self.ensure_csw_loaded()
        return word.upper() in self.csw_only
        
    def is_added_word(self, word):
        """Check if word is in the custom moderator-added list"""
        return word.upper() in self.added_words

    def is_valid_word_authoritative(self, word):
        """Check if word is in ANY official dictionary (excluding Added Words)"""
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

    def load_dictionary(self, dict_name):
        """Return the pre-calculated full set for the given dictionary."""
        if dict_name == 'CSW':
            self.ensure_csw_loaded()
            return self.full_csw_set
        else:  # NWL (default)
            return self.full_nwl_set

    def find_word_on_board(self, board, word, return_path=False):
        """Standard DFS search for a word on a Boggle board.
           Supports Either/Or cells (e.g. 'A/B') and Qu (Q matches QU)."""
        if not word or not board:
            return (False, None) if return_path else False
            
        word = word.upper()
        rows = len(board)
        cols = len(board[0])
        
        def dfs(r, c, index, visited_path):
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return None
            if (r, c) in visited_path:
                return None
                
            cell_val = str(board[r][c]).upper()
            # Support Either/Or tiles
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
                            res_path = dfs(r + dr, c + dc, index + match_len, current_path)
                            if res_path: return res_path
            return None

        for r in range(rows):
            for c in range(cols):
                path = dfs(r, c, 0, [])
                if path:
                    return (True, path) if return_path else True
        return (False, None) if return_path else False

# Global instance
word_validator = WordValidator()

