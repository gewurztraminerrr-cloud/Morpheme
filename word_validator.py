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
        self.long_trie = TrieNode()
        self.added_trie = TrieNode()
        self._load_dictionaries()
    
    def _load_dictionaries(self):
        """Load both dictionaries and the 16+ supplementary list into memory"""
        base_dir = os.path.dirname(__file__)
        
        # Load NWL
        nwl_path = os.path.join(base_dir, 'dictionaries', 'NWL.txt')
        with open(nwl_path, 'r') as f:
            self.nwl_words = {line.strip().upper() for line in f if line.strip()}
        
        # Load CSW
        csw_path = os.path.join(base_dir, 'dictionaries', 'CSW.txt')
        with open(csw_path, 'r') as f:
            self.csw_words = {line.strip().upper() for line in f if line.strip()}

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
        
        # Calculate CSW-only words
        self.csw_only = self.csw_words - self.nwl_words
        
        print(f"Loaded {len(self.nwl_words)} NWL words and {len(self.csw_words)} CSW words")
        print(f"Found {len(self.csw_only)} CSW-only words")
        
        # Build tries for fast prefix checking
        print("Building tries for fast prefix checking...")
        for word in self.nwl_words:
            self._add_to_trie(self.nwl_trie, word)
        for word in self.csw_words:
            self._add_to_trie(self.csw_trie, word)
        # Build a shared trie for the 16+ list
        for word in self.long_words:
            self._add_to_trie(self.long_trie, word)
        # Custom words trie is built in reload_added_words()
        print("Tries built successfully!")
    
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
            print(f"Loaded {len(self.added_words)} custom added words")

    def _add_to_trie(self, root, word):
        """Add a word to the trie"""
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    def has_valid_prefix(self, prefix, dictionary='NWL'):
        """Check if prefix could lead to a valid word.
        Checks main dictionary, 16+ list, AND custom added words."""
        prefix = prefix.upper()
        trie = self.nwl_trie if dictionary == 'NWL' else self.csw_trie
        
        # Check main dictionary trie
        node = trie
        for char in prefix:
            if char not in node.children:
                break
            node = node.children[char]
        else:
            return True

        # Check supplementary 16+ trie
        node = self.long_trie
        for char in prefix:
            if char not in node.children:
                break
            node = node.children[char]
        else:
            return True
            
        # Check custom added words trie
        node = self.added_trie
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def is_valid_word(self, word, dictionary='NWL'):
        """Check if word is valid in dictionary, 16+ list, OR custom added list"""
        word = word.upper()
        is_added = word in self.added_words
        if dictionary == 'CSW':
            return word in self.csw_words or word in self.long_words or is_added
        else:  # NWL
            return word in self.nwl_words or word in self.long_words or is_added
    
    def is_csw_only(self, word):
        """Check if word is in CSW but not NWL"""
        return word.upper() in self.csw_only
        
    def is_added_word(self, word):
        """Check if word is in the custom moderator-added list"""
        return word.upper() in self.added_words
    
    def filter_valid_words(self, words, dictionary='NWL'):
        """Filter list to only valid words"""
        return [w for w in words if self.is_valid_word(w, dictionary)]
    
    def load_dictionary(self, dict_name):
        """Return the word set for the given dictionary name, including 16+ and custom words."""
        if dict_name == 'CSW':
            return self.csw_words | self.long_words | self.added_words
        else:  # NWL (default)
            return self.nwl_words | self.long_words | self.added_words

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

