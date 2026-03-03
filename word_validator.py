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
        self.nwl_trie = TrieNode()
        self.csw_trie = TrieNode()
        self.long_trie = TrieNode()
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
        # Build a shared trie for the 16+ list (used regardless of dictionary choice)
        for word in self.long_words:
            self._add_to_trie(self.long_trie, word)
        print("Tries built successfully!")
    
    def _add_to_trie(self, root, word):
        """Add a word to the trie"""
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    def has_valid_prefix(self, prefix, dictionary='NWL'):
        """Check if prefix could lead to a valid word (O(k) where k is prefix length).
        Checks both the main dictionary trie AND the 16+ supplementary trie."""
        prefix = prefix.upper()
        trie = self.nwl_trie if dictionary == 'NWL' else self.csw_trie
        
        # Check main dictionary trie
        node = trie
        for char in prefix:
            if char not in node.children:
                break
            node = node.children[char]
        else:
            return True  # Prefix found in main dictionary

        # Check supplementary 16+ trie
        node = self.long_trie
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True  # Prefix found in 16+ list
    
    def is_valid_word(self, word, dictionary='NWL'):
        """Check if word is valid in specified dictionary OR in the 16+ supplementary list"""
        word = word.upper()
        if dictionary == 'CSW':
            return word in self.csw_words or word in self.long_words
        else:  # NWL
            return word in self.nwl_words or word in self.long_words
    
    def is_csw_only(self, word):
        """Check if word is in CSW but not NWL"""
        return word.upper() in self.csw_only
    
    def filter_valid_words(self, words, dictionary='NWL'):
        """Filter list to only valid words"""
        return [w for w in words if self.is_valid_word(w, dictionary)]
    
    def load_dictionary(self, dict_name):
        """Return the word set for the given dictionary name (used by tools/API routes).
        The 16+ supplementary words are merged in so callers automatically get them."""
        if dict_name == 'CSW':
            return self.csw_words | self.long_words
        else:  # NWL (default)
            return self.nwl_words | self.long_words

# Global instance
word_validator = WordValidator()
