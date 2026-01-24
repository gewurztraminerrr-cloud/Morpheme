"""
Word Validator for Morpheme Boggle Game
Loads NWL and CSW dictionaries and validates words
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
        self.nwl_trie = TrieNode()
        self.csw_trie = TrieNode()
        self._load_dictionaries()
    
    def _load_dictionaries(self):
        """Load both dictionaries into memory"""
        base_dir = os.path.dirname(__file__)
        
        # Load NWL
        nwl_path = os.path.join(base_dir, 'dictionaries', 'NWL.txt')
        with open(nwl_path, 'r') as f:
            self.nwl_words = {line.strip().upper() for line in f if line.strip()}
        
        # Load CSW
        csw_path = os.path.join(base_dir, 'dictionaries', 'CSW.txt')
        with open(csw_path, 'r') as f:
            self.csw_words = {line.strip().upper() for line in f if line.strip()}
        
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
        """Check if prefix could lead to a valid word (O(k) where k is prefix length)"""
        prefix = prefix.upper()
        trie = self.nwl_trie if dictionary == 'NWL' else self.csw_trie
        
        node = trie
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True  # Prefix exists in trie
    
    def is_valid_word(self, word, dictionary='NWL'):
        """Check if word is valid in specified dictionary"""
        word = word.upper()
        if dictionary == 'CSW':
            return word in self.csw_words
        else:  # NWL
            return word in self.nwl_words
    
    def is_csw_only(self, word):
        """Check if word is in CSW but not NWL"""
        return word.upper() in self.csw_only
    
    def filter_valid_words(self, words, dictionary='NWL'):
        """Filter list to only valid words"""
        return [w for w in words if self.is_valid_word(w, dictionary)]

# Global instance
word_validator = WordValidator()
