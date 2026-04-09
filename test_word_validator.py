import sys
import os
import time

# Add current path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from word_validator import word_validator

def test_unique():
    print(f"UniqueNWL size: {len(word_validator.unique_nwl_words)}")
    print(f"Is ABAMP in UniqueNWL? {'ABAMP' in word_validator.unique_nwl_words}")
    print(f"is_valid_word('ABAMP', 'UniqueNWL'): {word_validator.is_valid_word('ABAMP', 'UniqueNWL')}")
    print(f"has_valid_prefix('ABA', 'UniqueNWL'): {word_validator.has_valid_prefix('ABA', 'UniqueNWL')}")

if __name__ == "__main__":
    test_unique()
