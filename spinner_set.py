"""
Spinner Set System for Accumulative Game Mode
Generates randomized parameters for each round
"""

import random

from word_validator import word_validator

class SpinnerSet:
    @staticmethod
    def generate_params(board_dimensions):
        """Generate all spinner parameters for a round"""
        # Generate dictionary FIRST so we can use its size for word count
        dictionary = SpinnerSet._spin_dictionary()
        
        return {
            'bonus_word_length': SpinnerSet._spin_bonus_word_length(),
            'min_word_length': SpinnerSet._spin_min_word_length(board_dimensions),
            'difficulty': SpinnerSet._spin_difficulty(),
            'word_count_range': SpinnerSet._spin_word_count(dictionary),
            'dictionary': dictionary,
            'board_format': SpinnerSet._spin_board_format()
        }
    
    @staticmethod
    def _spin_bonus_word_length():
        """33.3% each for 8, 9, 10 letters"""
        return random.choice([8, 9, 10])
    
    @staticmethod
    def _spin_min_word_length(board_dimensions):
        """Based on board dimensions with specified percentages"""
        dims = board_dimensions.lower()
        
        if dims == '4x4':
            return random.choices([3, 4, 5], weights=[25, 50, 25])[0]
        elif dims == '4x6':
            return random.choices([4, 5, 6], weights=[25, 50, 25])[0]
        elif dims == '5x7':
            return random.choices([5, 6, 7], weights=[25, 50, 25])[0]
        elif dims == '6x8':
            return random.choices([6, 7, 8], weights=[25, 50, 25])[0]
        else:
            return 3  # Default
    
    @staticmethod
    def _spin_difficulty():
        """25% Easy, 50% Medium, 25% Hard"""
        return random.choices(['Easy', 'Medium', 'Hard'], weights=[25, 50, 25])[0]
    
    @staticmethod
    def _spin_word_count(dictionary_name='NWL'):
        """25% 50-100, 50% 100-200, 25% 200+ (max dict size)"""
        # Get actual size of selected dictionary
        if dictionary_name == 'CSW':
            max_words = len(word_validator.csw_words)
        else:
            max_words = len(word_validator.nwl_words)
            
        ranges = [
            (50, 100),
            (100, 200),
            (200, max_words)
        ]
        return random.choices(ranges, weights=[25, 50, 25])[0]
    
    @staticmethod
    def _spin_dictionary():
        """50% NWL, 50% CSW"""
        return random.choice(['NWL', 'CSW'])
    
    @staticmethod
    def _spin_board_format():
        """95% Normal, 5% Checkerboard"""
        return random.choices(['Normal', 'Checkerboard'], weights=[95, 5])[0]

# Test
if __name__ == '__main__':
    print("Testing Spinner Set:")
    for dim in ['4x4', '4x6', '5x7', '6x8']:
        params = SpinnerSet.generate_params(dim)
        print(f"\n{dim}: {params}")
