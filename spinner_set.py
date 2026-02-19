"""
Spinner Set System for Accumulative Game Mode
Generates randomized parameters for each round
"""

import random

from word_validator import word_validator

class SpinnerSet:
    @staticmethod
    def generate_tournament_params():
        """Generate ALL tournament parameters using lobby-standard dimensions and times"""
        # 1. Broad Parameters (matching lobby options for mobile compatibility)
        board_dims = random.choice(['4x4', '4x6']) # Sticking to rectangular mobile-friendly dims
        time_limit = random.choice([45, 60, 120, 180, 300]) # Standard lobby times
        
        # 2. Granular Parameters
        params = SpinnerSet.generate_params(board_dims, is_24h=False)
        
        # Merge
        params['board_dimensions'] = board_dims
        params['time_limit'] = time_limit
        return params

    @staticmethod
    def generate_params(board_dimensions, is_24h=False):
        """Generate granular spinner parameters given dimensions"""
        # Generate dictionary FIRST so we can use its size for word count
        dictionary = SpinnerSet._spin_dictionary()
        
        return {
            'bonus_word_length': SpinnerSet._spin_bonus_word_length(),
            'min_word_length': SpinnerSet._spin_min_word_length(board_dimensions),
            'difficulty': SpinnerSet._spin_difficulty(),
            'word_count_range': SpinnerSet._spin_word_count(dictionary),
            'dictionary': dictionary,
            'board_format': SpinnerSet._spin_board_format(is_24h)
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
    def _spin_board_format(is_24h=False):
        """
        Normal rooms: 90% Normal, 5% Checkerboard, 3% Penalty, 2% [letter] Mania
        24h rooms: 95% Normal, 5% Checkerboard (No Penalty or Mania)
        """
        if is_24h:
            return random.choices(['Normal', 'Checkerboard'], weights=[95, 5])[0]
        else:
            result = random.choices(
                ['Normal', 'Checkerboard', 'Penalty', 'Mania'],
                weights=[90, 5, 3, 2]
            )[0]
            if result == 'Mania':
                # Pick a random letter for Mania mode
                mania_letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                return f'{mania_letter} Mania'
            return result

# Test
if __name__ == '__main__':
    print("Testing Spinner Set (Normal Rooms):")
    for dim in ['4x4', '4x6']:
        params = SpinnerSet.generate_params(dim, is_24h=False)
        print(f"{dim}: {params['board_format']}")
    
    print("\nTesting Spinner Set (24h Rooms):")
    for dim in ['4x4', '4x6']:
        params = SpinnerSet.generate_params(dim, is_24h=True)
        print(f"{dim}: {params['board_format']}")
