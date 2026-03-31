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
    def generate_params(board_dimensions, is_24h=False, is_split=False):
        """Generate granular spinner parameters given dimensions"""
        # Generate dictionary FIRST so we can use its size for word count
        dictionary = SpinnerSet._spin_dictionary()
        wc_range = SpinnerSet._spin_word_count(dictionary)
        
        # Determine format (Force Normal for 500+ density to ensure solvability)
        board_format = SpinnerSet._spin_board_format(is_24h, board_dimensions)
        if wc_range == '500+':
            board_format = 'Normal'

        # Minimum word length must be established first to cap bonus len
        min_word_length = SpinnerSet._spin_min_word_length(board_dimensions)
        bonus_len = max(min_word_length, SpinnerSet._spin_bonus_word_length())
            
        return {
            'bonus_word_length': bonus_len,
            'min_word_length': min_word_length,
            'difficulty': SpinnerSet._spin_difficulty(),
            'word_count_range': wc_range,
            'dictionary': dictionary,
            'board_format': board_format
        }
    
    @staticmethod
    def _spin_bonus_word_length():
        """20% each for 6, 7, 8, 9, 10 letters"""
        return random.choice([6, 7, 8, 9, 10])
    
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
        elif dims == '3x3x3':
            return random.choices([6, 7, 8], weights=[25, 50, 25])[0]
        else:
            return 3  # Default
    
    @staticmethod
    def _spin_difficulty():
        """25% Easy, 50% Medium, 25% Hard"""
        return random.choices(['Easy', 'Medium', 'Hard'], weights=[25, 50, 25])[0]
    
    @staticmethod
    def _spin_word_count(dictionary_name='NWL'):
        """24% 50-100, 50% 100-200, 25% 200+, 1% 500+"""
        return random.choices(['50-100', '100-200', '200+', '500+'], weights=[24, 50, 25, 1])[0]
    
    @staticmethod
    def _spin_dictionary():
        """50% NWL, 50% CSW"""
        return random.choices(['NWL', 'CSW'], weights=[50, 50])[0]
    
    @staticmethod
    def _spin_board_format(is_24h=False, dimensions='4x4'):
        """
        Normal rooms: 82% Normal, 8% Checkerboard, 2% Penalty, 2% Mania, 2% Either/Or, 2% Bonus Letter, 2% Valued Letters
        24h rooms: 100% Normal (No alternate formats allowed)
        3D rooms: 100% Normal (No alternate formats allowed initially)
        """
        if is_24h:
            return 'Normal'
        elif dimensions == '3x3x3':
            return 'Normal'
        else:
            result = random.choices(
                ['Normal', 'Checkerboard', 'Penalty', 'Mania', 'Either/Or', 'Bonus Letter', 'Valued Letters'],
                weights=[82, 8, 2, 2, 2, 2, 2]
            )[0]
            if result == 'Mania':
                # User Request: 30% vowels, 70% consonants for Mania formats
                if random.random() < 0.30:
                    mania_letter = random.choice('AEIOU')
                else:
                    mania_letter = random.choice('BCDFGHJKLMNPQRSTVWXYZ')
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
