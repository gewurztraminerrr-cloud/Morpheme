"""
Spinner Set System for Accumulative Game Mode
Generates randomized parameters for each round
"""

import random
import time

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
    def generate_params(board_dimensions, is_24h=False, is_split=False, previous_params=None):
        """Generate granular spinner parameters given dimensions"""
        try:
            # Loop to ensure parameters are DIFFERENT from previous (User Request)
            # We allow up to 30 attempts to find a unique combination
            best_res = None
            
            for _ in range(30):
                # Randomize dictionary
                dictionary = SpinnerSet._spin_dictionary()
                
                # Minimum word length (spin BEFORE word count to inform density)
                min_word_length = SpinnerSet._spin_min_word_length(board_dimensions)

                # Difficulty (spin first to inform density)
                difficulty = SpinnerSet._spin_difficulty()
                
                # Weighted word count (density cap for Hard/Expert and long-word rounds)
                wc_range = SpinnerSet._spin_word_count(dictionary, min_word_length, difficulty, board_dimensions)
                
                # Determine format (Allow Either/Or to persist at high density)
                board_format = SpinnerSet._spin_board_format(is_24h, board_dimensions)
                if wc_range == '500+' or wc_range == '200+':
                    if min_word_length >= 5:
                        # Keep Either/Or and Checkerboard as they are highly requested/stable
                        if board_format not in ['Either/Or', 'Checkerboard']:
                            board_format = 'Normal'
                bonus_len = max(min_word_length, SpinnerSet._spin_bonus_word_length())
                
                res = {
                    'bonus_word_length': bonus_len,
                    'min_word_length': min_word_length,
                    'difficulty': difficulty,
                    'word_count_range': wc_range,
                    'dictionary': dictionary,
                    'board_format': board_format,
                    'generated_at': time.time()
                }
                
                if not previous_params:
                    return res
                
                # Uniqueness Check: Ensure at least one major visual parameter changed
                # Keys to check for variety:
                major_keys = ['difficulty', 'min_word_length', 'word_count_range', 'dictionary', 'board_format']
                is_different = False
                for k in major_keys:
                    if str(res.get(k)) != str(previous_params.get(k)):
                        is_different = True
                        break
                
                if is_different:
                    return res
                
                best_res = res # Fallback to last attempt if we somehow fail 30 times

            print(f"[SpinnerSet] WARNING: Could not find unique params after 30 attempts. Using last roll.")
            return best_res or res
            
        except Exception as e:
            print(f"[SpinnerSet] CRITICAL GENERATOR ERROR: {e}")
            # Emergency dynamic fallback to avoid static repetition
            return {
                'difficulty': random.choice(['Easy', 'Medium', 'Hard']),
                'dictionary': random.choice(['NWL', 'CSW']),
                'word_count_range': random.choice(['50-100', '100-200', '200+', '500+']),
                'board_format': 'Normal',
                'min_word_length': 3,
                'bonus_word_length': 8,
                'generated_at': time.time()
            }
    
    @staticmethod
    def _spin_bonus_word_length():
        """20% each for 6, 7, 8, 9, 10 letters"""
        return random.choice([6, 7, 8, 9, 10])
    
    @staticmethod
    def _spin_min_word_length(board_dimensions):
        """Based on board dimensions with specified percentages"""
        dims = str(board_dimensions).lower()
        if '4x4' in dims:
            return random.choices([3, 4, 5], weights=[25, 50, 25])[0]
        elif '4x6' in dims:
            return random.choices([4, 5, 6], weights=[25, 50, 25])[0]
        elif '5x7' in dims:
            return random.choices([5, 6, 7], weights=[25, 50, 25])[0]
        elif '6x8' in dims:
            # User Request: Never exceed 7-letter minimum for 6x8 playability
            return random.choices([6, 7], weights=[40, 60])[0]
        elif '3x3x3' in dims:
            # User Request: 25% 6LM, 50% 7LM, 25% 8LM for 3x3x3
            return random.choices([6, 7, 8], weights=[25, 50, 25])[0]
        else:
            return 3  # Default
    
    @staticmethod
    def _spin_difficulty():
        """Balanced difficulty selection: 25% Easy, 50% Medium, 25% Hard"""
        return random.choices(['Easy', 'Medium', 'Hard'], weights=[25, 50, 25])[0]
    
    @staticmethod
    def _spin_word_count(dictionary_name='NWL', min_word_length=3, difficulty='Medium', board_dimensions='4x4'):
        # CRITICAL: For cubes, we must check total tiles via something other than rows*cols 
        # since depth isn't passed here. We default to is_large if caller uses rows/cols correctly or detect via values.
        # 3x3x3 cube has 27 tiles, but surface area logic often treats it as large.
        is_large = ('3x3x3' in str(board_dimensions)) or ('6x8' in str(board_dimensions))
        choices = ['50-100', '100-200', '200+', '500+']
        weights = [24, 50, 25, 1]
        return random.choices(choices, weights=weights)[0]
    
    @staticmethod
    def _spin_dictionary():
        """50% NWL, 50% CSW"""
        return random.choices(['NWL', 'CSW'], weights=[50, 50])[0]
    
    @staticmethod
    def _spin_board_format(is_24h=False, dimensions='4x4'):
        """
        Normal: 82% Normal, 8% Checkerboard, rest 2% each
        """
        if is_24h:
            return 'Valued Letters'
            
        if '3x3x3' in str(dimensions):
            return 'Normal'
            
        result = random.choices(
            ['Normal', 'Checkerboard', 'Penalty', 'Mania', 'Either/Or', 'Bonus Letter', 'Valued Letters'],
            weights=[60, 10, 6, 6, 6, 6, 6]
        )[0]
        
        if result == 'Mania':
            # User Request: 30% vowels, 70% consonants for Mania formats
            if random.random() < 0.30:
                mania_letter = random.choice('AEIOU')
            else:
                mania_letter = random.choice('BCDFGHJKLMNPQRSTVWXYZ')
            return f'{mania_letter} Mania'
        return result
