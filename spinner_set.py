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
        """Generate granular spinner parameters, enforcing Valued Letters for 24h matches."""
        try:
            res = SpinnerSet._generate_params_raw(board_dimensions, is_24h, is_split, previous_params)
            if is_24h and isinstance(res, dict):
                res['board_format'] = 'Valued Letters'
            return res
        except Exception as e:
            print(f"[SpinnerSet] CRITICAL WRAPPER ERROR: {e}")
            return {
                'difficulty': random.choice(['Easy', 'Medium', 'Hard']),
                'dictionary': random.choice(['NWL', 'CSW']),
                'word_count_range': random.choice(['50-100', '100-200', '200-300', '300-400']),
                'board_format': 'Valued Letters' if is_24h else 'Normal',
                'min_word_length': 3,
                'generated_at': time.time()
            }

    @staticmethod
    def _generate_params_raw(board_dimensions, is_24h=False, is_split=False, previous_params=None):
        """Generate granular spinner parameters given dimensions (raw implementation)"""
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
                difficulty = SpinnerSet._spin_difficulty(board_dimensions, min_word_length)
                
                # Weighted word count (density cap for Hard/Expert and long-word rounds)
                wc_range = SpinnerSet._spin_word_count(dictionary, min_word_length, difficulty, board_dimensions)
                
                # RE-SYNC: High density on 4x4 ALWAYS results in high uniqueness (Hard).
                # If we rolled 200+ or 500+ on a 4x4, we must promote the difficulty to Hard.
                if wc_range in ['200+', '500+'] and ('4x4' in str(board_dimensions) or '4x6' in str(board_dimensions)):
                    difficulty = 'Hard'
                
                # Determine format (Allow Either/Or to persist at high density)
                board_format = SpinnerSet._spin_board_format(is_24h, board_dimensions)
                if wc_range == '500+' or wc_range == '200+':
                    if min_word_length >= 5:
                        # Keep Either/Or, Checkerboard, and Density as they are highly requested/stable
                        if board_format not in ['Either/Or', 'Checkerboard', 'Density', 'Valued Letters']:
                            board_format = 'Normal'
                if board_format == 'Checkerboard':
                    wc_range = random.choice(['100-200', '200-300'])
                
                res = {
                    'min_word_length': min_word_length,
                    'difficulty': difficulty,
                    'word_count_range': wc_range or SpinnerSet._spin_word_count(dictionary, min_word_length, difficulty, board_dimensions),
                    'dictionary': dictionary,
                    'board_format': board_format,
                    'generated_at': time.time()
                }
                
                if not previous_params:
                    # USER REQUEST: Specific initial state for the first round (Density Focused)
                    # We must still respect dimension-based minimum lengths (especially for 6x8/3x3x3)
                    initial_min = SpinnerSet._spin_min_word_length(board_dimensions)
                    return {
                        'min_word_length': initial_min,
                        'difficulty': SpinnerSet._spin_difficulty(board_dimensions, initial_min),
                        'word_count_range': random.choice(['50-100', '100-200', '200-300', '300-400']),
                        'dictionary': 'NWL',
                        'board_format': 'Normal',
                        'generated_at': time.time()
                    }
                
                # VARIETY ENFORCEMENT: Avoid repeating the same board format (User Request)
                # If we rolled the exact same format (e.g., 'Either/Or' twice), we re-roll 
                # up to the loop limit to find a different experience.
                # NOTE: We allow 'Normal' to repeat to maintain its intended 80% frequency.
                if res.get('board_format') == previous_params.get('board_format'):
                     if res.get('board_format') != 'Normal' and _ < 25:
                         continue

                # Uniqueness Check: Ensure at least one major parameter changed
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
                'word_count_range': random.choice(['50-100', '100-200', '200-300', '300-400']),
                'board_format': 'Normal',
                'min_word_length': 3,
                'generated_at': time.time()
            }
    
    
    @staticmethod
    def _spin_min_word_length(board_dimensions):
        """Based on board dimensions with specified percentages"""
        dims = str(board_dimensions).lower().replace(" ", "")
        if '4x4' in dims:
            return random.choices([3, 4, 5], weights=[25, 50, 25])[0]
        elif '4x6' in dims:
            return random.choices([4, 5, 6], weights=[25, 50, 25])[0]
        elif '5x7' in dims:
            return random.choices([5, 6, 7], weights=[25, 50, 25])[0]
        elif '6x8' in dims:
            # User Request: 25% 6LM, 50% 7LM, 25% 8LM for 6x8
            return random.choices([6, 7, 8], weights=[25, 50, 25])[0]
        elif '3x3x3' in dims:
            # User Request: 25% 6LM, 50% 7LM, 25% 8LM for 3x3x3
            return random.choices([6, 7, 8], weights=[25, 50, 25])[0]
        else:
            return 3  # Default
    
    @staticmethod
    def _spin_difficulty(board_dimensions='4x4', min_word_length=3):
        """Balanced difficulty selection: 25% Easy, 50% Medium, 25% Hard.
        For 5L+ minimum length rounds, we force Hard as Easy uniqueness is physically impossible."""
        dims_str = str(board_dimensions)
        if min_word_length >= 5 and ('4x4' in dims_str or '4x6' in dims_str):
            return 'Hard'

        choices = ['Easy', 'Medium', 'Hard']
        weights = [25, 50, 25] # Strictly 25% Easy, 50% Medium, 25% Hard
        return random.choices(choices, weights=weights)[0]
    
    @staticmethod
    def _spin_word_count(dictionary, min_word_length, difficulty, board_dimensions):
        # USER REQUEST: 16% 50-100, 33% 100-200, 33% 200-300, 17% 300-400, 1% 500+
        choices = ['50-100', '100-200', '200-300', '300-400', '500+']
        weights = [16, 33, 33, 17, 1]
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
            ['Normal', 'Checkerboard', 'Penalty', 'Mania', 'Either/Or', 'Bonus Letter', 'Valued Letters', 'Density', 'Rotating Letters'],
            weights=[78, 8, 2, 2, 2, 2, 2, 2, 2]
        )[0]
        
        if result == 'Mania':
            # User Request: 33% vowels, 67% consonants for Mania formats
            if random.random() < 0.33:
                mania_letter = random.choice('AEIOU')
            else:
                mania_letter = random.choice('BCDFGHJKLMNPQRSTVWXYZ')
            return f'{mania_letter} Mania'
        return result
