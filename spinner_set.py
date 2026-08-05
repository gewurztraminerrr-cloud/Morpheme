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
        # 1. Broad Parameters
        board_dims = random.choice(['4x4', '4x6', '5x7', '6x8'])
        time_limit = random.choice([60, 90, 120, 180, 240, 300])
        
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
                res['word_count_range'] = '200-300'
                
            # Ironclad validation to ensure 50-100 word count is strictly restricted to greatest minimum lengths
            if isinstance(res, dict) and res.get('word_count_range') == '50-100':
                dims = str(board_dimensions).lower().replace(" ", "")
                min_word_length = res.get('min_word_length', 3)
            # Roll for Added Words configuration
            if isinstance(res, dict):
                dict_val = res.get('dictionary')
                if dict_val:
                    if '+ AW' in str(dict_val) or '+AW' in str(dict_val):
                        res['use_added_words'] = True
                    else:
                        res['use_added_words'] = (str(dict_val).upper() == 'AW' or res.get('use_added_words') is True)
                    
                    if res.get('use_added_words') is True:
                        clean_dict = str(dict_val).replace('+ AW', '').replace('+AW', '').strip()
                        if clean_dict == 'AW':
                            clean_dict = 'NWL'
                        res['dictionary'] = f"{clean_dict} + AW"
                else:
                    res['dictionary'] = SpinnerSet._spin_dictionary()
                    res['use_added_words'] = ('+ AW' in str(res['dictionary']) or '+AW' in str(res['dictionary']))

            # Apply our ironclad safety sanitizer
            res = SpinnerSet.sanitize_params(res, board_dimensions, is_24h)
            return res
        except Exception as e:
            print(f"[SpinnerSet] CRITICAL WRAPPER ERROR: {e}")
            # Derive fallback min length dynamically based on dimensions
            dims_lower = str(board_dimensions).lower()
            fallback_min = 4 if '4x6' in dims_lower else (5 if '5x7' in dims_lower else (6 if '6x8' in dims_lower or '3x3x3' in dims_lower else 3))
            
            fallback = {
                'difficulty': random.choices(['Easy', 'Medium', 'Hard'], weights=[25, 50, 25])[0],
                'dictionary': random.choice(['NWL', 'CSW', 'NWL + AW', 'CSW + AW']),
                'word_count_range': '200-300' if is_24h else random.choices(['100-200', '200-300', '300-400', '500+'], weights=[30, 30, 30, 1])[0],
                'board_format': 'Valued Letters' if is_24h else 'Normal',
                'min_word_length': fallback_min,
                'bonus_word_length': 8,
                'generated_at': time.time()
            }
            # Roll for Added Words fallback
            fallback_dict = fallback.get('dictionary')
            if fallback_dict:
                if '+ AW' in str(fallback_dict) or '+AW' in str(fallback_dict):
                    fallback['use_added_words'] = True
                else:
                    fallback['use_added_words'] = (str(fallback_dict).upper() == 'AW')
            else:
                fallback['use_added_words'] = False
                
            return SpinnerSet.sanitize_params(fallback, board_dimensions, is_24h)

    @staticmethod
    def sanitize_params(res, board_dimensions='4x4', is_24h=False):
        """Ironclad density/min-length sanitization to enforce exact Spinner Set rules."""
        if not isinstance(res, dict):
            return res
            
        p = dict(res)
        dims = str(board_dimensions).lower().replace(" ", "")
        
        # Normalize word_count_range string format
        wc = p.get('word_count_range')
        if isinstance(wc, (list, tuple)):
            if len(wc) >= 2: p['word_count_range'] = f"{wc[0]}-{wc[1]}"
            elif len(wc) == 1: p['word_count_range'] = str(wc[0])
        elif isinstance(wc, str):
            p['word_count_range'] = wc.replace(',', '-')

        raw_dict_name = str(p.get('dictionary', 'NWL')).upper()
        is_aw_effective = (
            raw_dict_name in ['AW', 'ADDED_WORDS', 'ALL']
            or p.get('use_added_words') is True
            or '+ AW' in raw_dict_name
            or '+AW' in raw_dict_name
        )
        clean_base_dict = raw_dict_name.replace('+ AW', '').replace('+AW', '').replace('ADDED_WORDS', '').strip()
        if clean_base_dict in ['', 'AW', 'ALL']:
            clean_base_dict = 'NWL'
            
        if is_aw_effective:
            p['dictionary'] = f"{clean_base_dict} + AW"
            p['use_added_words'] = True
        else:
            p['dictionary'] = clean_base_dict
            p['use_added_words'] = False
        
        try:
            min_word_length = int(p.get('min_word_length'))
        except:
            min_word_length = 4 if '4x6' in dims else (5 if '5x7' in dims else (6 if '6x8' in dims or '3x3x3' in dims else 3))

        # Enforce grid floor limits for min_word_length
        floor_len = 3
        if '4x6' in dims: floor_len = 4
        elif '5x7' in dims: floor_len = 5
        elif '6x8' in dims or '3x3x3' in dims: floor_len = 6
        
        if min_word_length < floor_len:
            min_word_length = floor_len

        # Hard limits on max min_word_length per grid size to prevent overflows
        if '4x4' in dims:
            if min_word_length > 5: min_word_length = 5
        elif '4x6' in dims:
            if min_word_length > 6: min_word_length = 6
        elif '5x7' in dims:
            if min_word_length > 7: min_word_length = 7
        elif '6x8' in dims or '3x3x3' in dims:
            if min_word_length > 8: min_word_length = 8

        wc_range = p.get('word_count_range')
        if is_aw_effective:
            if wc_range not in ['300-400', '400-500', '500+']:
                wc_range = random.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
        else:
            if not wc_range or wc_range not in ['50-100', '100-200', '200-300', '300-400', '500+']:
                wc_range = random.choices(['50-100', '100-200', '200-300', '300-400', '500+'], weights=[9, 30, 30, 30, 1])[0]
            
            # Enforce physical achievable ceiling caps per min_word_length on standard dictionaries
            if min_word_length >= 8:
                if wc_range in ['200-300', '300-400', '400-500', '500+']:
                    wc_range = '100-200'
            elif min_word_length >= 7:
                if wc_range in ['300-400', '400-500', '500+']:
                    wc_range = '200-300'
            elif min_word_length >= 6:
                if wc_range in ['400-500', '500+']:
                    wc_range = '200-300'

        p['min_word_length'] = min_word_length
        p['word_count_range'] = wc_range

        if is_24h:
            p['board_format'] = 'Valued Letters'
            if not is_aw_effective:
                p['word_count_range'] = '200-300'

        bw_raw = p.get('bonus_word_length')
        try:
            bw_val = int(bw_raw)
            if bw_val < min_word_length or bw_val > 10:
                p['bonus_word_length'] = max(min_word_length, random.choice([6, 7, 8, 9, 10]))
            else:
                p['bonus_word_length'] = bw_val
        except:
            p['bonus_word_length'] = max(min_word_length, random.choice([6, 7, 8, 9, 10]))

        return p

    @staticmethod
    def _generate_params_raw(board_dimensions, is_24h=False, is_split=False, previous_params=None):
        """Generate granular spinner parameters matching exact Odds Window probabilities."""
        try:
            dims = str(board_dimensions).lower().replace(" ", "")
            
            # 1. Min Word Length (25% Low | 50% Med | 25% High)
            if '4x4' in dims:
                min_word_length = random.choices([3, 4, 5], weights=[25, 50, 25])[0]
            elif '4x6' in dims:
                min_word_length = random.choices([4, 5, 6], weights=[25, 50, 25])[0]
            elif '5x7' in dims:
                min_word_length = random.choices([5, 6, 7], weights=[25, 50, 25])[0]
            else: # 6x8, 3x3x3, etc.
                min_word_length = random.choices([6, 7, 8], weights=[25, 50, 25])[0]
            
            # 2. Dictionary (25% NWL | 25% CSW | 25% NWL + AW | 25% CSW + AW)
            dictionary = random.choices(['NWL', 'CSW', 'NWL + AW', 'CSW + AW'], weights=[25, 25, 25, 25])[0]

            is_aw_effective = '+ AW' in dictionary
            use_added_words = is_aw_effective

            # 3. Difficulty (25% Easy | 50% Medium | 25% Hard)
            difficulty = random.choices(['Easy', 'Medium', 'Hard'], weights=[25, 50, 25])[0]

            # 4. Word Count Range
            if is_aw_effective:
                wc_range = random.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
            else:
                wc_range = random.choices(['50-100', '100-200', '200-300', '300-400', '500+'], weights=[9, 30, 30, 30, 1])[0]

            # 5. Board Format
            board_format = SpinnerSet._spin_board_format(is_24h, board_dimensions)

            # 6. Bonus Word Length
            bw_len = random.choice([6, 7, 8, 9, 10])
            if bw_len < min_word_length:
                bw_len = min_word_length
            
            res = {
                'min_word_length': min_word_length,
                'difficulty': difficulty,
                'word_count_range': wc_range,
                'dictionary': dictionary,
                'use_added_words': use_added_words,
                'board_format': board_format,
                'bonus_word_length': bw_len,
                'generated_at': time.time()
            }
            
            return res
        except Exception as e:
            print(f"[SpinnerSet] CRITICAL GENERATOR ERROR: {e}")
            # Emergency dynamic fallback to avoid static repetition
            return {
                'difficulty': random.choices(['Easy', 'Medium', 'Hard'], weights=[25, 50, 25])[0],
                'dictionary': random.choice(['NWL', 'CSW']),
                'word_count_range': random.choices(['100-200', '200-300', '300-400', '500+'], weights=[30, 30, 30, 1])[0],
                'board_format': 'Normal',
                'min_word_length': 3,
                'bonus_word_length': 8,
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
        """Balanced difficulty selection: 25% Easy, 50% Medium, 25% Hard."""
        choices = ['Easy', 'Medium', 'Hard']
        weights = [25, 50, 25] # Strictly 25% Easy, 50% Medium, 25% Hard
        return random.choices(choices, weights=weights)[0]
    
    @staticmethod
    def _spin_word_count(dictionary, min_word_length, difficulty, board_dimensions, use_added_words=False):
        d_upper = str(dictionary).upper()
        if use_added_words or "+ AW" in d_upper or "+AW" in d_upper or "ADDED" in d_upper:
            choices = ['300-400', '400-500', '500+']
            weights = [40, 40, 20]
        else:
            choices = ['50-100', '100-200', '200-300', '300-400']
            weights = [10, 30, 30, 30]
        return random.choices(choices, weights=weights)[0]
    
    @staticmethod
    def _spin_dictionary():
        """Equal probability: 25% NWL, 25% CSW, 25% NWL + AW, 25% CSW + AW"""
        choices = ['NWL', 'CSW', 'NWL + AW', 'CSW + AW']
        weights = [25, 25, 25, 25]
        return random.choices(choices, weights=weights)[0]
    
    @staticmethod
    def _spin_board_format(is_24h=False, dimensions='4x4'):
        """
        Original board format weights
        """
        if is_24h:
            return 'Valued Letters'
            
        if '3x3x3' in str(dimensions):
            return 'Normal'
            
        result = random.choices(
            ['Normal', 'Checkerboard', 'Equality Freq', 'Bounce', 'Density', 'Mania', 'Penalty', 'Either/Or', 'Bonus Letter', 'Valued Letters', 'Rotation', 'Double', 'Triple'],
            weights=[66, 12, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
        )[0]
        
        if result == 'Bounce':
            result = random.choices(['Bounce 1x', 'Bounce 2x', 'Bounce 3x'], weights=[33, 33, 34])[0]
        
        if result == 'Mania':
            # 33% vowels (A, E, I, O, U), 67% consonants
            if random.random() < 0.33:
                mania_letter = random.choice(['A', 'E', 'I', 'O', 'U'])
            else:
                mania_letter = random.choice(['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'])
            return f'{mania_letter} Mania'
        return result


