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
                import random
                wc_range = random.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
        else:
            if not wc_range or wc_range not in ['50-100', '100-200', '200-300', '300-400', '500+']:
                import random
                wc_range = random.choices(['50-100', '100-200', '200-300', '300-400', '500+'], weights=[9, 30, 30, 30, 1])[0]

        # Enforce 4x4 feasibility:
        # USER MANDATE: The ONLY Minimum letter count allowable for 50-100 words in a 4x4 room is 5L!
        if '4x4' in dims and not is_aw_effective:
            if wc_range == '50-100':
                min_word_length = 5
            elif min_word_length >= 5:
                wc_range = '50-100'
            elif min_word_length == 4:
                if wc_range not in ['100-200', '200-300']:
                    wc_range = '100-200'
            elif min_word_length == 3:
                if wc_range not in ['100-200', '200-300', '300-400']:
                    wc_range = '200-300'

        p['min_word_length'] = min_word_length
        p['word_count_range'] = wc_range

        if is_24h:
            if not is_aw_effective:
                p['word_count_range'] = '200-300'
            if '4x4' in dims and min_word_length >= 5:
                p['min_word_length'] = 4
            elif '4x6' in dims and min_word_length >= 6:
                p['min_word_length'] = 5
            elif '5x7' in dims and min_word_length >= 7:
                p['min_word_length'] = 6

        return p
        bw_raw = res.get('bonus_word_length')
        try:
            bw_val = int(bw_raw)
            if bw_val < 6 or bw_val > 10:
                import random
                res['bonus_word_length'] = random.choice([6, 7, 8, 9, 10])
            else:
                res['bonus_word_length'] = bw_val
        except:
            import random
            res['bonus_word_length'] = random.choice([6, 7, 8, 9, 10])

        return res

    @staticmethod
    def _generate_params_raw(board_dimensions, is_24h=False, is_split=False, previous_params=None):
        """Generate granular spinner parameters given dimensions (raw implementation)"""
        try:
            # Loop to ensure parameters are DIFFERENT from previous (User Request)
            # We allow up to 30 attempts to find a unique combination
            best_res = None
            res = None  # Guard: ensure res is always defined even if loop body raises on first iteration
            
            for _ in range(30):
                # Randomize dictionary and added words configuration based on dimensions
                dims = str(board_dimensions).lower().replace(" ", "")
                
                # Determine floor, middle, ceiling min-lengths for the board dimensions
                if '4x4' in dims:
                    floor, middle, ceiling = 3, 4, 5
                elif '4x6' in dims:
                    floor, middle, ceiling = 4, 5, 6
                elif '5x7' in dims:
                    floor, middle, ceiling = 5, 6, 7
                else: # 6x8, 3x3x3, etc.
                    floor, middle, ceiling = 6, 7, 8
                
                # Enforce explicit weights: floor (25%), middle (50%), ceiling (25%)
                min_word_length = random.choices([floor, middle, ceiling], weights=[25, 50, 25])[0]
                
                # Equal 25% probability for each dictionary type
                dictionary = random.choices(['NWL', 'CSW', 'NWL + AW', 'CSW + AW'], weights=[25, 25, 25, 25])[0]

                is_aw_effective = (
                    '+ AW' in str(dictionary).upper()
                    or '+AW' in str(dictionary).upper()
                    or str(dictionary).upper() in ['AW', 'ADDED_WORDS', 'ALL']
                )
                use_added_words = is_aw_effective

                # Difficulty (spin independently)
                difficulty = SpinnerSet._spin_difficulty(board_dimensions, 3)

                # Word count range strictly drawn from Odds Window probabilities
                if is_aw_effective:
                    wc_range = random.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
                else:
                    wc_range = random.choices(['50-100', '100-200', '200-300', '300-400', '500+'], weights=[9, 30, 30, 30, 1])[0]

                # Determine board format
                board_format = SpinnerSet._spin_board_format(is_24h, board_dimensions)

                bw_len = random.choice([6, 7, 8, 9, 10])
                if bw_len < min_word_length:
                    bw_len = min_word_length
                
                res = {
                    'min_word_length': min_word_length,
                    'difficulty': difficulty,
                    'word_count_range': wc_range or SpinnerSet._spin_word_count(dictionary, min_word_length, difficulty, board_dimensions),
                    'dictionary': dictionary,
                    'use_added_words': use_added_words,
                    'board_format': board_format,
                    'bonus_word_length': bw_len,
                    'generated_at': time.time()
                }
                
                if previous_params and isinstance(previous_params, dict) and _ < 25:
                    def get_base_fmt(f):
                        s = str(f).lower()
                        if 'bounce' in s: return 'bounce'
                        if 'mania' in s: return 'mania'
                        if 'checkerboard' in s: return 'checkerboard'
                        if 'equality' in s: return 'equality'
                        if 'density' in s: return 'density'
                        if 'penalty' in s: return 'penalty'
                        if 'either' in s: return 'either'
                        if 'bonus' in s: return 'bonus'
                        if 'valued' in s: return 'valued'
                        if 'rotation' in s: return 'rotation'
                        if 'double' in s: return 'double'
                        if 'triple' in s: return 'triple'
                        return 'normal'

                    prev_base = get_base_fmt(previous_params.get('board_format', ''))
                    cur_base = get_base_fmt(res.get('board_format', ''))
                    
                    # 1. SPECIAL FORMAT ANTI-STREAK: Non-Normal special formats must never repeat back-to-back
                    if cur_base != 'normal' and cur_base == prev_base:
                        continue

                    # 2. MULTI-ATTRIBUTE ANTI-REPEAT: Count matching fields against previous round
                    matches = 0
                    if str(res.get('dictionary')).upper() == str(previous_params.get('dictionary')).upper():
                        matches += 1
                    if bool(res.get('use_added_words')) == bool(previous_params.get('use_added_words')):
                        matches += 1
                    if int(res.get('min_word_length', 3)) == int(previous_params.get('min_word_length', 3)):
                        matches += 1
                    if str(res.get('word_count_range')) == str(previous_params.get('word_count_range')):
                        matches += 1
                    if str(res.get('difficulty')) == str(previous_params.get('difficulty')):
                        matches += 1
                    if cur_base == prev_base:
                        matches += 1

                    # REJECT if 3 or more attributes match previous round (forces parameter variety!)
                    if matches >= 3:
                        continue

                return res
                
                best_res = res # Fallback to last attempt if we somehow fail 30 times

            print(f"[SpinnerSet] WARNING: Could not find unique params after 30 attempts. Using last roll.")
            return best_res or res or {
                'difficulty': random.choices(['Easy', 'Medium', 'Hard'], weights=[25, 50, 25])[0],
                'dictionary': random.choice(['NWL', 'CSW']),
                'word_count_range': '200-300',
                'board_format': 'Normal',
                'min_word_length': SpinnerSet._spin_min_word_length(board_dimensions),
                'bonus_word_length': 8,
                'use_added_words': False,
                'generated_at': time.time()
            }
            
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


