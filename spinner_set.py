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
                
            # Ironclad validation to ensure 50-100 word count is strictly restricted to greatest minimum lengths and NOT allowed for + AW
            if isinstance(res, dict) and res.get('word_count_range') == '50-100':
                dict_name = str(res.get('dictionary', '')).upper()
                is_aw = res.get('use_added_words') is True or '+ AW' in dict_name or '+AW' in dict_name or dict_name in ['AW', 'ADDED_WORDS']
                if is_aw:
                    res['word_count_range'] = '300-400'
                    print(f"[SpinnerSet] Ironclad corrected 50-100 word count range for + AW dictionary to 300-400.")
                else:
                    dims = str(board_dimensions).lower().replace(" ", "")
                    min_word_length = res.get('min_word_length', 3)
                    
                    is_greatest = False
                    if '4x4' in dims and min_word_length >= 5:
                        is_greatest = True
                    elif '4x6' in dims and min_word_length >= 6:
                        is_greatest = True
                    elif '5x7' in dims and min_word_length >= 7:
                        is_greatest = True
                    elif '6x8' in dims and min_word_length >= 8:
                        is_greatest = True
                    elif '3x3x3' in dims and min_word_length >= 8:
                        is_greatest = True
                    
                    if not is_greatest:
                        # If it slipped through or was loaded from a stale state, upgrade the word count range to 100-200
                        res['word_count_range'] = '100-200'
                        print(f"[SpinnerSet] Ironclad corrected mismatched word_count_range for dims={board_dimensions}, min_len={min_word_length} from 50-100 to 100-200.")
            
            # Ironclad validation to ensure 100-200 word count does not include forbidden lengths
            if isinstance(res, dict) and res.get('word_count_range') == '100-200':
                dims = str(board_dimensions).lower().replace(" ", "")
                min_word_length = res.get('min_word_length', 3)
                if '4x4' in dims and min_word_length == 3:
                    res['min_word_length'] = random.choice([4, 5])
                    print(f"[SpinnerSet] Wrapper adjusted 4x4 min_len for 100-200 words from 3 to {res['min_word_length']}")
                elif '4x6' in dims and min_word_length == 4:
                    res['min_word_length'] = random.choice([5, 6])
                    print(f"[SpinnerSet] Wrapper adjusted 4x6 min_len for 100-200 words from 4 to {res['min_word_length']}")
                elif '5x7' in dims and min_word_length == 5:
                    res['min_word_length'] = random.choice([6, 7])
                    print(f"[SpinnerSet] Wrapper adjusted 5x7 min_len for 100-200 words from 5 to {res['min_word_length']}")
                elif '6x8' in dims and min_word_length == 6:
                    res['min_word_length'] = random.choice([7, 8])
                    print(f"[SpinnerSet] Wrapper adjusted 6x8 min_len for 100-200 words from 6 to {res['min_word_length']}")
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
    def sanitize_params(res, board_dimensions, is_24h=False):
        """Ironclad density/min-length sanitization to prevent impossible board generator CPU hangs."""
        if not isinstance(res, dict):
            return res
            
        dims = str(board_dimensions).lower().replace(" ", "")
        
        dict_name = str(res.get('dictionary', 'NWL')).upper()
        # Detect compound AW dict names like 'NWL + AW' and 'CSW + AW'
        is_aw_effective = (
            dict_name in ['AW', 'ADDED_WORDS', 'ALL']
            or res.get('use_added_words') is True
            or '+ AW' in dict_name
            or '+AW' in dict_name
        )
        
        try:
            min_word_length = int(res.get('min_word_length'))
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
            if min_word_length > 4: min_word_length = 4
        elif '4x6' in dims:
            if min_word_length > 6: min_word_length = 6
        elif '5x7' in dims:
            if min_word_length > 7: min_word_length = 7
        elif '6x8' in dims or '3x3x3' in dims:
            if min_word_length > 8: min_word_length = 8

        wc_range = res.get('word_count_range', '100-200')

        # Enforce agreement: in 4x6 rooms, only allow 6L on 50-100 word counts.
        if '4x6' in dims:
            if wc_range == '50-100' or min_word_length == 6:
                wc_range = '50-100'
                min_word_length = 6

        if is_aw_effective:
            # Scale target range according to min_word_length to keep rounds mathematically possible and prevent hangs
            # BUT: NEVER allow word counts lower than 300-400 for AW rounds (strictly enforce 300-400, 400-500, or 500+).
            if wc_range not in ['300-400', '400-500', '500+']:
                import random
                wc_range = random.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
            # Safety caps for AW: Cap length to maximum possible that can hit 300+ words
            if '4x4' in dims:
                min_word_length = 3
            elif '4x6' in dims:
                min_word_length = min(min_word_length, 4)
            elif '5x7' in dims:
                min_word_length = min(min_word_length, 5)
            elif '3x3x3' in dims:
                min_word_length = min(min_word_length, 5)
            elif '6x8' in dims:
                min_word_length = min(min_word_length, 6)
        else:
            if wc_range not in ['50-100', '100-200', '200-300', '300-400']:
                import random
                wc_range = random.choices(['50-100', '100-200', '200-300', '300-400'], weights=[10, 30, 30, 30])[0]

            if '4x4' in dims:
                if min_word_length == 4:
                    wc_range = '50-100'
                else: # min_word_length <= 3
                    if wc_range not in ['100-200', '200-300', '300-400']:
                        wc_range = '100-200'
            elif '4x6' in dims:
                if min_word_length == 6:
                    if wc_range not in ['50-100', '100-200']:
                        wc_range = '50-100'
                elif min_word_length == 5:
                    if wc_range not in ['100-200', '200-300']:
                        wc_range = '100-200'
                elif min_word_length == 4:
                    if wc_range not in ['100-200', '200-300', '300-400']:
                        wc_range = '100-200'
                else: # min_word_length <= 3
                    if wc_range not in ['100-200', '200-300', '300-400']:
                        wc_range = '200-300'
            elif '5x7' in dims:
                if min_word_length == 7:
                    if wc_range not in ['50-100', '100-200']:
                        wc_range = '50-100'
                elif min_word_length == 6:
                    if wc_range not in ['100-200', '200-300']:
                        wc_range = '100-200'
                elif min_word_length == 5:
                    if wc_range not in ['100-200', '200-300', '300-400']:
                        wc_range = '200-300'
                else: # min_word_length <= 4
                    if wc_range not in ['200-300', '300-400']:
                        wc_range = '200-300'
            elif '6x8' in dims or '3x3x3' in dims:
                if min_word_length == 8:
                    if wc_range not in ['50-100', '100-200']:
                        wc_range = '50-100'
                elif min_word_length == 7:
                    if wc_range not in ['100-200', '200-300']:
                        wc_range = '100-200'
                elif min_word_length == 6:
                    if wc_range not in ['100-200', '200-300', '300-400']:
                        wc_range = '200-300'
                else: # min_word_length <= 5
                    if wc_range not in ['200-300', '300-400']:
                        wc_range = '200-300'

        res['min_word_length'] = min_word_length
        res['word_count_range'] = wc_range

        if is_24h:
            if not is_aw_effective:
                res['word_count_range'] = '200-300'
            if '4x4' in dims and min_word_length >= 5:
                res['min_word_length'] = 4
            elif '4x6' in dims and min_word_length >= 6:
                res['min_word_length'] = 5
            elif '5x7' in dims and min_word_length >= 7:
                res['min_word_length'] = 6
            elif ('6x8' in dims or '3x3x3' in dims) and min_word_length >= 8:
                res['min_word_length'] = 7

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
                # Randomize dictionary and added words configuration
                dictionary = SpinnerSet._spin_dictionary()
                is_aw_effective = (
                    '+ AW' in str(dictionary).upper()
                    or '+AW' in str(dictionary).upper()
                    or str(dictionary).upper() in ['AW', 'ADDED_WORDS', 'ALL']
                )
                use_added_words = is_aw_effective

                # Difficulty (spin independently)
                difficulty = SpinnerSet._spin_difficulty(board_dimensions, 3)

                # Word count range based on dictionary (forces exactly 9% weight for 50-100!)
                if is_aw_effective:
                    wc_range = random.choices(['300-400', '400-500', '500+'], weights=[40, 40, 20])[0]
                else:
                    wc_range = random.choices(['50-100', '100-200', '200-300', '300-400'], weights=[9, 31, 30, 30])[0]

                # Determine minimum word length based on wc_range and dimensions
                dims = str(board_dimensions).lower().replace(" ", "")
                if ('6x8' in dims or '3x3x3' in dims) and not is_aw_effective:
                    # Enforce the user's explicit weights: 6L (25%), 7L (50%), 8L (25%)
                    min_word_length = random.choices([6, 7, 8], weights=[25, 50, 25])[0]
                    # Select wc_range matching the length
                    if min_word_length == 8:
                        wc_range = random.choices(['50-100', '100-200'], weights=[25, 75])[0]
                    elif min_word_length == 7:
                        wc_range = random.choices(['100-200', '200-300'], weights=[50, 50])[0]
                    else: # 6L
                        wc_range = random.choices(['100-200', '200-300', '300-400'], weights=[30, 40, 30])[0]
                else:
                    if is_aw_effective:
                        if '4x4' in dims:
                            min_word_length = 3
                        elif '4x6' in dims:
                            min_word_length = random.choices([3, 4], weights=[40, 60])[0]
                        elif '5x7' in dims:
                            min_word_length = random.choices([4, 5], weights=[40, 60])[0]
                        elif '3x3x3' in dims:
                            min_word_length = random.choices([4, 5], weights=[40, 60])[0]
                        else: # 6x8
                            min_word_length = random.choices([5, 6], weights=[40, 60])[0]
                    else:
                        if wc_range == '50-100':
                            # Must be the greatest length
                            if '4x4' in dims: min_word_length = 4
                            elif '4x6' in dims: min_word_length = 6
                            elif '5x7' in dims: min_word_length = 7
                            else: min_word_length = 8
                        elif wc_range == '100-200':
                            # Must be second greatest length (or below if rolled)
                            if '4x4' in dims:
                                min_word_length = random.choices([3, 4], weights=[75, 25])[0]
                            elif '4x6' in dims:
                                min_word_length = random.choices([4, 5, 6], weights=[20, 50, 30])[0]
                            elif '5x7' in dims:
                                min_word_length = random.choices([5, 6, 7], weights=[20, 50, 30])[0]
                            else:
                                min_word_length = random.choices([6, 7, 8], weights=[20, 50, 30])[0]
                        elif wc_range == '200-300':
                            # Must be lower length (4x6 allows 6L at 10% to ensure it appears)
                            if '4x4' in dims:
                                min_word_length = 3
                            elif '4x6' in dims:
                                min_word_length = random.choices([4, 5, 6], weights=[60, 30, 10])[0]
                            elif '5x7' in dims:
                                min_word_length = random.choices([5, 6], weights=[70, 30])[0]
                            else:
                                min_word_length = random.choices([6, 7], weights=[70, 30])[0]
                        else: # 300-400
                            # Must be lowest length
                            if '4x4' in dims: min_word_length = 3
                            elif '4x6' in dims: min_word_length = random.choices([3, 4], weights=[50, 50])[0]
                            elif '5x7' in dims: min_word_length = random.choices([4, 5], weights=[50, 50])[0]
                            else: min_word_length = random.choices([5, 6], weights=[50, 50])[0]

                # Determine board format
                board_format = SpinnerSet._spin_board_format(is_24h, board_dimensions)
                if board_format == 'Checkerboard':
                    if is_aw_effective:
                        # Allow Checkerboard format in "+ AW" rounds! Keep the 300+ word count range.
                        pass
                    else:
                        wc_range = random.choice(['100-200', '200-300'])
                        # Re-sync min_word_length for the new wc_range
                        # FIX Issue 5: For 6x8 rooms, preserve the [6L=25%, 7L=50%, 8L=25%] user agreement.
                        # The original code excluded 8L from Checkerboard selections (only offered [6,7]).
                        if wc_range == '100-200':
                            if '4x4' in dims: min_word_length = random.choices([3, 4], weights=[75, 25])[0]
                            elif '4x6' in dims: min_word_length = random.choices([4, 5], weights=[25, 75])[0]
                            elif '5x7' in dims: min_word_length = random.choices([5, 6], weights=[25, 75])[0]
                            elif '6x8' in dims or '3x3x3' in dims:
                                # Respect [6L=25%, 7L=50%, 8L=25%] even in Checkerboard
                                min_word_length = random.choices([6, 7, 8], weights=[25, 50, 25])[0]
                                # Checkerboard with 8L forces 50-100 count
                                if min_word_length == 8: wc_range = '50-100'
                            else: min_word_length = random.choices([6, 7], weights=[25, 75])[0]
                        else: # 200-300
                            if '4x4' in dims: min_word_length = 3
                            elif '4x6' in dims: min_word_length = 4
                            elif '5x7' in dims: min_word_length = random.choices([5, 6], weights=[75, 25])[0]
                            elif '6x8' in dims or '3x3x3' in dims:
                                # Respect [6L=25%, 7L=50%, 8L=25%] — 8L needs fewer words, so pick 100-200
                                min_word_length = random.choices([6, 7, 8], weights=[25, 50, 25])[0]
                                if min_word_length == 8: wc_range = '100-200'
                            else: min_word_length = random.choices([6, 7], weights=[75, 25])[0]
                elif board_format == 'Equality Freq':
                    if is_aw_effective:
                        # AW dict with Equality Freq must still follow 300+ word counts; keep rolled range
                        pass
                    else:
                        # Non-AW Equality Freq is naturally low density; limit to 50-100 or 100-200
                        wc_range = random.choice(['50-100', '100-200'])
                        if wc_range == '50-100':
                            # Use highest min_len per dimension to correctly produce 50-100 words
                            if '4x4' in dims: min_word_length = 4
                            elif '4x6' in dims: min_word_length = 6  # was 5, bumped to match true 50-100 threshold
                            elif '5x7' in dims: min_word_length = 7  # was 6
                            elif '6x8' in dims or '3x3x3' in dims:
                                # FIX Issue 5: Equality Freq 50-100 → 8L only (to achieve low count)
                                min_word_length = 8
                            else: min_word_length = 8  # was 7
                        else: # 100-200
                            if '4x4' in dims: min_word_length = random.choice([3, 4])
                            elif '4x6' in dims: min_word_length = random.choices([4, 5, 6], weights=[20, 50, 30])[0]
                            elif '5x7' in dims: min_word_length = random.choices([5, 6, 7], weights=[20, 50, 30])[0]
                            elif '6x8' in dims or '3x3x3' in dims:
                                # FIX Issue 5: Respect [6L=25%, 7L=50%, 8L=25%] in Equality Freq 100-200
                                min_word_length = random.choices([6, 7, 8], weights=[25, 50, 25])[0]
                            else: min_word_length = random.choices([6, 7, 8], weights=[20, 50, 30])[0]

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
                
                if not previous_params:
                    # USER REQUEST: Specific initial state for the first round (Density Focused)
                    # We must still respect dimension-based minimum lengths (especially for 6x8/3x3x3)
                    initial_min = SpinnerSet._spin_min_word_length(board_dimensions)
                    initial_diff = SpinnerSet._spin_difficulty(board_dimensions, initial_min)
                    initial_wc = SpinnerSet._spin_word_count('NWL', initial_min, initial_diff, board_dimensions)
                    
                    if initial_wc == '100-200':
                        if '4x4' in dims and initial_min == 3:
                            initial_min = random.choice([4, 5])
                        elif '4x6' in dims and initial_min == 4:
                            initial_min = random.choice([5, 6])
                        elif '5x7' in dims and initial_min == 5:
                            initial_min = random.choice([6, 7])
                        elif '6x8' in dims and initial_min == 6:
                            initial_min = random.choice([7, 8])
                            
                    initial_dict = SpinnerSet._spin_dictionary()
                    initial_use_aw = ('+ AW' in initial_dict or '+AW' in initial_dict)
                    return {
                        'min_word_length': initial_min,
                        'difficulty': initial_diff,
                        'word_count_range': initial_wc,
                        'dictionary': initial_dict,
                        'use_added_words': initial_use_aw,
                        'board_format': 'Normal',
                        'bonus_word_length': max(8, initial_min),
                        'generated_at': time.time()
                    }
                
                # VARIETY ENFORCEMENT: Avoid repeating the same board format (User Request)
                # If we rolled the exact same format (e.g., 'Either/Or' twice), we re-roll 
                # up to the loop limit to find a different experience.
                # NOTE: We allow 'Normal' to repeat to maintain its intended 80% frequency.
                if res.get('board_format') == previous_params.get('board_format'):
                     if res.get('board_format') not in ['Normal', 'Either/Or'] and _ < 25:
                          continue

                # Uniqueness Check: Ensure at least one major parameter changed
                major_keys = ['difficulty', 'min_word_length', 'word_count_range', 'dictionary', 'use_added_words', 'board_format']
                is_different = False
                for k in major_keys:
                    if str(res.get(k)) != str(previous_params.get(k)):
                        is_different = True
                        break
                
                if is_different:
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
    def _spin_word_count(dictionary, min_word_length, difficulty, board_dimensions):
        d_upper = str(dictionary).upper()
        if "+ AW" in d_upper or "+AW" in d_upper or "ADDED" in d_upper:
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
            ['Normal', 'Bounce', 'Checkerboard', 'Equality Freq', 'Density', 'Penalty', 'Mania', 'Either/Or', 'Bonus Letter', 'Valued Letters', 'Rotation', 'Double', 'Triple'],
            weights=[66, 2, 12, 4, 2, 2, 2, 2, 2, 2, 2, 1, 1]
        )[0]
        
        if result == 'Bounce':
            result = random.choices(['Bounce 1x', 'Bounce 2x', 'Bounce 3x'], weights=[33, 33, 34])[0]
        
        if result == 'Mania':
            # User Request: 33% vowels, 67% consonants for Mania formats
            if random.random() < 0.33:
                mania_letter = random.choice('AEIOU')
            else:
                mania_letter = random.choice('BCDFGHJKLMNPQRSTVWXYZ')
            return f'{mania_letter} Mania'
        return result
