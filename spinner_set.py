"""
Spinner Set System for Accumulative Game Mode
Generates randomized parameters for each round

ARCHITECTURAL PREFACE MANDATE (DO NOT VIOLATE):
1. All board generation and Spinner Set rules/logic must be applied identically across
   Standard Game Rooms, Solo, With Friends (Private Matches), and Tournaments
   (excluding Multiplayer Subanagrams).
2. The Spinner Set declared at 0:45 during intermission is an ironclad contract:
   parameters must NEVER mutate at 0:00 or mid-round, and the generated board must
   strictly abide by the declared Spinner Set parameters.
"""

import random
import time

from word_validator import word_validator

# Use OS-backed cryptographic entropy so Spinner Set parameters never repeat cyclically
_rng = random.SystemRandom()


class SpinnerSet:
    @staticmethod
    def strip_difficulty_pct(difficulty):
        """Return clean base difficulty ('Easy', 'Medium', 'Hard', or 'Practice') without '(XX%)'."""
        if not difficulty:
            return 'Medium'
        raw = str(difficulty).split('(')[0].strip()
        low = raw.lower()
        if 'practice' in low:
            return 'Practice'
        if 'easy' in low or 'beginner' in low:
            return 'Easy'
        if 'hard' in low or 'expert' in low or 'difficult' in low:
            return 'Hard'
        if 'med' in low:
            return 'Medium'
        return 'Medium'

    @staticmethod
    def format_difficulty_with_pct(difficulty, ratio=None):
        """Ensure difficulty string ALWAYS includes a percentage in brackets, e.g. 'Medium (45%)'."""
        base = SpinnerSet.strip_difficulty_pct(difficulty)
        if base == 'Practice':
            return 'Practice'
        if ratio is not None:
            try:
                pct = int(round(float(ratio) * 100)) if float(ratio) <= 1.0 else int(round(float(ratio)))
                pct = max(1, min(99, pct))
                return f"{base} ({pct}%)"
            except Exception:
                pass
        s = str(difficulty or '')
        if '(' in s and '%' in s:
            try:
                inside = s.split('(', 1)[1].split(')', 1)[0].replace('%', '').strip()
                pct = int(round(float(inside)))
                return f"{base} ({pct}%)"
            except Exception:
                pass
        # Default representative percentage within the difficulty's uniqueness band (aligned with 4x4 thresholds)
        if base == 'Easy':
            pct = _rng.randint(11, 14)
        elif base == 'Hard':
            pct = _rng.randint(34, 44)
        else:
            pct = _rng.randint(20, 26)
        return f"{base} ({pct}%)"

    @staticmethod
    def get_allowed_min_lengths(board_dimensions):
        """Return ordered [low, med, high] minimum word lengths for the given board dimensions."""
        dims = str(board_dimensions).lower().replace(" ", "")
        if '4x4' in dims:
            return [3, 4, 5]
        elif '4x6' in dims:
            return [4, 5, 6]
        elif '5x7' in dims:
            return [5, 6, 7]
        else:  # 6x8, 3x3x3, etc.
            return [6, 7, 8]

    @staticmethod
    def generate_tournament_params():
        """Generate ALL tournament parameters using lobby-standard dimensions and times"""
        # 1. Broad Parameters
        board_dims = _rng.choice(['4x4', '4x6', '5x7', '6x8'])
        time_limit = _rng.choice([60, 90, 120, 180, 240, 300])

        # 2. Granular Parameters
        params = SpinnerSet.generate_params(board_dims, is_24h=False)

        # Merge
        params['board_dimensions'] = board_dims
        params['time_limit'] = time_limit
        return params

    @staticmethod
    def generate_params(board_dimensions, is_24h=False, is_split=False, previous_params=None):
        """Generate granular spinner parameters, enforcing Valued Letters and 300-400 words for 24h matches."""
        try:
            res = None
            for _attempt in range(20):
                cand = SpinnerSet._generate_params_raw(board_dimensions, is_24h, is_split, previous_params)
                if is_24h and isinstance(cand, dict):
                    cand['board_format'] = 'Valued Letters'
                    cand['word_count_range'] = '300-400'
                    cand['use_added_words'] = False
                    clean_dict = str(cand.get('dictionary', 'CSW')).replace('+ AW', '').replace('+AW', '').replace('ADDED_WORDS', '').replace('AW', '').strip()
                    if clean_dict not in ['NWL', 'CSW']:
                        clean_dict = 'CSW'
                    cand['dictionary'] = clean_dict

                if isinstance(cand, dict):
                    dict_val = cand.get('dictionary')
                    if dict_val:
                        if '+ AW' in str(dict_val) or '+AW' in str(dict_val):
                            cand['use_added_words'] = True
                        else:
                            cand['use_added_words'] = (str(dict_val).upper() == 'AW' or cand.get('use_added_words') is True)

                        if cand.get('use_added_words') is True:
                            clean_dict = str(dict_val).replace('+ AW', '').replace('+AW', '').strip()
                            if clean_dict == 'AW':
                                clean_dict = 'NWL'
                            cand['dictionary'] = f"{clean_dict} + AW"
                    else:
                        cand['dictionary'] = SpinnerSet._spin_dictionary()
                        cand['use_added_words'] = ('+ AW' in str(cand['dictionary']) or '+AW' in str(cand['dictionary']))

                cand = SpinnerSet.sanitize_params(cand, board_dimensions, is_24h)
                if is_24h and isinstance(cand, dict):
                    cand['board_format'] = 'Valued Letters'
                    cand['word_count_range'] = '300-400'
                    cand['use_added_words'] = False
                res = cand
                # Avoid repeating the exact same Spinner Set tuple back-to-back when previous_params is given
                if isinstance(previous_params, dict) and not is_24h:
                    prev_fmt_raw = str(previous_params.get('board_format', '')).strip()
                    curr_fmt_raw = str(cand.get('board_format', '')).strip()
                    # Strict safeguard: Do not repeat rare non-Normal formats back-to-back
                    if prev_fmt_raw and prev_fmt_raw.lower() != 'normal':
                        if curr_fmt_raw.lower() == prev_fmt_raw.lower():
                            continue
                        if 'mania' in curr_fmt_raw.lower() and 'mania' in prev_fmt_raw.lower():
                            continue
                        if 'bounce' in curr_fmt_raw.lower() and 'bounce' in prev_fmt_raw.lower():
                            continue

                    prev_sig = (
                        str(previous_params.get('dictionary', '')),
                        str(previous_params.get('word_count_range', '')),
                        str(previous_params.get('board_format', '')),
                        int(previous_params.get('min_word_length', 0) or 0),
                        SpinnerSet.strip_difficulty_pct(previous_params.get('difficulty', ''))
                    )
                    curr_sig = (
                        str(cand.get('dictionary', '')),
                        str(cand.get('word_count_range', '')),
                        str(cand.get('board_format', '')),
                        int(cand.get('min_word_length', 0) or 0),
                        SpinnerSet.strip_difficulty_pct(cand.get('difficulty', ''))
                    )
                    if curr_sig == prev_sig:
                        continue
                break

            return res
        except Exception as e:
            print(f"[SpinnerSet] CRITICAL WRAPPER ERROR: {e}")
            allowed_mins = SpinnerSet.get_allowed_min_lengths(board_dimensions)
            fallback_min = allowed_mins[0]
            fallback_dict = _rng.choice(['NWL', 'CSW']) if is_24h else _rng.choice(['NWL', 'CSW', 'NWL + AW', 'CSW + AW'])
            fallback_aw = ('+ AW' in fallback_dict) and not is_24h
            if is_24h:
                fallback_wc = '300-400'
            elif fallback_aw:
                if '4x4' in str(board_dimensions):
                    fallback_wc = '300-400'
                elif '4x6' in str(board_dimensions):
                    fallback_wc = _rng.choices(['300-400', '400-500'], weights=[50, 50])[0]
                else:
                    fallback_wc = _rng.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
            else:
                if '4x4' in str(board_dimensions):
                    fallback_wc = _rng.choices(['100-200', '200-300', '300-400'], weights=[34, 33, 33])[0]
                elif '4x6' in str(board_dimensions):
                    fallback_wc = _rng.choices(['100-200', '200-300', '300-400'], weights=[34, 33, 33])[0]
                else:
                    fallback_wc = _rng.choices(['100-200', '200-300', '300-400', '500+'], weights=[33, 33, 33, 1])[0]

            fallback = {
                'difficulty': SpinnerSet.format_difficulty_with_pct(_rng.choices(['Easy', 'Medium', 'Hard'], weights=[25, 50, 25])[0]),
                'dictionary': fallback_dict,
                'use_added_words': fallback_aw,
                'word_count_range': fallback_wc,
                'board_format': 'Valued Letters' if is_24h else 'Normal',
                'min_word_length': fallback_min,
                'bonus_word_length': 8,
                'generated_at': time.time()
            }
            return SpinnerSet.sanitize_params(fallback, board_dimensions, is_24h)

    @staticmethod
    def sanitize_params(res, board_dimensions='4x4', is_24h=False):
        """Ironclad density/min-length sanitization to enforce exact Spinner Set rules."""
        if not isinstance(res, dict):
            return res

        p = dict(res)
        dims = str(p.get('board_dimensions') or board_dimensions).lower().replace(" ", "")
        allowed_mins = SpinnerSet.get_allowed_min_lengths(dims)
        floor_len = allowed_mins[0]
        max_len_for_dims = allowed_mins[-1]
        low_wc_allowed_mins = allowed_mins[1:]  # Top two min lengths per dimension (e.g. 4L & 5L for 4x4)

        # Normalize word_count_range string format
        wc = p.get('word_count_range')
        if isinstance(wc, (list, tuple)):
            if len(wc) >= 2:
                p['word_count_range'] = f"{wc[0]}-{wc[1]}"
            elif len(wc) == 1:
                p['word_count_range'] = str(wc[0])
        elif isinstance(wc, str):
            p['word_count_range'] = wc.replace(',', '-').strip()

        raw_dict_name = str(p.get('dictionary', 'NWL')).upper()
        is_aw_effective = (
            raw_dict_name in ['AW', 'ADDED_WORDS', 'ALL']
            or p.get('use_added_words') is True
            or '+ AW' in raw_dict_name
            or '+AW' in raw_dict_name
        ) and not is_24h

        clean_base_dict = raw_dict_name.replace('+ AW', '').replace('+AW', '').replace('ADDED_WORDS', '').strip()
        if clean_base_dict not in ['NWL', 'CSW']:
            clean_base_dict = 'CSW' if 'CSW' in clean_base_dict else 'NWL'

        if is_aw_effective:
            p['dictionary'] = f"{clean_base_dict} + AW"
            p['use_added_words'] = True
        else:
            p['dictionary'] = clean_base_dict
            p['use_added_words'] = False

        try:
            min_word_length = int(p.get('min_word_length'))
        except Exception:
            min_word_length = floor_len

        if min_word_length < floor_len:
            min_word_length = floor_len
        if min_word_length > max_len_for_dims:
            min_word_length = max_len_for_dims

        wc_range = p.get('word_count_range')

        if is_24h:
            p['board_format'] = 'Valued Letters'
            wc_range = '300-400'
            p['use_added_words'] = False
            p['dictionary'] = clean_base_dict
            # Ensure 300-400 is physically achievable on 24h boards without AW
            if min_word_length > floor_len:
                min_word_length = floor_len
        elif p.get('_exact_wc_calculated') or p.get('_lock_word_count_range'):
            # Pre-calculated from an actual solved board — preserve the exact truthful range!
            wc_range = p.get('word_count_range') or ('300-400' if is_aw_effective else '100-200')
            if is_aw_effective and wc_range not in ['300-400', '400-500', '500+']:
                wc_range = '300-400'
        elif is_aw_effective:
            # + AW dictionaries target high counts when spinning random params
            if '4x4' in dims:
                if min_word_length >= 5:
                    wc_range = '100-200'
                elif min_word_length == 4:
                    if wc_range not in ['200-300', '300-400']:
                        wc_range = _rng.choices(['200-300', '300-400'], weights=[50, 50])[0]
                else:
                    wc_range = '300-400'
            elif '4x6' in dims:
                if min_word_length >= 6:
                    wc_range = '100-200'
                elif min_word_length == 5:
                    if wc_range not in ['200-300', '300-400']:
                        wc_range = _rng.choices(['200-300', '300-400'], weights=[50, 50])[0]
                else:
                    if wc_range not in ['300-400', '400-500']:
                        wc_range = _rng.choices(['300-400', '400-500'], weights=[50, 50])[0]
            else:
                if wc_range not in ['300-400', '400-500', '500+']:
                    wc_range = _rng.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
        else:
            # Standard NWL and CSW dictionaries:
            # Maximum normal range is 300-400 (NEVER 400-500). 500+ is only available on large boards (5x7, 6x8, Cube).
            # 50-100 is allowed for the two highest min word lengths of the board dimension (e.g. 4L & 5L on 4x4, 5L & 6L on 4x6, 6L & 7L on 5x7, 7L & 8L on 6x8/Cube).
            if '4x4' in dims:
                if min_word_length >= 5:
                    if wc_range not in ['50-100', '100-200']:
                        wc_range = _rng.choices(['50-100', '100-200'], weights=[40, 60])[0]
                elif min_word_length == 4:
                    if wc_range not in ['50-100', '100-200', '200-300']:
                        wc_range = _rng.choices(['50-100', '100-200', '200-300'], weights=[20, 40, 40])[0]
                else:
                    if wc_range not in ['100-200', '200-300', '300-400']:
                        wc_range = _rng.choices(['100-200', '200-300', '300-400'], weights=[34, 33, 33])[0]
            elif '4x6' in dims:
                if min_word_length >= 6:
                    if wc_range not in ['50-100', '100-200']:
                        wc_range = _rng.choices(['50-100', '100-200'], weights=[40, 60])[0]
                elif min_word_length == 5:
                    if wc_range not in ['50-100', '100-200', '200-300']:
                        wc_range = _rng.choices(['50-100', '100-200', '200-300'], weights=[20, 40, 40])[0]
                else:
                    if wc_range not in ['100-200', '200-300', '300-400']:
                        wc_range = _rng.choices(['100-200', '200-300', '300-400'], weights=[34, 33, 33])[0]
            else:
                if wc_range == '400-500' or not wc_range or wc_range not in ['50-100', '100-200', '200-300', '300-400', '500+']:
                    if min_word_length in low_wc_allowed_mins:
                        wc_range = _rng.choices(['50-100', '100-200', '200-300', '300-400', '500+'], weights=[9, 30, 30, 30, 1])[0]
                    else:
                        wc_range = _rng.choices(['100-200', '200-300', '300-400', '500+'], weights=[33, 33, 33, 1])[0]
                if wc_range == '50-100' and min_word_length not in low_wc_allowed_mins:
                    wc_range = _rng.choices(['100-200', '200-300', '300-400', '500+'], weights=[33, 33, 33, 1])[0]

        # Enforce physical ceilings for small grids regardless of origin
        if '4x4' in dims:
            if min_word_length >= 5 and wc_range not in ['50-100', '100-200']:
                wc_range = '100-200'
            elif min_word_length == 4 and wc_range in ['400-500', '500+']:
                wc_range = '300-400' if is_aw_effective else '200-300'
            elif wc_range in ['400-500', '500+']:
                wc_range = '300-400'
        elif '4x6' in dims:
            if min_word_length >= 6 and wc_range not in ['50-100', '100-200']:
                wc_range = '100-200'
            elif min_word_length == 5 and wc_range in ['400-500', '500+']:
                wc_range = '300-400' if is_aw_effective else '200-300'
            elif wc_range == '500+':
                wc_range = '400-500' if is_aw_effective else '300-400'

        p['min_word_length'] = min_word_length
        p['word_count_range'] = wc_range

        # Ensure difficulty always has a percentage in brackets (unless Practice)
        if 'difficulty' in p and p['difficulty']:
            p['difficulty'] = SpinnerSet.format_difficulty_with_pct(p['difficulty'])
        else:
            p['difficulty'] = SpinnerSet.format_difficulty_with_pct(_rng.choices(['Easy', 'Medium', 'Hard'], weights=[25, 50, 25])[0])

        # Bonus word length MUST always be >= min_word_length and in [6..10]
        valid_bw_Choices = [l for l in [6, 7, 8, 9, 10] if l >= min_word_length] or [max(min_word_length, 6)]
        bw_raw = p.get('bonus_word_length')
        try:
            bw_val = int(bw_raw)
            if bw_val < min_word_length or bw_val > 10:
                p['bonus_word_length'] = _rng.choice(valid_bw_Choices)
            else:
                p['bonus_word_length'] = bw_val
        except Exception:
            p['bonus_word_length'] = _rng.choice(valid_bw_Choices)

        return p

    @staticmethod
    def _generate_params_raw(board_dimensions, is_24h=False, is_split=False, previous_params=None):
        """Generate granular spinner parameters matching exact Odds Window probabilities."""
        try:
            dims = str(board_dimensions).lower().replace(" ", "")
            allowed_mins = SpinnerSet.get_allowed_min_lengths(dims)

            # 1. Min Word Length (25% Low | 50% Med | 25% High)
            min_word_length = _rng.choices(allowed_mins, weights=[25, 50, 25])[0]

            # 2. Dictionary (25% NWL | 25% CSW | 25% NWL + AW | 25% CSW + AW, no AW for 24h rooms)
            if is_24h:
                dictionary = _rng.choice(['NWL', 'CSW'])
            else:
                dictionary = _rng.choices(['NWL', 'CSW', 'NWL + AW', 'CSW + AW'], weights=[25, 25, 25, 25])[0]

            is_aw_effective = ('+ AW' in dictionary or '+AW' in dictionary) and not is_24h
            use_added_words = is_aw_effective

            # 3. Difficulty (25% Easy | 50% Medium | 25% Hard, 21% Hard on 6x8/Cube) with percentage in brackets
            difficulty = SpinnerSet._spin_difficulty(board_dimensions, min_word_length)

            # 4. Word Count Range (ALWAYS 300-400 for 24h rooms)
            if is_24h:
                wc_range = '300-400'
            else:
                wc_range = SpinnerSet._spin_word_count(dictionary, min_word_length, SpinnerSet.strip_difficulty_pct(difficulty), dims, use_added_words=is_aw_effective)
                if wc_range == '500+' and ('4x4' in dims or '4x6' in dims) and min_word_length > allowed_mins[0]:
                    min_word_length = allowed_mins[0]

            # 5. Board Format
            board_format = SpinnerSet._spin_board_format(is_24h, board_dimensions, previous_params=previous_params)
            prev_normal_streak = int(previous_params.get('normal_streak', 0)) if isinstance(previous_params, dict) else 0
            normal_streak = (prev_normal_streak + 1) if (board_format and board_format.lower() == 'normal') else 0

            # 6. Bonus Word Length (20% each for 6L, 7L, 8L, 9L, 10L; automatically capped to >= min_word_length)
            rolled_bw = _rng.choices([6, 7, 8, 9, 10], weights=[20, 20, 20, 20, 20])[0]
            bw_len = max(min_word_length, rolled_bw)

            res = {
                'min_word_length': min_word_length,
                'difficulty': difficulty,
                'word_count_range': wc_range,
                'dictionary': dictionary,
                'use_added_words': use_added_words,
                'board_format': board_format,
                'normal_streak': normal_streak,
                'bonus_word_length': bw_len,
                'generated_at': time.time()
            }

            return res
        except Exception as e:
            print(f"[SpinnerSet] CRITICAL GENERATOR ERROR: {e}")
            return {
                'difficulty': SpinnerSet.format_difficulty_with_pct(_rng.choices(['Easy', 'Medium', 'Hard'], weights=[25, 50, 25])[0]),
                'dictionary': _rng.choice(['NWL', 'CSW']),
                'word_count_range': '300-400' if is_24h else _rng.choices(['100-200', '200-300', '300-400', '500+'], weights=[33, 33, 33, 1])[0],
                'board_format': 'Valued Letters' if is_24h else 'Normal',
                'min_word_length': 3,
                'bonus_word_length': 8,
                'generated_at': time.time()
            }

    @staticmethod
    def _spin_min_word_length(board_dimensions):
        """Based on board dimensions with specified percentages"""
        allowed_mins = SpinnerSet.get_allowed_min_lengths(board_dimensions)
        return _rng.choices(allowed_mins, weights=[25, 50, 25])[0]

    @staticmethod
    def _spin_difficulty(board_dimensions='4x4', min_word_length=3):
        """Balanced difficulty selection: 25% Easy, 50% Medium, 25% Hard (21% on 6x8/Cube) with percentage in brackets."""
        dims = str(board_dimensions).lower().replace(" ", "")
        if '6x8' in dims or '3x3x3' in dims:
            choices = ['Easy', 'Medium', 'Hard']
            weights = [25, 50, 21]
        else:
            choices = ['Easy', 'Medium', 'Hard']
            weights = [25, 50, 25]
        return SpinnerSet.format_difficulty_with_pct(_rng.choices(choices, weights=weights)[0])

    @staticmethod
    def _spin_word_count(dictionary, min_word_length, difficulty, board_dimensions, use_added_words=False):
        d_upper = str(dictionary).upper()
        is_aw = use_added_words or "+ AW" in d_upper or "+AW" in d_upper or "ADDED" in d_upper
        dims = str(board_dimensions).lower().replace(" ", "")
        try:
            mwl = int(min_word_length)
        except Exception:
            mwl = 3

        if is_aw:
            if '4x4' in dims:
                if mwl >= 5:
                    return '100-200'
                elif mwl == 4:
                    return _rng.choices(['200-300', '300-400'], weights=[50, 50])[0]
                else:
                    return '300-400'
            elif '4x6' in dims:
                if mwl >= 6:
                    return '100-200'
                elif mwl == 5:
                    return _rng.choices(['200-300', '300-400'], weights=[50, 50])[0]
                else:
                    return _rng.choices(['300-400', '400-500'], weights=[50, 50])[0]
            else:
                choices = ['300-400', '400-500', '500+']
                weights = [33, 33, 34]
                return _rng.choices(choices, weights=weights)[0]
        else:
            if '4x4' in dims:
                if mwl >= 5:
                    return _rng.choices(['50-100', '100-200'], weights=[40, 60])[0]
                elif mwl == 4:
                    return _rng.choices(['50-100', '100-200', '200-300'], weights=[20, 40, 40])[0]
                else:
                    return _rng.choices(['100-200', '200-300', '300-400'], weights=[34, 33, 33])[0]
            elif '4x6' in dims:
                if mwl >= 6:
                    return _rng.choices(['50-100', '100-200'], weights=[40, 60])[0]
                elif mwl == 5:
                    return _rng.choices(['50-100', '100-200', '200-300'], weights=[20, 40, 40])[0]
                else:
                    return _rng.choices(['100-200', '200-300', '300-400'], weights=[34, 33, 33])[0]
            else:
                allowed_mins = SpinnerSet.get_allowed_min_lengths(board_dimensions)
                low_wc_allowed_mins = allowed_mins[1:]
                if mwl in low_wc_allowed_mins:
                    choices = ['50-100', '100-200', '200-300', '300-400', '500+']
                    weights = [9, 30, 30, 30, 1]
                else:
                    choices = ['100-200', '200-300', '300-400', '500+']
                    weights = [33, 33, 33, 1]
                return _rng.choices(choices, weights=weights)[0]

    @staticmethod
    def _spin_dictionary():
        """Equal probability: 25% NWL, 25% CSW, 25% NWL + AW, 25% CSW + AW"""
        choices = ['NWL', 'CSW', 'NWL + AW', 'CSW + AW']
        weights = [25, 25, 25, 25]
        return _rng.choices(choices, weights=weights)[0]

    @staticmethod
    def _spin_board_format(is_24h=False, dimensions='4x4', previous_params=None):
        """Original board format weights from Spinner Set Odds window."""
        if is_24h:
            return 'Valued Letters'

        if '3x3x3' in str(dimensions):
            return 'Normal'

        prev_fmt = None
        if isinstance(previous_params, dict):
            prev_fmt = str(previous_params.get('board_format', '')).strip()

        # Odds window exact weights:
        # Normal 66%, Checkerboard 12%, Equality Freq 4%,
        # Bounce 2%, Penalty 2%, Mania 2%, Either/Or 2%, Bonus Letter 2%, Valued Letters 2%, Density 2%, Rotation 2%,
        # Double 1%, Triple 1%
        formats = [
            'Normal', 'Checkerboard', 'Equality Freq', 'Bounce', 'Density',
            'Mania', 'Penalty', 'Either/Or', 'Bonus Letter', 'Valued Letters',
            'Rotation', 'Double', 'Triple'
        ]
        weights = [66, 12, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]

        for _ in range(25):
            result = _rng.choices(formats, weights=weights)[0]

            if result == 'Bounce':
                result = _rng.choices(['Bounce 1x', 'Bounce 2x', 'Bounce 3x'], weights=[33, 33, 34])[0]

            if result == 'Mania':
                # 33% vowels (A, E, I, O, U), 67% consonants
                if _rng.random() < 0.33:
                    mania_letter = _rng.choice(['A', 'E', 'I', 'O', 'U'])
                else:
                    mania_letter = _rng.choice(['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'])
                result = f'{mania_letter} Mania'

            # Strict anti-repetition safeguard: Do not repeat rare non-Normal format back-to-back
            if prev_fmt:
                if prev_fmt.lower() != 'normal' and result.strip().lower() == prev_fmt.strip().lower():
                    continue
                if 'mania' in result.lower() and 'mania' in prev_fmt.lower():
                    continue
                if 'bounce' in result.lower() and 'bounce' in prev_fmt.lower():
                    continue
            break
        return result

    @staticmethod
    def resolve_specific_format(requested, actual=None):
        """Turn a generic format request into a concrete, displayable one.
        'Mania' -> '<L> Mania' and 'Bounce' -> 'Bounce <N>x'. If the board's actual format
        is already a concrete variant of the request, keep it so label and board always match."""
        req = str(requested or '').strip()
        low = req.lower()
        act = str(actual or '').strip()
        if low == 'mania':
            parts = act.split()
            if len(parts) == 2 and parts[1].lower() == 'mania' and len(parts[0]) == 1 and parts[0].isalpha():
                return f"{parts[0].upper()} Mania"
            if _rng.random() < 0.33:
                letter = _rng.choice(['A', 'E', 'I', 'O', 'U'])
            else:
                letter = _rng.choice(list('BCDFGHJKLMNPQRSTVWXYZ'))
            return f"{letter} Mania"
        if low == 'bounce':
            if act.lower().startswith('bounce ') and act.lower().endswith('x'):
                return act
            return _rng.choice(['Bounce 1x', 'Bounce 2x', 'Bounce 3x'])
        return requested

    @staticmethod
    def generate_subanagrams_params(previous_params=None):
        """Generate randomized parameters for Subanagrams mode adhering strictly to
        the Subanagrams Spinner Set Odds window in templates/index.html.
        
        Odds Specifications:
        1. Letters: Sequence Length (20% each):
           6 Letters (20%) | 7 Letters (20%) | 8 Letters (20%) | 9 Letters (20%) | 10 Letters (20%)
        2. Min: Minimum Word Length:
           - 6 Letters: 3L Only (100%)
           - 7 Letters: 3L (50%) | 4L (50%)
           - 8, 9, 10 Letters: 3L (33.3%) | 4L (33.3%) | 5L (33.3%)
        3. Dict: Word Lists (25% each):
           CSW (25%) | NWL (25%) | CSW + AW (25%) | NWL + AW (25%)
        4. Words: Target Word Count Range (20% each spun):
           50-100 words (20%) | 100-200 words (20%) | 200-300 words (20%) | 300-400 words (20%) | 500+ words (20%)
        5. Format: Sequence Generation:
           Word Guaranteed (50%) | Totally Random (50%)
        6. Bonus Word: Secret bonus word chosen at random, length displayed in parameters.
        """
        prev_p = previous_params if isinstance(previous_params, dict) else {}
        prev_seq = prev_p.get('sequence_length')
        prev_wc = prev_p.get('word_count_range')
        prev_dict = prev_p.get('dictionary')
        prev_fmt = prev_p.get('board_format')

        # Joint distribution sampling so marginals are strictly 20% for each Sequence Length
        # and strictly 20% for each Target Word Count Range, while guaranteeing physical feasibility:
        joint_choices = [
            # 6 Letters (total 0.20): 50-100 (0.10), 100-200 (0.10)
            (6, '50-100', 0.100),
            (6, '100-200', 0.100),
            # 7 Letters (total 0.20): 50-100 (0.04), 100-200 (0.04), 200-300 (0.08), 300-400 (0.04)
            (7, '50-100', 0.040),
            (7, '100-200', 0.040),
            (7, '200-300', 0.080),
            (7, '300-400', 0.040),
            # 8 Letters (total 0.20): 50-100 (0.03), 100-200 (0.03), 200-300 (0.04), 300-400 (0.05), 500+ (0.05)
            (8, '50-100', 0.030),
            (8, '100-200', 0.030),
            (8, '200-300', 0.040),
            (8, '300-400', 0.050),
            (8, '500+', 0.050),
            # 9 Letters (total 0.20): 50-100 (0.015), 100-200 (0.015), 200-300 (0.02), 300-400 (0.055), 500+ (0.095)
            (9, '50-100', 0.015),
            (9, '100-200', 0.015),
            (9, '200-300', 0.020),
            (9, '300-400', 0.055),
            (9, '500+', 0.095),
            # 10 Letters (total 0.20): 50-100 (0.015), 100-200 (0.015), 200-300 (0.02), 300-400 (0.055), 500+ (0.095)
            (10, '50-100', 0.015),
            (10, '100-200', 0.015),
            (10, '200-300', 0.020),
            (10, '300-400', 0.055),
            (10, '500+', 0.095),
        ]

        # Apply strict anti-repetition against previous round if available
        filtered_choices = []
        if prev_seq is not None:
            filtered_choices = [c for c in joint_choices if c[0] != prev_seq and c[1] != prev_wc]
            if not filtered_choices:
                filtered_choices = [c for c in joint_choices if c[0] != prev_seq]
        if not filtered_choices:
            filtered_choices = joint_choices

        weights = [c[2] for c in filtered_choices]
        chosen_pair = _rng.choices(filtered_choices, weights=weights, k=1)[0]
        sequence_length, word_count_range = chosen_pair[0], chosen_pair[1]

        # Min word length calibrated strictly per odds modal
        if sequence_length == 6:
            min_word_length = 3
        elif sequence_length == 7:
            if word_count_range == '300-400':
                min_word_length = 3
            else:
                min_word_length = _rng.choice([3, 4])
        elif sequence_length == 8:
            if word_count_range == '500+':
                min_word_length = _rng.choice([3, 4])
            else:
                min_word_length = _rng.choices([3, 4, 5], weights=[1/3, 1/3, 1/3], k=1)[0]
        else: # 9, 10 Letters
            min_word_length = _rng.choices([3, 4, 5], weights=[1/3, 1/3, 1/3], k=1)[0]

        # Dictionary: 25% each with anti-repetition
        dict_options = ['CSW', 'NWL', 'CSW + AW', 'NWL + AW']
        if prev_dict in dict_options:
            dict_options = [d for d in dict_options if d != prev_dict]
        dict_choice = _rng.choice(dict_options)
        use_added_words = ('+ AW' in dict_choice or '+AW' in dict_choice)
        base_dict = dict_choice.replace('+ AW', '').replace('+AW', '').strip()

        # Format: 50% Word Guaranteed, 50% Totally Random
        format_options = ['word', 'random']
        if prev_fmt:
            p_mode = 'word' if 'word' in str(prev_fmt).lower() else 'random'
            # 50/50 random selection
        mode = _rng.choice(format_options)
        board_format = 'Word Guaranteed' if mode == 'word' else 'Totally Random'

        bonus_len = _rng.choice(range(min_word_length, sequence_length + 1))

        return {
            'board_dimensions': 'subanagrams',
            'time_limit': 120,
            'min_word_length': min_word_length,
            'sequence_length': sequence_length,
            'dictionary': dict_choice,
            'base_dictionary': base_dict,
            'use_added_words': use_added_words,
            'mode': mode,
            'board_format': board_format,
            'word_count_range': word_count_range,
            'difficulty': 'Practice',
            'bonus_word': '',
            'bonus_word_length': bonus_len,
            'generated_at': time.time()
        }

    @staticmethod
    def get_subanagrams_factchecked_wc_range(count):
        """Map actual word count to the exact truthful range bucket for Subanagrams:
             - >= 500 words -> '500+' (top bracket in Subanagrams odds)
             - 300 - 400 words -> '300-400'
             - 200 - 299 words -> '200-300'
             - 100 - 199 words -> '100-200'
             - < 100 words -> '50-100'
        """
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 0

        if count >= 500:
            return '500+'
        if count >= 300:
            return '300-400'
        if count >= 200:
            return '200-300'
        if count >= 100:
            return '100-200'
        return '50-100'

    @staticmethod
    def get_factchecked_wc_range(count, use_added_words=False, dictionary=None):
        """Map actual word count to the exact truthful range bucket:
             - 500+ words -> '500+'
             - 400 - 499 words -> '400-500'
             - 300 - 399 words -> '300-400'
             - 200 - 299 words -> '200-300'
             - 100 - 199 words -> '100-200'
             - < 100 words -> '50-100'

           CRITICAL RULE:
           "NWL + AW" and "CSW + AW" word lists should NEVER have a word count below 300.
           The only options for them are 300-400, 400-500, and 500+ words. Nothing else.
        """
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 0

        if count >= 500:
            return '500+'
        if count >= 400:
            return '400-500'
        if count >= 300:
            return '300-400'
        if count >= 200:
            return '200-300'
        if count >= 100:
            return '100-200'
        return '50-100'


