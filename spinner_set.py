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
        # Default representative percentage within the difficulty's uniqueness band
        if base == 'Easy':
            pct = _rng.randint(22, 36)
        elif base == 'Hard':
            pct = _rng.randint(54, 68)
        else:
            pct = _rng.randint(39, 51)
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
            for _attempt in range(6):
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
                res = cand
                # Avoid repeating the exact same Spinner Set tuple back-to-back when previous_params is given
                if isinstance(previous_params, dict) and not is_24h:
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
                fallback_wc = _rng.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
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

        if p.get('_exact_wc_calculated') or p.get('_lock_word_count_range'):
            # Pre-calculated from an actual solved board; do not randomly reroll
            pass
        elif is_24h:
            p['board_format'] = 'Valued Letters'
            wc_range = '300-400'
            p['use_added_words'] = False
            p['dictionary'] = clean_base_dict
            # Ensure 300-400 is physically achievable on 24h boards without AW
            if min_word_length >= 7:
                min_word_length = floor_len
        elif is_aw_effective:
            # + AW dictionaries MUST ALWAYS be 300-400, 400-500, or 500+ (never 50-100, 100-200, or 200-300)
            if wc_range not in ['300-400', '400-500', '500+']:
                wc_range = _rng.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
            # For 8L min length with + AW, 300-400 is the reliable high-density target
            if min_word_length >= 8 and wc_range in ['400-500', '500+']:
                wc_range = '300-400'
        else:
            # Standard NWL and CSW dictionaries:
            # Maximum normal range is 300-400 (NEVER 400-500), plus 1% chance of 500+.
            # 50-100 is allowed for the top two min word lengths of the board dimension (e.g., 4L & 5L on 4x4).
            if wc_range == '400-500' or not wc_range or wc_range not in ['50-100', '100-200', '200-300', '300-400', '500+']:
                if min_word_length in low_wc_allowed_mins:
                    wc_range = _rng.choices(['50-100', '100-200', '200-300', '300-400', '500+'], weights=[15, 28, 28, 28, 1])[0]
                else:
                    wc_range = _rng.choices(['100-200', '200-300', '300-400', '500+'], weights=[33, 33, 33, 1])[0]

            # 50-100 is restricted to the top two min lengths for the board dimension
            if wc_range == '50-100' and min_word_length not in low_wc_allowed_mins:
                wc_range = _rng.choices(['100-200', '200-300', '300-400', '500+'], weights=[33, 33, 33, 1])[0]

            # Preserve the 1% 500+ roll by adjusting min_word_length down to floor_len if needed
            if wc_range == '500+' and min_word_length > floor_len:
                min_word_length = floor_len
            elif min_word_length >= 8 and wc_range == '300-400':
                wc_range = _rng.choices(['50-100', '100-200', '200-300'], weights=[20, 40, 40])[0]

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

            # 3. Difficulty (25% Easy | 50% Medium | 25% Hard) with percentage in brackets
            base_diff = _rng.choices(['Easy', 'Medium', 'Hard'], weights=[25, 50, 25])[0]
            difficulty = SpinnerSet.format_difficulty_with_pct(base_diff)

            # 4. Word Count Range (ALWAYS 300-400 for 24h rooms)
            if is_24h:
                wc_range = '300-400'
                if min_word_length >= 7:
                    min_word_length = allowed_mins[0]
            elif is_aw_effective:
                wc_range = _rng.choices(['300-400', '400-500', '500+'], weights=[33, 33, 34])[0]
                if min_word_length >= 8 and wc_range in ['400-500', '500+']:
                    wc_range = '300-400'
            else:
                wc_range = SpinnerSet._spin_word_count(dictionary, min_word_length, base_diff, dims, use_added_words=False)
                if wc_range == '500+' and min_word_length > allowed_mins[0]:
                    min_word_length = allowed_mins[0]

            # 5. Board Format
            board_format = SpinnerSet._spin_board_format(is_24h, board_dimensions, previous_params=previous_params)

            # 6. Bonus Word Length (always >= min_word_length)
            valid_bw = [l for l in [6, 7, 8, 9, 10] if l >= min_word_length] or [max(min_word_length, 6)]
            bw_len = _rng.choice(valid_bw)

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
        """Balanced difficulty selection: 25% Easy, 50% Medium, 25% Hard with percentage in brackets."""
        choices = ['Easy', 'Medium', 'Hard']
        weights = [25, 50, 25]
        return SpinnerSet.format_difficulty_with_pct(_rng.choices(choices, weights=weights)[0])

    @staticmethod
    def _spin_word_count(dictionary, min_word_length, difficulty, board_dimensions, use_added_words=False):
        d_upper = str(dictionary).upper()
        is_aw = use_added_words or "+ AW" in d_upper or "+AW" in d_upper or "ADDED" in d_upper
        try:
            mwl = int(min_word_length)
        except Exception:
            mwl = 3
        if is_aw:
            if mwl >= 8:
                return '300-400'
            choices = ['300-400', '400-500', '500+']
            weights = [33, 33, 34]
            return _rng.choices(choices, weights=weights)[0]
        else:
            allowed_mins = SpinnerSet.get_allowed_min_lengths(board_dimensions)
            low_wc_allowed_mins = allowed_mins[1:]  # Top two min lengths per dimension (e.g. 4L & 5L for 4x4)
            if mwl >= 8:
                choices = ['50-100', '100-200', '200-300', '500+']
                weights = [20, 39, 40, 1]
            elif mwl in low_wc_allowed_mins:
                choices = ['50-100', '100-200', '200-300', '300-400', '500+']
                weights = [15, 28, 28, 28, 1]
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
        """Original board format weights with repetition safeguard for rare modifiers."""
        if is_24h:
            return 'Valued Letters'

        if '3x3x3' in str(dimensions):
            return 'Normal'

        prev_fmt = None
        if isinstance(previous_params, dict):
            prev_fmt = str(previous_params.get('board_format', '')).strip()

        for _ in range(5):
            result = _rng.choices(
                ['Normal', 'Checkerboard', 'Equality Freq', 'Bounce', 'Density', 'Mania', 'Penalty', 'Either/Or', 'Bonus Letter', 'Valued Letters', 'Rotation', 'Double', 'Triple'],
                weights=[66, 12, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
            )[0]

            if result == 'Bounce':
                result = _rng.choices(['Bounce 1x', 'Bounce 2x', 'Bounce 3x'], weights=[33, 33, 34])[0]

            if result == 'Mania':
                # 33% vowels (A, E, I, O, U), 67% consonants
                if _rng.random() < 0.33:
                    mania_letter = _rng.choice(['A', 'E', 'I', 'O', 'U'])
                else:
                    mania_letter = _rng.choice(['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'])
                result = f'{mania_letter} Mania'

            # Safeguard: do not repeat rare non-Normal formats back-to-back
            if prev_fmt and prev_fmt != 'Normal' and result == prev_fmt:
                continue
            break
        return result

    @staticmethod
    def generate_subanagrams_params(previous_params=None):
        """Generate randomized parameters for Subanagrams (Practice) mode."""
        sequence_length = _rng.choice([6, 7, 8, 9, 10])
        # Calibrate min_word_length based on sequence length so >= 50 words is physically achievable
        if sequence_length == 6:
            min_word_length = 3
        elif sequence_length in [7, 8]:
            min_word_length = _rng.choice([3, 4])
        else:
            min_word_length = _rng.choice([3, 4, 5])

        dict_choice = _rng.choice(['CSW', 'NWL', 'CSW + AW', 'NWL + AW'])
        use_added_words = ('+ AW' in dict_choice or '+AW' in dict_choice)
        base_dict = dict_choice.replace('+ AW', '').replace('+AW', '').strip()
        mode = _rng.choice(['word', 'random'])

        # Realistic word count ranges calibrated to sequence length
        if sequence_length == 6:
            word_count_range = _rng.choice(['50-100', '100-200'])
        elif sequence_length == 7:
            word_count_range = _rng.choice(['50-100', '100-200', '200-300'])
        elif sequence_length == 8:
            word_count_range = _rng.choice(['100-200', '200-300', '300-400'])
        else:  # 9 or 10 letters
            word_count_range = _rng.choice(['100-200', '200-300', '300-400', '500+'])

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
            'board_format': 'Word Guaranteed' if mode == 'word' else 'Totally Random',
            'word_count_range': word_count_range,
            'difficulty': 'Practice',
            'bonus_word': '',
            'bonus_word_length': bonus_len,
            'generated_at': time.time()
        }

    @staticmethod
    def get_factchecked_wc_range(count, use_added_words=False, dictionary=None):
        """Map actual word count to the exact corresponding standard range bucket,
           respecting dictionary constraints (+AW is always >= 300-400; non-AW never 400-500).
        """
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 0

        d_upper = str(dictionary or '').upper()
        is_aw = bool(use_added_words or '+ AW' in d_upper or '+AW' in d_upper or 'ADDED' in d_upper)

        if is_aw:
            if count >= 500:
                return '500+'
            if count >= 400:
                return '400-500'
            return '300-400'

        if count >= 500:
            return '500+'
        if count >= 300:
            return '300-400'
        if count >= 200:
            return '200-300'
        if count >= 100:
            return '100-200'
        return '50-100'


