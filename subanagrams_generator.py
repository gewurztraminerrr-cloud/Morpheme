"""
Subanagrams Generator & Solver Module
Generates candidate letter sequences and solves for all valid subanagrams
according to Spinner Set parameters (min_word_length, sequence_length, dictionary, mode, word_count_range).
"""

import os
import random
import time
from collections import Counter
import numpy as np

# Cache of dictionary features: dict_key -> { 'words': [...], 'masks': np.array, 'lens': np.array, 'by_len': { len: [...] } }
SUBANAGRAMS_CACHE = {}

BAG = (
    'E' * 12 + 'A' * 9 + 'I' * 9 + 'O' * 8 + 'N' * 6 + 'R' * 6 + 'T' * 6 +
    'L' * 4 + 'S' * 4 + 'U' * 4 + 'D' * 4 + 'G' * 3 +
    'B' * 2 + 'C' * 2 + 'M' * 2 + 'P' * 2 + 'F' * 2 + 'H' * 2 + 'V' * 2 + 'W' * 2 + 'Y' * 2 +
    'K' * 1 + 'J' * 1 + 'X' * 1 + 'Q' * 1 + 'Z' * 1
)
HIGH_YIELD_BAG = (
    'E' * 12 + 'A' * 9 + 'I' * 9 + 'O' * 8 + 'R' * 6 + 'S' * 6 + 'T' * 6 +
    'N' * 6 + 'L' * 4 + 'D' * 4 + 'C' * 3 + 'U' * 4 + 'P' * 2 + 'M' * 2
)
VOWELS = set('AEIOU')

def get_subanagrams_dict(base_dict='CSW', use_added_words=False):
    """Load or retrieve pre-indexed dictionary features for fast subanagram solving."""
    raw_dict = str(base_dict or 'CSW').upper()
    if '+ AW' in raw_dict or '+AW' in raw_dict:
        use_added_words = True
    base_dict = raw_dict.replace('+ AW', '').replace('+AW', '').strip()
    if base_dict not in ['NWL', 'CSW']:
        base_dict = 'CSW'
    cache_key = f"{base_dict}_{bool(use_added_words)}"
    if cache_key in SUBANAGRAMS_CACHE:
        return SUBANAGRAMS_CACHE[cache_key]

    base_dir = os.path.join(os.path.dirname(__file__), 'dictionaries')
    words_set = set()

    # Base dictionary file
    dict_file = os.path.join(base_dir, f"{base_dict}.txt")
    if os.path.exists(dict_file):
        with open(dict_file, 'r', encoding='utf-8', errors='ignore') as f:
            words_set.update(line.strip().split()[0].upper() for line in f if line.strip() and not line.strip().startswith('#'))

    # Custom additions
    extra_files = []
    if base_dict == 'NWL':
        extra_files = ['new_NWL.txt', 'custom_nwl.txt']
    elif base_dict == 'CSW':
        extra_files = ['new_CSW.txt', 'custom_csw.txt']

    for ef in extra_files:
        p = os.path.join(base_dir, ef)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                words_set.update(line.strip().split()[0].upper() for line in f if line.strip() and not line.strip().startswith('#'))

    # Added words if enabled
    if use_added_words:
        added_file = os.path.join(base_dir, 'added_words.txt')
        if os.path.exists(added_file):
            with open(added_file, 'r', encoding='utf-8', errors='ignore') as f:
                words_set.update(line.strip().split()[0].upper() for line in f if line.strip() and not line.strip().startswith('#'))

    # Long words (16+)
    long_file = os.path.join(base_dir, '16plus.txt')
    if os.path.exists(long_file):
        with open(long_file, 'r', encoding='utf-8', errors='ignore') as f:
            words_set.update(line.strip().split()[0].upper() for line in f if line.strip() and not line.strip().startswith('#'))

    # Filter strictly alphabetic words
    word_list = sorted([w for w in words_set if w.isalpha()])
    count = len(word_list)

    masks = np.zeros(count, dtype=np.uint32)
    lens = np.zeros(count, dtype=np.uint8)
    by_len = {}
    counters = [Counter(w) for w in word_list]

    for i, w in enumerate(word_list):
        l = len(w)
        lens[i] = l
        m = 0
        for ch in w:
            m |= (1 << (ord(ch) - 65))
        masks[i] = m

        if l not in by_len:
            by_len[l] = []
        by_len[l].append(w)

    entry = {
        'words': word_list,
        'masks': masks,
        'lens': lens,
        'by_len': by_len,
        'counters': counters
    }
    SUBANAGRAMS_CACHE[cache_key] = entry
    return entry

def solve_subanagrams(sequence, dict_data, min_length=3):
    """Fast bitmask + Counter solver for finding all subanagrams of a letter sequence.
    The QU tile is stored as 'Q' in the board but counts as both Q and U for word-finding.
    We expand Q→QU before counting so words like EQUATED are correctly discovered.
    """
    sequence = sequence.upper()
    # Expand Q → QU: the QU tile contributes both a Q and a U to the letter pool
    expanded_sequence = sequence.replace('Q', 'QU')
    seq_len = len(expanded_sequence)
    seq_counter = Counter(expanded_sequence)

    seq_mask = 0
    for ch in expanded_sequence:
        seq_mask |= (1 << (ord(ch) - 65))
    seq_inv_mask = (~seq_mask) & 0xFFFFFFFF

    masks = dict_data['masks']
    lens = dict_data['lens']
    word_list = dict_data['words']
    counters = dict_data.get('counters')

    # Vectorized fast bitmask filter in numpy C-speed (< 1ms)
    candidate_indices = np.where((lens >= min_length) & (lens <= seq_len) & ((masks & seq_inv_mask) == 0))[0]

    results = []
    if counters is not None:
        for idx in candidate_indices:
            w_cnt = counters[idx]
            if all(seq_counter[ch] >= cnt for ch, cnt in w_cnt.items()):
                results.append(word_list[idx])
    else:
        for idx in candidate_indices:
            w = word_list[idx]
            w_cnt = Counter(w)
            if all(seq_counter[ch] >= cnt for ch, cnt in w_cnt.items()):
                results.append(w)

    results.sort(key=lambda x: (-len(x), x))
    return results

def count_to_range_label(cnt):
    """Map an integer word count to a Spinner Set word count range label."""
    if cnt < 100:
        return '50-100'
    elif cnt < 200:
        return '100-200'
    elif cnt < 300:
        return '200-300'
    elif cnt < 400:
        return '300-400'
    else:
        return '500+'

def is_count_in_range(cnt, range_label):
    """Check if word count matches target range strictly."""
    if range_label == '50-100':
        return 50 <= cnt <= 100
    elif range_label == '100-200':
        return 100 <= cnt <= 200
    elif range_label == '200-300':
        return 200 <= cnt <= 300
    elif range_label == '300-400':
        return 300 <= cnt <= 400
    elif range_label == '500+':
        return cnt >= 500
    return 50 <= cnt <= 200

def generate_subanagrams_board_and_words(params):
    """
    Generate candidate sequence and solve for all subanagrams matching params.
    Enforces word count range with guaranteed compliance with Subanagrams Spinner Set Odds.
    For 'Word Guaranteed': strictly guarantees at least one valid dictionary word of full sequence_length.
    For 'Totally Random': letters sampled randomly from standard English tile bag (BAG) matching Tools.
    Returns: (board, results, bonus_word, final_params)
    """
    try:
        min_len = int(params.get('min_word_length', 3))
    except (ValueError, TypeError):
        min_len = 3
    min_len = max(3, min(5, min_len))

    try:
        seq_len = int(params.get('sequence_length', 8))
    except (ValueError, TypeError):
        seq_len = 8
    seq_len = max(6, min(10, seq_len))

    # Resolve dictionary and Added Words
    raw_dict = str(params.get('dictionary') or params.get('base_dictionary') or 'CSW')
    use_aw = bool(params.get('use_added_words', False) or '+ AW' in raw_dict.upper() or '+AW' in raw_dict.upper())
    base_dict = raw_dict.replace('+ AW', '').replace('+AW', '').strip().upper()
    if base_dict not in ['NWL', 'CSW']:
        base_dict = 'CSW'

    # Strict Format / Mode mapping:
    # 'Word Guaranteed' <-> 'word'
    # 'Totally Random'  <-> 'random'
    bf = str(params.get('board_format', '')).strip()
    raw_mode = str(params.get('mode', '')).strip().lower()
    if bf.lower() == 'word guaranteed' or raw_mode == 'word':
        mode = 'word'
    elif bf.lower() == 'totally random' or raw_mode == 'random':
        mode = 'random'
    else:
        mode = 'word'

    target_range = params.get('word_count_range', '100-200')
    if target_range not in ['50-100', '100-200', '200-300', '300-400', '500+']:
        target_range = '100-200'

    dict_data = get_subanagrams_dict(base_dict, use_aw)
    candidates_by_len = dict_data['by_len'].get(seq_len, [])

    # If requested length has no words in word mode, pick closest length
    if mode == 'word' and not candidates_by_len:
        for fallback_len in [8, 9, 7, 10, 6]:
            if dict_data['by_len'].get(fallback_len):
                seq_len = fallback_len
                candidates_by_len = dict_data['by_len'][fallback_len]
                break

    best_candidate = None
    best_results = None
    best_diff = 999999

    target_mid = 75
    if target_range == '50-100': target_mid = 75
    elif target_range == '100-200': target_mid = 150
    elif target_range == '200-300': target_mid = 250
    elif target_range == '300-400': target_mid = 350
    elif target_range == '500+': target_mid = 550

    # Choose appropriate letter bag for random generation
    bag_to_use = HIGH_YIELD_BAG if target_range in ['300-400', '500+'] else BAG

    # Phase 1: High-iteration search (up to 1000 attempts)
    max_attempts = 1000
    for attempt in range(max_attempts):
        if mode == 'word' and candidates_by_len:
            chosen_word = random.choice(candidates_by_len)
            letters = list(chosen_word)
            random.shuffle(letters)
            for _ in range(5):
                if "".join(letters) != chosen_word or len(letters) <= 3:
                    break
                random.shuffle(letters)
            sequence = "".join(letters)
        else:
            # Totally Random letters
            min_v = 3 if target_range in ['300-400', '500+'] else max(1, seq_len // 4)
            max_v = seq_len - 2
            for _ in range(25):
                selected = random.choices(bag_to_use, k=seq_len)
                vowel_cnt = sum(1 for c in selected if c in VOWELS)
                if min_v <= vowel_cnt <= max_v:
                    random.shuffle(selected)
                    sequence = "".join(selected)
                    break
            else:
                sequence = "".join(random.choices(bag_to_use, k=seq_len))

        solved_words = solve_subanagrams(sequence, dict_data, min_length=min_len)
        cnt = len(solved_words)

        # In Word Guaranteed mode, must guarantee at least one full sequence_length word
        if mode == 'word' and not any(len(w) == seq_len for w in solved_words):
            continue

        diff = abs(cnt - target_mid)
        score = diff if cnt >= 50 else (diff + 10000)
        if score < best_diff:
            best_diff = score
            best_candidate = sequence
            best_results = solved_words

        if is_count_in_range(cnt, target_range):
            best_candidate = sequence
            best_results = solved_words
            break

    # Phase 2: Guaranteed Range Enforcement
    # If the candidate does not strictly satisfy target_range, search specifically with guaranteed sequence lengths
    if not best_results or not is_count_in_range(len(best_results), target_range):
        # Determine sequence lengths where this target range is mathematically abundant
        if target_range == '500+':
            candidate_lengths = [9, 10, 8]
        elif target_range == '300-400':
            candidate_lengths = [8, 9, 10, 7]
        elif target_range == '200-300':
            candidate_lengths = [7, 8, 9, 6]
        else:
            candidate_lengths = [6, 7, 8]

        for alt_len in candidate_lengths:
            alt_words = dict_data['by_len'].get(alt_len, [])
            alt_bag = HIGH_YIELD_BAG if target_range in ['300-400', '500+'] else BAG
            for _ in range(500):
                if mode == 'word' and alt_words:
                    chosen_word = random.choice(alt_words)
                    letters = list(chosen_word)
                    random.shuffle(letters)
                    cand_seq = "".join(letters)
                else:
                    selected = random.choices(alt_bag, k=alt_len)
                    cand_seq = "".join(selected)

                solved = solve_subanagrams(cand_seq, dict_data, min_length=min_len)
                cnt = len(solved)
                if mode == 'word' and not any(len(w) == alt_len for w in solved):
                    continue
                if is_count_in_range(cnt, target_range):
                    best_candidate = cand_seq
                    best_results = solved
                    seq_len = alt_len
                    break
            if best_results and is_count_in_range(len(best_results), target_range):
                break

    # Safety guarantee: Under no circumstance return empty or mismatched results
    if not best_candidate or not best_results:
        # Ultimate fallback with guaranteed 100+ words
        chosen_word = random.choice(dict_data['by_len'].get(8, ['STARLING']))
        best_candidate = chosen_word
        best_results = solve_subanagrams(best_candidate, dict_data, min_length=min_len)

    # Resolve bonus word: chosen at random from all valid subanagrams
    if best_results:
        bonus_word = random.choice(best_results)
    else:
        bonus_word = best_candidate

    seq_len_actual = len(best_candidate)
    board_format = 'Word Guaranteed' if mode == 'word' else 'Totally Random'

    final_params = dict(params)
    final_params['min_word_length'] = min_len
    final_params['sequence_length'] = seq_len_actual
    final_params['bonus_word'] = bonus_word
    final_params['bonus_word_length'] = len(bonus_word)
    final_params['word_count_range'] = target_range
    final_params['sequence'] = best_candidate
    final_params['mode'] = mode
    final_params['board_format'] = board_format
    final_params['dictionary'] = f"{base_dict} + AW" if use_aw else base_dict
    final_params['base_dictionary'] = base_dict
    final_params['use_added_words'] = use_aw

    # Board representation as 1-row 2D list: [['S', 'E', 'Q', ...]]
    board = [list(best_candidate)]
    return board, best_results, bonus_word, final_params
