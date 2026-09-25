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
VOWELS = set('AEIOU')

def get_subanagrams_dict(base_dict='CSW', use_added_words=False):
    """Load or retrieve pre-indexed dictionary features for fast subanagram solving."""
    cache_key = f"{base_dict.upper()}_{bool(use_added_words)}"
    if cache_key in SUBANAGRAMS_CACHE:
        return SUBANAGRAMS_CACHE[cache_key]

    base_dir = os.path.join(os.path.dirname(__file__), 'dictionaries')
    words_set = set()

    # Base dictionary file
    dict_file = os.path.join(base_dir, f"{base_dict.upper()}.txt")
    if os.path.exists(dict_file):
        with open(dict_file, 'r', encoding='utf-8', errors='ignore') as f:
            words_set.update(line.strip().split()[0].upper() for line in f if line.strip() and not line.strip().startswith('#'))

    # Custom additions
    extra_files = []
    if base_dict.upper() == 'NWL':
        extra_files = ['new_NWL.txt', 'custom_nwl.txt']
    elif base_dict.upper() == 'CSW':
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
        'by_len': by_len
    }
    SUBANAGRAMS_CACHE[cache_key] = entry
    return entry

def solve_subanagrams(sequence, dict_data, min_length=3):
    """Fast bitmask + Counter solver for finding all subanagrams of a letter sequence."""
    sequence = sequence.upper()
    seq_len = len(sequence)
    seq_counter = Counter(sequence)

    seq_mask = 0
    for ch in sequence:
        seq_mask |= (1 << (ord(ch) - 65))
    seq_inv_mask = (~seq_mask) & 0xFFFFFFFF

    masks = dict_data['masks']
    lens = dict_data['lens']
    word_list = dict_data['words']

    # Vectorized / fast loop
    results = []
    for i in range(len(word_list)):
        l = lens[i]
        if l < min_length or l > seq_len:
            continue
        if (masks[i] & seq_inv_mask) != 0:
            continue

        w = word_list[i]
        # Check letter counts
        w_counter = Counter(w)
        match = True
        for ch, cnt in w_counter.items():
            if seq_counter[ch] < cnt:
                match = False
                break
        if match:
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
    """Check if word count matches target range."""
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
    Enforces word count range with graceful fallback if attempt limit is reached.
    Returns: (board, results, bonus_word, final_params)
    """
    min_len = int(params.get('min_word_length', 3))
    seq_len = int(params.get('sequence_length', 8))
    base_dict = params.get('base_dictionary', 'CSW')
    use_aw = bool(params.get('use_added_words', False))
    mode = params.get('mode', 'word')
    target_range = params.get('word_count_range', '100-200')

    dict_data = get_subanagrams_dict(base_dict, use_aw)
    candidates_by_len = dict_data['by_len'].get(seq_len, [])

    # If word mode requested but no exact length words exist, fallback to random
    if mode == 'word' and not candidates_by_len:
        mode = 'random'

    best_candidate = None
    best_results = None
    best_diff = 999999

    target_mid = 75
    if target_range == '50-100': target_mid = 75
    elif target_range == '100-200': target_mid = 150
    elif target_range == '200-300': target_mid = 250
    elif target_range == '300-400': target_mid = 350
    elif target_range == '500+': target_mid = 550

    max_attempts = 50
    for attempt in range(max_attempts):
        if mode == 'word':
            chosen_word = random.choice(candidates_by_len)
            letters = list(chosen_word)
            random.shuffle(letters)
            for _ in range(5):
                if "".join(letters) != chosen_word or len(letters) <= 3:
                    break
                random.shuffle(letters)
            sequence = "".join(letters)
        else:
            # Totally Random letters with vowel constraint
            min_vowels = max(1, seq_len // 4)
            for _ in range(30):
                selected = random.choices(BAG, k=seq_len)
                vowel_cnt = sum(1 for c in selected if c in VOWELS)
                if min_vowels <= vowel_cnt < seq_len:
                    random.shuffle(selected)
                    sequence = "".join(selected)
                    break
            else:
                sequence = "".join(random.choices(BAG, k=seq_len))

        solved_words = solve_subanagrams(sequence, dict_data, min_length=min_len)
        cnt = len(solved_words)

        # Track best fallback
        diff = abs(cnt - target_mid)
        # Prioritize candidates with at least 50 words
        score = diff if cnt >= 50 else (diff + 10000)
        if score < best_diff:
            best_diff = score
            best_candidate = sequence
            best_results = solved_words

        if is_count_in_range(cnt, target_range):
            # Perfect match!
            best_candidate = sequence
            best_results = solved_words
            break

    # If best candidate has < 50 words, do passes in 'word' mode or gracefully reduce min_len
    while (not best_results or len(best_results) < 50):
        if candidates_by_len:
            for _ in range(30):
                chosen_word = random.choice(candidates_by_len)
                letters = list(chosen_word)
                random.shuffle(letters)
                sequence = "".join(letters)
                solved = solve_subanagrams(sequence, dict_data, min_length=min_len)
                if len(solved) >= 50:
                    best_candidate = sequence
                    best_results = solved
                    mode = 'word'
                    break
        if best_results and len(best_results) >= 50:
            break
        if min_len > 3:
            min_len -= 1
        else:
            break

    # Graceful fallback: If no sequence matches all parameters within attempt limit, default gracefully
    if not best_candidate or not best_results or len(best_results) < 50:
        # Guaranteed rich sequences
        rich_pool = ["EDUCATION", "REACTIONS", "STREAMING", "ORCHESTRA", "RELATIONS"]
        best_candidate = random.choice(rich_pool)[:seq_len]
        best_results = solve_subanagrams(best_candidate, dict_data, min_length=min_len)

    # Resolve bonus word: chosen at random from the solved subanagrams
    if best_results:
        bonus_word = random.choice(best_results)
    else:
        bonus_word = best_candidate

    final_params = dict(params)
    final_params['min_word_length'] = min_len
    final_params['sequence_length'] = len(best_candidate)
    final_params['bonus_word'] = bonus_word
    final_params['bonus_word_length'] = len(bonus_word)
    # Update word count range to reflect actual candidate
    actual_range = count_to_range_label(len(best_results))
    final_params['word_count_range'] = actual_range
    final_params['sequence'] = best_candidate
    final_params['mode'] = mode
    final_params['board_format'] = 'Word Guaranteed' if mode == 'word' else 'Totally Random'

    # Board representation as 1-row 2D list: [['S', 'E', 'Q', ...]]
    board = [list(best_candidate)]
    return board, best_results, bonus_word, final_params
