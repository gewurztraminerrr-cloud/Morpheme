import time
import os
import sys
import numpy as np

# Mocking the dictionary loading logic from app.py
def load_tools_dictionary(dict_name):
    dict_path = os.path.join('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries', f'{dict_name}.txt')
    with open(dict_path, 'r') as f:
        words = set(word.strip().upper() for word in f)
    
    long_path = os.path.join('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries', '16plus.txt')
    if os.path.exists(long_path):
        with open(long_path, 'r') as f:
            long_words = {line.strip().upper() for line in f if line.strip()}
        words = words | long_words

    word_list = sorted(list(words))
    matrix = np.zeros((len(word_list), 26), dtype=np.uint8)
    masks = np.zeros(len(word_list), dtype=np.uint32)
    
    for i, word in enumerate(word_list):
        mask = 0
        for char in word:
            if 'A' <= char <= 'Z':
                c_idx = ord(char) - ord('A')
                matrix[i, c_idx] += 1
                mask |= (1 << c_idx)
        masks[i] = mask
    
    lens = np.array([len(w) for w in word_list], dtype=np.uint8)
    return {'words': word_list, 'matrix': matrix, 'lens': lens, 'masks': masks}

def get_lis(nums):
    if not nums: return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(len(nums[:i])):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if dp else 0

def calculate_morpheme_metric(source, target):
    s_len, t_len = len(source), len(target)
    if s_len == 0 or t_len == 0: return 99, 0
    
    prev = [0] * (t_len + 1)
    curr = [0] * (t_len + 1)
    for char_s in source:
        for j in range(1, t_len + 1):
            if char_s == target[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                p_v = prev[j]
                c_v = curr[j-1]
                curr[j] = p_v if p_v > c_v else c_v
        prev[:] = curr
    
    linearity = prev[t_len]
    if linearity == 0: return 99, 0
    if t_len - linearity > 6: return 99, linearity

    dp = [[0] * (t_len + 1) for _ in range(s_len + 1)]
    for i in range(1, s_len + 1):
        s_i = source[i-1]
        dp_prev = dp[i-1]
        dp_curr = dp[i]
        for j in range(1, t_len + 1):
            if s_i == target[j-1]: dp_curr[j] = dp_prev[j-1] + 1
            else:
                v1 = dp_prev[j]
                v2 = dp_curr[j-1]
                dp_curr[j] = v1 if v1 >= v2 else v2
            
    matched_s_indices = []
    i, j = s_len, t_len
    while i > 0 and j > 0:
        if source[i-1] == target[j-1]:
            matched_s_indices.append(i-1)
            i -= 1; j -= 1
        elif dp[i-1][j] >= dp[i][j-1]: i -= 1
        else: j -= 1
    matched_s_indices.reverse()
    
    lis_len = get_lis(matched_s_indices)
    relocations = len(matched_s_indices) - lis_len
    first_idx = matched_s_indices[0]
    last_idx = matched_s_indices[-1]
    paid_deletions = (last_idx - first_idx + 1) - len(matched_s_indices)
    insertions = t_len - len(matched_s_indices)
    
    return relocations + paid_deletions + insertions, linearity

def profile_combo(search_term):
    print(f"Profiling word: {search_term} with OPTIMIZED logic")
    start_total = time.time()
    
    dict_data = load_tools_dictionary('CSW')
    load_time = time.time() - start_total
    print(f"Dictionary load time: {load_time:.4f}s")
    
    word_list = dict_data['words']
    dict_matrix = dict_data['matrix']
    dict_lens = dict_data['lens']
    dict_masks = dict_data['masks']
    
    source_len = len(search_term)
    search_term_rev = search_term[::-1]
    
    s_vec = np.zeros(26, dtype=np.uint8)
    s_mask = 0
    for char in search_term:
        if 'A' <= char <= 'Z':
            c_idx = ord(char) - ord('A')
            s_vec[c_idx] += 1
            s_mask |= (1 << c_idx)
            
    start_pruning = time.time()
    mask_intersection = dict_masks & s_mask
    m = mask_intersection.astype(np.uint32)
    m = (m & 0x55555555) + ((m >> 1) & 0x55555555)
    m = (m & 0x33333333) + ((m >> 2) & 0x33333333)
    m = (m & 0x0F0F0F0F) + ((m >> 4) & 0x0F0F0F0F)
    m = (m & 0x00FF00FF) + ((m >> 8) & 0x00FF00FF)
    m = (m & 0x0000FFFF) + ((m >> 16) & 0x0000FFFF)
    passed_mask = (m >= 3)
    
    shared_counts = np.minimum(dict_matrix, s_vec).sum(axis=1)
    
    min_shared = 1
    if source_len == 5: min_shared = 3
    if source_len == 6: min_shared = 4
    if source_len >= 7: min_shared = 5
    
    len_diffs = np.abs(dict_lens.astype(np.int16) - source_len)
    candidates = np.where(
        passed_mask & 
        (len_diffs <= 6) & 
        (shared_counts >= min_shared) &
        (dict_lens.astype(np.int16) - shared_counts <= 6)
    )[0]
    pruning_time = time.time() - start_pruning
    print(f"Pruning time: {pruning_time:.4f}s, Candidates: {len(candidates)}")
    
    # Initialize Groups (Using sets)
    mp_groups = {i: set() for i in range(7)} # 0MP to 6MP
    lic_groups = {}
    
    start_loop = time.time()
    for idx in candidates:
        word = word_list[idx]
        target_len = int(dict_lens[idx])
        shared_count = int(shared_counts[idx])
        
        # 1-pass primary check
        best_mp, _ = calculate_morpheme_metric(search_term, word)
        
        # Subsequent passes only if promising
        if best_mp > 1:
            m2, _ = calculate_morpheme_metric(search_term, word[::-1])
            best_mp = min(best_mp, m2)
        if best_mp > 2:
            m3, _ = calculate_morpheme_metric(search_term_rev, word)
            best_mp = min(best_mp, m3)
            
        if best_mp <= 6:
            mp_groups[best_mp].add(word)
                
        if shared_count >= 1:
            if shared_count not in lic_groups: lic_groups[shared_count] = set()
            # LIC logic check (simplified for profile)
            valid = False
            if shared_count == 5: valid = (target_len < 7)
            elif shared_count == 6: valid = (target_len < 8)
            elif shared_count >= 7: valid = (target_len < 10)
            if valid:
                lic_groups[shared_count].add(word)
            
    loop_time = time.time() - start_loop
    print(f"Loop time: {loop_time:.4f}s")
    print(f"Total time: {time.time() - start_total:.4f}s")

if __name__ == "__main__":
    profile_combo("WASTRIE")
