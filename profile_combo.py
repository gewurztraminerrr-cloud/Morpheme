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
    
    best_mp = t_len + s_len
    
    char_to_s_indices = {}
    for idx, char in enumerate(source):
        if char not in char_to_s_indices:
            char_to_s_indices[char] = []
        char_to_s_indices[char].append(idx)
        
    def backtrack(t_idx, used_mask, matched):
        nonlocal best_mp
        
        m_len = len(matched)
        if m_len > 0:
            sub_lis = get_lis(matched)
            current_relocations = m_len - sub_lis
            current_paid_deletions = (max(matched) - min(matched) + 1) - m_len
        else:
            current_relocations = 0
            current_paid_deletions = 0
            
        # Lower bound on insertions is insertions made so far (t_idx - m_len)
        min_possible_cost = current_relocations + current_paid_deletions + (t_idx - m_len)
        
        if min_possible_cost >= best_mp:
            return
            
        if t_idx == t_len:
            actual_cost = current_relocations + current_paid_deletions + (t_len - m_len)
            if actual_cost < best_mp:
                best_mp = actual_cost
            return
            
        char = target[t_idx]
        if char in char_to_s_indices:
            for s_idx in char_to_s_indices[char]:
                if not (used_mask & (1 << s_idx)):
                    backtrack(t_idx + 1, used_mask | (1 << s_idx), matched + [s_idx])
                    
        backtrack(t_idx + 1, used_mask, matched)

    backtrack(0, 0, [])
    
    if best_mp > 6:
        return 99, linearity
        
    return best_mp, linearity

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
    
    dict_lens_int = dict_lens.astype(np.int16)
    candidates = np.where(
        passed_mask & 
        (np.abs(dict_lens_int - source_len) <= 3) & 
        (shared_counts >= dict_lens_int - 6)
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
        
        # Calculate forward MP (search_term -> candidate)
        m1_f, linearity = calculate_morpheme_metric(search_term, word)
        m2_f, _ = calculate_morpheme_metric(search_term, word[::-1])
        m3_f, _ = calculate_morpheme_metric(search_term_rev, word)
        forward_mp = min(m1_f, m2_f, m3_f)
        
        # Calculate backward MP (candidate -> search_term)
        m1_b, _ = calculate_morpheme_metric(word, search_term)
        m2_b, _ = calculate_morpheme_metric(word[::-1], search_term)
        m3_b, _ = calculate_morpheme_metric(word, search_term_rev)
        backward_mp = min(m1_b, m2_b, m3_b)
        
        # Apply asymmetric combination logic:
        if backward_mp == 0:
            best_mp = forward_mp
        elif forward_mp == 0:
            best_mp = 0
        else:
            best_mp = min(forward_mp, backward_mp)
            
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
