
def get_lis(nums):
    if not nums: return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def calculate_mp_fair(source, target):
    source_len = len(source)
    target_len = len(target)
    
    # 1. Best possible alignment (greedy position mapping)
    position = [-1] * target_len
    target_matched = []
    source_matched = []
    for s_idx, s_char in enumerate(source):
        # Find FIRST available occurrence in target
        for t_idx, t_char in enumerate(target):
            if s_char == t_char and position[t_idx] == -1:
                position[t_idx] = s_idx
                target_matched.append(t_idx)
                source_matched.append(s_idx)
                break
                
    if not target_matched:
        return 99
        
    count = len(target_matched)
    
    # 2. Longest Increasing Subsequence of the MATCHED source indices
    # (as they appear in the target order)
    matched_pos_in_source = [position[i] for i in sorted(target_matched)]
    lis_len = get_lis(matched_pos_in_source)
    moves = count - lis_len
    
    # 3. Internal Edits
    min_t, max_t = min(target_matched), max(target_matched)
    min_s, max_s = min(source_matched), max(source_matched)
    
    internal_inserts = (max_t - min_t + 1) - count
    internal_deletes = (max_s - min_s + 1) - count
    
    # Prefix/Suffix extensions are IGNORED (MP 0 for those)
    mp = moves + internal_inserts + internal_deletes
    
    return mp

# Tests
examples = [
    ("WASTRIE", "ASTRIDE"),  # Expected: 1 (Internal 'D')
    ("ANTED", "CANTED"),    # Expected: 0 (Prefix 'C' ignored)
    ("NAILER", "RELIANT"),   # Expected: 0? (Set AEILNR in both)
    ("NAILER", "RELIANTLY"), # Expected: 0? (Subset AEILNR)
    ("CAT", "CORTES"),       # Expected: ? (C and T match? No, C and T match with O R E S around them)
]

for s, t in examples:
    print(f"{s} -> {t}: MP = {calculate_mp_fair(s, t)}")
