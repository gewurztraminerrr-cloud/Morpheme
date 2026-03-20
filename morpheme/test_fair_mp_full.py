
def get_lis(nums):
    if not nums: return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def calculate_mp_pass_fair(source, target):
    source_len = len(source)
    target_len = len(target)
    
    # 1. Best possible alignment (greedy)
    position = [-1] * target_len
    target_matched = []
    source_matched = []
    for s_idx, s_char in enumerate(source):
        for t_idx, t_char in enumerate(target):
            if s_char == t_char and position[t_idx] == -1:
                position[t_idx] = s_idx
                target_matched.append(t_idx)
                source_matched.append(s_idx)
                break
                
    if not target_matched:
        return 99, 0
        
    count = len(target_matched)
    
    # Order matched indices as they appear in the target
    matched_pos_in_source = [position[i] for i in sorted(target_matched)]
    lis_len = get_lis(matched_pos_in_source)
    moves = count - lis_len
    
    # Internal Edits
    min_t, max_t = min(target_matched), max(target_matched)
    min_s, max_s = min(source_matched), max(source_matched)
    
    internal_inserts = (max_t - min_t + 1) - count
    internal_deletes = (max_s - min_s + 1) - count
    
    mp = moves + internal_inserts + internal_deletes
    return mp, count

def calculate_mp_fair_full(source, target):
    source_rev = source[::-1]
    target_rev = target[::-1]
    
    passes = [
        calculate_mp_pass_fair(source, target),
        calculate_mp_pass_fair(source, target_rev),
        calculate_mp_pass_fair(source_rev, target),
        calculate_mp_pass_fair(source_rev, target_rev)
    ]
    
    return min(passes, key=lambda x: x[0])[0]

# Tests
examples = [
    ("WASTRIE", "ASTRIDE"),
    ("ANTED", "CANTED"),
    ("NAILER", "RELIANT"),
    ("NAILER", "RELIANTLY"),
]

for s, t in examples:
    print(f"{s} -> {t}: MP = {calculate_mp_fair_full(s, t)}")
