import traceback

def get_lis(nums):
    """Calculates Longest Increasing Subsequence length."""
    if not nums:
        return 0
    # Standard O(n log n) or O(n^2) approach. Words are short, O(n^2) is negligible.
    # Using DP (O(n^2)) for simplicity and correctness with small N.
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if dp else 0

def calculate_mp_score(source, target, source_len, target_len):
    """Calculates MP score (Moves + Inserts + Deletes) using LIS."""
    # Map source indices to target indices (First Fit)
    position = [-1] * target_len
    for s_idx, s_char in enumerate(source):
        for t_idx, t_char in enumerate(target):
            if s_char == t_char and position[t_idx] == -1:
                position[t_idx] = s_idx
                break
    
    matched_indices = [p for p in position if p != -1]
    matched_count = len(matched_indices)
    
    lis_len = get_lis(matched_indices)
    
    moves = matched_count - lis_len
    inserts = target_len - matched_count
    deletes = source_len - matched_count
    
    return moves + inserts + deletes, matched_count

try:
    source = "GATEMAN"
    target_word = "NAMETAG"
    
    print(f"Testing Source: {source}, Target: {target_word}")
    
    # 1. Forward
    print("Running Forward...")
    mp_fwd, cnt_fwd = calculate_mp_score(source, target_word, len(source), len(target_word))
    print(f"Forward Result: MP={mp_fwd}, Matches={cnt_fwd}")
    
    # 2. Reverse
    print("Running Reverse...")
    target_rev = target_word[::-1]
    mp_rev, cnt_rev = calculate_mp_score(source, target_rev, len(source), len(target_word))
    print(f"Reverse Result: MP={mp_rev}, Matches={cnt_rev}")
    
    print("Success!")

except Exception:
    traceback.print_exc()
