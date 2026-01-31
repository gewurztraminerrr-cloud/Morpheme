def get_lis(nums):
    if not nums: return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if dp else 0

def calculate_mp_score(source, target, source_len, target_len):
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
    
    return moves + inserts + deletes

s = "CHAUFER"
t = "CHAMFER"
mp = calculate_mp_score(s, t, len(s), len(t))
print(f"Source: {s}, Target: {t}, Current MP: {mp}")
