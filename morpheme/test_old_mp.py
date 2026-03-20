
def get_lis(nums):
    if not nums: return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def calculate_mp_pass_v1(source, target):
    source_len = len(source)
    target_len = len(target)
    position = [-1] * target_len
    for s_idx, s_char in enumerate(source):
        for t_idx, t_char in enumerate(target):
            if s_char == t_char and position[t_idx] == -1:
                position[t_idx] = s_idx
                break
    matched_indices = [p for p in position if p != -1]
    count = len(matched_indices)
    count2 = get_lis(matched_indices)
    micro_procedures = (count - count2) + (target_len - count) + (source_len - count)
    if source_len == target_len:
        count3 = sum(1 for a, b in zip(source, target) if a != b)
        if micro_procedures > count3: micro_procedures = count3
    return micro_procedures

print(f"WASTRIE vs ASTRIDE: {calculate_mp_pass_v1('WASTRIE', 'ASTRIDE')}")
print(f"RELIANTLY vs RELIANT: {calculate_mp_pass_v1('RELIANTLY', 'RELIANT')}")
print(f"ANTED vs CANTED: {calculate_mp_pass_v1('ANTED', 'CANTED')}")
