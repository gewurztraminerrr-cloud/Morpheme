
import time

def get_lis(nums):
    if not nums: return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def calculate_mp_pass_new(source, target, source_len, target_len):
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
    if not target_matched: return 99, 0
    count = len(target_matched)
    matched_pos_in_source = [position[i] for i in sorted(target_matched)]
    lis_len = get_lis(matched_pos_in_source)
    moves = count - lis_len
    min_t, max_t = min(target_matched), max(target_matched)
    min_s, max_s = min(source_matched), max(source_matched)
    internal_inserts = (max_t - min_t + 1) - count
    internal_deletes = (max_s - min_s + 1) - count
    mp = moves + internal_inserts + internal_deletes
    if source_len == target_len:
        hamming = sum(1 for a, b in zip(source, target) if a != b)
        if mp > hamming: mp = hamming
    return mp, count

source = "WASTRIE"
target = "ASTRIDE"
n = 100000
start = time.time()
for _ in range(n):
    calculate_mp_pass_new(source, target, 7, 7)
end = time.time()
print(f"Time for {n} calls: {end - start:.4f}s")
print(f"Est time for 720k calls: {(end-start)*7.2:.2f}s")
