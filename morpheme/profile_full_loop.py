
from collections import Counter
import time

def get_lis(nums):
    if not nums: return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def calculate_mp_pass(source, target, source_len, target_len):
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
    mp = moves + ((max_t - min_t + 1) - count) + ((max_s - min_s + 1) - count)
    return mp, count

# Simulate dictionary
words = ["ASTRIDE", "CANTED", "RELIANT", "RELIANTLY", "TESTING", "DICTIONARY"] * 30000 # 180k words
source = "WASTRIE"
source_len = 7
source_counter = Counter(source)
source_set = set(source)
source_rev = source[::-1]

start = time.time()
n_processed = 0
for word in words:
    target_len = len(word)
    if abs(source_len - target_len) > 5: continue
    
    # Speed check
    target_set = set(word)
    if len(source_set & target_set) < 3: continue
    
    if source in word or source_rev in word:
        n_processed += 1
        continue
        
    mp, count = calculate_mp_pass(source, word, source_len, target_len)
    n_processed += 1
    
end = time.time()
print(f"Total time for {len(words)} words: {end - start:.2f}s")
print(f"Processed: {n_processed}")
