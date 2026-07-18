import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spinner_set import SpinnerSet

print("Spinning 100 times for 6x8:")
counts = {}
for i in range(100):
    params = SpinnerSet.generate_params("6x8", is_24h=False)
    # convert to tuple key
    key = (params.get('dictionary'), params.get('word_count_range'), params.get('min_word_length'))
    counts[key] = counts.get(key, 0) + 1

for key, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{key}: {count}")
