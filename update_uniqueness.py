
import os
import random

nwl_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/NWL.txt'
unique_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/uniqueNWL.txt'

def load_dict(path):
    if not os.path.exists(path): return []
    with open(path, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

nwl_words = load_dict(nwl_path)
unique_words = load_dict(unique_path)

# Extract 4-letter words from NWL
nwl_4l = [w for w in nwl_words if len(w) == 4]
print(f"Total 4-letter words in NWL: {len(nwl_4l)}")

# Add a subset of 4-letter words to unique set (to make 60% uniqueness achievable)
# We'll avoid the most common ones like THEM, THAT, TIME, etc.
# Since we don't have a frequency list, we'll just pick those with rare letters or just a large chunk.
common_avoid = ["THEM", "THAT", "TIME", "WITH", "THIS", "THEY", "HAVE", "FROM", "WORD", "WHAT", "SOME", "YOUR", "GOOD", "TAKE", "GIVE", "LOOK"]
candidate_4l = [w for w in nwl_4l if w not in common_avoid]

# Add 2500 words to the unique list
added_count = 0
unique_set = set(unique_words)
random.shuffle(candidate_4l)

for w in candidate_4l[:2500]:
    if w not in unique_set:
        unique_set.add(w)
        added_count += 1

# Write back to uniqueNWL.txt
with open(unique_path, 'w') as f:
    for w in sorted(list(unique_set)):
        f.write(w + "\n")

print(f"Added {added_count} four-letter words to {unique_path}.")
print(f"Total unique words now: {len(unique_set)}")
