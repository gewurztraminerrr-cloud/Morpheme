
import os

desk_unique = "/Users/jeffbabiak/Desktop/randomTWLunique.txt"
nwl_full = "/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/NWL.txt"
current_unique = "/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/uniqueNWL.txt"

def load_dict(path):
    if not os.path.exists(path): return set()
    with open(path, 'r') as f:
        return {line.strip().upper() for line in f if line.strip()}

desk_set = load_dict(desk_unique)
nwl_set = load_dict(nwl_full)
current_set = load_dict(current_unique)

nwl_4l = {w for w in nwl_set if len(w) == 4}
desk_4l = {w for w in desk_set if len(w) == 4}

print(f"Total 4-letter words in NWL: {len(nwl_4l)}")
print(f"Total 4-letter words in Desktop Unique set: {len(desk_4l)}")

# Let's see some of the 4-letter words the Desktop tool considers "Unique"
print(f"Sample desktop 'unique' 4L: {list(desk_4l)[:20]}")

# If we want to reach 60% with 100-200 words, we need about 500-1000 4-letter words in our unique set.
# Or better yet, we can move a portion of NWL words into our unique set.
# If we want to be "scientific", we could rank them by frequency, but for now let's see why they are missing.
