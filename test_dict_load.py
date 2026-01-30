import os
import time

def test_load():
    start = time.time()
    try:
        path = os.path.expanduser('~/Desktop/Definitions.txt')
        if not os.path.exists(path):
            print("File not found")
            return

        cache = {}
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.split(' - ', 1)
                if len(parts) == 2:
                    word = parts[0].strip()
                    definition = parts[1].strip()
                    cache[word] = definition
        
        end = time.time()
        print(f"Loaded {len(cache)} words in {end - start:.2f} seconds.")
        print(f"Sample: AA -> {cache.get('AA')}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_load()
