import os
import time

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    added_path = os.path.join(base_dir, 'dictionaries', 'added_words.txt')
    mwik_path = '/Users/jeffbabiak/Documents/mwik.txt'
    
    print("Reading current added_words.txt...")
    existing_words = set()
    ordered_existing = []
    if os.path.exists(added_path):
        with open(added_path, 'r') as f:
            for line in f:
                w = line.strip().upper()
                if w and w not in existing_words:
                    existing_words.add(w)
                    ordered_existing.append(w)
                    
    print(f"Loaded {len(existing_words)} existing unique added words.")
    
    print(f"Reading and merging {mwik_path}...")
    start_time = time.time()
    
    # We will read mwik.txt and add any new words
    with open(mwik_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            w = line.strip().upper()
            if w and w not in existing_words:
                existing_words.add(w)
                ordered_existing.append(w)
                
    print(f"Merged set contains {len(existing_words)} unique words.")
    
    # User Request: "so organize or sort the list how you need to so as not to lag"
    # Sorting alphabetically is highly standard and efficient for loading and display.
    print("Sorting the merged words list...")
    ordered_existing.sort()
    
    print(f"Writing merged words back to {added_path}...")
    with open(added_path, 'w') as f:
        for w in ordered_existing:
            f.write(f"{w}\n")
            
    print(f"Successfully merged and sorted in {time.time() - start_time:.2f} seconds!")

if __name__ == '__main__':
    main()
