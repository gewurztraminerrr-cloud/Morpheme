import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    added_path = os.path.join(base_dir, 'dictionaries', 'added_words.txt')
    csw_path = os.path.join(base_dir, 'dictionaries', 'CSW.txt')
    
    if not os.path.exists(csw_path):
        print(f"CSW.txt not found at {csw_path}")
        return
        
    if not os.path.exists(added_path):
        print(f"added_words.txt not found at {added_path}")
        return
        
    print("Loading CSW words...")
    with open(csw_path, 'r') as f:
        csw_words = {line.strip().upper() for line in f if line.strip()}
        
    print("Loading current added words...")
    with open(added_path, 'r') as f:
        added_words = [line.strip().upper() for line in f if line.strip()]
        
    # Filter
    filtered_added_words = []
    seen = set()
    for w in added_words:
        if w not in csw_words and w not in seen:
            seen.add(w)
            filtered_added_words.append(w)
            
    print(f"Original lines count: {len(added_words)}")
    print(f"Filtered lines count: {len(filtered_added_words)}")
    
    # Write back
    with open(added_path, 'w') as f:
        for w in filtered_added_words:
            f.write(w + '\n')
            
    print("Write back complete!")

if __name__ == "__main__":
    main()
