import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dict_dir = os.path.join(base_dir, 'dictionaries')
    trace_path = os.path.join(dict_dir, 'stats_trace.log')
    added_path = os.path.join(dict_dir, 'added_words.txt')

    # Load WordValidator to filter out words that are in official dictionaries
    print("[Restore] Initializing Word Validator to check official dictionaries...")
    from word_validator import word_validator

    print(f"[Restore] Reading trace log: {trace_path}")
    
    # 1. Parse trace log for active additions
    active_words = set()
    if os.path.exists(trace_path):
        with open(trace_path, 'r') as f:
            for line in f:
                if 'ADD_START:' in line:
                    parts = line.split("'")
                    if len(parts) >= 2:
                        word = parts[1].strip().upper()
                        if word:
                            active_words.add(word)
                elif 'REMOVE_START:' in line:
                    parts = line.split("'")
                    if len(parts) >= 2:
                        word = parts[1].strip().upper()
                        if word:
                            active_words.discard(word)
    else:
        print("[Restore] Error: trace log not found!")
        return

    print(f"[Restore] Found {len(active_words)} unique active custom words in trace log.")

    # 2. Load current words in added_words.txt
    current_words = []
    if os.path.exists(added_path):
        with open(added_path, 'r') as f:
            current_words = [line.strip().upper() for line in f if line.strip()]
    
    # 3. Combine both lists to get the full list of potential added words
    combined = list(active_words) + current_words
    
    # Remove duplicates while preserving order
    unique_combined = []
    seen = set()
    for w in combined:
        if w not in seen:
            unique_combined.append(w)
            seen.add(w)
            
    # 4. Filter out any words that are already in official dictionaries
    clean_combined = []
    filtered_words = []
    for w in unique_combined:
        if word_validator.is_valid_word_authoritative(w):
            filtered_words.append(w)
        else:
            clean_combined.append(w)

    print(f"[Restore] Filtered out {len(filtered_words)} words that are already valid in CSW/NWL/16plus.")
    if len(filtered_words) > 0:
        print(f"[Restore] Examples of filtered words: {filtered_words[:10]}")

    # Write back to file
    with open(added_path, 'w') as f:
        for w in clean_combined:
            f.write(f"{w}\n")
            
    print(f"[Restore] SUCCESS! Cleaned and updated added_words.txt. Total words now: {len(clean_combined)}")

if __name__ == '__main__':
    main()
