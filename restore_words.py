import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dict_dir = os.path.join(base_dir, 'dictionaries')
    trace_path = os.path.join(dict_dir, 'stats_trace.log')
    added_path = os.path.join(dict_dir, 'added_words.txt')

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
    
    # 3. Determine which words are missing
    new_to_inject = [w for w in sorted(active_words) if w not in current_words]
    
    if not new_to_inject:
        print("[Restore] All words are already present in added_words.txt. No restore needed.")
        return

    print(f"[Restore] Restoring {len(new_to_inject)} missing words to added_words.txt...")
    
    # Prepend new words (preserving chronological reverse order style)
    combined = new_to_inject + current_words
    
    # Write back to file
    with open(added_path, 'w') as f:
        for w in combined:
            f.write(f"{w}\n")
            
    print(f"[Restore] SUCCESS! {len(new_to_inject)} words restored. Total words now: {len(combined)}")

if __name__ == '__main__':
    main()
