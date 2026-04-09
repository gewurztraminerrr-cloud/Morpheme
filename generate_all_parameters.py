
import random
import sys
import time
import os
import json

# Frequency and Alphabet
LETTER_FREQ_USER = [114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def load_dictionary(path):
    if not os.path.exists(path): return []
    with open(path, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

class WordTrie:
    def __init__(self, words, min_len=3):
        self.trie = {}
        for word in words:
            if len(word) < min_len: continue
            node = self.trie
            for char in word:
                if char not in node: node[char] = {}
                node = node[char]
            node['#'] = True

    def solve(self, board, rows, cols):
        found = set()
        def dfs(r, c, node, word, visited):
            if '#' in node: found.add(word)
            if len(word) >= 15: return
            for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited:
                    char = board[nr][nc]
                    if char in node:
                        visited.add((nr,nc))
                        dfs(nr,nc,node[char],word+char,visited)
                        visited.remove((nr,nc))
        for r in range(rows):
            for c in range(cols):
                char = board[r][c]
                if char in self.trie: dfs(r, c, self.trie[char], char, {(r,c)})
        return found

def embed(b, rows, cols, word):
    r, c = random.randint(0, rows-1), random.randint(0, cols-1)
    path = [(r, c)]
    for char in word[1:]:
        neighbors = [(r+dr, c+dc) for dr in [-1,0,1] for dc in [-1,0,1] if (dr!=0 or dc!=0) and 0<=r+dr<rows and 0<=c+dc<cols and (r+dr,c+dc) not in path]
        if not neighbors: return None
        nr, nc = random.choice(neighbors)
        path.append((nr, nc))
        r, c = nr, nc
    for i, (pr, pc) in enumerate(path): b[pr][pc] = word[i]
    return path

def generate_board_optimized(nwl_trie, unique_set, bonus_pool, rows, cols, min_words, max_words, target_u, min_len, max_time=15):
    start_time = time.time()
    best_u = -1
    best_data = None
    
    # Pre-select test pool for speed
    test_pool = list("ETAOINSR") + list("QZJXVWKBC")
    
    for attempt in range(50):
        if not bonus_pool: bonus_word = "SEED"
        else: bonus_word = random.choice(bonus_pool)
        
        board = [[None for _ in range(cols)] for _ in range(rows)]
        path = embed(board, rows, cols, bonus_word)
        if not path: continue
        
        # Initial fill
        for r in range(rows):
            for c in range(cols):
                if board[r][c] is None:
                    if random.random() < 0.3: board[r][c] = random.choice("QZJXV")
                    else: board[r][c] = random.choices(ALPHABET, weights=LETTER_FREQ_USER, k=1)[0]
        
        # IO Pass (1 pass for speed in batch)
        tiles = [(r,c) for r in range(rows) for c in range(cols) if (r,c) not in path]
        random.shuffle(tiles)
        for r, c in tiles:
            best_char = board[r][c]
            max_score = -1e9
            random.shuffle(test_pool)
            for char in test_pool[:10]:
                board[r][c] = char
                found = nwl_trie.solve(board, rows, cols)
                count = len(found)
                u = sum(1 for w in found if w in unique_set)
                u_pct = u / count if count > 0 else 0
                
                score = u_pct * 100.0
                if count < min_words: score -= (min_words - count)
                if count > max_words: score -= (count - max_words)
                
                if score > max_score:
                    max_score = score
                    best_char = char
            board[r][c] = best_char
        
        found = nwl_trie.solve(board, rows, cols)
        count = len(found)
        u_pct = sum(1 for w in found if w in unique_set) / count * 100 if count > 0 else 0
        
        if u_pct > best_u:
            best_u = u_pct
            best_data = (board, bonus_word, count, u_pct, list(found))
            if best_u >= target_u and min_words <= count <= max_words: break
            
        if (time.time() - start_time) > max_time: break
    
    return best_data

def main():
    params_path = '/Users/jeffbabiak/Desktop/parameters.txt'
    subset_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/params_subset.txt'
    if os.path.exists(subset_path):
        params_path = subset_path
        print(f"Using subset parameters from {subset_path}")
    
    nwl_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/NWL.txt'
    unique_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/uniqueNWL.txt'
    
    nwl_words = load_dictionary(nwl_path)
    unique_words = load_dictionary(unique_path)
    unique_set = set(unique_words)
    
    results = []
    
    with open(params_path, 'r') as f:
        lines = f.readlines()
        
    total_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
    current_count = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        
        current_count += 1
        # Parsers: 4x4 - 100-200 words - hard - 4LM - NWL
        parts = [p.strip() for p in line.split('-')]
        if len(parts) < 5: continue
        
        dims = parts[0]
        words_range_raw = parts[1].split(' ')[0]
        difficulty = parts[2]
        min_len_str = parts[3]
        dict_name = parts[4]
        
        try:
            rows, cols = map(int, dims.split('x'))
            if '200+' in words_range_raw: min_w, max_w = 200, 300
            elif '-' in words_range_raw: min_w, max_w = map(int, words_range_raw.split('-'))
            else: min_w, max_w = int(words_range_raw), int(words_range_raw)
            
            min_len = int(min_len_str.replace('LM', '').replace('LW', ''))
            
            target_u = 60 if difficulty.lower() == 'hard' else (40 if difficulty.lower() == 'medium' else 0)
            
            # Setup specific trie for this min_len
            trie = WordTrie(nwl_words, min_len=min_len)
            bp = [w for w in nwl_words if min_len+2 <= len(w) <= min_len+5]
            
            print(f"[{current_count}/{total_lines}] Processing: {line}...")
            board_data = generate_board_optimized(trie, unique_set, bp, rows, cols, min_w, max_w, target_u, min_len)
            
            if board_data:
                b, bonus, cnt, ratio, all_w = board_data
                results.append({
                    "params": line,
                    "grid": b,
                    "bonus": bonus,
                    "count": cnt,
                    "uniqueness": ratio,
                    "top_words": sorted(all_w, key=lambda x: (-len(x), x))[:10]
                })
            else:
                print(f"Failed to generate for: {line}")
                
        except Exception as e:
            print(f"Error parsing line '{line}': {e}")
            continue

    with open('all_parameter_boards.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nCompleted! Generated {len(results)} boards.")

if __name__ == "__main__": main()
