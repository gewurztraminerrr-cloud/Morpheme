
import random
import sys
import time
import os

# Frequencies
LETTER_FREQ_USER = [114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
UNLIKELY_POOL = list("CMPHVFGWYBKJXQZ")

def load_dictionary(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

class WordTrie:
    def __init__(self, words, min_len=4):
        self.trie = {}
        self.min_len = min_len
        for word in words:
            if len(word) < min_len: continue
            node = self.trie
            for char in word:
                if char not in node: node[char] = {}
                node = node[char]
            node['#'] = True

    def solve(self, board):
        found = set()
        def dfs(r, c, node, word, visited):
            if '#' in node: found.add(word)
            if len(word) >= 15: return
            for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<4 and 0<=nc<4 and (nr,nc) not in visited:
                    char = board[nr][nc]
                    if char in node:
                        visited.add((nr,nc))
                        dfs(nr,nc,node[char],word+char,visited)
                        visited.remove((nr,nc))
        for r in range(4):
            for c in range(4):
                char = board[r][c]
                if char in self.trie: dfs(r, c, self.trie[char], char, {(r,c)})
        return found

def main():
    nwl_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/NWL.txt'
    unique_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/uniqueNWL.txt'
    nwl_words = load_dictionary(nwl_path)
    unique_words = load_dictionary(unique_path)
    if not nwl_words:
        print("Dictionary not found!")
        return
        
    nwl_trie = WordTrie(nwl_words, min_len=4)
    unique_set = set(unique_words)
    
    start_time = time.time()
    best_u = 0
    best_board = None
    best_all_words = []
    
    print("Searching for HARD board: 100-200 words, 60%+ uniqueness ratio...")
    
    # Target 5 minutes max search
    for i in range(5000):
        # 1. Start with a semi-random board favoring some consonants for density
        board = [[random.choices(ALPHABET, weights=LETTER_FREQ_USER, k=1)[0] for _ in range(4)] for _ in range(4)]
        
        # 2. IO Step 1: Optimize for density first to get into 100-200 range
        for _ in range(1): # Small density pass
            for r, c in [(r,c) for r in range(4) for c in range(4)]:
                best_char = board[r][c]
                max_score = -1
                for char in ALPHABET:
                    board[r][c] = char
                    found = nwl_trie.solve(board)
                    score = len(found)
                    if 100 <= score <= 200:
                        u = [w for w in found if w in unique_set]
                        # Score is a combination to get us in range but push uniqueness
                        ratio = len(u) / score if score > 0 else 0
                        final_score = score + (ratio * 1000)
                        if final_score > max_score:
                            max_score = final_score
                            best_char = char
                board[r][c] = best_char
        
        # 3. IO Step 2: Optimize specifically for UNIQUENESS while staying in range
        for _ in range(2): # Two uniqueness passes
            pos = [(r, c) for r in range(4) for c in range(4)]
            random.shuffle(pos)
            for r, c in pos:
                best_char = board[r][c]
                max_u_score = -1
                # Try rare letters and common ones
                pool = UNLIKELY_POOL + ['A','E','I','O','U','S','T','N','R','L']
                for char in pool:
                    orig = board[r][c]
                    board[r][c] = char
                    found = nwl_trie.solve(board)
                    count = len(found)
                    if 100 <= count <= 200:
                        u = [w for w in found if w in unique_set]
                        u_pct = len(u) / count
                        if u_pct > max_u_score:
                            max_u_score = u_pct
                            best_char = char
                    else:
                        # If outside range, we only keep it if it's closer to range than before?
                        # No, let's keep it if it improves uniqueness BUT doesn't tank count too much
                        u = [w for w in found if w in unique_set]
                        u_pct = len(u) / count if count > 0 else 0
                        # Penalize being out of range
                        penalty = abs(count - 150) / 100 
                        score = u_pct - penalty
                        if score > max_u_score:
                            max_u_score = score
                            best_char = char
                board[r][c] = best_char
            
        all_found = nwl_trie.solve(board)
        word_count = len(all_found)
        unique_found = [w for w in all_found if w in unique_set]
        u_pct = (len(unique_found) / word_count * 100) if word_count > 0 else 0
        
        if 100 <= word_count <= 200:
            if u_pct > best_u:
                best_u = u_pct
                best_board = [row[:] for row in board]
                best_all_words = list(all_found)
                print(f"  Loop {i}: New Best! {word_count} words, {u_pct:.1f}% unique")
                if best_u >= 60.0: break

        if (time.time() - start_time) > 240: break

    if not best_board:
        print("Could not find a board in range. Returning last attempt.")
        return

    print("\n" + "="*40)
    print("FINAL BOARD")
    print("="*40)
    for row in best_board: print(" ".join(row))
    print("="*40)
    print(f"COUNT: {len(best_all_words)}")
    print(f"RATIO: {best_u:.1f}%")
    print("WORDS:")
    for w in sorted(best_all_words, key=lambda x: (-len(x), x)): print(w)

if __name__ == "__main__": main()
