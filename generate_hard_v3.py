
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
    
    for i in range(10000):
        # Start board
        board = [[random.choices(ALPHABET, weights=LETTER_FREQ_USER, k=1)[0] for _ in range(4)] for _ in range(4)]
        
        # IO Optimization
        for pass_num in range(3): # More passes
            pos = [(r, c) for r in range(4) for c in range(4)]
            random.shuffle(pos)
            for r, c in pos:
                best_char = board[r][c]
                max_u_score = -100
                
                # Check current
                # Try all letters
                for char in ALPHABET:
                    board[r][c] = char
                    found = nwl_trie.solve(board)
                    count = len(found)
                    u = [w for w in found if w in unique_set]
                    u_pct = len(u) / count if count > 0 else 0
                    
                    # Score: Uniqueness + Range Bonus
                    # Range: 100-200. Ideal: 150.
                    range_score = 0
                    if 100 <= count <= 200:
                        range_score = 2.0 # Huge bonus for being in range
                    else:
                        # Dist-based penalty
                        range_score = -abs(count - 150) / 100.0
                    
                    final_score = u_pct + range_score
                    if final_score > max_u_score:
                        max_u_score = final_score
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

        if (time.time() - start_time) > 400: break

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
