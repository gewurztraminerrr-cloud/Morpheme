
import random
import sys
import time

# Frequencies
LETTER_FREQ_USER = [114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
UNLIKELY_POOL = list("CMPHVFGWYBKJXQZ")

def load_dictionary(path):
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
            if len(word) >= 12: return
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
    nwl_trie = WordTrie(nwl_words, min_len=4)
    unique_set = set(unique_words)
    
    start_time = time.time()
    best_u = 0
    best_board = None
    best_all_words = []
    
    for _ in range(3000):
        # Full board random base
        board = [[random.choices(ALPHABET, weights=LETTER_FREQ_USER, k=1)[0] for _ in range(4)] for _ in range(4)]
        
        # IO Pattern for high ratio
        # We find positions and pick letters that MAXIMIZE (unique_count^2 / total_count)
        # This favors unique words heavily while keeping total words reasonable.
        for r, c in [(r,c) for r in range(4) for c in range(4)]:
            best_char = board[r][c]
            max_score = -1
            
            # Using unlikely pool for IO candidates
            pool = UNLIKELY_POOL if random.random() < 0.8 else ALPHABET
            for char in pool:
                board[r][c] = char
                found = nwl_trie.solve(board)
                if not found: continue
                u = [w for w in found if w in unique_set]
                score = (len(u)**2) / len(found)
                if score > max_score:
                    max_score = score
                    best_char = char
            board[r][c] = best_char
            
        all_found = nwl_trie.solve(board)
        unique_found = [w for w in all_found if w in unique_set]
        word_count = len(all_found)
        u_pct = (len(unique_found) / word_count * 100) if word_count > 0 else 0
        
        if 100 <= word_count <= 200:
            if u_pct > best_u:
                best_u = u_pct
                best_board = [row[:] for row in board]
                best_all_words = list(all_found)
                if best_u >= 60.0: break

        if (time.time() - start_time) > 120: break

    print(f"COUNT: {len(best_all_words)}")
    print(f"RATIO: {best_u:.1%}")
    print("BOARD:")
    for row in best_board: print(" ".join(row))
    print("WORDS:")
    for w in sorted(best_all_words, key=lambda x: (-len(x), x)): print(w)

if __name__ == "__main__": main()
