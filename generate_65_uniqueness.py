
import time
import random
import sys

# Frequency and pools
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
    
    # Pre-load lists and Tries for 4LM
    nwl_words = load_dictionary(nwl_path)
    unique_words = load_dictionary(unique_path)
    nwl_trie = WordTrie(nwl_words, min_len=4)
    unique_trie = WordTrie(unique_words, min_len=4)
    unique_set = set(unique_words)
    bonus_pool = [w for w in nwl_words if len(w) == 5]
    
    start_time = time.time()
    board_count = 0
    best_ratio = 0
    
    while True:
        board_count += 1
        bonus_word = random.choice(bonus_pool)
        board = [[None for _ in range(4)] for _ in range(4)]
        
        # 1. Embed Bonus
        def embed(b, word):
            r, c = random.randint(0, 3), random.randint(0, 3)
            path = [(r, c)]
            for char in word[1:]:
                neighbors = [(r+dr, c+dc) for dr in [-1,0,1] for dc in [-1,0,1] if (dr!=0 or dc!=0) and 0<=r+dr<4 and 0<=c+dc<4 and (r+dr,c+dc) not in path]
                if not neighbors: return None
                nr, nc = random.choice(neighbors)
                path.append((nr, nc))
                r, c = nr, nc
            for i, (pr, pc) in enumerate(path): b[pr][pc] = word[i]
            return path

        path = None
        while path is None: path = embed([[None for _ in range(4)] for _ in range(4)], bonus_word)
        # Re-fill a clean board with the successful path
        board = [[None for _ in range(4)] for _ in range(4)]
        for i, (pr, pc) in enumerate(path): board[pr][pc] = bonus_word[i]
        
        # 2. Base filling (Checkerboard)
        for r in range(4):
            for c in range(4):
                if board[r][c] is None and (r+c)%2 == 0:
                    board[r][c] = random.choices(ALPHABET, weights=LETTER_FREQ_USER, k=1)[0]
        
        # 3. IO Filling (Maximize Count of Unique words)
        io_positions = [(r,c) for r in range(4) for c in range(4) if board[r][c] is None and (r+c)%2 == 1]
        random.shuffle(io_positions)
        
        for r, c in io_positions:
            pool = UNLIKELY_POOL if random.random() < 0.5 else ALPHABET
            best_char = pool[0]
            max_unique = -1
            for char in pool:
                board[r][c] = char
                unique_set_found = unique_trie.solve(board)
                if len(unique_set_found) > max_unique:
                    max_unique = len(unique_set_found)
                    best_char = char
            board[r][c] = best_char
            
        # Final evaluation
        all_words = nwl_trie.solve(board)
        unique_found = [w for w in all_words if w in unique_set]
        word_count = len(all_words)
        unique_ratio = len(unique_found) / word_count if word_count > 0 else 0
        
        # Log 200 words board if hit
        if word_count == 200:
             print(f"!!! FOUND 200 WORD BOARD !!! Board #{board_count}")
             for row in board: print(" ".join(row))
        
        # Success criteria check
        if 100 <= word_count <= 200 and unique_ratio >= 0.65:
             duration = time.time() - start_time
             print(f"SUCCESS: Found 65%+ Uniqueness Board at Board #{board_count} in {duration:.2f}s")
             print(f"Bonus: {bonus_word}, Words: {word_count}, Unique: {unique_ratio:.1%}")
             for row in board: print(" ".join(row))
             print("WORDS:")
             sorted_words = sorted(all_words, key=lambda x: (len(x), x), reverse=True)
             for w in sorted_words: print(w)
             break

        if board_count % 500 == 0:
             print(f"Board #{board_count}: best unique so far: {best_ratio:.1%} (Current: {word_count} words, {unique_ratio:.1%})")
        
        if unique_ratio > best_ratio and 50 <= word_count <= 150:
             best_ratio = unique_ratio
             print(f"*** {board_count}: NEW BEST UNIQUENESS: {best_ratio:.1%} ({word_count} words) ***")
             for row in board: print(" ".join(row))
             print("-" * 20)

        if (time.time() - start_time) > 180: # 3 minute limit
             print("Time limit reached.")
             break

if __name__ == "__main__": main()
