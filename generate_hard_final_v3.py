
import random
import sys
import time
import os

LETTER_FREQ_USER = [114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def load_dictionary(path):
    if not os.path.exists(path): return []
    with open(path, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

class WordTrie:
    def __init__(self, words, min_len=4):
        self.trie = {}
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

def embed(b, word):
    r, c = random.randint(0, 3) , random.randint(0, 3)
    path = [(r, c)]
    for char in word[1:]:
        neighbors = [(r+dr, c+dc) for dr in [-1,0,1] for dc in [-1,0,1] if (dr!=0 or dc!=0) and 0<=r+dr<4 and 0<=c+dc<4 and (r+dr,c+dc) not in path]
        if not neighbors: return None
        nr, nc = random.choice(neighbors)
        path.append((nr, nc))
        r, c = nr, nc
    for i, (pr, pc) in enumerate(path): b[pr][pc] = word[i]
    return path

def main():
    nwl_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/NWL.txt'
    unique_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/uniqueNWL.txt'
    nwl_words = load_dictionary(nwl_path)
    unique_words = load_dictionary(unique_path)
    nwl_trie = WordTrie(nwl_words, min_len=4)
    unique_set = set(unique_words)
    bonus_pool = [w for w in nwl_words if 9 <= len(w) <= 12]
    
    start_time = time.time()
    best_u = 0
    best_board_data = None

    print("Generating board seeking 60%+ uniqueness for 100-200 words...")
    
    for attempt in range(500):
        bonus_word = random.choice(bonus_pool)
        board = [[None for _ in range(4)] for _ in range(4)]
        path = embed(board, bonus_word)
        if not path: continue
        
        # Fill rest with rare letters mostly, then common
        for r in range(4):
            for c in range(4):
                if board[r][c] is None:
                    if random.random() < 0.4:
                        board[r][c] = random.choice("QZJXV")
                    else:
                        board[r][c] = random.choices(ALPHABET, weights=LETTER_FREQ_USER, k=1)[0]
        
        # IO Pass
        for pass_num in range(2):
            tiles = [(r,c) for r in range(4) for c in range(4) if (r,c) not in path]
            random.shuffle(tiles)
            for r, c in tiles:
                best_char = board[r][c]
                max_score = -1e9
                test_pool = list("ETAOINSR") + list("QZJXVWKBC")
                random.shuffle(test_pool)
                for char in test_pool[:12]:
                    board[r][c] = char
                    found = nwl_trie.solve(board)
                    count = len(found)
                    u = sum(1 for w in found if w in unique_set)
                    u_pct = u / count if count > 0 else 0
                    
                    # Target: 100-200 words, high uniqueness
                    score = u_pct * 100.0
                    if count < 100: score -= (100 - count)
                    if count > 200: score -= (count - 200)
                    
                    if score > max_score:
                        max_score = score
                        best_char = char
                board[r][c] = best_char
        
        found = nwl_trie.solve(board)
        count = len(found)
        u_pct = sum(1 for w in found if w in unique_set) / count * 100 if count > 0 else 0
        if 100 <= count <= 200:
            if u_pct > best_u:
                best_u = u_pct
                best_board_data = (board, bonus_word, count, u_pct, list(found))
                print(f"  Attempt {attempt}: {count} words, {u_pct:.1f}% unique (Bonus: {bonus_word})")
                if best_u >= 60.0: break
        
        if (time.time() - start_time) > 120: break

    if best_board_data:
        board, bonus_word, count, ratio, words = best_board_data
        print("\nFINAL BOARD:")
        for row in board: print(" ".join(row))
        print(f"BONUS: {bonus_word}")
        print(f"COUNT: {count}")
        print(f"RATIO: {ratio:.1f}%")
        print("WORDS (Top 20):")
        for w in sorted(words, key=lambda x: (-len(x), x))[:20]: print(f"  {w}")

if __name__ == "__main__": main()
