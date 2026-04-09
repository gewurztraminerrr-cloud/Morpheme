
import time
import random
import sys
import os

# Frequencies from screenshot
LETTER_FREQ_USER = [114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
UNLIKELY_POOL = list("CMPHVFGWYBKJXQZ")

def load_dictionary(path):
    if not os.path.exists(path): return []
    with open(path, 'r') as f:
        return [line.upper().strip() for line in f if line.strip()]

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
    twl_path = '/Users/jeffbabiak/Desktop/TWL.txt'
    unique_path = '/Users/jeffbabiak/Desktop/randomTWLunique.txt'
    
    twl_words = load_dictionary(twl_path)
    unique_words = load_dictionary(unique_path)
    twl_trie = WordTrie(twl_words, min_len=4)
    unique_set = set(unique_words)
    bonus_pool = [w for w in twl_words if 6 <= len(w) <= 9]
    
    start_time = time.time()
    board_count = 0
    best_u = 0
    
    while True:
        board_count += 1
        bonus_word = random.choice(bonus_pool)
        
        # IO Pattern: r+c % 2
        def embed(word):
            path = [(random.randint(0, 3), random.randint(0, 3))]
            for char in word[1:]:
                r, c = path[-1]
                neighbors = [(r+dr, c+dc) for dr in [-1,0,1] for dc in [-1,0,1] if (dr!=0 or dc!=0) and 0<=r+dr<4 and 0<=c+dc<4 and (r+dr,c+dc) not in path]
                if not neighbors: return None
                path.append(random.choice(neighbors))
            b = [[None for _ in range(4)] for _ in range(4)]
            for i, (pr, pc) in enumerate(path): b[pr][pc] = word[i]
            return b
        
        board = None
        while board is None: board = embed(bonus_word)
        
        # Base cells: (r+c)%2 == 0
        for r in range(4):
            for c in range(4):
                if board[r][c] is None and (r+c)%2 == 0:
                    board[r][c] = random.choices(ALPHABET, weights=LETTER_FREQ_USER, k=1)[0]
                    
        # IO cells: (r+c)%2 == 1 (Optimize for RATIO)
        io_positions = [(r,c) for r in range(4) for c in range(4) if board[r][c] is None and (r+c)%2 == 1]
        random.shuffle(io_positions)
        for r, c in io_positions:
            # Randomly alternate target: Ratio vs Unique Count
            target = "Ratio" if random.random() < 0.7 else "Unique"
            best_char = ALPHABET[0]
            max_val = -1
            
            pool = UNLIKELY_POOL if random.random() < 0.4 else ALPHABET
            for char in pool:
                board[r][c] = char
                found = twl_trie.solve(board)
                unique_found = [w for w in found if w in unique_set]
                count_all = len(found)
                count_u = len(unique_found)
                
                if target == "Ratio":
                    # Ratio biased by minimum word count to keep board alive
                    val = (count_u / count_all) if count_all >= 10 else 0
                else:
                    val = count_u
                
                if val > max_val:
                    max_val = val
                    best_char = char
            board[r][c] = best_char

        # Final evaluation
        all_words = twl_trie.solve(board)
        unique_matches = [w for w in all_words if w in unique_set]
        word_count = len(all_words)
        u_ratio = (len(unique_matches) / word_count) if word_count > 0 else 0
        
        # Target from screenshot: 100+ words, 60%+ uniqueness
        # Since 60% is "Very Difficult", we'll aim for 35% first to show a fresh board
        if word_count >= 100 and u_ratio >= 0.35: 
            duration = time.time() - start_time
            print(f"CREATED NEW BOARD: Bonus={bonus_word}, Count={word_count}, Unique={u_ratio:.1%}, Time={duration:.2f}s")
            for row in board: print(" ".join(row))
            print("WORDS:")
            sorted_words = sorted(all_words, key=lambda x: (len(x), x), reverse=True)
            for w in sorted_words: print(w)
            break
            
        if u_ratio > best_u and word_count >= 100:
             best_u = u_ratio
             best_board = [row[:] for row in board]
             best_words = list(all_words)
             best_bonus = bonus_word
             
        if board_count % 100 == 0:
             print(f"Board #{board_count}: best unique for >=100 words: {best_u:.1%}")
             
        if (time.time() - start_time) > 240: 
             print("Reached 4 min limit. Returning BEST matched board.")
             if best_u > 0:
                  print(f"BEST FOUND: Bonus={best_bonus}, Count={len(best_words)}, Unique={best_u:.1%}")
                  for row in best_board: print(" ".join(row))
                  print("WORDS:")
                  for w in sorted(best_words, key=lambda x: (-len(x), x)): print(w)
             break

if __name__ == "__main__": main()

if __name__ == "__main__": main()
