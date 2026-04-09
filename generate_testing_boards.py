
import random
import os
import sys

# Add current directory to path to import local modules
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator

# Frequencies from prompt
LETTER_FREQ_USER = [114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
UNLIKELY_POOL = list("CMPHVFGWYBKJXQZ")

def load_dictionary(path):
    with open(path, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

class WordTrie:
    def __init__(self, words):
        self.trie = {}
        for word in words:
            node = self.trie
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['#'] = True

    def get_words(self, board):
        rows, cols = len(board), len(board[0])
        found = set()
        
        def solve(r, c, node, word, visited):
            if '#' in node and len(word) >= 4:
                found.add(word)
            
            if len(word) >= 12: return

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        char = board[nr][nc]
                        if char in node:
                            visited.add((nr, nc))
                            solve(nr, nc, node[char], word + char, visited)
                            visited.remove((nr, nc))

        for r in range(rows):
            for c in range(cols):
                char = board[r][c]
                if char in self.trie:
                    solve(r, c, self.trie[char], char, {(r, c)})
        return found

def generate_board():
    nwl_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/NWL.txt'
    unique_path = '/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/dictionaries/uniqueNWL.txt'
    
    print("Loading NWL...")
    nwl_words = load_dictionary(nwl_path)
    print("Loading uniqueNWL...")
    unique_words = load_dictionary(unique_path)
    
    unique_set = set(unique_words)
    bonus_pool = [w for w in nwl_words if len(w) == 4]
    
    print("Creating Tries...")
    unique_trie = WordTrie(unique_words)
    full_trie = WordTrie(nwl_words)
    
    board_count = 0
    best_u = 0
    best_u_board = None
    best_u_words = 0

    while True:
        board_count += 1
        bonus_word = random.choice(bonus_pool)
        board = [[None for _ in range(4)] for _ in range(4)]
        
        def embed(b, word):
            r, c = random.randint(0, 3), random.randint(0, 3)
            path = [(r, c)]
            for char in word[1:]:
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 4 and 0 <= nc < 4 and (nr, nc) not in path:
                            neighbors.append((nr, nc))
                if not neighbors: return None
                nr, nc = random.choice(neighbors)
                path.append((nr, nc))
                r, c = nr, nc
            for i, (pr, pc) in enumerate(path):
                b[pr][pc] = word[i]
            return path

        path = None
        while path is None:
            temp_board = [[None for _ in range(4)] for _ in range(4)]
            path = embed(temp_board, bonus_word)
        board = temp_board
        
        for r in range(4):
            for c in range(4):
                if board[r][c] is None and (r + c) % 2 == 0:
                    board[r][c] = random.choices(ALPHABET, weights=LETTER_FREQ_USER, k=1)[0]
        
        io_positions = [(r, c) for r in range(4) for c in range(4) if board[r][c] is None and (r + c) % 2 == 1]
        random.shuffle(io_positions)
        
        for r, c in io_positions:
            pool = UNLIKELY_POOL if random.random() < 0.5 else ALPHABET
            best_char = ALPHABET[0]
            max_unique = -1
            for char in pool:
                board[r][c] = char
                found_count = len(unique_trie.get_words(board))
                if found_count > max_unique:
                    max_unique = found_count
                    best_char = char
            board[r][c] = best_char

        for r in range(4):
            for c in range(4):
                if board[r][c] is None:
                    board[r][c] = random.choices(ALPHABET, weights=LETTER_FREQ_USER, k=1)[0]

        all_found = full_trie.get_words(board)
        unique_found = [w for w in all_found if w in unique_set]
        word_count = len(all_found)
        u_pct = (len(unique_found) / word_count * 100) if word_count > 0 else 0
        
        if board_count % 50 == 0:
            print(f"Board #{board_count}: Bonus={bonus_word}, Words={word_count}, Unique={u_pct:.1f}%")
        
        if u_pct > best_u and word_count >= 80: 
            best_u = u_pct
            best_u_board = [row[:] for row in board]
            best_u_words = word_count
            print(f"*** {board_count}: NEW BEST UNIQUENESS: {best_u:.1f}% ({best_u_words} words) ***")

        if u_pct >= 65 and 100 <= word_count <= 200:
            print(f"!!! SUCCESS: Found a board matching all criteria: {word_count} words, {u_pct:.1f}% unique !!!")
            for row in board:
                print(" ".join(row))
            print("-" * 20)
            break
            
        if board_count > 5000:
             print("Reached limit. Best found:")
             print(f"{best_u:.1f}% uniqueness with {best_u_words} words.")
             if best_u_board:
                 for row in best_u_board:
                     print(" ".join(row))
             break

if __name__ == "__main__":
    generate_board()
