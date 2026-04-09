
import random
import sys
import os
from recreate_boggle_difficult import load_dictionary, WordTrie

twl = load_dictionary('/Users/jeffbabiak/Desktop/TWL.txt')
uniq = set(load_dictionary('/Users/jeffbabiak/Desktop/randomTWLunique.txt'))
trie = WordTrie(twl, min_len=4)
pool = list('CMPHVFGWYBKJXQZ')
alph = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
freq = [114, 37, 54, 49, 127, 24, 34, 35, 102, 5, 23, 77, 46, 69, 86, 44, 3, 81, 90, 62, 62, 13, 17, 7, 38, 8]

best_ratio = 0
best_board = None

for _ in range(5000):
    b = [[None]*4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            if (r + c) % 2 == 0: 
                b[r][c] = random.choices(alph, weights=freq, k=1)[0]
            else: 
                b[r][c] = random.choice(pool)
    
    words = trie.solve(b)
    if len(words) >= 70:
        u = [w for w in words if w in uniq]
        ratio = len(u) / len(words) if words else 0
        if ratio > best_ratio:
            best_ratio = ratio
            best_board = [row[:] for row in b]
            best_words = list(words)
            if ratio >= 0.5 and len(words) >= 80: break

if best_board:
    print(f"Board found with {len(best_words)} words, Uniqueness: {best_ratio:.1%}")
    for row in best_board: print(" ".join(row))
    print("WORDS:")
    for w in sorted(best_words, key=lambda x: (-len(x), x)): print(w)
