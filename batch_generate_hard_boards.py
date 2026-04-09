
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

def find_board(target_words=100):
    best_b = None
    best_w = []
    best_r = 0
    
    for _ in range(2000): # Iterations per board search
        b = [[None]*4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                if (r + c) % 2 == 0: b[r][c] = random.choices(alph, weights=freq, k=1)[0]
                else: b[r][c] = random.choice(pool) if random.random() < 0.6 else random.choices(alph, weights=freq, k=1)[0]
        
        words = trie.solve(b)
        count = len(words)
        if count >= target_words:
            u = [w for w in words if w in uniq]
            ratio = len(u) / count if count > 0 else 0
            # Higher is better, but anything above 100 works for difficult categorization
            if ratio > best_r:
                best_r = ratio
                best_b = [row[:] for row in b]
                best_w = list(words)
            if ratio >= 0.2: break # Early Exit if decent uniqueness found
            
    return best_b, best_w, best_r

for i in range(1, 4): # Find 3 boards
    b, w, r = find_board(100)
    print(f"BOARD {i}: {len(w)} words, Uniqueness: {r:.1%}")
    for row in b: print(" ".join(row))
    # Print top 15 words
    sorted_w = sorted(w, key=lambda x: (-len(x), x))
    print(f"Top Words: {', '.join(sorted_w[:15])}")
    print("-" * 20)
