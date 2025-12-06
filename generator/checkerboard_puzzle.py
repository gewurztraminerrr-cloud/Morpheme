import random
from typing import List, Set, Tuple, Dict, Optional
from collections import Counter
import string
import time

def load_word_list(file_path: str) -> Set[str]:
    """Load words from file and convert to uppercase."""
    try:
        with open(file_path, 'r') as file:
            words = file.readlines()
            words = [word.strip().upper() for word in words]
            return set(words)
    except Exception as e:
        print(f"Error loading word list: {e}")
        return set()

def generate_random_grid(width: int, height: int) -> List[List[str]]:
    """Generate a completely random grid of letters of the specified dimensions."""
    letter_dist = {
        'A': 800, 'B': 230, 'C': 360, 'D': 410, 'E': 1180, 'F': 150, 'G': 300, 'H': 240, 'I': 750,
        'J': 20, 'K': 140, 'L': 560, 'M': 280, 'N': 580, 'O': 610, 'P': 290, 'Q': 20, 'R': 730,
        'S': 940, 'T': 570, 'U': 370, 'V': 100, 'W': 120, 'X': 30, 'Y': 180, 'Z': 40
    }

    total = sum(letter_dist.values())
    letter_weights = {k: v/total for k, v in letter_dist.items()}
    
    letters = list(letter_weights.keys())
    weights = list(letter_weights.values())
    
    return [[random.choices(letters, weights=weights)[0] for _ in range(width)] for _ in range(height)]

def is_optimizable_cell(i: int, j: int) -> bool:
    """Determine if a cell should be optimized based on checkerboard pattern."""
    return (i + j) % 2 == 0

def build_prefix_set(word_set: Set[str]) -> Set[str]:
    """Build a set of all prefixes from the words."""
    prefixes = set()
    for word in word_set:
        for i in range(1, len(word)):
            prefixes.add(word[:i])
    return prefixes

def find_all_word_occurrences(grid: List[List[str]], word_set: Set[str], prefixes: Set[str], min_length: int = 3) -> Counter:
    """
    Find all words in the grid and count how many times each word can be formed.
    Returns a Counter of word -> occurrence count.
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    word_counts = Counter()
    
    def dfs(i: int, j: int, current_word: str, path: Set[Tuple[int, int]]):
        if current_word and current_word not in prefixes and current_word not in word_set:
            return
            
        if len(current_word) >= min_length and current_word in word_set:
            word_counts[current_word] += 1
            
        if len(current_word) >= 16:
            return
            
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in path:
                    dfs(ni, nj, current_word + grid[ni][nj], path | {(ni, nj)})
    
    for i in range(height):
        for j in range(width):
            dfs(i, j, grid[i][j], {(i, j)})
            
    return word_counts

def count_distinct_difficult(word_counts: Counter, difficult_set: Set[str]) -> int:
    """Count how many difficult words appear exactly once in the puzzle."""
    return sum(1 for word, count in word_counts.items() if count == 1 and word in difficult_set)

def optimize_grid_iterative(grid: List[List[str]], word_set: Set[str], difficult_set: Set[str], prefixes: Set[str], invert_checkerboard: bool = False, verbose: bool = True) -> List[List[str]]:
    """
    Optimize the grid using the checkerboard pattern.
    For each IO cell, try all letters and keep the one that maximizes distinct difficult words.
    """
    if verbose:
        print("Starting grid optimization (Distinct Difficult Word Tracking)...")
    start = time.time()
    
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    # Identify IO cells (Checkerboard)
    optimizable_cells = [(i, j) for i in range(height) for j in range(width) if is_optimizable_cell(i, j) == invert_checkerboard]
    random.shuffle(optimizable_cells)
    
    alphabet = string.ascii_uppercase
    
    # Get baseline score
    baseline_counts = find_all_word_occurrences(grid, word_set, prefixes)
    baseline_distinct = count_distinct_difficult(baseline_counts, difficult_set)
    baseline_total = len([w for w in baseline_counts if w in difficult_set])
    
    if verbose:
        print(f"  Baseline: {baseline_distinct} distinct difficult words, {baseline_total} total difficult")
    
    improvements = 0
    
    for idx, (r, c) in enumerate(optimizable_cells):
        original_letter = grid[r][c]
        best_letter = original_letter
        best_distinct = -1
        best_total = -1
        
        for letter in alphabet:
            grid[r][c] = letter
            
            # Solve the entire puzzle with this letter
            word_counts = find_all_word_occurrences(grid, word_set, prefixes)
            distinct_count = count_distinct_difficult(word_counts, difficult_set)
            total_diff = len([w for w in word_counts if w in difficult_set])
            
            # Primary: maximize distinct difficult words
            # Secondary: maximize total difficult words (tiebreaker)
            if distinct_count > best_distinct or (distinct_count == best_distinct and total_diff > best_total):
                best_distinct = distinct_count
                best_total = total_diff
                best_letter = letter
        
        grid[r][c] = best_letter
        
        if best_letter != original_letter:
            improvements += 1
            
        if verbose and (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(optimizable_cells)} cells ({improvements} improvements)")
    
    # Final stats
    final_counts = find_all_word_occurrences(grid, word_set, prefixes)
    final_distinct = count_distinct_difficult(final_counts, difficult_set)
    final_total = len([w for w in final_counts if w in difficult_set])
    
    if verbose:
        end = time.time()
        print(f"Grid optimization completed in {end - start:.2f} seconds")
        print(f"  Final: {final_distinct} distinct difficult words, {final_total} total difficult")
        print(f"  Improvement: +{final_distinct - baseline_distinct} distinct, +{final_total - baseline_total} total")
    
    return grid

def find_words_in_grid(grid: List[List[str]], word_set: Set[str], prefixes: Set[str] = None, min_length: int = 3) -> Set[str]:
    """Find all valid words in the grid (unique set)."""
    word_counts = find_all_word_occurrences(grid, word_set, prefixes or build_prefix_set(word_set), min_length)
    return set(word_counts.keys())

def track_letter_usage(grid: List[List[str]]) -> Dict[str, int]:
    """Count frequency of each letter in the current grid."""
    usage = {letter: 0 for letter in string.ascii_uppercase}
    for row in grid:
        for letter in row:
            usage[letter] += 1
    return usage

def generate_single_puzzle(width: int, height: int, word_list_path: str, difficult_list_path: str, seed: int = None, iterations: int = 1, verbose: bool = True) -> Tuple[List[List[str]], Set[str], Set[str], Set[str], int]:
    """Generate a word puzzle using the checkerboard optimization approach."""
    if verbose:
        print(f"Generating {width}x{height} puzzle...")
    if seed is not None:
        random.seed(seed)
        if verbose:
            print(f"Using seed: {seed}")
    else:
        seed = random.randint(1, 100000)
        random.seed(seed)
    
    # Load word lists
    if 'word_set' not in globals() or globals()['word_set'] is None:
        start = time.time()
        globals()['word_set'] = load_word_list(word_list_path)
        if verbose:
            print(f"Loaded {len(globals()['word_set'])} total words in {time.time() - start:.2f} seconds")
    word_set = globals()['word_set']
        
    if 'difficult_set' not in globals() or globals()['difficult_set'] is None:
        start = time.time()
        globals()['difficult_set'] = load_word_list(difficult_list_path)
        if verbose:
            print(f"Loaded {len(globals()['difficult_set'])} difficult words in {time.time() - start:.2f} seconds")
    difficult_set = globals()['difficult_set']
        
    if 'word_prefixes' not in globals() or globals()['word_prefixes'] is None:
        start = time.time()
        globals()['word_prefixes'] = build_prefix_set(word_set)
        if verbose:
            print(f"Built prefix set in {time.time() - start:.2f} seconds")
    prefixes = globals()['word_prefixes']
    
    # Generate random initial grid
    if verbose:
        start = time.time()
    grid = generate_random_grid(width, height)
    if verbose:
        print(f"Generated random grid in {time.time() - start:.2f} seconds")
    
    # Apply optimization, iterating with alternating checkerboard patterns
    for i in range(iterations):
        grid = optimize_grid_iterative(grid, word_set, difficult_set, prefixes, i % 2 == 0, verbose)

    # Find all words in the grid
    if verbose:
        start = time.time()
        print("Finding all words in grid...")
    all_words_found = find_words_in_grid(grid, word_set, prefixes)
    if verbose:
        print(f"Found {len(all_words_found)} total words in {time.time() - start:.2f} seconds")
        
    difficult_words_found = {w for w in all_words_found if w in difficult_set}
    long_words = {word for word in all_words_found if len(word) >= 7}
    
    return grid, all_words_found, difficult_words_found, long_words, seed

def generate_best_puzzle(width: int, height: int, word_list_path: str, difficult_list_path: str, iterations: int = 1) -> Tuple[List[List[str]], Set[str], Set[str], Set[str], int, Dict]:
    """Generate multiple puzzles and return the best one based on difficult word count."""
    print(f"Generating {iterations} puzzle candidates of size {width}x{height} to find the best one...")
    
    candidates = []
    best_score = -1
    best_index = -1
    
    for i in range(iterations):
        seed = random.randint(1, 100000)
        grid, all_words, difficult_words, long_words, _ = generate_single_puzzle(
            width=width,
            height=height,
            word_list_path=word_list_path,
            difficult_list_path=difficult_list_path,
            iterations=iterations,
            seed=seed,
            verbose=False
        )
        
        score = len(difficult_words) * 2 + len(long_words)
        
        candidates.append({
            'seed': seed,
            'grid': grid,
            'all_words': all_words,
            'difficult_words': difficult_words,
            'long_words': long_words,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_index = i
    
    best = candidates[best_index]
    print(f"\nSelected best puzzle (candidate {best_index+1}, seed: {best['seed']}):")
    print(f"  - {len(best['difficult_words'])} difficult words")
    print(f"  - {len(best['long_words'])} words with 7+ letters") 
    print(f"  - Score: {best['score']}")
    
    comparison = {}
    for i, candidate in enumerate(candidates):
        if i != best_index:
            comparison[f"candidate_{i+1}"] = {
                'seed': candidate['seed'],
                'score_diff': best['score'] - candidate['score']
            }
    
    return best['grid'], best['all_words'], best['difficult_words'], best['long_words'], best['seed'], comparison

def generate_puzzle(
    width: int,
    height: int,
    word_list_path: str,
    difficult_list_path: str,
    seed: Optional[int] = None,
    params: Optional[Dict] = None
) -> Dict:
    """
    Standard interface for puzzle generation.
    """
    if params is None:
        params = {}
        
    iterations = params.get('iterations', 1)
    verbose = params.get('verbose', False)
    
    if iterations > 1:
        grid, all_words, difficult_words, long_words, used_seed, comparison = generate_best_puzzle(
            width, height, word_list_path, difficult_list_path, iterations
        )
        stats = {
            'comparison': comparison,
            'long_words_count': len(long_words)
        }
    else:
        grid, all_words, difficult_words, long_words, used_seed = generate_single_puzzle(
            width, height, word_list_path, difficult_list_path, seed, verbose
        )
        stats = {
            'long_words_count': len(long_words)
        }
        
    return {
        'grid': grid,
        'found_words': all_words,
        'difficult_words': difficult_words,
        'score': len(difficult_words) * 2 + len(long_words),
        'metadata': stats,
        'seed': used_seed
    }

if __name__ == "__main__":
    result = generate_puzzle(
        width=6,
        height=6,
        word_list_path="generator/TWL.txt",
        difficult_list_path="generator/randomTWLunique.txt",
        params={'iterations': 1, 'verbose': True}
    )
    print("\nFinal Grid:")
    for row in result['grid']:
        print(" ".join(row))
    print(f"\nDifficult words found: {len(result['difficult_words'])}")