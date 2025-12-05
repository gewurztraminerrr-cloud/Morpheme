import random
from typing import List, Set, Tuple, Dict, Optional
from collections import Counter
from enum import Enum
import string
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import copy

class OptimizeStrategy(Enum):
    """Strategy for optimizing IO cells."""
    TOTAL_DIFFICULT = "total_difficult"           # Maximize total difficult word count
    DISTINCT_DIFFICULT = "distinct_difficult"     # Maximize words appearing exactly once
    DIFFICULT_RATIO_6PLUS = "difficult_ratio_6plus"  # Maximize ratio of 6+ letter difficult words

def load_word_list(file_path: str) -> Set[str]:
    """Load words from file and convert to uppercase."""
    try:
        with open(file_path, 'r') as file:
            words = [line.strip().upper() for line in file]
            return set([w for w in words if len(w) >= 3])
    except Exception as e:
        print(f"Error loading word list: {e}")
        return set()

def generate_random_grid(width: int, height: int) -> List[List[str]]:
    """Generate a completely random grid of letters."""
    letter_dist = {
        'A': 343, 'B': 100, 'C': 157, 'D': 161, 'E': 455, 'F': 64, 'G': 106, 'H': 108, 'I': 326,
        'J': 11, 'K': 64, 'L': 236, 'M': 131, 'N': 232, 'O': 266, 'P': 123, 'Q': 8, 'R': 272,
        'S': 283, 'T': 224, 'U': 168, 'V': 40, 'W': 49,
         'X': 15, 'Y': 92, 'Z': 22
    }
    total = sum(letter_dist.values())
    letters = list(letter_dist.keys())
    weights = [v/total for v in letter_dist.values()]
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

def find_all_word_occurrences(grid: List[List[str]], word_set: Set[str], prefixes: Set[str]) -> Counter:
    """Find all words in the grid and count occurrences."""
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    word_counts = Counter()
    
    def dfs(i: int, j: int, current_word: str, path: Set[Tuple[int, int]]):
        if current_word and current_word not in prefixes and current_word not in word_set:
            return
        if len(current_word) >= 3 and current_word in word_set:
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

def find_words_through_cell(grid: List[List[str]], target_row: int, target_col: int, 
                            word_set: Set[str], prefixes: Set[str]) -> Counter:
    """
    Find all words that pass through a specific cell.
    Much faster than full grid search when only one cell changed.
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    word_counts = Counter()
    target = (target_row, target_col)
    
    def dfs(i: int, j: int, current_word: str, path: List[Tuple[int, int]], has_target: bool):
        """DFS that tracks whether path includes target cell."""
        if current_word and current_word not in prefixes and current_word not in word_set:
            return
        
        path_set = set(path)
        
        # Only count words that pass through target cell
        if len(current_word) >= 3 and current_word in word_set and has_target:
            word_counts[current_word] += 1
        
        if len(current_word) >= 16:
            return
            
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in path_set:
                    next_has_target = has_target or (ni, nj) == target
                    dfs(ni, nj, current_word + grid[ni][nj], path + [(ni, nj)], next_has_target)
    
    # Only start from cells that could reach the target (within path length)
    # For a 6x6 grid, max path is 16, so all cells could potentially reach target
    for i in range(height):
        for j in range(width):
            starts_at_target = (i, j) == target
            dfs(i, j, grid[i][j], [(i, j)], starts_at_target)
    
    return word_counts

def evaluate_cell_fast(grid: List[List[str]], r: int, c: int, 
                       base_counts: Counter, word_set: Set[str], 
                       difficult_set: Set[str], prefixes: Set[str],
                       scorer) -> Tuple[str, float]:
    """
    Evaluate all letters for a cell using differential updates.
    Instead of full grid search, only recompute words through the changed cell.
    """
    original_letter = grid[r][c]
    
    # Get words that pass through this cell with original letter
    original_cell_words = find_words_through_cell(grid, r, c, word_set, prefixes)
    
    best_letter = original_letter
    best_score = -1.0
    
    for letter in string.ascii_uppercase:
        grid[r][c] = letter
        
        if letter == original_letter:
            # Use cached counts
            score, _ = scorer(base_counts, difficult_set)
        else:
            # Get words with new letter
            new_cell_words = find_words_through_cell(grid, r, c, word_set, prefixes)
            
            # Compute new total: base - old_cell_words + new_cell_words
            new_counts = base_counts.copy()
            new_counts.subtract(original_cell_words)
            new_counts.update(new_cell_words)
            # Remove zero/negative counts
            new_counts = Counter({k: v for k, v in new_counts.items() if v > 0})
            
            score, _ = scorer(new_counts, difficult_set)
        
        if score > best_score:
            best_score = score
            best_letter = letter
    
    grid[r][c] = original_letter  # Restore
    return best_letter, best_score

# ============ SCORING FUNCTIONS ============

def score_total_difficult(word_counts: Counter, difficult_set: Set[str]) -> Tuple[float, Dict]:
    """Score: Total count of difficult words (including duplicates)."""
    total = sum(count for word, count in word_counts.items() if word in difficult_set)
    return float(total), {'total_difficult': total}

def score_distinct_difficult(word_counts: Counter, difficult_set: Set[str]) -> Tuple[float, Dict]:
    """Score: Count of difficult words appearing exactly once."""
    distinct = sum(1 for word, count in word_counts.items() if count == 1 and word in difficult_set)
    total = len([w for w in word_counts if w in difficult_set])
    return float(distinct), {'distinct_difficult': distinct, 'total_difficult': total}

def score_difficult_ratio_6plus(word_counts: Counter, difficult_set: Set[str]) -> Tuple[float, Dict]:
    """
    Score: Ratio of 6+ letter difficult words to total 6+ letter words.
    Penalizes low difficult word counts by multiplying ratio by count.
    This prevents achieving 100% ratio by reducing total words.
    """
    words_6plus = [w for w in word_counts if len(w) >= 6]
    difficult_6plus = [w for w in words_6plus if w in difficult_set]
    
    total_6plus = len(words_6plus)
    diff_6plus = len(difficult_6plus)
    ratio = diff_6plus / total_6plus if total_6plus > 0 else 0.0
    
    # Score combines ratio with difficult count to prevent reducing total words
    # A puzzle with 50 difficult / 100 total (50%) is better than 5/5 (100%)
    score = ratio * diff_6plus
    
    return score, {
        'difficult_6plus': diff_6plus,
        'total_6plus': total_6plus,
        'ratio': ratio
    }

def get_scorer(strategy: OptimizeStrategy):
    """Return the scoring function for a given strategy."""
    scorers = {
        OptimizeStrategy.TOTAL_DIFFICULT: score_total_difficult,
        OptimizeStrategy.DISTINCT_DIFFICULT: score_distinct_difficult,
        OptimizeStrategy.DIFFICULT_RATIO_6PLUS: score_difficult_ratio_6plus,
    }
    return scorers[strategy]

# ============ OPTIMIZATION ============

# Worker process globals - loaded once per process via initializer
_worker_word_set = None
_worker_difficult_set = None
_worker_prefixes = None

def _init_worker(word_set, difficult_set, prefixes):
    """Initialize worker process with shared data."""
    global _worker_word_set, _worker_difficult_set, _worker_prefixes
    _worker_word_set = word_set
    _worker_difficult_set = difficult_set
    _worker_prefixes = prefixes

def _evaluate_cell_letters(args):
    """Worker function to evaluate all letters for a single cell using fast differential updates."""
    grid_copy, r, c, strategy_value, base_counts = args
    
    # Use pre-loaded global data
    global _worker_word_set, _worker_difficult_set, _worker_prefixes
    
    # Recreate strategy enum from value
    strategy = OptimizeStrategy(strategy_value)
    scorer = get_scorer(strategy)
    
    original_letter = grid_copy[r][c]
    
    # Get words that pass through this cell with original letter
    original_cell_words = find_words_through_cell(grid_copy, r, c, _worker_word_set, _worker_prefixes)
    
    best_letter = original_letter
    best_score = -1.0
    
    for letter in string.ascii_uppercase:
        grid_copy[r][c] = letter
        
        if letter == original_letter:
            # Use base counts directly
            score, _ = scorer(base_counts, _worker_difficult_set)
        else:
            # Get words with new letter
            new_cell_words = find_words_through_cell(grid_copy, r, c, _worker_word_set, _worker_prefixes)
            
            # Compute new total: base - old_cell_words + new_cell_words
            new_counts = Counter(base_counts)
            new_counts.subtract(original_cell_words)
            new_counts.update(new_cell_words)
            # Remove zero/negative counts
            new_counts = Counter({k: v for k, v in new_counts.items() if v > 0})
            
            score, _ = scorer(new_counts, _worker_difficult_set)
        
        if score > best_score:
            best_score = score
            best_letter = letter
    
    return r, c, best_letter, best_score

def optimize_grid_iterative(
    grid: List[List[str]], 
    word_set: Set[str], 
    difficult_set: Set[str], 
    prefixes: Set[str], 
    strategy: OptimizeStrategy,
    invert_checkerboard: bool = False, 
    verbose: bool = True,
    parallel: bool = True,
    num_workers: int = None
) -> Tuple[List[List[str]], int, Dict]:
    """
    Optimize the grid using the checkerboard pattern.
    Returns (grid, improvements_count, final_stats).
    
    With parallel=True, uses multiple CPU cores to evaluate cells faster.
    """
    if verbose:
        print(f"  Optimization pass ({strategy.value}, invert={invert_checkerboard}, parallel={parallel})...")
    start = time.time()
    
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    scorer = get_scorer(strategy)

    # Identify IO cells (Checkerboard)
    optimizable_cells = [(i, j) for i in range(height) for j in range(width) 
                         if is_optimizable_cell(i, j) != invert_checkerboard]
    random.shuffle(optimizable_cells)
    
    improvements = 0
    
    # Get baseline
    baseline_counts = find_all_word_occurrences(grid, word_set, prefixes)
    baseline_score, baseline_stats = scorer(baseline_counts, difficult_set)
    
    if verbose:
        print(f"    Baseline: {baseline_stats}")
    
    if parallel and len(optimizable_cells) > 4:
        # Parallel optimization - process all cells at once
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 32)
        
        if verbose:
            print(f"    Using {num_workers} workers for {len(optimizable_cells)} cells (differential mode)...")
        
        # Prepare work items with base counts for differential updates
        work_items = []
        for r, c in optimizable_cells:
            grid_copy = [row[:] for row in grid]
            work_items.append((grid_copy, r, c, strategy.value, dict(baseline_counts)))
        
        # Process all cells in parallel - initializer loads word sets once per worker
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(word_set, difficult_set, prefixes)
        ) as executor:
            results = list(executor.map(_evaluate_cell_letters, work_items))
        
        # Apply all results to the grid
        for r, c, best_letter, best_score in results:
            if grid[r][c] != best_letter:
                grid[r][c] = best_letter
                improvements += 1
        
        if verbose:
            elapsed = time.time() - start
            cells_per_sec = len(optimizable_cells) / elapsed if elapsed > 0 else 0
            print(f"    Parallel: {len(optimizable_cells)} cells, {cells_per_sec:.1f} cells/sec")
    else:
        # Sequential fallback
        alphabet = string.ascii_uppercase
        
        for r, c in optimizable_cells:
            original_letter = grid[r][c]
            best_letter = original_letter
            best_score = -1.0
            
            for letter in alphabet:
                grid[r][c] = letter
                word_counts = find_all_word_occurrences(grid, word_set, prefixes)
                score, _ = scorer(word_counts, difficult_set)
                
                if score > best_score:
                    best_score = score
                    best_letter = letter
            
            grid[r][c] = best_letter
            if best_letter != original_letter:
                improvements += 1
    
    # Final stats
    final_counts = find_all_word_occurrences(grid, word_set, prefixes)
    final_score, final_stats = scorer(final_counts, difficult_set)
    
    if verbose:
        print(f"    Final: {final_stats} ({improvements} improvements, {time.time() - start:.1f}s)")
    
    return grid, improvements, final_stats

def optimize_until_target(
    grid: List[List[str]], 
    word_set: Set[str], 
    difficult_set: Set[str], 
    prefixes: Set[str],
    target_ratio: float = 0.80,
    buildup_iterations: int = 1,
    max_iterations: int = 20,
    verbose: bool = True
) -> List[List[str]]:
    """
    Two-phase adaptive optimization:
    Phase 1: Build up difficult word count using DISTINCT_DIFFICULT strategy
    Phase 2: Optimize ratio using DIFFICULT_RATIO_6PLUS until target reached
    """
    if verbose:
        print(f"Adaptive optimization (target ratio: {target_ratio*100:.0f}%)...")
    
    # Phase 1: Build up difficult word count
    if verbose:
        print(f"  Phase 1: Building difficult word count ({buildup_iterations} iterations)...")
    for i in range(buildup_iterations):
        grid, improvements, stats = optimize_grid_iterative(
            grid, word_set, difficult_set, prefixes,
            strategy=OptimizeStrategy.TOTAL_DIFFICULT,
            invert_checkerboard=(i % 2 == 1),
            verbose=verbose
        )
        if improvements == 0:
            if verbose:
                print(f"    No improvements at buildup iteration {i+1}, moving to Phase 2")
            break
    
    # Phase 2: Optimize ratio
    if verbose:
        print(f"  Phase 2: Optimizing 6+ letter ratio...")
    for i in range(max_iterations):
        grid, improvements, stats = optimize_grid_iterative(
            grid, word_set, difficult_set, prefixes,
            strategy=OptimizeStrategy.DIFFICULT_RATIO_6PLUS,
            invert_checkerboard=(i % 2 == 1),
            verbose=verbose
        )
        
        current_ratio = stats.get('ratio', 0.0)
        
        if current_ratio >= target_ratio:
            if verbose:
                print(f"  Target reached at iteration {i+1}: {current_ratio*100:.1f}%")
            break
            
        if improvements == 0:
            if verbose:
                print(f"  No improvements at iteration {i+1}, stopping early")
            break
    else:
        if verbose:
            print(f"  Reached max iterations ({max_iterations})")
    
    return grid

# ============ PUZZLE GENERATION ============

def generate_single_puzzle(
    width: int, 
    height: int, 
    word_list_path: str, 
    difficult_list_path: str, 
    seed: int = None, 
    strategy: OptimizeStrategy = OptimizeStrategy.DISTINCT_DIFFICULT,
    iterations: int = 1,
    adaptive: bool = False,
    target_ratio: float = 0.80,
    verbose: bool = True
) -> Tuple[List[List[str]], Set[str], Set[str], Set[str], int, Dict]:
    """Generate a word puzzle using the checkerboard optimization approach."""
    if verbose:
        print(f"Generating {width}x{height} puzzle (strategy: {strategy.value})...")
    
    if seed is not None:
        random.seed(seed)
    else:
        seed = random.randint(1, 100000)
        random.seed(seed)
    if verbose:
        print(f"  Seed: {seed}")
    
    # Load word lists (cached in globals)
    if 'word_set' not in globals() or globals()['word_set'] is None:
        globals()['word_set'] = load_word_list(word_list_path)
        if verbose:
            print(f"  Loaded {len(globals()['word_set'])} words")
    word_set = globals()['word_set']
        
    if 'difficult_set' not in globals() or globals()['difficult_set'] is None:
        globals()['difficult_set'] = load_word_list(difficult_list_path)
        if verbose:
            print(f"  Loaded {len(globals()['difficult_set'])} difficult words")
    difficult_set = globals()['difficult_set']
        
    if 'word_prefixes' not in globals() or globals()['word_prefixes'] is None:
        globals()['word_prefixes'] = build_prefix_set(word_set)
    prefixes = globals()['word_prefixes']
    
    # Generate random initial grid
    grid = generate_random_grid(width, height)
    
    # Apply optimization
    final_stats = {}
    if adaptive:
        grid = optimize_until_target(grid, word_set, difficult_set, prefixes, target_ratio, verbose=verbose)
    else:
        for i in range(iterations):
            grid, _, final_stats = optimize_grid_iterative(
                grid, word_set, difficult_set, prefixes,
                strategy=strategy,
                invert_checkerboard=(i % 2 == 1),
                verbose=verbose
            )
    
    # Find all words
    word_counts = find_all_word_occurrences(grid, word_set, prefixes)
    all_words_found = set(word_counts.keys())
    difficult_words_found = {w for w in all_words_found if w in difficult_set}
    long_words = {w for w in all_words_found if len(w) >= 7}
    
    if verbose:
        print(f"  Found {len(all_words_found)} total, {len(difficult_words_found)} difficult, {len(long_words)} long (7+)")
    
    return grid, all_words_found, difficult_words_found, long_words, seed, final_stats

def generate_puzzle(
    width: int,
    height: int,
    word_list_path: str,
    difficult_list_path: str,
    seed: Optional[int] = None,
    params: Optional[Dict] = None
) -> Dict:
    """Standard interface for puzzle generation."""
    if params is None:
        params = {}
    
    # Strategy selection
    strategy_str = params.get('strategy', 'distinct_difficult')
    try:
        strategy = OptimizeStrategy(strategy_str)
    except ValueError:
        strategy = OptimizeStrategy.DISTINCT_DIFFICULT
    
    iterations = params.get('iterations', 1)
    adaptive = params.get('adaptive', False)
    target_ratio = params.get('target_ratio', 0.80)
    verbose = params.get('verbose', False)
    
    grid, all_words, difficult_words, long_words, used_seed, stats = generate_single_puzzle(
        width, height, word_list_path, difficult_list_path,
        seed=seed,
        strategy=strategy,
        iterations=iterations,
        adaptive=adaptive,
        target_ratio=target_ratio,
        verbose=verbose
    )
    
    return {
        'grid': grid,
        'found_words': all_words,
        'difficult_words': difficult_words,
        'score': len(difficult_words) * 2 + len(long_words),
        'metadata': {
            'long_words_count': len(long_words),
            'strategy': strategy.value,
            **stats
        },
        'seed': used_seed
    }

if __name__ == "__main__":
    print("=== Testing DISTINCT_DIFFICULT strategy ===")
    result = generate_puzzle(
        width=6, height=6,
        word_list_path="generator/CSW22.txt",
        difficult_list_path="generator/difficult.txt",
        params={'strategy': 'distinct_difficult', 'iterations': 2, 'verbose': True}
    )
    for row in result['grid']:
        print(" ".join(row))
    print(f"Difficult: {len(result['difficult_words'])}")
    
    print("\n=== Testing ADAPTIVE with 80% target ratio ===")
    # Clear cache to get fresh results
    globals()['word_set'] = None
    globals()['difficult_set'] = None
    globals()['word_prefixes'] = None
    
    result = generate_puzzle(
        width=6, height=6,
        word_list_path="generator/CSW22.txt",
        difficult_list_path="generator/difficult.txt",
        params={'adaptive': True, 'target_ratio': 0.80, 'verbose': True}
    )
    for row in result['grid']:
        print(" ".join(row))
    print(f"Metadata: {result['metadata']}")
