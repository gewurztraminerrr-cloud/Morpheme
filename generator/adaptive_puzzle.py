"""
Adaptive Density Generator with Constraint Satisfaction
Combines strengths of both checkerboard and word-placement approaches.
"""

import random
import string
import math
from typing import List, Set, Dict, Tuple, Optional


# Letter frequency distribution (per 10,000 letters in English)
LETTER_FREQUENCIES = {
    'A': 800, 'B': 230, 'C': 360, 'D': 410, 'E': 1180, 'F': 150, 'G': 300, 'H': 240, 'I': 750,
    'J': 20, 'K': 140, 'L': 560, 'M': 280, 'N': 580, 'O': 610, 'P': 290, 'Q': 20, 'R': 730,
    'S': 940, 'T': 570, 'U': 370, 'V': 100, 'W': 120, 'X': 30, 'Y': 180, 'Z': 40
}

TOTAL_FREQ = sum(LETTER_FREQUENCIES.values())
NORMALIZED_FREQ = {k: v / TOTAL_FREQ for k, v in LETTER_FREQUENCIES.items()}

# Rare letters that should get bonuses (frequency < 100 per 10,000)
RARE_LETTERS = {'Q', 'J', 'X', 'Z', 'V', 'K', 'W', 'F'}
# Very rare letters get extra bonus
VERY_RARE_LETTERS = {'Q', 'J', 'X', 'Z'}


def load_word_list(file_path: str) -> Set[str]:
    """Load words from file and convert to uppercase."""
    try:
        with open(file_path, 'r') as file:
            words = [word.strip().upper() for word in file.readlines()]
            return {w for w in words if 3 <= len(w) <= 16}
    except Exception as e:
        print(f"Error loading word list: {e}")
        return set()


def build_prefix_set(word_set: Set[str]) -> Set[str]:
    """Build a set of all prefixes from the words."""
    prefixes = set()
    for word in word_set:
        for i in range(1, len(word)):
            prefixes.add(word[:i])
    return prefixes


def create_radial_density_map(width: int, height: int, target_difficulty: float) -> List[List[float]]:
    """Create a density map with difficulty concentrated in center."""
    density = [[0.0 for _ in range(width)] for _ in range(height)]
    center_i, center_j = height / 2 - 0.5, width / 2 - 0.5
    max_dist = math.sqrt(center_i**2 + center_j**2)
    
    for i in range(height):
        for j in range(width):
            dist = math.sqrt((i - center_i)**2 + (j - center_j)**2)
            normalized = 1.0 - (dist / max_dist)
            density[i][j] = normalized * target_difficulty
    
    return density


def create_gradient_density_map(width: int, height: int, target_difficulty: float, 
                                direction: str = 'horizontal') -> List[List[float]]:
    """Create a gradient density map (easy to hard)."""
    density = [[0.0 for _ in range(width)] for _ in range(height)]
    
    if direction == 'horizontal':
        for j in range(width):
            value = (j / (width - 1)) * target_difficulty
            for i in range(height):
                density[i][j] = value
    else:  # vertical
        for i in range(height):
            value = (i / (height - 1)) * target_difficulty
            for j in range(width):
                density[i][j] = value
    
    return density


def create_uniform_density_map(width: int, height: int, target_difficulty: float) -> List[List[float]]:
    """Create a uniform density map."""
    return [[target_difficulty for _ in range(width)] for _ in range(height)]


def analyze_difficult_letter_frequencies(difficult_set: Set[str]) -> Dict[str, float]:
    """Analyze which letters appear most frequently in difficult words."""
    letter_counts = {letter: 0 for letter in string.ascii_uppercase}
    total = 0
    
    for word in difficult_set:
        for letter in word:
            letter_counts[letter] += 1
            total += 1
    
    if total == 0:
        return NORMALIZED_FREQ.copy()
    
    return {k: v / total for k, v in letter_counts.items()}


def count_potential_words(grid: List[List[str]], pos: Tuple[int, int], letter: str,
                         prefixes: Set[str], word_set: Set[str]) -> int:
    """Count how many word paths this letter could enable."""
    i, j = pos
    height, width = len(grid), len(grid[0])
    potential = 0
    
    # Check all 8 neighbors
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width and grid[ni][nj]:
                # Check if neighbor + letter forms valid prefix
                prefix1 = grid[ni][nj] + letter
                prefix2 = letter + grid[ni][nj]
                
                if prefix1 in prefixes or prefix1 in word_set:
                    potential += 1
                if prefix2 in prefixes or prefix2 in word_set:
                    potential += 1
    
    return potential


def calculate_neighbor_synergy(grid: List[List[str]], pos: Tuple[int, int], 
                               letter: str, prefixes: Set[str]) -> float:
    """Calculate how well this letter works with existing neighbors."""
    i, j = pos
    height, width = len(grid), len(grid[0])
    synergy = 0.0
    neighbor_count = 0
    
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width and grid[ni][nj]:
                neighbor_count += 1
                # Check if this creates common letter pairs
                pair1 = grid[ni][nj] + letter
                pair2 = letter + grid[ni][nj]
                
                # Bonus for common English letter pairs
                common_pairs = {'TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ON', 'AT', 'EN', 'ND'}
                if pair1 in common_pairs or pair2 in common_pairs:
                    synergy += 2.0
                
                # Check if forms valid prefix
                if pair1 in prefixes or pair2 in prefixes:
                    synergy += 1.0
    
    return synergy / max(neighbor_count, 1)


def count_letter_usage(grid: List[List[str]]) -> Dict[str, int]:
    """Count how many times each letter appears in the grid."""
    usage = {letter: 0 for letter in string.ascii_uppercase}
    for row in grid:
        for cell in row:
            if cell:
                usage[cell] += 1
    return usage


def score_letter_candidates(grid: List[List[str]], pos: Tuple[int, int], 
                           density: float, word_set: Set[str], difficult_set: Set[str],
                           prefixes: Set[str], difficult_prefixes: Set[str],
                           difficult_letter_freq: Dict[str, float]) -> List[Tuple[str, float]]:
    """Score each letter based on multiple factors."""
    scores = {}
    
    # Get current letter usage to penalize over-used letters
    letter_usage = count_letter_usage(grid)
    total_cells = len(grid) * len(grid[0])
    filled_cells = sum(letter_usage.values())
    
    # Calculate unique letter count for diversity bonus
    unique_letters_used = sum(1 for count in letter_usage.values() if count > 0)
    
    for letter in string.ascii_uppercase:
        score = 0.0
        
        # Factor 1: Natural letter frequency (weighted by inverse density)
        score += NORMALIZED_FREQ[letter] * 100 * (1 - density * 0.7)
        
        # Factor 2: Difficult word letter frequency (weighted by density)
        score += difficult_letter_freq.get(letter, 0) * 100 * density * 2.0
        
        # Factor 3: Difficult word potential
        difficult_potential = count_potential_words(grid, pos, letter, difficult_prefixes, difficult_set)
        score += difficult_potential * density * 5.0
        
        # Factor 4: General word potential
        word_potential = count_potential_words(grid, pos, letter, prefixes, word_set)
        score += word_potential * 2.0
        
        # Factor 5: Neighbor synergy
        neighbor_score = calculate_neighbor_synergy(grid, pos, letter, prefixes)
        score += neighbor_score * 3.0
        
        # Factor 6: Rare letter bonus - encourage interesting letters
        if letter in VERY_RARE_LETTERS:
            # Very rare letters (Q, J, X, Z) get big bonus if not yet used
            if letter_usage[letter] == 0:
                score += 25.0 * density  # Scale with difficulty
            elif letter_usage[letter] == 1:
                score += 10.0 * density
        elif letter in RARE_LETTERS:
            # Rare letters (V, K, W, F) get moderate bonus
            if letter_usage[letter] == 0:
                score += 15.0 * density
            elif letter_usage[letter] == 1:
                score += 5.0 * density
        
        # Factor 7: Diversity bonus - reward using new letters
        if letter_usage[letter] == 0 and filled_cells > 3:
            # Bonus for introducing a new letter (scales with grid fill)
            diversity_bonus = 20.0 * (filled_cells / total_cells)
            score += diversity_bonus
        
        # Factor 8: Diversity penalty - heavily penalize overused letters
        if filled_cells > 0:
            current_usage_ratio = letter_usage[letter] / filled_cells
            expected_ratio = NORMALIZED_FREQ[letter]
            
            # Progressive penalty that gets stronger with overuse
            if current_usage_ratio > expected_ratio * 1.3:
                overuse_factor = current_usage_ratio / expected_ratio
                overuse_penalty = (overuse_factor - 1.3) * 300
                score -= overuse_penalty
            
            # Extra harsh penalty for extreme overuse (>2x expected)
            if current_usage_ratio > expected_ratio * 2.0:
                score -= 500
        
        # Add small random factor to break ties
        score += random.random() * 0.1
        
        scores[letter] = score
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def find_words_in_grid(grid: List[List[str]], word_set: Set[str], 
                      prefixes: Set[str], min_length: int = 3) -> Set[str]:
    """Find all valid words in the grid using DFS."""
    height, width = len(grid), len(grid[0])
    found_words = set()
    
    def dfs(i: int, j: int, current_word: str, path: List[Tuple[int, int]]):
        if current_word and current_word not in prefixes and current_word not in word_set:
            return
        
        if len(current_word) >= min_length and current_word in word_set:
            found_words.add(current_word)
        
        if len(current_word) >= 16:
            return
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in path:
                    dfs(ni, nj, current_word + grid[ni][nj], path + [(ni, nj)])
    
    for i in range(height):
        for j in range(width):
            dfs(i, j, "", [])
    
    return found_words


def select_anchor_words(difficult_set: Set[str], min_difficult: int) -> List[str]:
    """Select anchor words to guarantee placement."""
    if min_difficult == 0:
        return []
    
    # Prioritize longer, more unique words
    sorted_words = sorted(difficult_set, key=lambda w: (len(w), sum(1 for c in w if c in 'QXZJK')), reverse=True)
    
    # Select top words, but not too many (leave room for organic formation)
    anchor_count = min(min_difficult // 2, len(sorted_words))
    return sorted_words[:anchor_count]


def try_place_anchor(grid: List[List[str]], word: str, density_map: List[List[float]],
                    prefixes: Set[str], max_attempts: int = 50) -> bool:
    """Try to place an anchor word in a high-density area."""
    height, width = len(grid), len(grid[0])
    
    # Find high-density positions
    high_density_positions = []
    for i in range(height):
        for j in range(width):
            if density_map[i][j] > 0.5:
                high_density_positions.append((i, j))
    
    if not high_density_positions:
        high_density_positions = [(i, j) for i in range(height) for j in range(width)]
    
    random.shuffle(high_density_positions)
    
    for _ in range(max_attempts):
        if not high_density_positions:
            break
        start_pos = high_density_positions.pop()
        
        if place_word_dfs(grid, word, start_pos, 0, set()):
            return True
    
    return False


def place_word_dfs(grid: List[List[str]], word: str, pos: Tuple[int, int],
                  index: int, path: Set[Tuple[int, int]]) -> bool:
    """Recursively place a word using DFS with backtracking."""
    if index == len(word):
        return True
    
    i, j = pos
    height, width = len(grid), len(grid[0])
    
    if not (0 <= i < height and 0 <= j < width) or pos in path:
        return False
    
    if grid[i][j] is not None and grid[i][j] != word[index]:
        return False
    
    original = grid[i][j]
    if grid[i][j] is None:
        grid[i][j] = word[index]
    path.add(pos)
    
    if index == len(word) - 1:
        return True
    
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    random.shuffle(directions)
    
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if place_word_dfs(grid, word, (ni, nj), index + 1, path):
            return True
    
    path.remove(pos)
    if original is None:
        grid[i][j] = None
    
    return False


def place_anchors(grid: List[List[str]], anchor_words: List[str], 
                 density_map: List[List[float]], prefixes: Set[str]) -> int:
    """Place anchor words in the grid."""
    placed = 0
    for word in anchor_words:
        if try_place_anchor(grid, word, density_map, prefixes):
            placed += 1
    return placed


def get_fill_order(grid: List[List[str]], density_map: List[List[float]]) -> List[Tuple[int, int]]:
    """Get the order to fill cells (high density first, then random)."""
    height, width = len(grid), len(grid[0])
    positions = []
    
    for i in range(height):
        for j in range(width):
            if grid[i][j] is None:
                positions.append((i, j, density_map[i][j]))
    
    # Sort by density (high first), then randomize within density bands
    positions.sort(key=lambda x: (-x[2], random.random()))
    
    return [(p[0], p[1]) for p in positions]


def estimate_achievable_words(grid: List[List[str]], word_set: Set[str],
                             difficult_set: Set[str], prefixes: Set[str]) -> Tuple[int, int]:
    """Quick estimate of how many words could still be formed."""
    # Count filled cells
    filled = sum(1 for row in grid for cell in row if cell is not None)
    total = len(grid) * len(grid[0])
    fill_ratio = filled / total
    
    # Rough heuristic: assume linear relationship
    if fill_ratio < 0.3:
        return len(word_set), len(difficult_set)
    
    # Do a quick partial search for better estimate
    current_words = find_words_in_grid(grid, word_set, prefixes)
    current_difficult = current_words & difficult_set
    
    # Extrapolate based on fill ratio
    estimated_total = int(len(current_words) / max(fill_ratio, 0.3))
    estimated_difficult = int(len(current_difficult) / max(fill_ratio, 0.3))
    
    return estimated_total, estimated_difficult


def generate_adaptive_puzzle(
    width: int,
    height: int,
    word_list_path: str,
    difficult_list_path: str,
    constraints: Optional[Dict] = None,
    density_map: Optional[List[List[float]]] = None,
    density_type: str = 'radial',
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[List[List[str]], Set[str], Set[str], Dict]:
    """
    Generate a puzzle using adaptive density and constraint satisfaction.
    
    Args:
        width: Grid width
        height: Grid height
        word_list_path: Path to main word list
        difficult_list_path: Path to difficult words list
        constraints: Dict with keys: min_words, min_difficult, difficulty_score (0.0-1.0)
        density_map: Optional custom density map (height x width array)
        density_type: 'radial', 'gradient_h', 'gradient_v', or 'uniform'
        seed: Random seed
        verbose: Print progress
    
    Returns:
        (grid, found_words, found_difficult, stats)
    """
    if seed is not None:
        random.seed(seed)
    
    # Default constraints
    if constraints is None:
        constraints = {
            'min_words': 20,
            'min_difficult': 3,
            'difficulty_score': 0.5
        }
    
    # Load word lists
    word_set = load_word_list(word_list_path)
    difficult_set = load_word_list(difficult_list_path)
    
    if not word_set:
        raise ValueError("Failed to load word list")
    
    if verbose:
        print(f"Loaded {len(word_set)} words, {len(difficult_set)} difficult words")
    
    # Build prefix sets
    prefixes = build_prefix_set(word_set)
    difficult_prefixes = build_prefix_set(difficult_set)
    
    # Analyze difficult word letter frequencies
    difficult_letter_freq = analyze_difficult_letter_frequencies(difficult_set)
    
    # Create density map
    if density_map is None:
        difficulty_score = constraints.get('difficulty_score', 0.5)
        if density_type == 'radial':
            density_map = create_radial_density_map(width, height, difficulty_score)
        elif density_type == 'gradient_h':
            density_map = create_gradient_density_map(width, height, difficulty_score, 'horizontal')
        elif density_type == 'gradient_v':
            density_map = create_gradient_density_map(width, height, difficulty_score, 'vertical')
        else:  # uniform
            density_map = create_uniform_density_map(width, height, difficulty_score)
    
    if verbose:
        print(f"Using {density_type} density map with difficulty {constraints.get('difficulty_score', 0.5):.2f}")
    
    # Initialize grid
    grid = [[None for _ in range(width)] for _ in range(height)]
    
    # Phase 1: Place anchor words
    min_difficult = constraints.get('min_difficult', 0)
    anchor_words = select_anchor_words(difficult_set, min_difficult)
    
    if anchor_words and verbose:
        print(f"Placing {len(anchor_words)} anchor words...")
    
    placed_anchors = place_anchors(grid, anchor_words, density_map, prefixes)
    
    if verbose and anchor_words:
        print(f"Successfully placed {placed_anchors}/{len(anchor_words)} anchors")
    
    # Phase 2: Fill remaining cells with constraint-aware selection
    fill_order = get_fill_order(grid, density_map)
    
    if verbose:
        print(f"Filling {len(fill_order)} remaining cells...")
    
    for idx, pos in enumerate(fill_order):
        i, j = pos
        
        if grid[i][j] is not None:
            continue
        
        # Score all letter candidates
        candidates = score_letter_candidates(
            grid, pos, density_map[i][j],
            word_set, difficult_set,
            prefixes, difficult_prefixes,
            difficult_letter_freq
        )
        
        # Select top candidate (already sorted by score)
        grid[i][j] = candidates[0][0]
        
        if verbose and (idx + 1) % 5 == 0:
            progress = (idx + 1) / len(fill_order) * 100
            print(f"  Progress: {progress:.1f}%", end='\r')
    
    if verbose:
        print(f"  Progress: 100.0%")
    
    # Phase 3: Find all words
    if verbose:
        print("Searching for words in completed grid...")
    
    found_words = find_words_in_grid(grid, word_set, prefixes)
    found_difficult = found_words & difficult_set
    
    # Compile stats
    stats = {
        'total_words': len(found_words),
        'difficult_words': len(found_difficult),
        'difficulty_score': constraints.get('difficulty_score', 0.5),
        'anchors_placed': placed_anchors,
        'anchors_attempted': len(anchor_words),
        'meets_min_words': len(found_words) >= constraints.get('min_words', 0),
        'meets_min_difficult': len(found_difficult) >= constraints.get('min_difficult', 0)
    }
    
    if verbose:
        print(f"\nResults:")
        print(f"  Total words found: {stats['total_words']}")
        print(f"  Difficult words: {stats['difficult_words']}")
        print(f"  Meets constraints: words={stats['meets_min_words']}, difficult={stats['meets_min_difficult']}")
    
    return grid, found_words, found_difficult, stats


def generate_best_adaptive_puzzle(
    width: int,
    height: int,
    word_list_path: str,
    difficult_list_path: str,
    constraints: Optional[Dict] = None,
    density_type: str = 'radial',
    iterations: int = 5,
    verbose: bool = True
) -> Tuple[List[List[str]], Set[str], Set[str], Dict]:
    """
    Generate multiple puzzles and return the best one based on constraints.
    
    Args:
        iterations: Number of puzzles to generate
        Other args same as generate_adaptive_puzzle
    
    Returns:
        Best (grid, found_words, found_difficult, stats)
    """
    if verbose:
        print(f"Generating {iterations} puzzles to find the best...\n")
    
    best_result = None
    best_score = -1
    
    for i in range(iterations):
        if verbose:
            print(f"=== Iteration {i + 1}/{iterations} ===")
        
        try:
            result = generate_adaptive_puzzle(
                width, height, word_list_path, difficult_list_path,
                constraints, None, density_type, None, verbose
            )
            
            grid, found_words, found_difficult, stats = result
            
            # Score this puzzle
            score = len(found_difficult) * 10 + len(found_words)
            
            # Bonus for meeting constraints
            if constraints:
                if len(found_words) >= constraints.get('min_words', 0):
                    score += 50
                if len(found_difficult) >= constraints.get('min_difficult', 0):
                    score += 100
            
            if verbose:
                print(f"  Score: {score}\n")
            
            if score > best_score:
                best_score = score
                best_result = result
        
        except Exception as e:
            if verbose:
                print(f"  Error in iteration {i + 1}: {e}\n")
            continue
    
    if best_result is None:
        raise ValueError("Failed to generate any valid puzzles")
    
    if verbose:
        print(f"=== Best Puzzle (Score: {best_score}) ===")
        grid, found_words, found_difficult, stats = best_result
        print(f"Total words: {len(found_words)}, Difficult: {len(found_difficult)}")
    
    return best_result


def print_grid(grid: List[List[str]]):
    """Print the grid in a readable format."""
    for row in grid:
        print(" ".join(row))


def visualize_density_map(density_map: List[List[float]]):
    """Visualize the density map with ASCII art."""
    height = len(density_map)
    width = len(density_map[0]) if height > 0 else 0
    print("\nDensity Map (0=easy, 9=hard):")
    for i in range(height):
        row = []
        for j in range(width):
            level = int(density_map[i][j] * 9)
            row.append(str(level))
        print(" ".join(row))


def print_word_list(words: Set[str], title: str = "Words", max_display: int = 50):
    """Print a list of words in columns."""
    word_list = sorted(words)
    print(f"\n{title} ({len(word_list)} total):")
    
    if len(word_list) <= max_display:
        # Print in columns
        cols = 4
        for i in range(0, len(word_list), cols):
            row = word_list[i:i+cols]
            print("  " + "  ".join(f"{w:12}" for w in row))
    else:
        # Print first few and last few
        print(f"  First {max_display//2}:")
        for i in range(0, max_display//2, 4):
            row = word_list[i:i+4]
            print("    " + "  ".join(f"{w:12}" for w in row))
        print(f"  ... ({len(word_list) - max_display} more) ...")
        print(f"  Last {max_display//2}:")
        start = len(word_list) - max_display//2
        for i in range(start, len(word_list), 4):
            row = word_list[i:i+4]
            print("    " + "  ".join(f"{w:12}" for w in row))


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
        
    # Extract params specific to this generator
    constraints = params.get('constraints', {
        'min_words': 20,
        'min_difficult': 3,
        'difficulty_score': 0.5
    })
    density_type = params.get('density_type', 'radial')
    verbose = params.get('verbose', False)
    
    # Call the internal implementation
    grid, found_words, found_difficult, stats = generate_adaptive_puzzle(
        width, height, word_list_path, difficult_list_path,
        constraints, None, density_type, seed, verbose
    )
    
    return {
        'grid': grid,
        'found_words': found_words,
        'difficult_words': found_difficult,
        'score': len(found_difficult) * 10 + len(found_words),
        'metadata': stats
    }

if __name__ == "__main__":
    # Example usage
    WIDTH = 6
    HEIGHT = 6
    WORD_LIST = "generator/TWL.txt"
    DIFFICULT_LIST = "generator/difficult.txt"
    
    result = generate_puzzle(
        WIDTH, HEIGHT, WORD_LIST, DIFFICULT_LIST, 
        seed=42, 
        params={'verbose': True}
    )
    
    print_grid(result['grid'])
    print(f"Found {len(result['found_words'])} words, {len(result['difficult_words'])} difficult.")
