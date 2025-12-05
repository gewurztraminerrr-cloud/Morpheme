import random
from typing import List, Set, Tuple, Dict, Optional
import time
import string

def load_word_list(word_list: List[str]) -> Set[str]:
    """Prepare the word list by filtering and converting to uppercase."""
    return {w.upper() for w in word_list if 3 <= len(w) <= 16}

def try_place_word(grid: List[List[str]], word: str, start_pos: Tuple[int, int]) -> bool:
    """Place a word in the grid starting from start_pos using backtracking with randomized direction."""
    # Fast path for words that are too long
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    if len(word) > height * width:
        return False
        
    def place(index: int, pos: Tuple[int, int], path: Set[Tuple[int, int]]) -> bool:
        if index == len(word):
            return True
        i, j = pos
        
        # Early termination checks
        if not (0 <= i < height and 0 <= j < width) or pos in path:
            return False
            
        # If cell is filled, it must match the current letter
        if grid[i][j] is not None and grid[i][j] != word[index]:
            return False
            
        # Store original value for backtracking
        original = grid[i][j]
        if grid[i][j] is None:
            grid[i][j] = word[index]
        path.add(pos)
        
        # Randomize the order of directions to try
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        random.shuffle(directions)
        
        # Try all adjacent directions in randomized order
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if place(index + 1, (ni, nj), path):
                return True
                
        # Backtrack if placement fails
        path.remove(pos)
        if original is None:
            grid[i][j] = None
        return False
        
    return place(0, start_pos, set())

def build_prefix_set(word_set: Set[str]) -> Set[str]:
    """Build a set of all prefixes from the words."""
    prefixes = set()
    for word in word_set:
        for i in range(1, len(word)):
            prefixes.add(word[:i])
    return prefixes

def dfs(grid: List[List[str]], pos: Tuple[int, int], current_word: str, 
        path: List[Tuple[int, int]], word_set: Set[str], prefixes: Set[str], found_words: Set[str]):
    """Search the grid for all possible words using DFS with prefix optimization."""
    i, j = pos
    current_word += grid[i][j]
    
    # Early termination - check using precomputed prefix set
    if len(current_word) > 2 and current_word not in prefixes and current_word not in word_set:
        return
        
    if len(current_word) >= 3 and current_word in word_set:
        found_words.add(current_word)
        
    if len(current_word) == 16:  # Maximum path length in grid
        return
        
    height = len(grid)
    width = len(grid[0])
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in path:
                dfs(grid, (ni, nj), current_word, path + [(ni, nj)], word_set, prefixes, found_words)

def generate_puzzle_internal(word_list: List[str], min_word_count: int, seed: int, 
                   difficult_list: List[str] = None, difficult_percentage: float = 0.0, 
                   max_attempts: int = 10, initial_word_length: Tuple[int, int] = None,
                   grid_width: int = 4, grid_height: int = 4, min_difficult_words: int = 0) -> Tuple[List[List[str]], int, Set[str], Set[str]]:
    """
    Generate a word puzzle meeting the minimum word count and difficulty requirements.
    """
    start_time = time.time()
    
    if difficult_percentage < 0.0 or difficult_percentage > 100.0:
        raise ValueError("difficult_percentage must be between 0.0 and 100.0")
        
    print(f"Loading word lists...")
    word_set = load_word_list(word_list)
    difficult_set = load_word_list(difficult_list) if difficult_list else set()
    
    # Pre-sort all lists once for determinism
    sorted_word_list = sorted(list(word_set))
    sorted_difficult_list = sorted(list(difficult_set)) if difficult_set else []
    
    # Prepare initial word candidates based on length range or random selection
    if initial_word_length is not None:
        min_len, max_len = initial_word_length
        initial_candidates = [w for w in sorted_word_list if min_len <= len(w) <= max_len]
        if not initial_candidates:
            raise ValueError(f"No words found in length range {min_len}-{max_len}")
        print(f"Using {len(initial_candidates)} words of length {min_len}-{max_len} for initial placement")
    else:
        # Use all words as candidates, will be randomly selected using the seed
        initial_candidates = sorted_word_list
        print(f"Using all {len(initial_candidates)} words for initial placement")

    # If difficult list is provided, prioritize some difficult words in initial placement
    difficult_candidates = [w for w in initial_candidates if w in difficult_set] if difficult_set else []
    print(f"Word lists loaded in {time.time() - start_time:.2f} seconds")
    
    # Pre-compute prefix set for faster word finding
    prefix_time = time.time()
    prefixes = build_prefix_set(word_set)
    print(f"Built prefix set in {time.time() - prefix_time:.2f} seconds")

    # Pre-compute word length groups for faster selection
    word_length_groups = {}
    for word in sorted_word_list:
        length = len(word)
        if length not in word_length_groups:
            word_length_groups[length] = []
        word_length_groups[length].append(word)
    # No need to sort each group since we're using pre-sorted word_list

    for attempt in range(max_attempts):
        attempt_start = time.time()
        current_seed = seed + attempt
        random.seed(current_seed)
        grid = [[None]*grid_width for _ in range(grid_height)]

        print(f"Attempt {attempt+1}/{max_attempts} with seed {current_seed}...")
        
        # Place the first word, prioritize difficult if available and percentage > 0
        first_word_candidates = difficult_candidates if difficult_set and difficult_percentage > 0 and difficult_candidates else initial_candidates
        first_word = random.choice(first_word_candidates)
        start_i = random.randint(0, grid_height-1)
        start_j = random.randint(0, grid_width-1)
        if not try_place_word(grid, first_word, (start_i, start_j)):
            print(f"  Failed to place first word: {first_word}")
            continue
        print(f"  Placed first word: {first_word}")

        # Place additional words strategically
        placed_words = 1
        max_placement_attempts = 100
        for _ in range(max_placement_attempts):
            # Adjust probability to pick from difficult list based on required percentage
            if difficult_set and difficult_percentage > 0 and random.random() < difficult_percentage / 100.0:
                word_candidates = sorted_difficult_list
                if not word_candidates:
                    continue
                word = random.choice(word_candidates)
            else:
                # Use length-based selection for better performance
                target_length = min(int(random.expovariate(0.5)) + 3, 16)  # Bias towards longer words
                if target_length in word_length_groups:
                    word = random.choice(word_length_groups[target_length])
                else:
                    # Fallback to random word if no words of target length
                    word = random.choice(sorted_word_list)
                
            # Try placing at different positions
            for _ in range(8):  # Limit attempts per word
                start_i = random.randint(0, grid_height-1)
                start_j = random.randint(0, grid_width-1)
                if try_place_word(grid, word, (start_i, start_j)):
                    placed_words += 1
                    break
        
        print(f"  Placed {placed_words} words")

        # Fill remaining empty cells with reasonable letters
        empty_cells = [(i, j) for i in range(grid_height) for j in range(grid_width) if grid[i][j] is None]
        if empty_cells:
            # For each empty cell, find letters that can form words with existing letters
            for i, j in empty_cells:
                possible_letters = set()
                # Check all 8 directions around the cell
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_height and 0 <= nj < grid_width and grid[ni][nj] is not None:
                            # Try each letter to see if it forms a word with neighbors
                            for letter in string.ascii_uppercase:
                                grid[i][j] = letter
                                # Check if this letter forms a word in any direction
                                for word_length in range(3, 5):  # Check for 3-4 letter words
                                    # Check in all 8 directions
                                    for ddi in [-1, 0, 1]:
                                        for ddj in [-1, 0, 1]:
                                            if ddi == 0 and ddj == 0:
                                                continue
                                            # Build potential word
                                            word = letter
                                            ci, cj = i, j
                                            for _ in range(word_length - 1):
                                                ci += ddi
                                                cj += ddj
                                                if 0 <= ci < grid_height and 0 <= cj < grid_width and grid[ci][cj] is not None:
                                                    word += grid[ci][cj]
                                                else:
                                                    break
                                            if len(word) >= 3 and word in word_set:
                                                possible_letters.add(letter)
                                                break
                                grid[i][j] = None
                
                if not possible_letters:
                    print(f"  Failed to find valid letter for cell ({i}, {j})")
                    break  # Reject this attempt and try next seed
                
                # Choose a random letter from those that form words
                grid[i][j] = random.choice(list(possible_letters))
            else:
                # Only continue if we successfully filled all empty cells
                continue
            # If we broke out of the loop, skip to next attempt
            continue
        
        # Find all possible words
        find_start = time.time()
        found_words = set()
        for i in range(grid_height):
            for j in range(grid_width):
                dfs(grid, (i, j), '', [(i, j)], word_set, prefixes, found_words)
        
        find_time = time.time() - find_start
        print(f"  Found {len(found_words)} words in {find_time:.2f} seconds")

        # Check minimum word count
        if len(found_words) < min_word_count:
            print(f"  Insufficient words found: {len(found_words)} < {min_word_count}")
            continue
            
        # Check difficult percentage and minimum number for 6-letter words if applicable
        difficult_words = set()
        if difficult_set and (difficult_percentage > 0 or min_difficult_words > 0):
            long_words = [w for w in found_words if len(w) >= 6]
            difficult_words = {w for w in long_words if w in difficult_set}
            
            # Check minimum number of difficult words
            if min_difficult_words > 0 and len(difficult_words) < min_difficult_words:
                print(f"  Insufficient difficult words count: {len(difficult_words)} < {min_difficult_words}")
                continue
                
            # Check percentage requirement if specified
            if difficult_percentage > 0:
                actual_percentage = (len(difficult_words) / len(long_words)) * 100.0 if long_words else 0.0
                if actual_percentage < difficult_percentage:
                    print(f"  Insufficient difficult words percentage: {actual_percentage:.1f}% < {difficult_percentage:.1f}%")
                    continue

        print(f"Success! Found {len(found_words)} words in seed #{current_seed}")
        print(f"Attempt completed in {time.time() - attempt_start:.2f} seconds")
        
        return grid, current_seed, found_words, difficult_words

    raise Exception(f"Failed to generate puzzle after {max_attempts} attempts")

def read_word_list(file_path):
    with open(file_path, 'r') as file:
        words = file.readlines()
        words = [word.strip() for word in words]  # Remove trailing newlines
        words = [word for word in words if 3 <= len(word) <= 16]  # Filter words by length
    return words

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
        
    # Load word lists
    word_list = read_word_list(word_list_path)
    difficult_list = read_word_list(difficult_list_path)
    
    # Extract params
    min_word_count = params.get('min_word_count', 20)
    difficult_percentage = params.get('difficult_percentage', 0.0)
    min_difficult_words = params.get('min_difficult_words', 3)
    max_attempts = params.get('max_attempts', 100)
    
    if seed is None:
        seed = random.randint(1, 1000000)
        
    grid, used_seed, found_words, difficult_words = generate_puzzle_internal(
        word_list,
        min_word_count=min_word_count,
        seed=seed,
        difficult_list=difficult_list,
        difficult_percentage=difficult_percentage,
        max_attempts=max_attempts,
        grid_width=width,
        grid_height=height,
        min_difficult_words=min_difficult_words
    )
    
    return {
        'grid': grid,
        'found_words': found_words,
        'difficult_words': difficult_words,
        'score': len(difficult_words) * 10 + len(found_words),
        'metadata': {
            'seed': used_seed
        },
        'seed': used_seed
    }

# Example usage
if __name__ == "__main__":
    overall_start = time.time()
    try:
        print(f"Starting puzzle generation...")
        result = generate_puzzle(
            width=6,
            height=6,
            word_list_path="generator/CSW22.txt",
            difficult_list_path="generator/difficult.txt",
            seed=25364461,
            params={
                'min_word_count': 800,
                'difficult_percentage': 60,
                'min_difficult_words': 44,
                'max_attempts': 1000000
            }
        )
        
        print(f"\nPuzzle generated with seed {result.get('seed')}:")
        for row in result['grid']:
            print("  ".join(row))
            
        print(f"\nTotal generation time: {time.time() - overall_start:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")