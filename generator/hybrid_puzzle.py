import random
import time
import string
import math
from typing import List, Set, Tuple, Dict, Optional, Any
import copy

# Import helper functions from other modules if needed, or reimplement for independence
# For now, we'll implement necessary helpers to keep it self-contained but consistent

def load_word_list(path: str) -> Set[str]:
    with open(path, 'r') as f:
        return {line.strip().upper() for line in f if 3 <= len(line.strip()) <= 16}

def create_radial_density_map(width: int, height: int) -> List[List[float]]:
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    max_dist = math.sqrt(center_x**2 + center_y**2)
    density_map = [[0.0] * width for _ in range(height)]
    
    for y in range(height):
        for x in range(width):
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            # Higher density in the center
            density_map[y][x] = 1.0 - (dist / (max_dist * 1.5))
            
    return density_map

def get_neighbors(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                neighbors.append((nr, nc))
    return neighbors

def find_words(grid: List[List[str]], word_set: Set[str]) -> Set[str]:
    found = set()
    h, w = len(grid), len(grid[0])
    
    # Pre-check: if grid has None, we can't fully check, but we can check what's there
    # For speed, we might want a prefix tree, but for now standard DFS
    
    # Optimization: Build prefix set for fast pruning
    prefixes = set()
    for word in word_set:
        for i in range(1, len(word) + 1):
            prefixes.add(word[:i])
            
    visited = set()
    
    def dfs(r, c, current_word, path_set):
        if current_word not in prefixes:
            return
        
        if current_word in word_set:
            found.add(current_word)
            
        if len(current_word) >= 16:
            return
            
        for nr, nc in get_neighbors(r, c, h, w):
            if (nr, nc) not in path_set:
                if grid[nr][nc] is not None:
                    dfs(nr, nc, current_word + grid[nr][nc], path_set | {(nr, nc)})

    for r in range(h):
        for c in range(w):
            if grid[r][c] is not None:
                dfs(r, c, grid[r][c], {(r, c)})
                
    return found

class HybridGenerator:
    def __init__(self, width: int, height: int, word_list: Set[str], difficult_list: Set[str], seed: int):
        self.width = width
        self.height = height
        self.word_set = word_list
        self.difficult_set = difficult_list
        self.rng = random.Random(seed)
        self.density_map = create_radial_density_map(width, height)
        
        # Precompute letter scores based on difficult words
        self.letter_scores = {l: 1 for l in string.ascii_uppercase}
        total_diff_letters = 0
        for word in difficult_list:
            for char in word:
                self.letter_scores[char] += 1
                total_diff_letters += 1
        
        # Normalize scores
        for char in self.letter_scores:
            self.letter_scores[char] /= (total_diff_letters + 26) # Smoothing

    def generate(self, beam_width: int = 5, max_steps: int = 100) -> Tuple[List[List[str]], Set[str], Set[str]]:
        # 1. Initialization
        grid = [[None] * self.width for _ in range(self.height)]
        
        # 2. Seeding: Place a difficult word in the center
        difficult_candidates = sorted(list(self.difficult_set))
        if difficult_candidates:
            seed_word = self.rng.choice(difficult_candidates)
            self._place_word_centered(grid, seed_word)
            
        # 3. Beam Search to fill
        # State: (grid, score)
        # Score heuristic: density_score + difficult_potential
        beams = [(grid, 0.0)]
        
        filled_count = sum(1 for r in range(self.height) for c in range(self.width) if grid[r][c] is not None)
        total_cells = self.width * self.height
        
        while filled_count < total_cells:
            next_beams = []
            
            for current_grid, current_score in beams:
                # Find empty cells adjacent to filled cells
                candidates = []
                for r in range(self.height):
                    for c in range(self.width):
                        if current_grid[r][c] is None:
                            # Check if neighbor is filled
                            has_neighbor = False
                            for nr, nc in get_neighbors(r, c, self.height, self.width):
                                if current_grid[nr][nc] is not None:
                                    has_neighbor = True
                                    break
                            if has_neighbor or filled_count == 0: # If empty grid, pick center
                                candidates.append((r, c))
                
                if not candidates and filled_count == 0:
                    candidates.append((self.height//2, self.width//2))
                
                # Limit candidates to top N by density to save time
                candidates.sort(key=lambda p: self.density_map[p[0]][p[1]], reverse=True)
                candidates = candidates[:5] 
                
                for r, c in candidates:
                    # Try top letters
                    # Heuristic: Pick letters that form valid prefixes with neighbors
                    # For simplicity in this version, pick random weighted by letter_scores
                    # improved: check local constraints
                    
                    possible_letters = self._get_promising_letters(current_grid, r, c)
                    
                    for letter in possible_letters[:4]: # Top 4 letters
                        new_grid = [row[:] for row in current_grid]
                        new_grid[r][c] = letter
                        
                        # Calculate score
                        score = current_score + self.letter_scores[letter] * self.density_map[r][c] * 10
                        
                        # Bonus for forming words (expensive check, maybe do partial?)
                        # For now, just trust the heuristic
                        
                        next_beams.append((new_grid, score))
            
            # Prune
            next_beams.sort(key=lambda x: x[1], reverse=True)
            beams = next_beams[:beam_width]
            
            if not beams:
                break
                
            filled_count += 1
            
        best_grid = beams[0][0]
        
        # Fill any remaining None (if beam search got stuck)
        for r in range(self.height):
            for c in range(self.width):
                if best_grid[r][c] is None:
                    best_grid[r][c] = self.rng.choice(string.ascii_uppercase)
                    
        # 4. Final Optimization (Local Search)
        # Swap letters to maximize difficult words
        best_grid = self._optimize_grid(best_grid)
        
        found_words = find_words(best_grid, self.word_set)
        difficult_words = found_words.intersection(self.difficult_set)
        
        return best_grid, found_words, difficult_words

    def _place_word_centered(self, grid, word):
        # Place horizontally or vertically centered
        direction = self.rng.choice(['H', 'V'])
        if direction == 'H':
            r = self.height // 2
            c_start = (self.width - len(word)) // 2
            if c_start < 0: c_start = 0
            for i, char in enumerate(word):
                if 0 <= c_start + i < self.width:
                    grid[r][c_start + i] = char
        else:
            c = self.width // 2
            r_start = (self.height - len(word)) // 2
            if r_start < 0: r_start = 0
            for i, char in enumerate(word):
                if 0 <= r_start + i < self.height:
                    grid[r_start + i][c] = char

    def _get_promising_letters(self, grid, r, c) -> List[str]:
        # Simple heuristic: return high scoring letters
        # Better: check neighbors and see what letters could complete words
        # For now, return top scoring letters + some random
        
        sorted_letters = sorted(self.letter_scores.keys(), key=lambda l: self.letter_scores[l], reverse=True)
        return sorted_letters[:10]

    def _optimize_grid(self, grid):
        # Simple hill climbing
        current_grid = [row[:] for row in grid]
        current_found = find_words(current_grid, self.word_set)
        current_score = len(current_found.intersection(self.difficult_set)) * 10 + len(current_found)
        
        for _ in range(50): # Iterations
            # Pick random cell
            r, c = self.rng.randint(0, self.height-1), self.rng.randint(0, self.width-1)
            old_char = current_grid[r][c]
            
            # Try changing to a high scoring letter
            new_char = self.rng.choice(string.ascii_uppercase)
            current_grid[r][c] = new_char
            
            new_found = find_words(current_grid, self.word_set)
            new_score = len(new_found.intersection(self.difficult_set)) * 10 + len(new_found)
            
            if new_score > current_score:
                current_score = new_score
            else:
                current_grid[r][c] = old_char # Revert
                
        return current_grid

def generate_puzzle(
    width: int,
    height: int,
    word_list_path: str,
    difficult_list_path: str,
    seed: Optional[int] = None,
    params: Optional[Dict] = None
) -> Dict:
    if params is None:
        params = {}
        
    if seed is None:
        seed = random.randint(1, 1000000)
        
    word_list = load_word_list(word_list_path)
    difficult_list = load_word_list(difficult_list_path)
    
    generator = HybridGenerator(width, height, word_list, difficult_list, seed)
    
    beam_width = params.get('beam_width', 5)
    
    grid, found_words, difficult_words = generator.generate(beam_width=beam_width)
    
    return {
        'grid': grid,
        'found_words': found_words,
        'difficult_words': difficult_words,
        'score': len(difficult_words) * 10 + len(found_words),
        'metadata': {
            'seed': seed,
            'algorithm': 'Hybrid Density-Guided Beam Search'
        },
        'seed': seed
    }
