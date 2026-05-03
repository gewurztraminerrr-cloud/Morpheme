import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from board_generator import BoardGenerator

def analyze_board():
    bg = BoardGenerator()
    
    board = [
        ['T', 'O', 'A', 'S'],
        ['U', 'L', 'L', 'W'],
        ['B', 'T', 'S', 'B'],
        ['S', 'T', 'C', 'R']
    ]
    
    print("Analyzing User's Board...")
    
    # Solve with UniqueNWL
    unique_nwl = bg._solve_board(board, "UniqueNWL", (0, 99999), 3, max_depth=12, store_paths=True)
    print(f"UniqueNWL words found: {len(unique_nwl)}")
    print("First 20 UniqueNWL words:")
    print(", ".join(list(unique_nwl.keys())[:20]))
    
    # Solve with NWL
    all_nwl = bg._solve_board(board, "NWL", (0, 99999), 3, max_depth=12, store_paths=True)
    print(f"\nTotal NWL words found: {len(all_nwl)}")
    print("First 20 NWL words:")
    print(", ".join(list(all_nwl.keys())[:20]))

if __name__ == "__main__":
    analyze_board()
