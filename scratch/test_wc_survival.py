import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spinner_set import SpinnerSet

def main():
    dims_list = ['4x4', '4x6', '5x7', '6x8']
    iterations = 10000
    
    for dims in dims_list:
        counts = {"50-100": 0, "100-200": 0, "200-300": 0, "300-400": 0, "500+": 0}
        min_lens = {}
        for _ in range(iterations):
            params = SpinnerSet.generate_params(dims, is_24h=False)
            wc = params.get('word_count_range', 'unknown')
            counts[wc] = counts.get(wc, 0) + 1
            
            ml = params.get('min_word_length', 3)
            min_lens[ml] = min_lens.get(ml, 0) + 1
            
        print(f"\n--- Dimension: {dims} (over {iterations} iterations) ---")
        print("Word Count Range Distributions:")
        for wc, count in sorted(counts.items()):
            pct = (count / iterations) * 100
            print(f"  {wc}: {count} ({pct:.2f}%)")
        print("Min Word Lengths Generated:")
        for ml, count in sorted(min_lens.items()):
            pct = (count / iterations) * 100
            print(f"  {ml}L: {count} ({pct:.2f}%)")

if __name__ == "__main__":
    main()
