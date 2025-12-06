import time
import statistics
import sys
import os
from typing import List, Dict, Any
import importlib.util

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import adaptive_puzzle
import checkerboard_puzzle
import generate_optimized
import hybrid_puzzle

def run_benchmark(
    generators: List[Dict[str, Any]],
    width: int,
    height: int,
    word_list_path: str,
    difficult_list_path: str,
    iterations: int = 5
):
    print(f"Benchmarking {len(generators)} generators ({iterations} iterations each)...")
    print(f"Grid Size: {width}x{height}")
    print("-" * 80)
    print(f"{'Generator':<20} | {'Avg Time':<10} | {'Avg Words':<10} | {'Avg Diff':<10} | {'Max Diff':<10} | {'Score':<10}")
    print("-" * 80)
    
    results = {}
    
    for gen_config in generators:
        name = gen_config['name']
        module = gen_config['module']
        params = gen_config.get('params', {})
        
        times = []
        word_counts = []
        diff_counts = []
        scores = []
        
        for i in range(iterations):
            start_time = time.time()
            try:
                # Use a different seed for each iteration
                seed = 42 + i
                
                result = module.generate_puzzle(
                    width, height, word_list_path, difficult_list_path,
                    seed=seed,
                    params=params
                )
                
                duration = time.time() - start_time
                times.append(duration)
                word_counts.append(len(result['found_words']))
                diff_counts.append(len(result['difficult_words']))
                scores.append(result['score'])
                
            except Exception as e:
                print(f"Error running {name}: {e}")
                import traceback
                traceback.print_exc()
        
        if times:
            avg_time = statistics.mean(times)
            avg_words = statistics.mean(word_counts)
            avg_diff = statistics.mean(diff_counts)
            max_diff = max(diff_counts)
            avg_score = statistics.mean(scores)
            
            print(f"{name:<20} | {avg_time:<10.4f} | {avg_words:<10.1f} | {avg_diff:<10.1f} | {max_diff:<10d} | {avg_score:<10.1f}")
            
            results[name] = {
                'avg_time': avg_time,
                'avg_words': avg_words,
                'avg_diff': avg_diff,
                'max_diff': max_diff,
                'avg_score': avg_score
            }
            
    print("-" * 80)
    return results

def main():
    # Configuration
    WIDTH = 6
    HEIGHT = 6
    WORD_LIST = "generator/TWL.txt"
    DIFFICULT_LIST = "generator/difficult.txt"
    ITERATIONS = 2
    
    # Define generators to test
    generators = [
        {
            'name': 'Adaptive (Radial)',
            'module': adaptive_puzzle,
            'params': {'density_type': 'radial', 'constraints': {'min_words': 20, 'min_difficult': 5}}
        },
        {
            'name': 'Checkerboard',
            'module': checkerboard_puzzle,
            'params': {'iterations': 1} # Single run for fair comparison of speed
        },
        {
            'name': 'Optimized (Legacy)',
            'module': generate_optimized,
            'params': {
                'min_word_count': 100,
                'difficult_percentage': 50,
                'min_difficult_words': 10,
                'max_attempts': 100
            }
        },
        {
            'name': 'Hybrid Beam',
            'module': hybrid_puzzle,
            'params': {'beam_width': 5}
        }
    ]
    
    # Run benchmark
    run_benchmark(generators, WIDTH, HEIGHT, WORD_LIST, DIFFICULT_LIST, ITERATIONS)

if __name__ == "__main__":
    main()
