import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Add parent directory to path to allow imports from generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import adaptive_puzzle, checkerboard_puzzle, checkerboard_v2, checkerboard_v3, generate_optimized, hybrid_puzzle

app = Flask(__name__)

# Configuration
WORD_LIST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generator", "TWL.txt")
DIFFICULT_LIST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generator", "randomTWLunique.txt")

ALGORITHMS = {
    'adaptive': adaptive_puzzle,
    'checkerboard': checkerboard_puzzle,
    'checkerboard_v2': checkerboard_v2,
    'checkerboard_v3': checkerboard_v3,
    'optimized': generate_optimized,
    'hybrid': hybrid_puzzle
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    return jsonify({
        'algorithms': [
            {'id': 'adaptive', 'name': 'Adaptive (Radial)', 'description': 'Balanced difficulty using radial density.'},
            {'id': 'checkerboard', 'name': 'Checkerboard', 'description': 'Distinct difficult word optimization.'},
            {'id': 'checkerboard_v2', 'name': 'Checkerboard v2', 'description': 'Enum strategies: total, distinct, or 6+ ratio.'},
            {'id': 'checkerboard_v3', 'name': 'Checkerboard v3 (Parallel)', 'description': 'Parallel Trie-based optimization with persistent pool.'},
            {'id': 'optimized', 'name': 'Optimized (Legacy)', 'description': 'Original optimization algorithm.'},
            {'id': 'hybrid', 'name': 'Hybrid Beam', 'description': 'High difficulty using beam search.'}
        ]
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        algo_id = data.get('algorithm', 'adaptive')
        width = int(data.get('width', 6))
        height = int(data.get('height', 6))
        seed = data.get('seed')
        if seed is not None:
            seed = int(seed)
        
        params = data.get('params', {})
        
        min_diff = int(data.get('min_difficult', 0))
        min_total = int(data.get('min_total', 0))
        min_ratio = int(data.get('min_ratio', 0))
        max_attempts = int(data.get('max_attempts', 10))

        if algo_id not in ALGORITHMS:
            return jsonify({'error': f'Unknown algorithm: {algo_id}'}), 400
            
        module = ALGORITHMS[algo_id]
        
        best_result = None
        best_score = -1.0
        
        current_seed = seed
        
        for attempt in range(max_attempts):
            # If retrying and using a fixed seed, increment it to explore
            if attempt > 0 and current_seed is not None:
                current_seed += 1
                
            result = module.generate_puzzle(
                width=width,
                height=height,
                word_list_path=WORD_LIST_PATH,
                difficult_list_path=DIFFICULT_LIST_PATH,
                seed=current_seed,
                params=params
            )
            
            # Check constraints
            found = result['found_words']
            difficult = result['difficult_words']
            
            c_total = len(found)
            c_diff = len(difficult)
            
            # Calculate 6+ ratio
            # Convert to list if it's a set to iterate, though iterating set is fine
            w6 = [w for w in found if len(w) >= 6]
            d6 = [w for w in difficult if len(w) >= 6]
            c_ratio = (len(d6) / len(w6) * 100) if w6 else 0
            
            # Check satisfied
            sat_diff = c_diff >= min_diff
            sat_total = c_total >= min_total
            sat_ratio = c_ratio >= min_ratio
            
            if sat_diff and sat_total and sat_ratio:
                # Success
                result['found_words'] = sorted(list(found))
                result['difficult_words'] = sorted(list(difficult))
                result['attempts_taken'] = attempt + 1
                return jsonify(result)
            
            # Score for best effort (sum of completion percentages)
            s_diff = min(1.0, c_diff / min_diff) if min_diff > 0 else 1.0
            s_total = min(1.0, c_total / min_total) if min_total > 0 else 1.0
            s_ratio = min(1.0, c_ratio / min_ratio) if min_ratio > 0 else 1.0
            
            score = s_diff + s_total + s_ratio
            
            if score > best_score:
                best_score = score
                best_result = result
        
        # If loop finishes without perfect match, return best result
        result = best_result
        if result:
            result['found_words'] = sorted(list(result['found_words']))
            result['difficult_words'] = sorted(list(result['difficult_words']))
            result['attempts_taken'] = max_attempts
            return jsonify(result)
            
        return jsonify({'error': 'Generation failed completely'}), 500
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
