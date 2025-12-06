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
        
        if algo_id not in ALGORITHMS:
            return jsonify({'error': f'Unknown algorithm: {algo_id}'}), 400
            
        module = ALGORITHMS[algo_id]
        
        # Normalize params based on algorithm
        # This allows the frontend to send generic params and we map them here if needed
        # For now, we pass them through
        
        result = module.generate_puzzle(
            width=width,
            height=height,
            word_list_path=WORD_LIST_PATH,
            difficult_list_path=DIFFICULT_LIST_PATH,
            seed=seed,
            params=params
        )
        
        # Convert sets to lists for JSON serialization
        result['found_words'] = sorted(list(result['found_words']))
        result['difficult_words'] = sorted(list(result['difficult_words']))
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
