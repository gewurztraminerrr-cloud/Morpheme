import sys
import os
import time

# Add current path
sys.path.append('/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme')

from board_generator import BoardGenerator

def test_rarer_weights():
    gen = BoardGenerator()
    
    # 4x4 - Hard (60%+ uniqueness) - 3LM - NWL
    dimensions = '4x4'
    bonus_word = 'MORPHEME' 
    dictionary = 'NWL'
    
    # Rare-heavy weights (Hypothetically used by Java for Hard?)
    # A=10, B=20, C=20, D=20, E=10, F=20, G=20, H=20, I=10, J=20, K=20, L=20, M=20, N=20, O=10, P=20, Q=20, R=20, S=20, T=20, U=10, V=20, W=20, X=20, Y=20, Z=20
    RARE_WEIGHTS = [10, 20, 20, 20, 10, 20, 20, 20, 10, 20, 20, 20, 20, 20, 10, 20, 20, 20, 20, 20, 10, 20, 20, 20, 20, 20]
    
    max_ratio = 0
    total_attempts = 100
    
    results = []
    for i in range(total_attempts):
        board = gen._create_normal_board(4, 4, RARE_WEIGHTS)
        gen._embed_bonus_word(board, bonus_word)
        
        words_dict = gen._solve_board(board, dictionary, (0, 99999), 3)
        if words_dict:
            found = list(words_dict.keys())
            unique_set = gen._get_difficulty_set(dictionary)
            count_total = len(found)
            count_unique = sum(1 for w in found if w.upper() in unique_set)
            ratio = count_unique / count_total if count_total > 0 else 0
            
            if ratio > max_ratio:
                max_ratio = ratio
                print(f"Attempt {i}: Ratio {ratio:.2%} ({count_total} words)")
            
            if ratio >= 0.60 and count_total >= 50:
                print(f"Goal met (60% + 50 words) at attempt {i}!")
                break
                
    print(f"Max Uniqueness: {max_ratio:.2%}")

if __name__ == "__main__":
    test_rarer_weights()
