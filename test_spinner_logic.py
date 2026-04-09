from spinner_set import SpinnerSet

def test_spinner_min_length():
    print("Testing SpinnerSet._spin_min_word_length for '3x3x3'...")
    results = []
    for _ in range(100):
        res = SpinnerSet._spin_min_word_length('3x3x3')
        results.append(res)
    
    counts = {x: results.count(x) for x in set(results)}
    print(f"Results for '3x3x3': {counts}")
    
    print("\nTesting for '4x4'...")
    results_4x4 = []
    for _ in range(100):
        results_4x4.append(SpinnerSet._spin_min_word_length('4x4'))
    counts_4x4 = {x: results_4x4.count(x) for x in set(results_4x4)}
    print(f"Results for '4x4': {counts_4x4}")

if __name__ == '__main__':
    test_spinner_min_length()
