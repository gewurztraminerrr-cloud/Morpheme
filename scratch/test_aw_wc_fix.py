import sys
from spinner_set import SpinnerSet

print("=== 1. Testing get_factchecked_wc_range ===")
assert SpinnerSet.get_factchecked_wc_range(270, use_added_words=True, dictionary='NWL + AW') == '300-400', "Failed for NWL + AW 270"
assert SpinnerSet.get_factchecked_wc_range(270, use_added_words=True, dictionary='CSW + AW') == '300-400', "Failed for CSW + AW 270"
assert SpinnerSet.get_factchecked_wc_range(50, use_added_words=True, dictionary='NWL + AW') == '300-400', "Failed for NWL + AW 50"
assert SpinnerSet.get_factchecked_wc_range(350, use_added_words=True, dictionary='NWL + AW') == '300-400', "Failed for NWL + AW 350"
assert SpinnerSet.get_factchecked_wc_range(420, use_added_words=True, dictionary='NWL + AW') == '400-500', "Failed for NWL + AW 420"
assert SpinnerSet.get_factchecked_wc_range(530, use_added_words=True, dictionary='NWL + AW') == '500+', "Failed for NWL + AW 530"

# Standard dict must still allow 200-300
assert SpinnerSet.get_factchecked_wc_range(270, use_added_words=False, dictionary='NWL') == '200-300', "Failed for standard NWL 270"
assert SpinnerSet.get_factchecked_wc_range(270, use_added_words=False, dictionary='CSW') == '200-300', "Failed for standard CSW 270"
print("✓ get_factchecked_wc_range assertions passed!")

print("\n=== 2. Testing SpinnerSet.sanitize_params ===")
for d in ['NWL + AW', 'CSW + AW']:
    p = {'dictionary': d, 'word_count_range': '200-300', 'min_word_length': 6, 'difficulty': 'Easy'}
    san = SpinnerSet.sanitize_params(p, board_dimensions='4x6')
    assert san['word_count_range'] in ['300-400', '400-500', '500+'], f"Sanitize failed to clamp {d} 200-300: {san['word_count_range']}"
    assert san['word_count_range'] != '200-300', f"{d} still has 200-300!"
print("✓ sanitize_params assertions passed!")

print("\n=== 3. Testing SpinnerSet._spin_word_count across 100 trials ===")
for d in ['NWL + AW', 'CSW + AW']:
    for _ in range(100):
        wc = SpinnerSet._spin_word_count(d, 6, 'Easy', '4x6', use_added_words=True)
        assert wc in ['300-400', '400-500', '500+'], f"Spin returned forbidden range {wc} for {d}"
print("✓ _spin_word_count 100-trial assertions passed for NWL + AW and CSW + AW!")

print("\n=== 4. Testing BoardGenerator with NWL + AW ===")
from board_generator import BoardGenerator
bg = BoardGenerator()
res = bg.generate_board('4x6', 'RESTORATION', '200-300', 'NWL + AW', 'Normal', 6, 'Easy', timeout=20.0, use_added_words=True)
words = res[1]
print(f"Generated board with NWL + AW (requested 200-300) yielded: {len(words)} words")
assert len(words) >= 300, f"Board count must be >= 300, got {len(words)}"
final_factcheck = SpinnerSet.get_factchecked_wc_range(len(words), use_added_words=True, dictionary='NWL + AW')
print(f"Factchecked range label: {final_factcheck}")
assert final_factcheck in ['300-400', '400-500', '500+']
assert final_factcheck != '200-300'
print("✓ BoardGenerator NWL + AW verification passed!")

print("\nALL TESTS PASSED SUCCESSFULLY!")
