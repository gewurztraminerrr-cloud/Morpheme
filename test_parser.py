
def _parse_word_count_range(word_count_range):
    """A direct copy of the April 29 parser logic"""
    if isinstance(word_count_range, tuple):
        return word_count_range
    if not isinstance(word_count_range, str):
        return (0, 99999)

    # The current logic
    if word_count_range == "50-100":
        return (50, 100)
    elif word_count_range == "100-200":
        return (100, 200)

    if "-" in word_count_range:
        try:
            parts = word_count_range.split("-")
            return (int(parts[0]), int(parts[1])) # This fails if it contains 'Words:'
        except (ValueError, IndexError):
            pass

    return (0, 99999) # Default failure state

# TEST CASES
test_strings = [
    "50-100",           # Should pass
    "Words: 50-100",    # Fails (int('Words: 50'))
    "Range: 100-200"    # Fails
]

for s in test_strings:
    res = _parse_word_count_range(s)
    print(f"Input: '{s}' -> Parsed Range: {res} ({'SUCCESS' if res[0] > 0 else 'FAILURE (Defaults to 0)'})")
