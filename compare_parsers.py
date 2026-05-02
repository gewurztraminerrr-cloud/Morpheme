
import re

def parse_with_regex(s):
    nums = re.findall(r'\d+', s)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    return (0, 0)

def parse_with_split(s):
    # What you suggested: split by space
    parts = s.split()
    clean = parts[-1] if parts else s
    # Then split by dash
    if "-" in clean:
        try:
            p = clean.split("-")
            return (int(p[0].strip()), int(p[1].strip()))
        except:
            pass
    return (0, 0)

test_cases = [
    "Words: 50-100",  # Success for both
    "Words:50-100",   # FAIL for split, SUCCESS for regex
    "Range: 100-200", # Success for both
    "50 - 100"        # SUCCESS for regex, MAYBE fail for split depending on logic
]

print(f"{'Input':<20} | {'Split Result':<15} | {'Regex Result':<15}")
print("-" * 55)
for s in test_cases:
    r_split = parse_with_split(s)
    r_regex = parse_with_regex(s)
    print(f"{s:<20} | {str(r_split):<15} | {str(r_regex):<15}")
