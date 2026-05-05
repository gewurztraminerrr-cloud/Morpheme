def lcs_bit(char_masks, m, target):
    v = 0
    for char in target:
        m_mask = char_masks.get(char, 0)
        x = m_mask | v
        # This is the bit-parallel LCS algorithm
        v = (x & ((x - ((m_mask << 1) | 1)) ^ x)) # No, this is also wrong.
    return bin(v).count('1')

# Let's use the correct bit-parallel LCS (Hyyrö's)
def lcs_hyyro(char_masks, m, target):
    # This is for strings up to 64 chars
    # char_masks is a dict of bitmasks for each char in source
    # where bit i is 1 if source[i] == char
    V = (1 << m) - 1
    for char in target:
        M = char_masks.get(char, 0)
        X = M | V
        V = (X & (X - ((M << 1) | 1))) # Still not quite right.
    return 0 # ...

# Okay, I'll just use the optimized iterative one. It's safe.
