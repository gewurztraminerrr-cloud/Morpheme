import nltk
import re
from g2p_en import G2p
import os

g2p = G2p()
cmu = nltk.corpus.cmudict.dict()

# -------------------------------------------------------
# R-colored vowel pairs (vowel phoneme + R -> respelling)
# These must be merged BEFORE syllabification.
# -------------------------------------------------------
RHOTIC_MAP = {
    ('AA', 'R'): 'AR',   # far, car
    ('AE', 'R'): 'AIR',  # care, rare
    ('AH', 'R'): 'UR',   # fur, blur
    ('AO', 'R'): 'OR',   # more, store
    ('AW', 'R'): 'OWR',  # hour (before r)
    ('AY', 'R'): 'IGHR', # fire
    ('EH', 'R'): 'AIR',  # bare, there, care  <-- BARYE fix
    ('EY', 'R'): 'AIR',  # prayer
    ('IH', 'R'): 'EAR',  # near
    ('IY', 'R'): 'EAR',  # fear, clear
    ('OW', 'R'): 'OR',   # more
    ('OY', 'R'): 'OYR',  # loyal
    ('UH', 'R'): 'UR',   # lure
    ('UW', 'R'): 'OOR',  # poor
}

# Standard vowel map (non-rhotic)
VOWEL_MAP = {
    'AA': 'AH', 'AE': 'A', 'AH': 'UH', 'AO': 'AW', 'AW': 'OW', 'AY': 'IGH',
    'EH': 'E',  'ER': 'UR', 'EY': 'AY', 'IH': 'IH', 'IY': 'EE', 'OW': 'OH',
    'OY': 'OY', 'UH': 'UU', 'UW': 'OO'
}

VOWEL_MAP_UNSTRESSED = {
    'AA': 'UH', 'AE': 'UH', 'AH': 'UH', 'AO': 'UH', 'AW': 'OW', 'AY': 'IGH',
    'EH': 'UH', 'ER': 'UR', 'EY': 'AY', 'IH': 'IH', 'IY': 'EE', 'OW': 'OH',
    'OY': 'OY', 'UH': 'UU', 'UW': 'OO'
}

# Manual overrides for problematic g2p/cmudict entries
CUSTOM_FIXES = {
    'LINALOOL': 'LIH-NAH-LOW-WOLL',
    'LINALOOLS': 'LIH-NAH-LOW-WOLLS',
    'PORTAMENTI': 'POR-TUH-MEN-TEE',
    'PORTAMENTO': 'POR-TUH-MEN-TOH',
    'PORTAMENTOS': 'POR-TUH-MEN-TOHS',
}

CONS_MAP = {
    'B': 'B', 'CH': 'CH', 'D': 'D', 'DH': 'TH', 'F': 'F', 'G': 'G', 'HH': 'H',
    'JH': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'NG': 'NG', 'P': 'P',
    'R': 'R', 'S': 'S', 'SH': 'SH', 'T': 'T', 'TH': 'TH', 'V': 'V', 'W': 'W',
    'Y': 'Y', 'Z': 'Z', 'ZH': 'ZH'
}


def strip_stress(p):
    return re.sub(r'\d+', '', p)


def get_stress(p):
    m = re.search(r'\d', p)
    return int(m.group()) if m else 1


def is_vowel_base(base):
    return base in VOWEL_MAP


def merge_rhotics(phonemes):
    """
    Pre-process phoneme list to merge vowel+R pairs into a single token
    that encodes the rhotic vowel sound.
    Returns a list of items that are either original phoneme strings
    or ('RHOTIC', respelling) tuples.
    """
    result = []
    i = 0
    while i < len(phonemes):
        base = strip_stress(phonemes[i])
        # Look ahead for R
        if is_vowel_base(base) and i + 1 < len(phonemes) and strip_stress(phonemes[i + 1]) == 'R':
            rhotic_resp = RHOTIC_MAP.get((base, 'R'))
            if rhotic_resp:
                result.append(('RHOTIC', rhotic_resp))
                i += 2  # skip the R too
                continue
        result.append(phonemes[i])
        i += 1
    return result


def is_vowel_token(token):
    if isinstance(token, tuple):
        return True  # rhotic tokens are always vowel-based
    return is_vowel_base(strip_stress(token))


def syllabify(tokens):
    """
    Split a list of tokens (strings or RHOTIC tuples) into syllables
    using a vowel-nucleus grouping approach.
    """
    vowel_indices = [i for i, t in enumerate(tokens) if is_vowel_token(t)]
    if not vowel_indices:
        return [tokens]

    syllables = []
    start = 0
    for k in range(len(vowel_indices)):
        if k == len(vowel_indices) - 1:
            syllables.append(tokens[start:])
        else:
            curr_v = vowel_indices[k]
            next_v = vowel_indices[k + 1]
            gap = next_v - curr_v - 1
            # One consonant between vowels: attach to next syllable (V-CV rule)
            # Two+ consonants: split in the middle (VC-CV rule)
            if gap <= 1:
                split = curr_v + 1
            else:
                split = curr_v + 1 + gap // 2
            syllables.append(tokens[start:split])
            start = split
    return syllables


def format_token(token):
    if isinstance(token, tuple):
        # ('RHOTIC', 'AIR') etc
        return token[1]
    p = token
    base = strip_stress(p)
    stress = get_stress(p)
    if base == 'ER':
        return 'UR'
    if is_vowel_base(base):
        if stress == 0:
            return VOWEL_MAP_UNSTRESSED.get(base, base)
        else:
            return VOWEL_MAP.get(base, base)
    return CONS_MAP.get(base, base)


def phonemes_to_respelling(phonemes):
    tokens = merge_rhotics(phonemes)
    sylls = syllabify(tokens)
    parts = []
    for s in sylls:
        syll_str = ''.join(format_token(t) for t in s)
        if syll_str:
            parts.append(syll_str)
    return '-'.join(parts)


def get_phonetic_respelling(word):
    wl = word.upper()
    if wl in CUSTOM_FIXES:
        return CUSTOM_FIXES[wl]
    
    wl_low = wl.lower()
    if wl_low in cmu:
        phonemes = cmu[wl_low][0]
    else:
        phonemes = [p for p in g2p(wl_low) if p != ' ']
    return phonemes_to_respelling(phonemes)


def generate_pronunciations_file(input_path, output_path):
    with open(input_path, 'r') as f:
        words = [line.strip().upper() for line in f if 3 <= len(line.strip()) <= 10]
    words.sort()

    existing = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                parts = line.split()
                if parts:
                    existing.add(parts[0].upper())

    to_process = [w for w in words if w not in existing]
    print(f"Total: {len(words)}, Existing: {len(existing)}, To Process: {len(to_process)}")

    batch_size = 1000
    with open(output_path, 'a') as out_f:
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i + batch_size]
            for w in batch:
                respelling = get_phonetic_respelling(w)
                out_f.write(f"{w} {respelling}\n")
            out_f.flush()
            print(f"Processed up to {batch[-1]} ({min(i+batch_size, len(to_process))}/{len(to_process)})")


# Quick test
if __name__ == '__main__':
    tests = [
        ('SECERNENT', 'SE-SUR-UHNT'),
        ('BARYE', 'BARE-EE'),
        ('HELLO', 'HUH-LOH'),
        ('CARE', 'KAIR'),
        ('BARE', 'BAIR'),
        ('FEAR', 'FEAR'),
        ('MORE', 'MOR'),
        ('FAR', 'FAR'),
        ('PHONETIC', 'FUH-NE-TUHK'),
    ]
    for w, expected in tests:
        result = get_phonetic_respelling(w)
        status = '✓' if result == expected else f'✗ (expected {expected})'
        print(f"{w}: {result}  {status}")
