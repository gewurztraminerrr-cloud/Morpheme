
import os
import re

def load_cmudict(path):
    cmudict = {}
    if not os.path.exists(path):
        return cmudict
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.startswith(';;;'):
                continue
            parts = line.strip().split('  ')
            if len(parts) == 2:
                word = parts[0].split('(')[0].upper()
                phonemes = parts[1].split(' ')
                phonemes = [re.sub(r'\d', '', p) for p in phonemes]
                cmudict[word] = phonemes
    return cmudict

def arpa_to_respell(phonemes):
    mapping = {
        'AA': 'AH', 'AE': 'A', 'AH': 'UH', 'AO': 'AW', 'AW': 'OW', 'AY': 'EYE',
        'EH': 'EH', 'ER': 'ER', 'EY': 'AY', 'IH': 'IH', 'IY': 'EE', 'OW': 'OH',
        'OY': 'OY', 'UH': 'UU', 'UW': 'OO', 'B': 'B', 'CH': 'CH', 'D': 'D',
        'DH': 'TH', 'F': 'F', 'G': 'G', 'HH': 'H', 'JH': 'J', 'K': 'K',
        'L': 'L', 'M': 'M', 'N': 'N', 'NG': 'NG', 'P': 'P', 'R': 'R',
        'S': 'S', 'SH': 'SH', 'T': 'T', 'TH': 'TH', 'V': 'V', 'W': 'W',
        'Y': 'Y', 'Z': 'Z', 'ZH': 'ZH'
    }
    
    respelled = []
    i = 0
    while i < len(phonemes):
        p = phonemes[i]
        next_p = phonemes[i+1] if i + 1 < len(phonemes) else None
        
        if p == 'AO' and next_p == 'R':
             respelled.append('ORE')
             i += 1
        elif p == 'EH' and next_p == 'R':
             respelled.append('AIR')
             i += 1
        elif p == 'AE' and next_p == 'R':
             respelled.append('AIR')
             i += 1
        elif p == 'AA' and next_p == 'R':
             respelled.append('AR')
             i += 1
        else:
             respelled.append(mapping.get(p, p))
        i += 1
    
    # Syllabic joining logic for ARPAbet
    vowels = {'AH', 'A', 'AW', 'OW', 'EYE', 'EH', 'ER', 'AY', 'IH', 'EE', 'OH', 'OY', 'UU', 'OO', 'AIR', 'ORE', 'AR', 'UH'}
    
    parts = []
    curr = ""
    for p in respelled:
        if p in vowels:
            parts.append(curr + p)
            curr = ""
        else:
            curr += p
    if curr:
        if parts: parts[-1] += curr
        else: parts.append(curr)
        
    return "-".join(parts)

def get_compound_pron(word, cmudict):
    """Try to find the pronunciation by splitting into common words."""
    if len(word) < 6: return None
    
    # Try every possible split point (e.g. CHICK + PEA)
    for i in range(3, len(word) - 2):
        left, right = word[:i], word[i:]
        if left in cmudict and right in cmudict:
            left_respell = arpa_to_respell(cmudict[left])
            right_respell = arpa_to_respell(cmudict[right])
            if left_respell and right_respell:
                return f"{left_respell}-{right_respell}"
    
    return None

def simple_g2p(word):
    """Cautious rule-based fallback for words not in CMU or compounds."""
    word = word.upper().strip()
    
    # PRECISE OVERRIDES (Requested by User and Verified)
    overrides = {
        "ORCINOL": "ORE-SIN-AWL",
        "OVERKEEN": "OH-VUR-KEEN",
        "CHICKPEA": "CHIK-PEE",
        "CALCANEA": "KAL-KAY-NEE-AH",
        "BARYE": "BAIR-EE",
        "PORTAMENTI": "PORE-TAH-MEN-TEE",
        "LINALOOL": "LIH-NAH-LOW-WOLL",
        "CARATE": "KAH-RAH-TEE",
        "PLEIAD": "PLEE-ADD",
        "MALACHITE": "MAL-UH-KITE",
        "SECERNENT": "SIH-SUR-NUNT"
    }
    if word in overrides: return overrides[word]

    # Don't guess for very long or very rare words not in source
    if len(word) > 11: return ""

    # Rule-based G2P (Single pass or protected)
    res = word
    
    # 1. Protect Clues and Phonemes
    res = res.replace('PH', ' F ')
    res = res.replace('QU', ' KW ')
    res = res.replace('TION', ' SHUN ')
    res = res.replace('CK', ' K ')
    res = res.replace('CH', ' CH ')
    res = res.replace('SH', ' SH ')
    res = res.replace('TH', ' TH ')
    
    # Soft C/G (Before vowels)
    res = re.sub(r'C([EIY])', r' S \1', res)
    res = re.sub(r'G([EIY])', r' J \1', res)
    res = res.replace('C', ' K ')
    res = res.replace('G', ' G ')

    # 2. Vowel Clusters with precise respelling
    vowel_mappings = [
        (r'AI|AY', ' AY '), (r'EE|EA|IE|EI', ' EE '),
        (r'OA|OW|OE', ' OH '), (r'OO', ' OO '),
        (r'OU|OW', ' OW '), (r'OI|OY', ' OY '),
        (r'AU|AW', ' AW '), (r'AR', ' AR '), 
        (r'ER|IR|UR|OR(?=[B-DF-HJ-NP-TV-Z])', ' ER '),
        (r'OR', ' ORE '),
        (r'A(?=[B-DF-HJ-NP-TV-Z]E)', ' AY '), # cake
        (r'I(?=[B-DF-HJ-NP-TV-Z]E)', ' EYE '), # kite
        (r'O(?=[B-DF-HJ-NP-TV-Z]E)', ' OH '), # note
        (r'U(?=[B-DF-HJ-NP-TV-Z]E)', ' OO '), # cute
        # Lone vowels (cautious, avoid H overuse)
        (r'A$', ' AH '), (r'A', ' A '), 
        (r'E$', ' EE '), (r'E', ' EH '),
        (r'I', ' IH '),
        (r'O', ' OH '),
        (r'U', ' UH ')
    ]
    
    for pat, repl in vowel_mappings:
        res = re.sub(pat, repl, res)
    
    # 3. Syllabification
    parts = [p.strip() for p in res.split() if p.strip()]
    
    vowels = {'AH', 'A', 'AW', 'OW', 'EYE', 'EH', 'ER', 'AY', 'IH', 'EE', 'OH', 'OY', 'UU', 'OO', 'AIR', 'ORE', 'AR', 'UH'}
    
    final_parts = []
    curr = ""
    for p in parts:
        if any(v in p for v in vowels):
            final_parts.append(curr + p)
            curr = ""
        else:
            curr += p
    if curr:
        if final_parts: final_parts[-1] += curr
        else: final_parts.append(curr)
        
    if not final_parts: return ""
    return "-".join(final_parts)

def main():
    base_dir = "/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme"
    dict_dir = os.path.join(base_dir, "dictionaries")
    cmu_path = os.path.join(dict_dir, "cmudict.txt")
    
    print("Loading CMU Dictionary...")
    cmu = load_cmudict(cmu_path)
    
    words = set()
    for d in ["NWL.txt", "CSW.txt"]:
        path = os.path.join(dict_dir, d)
        if not os.path.exists(path): continue
        with open(path, 'r') as f:
            for line in f:
                w = line.strip().upper()
                if 3 <= len(w) <= 10:
                    words.add(w)
    
    output_path = os.path.join(dict_dir, "pronunciations.txt")
    
    with open(output_path, 'w') as f:
        for word in sorted(words):
            pron = ""
            if word in cmu:
                pron = arpa_to_respell(cmu[word])
            else:
                # Try common compounds (e.g. CHICKPEA)
                pron = get_compound_pron(word, cmu)
                if not pron:
                    # Fallback to cautious G2P
                    pron = simple_g2p(word)
            
            if pron:
                pron = re.sub(r'-+', '-', pron).strip('-').upper()
                f.write(f"{word} - {pron}\n")

    print(f"Saved pronunciations to {output_path}")

if __name__ == "__main__":
    main()
