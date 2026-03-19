
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

def simple_g2p(word):
    word = word.upper().strip()
    
    # DEFINITIVE OVERRIDES (Reliable Sources)
    overrides = {
        "BARYE": "BAIR-EE",
        "PORTAMENTI": "PORE-TAH-MEN-TEE",
        "LINALOOL": "LIH-NAH-LOW-WOLL",
        "CARATE": "KAH-RAH-TEE",
        "PLEIAD": "PLEE-ADD",
        "MALACHITE": "MAL-UH-KITE",
        "SECERNENT": "SIH-SUR-NUNT"
    }
    if word in overrides: return overrides[word]

    # Rule-based G2P Refinement
    res = word
    
    # Pre-processing Clusters
    res = res.replace('TION', 'SHUN-')
    res = res.replace('PH', 'F')
    res = res.replace('QU', 'KW')
    res = res.replace('CK', 'K')
    
    # Common Scrabble word patterns
    if res.endswith('ITE'): res = res[:-3] + 'EYET'
    if res.endswith('ATE'): res = res[:-3] + 'AYT'
    if res.endswith('OUS'): res = res[:-3] + 'UHS'
    if res.endswith('ISM'): res = res[:-3] + 'IH-ZUM'
    if res.endswith('OLOGY'): res = res[:-5] + 'OL-UH-JEE'

    # Handle C and G
    res = re.sub(r'C([EIY])', r'S\1', res)
    res = re.sub(r'G([EIY])', r'J\1', res)
    res = res.replace('CH', 'CH') # Default
    # Specific Greek-root CH as K (very common in minerals/science)
    if 'CH' in res:
        if any(x in res for x in ["MALACH", "ARCHI", "MECHAN", "CHLOR", "CHRON", "ECH"]):
            res = res.replace('CH', 'K')

    res = res.replace('C', 'K')
    
    # Silent E (very basic)
    if res.endswith('E') and len(res) > 3:
        # Check if preceded by V-C
        if re.search(r'[AEIOUY][B-DF-HJ-NP-TV-Z]E$', word):
            # This logic is usually better handled by cluster detection
            pass

    # Vowel Mappings (English sounds)
    mapping = [
        (r'EE', 'EE'), (r'EA', 'EE'), (r'IE', 'EE'), (r'EI', 'EE'),
        (r'AI', 'AY'), (r'AY', 'AY'), (r'OA', 'OH'), (r'OO', 'OO'),
        (r'OU', 'OW'), (r'OW', 'OW'), (r'OI', 'OY'), (r'OY', 'OY'),
        (r'AU', 'AW'), (r'AW', 'AW'),
        (r'AR', 'AR'), (r'OR', 'ORE'), (r'ER', 'ER'), (r'IR', 'ER'), (r'UR', 'ER'),
        # Lone vowels
        (r'A(?=[B-DF-HJ-NP-TV-Z]E$)', 'AY'), # cake
        (r'A', 'AH'),
        (r'E(?=[B-DF-HJ-NP-TV-Z]E$)', 'EE'), # mete
        (r'E', 'EH'),
        (r'I(?=[B-DF-HJ-NP-TV-Z]E$)', 'EYE'), # kite
        (r'I', 'IH'),
        (r'O(?=[B-DF-HJ-NP-TV-Z]E$)', 'OH'), # note
        (r'O', 'OH'),
        (r'U(?=[B-DF-HJ-NP-TV-Z]E$)', 'OO'), # cute
        (r'U', 'UH')
    ]
    
    # Protected tokens to avoid double-processing
    tokens = []
    
    # Syllable splitting (crude but better)
    # 1. Identify vowels
    v_regex = r'(EE|EA|IE|EI|AY|AI|OH|OA|OO|OW|OU|OY|OI|AW|AU|AR|ORE|ER|IR|UR|EYE|AY|AH|EH|IH|OH|UH|EH)'
    
    # We'll just apply the core vowel mappings and then hyphenate
    for pat, repl in mapping:
        # Use a placeholder to protect
        res = re.sub(pat, " " + repl + " ", res)
    
    res = res.replace("  ", " ").strip()
    parts = res.split(" ")
    
    # Join consonants into the previous or next syllable
    final_parts = []
    curr = ""
    for p in parts:
        if not p: continue
        if any(v in p for v in ['AH', 'EH', 'IH', 'OH', 'UH', 'EE', 'AY', 'OW', 'OY', 'AW', 'AR', 'ER', 'ORE', 'EYE', 'OO']):
            final_parts.append(curr + p)
            curr = ""
        else:
            curr += p
    if curr:
        if final_parts: final_parts[-1] += curr
        else: final_parts.append(curr)
        
    # Schwa logic: Unstressed AH often becomes UH
    # If more than 2 syllables, change internal AH to UH
    if len(final_parts) > 2:
        for i in range(1, len(final_parts) - 1):
             if final_parts[i] == "AH": final_parts[i] = "UH"
             # If part is like LAH, change to LUH? No, too risky.
             
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
            pron = arpa_to_respell(cmu[word]) if word in cmu else simple_g2p(word)
            if pron:
                pron = re.sub(r'-+', '-', pron).strip('-').upper()
                f.write(f"{word} - {pron}\n")

    print(f"Saved {len(words)} pronunciations to {output_path}")

if __name__ == "__main__":
    main()
