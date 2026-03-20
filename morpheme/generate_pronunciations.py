
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

def load_moby(path):
    """Load Moby Pronunciator II format (Word /Phonemes/)."""
    moby = {}
    if not os.path.exists(path):
        return moby
    
    # Mapping for Moby ASCII phonemes to our respelled vowels
    # Refining to use fewer 'H's where possible (e.g. A -> A instead of AH)
    VOWELS = {
        'A': 'A', '@': 'UH', 'E': 'EH', 'I': 'IH', 'O': 'AW', 'U': 'UU',
        'i': 'EE', 'u': 'OO', 'eI': 'AY', 'oU': 'OH', 'aI': 'EYE', 'aU': 'OW', 'OI': 'OY',
    }
    CONSONANTS = {
        'S': 'SH', 'Z': 'ZH', 'T': 'TH', 'D': 'TH', 'C': 'CH', 'J': 'J', 'N': 'NG',
        'b':'B', 'd':'D', 'f':'F', 'g':'G', 'h':'H', 'k':'K', 'l':'L', 'm':'M', 
        'n':'N', 'p':'P', 'r':'R', 's':'S', 't':'T', 'v':'V', 'w':'W', 'y':'Y', 'z':'Z'
    }

    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line or '/' not in line: continue
            
            # Split "Word /Phonemes/"
            parts = line.split('/', 1)
            word = re.sub(r'[^a-zA-Z]', '', parts[0]).upper()
            phones_raw = parts[1].strip('/')
            
            # Remove stress and breaks: ' , _
            phones_raw = re.sub(r"[' ,_]", "", phones_raw)
            
            # Tokenize Moby (greedy match double-char eI, oU, etc)
            res = []
            it = iter(range(len(phones_raw)))
            for i in it:
                if i + 1 < len(phones_raw) and phones_raw[i:i+2] in VOWELS:
                    res.append(VOWELS[phones_raw[i:i+2]])
                    next(it, None)
                elif phones_raw[i] in VOWELS:
                    res.append(VOWELS[phones_raw[i]])
                elif phones_raw[i] in CONSONANTS:
                    res.append(CONSONANTS[phones_raw[i]])
            
            if res:
                vowel_set = set(VOWELS.values()) | {'AH', 'AR', 'AIR', 'ORE'}
                syllables = []
                curr = ""
                for p in res:
                    if p in vowel_set:
                        syllables.append(curr + p)
                        curr = ""
                    else:
                        curr += p
                if curr:
                    if syllables: syllables[-1] += curr
                    else: syllables.append(curr)
                
                moby[word] = "-".join(syllables)
    return moby

def load_wiktionary(path):
    """Load wiktionary.json (Word: [IPA, ...])."""
    import json
    wikidict = {}
    if not os.path.exists(path):
        return wikidict
    
    # Mapping for IPA tokens to Respelled Sounds
    IPA_MAPPING = {
        'ɑ': 'AH', 'æ': 'A', 'ʌ': 'UH', 'ɔ': 'AW', 'ə': 'UH', 'aɪ': 'EYE',
        'ɛ': 'EH', 'ɜ': 'ER', 'eɪ': 'AY', 'ɪ': 'IH', 'i': 'EE', 'oʊ': 'OH',
        'ʊ': 'UU', 'u': 'OO', 'aʊ': 'OW', 'ɔɪ': 'OY', 'ɒ': 'AW',
        'ʃ': 'SH', 'ʒ': 'ZH', 'θ': 'TH', 'ð': 'TH', 'tʃ': 'CH', 'dʒ': 'J', 'ŋ': 'NG',
        'ɹ': 'R', 'j': 'Y', 'w': 'W', 'b': 'B', 'd': 'D', 'f': 'F', 'g': 'G', 
        'h': 'H', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'p': 'P', 's': 'S', 
        't': 'T', 'v': 'V', 'z': 'Z'
    }

    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            return wikidict

        for word, p_list in data.items():
            if not p_list: continue
            raw_ipa = p_list[0]
            tokens = raw_ipa.split()
            
            res = []
            for t in tokens:
                t = re.sub(r'[ˈˌ.ː]', '', t)
                if t in IPA_MAPPING:
                    res.append(IPA_MAPPING[t])
                elif len(t) > 1: # Try composite
                    if t in IPA_MAPPING: res.append(IPA_MAPPING[t])
            
            if res:
                vowel_set = {'AH', 'A', 'UH', 'AW', 'EYE', 'EH', 'ER', 'AY', 'IH', 'EE', 'OH', 'UU', 'OO', 'OW', 'OY'}
                parts = []
                curr = ""
                for p in res:
                    if p in vowel_set:
                        parts.append(curr + p)
                        curr = ""
                    else:
                        curr += p
                if curr:
                    if parts: parts[-1] += curr
                    else: parts.append(curr)
                wikidict[word.upper()] = "-".join(parts)
    return wikidict

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

OVERRIDES = {
    "CHAMOIS": "SHAMMY",
    "WARISON": "WARE-IH-SUN",
    "CALCANEA": "KAL-KAY-NEE-AH",
    "ORCINOL": "ORE-SIN-AWL",
    "OVERKEEN": "OH-VUR-KEEN",
    "CHICKPEA": "CHIK-PEE",
    "BARYE": "BAIR-EE",
    "PORTAMENTI": "PORE-TAH-MEN-TEE",
    "LINALOOL": "LIH-NAH-LOW-WOLL",
    "CARATE": "KAH-RAH-TEE",
    "PLEIAD": "PLEE-ADD",
    "MALACHITE": "MAL-UH-KITE",
    "SECERNENT": "SIH-SUR-NUNT",
    "AA": "AH-AH",
    "AAH": "AH",
    "QI": "CHEE",
    "ZA": "ZAH",
    "XU": "SOO",
    "JO": "JOH",
    "OE": "OH",
    "KA": "KAH"
}

def main():
    base_dir = "/Users/jeffbabiak/morpheme"
    dict_dir = os.path.join(base_dir, "dictionaries")
    cmu_path = os.path.join(dict_dir, "cmudict.txt")
    
    print("Loading CMU Dictionary...")
    cmu = load_cmudict(cmu_path)
    
    moby_path = os.path.join(dict_dir, "moby.txt")
    print("Loading Moby Dictionary...")
    moby = load_moby(moby_path)
    
    words = set()
    for d in ["NWL.txt", "CSW.txt"]:
        path = os.path.join(dict_dir, d)
        if not os.path.exists(path): continue
        with open(path, 'r') as f:
            for line in f:
                w = line.strip().upper()
                if 2 <= len(w) <= 15:
                    words.add(w)
    
    output_path = os.path.join(dict_dir, "pronunciations.txt")
    
    wiki_path = os.path.join(dict_dir, "wiktionary.json")
    print("Loading Wiktionary...")
    wiki = load_wiktionary(wiki_path)
    
    with open(output_path, 'w') as f:
        for word in sorted(words):
            pron = ""
            if word in OVERRIDES:
                pron = OVERRIDES[word]
            elif word in cmu:
                pron = arpa_to_respell(cmu[word])
            elif word in moby:
                pron = moby[word]
            elif word in wiki:
                pron = wiki[word]
            
            # NOTE: Removed inflected and compound guesses per user request:
            # "DO NOT MAKE UP A PRONUNCIATION IF YOU DON’T KNOW IT. Let it be blank... if you don’t know it."
            
            if pron:
                pron = re.sub(r'-+', '-', pron).strip('-').upper()
                f.write(f"{word} - {pron}\n")

    print(f"Saved {len(words)} words, including Moby coverage, to {output_path}")

if __name__ == "__main__":
    main()
