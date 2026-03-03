import re
from g2p_en import G2p

g2p = G2p()

VOWELS_MAP = {
    'AA': 'AH', 'AE': 'A', 'AH': 'UH', 'AO': 'AW', 'AW': 'OW', 'AY': 'IGH',
    'EH': 'E', 'ER': 'UR', 'EY': 'AY', 'IH': 'I', 'IY': 'EE', 'OW': 'OH',
    'OY': 'OY', 'UH': 'UU', 'UW': 'OO'
}

def get_phonemes(word):
    ph_list = g2p(word)
    return [p for p in ph_list if p.strip() and p != ' ']

def chunk_syllables(phonemes):
    is_vowel = lambda p: any(v in p for v in VOWELS_MAP)
    structure = ['V' if is_vowel(p) else 'C' for p in phonemes]
    
    syllable_breaks = [False] * len(phonemes)
    last_v_idx = -1
    for i, t in enumerate(structure):
        if t == 'V':
            if last_v_idx != -1:
                dist = i - last_v_idx
                if dist == 1:
                    syllable_breaks[i] = True
                elif dist == 2:
                    syllable_breaks[last_v_idx + 1] = True
                elif dist == 3:
                    syllable_breaks[last_v_idx + 2] = True
                else:
                    syllable_breaks[last_v_idx + 2] = True
            last_v_idx = i
            
    sylls = []
    curr = []
    for i, p in enumerate(phonemes):
        if syllable_breaks[i] and curr:
            sylls.append(curr)
            curr = []
        curr.append(p)
    if curr:
        sylls.append(curr)
        
    # fallback if no vowels
    if not any(t == 'V' for t in structure):
        return [phonemes]
        
    return sylls

def phonetic_respell(word):
    phonemes = get_phonemes(word)
    sylls = chunk_syllables(phonemes)
    res = []
    for s in sylls:
        s_res = ""
        for p in s:
            p_strip = re.sub(r'\d+', '', p)
            if p_strip in VOWELS_MAP:
                s_res += VOWELS_MAP[p_strip]
            else:
                s_res += p_strip
        res.append(s_res)
    return '-'.join(res)

for w in ['SECERNENT', 'HELLO', 'WORLD', 'PHONETIC']:
    print(w, phonetic_respell(w))
