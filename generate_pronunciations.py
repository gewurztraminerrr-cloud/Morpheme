import nltk
import re
from g2p_en import G2p
import pyphen
import sys
import os

dic = pyphen.Pyphen(lang='en_US')
g2p = G2p()
cmu = nltk.corpus.cmudict.dict()

VOWEL_MAP = {
    # Stressed
    'AA1': 'AH', 'AE1': 'A', 'AH1': 'UH', 'AO1': 'AW', 'AW1': 'OW', 'AY1': 'IGH',
    'EH1': 'E', 'ER1': 'UR', 'EY1': 'AY', 'IH1': 'I', 'IY1': 'EE', 'OW1': 'OH',
    'OY1': 'OY', 'UH1': 'UU', 'UW1': 'OO',

    # Secondary stress
    'AA2': 'AH', 'AE2': 'A', 'AH2': 'UH', 'AO2': 'AW', 'AW2': 'OW', 'AY2': 'IGH',
    'EH2': 'E', 'ER2': 'UR', 'EY2': 'AY', 'IH2': 'I', 'IY2': 'EE', 'OW2': 'OH',
    'OY2': 'OY', 'UH2': 'UU', 'UW2': 'OO',

    # Unstressed (Schwa and variants)
    'AA0': 'UH', 'AE0': 'UH', 'AH0': 'UH', 'AO0': 'UH', 'AW0': 'OW', 'AY0': 'IGH',
    'EH0': 'UH', 'ER0': 'UR', 'EY0': 'AY', 'IH0': 'UH', 'IY0': 'EE', 'OW0': 'OH',
    'OY0': 'OY', 'UH0': 'UU', 'UW0': 'OO'
}

for k, v in list(VOWEL_MAP.items()):
    no_num = re.sub(r'\d+', '', k)
    if no_num not in VOWEL_MAP:
        VOWEL_MAP[no_num] = v

CONS_MAP = {
    'B': 'B', 'CH': 'CH', 'D': 'D', 'DH': 'TH', 'F': 'F', 'G': 'G', 'HH': 'H',
    'JH': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'NG': 'NG', 'P': 'P',
    'R': 'R', 'S': 'S', 'SH': 'SH', 'T': 'T', 'TH': 'TH', 'V': 'V', 'W': 'W',
    'Y': 'Y', 'Z': 'Z', 'ZH': 'ZH'
}

def is_vowel(phoneme):
    return re.sub(r'\d+', '', phoneme) in VOWEL_MAP

def align_phonemes_to_syllables(word, phonemes):
    vowel_indices = [i for i, p in enumerate(phonemes) if is_vowel(p)]
    if not vowel_indices:
        return [phonemes]
    
    syllables = []
    start = 0
    for i in range(len(vowel_indices)):
        if i == len(vowel_indices) - 1:
            syllables.append(phonemes[start:])
        else:
            curr_v_idx = vowel_indices[i]
            next_v_idx = vowel_indices[i+1]
            split_idx = curr_v_idx + 1 + (next_v_idx - curr_v_idx - 1) // 2
            syllables.append(phonemes[start:split_idx])
            start = split_idx
    return syllables

def format_syllable(syllable):
    res = ""
    for p in syllable:
        base_p = re.sub(r'\d+', '', p)
        if p in VOWEL_MAP:
            res += VOWEL_MAP[p]
        elif base_p in VOWEL_MAP:
             res += VOWEL_MAP[base_p]
        elif p in CONS_MAP:
            res += CONS_MAP[p]
    return res

def get_phonetic_respelling(word):
    word = word.lower()
    
    # Specific overrides
    if word == 'secernent':
        return 'SUH-SUR-NUNT'
        
    if word in cmu:
        phonemes = cmu[word][0]
    else:
        phonemes = [p for p in g2p(word) if p != ' ']
        
    sylls = align_phonemes_to_syllables(word, phonemes)
    
    respelled_sylls = [format_syllable(s) for s in sylls]
    return "-".join(respelled_sylls).strip("-")

def generate_pronunciations_file(input_path, output_path):
    # Load all words
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
    
    print(f"Total words: {len(words)}, Existing: {len(existing)}, To Process: {len(to_process)}")
    
    batch_size = 1000
    with open(output_path, 'a') as out_f:
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i+batch_size]
            
            for w in batch:
                respelling = get_phonetic_respelling(w)
                out_f.write(f"{w} {respelling}\n")
                
            out_f.flush()
            print(f"Processed up to {batch[-1]} ({min(i+batch_size, len(to_process))}/{len(to_process)})")

if __name__ == '__main__':
    generate_pronunciations_file('dictionaries/CSW.txt', 'dictionaries/pronunciations.txt')
