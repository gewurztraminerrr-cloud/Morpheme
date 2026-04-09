import time
from board_generator import BoardGenerator

start = time.time()
bg = BoardGenerator()
print("Init took:", time.time() - start)

start2 = time.time()
bg.generate_board(dimensions='4x4', bonus_word='SMILE', word_count_range='100-9999', dictionary='NWL', board_format='Normal', min_word_length=4, difficulty='Hard')
print("Generation took:", time.time() - start2)
