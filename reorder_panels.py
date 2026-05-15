path = 'templates/index.html'
with open(path, 'r') as f: content = f.read()
# We need to make sure the order is: Players -> Board -> Words
# (Since the patch might have placed them differently)
import re
# 1. Extract the three main containers
players_match = re.search(r'<div class="left-panel-container".*?</div>\s*</div>\s*</div>', content, re.DOTALL)
board_match = re.search(r'<div class="board-panel".*?<!-- Right Panel', content, re.DOTALL)
words_match = re.search(r'<div class="words-panel".*?</div>\s*</div>', content, re.DOTALL)
if players_match and board_match and words_match:
    players = players_match.group(0)
    board = board_match.group(0).replace('<!-- Right Panel', '')
    words = words_match.group(0)
    
    # 2. Build the new grid order
    new_grid = f'<div class="play-grid" id="play-grid-carousel">\n{players}\n{board}\n{words}\n'
    
    # 3. Replace the old grid with the new order
    content = re.sub(r'<div class="play-grid".*?</div>\s*</div>\s*<!-- End Play Grid', new_grid + '</div> <!-- End Play Grid', content, flags=re.DOTALL)
    
    with open(path, 'w') as f: f.write(content)
    print('Panels Reordered: Players(Left) - Board(Center) - Words(Right)')
else:
    print('Could not find all panels to reorder.')
