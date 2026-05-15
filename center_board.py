import re
path = 'templates/index.html'
with open(path, 'r') as f: content = f.read()
# 1. Clean out the existing grid entirely
# We find the start of the play-grid and the end of the play-grid
start_marker = '<div class="play-grid" id="play-grid-carousel">'
end_marker = '</div> <!-- End Play Grid -->'
if start_marker in content and end_marker in content:
    # 2. Re-create the structure from scratch with BOARD IN THE MIDDLE
    # We will use simplified placeholders for the panels to ensure it works
    
    # Extract the Players block
    players_block = re.search(r'<div class="left-panel-container".*?<!-- Center Panel', content, re.DOTALL).group(0).replace('<!-- Center Panel', '')
    # Extract the Board block
    board_block = re.search(r'<div class="board-panel".*?<!-- Right Panel', content, re.DOTALL).group(0).replace('<!-- Right Panel', '')
    # Extract the Words block
    words_block = re.search(r'<div class="words-panel".*?</div>\s*</div>', content, re.DOTALL).group(0)
    new_html = f"""{start_marker}
        {players_block}
        {board_block}
        {words_block}
    {end_marker}"""
    # Replace the old messy grid with the clean Centered Board version
    pattern = re.escape(start_marker) + r'.*?' + re.escape(end_marker)
    updated = re.sub(pattern, new_html, content, flags=re.DOTALL)
    
    with open(path, 'w') as f: f.write(updated)
    print('SUCCESS: Board is now Centered. Swipe Left for Words, Swipe Right for Players.')
else:
    print('Error: Could not find the grid markers.')
