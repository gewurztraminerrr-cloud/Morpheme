import re
path = 'templates/index.html'
with open(path, 'r') as f: content = f.read()
# 1. Grab the clean panels
players = re.search(r'<div class=\"left-panel-container\".*?</div>\s*</div>\s*</div>', content, re.DOTALL).group(0)
board = re.search(r'<div class=\"board-panel\".*?</div>\s*</div>', content, re.DOTALL).group(0)
words = re.search(r'<div class=\"words-panel\".*?</div>\s*</div>', content, re.DOTALL).group(0)
# 2. Re-build the entire Swipe section with Board in the MIDDLE
new_grid = f"""
            <div class="play-grid" id="play-grid-carousel">
                {players}
                {board}
                {words}
            </div>
"""
# 3. Replace the old messy area with this clean one
# (Finding the spot between the param-bonus and the leaderboards)
pattern = r'<div class=\"play-grid\".*?<!-- Leaderboards Page -->'
updated = re.sub(pattern, new_grid + '\n        <!-- Leaderboards Page -->', content, flags=re.DOTALL)
with open(path, 'w') as f: f.write(updated)
print('SUCCESS: Perfect Sandwich Layout Rebuilt!')
