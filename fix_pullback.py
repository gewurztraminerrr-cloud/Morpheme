path = 'static/js/app.js'
with open(path, 'r') as f: content = f.read()
# This part is causing the "snap back" effect
if 'boardPanel.scrollIntoView' in content:
    # We will only allow it to happen ONCE when you first join
    new_logic = """
        if (window.innerWidth <= 992 && !window.hasCenteredBoard) {
            window.hasCenteredBoard = true;
            setTimeout(() => {
                const boardPanel = document.getElementById('play-panel-board');
                if (boardPanel) boardPanel.scrollIntoView({ behavior: 'auto', block: 'nearest', inline: 'start' });
            }, 100);
        }
    """
    # Replace the aggressive logic with the one-time logic
    import re
    content = re.sub(r'if \(window\.innerWidth <= 992\).*?inline: \'start\' \}\);\s*\}, 100\);\s*\}', new_logic, content, flags=re.DOTALL)
    with open(path, 'w') as f: f.write(content)
    print('Pullback Fixed!')
