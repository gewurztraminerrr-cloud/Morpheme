path = 'static/js/app.js'
with open(path, 'r') as f: content = f.read()
# Only run if it hasn't been added yet
if 'window.hasCenteredBoard' not in content:
    js_fix = """
    if (window.innerWidth <= 768 && !window.hasCenteredBoard) {
        window.hasCenteredBoard = true;
        setTimeout(() => {
            const board = document.getElementById('play-panel-board');
            if (board) board.scrollIntoView({ behavior: 'auto', block: 'nearest', inline: 'start' });
        }, 500);
    }
    """
    content = content.replace('input.focus();', 'input.focus();' + js_fix)
    with open(path, 'w') as f: f.write(content)
    print('JavaScript Repaired!')
