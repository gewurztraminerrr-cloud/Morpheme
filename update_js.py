# Update app.js
with open('static/js/app.js', 'r') as f: content = f.read()
js_fix = """
        // Mobile Carousel: Center the board automatically on entry
        if (window.innerWidth <= 992) {
            setTimeout(() => {
                const boardPanel = document.getElementById('play-panel-board');
                if (boardPanel) {
                    boardPanel.scrollIntoView({ behavior: 'auto', block: 'nearest', inline: 'start' });
                }
            }, 100);
        }
"""
if "input.focus();" in content:
    content = content.replace("input.focus();", "input.focus();" + js_fix)
    with open('static/js/app.js', 'w') as f: f.write(content)
    print('JavaScript Updated Successfully!')
