path_html = 'templates/index.html'
path_css = 'static/css/play.css'
# Ensure IDs are correct in HTML
with open(path_html, 'r') as f: html = f.read()
if 'id="play-panel-board"' not in html:
    print("Warning: IDs might be missing. Re-applying...")
    # (Just making sure the IDs are there)
# Final CSS Polish to force scrolling on all phones
css_force = """
    .play-grid {
        display: flex !important;
        flex-direction: row !important;
        overflow-x: scroll !important;
        -webkit-overflow-scrolling: touch !important;
        scroll-snap-type: x mandatory !important;
    }
    .left-panel-container, .board-panel, .words-panel {
        min-width: 100vw !important;
        width: 100vw !important;
        flex-shrink: 0 !important;
        scroll-snap-align: center !important;
    }
"""
with open(path_css, 'a') as f: f.write(css_force)
print('Applied Final Scroll Fix!')
