path = 'static/css/play.css'
with open(path, 'r') as f: content = f.read()
new_css = """
    .play-grid {
        display: flex !important;
        flex-direction: row !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        scroll-snap-type: x mandatory !important;
        scrollbar-width: none !important;
        -webkit-overflow-scrolling: touch !important;
        overscroll-behavior-x: contain !important;
        gap: 0 !important;
        padding: 0 !important;
        width: 100vw !important;
        height: calc(100vh - 120px) !important;
        flex: 1 !important;
    }
    .play-grid::-webkit-scrollbar { display: none !important; }
    .left-panel-container, .board-panel, .words-panel {
        min-width: 100vw !important;
        width: 100vw !important;
        scroll-snap-align: center !important;
        scroll-snap-stop: always !important;
        height: 100% !important;
        box-sizing: border-box !important;
        flex-shrink: 0 !important;
        overflow-y: auto !important;
    }
"""
if '@media (max-width: 992px) {' in content:
    updated = content.replace('@media (max-width: 992px) {', '@media (max-width: 992px) {' + new_css)
    with open(path, 'w') as f: f.write(updated)
    print('CSS Updated Successfully!')
