path = 'static/css/play.css'
with open(path, 'a') as f:
    f.write("""
@media (max-width: 992px) {
    .play-grid {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        scroll-snap-type: x mandatory !important;
        -webkit-overflow-scrolling: touch !important;
        width: 100vw !important;
    }
    .left-panel-container, .board-panel, .words-panel {
        min-width: 100vw !important;
        width: 100vw !important;
        flex-shrink: 0 !important;
        scroll-snap-align: center !important;
    }
}
""")
