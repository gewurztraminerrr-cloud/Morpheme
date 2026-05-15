path = 'static/css/play.css'
with open(path, 'r') as f: content = f.read()
mobile_swipe_css = """
@media (max-width: 992px) {
    .play-grid {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        scroll-snap-type: x mandatory !important;
        -webkit-overflow-scrolling: touch !important;
        width: 100vw !important;
        gap: 0 !important;
        padding: 0 !important;
    }
    .left-panel-container, .board-panel, .words-panel {
        min-width: 100vw !important;
        width: 100vw !important;
        flex-shrink: 0 !important;
        scroll-snap-align: center !important;
        height: auto !important;
        overflow-y: auto !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .mobile-carousel-nav {
        display: flex !important;
        justify-content: space-between !important;
        padding: 10px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    .mobile-nav-btn {
        background: rgba(255,255,255,0.1) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        padding: 8px 15px !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        font-size: 0.8rem !important;
    }
}
"""
with open(path, 'a') as f: f.write(mobile_swipe_css)
print('Step 2 Complete: Mobile Swipe Active!')
