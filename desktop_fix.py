path = 'static/css/play.css'
with open(path, 'a') as f:
    f.write("""
@media (min-width: 993px) {
    .play-grid {
        display: grid !important;
        grid-template-columns: 300px 1fr 300px !important;
        width: 100% !important;
        gap: 20px !important;
    }
    .words-panel, .left-panel-container {
        display: flex !important;
        width: 300px !important;
        min-width: 300px !important;
    }
}
""")
