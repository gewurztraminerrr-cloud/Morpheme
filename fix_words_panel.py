path = 'static/css/play.css'
with open(path, 'a') as f: f.write("""
    /* Force Words panel visibility on mobile */
    #play-panel-words {
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
        min-height: 500px !important;
        padding: 10px !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    .words-panel .words-table-box {
        display: block !important;
        width: 100% !important;
        height: 100% !important;
    }
""")
print('Words Panel Fix Applied!')
