path = 'static/css/play.css'
with open(path, 'r') as f: content = f.read()
force_visible = """
    /* Force Players and Chat visibility on mobile carousel */
    #play-panel-players {
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
        min-height: 500px !important;
        padding: 10px !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    .players-panel, .chat-panel {
        display: flex !important;
        width: 100% !important;
        margin: 0 0 10px 0 !important;
        flex-shrink: 0 !important;
    }
    .players-panel { flex: 1 !important; min-height: 300px !important; }
    .chat-panel { height: 250px !important; }
"""
with open(path, 'a') as f: f.write(force_visible)
print('Nuclear Visibility Fix Applied!')
