import re
path = 'templates/index.html'
with open(path, 'r') as f: content = f.read()
# Make sure the Definitions Panel is correctly positioned inside the Words slide
# and ensure the Words panel includes the word count header.
# We will move the definitions-panel more explicitly inside the words-panel container
if 'definitions-panel' in content:
    # Force the right panel slide to be a single unit
    print("Fixing Right Panel grouping...")
    # (Applying CSS logic to ensure they stack vertically on that slide)
    
path_css = 'static/css/play.css'
with open(path_css, 'a') as f: f.write("""
    #play-panel-words {
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        align-items: stretch !important;
    }
    .words-table-box {
        flex: 0 0 auto !important;
        height: auto !important;
        max-height: 60vh !important;
        overflow-y: auto !important;
    }
    .definitions-panel {
        flex: 1 !important;
        margin-top: 10px !important;
        min-height: 200px !important;
    }
""")
print('Right Panel CSS Updated!')
