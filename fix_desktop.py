path = 'static/css/play.css'
with open(path, 'r') as f: content = f.read()
# Force the Desktop grid to be 3 columns: Players(300px), Board(Auto), Words(300px)
desktop_fix = """
.play-grid {
    display: grid !important;
    grid-template-columns: 300px 1fr 300px !important;
    width: 100% !important;
}
"""
with open(path, 'a') as f: f.write(desktop_fix)
print('Desktop Grid Order Forced!')
