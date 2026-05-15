import os
path = 'templates/index.html'
with open(path, 'r') as f: content = f.read()
# Define the CLEAN structure for the entire Play Grid
clean_grid = """
            <div class="play-grid" id="play-grid-carousel">
                <!-- Left Panel: Players -->
                <div class="left-panel-container" id="play-panel-players">
                    <button class="mobile-nav-btn return-to-main-right" onclick="document.getElementById('play-panel-board').scrollIntoView({behavior: 'smooth'})">Go Back to Board ➜</button>
                    <div class="players-panel">
                        <h3 id="players-heading">Players</h3>
                        <div class="player-actions-row">
                            <button id="find-me-btn" class="find-me-btn" style="display:none;">Find Me</button>
                            <button id="find-friends-btn" class="find-me-btn" style="display:none;">Find Friends</button>
                            <button id="show-everyone-btn" class="find-me-btn" style="display:none;">Show Everyone</button>
                        </div>
                        <div id="players-list">
                            <p class="placeholder">Loading...</p>
                        </div>
                    </div>
                    <div class="chat-panel">
                        <div id="chat-history">
                            <p class="placeholder">No messages yet</p>
                        </div>
                        <div class="chat-input-section">
                            <input type="text" id="chat-input" placeholder="Say something..." autocomplete="off">
                            <button id="chat-send-btn">Send</button>
                        </div>
                    </div>
                </div>
                <!-- Center Panel: Board -->
                <div class="board-panel" id="play-panel-board">
                    <div class="mobile-carousel-nav">
                        <button class="mobile-nav-btn nav-to-left" onclick="document.getElementById('play-panel-players').scrollIntoView({behavior: 'smooth'})">⇠ Players/Chat</button>
                        <button class="mobile-nav-btn nav-to-right" onclick="document.getElementById('play-panel-words').scrollIntoView({behavior: 'smooth'})">Words ⇢</button>
                    </div>
                    <div class="timer-display">
                        <div class="timer-label">Time:</div>
                        <div class="timer-value" id="timer-value">0:00</div>
                    </div>
                    <div id="game-board" class="game-board"></div>
                    <div id="word-validation-status" class="validation-status"></div>
                    <div class="word-input-section">
                        <input type="text" id="word-input" placeholder="Enter word" autocomplete="off" onkeydown="if(event.key==='Enter'||event.keyCode===13){event.preventDefault();submitWord(this.value);}">
                        <button id="submit-word-btn">Submit</button>
                        <button id="rotate-board-btn" class="rotate-btn">Rotate</button>
                    </div>
                    <div id="cube-rotate-hint" class="cube-rotate-hint hidden">ARROWS TO ROTATE</div>
                </div>
                <!-- Right Panel: Words and Definitions -->
                <div class="words-panel" id="play-panel-words">
                    <button class="mobile-nav-btn return-to-main-left" onclick="document.getElementById('play-panel-board').scrollIntoView({behavior: 'smooth'})">⇠ Go Back to Board</button>
                    <div class="words-table-box">
                        <div class="words-tabs" id="words-tabs-container">
                            <button class="word-tab active" data-tab="found">Words</button>
                            <button class="word-tab" data-tab="remaining" style="display: none;">Remaining</button>
                            <button class="word-tab" data-tab="clues" style="display: none;">Clues</button>
                            <button class="word-tab" data-tab="previous" style="display: none;">Previous Day</button>
                            <button class="word-tab" data-tab="history">History</button>
                        </div>
                        <div id="tab-content-found" class="tab-content active">
                            <div class="words-header-group">
                                <h3 id="words-panel-title">Your Words</h3>
                                <p class="words-stats" id="words-stats"></p>
                            </div>
                            <div id="submitted-words-list">
                                <p class="placeholder">No words yet</p>
                            </div>
                        </div>
                        <div id="tab-content-remaining" class="tab-content">
                            <div class="words-header-group"><h3>Remaining</h3></div>
                            <div id="remaining-words-list"><p class="placeholder">Calculating...</p></div>
                        </div>
                        <div id="tab-content-clues" class="tab-content">
                            <div class="words-header-group"><h3>Clues</h3></div>
                            <div id="clues-list" class="clues-grid"><p class="placeholder">Loading...</p></div>
                        </div>
                        <div id="tab-content-previous" class="tab-content">
                            <div class="words-header-group"><h3>Previous Day</h3></div>
                            <div id="previous-words-list"><p class="placeholder">No data</p></div>
                        </div>
                        <div id="tab-content-history" class="tab-content">
                            <div class="words-header-group"><h3>Past Winners</h3></div>
                            <div id="winners-list" class="winners-list-container"><p class="placeholder">No history...</p></div>
                        </div>
                    </div>
                    <div class="definitions-panel">
                        <div id="definition-header" class="definition-header" style="display: none;"><span id="definition-word" class="definition-word"></span></div>
                        <div id="definition-content"><p class="placeholder">Select a word for definition</p></div>
                    </div>
                </div>
            </div>
"""
import re
# Find the start and end of the messy section
start_pattern = r'<div class=\"play-grid\".*?id=\"play-grid-carousel\">'
end_pattern = r'</div> <!-- End Words Panel -->\s*</div>'
# Perform the replacement
new_content = re.sub(start_pattern + r'.*?' + end_pattern, clean_grid, content, flags=re.DOTALL)
with open(path, 'w') as f: f.write(new_content)
print('SUCCESS: Game Room structure is now perfectly organized!')
