
import os

file_path = 'static/index.html'

with open(file_path, 'r') as f:
    lines = f.readlines()

# The target block is inside "profile-display-container". 
# We saw in previous steps that line 419 is the container start: <div id="profile-display-container" ...
# Line 420 is the card start.
# Line 480 (approx) is the card end.
# We want to replace everything inside the container.

# Let's find the line index for profile-display-container
start_idx = -1
for i, line in enumerate(lines):
    if 'id="profile-display-container"' in line:
        start_idx = i
        break

if start_idx == -1:
    print("Could not find start index")
    exit(1)

# Find the end of this div. It's risky to guess, but we know the structure is stuck.
# We will verify if line[start_idx+1] contains 'profile-card'.
if 'profile-card' not in lines[start_idx+1]:
    print(f"Unexpected content at {start_idx+1}: {lines[start_idx+1]}")
    # Force proceed anyway if it looks like the old split header
    # exit(1) 

# We will replace from start_idx + 1 until we see the closing div of the container?
# Or just replace a fixed chunk if we trust the line count.
# Better: Replace from start_idx+1 to start_idx + 61 (480-419 = 61 lines).
# Let's be safer: Replace until we see line 481 '                        </div>' or similar?
# Or just find the next '</div>' with same indentation as profile-card?
# Profile card (line 420) has 28 spaces indent.
# We will delete lines until we find a line with 28 spaces and '</div>'.

end_idx = start_idx + 1
while end_idx < len(lines):
    if lines[end_idx].strip() == '</div>' and len(lines[end_idx]) - len(lines[end_idx].lstrip()) >= 28:
        break
    end_idx += 1

print(f"Replacing lines {start_idx+1} to {end_idx}")

new_html = """                            <div class="profile-card">
                                <div class="profile-header-section">
                                    <div class="profile-avatar-container">
                                        <div class="profile-avatar">?</div>
                                    </div>
                                    <div class="profile-main-info">
                                        <div class="profile-name-row">
                                            <h2 id="profile-username">Player</h2>
                                            <span class="profile-flag" id="profile-flag" title="Country">🏳️</span>
                                        </div>
                                        <div class="profile-meta-row">
                                            <span class="profile-rank-badge" id="profile-rank-text">Unranked</span>
                                            <span class="profile-rating-value" id="profile-rating">0</span>
                                        </div>
                                        <div class="profile-personal-row">
                                            <span id="profile-age-val">-</span> <span class="meta-label">AGE</span>
                                            <span class="sep">|</span>
                                            <span id="profile-gender-val">-</span> <span class="meta-label">SEX</span>
                                        </div>
                                        <div class="profile-quote">
                                            "<span id="profile-quote-val">Welcome to Morpheme.</span>"
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="profile-stats-section">
                                    <div class="stat-item">
                                        <span class="stat-value" id="profile-games">0</span>
                                        <span class="stat-label">GAMES</span>
                                    </div>
                                    <div class="stat-item">
                                        <span class="stat-value">0%</span>
                                        <span class="stat-label">WIN RATE</span>
                                    </div>
                                    <div class="stat-item">
                                        <span class="stat-value">-</span>
                                        <span class="stat-label">BEST</span>
                                    </div>
                                </div>

                                <div class="profile-actions">
                                    <button class="profile-action-btn">View History</button>
                                </div>
                            </div>
"""

# Replace the lines
lines[start_idx+1 : end_idx+1] = [new_html]

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Successfully updated index.html")
