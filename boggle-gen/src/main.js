// Boggle Dice Configuration (New Boggle / Standard 4x4)
const DICE_CONFIG = [
    "AAEEGN", "ABBJOO", "ACHOPS", "AFFKPS",
    "AOOTTW", "CIMOTU", "DEILRX", "DELRVY",
    "DISTTY", "EEGHNW", "EEINSU", "EHRTVW",
    "EIOSST", "ELRTTY", "HIMNQU", "HLNNRZ"
];

class BoggleGame {
    constructor() {
        this.boardElement = document.getElementById('board');
        this.shakeButton = document.getElementById('shake-btn');
        this.isShaking = false;
        this.isCrashed = true; // Start in crashed state per request
        this.savesList = document.getElementById('saves-list');
        this.restartBtn = document.getElementById('restart-server-btn');
        this.checkpointBtn = document.getElementById('checkpoint-btn');
        this.pastSavesPanel = document.getElementById('past-saves');
        this.history = [];

        // Bind events
        this.shakeButton.addEventListener('click', () => this.shake());
        this.restartBtn.addEventListener('click', () => this.restartServer());
        this.checkpointBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.pastSavesPanel.classList.toggle('collapsed');
        });

        // Close panel when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.pastSavesPanel.contains(e.target) && !this.checkpointBtn.contains(e.target)) {
                this.pastSavesPanel.classList.add('collapsed');
            }
        });

        // Initial render
        this.generateBoard();
        this.loadHistory();
    }

    getRandomFace(die) {
        const face = die[Math.floor(Math.random() * die.length)];
        return face === 'Q' ? 'Qu' : face;
    }

    shuffle(array) {
        let currentIndex = array.length, randomIndex;

        // While there remain elements to shuffle.
        while (currentIndex != 0) {
            // Pick a remaining element.
            randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex--;

            // And swap it with the current element.
            [array[currentIndex], array[randomIndex]] = [
                array[randomIndex], array[currentIndex]];
        }

        return array;
    }

    generateBoard() {
        // 1. Shuffle dice positions
        const shuffledDice = this.shuffle([...DICE_CONFIG]);

        // 2. Select faces
        const boardState = shuffledDice.map(die => this.getRandomFace(die));

        // 3. Update history
        this.addToHistory(boardState);

        // 4. Render
        this.render(boardState);
    }

    async addToHistory(boardState) {
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const lettersStr = JSON.stringify(boardState);
        
        this.history.unshift({
            time: time,
            letters: lettersStr
        });
        
        // Limit to 20
        if (this.history.length > 20) this.history.pop();
        
        this.updateSavesUI();

        // Persist to server
        try {
            await fetch('http://127.0.0.1:5005/history', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ letters: lettersStr })
            });
        } catch (error) {
            console.warn("Could not persist board to server history", error);
        }
    }

    async loadHistory() {
        try {
            const response = await fetch('http://127.0.0.1:5005/history');
            if (response.ok) {
                const data = await response.json();
                this.history = data;
                this.isCrashed = false;
            } else {
                throw new Error("Server returned an error");
            }
        } catch (error) {
            console.error("Could not fetch board history:", error);
            // Don't set isCrashed here, let the caller decide if it's a fatal crash
        } finally {
            this.updateSavesUI();
        }
    }

    updateSavesUI() {
        this.savesList.innerHTML = '';
        if (this.isCrashed) {
            this.savesList.innerHTML = `
                <div class="crash-status">
                    <span class="crash-icon">⚠️</span>
                    <p class="crash-msg">Antigravity server crashed unexpectedly. Please restart to fully restore AI features.</p>
                </div>
            `;
            return;
        }
        if (this.history.length === 0) {
            this.savesList.innerHTML = '<div class="save-item" style="color:#999;cursor:default;">No history found</div>';
            return;
        }
        this.history.forEach(item => {
            const div = document.createElement('div');
            div.className = 'save-item';
            
            // Extract preview letters
            let preview = '...';
            try {
                const letters = typeof item.letters === 'string' ? JSON.parse(item.letters) : item.letters;
                if (Array.isArray(letters)) {
                    preview = letters.slice(0, 4).join(' ');
                }
            } catch (e) {}

            div.innerHTML = `
                <span>${item.time}</span>
                <span class="timestamp">${preview}</span>
            `;
            div.onclick = () => {
                try {
                    const letters = typeof item.letters === 'string' ? JSON.parse(item.letters) : item.letters;
                    this.render(letters.flat ? letters.flat() : letters);
                } catch (e) {
                    console.error("Failed to parse board state", e);
                }
            };
            this.savesList.appendChild(div);
        });
    }

    async restartServer() {
        const originalText = this.restartBtn.textContent;
        this.restartBtn.textContent = "Restoring...";
        this.restartBtn.disabled = true;
        
        try {
            // Give the backend a moment
            await new Promise(r => setTimeout(r, 1500));
            
            const response = await fetch('http://127.0.0.1:5005/history');
            if (!response.ok) throw new Error("Backend connection failed");

            this.isCrashed = false;
            const data = await response.json();
            this.history = data;
            
            this.restartBtn.textContent = "Restored!";
            this.generateBoard();
        } catch (error) {
            console.error("Restart failed:", error);
            this.restartBtn.textContent = "Failed!";
            this.isCrashed = true;
        } finally {
            this.updateSavesUI();
            setTimeout(() => {
                this.restartBtn.textContent = originalText;
                this.restartBtn.disabled = false;
            }, 2000);
        }
    }

    render(boardState) {
        this.boardElement.innerHTML = '';

        boardState.forEach(face => {
            const cube = document.createElement('div');
            cube.className = 'cube';
            cube.textContent = face;

            // Add special class for 'Qu' to handle font size if needed
            if (face === 'Qu') {
                cube.classList.add('span-qu');
            }

            this.boardElement.appendChild(cube);
        });
    }

    shake() {
        if (this.isShaking) return;
        this.isShaking = true;

        // Add shaking animation class
        this.boardElement.classList.add('shaking');

        // Disable button temporarily
        this.shakeButton.disabled = true;

        // Generate new board halfway through animation or after
        setTimeout(() => {
            this.generateBoard();
        }, 250);

        // Remove class and re-enable button after animation
        setTimeout(() => {
            this.boardElement.classList.remove('shaking');
            this.isShaking = false;
            this.shakeButton.disabled = false;
        }, 500);
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    new BoggleGame();
});
