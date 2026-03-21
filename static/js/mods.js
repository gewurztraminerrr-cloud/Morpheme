
async function checkModStatus() {
    try {
        const response = await fetch('/api/mods/status');
        const data = await response.json();
        
        window.currentUserIsMod = data.is_mod;
        
        if (typeof updateAuthUI === 'function') {
            updateAuthUI();
        }
        
        if (data.is_mod) {
            loadModList();
        }
    } catch (err) {
        console.error("Error checking mod status:", err);
    }
}


async function loadModList() {
    const listEl = document.getElementById('mod-list-container');
    if (!listEl) return;

    try {
        const response = await fetch('/api/mods/list');
        const data = await response.json();
        if (data.mods) {
            listEl.innerHTML = data.mods.map(m => `
                <div class="mod-item">
                    <span class="mod-name">${m}</span>
                    <button class="remove-mod-btn" onclick="removeModerator('${m}')" title="Remove Moderator">&times;</button>
                </div>
            `).join('');

        }
    } catch (err) {
        console.error("Error loading mod list:", err);
    }
}

async function addModerator() {
    const input = document.getElementById('new-mod-username');
    const username = input ? input.value.trim() : '';
    if (!username) return;

    try {
        const response = await fetch('/api/mods/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username })
        });
        const data = await response.json();
        if (data.success) {
            if (input) input.value = '';
            alert(`User ${username} added as moderator.`);
            loadModList();
        } else {
            alert(data.error || "Failed to add moderator.");
        }
    } catch (err) {
        console.error("Error adding mod:", err);
    }
}

async function removeModerator(username) {
    if (!confirm(`Are you sure you want to remove ${username} as moderator?`)) return;

    try {
        const response = await fetch('/api/mods/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username })
        });
        const data = await response.json();
        if (data.success) {
            alert(`User ${username} removed from moderators.`);
            loadModList();
        } else {
            alert(data.error || "Failed to remove moderator.");
        }
    } catch (err) {
        console.error("Error removing mod:", err);
    }
}


async function addPronunciation() {
    const wordInput = document.getElementById('pron-word-input');
    const pronInput = document.getElementById('pron-value-input');
    const word = wordInput ? wordInput.value.trim() : '';
    const pronunciation = pronInput ? pronInput.value.trim() : '';

    if (!word || !pronunciation) {
        alert("Both word and pronunciation are required.");
        return;
    }

    try {
        const response = await fetch('/api/pronunciations/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word, pronunciation })
        });
        const data = await response.json();
        if (data.success) {
            if (wordInput) wordInput.value = '';
            if (pronInput) pronInput.value = '';
            alert(`Pronunciation for ${word} added successfully.`);
        } else {
            alert(data.error || "Failed to add pronunciation.");
        }
    } catch (err) {
        console.error("Error adding pronunciation:", err);
    }
}

async function removePronunciation() {
    const wordInput = document.getElementById('pron-word-input');
    const word = wordInput ? wordInput.value.trim() : '';

    if (!word) {
        alert("Word is required to remove pronunciation.");
        return;
    }

    if (!confirm(`Are you sure you want to remove the pronunciation for ${word}?`)) return;

    try {
        const response = await fetch('/api/pronunciations/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word })
        });
        const data = await response.json();
        if (data.success) {
            if (wordInput) wordInput.value = '';
            alert(`Pronunciation for ${word} removed successfully.`);
        } else {
            alert(data.error || "Failed to remove pronunciation.");
        }
    } catch (err) {
        console.error("Error removing pronunciation:", err);
    }
}

// Global initialization
document.addEventListener('DOMContentLoaded', () => {
    checkModStatus();
    
    // Moderators
    const addModBtn = document.getElementById('add-mod-btn');
    if (addModBtn) {
        addModBtn.addEventListener('click', addModerator);
    }

    // Pronunciations
    const addPronBtn = document.getElementById('add-pron-btn');
    if (addPronBtn) {
        addPronBtn.addEventListener('click', addPronunciation);
    }

    const removePronBtn = document.getElementById('remove-pron-btn');
    if (removePronBtn) {
        removePronBtn.addEventListener('click', removePronunciation);
    }
});

// Re-check on login/logout events if they exist
window.addEventListener('user-login', checkModStatus);

