
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
            document.querySelectorAll('.mod-only-btn').forEach(btn => btn.style.display = 'inline-block');
        }
    } catch (err) {
        console.error("Error checking mod status:", err);
    }
}

function showModStatus(message, isError = false, targetId = 'mod-status-area') {
    const statusArea = document.getElementById(targetId);
    if (!statusArea) {
        // Fallback to alert if status area not found
        alert(message);
        return;
    }
    statusArea.textContent = message;
    statusArea.style.color = isError ? '#f43f5e' : '#4ade80';
    statusArea.style.opacity = '1';
    
    // Clear after 5 seconds
    setTimeout(() => {
        if (statusArea.textContent === message) {
            statusArea.style.transition = 'opacity 1s ease';
            statusArea.style.opacity = '0';
            setTimeout(() => {
                if (statusArea.textContent === message) statusArea.textContent = '';
                statusArea.style.opacity = '1';
                statusArea.style.transition = '';
            }, 1000);
        }
    }, 5000);
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
            showModStatus(`User ${username} added as moderator.`);
            loadModList();
        } else {
            showModStatus(data.error || "Failed to add moderator.", true);
        }
    } catch (err) {
        console.error("Error adding mod:", err);
        showModStatus("Network error adding moderator.", true);
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
            showModStatus(`User ${username} removed from moderators.`);
            loadModList();
        } else {
            showModStatus(data.error || "Failed to remove moderator.", true);
        }
    } catch (err) {
        console.error("Error removing mod:", err);
        showModStatus("Network error removing moderator.", true);
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
            showModStatus(`Pronunciation for ${word} added successfully.`, false, 'pron-status-area');
        } else {
            showModStatus(data.error || "Failed to add pronunciation.", true, 'pron-status-area');
        }
    } catch (err) {
        console.error("Error adding pronunciation:", err);
        showModStatus("Network error adding pronunciation.", true, 'pron-status-area');
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
            showModStatus(`Pronunciation for ${word} removed successfully.`, false, 'pron-status-area');
        } else {
            showModStatus(data.error || "Failed to remove pronunciation.", true, 'pron-status-area');
        }
    } catch (err) {
        console.error("Error removing pronunciation:", err);
        showModStatus("Network error removing pronunciation.", true, 'pron-status-area');
    }
}

async function addAddedWord() {
    console.info("[Mods] addAddedWord triggered.");
    const wordInput = document.getElementById('added-word-input');
    const word = wordInput ? wordInput.value.trim() : '';

    if (!word) {
        alert("Word is required.");
        return;
    }

    console.info(`[Mods] Attempting to add word: "${word}"`);

    try {
        const response = await fetch('/api/mods/added_words/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word })
        });
        const data = await response.json();
        if (data.success) {
            if (wordInput) wordInput.value = '';
            showModStatus(`Word "${word}" added to Added Words list.`, false, 'added-word-status-area');
            
            if (window.loadAddedWords) window.loadAddedWords('added');
        } else {
            showModStatus(data.error || "Failed to add word.", true, 'added-word-status-area');
        }
    } catch (err) {
        console.error("Error adding added word:", err);
        showModStatus("Network error adding word.", true, 'added-word-status-area');
    }
}

async function removeAddedWord() {
    console.info("[Mods] removeAddedWord triggered.");
    const wordInput = document.getElementById('added-word-input');
    const word = wordInput ? wordInput.value.trim() : '';

    if (!word) {
        alert("Word is required to remove.");
        return;
    }

    console.info(`[Mods] Attempting to remove word: "${word}"`);

    try {
        const response = await fetch('/api/mods/added_words/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word.toUpperCase() })
        });
        const data = await response.json();
        if (data.success) {
            if (wordInput) wordInput.value = '';
            showModStatus(`Word "${word}" removed from Added Words list.`, false, 'added-word-status-area');
            alert(`Success: Word "${word}" was removed from the dictionary.`);
            
            if (window.loadAddedWords) window.loadAddedWords('added');
        } else {
            showModStatus(data.error || "Failed to remove word.", true, 'added-word-status-area');
            alert("Error: " + (data.error || "Failed to remove word."));
        }
    } catch (err) {
        console.error("Error removing added word:", err);
        showModStatus("Network error removing word.", true, 'added-word-status-area');
        alert("Network error: Could not reach the server.");
    }
}
window.addAddedWord = addAddedWord;
window.removeAddedWord = removeAddedWord;


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

    // Added Words
    const addAddedWordBtn = document.getElementById('add-added-word-btn');
    if (addAddedWordBtn) {
        addAddedWordBtn.addEventListener('click', addAddedWord);
    }

    const removeAddedWordBtn = document.getElementById('remove-added-word-btn');
    if (removeAddedWordBtn) {
        removeAddedWordBtn.addEventListener('click', removeAddedWord);
    }

    // Definitions
    const addDefBtn = document.getElementById('add-def-btn');
    if (addDefBtn) {
        addDefBtn.addEventListener('click', addDefinition);
    }

    const removeDefBtn = document.getElementById('remove-def-btn');
    if (removeDefBtn) {
        removeDefBtn.addEventListener('click', removeDefinition);
    }
});

async function addDefinition() {
    const wordInput = document.getElementById('def-word-input');
    const textInput = document.getElementById('def-text-input');
    const word = wordInput ? wordInput.value.trim().toUpperCase() : '';
    const def = textInput ? textInput.value.trim() : '';

    if (!word || !def) {
        alert("Both word and definition are required.");
        return;
    }

    try {
        const response = await fetch('/api/mods/definitions/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word, definition: def })
        });
        const data = await response.json();
        if (data.success) {
            if (wordInput) wordInput.value = '';
            if (textInput) textInput.value = '';
            showModStatus(`Definition for "${word}" updated.`, false, 'def-status-area');
            alert(`Success: Definition for "${word}" has been set.`);
        } else {
            alert("Error: " + (data.error || "Failed to set definition."));
        }
    } catch (err) {
        console.error("Error setting definition:", err);
        showModStatus("Network error setting definition.", true, 'def-status-area');
    }
}

async function removeDefinition() {
    const wordInput = document.getElementById('def-word-input');
    const word = wordInput ? wordInput.value.trim().toUpperCase() : '';

    if (!word) {
        alert("Word is required to remove definition.");
        return;
    }

    try {
        const response = await fetch('/api/mods/definitions/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word })
        });
        const data = await response.json();
        if (data.success) {
            if (wordInput) wordInput.value = '';
            showModStatus(`Definition for "${word}" removed.`, false, 'def-status-area');
            alert(`Success: Definition for "${word}" has been removed.`);
        } else {
            alert("Error: " + (data.error || "Failed to remove definition."));
        }
    } catch (err) {
        console.error("Error removing definition:", err);
        showModStatus("Network error removing definition.", true, 'def-status-area');
    }
}


window.promptAddAddedWord = async function() {
    const word = prompt("Enter word to add to Added Words list:");
    if (!word) return;
    
    try {
        const response = await fetch('/api/mods/added_words/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word.trim() })
        });
        const data = await response.json();
        
        if (data.success) {
            alert(data.message);
            if (window.loadAddedWords) window.loadAddedWords();
        } else {
            alert(data.error);
        }
    } catch (err) {
        console.error("Error adding word:", err);
        alert("Failed to add word.");
    }
};

window.promptRemoveAddedWord = async function() {
    const word = prompt("Enter word to remove from Added Words list:");
    if (!word) return;
    
    try {
        const response = await fetch('/api/mods/added_words/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word.trim().toUpperCase() })
        });
        const data = await response.json();
        
        if (data.success) {
            alert(data.message || `Word "${word}" removed.`);
            if (window.loadAddedWords) window.loadAddedWords('added');
        } else {
            alert("Error: " + (data.error || "Failed to remove word."));
        }
    } catch (err) {
        console.error("Error removing word:", err);
        alert("Network error: Could not reach the server.");
    }
};


