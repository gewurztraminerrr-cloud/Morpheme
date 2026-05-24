
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
                    ${m.toLowerCase() === 'jeffbabiak' ? '' : `<button class="remove-mod-btn" onclick="removeModerator('${m}')" title="Remove Moderator">&times;</button>`}
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
        
        // UNCONDITIONALLY clear input and restore focus to facilitate rapid typing flow
        if (wordInput) {
            wordInput.value = '';
            wordInput.focus();
        }

        if (data.success) {
            const msg = data.message || `Word "${word}" added (V2 Success)`;
            showModStatus(msg, false, 'added-word-status-area');
            if (window.loadAddedWords) window.loadAddedWords('added');
        } else {
            const errorMsg = data.error || "Failed to add word (V2 Error)";
            showModStatus(errorMsg, true, 'added-word-status-area');
        }
    } catch (err) {
        console.error("Error adding added word:", err);
        if (wordInput) {
            wordInput.value = '';
            wordInput.focus();
        }
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
        
        // Clear input regardless of success/fail (consistency)
        if (wordInput) {
            wordInput.value = '';
            wordInput.focus();
        }

        if (data.success) {
            showModStatus(`Word "${word}" removed from Added Words list.`, false, 'added-word-status-area');
            alert(`Success: Word "${word}" was removed from the dictionary.`);
            
            if (window.loadAddedWords) window.loadAddedWords('added');
        } else {
            showModStatus(data.error || "Failed to remove word.", true, 'added-word-status-area');
            alert("Error: " + (data.error || "Failed to remove word."));
        }
        
        // Final focus catch specifically after alerts
        if (wordInput) wordInput.focus();
    } catch (err) {
        console.error("Error removing added word:", err);
        if (wordInput) {
            wordInput.value = '';
            wordInput.focus();
        }
        showModStatus("Network error removing word.", true, 'added-word-status-area');
        alert("Network error: Could not reach the server.");
        if (wordInput) wordInput.focus();
    }
}
window.addAddedWord = addAddedWord;
window.removeAddedWord = removeAddedWord;

async function loadAddedWordsConfig() {
    const toggle = document.getElementById('toggle-use-added-words');
    if (!toggle) return;
    
    try {
        const response = await fetch('/api/mods/added_words/config');
        const data = await response.json();
        if (data.hasOwnProperty('use_added_words')) {
            toggle.checked = data.use_added_words;
        }
    } catch (err) {
        console.error("Error loading added words config:", err);
    }
}


async function toggleAddedWords(enabled) {
    try {
        const response = await fetch('/api/mods/added_words/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled })
        });
        const data = await response.json();
        if (data.success) {
            showModStatus(`Added Words are now ${enabled ? 'ENABLED' : 'DISABLED'} game-wide.`, false, 'added-word-status-area');
        } else {
            showModStatus(data.error || "Failed to toggle added words.", true, 'added-word-status-area');
        }
    } catch (err) {
        console.error("Error toggling added words:", err);
        showModStatus("Network error toggling added words.", true, 'added-word-status-area');
    }
}

// Dictionary Database Submission
async function submitDictionaryToDatabase() {
    const fileInput = document.getElementById('dict-upload-input');
    const statusEl = document.getElementById('dict-upload-status');
    
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        if (statusEl) {
            statusEl.innerText = "Error: No file selected.";
            statusEl.style.color = "#f43f5e";
        }
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    if (statusEl) {
        statusEl.innerText = "Submitting to database...";
        statusEl.style.color = "#00d2ff";
    }

    try {
        const response = await fetch('/api/mods/dictionary/submit', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (data.success) {
            if (statusEl) {
                statusEl.innerText = `Success: ${data.added_count} words added to ${data.target}.`;
                statusEl.style.color = "#4ade80";
            }
            alert(`Dictionary Updated: ${data.added_count} words successfully merged into ${data.target}. Staging list cleaned.`);
            
            // Reset input
            fileInput.value = '';
            const triggerBtn = document.getElementById('dict-upload-trigger-btn');
            if (triggerBtn) triggerBtn.innerText = "Select .txt File";
            
            // Refresh added words list if open
            if (window.loadAddedWords) window.loadAddedWords('added');
        } else {
            if (statusEl) {
                statusEl.innerText = "Error: " + (data.error || "Upload failed.");
                statusEl.style.color = "#f43f5e";
            }
        }
    } catch (err) {
        console.error("Dict Upload Error:", err);
        if (statusEl) {
            statusEl.innerText = "Network error during submission.";
            statusEl.style.color = "#f43f5e";
        }
    }
}


// Global initialization

async function loadLobbyNotice() {
    const input = document.getElementById('lobby-notice-input');
    if (!input) return;
    
    try {
        const response = await fetch('/api/mods/lobby-notice');
        const data = await response.json();
        if (data.notice) {
            input.value = data.notice;
        } else {
            input.value = '';
        }
    } catch (err) {
        console.error("Error loading lobby notice:", err);
    }
}

async function updateLobbyNotice() {
    const input = document.getElementById('lobby-notice-input');
    if (!input) return;
    const notice = input.value.trim();
    
    try {
        const response = await fetch('/api/mods/lobby-notice/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ notice })
        });
        const data = await response.json();
        if (data.success) {
            showModStatus("Lobby notice updated successfully.", false, 'notice-status-area');
        } else {
            showModStatus(data.error || "Failed to update notice.", true, 'notice-status-area');
        }
    } catch (err) {
        console.error("Error updating notice:", err);
        showModStatus("Network error updating notice.", true, 'notice-status-area');
    }
}

async function clearLobbyNotice() {
    const input = document.getElementById('lobby-notice-input');
    if (input) input.value = '';
    await updateLobbyNotice();
}

document.addEventListener('DOMContentLoaded', () => {
    loadLobbyNotice(); // Load initial notice
    checkModStatus();
    loadAddedWordsConfig();
    
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

    const addedWordInput = document.getElementById('added-word-input');
    if (addedWordInput) {
        addedWordInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                addAddedWord();
            }
        });
    }

    const toggleAddedWordsEl = document.getElementById('toggle-use-added-words');
    if (toggleAddedWordsEl) {
        toggleAddedWordsEl.addEventListener('change', (e) => toggleAddedWords(e.target.checked));
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

    // Ban User
    const banUserBtn = document.getElementById('ban-user-btn');
    if (banUserBtn) {
        banUserBtn.addEventListener('click', banUser);
    }


    const setNoticeBtn = document.getElementById('set-notice-btn');
    if (setNoticeBtn) {
        setNoticeBtn.addEventListener('click', updateLobbyNotice);
    }

    const clearNoticeBtn = document.getElementById('clear-notice-btn');
    if (clearNoticeBtn) {
        clearNoticeBtn.addEventListener('click', clearLobbyNotice);
    }

    // Dictionary Upload
    const dictTriggerBtn = document.getElementById('dict-upload-trigger-btn');
    const dictFileInput = document.getElementById('dict-upload-input');
    const dictSubmitBtn = document.getElementById('dict-submit-db-btn');

    if (dictTriggerBtn && dictFileInput) {
        dictTriggerBtn.addEventListener('click', () => dictFileInput.click());
        dictFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                const name = e.target.files[0].name;
                dictTriggerBtn.innerText = `File: ${name}`;
                const statusEl = document.getElementById('dict-upload-status');
                if (statusEl) {
                    statusEl.innerText = `Selected: ${name}`;
                    statusEl.style.color = "#fff";
                }
            }
        });
    }

    if (dictSubmitBtn) {
        dictSubmitBtn.addEventListener('click', submitDictionaryToDatabase);
    }
});

async function banUser() {
    const input = document.getElementById('ban-username-input');
    const username = input ? input.value.trim() : '';
    if (!username) {
        alert("Please enter a username to ban.");
        return;
    }

    if (username.toLowerCase() === 'jeffbabiak') {
        alert("Action Prohibited: User 'JeffBabiak' cannot be banned.");
        return;
    }

    if (!confirm(`ARE YOU SURE? This will permanently ERASE all data for user "${username}". This cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch('/api/mods/ban_user', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username })
        });
        const data = await response.json();
        if (data.success) {
            if (input) input.value = '';
            showModStatus(data.message, false, 'ban-status-area');
            alert(`User "${username}" has been permanently erased from the database.`);
        } else {
            showModStatus(data.error || "Failed to ban user.", true, 'ban-status-area');
            alert("Error: " + (data.error || "Failed to ban user."));
        }
    } catch (err) {
        console.error("Error banning user:", err);
        showModStatus("Network error banning user.", true, 'ban-status-area');
    }
}

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


