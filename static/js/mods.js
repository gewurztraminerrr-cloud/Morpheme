
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
window.escapeHtml = escapeHtml;
window.escapeHTML = escapeHtml;

async function checkModStatus() {
    try {
        const response = await fetch('/api/mods/status');
        const data = await response.json();
        
        window.currentUserIsMod = Boolean(data.is_mod);
        window.currentUserIsRootMod = Boolean(data.is_root);
        if (data.username) {
            window.currentUser = data.username;
        }
        
        const modsBtn = document.getElementById('nav-mods-btn');
        if (modsBtn) {
            modsBtn.style.display = data.is_mod ? 'block' : 'none';
        }
        
        if (typeof updateAuthUI === 'function') {
            updateAuthUI();
        }
        
        if (data.is_mod) {
            loadModList();
            loadIpBans();
            if (typeof loadUndefinedWords === 'function') {
                loadUndefinedWords();
            }
            document.querySelectorAll('.mod-only-btn').forEach(btn => btn.style.display = 'inline-block');
        } else {
            document.querySelectorAll('.mod-only-btn').forEach(btn => btn.style.display = 'none');
            if (modsBtn) {
                modsBtn.style.display = 'none';
            }
        }
    } catch (err) {
        console.error("Error checking mod status:", err);
    }
}

const _modStatusTimeouts = {};

function showModStatus(message, isError = false, targetId = 'mod-status-area') {
    const statusArea = document.getElementById(targetId);
    if (!statusArea) {
        alert(message);
        return;
    }
    
    if (_modStatusTimeouts[targetId]) {
        clearTimeout(_modStatusTimeouts[targetId]);
        delete _modStatusTimeouts[targetId];
    }
    
    statusArea.style.transition = '';
    statusArea.textContent = message;
    statusArea.style.color = isError ? '#f43f5e' : (message.includes('Adding') || message.includes('Removing')) ? '#38bdf8' : '#4ade80';
    statusArea.style.opacity = '1';
    
    // Clear after 5 seconds if not a loading message
    if (!message.endsWith('...')) {
        _modStatusTimeouts[targetId] = setTimeout(() => {
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
}


async function loadModList() {
    const listEl = document.getElementById('mod-list-container');
    if (!listEl) return;

    const isJeffb = Boolean(window.currentUserIsRootMod) || 
                    (window.currentUser || localStorage.getItem('morpheme_username') || '').toLowerCase().trim() === 'jeffb';

    const addSection = document.querySelector('#mod-tab-access .mod-add-section');
    if (addSection) {
        addSection.style.display = isJeffb ? 'flex' : 'none';
    }

    const descEl = document.querySelector('#mod-tab-access .tool-header p');
    if (descEl && !isJeffb) {
        descEl.textContent = "View active moderators. Only jeffb can add or remove moderators.";
    }

    try {
        const response = await fetch('/api/mods/list');
        const data = await response.json();
        if (data.mods) {
            listEl.innerHTML = data.mods.map(m => `
                <div class="mod-item">
                    <span class="mod-name">${m}</span>
                    ${isJeffb && !['jeffb', 'system'].includes(m.toLowerCase()) ? `<button class="remove-mod-btn" onclick="removeModerator('${m}')" title="Remove Moderator">&times;</button>` : ''}
                </div>
            `).join('');
        }
    } catch (err) {
        console.error("Error loading mod list:", err);
    }
}


async function addModerator() {
    const isJeffb = Boolean(window.currentUserIsRootMod) || 
                    (window.currentUser || localStorage.getItem('morpheme_username') || '').toLowerCase().trim() === 'jeffb';
    if (!isJeffb) {
        showModStatus("Unauthorized: Only jeffb can add moderators.", true);
        return;
    }

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
    const isJeffb = Boolean(window.currentUserIsRootMod) || 
                    (window.currentUser || localStorage.getItem('morpheme_username') || '').toLowerCase().trim() === 'jeffb';
    if (!isJeffb) {
        showModStatus("Unauthorized: Only jeffb can remove moderators.", true);
        return;
    }

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
    const wordInput = document.getElementById('added-word-input');
    const rawVal = wordInput ? wordInput.value.trim() : '';

    if (!rawVal) {
        showModStatus('Please enter a word.', true, 'added-word-status-area');
        if (wordInput) wordInput.focus();
        return;
    }

    // Letters, commas, spaces allowed
    if (!/^[A-Za-z,\s]+$/.test(rawVal)) {
        showModStatus('❌ Invalid entry: only letters (A–Z) and commas are allowed.', true, 'added-word-status-area');
        if (wordInput) wordInput.focus();
        return;
    }

    // Instantly clear input and maintain focus for continuous typing flow
    if (wordInput) {
        wordInput.value = '';
        wordInput.focus();
    }

    showModStatus(`Adding word(s)...`, false, 'added-word-status-area');

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 20000);

    try {
        const response = await fetch('/api/mods/added_words/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: rawVal }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        const data = await response.json();

        if (data.success) {
            const msg = data.message || `Word(s) added successfully.`;
            showModStatus(`✅ ${msg}`, false, 'added-word-status-area');
            if (window.loadAddedWords) window.loadAddedWords('added');
        } else {
            const errorMsg = data.error || "Failed to add word.";
            showModStatus(`❌ ${errorMsg}`, true, 'added-word-status-area');
        }
        if (wordInput) wordInput.focus();
    } catch (err) {
        clearTimeout(timeoutId);
        console.error("Error adding word:", err);
        const errMsg = (err.name === 'AbortError') ? '❌ Request timed out adding word.' : '❌ Network error adding word.';
        showModStatus(errMsg, true, 'added-word-status-area');
        if (wordInput) wordInput.focus();
    }
}

async function removeAddedWord() {
    const wordInput = document.getElementById('added-word-input');
    const rawVal = wordInput ? wordInput.value.trim() : '';

    if (!rawVal) {
        showModStatus('Please enter a word to remove.', true, 'added-word-status-area');
        if (wordInput) wordInput.focus();
        return;
    }

    // Letters, commas, spaces allowed
    if (!/^[A-Za-z,\s]+$/.test(rawVal)) {
        showModStatus('❌ Invalid entry: only letters (A–Z) and commas are allowed.', true, 'added-word-status-area');
        if (wordInput) wordInput.focus();
        return;
    }

    // Instantly clear input and keep focus for lightning-fast consecutive removal
    if (wordInput) {
        wordInput.value = '';
        wordInput.focus();
    }

    showModStatus(`Removing word(s)...`, false, 'added-word-status-area');

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 20000);

    try {
        const response = await fetch('/api/mods/added_words/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: rawVal }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        const data = await response.json();

        if (data.success) {
            const msg = data.message || `Word(s) removed from Added Words list.`;
            showModStatus(`🗑️ ${msg}`, false, 'added-word-status-area');
            if (window.loadAddedWords) window.loadAddedWords('added');
        } else {
            showModStatus(`❌ ${data.error || "Failed to remove word."}`, true, 'added-word-status-area');
        }
        if (wordInput) wordInput.focus();
    } catch (err) {
        clearTimeout(timeoutId);
        console.error("Error removing added word:", err);
        const errMsg = (err.name === 'AbortError') ? '❌ Request timed out removing word.' : '❌ Network error removing word.';
        showModStatus(errMsg, true, 'added-word-status-area');
        if (wordInput) wordInput.focus();
    }
}
window.addAddedWord = addAddedWord;
window.removeAddedWord = removeAddedWord;
window.loadAddedWordsConfig = loadAddedWordsConfig;

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
    const previewEl = document.getElementById('current-notice-preview');
    const badgeEl = document.getElementById('current-notice-badge');
    if (!input) return;
    
    try {
        const response = await fetch('/api/mods/lobby-notice');
        const data = await response.json();
        if (data.notice && data.notice.trim() !== '') {
            input.value = data.notice;
            if (previewEl) {
                previewEl.textContent = data.notice;
                previewEl.style.fontStyle = 'normal';
                previewEl.style.opacity = '1';
            }
            if (badgeEl) {
                badgeEl.textContent = 'Active Notice';
                badgeEl.style.background = 'rgba(16, 185, 129, 0.2)';
                badgeEl.style.color = '#34d399';
            }
        } else {
            input.value = '';
            if (previewEl) {
                previewEl.textContent = 'No announcement is currently set. Type a message below and click "Broadcast Notice" to publish.';
                previewEl.style.fontStyle = 'italic';
                previewEl.style.opacity = '0.7';
            }
            if (badgeEl) {
                badgeEl.textContent = 'No Active Notice';
                badgeEl.style.background = 'rgba(255, 255, 255, 0.1)';
                badgeEl.style.color = 'var(--text-secondary)';
            }
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
            if (notice) {
                showModStatus("Lobby announcement broadcasted successfully!", false, 'notice-status-area');
            } else {
                showModStatus("Lobby notice cleared successfully.", false, 'notice-status-area');
            }
            await loadLobbyNotice();
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
        // Real-time: strip invalid characters as the user types (letters, commas, spaces allowed).
        addedWordInput.addEventListener('input', () => {
            const before = addedWordInput.value;
            const after = before.replace(/[^A-Za-z,\s]/g, '');
            if (before !== after) {
                // Preserve cursor position after stripping
                const sel = addedWordInput.selectionStart - (before.length - after.length);
                addedWordInput.value = after;
                addedWordInput.setSelectionRange(Math.max(0, sel), Math.max(0, sel));
                showModStatus('❌ Only letters (A–Z) and commas are allowed.', true, 'added-word-status-area');
            }
        });
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

    const refreshUndefBtn = document.getElementById('refresh-undef-btn');
    if (refreshUndefBtn) {
        refreshUndefBtn.addEventListener('click', () => loadUndefinedWords(true));
    }

    const undefSearchInput = document.getElementById('undef-search-input');
    if (undefSearchInput) {
        undefSearchInput.addEventListener('input', renderUndefinedWords);
    }

    const undefScrollArea = document.querySelector('.undef-words-list-scroll');
    if (undefScrollArea) {
        undefScrollArea.addEventListener('scroll', () => {
            if (undefScrollArea.scrollTop + undefScrollArea.clientHeight >= undefScrollArea.scrollHeight - 150) {
                appendUndefBatch();
            }
        });
    }

    // Ban / Timeout User
    const timeoutUserBtn = document.getElementById('timeout-user-btn');
    if (timeoutUserBtn) {
        timeoutUserBtn.addEventListener('click', timeoutUser);
    }

    const checkTimeoutBtn = document.getElementById('check-timeout-btn');
    if (checkTimeoutBtn) {
        checkTimeoutBtn.addEventListener('click', checkTimeoutStatus);
    }

    const liftTimeoutBtn = document.getElementById('lift-timeout-btn');
    if (liftTimeoutBtn) {
        liftTimeoutBtn.addEventListener('click', liftTimeout);
    }

    const banUserBtn = document.getElementById('ban-user-btn');
    if (banUserBtn) {
        banUserBtn.addEventListener('click', banUser);
    }

    const refreshIpBansBtn = document.getElementById('refresh-ip-bans-btn');
    if (refreshIpBansBtn) {
        refreshIpBansBtn.addEventListener('click', loadIpBans);
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
    const dictUploadWrapper = document.getElementById('dict-upload-wrapper');

    if (dictTriggerBtn && dictFileInput) {
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

    if (dictUploadWrapper && dictFileInput) {
        dictUploadWrapper.addEventListener('dragover', (e) => {
            e.preventDefault();
            if (dictTriggerBtn) {
                dictTriggerBtn.style.background = 'rgba(var(--text-primary-rgb), 0.15)';
                dictTriggerBtn.style.borderColor = 'var(--accent-color)';
            }
        });
        dictUploadWrapper.addEventListener('dragleave', () => {
            if (dictTriggerBtn) {
                dictTriggerBtn.style.background = 'var(--input-bg)';
                dictTriggerBtn.style.borderColor = 'var(--input-border)';
            }
        });
        dictUploadWrapper.addEventListener('drop', (e) => {
            e.preventDefault();
            if (dictTriggerBtn) {
                dictTriggerBtn.style.background = 'var(--input-bg)';
                dictTriggerBtn.style.borderColor = 'var(--input-border)';
            }
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                dictFileInput.files = e.dataTransfer.files;
                dictFileInput.dispatchEvent(new Event('change'));
            }
        });
    }

    if (dictSubmitBtn) {
        dictSubmitBtn.addEventListener('click', submitDictionaryToDatabase);
    }
});

async function timeoutUser() {
    const input = document.getElementById('timeout-username-input');
    const username = input ? input.value.trim() : '';
    if (!username) {
        alert("Please enter a username to timeout.");
        return;
    }

    const reasonInput = document.getElementById('timeout-reason-input');
    const reason = reasonInput ? reasonInput.value.trim() : '';

    const hoursInput = document.getElementById('timeout-hours-input');
    const hoursVal = hoursInput ? hoursInput.value.trim() : '';

    if (['jeffbabiak', 'jeffb', 'system'].includes(username.toLowerCase())) {
        alert(`Action Prohibited: User '${username}' cannot be timed out.`);
        return;
    }

    const durPrompt = hoursVal ? `\nDuration: ${hoursVal} hour(s)` : '';
    const reasonPrompt = reason ? `\nReason: "${reason}"` : '';
    if (!confirm(`Are you sure you want to timeout user "${username}"?${durPrompt}${reasonPrompt}\n\nThey will be evicted from their current room and temporarily banned from playing in all rooms.`)) {
        return;
    }

    try {
        const response = await fetch('/api/mods/timeout_user', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                username, 
                reason: reason || 'Moderator timeout',
                hours: hoursVal || null
            })
        });
        const data = await response.json();
        if (data.success) {
            showModStatus(data.message, false, 'ban-status-area');
            const infoEl = document.getElementById('timeout-info-display');
            if (infoEl) {
                const rDisplay = reason ? ` | Reason: <em>${reason}</em>` : '';
                infoEl.innerHTML = `✅ <strong>${username}</strong> timed out for <strong>${data.duration}</strong> (Offense #${data.offense_count})${rDisplay}. Until: ${data.timeout_until} UTC`;
            }
            alert(`User "${username}" has been timed out for ${data.duration}.`);
        } else {
            showModStatus(data.error || "Failed to timeout user.", true, 'ban-status-area');
            alert("Error: " + (data.error || "Failed to timeout user."));
        }
    } catch (err) {
        console.error("Error timing out user:", err);
        showModStatus("Network error timing out user.", true, 'ban-status-area');
    }
}

async function checkTimeoutStatus() {
    const input = document.getElementById('timeout-username-input');
    const username = input ? input.value.trim() : '';
    if (!username) {
        alert("Please enter a username to check timeout status.");
        return;
    }

    try {
        const response = await fetch(`/api/mods/user_timeout_status/${encodeURIComponent(username)}`);
        const data = await response.json();
        const infoEl = document.getElementById('timeout-info-display');
        if (data.error) {
            showModStatus(data.error, true, 'ban-status-area');
            if (infoEl) infoEl.innerHTML = `<span style="color: #f43f5e;">❌ ${data.error}</span>`;
            return;
        }

        let statusHtml = `<strong>Status for ${data.username}:</strong><br>`;
        if (data.is_timed_out) {
            statusHtml += `<span style="color: #f59e0b;">⏱️ CURRENTLY TIMED OUT</span> — Remaining: <strong>${data.remaining}</strong> (Until: ${data.timeout_until} UTC)<br>`;
            if (data.reason) {
                statusHtml += `Reason: <strong>${data.reason}</strong><br>`;
            }
        } else {
            statusHtml += `<span style="color: #10b981;">✅ Active (Not timed out)</span><br>`;
        }
        statusHtml += `Total Offenses on Record: <strong>${data.offense_count}</strong> | Effective Offense Level (after decay): <strong>${data.effective_offenses}</strong><br>`;
        statusHtml += `Next Offense Timeout Duration: <strong>${data.next_duration}</strong>`;

        if (infoEl) infoEl.innerHTML = statusHtml;
        showModStatus(`Status fetched for ${data.username}`, false, 'ban-status-area');
    } catch (err) {
        console.error("Error checking timeout status:", err);
        showModStatus("Network error checking timeout status.", true, 'ban-status-area');
    }
}

async function liftTimeout() {
    const input = document.getElementById('timeout-username-input');
    const username = input ? input.value.trim() : '';
    if (!username) {
        alert("Please enter a username to lift timeout.");
        return;
    }

    if (!confirm(`Lift timeout for "${username}" and reset their offense record back to 10 minutes?`)) {
        return;
    }

    try {
        const response = await fetch('/api/mods/lift_timeout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username })
        });
        const data = await response.json();
        if (data.success) {
            showModStatus(data.message, false, 'ban-status-area');
            const infoEl = document.getElementById('timeout-info-display');
            if (infoEl) infoEl.innerHTML = `🔓 <strong>${username}</strong> timeout has been lifted and reset back to <strong>10 minutes</strong>.`;
            alert(data.message);
        } else {
            showModStatus(data.error || "Failed to lift timeout.", true, 'ban-status-area');
            alert("Error: " + (data.error || "Failed to lift timeout."));
        }
    } catch (err) {
        console.error("Error lifting timeout:", err);
        showModStatus("Network error lifting timeout.", true, 'ban-status-area');
    }
}

async function banUser() {
    const input = document.getElementById('ban-username-input');
    const reasonInput = document.getElementById('ban-reason-input');
    const username = input ? input.value.trim() : '';
    const reason = reasonInput ? reasonInput.value.trim() : '';
    if (!username) {
        alert("Please enter a username to ban.");
        return;
    }

    if (['jeffbabiak', 'jeffb', 'system'].includes(username.toLowerCase())) {
        alert(`Action Prohibited: User '${username}' cannot be banned.`);
        return;
    }

    if (!confirm(`ARE YOU SURE? This will permanently ERASE all data for user "${username}". This cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch('/api/mods/ban_user', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, reason })
        });
        const data = await response.json();
        if (data.success) {
            if (input) input.value = '';
            if (reasonInput) reasonInput.value = '';
            showModStatus(data.message, false, 'ban-status-area');
            alert(data.message || `User "${username}" has been permanently erased from the database.`);
            loadIpBans();
        } else {
            showModStatus(data.error || "Failed to ban user.", true, 'ban-status-area');
            alert("Error: " + (data.error || "Failed to ban user."));
        }
    } catch (err) {
        console.error("Error banning user:", err);
        showModStatus("Network error banning user.", true, 'ban-status-area');
    }
}

// --- IP BAN MANAGEMENT ---
async function loadIpBans() {
    const tbody = document.getElementById('ip-bans-tbody');
    const tableWrap = document.getElementById('ip-bans-table-wrap');
    const emptyEl = document.getElementById('ip-bans-empty');
    const countEl = document.getElementById('ip-bans-count');
    if (!tbody) return;

    try {
        const response = await fetch('/api/mods/ip_bans');
        const data = await response.json();
        if (data.success && Array.isArray(data.bans)) {
            if (countEl) countEl.textContent = data.bans.length;
            if (data.bans.length === 0) {
                if (emptyEl) emptyEl.style.display = 'block';
                if (tableWrap) tableWrap.style.display = 'none';
                tbody.innerHTML = '';
            } else {
                if (emptyEl) emptyEl.style.display = 'none';
                if (tableWrap) tableWrap.style.display = 'block';
                tbody.innerHTML = data.bans.map(b => `
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <td style="padding: 8px; font-family: monospace; font-weight: 600; color: #f43f5e;">${escapeHtml(b.ip_address)}</td>
                        <td style="padding: 8px; color: var(--text-secondary);">${escapeHtml(b.banned_username || 'Manual / Unknown')}</td>
                        <td style="padding: 8px; color: var(--text-secondary);">${escapeHtml(b.banned_by || 'Moderator')}</td>
                        <td style="padding: 8px; color: var(--text-secondary);">${escapeHtml(b.reason || '-')}</td>
                        <td style="padding: 8px; font-size: 0.8rem; color: var(--text-secondary);">${typeof window.formatAppDate === 'function' ? window.formatAppDate(b.created_at, true) : escapeHtml(b.created_at || '-')}</td>
                        <td style="padding: 8px; text-align: right;">
                            <button class="mini-action-btn secondary" style="background: #059669; border-color: #047857; color: #fff; padding: 3px 8px; font-size: 0.75rem;" onclick="liftIpBan('${escapeHtml(b.ip_address)}', ${b.id})">🔓 Lift Ban</button>
                        </td>
                    </tr>
                `).join('');
            }
        }
    } catch (err) {
        console.error("Error loading IP bans:", err);
    }
}

async function liftIpBan(ip, id) {
    if (!confirm(`Are you sure you want to lift the IP ban for "${ip}"?`)) {
        return;
    }

    try {
        const response = await fetch('/api/mods/lift_ip_ban', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ip_address: ip, id: id })
        });
        const data = await response.json();
        if (data.success) {
            showModStatus(data.message || `IP ban lifted for ${ip}.`, false, 'ban-status-area');
            loadIpBans();
        } else {
            alert("Error: " + (data.error || "Failed to lift IP ban."));
        }
    } catch (err) {
        console.error("Error lifting IP ban:", err);
        showModStatus("Network error lifting IP ban.", true, 'ban-status-area');
    }
}

window.loadIpBans = loadIpBans;
window.liftIpBan = liftIpBan;

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
            const wordsStr = data.words ? data.words.join(', ') : word;
            showModStatus(`Definition for "${wordsStr}" updated.`, false, 'def-status-area');
            alert(`Success: Definition for "${wordsStr}" has been set.`);
            if (typeof loadUndefinedWords === 'function') loadUndefinedWords();
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
            const wordsStr = data.words ? data.words.join(', ') : word;
            showModStatus(`Definition for "${wordsStr}" removed.`, false, 'def-status-area');
            alert(`Success: Definition for "${wordsStr}" has been removed.`);
            if (typeof loadUndefinedWords === 'function') loadUndefinedWords();
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

// Undefined Words Table Support
let undefinedWordsList = [];
let undefRenderedCount = 0;
const UNDEF_BATCH_SIZE = 500;
let currentFilteredUndef = [];

async function loadUndefinedWords(force = false) {
    const container = document.getElementById('undef-words-container');
    if (!container) return;
    
    if (!window.currentUserIsMod) {
        container.style.display = 'none';
        return;
    }
    
    container.style.display = 'block';
    
    const countEl = document.getElementById('undef-words-count');
    const tbody = document.getElementById('undef-words-tbody');
    
    if (countEl) countEl.textContent = '(Loading...)';
    
    try {
        const url = force ? '/api/mods/definitions/undefined?force=true' : '/api/mods/definitions/undefined';
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success && Array.isArray(data.words)) {
            undefinedWordsList = data.words;
            renderUndefinedWords();
        } else {
            if (tbody) tbody.innerHTML = `<tr><td style="padding: 10px; color: #f43f5e; font-size: 0.85rem;">Failed to load words.</td></tr>`;
        }
    } catch (err) {
        console.error("Error loading undefined words:", err);
        if (tbody) tbody.innerHTML = `<tr><td style="padding: 10px; color: #f43f5e; font-size: 0.85rem;">Error loading words.</td></tr>`;
    }
}

function renderUndefinedWords() {
    const countEl = document.getElementById('undef-words-count');
    const tbody = document.getElementById('undef-words-tbody');
    const searchInput = document.getElementById('undef-search-input');
    const query = searchInput ? searchInput.value.trim().toUpperCase() : '';
    
    if (!tbody) return;
    
    currentFilteredUndef = query 
        ? undefinedWordsList.filter(w => w.includes(query))
        : undefinedWordsList;
        
    if (countEl) {
        countEl.textContent = `(${currentFilteredUndef.length.toLocaleString()} undefined)`;
    }
    
    if (currentFilteredUndef.length === 0) {
        tbody.innerHTML = `<tr><td style="padding: 10px; color: rgba(255,255,255,0.4); font-size: 0.85rem; text-align: center;">No undefined words found.</td></tr>`;
        return;
    }
    
    undefRenderedCount = 0;
    tbody.innerHTML = '';
    appendUndefBatch();
}

function appendUndefBatch() {
    const tbody = document.getElementById('undef-words-tbody');
    if (!tbody || undefRenderedCount >= currentFilteredUndef.length) return;
    
    const nextWords = currentFilteredUndef.slice(undefRenderedCount, undefRenderedCount + UNDEF_BATCH_SIZE);
    if (nextWords.length === 0) return;
    
    // Group words by length
    const groups = {};
    nextWords.forEach(word => {
        const len = word.length;
        if (!groups[len]) groups[len] = [];
        groups[len].push(word);
    });
    
    // Sort lengths ascending (fewest letters first)
    const lengths = Object.keys(groups).map(Number).sort((a, b) => a - b);
    
    // Build HTML rows with category headers
    const rowsHTML = [];
    lengths.forEach(len => {
        // Sort words within this length group alphabetically
        groups[len].sort((a, b) => a.localeCompare(b));
        
        // Check if header for this length already exists in DOM
        const existingHeader = tbody.querySelector(`.undef-header-${len}`);
        if (!existingHeader) {
            rowsHTML.push(`
                <tr class="undef-header-${len}" style="background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.08); pointer-events: none;">
                    <td style="padding: 6px 12px; font-weight: 800; font-size: 0.65rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 1px;">${len} Letters</td>
                </tr>
            `);
        }
        
        // Word Rows
        groups[len].forEach(word => {
            rowsHTML.push(`
                <tr onclick="selectUndefinedWord('${word}')" style="border-bottom: 1px solid rgba(255,255,255,0.04); cursor: pointer; transition: background 0.2s;" onmouseover="this.style.background='rgba(255,255,255,0.05)'" onmouseout="this.style.background='transparent'">
                    <td style="padding: 6px 12px; padding-left: 24px; font-weight: 600; font-size: 0.85rem; color: #a5b4fc;">${word}</td>
                </tr>
            `);
        });
    });
    
    tbody.insertAdjacentHTML('beforeend', rowsHTML.join(''));
    undefRenderedCount += nextWords.length;
}

window.selectUndefinedWord = function(word) {
    const wordInput = document.getElementById('def-word-input');
    const textInput = document.getElementById('def-text-input');
    if (wordInput) {
        wordInput.value = word;
    }
    if (textInput) {
        textInput.focus();
    }
};

window.loadUndefinedWords = loadUndefinedWords;
window.renderUndefinedWords = renderUndefinedWords;
window.appendUndefBatch = appendUndefBatch;

window.showModTab = function(tabId) {
    const sidebar = document.querySelector('#page-mods .tools-sidebar');
    const content = document.querySelector('#page-mods .tools-content');
    if (!sidebar || !content) return;

    // Update active class on buttons
    sidebar.querySelectorAll('.tool-nav-btn').forEach(btn => {
        if (btn.dataset.modTab === tabId) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Update active class on panes
    content.querySelectorAll('.tool-pane').forEach(pane => {
        if (pane.id === `mod-tab-${tabId}`) {
            pane.classList.add('active');
        } else {
            pane.classList.remove('active');
        }
    });

    if (tabId === 'ban') {
        loadIpBans();
    } else if (tabId === 'notice') {
        loadLobbyNotice();
    } else if (tabId === 'donations') {
        loadModDonations();
    } else if (tabId === 'def') {
        loadUndefinedWords();
    } else if (tabId === 'access') {
        loadModList();
    } else if (tabId === 'added') {
        if (typeof loadAddedWordsConfig === 'function') loadAddedWordsConfig();
    }

    // Trigger scroll to content area on mobile with smooth sliding animation
    const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    if (isMobile) {
        const layout = document.querySelector('#page-mods .tools-split-layout');
        if (layout) {
            layout.scrollTo({ left: layout.clientWidth || layout.scrollWidth, behavior: 'smooth' });
        }
    }
};

function setupModsNavigation() {
    const sidebar = document.querySelector('#page-mods .tools-sidebar');
    if (!sidebar) return;

    sidebar.addEventListener('click', (e) => {
        const btn = e.target.closest('.tool-nav-btn');
        if (!btn) return;

        const tabId = btn.dataset.modTab;
        if (tabId) {
            window.showModTab(tabId);
        }
    });

    // Mobile Layout snapping on navigation
    const modsPage = document.getElementById('page-mods');
    if (modsPage) {
        const observer = new MutationObserver(() => {
            if (modsPage.classList.contains('active')) {
                const isMobile = (window.innerWidth <= 900) || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
                if (isMobile) {
                    setTimeout(() => {
                        const layoutEl = document.querySelector('#page-mods .tools-split-layout');
                        if (layoutEl) layoutEl.scrollLeft = 0;
                    }, 100);
                }
            }
        });
        observer.observe(modsPage, {
            attributes: true,
            attributeFilter: ['class']
        });
    }


    // Mobile touch swipe handling for sliding back to mods list
    const modsContent = document.querySelector('#page-mods .tools-content');
    const modsSidebar = document.querySelector('#page-mods .tools-sidebar');
    if (modsContent && modsSidebar) {
        let touchStartX = 0;
        let touchStartY = 0;
        modsContent.addEventListener('touchstart', (e) => {
            touchStartX = e.changedTouches[0].screenX;
            touchStartY = e.changedTouches[0].screenY;
        }, { passive: true });
        
        modsContent.addEventListener('touchend', (e) => {
            const touchEndX = e.changedTouches[0].screenX;
            const touchEndY = e.changedTouches[0].screenY;
            const diffX = touchEndX - touchStartX;
            const diffY = touchEndY - touchStartY;
            
            // If swiped right (diffX > 80) and horizontal movement was dominant
            if (diffX > 80 && Math.abs(diffX) > Math.abs(diffY)) {
                const layoutEl = document.querySelector('#page-mods .tools-split-layout');
                if (layoutEl) layoutEl.scrollTo({ left: 0, behavior: 'smooth' });
            }
        }, { passive: true });
    }

    // Mobile back button inside mods content
    const mobileBackBtn = document.getElementById('mods-mobile-back-btn');
    if (mobileBackBtn) {
        mobileBackBtn.addEventListener('click', () => {
            const layoutEl = document.querySelector('#page-mods .tools-split-layout');
            if (layoutEl) layoutEl.scrollTo({ left: 0, behavior: 'smooth' });
        });
    }
    // Setup donation management listeners
    const addDonationBtn = document.getElementById('add-donation-btn');
    if (addDonationBtn) {
        addDonationBtn.addEventListener('click', addModDonation);
    }
    const refreshDonationsBtn = document.getElementById('refresh-donations-btn');
    if (refreshDonationsBtn) {
        refreshDonationsBtn.addEventListener('click', loadModDonations);
    }
    const donorAmountInput = document.getElementById('mod-donor-amount');
    if (donorAmountInput) {
        donorAmountInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                addModDonation();
            }
        });
    }
    const donorNameInput = document.getElementById('mod-donor-name');
    if (donorNameInput) {
        donorNameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                addModDonation();
            }
        });
    }
}

async function loadModDonations() {
    const emptyEl = document.getElementById('mod-donations-empty');
    const tableWrap = document.getElementById('mod-donations-table-wrap');
    const tbody = document.getElementById('mod-donations-tbody');
    const countEl = document.getElementById('mod-donations-count');
    if (!tbody) return;

    try {
        const res = await fetch('/api/mods/donations/list');
        const data = await res.json();
        const donations = data.donations || [];
        if (countEl) countEl.innerText = donations.length;

        if (donations.length === 0) {
            if (emptyEl) emptyEl.style.display = 'block';
            if (tableWrap) tableWrap.style.display = 'none';
            tbody.innerHTML = '';
            return;
        }

        if (emptyEl) emptyEl.style.display = 'none';
        if (tableWrap) tableWrap.style.display = 'block';

        tbody.innerHTML = donations.map(d => {
            const dateStr = typeof window.formatAppDate === 'function' ? window.formatAppDate(d.timestamp, true) : (d.timestamp || '');
            const isAnon = Boolean(d.is_anonymous);
            return `
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px; color: var(--text-secondary); white-space: nowrap;">${dateStr}</td>
                    <td style="padding: 8px; font-weight: 700; color: #fff;">${escapeHTML(d.donor_name || 'Anonymous')}</td>
                    <td style="padding: 8px; font-weight: 800; color: #4ade80;">$${parseFloat(d.amount).toFixed(2)}</td>
                    <td style="padding: 8px; color: ${isAnon ? '#f43f5e' : 'var(--text-secondary)'};">${isAnon ? 'Yes (Anon)' : 'Public'}</td>
                    <td style="padding: 8px;"><span style="font-size: 0.75rem; padding: 2px 8px; border-radius: 10px; background: rgba(74, 222, 128, 0.15); color: #4ade80; font-weight: 600;">Confirmed</span></td>
                    <td style="padding: 8px; text-align: right;">
                        <button class="mini-action-btn remove" onclick="deleteModDonation(${d.id})" style="padding: 3px 8px; font-size: 0.75rem; background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 4px; cursor: pointer;">🗑️ Delete</button>
                    </td>
                </tr>
            `;
        }).join('');
    } catch (err) {
        console.error("Error loading mod donations:", err);
    }
}

async function addModDonation() {
    const nameInput = document.getElementById('mod-donor-name');
    const amountInput = document.getElementById('mod-donor-amount');
    const anonCheckbox = document.getElementById('mod-donor-anonymous');

    const donor_name = nameInput ? nameInput.value.trim() : '';
    const amount = amountInput ? parseFloat(amountInput.value) : 0;
    const is_anonymous = anonCheckbox ? anonCheckbox.checked : false;

    if (!donor_name) {
        showModStatus("Please enter a donor name or username.", true, 'donation-status-area');
        return;
    }
    if (!amount || isNaN(amount) || amount <= 0) {
        showModStatus("Please enter a valid donation amount greater than $0.", true, 'donation-status-area');
        return;
    }

    try {
        const res = await fetch('/api/mods/donations/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ donor_name, amount, is_anonymous })
        });
        const data = await res.json();
        if (data.success) {
            showModStatus(`Successfully recorded $${amount.toFixed(2)} donation for "${donor_name}"!`, false, 'donation-status-area');
            if (nameInput) nameInput.value = '';
            if (amountInput) amountInput.value = '';
            if (anonCheckbox) anonCheckbox.checked = false;
            loadModDonations();
            if (typeof window.initDonatePage === 'function') {
                window.initDonatePage();
            }
        } else {
            showModStatus(data.error || "Failed to record donation.", true, 'donation-status-area');
        }
    } catch (err) {
        console.error("Error adding donation:", err);
        showModStatus("Network error recording donation.", true, 'donation-status-area');
    }
}

async function deleteModDonation(id) {
    if (!confirm("Are you sure you want to delete this donation record? This will remove it from the Hall of Fame and progress meter.")) {
        return;
    }

    try {
        const res = await fetch('/api/mods/donations/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id })
        });
        const data = await res.json();
        if (data.success) {
            showModStatus("Donation record deleted.", false, 'donation-status-area');
            loadModDonations();
            if (typeof window.initDonatePage === 'function') {
                window.initDonatePage();
            }
        } else {
            showModStatus(data.error || "Failed to delete donation.", true, 'donation-status-area');
        }
    } catch (err) {
        console.error("Error deleting donation:", err);
        showModStatus("Network error deleting donation.", true, 'donation-status-area');
    }
}

window.loadModDonations = loadModDonations;
window.addModDonation = addModDonation;
window.deleteModDonation = deleteModDonation;

// Add event listener to initialize setupModsNavigation when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    setupModsNavigation();
});


