
async function checkModStatus() {
    try {
        const response = await fetch('/api/mods/status');
        const data = await response.json();
        const modsBtn = document.getElementById('nav-mods-btn');
        if (data.is_mod) {
            if (modsBtn) modsBtn.classList.remove('hidden');
            loadModList();
        } else {
            if (modsBtn) modsBtn.classList.add('hidden');
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

// Global initialization
document.addEventListener('DOMContentLoaded', () => {
    checkModStatus();
    
    const addBtn = document.getElementById('add-mod-btn');
    if (addBtn) {
        addBtn.addEventListener('click', addModerator);
    }
});

// Re-check on login/logout events if they exist
window.addEventListener('user-login', checkModStatus);
