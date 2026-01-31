document.addEventListener('DOMContentLoaded', () => {
    setupToolsNavigation();
    setupComboChecker();
});

function setupToolsNavigation() {
    const navBtns = document.querySelectorAll('.tool-nav-btn');
    const panes = document.querySelectorAll('.tool-pane');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update buttons
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Show pane
            const toolId = btn.dataset.tool; // e.g. "combo"
            panes.forEach(p => p.classList.remove('active'));
            const targetPane = document.getElementById(`tool-${toolId}`);
            if (targetPane) targetPane.classList.add('active');
        });
    });
}

function setupComboChecker() {
    const searchBtn = document.getElementById('combo-search-btn');
    const input = document.getElementById('combo-input');

    if (searchBtn) {
        searchBtn.addEventListener('click', runComboSearch);
    }

    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runComboSearch();
        });
    }
}

async function runComboSearch() {
    const inputEl = document.getElementById('combo-input');
    const dictEl = document.getElementById('combo-dict');
    const resultsContainer = document.getElementById('combo-results');

    const searchTerm = inputEl.value.trim();
    const dictionary = dictEl.value;

    if (!searchTerm) return;

    // Clear previous results
    document.getElementById('mp-container').innerHTML = '';
    document.getElementById('lic-container').innerHTML = '';

    resultsContainer.classList.remove('hidden');

    try {
        const response = await fetch('/api/tools/combo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ search_term: searchTerm, dictionary: dictionary })
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Render MP Groups (0MP to 5MP)
        renderGroups(data.mp_groups, 'mp-container', 'MP');

        // Render LIC Groups (Shared Count)
        renderGroups(data.lic_groups, 'lic-container', 'LIC');

    } catch (error) {
        console.error('Combo check failed:', error);
        alert('An error occurred while checking combo.');
    }
}

function renderGroups(groupsData, containerId, type) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Sort keys logically
    // MP keys are 0, 1, 2... (Integers)
    // LIC keys are Lengths (Integers)
    const keys = Object.keys(groupsData).map(Number).sort((a, b) => a - b);

    keys.forEach(key => {
        const words = groupsData[key];
        if (words.length === 0) return; // Skip empty groups? Or show empty columns? User implies "0MP...5MP", better skip empty to save space or show if desired. Let's skip empty for now.

        // Label logic
        let label = '';
        if (type === 'MP') {
            label = `${key}MP`; // e.g. 0MP (0 Ops)
        } else {
            label = `${key} Shared`; // e.g. 5 Shared
        }

        const colDiv = document.createElement('div');
        colDiv.className = 'group-column';

        colDiv.innerHTML = `
            <div class="group-header">${label}</div>
            <div class="group-table-container">
                <table class="group-table">
                    <tbody>
                        ${words.map(w => `<tr><td>${w}</td></tr>`).join('')}
                    </tbody>
                </table>
            </div>
        `;

        container.appendChild(colDiv);
    });
}
