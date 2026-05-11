/* --- PREMIUM DONATE PAGE JS CONTROLLER --- */

(function() {
    console.log('[donate.js] Initializing Morpheme Support module...');

    window.initDonatePage = function() {
        console.log('[donate.js] initDonatePage invoked.');
        
        // 1. Reset progress bar fill and animate it beautifully on entry
        const progressFill = document.querySelector('.progress-bar-fill');
        if (progressFill) {
            progressFill.style.width = '0%';
            setTimeout(() => {
                progressFill.style.width = '78%'; // Target funding percentage
            }, 150);
        }

        // 2. Setup tier support buttons
        const tierButtons = document.querySelectorAll('.donate-tier-btn');
        const customAmountInput = document.getElementById('custom-donate-amount');
        
        tierButtons.forEach(btn => {
            // Remove previous listeners
            const newBtn = btn.cloneNode(true);
            btn.parentNode.replaceChild(newBtn, btn);

            newBtn.addEventListener('click', () => {
                const amount = newBtn.getAttribute('data-amount');
                if (customAmountInput) {
                    customAmountInput.value = amount;
                    // Trigger flash effect on the input
                    customAmountInput.style.borderColor = 'var(--accent-color, #f43f5e)';
                    customAmountInput.style.boxShadow = '0 0 20px rgba(244, 63, 94, 0.4)';
                    setTimeout(() => {
                        customAmountInput.style.borderColor = '';
                        customAmountInput.style.boxShadow = '';
                    }, 500);
                }
                
                showToast(`Support tier $${amount} selected! Please choose a channel below to proceed.`, "💖");
                
                // Smooth scroll down to payment methods card
                const paymentCard = document.querySelector('.payment-methods-card');
                if (paymentCard) {
                    paymentCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            });
        });

        // 3. Setup custom amount submission
        const customSubmit = document.getElementById('custom-donate-submit');
        if (customSubmit) {
            const newSubmit = customSubmit.cloneNode(true);
            customSubmit.parentNode.replaceChild(newSubmit, customSubmit);

            newSubmit.addEventListener('click', () => {
                if (customAmountInput) {
                    const amount = parseFloat(customAmountInput.value);
                    if (isNaN(amount) || amount <= 0) {
                        showToast("Please enter a valid support amount.", "⚠️");
                        return;
                    }
                    showToast(`Custom amount $${amount} chosen! Select a support channel below to continue.`, "💖");
                    
                    // Smooth scroll down to payment methods card
                    const paymentCard = document.querySelector('.payment-methods-card');
                    if (paymentCard) {
                        paymentCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                }
            });
        }
    };

    // Helper to trigger standard toasts
    function showToast(message, icon = "🔔") {
        if (typeof window.showToastNotification === 'function') {
            window.showToastNotification(message, icon);
            return;
        }

        // Fallback simple toast
        const toast = document.createElement('div');
        toast.className = 'm-toast';
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">${icon}</span>
                <div class="toast-body">
                    <p>${message}</p>
                </div>
            </div>
        `;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('hiding');
            setTimeout(() => toast.remove(), 400);
        }, 4000);
    }
})();
