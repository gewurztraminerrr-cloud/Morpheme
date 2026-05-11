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

        // 2. Setup PayPal URL mapping
        const basePayPalUrl = 'https://paypal.me/jeffbabiak';
        const paypalBtn = document.querySelector('.payment-btn.paypal');
        
        function updatePayPalLink(amount) {
            if (paypalBtn) {
                if (amount && amount > 0) {
                    paypalBtn.href = `${basePayPalUrl}/${amount}`;
                } else {
                    paypalBtn.href = basePayPalUrl;
                }
            }
        }

        // Initialize PayPal link with default custom amount input value ($10)
        const customAmountInput = document.getElementById('custom-donate-amount');
        if (customAmountInput) {
            const initialAmount = parseFloat(customAmountInput.value) || 10;
            updatePayPalLink(initialAmount);

            // Listen to real-time input typing to update PayPal URL dynamically
            customAmountInput.addEventListener('input', () => {
                const amount = parseFloat(customAmountInput.value);
                if (!isNaN(amount) && amount > 0) {
                    updatePayPalLink(amount);
                } else {
                    updatePayPalLink(null);
                }
            });
        }

        // 3. Setup tier support buttons
        const tierButtons = document.querySelectorAll('.donate-tier-btn');
        
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
                
                // Update the PayPal link dynamically to matching tier amount!
                updatePayPalLink(amount);
                
                showToast(`Support tier $${amount} selected! PayPal button updated to $${amount}.`, "💖");
                
                // Smooth scroll down to payment methods card
                const paymentCard = document.querySelector('.payment-methods-card');
                if (paymentCard) {
                    paymentCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            });
        });

        // 4. Setup custom amount submission
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
                    
                    updatePayPalLink(amount);
                    showToast(`Custom amount $${amount} chosen! PayPal button updated to $${amount}.`, "💖");
                    
                    // Smooth scroll down to payment methods card
                    const paymentCard = document.querySelector('.payment-methods-card');
                    if (paymentCard) {
                        paymentCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                }
            });
        }

        // 5. Fetch and Render Real Recent Donators from SQLite DB
        loadRecentDonations();
    };

    function loadRecentDonations() {
        const hofGrid = document.querySelector('.hof-grid');
        if (!hofGrid) return;

        fetch('/api/donations/recent')
            .then(res => res.json())
            .then(data => {
                if (data.donations && data.donations.length > 0) {
                    // Render actual donations from DB
                    hofGrid.innerHTML = data.donations.map(donation => {
                        const amount = parseFloat(donation.amount);
                        let tierClass = 'bronze-tier';
                        let tierName = 'Bronze Supporter';
                        let avatar = '🛡️';

                        if (amount >= 30) {
                            tierClass = 'gold-tier';
                            tierName = 'Gold Supporter';
                            avatar = '👑';
                        } else if (amount >= 15) {
                            tierClass = 'silver-tier';
                            tierName = 'Silver Supporter';
                            avatar = '✨';
                        }

                        // Format timestamp nicely
                        let dateStr = '';
                        try {
                            const date = new Date(donation.timestamp);
                            dateStr = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
                        } catch (e) {
                            dateStr = donation.timestamp;
                        }

                        return `
                            <div class="hof-item">
                                <div class="hof-avatar">${avatar}</div>
                                <div class="hof-info">
                                    <div class="hof-name">${escapeHTML(donation.donor_name)}</div>
                                    <div class="hof-tier ${tierClass}">${tierName} • $${amount}</div>
                                    <div style="font-size:0.65rem; color:var(--muted-text); margin-top:2px;">${dateStr}</div>
                                </div>
                            </div>
                        `;
                    }).join('');
                } else {
                    // If DB has no records, show premium placeholder instructions
                    hofGrid.innerHTML = `
                        <div class="hof-item" style="grid-column: 1 / -1; justify-content: center; padding: 25px; border-style: dashed; border-color: rgba(255,255,255,0.15); background: rgba(0,0,0,0.15);">
                            <div style="text-align: center;">
                                <div style="font-size: 1.5rem; margin-bottom: 8px;">🤝</div>
                                <div style="font-weight: 700; color: #fff; margin-bottom: 4px;">No Donations Yet</div>
                                <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0;">Be the very first player to support Morpheme and claim your spot on the Hall of Fame!</p>
                            </div>
                        </div>
                    `;
                }
            })
            .catch(err => {
                console.error('[donate.js] Error loading donations:', err);
            });
    }

    // Helper to escape HTML to prevent XSS
    function escapeHTML(str) {
        if (!str) return '';
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    }

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
