/* --- PREMIUM DONATE PAGE JS CONTROLLER --- */

(function() {
    console.log('[donate.js] Initializing Morpheme Support module...');

    window.initDonatePage = function() {
        console.log('[donate.js] initDonatePage invoked.');
        
        // 1. Setup PayPal URL mapping
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

            // Bulletproof backup: Update link on actual click to capture latest input value
            if (paypalBtn) {
                paypalBtn.addEventListener('click', () => {
                    const amount = parseFloat(customAmountInput.value);
                    if (!isNaN(amount) && amount > 0) {
                        paypalBtn.href = `${basePayPalUrl}/${amount}`;
                    } else {
                        paypalBtn.href = basePayPalUrl;
                    }
                });
            }
        }

        // 2. Fetch and Render Real Recent Donators and Monthly Progress
        loadRecentDonations();
    };

    function loadRecentDonations() {
        const topList = document.getElementById('top-supporters-list');
        const recentList = document.getElementById('recent-supporters-list');
        if (!topList || !recentList) return;

        fetch('/api/donations/recent')
            .then(res => res.json())
            .then(data => {
                // Update Cost Progress Meter
                const monthlyTotal = data.monthly_total || 0;
                const target = 400;
                const percentage = Math.round((monthlyTotal / target) * 100);

                // 1. Update status text
                const statusText = document.getElementById('funding-status-text');
                if (statusText) {
                    statusText.textContent = `$${Math.round(monthlyTotal)} / $${target} USD`;
                }

                // 2. Update progress bar fill
                const progressFill = document.querySelector('.progress-bar-fill');
                if (progressFill) {
                    progressFill.style.width = '0%';
                    setTimeout(() => {
                        progressFill.style.width = `${Math.min(100, percentage)}%`;
                    }, 150);
                }

                // 3. Update percentage footer
                const percentageText = document.getElementById('funding-percentage-text');
                if (percentageText) {
                    percentageText.textContent = `${percentage}% Funded this month`;
                }

                // 4. Update days remaining
                const daysText = document.getElementById('funding-days-text');
                if (daysText) {
                    const now = new Date();
                    const lastDay = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate();
                    const daysRemaining = lastDay - now.getDate();
                    if (daysRemaining === 0) {
                        daysText.textContent = 'Ends today';
                    } else if (daysRemaining === 1) {
                        daysText.textContent = '1 day remaining';
                    } else {
                        daysText.textContent = `${daysRemaining} days remaining`;
                    }
                }

                // Render Top Lifetime Supporters
                if (data.top && data.top.length > 0) {
                    topList.innerHTML = data.top.map(donation => {
                        const amount = parseFloat(donation.amount);
                        let tierClass = 'bronze-tier';
                        let tierName = 'Bronze Lifetime';
                        let avatar = '🛡️';

                        if (amount >= 30) {
                            tierClass = 'gold-tier';
                            tierName = 'Gold Lifetime';
                            avatar = '👑';
                        } else if (amount >= 15) {
                            tierClass = 'silver-tier';
                            tierName = 'Silver Lifetime';
                            avatar = '✨';
                        }

                        return `
                            <div class="hof-item">
                                <div class="hof-avatar">${avatar}</div>
                                <div class="hof-info">
                                    <div class="hof-name">${escapeHTML(donation.donor_name)}</div>
                                    <div class="hof-tier ${tierClass}">${tierName} • $${amount}</div>
                                </div>
                            </div>
                        `;
                    }).join('');
                } else {
                    topList.innerHTML = getEmptyPlaceholderHTML("Become a Top Supporter!");
                }

                // Render Recent Support Activity
                if (data.recent && data.recent.length > 0) {
                    recentList.innerHTML = data.recent.map(donation => {
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

                        let dateStr = '';
                        try {
                            const date = (typeof window.parseUTCTimestamp === 'function') ? window.parseUTCTimestamp(donation.timestamp) : new Date(donation.timestamp);
                            dateStr = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
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
                    recentList.innerHTML = getEmptyPlaceholderHTML("Be the first to donate!");
                }
            })
            .catch(err => {
                console.error('[donate.js] Error loading donations:', err);
            });
    }

    function getEmptyPlaceholderHTML(subtext) {
        return `
            <div class="hof-item" style="grid-column: 1 / -1; justify-content: center; padding: 25px; border-style: dashed; border-color: rgba(255,255,255,0.15); background: rgba(0,0,0,0.15);">
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; margin-bottom: 8px;">🤝</div>
                    <div style="font-weight: 700; color: #fff; margin-bottom: 4px;">No Donations Yet</div>
                    <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0;">${subtext}</p>
                </div>
            </div>
        `;
    }

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
})();
