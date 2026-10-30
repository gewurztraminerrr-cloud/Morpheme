/**
 * Morpheme Store Module
 * Automatically renders competitive tournament hardware, books, and accessories.
 * Dynamically adjusts all prices based on the player's selected country flag.
 */

(function() {
    'use strict';

    // Comprehensive Flag & ISO Code to Currency Mapping
    const CURRENCY_DATABASE = {
        'US': { code: 'USD', symbol: '$', rate: 1.0, flag: '🇺🇸', name: 'United States' },
        'CA': { code: 'CAD', symbol: '$', rate: 1.38, flag: '🇨🇦', name: 'Canada' },
        'GB': { code: 'GBP', symbol: '£', rate: 0.77, flag: '🇬🇧', name: 'United Kingdom' },
        'AU': { code: 'AUD', symbol: '$', rate: 1.51, flag: '🇦🇺', name: 'Australia' },
        'NZ': { code: 'NZD', symbol: '$', rate: 1.66, flag: '🇳🇿', name: 'New Zealand' },
        'JP': { code: 'JPY', symbol: '¥', rate: 152.0, flag: '🇯🇵', name: 'Japan', decimals: 0 },
        'CN': { code: 'CNY', symbol: '¥', rate: 7.12, flag: '🇨🇳', name: 'China' },
        'IN': { code: 'INR', symbol: '₹', rate: 84.0, flag: '🇮🇳', name: 'India' },
        'CH': { code: 'CHF', symbol: 'CHF ', rate: 0.87, flag: '🇨🇭', name: 'Switzerland' },
        'KR': { code: 'KRW', symbol: '₩', rate: 1380.0, flag: '🇰🇷', name: 'South Korea', decimals: 0 },
        'MX': { code: 'MXN', symbol: '$', rate: 19.90, flag: '🇲🇽', name: 'Mexico' },
        'BR': { code: 'BRL', symbol: 'R$ ', rate: 5.70, flag: '🇧🇷', name: 'Brazil' },
        'SE': { code: 'SEK', symbol: 'kr ', rate: 10.60, flag: '🇸🇪', name: 'Sweden' },
        'NO': { code: 'NOK', symbol: 'kr ', rate: 11.00, flag: '🇳🇴', name: 'Norway' },
        'DK': { code: 'DKK', symbol: 'kr ', rate: 6.85, flag: '🇩🇰', name: 'Denmark' },
        'PL': { code: 'PLN', symbol: 'zł ', rate: 4.00, flag: '🇵🇱', name: 'Poland' },
        'SG': { code: 'SGD', symbol: '$', rate: 1.32, flag: '🇸🇬', name: 'Singapore' },
        'HK': { code: 'HKD', symbol: '$', rate: 7.77, flag: '🇭🇰', name: 'Hong Kong' },
        'ZA': { code: 'ZAR', symbol: 'R ', rate: 17.60, flag: '🇿🇦', name: 'South Africa' },
        'PH': { code: 'PHP', symbol: '₱', rate: 57.50, flag: '🇵🇭', name: 'Philippines' },
        'TR': { code: 'TRY', symbol: '₺', rate: 34.20, flag: '🇹🇷', name: 'Turkey' },
        'IL': { code: 'ILS', symbol: '₪', rate: 3.74, flag: '🇮🇱', name: 'Israel' },
        'AE': { code: 'AED', symbol: 'AED ', rate: 3.67, flag: '🇦🇪', name: 'UAE' },
        'SA': { code: 'SAR', symbol: 'SAR ', rate: 3.75, flag: '🇸🇦', name: 'Saudi Arabia' },
        'TW': { code: 'TWD', symbol: 'NT$ ', rate: 32.0, flag: '🇹🇼', name: 'Taiwan' },
        'TH': { code: 'THB', symbol: '฿', rate: 33.20, flag: '🇹🇭', name: 'Thailand' },
        'ID': { code: 'IDR', symbol: 'Rp ', rate: 15600.0, flag: '🇮🇩', name: 'Indonesia', decimals: 0 },
        'MY': { code: 'MYR', symbol: 'RM ', rate: 4.32, flag: '🇲🇾', name: 'Malaysia' },
        'VN': { code: 'VND', symbol: '₫', rate: 25300.0, flag: '🇻🇳', name: 'Vietnam', decimals: 0 },
        'NG': { code: 'NGN', symbol: '₦', rate: 1670.0, flag: '🇳🇬', name: 'Nigeria' },
        'AR': { code: 'ARS', symbol: '$', rate: 980.0, flag: '🇦🇷', name: 'Argentina' },
        'CL': { code: 'CLP', symbol: '$', rate: 945.0, flag: '🇨🇱', name: 'Chile', decimals: 0 },
        'CO': { code: 'COP', symbol: '$', rate: 4260.0, flag: '🇨🇴', name: 'Colombia', decimals: 0 },
        'CZ': { code: 'CZK', symbol: 'Kč ', rate: 23.20, flag: '🇨🇿', name: 'Czech Republic' },
        'HU': { code: 'HUF', symbol: 'Ft ', rate: 370.0, flag: '🇭🇺', name: 'Hungary', decimals: 0 },
        'RO': { code: 'RON', symbol: 'lei ', rate: 4.58, flag: '🇷🇴', name: 'Romania' },
        // Eurozone Nations
        'DE': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇩🇪', name: 'Germany' },
        'FR': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇫🇷', name: 'France' },
        'IT': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇮🇹', name: 'Italy' },
        'ES': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇪🇸', name: 'Spain' },
        'NL': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇳🇱', name: 'Netherlands' },
        'BE': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇧🇪', name: 'Belgium' },
        'AT': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇦🇹', name: 'Austria' },
        'IE': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇮🇪', name: 'Ireland' },
        'FI': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇫🇮', name: 'Finland' },
        'PT': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇵🇹', name: 'Portugal' },
        'GR': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇬🇷', name: 'Greece' },
        'LU': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇱🇺', name: 'Luxembourg' },
        'EE': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇪🇪', name: 'Estonia' },
        'LV': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇱🇻', name: 'Latvia' },
        'LT': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇱🇹', name: 'Lithuania' },
        'SK': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇸🇰', name: 'Slovakia' },
        'SI': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇸🇮', name: 'Slovenia' },
        'CY': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇨🇾', name: 'Cyprus' },
        'MT': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇲🇹', name: 'Malta' },
        'HR': { code: 'EUR', symbol: '€', rate: 0.92, flag: '🇭🇷', name: 'Croatia' }
    };

    let _storeCatalog = null;
    let _activeCategory = 'all';

    /**
     * Resolves the player's active flag and corresponding currency object.
     */
    function getActiveCurrency() {
        const flagEmoji = (window.currentUserFlag || localStorage.getItem('morpheme_country_flag') || '').trim();
        let countryCode = null;

        if (flagEmoji && typeof window.emojiToCountryCode === 'function') {
            countryCode = window.emojiToCountryCode(flagEmoji);
        } else if (flagEmoji && typeof window.ALL_FLAGS !== 'undefined') {
            const match = window.ALL_FLAGS.find(f => f.flag === flagEmoji);
            if (match) countryCode = match.code;
        }

        if (countryCode && CURRENCY_DATABASE[countryCode]) {
            return {
                ...CURRENCY_DATABASE[countryCode],
                flagEmoji: flagEmoji || CURRENCY_DATABASE[countryCode].flag
            };
        }

        // Default to USD
        return {
            ...CURRENCY_DATABASE['US'],
            flagEmoji: flagEmoji || '🇺🇸'
        };
    }

    /**
     * Formats a USD base price into the target currency.
     */
    function formatPrice(basePriceUsd, curr) {
        if (!basePriceUsd && basePriceUsd !== 0) return '';
        const converted = basePriceUsd * (curr.rate || 1.0);
        const dec = curr.decimals !== undefined ? curr.decimals : 2;

        const formattedNumber = converted.toLocaleString(undefined, {
            minimumFractionDigits: dec,
            maximumFractionDigits: dec
        });

        // If local currency is CAD or USD, show both or clean standard display
        if (curr.code === 'USD') {
            return `$${formattedNumber} USD`;
        }
        if (curr.code === 'CAD') {
            const usdEquiv = basePriceUsd.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            return `$${formattedNumber} CAD <span class="store-price-subtext">(~$${usdEquiv} USD)</span>`;
        }

        const usdEquiv = basePriceUsd.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        return `${curr.symbol}${formattedNumber} ${curr.code} <span class="store-price-subtext">(~$${usdEquiv} USD)</span>`;
    }

    /**
     * Fetch store catalog from server or cache.
     */
    async function loadStoreCatalog() {
        if (_storeCatalog) return _storeCatalog;
        try {
            const res = await fetch(`/api/store/catalog?t=${Date.now()}`);
            if (res.ok) {
                _storeCatalog = await res.json();
                return _storeCatalog;
            }
        } catch (e) {
            console.warn('[Store] API catalog fetch failed, trying static fallback:', e);
        }

        try {
            const resStatic = await fetch(`/static/data/store_catalog.json?t=${Date.now()}`);
            if (resStatic.ok) {
                _storeCatalog = await resStatic.json();
                return _storeCatalog;
            }
        } catch (e2) {
            console.error('[Store] Static catalog fetch failed:', e2);
        }

        return null;
    }

    /**
     * Renders the complete Store UI inside #tool-store.
     */
    async function renderMorphemeStore() {
        const storePane = document.getElementById('tool-store');
        if (!storePane) return;

        const catalog = await loadStoreCatalog();
        if (!catalog || !catalog.products) {
            console.error('[Store] Unable to load store catalog data.');
            return;
        }

        const curr = getActiveCurrency();
        const flagHtml = window.getFlagHtml ? window.getFlagHtml(curr.flagEmoji, 'width: 22px; height: auto; margin-right: 6px;') : `${curr.flagEmoji} `;

        // Update or inject Top Header & Banner
        let headerEl = storePane.querySelector('.tool-header');
        if (!headerEl) {
            headerEl = document.createElement('div');
            headerEl.className = 'tool-header';
            storePane.prepend(headerEl);
        }

        headerEl.innerHTML = `
            <h2>Morpheme Store</h2>
            <p>Tournament-grade hardware, official dictionaries, and accessories optimized for competitive play.</p>
            <div class="store-top-banner">
                <div class="store-status-dates-card">
                    <div class="store-banner-side-visual store-banner-visual-left" title="Click to enlarge Morpheme Flagship Boutique">
                        <img src="/static/images/morpheme_store_front.jpg" alt="Morpheme Flagship Boutique" class="store-banner-showcase-img" loading="lazy">
                        <span class="store-banner-zoom-hint" title="Click to enlarge"><i class="fas fa-search-plus"></i></span>
                        <span class="store-banner-showcase-tag"><i class="fas fa-store-alt"></i> Morpheme Boutique</span>
                    </div>
                    <div class="store-status-center-col">
                        <div class="store-status-row store-last-updated-row">
                            <span class="store-status-icon-wrap"><span class="store-pulse-dot"></span></span>
                            <div class="store-status-content">
                                <span class="store-status-label">Prices and products last updated on</span>
                                <span class="store-status-date">${catalog.last_catalog_update || 'October 16, 2026'}</span>
                            </div>
                        </div>
                        <div class="store-status-divider"></div>
                        <div class="store-status-row store-next-update-row">
                            <span class="store-status-icon-wrap"><i class="fas fa-calendar-alt"></i></span>
                            <div class="store-status-content">
                                <span class="store-status-label">Next Store update will occur on</span>
                                <span class="store-status-date">${catalog.next_scheduled_update || 'November 16, 2026'}</span>
                            </div>
                        </div>
                        <div class="store-status-divider"></div>
                        <div class="store-currency-card">
                            <span class="store-context-label">Prices reflected for:</span>
                            <div class="store-flag-badge" title="Automatically tailored to your profile country flag (${curr.name})">
                                ${flagHtml}
                                <span class="store-flag-country-name">${curr.name}</span>
                                <span class="store-flag-currency-code">(${curr.code})</span>
                            </div>
                        </div>
                    </div>
                    <div class="store-banner-side-visual store-banner-visual-right" title="Click to enlarge Tournament Grid Showcase">
                        <img src="/static/images/morpheme_store_display.jpg" alt="Tournament Grid Showcase" class="store-banner-showcase-img" loading="lazy">
                        <span class="store-banner-zoom-hint" title="Click to enlarge"><i class="fas fa-search-plus"></i></span>
                        <span class="store-banner-showcase-tag"><i class="fas fa-th"></i> Tournament Grid Showcase</span>
                    </div>
                </div>
            </div>
            <div class="store-category-filters" id="store-category-filters">
                <button class="store-filter-btn ${_activeCategory === 'all' ? 'active' : ''}" data-cat="all"><i class="fas fa-th-large"></i><span>All Products (${catalog.products.length})</span></button>
                <button class="store-filter-btn ${_activeCategory === 'hardware' ? 'active' : ''}" data-cat="hardware"><i class="fas fa-microchip"></i><span>Tournament Hardware</span></button>
                <button class="store-filter-btn ${_activeCategory === 'mobile' ? 'active' : ''}" data-cat="mobile"><i class="fas fa-mobile-alt"></i><span>Phones & Tablets</span></button>
                <button class="store-filter-btn ${_activeCategory === 'computers' ? 'active' : ''}" data-cat="computers"><i class="fas fa-laptop"></i><span>PC & Mac</span></button>
                <button class="store-filter-btn ${_activeCategory === 'books' ? 'active' : ''}" data-cat="books"><i class="fas fa-book"></i><span>Dictionaries</span></button>
                <button class="store-filter-btn ${_activeCategory === 'games' ? 'active' : ''}" data-cat="games"><i class="fas fa-cube"></i><span>Games & Gear</span></button>
            </div>
        `;

        // Wire up Expandable Showcase Images (Lightbox)
        const visualLeft = headerEl.querySelector('.store-banner-visual-left');
        if (visualLeft) {
            visualLeft.addEventListener('click', () => {
                if (typeof window.showImageLightbox === 'function') {
                    window.showImageLightbox('/static/images/morpheme_store_front.jpg', 'Morpheme Flagship Boutique — Specialists in Competitive Word Games');
                }
            });
        }
        const visualRight = headerEl.querySelector('.store-banner-visual-right');
        if (visualRight) {
            visualRight.addEventListener('click', () => {
                if (typeof window.showImageLightbox === 'function') {
                    window.showImageLightbox('/static/images/morpheme_store_display.jpg', 'Tournament Grid Showcase — 4×4 Grid, Dome Shaker & Official Dictionaries');
                }
            });
        }

        // Wire up Category Filters
        const filterBtns = headerEl.querySelectorAll('.store-filter-btn');
        filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                _activeCategory = btn.dataset.cat;
                filterBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                renderProductGrid(catalog.products, curr);
            });
        });

        // Ensure Grid Container Exists
        let gridEl = document.getElementById('store-items-grid');
        if (!gridEl) {
            gridEl = document.createElement('div');
            gridEl.id = 'store-items-grid';
            gridEl.className = 'store-grid';
            storePane.appendChild(gridEl);
        }

        renderProductGrid(catalog.products, curr);
    }

    /**
     * Renders filtered product cards into #store-items-grid.
     */
    function renderProductGrid(products, curr) {
        const gridEl = document.getElementById('store-items-grid');
        if (!gridEl) return;

        const filtered = products.filter(p => {
            if (_activeCategory === 'all') return true;
            if (_activeCategory === 'hardware') {
                return ['mobile', 'tablet', 'pc_laptop', 'pc_desktop', 'mac_laptop', 'mac_desktop'].includes(p.category);
            }
            if (_activeCategory === 'mobile') {
                return ['mobile', 'tablet'].includes(p.category);
            }
            if (_activeCategory === 'computers') {
                return ['pc_laptop', 'pc_desktop', 'mac_laptop', 'mac_desktop'].includes(p.category);
            }
            if (_activeCategory === 'books') {
                return p.category === 'books';
            }
            if (_activeCategory === 'games') {
                return ['games', 'accessories'].includes(p.category);
            }
            return p.category === _activeCategory;
        });

        gridEl.innerHTML = filtered.map(item => {
            const priceHtml = formatPrice(item.base_price_usd, curr);
            const descHtml = (item.desc || []).map(d => `<p class="store-item-desc">${escapeHtml(d)}</p>`).join('');
            const featuresHtml = (item.features && item.features.length) ? `
                <ul class="store-item-features">
                    ${item.features.map(f => `<li>${escapeHtml(f)}</li>`).join('')}
                </ul>
            ` : '';

            return `
                <div class="store-item" data-category="${escapeHtml(item.category)}" data-id="${escapeHtml(item.id)}">
                    <div class="store-item-image">
                        <img src="${escapeHtml(item.image_url)}" alt="${escapeHtml(item.title)}" loading="lazy" style="cursor: zoom-in;" title="Click to enlarge ${escapeHtml(item.title)}">
                    </div>
                    <div class="store-item-info">
                        <div class="store-item-header">
                            <h3>${escapeHtml(item.title)}</h3>
                            <span class="store-item-price">${priceHtml}</span>
                            <div class="store-item-last-changed" title="Last date this product was introduced, verified, or updated">
                                <i class="fas fa-calendar-check" style="margin-right: 4px; opacity: 0.85;"></i>
                                Last changed on ${escapeHtml(item.last_changed_date || 'October 16, 2026')}
                            </div>
                        </div>
                        ${descHtml}
                        ${featuresHtml}
                        <a href="${escapeHtml(item.buy_url)}" target="_blank" rel="noopener noreferrer" class="buy-now-btn">
                            ${escapeHtml(item.buy_label || 'View Store Listing')}
                        </a>
                    </div>
                </div>
            `;
        }).join('');

        // Wire up product image lightbox zoom
        gridEl.querySelectorAll('.store-item-image img').forEach(img => {
            img.addEventListener('click', () => {
                if (typeof window.showImageLightbox === 'function') {
                    window.showImageLightbox(img.src, img.alt || 'Product Image');
                }
            });
        });
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Expose to window for global invocation and live re-rendering on flag change
    window.renderMorphemeStore = renderMorphemeStore;
    window.CURRENCY_DATABASE = CURRENCY_DATABASE;

    // Listen for tab switch to Store
    document.addEventListener('DOMContentLoaded', () => {
        // If Store is currently open or loaded
        renderMorphemeStore();

        // Also trigger whenever Tools -> Store tab button is pressed
        const storeToolBtn = document.querySelector('[data-tool="store"], [data-pane="tool-store"], #tool-btn-store');
        if (storeToolBtn) {
            storeToolBtn.addEventListener('click', () => {
                setTimeout(renderMorphemeStore, 50);
            });
        }
    });

})();
