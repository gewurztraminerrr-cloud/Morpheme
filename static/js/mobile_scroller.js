// Mobile custom vertical scroller (track + draggable thumb) for Leaderboards, Tournaments, and Round Replay modal.
// The track is appended to <body> and positioned over the right edge of the scrolling element,
// so no ancestor styling can clip or hide it.
(function () {
    const MOBILE_MAX = 992;

    function attachScroller(targetSpec) {
        let getTarget;
        let isOverlayTarget = false;

        if (typeof targetSpec === 'string') {
            getTarget = () => document.getElementById(targetSpec);
        } else if (typeof targetSpec === 'object' && typeof targetSpec.getTarget === 'function') {
            getTarget = targetSpec.getTarget;
            isOverlayTarget = !!targetSpec.isOverlay;
        } else {
            return;
        }

        const track = document.createElement('div');
        track.className = 'm-scroller-track';
        if (isOverlayTarget) {
            track.classList.add('m-scroller-track-overlay');
        }
        const thumb = document.createElement('div');
        thumb.className = 'm-scroller-thumb';
        track.appendChild(thumb);
        document.body.appendChild(track);

        let dragging = false;
        let startY = 0;
        let startThumbTop = 0;

        function metrics() {
            const trackH = track.clientHeight;
            const thumbH = thumb.offsetHeight;
            return { trackH, thumbH, maxThumbTop: Math.max(0, trackH - thumbH) };
        }

        function update() {
            const el = getTarget();
            if (!el) {
                track.style.display = 'none';
                return;
            }

            const replayOverlay = document.getElementById('history-review-overlay');
            const isReplayOpen = replayOverlay &&
                !replayOverlay.classList.contains('hidden') &&
                (replayOverlay.classList.contains('forced-show') || replayOverlay.offsetParent !== null || window.getComputedStyle(replayOverlay).display !== 'none');

            if (isOverlayTarget) {
                // Round Replay overlay custom scroller
                const isRendered = el && (el.offsetParent !== null || window.getComputedStyle(el).display !== 'none');
                const visible = window.innerWidth <= MOBILE_MAX &&
                    isReplayOpen &&
                    isRendered &&
                    (el.scrollHeight > el.clientHeight + 5);

                if (!visible) {
                    track.style.display = 'none';
                    return;
                }
            } else {
                // Background page scroller (leaderboards / tournaments)
                const isAnyModalOpen = isReplayOpen || !!document.querySelector('.overlay:not(.hidden), .modal-overlay:not(.hidden), .mini-profile-overlay:not(.hidden), #full-list-modal:not(.hidden)');
                const visible = window.innerWidth <= MOBILE_MAX &&
                    !isAnyModalOpen &&
                    el.classList.contains('active') &&
                    el.offsetParent !== null &&
                    (el.scrollHeight > el.clientHeight + 15);

                if (!visible) {
                    track.style.display = 'none';
                    return;
                }
            }

            const r = el.getBoundingClientRect();
            track.style.display = 'block';
            track.style.top = Math.max(0, r.top + 6) + 'px';
            track.style.height = Math.max(40, r.height - 12) + 'px';

            if (dragging) return;
            const trackH = track.clientHeight;
            const ratio = el.clientHeight / el.scrollHeight;
            const thumbH = Math.max(40, Math.min(trackH, trackH * ratio));
            thumb.style.height = thumbH + 'px';
            const maxScroll = el.scrollHeight - el.clientHeight;
            const maxTop = Math.max(0, trackH - thumbH);
            thumb.style.top = (maxScroll > 0 ? (el.scrollTop / maxScroll) * maxTop : 0) + 'px';
        }

        function onMove(e) {
            const el = getTarget();
            if (!dragging || !el) return;
            const y = e.touches ? e.touches[0].clientY : e.clientY;
            const m = metrics();
            const top = Math.max(0, Math.min(m.maxThumbTop, startThumbTop + (y - startY)));
            thumb.style.top = top + 'px';
            const maxScroll = el.scrollHeight - el.clientHeight;
            if (m.maxThumbTop > 0) el.scrollTop = (top / m.maxThumbTop) * maxScroll;
            if (e.cancelable) e.preventDefault();
        }

        function onEnd() {
            dragging = false;
            thumb.classList.remove('dragging');
            document.removeEventListener('touchmove', onMove);
            document.removeEventListener('touchend', onEnd);
            document.removeEventListener('touchcancel', onEnd);
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onEnd);
            update();
        }

        function onStart(e) {
            dragging = true;
            thumb.classList.add('dragging');
            startY = e.touches ? e.touches[0].clientY : e.clientY;
            startThumbTop = parseFloat(thumb.style.top) || 0;
            document.addEventListener('touchmove', onMove, { passive: false });
            document.addEventListener('touchend', onEnd);
            document.addEventListener('touchcancel', onEnd);
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onEnd);
            if (e.cancelable) e.preventDefault();
            e.stopPropagation();
        }

        thumb.addEventListener('touchstart', onStart, { passive: false });
        thumb.addEventListener('mousedown', onStart);

        track.addEventListener('click', (e) => {
            const el = getTarget();
            if (!el || e.target === thumb) return;
            const rect = track.getBoundingClientRect();
            const m = metrics();
            const top = Math.max(0, Math.min(m.maxThumbTop, e.clientY - rect.top - m.thumbH / 2));
            const maxScroll = el.scrollHeight - el.clientHeight;
            if (m.maxThumbTop > 0) {
                el.scrollTo({ top: (top / m.maxThumbTop) * maxScroll, behavior: 'smooth' });
            }
        });

        // Track target scroll
        let attachedEl = null;
        function bindTargetScroll() {
            const curEl = getTarget();
            if (curEl && curEl !== attachedEl) {
                if (attachedEl) attachedEl.removeEventListener('scroll', update);
                attachedEl = curEl;
                attachedEl.addEventListener('scroll', update, { passive: true });
            }
        }

        window.addEventListener('resize', update, { passive: true });
        window.addEventListener('orientationchange', update, { passive: true });
        setInterval(() => {
            bindTargetScroll();
            update();
        }, 200);
        bindTargetScroll();
        update();

        // Expose update for modal opening
        if (isOverlayTarget) {
            window.updateReplayScroller = update;
        }
    }

    function init() {
        attachScroller('page-leaderboards');
        attachScroller('page-tournaments');
        // Round Replay overlay layout on mobile
        attachScroller({
            getTarget: () => document.querySelector('#history-review-overlay .history-review-layout'),
            isOverlay: true
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
