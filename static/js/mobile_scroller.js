// Mobile custom vertical scroller (track + draggable thumb) for Leaderboards and Tournaments.
// The track is appended to <body> and positioned over the right edge of the scrolling page,
// so no ancestor styling can clip or hide it.
(function () {
    const MOBILE_MAX = 992;

    function attachScroller(pageId) {
        const page = document.getElementById(pageId);
        if (!page) return;

        const track = document.createElement('div');
        track.className = 'm-scroller-track';
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
            // If an overlay or modal is open (e.g. Round Replay, Full List, Profile, etc.), hide the background page scroller
            const replayOverlay = document.getElementById('history-review-overlay');
            const isReplayOpen = replayOverlay && !replayOverlay.classList.contains('hidden') && replayOverlay.offsetParent !== null;
            const isAnyModalOpen = isReplayOpen || !!document.querySelector('.overlay:not(.hidden), .modal-overlay:not(.hidden), .mini-profile-overlay:not(.hidden), #full-list-modal:not(.hidden)');

            const visible = window.innerWidth <= MOBILE_MAX &&
                !isAnyModalOpen &&
                page.classList.contains('active') &&
                page.offsetParent !== null &&
                page.scrollHeight > page.clientHeight + 15;
            if (!visible) {
                track.style.display = 'none';
                return;
            }
            const r = page.getBoundingClientRect();
            track.style.display = 'block';
            track.style.top = (r.top + 6) + 'px';
            track.style.height = Math.max(40, r.height - 12) + 'px';

            if (dragging) return;
            const trackH = track.clientHeight;
            const ratio = page.clientHeight / page.scrollHeight;
            const thumbH = Math.max(40, Math.min(trackH, trackH * ratio));
            thumb.style.height = thumbH + 'px';
            const maxScroll = page.scrollHeight - page.clientHeight;
            const maxTop = Math.max(0, trackH - thumbH);
            thumb.style.top = (maxScroll > 0 ? (page.scrollTop / maxScroll) * maxTop : 0) + 'px';
        }

        function onMove(e) {
            if (!dragging) return;
            const y = e.touches ? e.touches[0].clientY : e.clientY;
            const m = metrics();
            const top = Math.max(0, Math.min(m.maxThumbTop, startThumbTop + (y - startY)));
            thumb.style.top = top + 'px';
            const maxScroll = page.scrollHeight - page.clientHeight;
            if (m.maxThumbTop > 0) page.scrollTop = (top / m.maxThumbTop) * maxScroll;
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
            if (e.target === thumb) return;
            const rect = track.getBoundingClientRect();
            const m = metrics();
            const top = Math.max(0, Math.min(m.maxThumbTop, e.clientY - rect.top - m.thumbH / 2));
            const maxScroll = page.scrollHeight - page.clientHeight;
            if (m.maxThumbTop > 0) {
                page.scrollTo({ top: (top / m.maxThumbTop) * maxScroll, behavior: 'smooth' });
            }
        });

        page.addEventListener('scroll', update, { passive: true });
        window.addEventListener('resize', update, { passive: true });
        window.addEventListener('orientationchange', update, { passive: true });
        // Content and visibility change dynamically (data loads, tab switches), so re-check regularly.
        setInterval(update, 400);
        update();
    }

    function init() {
        attachScroller('page-leaderboards');
        attachScroller('page-tournaments');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
