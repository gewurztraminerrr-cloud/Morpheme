# Morpheme Stable State Summary – November 23, 2026

**Synchronization Checkpoint & Stable Anchor**  
- **Canonical Git Commit ID**: `eb51f92500c25a07530c00827282b09eb321e9c2` (`eb51f925`)
- **Git Branch**: `main` (synchronized: local `main` == `origin/main` == VPS `main`)
- **Production Server**: `132.148.72.249` ([morpheme.games](https://morpheme.games)) – Live (HTTP 200 OK)
- **Localhost Development**: Port `5001` – Running & healthy (Tally Hardened version)
- **Date/Time**: November 23, 2026 (15:55 UTC / 10:55 EST)

---

## 1. Key Achievements & Updates in this Stable Point

### A. Words User is Good at (Tools Tab) Mobile & Desktop Overhaul
- **Control Organization Aligned with Lists**:
  - Restructured the controls in `#tool-good-at` into an `.input-row` identical in organization to `#tool-lists`.
  - **Mobile 2×2 Grid + Full-Width Button**:
    - **Row 1**: Player textbox (`#good-at-user-input`) & Word Filter textbox (`#good-at-word-filter`)
    - **Row 2**: Length dropdown (`#good-at-length-select`) & Sort dropdown (`#good-at-sort-select`)
    - **Row 3**: Full-width **Update** button spanning both columns (`grid-column: 1 / -1`)
  - **Desktop Single Horizontal Row**: Aligned inputs, dropdowns, and Update button in a clean horizontal flex line matching Lists.
- **Un-Crammed Horizontal Touch Scrolling on Mobile**:
  - Eliminated cramped column percentages and abbreviated cell values.
  - Set comfortable table dimensions (`min-width: 540px`) with natural padding (`8px 12px` to `10px 14px`).
  - Rendered full readable values: Word text with definition popup triggers, length indicator (`3L`), bold find counts, full division badges (`Division 1 (1 pt)`, `Division 2 (3 pts)`, `Division 5 (9 pts)`), and full dates (`Nov 23, 2026`).
  - Enabled smooth horizontal swiping on mobile viewports via `-webkit-overflow-scrolling: touch` with subtle native scroll indicators.
- **Universal Internal Vertical Scrolling with Sticky Headers**:
  - **Mobile**: Configured `#good-at-results-wrapper` to fill all remaining vertical viewport space (`flex: 1 1 0px !important; min-height: 0 !important; height: 100% !important; overflow-y: auto !important;`). Sticky header (`position: sticky; top: 0; z-index: 10`) keeps column titles anchored while words scroll beneath.
  - **Desktop**: Isolated vertical flex sizing (`min-height: 160px; height: 100%`) with internal vertical scroll and custom thin scrollbars, preventing outer page displacement.
- **Compact Summary Banner**:
  - Streamlined the 4 summary statistics into a single-row compact strip on mobile (`PLAYER: <user>`, `WORDS: <count>`, `FINDS: <count>`, `RAREST: Div <n>`).
  - Avoided label truncation and ellipsis (`...`) while preserving bold contrast.
- **Description & Featured Players Spacing**:
  - Cleaned up top player chip pills with dynamic toggle (hidden when empty).
  - Adjusted header description text padding and max-height to ensure text remains fully legible without clipping descenders.

---

### B. Moderation Policy, Hierarchy & Permanent Ban Protocol
- **Permanent Ban Enforcement & Removal of "Lift Ban"**:
  - Fully eliminated the "Lift Ban" button and associated frontend/backend handlers from **Mods $\to$ Ban/Timeout User**.
  - Enforced a strictly permanent ban model: when a user account is banned, the record is permanently eradicated, active sessions are terminated, and their IP address is blacklisted.
- **Hardened Root Administrator Protection (`jeffb`)**:
  - Client-side and server-side hard checks prevent any moderator from timing out, banning, or IP-banning user `jeffb`.
  - Rejection with explicit authorization error (`403 Forbidden: Protected user cannot be timed out or banned`) if attempted.
- **Moderator Hierarchy Rules**:
  - Only root administrator `jeffb` possesses the privilege to timeout or ban fellow moderators.
  - Standard moderators are prevented from taking moderation actions against other moderators.
- **Moderator Record & Privilege Purging**:
  - If a moderator account is banned, all associated moderator permissions, credentials, and cache entries are completely purged from the system.

---

## 2. Infrastructure & System Status

| Component | Status | Details |
|---|---|---|
| **Production VPS** | `ONLINE` | `132.148.72.249` ([morpheme.games](https://morpheme.games)), Ubuntu 24.04 LTS, PM2 Process ID 0 |
| **Localhost** | `ONLINE` | Port `5001`, Python 3 Flask / SocketIO daemon (Tally Hardened) |
| **Git Canonical Branch** | `SYNCHRONIZED` | `origin/main` == `HEAD` (`7c50d72b`) |
| **Asset Cache Buster** | `1795290000` | Applied across `play.css` and `tools.js` in `templates/index.html` |
| **RAM Utilization** | `OPTIMAL` | SQLite definitions proxy + LRU cache maintaining low ~1.1 GB memory profile |

---

## 3. Verification & Test Evidence
- **Mobile Emulation (390×844 iPhone Viewport)**:
  - Verified 2×2 control grid + full-width Update button matches Lists.
  - Verified horizontal swiping (`scrollLeft`) reveals all columns comfortably without clipping.
  - Verified vertical scrolling (`scrollTop`) with sticky headers anchored at table top.
- **Desktop (1440×900 Viewport)**:
  - Verified horizontal input controls row.
  - Verified vertical table scrolling within `#good-at-results-wrapper` with 50+ rows.
  - Verified zero outer page layout displacement.
- **Live Production HTTP**:
  - `curl -sL https://morpheme.games/` confirmed serving cache buster `v=1795290000`.
  - PM2 logs confirmed clean process restart with zero errors.
