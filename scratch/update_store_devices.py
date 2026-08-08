#!/usr/bin/env python3
"""
Morpheme Store Monthly Device Updater
======================================
Run this script once a month to automatically update the Store section
in Tools with the best current Android phones, iPhones, iPads, and
PC/Mac desktops & laptops. It uses the Gemini API to research current
best devices, rewrites the relevant store items in index.html, bumps
version numbers, commits to git, and deploys to production.

Usage:
    python3 scratch/update_store_devices.py

Schedule (cron — runs on the 1st of each month at 3:00 AM server time):
    0 3 1 * * cd /path/to/morpheme && python3 scratch/update_store_devices.py >> logs/store_update.log 2>&1

Requirements:
    pip install google-generativeai

Setup:
    1. Set GEMINI_API_KEY environment variable.
    2. Add sentinel comments around device items in index.html:
           <!-- AUTO-UPDATED DEVICES START -->
           ...existing device store items...
           <!-- AUTO-UPDATED DEVICES END -->
    3. Add this file's cron line to the server crontab.
"""

import os
import re
import subprocess
import sys
import datetime
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("store-updater")

# Paths
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_HTML   = os.path.join(REPO_ROOT, "templates", "index.html")
DEPLOY_SCRIPT = os.path.join(REPO_ROOT, "scratch", "deploy_all_fixes.py")

# Sentinel comments that wrap the auto-updated section in index.html
START_MARKER = "<!-- AUTO-UPDATED DEVICES START -->"
END_MARKER   = "<!-- AUTO-UPDATED DEVICES END -->"

# Device categories to research
DEVICE_CATEGORIES = [
    "best Android smartphone",
    "best iPhone",
    "best iPad",
    "best Windows laptop",
    "best Mac laptop",
    "best Windows desktop",
    "best Mac desktop",
]


def run(cmd, cwd=REPO_ROOT, check=True):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        log.error(f"Command failed: {cmd}\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def get_current_lobby_css_version():
    with open(INDEX_HTML) as f:
        content = f.read()
    m = re.search(r'lobby\.css\?v=(\d+)', content)
    return int(m.group(1)) if m else 81


def research_best_devices():
    """Use Gemini API to look up the best devices for this month."""
    try:
        import google.generativeai as genai
    except ImportError:
        log.error("google-generativeai not installed. Run: pip install google-generativeai")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("Set GEMINI_API_KEY environment variable.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    month_year = datetime.date.today().strftime("%B %Y")
    prompt = f"""
You are updating the Morpheme word-game Store page for {month_year}.
Morpheme is a fast-paced touchscreen word-finding game (like Boggle but digital).
Players drag paths across letter grids on mobile/tablet/desktop.

For each category below, pick the single BEST device available as of {month_year}.
Focus on: touch responsiveness, refresh rate, display quality, and performance.
Prefer devices available on Amazon.ca.

Categories:
{json.dumps(DEVICE_CATEGORIES, indent=2)}

Return a JSON array (no markdown fences, no extra text) of objects with these keys:
- category       : one of the category strings above
- name           : full product name  (e.g. "Samsung Galaxy S25 Ultra")
- subtitle       : 3-6 word tagline   (e.g. "The Ultimate Mobile Platform")
- price_cad      : approx price       (e.g. "$1,799 CAD")
- amazon_url     : best Amazon.ca or BestBuy.ca URL
- image_url      : Amazon CDN image URL (https://m.media-amazon.com/images/...)
- headline_label : bold intro phrase  (e.g. "The Perfect Choice")
- desc1          : 2-3 sentences about why ideal for Morpheme
- desc2          : 2-3 sentences about a key display/touch spec for Morpheme
- desc3          : 2-3 sentences about performance/reliability for Morpheme
- features       : array of exactly 4 short feature strings

Be specific with real current specs. Output ONLY valid JSON.
"""

    log.info(f"Querying Gemini for best devices ({month_year})...")
    response = model.generate_content(prompt)
    raw = response.text.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'```\s*$', '', raw)

    devices = json.loads(raw)
    log.info(f"Received {len(devices)} device records from Gemini.")
    return devices


def build_store_item_html(device):
    features_html = "\n".join(
        f"                            <li>{f}</li>" for f in device["features"]
    )
    return f"""
                <!-- AUTO: {device['category']} -->
                <div class="store-item" data-category="hardware">
                    <div class="store-item-image">
                        <img src="{device['image_url']}" alt="{device['name']}">
                    </div>
                    <div class="store-item-info">
                        <div class="store-item-header">
                            <h3>{device['name']} ({device['subtitle']})</h3>
                            <span class="store-item-price">{device['price_cad']}</span>
                        </div>
                        <p class="store-item-desc"><strong>{device['headline_label']}:</strong> {device['desc1']}</p>
                        <p class="store-item-desc"><strong>Touch &amp; Display:</strong> {device['desc2']}</p>
                        <p class="store-item-desc"><strong>Performance:</strong> {device['desc3']}</p>
                        <ul class="store-item-features">
{features_html}
                        </ul>
                        <a href="{device['amazon_url']}" target="_blank" class="buy-now-btn">View on Amazon</a>
                    </div>
                </div>"""


def update_index_html(devices):
    with open(INDEX_HTML) as f:
        content = f.read()

    if START_MARKER not in content:
        log.error(
            f"Sentinel '{START_MARKER}' not found in index.html.\n"
            "Add <!-- AUTO-UPDATED DEVICES START --> and <!-- AUTO-UPDATED DEVICES END -->\n"
            "around the device store items manually first."
        )
        sys.exit(1)

    new_items_html = "\n".join(build_store_item_html(d) for d in devices)
    month_year = datetime.date.today().strftime("%B %Y")

    new_section = (
        f"{START_MARKER}\n"
        f"                <!-- Last auto-updated: {month_year} -->\n"
        f"{new_items_html}\n"
        f"                {END_MARKER}"
    )

    updated = re.sub(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
        new_section,
        content,
        flags=re.DOTALL
    )

    with open(INDEX_HTML, "w") as f:
        f.write(updated)

    log.info("index.html updated with new device listings.")


def bump_lobby_css_version():
    with open(INDEX_HTML) as f:
        content = f.read()

    current = get_current_lobby_css_version()
    new_ver = current + 1
    updated = re.sub(r'(lobby\.css\?v=)\d+', f'\\g<1>{new_ver}', content)

    with open(INDEX_HTML, "w") as f:
        f.write(updated)

    log.info(f"lobby.css version bumped v{current} → v{new_ver}")


def git_commit_and_push(month_year):
    run("git add templates/index.html")
    run(f'git commit -m "auto: update Store device listings for {month_year}"')
    run("git push origin main")
    log.info("Committed and pushed to GitHub.")


def deploy():
    if not os.path.exists(DEPLOY_SCRIPT):
        log.warning(f"Deploy script not found at {DEPLOY_SCRIPT} — skipping.")
        return
    log.info("Deploying to production...")
    run(f"python3 {DEPLOY_SCRIPT}")
    log.info("Deployment complete.")


def main():
    month_year = datetime.date.today().strftime("%B %Y")
    log.info(f"=== Morpheme Store Device Update — {month_year} ===")
    devices = research_best_devices()
    update_index_html(devices)
    bump_lobby_css_version()
    git_commit_and_push(month_year)
    deploy()
    log.info("=== Store update finished ===")


if __name__ == "__main__":
    main()
