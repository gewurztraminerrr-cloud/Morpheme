import time
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def run_test(is_mobile=False):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')

    if is_mobile:
        options.add_argument('--window-size=390,844')
        options.add_argument('--user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1')
    else:
        options.add_argument('--window-size=1280,800')

    driver = webdriver.Chrome(options=options)
    try:
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': """
                const origFetch = window.fetch;
                window.fetch = async function(...args) {
                    const url = typeof args[0] === 'string' ? args[0] : (args[0] ? args[0].url : '');
                    if (url.includes('/api/session')) {
                        return new Response(JSON.stringify({
                            authenticated: true,
                            username: 'jeffb',
                            user_id: 1,
                            is_mod: true,
                            rating: 1500
                        }), { status: 200, headers: { 'Content-Type': 'application/json' } });
                    }
                    return origFetch.apply(this, args);
                };
                localStorage.setItem('morpheme_username', 'jeffb');
                sessionStorage.removeItem('morpheme_logged_out');
                localStorage.removeItem('morpheme_logged_out');
                window.currentUserIsMod = true;
            """
        })

        target_url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:5001/'
        print(f"\n--- Testing {'Mobile (390x844)' if is_mobile else 'Desktop (1280x800)'} on {target_url} ---")
        driver.get(target_url)
        time.sleep(1.5)

        # Enter lobby
        gw_btn = driver.find_element(By.ID, 'btn-enter-lobby-gateway')
        gw_btn.click()
        time.sleep(1)

        # 1. TEST MODS TAB
        print("Navigating to Mods tab...")
        driver.execute_script("window.currentUserIsMod = true; window.showPage('page-mods');")
        time.sleep(0.5)

        print("Opening Tally Divisions tab...")
        driver.execute_script("window.showModTab('tally-divisions');")
        time.sleep(1.5)

        badge_text = driver.find_element(By.ID, 'tally-divisions-status-badge').text
        toggle_input = driver.find_element(By.ID, 'tally-divisions-toggle-input')
        print(f"Initial Tally Badge: '{badge_text}', Toggle Switch Checked: {toggle_input.is_selected()}")

        # Test toggling the switch as jeffb
        initial_checked = toggle_input.is_selected()
        driver.execute_script("document.getElementById('tally-divisions-toggle-input').click();")
        time.sleep(1.2)
        assert toggle_input.is_selected() != initial_checked, "Toggle switch failed to flip state"
        print(f"✓ Toggle switch flipped to: {toggle_input.is_selected()} ('{driver.find_element(By.ID, 'tally-divisions-status-badge').text}')")

        # Flip back to initial state
        driver.execute_script("document.getElementById('tally-divisions-toggle-input').click();")
        time.sleep(1.2)
        assert toggle_input.is_selected() == initial_checked, "Toggle switch failed to revert state"
        print(f"✓ Toggle switch cleanly reverted to: {toggle_input.is_selected()} ('{driver.find_element(By.ID, 'tally-divisions-status-badge').text}')")

        # Test repopulate examples
        driver.execute_script("window.loadTallyDivisionsExamples(true);")
        time.sleep(1.5)
        examples_html = driver.find_element(By.ID, 'tally-examples-container').text
        print(f"Examples container loaded ({len(examples_html)} chars)")
        assert "5-Letter Words" in examples_html
        assert "6-Letter Words" in examples_html
        assert "7-Letter Words" in examples_html
        assert "8-Letter Words" in examples_html
        print("✓ All 5L, 6L, 7L, 8L division example sections present!")

        # 2. TEST FIND COUNT IN TOOLS
        print("\nNavigating to Tools -> Find Count...")
        driver.execute_script("window.showPage('page-tools');")
        time.sleep(0.5)
        driver.execute_script("window.showTool('find-count');")
        time.sleep(0.5)

        # Search 5L word 'CREEP'
        fc_input = driver.find_element(By.ID, 'find-count-input')
        fc_input.clear()
        fc_input.send_keys('CREEP')
        fc_btn = driver.find_element(By.ID, 'find-count-btn')
        fc_btn.click()
        time.sleep(1)

        summary_text = driver.find_element(By.ID, 'find-count-summary').text
        print(f"Find Count summary for CREEP: '{summary_text}'")
        assert "Division" in summary_text
        assert "pts" in summary_text
        print("✓ Division and points correctly displayed for 5L word!")

        # Search 7L word 'TESTING' (should default to Division 1 of 5 (5 of 9 pts))
        fc_input.clear()
        fc_input.send_keys('TESTING')
        fc_btn.click()
        time.sleep(1)

        summary_testing = driver.find_element(By.ID, 'find-count-summary').text
        print(f"Find Count summary for TESTING: '{summary_testing}'")
        assert "Division 1 of 5 (5 of 9 pts)" in summary_testing
        print("✓ TESTING correctly defaults to Division 1 of 5 (5 of 9 pts)!")

        # Search 4L word 'BIRD'
        fc_input.clear()
        fc_input.send_keys('BIRD')
        fc_btn.click()
        time.sleep(1)

        summary_text4 = driver.find_element(By.ID, 'find-count-summary').text
        print(f"Find Count summary for BIRD: '{summary_text4}'")
        assert "Division" not in summary_text4
        print("✓ Concept of divisions correctly NOT applied to 4-letter words!")

        # 3. TEST DEFINITION POPUP DIVISION BADGE
        print("\nTesting Definition Popup Division Badge for TESTING...")
        driver.execute_script("window.showWordDefinitionPopup('TESTING');")
        time.sleep(1.5)

        div_badge = driver.find_element(By.ID, 'tool-def-division-badge')
        badge_text = div_badge.text
        print(f"Definition popup division badge for TESTING: '{badge_text}'")
        assert "Div 1 of 5 (5 of 9 pts)" in badge_text
        print("✓ Definition popup shows correct dynamic division and points for TESTING!")

        # Test definition popup for larger words (15L and 20L)
        driver.execute_script("window.showWordDefinitionPopup('SUBCOMPARTMENTS');")
        time.sleep(1.2)
        badge_text_15L = driver.find_element(By.ID, 'tool-def-division-badge').text
        print(f"Definition popup division badge for SUBCOMPARTMENTS (15L): '{badge_text_15L}'")
        assert "Div 1 of 5 (37 of 41 pts)" in badge_text_15L
        print("✓ Definition popup shows correct 5 divisions for 15L word SUBCOMPARTMENTS!")

        driver.execute_script("window.showWordDefinitionPopup('ILIOTROCHANTERICUSES');")
        time.sleep(1.2)
        badge_text_20L = driver.find_element(By.ID, 'tool-def-division-badge').text
        print(f"Definition popup division badge for ILIOTROCHANTERICUSES (20L): '{badge_text_20L}'")
        assert "Div 1 of 5 (57 of 61 pts)" in badge_text_20L
        print("✓ Definition popup shows correct 5 divisions for 20L word ILIOTROCHANTERICUSES!")

        # Test definition popup for 4L word BIRD
        driver.execute_script("window.showWordDefinitionPopup('BIRD');")
        time.sleep(1.5)
        badge_text_bird = driver.find_element(By.ID, 'tool-def-division-badge').text
        print(f"Definition popup division badge for BIRD: '{badge_text_bird}'")
        assert "1 pt" in badge_text_bird
        print("✓ Definition popup shows correct points (1 pt) for 4L word BIRD!")

        # Check JS errors relating to our features or uncaught exceptions
        logs = driver.get_log('browser')
        severe = [l for l in logs if l['level'] == 'SEVERE' and ('tally' in l['message'].lower() or 'find-count' in l['message'].lower() or 'syntaxerror' in l['message'].lower() or 'referenceerror' in l['message'].lower())]
        print(f"Severe JS errors for features ({len(severe)}):")
        for err in severe:
            print("  ", err['message'])
        assert len(severe) == 0, f"Found {len(severe)} severe JS errors"

        print("✓ Test completed successfully!")

    finally:
        driver.quit()

if __name__ == '__main__':
    run_test(is_mobile=False)
    run_test(is_mobile=True)
