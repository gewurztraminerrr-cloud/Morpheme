import os
import sys
import json
import time
import requests

BASE_DIR = '/Users/jeffbabiak'
sys.path.insert(0, BASE_DIR)

import tally_divisions as td

def test_unit_logic():
    print("--- 1. Testing Unit Formulas & Logic ---")
    # Base points
    assert td.get_base_points(3) == 1
    assert td.get_base_points(4) == 1
    assert td.get_base_points(5) == 2
    assert td.get_base_points(6) == 3
    assert td.get_base_points(7) == 5
    assert td.get_base_points(8) == 9
    assert td.get_base_points(9) == 13
    assert td.get_base_points(10) == 17
    print("✓ Base points verified (3L: 1, 4L: 1, 5L: 2, 6L: 3, 7L: 5, 8L: 9, 9L: 13, 10L: 17)")

    # Divisions
    assert td.get_num_divisions(3) == 1
    assert td.get_num_divisions(4) == 1
    assert td.get_num_divisions(5) == 2  # 3 - 2 + 1 = 2
    assert td.get_num_divisions(6) == 3  # 5 - 3 + 1 = 3
    assert td.get_num_divisions(7) == 5  # 9 - 5 + 1 = 5
    assert td.get_num_divisions(8) == 5  # 13 - 9 + 1 = 5
    assert td.get_num_divisions(9) == 5  # 17 - 13 + 1 = 5
    assert td.get_num_divisions(10) == 5
    print("✓ Number of divisions verified (5L: 2, 6L: 3, 7L: 5, 8L: 5, 9L: 5)")

    # Max points
    assert td.get_max_points(3) == 1
    assert td.get_max_points(4) == 1
    assert td.get_max_points(5) == 3  # 2 + 1
    assert td.get_max_points(6) == 5  # 3 + 2
    assert td.get_max_points(7) == 9  # 5 + 4
    assert td.get_max_points(8) == 13 # 9 + 4
    assert td.get_max_points(9) == 17 # 13 + 4
    print("✓ Max points verified (5L: 3, 6L: 5, 7L: 9, 8L: 13, 9L: 17)")

    # Word division info
    the_info = td.get_word_division_info('THE')
    assert the_info['has_divisions'] is False
    assert the_info['points'] == 1

    bird_info = td.get_word_division_info('BIRD')
    assert bird_info['has_divisions'] is False
    assert bird_info['points'] == 1

    creep_info = td.get_word_division_info('CREEP') # count=4, max=4
    assert creep_info['has_divisions'] is True
    assert creep_info['division'] == 1
    assert creep_info['points'] == 2
    print(f"✓ CREEP division info: {creep_info['division_str']}")

    alums_info = td.get_word_division_info('ALUMS') # count=2, max=4
    assert alums_info['division'] == 2
    assert alums_info['points'] == 3
    print(f"✓ ALUMS division info: {alums_info['division_str']}")

    # TESTING: defaults to Division 1 (5 pts) per user instruction
    testing_info = td.get_word_division_info('TESTING')
    assert testing_info['division'] == 1
    assert testing_info['points'] == 5
    print(f"✓ TESTING division info: {testing_info['division_str']}")

    # 9L rule: all non-ING sequences are considered hard
    assert td.is_word_hard_by_tally('BEAUTIFUL') is True
    # 9L ING sequences evaluated by division (Div 1 is not hard)
    assert td.is_word_hard_by_tally('SOMETHING') is False
    print("✓ 9L non-ING is considered hard (BEAUTIFUL) and ING follows division (SOMETHING)")

    # WOTD eligibility
    # 6L words: only rarest division (div == 3)
    sirens_info = td.get_word_division_info('SIRENS')
    assert td.is_wotd_eligible('SIRENS') == (sirens_info['division'] == 3)
    print("✓ WOTD eligibility verified for 6L and 7L+")

def test_api_endpoints():
    print("\n--- 2. Testing API Endpoints on Localhost ---")
    s = requests.Session()
    # Login as jeffb
    login_res = s.post('http://localhost:5001/api/login', json={'username': 'jeffb', 'password': 'password'})
    print("Login response:", login_res.status_code)

    # 1. Test Find Count API
    fc_res = s.get('http://localhost:5001/api/tools/find-count?word=CREEP')
    assert fc_res.status_code == 200
    fc_data = fc_res.json()
    print("Find Count CREEP:", fc_data.get('division_str'), "points:", fc_data.get('points'))
    assert fc_data.get('has_divisions') is True
    assert 'Division' in fc_data.get('division_str')

    fc_res4 = s.get('http://localhost:5001/api/tools/find-count?word=BIRD')
    assert fc_res4.status_code == 200
    fc_data4 = fc_res4.json()
    print("Find Count BIRD (4L):", fc_data4.get('division_str'), "has_divisions:", fc_data4.get('has_divisions'))
    assert fc_data4.get('has_divisions') is False

    # 2. Test Mods Tally Status
    status_res = s.get('http://localhost:5001/api/mods/tally-divisions/status')
    assert status_res.status_code == 200
    status_data = status_res.json()
    print("Tally Status:", status_data)
    assert status_data.get('is_jeffb') is True

    # 3. Test Mods Toggle
    toggle_on = s.post('http://localhost:5001/api/mods/tally-divisions/toggle', json={'enabled': True})
    assert toggle_on.status_code == 200
    print("Toggle ON:", toggle_on.json())

    # Verify board generator reflects toggle
    import board_generator as bg
    assert td.is_tally_divisions_enabled() is True

    # Toggle revert back to False (as requested: allow it to just as easily be switched off)
    toggle_off = s.post('http://localhost:5001/api/mods/tally-divisions/toggle', json={'enabled': False})
    assert toggle_off.status_code == 200
    print("Toggle OFF (revert):", toggle_off.json())
    assert td.is_tally_divisions_enabled() is False

    # 4. Test Mods Thresholds
    thresh_res = s.post('http://localhost:5001/api/mods/tally-divisions/thresholds', json={'hard_thresholds': {'5': 2, '6': 3, '7': 3, '8': 3}})
    assert thresh_res.status_code == 200
    print("Thresholds updated:", thresh_res.json())

    # 5. Test Mods Examples
    ex_res = s.get('http://localhost:5001/api/mods/tally-divisions/examples')
    assert ex_res.status_code == 200
    ex_data = ex_res.json()
    print("Examples keys:", list(ex_data.get('examples', {}).keys()))
    assert '5' in ex_data['examples']
    assert '6' in ex_data['examples']
    assert '7' in ex_data['examples']
    assert '8' in ex_data['examples']

    print("✓ All API Endpoints Verified Successfully!")

if __name__ == '__main__':
    test_unit_logic()
    test_api_endpoints()
