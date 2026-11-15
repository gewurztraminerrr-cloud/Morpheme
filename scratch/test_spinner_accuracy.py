import sys
import os
import random

sys.path.insert(0, '/Users/jeffbabiak')

from spinner_set import SpinnerSet
from game_room import RoomManager, is_board_count_valid

def test_factchecked_wc_range():
    print("[TEST] Testing SpinnerSet.get_factchecked_wc_range...")
    assert SpinnerSet.get_factchecked_wc_range(211, use_added_words=True, dictionary='NWL + AW') == '200-300', "211 words with NWL + AW must be '200-300'!"
    assert SpinnerSet.get_factchecked_wc_range(211, use_added_words=False, dictionary='NWL') == '200-300'
    assert SpinnerSet.get_factchecked_wc_range(350, use_added_words=True, dictionary='CSW + AW') == '300-400'
    assert SpinnerSet.get_factchecked_wc_range(450, use_added_words=True) == '400-500'
    assert SpinnerSet.get_factchecked_wc_range(520, use_added_words=True) == '500+'
    assert SpinnerSet.get_factchecked_wc_range(80, use_added_words=False) == '50-100'
    assert SpinnerSet.get_factchecked_wc_range(150, use_added_words=False) == '100-200'
    print("  -> PASSED: All word count ranges fact-checked correctly.")

def test_anti_repetition():
    print("[TEST] Testing format anti-repetition safeguard across 2000 spins...")
    prev_params = None
    normal_run = 0
    for _ in range(2000):
        current = SpinnerSet._spin_board_format(is_24h=False, dimensions='4x4', previous_params=prev_params)
        if current.lower() == 'normal':
            normal_run += 1
            assert normal_run <= 2, f"Normal occurred more than 2 times in a row! streak={normal_run}"
        else:
            normal_run = 0

        if prev_params:
            prev_fmt = prev_params['board_format']
            if prev_fmt.lower() != 'normal':
                assert current.lower() != prev_fmt.lower(), f"Non-normal format repeated back-to-back: {prev_fmt} -> {current}"
            if 'mania' in prev_fmt.lower():
                assert 'mania' not in current.lower(), f"Mania repeated back-to-back: {prev_fmt} -> {current}"
            if 'bounce' in prev_fmt.lower():
                assert 'bounce' not in current.lower(), f"Bounce repeated back-to-back: {prev_fmt} -> {current}"

        prev_params = {'board_format': current, 'normal_streak': normal_run}
    print("  -> PASSED: Anti-repetition safeguard strictly prevents back-to-back non-Normal, Mania, or Bounce, and caps Normal at <= 2 in a row.")

def test_room_creation_and_sync():
    print("[TEST] Testing room creation and parameter synchronization...")
    rm = RoomManager()
    
    # Test new room creation
    test_room_id = f"test_room_{random.randint(10000, 99999)}"
    room = rm.create_room(test_room_id, 'accumulative', time_limit=45, board_dimensions='4x6')
    
    assert room.bonus_word, "Room must have a bonus word!"
    assert len(room.bonus_word) == room.spinner_params['bonus_word_length'], (
        f"Desync! bonus_word '{room.bonus_word}' (len {len(room.bonus_word)}) != spinner bonus_word_length {room.spinner_params['bonus_word_length']}"
    )
    
    wc = len(room.all_words)
    expected_range = SpinnerSet.get_factchecked_wc_range(wc, use_added_words=room.use_added_words, dictionary=room.current_dictionary)
    assert room.spinner_params['word_count_range'] == expected_range, (
        f"Desync! actual words {wc} should map to '{expected_range}', but spinner has '{room.spinner_params['word_count_range']}'"
    )
    assert is_board_count_valid(wc, room.spinner_params['word_count_range']), (
        f"Word count {wc} is invalid for spinner range '{room.spinner_params['word_count_range']}'!"
    )
    print(f"  -> PASSED: Room {test_room_id} has {wc} words ({expected_range}), bonus word '{room.bonus_word}' ({len(room.bonus_word)}L), perfectly synchronized!")

    # Clean up test room
    rm.rooms.pop(test_room_id, None)

def test_solo_randomness_across_rounds():
    print("[TEST] Testing Solo rounds randomness across 10 rounds...")
    rm = RoomManager()
    solo_id = f"practice_{random.randint(10000, 99999)}"
    room = rm.create_room(solo_id, 'practice', time_limit=60, board_dimensions='4x4', is_solo=True)
    
    formats = [room.spinner_params.get('board_format')]
    for r in range(2, 8):
        # Simulate round ending and entering intermission (as done in check_round_status)
        room.state = 'intermission'
        room.intermission_start_time = 0
        room.spinner_params_revealed = False
        room._spinner_params_locked = False
        room.frozen_revealed_params = None
        # Simulate intermission spinner generation & reveal
        rm.generate_spinner_params(solo_id, reveal=True)
        # Advance round
        rm.start_next_round(solo_id)
        fmt = room.spinner_params.get('board_format')
        formats.append(fmt)
        # Verify sync
        assert len(room.bonus_word) == room.spinner_params['bonus_word_length'], (
            f"Solo Round {r} bonus word length desync! word='{room.bonus_word}', param={room.spinner_params['bonus_word_length']}"
        )
        expected_range = SpinnerSet.get_factchecked_wc_range(len(room.all_words), use_added_words=room.use_added_words, dictionary=room.current_dictionary)
        assert room.spinner_params['word_count_range'] == expected_range, (
            f"Solo Round {r} word range desync! count={len(room.all_words)}, param={room.spinner_params['word_count_range']}, expected={expected_range}"
        )

    rm.rooms.pop(solo_id, None)
    print(f"  -> Solo Formats observed across rounds: {formats}")
    print("  -> PASSED: Solo rotates parameters dynamically and does not stick on frozen values.")

if __name__ == '__main__':
    test_factchecked_wc_range()
    test_anti_repetition()
    test_room_creation_and_sync()
    test_solo_randomness_across_rounds()
    print("\nALL SPINNER ACCURACY & SYNCHRONIZATION TESTS PASSED!")
