import requests
import time

BASE_URL = "http://localhost:3000"

def test_unlimited_players():
    # 1. Create an Accumulative room (10m)
    print("Creating Accumulative room...")
    resp = requests.post(f"{BASE_URL}/api/guest-login")
    session = resp.cookies
    
    create_resp = requests.post(f"{BASE_URL}/api/room/create", json={
        "game_type": "accumulative",
        "time_limit": 600,
        "board_dimensions": "4x4"
    }, cookies=session)
    room_data = create_resp.json()
    room_id = room_data['room_id']
    print(f"Created room: {room_id}")

    # 2. Verify room listing has max_players
    print("Checking room listing...")
    list_resp = requests.get(f"{BASE_URL}/api/rooms?game_type=accumulative&board_dimensions=4x4&time_limit=600")
    list_data = list_resp.json()
    room_info = list_data['rooms'][0]
    print(f"Room Info: {room_info}")
    if room_info.get('max_players') == 9999:
        print("SUCCESS: max_players is 9999")
    else:
        print(f"FAILURE: max_players is {room_info.get('max_players')}")

    # 3. Simulate joining many people (e.g. 10)
    print("Joining 10 players...")
    for i in range(10):
        guest_resp = requests.post(f"{BASE_URL}/api/guest-login")
        guest_session = guest_resp.cookies
        join_resp = requests.post(f"{BASE_URL}/api/room/{room_id}/join", json={}, cookies=guest_session)
        print(f"Player {i+1} join status: {join_resp.status_code}")
        if join_resp.status_code != 200:
            print(f"FAILURE: Join failed: {join_resp.json()}")
            break
    else:
        print("SUCCESS: Joined 10 players to Accumulative room")

    # 4. Check status
    status_resp = requests.get(f"{BASE_URL}/api/room/{room_id}/state", cookies=session)
    status_data = status_resp.json()
    print(f"Player count in state: {len(status_data['players'])}")
    if len(status_data['players']) >= 11: # 1 creator + 10 joined
        print("SUCCESS: Player count verified")
    else:
        print(f"FAILURE: Player count is {len(status_data['players'])}")

if __name__ == "__main__":
    try:
        test_unlimited_players()
    except Exception as e:
        print(f"Error: {e}")
