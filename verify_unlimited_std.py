import urllib.request
import urllib.parse
import json
import http.cookiejar

BASE_URL = "http://localhost:3000"

def make_request(url, data=None, cookies=None, method='GET'):
    cj = http.cookiejar.CookieJar()
    if cookies:
        for cookie in cookies:
            cj.set_cookie(cookie)
    
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    
    req_data = None
    if data is not None:
        req_data = json.dumps(data).encode('utf-8')
    
    req = urllib.request.Request(url, data=req_data, method=method)
    req.add_header('Content-Type', 'application/json')
    
    try:
        with opener.open(req) as response:
            res_data = response.read().decode('utf-8')
            return json.loads(res_data), cj
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.read().decode('utf-8')}")
        return None, cj

def test_unlimited_players():
    # 1. Guest Login
    print("Logging in as guest...")
    data, cj = make_request(f"{BASE_URL}/api/guest-login", data={}, method='POST')
    if not data or not data.get('success'):
        print("Login failed")
        return
    
    session = cj
    print(f"Logged in as {data.get('username')}")

    # 2. Create an Accumulative room (10m)
    print("Creating Accumulative room...")
    data, _ = make_request(f"{BASE_URL}/api/room/create", data={
        "game_type": "accumulative",
        "time_limit": 600,
        "board_dimensions": "4x4"
    }, cookies=session, method='POST')
    
    if not data or not data.get('success'):
        print("Create failed")
        return
        
    room_id = data['room_id']
    print(f"Created room: {room_id}")

    # 3. Verify room listing has max_players
    print("Checking room listing...")
    data, _ = make_request(f"{BASE_URL}/api/rooms?game_type=accumulative&board_dimensions=4x4&time_limit=600")
    
    rooms = data.get('rooms', [])
    if not rooms:
        print("No rooms found")
        return
        
    room_info = rooms[0]
    print(f"Room Info: {room_info}")
    if room_info.get('max_players') == 9999:
        print("SUCCESS: max_players is 9999")
    else:
        print(f"FAILURE: max_players is {room_info.get('max_players')}")

    # 4. Simulate joining many people
    print("Joining 10 players...")
    for i in range(10):
        # New guest session for each
        _, guest_cj = make_request(f"{BASE_URL}/api/guest-login", data={}, method='POST')
        join_data, _ = make_request(f"{BASE_URL}/api/room/{room_id}/join", data={}, cookies=guest_cj, method='POST')
        
        if join_data and join_data.get('success'):
            print(f"Player {i+1} joined")
        else:
            print(f"Player {i+1} failed to join")
            break
    else:
        print("SUCCESS: Joined 10 players to Accumulative room")

    # 5. Check status
    print("Final status check...")
    data, _ = make_request(f"{BASE_URL}/api/room/{room_id}/state", cookies=session)
    if not data:
        print("State check failed")
        return
        
    players = data.get('players', [])
    print(f"Player count in state: {len(players)}")
    if len(players) >= 11:
        print("SUCCESS: Final player count verified")
    else:
        print(f"FAILURE: Final player count is {len(players)}")

if __name__ == "__main__":
    test_unlimited_players()
