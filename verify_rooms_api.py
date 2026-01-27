
import urllib.request
import urllib.parse
import json
import http.cookiejar
import sys
import time

BASE_URL = 'http://localhost:3000'
cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

def json_request(url, method='GET', data=None):
    try:
        req = urllib.request.Request(url, method=method)
        req.add_header('Content-Type', 'application/json')
        
        if data:
            json_data = json.dumps(data).encode('utf-8')
            req.data = json_data
            
        with opener.open(req) as response:
            if response.status != 200:
                print(f"Request failed: {url} -> {response.status}")
                return None
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Error ({url}): {e}")
        return None

def verify_rooms_list():
    print(f"Verifying Rooms List API against {BASE_URL}...")
    
    # 1. Login
    print("\n1. Logging in...")
    login_data = json_request(f"{BASE_URL}/api/guest-login", method='POST')
    username = login_data.get('username')
    print(f"Logged in as: {username}")
    
    # 2. Create Room (FCFS, 6x8, 180s) - using create directly to ensure it exists
    print("\n2. Creating room (fcfs, 6x8, 180s)...")
    room_config = {
        'game_type': 'fcfs',
        'time_limit': 180,
        'board_dimensions': '6x8'
    }
    resp = json_request(f"{BASE_URL}/api/room/create", method='POST', data=room_config)
    room_id = resp.get('room_id')
    print(f"Room: {room_id}")
    
    # 3. List Rooms
    print("\n3. Listing rooms for fcfs/6x8/180...")
    params = urllib.parse.urlencode({
        'game_type': 'fcfs',
        'time_limit': 180,
        'board_dimensions': '6x8'
    })
    data = json_request(f"{BASE_URL}/api/rooms?{params}")
    
    rooms = data.get('rooms', [])
    print(f"Found {len(rooms)} rooms")
    
    target_room = next((r for r in rooms if r['room_id'] == room_id), None)
    
    if target_room:
        print(f"Target Room Found!")
        players = target_room.get('players', [])
        print(f"Players: {players}")
        
        # Check if username is in players
        player_names = [p['username'] for p in players]
        if username in player_names:
             print("SUCCESS: Current user found in room players list.")
             return True
        else:
             print("FAILURE: Current user NOT found in room players list.")
             return False
    else:
        print("FAILURE: Room not in list.")
        return False

if __name__ == '__main__':
    if not verify_rooms_list():
        sys.exit(1)
