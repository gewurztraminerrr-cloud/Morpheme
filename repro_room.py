
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
    except urllib.error.HTTPError as e:
        # Handle 404 for room deletion check
        if e.code == 404 and 'room' in url and method == 'GET':
            return {'status': 404}
        print(f"HTTP Error: {e.code} {e.reason}")
        try:
            print(e.read().decode('utf-8'))
        except:
            pass
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_room_lifecycle():
    print(f"Testing room lifecycle against {BASE_URL}...")
    
    # 1. Login/Guest Login
    print("\n1. Logging in as guest...")
    data = json_request(f"{BASE_URL}/api/guest-login", method='POST')
    if not data: return False
    print(f"Logged in as: {data.get('username')}")
    
    # 2. Create Room
    print("\n2. Creating room...")
    room_config = {
        'game_type': 'accumulative',
        'time_limit': 60,
        'board_dimensions': '4x4'
    }
    data = json_request(f"{BASE_URL}/api/room/create", method='POST', data=room_config)
    if not data: return False
    room_id = data.get('room_id')
    print(f"Room created: {room_id}")
    
    # 3. Verify Room Exists
    print("\n3. Verifying room exists...")
    data = json_request(f"{BASE_URL}/api/room/{room_id}/state")
    if not data: return False
    print("Room is active and accessible.")
    
    # 4. Leave Room
    print("\n4. Leaving room...")
    data = json_request(f"{BASE_URL}/api/room/{room_id}/leave", method='POST')
    if not data: return False
    print("Left room successfully.")
    
    # 5. Check if Room Deleted
    print("\n5. Checking if room is deleted...")
    data = json_request(f"{BASE_URL}/api/room/{room_id}/state")
    
    if data and data.get('status') == 404:
        print("SUCCESS: Room was deleted (404 Not Found).")
        return True
    elif data:
        players = data.get('players', [])
        print(f"FAILURE: Room still exists with {len(players)} players.")
        print(f"Players in room: {[p['username'] for p in players]}")
        return False
    else:
        return False

if __name__ == '__main__':
    try:
        success = test_room_lifecycle()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
