
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

def verify_stats():
    print(f"Verifying Lobby Stats against {BASE_URL}...")
    
    # 1. Check Initial Stats
    print("\n1. Checking initial stats...")
    data = json_request(f"{BASE_URL}/api/lobby-stats")
    if not data: return False
    print(f"Initial stats: {data}")
    
    # 2. Login
    print("\n2. Logging in...")
    json_request(f"{BASE_URL}/api/guest-login", method='POST')
    
    # 3. Create Room (Accumulative, 4x4, 60s)
    print("\n3. Creating room (accumulative, 4x4, 60s)...")
    room_config = {
        'game_type': 'accumulative',
        'time_limit': 60,
        'board_dimensions': '4x4'
    }
    resp = json_request(f"{BASE_URL}/api/room/create", method='POST', data=room_config)
    room_id = resp.get('room_id')
    print(f"Room: {room_id}")
    
    # 4. Check Stats - Should be 1
    print("\n4. Checking stats (Expect 1)...")
    data = json_request(f"{BASE_URL}/api/lobby-stats")
    stats = data.get('stats', {})
    key = "accumulative|4x4|60"
    count = stats.get(key, 0)
    print(f"Stats for {key}: {count}")
    
    if count != 1:
        print("FAILURE: Expected count 1")
        return False
        
    # 5. Leave Room
    print("\n5. Leaving room...")
    json_request(f"{BASE_URL}/api/room/{room_id}/leave", method='POST')
    
    # 6. Check Stats - Should be 0 (key might be missing or 0)
    print("\n6. Checking stats (Expect 0 or missing)...")
    data = json_request(f"{BASE_URL}/api/lobby-stats")
    stats = data.get('stats', {})
    count = stats.get(key, 0)
    print(f"Stats for {key}: {count}")
    
    if count != 0:
        print("FAILURE: Expected count 0")
        return False
        
    print("\nSUCCESS: Lobby stats tracked active players correctly.")
    return True

if __name__ == '__main__':
    if not verify_stats():
        sys.exit(1)
