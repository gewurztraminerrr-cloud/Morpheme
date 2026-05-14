import urllib.request
import urllib.parse
import http.cookiejar
import json

def test_flow():
    # Setup cookie handler for session persistence
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    urllib.request.install_opener(opener)
    
    # 1. Guest login
    print("--- 1. Guest Login ---")
    req = urllib.request.Request(
        'http://127.0.0.1:5001/api/guest-login',
        data=b'',
        method='POST'
    )
    try:
        with opener.open(req) as response:
            body = response.read().decode('utf-8')
            login_data = json.loads(body)
            print(json.dumps(login_data, indent=2))
            guest_username = login_data['username']
    except Exception as e:
        print("Guest Login failed:", e)
        return
        
    # 2. Get rooms for accumulative format
    print("\n--- 2. Get Rooms ---")
    url = 'http://127.0.0.1:5001/api/rooms?game_type=accumulative&board_dimensions=4x4&time_limit=45'
    try:
        with opener.open(url) as response:
            body = response.read().decode('utf-8')
            rooms_data = json.loads(body)
            print(json.dumps(rooms_data, indent=2))
    except Exception as e:
        print("Get Rooms failed:", e)
        return
    
    room_id = None
    if rooms_data.get('rooms'):
        room_id = rooms_data['rooms'][0]['room_id']
        print(f"Found existing room: {room_id}")
    else:
        # Create room
        print("\n--- 2b. Create Room ---")
        create_data_bytes = json.dumps({
            'game_type': 'accumulative',
            'time_limit': 45,
            'board_dimensions': '4x4'
        }).encode('utf-8')
        req = urllib.request.Request(
            'http://127.0.0.1:5001/api/room/create',
            data=create_data_bytes,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with opener.open(req) as response:
                body = response.read().decode('utf-8')
                create_data = json.loads(body)
                print(json.dumps(create_data, indent=2))
                if create_data.get('success'):
                    room_id = create_data['room_id']
        except Exception as e:
            print("Create Room failed:", e)
            return
            
    if not room_id:
        print("Failed to get or create room")
        return
        
    # 3. Join Room
    print("\n--- 3. Join Room ---")
    join_data_bytes = json.dumps({
        'as_spectator': False
    }).encode('utf-8')
    req = urllib.request.Request(
        f'http://127.0.0.1:5001/api/room/{room_id}/join',
        data=join_data_bytes,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with opener.open(req) as response:
            body = response.read().decode('utf-8')
            join_data = json.loads(body)
            print(json.dumps(join_data, indent=2))
    except Exception as e:
        if hasattr(e, 'read'):
            print("Join Room failed:", e.read().decode('utf-8'))
        else:
            print("Join Room failed:", e)
        return
        
    # 4. Get Room State
    print("\n--- 4. Get Room State ---")
    try:
        with opener.open(f'http://127.0.0.1:5001/api/room/{room_id}/state') as response:
            body = response.read().decode('utf-8')
            state_data = json.loads(body)
            print(f"Room State keys: {list(state_data.keys())}")
            print(f"Room state field: {state_data.get('state')}")
            print(f"Board (is None?): {state_data.get('board') is None}")
            print(f"Board dimensions: {state_data.get('board_dimensions')}")
            print(f"Players count: {len(state_data.get('players', []))}")
            print("Players detail:")
            print(json.dumps(state_data.get('players'), indent=2))
    except Exception as e:
        if hasattr(e, 'read'):
            print("Get Room State failed:", e.read().decode('utf-8'))
        else:
            print("Get Room State failed:", e)

if __name__ == '__main__':
    test_flow()
