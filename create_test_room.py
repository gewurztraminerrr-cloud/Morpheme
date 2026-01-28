import requests
import json
import time

BASE_URL = 'http://localhost:3000'

def create_room():
    print("Creating room...")
    # 1. Session for Host
    s = requests.Session()
    # Login as Host
    s.post(f'{BASE_URL}/api/guest-login')
    
    # Create Room
    resp = s.post(f'{BASE_URL}/api/room/create', json={
        'game_type': 'fcfs',
        'time_limit': 300,
        'board_dimensions': '4x4'
    })
    print("Create resp:", resp.text)
    data = resp.json()
    room_id = data['room_id']
    print(f"Room created: {room_id}")
    return room_id

if __name__ == "__main__":
    create_room()
