
import sys
import os
import time

# Mocking Flask environment partially
sys.path.append('/Users/jeffbabiak/')
from app import room_manager

def dump_rooms():
    print(f"Active rooms: {list(room_manager.rooms.keys())}")
    for rid, room in room_manager.rooms.items():
        print(f"Room {rid}: State={room.state}, Round={room.current_round}")
        for p in room.players:
            print(f"  Player {p.username}: Score={p.score}, Rating={p.rating}, Change={p.rating_change}")

if __name__ == "__main__":
    while True:
        dump_rooms()
        time.sleep(5)
