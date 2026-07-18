import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=20)
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect([r"\$", r"#"])
    
    # Run python query to inspect rooms
    py_cmd = (
        "python3 -c \""
        "import sqlite3, sys; "
        "sys.path.append('/home/morpheme/morpheme'); "
        "from game_room import room_manager; "
        "print('Rooms:'); "
        "for rid, r in room_manager.rooms.items(): "
        "    print(f'Room: {rid}, state: {r.state}, players: {len(r.players)}, loading: {getattr(r, \"board_search_loading\", False)}, started: {getattr(r, \"board_search_started\", False)}'); "
        "\""
    )
    child.sendline(py_cmd)
    
    child.expect([r"\$", r"#"])
    print("\n--- Room Status ---")
    print(child.before)
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
