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
    
    # Run python query
    py_cmd = "python3 -c \"import sqlite3, json; conn = sqlite3.connect('/home/morpheme/morpheme/morpheme.db'); cursor = conn.cursor(); cursor.execute('SELECT room_id, board_data, bonus_word, board_format FROM active_boards'); [print(f'Room: {r}\\nFormat: {fmt}\\nBonus: {b}\\nBoard: {bd}\\n') for r, bd, b, fmt in cursor.fetchall()]\""
    child.sendline(py_cmd)
    
    child.expect([r"\$", r"#"])
    print("\n--- Active Boards on Remote ---")
    print(child.before)
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
