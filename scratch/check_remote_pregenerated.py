import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=20)
    child.expect([r'[Pp]assword:'])
    child.sendline(password)
    child.expect(['morpheme@249:'])
    
    cmd = """cat << 'EOF' > purge_planets.py
import sqlite3

conn = sqlite3.connect('/home/morpheme/morpheme/morpheme.db')
cursor = conn.cursor()

rows = cursor.execute("SELECT id, param_key, board_json FROM pregenerated_boards").fetchall()
print("Total pregenerated boards in DB:", len(rows))

planets_ids = [r[0] for r in rows if "PLANETS" in str(r[1]).upper() or "PLANETS" in str(r[2]).upper()]
print("Found pregenerated boards containing PLANETS:", len(planets_ids))

if planets_ids:
    placeholders = ",".join("?" for _ in planets_ids)
    cursor.execute(f"DELETE FROM pregenerated_boards WHERE id IN ({placeholders})", planets_ids)
    conn.commit()
    print(f"Purged {len(planets_ids)} pregenerated boards with PLANETS.")
else:
    print("Zero pregenerated PLANETS boards found in DB.")

EOF
python3 purge_planets.py
"""
    child.sendline(cmd)
    child.expect(['morpheme@249:'])
    print(child.before)
    child.sendline('exit')
    child.close()

if __name__ == '__main__':
    main()
