import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=20)
    child.logfile = sys.stdout
    
    child.expect([r"Are you sure you want to continue connecting", r"[Pp]assword:"])
    child.sendline(password)
    child.expect([r"\$", r"#", r">"])
    
    child.sendline("""cat << 'EOF' > /tmp/check_rh.py
import sqlite3
import json
conn = sqlite3.connect('/home/morpheme/morpheme/morpheme.db')
c = conn.cursor()
try:
    c.execute('SELECT room_id, round_number, username, invalid_words FROM round_history ORDER BY id DESC LIMIT 1000')
    rows = c.fetchall()
    print('Recent round history invalid words:')
    for row in rows:
        invalids = json.loads(row[3]) if row[3] else []
        if any(w in ['SHALST', 'ASSRUN', 'SPLASHT', 'STHAL'] for w in [x.upper() for x in invalids]):
            print(f"  Room: {row[0]}, Round: {row[1]}, User: {row[2]}, Invalids: {invalids}")
except Exception as e:
    print('Error:', e)

print('=== END OF CHECK ===')
conn.close()
EOF
""")
    child.expect([r"\$", r"#"])
    
    child.sendline("python3 /tmp/check_rh.py")
    child.expect("=== END OF CHECK ===")
    print("\n--- DB Invalid Words Search on Remote ---")
    print(child.before)
    child.expect([r"\$", r"#"])
    
    child.sendline("rm /tmp/check_rh.py")
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
