import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    cmd = f'ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no morpheme@{ip}'
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(cmd, encoding='utf-8', timeout=20)
    
    child.expect([r"password:"])
    child.sendline(password)
    child.expect([r"\$", r"#"])
    
    # Run python query
    py_cmd = "python3 -c \"import sqlite3; conn = sqlite3.connect('/home/morpheme/morpheme/morpheme.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM pregenerated_boards'); print('Pregenerated count:', cursor.fetchone()[0]); cursor.execute('SELECT DISTINCT param_key FROM pregenerated_boards LIMIT 10'); [print(r[0]) for r in cursor.fetchall()]\""
    child.sendline(py_cmd)
    
    child.expect([r"\$", r"#"])
    print("\n--- Remote Cache Info ---")
    print(child.before)
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
