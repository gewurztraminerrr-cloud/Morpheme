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
    # Create remote script
    child.sendline("""cat << 'EOF' > /tmp/check_db.py
import sqlite3
conn = sqlite3.connect('/home/morpheme/morpheme/morpheme.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*), COUNT(CASE WHEN param_key LIKE "%AW%" THEN 1 END) FROM pregenerated_boards')
total, aw_count = cursor.fetchone()
print(f'Total pregenerated: {total}, AW count: {aw_count}')
cursor.execute('SELECT DISTINCT param_key FROM pregenerated_boards')
for r in cursor.fetchall():
    print(r[0])
conn.close()
EOF
""")
    child.expect([r"\$", r"#"])
    
    # Run the script
    child.sendline("python3 /tmp/check_db.py")
    child.expect([r"\$", r"#"])
    print("\n--- DB Search Results on Remote ---")
    print(child.before)
    
    child.sendline("rm /tmp/check_db.py")
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
