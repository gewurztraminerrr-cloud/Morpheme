import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=20)
    child.logfile = sys.stdout
    
    # Handle optional authenticity prompt
    idx = child.expect([r"Are you sure you want to continue connecting", r"[Pp]assword:"])
    if idx == 0:
        child.sendline("yes")
        child.expect(r"[Pp]assword:")
    child.sendline(password)
    
    child.expect([r"\$", r"#", r">"])
    
    # Create remote script to check columns and cache stats
    child.sendline("""cat << 'EOF' > /tmp/check_db.py
import sqlite3
conn = sqlite3.connect('/home/morpheme/morpheme/morpheme.db')
c = conn.cursor()
try:
    c.execute('PRAGMA table_info(pregenerated_boards)')
    print('Columns of pregenerated_boards:', c.fetchall())
except Exception as e:
    print('Error table_info:', e)

try:
    c.execute('SELECT COUNT(*) FROM pregenerated_boards')
    print('Total pregenerated:', c.fetchone()[0])
except Exception as e:
    print('Error count:', e)

try:
    c.execute('SELECT param_key, COUNT(*) FROM pregenerated_boards GROUP BY param_key')
    print('Pregenerated counts by param_key:')
    for row in c.fetchall():
        print(f"  {row[0][:100]}...: {row[1]}")
except Exception as e:
    print('Error group count:', e)

print('=== END OF CHECK ===')
conn.close()
EOF
""")
    child.expect([r"\$", r"#"])
    
    # Run the script
    child.sendline("python3 /tmp/check_db.py")
    child.expect("=== END OF CHECK ===")
    print("\n--- DB Column and Cache Info on Remote ---")
    print(child.before)
    child.expect([r"\$", r"#"])
    
    child.sendline("rm /tmp/check_db.py")
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
