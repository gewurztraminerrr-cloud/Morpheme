import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=20)
    child.logfile = sys.stdout
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect([r"\$", r"#"])
    # Create remote script
    child.sendline("cat << 'EOF' > /tmp/transfer_email.py")
    child.sendline("import sqlite3")
    child.sendline("conn = sqlite3.connect('/home/morpheme/morpheme/morpheme.db')")
    child.sendline("conn.row_factory = sqlite3.Row")
    child.sendline("cursor = conn.cursor()")
    child.sendline("print('--- BEFORE UPDATE ---')")
    child.sendline("rows = cursor.execute('SELECT id, username, email FROM users WHERE email LIKE \"%%jeffbabiak@outlook.com%%\" OR username IN (\"jeffb\", \"jeffbabiak\", \"username12\")').fetchall()")
    child.sendline("for r in rows: print(dict(r))")
    child.sendline("# Clear email from any other account that has jeffbabiak@outlook.com")
    child.sendline("cursor.execute('UPDATE users SET email = \"\" WHERE LOWER(email) = \"jeffbabiak@outlook.com\" AND LOWER(username) != \"jeffb\"')")
    child.sendline("# Set email of jeffb to jeffbabiak@outlook.com")
    child.sendline("cursor.execute('UPDATE users SET email = \"jeffbabiak@outlook.com\" WHERE LOWER(username) = \"jeffb\"')")
    child.sendline("conn.commit()")
    child.sendline("print('--- AFTER UPDATE ---')")
    child.sendline("rows = cursor.execute('SELECT id, username, email FROM users WHERE email LIKE \"%%jeffbabiak@outlook.com%%\" OR username IN (\"jeffb\", \"jeffbabiak\", \"username12\")').fetchall()")
    child.sendline("for r in rows: print(dict(r))")
    child.sendline("conn.close()")
    child.sendline("EOF")
    child.expect([r"\$", r"#"])
    
    # Run the script
    child.sendline("python3 /tmp/transfer_email.py")
    child.expect([r"\$", r"#"])
    print("\n--- Execution Finished ---")
    
    child.sendline("rm /tmp/transfer_email.py")
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
