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
    child.sendline("cat << 'EOF' > /tmp/check_db.py")
    child.sendline("import sqlite3")
    child.sendline("conn = sqlite3.connect('/home/morpheme/morpheme/morpheme.db')")
    child.sendline("cursor = conn.cursor()")
    child.sendline("cursor.execute('SELECT username, rating FROM users WHERE username = \"jeffjeff\"')")
    child.sendline("print('USERS TABLE RATING:', cursor.fetchall())")
    child.sendline("cursor.execute('SELECT config_key, rating FROM user_ratings WHERE user_id = (SELECT id FROM users WHERE username = \"jeffjeff\")')")
    child.sendline("print('USER_RATINGS TABLE:', cursor.fetchall())")
    child.sendline("conn.close()")
    child.sendline("EOF")
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
