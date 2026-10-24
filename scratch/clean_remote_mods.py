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
    child.sendline("cat << 'EOF' > /tmp/clean_mods.py")
    child.sendline("import sqlite3")
    child.sendline("conn = sqlite3.connect('/home/morpheme/morpheme/morpheme.db')")
    child.sendline("conn.execute('CREATE TABLE IF NOT EXISTS moderators (username TEXT PRIMARY KEY, added_at REAL)')")
    child.sendline("conn.execute('DELETE FROM moderators WHERE LOWER(username) = \"jeffbabiak\"')")
    child.sendline("conn.execute('INSERT OR REPLACE INTO moderators (username, added_at) VALUES (\"jeffb\", 1700000000)')")
    child.sendline("conn.commit()")
    child.sendline("print('Remote DB Moderators:', conn.execute('SELECT * FROM moderators').fetchall())")
    child.sendline("conn.close()")
    child.sendline("EOF")
    child.expect([r"\$", r"#"])
    
    # Run the script
    child.sendline("python3 /tmp/clean_mods.py")
    child.expect([r"\$", r"#"])
    print("\n--- Clean Mods Finished ---")
    
    child.sendline("rm /tmp/clean_mods.py")
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
