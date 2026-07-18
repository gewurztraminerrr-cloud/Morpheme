import pexpect
import sys

def run_remote_db_cleanup():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(f"ssh morpheme@{ip}", encoding="utf-8", timeout=20)
    child.logfile_read = sys.stdout
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect([r"\$", r"#"])
    print("\nConnected! Deleting 'AW' dictionary pregenerated boards from database using python3...")
    
    # Run the sqlite3 commands via python3 to clear all caches and used boards
    py_delete = "python3 -c \"import sqlite3; conn = sqlite3.connect('/home/morpheme/morpheme/morpheme.db'); c = conn.cursor(); c.execute('DELETE FROM pregenerated_boards'); c.execute('DELETE FROM used_boards'); conn.commit(); print('Deleted all cached and used boards'); conn.close()\""
    child.sendline(py_delete)
    child.expect([r"\$", r"#"])
    
    child.sendline("exit")
    child.expect(pexpect.EOF)
    print("\nRemote DB cleanup completed successfully!")

if __name__ == "__main__":
    run_remote_db_cleanup()
