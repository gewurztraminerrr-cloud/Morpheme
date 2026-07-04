import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=20)
    child.logfile_read = sys.stdout
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect([r"\$", r"#"])
    
    child.sendline("cd /home/morpheme/morpheme")
    child.expect([r"\$", r"#"])
    
    child.sendline("python3 -c \"import sqlite3; conn=sqlite3.connect('morpheme.db'); print('AI bots in users:', conn.execute(\\\"SELECT COUNT(*) FROM users WHERE username LIKE 'AI_%'\\\").fetchone()[0]); print('All AI usernames:', [r[0] for r in conn.execute(\\\"SELECT username FROM users WHERE username LIKE 'AI_%'\\\").fetchall()])\"")
    child.expect([r"\$", r"#"])
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
