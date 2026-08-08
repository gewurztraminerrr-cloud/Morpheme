import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=20)
    child.expect([r'[Pp]assword:'])
    child.sendline(password)
    child.expect(['morpheme@249:'])
    
    child.sendline("python3 -c \"import sqlite3; conn = sqlite3.connect('morpheme.db'); print(conn.execute('SELECT name FROM sqlite_master WHERE type=\\\"table\\\"').fetchall())\"")
    child.expect(['morpheme@249:'])
    print(child.before)
    child.sendline('exit')
    child.close()

if __name__ == '__main__':
    main()
