import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=20)
    # Redirect stdout so we see the output
    child.logfile_read = sys.stdout
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect([r"\$", r"#"])
    
    # Run the SQLite commands on production database
    child.sendline("cd /home/morpheme/morpheme")
    child.expect([r"\$", r"#"])
    
    print("\n--- Tournaments on Production ---")
    child.sendline("python3 -c \"import sqlite3; conn=sqlite3.connect('morpheme.db'); cursor=conn.execute('SELECT * FROM tournaments ORDER BY id DESC LIMIT 5'); [print(row) for row in cursor.fetchall()]\"")
    child.expect([r"\$", r"#"])
    
    print("\n--- Current Tournament Participants ---")
    child.sendline("python3 -c \"import sqlite3; conn=sqlite3.connect('morpheme.db'); cursor=conn.execute('SELECT * FROM tournament_participants WHERE tournament_id = (SELECT id FROM tournaments ORDER BY id DESC LIMIT 1)'); [print(row) for row in cursor.fetchall()]\"")
    child.expect([r"\$", r"#"])
    
    print("\n--- Recent Tournament Participants ---")
    child.sendline("python3 -c \"import sqlite3; conn=sqlite3.connect('morpheme.db'); cursor=conn.execute('SELECT * FROM tournament_participants ORDER BY joined_at DESC LIMIT 10'); [print(row) for row in cursor.fetchall()]\"")
    child.expect([r"\$", r"#"])
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
