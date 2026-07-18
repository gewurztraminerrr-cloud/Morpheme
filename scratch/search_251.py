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
    
    # Search for Background pre-gen complete with count 251 in the log
    sqlite_query = "grep -i -a \"Background pre-gen complete.*251\" /home/morpheme/.pm2/logs/morpheme-out.log"
    child.sendline(sqlite_query)
    child.expect([r"\$", r"#"])
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
