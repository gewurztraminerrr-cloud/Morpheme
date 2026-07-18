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
    
    # Search for all occurrences of "Range: 500+" and check for the generated count
    child.sendline("grep -a -n -C 5 \"Range: 500+.*MinLen:.*\" /home/morpheme/.pm2/logs/morpheme-out.log | tail -n 40")
    child.expect([r"\$", r"#"])
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
