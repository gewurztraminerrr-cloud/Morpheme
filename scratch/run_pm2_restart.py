import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(f"ssh morpheme@{ip}", encoding="utf-8", timeout=20)
    child.logfile_read = sys.stdout
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect([r"\$", r"#"])
    print("\nLogged in! Restarting PM2 process...")
    child.sendline("pm2 restart 0")
    
    child.expect([r"\$", r"#"])
    print("\nPM2 restarted!")
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
