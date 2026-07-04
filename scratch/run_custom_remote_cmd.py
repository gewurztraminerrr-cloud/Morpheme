import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=20)
    child.logfile_read = sys.stdout
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect([r"\$", r"#"])
    print("\nLogged in successfully!")
    
    print("\n--- Running git pull and pm2 restart all ---")
    child.sendline("cd /home/morpheme/morpheme && git pull origin main && pm2 restart all")
    child.expect([r"\$", r"#"])
    
    child.sendline("exit")
    child.close()
    print("\nRemote command completed successfully!")

if __name__ == "__main__":
    main()
