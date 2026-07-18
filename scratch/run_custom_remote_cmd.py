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
    
    print("\n--- Grepping PM2 out log for RESCUE ---")
    child.sendline("grep -i -C 2 rescue ~/.pm2/logs/morpheme-out.log | tail -n 100")
    child.expect([r"\$", r"#"], timeout=15)
    
    child.sendline("exit")
    child.close()
    print("\nRemote command completed successfully!")

if __name__ == "__main__":
    main()
