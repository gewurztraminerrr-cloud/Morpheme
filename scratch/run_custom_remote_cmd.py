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
    
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "tail -n 100 ~/.pm2/logs/morpheme-error.log"
    print(f"\n--- Executing remote command: {cmd} ---")
    child.sendline(cmd)
    child.expect([r"\$", r"#"], timeout=15)
    
    child.sendline("exit")
    child.close()
    print("\nRemote command completed successfully!")

if __name__ == "__main__":
    main()
