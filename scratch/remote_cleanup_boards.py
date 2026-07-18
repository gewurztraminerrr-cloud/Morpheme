import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"Connecting to {ip} via SSH to clean up legacy pregenerated boards on remote...")
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=60)
    child.logfile_read = sys.stdout
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect([r"\$", r"#"])
    print("\nLogged in successfully!")
    
    print("\n--- Running cleanup_pregenerated_boards.py on remote ---")
    child.sendline("cd /home/morpheme/morpheme && python3 scratch/cleanup_pregenerated_boards.py")
    child.expect([r"\$", r"#"])
    
    child.sendline("exit")
    child.close()
    print("\nRemote board cleanup completed successfully!")

if __name__ == "__main__":
    main()
