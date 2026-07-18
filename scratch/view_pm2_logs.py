import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    cmd = f'ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no morpheme@{ip}'
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(cmd, encoding='utf-8', timeout=20)
    child.logfile_read = sys.stdout
    
    try:
        idx = child.expect([r"password:", pexpect.EOF, pexpect.TIMEOUT])
        if idx == 0:
            child.sendline(password)
            child.expect([r"\$", r"#", pexpect.EOF, pexpect.TIMEOUT], timeout=10)
            
            print("\n--- Reading tail of morpheme-out.log ---")
            child.sendline("tail -n 30 ~/.pm2/logs/morpheme-out.log")
            child.expect([r"\$", r"#", pexpect.EOF, pexpect.TIMEOUT], timeout=10)

            print("\n--- Reading tail of morpheme-error.log ---")
            child.sendline("tail -n 30 ~/.pm2/logs/morpheme-error.log")
            child.expect([r"\$", r"#", pexpect.EOF, pexpect.TIMEOUT], timeout=10)

            child.sendline("exit")
            child.expect([pexpect.EOF])
            print("\nDone!")
        else:
            print("Failed to reach password prompt.")
    except Exception as e:
        print(f"\nSSH Exception: {e}")
    finally:
        child.close()

if __name__ == "__main__":
    main()
