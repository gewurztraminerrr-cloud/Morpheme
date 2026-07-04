import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    cmd = f'ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no morpheme@{ip}'
    
    print(f"Connecting to {ip} via SSH to run server manually...")
    child = pexpect.spawn(cmd, encoding='utf-8', timeout=40)
    child.logfile_read = sys.stdout
    
    try:
        idx = child.expect([r"password:", pexpect.EOF, pexpect.TIMEOUT])
        if idx == 0:
            child.sendline(password)
            child.expect([r"\$", r"#", pexpect.EOF, pexpect.TIMEOUT], timeout=10)
            
            print("\n--- Starting Morpheme Server Manually ---")
            child.sendline("cd /home/morpheme/morpheme && venv/bin/python3 app.py")
            
            # Let it run for 15 seconds to see what it prints (startup logs / errors)
            print("\nWaiting for 15 seconds of server logs...")
            child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=15)
            
            print("\nInterrupting server (Ctrl+C)...")
            child.sendintr() # Send Ctrl+C
            child.expect([r"\$", r"#", pexpect.EOF, pexpect.TIMEOUT], timeout=5)
            
            child.sendline("exit")
            child.expect([pexpect.EOF])
            print("\nManual run complete!")
        else:
            print("Failed to reach password prompt.")
    except Exception as e:
        print(f"\nSSH Exception: {e}")
    finally:
        child.close()

if __name__ == "__main__":
    main()
