import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=20)
    child.logfile = sys.stdout
    
    child.expect([r"Are you sure you want to continue connecting", r"[Pp]assword:"])
    child.sendline(password)
    child.expect([r"\$", r"#", r">"])
    
    # Check pm2 status and tail logs
    child.sendline("pm2 status")
    child.expect([r"\$", r"#", r">"])
    
    # Print the last 100 lines of morpheme log from pm2
    child.sendline("pm2 logs morpheme --lines 100 --nostream")
    child.expect([r"\$", r"#", r">"])
    
    # Also tail /home/morpheme/morpheme/boggle_server.log or similar
    child.sendline("tail -n 100 /home/morpheme/morpheme/boggle_server_console.log")
    child.expect([r"\$", r"#", r">"])
    
    child.sendline("tail -n 100 /home/morpheme/morpheme/morpheme.log")
    child.expect([r"\$", r"#", r">"])
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
