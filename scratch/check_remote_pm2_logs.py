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
    
    child.sendline("pm2 logs morpheme --lines 50 --nostream")
    child.expect([r"\$", r"#", r">"])
    print("\n--- PM2 Logs Output ---")
    print(child.before)
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
