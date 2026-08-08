import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print("Connecting to remote server...")
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=30)
    idx = child.expect([r"Are you sure you want to continue connecting", r"[Pp]assword:"])
    if idx == 0:
        child.sendline("yes")
        child.expect([r"[Pp]assword:"])
    child.sendline(password)
    child.expect([r"\$", r"#", r">"])
    
    child.sendline("pm2 status")
    child.expect([r"\$", r"#", r">"])
    print("\n--- PM2 Status ---")
    print(child.before)
    
    child.sendline("pm2 logs morpheme --lines 100 --nostream")
    child.expect([r"\$", r"#", r">"])
    print("\n--- PM2 Logs ---")
    print(child.before)

    child.sendline("pm2 restart morpheme")
    child.expect([r"\$", r"#", r">"])
    print("\n--- PM2 Restarted ---")
    print(child.before)
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
