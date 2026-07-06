import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    cmd = f'ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no morpheme@{ip}'
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(cmd, encoding='utf-8', timeout=20)
    
    child.expect([r"password:"])
    child.sendline(password)
    child.expect([r"\$", r"#"])
    
    # Check added_words.txt lines
    child.sendline("wc -l /home/morpheme/morpheme/dictionaries/added_words.txt")
    child.expect([r"\$", r"#"])
    print("\n--- Remote added_words.txt info ---")
    print(child.before)
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
