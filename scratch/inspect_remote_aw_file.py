import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(f"ssh morpheme@{ip}", encoding="utf-8", timeout=20)
    child.logfile_read = sys.stdout
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect([r"\$", r"#"])
    print("\nLogged in! Checking remote file lines count...")
    child.sendline("wc -l /home/morpheme/morpheme/dictionaries/added_words.txt")
    
    child.expect([r"\$", r"#"])
    child.sendline("head -n 20 /home/morpheme/morpheme/dictionaries/added_words.txt")
    
    child.expect([r"\$", r"#"])
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
