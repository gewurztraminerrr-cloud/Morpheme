import pexpect
import sys
import os

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dictionaries', 'added_words.txt')
    
    cmd = f"scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no morpheme@{ip}:/home/morpheme/morpheme/dictionaries/added_words.txt {local_path}"
    
    print(f"Downloading remote added_words.txt to {local_path}...")
    child = pexpect.spawn(cmd, encoding='utf-8', timeout=60)
    child.logfile_read = sys.stdout
    
    try:
        idx = child.expect([r"password:", pexpect.EOF, pexpect.TIMEOUT])
        if idx == 0:
            child.sendline(password)
            child.expect(pexpect.EOF)
            print("\nDownload complete!")
        else:
            print("Failed to reach password prompt or transfer finished early.")
    except Exception as e:
        print(f"\nException during copy: {e}")
    finally:
        child.close()

if __name__ == "__main__":
    main()
