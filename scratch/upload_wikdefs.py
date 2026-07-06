import pexpect
import sys
import os

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dictionaries', 'wikdefs.txt')
    remote_path = f"morpheme@{ip}:/home/morpheme/morpheme/dictionaries/wikdefs.txt"
    
    print(f"Uploading local {local_path} to remote {remote_path} via scp...")
    cmd = f'scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no "{local_path}" "{remote_path}"'
    
    child = pexpect.spawn(cmd, encoding='utf-8', timeout=300)
    child.logfile_read = sys.stdout
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect(pexpect.EOF)
    child.close()
    print("\nUpload completed successfully!")

if __name__ == "__main__":
    main()
