import pexpect
import sys
import subprocess

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    files_to_transfer = [
        "game_room.py",
        "app.py",
        "spinner_set.py",
        "word_validator.py",
        "private_match_logic.py",
        "tournament_logic.py",
        "board_generator.py",
        "static/js/play.js",
        "static/js/lobby.js",
        "static/js/tools.js",
        "static/js/app.js",
        "static/css/lobby.css",
        "static/css/style.css",
        "static/css/forum.css",
        "templates/index.html"
    ]
    
    print("Uploading updated files to production server...")
    for f in files_to_transfer:
        print(f"Uploading {f}...")
        cmd = f"scp {f} morpheme@{ip}:/home/morpheme/morpheme/{f}"
        child = pexpect.spawn(cmd, encoding='utf-8', timeout=30)
        idx = child.expect([r"Are you sure you want to continue connecting", r"[Pp]assword:"])
        if idx == 0:
            child.sendline("yes")
            child.expect([r"[Pp]assword:"])
        child.sendline(password)
        child.expect(pexpect.EOF)
        print(f"Uploaded {f} successfully.")
        
    print("\nRestarting PM2 application on remote server...")
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=30)
    child.expect([r"Are you sure you want to continue connecting", r"[Pp]assword:"])
    child.sendline(password)
    child.expect([r"\$", r"#", r">"])
    
    child.sendline("pm2 restart morpheme")
    child.expect([r"\$", r"#", r">"])
    print("\n--- PM2 Restart Output ---")
    print(child.before)
    
    child.sendline("pm2 status")
    child.expect([r"\$", r"#", r">"])
    print("\n--- PM2 Status Output ---")
    print(child.before)
    
    child.sendline("exit")
    child.close()
    print("\nDeployment complete!")

if __name__ == "__main__":
    main()
