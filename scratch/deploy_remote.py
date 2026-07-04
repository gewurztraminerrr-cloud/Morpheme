import subprocess
import pexpect
import sys

def run_local_cmd(cmd):
    print(f"Running local command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print(result.stdout)

def main():
    # 1. Commit and push local changes to GitHub
    run_local_cmd("git add templates/index.html static/js/settings.js static/css/style.css static/css/play.css static/js/app.js static/js/mods.js static/js/play.js static/js/lobby.js game_room.py app.py spinner_set.py board_generator.py tournament_logic.py")
    # Check if there are changes to commit
    status = subprocess.run("git diff --cached --quiet", shell=True)
    if status.returncode == 1: # 1 means there are cached changes
        run_local_cmd('git commit -m "Deploy updates"')
        run_local_cmd("git push origin main")
    else:
        print("No local changes to commit. Proceeding with remote deployment...")

    # 2. Connect to remote server and pull/restart
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"\nConnecting to {ip} via SSH for deployment...")
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=30)
    child.logfile_read = sys.stdout
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect([r"\$", r"#"])
    print("\nLogged in successfully!")
    
    # Navigate to the repo
    print("\n--- Navigating to repository ---")
    child.sendline("cd /home/morpheme/morpheme")
    child.expect([r"\$", r"#"])
    
    # Pull changes
    print("\n--- Pulling latest changes from GitHub ---")
    child.sendline("git fetch && git reset --hard origin/main")
    child.expect([r"\$", r"#"])
    
    # Restart PM2 process
    print("\n--- Restarting PM2 process ---")
    child.sendline("pm2 restart 0")
    child.expect([r"\$", r"#"])
    
    # Show PM2 status
    print("\n--- PM2 Status ---")
    child.sendline("pm2 status")
    child.expect([r"\$", r"#"])
    
    child.sendline("exit")
    child.close()
    print("\nDeployment completed successfully!")

if __name__ == "__main__":
    main()
