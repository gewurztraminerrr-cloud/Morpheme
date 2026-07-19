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
    run_local_cmd("git add templates/index.html static/js/settings.js static/css/style.css static/css/play.css static/js/app.js static/js/mods.js static/js/play.js static/js/lobby.js game_room.py app.py spinner_set.py board_generator.py tournament_logic.py static/js/tools.js static/js/private_matches.js word_validator.py dictionaries/static_fallbacks.json scratch/deploy_remote.py scratch/query_remote_db.py scratch/view_pm2_logs.py")
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
    
    # Clear remote DB active_boards to reset round states on deploy
    print("\n--- Clearing remote database active_boards ---")
    child.sendline('python3 -c "import sqlite3; conn = sqlite3.connect(\'morpheme.db\'); c = conn.cursor(); c.execute(\'DELETE FROM active_boards;\'); conn.commit(); conn.close(); print(\'Remote active boards cleared successfully\')"')
    child.expect([r"\$", r"#"])
    
    # Restart PM2 process
    print("\n--- Restarting PM2 process ---")
    child.sendline("pm2 restart morpheme || pm2 start app.py --name morpheme --interpreter venv/bin/python3")
    child.expect([r"\$", r"#"])
    child.sendline("pm2 save")
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
