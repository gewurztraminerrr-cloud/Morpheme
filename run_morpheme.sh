#!/bin/bash
# run_morpheme.sh - Robust Multi-Server Launcher for Morpheme

# Navigate to script directory
cd "$(dirname "$0")"

LOCKFILE="run_morpheme.lock"
if [ -f "$LOCKFILE" ]; then
    LAST_PID=$(cat "$LOCKFILE")
    if kill -0 "$LAST_PID" 2>/dev/null; then
        echo "Morpheme launcher is already running with PID $LAST_PID. Exiting."
        exit 1
    fi
fi
echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

# Configure Logging
MAIN_LOG="server.log"
BOGGLE_LOG="boggle_server_console.log"
echo "--- Morpheme Startup: $(date) ---" | tee -a "$MAIN_LOG" "$BOGGLE_LOG"

# Helper to kill processes on a port
kill_port() {
    local port=$1
    local pids=$(lsof -t -i :$port)
    if [ ! -z "$pids" ]; then
        echo "Cleaning up port $port (PIDs: $pids)..."
        kill -9 $pids 2>/dev/null || true
    fi
}

# Cleanup existing servers to prevent "Address already in use"
kill_port 3000
kill_port 5005

# Start Boggle-Gen History Service (Port 5005) in background
echo "Starting Boggle-Gen service on http://localhost:5005..."
nohup venv/bin/python3 boggle-gen/web/app.py >> "$BOGGLE_LOG" 2>&1 &
BOGGLE_PID=$!

# Start Main Morpheme Server (Port 5001) with Auto-Restart Loop
echo "Starting Morpheme main server on http://localhost:3000..."
echo "Auto-restart protection active."

while true; do
    # Ensure Boggle-Gen service is running
    if ! pgrep -f "boggle-gen/web/app.py" > /dev/null; then
        echo "Boggle-Gen service is offline. Restarting..." | tee -a "$BOGGLE_LOG"
        nohup venv/bin/python3 boggle-gen/web/app.py >> "$BOGGLE_LOG" 2>&1 &
    fi

    # Kill any existing processes on port 5001
    echo "Cleaning up existing Morpheme processes..."
    PIDS=$(lsof -t -i :3000)
    if [ ! -z "$PIDS" ]; then
        kill -9 $PIDS
        sleep 3
    fi

    # Run the server and CAPTURE its completion/crash
    echo "Starting Morpheme Server (app.py)..."
    venv/bin/python3 app.py 2>&1 | tee server.log
    
    echo "Main server crashed or stopped. Checking for termios issues..."
    if grep -q "termios.error" "$MAIN_LOG"; then
        echo "Detected termios.error crash. Restarting in 2s..."
    fi
    
    sleep 2
    echo "Restarting main loop..."
done
