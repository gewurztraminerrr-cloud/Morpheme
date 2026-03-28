#!/bin/bash
# run_morpheme.sh - Robust Multi-Server Launcher for Morpheme

# Navigate to script directory
cd "$(dirname "$0")"

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
nohup python3 boggle-gen/web/app.py >> "$BOGGLE_LOG" 2>&1 &
BOGGLE_PID=$!

# Start Main Morpheme Server (Port 3000) with Auto-Restart Loop
echo "Starting Morpheme main server on http://localhost:3000..."
echo "Auto-restart protection active."

while true; do
    # Ensure Boggle-Gen service is running
    if ! pgrep -f "boggle-gen/web/app.py" > /dev/null; then
        echo "Boggle-Gen service is offline. Restarting..." | tee -a "$BOGGLE_LOG"
        nohup python3 boggle-gen/web/app.py >> "$BOGGLE_LOG" 2>&1 &
    fi

    # Run main Morpheme server in foreground of this loop
    python3 app.py >> "$MAIN_LOG" 2>&1
    
    echo "Main server crashed or stopped. Checking for termios issues..."
    if grep -q "termios.error" "$MAIN_LOG"; then
        echo "Detected termios.error crash. Restarting in 2s..."
    fi
    
    sleep 2
    echo "Restarting main loop..."
done
