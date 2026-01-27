#!/bin/bash
# Run Morpheme Application

# Navigate to script directory
cd "$(dirname "$0")"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

echo "Starting Morpheme..."
echo "Access at http://localhost:3000"

# Run the app
python3 app.py
