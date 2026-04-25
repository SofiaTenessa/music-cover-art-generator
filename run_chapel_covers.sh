#!/bin/bash

# Chapel Covers - Quick Start Script
# Run the Flask server for the beautiful custom UI

echo "🏛️  Chapel Covers"
echo "=================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Check if CNN checkpoint exists
if [ ! -f "models/cnn_default_best.pt" ]; then
    echo ""
    echo "⚠️  ERROR: CNN checkpoint not found at models/cnn_default_best.pt"
    echo ""
    echo "Please train the CNN first:"
    echo "  python -m src.train --data-dir data/genres_original --epochs 50 --run-name cnn_v1"
    echo ""
    echo "Then copy the checkpoint:"
    echo "  cp models/cnn_v1_best.pt models/cnn_default_best.pt"
    echo ""
    exit 1
fi

# Start Flask server
echo ""
echo "Starting Flask server..."
echo "Opening http://localhost:5000 in your browser..."
echo ""
echo "Press Ctrl+C to stop"
echo ""

python app/flask_server.py
