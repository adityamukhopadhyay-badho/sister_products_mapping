#!/bin/bash

echo "🚀 Starting Sister Products Mapping API"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p output logs checkpoints visualizations

# Start the API
echo "Starting FastAPI server..."
echo "Web interface: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo "Health check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
