#!/bin/bash

# Load environment variables from .env
export $(grep -v '^#' .env | xargs)

# Activate virtual environment
# source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the server using values from .env
nohup uvicorn socket_server:asgi_app \
  --host 0.0.0.0 \
  --port ${SOCKET_PORT:-5001} \
  --workers ${SOCKET_WORKERS:-4} > logs/socket_server.log 2>&1 &

# Save the PID
echo $! > socket_server.pid
echo "Inference server started with PID $(cat socket_server.pid) on port ${PORT:-5001} with ${WORKERS:-4} workers"
