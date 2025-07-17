source venv/bin/activate

# Start the server in background
nohup python inference_server.py > logs/inference_server.log 2>&1 &

# Save the PID
echo $! > inference_server.pid
echo "Inference server started with PID $(cat inference_server.pid)"
