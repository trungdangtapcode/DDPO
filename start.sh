#!/bin/bash

echo "🚀 Starting Mock Diffusion Model..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $PYTHON_PID $NODE_PID $VITE_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Python API
echo "Starting Python API on port 8000..."
cd python-api
source venv/bin/activate
python main.py &
PYTHON_PID=$!
cd ..

# Wait for Python API to start
sleep 3

# Start Node.js Backend
echo "Starting Node.js Backend on port 3000..."
cd backend
npm start &
NODE_PID=$!
cd ..

# Wait for Backend to start
sleep 2

# Start Frontend
echo "Starting Frontend on port 5173..."
cd frontend
npm run dev &
VITE_PID=$!
cd ..

echo ""
echo "✅ All services started!"
echo ""
echo "  Python API:    http://localhost:8000"
echo "  Node Backend:  http://localhost:3001"
echo "  Frontend:      http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for processes
wait $PYTHON_PID $NODE_PID $VITE_PID
