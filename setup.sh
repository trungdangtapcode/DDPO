#!/bin/bash

echo "🚀 Setting up Mock Diffusion Model Project..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed. Please install Node.js 18 or higher.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed!${NC}"
echo ""

# Setup Python API
echo -e "${BLUE}📦 Setting up Python API...${NC}"
cd python-api
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..
echo -e "${GREEN}✅ Python API setup complete!${NC}"
echo ""

# Setup Node.js Backend
echo -e "${BLUE}📦 Setting up Node.js Backend...${NC}"
cd backend
npm install
cd ..
echo -e "${GREEN}✅ Node.js Backend setup complete!${NC}"
echo ""

# Setup Frontend
echo -e "${BLUE}📦 Setting up Frontend...${NC}"
cd frontend
npm install
cd ..
echo -e "${GREEN}✅ Frontend setup complete!${NC}"
echo ""

echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo "To start the application:"
echo ""
echo "1. Start Python API:"
echo "   cd python-api && source venv/bin/activate && python main.py"
echo ""
echo "2. Start Node.js Backend (in new terminal):"
echo "   cd backend && npm start"
echo ""
echo "3. Start Frontend (in new terminal):"
echo "   cd frontend && npm run dev"
echo ""
echo "4. Open browser:"
echo "   http://localhost:5173"
echo ""
echo -e "${BLUE}Happy coding! 🚀${NC}"
