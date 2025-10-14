#!/bin/bash
# ============================================================================
# KERTAS Paleographer - Docker Quick Start Script
# ============================================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🐳 KERTAS Paleographer - Docker Deployment              ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed!${NC}"
    echo -e "${YELLOW}Please install Docker from: https://www.docker.com/products/docker-desktop${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed!${NC}"
    echo -e "${YELLOW}Please install Docker Compose${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is installed${NC}"
echo -e "${GREEN}✓ Docker Compose is installed${NC}"
echo ""

# Menu
echo -e "${BLUE}Select deployment option:${NC}"
echo ""
echo "  1) 🚀 Start Web Application (Recommended)"
echo "  2) 💻 Start CLI Interface"
echo "  3) 🔨 Build Docker Image Only"
echo "  4) 🛑 Stop All Services"
echo "  5) 📊 View Logs"
echo "  6) 🔄 Rebuild and Restart"
echo "  7) 🗑️  Clean Up (Remove containers & images)"
echo ""

read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo -e "${BLUE}Starting Web Application...${NC}"
        docker-compose up -d app
        echo ""
        echo -e "${GREEN}✅ Application started successfully!${NC}"
        echo -e "${YELLOW}Access the web interface at: ${NC}${GREEN}http://localhost:8501${NC}"
        echo ""
        echo -e "${BLUE}View logs with: ${NC}docker-compose logs -f app"
        ;;
    2)
        echo -e "${BLUE}Starting CLI Interface...${NC}"
        docker-compose --profile cli run --rm cli
        ;;
    3)
        echo -e "${BLUE}Building Docker image...${NC}"
        docker build -t kertas-paleographer:latest .
        echo -e "${GREEN}✅ Build complete!${NC}"
        ;;
    4)
        echo -e "${BLUE}Stopping all services...${NC}"
        docker-compose down
        echo -e "${GREEN}✅ All services stopped${NC}"
        ;;
    5)
        echo -e "${BLUE}Showing logs (Ctrl+C to exit)...${NC}"
        docker-compose logs -f
        ;;
    6)
        echo -e "${BLUE}Rebuilding and restarting...${NC}"
        docker-compose down
        docker-compose up -d --build
        echo -e "${GREEN}✅ Application rebuilt and restarted!${NC}"
        echo -e "${YELLOW}Access at: ${NC}${GREEN}http://localhost:8501${NC}"
        ;;
    7)
        echo -e "${YELLOW}⚠️  This will remove all containers and images${NC}"
        read -p "Are you sure? (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            echo -e "${BLUE}Cleaning up...${NC}"
            docker-compose down -v
            docker rmi kertas-paleographer:latest 2>/dev/null || true
            echo -e "${GREEN}✅ Cleanup complete${NC}"
        else
            echo -e "${YELLOW}Cleanup cancelled${NC}"
        fi
        ;;
    *)
        echo -e "${RED}Invalid choice!${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}For more information, see: ${NC}${GREEN}DOCKER.md${NC}"

