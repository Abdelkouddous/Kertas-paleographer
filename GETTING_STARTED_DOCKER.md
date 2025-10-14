# 🚀 Getting Started with Docker - Quick Guide

This guide will help you run the KERTAS Paleographer ML application using Docker. No Python installation or dependency management required - everything runs in a container!

## 📋 Prerequisites

Before you begin, make sure you have the following installed:

### 1. Git

You'll need Git to clone the repository:

```bash
# Check if Git is installed
git --version

# If not installed:
# macOS: brew install git
# Windows: Download from https://git-scm.com/download/win
# Linux: sudo apt-get install git
```

### 2. Docker

Make sure Docker is installed and running:

### Check if Docker is Installed

```bash
docker --version
docker-compose --version
```

### Install Docker (if needed)

**macOS:**

```bash
# Option 1: Using Homebrew
brew install --cask docker

# Option 2: Download Docker Desktop manually
# Visit: https://www.docker.com/products/docker-desktop/
# Download and install the .dmg file
```

**Windows:**

```bash
# Download Docker Desktop for Windows
# Visit: https://www.docker.com/products/docker-desktop/
# Download and run the installer

# Note: Requires Windows 10/11 64-bit Pro, Enterprise, or Education
# Or Windows 10/11 Home with WSL 2
```

**Linux (Ubuntu/Debian):**

```bash
# Install using the official Docker script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to the docker group (to avoid using sudo)
sudo usermod -aG docker $USER

# Log out and back in for group changes to take effect
# Or run: newgrp docker
```

**After Installation:**

Restart your computer to ensure Docker is fully initialized, then verify:

```bash
docker --version
docker-compose --version
```

---

## ⚡ Quick Start (4 Steps)

### Step 1: Clone the Repository

First, clone the project from GitHub:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Kertas-paleographer.git

# Navigate into the project directory
cd Kertas-paleographer
```

> **Note:** Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 2: Verify Project Files

Make sure you're in the correct directory with all necessary files:

```bash
# List files to verify
ls -la

# You should see:
# - Dockerfile
# - docker-compose.yml
# - docker-start.sh (or docker-start.bat on Windows)
# - app.py
# - main.py
# - requirements.txt
# - KERTASpaleographer/ (data directory)
```

If you see all these files, you're ready to proceed! ✅

### Step 3: Start the Application

**Option A: Using Quick Start Script (Recommended)**

```bash
# macOS/Linux
./docker-start.sh

# Windows
docker-start.bat
```

**Option B: Using Docker Compose Directly**

```bash
docker-compose up -d
```

### Step 4: Access the Application

Open your browser and go to:

```
http://localhost:8501
```

**That's it! Your ML application is running! 🎉**

---

## 🛠️ Common Commands

### Start the Application

```bash
docker-compose up -d
```

- `-d` runs in detached mode (background)
- Without `-d` you'll see live logs

### Stop the Application

```bash
docker-compose down
```

### View Logs

```bash
# All logs
docker-compose logs

# Follow logs (real-time)
docker-compose logs -f

# Logs for specific service
docker-compose logs -f app
```

### Check Status

```bash
# List running containers
docker-compose ps

# Or
docker ps
```

### Restart Application

```bash
docker-compose restart
```

### Rebuild After Code Changes

```bash
docker-compose up -d --build
```

### Clean Up Everything

```bash
# Stop and remove containers
docker-compose down

# Also remove volumes
docker-compose down -v

# Remove images
docker rmi kertas-paleographer:latest
```

---

## 🎯 What's Happening Behind the Scenes?

When you run `docker-compose up -d`:

1. **Docker reads** `docker-compose.yml`
2. **Builds** the image from `Dockerfile` (if needed)
3. **Creates** a container from the image
4. **Mounts** your data directory
5. **Exposes** port 8501
6. **Starts** the Streamlit application
7. **Health checks** ensure it's running

---

## 📊 Verify Everything is Working

### Check Container Status

```bash
docker ps
```

You should see:

```
CONTAINER ID   IMAGE                      STATUS         PORTS
xxxxx         kertas-paleographer:latest  Up 2 minutes   0.0.0.0:8501->8501/tcp
```

### Check Health

```bash
docker inspect --format='{{.State.Health.Status}}' kertas-paleographer
```

Should return: `healthy`

### Test the Web Interface

```bash
curl http://localhost:8501/_stcore/health
```

Should return: `ok`

---

## 🐛 Troubleshooting

### Problem: Port Already in Use

**Error:**

```
Error starting userland proxy: listen tcp4 0.0.0.0:8501: bind: address already in use
```

**Solution 1:** Stop the conflicting service

```bash
# Find what's using port 8501 (macOS/Linux)
lsof -i :8501

# Windows (in PowerShell)
netstat -ano | findstr :8501

# Kill the process (macOS/Linux)
kill -9 <PID>

# Windows (in PowerShell as Administrator)
taskkill /PID <PID> /F
```

**Solution 2:** Use a different port

```bash
# Edit docker-compose.yml and change:
ports:
  - "8502:8501"  # Use 8502 instead

# Then access at http://localhost:8502
```

### Problem: Permission Denied (Linux)

**Error:**

```
permission denied while trying to connect to the Docker daemon socket
```

**Solution:**

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Problem: Container Exits Immediately

**Check logs:**

```bash
docker-compose logs app
```

**Common causes:**

- Missing data files
- Python dependency issues
- Port conflicts

**Solution:**

```bash
# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Problem: Slow Performance

**Solution 1:** Allocate more resources to Docker

- Docker Desktop → Settings → Resources
- Increase CPUs to 4
- Increase Memory to 6-8GB

**Solution 2:** Check resource usage

```bash
docker stats kertas-paleographer
```

### Problem: Can't Access from Browser

**Check:**

1. Container is running: `docker ps`
2. Port is correct: `http://localhost:8501`
3. Firewall isn't blocking
4. Try `http://127.0.0.1:8501`

---

## 🎓 Understanding the Files

### `Dockerfile`

- Instructions to build the Docker image
- Installs Python, dependencies, and your application
- Multi-stage build for optimization

### `docker-compose.yml`

- Defines services, networks, and volumes
- Makes it easy to run multi-container apps
- Configuration for ports, environment variables

### `.dockerignore`

- Excludes files from Docker build
- Speeds up build process
- Reduces image size

### `docker-start.sh` / `docker-start.bat`

- Convenience scripts for quick start
- Menu-driven interface
- Checks for Docker installation

---

## 📈 Next Steps

### 1. Explore the Application

- Try training different models
- Compare ChainCode vs Polygon features
- Enable Grid Search for optimization

### 2. Check Resource Usage

```bash
# Real-time stats
docker stats kertas-paleographer

# Image size
docker images kertas-paleographer
```

### 3. Learn Docker Commands

```bash
# List images
docker images

# List containers
docker ps -a

# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# View detailed info
docker inspect kertas-paleographer
```

### 4. Customize Configuration

Edit `docker-compose.yml` to:

- Change ports
- Add environment variables
- Adjust resource limits
- Add more services

### 5. Deploy to Cloud

See `DOCKER.md` for:

- AWS ECS deployment
- Google Cloud Run
- Azure Container Instances
- Kubernetes

---

## 🎨 Development Workflow

### Making Code Changes

```bash
# 1. Edit your code (app.py, main.py, etc.)
# Use your favorite text editor or IDE

# 2. Rebuild and restart the container
docker-compose up -d --build

# 3. Watch logs to check for errors
docker-compose logs -f app

# 4. Test changes in browser
# macOS: open http://localhost:8501
# Linux: xdg-open http://localhost:8501
# Windows: start http://localhost:8501
# Or just manually open: http://localhost:8501
```

### Debugging

```bash
# Enter the running container (interactive shell)
docker exec -it kertas-paleographer bash

# Once inside the container:
# - Check Python version
python --version

# - List files
ls -la

# - Check environment variables
env

# - Check if app is running
ps aux | grep streamlit

# Exit container (type or press Ctrl+D)
exit
```

### Running Tests

```bash
# Run tests in container
docker-compose exec app pytest test_app.py
```

---

## 💡 Tips & Best Practices

### Do's ✅

- ✅ Always use `docker-compose down` before making changes
- ✅ Check logs when something doesn't work
- ✅ Use `-d` flag for production, omit for debugging
- ✅ Regularly clean up with `docker system prune`
- ✅ Keep data files outside container (use volumes)

### Don'ts ❌

- ❌ Don't use `sudo` with Docker (on Linux, add to docker group)
- ❌ Don't edit files inside running container (changes won't persist)
- ❌ Don't commit `.env` files to Git
- ❌ Don't ignore health check failures
- ❌ Don't run production without resource limits

---

## 📚 Additional Resources

### Documentation

- [DOCKER.md](DOCKER.md) - Complete Docker deployment guide
- [DOCKER_SETUP_COMPLETE.md](DOCKER_SETUP_COMPLETE.md) - Setup summary and next steps
- [README.md](README.md) - Main project documentation

### Learning Docker

- [Docker Official Docs](https://docs.docker.com/)
- [Docker Compose Docs](https://docs.docker.com/compose/)
- [Play with Docker](https://labs.play-with-docker.com/)

### Community

- [Docker Forum](https://forums.docker.com/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/docker)
- [Reddit r/docker](https://reddit.com/r/docker)

---

## 🎯 Quick Reference Card

### Essential Commands

| Command                                    | What it does        |
| ------------------------------------------ | ------------------- |
| `docker-compose up -d`                     | Start application   |
| `docker-compose down`                      | Stop application    |
| `docker-compose logs -f`                   | View live logs      |
| `docker-compose ps`                        | Check status        |
| `docker-compose restart`                   | Restart application |
| `docker-compose up -d --build`             | Rebuild & restart   |
| `docker exec -it kertas-paleographer bash` | Enter container     |

### Shortcuts (macOS/Linux)

Create aliases in `~/.bashrc` or `~/.zshrc`:

```bash
alias dup='docker-compose up -d'
alias ddown='docker-compose down'
alias dlogs='docker-compose logs -f'
alias dps='docker-compose ps'
alias drestart='docker-compose restart'
alias drebuild='docker-compose up -d --build'
```

Then use:

```bash
dup      # Start
dlogs    # View logs
ddown    # Stop
```

---

## 🏆 Success Checklist

After following this guide, you should be able to:

- [ ] Check if Docker is installed on your system
- [ ] Clone the repository from GitHub
- [ ] Navigate to the project directory
- [ ] Start the application with one command
- [ ] Access the web interface at http://localhost:8501
- [ ] View and follow container logs
- [ ] Stop and restart the application
- [ ] Rebuild containers after code changes
- [ ] Troubleshoot common Docker issues
- [ ] Enter the container for debugging
- [ ] Check resource usage and performance
- [ ] Clean up containers and images when done

---

## 🎉 Congratulations!

You now know how to:

- Run ML applications in Docker
- Manage containers with Docker Compose
- Debug containerized applications
- Follow MLDevOps best practices

**Next:** Check out [DOCKER.md](DOCKER.md) for advanced Docker deployment options including cloud deployment!

---

**Need Help?**

1. Check [DOCKER.md](DOCKER.md) for detailed documentation
2. Review [Troubleshooting](#-troubleshooting) section above
3. Check Docker logs: `docker-compose logs -f`
4. Open an issue on [GitHub Issues](https://github.com/YOUR_USERNAME/Kertas-paleographer/issues)
5. Ask in Docker community forums

> **Note:** Replace `YOUR_USERNAME` with your actual GitHub username in the issues link.

---

_Happy Dockerizing! 🐳_
