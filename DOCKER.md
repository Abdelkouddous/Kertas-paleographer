# 🐳 Docker Deployment Guide - KERTAS Paleographer

## 📋 Table of Contents

1. [Introduction to Docker](#introduction-to-docker)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Building the Image](#building-the-image)
5. [Running the Container](#running-the-container)
6. [Docker Compose](#docker-compose)
7. [Production Deployment](#production-deployment)
8. [Cloud Deployment](#cloud-deployment)
9. [MLDevOps Best Practices](#mldevops-best-practices)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Introduction to Docker

### What is Docker?

Docker is a **containerization platform** that packages your ML application with all its dependencies into a standardized, portable unit called a **container**.

### Why Docker for KERTAS Paleographer?

✅ **Reproducibility** - Run the same environment everywhere  
✅ **Dependency Management** - No installation conflicts  
✅ **Scalability** - Easy horizontal scaling  
✅ **Portability** - Deploy to any cloud provider  
✅ **Isolation** - Multiple projects without interference  
✅ **CI/CD Ready** - Automated deployment pipelines

### MLDevOps Benefits

This Docker setup demonstrates:

- **Infrastructure as Code** (IaC)
- **Container Orchestration** with Docker Compose
- **Multi-Stage Builds** for optimization
- **Security Best Practices** (non-root user)
- **Health Checks** for reliability
- **Cloud-Ready Architecture**

---

## 🔧 Prerequisites

### Required Software

1. **Docker Engine** (20.10+)

   - [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Includes Docker Compose

2. **Verify Installation:**

```bash
docker --version
# Docker version 24.0.0 or higher

docker-compose --version
# Docker Compose version 2.20.0 or higher
```

### System Requirements

- **RAM**: Minimum 4GB, Recommended 8GB+
- **Disk Space**: ~2GB for image + data
- **OS**: Linux, macOS, Windows 10/11 Pro

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone or navigate to project directory
cd Kertas-paleographer

# Start the application
docker-compose up -d

# Access the web interface
open http://localhost:8501
```

That's it! Your ML application is running in Docker! 🎉

### Option 2: Docker CLI

```bash
# Build the image
docker build -t kertas-paleographer .

# Run the container
docker run -d -p 8501:8501 --name kertas kertas-paleographer

# Access the web interface
open http://localhost:8501
```

---

## 🏗️ Building the Image

### Basic Build

```bash
docker build -t kertas-paleographer:latest .
```

### Build with Custom Tag

```bash
docker build -t kertas-paleographer:v1.0 .
```

### Build Arguments (Advanced)

```bash
docker build \
  --build-arg PYTHON_VERSION=3.9 \
  -t kertas-paleographer:custom .
```

### View Built Images

```bash
docker images | grep kertas
```

---

## 🏃 Running the Container

### Web UI Mode (Default)

```bash
# Run detached (background)
docker run -d \
  -p 8501:8501 \
  --name kertas-web \
  kertas-paleographer

# Run interactive (see logs)
docker run -it \
  -p 8501:8501 \
  --name kertas-web \
  kertas-paleographer
```

### CLI Mode

```bash
docker run -it \
  --name kertas-cli \
  kertas-paleographer \
  python main.py
```

### With Volume Mounting (Development)

```bash
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/KERTASpaleographer:/app/KERTASpaleographer:ro \
  --name kertas-dev \
  kertas-paleographer
```

### Container Management

```bash
# View running containers
docker ps

# View all containers
docker ps -a

# View logs
docker logs kertas-web
docker logs -f kertas-web  # Follow logs

# Stop container
docker stop kertas-web

# Start container
docker start kertas-web

# Restart container
docker restart kertas-web

# Remove container
docker rm kertas-web

# Remove container (force)
docker rm -f kertas-web
```

---

## 🎼 Docker Compose

### Configuration Overview

Our `docker-compose.yml` defines:

- **app** - Streamlit web interface
- **cli** - Command-line interface (optional)
- **networks** - Isolated networking
- **volumes** - Data persistence

### Basic Commands

```bash
# Start all services
docker-compose up -d

# Start with CLI
docker-compose --profile cli up cli

# View logs
docker-compose logs -f app

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild and restart
docker-compose up -d --build

# View service status
docker-compose ps

# Execute command in running container
docker-compose exec app bash
```

### Environment Configuration

Create `.env` file for custom settings:

```bash
# .env
STREAMLIT_PORT=8501
PYTHON_VERSION=3.9
APP_ENV=production
```

Update `docker-compose.yml`:

```yaml
services:
  app:
    environment:
      - APP_ENV=${APP_ENV}
    ports:
      - "${STREAMLIT_PORT}:8501"
```

---

## 🌐 Production Deployment

### Production Best Practices

#### 1. Multi-Stage Build

Our Dockerfile uses multi-stage builds to minimize image size:

```dockerfile
FROM python:3.9-slim as base
# ... base setup

FROM base as dependencies
# ... install dependencies

FROM dependencies as production
# ... final production image
```

**Benefits:**

- Smaller image size (~400MB vs ~1.5GB)
- Faster deployment
- Reduced attack surface

#### 2. Security Hardening

```bash
# Run as non-root user
USER mluser

# Read-only root filesystem
docker run --read-only -p 8501:8501 kertas-paleographer

# Security scanning
docker scan kertas-paleographer
```

#### 3. Resource Limits

```bash
# Limit CPU and memory
docker run -d \
  --cpus="2.0" \
  --memory="4g" \
  --memory-swap="4g" \
  -p 8501:8501 \
  kertas-paleographer
```

In `docker-compose.yml`:

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 2G
```

#### 4. Health Checks

Already configured in our setup:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

#### 5. Logging

```bash
# Configure logging driver
docker run -d \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  kertas-paleographer
```

---

## ☁️ Cloud Deployment

### AWS ECS (Elastic Container Service)

```bash
# 1. Authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

# 2. Tag image
docker tag kertas-paleographer:latest \
  123456789.dkr.ecr.us-east-1.amazonaws.com/kertas-paleographer:latest

# 3. Push to ECR
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/kertas-paleographer:latest

# 4. Deploy to ECS (via AWS Console or CLI)
aws ecs update-service \
  --cluster ml-cluster \
  --service kertas-service \
  --force-new-deployment
```

### Google Cloud Run

```bash
# 1. Tag image
docker tag kertas-paleographer gcr.io/PROJECT-ID/kertas-paleographer

# 2. Push to Container Registry
docker push gcr.io/PROJECT-ID/kertas-paleographer

# 3. Deploy to Cloud Run
gcloud run deploy kertas-paleographer \
  --image gcr.io/PROJECT-ID/kertas-paleographer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# 1. Login to Azure
az login

# 2. Create container registry
az acr create --resource-group myResourceGroup \
  --name kertasregistry --sku Basic

# 3. Push image
az acr login --name kertasregistry
docker tag kertas-paleographer kertasregistry.azurecr.io/kertas-paleographer
docker push kertasregistry.azurecr.io/kertas-paleographer

# 4. Deploy
az container create \
  --resource-group myResourceGroup \
  --name kertas-app \
  --image kertasregistry.azurecr.io/kertas-paleographer \
  --dns-name-label kertas-ml \
  --ports 8501
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kertas-paleographer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kertas
  template:
    metadata:
      labels:
        app: kertas
    spec:
      containers:
        - name: kertas
          image: kertas-paleographer:latest
          ports:
            - containerPort: 8501
          resources:
            limits:
              memory: "4Gi"
              cpu: "2"
            requests:
              memory: "2Gi"
              cpu: "1"
          livenessProbe:
            httpGet:
              path: /_stcore/health
              port: 8501
            initialDelaySeconds: 30
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: kertas-service
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 8501
  selector:
    app: kertas
```

Deploy:

```bash
kubectl apply -f deployment.yaml
kubectl get services
```

---

## 🎓 MLDevOps Best Practices

### 1. Version Control

```bash
# Tag images with git commit
GIT_COMMIT=$(git rev-parse --short HEAD)
docker build -t kertas-paleographer:$GIT_COMMIT .
docker tag kertas-paleographer:$GIT_COMMIT kertas-paleographer:latest
```

### 2. CI/CD Pipeline (.github/workflows/docker.yml)

```yaml
name: Docker Build & Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t kertas-paleographer .

      - name: Run tests in container
        run: docker run kertas-paleographer pytest

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push kertas-paleographer:latest
```

### 3. Model Versioning

```bash
# Tag with model version
docker build -t kertas-paleographer:model-v1.2 .

# Save model metadata
docker inspect kertas-paleographer:model-v1.2
```

### 4. Monitoring & Observability

```yaml
# docker-compose with monitoring
services:
  app:
    # ... your app config

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

### 5. Data Pipeline Integration

```bash
# Mount data processing volumes
docker run -v /data/input:/app/input:ro \
           -v /data/output:/app/output:rw \
           kertas-paleographer python main.py
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Error: Bind for 0.0.0.0:8501 failed: port is already allocated

# Solution: Use different port
docker run -p 8502:8501 kertas-paleographer

# Or kill process using port 8501
lsof -ti:8501 | xargs kill -9
```

#### 2. Permission Denied

```bash
# Error: permission denied while trying to connect to Docker daemon

# Solution (Linux): Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### 3. Container Exits Immediately

```bash
# Check logs
docker logs kertas-web

# Run interactively to debug
docker run -it kertas-paleographer bash
```

#### 4. Out of Memory

```bash
# Increase Docker memory limit (Docker Desktop)
# Settings → Resources → Memory → 8GB

# Or run with specific limits
docker run --memory="4g" kertas-paleographer
```

#### 5. Build Fails

```bash
# Clear build cache
docker builder prune -a

# Build with no cache
docker build --no-cache -t kertas-paleographer .
```

### Debugging Commands

```bash
# Inspect container
docker inspect kertas-web

# View resource usage
docker stats kertas-web

# Execute shell in running container
docker exec -it kertas-web bash

# View container processes
docker top kertas-web

# Check container health
docker inspect --format='{{.State.Health.Status}}' kertas-web
```

### Performance Optimization

```bash
# View image layers
docker history kertas-paleographer

# Check image size
docker images kertas-paleographer

# Remove unused images/containers
docker system prune -a
```

---

## 📊 Monitoring & Metrics

### Built-in Health Check

```bash
# Check if container is healthy
docker inspect --format='{{json .State.Health}}' kertas-web | jq
```

### Resource Monitoring

```bash
# Real-time stats
docker stats kertas-web

# Export metrics
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

---

## 🎯 Portfolio Showcase

### MLDevOps Skills Demonstrated

When showcasing this in your portfolio, highlight:

1. **Container Orchestration** ✅

   - Multi-container architecture with Docker Compose
   - Service discovery and networking
   - Volume management for data persistence

2. **Infrastructure as Code** ✅

   - Declarative configuration
   - Reproducible environments
   - Version-controlled infrastructure

3. **Security Best Practices** ✅

   - Multi-stage builds
   - Non-root user execution
   - Health checks and monitoring

4. **Cloud-Ready Architecture** ✅

   - Portable across AWS, GCP, Azure
   - Kubernetes-ready
   - Scalable design

5. **CI/CD Integration** ✅
   - Automated builds
   - Testing in containers
   - Continuous deployment

---

## 📚 Additional Resources

### Learning Resources

- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [MLOps Community](https://mlops.community/)

### Related Files

- `Dockerfile` - Container build instructions
- `docker-compose.yml` - Multi-container orchestration
- `.dockerignore` - Build context exclusions
- `requirements.txt` - Python dependencies

---

## 🤝 Contributing

Improvements to Docker configuration are welcome! Consider:

- GPU support for ML training
- Redis for caching predictions
- PostgreSQL for result storage
- Nginx reverse proxy
- SSL/TLS configuration

---

## 📄 License

This Docker configuration is part of the KERTAS Paleographer project.

---

## 👤 Author

**Aymen Abdelkouddous Hamel**

MLDevOps Engineer | Full Stack Developer

---

**🎉 Your ML application is now production-ready with Docker! 🐳**

For questions or issues, please refer to the main README or create an issue on GitHub.
