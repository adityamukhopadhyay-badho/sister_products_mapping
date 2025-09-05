# 🐳 Docker Quick Guide

## 🚀 **Easy Docker Commands**

### **Start Services**
```bash
# Quick start
./start_docker.sh

# Or manually
docker-compose up -d --build
```

### **Stop Services**
```bash
# Quick stop
./stop_docker.sh

# Or manually
docker-compose down
```

### **Restart with Code Changes**
```bash
# Quick restart
./restart_docker.sh

# Or manually
docker-compose down && docker-compose up -d --build
```

## 📊 **Monitor Services**

### **View Logs**
```bash
# All services
docker-compose logs -f

# Just the API
docker-compose logs -f sister-products-api
```

### **Check Status**
```bash
# See running containers
docker-compose ps

# Check health
curl http://localhost:8000/health
```

## 🔄 **When You Make Code Changes**

### **Method 1: Quick Restart (Recommended)**
```bash
./restart_docker.sh
```

### **Method 2: Manual Steps**
```bash
# 1. Stop services
docker-compose down

# 2. Rebuild and start
docker-compose up -d --build
```

### **Method 3: Force Rebuild (if issues)**
```bash
# 1. Stop everything
docker-compose down --remove-orphans

# 2. Rebuild from scratch
docker-compose build --no-cache

# 3. Start services
docker-compose up -d
```

## 🧹 **Cleanup Commands**

### **Remove Everything**
```bash
# Stop and remove all
docker-compose down --remove-orphans --volumes

# Remove unused images
docker image prune -a
```

## �� **Common Workflows**

### **Daily Development**
```bash
# Start services
./start_docker.sh

# Make code changes...

# Restart with changes
./restart_docker.sh

# View logs if needed
docker-compose logs -f sister-products-api
```

### **Production Deployment**
```bash
# Stop old version
docker-compose down

# Pull latest code
git pull

# Start new version
./start_docker.sh

# Verify it's working
curl http://localhost:8000/health
```

## ⚠️ **Troubleshooting**

### **If containers won't start**
```bash
# Check logs
docker-compose logs

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### **If port is already in use**
```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process
sudo kill -9 <PID>
```

## 📝 **Summary**

- **Start**: `./start_docker.sh` or `docker-compose up -d --build`
- **Stop**: `./stop_docker.sh` or `docker-compose down`
- **Restart**: `./restart_docker.sh` or `docker-compose down && docker-compose up -d --build`
- **Logs**: `docker-compose logs -f`
- **Status**: `docker-compose ps`
- **Health**: `curl http://localhost:8000/health`
