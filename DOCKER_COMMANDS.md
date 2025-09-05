# 🐳 Docker Management Commands

## 📋 **Basic Docker Commands**

### **Start Services (Up)**
```bash
# Start all services in background
docker-compose up -d

# Start with build (when code changes)
docker-compose up -d --build

# Start specific service only
docker-compose up -d sister-products-api

# Start with logs visible
docker-compose up
```

### **Stop Services (Down)**
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop and remove everything (containers, networks, volumes)
docker-compose down --remove-orphans

# Stop specific service
docker-compose stop sister-products-api
```

### **Restart Services**
```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart sister-products-api

# Restart with rebuild
docker-compose up -d --build --force-recreate
```

## 🔄 **When Code Changes**

### **Method 1: Quick Restart (Recommended)**
```bash
# Stop services
docker-compose down

# Rebuild and start
docker-compose up -d --build
```

### **Method 2: Force Recreate**
```bash
# Stop and remove everything
docker-compose down --remove-orphans

# Rebuild from scratch
docker-compose build --no-cache

# Start services
docker-compose up -d
```

## 📊 **Monitoring Commands**

### **View Logs**
```bash
# All services logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f sister-products-api

# Last 100 lines
docker-compose logs --tail=100 sister-products-api
```

### **Check Status**
```bash
# Show running containers
docker-compose ps

# Show all containers (including stopped)
docker-compose ps -a

# Show resource usage
docker stats
```

### **Health Check**
```bash
# Check if API is responding
curl http://localhost:8000/health

# Check specific service
docker-compose exec sister-products-api curl localhost:8000/health
```

## 🧹 **Cleanup Commands**

### **Remove Everything**
```bash
# Stop and remove all
docker-compose down --remove-orphans --volumes

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Complete cleanup (be careful!)
docker system prune -a --volumes
```

## 📝 **Common Workflows**

### **Daily Development**
```bash
# Start services
docker-compose up -d

# Make code changes...

# Restart with changes
docker-compose up -d --build

# View logs
docker-compose logs -f sister-products-api
```

### **Production Deployment**
```bash
# Stop old version
docker-compose down

# Pull latest code
git pull

# Build and start new version
docker-compose up -d --build

# Verify health
curl http://localhost:8000/health
```
