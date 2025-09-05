# Docker Troubleshooting Guide

## Common Docker Build Issues

### Issue 1: apt-get update fails (exit code 100)

**Error**: `process "/bin/sh -c apt-get update && apt-get install -y ..." did not complete successfully: exit code 100`

**Solutions**:

#### Solution A: Use the fixed Dockerfile
The main Dockerfile has been updated with better error handling and package management.

#### Solution B: Use the alternative Dockerfile
If the slim image continues to have issues, use the alternative Dockerfile:

```bash
# Use the alternative Dockerfile
docker-compose -f docker-compose.alternative.yml up --build
```

#### Solution C: Manual build with verbose output
```bash
# Build with verbose output to see the exact error
docker build --no-cache --progress=plain -t sister-products-api .

# Or build the alternative
docker build --no-cache --progress=plain -f Dockerfile.alternative -t sister-products-api .
```

### Issue 2: Package installation fails

**Error**: `E: Unable to locate package` or similar package errors

**Solution**: The alternative Dockerfile uses the full Python image which has more packages pre-installed.

### Issue 3: Memory issues during build

**Error**: Build process killed or out of memory

**Solution**: 
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory > Increase to 4GB+

# Or build with memory limit
docker build --memory=4g -t sister-products-api .
```

### Issue 4: Network connectivity issues

**Error**: Cannot fetch packages or connect to database

**Solution**:
```bash
# Check Docker network
docker network ls

# Build with network access
docker build --network=host -t sister-products-api .
```

## Testing the Fix

### Test 1: Build the main Dockerfile
```bash
docker build -t sister-products-api .
```

### Test 2: Build the alternative Dockerfile
```bash
docker build -f Dockerfile.alternative -t sister-products-api .
```

### Test 3: Run with docker-compose
```bash
# Main compose file
docker-compose up --build

# Alternative compose file
docker-compose -f docker-compose.alternative.yml up --build
```

### Test 4: Check if the container runs
```bash
# Run the built image
docker run -p 8000:8000 --env-file .env sister-products-api

# Check logs
docker logs <container_id>
```

## Alternative Deployment Methods

### Method 1: Direct Python (No Docker)
```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python app.py
```

### Method 2: Using the startup script
```bash
chmod +x start_api.sh
./start_api.sh
```

### Method 3: Using virtual environment
```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Verification Steps

### 1. Check if the application starts
```bash
# Test the API
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","timestamp":"2024-..."}
```

### 2. Check the web interface
Open http://localhost:8000 in your browser

### 3. Test API endpoints
```bash
# Get available brands
curl http://localhost:8000/api/brands

# Test processing (replace with actual brand ID)
curl -X POST "http://localhost:8000/api/process-brand" \
  -H "Content-Type: application/json" \
  -d '{"brandId": "test-brand-id"}'
```

## If All Else Fails

### Use the working Python setup
The application has been tested and works perfectly with direct Python execution:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py

# 3. Access at http://localhost:8000
```

This method bypasses Docker entirely and uses the system Python environment.
