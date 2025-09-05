# Docker Deployment Guide for Sister Products Mapping System

This guide explains how to deploy the Sister Products Mapping System using Docker and Docker Compose.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Access to the PostgreSQL database (db.badho.in)
- At least 4GB RAM available for the container

### 1. Clone and Setup
```bash
git clone <repository-url>
cd sister_products_mapping
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit the .env file with your database credentials
nano .env
```

### 3. Build and Run
```bash
# Build the Docker image
docker-compose build

# Start the services
docker-compose up -d

# View logs
docker-compose logs -f sister-products-api
```

### 4. Access the Application
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📋 Available Services

### Main Application
- **sister-products-api**: FastAPI application (Port 8000)
- **redis**: Caching and session storage (Port 6379)
- **nginx**: Reverse proxy and load balancer (Port 80)

### Optional Services
- **postgres**: Local PostgreSQL database (uncomment in docker-compose.yml)

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | Database host | db.badho.in |
| `DB_PORT` | Database port | 5432 |
| `DB_NAME` | Database name | badho-app |
| `DB_USER` | Database username | postgres |
| `DB_PASSWORD` | Database password | Badho_1301 |
| `APP_HOST` | Application host | 0.0.0.0 |
| `APP_PORT` | Application port | 8000 |
| `DEFAULT_MODEL` | AI model to use | all-MiniLM-L6-v2 |

### AI Model Configuration
- **all-MiniLM-L6-v2**: Default, balanced speed/accuracy
- **all-mpnet-base-v2**: More accurate, slower
- **paraphrase-MiniLM-L6-v2**: Faster, less accurate

## 🛠️ API Endpoints

### Core Endpoints
- `GET /` - Web interface for brand selection and processing
- `GET /api/brands` - List all available brands
- `POST /api/process-brand` - Process sister products for a specific brand
- `GET /api/status/{brand_id}` - Get processing status
- `GET /health` - Health check

### Example API Usage

#### Process Sister Products for a Brand
```bash
curl -X POST "http://localhost:8000/api/process-brand" \
  -H "Content-Type: application/json" \
  -d '{
    "brandId": "123e4567-e89b-12d3-a456-426614174000",
    "model_name": "all-MiniLM-L6-v2",
    "min_cluster_size": 2,
    "cluster_epsilon": 0.1,
    "enable_phonetic": false,
    "use_facets": false
  }'
```

#### Get Available Brands
```bash
curl "http://localhost:8000/api/brands"
```

## 📊 Processing Workflow

1. **Brand Selection**: User selects a brand from the web interface
2. **Data Fetching**: System fetches all products for the brand from database
3. **AI Processing**: 
   - Product normalization and embedding generation
   - HDBSCAN clustering to identify sister products
   - Results generation
4. **CSV Export**: Results are formatted and returned as downloadable CSV

## 🔍 Monitoring and Logs

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f sister-products-api

# Last 100 lines
docker-compose logs --tail=100 sister-products-api
```

### Health Monitoring
```bash
# Check service status
docker-compose ps

# Check health endpoint
curl http://localhost:8000/health
```

## 🚀 Production Deployment

### 1. Security Configuration
```bash
# Generate secret key
openssl rand -hex 32

# Update .env with production values
SECRET_KEY=your-generated-secret-key
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
```

### 2. SSL Configuration
```bash
# Create SSL directory
mkdir ssl

# Add your SSL certificates
# ssl/cert.pem - SSL certificate
# ssl/key.pem - Private key
```

### 3. Resource Limits
Update `docker-compose.yml` with resource limits:
```yaml
services:
  sister-products-api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

### 4. Database Connection Pooling
For high-traffic deployments, consider using connection pooling:
```yaml
environment:
  - DB_POOL_SIZE=20
  - DB_MAX_OVERFLOW=30
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```bash
# Check database connectivity
docker-compose exec sister-products-api python -c "
from src.database_manager import DatabaseManager
db = DatabaseManager()
print('DB connection:', db.get_db_engine() is not None)
"
```

#### 2. Out of Memory
```bash
# Check memory usage
docker stats sister-products-api

# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory
```

#### 3. Model Loading Issues
```bash
# Check if model is downloaded
docker-compose exec sister-products-api ls -la ~/.cache/torch/sentence_transformers/

# Clear model cache and restart
docker-compose down
docker-compose up -d
```

#### 4. Processing Timeout
- Large brands may take several minutes to process
- Check logs for progress updates
- Consider using faster models for large datasets

### Debug Mode
```bash
# Run in debug mode
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up
```

## 📈 Performance Optimization

### 1. Model Caching
The application caches loaded models in memory. First request may be slower.

### 2. Batch Processing
For multiple brands, consider using the bulk processing mode:
```bash
docker-compose exec sister-products-api python main.py --bulk-run
```

### 3. Resource Scaling
```bash
# Scale the API service
docker-compose up -d --scale sister-products-api=3
```

## 🔄 Updates and Maintenance

### Update Application
```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup Data
```bash
# Backup output files
tar -czf backup-$(date +%Y%m%d).tar.gz output/ logs/

# Backup database (if using local postgres)
docker-compose exec postgres pg_dump -U postgres sister_products > backup.sql
```

### Clean Up
```bash
# Remove unused containers and images
docker system prune -a

# Remove specific volumes
docker volume rm sister_products_mapping_postgres_data
```

## 📞 Support

For issues and questions:
1. Check the logs: `docker-compose logs -f`
2. Verify environment configuration
3. Test database connectivity
4. Check resource usage

## 🎯 Next Steps

1. **Monitoring**: Set up Prometheus/Grafana for metrics
2. **Logging**: Configure centralized logging (ELK stack)
3. **CI/CD**: Set up automated deployment pipeline
4. **Scaling**: Implement horizontal scaling with load balancer
5. **Security**: Add authentication and authorization
