# Sister Products Mapping API - Deployment

AI-powered sister products mapping system using vector embeddings and clustering.

## 🚀 Quick Start

### Local Development & Testing

1. **Start the service:**
   ```bash
   ./start.sh
   ```

2. **Test the API:**
   ```bash
   # Health check
   curl http://localhost:8000/health

   # Process a brand
   curl -X POST http://localhost:8000/api/process-brand-commit \
     -H "Content-Type: application/json" \
     -d '{"brandId":"a1e5abdc-d3f6-4eb1-9d12-7c5d460e8088"}'
   ```

3. **Stop the service:**
   ```bash
   ./stop.sh
   ```

## 📡 API Endpoints

### `GET /health`
Health check endpoint with database connectivity test.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected", 
  "connection_pool": "active",
  "timestamp": "2025-09-17T11:56:24.223309"
}
```

### `POST /api/process-brand-commit`
Process brand products and commit results to draft database table.

**Request:**
```json
{
  "brandId": "a1e5abdc-d3f6-4eb1-9d12-7c5d460e8088"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Sister products processed and committed to draft table successfully",
  "brandId": "a1e5abdc-d3f6-4eb1-9d12-7c5d460e8088",
  "brandName": "Madhusudan",
  "processing_stats": {
    "total_products": 8,
    "total_clusters": 3,
    "products_with_sisters": 7,
    "products_without_sisters": 1,
    "clustering_rate": 87.5
  },
  "database_operation": {
    "upserted_rows": 8,
    "table": "brands.sisterProductDraft",
    "operation": "INSERT_ON_CONFLICT_UPDATE",
    "columns_used": "brandId, brandSKUId, sisterProductId, clusterId"
  }
}
```

## 🗄️ Database Integration

### Target Table: `brands."sisterProductDraft"`

**Schema:**
- `id`: text, primary key, unique, default: gen_random_uuid()
- `brandId`: text, nullable
- `clusterId`: integer, nullable  
- `brandSKUId`: text, nullable
- `sisterProductId`: text, nullable
- `created_at`: timestamp with time zone, nullable, default: now()
- `updated_at`: timestamp with time zone, nullable, default: now()

### Sister Product Assignment Logic

1. **Cluster-based Assignment**: The first `brandSKUId` in each cluster becomes the `sisterProductId` for all products in that cluster
2. **Noise Points**: Products in cluster -1 (no sisters) reference themselves as `sisterProductId`
3. **Connection Pooling**: Uses SQLAlchemy connection pooling to prevent idle connection timeouts

## 🔧 Configuration

### Database Connection
Database credentials are hardcoded in `docker-compose.yml`:
- Host: `db.badho.in`
- Port: `5432`
- Database: `badho-app`
- User: `postgres`
- Password: `Badho_1301`

### Environment Variables
- `PYTHONUNBUFFERED=1`: Ensures Python output is sent straight to terminal
- `PYTHONPATH=/app`: Sets Python path for module imports

## 🏗️ Architecture

### Core Components
1. **FastAPI Application** (`app.py`): REST API endpoints
2. **Database Manager** (`src/database_manager.py`): PostgreSQL connection and operations
3. **Sister Products Mapper** (`src/sister_products_mapper.py`): AI clustering pipeline

### Processing Pipeline
1. **Fetch Products**: Retrieve brand products from database
2. **Normalization**: Clean and normalize product names
3. **Embeddings**: Generate vector embeddings using Sentence Transformers
4. **Clustering**: Apply HDBSCAN clustering to find sister products
5. **Database Commit**: Store results in `brands."sisterProductDraft"` table

## 🔒 Security Notes

- Database credentials are hardcoded in `docker-compose.yml` for deployment convenience
- This folder is not pushed to GitHub to protect sensitive information
- Uses connection pooling with TCP keepalives for stable database connections

## 📦 Deployment

### For Coolify Deployment
1. Upload this entire `deploy/` folder as your build context
2. Coolify will automatically detect the `Dockerfile` and `docker-compose.yml`
3. The service will be available on port 8000
4. Health checks are configured for monitoring

### Manual Docker Deployment
```bash
# Build and start
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

## 🧹 Removed Components

The following components were removed for API-only deployment:
- Frontend HTML/CSS/JavaScript code
- CSV generation and download functionality  
- Nginx reverse proxy
- Authentication middleware
- File upload capabilities
- `visualizer.py` and `bulk_processor.py` modules

## 📊 Performance

- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Clustering**: HDBSCAN with automatic parameter optimization
- **Connection Pooling**: 5 base connections, 10 overflow, 1-hour recycle
- **Batch Processing**: 1000 rows per database batch

## 🐛 Troubleshooting

### Common Issues
1. **Database Connection Failed**: Check if credentials in `docker-compose.yml` are correct
2. **Port Already in Use**: Change port mapping in `docker-compose.yml`
3. **Out of Memory**: Reduce batch size or use smaller model
4. **Slow Processing**: Consider enabling `fast_clustering=True` for large datasets

### Logs
```bash
# View real-time logs
docker compose logs -f api

# View specific container logs
docker logs sister-products-api
``` 