# Database Integration for Sister Products Mapping

## Overview

The Sister Products Mapping System now supports direct integration with PostgreSQL databases, allowing you to process live product data without the need for CSV files. This enables real-time sister product analysis and automated processing pipelines.

## Key Features

### 🔗 **Database Connectivity**
- **PostgreSQL Support**: Direct connection to PostgreSQL databases using SQLAlchemy
- **Connection Pooling**: Automatic connection management with keepalive settings
- **Environment Variables**: Secure credential management
- **Connection Testing**: Automatic validation of database connectivity

### 📊 **Live Data Processing**
- **Dynamic Queries**: Processes all active, verified brands automatically
- **Brand-wise Processing**: Supports both single brand and multi-brand processing
- **Real-time Data**: Fetches latest product information directly from database
- **Facets Integration**: Uses `facetsV2Processed` JSON data for rich semantic analysis

### 💾 **Advanced Data Management**
- **Master CSV**: Consolidated results across all brands with live updates
- **Conflict Resolution**: Handles duplicate `brandSKUId` entries automatically
- **Incremental Updates**: Supports reprocessing individual brands without affecting others
- **Timestamped Results**: Tracks processing time for audit trails

## Database Schema Requirements

The system expects the following PostgreSQL schema structure:

### Required Tables

1. **`brands.brand`**
   - `id` (UUID): Primary brand identifier
   - `label` or `title` (VARCHAR): Brand name
   - `isActive` (BOOLEAN): Brand active status
   - `isBrandBadhoVerified` (BOOLEAN): Verification status

2. **`brands.brandSKU`**
   - `id` (UUID): Unique product identifier (`brandSKUId`)
   - `brandId` (UUID): Foreign key to brands.brand
   - `label` (VARCHAR): Product name/label
   - `facetsV2Processed` (JSONB): Product facets in JSON format
   - `isActive` (BOOLEAN): Product active status

3. **`brands.brandSKU_category`**
   - `brandSKUId` (UUID): Foreign key to brandSKU
   - `categoryId` (UUID): Foreign key to category

4. **`categories.category`**
   - `id` (UUID): Category identifier
   - `label` (VARCHAR): Category name
   - `isActive` (BOOLEAN): Category active status

5. **`users.seller_brand`**
   - `brandId` (UUID): Links sellers to brands

## Connection Configuration

### Environment Variables
```bash
export DB_USER="postgres"
export DB_PASSWORD="your_password"
export DB_HOST="db.badho.in"
export DB_PORT="5432"
export DB_NAME="badho-app"
```

### Default Connection
If environment variables are not set, the system uses these defaults:
```python
DB_USER = "postgres"
DB_PASSWORD = "Badho_1301"
DB_HOST = "db.badho.in"
DB_PORT = 5432
DB_NAME = "badho-app"
```

## Usage Examples

### 1. Process All Brands from Database
```bash
# Process all verified brands using facets-based embeddings
python3 main.py --from-database --use-facets --cluster-epsilon 0.1

# Traditional approach for all brands
python3 main.py --from-database --cluster-epsilon 0.2
```

### 2. Process Specific Brand
```bash
# Process single brand by ID
python3 main.py --from-database --brand-id "234976b1-4712-405d-88dd-0808aaf2168d" --use-facets

# With custom clustering parameters
python3 main.py --from-database --brand-id "52741216-daa7-43c8-ae92-1bd9543bd142" \
    --use-facets --cluster-epsilon 0.15 --min-cluster-size 3
```

### 3. Batch Processing with Custom Settings
```bash
# Large scale processing with custom batch size
python3 main.py --from-database --batch-size 2000 --use-facets \
    --cluster-epsilon 0.2 --enable-phonetic --no-visualizations
```

### 4. Development and Testing
```bash
# Test database connectivity
python3 test_database.py

# Process small dataset for testing
python3 main.py --from-database --brand-id "specific-id" \
    --min-cluster-size 2 --no-visualizations
```

## Output Files

### Individual Brand Results
For each processed brand, the system generates:
- `{brand_name}_sister_products.json`: Complete clustering results
- `{brand_name}_detailed_results.csv`: Detailed product data with cluster assignments
- `{brand_name}_cluster_summary.csv`: Summary of clusters and their products

### Master Results (Database Mode Only)
- `master_sister_products_results.csv`: Consolidated results across all brands
- `master_processing_summary.json`: Overall statistics and metadata
- `all_brands_database_results.json`: Combined JSON results for all brands

### Live Updates
- Master CSV is updated in real-time as each brand is processed
- Supports incremental updates with conflict resolution on `brandSKUId`
- Timestamps track when each product was last processed

## Data Processing Pipeline

### 1. **Data Extraction**
```sql
WITH cte AS (
    SELECT 
        bs.id AS "brandSKUId", 
        bs.label, 
        bs."facetsV2Processed" AS facets_jsonb, 
        c.label AS "categoryLabel",
        b.id "brandId",
        ROW_NUMBER() OVER (PARTITION BY bs.id ORDER BY random()) AS rn
    FROM brands."brandSKU" bs
    JOIN brands.brand b ON bs."brandId" = b.id
    JOIN users.seller_brand sb ON sb."brandId" = bs."brandId"
    JOIN brands."brandSKU_category" bsc ON bs.id = bsc."brandSKUId"
    JOIN categories.category c ON bsc."categoryId" = c.id
    WHERE bs."isActive" = true
    AND b."isActive" = true
    AND c."isActive" = true
    AND bs."facetsV2Processed" IS NOT NULL
    AND b."isBrandBadhoVerified" = true
)
SELECT DISTINCT("brandSKUId") "brandSKUId", "brandId", label, facets_jsonb, "categoryLabel"
FROM cte
WHERE rn = 1
ORDER BY "brandId";
```

### 2. **Sister Products Analysis**
- **Normalization**: Clean product names and extract core identity
- **Facets Processing**: Extract semantic information from JSON facets
- **Embedding Generation**: Create vector representations using sentence-transformers
- **Clustering**: Apply HDBSCAN for automatic sister product grouping
- **Results Generation**: Create structured outputs with cluster assignments

### 3. **Output Generation**
- Save individual brand results
- Update master CSV with conflict resolution
- Generate summary statistics
- Create visualizations (optional)

## Advanced Features

### Multi-Category Support
Products can belong to multiple categories (e.g., "Sweets/Snacks"):
```python
# Automatic parsing of category strings
categories = ["Sweets", "Snacks"]  # From "Sweets/Snacks"
```

### Facets-Based Embeddings
When using `--use-facets`, the system leverages ALL available facet data:
```json
{
  "tags": ["corn chips", "peri peri", "spicy"],
  "flavour": "Peri Peri",
  "texture": "Crispy",
  "productType": "Corn Chips",
  "spiceLevel": "Spicy",
  "keyIngredient": "Corn",
  "dietaryPreference": "Vegetarian"
}
```

### Conflict Resolution
The system handles data conflicts intelligently:
- **Brand Level**: Replaces existing brand data when reprocessing
- **Product Level**: Uses `brandSKUId` for unique identification
- **Timestamp Tracking**: Maintains processing history

### Performance Optimization
- **Connection Pooling**: Reuses database connections efficiently
- **Batch Processing**: Configurable batch sizes for large datasets
- **Memory Management**: Processes brands sequentially to manage memory usage
- **Progress Tracking**: Real-time progress indicators with rich console output

## Error Handling

### Database Connection Issues
```python
# Automatic retry and graceful degradation
try:
    engine = db_manager.get_db_engine()
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    # Falls back to CSV mode or exits gracefully
```

### Data Quality Issues
- **Missing Facets**: Products without `facetsV2Processed` are filtered out
- **Invalid JSON**: Malformed facets are logged and skipped
- **Empty Results**: Brands with no valid products are reported but don't stop processing

### Processing Failures
- **Brand-Level Failures**: Individual brand failures don't stop overall processing
- **Clustering Failures**: Fallback to noise assignment (-1 cluster)
- **Save Failures**: Logged but don't prevent continuation

## Monitoring and Debugging

### Logging
```python
# Detailed logging with multiple levels
logger.info("Successfully processed 1,234 products")
logger.warning("Brand XYZ has no valid products")
logger.error("Clustering failed for Brand ABC")
```

### Progress Tracking
- Real-time console output with rich formatting
- Brand-by-brand progress indicators
- Processing time tracking
- Memory usage monitoring

### Validation Tools
```bash
# Test database connectivity and basic functionality
python3 test_database.py

# Validate specific brand processing
python3 main.py --from-database --brand-id "test-id" --no-visualizations
```

## Best Practices

### Production Deployment
1. **Set Environment Variables**: Never hardcode credentials
2. **Use Connection Pooling**: Configure appropriate pool settings
3. **Monitor Performance**: Track processing times and memory usage
4. **Regular Backups**: Backup results before large processing runs
5. **Incremental Processing**: Process brands individually for large datasets

### Performance Tuning
```bash
# For large datasets
python3 main.py --from-database --batch-size 5000 --no-visualizations

# For high accuracy
python3 main.py --from-database --use-facets --enable-phonetic --cluster-epsilon 0.1

# For speed
python3 main.py --from-database --cluster-epsilon 0.3 --min-cluster-size 5
```

### Data Quality
- Ensure `facetsV2Processed` contains structured, clean JSON data
- Verify brand verification status (`isBrandBadhoVerified = true`)
- Monitor product activation status regularly
- Validate category assignments are current

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   ```bash
   # Increase keepalive settings in database_manager.py
   connect_args={
       "keepalives": 1,
       "keepalives_idle": 120,  # Increased from 60
       "keepalives_interval": 30  # Increased from 10
   }
   ```

2. **Memory Issues with Large Datasets**
   ```bash
   # Process brands individually
   python3 main.py --from-database --brand-id "specific-id"
   
   # Reduce batch size
   python3 main.py --from-database --batch-size 500
   ```

3. **Slow Processing**
   ```bash
   # Skip visualizations
   python3 main.py --from-database --no-visualizations
   
   # Use higher cluster epsilon
   python3 main.py --from-database --cluster-epsilon 0.3
   ```

4. **Empty Results**
   - Check brand verification status
   - Verify products have valid facets
   - Ensure products are active
   - Check category assignments

## Migration from CSV

To migrate from CSV-based processing to database processing:

1. **Verify Data Structure**: Ensure database schema matches expected format
2. **Test Connection**: Run `python3 test_database.py`
3. **Pilot Run**: Process single brand first
4. **Compare Results**: Validate against existing CSV results
5. **Full Migration**: Process all brands from database

## Security Considerations

- **Credentials**: Use environment variables or secure credential management
- **Network Security**: Ensure database connections are encrypted
- **Access Control**: Use dedicated database user with minimal required permissions
- **Audit Trails**: Monitor and log all database access
- **Data Privacy**: Ensure compliance with data protection regulations

This database integration provides a robust, scalable solution for enterprise-level sister product mapping with real-time data processing capabilities. 