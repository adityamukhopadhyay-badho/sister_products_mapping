#!/usr/bin/env python3
"""
Sister Products Mapping API

FastAPI application for processing sister products mapping using AI clustering.
This API processes brand products, creates embeddings, performs clustering, and
stores the results in the database.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text

# Import our modules
from src.database_manager import DatabaseManager
from src.sister_products_mapper import SisterProductsMapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database manager
db_manager = None

def get_db_manager():
    """Get or create database manager instance."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    try:
        db_manager = get_db_manager()
        engine = db_manager.get_db_engine()
        if engine:
            logger.info("Database connection established successfully")
        else:
            logger.error("Failed to establish database connection")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    
    yield
    
    # Shutdown (if needed)
    try:
        if db_manager:
            db_manager.close_connections()
            logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Sister Products Mapping API",
    description="AI-powered sister products mapping system using vector embeddings and clustering",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BrandProcessingRequest(BaseModel):
    brandId: str

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Sister Products Mapping API",
        "version": "1.0.0",
        "description": "AI-powered sister products mapping using vector embeddings and clustering",
        "endpoints": {
            "/health": "Health check endpoint",
            "/api/brands": "Get all active verified brands (GET)",
            "/api/process-brand-commit": "Process brand and commit to draft database table (POST)"
        },
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with database connectivity test."""
    try:
        db_manager = get_db_manager()
        engine = db_manager.get_db_engine()
        
        if engine is None:
            return {"status": "unhealthy", "database": "disconnected", "timestamp": datetime.now().isoformat()}
        
        # Test connection with a simple query
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        return {
            "status": "healthy", 
            "database": "connected",
            "connection_pool": "active",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "error", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/brands")
async def get_all_brands():
    """Get all active verified brands from the database."""
    try:
        logger.info("Fetching all brands from database...")
        
        db_manager = get_db_manager()
        brands_df = db_manager.get_all_brands()
        
        if brands_df.empty:
            return {
                "status": "success",
                "message": "No brands found",
                "brands": [],
                "count": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Convert DataFrame to list of dictionaries
        brands_list = brands_df.to_dict('records')
        
        logger.info(f"Successfully fetched {len(brands_list)} brands")
        
        return {
            "status": "success",
            "message": f"Successfully fetched {len(brands_list)} brands",
            "brands": brands_list,
            "count": len(brands_list),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching brands: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch brands: {e}")

def create_sister_products_payload(processed_df: pd.DataFrame, results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create sister products payload in the required format for database insertion.
    
    Args:
        processed_df: Processed DataFrame with cluster assignments
        results: Results dictionary from sister products mapper
        
    Returns:
        DataFrame with columns: brandId, brandSKUId, sisterProductId, clusterId
    """
    logger.info("Creating sister products payload...")
    
    payload_records = []
    
    # Process each cluster to create sister product mappings
    for cluster_name, cluster_info in results['sisterProductClusters'].items():
        if cluster_name == 'no_sisters':
            # Handle noise points (products without sisters)
            for product in cluster_info['products']:
                payload_records.append({
                    'brandId': product.get('brandId', results.get('brandId', '')),
                    'clusterId': -1,  # Noise cluster
                    'brandSKUId': product['brandSKUId'],
                    'sisterProductId': product['brandSKUId']  # Self-reference for noise
                })
        else:
            # Extract cluster ID from cluster name (e.g., "cluster_1" -> 1)
            cluster_id = int(cluster_name.split('_')[1]) if '_' in cluster_name else 0
            products = cluster_info['products']
            
            if products:
                # First product in cluster becomes the sister product representative
                representative_sku = products[0]['brandSKUId']
                
                # All products in this cluster point to the representative
                for product in products:
                    payload_records.append({
                        'brandId': product.get('brandId', results.get('brandId', '')),
                        'clusterId': cluster_id,
                        'brandSKUId': product['brandSKUId'],
                        'sisterProductId': representative_sku
                    })
    
    payload_df = pd.DataFrame(payload_records)
    
    logger.info(f"Payload created:")
    logger.info(f"  - Total records: {len(payload_df)}")
    logger.info(f"  - Valid clusters: {len(payload_df[payload_df['clusterId'] >= 0])}")
    logger.info(f"  - Noise points: {len(payload_df[payload_df['clusterId'] == -1])}")
    logger.info(f"  - Unique clusters: {payload_df[payload_df['clusterId'] >= 0]['clusterId'].nunique()}")
    
    return payload_df

@app.post("/api/process-brand-commit")
async def process_brand_and_commit_to_draft(request: BrandProcessingRequest):
    """
    Process brand products and commit results to draft database table.
    
    This endpoint:
    1. Fetches products for the specified brand from database
    2. Processes them through AI clustering pipeline  
    3. Creates sister product mappings
    4. Commits results to brands."sisterProductDraft" table
    """
    try:
        logger.info(f"Processing brand: {request.brandId}")
        
        # Initialize components
        db_manager = get_db_manager()
        
        # Phase 1: Fetch brand products from database
        logger.info("Phase 1: Fetching brand products from database...")
        df = db_manager.fetch_brand_products(request.brandId)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No products found for brand {request.brandId}")
        
        logger.info(f"Found {len(df)} products for brand {request.brandId}")
        
        # Get brand name from the fetched data
        brand_name = df['brandLabel'].iloc[0] if 'brandLabel' in df.columns else "Unknown Brand"
        logger.info(f"Processing brand: {brand_name}")
        
        # Phase 2: Initialize sister products mapper and process
        logger.info("Phase 1: Processing brand data and creating embeddings...")
        mapper = SisterProductsMapper()
        processed_df, embeddings, core_identities = mapper.process_brand_data(df, brand_name)
        
        logger.info("Phase 2: Performing clustering...")
        cluster_labels = mapper.perform_clustering(embeddings, brand_name)
        
        logger.info("Phase 3: Generating output...")
        results = mapper.generate_output(processed_df, cluster_labels, brand_name)
        
        # Phase 4: Create payload for database insertion
        logger.info("Phase 4: Creating sister products payload...")
        payload_data = create_sister_products_payload(processed_df, results)
        
        # === DATABASE UPSERT TO DRAFT TABLE ===
        logger.info("Phase 5: Upserting to draft database table...")
        upsert_count = db_manager.upsert_sister_products(
            payload_data, 
            table_full_name='brands."sisterProductDraft"'  # Using draft table for testing
        )
        logger.info(f"Successfully upserted {upsert_count} rows to draft table")

        # Log the payload summary for console viewing
        logger.info("=== PROCESSED PAYLOAD SUMMARY ===")
        logger.info(f"Brand: {brand_name} ({request.brandId})")
        logger.info(f"Total Products: {len(payload_data)}")
        logger.info(f"Clusters Found: {payload_data[payload_data['clusterId'] >= 0]['clusterId'].nunique()}")
        logger.info(f"Products with Sisters: {len(payload_data[payload_data['clusterId'] >= 0])}")
        logger.info(f"Noise Products: {len(payload_data[payload_data['clusterId'] == -1])}")
        logger.info("=== END PAYLOAD SUMMARY ===")

        # Return success response with summary
        return {
            "status": "success",
            "message": "Sister products processed and committed to draft table successfully",
            "brandId": request.brandId,
            "brandName": brand_name,
            "processing_stats": {
            "total_products": results['total_products'],
            "total_clusters": results['total_clusters'],
            "products_with_sisters": results['products_with_sisters'],
            "products_without_sisters": results['products_without_sisters'],
                "clustering_rate": round((results['products_with_sisters'] / results['total_products']) * 100, 1)
            },
            "database_operation": {
                "upserted_rows": upsert_count,
                "table": "brands.sisterProductDraft",
                "operation": "INSERT_ON_CONFLICT_UPDATE",
                "conflict_resolution": "ON (brandId, brandSKUId)",
                "columns_used": "brandId, brandSKUId, sisterProductId, clusterId"
            },
            "payload_summary": {
                "total_records": len(payload_data),
                "valid_clusters": len(payload_data[payload_data['clusterId'] >= 0]),
                "noise_points": len(payload_data[payload_data['clusterId'] == -1]),
                "unique_clusters": payload_data[payload_data['clusterId'] >= 0]['clusterId'].nunique()
            },
            "payload_schema": {
                "brandId": "UUID of the brand",
                "clusterId": "Cluster ID (>= 0, -1 for noise)",
                "brandSKUId": "UUID of the product", 
                "sisterProductId": "UUID of the representative sister product"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing brand {request.brandId}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process brand: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)