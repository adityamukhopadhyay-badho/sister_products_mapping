#!/usr/bin/env python3
"""
Enhanced FastAPI Application for Sister Products Mapping System

This application provides a REST API for processing sister products mapping
with the ability to process specific brands and view/download results.

Author: Sister Products Mapping System
Date: 2024
"""

import os
import io
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules
from src.sister_products_mapper import SisterProductsMapper
from src.database_manager import DatabaseManager
from post_process_pipeline import PostProcessingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sister Products Mapping API",
    description="AI-powered system for mapping sister products using vector embeddings and clustering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
mapper_instance = None
db_manager = None

class BrandProcessingRequest(BaseModel):
    brandId: str
    model_name: Optional[str] = "all-MiniLM-L6-v2"
    min_cluster_size: Optional[int] = 2
    min_samples: Optional[int] = 1
    cluster_epsilon: Optional[float] = 0.1
    enable_phonetic: Optional[bool] = False
    use_facets: Optional[bool] = False

class ProcessingStatus(BaseModel):
    status: str
    message: str
    brandId: str
    timestamp: str
    total_products: Optional[int] = None
    total_clusters: Optional[int] = None

# In-memory storage for processing status (in production, use Redis or database)
processing_status = {}

def get_mapper_instance():
    """Get or create mapper instance (singleton pattern)."""
    global mapper_instance
    if mapper_instance is None:
        mapper_instance = SisterProductsMapper(
            model_name="all-MiniLM-L6-v2",
            min_cluster_size=2,
            min_samples=1,
            cluster_selection_epsilon=0.1,
            enable_phonetic=False,
            use_facets=False,
            output_dir="output",
            logs_dir="logs"
        )
    return mapper_instance

def get_db_manager():
    """Get or create database manager instance."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sister Products Mapping System</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #34495e;
            }
            input, select {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            button {
                background-color: #3498db;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-right: 10px;
                margin-bottom: 10px;
            }
            button:hover {
                background-color: #2980b9;
            }
            button:disabled {
                background-color: #bdc3c7;
                cursor: not-allowed;
            }
            .button-group {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .status {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }
            .status.success {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.error {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .status.info {
                background-color: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .results {
                margin-top: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
                display: none;
            }
            .brand-list {
                max-height: 200px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background: white;
            }
            .brand-item {
                padding: 5px;
                cursor: pointer;
                border-radius: 3px;
            }
            .brand-item:hover {
                background-color: #e9ecef;
            }
            .data-preview {
                margin-top: 20px;
                max-height: 500px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .data-table {
                width: 100%;
                border-collapse: collapse;
                background: white;
            }
            .data-table th,
            .data-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .data-table th {
                background-color: #f8f9fa;
                font-weight: bold;
                position: sticky;
                top: 0;
            }
            .data-table tr:hover {
                background-color: #f5f5f5;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: white;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
            .stat-label {
                font-size: 14px;
                color: #7f8c8d;
                margin-top: 5px;
            }
            .cluster-info {
                background: #e8f4f8;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .cluster-header {
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
            }
            .cluster-products {
                font-size: 14px;
                color: #34495e;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Sister Products Mapping System</h1>
            <p style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">
                AI-powered system for identifying and clustering sister products using vector embeddings
            </p>
            
            <form id="processingForm">
                <div class="form-group">
                    <label for="brandId">Brand ID:</label>
                    <input type="text" id="brandId" name="brandId" placeholder="Enter brand ID (e.g., 123e4567-e89b-12d3-a456-426614174000)" required>
                    <div class="brand-list" id="brandList" style="display: none;"></div>
                </div>
                
                <div class="form-group">
                    <label for="model_name">AI Model:</label>
                    <select id="model_name" name="model_name">
                        <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2 (Default)</option>
                        <option value="all-mpnet-base-v2">all-mpnet-base-v2 (More Accurate)</option>
                        <option value="paraphrase-MiniLM-L6-v2">paraphrase-MiniLM-L6-v2 (Faster)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="min_cluster_size">Minimum Cluster Size:</label>
                    <input type="number" id="min_cluster_size" name="min_cluster_size" value="2" min="2" max="10">
                </div>
                
                <div class="form-group">
                    <label for="cluster_epsilon">Cluster Sensitivity:</label>
                    <input type="range" id="cluster_epsilon" name="cluster_epsilon" min="0" max="0.5" step="0.05" value="0.1">
                    <span id="epsilonValue">0.1</span>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="enable_phonetic" name="enable_phonetic"> 
                        Enable Phonetic Similarity (for similar-sounding words)
                    </label>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="use_facets" name="use_facets"> 
                        Use Rich Facets Data (more detailed analysis)
                    </label>
                </div>
                
                <div class="button-group">
                    <button type="submit" id="processBtn">🚀 Process Sister Products</button>
                    <button type="button" id="getBrandsBtn">📋 Get Available Brands</button>
                    <button type="button" id="downloadCsvBtn" style="display: none;">📥 Download CSV</button>
                    <button type="button" id="viewDataBtn" style="display: none;">👁️ View Data</button>
                </div>
            </form>
            
            <div class="status" id="status"></div>
            <div class="results" id="results"></div>
            <div class="data-preview" id="dataPreview" style="display: none;"></div>
        </div>

        <script>
            let currentData = null;
            let currentCsvData = null;

            // Update epsilon value display
            document.getElementById('cluster_epsilon').addEventListener('input', function() {
                document.getElementById('epsilonValue').textContent = this.value;
            });

            // Get available brands
            document.getElementById('getBrandsBtn').addEventListener('click', async function() {
                const brandList = document.getElementById('brandList');
                const status = document.getElementById('status');
                
                status.className = 'status info';
                status.style.display = 'block';
                status.innerHTML = '<div class="loading"></div>Fetching available brands...';
                
                try {
                    const response = await fetch('/api/brands');
                    const brands = await response.json();
                    
                    if (brands.length > 0) {
                        brandList.innerHTML = brands.map(brand => 
                            `<div class="brand-item" onclick="selectBrand('${brand.brandId}', '${brand.brandName}')">
                                <strong>${brand.brandName}</strong><br>
                                <small>ID: ${brand.brandId}</small>
                            </div>`
                        ).join('');
                        brandList.style.display = 'block';
                        status.innerHTML = `Found ${brands.length} brands. Click on a brand to select it.`;
                    } else {
                        status.innerHTML = 'No brands found.';
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.innerHTML = 'Error fetching brands: ' + error.message;
                }
            });

            function selectBrand(brandId, brandName) {
                document.getElementById('brandId').value = brandId;
                document.getElementById('brandList').style.display = 'none';
                document.getElementById('status').innerHTML = `Selected: ${brandName}`;
            }

            // Process form submission
            document.getElementById('processingForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const data = Object.fromEntries(formData);
                
                // Convert checkbox values
                data.enable_phonetic = document.getElementById('enable_phonetic').checked;
                data.use_facets = document.getElementById('use_facets').checked;
                
                const processBtn = document.getElementById('processBtn');
                const status = document.getElementById('status');
                const results = document.getElementById('results');
                const dataPreview = document.getElementById('dataPreview');
                const downloadCsvBtn = document.getElementById('downloadCsvBtn');
                const viewDataBtn = document.getElementById('viewDataBtn');
                
                processBtn.disabled = true;
                processBtn.innerHTML = '<div class="loading"></div>Processing...';
                
                status.className = 'status info';
                status.style.display = 'block';
                status.innerHTML = '<div class="loading"></div>Starting sister products processing...';
                
                // Hide previous results
                results.style.display = 'none';
                dataPreview.style.display = 'none';
                downloadCsvBtn.style.display = 'none';
                viewDataBtn.style.display = 'none';
                
                try {
                    // First, get the JSON data for display
                    const jsonResponse = await fetch('/api/process-brand-json', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    if (jsonResponse.ok) {
                        const jsonData = await jsonResponse.json();
                        currentData = jsonData;
                        
                        // Display statistics
                        results.style.display = 'block';
                        results.innerHTML = `
                            <h3>Processing Complete</h3>
                            <div class="stats-grid">
                                <div class="stat-card">
                                    <div class="stat-value">${jsonData.total_products}</div>
                                    <div class="stat-label">Total Products</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-value">${jsonData.total_clusters}</div>
                                    <div class="stat-label">Sister Product Clusters</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-value">${jsonData.products_with_sisters}</div>
                                    <div class="stat-label">Products with Sisters</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-value">${((jsonData.products_with_sisters / jsonData.total_products) * 100).toFixed(1)}%</div>
                                    <div class="stat-label">Clustering Rate</div>
                                </div>
                            </div>
                            <p><strong>Brand ID:</strong> ${data.brandId}</p>
                            <p><strong>Model Used:</strong> ${data.model_name}</p>
                            <p><strong>Processing Time:</strong> ${new Date().toLocaleString()}</p>
                        `;
                        
                        // Show action buttons
                        downloadCsvBtn.style.display = 'inline-block';
                        viewDataBtn.style.display = 'inline-block';
                        
                        status.className = 'status success';
                        status.innerHTML = '✅ Sister products processing completed! You can now view the data or download the CSV.';
                        
                    } else {
                        const error = await jsonResponse.json();
                        throw new Error(error.detail || 'Processing failed');
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.innerHTML = '❌ Error: ' + error.message;
                } finally {
                    processBtn.disabled = false;
                    processBtn.innerHTML = '🚀 Process Sister Products';
                }
            });

            // Download CSV button
            document.getElementById('downloadCsvBtn').addEventListener('click', async function() {
                if (!currentData) return;
                
                const formData = new FormData(document.getElementById('processingForm'));
                const data = Object.fromEntries(formData);
                data.enable_phonetic = document.getElementById('enable_phonetic').checked;
                data.use_facets = document.getElementById('use_facets').checked;
                
                try {
                    const response = await fetch('/api/process-brand', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `sister_products_${data.brandId}_${new Date().toISOString().split('T')[0]}.csv`;
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        document.body.removeChild(a);
                    } else {
                        throw new Error('CSV download failed');
                    }
                } catch (error) {
                    alert('Error downloading CSV: ' + error.message);
                }
            });

            // View data button
            document.getElementById('viewDataBtn').addEventListener('click', function() {
                if (!currentData) return;
                
                const dataPreview = document.getElementById('dataPreview');
                
                if (dataPreview.style.display === 'none') {
                    // Show data
                    let tableHtml = `
                        <h3>📊 Sister Products Data Preview</h3>
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Brand ID</th>
                                    <th>Cluster ID</th>
                                    <th>Product SKU ID</th>
                                    <th>Sister Product ID</th>
                                </tr>
                            </thead>
                            <tbody>
                    `;
                    
                    currentData.sister_products_data.forEach(row => {
                        tableHtml += `
                            <tr>
                                <td>${row.brandId}</td>
                                <td>${row.clusterId}</td>
                                <td>${row.brandSKUId}</td>
                                <td>${row.sisterProductId}</td>
                            </tr>
                        `;
                    });
                    
                    tableHtml += '</tbody></table>';
                    
                    // Add cluster information
                    if (currentData.cluster_details) {
                        tableHtml += '<h4>🔍 Cluster Details</h4>';
                        currentData.cluster_details.forEach((cluster, index) => {
                            if (cluster.products && cluster.products.length > 0) {
                                tableHtml += `
                                    <div class="cluster-info">
                                        <div class="cluster-header">Cluster ${cluster.cluster_id} (${cluster.products.length} products)</div>
                                        <div class="cluster-products">
                                            ${cluster.products.map(p => p.label).join(', ')}
                                        </div>
                                    </div>
                                `;
                            }
                        });
                    }
                    
                    dataPreview.innerHTML = tableHtml;
                    dataPreview.style.display = 'block';
                    this.innerHTML = '👁️ Hide Data';
                } else {
                    // Hide data
                    dataPreview.style.display = 'none';
                    this.innerHTML = '👁️ View Data';
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/api/brands")
async def get_available_brands():
    """Get list of available brands from the database."""
    try:
        db_manager = get_db_manager()
        brands_df = db_manager.get_all_brands()
        
        brands = []
        for _, row in brands_df.iterrows():
            brands.append({
                "brandId": row["brandId"],
                "brandName": row["brandName"]
            })
        
        return brands
    except Exception as e:
        logger.error(f"Error fetching brands: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch brands: {str(e)}")

@app.post("/api/process-brand-json")
async def process_brand_sister_products_json(request: BrandProcessingRequest):
    """
    Process sister products for a specific brand and return JSON data for display.
    """
    try:
        # Update processing status
        processing_status[request.brandId] = ProcessingStatus(
            status="processing",
            message="Starting sister products processing...",
            brandId=request.brandId,
            timestamp=datetime.now().isoformat()
        )
        
        # Get database manager and fetch brand products
        db_manager = get_db_manager()
        products_df = db_manager.fetch_brand_products(request.brandId)
        
        if products_df.empty:
            raise HTTPException(status_code=404, detail=f"No products found for brand ID: {request.brandId}")
        
        # Update status
        processing_status[request.brandId].total_products = len(products_df)
        processing_status[request.brandId].message = f"Found {len(products_df)} products. Starting AI processing..."
        
        # Create mapper instance with custom parameters
        mapper = SisterProductsMapper(
            model_name=request.model_name,
            min_cluster_size=request.min_cluster_size,
            min_samples=request.min_samples,
            cluster_selection_epsilon=request.cluster_epsilon,
            enable_phonetic=request.enable_phonetic,
            use_facets=request.use_facets,
            output_dir="output",
            logs_dir="logs"
        )
        
        # Get brand name for processing
        brand_name = products_df['brandLabel'].iloc[0] if 'brandLabel' in products_df.columns else request.brandId
        
        # Process through the sister products mapping pipeline
        processing_status[request.brandId].message = "Processing products through AI pipeline..."
        
        # Phase 1: Normalize and create embeddings
        processed_df, embeddings, core_identities = mapper.process_brand_data(products_df, brand_name)
        
        # Phase 2: Perform clustering
        processing_status[request.brandId].message = "Performing AI clustering..."
        cluster_labels = mapper.perform_clustering(embeddings, brand_name, processed_df)
        
        # Phase 3: Generate output
        processing_status[request.brandId].message = "Generating sister products mapping..."
        results = mapper.generate_output(processed_df, cluster_labels, brand_name)
        
        # Create sister products CSV data
        csv_data = create_sister_products_csv(processed_df, results)
        
        # Create cluster details for display
        cluster_details = []
        for cluster_id in set(cluster_labels):
            if cluster_id != -1:  # Skip noise points
                cluster_products = processed_df[processed_df['cluster_id'] == cluster_id]
                cluster_details.append({
                    'cluster_id': cluster_id,
                    'products': [
                        {
                            'brandSKUId': row['brandSKUId'],
                            'label': row['label'],
                            'normalized_name': row['normalized_name']
                        }
                        for _, row in cluster_products.iterrows()
                    ]
                })
        
        # Update final status
        processing_status[request.brandId].status = "completed"
        processing_status[request.brandId].message = f"Successfully processed {len(products_df)} products into {results['total_clusters']} sister product clusters."
        
        # Return JSON response with all data
        return JSONResponse(content={
            "brandId": request.brandId,
            "brandName": brand_name,
            "total_products": results['total_products'],
            "total_clusters": results['total_clusters'],
            "products_with_sisters": results['products_with_sisters'],
            "products_without_sisters": results['products_without_sisters'],
            "sister_products_data": csv_data.to_dict('records'),
            "cluster_details": cluster_details,
            "processing_metadata": results['processing_metadata'],
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing brand {request.brandId}: {e}")
        
        # Update status with error
        if request.brandId in processing_status:
            processing_status[request.brandId].status = "error"
            processing_status[request.brandId].message = f"Processing failed: {str(e)}"
        
        raise HTTPException(status_code=500, detail=f"Failed to process brand: {str(e)}")

@app.post("/api/process-brand")
async def process_brand_sister_products(request: BrandProcessingRequest):
    """
    Process sister products for a specific brand and return CSV file.
    """
    try:
        # Get database manager and fetch brand products
        db_manager = get_db_manager()
        products_df = db_manager.fetch_brand_products(request.brandId)
        
        if products_df.empty:
            raise HTTPException(status_code=404, detail=f"No products found for brand ID: {request.brandId}")
        
        # Create mapper instance with custom parameters
        mapper = SisterProductsMapper(
            model_name=request.model_name,
            min_cluster_size=request.min_cluster_size,
            min_samples=request.min_samples,
            cluster_selection_epsilon=request.cluster_epsilon,
            enable_phonetic=request.enable_phonetic,
            use_facets=request.use_facets,
            output_dir="output",
            logs_dir="logs"
        )
        
        # Get brand name for processing
        brand_name = products_df['brandLabel'].iloc[0] if 'brandLabel' in products_df.columns else request.brandId
        
        # Process through the sister products mapping pipeline
        processed_df, embeddings, core_identities = mapper.process_brand_data(products_df, brand_name)
        cluster_labels = mapper.perform_clustering(embeddings, brand_name, processed_df)
        results = mapper.generate_output(processed_df, cluster_labels, brand_name)
        
        # Create CSV in memory using the post-processing pipeline format
        csv_data = create_sister_products_csv(processed_df, results)
        
        # Create CSV file in memory
        csv_buffer = io.StringIO()
        csv_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        csv_buffer.close()
        
        # Return CSV file as streaming response
        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=sister_products_{request.brandId}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing brand {request.brandId}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process brand: {str(e)}")

@app.get("/api/status/{brand_id}")
async def get_processing_status(brand_id: str):
    """Get processing status for a specific brand."""
    if brand_id not in processing_status:
        raise HTTPException(status_code=404, detail="Brand not found in processing status")
    
    return processing_status[brand_id]

def create_sister_products_csv(processed_df: pd.DataFrame, results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create CSV data in the format expected by the post-processing pipeline.
    """
    csv_data = []
    
    for _, row in processed_df.iterrows():
        cluster_id = row['cluster_id']
        brand_id = row['brandId']
        brand_sku_id = row['brandSKUId']
        
        # Find the sister product ID (first product in the cluster)
        if cluster_id != -1:  # Not a noise point
            cluster_products = processed_df[processed_df['cluster_id'] == cluster_id]
            sister_product_id = cluster_products['brandSKUId'].iloc[0]
        else:
            # For noise points, use the product itself as sister
            sister_product_id = brand_sku_id
        
        csv_data.append({
            'brandId': brand_id,
            'clusterId': cluster_id,
            'brandSKUId': brand_sku_id,
            'sisterProductId': sister_product_id
        })
    
    return pd.DataFrame(csv_data)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
