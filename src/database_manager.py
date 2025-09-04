#!/usr/bin/env python3
"""
Database Manager for Sister Products Mapping System

This module handles database connections and data retrieval from PostgreSQL
for the sister products mapping system.
"""

import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database connections and data retrieval."""
    
    def __init__(self):
        self.db_engine = None
        self.console = Console()
        
    def get_db_engine(self):
        """Creates a SQLAlchemy engine using environment variables."""
        if self.db_engine is not None:
            return self.db_engine
            
        try:
            # Try environment variables first
            db_user = os.environ.get('DB_USER', 'postgres')
            db_pass = os.environ.get('DB_PASSWORD', 'Badho_1301')
            db_host = os.environ.get('DB_HOST', 'db.badho.in')
            db_port = os.environ.get('DB_PORT', 5432)
            db_name = os.environ.get('DB_NAME', 'badho-app')
            
            # The connection string format for SQLAlchemy is:
            # "postgresql+psycopg2://user:password@host:port/dbname"
            db_url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            
            # Add TCP keepalives to prevent network timeouts on long queries
            self.db_engine = create_engine(
                db_url, 
                connect_args={
                    "keepalives": 1,
                    "keepalives_idle": 60,
                    "keepalives_interval": 10,
                }
            )
            # Test the connection
            with self.db_engine.connect() as connection:
                logger.info("Database connection successful.")
                self.console.print("[green]✓ Database connection established[/green]")
            return self.db_engine
            
        except Exception as error:
            logger.error(f"Error creating database engine: {error}")
            self.console.print(f"[red]✗ Database connection failed: {error}[/red]")
            return None

    def fetch_brand_products(self, brand_id: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch brand products from the database using the provided SQL query.
        
        Args:
            brand_id: Optional specific brand ID to filter for
            
        Returns:
            DataFrame with columns: brandSKUId, brandId, label, facets_jsonb, categoryLabel
        """
        engine = self.get_db_engine()
        if engine is None:
            raise Exception("Failed to establish database connection")
        
        # Use the exact query provided by the user, updated with brand exclusion
        query = """
        WITH cte AS (
            SELECT 
                bs.id AS "brandSKUId", 
                bs.label, 
                bs."facetsV2Processed" AS facets_jsonb, 
                c.label AS "categoryLabel",
                b.id "brandId",
                b.label "brandLabel",
                ROW_NUMBER() OVER (PARTITION BY bs.id ORDER BY random()) AS rn
            FROM brands."brandSKU" bs
            JOIN brands.brand b ON bs."brandId" = b.id
            JOIN users.seller_brand sb ON sb."brandId" = bs."brandId"
            JOIN brands."brandSKU_category" bsc ON bs.id = bsc."brandSKUId"
            JOIN categories.category c ON bsc."categoryId" = c.id
            WHERE bs."isActive" = true
            AND b."isActive" = true
            AND c."isActive" = true
            AND b.id != 'd0f115f8-4127-4081-8948-ee56070a2a0a'
            -- AND bs."facetsV2Processed" IS NOT NULL
            -- AND b."isBrandBadhoVerified" = true
        """
        
        if brand_id:
            query += f"\n            AND bs.\"brandId\" = '{brand_id}'"
        
        query += """
        )
        SELECT DISTINCT("brandSKUId") "brandSKUId", "brandId", "brandLabel", label, "categoryLabel"
        FROM cte
        WHERE rn = 1
        ORDER BY "brandId";
        """
        
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]Fetching products from database...", total=None)
                
                with engine.connect() as connection:
                    df = pd.read_sql_query(text(query), connection)
                    
                progress.update(task, completed=100, total=100)
                
            self.console.print(f"[green]✓ Fetched {len(df)} products from database[/green]")
            logger.info(f"Successfully fetched {len(df)} products from database")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from database: {e}")
            self.console.print(f"[red]✗ Error fetching data: {e}[/red]")
            raise

    def get_all_brands(self) -> pd.DataFrame:
        """Get list of all active verified brands."""
        engine = self.get_db_engine()
        if engine is None:
            raise Exception("Failed to establish database connection")
            
        query = """
        SELECT DISTINCT b.id as "brandId", 
               COALESCE(b.label, b.id::text) as "brandName"
        FROM brands.brand b
        JOIN users.seller_brand sb ON sb."brandId" = b.id
        WHERE b."isActive" = true
        AND b."isBrandBadhoVerified" = true
        ORDER BY "brandName";
        """
        
        try:
            with engine.connect() as connection:
                df = pd.read_sql_query(text(query), connection)
            
            self.console.print(f"[blue]Found {len(df)} verified brands[/blue]")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching brands: {e}")
            raise

    def save_results_to_db(self, results_df: pd.DataFrame, table_name: str = "sister_products_results"):
        """
        Save clustering results back to database.
        
        Args:
            results_df: DataFrame with clustering results
            table_name: Target table name
        """
        engine = self.get_db_engine()
        if engine is None:
            raise Exception("Failed to establish database connection")
            
        try:
            # Save with conflict resolution on brandSKUId
            results_df.to_sql(
                table_name,
                engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            self.console.print(f"[green]✓ Saved {len(results_df)} results to database table '{table_name}'[/green]")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            self.console.print(f"[red]✗ Error saving to database: {e}[/red]")
            raise 