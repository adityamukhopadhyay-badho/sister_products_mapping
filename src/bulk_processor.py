#!/usr/bin/env python3
"""
Bulk Processor for Sister Products Mapping System

This module handles large-scale batch processing of sister products mapping
for 1.4M+ products across 53K+ brands. It includes:

- Batch processing of 10K brands per batch
- Progress tracking and resumption
- Human-in-the-loop approval mechanism
- Detailed logging and statistics
- Master CSV files per batch

Author: Sister Products Mapping System
Date: 2024
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from rich.console import Console
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import print as rprint

from .database_manager import DatabaseManager
from .sister_products_mapper import SisterProductsMapper

class BulkProcessor:
    """
    Handles large-scale batch processing of sister products mapping.
    
    Features:
    - Processes brands in batches of 10K
    - Tracks progress and allows resumption
    - Human approval for each batch
    - Detailed statistics and logging
    - Master CSV files per batch
    """
    
    def __init__(self, 
                 batch_size: int = 10000,
                 auto_approve: bool = False,
                 output_dir: str = 'output',
                 logs_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints',
                 **mapper_kwargs):
        """
        Initialize the Bulk Processor.
        
        Args:
            batch_size: Number of brands per batch (default: 10000)
            auto_approve: Auto-approve all batches without human confirmation
            output_dir: Directory for output files
            logs_dir: Directory for log files
            checkpoint_dir: Directory for checkpoint files
            **mapper_kwargs: Additional arguments for SisterProductsMapper
        """
        self.batch_size = batch_size
        self.auto_approve = auto_approve
        self.output_dir = Path(output_dir)
        self.logs_dir = Path(logs_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.console = Console()
        self.db_manager = DatabaseManager()
        self.mapper_kwargs = mapper_kwargs
        
        # Progress tracking
        self.processed_brands = set()
        self.failed_brands = set()
        self.batch_stats = {}
        self.global_stats = {
            'total_brands': 0,
            'total_products': 0,
            'total_clusters': 0,
            'products_clustered': 0,
            'processing_start_time': None,
            'processing_end_time': None,
            'batches_completed': 0,
            'batches_failed': 0
        }
        
        # Setup logging
        self._setup_logging()
        
        # Load checkpoint if exists
        self._load_checkpoint()
    
    def _setup_logging(self):
        """Setup comprehensive logging for bulk processing."""
        log_file = self.logs_dir / f"bulk_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Bulk Processor initialized")
    
    def _load_checkpoint(self):
        """Load processing checkpoint if it exists."""
        checkpoint_file = self.checkpoint_dir / "bulk_processing_checkpoint.json"
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                self.processed_brands = set(checkpoint_data.get('processed_brands', []))
                self.failed_brands = set(checkpoint_data.get('failed_brands', []))
                self.batch_stats = checkpoint_data.get('batch_stats', {})
                self.global_stats.update(checkpoint_data.get('global_stats', {}))
                
                self.console.print(f"[yellow]📂 Loaded checkpoint: {len(self.processed_brands)} brands processed[/yellow]")
                self.logger.info(f"Checkpoint loaded: {len(self.processed_brands)} processed, {len(self.failed_brands)} failed")
                
            except Exception as e:
                self.console.print(f"[red]⚠️ Failed to load checkpoint: {e}[/red]")
                self.logger.error(f"Failed to load checkpoint: {e}")
    
    def _check_existing_batch_files(self) -> int:
        """
        Check for existing batch CSV files to determine the last completed batch.
        
        Returns:
            Last completed batch number (0 if no files found)
        """
        last_completed_batch = 0
        
        # Check for existing batch files
        for i in range(1, 1000):  # Reasonable upper limit
            batch_file = self.output_dir / f"master_sister_products_batch{i}.csv"
            if batch_file.exists():
                last_completed_batch = i
            else:
                break
        
        if last_completed_batch > 0:
            self.console.print(f"[blue]📁 Found existing batch files up to batch {last_completed_batch}[/blue]")
            self.logger.info(f"Found existing batch files up to batch {last_completed_batch}")
        
        return last_completed_batch
    
    def _get_remaining_brands_info(self, all_brands: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
        """
        Get information about remaining brands to process.
        
        Args:
            all_brands: DataFrame with all brands
            
        Returns:
            Tuple of (remaining_brands_df, total_brands, remaining_brands_count)
        """
        total_brands = len(all_brands)
        remaining_brands = all_brands[~all_brands['brandId'].isin(self.processed_brands)]
        remaining_count = len(remaining_brands)
        
        return remaining_brands, total_brands, remaining_count
    
    def _save_checkpoint(self):
        """Save current processing state to checkpoint."""
        checkpoint_file = self.checkpoint_dir / "bulk_processing_checkpoint.json"
        
        checkpoint_data = {
            'processed_brands': list(self.processed_brands),
            'failed_brands': list(self.failed_brands),
            'batch_stats': self.batch_stats,
            'global_stats': self.global_stats,
            'last_saved': datetime.now().isoformat()
        }
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.logger.info("Checkpoint saved successfully")
            
        except Exception as e:
            self.console.print(f"[red]⚠️ Failed to save checkpoint: {e}[/red]")
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def get_all_brands(self) -> pd.DataFrame:
        """
        Fetch all active brands from the database, excluding brand IDs from CSV file.
        
        Returns:
            DataFrame with brand information
        """
        self.console.print("[blue]🔍 Fetching all brands from database...[/blue]")
        
        engine = self.db_manager.get_db_engine()
        if engine is None:
            raise Exception("Failed to establish database connection")
        
        # Load excluded brand IDs from CSV file
        excluded_brand_ids = []
        csv_file_path = "brand_ids_not_to_process.csv"
        
        try:
            if os.path.exists(csv_file_path):
                excluded_df = pd.read_csv(csv_file_path)
                excluded_brand_ids = excluded_df['processed_brand_ids'].tolist()
                self.console.print(f"[yellow]📋 Loaded {len(excluded_brand_ids)} brand IDs to exclude[/yellow]")
                self.logger.info(f"Loaded {len(excluded_brand_ids)} brand IDs to exclude from {csv_file_path}")
            else:
                self.console.print(f"[yellow]⚠️ CSV file {csv_file_path} not found, proceeding without exclusions[/yellow]")
                self.logger.warning(f"CSV file {csv_file_path} not found")
        except Exception as e:
            self.console.print(f"[red]⚠️ Error loading excluded brand IDs: {e}[/red]")
            self.logger.error(f"Error loading excluded brand IDs: {e}")
        
        # Build the exclusion condition for the query
        exclusion_condition = ""
        if excluded_brand_ids:
            # Format the excluded brand IDs for SQL NOT IN clause
            excluded_ids_str = "', '".join(excluded_brand_ids)
            exclusion_condition = f"AND b.id NOT IN ('{excluded_ids_str}')"
        
        query = f"""
        SELECT DISTINCT 
            b.id as "brandId",
            b.label as "brandName",
            COUNT(distinct bs.id) as product_count
        FROM brands.brand b
        JOIN brands."brandSKU" bs ON bs."brandId" = b.id
        JOIN brands."brandSKU_category" bsc ON bs.id = bsc."brandSKUId"
        JOIN categories.category c ON bsc."categoryId" = c.id
        JOIN users.seller_brand sb ON b.id = sb."brandId"
        WHERE b."isActive" = true
        AND (b."isTest" = 'false' OR b."isTest" IS NULL)
        AND bs."isActive" = true
        AND c."isActive" = true
        AND b.id != 'd0f115f8-4127-4081-8948-ee56070a2a0a'
        {exclusion_condition}
        GROUP BY b.id, b.label
        HAVING COUNT(distinct bs.id) > 0
        ORDER BY product_count DESC, b.id
        """
        
        try:
            with engine.connect() as connection:
                brands_df = pd.read_sql(query, connection)
            
            self.console.print(f"[green]✓ Found {len(brands_df)} brands with products[/green]")
            self.logger.info(f"Fetched {len(brands_df)} brands from database")
            
            return brands_df
            
        except Exception as e:
            self.console.print(f"[red]✗ Failed to fetch brands: {e}[/red]")
            self.logger.error(f"Failed to fetch brands: {e}")
            raise
    
    def fetch_brand_products_bulk(self, brand_ids: List[str]) -> pd.DataFrame:
        """
        Fetch products for multiple brands using the updated query.
        
        Args:
            brand_ids: List of brand IDs to fetch
            
        Returns:
            DataFrame with product information
        """
        engine = self.db_manager.get_db_engine()
        if engine is None:
            raise Exception("Failed to establish database connection")
        
        # Use the updated query provided by the user
        brand_ids_str = "', '".join(brand_ids)
        query = f"""
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
            JOIN brands."brandSKU_category" bsc ON bs.id = bsc."brandSKUId"
            JOIN categories.category c ON bsc."categoryId" = c.id
            WHERE bs."isActive" = true
            AND b."isActive" = true
            AND c."isActive" = true
            AND b.id IN ('{brand_ids_str}')
            AND b.id != 'd0f115f8-4127-4081-8948-ee56070a2a0a'
        )
        SELECT DISTINCT("brandSKUId") "brandSKUId", "brandId", "brandLabel", label, facets_jsonb, "categoryLabel"
        FROM cte
        WHERE rn = 1
        ORDER BY "brandId"
        """
        
        try:
            with engine.connect() as connection:
                products_df = pd.read_sql(query, connection)
            
            return products_df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch products for brands: {e}")
            raise
    
    def process_batch(self, batch_num: int, brand_batch: pd.DataFrame) -> Dict[str, Any]:
        """
        Process a single batch of brands.
        
        Args:
            batch_num: Batch number
            brand_batch: DataFrame with brand information for this batch
            
        Returns:
            Batch processing results
        """
        batch_start_time = time.time()
        batch_brands = brand_batch['brandId'].tolist()
        batch_results = {
            'batch_number': batch_num,
            'brands_in_batch': len(batch_brands),
            'brands_processed': 0,
            'brands_failed': 0,
            'total_products': 0,
            'total_clusters': 0,
            'products_clustered': 0,
            'processing_time': 0,
            'start_time': datetime.now().isoformat(),
            'failed_brands': []
        }
        
        # Create batch-specific output file
        batch_output_file = self.output_dir / f"master_sister_products_batch{batch_num}.csv"
        batch_results_df = pd.DataFrame()
        
        self.console.print(f"\n[bold blue]🚀 Processing Batch {batch_num}[/bold blue]")
        self.console.print(f"[cyan]Brands in batch: {len(batch_brands)}[/cyan]")
        
        # Initialize mapper for this batch
        mapper = SisterProductsMapper(**self.mapper_kwargs)
        
        # Process each brand in the batch
        for idx, (_, brand_row) in enumerate(brand_batch.iterrows(), 1):
            brand_id = brand_row['brandId']
            brand_name = brand_row['brandName']
            
            # Skip if already processed
            if brand_id in self.processed_brands:
                self.console.print(f"[yellow]⏭️ Skipping {brand_name} (already processed)[/yellow]")
                continue
            
            try:
                self.console.print(f"\n[green]═══ Brand {idx}/{len(brand_batch)} in Batch {batch_num} ═══[/green]")
                self.console.print(f"[cyan]Processing: {brand_name} ({brand_id})[/cyan]")
                
                # Fetch products for this brand
                brand_products = self.fetch_brand_products_bulk([brand_id])
                
                if brand_products.empty:
                    self.console.print(f"[yellow]⚠️ No products found for {brand_name}[/yellow]")
                    self.processed_brands.add(brand_id)
                    continue
                
                # Process through the mapper
                processed_df, embeddings, core_identities = mapper.process_brand_data(brand_products, brand_name)
                cluster_labels = mapper.perform_clustering(embeddings, brand_name, processed_df)
                results = mapper.generate_output(processed_df, cluster_labels, brand_name)
                
                # Update batch results
                batch_results['brands_processed'] += 1
                batch_results['total_products'] += results['total_products']
                batch_results['total_clusters'] += results['total_clusters']
                batch_results['products_clustered'] += results['products_with_sisters']
                
                # Add to batch CSV
                brand_results_df = processed_df.copy()
                brand_results_df['brand_name'] = brand_name
                brand_results_df['batch_number'] = batch_num
                brand_results_df['processing_timestamp'] = pd.Timestamp.now()
                
                if batch_results_df.empty:
                    batch_results_df = brand_results_df.copy()
                else:
                    batch_results_df = pd.concat([batch_results_df, brand_results_df], ignore_index=True)
                
                # LIVE SAVE: Save CSV after each brand (not just at end of batch)
                batch_results_df.to_csv(batch_output_file, index=False)
                
                # Mark as processed
                self.processed_brands.add(brand_id)
                
                # LIVE CHECKPOINT: Save checkpoint after each brand
                self._save_checkpoint()
                
                # Display progress
                mapper.display_results_summary(results)
                self.console.print(f"[green]✅ {brand_name} completed ({results['total_products']} products, {results['total_clusters']} clusters)[/green]")
                
            except Exception as e:
                batch_results['brands_failed'] += 1
                batch_results['failed_brands'].append({
                    'brand_id': brand_id,
                    'brand_name': brand_name,
                    'error': str(e)
                })
                self.failed_brands.add(brand_id)
                
                self.console.print(f"[red]✗ Failed to process {brand_name}: {e}[/red]")
                self.logger.error(f"Failed to process brand {brand_name} ({brand_id}): {e}")
        
        # Batch results already saved live during processing
        if not batch_results_df.empty:
            self.console.print(f"[green]💾 Batch {batch_num} results live-saved to {batch_output_file}[/green]")
        
        # Calculate batch processing time
        batch_results['processing_time'] = time.time() - batch_start_time
        batch_results['end_time'] = datetime.now().isoformat()
        
        # Save batch statistics
        self.batch_stats[f"batch_{batch_num}"] = batch_results
        
        return batch_results
    
    def display_batch_summary(self, batch_results: Dict[str, Any]):
        """Display batch processing summary."""
        batch_num = batch_results['batch_number']
        
        # Create summary table
        table = Table(title=f"Batch {batch_num} Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Brands in Batch", str(batch_results['brands_in_batch']))
        table.add_row("Brands Processed", str(batch_results['brands_processed']))
        table.add_row("Brands Failed", str(batch_results['brands_failed']))
        table.add_row("Total Products", str(batch_results['total_products']))
        table.add_row("Total Clusters", str(batch_results['total_clusters']))
        table.add_row("Products Clustered", str(batch_results['products_clustered']))
        
        if batch_results['total_products'] > 0:
            clustering_rate = (batch_results['products_clustered'] / batch_results['total_products']) * 100
            table.add_row("Clustering Rate", f"{clustering_rate:.1f}%")
        
        processing_time = batch_results['processing_time']
        table.add_row("Processing Time", f"{processing_time/60:.1f} minutes")
        
        self.console.print(table)
        
        # Show failed brands if any
        if batch_results['failed_brands']:
            self.console.print(f"\n[red]❌ Failed Brands in Batch {batch_num}:[/red]")
            for failed_brand in batch_results['failed_brands']:
                self.console.print(f"  - {failed_brand['brand_name']}: {failed_brand['error']}")
    
    def get_human_approval(self, batch_num: int, total_batches: int) -> bool:
        """
        Get human approval to proceed with the next batch.
        
        Args:
            batch_num: Current batch number
            total_batches: Total number of batches
            
        Returns:
            True if approved, False if rejected
        """
        # Check if auto-approval is enabled
        if self.auto_approve:
            self.console.print(f"\n[bold green]🤖 Auto-Approval Enabled[/bold green]")
            self.console.print(f"[cyan]Auto-processing Batch {batch_num} of {total_batches}[/cyan]")
            
            # Show overall progress
            progress_pct = ((batch_num - 1) / total_batches) * 100
            self.console.print(f"[blue]Overall Progress: {progress_pct:.1f}% ({batch_num-1}/{total_batches} batches completed)[/blue]")
            
            # Show current statistics
            self.console.print(f"[green]Processed so far: {len(self.processed_brands):,} brands, {self.global_stats['total_products']:,} products[/green]")
            
            approved = True
            self.console.print("[green]✅ Auto-approved! Starting batch processing...[/green]")
            self.logger.info(f"Auto-approved processing of batch {batch_num}")
        else:
            self.console.print(f"\n[bold yellow]🤔 Human Approval Required[/bold yellow]")
            self.console.print(f"[cyan]Ready to process Batch {batch_num} of {total_batches}[/cyan]")
            
            # Show overall progress
            progress_pct = ((batch_num - 1) / total_batches) * 100
            self.console.print(f"[blue]Overall Progress: {progress_pct:.1f}% ({batch_num-1}/{total_batches} batches completed)[/blue]")
            
            # Show current statistics
            self.console.print(f"[green]Processed so far: {len(self.processed_brands):,} brands, {self.global_stats['total_products']:,} products[/green]")
            
            # Get approval
            approved = Confirm.ask(f"[bold]Proceed with Batch {batch_num}?[/bold]", default=True)
        
            if not approved:
                self.console.print("[red]❌ Processing halted by user[/red]")
                self.logger.info(f"Processing halted by user at batch {batch_num}")
            else:
                self.console.print("[green]✅ Approved! Starting batch processing...[/green]")
                self.logger.info(f"User approved processing of batch {batch_num}")
        
        return approved
    
    def display_global_summary(self):
        """Display final global processing summary."""
        # Calculate final statistics
        total_processing_time = 0
        if self.global_stats['processing_start_time'] and self.global_stats['processing_end_time']:
            start_time = datetime.fromisoformat(self.global_stats['processing_start_time'])
            end_time = datetime.fromisoformat(self.global_stats['processing_end_time'])
            total_processing_time = (end_time - start_time).total_seconds()
        
        # Create global summary table
        table = Table(title="🎉 Global Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Brands Processed", str(len(self.processed_brands)))
        table.add_row("Total Brands Failed", str(len(self.failed_brands)))
        table.add_row("Total Products", str(self.global_stats['total_products']))
        table.add_row("Total Clusters", str(self.global_stats['total_clusters']))
        table.add_row("Products Clustered", str(self.global_stats['products_clustered']))
        
        if self.global_stats['total_products'] > 0:
            clustering_rate = (self.global_stats['products_clustered'] / self.global_stats['total_products']) * 100
            table.add_row("Overall Clustering Rate", f"{clustering_rate:.1f}%")
        
        table.add_row("Batches Completed", str(self.global_stats['batches_completed']))
        table.add_row("Batches Failed", str(self.global_stats['batches_failed']))
        
        if total_processing_time > 0:
            table.add_row("Total Processing Time", f"{total_processing_time/3600:.1f} hours")
        
        self.console.print(table)
        
        # Show batch breakdown
        if self.batch_stats:
            self.console.print("\n[bold]📊 Batch Breakdown:[/bold]")
            batch_table = Table()
            batch_table.add_column("Batch", style="cyan")
            batch_table.add_column("Brands", style="blue")
            batch_table.add_column("Products", style="green")
            batch_table.add_column("Clusters", style="magenta")
            batch_table.add_column("Time (min)", style="yellow")
            
            for batch_key, batch_data in self.batch_stats.items():
                batch_table.add_row(
                    str(batch_data['batch_number']),
                    str(batch_data['brands_processed']),
                    str(batch_data['total_products']),
                    str(batch_data['total_clusters']),
                    f"{batch_data['processing_time']/60:.1f}"
                )
            
            self.console.print(batch_table)
    
    def run_bulk_processing(self) -> bool:
        """
        Main method to run bulk processing of all brands with live resume functionality.
        
        Returns:
            True if successful, False if failed or interrupted
        """
        try:
            self.console.print(Panel.fit(
                "[bold blue]Sister Products Mapping - Bulk Processing Mode[/bold blue]\n"
                "[cyan]Processing brands with live resume capability[/cyan]\n"
                f"[yellow]Batch Size: {self.batch_size} brands per batch[/yellow]",
                title="Bulk Processor",
                border_style="blue"
            ))
            
            # Check for existing batch files to determine resume point
            last_completed_batch = self._check_existing_batch_files()
            
            # Record start time if not already set
            if not self.global_stats['processing_start_time']:
                self.global_stats['processing_start_time'] = datetime.now().isoformat()
            
            # Fetch all brands
            all_brands = self.get_all_brands()
            total_brands = len(all_brands)
            
            # Get remaining brands information
            remaining_brands, total_brands, remaining_count = self._get_remaining_brands_info(all_brands)
            
            self.console.print(f"[green]📊 Total brands available: {total_brands:,}[/green]")
            self.console.print(f"[yellow]📊 Already processed: {len(self.processed_brands):,}[/yellow]")
            self.console.print(f"[blue]📊 Remaining to process: {remaining_count:,}[/blue]")
            
            if remaining_brands.empty:
                self.console.print("[green]🎉 All brands have been processed![/green]")
                return True
            
            # Calculate number of batches dynamically based on remaining brands
            total_batches = (remaining_count + self.batch_size - 1) // self.batch_size
            self.console.print(f"[cyan]📦 Total batches needed: {total_batches}[/cyan]")
            
            # Determine starting batch number based on last completed batch
            if last_completed_batch > 0:
                self.console.print(f"[blue]🔄 Resuming from batch {last_completed_batch + 1}[/blue]")
                start_batch = last_completed_batch + 1
            else:
                start_batch = 1
            
            # Process each batch
            for batch_num in range(start_batch, start_batch + total_batches):
                # Calculate the correct slice of remaining brands for this batch
                batch_start_idx = (batch_num - start_batch) * self.batch_size
                batch_end_idx = min(batch_start_idx + self.batch_size, len(remaining_brands))
                brand_batch = remaining_brands.iloc[batch_start_idx:batch_end_idx]
                
                # Show batch information
                self.console.print(f"\n[bold]Batch {batch_num} of {start_batch + total_batches - 1}[/bold]")
                self.console.print(f"[cyan]Brands in this batch: {len(brand_batch)}[/cyan]")
                self.console.print(f"[blue]Brands range: {batch_start_idx + 1}-{batch_end_idx} of {remaining_count}[/blue]")
                
                # Get human approval for this batch
                if not self.get_human_approval(batch_num, start_batch + total_batches - 1):
                    self.console.print("[yellow]⏸️ Processing paused by user[/yellow]")
                    return False
                
                try:
                    # Process the batch
                    batch_results = self.process_batch(batch_num, brand_batch)
                    
                    # Update global statistics
                    self.global_stats['total_products'] += batch_results['total_products']
                    self.global_stats['total_clusters'] += batch_results['total_clusters']
                    self.global_stats['products_clustered'] += batch_results['products_clustered']
                    self.global_stats['batches_completed'] += 1
                    
                    # Display batch summary
                    self.display_batch_summary(batch_results)
                    
                    # Checkpoint already saved live during processing
                    self.console.print(f"[green]✅ Batch {batch_num} completed successfully[/green]")
                    
                except Exception as e:
                    self.global_stats['batches_failed'] += 1
                    self.console.print(f"[red]❌ Batch {batch_num} failed: {e}[/red]")
                    self.logger.error(f"Batch {batch_num} failed: {e}")
                    
                    # Ask if user wants to continue
                    if not Confirm.ask("Continue with next batch?", default=True):
                        return False
            
            # Record end time
            self.global_stats['processing_end_time'] = datetime.now().isoformat()
            
            # Display final summary
            self.display_global_summary()
            
            # Save final checkpoint
            self._save_checkpoint()
            
            self.console.print("[bold green]🎉 Bulk processing completed successfully![/bold green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]💥 Bulk processing failed: {e}[/red]")
            self.logger.error(f"Bulk processing failed: {e}")
            return False
    
    def retry_failed_brands(self) -> bool:
        """
        Retry processing for failed brands.
        
        Returns:
            True if successful, False if failed
        """
        if not self.failed_brands:
            self.console.print("[green]✅ No failed brands to retry[/green]")
            return True
        
        self.console.print(f"[yellow]🔄 Retrying {len(self.failed_brands)} failed brands...[/yellow]")
        
        # Get failed brand information
        all_brands = self.get_all_brands()
        failed_brands_df = all_brands[all_brands['brandId'].isin(self.failed_brands)]
        
        # Clear failed brands set for retry
        self.failed_brands.clear()
        
        # Process failed brands as a special batch
        batch_results = self.process_batch(999, failed_brands_df)  # Use batch 999 for retries
        
        # Display results
        self.display_batch_summary(batch_results)
        
        return batch_results['brands_failed'] == 0


def test_bulk_processor():
    """Test the bulk processor functionality."""
    console = Console()
    
    console.print("[bold blue]🧪 Testing Bulk Processor Functionality[/bold blue]")
    
    # Test 1: Initialize processor
    console.print("\n[cyan]Test 1: Initialize Bulk Processor[/cyan]")
    try:
        processor = BulkProcessor(
            batch_size=5,  # Small batch for testing
            model_name='all-MiniLM-L6-v2',
            min_cluster_size=2,
            output_dir='test_output',
            logs_dir='test_logs',
            checkpoint_dir='test_checkpoints'
        )
        console.print("[green]✅ Bulk Processor initialized successfully[/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize: {e}[/red]")
        return False
    
    # Test 2: Database connection
    console.print("\n[cyan]Test 2: Database Connection[/cyan]")
    try:
        engine = processor.db_manager.get_db_engine()
        if engine:
            console.print("[green]✅ Database connection successful[/green]")
        else:
            console.print("[red]❌ Database connection failed[/red]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Database connection error: {e}[/red]")
        return False
    
    # Test 3: Fetch brands
    console.print("\n[cyan]Test 3: Fetch All Brands[/cyan]")
    try:
        brands_df = processor.get_all_brands()
        console.print(f"[green]✅ Fetched {len(brands_df)} brands[/green]")
        if len(brands_df) > 0:
            console.print(f"[blue]Sample brands: {brands_df.head(3)['brandName'].tolist()}[/blue]")
    except Exception as e:
        console.print(f"[red]❌ Failed to fetch brands: {e}[/red]")
        return False
    
    # Test 4: Fetch products for a small batch
    console.print("\n[cyan]Test 4: Fetch Products for Small Batch[/cyan]")
    try:
        test_brands = brands_df.head(2)['brandId'].tolist()
        products_df = processor.fetch_brand_products_bulk(test_brands)
        console.print(f"[green]✅ Fetched {len(products_df)} products for {len(test_brands)} brands[/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to fetch products: {e}[/red]")
        return False
    
    # Test 5: Checkpoint functionality
    console.print("\n[cyan]Test 5: Checkpoint Save/Load[/cyan]")
    try:
        # Add some test data
        processor.processed_brands.add("test-brand-1")
        processor.failed_brands.add("test-brand-2")
        processor.global_stats['test_metric'] = 100
        
        # Save checkpoint
        processor._save_checkpoint()
        
        # Create new processor and load checkpoint
        processor2 = BulkProcessor(
            batch_size=5,
            output_dir='test_output',
            logs_dir='test_logs',
            checkpoint_dir='test_checkpoints'
        )
        
        if ("test-brand-1" in processor2.processed_brands and 
            "test-brand-2" in processor2.failed_brands and
            processor2.global_stats.get('test_metric') == 100):
            console.print("[green]✅ Checkpoint save/load working correctly[/green]")
        else:
            console.print("[red]❌ Checkpoint data not loaded correctly[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ Checkpoint test failed: {e}[/red]")
        return False
    
    console.print("\n[bold green]🎉 All tests passed! Bulk Processor is ready for production.[/bold green]")
    
    # Clean up test files
    console.print("\n[yellow]🧹 Cleaning up test files...[/yellow]")
    try:
        import shutil
        for test_dir in ['test_output', 'test_logs', 'test_checkpoints']:
            if Path(test_dir).exists():
                shutil.rmtree(test_dir)
        console.print("[green]✅ Test files cleaned up[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Could not clean up all test files: {e}[/yellow]")
    
    return True


if __name__ == "__main__":
    # Run tests
    test_bulk_processor() 