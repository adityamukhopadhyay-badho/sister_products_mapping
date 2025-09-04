#!/usr/bin/env python3
"""
Post-Processing Pipeline for Sister Products Mapping

This script combines all batch CSV files into a master CSV file with the following fields:
1. brandId
2. clusterId  
3. brandSKUId
4. sisterProductId (one brandSKUId per cluster for each brandId)

Author: Sister Products Mapping System
Date: 2024
"""

import pandas as pd
import os
import glob
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

class PostProcessingPipeline:
    """
    Post-processing pipeline to combine batch CSV files into a master CSV.
    """
    
    def __init__(self, output_dir: str = 'output'):
        """
        Initialize the post-processing pipeline.
        
        Args:
            output_dir: Directory containing batch CSV files
        """
        self.output_dir = Path(output_dir)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the post-processing pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/post_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def find_batch_files(self) -> List[Path]:
        """
        Find all batch CSV files in the output directory.
        
        Returns:
            List of batch file paths
        """
        pattern = self.output_dir / "master_sister_products_batch*.csv"
        batch_files = list(Path(self.output_dir).glob("master_sister_products_batch*.csv"))
        
        # Sort by batch number
        batch_files.sort(key=lambda x: int(x.stem.split('batch')[1]))
        
        self.logger.info(f"Found {len(batch_files)} batch files: {[f.name for f in batch_files]}")
        return batch_files
    
    def load_batch_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single batch CSV file.
        
        Args:
            file_path: Path to the batch CSV file
            
        Returns:
            DataFrame with batch data
        """
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} rows from {file_path.name}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load {file_path.name}: {e}")
            return pd.DataFrame()
    
    def process_batch_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process batch data to extract required fields and create sisterProductId.
        
        Args:
            df: DataFrame with batch data
            
        Returns:
            Processed DataFrame with required fields
        """
        if df.empty:
            return df
            
        # Select required columns
        required_columns = ['brandId', 'cluster_id', 'brandSKUId']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"Missing columns in batch: {missing_columns}")
            return pd.DataFrame()
        
        # Create processed DataFrame with required fields
        processed_df = df[required_columns].copy()
        
        # Rename cluster_id to clusterId for consistency
        processed_df = processed_df.rename(columns={'cluster_id': 'clusterId'})
        
        # Filter out rows with invalid cluster IDs (like -1)
        processed_df = processed_df[processed_df['clusterId'] >= 0]
        
        # Create sisterProductId - use the first brandSKUId for each brandId+clusterId combination
        processed_df['sisterProductId'] = processed_df.groupby(['brandId', 'clusterId'])['brandSKUId'].transform('first')
        
        self.logger.info(f"Processed {len(processed_df)} valid rows")
        return processed_df
    
    def combine_all_batches(self) -> pd.DataFrame:
        """
        Combine all batch files into a single DataFrame.
        
        Returns:
            Combined DataFrame with all batch data
        """
        batch_files = self.find_batch_files()
        
        if not batch_files:
            self.logger.error("No batch files found!")
            return pd.DataFrame()
        
        all_data = []
        
        for batch_file in batch_files:
            self.logger.info(f"Processing {batch_file.name}...")
            
            # Load batch data
            batch_df = self.load_batch_file(batch_file)
            
            if not batch_df.empty:
                # Process batch data
                processed_df = self.process_batch_data(batch_df)
                
                if not processed_df.empty:
                    all_data.append(processed_df)
        
        if not all_data:
            self.logger.error("No valid data found in any batch files!")
            return pd.DataFrame()
        
        # Combine all processed data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        self.logger.info(f"Combined {len(combined_df)} total rows from {len(all_data)} batches")
        return combined_df
    
    def create_master_csv(self, output_filename: str = 'master_sister_products.csv') -> bool:
        """
        Create the master CSV file with all batch data.
        
        Args:
            output_filename: Name of the output master CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Starting post-processing pipeline...")
            
            # Combine all batches
            master_df = self.combine_all_batches()
            
            if master_df.empty:
                self.logger.error("No data to save!")
                return False
            
            # Save to master CSV file
            output_path = self.output_dir / output_filename
            master_df.to_csv(output_path, index=False)
            
            self.logger.info(f"Master CSV created successfully: {output_path}")
            self.logger.info(f"Total rows: {len(master_df):,}")
            self.logger.info(f"Unique brands: {master_df['brandId'].nunique():,}")
            self.logger.info(f"Unique clusters: {master_df['clusterId'].nunique():,}")
            self.logger.info(f"Unique products: {master_df['brandSKUId'].nunique():,}")
            
            # Show sample of the data
            self.logger.info("\nSample of master CSV data:")
            sample_df = master_df.head(10)
            for _, row in sample_df.iterrows():
                self.logger.info(f"  {row['brandId']} | Cluster {row['clusterId']} | {row['brandSKUId']} | Sister: {row['sisterProductId']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create master CSV: {e}")
            return False
    
    def validate_sister_product_ids(self, df: pd.DataFrame) -> bool:
        """
        Validate that sisterProductId is consistent for each brandId+clusterId combination.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Group by brandId and clusterId and check that sisterProductId is consistent
            validation_groups = df.groupby(['brandId', 'clusterId'])['sisterProductId'].nunique()
            
            # All groups should have exactly 1 unique sisterProductId
            invalid_groups = validation_groups[validation_groups > 1]
            
            if len(invalid_groups) > 0:
                self.logger.error(f"Found {len(invalid_groups)} groups with inconsistent sisterProductId:")
                for (brand_id, cluster_id), count in invalid_groups.items():
                    self.logger.error(f"  Brand {brand_id}, Cluster {cluster_id}: {count} different sisterProductIds")
                return False
            
            self.logger.info("✅ All sisterProductId assignments are consistent!")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False

def main():
    """Main function to run the post-processing pipeline."""
    print("🚀 Starting Post-Processing Pipeline")
    print("=" * 50)
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipeline = PostProcessingPipeline()
    
    # Create master CSV
    success = pipeline.create_master_csv()
    
    if success:
        print("\n✅ Post-processing completed successfully!")
        print("📁 Master CSV file created in output/ directory")
    else:
        print("\n❌ Post-processing failed!")
        return False
    
    # Load and validate the created file
    try:
        master_df = pd.read_csv('output/master_sister_products.csv')
        pipeline.validate_sister_product_ids(master_df)
        
        print(f"\n📊 Final Statistics:")
        print(f"   Total rows: {len(master_df):,}")
        print(f"   Unique brands: {master_df['brandId'].nunique():,}")
        print(f"   Unique clusters: {master_df['clusterId'].nunique():,}")
        print(f"   Unique products: {master_df['brandSKUId'].nunique():,}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 