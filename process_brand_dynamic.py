#!/usr/bin/env python3
"""
Dynamic script to process any brand ID from the database with configurable parameters.
Usage: python process_brand_dynamic.py <brand_id> [config_file]
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import with absolute import
from src.sister_products_mapper_dynamic import DynamicSisterProductsMapper
from config import ClusteringConfig

def process_brand_dynamic(brand_id, config_file=None):
    """Process a specific brand from the database with dynamic configuration."""
    
    print(f"Processing brand ID: {brand_id}")
    
    # Load configuration
    config = ClusteringConfig(config_file) if config_file else ClusteringConfig()
    
    # Initialize mapper with dynamic configuration
    mapper = DynamicSisterProductsMapper(
        config=config,
        output_dir='output',
        logs_dir='logs'
    )
    
    print("Running dynamic clustering with configurable parameters...")
    print(f"Similarity threshold: {config.similarity_threshold}")
    print(f"Core term boost: {config.core_term_boost}")
    print(f"Min word overlap similarity: {config.min_word_overlap_similarity}")
    
    try:
        # Process the specific brand from database
        results = mapper.process_from_database(brand_id=brand_id, batch_size=1000)
        
        if results:
            print(f"\n✅ Dynamic clustering complete! Results saved to 'output/' directory")
            return True
        else:
            print(f"\n⚠️ No results found for brand ID: {brand_id}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error processing brand from database: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_brand_dynamic.py <brand_id> [config_file]")
        print("Example: python process_brand_dynamic.py 2768f87a-9ec7-4dfc-8dfe-982538dc5547")
        print("Example: python process_brand_dynamic.py 2768f87a-9ec7-4dfc-8dfe-982538dc5547 custom_config.json")
        sys.exit(1)
    
    brand_id = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    process_brand_dynamic(brand_id, config_file)
