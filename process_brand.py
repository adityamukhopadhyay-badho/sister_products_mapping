#!/usr/bin/env python3
"""
Dynamic script to process any brand ID from the database.
Usage: python process_brand.py <brand_id>
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import with absolute import
from src.sister_products_mapper import SisterProductsMapper

def process_brand(brand_id):
    """Process a specific brand from the database."""
    
    print(f"Processing brand ID: {brand_id}")
    
    # Initialize mapper
    mapper = SisterProductsMapper(
        min_cluster_size=2,
        min_samples=1,
        output_dir='output',
        logs_dir='logs'
    )
    
    print("Running clustering with 0.5 similarity threshold from database...")
    
    try:
        # Process the specific brand from database
        results = mapper.process_from_database(brand_id=brand_id, batch_size=1000)
        
        if results:
            print(f"\n✅ Database clustering complete! Results saved to 'output/' directory")
            return True
        else:
            print(f"\n⚠️ No results found for brand ID: {brand_id}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error processing brand from database: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_brand.py <brand_id>")
        print("Example: python process_brand.py 2768f87a-9ec7-4dfc-8dfe-982538dc5547")
        sys.exit(1)
    
    brand_id = sys.argv[1]
    process_brand(brand_id)
