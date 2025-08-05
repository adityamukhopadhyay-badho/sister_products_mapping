#!/usr/bin/env python3
"""
Simple test script for database connectivity and sister products mapping.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database_manager import DatabaseManager
from src.sister_products_mapper import SisterProductsMapper
import logging

def test_database_connection():
    """Test basic database connectivity."""
    print("🔬 Testing Database Connection")
    print("=" * 50)
    
    try:
        db_manager = DatabaseManager()
        engine = db_manager.get_db_engine()
        
        if engine:
            print("✓ Database connection successful!")
            return True
        else:
            print("✗ Database connection failed!")
            return False
            
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

def test_data_fetching():
    """Test fetching data from database."""
    print("\n🔬 Testing Data Fetching")
    print("=" * 50)
    
    try:
        db_manager = DatabaseManager()
        
        # Test simple query first
        engine = db_manager.get_db_engine()
        if not engine:
            print("✗ No database connection")
            return False
            
        from sqlalchemy import text
        with engine.connect() as connection:
            # Test a simple query
            result = connection.execute(text("SELECT 1 as test"))
            print("✓ Simple query works")
            
            # Test brands query with different approach
            try:
                brands_result = connection.execute(text("""
                    SELECT b.id as "brandId"
                    FROM brands.brand b
                    WHERE b."isActive" = true 
                    AND b."isBrandBadhoVerified" = true
                    LIMIT 5;
                """))
                
                brand_ids = [row[0] for row in brands_result]
                print(f"✓ Found {len(brand_ids)} brand IDs")
                
                if brand_ids:
                    # Test products query for first brand
                    first_brand = brand_ids[0]
                    print(f"Testing products for brand: {first_brand}")
                    
                    products_result = connection.execute(text("""
                        SELECT bs.id AS "brandSKUId", 
                               bs.label, 
                               bs."facetsV2Processed" AS facets_jsonb
                        FROM brands."brandSKU" bs
                        WHERE bs."brandId" = :brand_id
                        AND bs."isActive" = true
                        AND bs."facetsV2Processed" IS NOT NULL
                        LIMIT 3;
                    """), {"brand_id": first_brand})
                    
                    products = list(products_result)
                    print(f"✓ Found {len(products)} sample products")
                    
                    if products:
                        print("Sample product:")
                        print(f"  ID: {products[0][0]}")
                        print(f"  Label: {products[0][1][:50]}...")
                        print(f"  Has facets: {products[0][2] is not None}")
                
                return True
                
            except Exception as e:
                print(f"✗ Brands query failed: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Data fetching error: {e}")
        return False

def test_sister_products_processing():
    """Test a small sister products processing run."""
    print("\n🔬 Testing Sister Products Processing")
    print("=" * 50)
    
    try:
        # Initialize mapper with database support
        mapper = SisterProductsMapper(
            min_cluster_size=2,
            use_facets=True,
            output_dir='test_output',
            logs_dir='test_logs'
        )
        
        print("✓ Mapper initialized")
        
        # Test database processing with specific brand
        # Get first available brand ID
        db_manager = DatabaseManager()
        engine = db_manager.get_db_engine()
        
        from sqlalchemy import text
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT b.id as "brandId"
                FROM brands.brand b
                WHERE b."isActive" = true 
                AND b."isBrandBadhoVerified" = true
                LIMIT 1;
            """))
            
            brand_ids = [row[0] for row in result]
            
        if not brand_ids:
            print("✗ No brands found for testing")
            return False
            
        test_brand_id = brand_ids[0]
        print(f"Testing with brand ID: {test_brand_id}")
        
        # Process single brand (limited dataset)
        results = mapper.process_from_database(brand_id=test_brand_id)
        
        if results:
            print("✓ Sister products processing completed!")
            for brand_name, brand_results in results.items():
                print(f"  Brand: {brand_name}")
                print(f"  Products: {brand_results.get('total_products', 0)}")
                print(f"  Clusters: {brand_results.get('total_clusters', 0)}")
            return True
        else:
            print("✗ No results generated")
            return False
            
    except Exception as e:
        print(f"✗ Processing error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Sister Products Database Testing Suite")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise for testing
    
    # Run tests
    tests = [
        ("Database Connection", test_database_connection),
        ("Data Fetching", test_data_fetching),
        ("Sister Products Processing", test_sister_products_processing),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Database integration is working.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    main() 