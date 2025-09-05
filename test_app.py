#!/usr/bin/env python3
"""
Test script for the Sister Products Mapping API
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported."""
    try:
        print("Testing imports...")
        
        # Test FastAPI imports
        from fastapi import FastAPI
        print("✓ FastAPI imported successfully")
        
        # Test our modules
        from src.sister_products_mapper import SisterProductsMapper
        print("✓ SisterProductsMapper imported successfully")
        
        from src.database_manager import DatabaseManager
        print("✓ DatabaseManager imported successfully")
        
        from post_process_pipeline import PostProcessingPipeline
        print("✓ PostProcessingPipeline imported successfully")
        
        # Test app import
        import app
        print("✓ App module imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_database_connection():
    """Test database connection."""
    try:
        print("\nTesting database connection...")
        from src.database_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        engine = db_manager.get_db_engine()
        
        if engine is not None:
            print("✅ Database connection successful!")
            return True
        else:
            print("❌ Database connection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def test_app_creation():
    """Test FastAPI app creation."""
    try:
        print("\nTesting FastAPI app creation...")
        from app import app
        
        if app is not None:
            print("✅ FastAPI app created successfully!")
            print(f"   - Title: {app.title}")
            print(f"   - Version: {app.version}")
            return True
        else:
            print("❌ FastAPI app creation failed!")
            return False
            
    except Exception as e:
        print(f"❌ App creation error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Sister Products Mapping API")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_imports,
        test_database_connection,
        test_app_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        print("\nTo start the application:")
        print("  python app.py")
        print("  or")
        print("  uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
