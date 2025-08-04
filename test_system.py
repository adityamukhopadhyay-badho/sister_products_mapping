#!/usr/bin/env python3
"""
Simple test script to verify the Sister Products Mapping System works correctly.

This script runs a quick test on the provided sample data to ensure all components
are working properly.
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / 'src'))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
import time

def test_imports():
    """Test that all required modules can be imported."""
    console = Console()
    console.print("[blue]Testing imports...[/blue]")
    
    try:
        import pandas as pd
        import numpy as np
        import sentence_transformers
        import hdbscan
        import plotly
        import matplotlib
        import seaborn
        import rich
        import networkx
        import umap
        console.print("[green]✓ All dependencies imported successfully[/green]")
        return True
    except ImportError as e:
        console.print(f"[red]✗ Import error: {e}[/red]")
        return False

def test_data_files():
    """Test that sample data files exist and are readable."""
    console = Console()
    console.print("[blue]Testing data files...[/blue]")
    
    data_dir = Path('data')
    if not data_dir.exists():
        console.print("[red]✗ Data directory not found[/red]")
        return False
    
    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        console.print("[red]✗ No CSV files found in data directory[/red]")
        return False
    
    for csv_file in csv_files:
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            required_columns = ['brandSKUId', 'label', 'facets_jsonb', 'categoryLabel']
            
            if not all(col in df.columns for col in required_columns):
                console.print(f"[red]✗ {csv_file.name} missing required columns[/red]")
                return False
                
            console.print(f"[green]✓ {csv_file.name} ({len(df)} products)[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Error reading {csv_file.name}: {e}[/red]")
            return False
    
    return True

def test_core_functionality():
    """Test the core mapping functionality on a small sample."""
    console = Console()
    console.print("[blue]Testing core functionality...[/blue]")
    
    try:
        from sister_products_mapper import SisterProductsMapper
        
        # Initialize mapper
        mapper = SisterProductsMapper(
            model_name='all-MiniLM-L6-v2',
            min_cluster_size=2,
            min_samples=1,
            output_dir='test_output',
            logs_dir='test_logs'
        )
        
        # Test normalization
        test_facets = {
            'flavour': 'Sweet Chili',
            'netWeight': '22g',
            'brandName': 'TestBrand'
        }
        
        normalized = mapper.normalize_product_name(
            "TestBrand Sweet Chili Chips 22g", 
            test_facets
        )
        
        if len(normalized) < 5:
            console.print(f"[red]✗ Normalization failed: '{normalized}'[/red]")
            return False
        
        console.print(f"[green]✓ Normalization works: 'TestBrand Sweet Chili Chips 22g' → '{normalized}'[/green]")
        
        # Test core identity creation
        core_identity = mapper.create_core_identity(normalized, "Snacks", "TestBrand")
        if not core_identity:
            console.print("[red]✗ Core identity creation failed[/red]")
            return False
            
        console.print(f"[green]✓ Core identity created: '{core_identity}'[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Core functionality test failed: {e}[/red]")
        return False

def run_quick_test():
    """Run a quick end-to-end test."""
    console = Console()
    console.print("[blue]Running quick end-to-end test...[/blue]")
    
    try:
        # Find a small CSV file to test with
        data_dir = Path('data')
        csv_files = list(data_dir.glob('*.csv'))
        
        if not csv_files:
            console.print("[red]✗ No CSV files found for testing[/red]")
            return False
        
        test_file = csv_files[0]  # Use first available file
        
        # Import main functionality
        from sister_products_mapper import SisterProductsMapper
        
        # Initialize with conservative settings for quick test
        mapper = SisterProductsMapper(
            model_name='all-MiniLM-L6-v2',
            min_cluster_size=2,
            min_samples=1,
            output_dir='test_output',
            logs_dir='test_logs'
        )
        
        # Load model (this is usually the slowest part)
        console.print("[yellow]Loading sentence transformer model (this may take a moment)...[/yellow]")
        mapper.load_model()
        
        # Read a small sample of data
        import pandas as pd
        df = pd.read_csv(test_file)
        
        # Test with just first 10 products for speed
        df_sample = df.head(10)
        brand_name = test_file.stem.replace('_products', '').title()
        
        console.print(f"[blue]Processing {len(df_sample)} sample products from {brand_name}...[/blue]")
        
        # Process sample
        processed_df, embeddings, core_identities = mapper.process_brand_data(df_sample, brand_name)
        
        if len(embeddings) != len(df_sample):
            console.print("[red]✗ Embedding generation failed[/red]")
            return False
            
        console.print(f"[green]✓ Generated {len(embeddings)} embeddings[/green]")
        
        # Test clustering
        cluster_labels = mapper.perform_clustering(embeddings, brand_name)
        
        if len(cluster_labels) != len(df_sample):
            console.print("[red]✗ Clustering failed[/red]")
            return False
            
        console.print(f"[green]✓ Clustering complete[/green]")
        
        # Test output generation
        results = mapper.generate_output(processed_df, cluster_labels, brand_name)
        
        if not results or 'brand' not in results:
            console.print("[red]✗ Output generation failed[/red]")
            return False
            
        console.print(f"[green]✓ Generated results for {results['total_products']} products[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Quick test failed: {e}[/red]")
        return False

def main():
    """Run all tests."""
    console = Console()
    
    # Display welcome message
    welcome_text = """
Sister Products Mapping System - Test Suite

This script performs basic verification tests to ensure
the system is properly installed and configured.
"""
    
    console.print(Panel(welcome_text.strip(), title="Test Suite", border_style="blue"))
    
    # Run tests
    tests = [
        ("Import Dependencies", test_imports),
        ("Data Files", test_data_files),
        ("Core Functionality", test_core_functionality),
        ("End-to-End Test", run_quick_test)
    ]
    
    results = []
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Running tests...", total=len(tests))
        
        for test_name, test_func in tests:
            console.print(f"\n[bold]Running: {test_name}[/bold]")
            
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            results.append((test_name, result, duration))
            progress.update(task, advance=1)
            
            if result:
                console.print(f"[green]✅ {test_name} passed ({duration:.1f}s)[/green]")
            else:
                console.print(f"[red]❌ {test_name} failed ({duration:.1f}s)[/red]")
    
    # Summary
    console.print("\n" + "="*50)
    console.print("[bold]Test Results Summary[/bold]")
    console.print("="*50)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, duration in results:
        status = "✅ PASS" if result else "❌ FAIL"
        console.print(f"{status} {test_name:<20} ({duration:.1f}s)")
    
    console.print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        console.print("[bold green]🎉 All tests passed! System is ready to use.[/bold green]")
        console.print("\n[blue]To run the full system:[/blue]")
        console.print("[cyan]python3 main.py data/*.csv[/cyan]")
        return 0
    else:
        console.print("[bold red]❌ Some tests failed. Please check the issues above.[/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 