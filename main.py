#!/usr/bin/env python3
"""
Sister Products Mapping - Main Execution Script

This script orchestrates the complete sister products mapping pipeline:
1. Loads brand product data from CSV files
2. Processes each brand through normalization, embedding, and clustering
3. Generates comprehensive visualizations and reports
4. Saves results in multiple formats with live progress tracking

Usage:
    python3 main.py [data_files...]
    
Examples:
    python3 main.py data/cornitos_products.csv data/haldiram_products.csv
    python3 main.py data/*.csv

Author: Sister Products Mapping System
Date: 2024
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint

# Import our modules
from src.sister_products_mapper import SisterProductsMapper
from src.visualizer import SisterProductVisualizer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sister Products Mapping System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py data/cornitos_products.csv data/haldiram_products.csv
  python3 main.py data/*.csv
  python3 main.py --model sentence-transformers/all-MiniLM-L6-v2 --min-cluster-size 3 data/*.csv
        """
    )
    
    parser.add_argument(
        'files',
        nargs='*',
        help='CSV files containing product data (optional when using --from-database)'
    )
    
    parser.add_argument(
        '--model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='FastEmbed model to use (default: sentence-transformers/all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=2,
        help='Minimum cluster size for HDBSCAN (default: 2)'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=1,
        help='Minimum samples for HDBSCAN (default: 1)'
    )
    
    parser.add_argument(
        '--cluster-epsilon',
        type=float,
        default=0.1,
        help='Distance threshold for cluster merging in HDBSCAN. Higher values (0.1-0.3) create larger, more inclusive clusters (default: 0.1)'
    )
    
    parser.add_argument(
        '--enable-phonetic',
        action='store_true',
        help='Enable phonetic similarity encoding for better clustering of similar-sounding words (e.g., burfi/burfee/barfee)'
    )
    
    parser.add_argument(
        '--phonetic-algorithm',
        choices=['soundex', 'metaphone', 'nysiis', 'match_rating_codex'],
        default='metaphone',
        help='Phonetic algorithm to use when --enable-phonetic is set (default: metaphone)'
    )
    
    parser.add_argument(
        '--use-facets',
        action='store_true',
        help='Use facets_jsonb data directly for embeddings instead of normalized product names. Provides richer semantic information for clustering.'
    )
    
    parser.add_argument(
        '--simple-identity',
        action='store_true',
        help='Use a simplified core identity (brand name + category) for embeddings'
    )
    
    parser.add_argument(
        '--fast-clustering',
        action='store_true',
        help='Force use of fast KMeans clustering for all datasets (trades accuracy for speed)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for results (default: output)'
    )
    
    parser.add_argument(
        '--visualizations-dir',
        default='visualizations',
        help='Visualizations directory (default: visualizations)'
    )
    
    parser.add_argument(
        '--logs-dir',
        default='logs',
        help='Logs directory (default: logs)'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        default=True,
        help='Skip generating visualizations (default: enabled)'
    )
    parser.add_argument(
        '--visualizations',
        dest='no_visualizations',
        action='store_false',
        help='Enable visualizations'
    )
    
    parser.add_argument(
        '--from-database',
        action='store_true',
        default=True,
        help='Process products from PostgreSQL database instead of CSV files (default: enabled)'
    )
    parser.add_argument(
        '--from-csv',
        dest='from_database',
        action='store_false',
        help='Process products from CSV files instead of the database'
    )
    
    parser.add_argument(
        '--brand-id',
        type=str,
        help='Specific brand ID to process when using --from-database'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for database processing (default: 1000)'
    )
    
    parser.add_argument(
        '--bulk-run',
        action='store_true',
        default=True,
        help='Enable bulk processing mode for processing brands in batches (default: enabled)'
    )
    parser.add_argument(
        '--no-bulk-run',
        dest='bulk_run',
        action='store_false',
        help='Disable bulk processing mode'
    )
    
    parser.add_argument(
        '--bulk-batch-size',
        type=int,
        default=1000,
        help='Number of brands per batch in bulk processing mode (default: 1000)'
    )
    
    parser.add_argument(
        '--auto-approve',
        action='store_true',
        default=True,
        help='Auto-approve all batches in bulk processing mode (skip human confirmation) (default: enabled)'
    )
    parser.add_argument(
        '--no-auto-approve',
        dest='auto_approve',
        action='store_false',
        help='Disable auto-approval for batches (requires manual confirmation)'
    )
    
    return parser.parse_args()

def validate_files(file_paths: List[str]) -> List[str]:
    """Validate input files exist and are CSV files."""
    if not file_paths:  # Allow empty file list for database mode
        return []
        
    valid_files = []
    console = Console()
    
    for file_path in file_paths:
        path = Path(file_path)
        
        if not path.exists():
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            continue
            
        if not path.suffix.lower() == '.csv':
            console.print(f"[yellow]Warning: Not a CSV file: {file_path}[/yellow]")
            continue
            
        valid_files.append(str(path))
    
    return valid_files

def display_database_summary(args):
    """Display database processing configuration."""
    console = Console()
    
    # Create configuration table
    config_table = Table(title="Database Processing Configuration", show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="magenta")
    
    config_table.add_row("Data Source", "PostgreSQL Database")
    config_table.add_row("Model", args.model)
    config_table.add_row("Min Cluster Size", str(args.min_cluster_size))
    config_table.add_row("Min Samples", str(args.min_samples))
    config_table.add_row("Cluster Epsilon", str(args.cluster_epsilon))
    config_table.add_row("Use Facets", "Yes" if args.use_facets else "No")
    config_table.add_row("Simple Identity", "Yes" if args.simple_identity else "No")
    config_table.add_row("Enable Phonetic", "Yes" if args.enable_phonetic else "No")
    config_table.add_row("Brand ID Filter", args.brand_id if args.brand_id else "All Brands")
    config_table.add_row("Batch Size", str(args.batch_size))
    config_table.add_row("Output Directory", args.output_dir)
    config_table.add_row("Generate Visualizations", "No" if args.no_visualizations else "Yes")
    
    console.print(config_table)

def display_welcome_message():
    """Display welcome message and system information."""
    console = Console()
    
    welcome_text = Text()
    welcome_text.append("Sister Products Mapping System\n", style="bold blue")
    welcome_text.append("Advanced AI-Powered Product Clustering\n\n", style="blue")
    welcome_text.append("This system uses vector embeddings and density-based clustering\n")
    welcome_text.append("to automatically identify sister products across brands.\n\n")
    welcome_text.append("🔍 Phase 1: Normalization & Embedding\n", style="green")
    welcome_text.append("🎯 Phase 2: HDBSCAN Clustering\n", style="green")
    welcome_text.append("📊 Phase 3: Visualization & Output\n", style="green")
    
    console.print(Panel(welcome_text, title="Welcome", border_style="blue"))

def display_processing_summary(file_paths: List[str], args):
    """Display processing configuration summary."""
    console = Console()
    
    table = Table(title="Processing Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Input Files", str(len(file_paths)))
    table.add_row("Model", args.model)
    table.add_row("Min Cluster Size", str(args.min_cluster_size))
    table.add_row("Min Samples", str(args.min_samples))
    table.add_row("Output Directory", args.output_dir)
    table.add_row("Generate Visualizations", "No" if args.no_visualizations else "Yes")
    
    console.print(table)
    
    # Show input files
    files_table = Table(title="Input Files")
    files_table.add_column("File", style="green")
    files_table.add_column("Brand", style="blue")
    
    for file_path in file_paths:
        path = Path(file_path)
        brand_name = path.stem.replace('_products', '').title()
        files_table.add_row(path.name, brand_name)
    
    console.print(files_table)

def display_final_summary(all_results: dict, processing_time: float):
    """Display final processing summary."""
    console = Console()
    
    # Overall statistics
    total_products = sum(results['total_products'] for results in all_results.values())
    total_clusters = sum(results['total_clusters'] for results in all_results.values())
    total_with_sisters = sum(results['products_with_sisters'] for results in all_results.values())
    
    summary_table = Table(title="🎉 Processing Complete - Final Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")
    
    summary_table.add_row("Brands Processed", str(len(all_results)))
    summary_table.add_row("Total Products", str(total_products))
    summary_table.add_row("Total Clusters Found", str(total_clusters))
    summary_table.add_row("Products with Sisters", str(total_with_sisters))
    summary_table.add_row("Overall Clustering Rate", f"{(total_with_sisters/total_products*100):.1f}%" if total_products > 0 else "0%")
    summary_table.add_row("Processing Time", f"{processing_time:.1f} seconds")
    
    console.print(summary_table)
    
    # Per-brand breakdown
    brands_table = Table(title="Per-Brand Results")
    brands_table.add_column("Brand", style="green")
    brands_table.add_column("Products", style="blue")
    brands_table.add_column("Clusters", style="cyan")
    brands_table.add_column("With Sisters", style="magenta")
    brands_table.add_column("Rate", style="yellow")
    
    for brand, results in all_results.items():
        rate = results['products_with_sisters'] / results['total_products'] * 100
        brands_table.add_row(
            brand,
            str(results['total_products']),
            str(results['total_clusters']),
            str(results['products_with_sisters']),
            f"{rate:.1f}%"
        )
    
    console.print(brands_table)

def main():
    """Main execution function."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize console
    console = Console()
    
    # Display welcome message
    display_welcome_message()
    
    # Validate approach: either files or database
    if args.from_database:
        if args.files:
            console.print("[yellow]⚠️ Ignoring file arguments when using --from-database[/yellow]")
        valid_files = []
    else:
        valid_files = validate_files(args.files)
        if not valid_files:
            console.print("[red]❌ No valid CSV files found. Use --from-database for database processing.[/red]")
            sys.exit(1)
    
    # Display processing summary
    if not args.from_database:
        display_processing_summary(valid_files, args)
    else:
        display_database_summary(args)
    
    # Initialize components
    try:
        console.print("\n[blue]🚀 Initializing Sister Products Mapper...[/blue]")
        
        mapper = SisterProductsMapper(
            model_name=args.model,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_epsilon,
            enable_phonetic=args.enable_phonetic,
            phonetic_algorithm=args.phonetic_algorithm,
            use_facets=args.use_facets,
            simple_identity=args.simple_identity,
            fast_clustering=args.fast_clustering,
            output_dir=args.output_dir,
            logs_dir=args.logs_dir
        )
        
        if not args.no_visualizations:
            visualizer = SisterProductVisualizer(
                visualizations_dir=args.visualizations_dir
            )
        
        console.print("[green]✓ Initialization complete[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Initialization failed: {e}[/red]")
        sys.exit(1)
    
    # Process brands (either from files or database)
    all_results = {}
    
    try:
        if args.bulk_run:
            # Bulk processing mode for 53K+ brands
            from src.bulk_processor import BulkProcessor
            
            console.print(f"\n[bold red]🚀 BULK PROCESSING MODE ACTIVATED[/bold red]")
            console.print(f"[yellow]⚠️ This will process 1.4M+ products across 53K+ brands in batches of {args.bulk_batch_size}![/yellow]")
            
            # Initialize bulk processor
            bulk_processor = BulkProcessor(
                batch_size=args.bulk_batch_size,
                auto_approve=args.auto_approve,
                model_name=args.model,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                cluster_selection_epsilon=args.cluster_epsilon,
                enable_phonetic=args.enable_phonetic,
                phonetic_algorithm=args.phonetic_algorithm,
                use_facets=args.use_facets,
                simple_identity=args.simple_identity,
                fast_clustering=args.fast_clustering,
                output_dir=args.output_dir,
                logs_dir=args.logs_dir
            )
            
            # Run bulk processing
            success = bulk_processor.run_bulk_processing()
            
            if success:
                console.print("[bold green]🎉 Bulk processing completed successfully![/bold green]")
            else:
                console.print("[yellow]⏸️ Bulk processing paused/interrupted[/yellow]")
                
        elif args.from_database:
            # Database processing mode
            console.print(f"\n[bold blue]🗄️ Processing from database...[/bold blue]")
            all_results = mapper.process_from_database(
                brand_id=args.brand_id,
                batch_size=args.batch_size
            )
            
            # Generate visualizations for database results
            if not args.no_visualizations and all_results:
                for brand_name, results in all_results.items():
                    try:
                        visualizer.generate_all_visualizations(results, brand_name)
                        console.print(f"[green]📊 Visualizations generated for {brand_name}[/green]")
                    except Exception as e:
                        console.print(f"[yellow]⚠️ Visualization failed for {brand_name}: {e}[/yellow]")
        
        else:
            # File processing mode (existing logic)
            console.print(f"\n[bold blue]📊 Processing {len(valid_files)} brand(s)...[/bold blue]")
            
            for i, file_path in enumerate(valid_files, 1):
                console.print(f"\n[bold]═══ Brand {i}/{len(valid_files)} ═══[/bold]")
                
                try:
                    # Process brand
                    results = mapper.process_brand_file(file_path)
                    brand_name = Path(file_path).stem.replace('_products', '').title()
                    all_results[brand_name] = results
                    
                    # Generate visualizations
                    if not args.no_visualizations:
                        visualizer.generate_all_visualizations(results, brand_name)
                    
                    console.print(f"[green]✅ {brand_name} processing complete[/green]")
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed to process {file_path}: {e}[/red]")
                    continue
        
        # Skip comparison visualizations and final summary for bulk mode
        if not args.bulk_run:
            # Generate comparison visualizations for multiple brands
            if not args.no_visualizations and len(all_results) > 1:
                console.print("\n[blue]📈 Generating multi-brand comparison...[/blue]")
                visualizer.generate_comparison_dashboard(all_results)
            
            # Display final summary
            processing_time = time.time() - start_time
            display_final_summary(all_results, processing_time)
            
            # Display output information
            console.print(f"\n[bold green]🎯 Results saved to:[/bold green]")
            console.print(f"  📁 Main results: {args.output_dir}/")
            if not args.no_visualizations:
                console.print(f"  📊 Visualizations: {args.visualizations_dir}/")
            console.print(f"  📋 Logs: {args.logs_dir}/")
            
            console.print("\n[bold blue]✨ Sister Products Mapping Complete![/bold blue]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Processing interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]💥 Fatal error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main() 