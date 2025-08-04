#!/usr/bin/env python3
"""
Test Script: Phonetic Similarity for Sister Products

This script demonstrates how phonetic encoding helps cluster similar-sounding
product names that have different spellings.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from sister_products_mapper import SisterProductsMapper
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import jellyfish

def test_phonetic_examples():
    """Test phonetic encoding on example similar-sounding words."""
    console = Console()
    
    console.print(Panel.fit(
        "🔊 [bold blue]Phonetic Similarity Testing[/bold blue]\n\n"
        "Testing how different phonetic algorithms handle similar-sounding words",
        title="Phonetic Test",
        border_style="blue"
    ))
    
    # Test cases for Indian food products
    test_cases = [
        ("burfi", "burfee", "barfee"),
        ("laddu", "ladoo", "laddoo"),
        ("rasgulla", "rasgulla", "rasogolla"),
        ("gulab jamun", "gulaab jamun", "gulab jaman"),
        ("samosa", "samossa", "samoosa"),
        ("kachori", "kachouri", "kachauri"),
        ("paneer", "panir", "paneeer"),
        ("tikka masala", "tika masala", "tikka massala"),
        ("biryani", "biriyani", "biryaani"),
        ("chole bhature", "chhole bhature", "chole bature")
    ]
    
    algorithms = ['soundex', 'metaphone', 'nysiis']
    
    for algorithm in algorithms:
        console.print(f"\n[bold yellow]Testing Algorithm: {algorithm.upper()}[/bold yellow]")
        
        table = Table(title=f"Phonetic Encoding - {algorithm}")
        table.add_column("Original", style="cyan")
        table.add_column("Variant 1", style="green")
        table.add_column("Variant 2", style="green")
        table.add_column("Code 1", style="magenta")
        table.add_column("Code 2", style="magenta")
        table.add_column("Code 3", style="magenta")
        table.add_column("Match?", style="yellow")
        
        for original, variant1, variant2 in test_cases:
            if algorithm == 'soundex':
                code1 = jellyfish.soundex(original)
                code2 = jellyfish.soundex(variant1) 
                code3 = jellyfish.soundex(variant2)
            elif algorithm == 'metaphone':
                code1 = jellyfish.metaphone(original)
                code2 = jellyfish.metaphone(variant1)
                code3 = jellyfish.metaphone(variant2)
            else:  # nysiis
                code1 = jellyfish.nysiis(original)
                code2 = jellyfish.nysiis(variant1)
                code3 = jellyfish.nysiis(variant2)
            
            # Check if codes match
            match = "✓" if code1 == code2 == code3 else "✗"
            if code1 == code2 or code1 == code3 or code2 == code3:
                match = "~"  # Partial match
                
            table.add_row(original, variant1, variant2, code1, code2, code3, match)
        
        console.print(table)

def test_with_real_data():
    """Test phonetic clustering with actual product data."""
    console = Console()
    
    console.print(Panel.fit(
        "🧪 [bold green]Real Data Test[/bold green]\n\n"
        "Testing phonetic clustering on Haldiram dataset which likely contains\n"
        "products with similar-sounding names in different spellings",
        title="Real Data Test",
        border_style="green"
    ))
    
    # Test with and without phonetic encoding
    console.print("\n[yellow]Running clustering WITHOUT phonetic encoding...[/yellow]")
    
    mapper_normal = SisterProductsMapper(
        enable_phonetic=False,
        min_cluster_size=2
    )
    
    try:
        results_normal = mapper_normal.process_brand("data/haldiram_products.csv")
        clusters_normal = results_normal['total_clusters']
        with_sisters_normal = results_normal['products_with_sisters']
        
        console.print(f"[cyan]Normal clustering: {clusters_normal} clusters, {with_sisters_normal} products with sisters[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error in normal clustering: {e}[/red]")
        clusters_normal = "Error"
        with_sisters_normal = "Error"
    
    console.print("\n[yellow]Running clustering WITH phonetic encoding...[/yellow]")
    
    mapper_phonetic = SisterProductsMapper(
        enable_phonetic=True,
        phonetic_algorithm='metaphone',
        min_cluster_size=2
    )
    
    try:
        results_phonetic = mapper_phonetic.process_brand("data/haldiram_products.csv")
        clusters_phonetic = results_phonetic['total_clusters']
        with_sisters_phonetic = results_phonetic['products_with_sisters']
        
        console.print(f"[cyan]Phonetic clustering: {clusters_phonetic} clusters, {with_sisters_phonetic} products with sisters[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error in phonetic clustering: {e}[/red]")
        clusters_phonetic = "Error"
        with_sisters_phonetic = "Error"
    
    # Compare results
    console.print("\n[bold]📊 Comparison[/bold]")
    
    comparison_table = Table(title="Phonetic vs Normal Clustering")
    comparison_table.add_column("Method", style="bold")
    comparison_table.add_column("Total Clusters", style="magenta")
    comparison_table.add_column("Products with Sisters", style="green")
    comparison_table.add_column("Impact", style="yellow")
    
    if isinstance(clusters_normal, int) and isinstance(clusters_phonetic, int):
        cluster_diff = clusters_phonetic - clusters_normal
        sister_diff = with_sisters_phonetic - with_sisters_normal
        
        if cluster_diff < 0 and sister_diff > 0:
            impact = f"Better grouping (-{abs(cluster_diff)} clusters, +{sister_diff} products)"
        elif cluster_diff < 0:
            impact = f"Fewer clusters (-{abs(cluster_diff)})"
        elif sister_diff > 0:
            impact = f"More products grouped (+{sister_diff})"
        else:
            impact = "Similar results"
    else:
        impact = "Cannot compare"
    
    comparison_table.add_row("Normal", str(clusters_normal), str(with_sisters_normal), "Baseline")
    comparison_table.add_row("Phonetic", str(clusters_phonetic), str(with_sisters_phonetic), impact)
    
    console.print(comparison_table)

def main():
    console = Console()
    
    console.print("[bold blue]Phonetic Similarity Testing for Sister Products[/bold blue]\n")
    
    # Test 1: Phonetic algorithm comparison
    test_phonetic_examples()
    
    # Test 2: Real data comparison
    test_with_real_data()
    
    console.print(Panel.fit(
        "[bold green]💡 Key Takeaways:[/bold green]\n\n"
        "• [bold]Metaphone[/bold] typically provides the best phonetic matching\n"
        "• [bold]Phonetic encoding[/bold] helps cluster products with spelling variations\n"
        "• [bold]Ideal for Indian food products[/bold] with transliteration variations\n\n"
        "[bold]Usage:[/bold]\n"
        "[code]python3 main.py --enable-phonetic data/*.csv[/code]\n"
        "[code]python3 main.py --enable-phonetic --phonetic-algorithm soundex data/*.csv[/code]",
        title="Summary",
        border_style="green"
    ))

if __name__ == "__main__":
    main() 