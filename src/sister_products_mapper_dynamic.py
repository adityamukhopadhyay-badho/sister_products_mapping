#!/usr/bin/env python3
"""
Dynamic Sister Products Mapping System

This module implements a fully dynamic system for mapping sister products using
configurable parameters that work across any brand and product category.

Author: Sister Products Mapping System
Date: 2024
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding
import hdbscan
import umap
import jellyfish
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Import database manager
try:
    from .database_manager import DatabaseManager
except ImportError:
    # Fallback for when running as standalone script
    from database_manager import DatabaseManager

# Import configuration
try:
    from ..config import ClusteringConfig
except ImportError:
    from config import ClusteringConfig

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DynamicSisterProductsMapper:
    """
    Dynamic Sister Products Mapper that works with any brand and product category.
    
    This class uses configurable parameters to adapt to different product types
    and categories without hardcoded values.
    """
    
    def __init__(self, 
                 config: ClusteringConfig = None,
                 output_dir: str = 'output',
                 logs_dir: str = 'logs'):
        """
        Initialize the Dynamic Sister Products Mapper.
        
        Args:
            config: ClusteringConfig instance with dynamic parameters
            output_dir: Directory for output files
            logs_dir: Directory for log files
        """
        self.config = config or ClusteringConfig()
        self.output_dir = Path(output_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize console and logging
        self.console = Console()
        self._setup_logging()
        
        # Initialize components
        self.model = None
        self.clusterer = None
        
        # Initialize database manager for live data processing
        self.db_manager = DatabaseManager()
        
        # Master results storage for all brands
        self.master_results_df = pd.DataFrame()
        
    def _setup_logging(self):
        """Setup rich logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(console=self.console, rich_tracebacks=True),
                logging.FileHandler(self.logs_dir / 'sister_products.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_base_product_name(self, product_name: str, brand_name: str = "", category: str = "") -> str:
        """
        Extract the base product name by removing flavor/variant words and keeping common core terms.
        Now fully dynamic based on configuration.
        
        Args:
            product_name: Product name to extract base name from
            brand_name: Brand name to remove
            category: Product category for context
            
        Returns:
            Base product name with common terms preserved
        """
        # Convert to lowercase for processing
        name = product_name.lower().strip()
        
        # Remove weight/quantity information first using configurable patterns
        for pattern in self.config.variant_patterns['weight'] + self.config.variant_patterns['volume']:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        # Remove common packaging terms
        for term in self.config.packaging_terms:
            name = re.sub(rf'\b{term}\b', '', name)
        
        # Detect category and get appropriate flavor indicators
        detected_category = self.config.detect_category(category) if category else 'food_snacks'
        flavor_indicators = self.config.get_flavor_indicators(detected_category)
        
        # Remove flavor indicators
        for flavor in flavor_indicators:
            name = re.sub(rf'\b{re.escape(flavor)}\b', '', name)
        
        # Remove common adjectives
        adjectives = ['premium', 'deluxe', 'special', 'original', 'classic', 'new', 'improved']
        for adj in adjectives:
            name = re.sub(rf'\b{adj}\b', '', name)
        
        # Remove brand name using configurable patterns
        for pattern in self.config.brand_removal_patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        # Add specific brand name if provided
        if brand_name:
            self.config.add_brand_pattern(brand_name)
            name = re.sub(rf'^{re.escape(brand_name.lower())}\s+', '', name)
        
        # Clean up extra spaces and special characters
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name

    def calculate_base_name_similarity(self, base1: str, base2: str, original1: str, original2: str, category: str = "") -> float:
        """
        Calculate similarity between two base product names using dynamic configuration.
        
        Args:
            base1: First base product name
            base2: Second base product name
            original1: Original product name 1 (not used in this simplified version)
            original2: Original product name 2 (not used in this simplified version)
            category: Product category for context
            
        Returns:
            Similarity score between 0 and 1
        """
        if not base1 or not base2:
            return 0.0
        
        # Split into words
        words1 = set(base1.split())
        words2 = set(base2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        
        # Get appropriate core terms for the category
        detected_category = self.config.detect_category(category) if category else 'food_snacks'
        core_terms = self.config.get_core_terms(detected_category)
        
        # Boost score if there are common core terms
        common_core_terms = intersection.intersection(set(core_terms))
        if common_core_terms:
            jaccard_sim += self.config.core_term_boost
        
        # More lenient: if there's any word overlap, give some similarity
        if len(intersection) > 0:
            jaccard_sim = max(jaccard_sim, self.config.min_word_overlap_similarity)
        
        return min(jaccard_sim, 1.0)

    def extract_numerical_data(self, product_name: str) -> List[float]:
        """
        Extract numerical data from product names using configurable patterns.
        
        Args:
            product_name: Product name to extract numerical data from
            
        Returns:
            List of numerical values found in the product name
        """
        # Use configurable patterns
        patterns = []
        for pattern_list in self.config.variant_patterns.values():
            patterns.extend(pattern_list)
        
        numerical_values = []
        
        for pattern in patterns:
            matches = re.findall(pattern, product_name, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle patterns that capture multiple groups
                    for group in match:
                        if group:
                            try:
                                numerical_values.append(float(group))
                            except ValueError:
                                continue
                else:
                    try:
                        numerical_values.append(float(match))
                    except ValueError:
                        continue
        
        return numerical_values

    def calculate_similarity_score(self, product1: Dict, product2: Dict) -> float:
        """
        Calculate similarity score between two products based on brand, category, and numerical data.
        
        Args:
            product1: First product dictionary with brand, category, and numerical data
            product2: Second product dictionary with brand, category, and numerical data
            
        Returns:
            Similarity score between 0 and 1
        """
        score = 0.0
        
        # Brand similarity (exact match = 1.0, no match = 0.0)
        brand1 = product1.get('brand', '').lower().strip()
        brand2 = product2.get('brand', '').lower().strip()
        if brand1 and brand2:
            if brand1 == brand2:
                score += 0.2  # 20% weight for brand match
            else:
                return 0.0  # Different brands = no similarity
        
        # Category similarity (exact match = 1.0, no match = 0.0)
        category1 = product1.get('category', '').lower().strip()
        category2 = product2.get('category', '').lower().strip()
        if category1 and category2:
            if category1 == category2:
                score += 0.2  # 20% weight for category match
            else:
                return 0.0  # Different categories = no similarity
        
        # Numerical data similarity (60% weight - highest priority)
        numerical1 = product1.get('numerical_data', [])
        numerical2 = product2.get('numerical_data', [])
        
        if not numerical1 and not numerical2:
            # Both have no numerical data - consider them similar
            score += 0.6
        elif not numerical1 or not numerical2:
            # One has numerical data, other doesn't - not similar
            score += 0.0
        else:
            # Both have numerical data - calculate similarity
            # Find the best matching numerical values
            max_similarity = 0.0
            for num1 in numerical1:
                for num2 in numerical2:
                    # Calculate similarity based on how close the numbers are
                    if num1 == 0 and num2 == 0:
                        similarity = 1.0
                    elif num1 == 0 or num2 == 0:
                        similarity = 0.0
                    else:
                        # Use relative difference for similarity
                        diff = abs(num1 - num2)
                        avg = (num1 + num2) / 2
                        similarity = max(0, 1 - (diff / avg))
                    
                    max_similarity = max(max_similarity, similarity)
            
            score += 0.6 * max_similarity
        
        return min(score, 1.0)

    def process_brand_data(self, df: pd.DataFrame, brand_name: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Process brand data with dynamic configuration.
        
        Args:
            df: DataFrame with product data
            brand_name: Name of the brand being processed
            
        Returns:
            Tuple of (processed_df, embeddings_array, core_identities_list)
        """
        self.console.print(Panel(f"[bold green]Processing Brand:[/bold green] {brand_name}", 
                                style="green"))
        
        processed_df = df.copy()
        core_identities = []
        
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Normalizing {brand_name} products...", 
                                   total=len(df))
            
            for idx, row in df.iterrows():
                try:
                    # Parse facets JSON if available, otherwise use empty dict
                    facets = {}
                    if 'facets_jsonb' in row and pd.notna(row['facets_jsonb']):
                        try:
                            if isinstance(row['facets_jsonb'], str):
                                facets = json.loads(row['facets_jsonb'])
                            elif isinstance(row['facets_jsonb'], dict):
                                facets = row['facets_jsonb']
                        except (json.JSONDecodeError, TypeError):
                            facets = {}
                    
                    # Extract brand name from facets for core identity
                    brand_from_facets = facets.get('brandName', brand_name) if facets else brand_name
                    
                    # Extract numerical data from product name
                    numerical_data = self.extract_numerical_data(row['label'])
                    
                    # Extract base product name for additional clustering layer
                    base_name = self.extract_base_product_name(
                        row['label'], 
                        brand_name, 
                        row.get('categoryLabel', '')
                    )
                    
                    # Create core identity string
                    core_identity = f"{base_name} | {row.get('categoryLabel', 'general')}"
                    core_identities.append(core_identity)
                    
                    # Store processed data
                    processed_df.at[idx, 'normalized_name'] = base_name
                    processed_df.at[idx, 'core_identity'] = core_identity
                    processed_df.at[idx, 'brand_extracted'] = brand_from_facets
                    processed_df.at[idx, 'categories_parsed'] = row.get('categoryLabel', '')
                    processed_df.at[idx, 'primary_category'] = row.get('categoryLabel', 'general')
                    
                except Exception as e:
                    self.logger.warning(f"Error processing row {idx}: {e}")
                    # Fallback processing
                    try:
                        fallback_base = self.extract_base_product_name(row['label'], brand_name)
                    except:
                        fallback_base = row['label']
                    
                    fallback_core = f"{fallback_base} | {row.get('categoryLabel', 'general')}"
                    core_identities.append(fallback_core)
                    processed_df.at[idx, 'normalized_name'] = fallback_base
                    processed_df.at[idx, 'core_identity'] = fallback_core
                    processed_df.at[idx, 'brand_extracted'] = brand_name
                    processed_df.at[idx, 'categories_parsed'] = row.get('categoryLabel', '')
                    processed_df.at[idx, 'primary_category'] = row.get('categoryLabel', 'general')
                
                progress.update(task, advance=1)
        
        # Create dummy embeddings array for compatibility
        embeddings = np.zeros((len(processed_df), 1), dtype=np.float32)
        
        return processed_df, embeddings, core_identities

    def perform_brand_category_numerical_clustering(self, processed_df: pd.DataFrame, brand_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform clustering based on brand, category, and numerical data from product names.
        Now uses dynamic configuration.
        
        Args:
            processed_df: Processed DataFrame with product data
            brand_name: Brand name for logging
            
        Returns:
            Tuple of (cluster_labels, base_cluster_labels)
        """
        self.console.print(f"[blue]Performing dynamic brand-category-numerical clustering for {brand_name}...[/blue]")
        
        n_products = len(processed_df)
        self.console.print(f"[cyan]Processing {n_products} products...[/cyan]")
        
        # Prepare product data for clustering
        products_data = []
        for idx, row in processed_df.iterrows():
            # Extract numerical data from product name
            numerical_data = self.extract_numerical_data(row['label'])
            
            # Extract base product name for additional clustering layer
            base_name = self.extract_base_product_name(
                row['label'], 
                brand_name, 
                row.get('categoryLabel', '')
            )
            
            product_data = {
                'index': idx,
                'brand': row.get('brand_extracted', brand_name),
                'category': row.get('primary_category', row.get('categoryLabel', '')),
                'numerical_data': numerical_data,
                'base_name': base_name,
                'label': row['label']
            }
            products_data.append(product_data)
        
        # Perform primary clustering using similarity matrix
        cluster_labels = np.full(n_products, -1)  # Initialize all as noise
        current_cluster_id = 0
        similarity_threshold = 0.8  # Primary clustering threshold
        
        # Create similarity matrix and cluster products
        for i in range(n_products):
            if cluster_labels[i] != -1:  # Already assigned to a cluster
                continue
                
            # Start a new cluster with product i
            cluster_labels[i] = current_cluster_id
            cluster_members = [i]
            
            # Find all products similar to product i
            for j in range(i + 1, n_products):
                if cluster_labels[j] != -1:  # Already assigned
                    continue
                    
                similarity = self.calculate_similarity_score(products_data[i], products_data[j])
                
                if similarity >= similarity_threshold:
                    cluster_labels[j] = current_cluster_id
                    cluster_members.append(j)
            
            # Only keep clusters with at least min_cluster_size members
            if len(cluster_members) >= self.config.min_cluster_size:
                current_cluster_id += 1
            else:
                # Mark as noise if cluster is too small
                for member_idx in cluster_members:
                    cluster_labels[member_idx] = -1
        
        # Perform additional base name clustering within each primary cluster
        self.console.print(f"[blue]Performing dynamic base name clustering within primary clusters...[/blue]")
        base_cluster_labels = np.full(n_products, -1)  # Initialize all as noise
        base_cluster_id = 0
        
        # Group products by primary cluster
        primary_clusters = {}
        for idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in primary_clusters:
                primary_clusters[cluster_id] = []
            primary_clusters[cluster_id].append(idx)
        
        # For each primary cluster, perform base name clustering
        for primary_cluster_id, product_indices in primary_clusters.items():
            if primary_cluster_id == -1:  # Skip noise points
                continue
                
            # Extract products in this primary cluster
            cluster_products = [products_data[i] for i in product_indices]
            
            # Perform base name clustering within this cluster
            for i, product_i in enumerate(cluster_products):
                if base_cluster_labels[product_i['index']] != -1:  # Already assigned
                    continue
                    
                # Start a new base cluster with product i
                base_cluster_labels[product_i['index']] = base_cluster_id
                base_cluster_members = [product_i['index']]
                
                # Find all products with similar base names
                for j, product_j in enumerate(cluster_products):
                    if i >= j or base_cluster_labels[product_j['index']] != -1:  # Skip self or already assigned
                        continue
                    
                    base_similarity = self.calculate_base_name_similarity(
                        product_i['base_name'], 
                        product_j['base_name'],
                        product_i['label'],
                        product_j['label'],
                        product_i['category']
                    )
                    
                    if base_similarity >= self.config.similarity_threshold:
                        base_cluster_labels[product_j['index']] = base_cluster_id
                        base_cluster_members.append(product_j['index'])
                
                # Only keep base clusters with at least 2 members
                if len(base_cluster_members) >= 2:
                    base_cluster_id += 1
                else:
                    # Mark as noise if base cluster is too small
                    for member_idx in base_cluster_members:
                        base_cluster_labels[member_idx] = -1
        
        # Log clustering results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        n_base_clusters = len(set(base_cluster_labels)) - (1 if -1 in base_cluster_labels else 0)
        n_base_noise = list(base_cluster_labels).count(-1)
        
        self.logger.info(f"Dynamic clustering results for {brand_name}:")
        self.logger.info(f"  - Primary clusters: {n_clusters}")
        self.logger.info(f"  - Primary noise points: {n_noise}")
        self.logger.info(f"  - Base name clusters: {n_base_clusters}")
        self.logger.info(f"  - Base name noise points: {n_base_noise}")
        self.logger.info(f"  - Products clustered: {len(cluster_labels) - n_noise}")
        self.logger.info(f"  - Similarity threshold: {self.config.similarity_threshold}")
        
        return cluster_labels, base_cluster_labels

    def perform_clustering(self, embeddings: np.ndarray, brand_name: str, processed_df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform clustering using dynamic configuration.
        
        Args:
            embeddings: Vector embeddings array (kept for compatibility, not used)
            brand_name: Brand name for logging
            processed_df: Processed DataFrame with product data
            
        Returns:
            Tuple of (cluster_labels, base_cluster_labels)
        """
        return self.perform_brand_category_numerical_clustering(processed_df, brand_name)

    def generate_output(self, processed_df: pd.DataFrame, cluster_data: Tuple[np.ndarray, np.ndarray], 
                       brand_name: str) -> Dict[str, Any]:
        """
        Generate final output structure with cluster mappings.
        
        Args:
            processed_df: Processed DataFrame with product data
            cluster_data: Tuple of (cluster_labels, base_cluster_labels)
            brand_name: Brand name
            
        Returns:
            Dictionary with sister product clusters
        """
        self.console.print(f"[blue]Generating output structure for {brand_name}...[/blue]")
        
        cluster_labels, base_cluster_labels = cluster_data
        
        # Add cluster labels to DataFrame
        processed_df['cluster_id'] = cluster_labels
        processed_df['base_cluster_id'] = base_cluster_labels
        
        # Add base product names to DataFrame
        processed_df['base_product_name'] = processed_df.apply(
            lambda row: self.extract_base_product_name(
                row['label'], 
                brand_name, 
                row.get('categoryLabel', '')
            ), axis=1
        )
        
        # Create sister product clusters mapping
        sister_clusters = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_products = processed_df[processed_df['cluster_id'] == cluster_id]
            
            # Group by base cluster within this primary cluster
            base_clusters = {}
            for _, product in cluster_products.iterrows():
                base_cluster_id = product['base_cluster_id']
                if base_cluster_id not in base_clusters:
                    base_clusters[base_cluster_id] = []
                base_clusters[base_cluster_id].append(product)
            
            cluster_data = []
            for base_cluster_id, base_products in base_clusters.items():
                base_cluster_data = []
                for product in base_products:
                    base_cluster_data.append({
                        'brandSKUId': product['brandSKUId'],
                        'label': product['label'],
                        'normalized_name': product['normalized_name'],
                        'core_identity': product['core_identity'],
                        'categoryLabel': product['categoryLabel'],
                        'categories_parsed': product.get('categories_parsed', ''),
                        'primary_category': product.get('primary_category', 'general'),
                        'base_product_name': product['base_product_name'],
                        'base_cluster_id': base_cluster_id
                    })
                
                cluster_data.append({
                    'base_cluster_id': base_cluster_id,
                    'base_product_name': base_products[0]['base_product_name'] if base_products else '',
                    'products': base_cluster_data
                })
            
            sister_clusters[f"cluster_{cluster_id}"] = {
                'base_clusters': cluster_data
            }
        
        # Handle noise points (products with no sisters)
        noise_products = processed_df[processed_df['cluster_id'] == -1]
        if len(noise_products) > 0:
            noise_data = []
            for _, product in noise_products.iterrows():
                noise_data.append({
                    'brandSKUId': product['brandSKUId'],
                    'label': product['label'],
                    'normalized_name': product['normalized_name'],
                    'core_identity': product['core_identity'],
                    'categoryLabel': product['categoryLabel'],
                    'categories_parsed': product.get('categories_parsed', ''),
                    'primary_category': product.get('primary_category', 'general'),
                    'base_product_name': product['base_product_name'],
                    'base_cluster_id': product['base_cluster_id']
                })
            sister_clusters['no_sisters'] = {
                'products': noise_data
            }
        
        # Create final output structure
        output = {
            'brand': brand_name,
            'total_products': len(processed_df),
            'total_clusters': len([k for k in sister_clusters.keys() if k != 'no_sisters']),
            'products_with_sisters': len(processed_df[processed_df['cluster_id'] != -1]),
            'products_without_sisters': len(noise_products),
            'sisterProductClusters': sister_clusters,
            'processing_metadata': {
                'clustering_method': 'dynamic_brand_category_numerical_with_base_name',
                'similarity_threshold': self.config.similarity_threshold,
                'core_term_boost': self.config.core_term_boost,
                'min_word_overlap_similarity': self.config.min_word_overlap_similarity,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        return output

    def save_results(self, results: Dict[str, Any], brand_name: str, processed_df: pd.DataFrame):
        """Save results to various output formats."""
        brand_clean = re.sub(r'[^\w\-_]', '_', brand_name.lower())
        
        # Save main results as JSON
        json_file = self.output_dir / f"{brand_clean}_sister_products.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save detailed CSV
        csv_file = self.output_dir / f"{brand_clean}_detailed_results.csv"
        processed_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Save cluster summary CSV
        summary_data = []
        for cluster_name, cluster_info in results['sisterProductClusters'].items():
            if cluster_name == 'no_sisters':
                # Handle noise products
                products = cluster_info.get('products', [])
                for product in products:
                    summary_data.append({
                        'cluster_id': cluster_name,
                        'base_cluster_id': product.get('base_cluster_id', -1),
                        'base_product_name': product.get('base_product_name', ''),
                        'brandSKUId': product['brandSKUId'],
                        'label': product['label'],
                        'normalized_name': product['normalized_name'],
                        'categoryLabel': product['categoryLabel'],
                        'categories_parsed': product.get('categories_parsed', ''),
                        'primary_category': product.get('primary_category', 'general')
                    })
            else:
                # Handle regular clusters with base clusters
                base_clusters = cluster_info.get('base_clusters', [])
                for base_cluster in base_clusters:
                    base_cluster_id = base_cluster.get('base_cluster_id', -1)
                    base_product_name = base_cluster.get('base_product_name', '')
                    products = base_cluster.get('products', [])
                    
                    for product in products:
                        summary_data.append({
                            'cluster_id': cluster_name,
                            'base_cluster_id': base_cluster_id,
                            'base_product_name': base_product_name,
                            'brandSKUId': product['brandSKUId'],
                            'label': product['label'],
                            'normalized_name': product['normalized_name'],
                            'categoryLabel': product['categoryLabel'],
                            'categories_parsed': product.get('categories_parsed', ''),
                            'primary_category': product.get('primary_category', 'general')
                        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / f"{brand_clean}_cluster_summary.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        self.console.print(f"[green]Results saved:[/green]")
        self.console.print(f"  - JSON: {json_file}")
        self.console.print(f"  - Detailed CSV: {csv_file}")
        self.console.print(f"  - Summary CSV: {summary_file}")

    def display_results_summary(self, results: Dict[str, Any]):
        """Display a summary table of clustering results."""
        
        table = Table(title=f"Dynamic Sister Products Mapping Results - {results['brand']}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Products", str(results['total_products']))
        table.add_row("Total Clusters", str(results['total_clusters']))
        table.add_row("Products with Sisters", str(results['products_with_sisters']))
        table.add_row("Products without Sisters", str(results['products_without_sisters']))
        table.add_row("Clustering Method", results['processing_metadata']['clustering_method'])
        table.add_row("Similarity Threshold", str(results['processing_metadata']['similarity_threshold']))
        
        self.console.print(table)
        
        # Show top clusters
        clusters = results['sisterProductClusters']
        if clusters:
            self.console.print("\n[bold]Top Sister Product Clusters:[/bold]")
            cluster_sizes = []
            for name, cluster_info in clusters.items():
                if name != 'no_sisters':
                    base_clusters = cluster_info.get('base_clusters', [])
                    total_products = sum(len(bc.get('products', [])) for bc in base_clusters)
                    cluster_sizes.append((name, total_products))
            
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            
            for cluster_name, size in cluster_sizes[:5]:  # Show top 5 clusters
                cluster_info = clusters[cluster_name]
                base_clusters = cluster_info.get('base_clusters', [])
                
                self.console.print(f"\n[green]{cluster_name}[/green] ({size} products):")
                
                # Show products from base clusters
                product_count = 0
                for base_cluster in base_clusters[:2]:  # Show first 2 base clusters
                    products = base_cluster.get('products', [])
                    base_name = base_cluster.get('base_product_name', 'Unknown')
                    self.console.print(f"  Base: {base_name}")
                    for product in products[:2]:  # Show first 2 products per base cluster
                        self.console.print(f"    {product_count+1}. {product['label']}")
                        product_count += 1
                        if product_count >= 3:  # Limit to 3 total products shown
                            break
                    if product_count >= 3:
                        break
                
                if size > 3:
                    self.console.print(f"  ... and {size - 3} more")

    def process_from_database(self, brand_id: Optional[str] = None, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Process sister products mapping using live data from PostgreSQL database.
        
        Args:
            brand_id: Optional specific brand ID to process. If None, processes all brands.
            batch_size: Number of products to process per brand batch
            
        Returns:
            Dictionary containing all brand results
        """
        self.console.print("[bold blue]🚀 Starting Dynamic Database-Driven Sister Products Mapping[/bold blue]")
        
        try:
            if brand_id:
                # Process specific brand
                return self._process_single_brand_from_db(brand_id)
            else:
                # Process all brands
                return self._process_all_brands_from_db(batch_size)
                
        except Exception as e:
            self.logger.error(f"Failed to process from database: {e}")
            self.console.print(f"[red]✗ Processing failed: {e}[/red]")
            raise

    def _process_single_brand_from_db(self, brand_id: str) -> Dict[str, Any]:
        """Process a single brand from the database."""
        try:
            # Fetch brand data
            df = self.db_manager.fetch_brand_products(brand_id)
            
            if df.empty:
                self.console.print(f"[yellow]⚠️ No products found for brand {brand_id}[/yellow]")
                return {}
            
            # Get brand name from first row
            brand_name = df['brandLabel'].iloc[0] if 'brandLabel' in df.columns else df['brandId'].iloc[0]
            
            self.console.print(Panel(
                f"[bold green]Processing Brand from Database[/bold green]\n"
                f"Brand ID: {brand_id}\nProducts: {len(df)}",
                style="green"
            ))
            
            # Process through the pipeline
            processed_df, embeddings, core_identities = self.process_brand_data(df, brand_name)
            cluster_data = self.perform_clustering(embeddings, brand_name, processed_df)
            results = self.generate_output(processed_df, cluster_data, brand_name)
            
            # Save results to CSV and append to master
            self._save_and_append_results(results, brand_name, processed_df)
            
            # Display results
            self.display_results_summary(results)
            
            return {brand_name: results}
            
        except Exception as e:
            self.logger.error(f"Failed to process brand {brand_id}: {e}")
            raise

    def _process_all_brands_from_db(self, batch_size: int = 1000) -> Dict[str, Any]:
        """Process all brands from the database."""
        try:
            # Get all brands
            brands_df = self.db_manager.get_all_brands()
            
            if brands_df.empty:
                self.console.print("[yellow]⚠️ No verified brands found[/yellow]")
                return {}
            
            all_results = {}
            total_brands = len(brands_df)
            
            self.console.print(f"[blue]📊 Processing {total_brands} brands from database[/blue]")
            
            for idx, brand_row in brands_df.iterrows():
                brand_id = brand_row['brandId']
                brand_name = brand_row['brandLabel']
                
                self.console.print(f"\n[cyan]═══ Brand {idx+1}/{total_brands}: {brand_name} ═══[/cyan]")
                
                try:
                    # Fetch and process brand data
                    df = self.db_manager.fetch_brand_products(brand_id)
                    
                    if df.empty:
                        self.console.print(f"[yellow]⚠️ No products found for {brand_name}[/yellow]")
                        continue
                    
                    # Use brand name from fetched data for consistent file naming
                    actual_brand_name = df['brandLabel'].iloc[0] if 'brandLabel' in df.columns else brand_name
                    
                    # Process through the pipeline
                    processed_df, embeddings, core_identities = self.process_brand_data(df, actual_brand_name)
                    cluster_data = self.perform_clustering(embeddings, actual_brand_name, processed_df)
                    results = self.generate_output(processed_df, cluster_data, actual_brand_name)
                    
                    # Save and append results
                    self._save_and_append_results(results, actual_brand_name, processed_df)
                    
                    # Display results
                    self.display_results_summary(results)
                    
                    all_results[actual_brand_name] = results
                    
                    self.console.print(f"[green]✅ {actual_brand_name} processing complete[/green]")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process brand {brand_name}: {e}")
                    self.console.print(f"[red]✗ Error processing {brand_name}: {e}[/red]")
                    continue
            
            # Save final master CSV
            self._save_master_results()
            
            # Save combined JSON results
            combined_file = self.output_dir / "all_brands_database_results.json"
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            self.console.print(f"[green]📁 Combined results saved to: {combined_file}[/green]")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Failed to process all brands: {e}")
            raise

    def _save_and_append_results(self, results: Dict[str, Any], brand_name: str, processed_df: pd.DataFrame):
        """Append brand results to master CSV with conflict resolution."""
        try:
            # Prepare data for master CSV with conflict resolution
            brand_results_df = processed_df.copy()
            brand_results_df['brand_name'] = brand_name
            brand_results_df['processing_timestamp'] = pd.Timestamp.now()
            
            # Append to master results with conflict resolution on brandSKUId
            if self.master_results_df.empty:
                self.master_results_df = brand_results_df.copy()
            else:
                # Remove existing entries for this brand to avoid duplicates
                self.master_results_df = self.master_results_df[
                    self.master_results_df['brandId'] != brand_results_df['brandId'].iloc[0]
                ]
                # Append new results
                self.master_results_df = pd.concat([self.master_results_df, brand_results_df], 
                                                 ignore_index=True)
            
            # Live save to master CSV
            master_csv_path = self.output_dir / "master_sister_products_results.csv"
            self.master_results_df.to_csv(master_csv_path, index=False)
            
            self.console.print(f"[green]💾 Live saved to master CSV ({len(self.master_results_df)} total products)[/green]")
            
        except Exception as e:
            self.logger.error(f"Error appending results for {brand_name}: {e}")
            self.console.print(f"[red]✗ Error saving results for {brand_name}: {e}[/red]")

    def _save_master_results(self):
        """Save final master results to multiple formats."""
        try:
            if self.master_results_df.empty:
                self.console.print("[yellow]⚠️ No master results to save[/yellow]")
                return
            
            # Save master CSV
            master_csv_path = self.output_dir / "master_sister_products_results.csv"
            self.master_results_df.to_csv(master_csv_path, index=False)
            
            # Save summary statistics
            summary_stats = {
                'total_products': len(self.master_results_df),
                'total_brands': self.master_results_df['brandId'].nunique(),
                'total_clusters': self.master_results_df['cluster_id'].nunique() - (1 if -1 in self.master_results_df['cluster_id'].values else 0),
                'products_with_sisters': len(self.master_results_df[self.master_results_df['cluster_id'] != -1]),
                'clustering_rate': len(self.master_results_df[self.master_results_df['cluster_id'] != -1]) / len(self.master_results_df) * 100,
                'processing_timestamp': pd.Timestamp.now().isoformat()
            }
            
            summary_file = self.output_dir / "master_processing_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_stats, f, indent=2, ensure_ascii=False)
            
            self.console.print(f"[green]📊 Master results saved: {len(self.master_results_df)} products across {summary_stats['total_brands']} brands[/green]")
            self.console.print(f"[blue]📈 Overall clustering rate: {summary_stats['clustering_rate']:.1f}%[/blue]")
            
        except Exception as e:
            self.logger.error(f"Error saving master results: {e}")
            self.console.print(f"[red]✗ Error saving master results: {e}[/red]")
