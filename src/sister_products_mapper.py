#!/usr/bin/env python3
"""
Sister Products Mapping System

This module implements a sophisticated system for mapping sister products using
vector embeddings and density-based clustering. The system follows a three-phase approach:

Phase 1: Normalization & Embedding - Clean product names and create vector embeddings
Phase 2: Clustering - Use HDBSCAN for automatic sister product grouping  
Phase 3: Output Generation - Create structured mappings with cluster assignments

Author: Sister Products Mapping System
Date: 2024
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import jellyfish
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SisterProductsMapper:
    """
    Main class for mapping sister products using vector embeddings and clustering.
    
    This class implements a three-phase approach:
    1. Normalization & Embedding: Clean product names and create embeddings
    2. Clustering: Use HDBSCAN for automatic sister product grouping
    3. Output Generation: Create structured mappings
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 min_cluster_size: int = 2,
                 min_samples: int = 1,
                 cluster_selection_epsilon: float = 0.0,
                 enable_phonetic: bool = False,
                 phonetic_algorithm: str = 'soundex',
                 output_dir: str = 'output',
                 logs_dir: str = 'logs'):
        """
        Initialize the Sister Products Mapper.
        
        Args:
            model_name: Sentence transformer model name
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
            cluster_selection_epsilon: Distance threshold for cluster merging in HDBSCAN. 
                                     Higher values (e.g., 0.1-0.3) create larger, more inclusive clusters.
                                     Default 0.0 uses standard HDBSCAN behavior.
            enable_phonetic: Enable phonetic similarity encoding for better clustering of
                           similar-sounding words (e.g., 'burfi', 'burfee', 'barfee')
            phonetic_algorithm: Phonetic algorithm to use ('soundex', 'metaphone', 'nysiis', 'match_rating_codex')
            output_dir: Directory for output files
            logs_dir: Directory for log files
        """
        self.model_name = model_name
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.enable_phonetic = enable_phonetic
        self.phonetic_algorithm = phonetic_algorithm
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
        
        # Variant removal patterns - these will be removed from product names
        self.variant_patterns = {
            'weight': [
                r'\b\d+(\.\d+)?\s*(grams?|g|kg|kilograms?|ounces?|oz|lbs?|pounds?)\b',
                r'\b\d+(\.\d+)?\s*gm\b',
                r'\b\d+(\.\d+)?\s*GM\b'
            ],
            'volume': [
                r'\b\d+(\.\d+)?\s*(ml|milliliters?|liters?|l|fl\.?\s*oz|fluid\s*ounces?)\b',
                r'\b\d+(\.\d+)?\s*ML\b'
            ],
            'size': [
                r'\b(small|medium|large|xl|xxl|s|m|l)\b',
                r'\b\d+\s*(inch|inches|cm|centimeters?|mm|millimeters?)\b'
            ],
            'price': [
                r'rs\.?\s*\d+',
                r'₹\s*\d+',
                r'\$\s*\d+(\.\d+)?'
            ],
            'packet_size': [
                r'\b(single-use|multi-use|family\s*pack|party\s*pack)\b'
            ]
        }
        
        # Common variant facet keys to extract values from
        self.variant_facet_keys = [
            'flavour', 'flavor', 'scent', 'netWeight', 'netVolume', 
            'packetSize', 'productWeight', 'productVolume', 'color',
            'texture', 'spiceLevel', 'specialTag', 'variant',
            'packQuantity', 'servingSize', 'seasonalUse'
        ]
        
    def _get_phonetic_encoding(self, text: str) -> str:
        """
        Generate phonetic encoding of text using the specified algorithm.
        
        Args:
            text: Text to encode
            
        Returns:
            Phonetic encoding string
        """
        if not text or not self.enable_phonetic:
            return ""
            
        # Clean text - remove numbers, special chars, extra spaces
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        clean_text = re.sub(r'\d+', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if not clean_text:
            return ""
            
        words = clean_text.split()
        phonetic_codes = []
        
        for word in words:
            if len(word) < 3:  # Skip very short words
                continue
                
            try:
                if self.phonetic_algorithm == 'soundex':
                    code = jellyfish.soundex(word)
                elif self.phonetic_algorithm == 'metaphone':
                    code = jellyfish.metaphone(word)
                elif self.phonetic_algorithm == 'nysiis':
                    code = jellyfish.nysiis(word)
                elif self.phonetic_algorithm == 'match_rating_codex':
                    code = jellyfish.match_rating_codex(word)
                else:
                    # Default to metaphone
                    code = jellyfish.metaphone(word)
                    
                if code:
                    phonetic_codes.append(code)
                    
            except Exception:
                # If phonetic encoding fails, skip this word
                continue
                
        return ' '.join(phonetic_codes)
    
    def _get_phonetic_similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate phonetic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.enable_phonetic or not text1 or not text2:
            return 0.0
            
        phonetic1 = self._get_phonetic_encoding(text1)
        phonetic2 = self._get_phonetic_encoding(text2)
        
        if not phonetic1 or not phonetic2:
            return 0.0
            
        # Simple word overlap similarity
        words1 = set(phonetic1.split())
        words2 = set(phonetic2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
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
        
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            self.console.print(f"[blue]Loading sentence transformer model: {self.model_name}[/blue]")
            try:
                self.model = SentenceTransformer(self.model_name)
                self.logger.info(f"Successfully loaded model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
                
    def normalize_product_name(self, label: str, facets: Dict[str, Any], brand_name: str = "") -> str:
        """
        Phase 1: Normalize product name by removing variant-specific information.
        
        This is the most critical step - it extracts the core product identity
        by removing flavors, sizes, weights, and other variant characteristics.
        
        Args:
            label: Original product label/name
            facets: Product facets dictionary
            brand_name: Brand name to remove from normalized text
            
        Returns:
            Normalized product name representing the core product
        """
        normalized = label.lower().strip()
        
        # Extract variant values from facets and remove them from the label
        # But be selective - only remove weight/volume/size info, not flavor/product identifiers
        variant_values_to_remove = set()
        
        # Only remove these specific types of variants that are truly just packaging/size variants
        size_weight_facets = ['netWeight', 'netVolume', 'packetSize', 'productWeight', 'productVolume', 'servingSize']
        
        for facet_key in size_weight_facets:
            if facet_key in facets and facets[facet_key]:
                value = str(facets[facet_key]).strip()
                if value and value.lower() not in ['n/a', 'null', 'none', '']:
                    # Clean and add to removal set
                    cleaned_value = re.sub(r'[^\w\s]', '', value.lower())
                    if len(cleaned_value) > 2:  # Only remove meaningful values
                        variant_values_to_remove.add(cleaned_value)
                        variant_values_to_remove.add(value.lower())
        
        # Remove variant values found in facets
        for variant_value in variant_values_to_remove:
            # Try exact match first
            if variant_value in normalized:
                normalized = normalized.replace(variant_value, ' ')
            
            # Try partial matches for compound values
            words = variant_value.split()
            if len(words) > 1:
                for word in words:
                    if len(word) > 3:  # Only remove meaningful words
                        normalized = re.sub(rf'\b{re.escape(word)}\b', ' ', normalized)
        
        # Remove brand name FIRST (before other processing affects it)
        if brand_name and brand_name.strip():
            brand_clean = brand_name.lower().strip()
            # Remove exact brand name matches (case-insensitive)
            normalized = re.sub(rf'\b{re.escape(brand_clean)}\b', ' ', normalized, flags=re.IGNORECASE)
            
            # Also try removing brand name without special characters
            brand_alphanumeric = re.sub(r'[^\w\s]', '', brand_clean)
            if brand_alphanumeric and brand_alphanumeric != brand_clean:
                normalized = re.sub(rf'\b{re.escape(brand_alphanumeric)}\b', ' ', normalized, flags=re.IGNORECASE)
            
            # Handle partial brand names that might remain (like "ram" from "Haldiram")
            # Also check for substrings of the brand name that might remain
            if len(brand_clean) > 3:
                # Remove substrings of 3+ characters from the brand name
                for i in range(len(brand_clean) - 2):
                    for j in range(i + 3, len(brand_clean) + 1):
                        substring = brand_clean[i:j]
                        if len(substring) >= 3:
                            normalized = re.sub(rf'\b{re.escape(substring)}\b', ' ', normalized, flags=re.IGNORECASE)
        
        # Remove dashes and special characters early
        normalized = re.sub(r'\s*[-–—]\s*', ' ', normalized)  # Dashes with optional spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)  # Other special characters
        
        # Apply pattern-based removal for weights, volumes, etc.
        for category, patterns in self.variant_patterns.items():
            for pattern in patterns:
                normalized = re.sub(pattern, ' ', normalized, flags=re.IGNORECASE)
        
        # Remove numbers along with the word that follows them (but be more specific)
        # This targets patterns like "250 grams", "13 gram", "180g", etc.
        normalized = re.sub(r'\b\d+\s*(g|gm|gram|grams|kg|ml|liter|liters|oz|pounds?|lbs?)\b', ' ', normalized, flags=re.IGNORECASE)
        
        # Remove standalone numbers that might be left
        normalized = re.sub(r'\b\d+\b', ' ', normalized)
        
        # Remove other noise patterns
        noise_patterns = [
            r'\b(rs\.?|₹)\s*\d+\b',  # Price mentions
            r'\b\d+%\s*(extra|off|free)\b',  # Promotional text
            r'\b(pack|packet|pouch|bag|tin|can|jar|bottle)\b',  # Packaging
            r'\b(new|improved|special|premium|original)\b',  # Marketing terms (excluding "classic" as it can be product identity)
            r'\b(available|now|today|limited|offer)\b',  # Availability terms
        ]
        
        for pattern in noise_patterns:
            normalized = re.sub(pattern, ' ', normalized, flags=re.IGNORECASE)
        
        # Final cleanup
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces to single
        normalized = normalized.strip()
        
        # If normalization resulted in too short a string, fall back to original
        if len(normalized) < 5:
            normalized = re.sub(r'[^\w\s]', ' ', label.lower())
            normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def parse_categories(self, category_input: str) -> List[str]:
        """
        Parse multiple categories from category input string.
        
        Args:
            category_input: Category string that may contain multiple categories
            
        Returns:
            List of individual categories
        """
        if not category_input or not isinstance(category_input, str):
            return ["general"]
        
        # Handle different separators: "/", ",", ";", "|"
        separators = ["/", ",", ";", "|"]
        categories = [category_input.strip()]
        
        for separator in separators:
            if separator in category_input:
                categories = [cat.strip() for cat in category_input.split(separator)]
                break
        
        # Clean and filter categories
        cleaned_categories = []
        for cat in categories:
            cat_clean = cat.strip()
            if cat_clean and cat_clean.lower() not in ['', 'n/a', 'null', 'none', 'n', 'a']:
                cleaned_categories.append(cat_clean)
        
        return cleaned_categories if cleaned_categories else ["general"]

    def create_core_identity(self, normalized_name: str, category: str, brand: str = "") -> str:
        """
        Create the core identity string for embedding generation.
        
        Args:
            normalized_name: Normalized product name
            category: Product category (can be multiple categories separated by delimiters)
            brand: Brand name (optional)
            
        Returns:
            Core identity string formatted for embedding
        """
        # Parse multiple categories
        categories = self.parse_categories(category)
        
        # Create category string - use primary category for core identity but include all for context
        primary_category = categories[0]
        if len(categories) > 1:
            # Include all categories for richer semantic understanding
            category_context = " ".join(categories)
        else:
            category_context = primary_category
        
        # Create base core identity
        core_identity = f"{normalized_name} | {category_context}"
        
        # Add phonetic encoding if enabled
        if self.enable_phonetic:
            phonetic_name = self._get_phonetic_encoding(normalized_name)
            if phonetic_name:
                core_identity += f" | PHONETIC:{phonetic_name}"
            
        return core_identity.lower().strip()
    
    def process_brand_data(self, df: pd.DataFrame, brand_name: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Process data for a single brand through normalization and embedding generation.
        
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
                    # Parse facets JSON
                    facets = json.loads(row['facets_jsonb']) if isinstance(row['facets_jsonb'], str) else row['facets_jsonb']
                    
                    # Extract brand name from facets for core identity (but use filename brand for normalization)
                    brand_from_facets = facets.get('brandName', brand_name)
                    
                    # Phase 1: Normalize the product name (use filename-based brand name for removal)
                    normalized_name = self.normalize_product_name(row['label'], facets, brand_name)
                    
                    # Parse categories for additional context
                    categories_list = self.parse_categories(row['categoryLabel'])
                    
                    # Create core identity string
                    core_identity = self.create_core_identity(
                        normalized_name, 
                        row['categoryLabel'], 
                        brand_from_facets
                    )
                    
                    core_identities.append(core_identity)
                    
                    # Store processed data
                    processed_df.at[idx, 'normalized_name'] = normalized_name
                    processed_df.at[idx, 'core_identity'] = core_identity
                    processed_df.at[idx, 'brand_extracted'] = brand_from_facets
                    processed_df.at[idx, 'categories_parsed'] = ', '.join(categories_list)
                    processed_df.at[idx, 'primary_category'] = categories_list[0]
                    
                except Exception as e:
                    self.logger.warning(f"Error processing row {idx}: {e}")
                    # Fallback processing - try basic normalization (use filename-based brand)
                    try:
                        fallback_normalized = self.normalize_product_name(row['label'], {}, brand_name)
                    except:
                        fallback_normalized = row['label']
                    
                    # Parse categories even in fallback
                    fallback_categories = self.parse_categories(row['categoryLabel'])
                    
                    fallback_core = f"{fallback_normalized} | {row['categoryLabel']}"
                    core_identities.append(fallback_core)
                    processed_df.at[idx, 'normalized_name'] = fallback_normalized
                    processed_df.at[idx, 'core_identity'] = fallback_core
                    processed_df.at[idx, 'brand_extracted'] = brand_name
                    processed_df.at[idx, 'categories_parsed'] = ', '.join(fallback_categories)
                    processed_df.at[idx, 'primary_category'] = fallback_categories[0]
                
                progress.update(task, advance=1)
        
        # Generate embeddings
        self.console.print("[blue]Generating vector embeddings...[/blue]")
        embeddings = self.model.encode(core_identities, 
                                     show_progress_bar=True,
                                     batch_size=32)
        
        return processed_df, embeddings, core_identities
    
    def perform_clustering(self, embeddings: np.ndarray, brand_name: str) -> np.ndarray:
        """
        Phase 2: Perform HDBSCAN clustering on embeddings.
        
        Args:
            embeddings: Vector embeddings array
            brand_name: Brand name for logging
            
        Returns:
            Cluster labels array
        """
        self.console.print(f"[blue]Performing HDBSCAN clustering for {brand_name}...[/blue]")
        
        # Initialize HDBSCAN clusterer
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        # Perform clustering
        cluster_labels = self.clusterer.fit_predict(embeddings)
        
        # Log clustering results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        self.logger.info(f"Clustering results for {brand_name}:")
        self.logger.info(f"  - Number of clusters: {n_clusters}")
        self.logger.info(f"  - Number of noise points: {n_noise}")
        self.logger.info(f"  - Products clustered: {len(cluster_labels) - n_noise}")
        
        return cluster_labels
    
    def generate_output(self, processed_df: pd.DataFrame, cluster_labels: np.ndarray, 
                       brand_name: str) -> Dict[str, Any]:
        """
        Phase 3: Generate final output structure with cluster mappings.
        
        Args:
            processed_df: Processed DataFrame with product data
            cluster_labels: Cluster labels from HDBSCAN
            brand_name: Brand name
            
        Returns:
            Dictionary with sister product clusters
        """
        self.console.print(f"[blue]Generating output structure for {brand_name}...[/blue]")
        
        # Add cluster labels to DataFrame
        processed_df['cluster_id'] = cluster_labels
        
        # Create sister product clusters mapping
        sister_clusters = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_products = processed_df[processed_df['cluster_id'] == cluster_id]
            
            cluster_data = []
            for _, product in cluster_products.iterrows():
                cluster_data.append({
                    'brandSKUId': product['brandSKUId'],
                    'label': product['label'],
                    'normalized_name': product['normalized_name'],
                    'core_identity': product['core_identity'],
                    'categoryLabel': product['categoryLabel'],
                    'categories_parsed': product.get('categories_parsed', ''),
                    'primary_category': product.get('primary_category', 'general')
                })
            
            sister_clusters[f"cluster_{cluster_id}"] = cluster_data
        
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
                    'primary_category': product.get('primary_category', 'general')
                })
            sister_clusters['no_sisters'] = noise_data
        
        # Create final output structure
        output = {
            'brand': brand_name,
            'total_products': len(processed_df),
            'total_clusters': len([k for k in sister_clusters.keys() if k != 'no_sisters']),
            'products_with_sisters': len(processed_df[processed_df['cluster_id'] != -1]),
            'products_without_sisters': len(noise_products),
            'sisterProductClusters': sister_clusters,
            'processing_metadata': {
                'model_used': self.model_name,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        return output
    
    def save_results(self, results: Dict[str, Any], brand_name: str, 
                    processed_df: pd.DataFrame):
        """
        Save results to various output formats.
        
        Args:
            results: Results dictionary
            brand_name: Brand name
            processed_df: Processed DataFrame
        """
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
        for cluster_name, products in results['sisterProductClusters'].items():
            for product in products:
                summary_data.append({
                    'cluster_id': cluster_name,
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
        
        table = Table(title=f"Sister Products Mapping Results - {results['brand']}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Products", str(results['total_products']))
        table.add_row("Total Clusters", str(results['total_clusters']))
        table.add_row("Products with Sisters", str(results['products_with_sisters']))
        table.add_row("Products without Sisters", str(results['products_without_sisters']))
        table.add_row("Model Used", results['processing_metadata']['model_used'])
        
        self.console.print(table)
        
        # Show top clusters
        clusters = results['sisterProductClusters']
        if clusters:
            self.console.print("\n[bold]Top Sister Product Clusters:[/bold]")
            cluster_sizes = [(name, len(products)) for name, products in clusters.items() 
                           if name != 'no_sisters']
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            
            for cluster_name, size in cluster_sizes[:5]:  # Show top 5 clusters
                products = clusters[cluster_name]
                self.console.print(f"\n[green]{cluster_name}[/green] ({size} products):")
                for i, product in enumerate(products[:3]):  # Show first 3 products
                    self.console.print(f"  {i+1}. {product['label']}")
                if len(products) > 3:
                    self.console.print(f"  ... and {len(products) - 3} more")
    
    def process_brand_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single brand CSV file through the complete pipeline.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Results dictionary
        """
        file_path = Path(file_path)
        brand_name = file_path.stem.replace('_products', '').title()
        
        self.console.print(Panel(
            f"[bold blue]Starting Sister Products Mapping[/bold blue]\n"
            f"File: {file_path.name}\nBrand: {brand_name}",
            style="blue"
        ))
        
        # Load data
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} products for {brand_name}")
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            raise
        
        # Load model if not already loaded
        self.load_model()
        
        # Process through the three phases
        processed_df, embeddings, core_identities = self.process_brand_data(df, brand_name)
        cluster_labels = self.perform_clustering(embeddings, brand_name)
        results = self.generate_output(processed_df, cluster_labels, brand_name)
        
        # Save and display results
        self.save_results(results, brand_name, processed_df)
        self.display_results_summary(results)
        
        return results
    
    def process_multiple_brands(self, file_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple brand files.
        
        Args:
            file_paths: List of CSV file paths
            
        Returns:
            Dictionary mapping brand names to their results
        """
        all_results = {}
        
        for file_path in file_paths:
            try:
                results = self.process_brand_file(file_path)
                brand_name = Path(file_path).stem.replace('_products', '').title()
                all_results[brand_name] = results
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        # Save combined results
        combined_file = self.output_dir / "all_brands_results.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        self.console.print(f"[green]Combined results saved to: {combined_file}[/green]")
        
        return all_results 